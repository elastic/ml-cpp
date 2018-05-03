/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CEventRateModel_h
#define INCLUDED_ml_model_CEventRateModel_h

#include <core/CMemory.h>
#include <core/CoreTypes.h>

#include <maths/CMultinomialConjugate.h>

#include <model/CFeatureData.h>
#include <model/CIndividualModel.h>
#include <model/CModelTools.h>
#include <model/CProbabilityAndInfluenceCalculator.h>
#include <model/ImportExport.h>
#include <model/ModelTypes.h>

#include <string>
#include <utility>
#include <vector>

#include <stdint.h>

namespace {
class CMockEventRateModel;
}
namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace model {

//! \brief The event rate model common functionality.
//!
//! DESCRIPTION:\n
//! It holds the priors used to compute time series' rarity and also
//! various statistics about the current bucketing time interval.
//! This model is used for computing the probability of each new sample
//! of some specified features of the event rate as a set of data is
//! continuously streamed to the model in time order.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The data about the current bucketing interval is stored on the model
//! so that the data gatherer objects can be shared by multiple models.
//! This is to reduce the model memory footprint when the event data is
//! being batched by time to support comparison in which case all models
//! share a data gatherer.
//!
//! It assumes data are supplied in time order since this means minimal
//! state can be maintained.
class MODEL_EXPORT CEventRateModel : public CIndividualModel {
public:
    using TFeatureData = SEventRateFeatureData;
    using TSizeFeatureDataPr = std::pair<std::size_t, TFeatureData>;
    using TSizeFeatureDataPrVec = std::vector<TSizeFeatureDataPr>;
    using TFeatureSizeFeatureDataPrVecPr = std::pair<model_t::EFeature, TSizeFeatureDataPrVec>;
    using TFeatureSizeFeatureDataPrVecPrVec = std::vector<TFeatureSizeFeatureDataPrVecPr>;
    using TCategoryProbabilityCache = CModelTools::CCategoryProbabilityCache;

    //! The statistics we maintain about a bucketing interval.
    struct MODEL_EXPORT SBucketStats {
        explicit SBucketStats(core_t::TTime startTime);

        //! The start time of this bucket.
        core_t::TTime s_StartTime;
        //! The non-zero person counts in the current bucket.
        TSizeUInt64PrVec s_PersonCounts;
        //! The total count in the current bucket.
        uint64_t s_TotalCount;
        //! The feature data samples for the current bucketing interval.
        TFeatureSizeFeatureDataPrVecPrVec s_FeatureData;
        //! A cache of the corrections applied to interim results.
        //! The key is <feature, pid, pid> for non-correlated corrections
        //! or <feature, pid, correlated_pid> for correlated corrections
        mutable TFeatureSizeSizeTripleDouble1VecUMap s_InterimCorrections;
    };

public:
    //! \name Life-cycle
    //@{
    //! \param[in] params The global configuration parameters.
    //! \param[in] dataGatherer The object that gathers time series data.
    //! \param[in] newFeatureModels The new models to use for each feature.
    //! \param[in] newFeatureCorrelateModelPriors The prior to use for the
    //! new model of correlates for each feature.
    //! \param[in] featureCorrelatesModels The model of all correlates for
    //! each feature.
    //! \param[in] probabilityPrior The prior used for the joint probabilities
    //! of seeing the people we are modeling.
    //! \param[in] influenceCalculators The influence calculators to use
    //! for each feature.
    CEventRateModel(const SModelParams& params,
                    const TDataGathererPtr& dataGatherer,
                    const TFeatureMathsModelPtrPrVec& newFeatureModels,
                    const TFeatureMultivariatePriorPtrPrVec& newFeatureCorrelateModelPriors,
                    const TFeatureCorrelationsPtrPrVec& featureCorrelatesModels,
                    const maths::CMultinomialConjugate& probabilityPrior,
                    const TFeatureInfluenceCalculatorCPtrPrVecVec& influenceCalculators);

    //! Constructor used for restoring persisted models.
    //!
    //! \note The current bucket statistics are left default initialized
    //! and so must be sampled for before this model can be used.
    CEventRateModel(const SModelParams& params,
                    const TDataGathererPtr& dataGatherer,
                    const TFeatureMathsModelPtrPrVec& newFeatureModels,
                    const TFeatureMultivariatePriorPtrPrVec& newFeatureCorrelateModelPriors,
                    const TFeatureCorrelationsPtrPrVec& featureCorrelatesModels,
                    const TFeatureInfluenceCalculatorCPtrPrVecVec& influenceCalculators,
                    core::CStateRestoreTraverser& traverser);

    //! Create a copy that will result in the same persisted state as the
    //! original.  This is effectively a copy constructor that creates a
    //! copy that's only valid for a single purpose.  The boolean flag is
    //! redundant except to create a signature that will not be mistaken
    //! for a general purpose copy constructor.
    CEventRateModel(bool isForPersistence, const CEventRateModel& other);
    //@}

    //! \name Persistence
    //@{
    //! Persist state by passing information to \p inserter.
    virtual void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Restore reading state from \p traverser.
    virtual bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Create a clone of this model that will result in the same persisted
    //! state.  The clone may be incomplete in ways that do not affect the
    //! persisted representation, and must not be used for any other
    //! purpose.
    //! \warning The caller owns the object returned.
    virtual CAnomalyDetectorModel* cloneForPersistence() const;
    //@}

    //! Get the model category.
    virtual model_t::EModelType category() const;

    //! Returns true.
    virtual bool isEventRate() const;

    //! Returns false.
    virtual bool isMetric() const;

    //! \name Bucket Statistics
    //@{
    //! Returns null.
    virtual TOptionalDouble baselineBucketCount(std::size_t pid) const;

    //! Get the value of \p feature for the person identified
    //! by \p pid in the bucketing interval containing \p time.
    //!
    //! \param[in] feature The feature of interest.
    //! \param[in] pid The identifier of the person of interest.
    //! \param[in] cid Ignored.
    //! \param[in] time The time of interest.
    virtual TDouble1Vec currentBucketValue(model_t::EFeature feature,
                                           std::size_t pid,
                                           std::size_t cid,
                                           core_t::TTime time) const;

    //! Get the baseline bucket value of \p feature for the person
    //! identified by \p pid as of the start of the current bucketing
    //! interval.
    //!
    //! \param[in] feature The feature of interest.
    //! \param[in] pid The identifier of the person of interest.
    //! \param[in] cid Ignored.
    //! \param[in] type A description of the type of result for which
    //! to get the baseline. See CResultType for more details.
    //! \param[in] correlated The correlated series' identifiers and
    //! their values if any.
    //! \param[in] time The time of interest.
    virtual TDouble1Vec baselineBucketMean(model_t::EFeature feature,
                                           std::size_t pid,
                                           std::size_t cid,
                                           model_t::CResultType type,
                                           const TSizeDoublePr1Vec& correlated,
                                           core_t::TTime time) const;

    //@}

    //! \name Person
    //@{
    //! Get the person unique identifiers which have a feature value
    //! in the bucketing time interval including \p time.
    virtual void currentBucketPersonIds(core_t::TTime time, TSizeVec& result) const;
    //@}

    //! \name Update
    //@{
    //! Sample any state needed by computeProbablity in the time
    //! interval [\p startTime, \p endTime] but do not update the
    //! model. This is needed by the results preview.
    //!
    //! \param[in] startTime The start of the time interval to sample.
    //! \param[in] endTime The end of the time interval to sample.
    virtual void sampleBucketStatistics(core_t::TTime startTime,
                                        core_t::TTime endTime,
                                        CResourceMonitor& resourceMonitor);

    //! Update the model with features samples from the time interval
    //! [\p startTime, \p endTime].
    //!
    //! \param[in] startTime The start of the time interval to sample.
    //! \param[in] endTime The end of the time interval to sample.
    //! \param[in] resourceMonitor The resourceMonitor.
    virtual void sample(core_t::TTime startTime, core_t::TTime endTime, CResourceMonitor& resourceMonitor);
    //@}

    //! \name Probability
    //@{
    //! Compute the probability of seeing the event counts in the
    //! time interval [\p startTime, \p endTime] for the person
    //! identified by \p pid.
    //!
    //! \param[in] pid The identifier of the person of interest.
    //! \param[in] startTime The start of the time interval of interest.
    //! \param[in] endTime The end of the time interval of interest.
    //! \param[in] partitioningFields The partitioning field (name, value)
    //! pairs for which to compute the the probability.
    //! \param[in] numberAttributeProbabilities Ignored.
    //! \param[out] result A structure containing the probability,
    //! the smallest \p numberAttributeProbabilities attribute
    //! probabilities, the influences and any extra descriptive data
    virtual bool computeProbability(std::size_t pid,
                                    core_t::TTime startTime,
                                    core_t::TTime endTime,
                                    CPartitioningFields& partitioningFields,
                                    std::size_t numberAttributeProbabilities,
                                    SAnnotatedProbability& result) const;
    //@}

    //! Get the checksum of this model.
    //!
    //! \param[in] includeCurrentBucketStats If true then include
    //! the current bucket statistics. (This is designed to handle
    //! serialization, for which we don't serialize the current
    //! bucket statistics.)
    virtual uint64_t checksum(bool includeCurrentBucketStats = true) const;

    //! Debug the memory used by this model.
    virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

    //! Get the memory used by this model.
    virtual std::size_t memoryUsage() const;

    //! Get the static size of this object - used for virtual hierarchies.
    virtual std::size_t staticSize() const;

    //! Get the non-estimated value of the the memory used by this model.
    virtual std::size_t computeMemoryUsage() const;

    //! Get a view of the internals of the model for visualization.
    virtual CModelDetailsViewPtr details() const;

    //! Get the value of the \p feature of the person identified
    //! by \p pid for the bucketing interval containing \p time.
    const TFeatureData*
    featureData(model_t::EFeature feature, std::size_t pid, core_t::TTime time) const;

private:
    //! Get the start time of the current bucket.
    virtual core_t::TTime currentBucketStartTime() const;

    //! Set the start time of the current bucket.
    virtual void currentBucketStartTime(core_t::TTime time);

    //! Get the person counts in the current bucket.
    virtual const TSizeUInt64PrVec& currentBucketPersonCounts() const;

    //! Get writable person counts in the current bucket.
    virtual TSizeUInt64PrVec& currentBucketPersonCounts();

    //! Set the current bucket total count.
    virtual void currentBucketTotalCount(uint64_t totalCount);

    //! Get the total count of the current bucket.
    uint64_t currentBucketTotalCount() const;

    //! Get the interim corrections of the current bucket.
    TFeatureSizeSizeTripleDouble1VecUMap& currentBucketInterimCorrections() const;

    //! Clear out large state objects for people that are pruned.
    virtual void clearPrunedResources(const TSizeVec& people, const TSizeVec& attributes);

    //! Check if there are correlates for \p feature and the person
    //! identified by \p pid.
    bool correlates(model_t::EFeature feature, std::size_t pid, core_t::TTime time) const;

    //! Fill in the probability calculation parameters for \p feature
    //! and person identified by \p pid.
    void fill(model_t::EFeature feature,
              std::size_t pid,
              core_t::TTime bucketTime,
              bool interim,
              CProbabilityAndInfluenceCalculator::SParams& params) const;

    //! Fill in the probability calculation parameters for the correlates
    //! of \p feature and the person identified by \p pid.
    void fill(model_t::EFeature feature,
              std::size_t pid,
              core_t::TTime bucketTime,
              bool interim,
              CProbabilityAndInfluenceCalculator::SCorrelateParams& params,
              TStrCRefDouble1VecDouble1VecPrPrVecVecVec& correlateInfluenceValues) const;

private:
    //! The statistics we maintain about the bucket.
    SBucketStats m_CurrentBucketStats;

    //! The prior for the joint probabilities of seeing the people
    //! we are modeling (this captures information about the person
    //! rarity).
    maths::CMultinomialConjugate m_ProbabilityPrior;

    //! A cache of the person probabilities as of the start of the
    //! for the bucketing interval.
    TCategoryProbabilityCache m_Probabilities;

    friend class CEventRateModelDetailsView;
    friend class ::CMockEventRateModel;
};
}
}

#endif // INCLUDED_ml_model_CEventRateModel_h
