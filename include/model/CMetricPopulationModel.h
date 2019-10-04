/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CMetricPopulationModel_h
#define INCLUDED_ml_model_CMetricPopulationModel_h

#include <core/CMemory.h>

#include <model/CMemoryUsageEstimator.h>
#include <model/CModelTools.h>
#include <model/CPopulationModel.h>
#include <model/CProbabilityAndInfluenceCalculator.h>
#include <model/ImportExport.h>
#include <model/ModelTypes.h>

#include <map>
#include <utility>
#include <vector>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace model {
//! \brief The model for computing the anomalousness of the values
//! each person in a population generates in a data stream.
//!
//! DESCRIPTION:\n
//! This model computes the probability of the metric value a person
//! generates in the bucketing time interval given the typical values
//! all people generate in that interval for each person in a population.
//! There are two distinct types of probability that it can compute:
//! a probability on the current bucketing interval and a probability
//! based on all the person's interactions to date. The later uses
//! (statistical) models of the person's metric values, and is only
//! available if person models are created.
//!
//! Various features of the data can be modeled. These include the
//! minimum, maximum and mean of a set of attributes across.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The data about the current bucketing interval is stored on the model
//! so that the data gatherer objects can be shared by multiple models.
//! This is to reduce the model memory footprint when the event data is
//! being batched by time for comparison in which case all models for
//! the same attribute share a data gatherer.
//!
//! It assumes data are supplied in time order since this means minimal
//! state can be maintained.
class MODEL_EXPORT CMetricPopulationModel : public CPopulationModel {
public:
    using TFeatureData = SMetricFeatureData;
    using TSizeSizePrFeatureDataPr = std::pair<TSizeSizePr, TFeatureData>;
    using TSizeSizePrFeatureDataPrVec = std::vector<TSizeSizePrFeatureDataPr>;
    using TFeatureSizeSizePrFeatureDataPrVecMap =
        std::map<model_t::EFeature, TSizeSizePrFeatureDataPrVec>;
    using TInterimBucketCorrectorCPtr = std::shared_ptr<const CInterimBucketCorrector>;
    using TProbabilityCache = CModelTools::CProbabilityCache;

    //! The statistics we maintain about a bucketing interval.
    struct MODEL_EXPORT SBucketStats {
        explicit SBucketStats(core_t::TTime startTime);

        //! The start time of this bucket.
        core_t::TTime s_StartTime;
        //! The non-zero counts of messages by people in the bucketing
        //! interval.
        TSizeUInt64PrVec s_PersonCounts;
        //! The metric features we are modeling.
        TFeatureSizeSizePrFeatureDataPrVecMap s_FeatureData;
        //! A cache of the corrections applied to interim results.
        mutable TCorrectionKeyDouble1VecUMap s_InterimCorrections;
    };

    //! Lift the overloads of currentBucketValue into the class scope.
    using CPopulationModel::currentBucketValue;

    //! Lift the overloads of baselineBucketMean into the class scope.
    using CAnomalyDetectorModel::baselineBucketMean;

    //! Lift the overloads of acceptPersistInserter into the class scope.
    using CPopulationModel::acceptPersistInserter;

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
    //! \param[in] influenceCalculators The influence calculators to use
    //! for each feature.
    //! \param[in] interimBucketCorrector Calculates corrections for interim
    //! buckets.
    //! \note The current bucket statistics are left default initialized
    //! and so must be sampled for before this model can be used.
    CMetricPopulationModel(const SModelParams& params,
                           const TDataGathererPtr& dataGatherer,
                           const TFeatureMathsModelSPtrPrVec& newFeatureModels,
                           const TFeatureMultivariatePriorSPtrPrVec& newFeatureCorrelateModelPriors,
                           TFeatureCorrelationsPtrPrVec&& featureCorrelatesModels,
                           const TFeatureInfluenceCalculatorCPtrPrVecVec& influenceCalculators,
                           const TInterimBucketCorrectorCPtr& interimBucketCorrector);

    //! Constructor used for restoring persisted models.
    //!
    //! \note The current bucket statistics are left default initialized
    //! and so must be sampled for before this model can be used.
    CMetricPopulationModel(const SModelParams& params,
                           const TDataGathererPtr& dataGatherer,
                           const TFeatureMathsModelSPtrPrVec& newFeatureModels,
                           const TFeatureMultivariatePriorSPtrPrVec& newFeatureCorrelateModelPriors,
                           TFeatureCorrelationsPtrPrVec&& featureCorrelatesModels,
                           const TFeatureInfluenceCalculatorCPtrPrVecVec& influenceCalculators,
                           const TInterimBucketCorrectorCPtr& interimBucketCorrector,
                           core::CStateRestoreTraverser& traverser);

    //! Create a copy that will result in the same persisted state as the
    //! original.  This is effectively a copy constructor that creates a
    //! copy that's only valid for a single purpose.  The boolean flag is
    //! redundant except to create a signature that will not be mistaken
    //! for a general purpose copy constructor.
    CMetricPopulationModel(bool isForPersistence, const CMetricPopulationModel& other);
    CMetricPopulationModel(const CMetricPopulationModel&) = delete;
    CMetricPopulationModel& operator=(const CMetricPopulationModel&) = delete;
    //@}

    //! Returns false.
    bool isEventRate() const override;

    //! Returns true.
    bool isMetric() const override;

    //! \name Persistence
    //@{
    //! Persist the state of the models only.
    void persistModelsState(core::CStatePersistInserter& /* inserter*/) const override {
        // NO-OP
    }

    //! Persist state by passing information to the supplied inserter
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const override;

    //! Add to the contents of the object.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) override;

    //! Create a clone of this model that will result in the same persisted
    //! state.  The clone may be incomplete in ways that do not affect the
    //! persisted representation, and must not be used for any other
    //! purpose.
    //! \warning The caller owns the object returned.
    CAnomalyDetectorModel* cloneForPersistence() const override;
    //@}

    //! Get the model category.
    model_t::EModelType category() const override;

    //! \name Bucket Statistics
    //@{
    //! Get the value of \p feature for the person identified
    //! by \p pid and the attribute identified by \p cid in the
    //! bucketing interval containing \p time.
    //!
    //! \param[in] feature The feature of interest
    //! \param[in] pid The identifier of the person of interest.
    //! \param[in] cid The identifier of the attribute of interest.
    //! \param[in] time The time of interest.
    TDouble1Vec currentBucketValue(model_t::EFeature feature,
                                   std::size_t pid,
                                   std::size_t cid,
                                   core_t::TTime time) const override;

    //! Get the population baseline mean of \p feature for the
    //! attribute identified by \p cid as of the start of the
    //! current bucketing interval.
    //!
    //! \param[in] feature The feature of interest
    //! \param[in] pid The identifier of the person of interest.
    //! \param[in] cid The identifier of the attribute of interest.
    //! \param[in] type A description of the type of result for which
    //! to get the baseline. See CResultType for more details.
    //! \param[in] correlated The correlated series' identifiers and
    //! their values if any.
    //! \param[in] time The time of interest.
    TDouble1Vec baselineBucketMean(model_t::EFeature feature,
                                   std::size_t pid,
                                   std::size_t cid,
                                   model_t::CResultType type,
                                   const TSizeDoublePr1Vec& correlated,
                                   core_t::TTime time) const override;

    //! Check if bucket statistics are available for the specified time.
    bool bucketStatsAvailable(core_t::TTime time) const override;
    //@}

    //! \name Update
    //@{
    //! This samples the bucket statistics, and any state needed
    //! by computeProbablity, in the time interval [\p startTime,
    //! \p endTime], but does not update the model. This is needed
    //! by the results preview.
    //!
    //! \param[in] startTime The start of the time interval to sample.
    //! \param[in] endTime The end of the time interval to sample.
    void sampleBucketStatistics(core_t::TTime startTime,
                                core_t::TTime endTime,
                                CResourceMonitor& resourceMonitor) override;

    //! Update the model with the samples of the various processes
    //! in the time interval [\p startTime, \p endTime].
    //!
    //! \param[in] startTime The start of the time interval to sample.
    //! \param[in] endTime The end of the time interval to sample.
    //! \param[in] resourceMonitor The resourceMonitor.
    void sample(core_t::TTime startTime, core_t::TTime endTime, CResourceMonitor& resourceMonitor) override;

    //! Prune any data for people and attributes which haven't been
    //! seen for a sufficiently long period. This is based on the
    //! prior decay rates and the number of batches into which we
    //! are partitioning time.
    void prune(std::size_t maximumAge) override;
    //@}

    //! \name Probability
    //@{
    //! Compute the probability of seeing \p person's attribute values
    //! for the buckets in the interval [\p startTime, \p endTime].
    //!
    //! \param[in] pid The identifier of the person of interest.
    //! \param[in] startTime The start of the interval of interest.
    //! \param[in] endTime The end of the interval of interest.
    //! \param[in] partitioningFields The partitioning field (name, value)
    //! pairs for which to compute the the probability.
    //! \param[in] numberAttributeProbabilities The maximum number of
    //! attribute probabilities to retrieve.
    //! \param[out] result A structure containing the probability,
    //! the smallest \p numberAttributeProbabilities attribute
    //! probabilities, the influences and any extra descriptive data
    bool computeProbability(std::size_t pid,
                            core_t::TTime startTime,
                            core_t::TTime endTime,
                            CPartitioningFields& partitioningFields,
                            std::size_t numberAttributeProbabilities,
                            SAnnotatedProbability& result) const override;

    //! Clears \p probability and \p attributeProbabilities.
    bool computeTotalProbability(const std::string& person,
                                 std::size_t numberAttributeProbabilities,
                                 TOptionalDouble& probability,
                                 TAttributeProbability1Vec& attributeProbabilities) const override;
    //@}

    //! Get the checksum of this model.
    //!
    //! \param[in] includeCurrentBucketStats If true then include
    //! the current bucket statistics. (This is designed to handle
    //! serialization, for which we don't serialize the current
    //! bucket statistics.)
    uint64_t checksum(bool includeCurrentBucketStats = true) const override;

    //! Get a view of the internals of the model for visualization.
    TModelDetailsViewUPtr details() const override;

    //! Get the feature data corresponding to \p feature at \p time.
    const TSizeSizePrFeatureDataPrVec& featureData(model_t::EFeature feature,
                                                   core_t::TTime time) const;

    //! Debug the memory used by this model.
    void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const override;

    //! Get the memory used by this model.
    std::size_t memoryUsage() const override;

    //! Get the static size of this object - used for virtual hierarchies
    std::size_t staticSize() const override;

    //! Get the non-estimated memory used by this model.
    std::size_t computeMemoryUsage() const override;

private:
    //! Initialize the feature models.
    void initialize(const TFeatureMathsModelSPtrPrVec& newFeatureModels,
                    const TFeatureMultivariatePriorSPtrPrVec& newFeatureCorrelateModelPriors,
                    TFeatureCorrelationsPtrPrVec&& featureCorrelatesModels);

    //! Get the start time of the current bucket.
    core_t::TTime currentBucketStartTime() const override;

    //! Set the start time of the current bucket.
    void currentBucketStartTime(core_t::TTime time) override;

    //! Get the current bucket person counts.
    const TSizeUInt64PrVec& personCounts() const override;

    //! Get the interim corrections of the current bucket.
    TCorrectionKeyDouble1VecUMap& currentBucketInterimCorrections() const;

    //! Initialize the time series models for "n" newly observed people
    //! and "m" attributes.
    void createNewModels(std::size_t n, std::size_t m) override;

    //! Initialize the time series models for recycled attributes and/or people
    void updateRecycledModels() override;

    //! Update the correlation models.
    void refreshCorrelationModels(std::size_t resourceLimit,
                                  CResourceMonitor& resourceMonitor) override;

    //! Clear out large state objects for people/attributes that are pruned
    void clearPrunedResources(const TSizeVec& people, const TSizeVec& attributes) override;

    //! Get the object which calculates corrections for interim buckets.
    const CInterimBucketCorrector& interimValueCorrector() const override;

    //! Skip sampling the interval \p endTime - \p startTime.
    void doSkipSampling(core_t::TTime startTime, core_t::TTime endTime) override;

    //! Get a read only model for \p feature and the attribute identified
    //! by \p cid.
    const maths::CModel* model(model_t::EFeature feature, std::size_t cid) const;

    //! Get a writable model for \p feature and the attribute identified
    //! by \p cid.
    maths::CModel* model(model_t::EFeature feature, std::size_t pid);

    //! Check if there are correlates for \p feature and the person and
    //! attribute identified by \p pid and \p cid, respectively.
    bool correlates(model_t::EFeature feature, std::size_t pid, std::size_t cid, core_t::TTime time) const;

    //! Fill in the probability calculation parameters for \p feature and
    //! person and attribute identified by \p pid and \p cid, respectively.
    bool fill(model_t::EFeature feature,
              std::size_t pid,
              std::size_t cid,
              core_t::TTime bucketTime,
              bool interim,
              CProbabilityAndInfluenceCalculator::SParams& params) const;

    //! Get the model memory usage estimator
    CMemoryUsageEstimator* memoryUsageEstimator() const override;

private:
    //! The statistics we maintain about the bucket.
    SBucketStats m_CurrentBucketStats;

    //! The models of all the attribute correlates for each feature.
    //!
    //! IMPORTANT this must come before m_FeatureModels in the class declaration
    //! so its destructor is called afterwards (12.6.2) because feature models
    //! unregister themselves from correlation models.
    TFeatureCorrelateModelsVec m_FeatureCorrelatesModels;

    //! The population attribute models for each feature.
    TFeatureModelsVec m_FeatureModels;

    //! Calculates corrections for interim buckets.
    TInterimBucketCorrectorCPtr m_InterimBucketCorrector;

    //! A cache of the probability calculation results.
    mutable TProbabilityCache m_Probabilities;

    //! The memory estimator.
    mutable CMemoryUsageEstimator m_MemoryEstimator;

    friend class CMetricPopulationModelDetailsView;
};
}
}

#endif // INCLUDED_ml_model_CMetricPopulationModel_h
