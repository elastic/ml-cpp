/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */

#ifndef INCLUDED_ml_model_CIndividualModel_h
#define INCLUDED_ml_model_CIndividualModel_h

#include <core/CMemory.h>
#include <core/CStatistics.h>
#include <core/CTriple.h>
#include <core/CoreTypes.h>

#include <model/CAnomalyDetectorModel.h>
#include <model/CMemoryUsageEstimator.h>
#include <model/ImportExport.h>

#include <boost/unordered_set.hpp>

#include <cstddef>
#include <utility>
#include <vector>

#include <stdint.h>

namespace ml {
namespace model {
class CAnnotatedProbabilityBuilder;
class CProbabilityAndInfluenceCalculator;

//! \brief The most basic individual model interface.
//!
//! DESCRIPTION:\n
//! This implements or stubs out the common portion of the CAnomalyDetectorModel
//! interface for all individual models. It holds the maths:: and
//! objects which are used to describe the R.V.s that we use to
//! model individual time series' features.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This gathers up the implementation which can be shared by all event
//! rate and metric individual times series models to avoid unnecessary
//! code duplication.
//!
//! It assumes data are supplied in time order since this means minimal
//! state can be maintained.
class MODEL_EXPORT CIndividualModel : public CAnomalyDetectorModel {
public:
    using TSizeTimeUMap = boost::unordered_map<std::size_t, core_t::TTime>;
    using TTimeVec = std::vector<core_t::TTime>;
    using TSizeUInt64Pr = std::pair<std::size_t, uint64_t>;
    using TSizeUInt64PrVec = std::vector<TSizeUInt64Pr>;
    using TFeatureSizeSizeTriple = core::CTriple<model_t::EFeature, std::size_t, std::size_t>;
    using TFeatureSizeSizeTripleDouble1VecUMap = boost::unordered_map<TFeatureSizeSizeTriple, TDouble1Vec>;
    using TFeatureMathsModelPtrPr = std::pair<model_t::EFeature, TMathsModelPtr>;
    using TFeatureMathsModelPtrPrVec = std::vector<TFeatureMathsModelPtrPr>;
    using TFeatureMathsModelPtrVecPr = std::pair<model_t::EFeature, TMathsModelPtrVec>;
    using TFeatureMathsModelPtrVecPrVec = std::vector<TFeatureMathsModelPtrVecPr>;
    using TFeatureCorrelationsPtrPr = std::pair<model_t::EFeature, TCorrelationsPtr>;
    using TFeatureCorrelationsPtrPrVec = std::vector<TFeatureCorrelationsPtrPr>;

public:
    //! \name Life-cycle
    //@{
    //! \param[in] params The global configuration parameters.
    //! \param[in] dataGatherer The object to gather time series data.
    //! \param[in] newFeatureModels The new models to use for each feature.
    //! \param[in] newFeatureCorrelateModelPriors The prior to use for the
    //! new model of correlates for each feature.
    //! \param[in] featureCorrelatesModels The model of all correlates for
    //! each feature.
    //! \param[in] influenceCalculators The influence calculators to use
    //! for each feature.
    //! \note The current bucket statistics are left default initialized
    //! and so must be sampled for before this model can be used.
    CIndividualModel(const SModelParams& params,
                     const TDataGathererPtr& dataGatherer,
                     const TFeatureMathsModelPtrPrVec& newFeatureModels,
                     const TFeatureMultivariatePriorPtrPrVec& newFeatureCorrelateModelPriors,
                     const TFeatureCorrelationsPtrPrVec& featureCorrelatesModels,
                     const TFeatureInfluenceCalculatorCPtrPrVecVec& influenceCalculators);

    //! Create a copy that will result in the same persisted state as the
    //! original.  This is effectively a copy constructor that creates a
    //! copy that's only valid for a single purpose.  The boolean flag is
    //! redundant except to create a signature that will not be mistaken
    //! for a general purpose copy constructor.
    CIndividualModel(bool isForPersistence, const CIndividualModel& other);
    //@}

    //! Returns false.
    virtual bool isPopulation() const;

    //! \name Bucket Statistics
    //@{
    //! Get the count in the bucketing interval containing \p time
    //! for the person identified by \p pid.
    //!
    //! \param[in] pid The person of interest.
    //! \param[in] time The time of interest.
    virtual TOptionalUInt64 currentBucketCount(std::size_t pid, core_t::TTime time) const;

    //! Check if bucket statistics are available for the specified time.
    virtual bool bucketStatsAvailable(core_t::TTime time) const;
    //@}

    //! \name Update
    //@{
    //! Sample any state needed by computeProbablity in the time
    //! interval [\p startTime, \p endTime] but do not update the
    //! model. This is needed by the results preview.
    //!
    //! \param[in] startTime The start of the time interval to sample.
    //! \param[in] endTime The end of the time interval to sample.
    virtual void sampleBucketStatistics(core_t::TTime startTime, core_t::TTime endTime, CResourceMonitor& resourceMonitor) = 0;

    //! Sample any state needed by computeProbablity for the out-
    //! of-phase bucket in the time interval [\p startTime, \p endTime]
    //! but do not update the model.
    //!
    //! \param[in] startTime The start of the time interval to sample.
    //! \param[in] endTime The end of the time interval to sample.
    virtual void sampleOutOfPhase(core_t::TTime startTime, core_t::TTime endTime, CResourceMonitor& resourceMonitor);

    //! Update the model with features samples from the time interval
    //! [\p startTime, \p endTime].
    //!
    //! \param[in] startTime The start of the time interval to sample.
    //! \param[in] endTime The end of the time interval to sample.
    //! \param[in] resourceMonitor The resourceMonitor.
    virtual void sample(core_t::TTime startTime, core_t::TTime endTime, CResourceMonitor& resourceMonitor) = 0;

    //! Prune any person models which haven't been updated for a
    //! specified period.
    virtual void prune(std::size_t maximumAge);
    //@}

    //! \name Probability
    //@{
    //! Clears \p probability and \p attributeProbabilities.
    virtual bool computeTotalProbability(const std::string& person,
                                         std::size_t numberAttributeProbabilities,
                                         TOptionalDouble& probability,
                                         TAttributeProbability1Vec& attributeProbabilities) const;
    //@}

    //! Get the checksum of this model.
    //!
    //! \param[in] includeCurrentBucketStats If true then include
    //! the current bucket statistics. (This is designed to handle
    //! serialization, for which we don't serialize the current
    //! bucket statistics.)
    virtual uint64_t checksum(bool includeCurrentBucketStats = true) const = 0;

    //! Debug the memory used by this model.
    virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const = 0;

    //! Get the memory used by this model.
    virtual std::size_t memoryUsage() const = 0;

    //! Get the static size of this object - used for virtual hierarchies.
    virtual std::size_t staticSize() const = 0;

    //! Get the non-estimated value of the the memory used by this model.
    virtual std::size_t computeMemoryUsage() const = 0;

protected:
    using TStrCRefDouble1VecDouble1VecPrPr = std::pair<TStrCRef, TDouble1VecDouble1VecPr>;
    using TStrCRefDouble1VecDouble1VecPrPrVec = std::vector<TStrCRefDouble1VecDouble1VecPrPr>;
    using TStrCRefDouble1VecDouble1VecPrPrVecVec = std::vector<TStrCRefDouble1VecDouble1VecPrPrVec>;
    using TStrCRefDouble1VecDouble1VecPrPrVecVecVec = std::vector<TStrCRefDouble1VecDouble1VecPrPrVecVec>;

protected:
    //! Persist state by passing information to the supplied inserter.
    void doAcceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Restore the model reading state from the supplied traverser.
    bool doAcceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Get the start time of the current bucket.
    virtual core_t::TTime currentBucketStartTime() const = 0;

    //! Set the start time of the current bucket.
    virtual void currentBucketStartTime(core_t::TTime time) = 0;

    //! Monitor the resource usage while creating new models.
    void createUpdateNewModels(core_t::TTime time, CResourceMonitor& resourceMonitor);

    //! Create the time series models for "n" newly observed people.
    virtual void createNewModels(std::size_t n, std::size_t m) = 0;

    //! Reinitialize the time series models for recycled people.
    virtual void updateRecycledModels() = 0;

    //! Update the correlation models.
    void refreshCorrelationModels(std::size_t resourceLimit, CResourceMonitor& resourceMonitor);

    //! Clear out large state objects for people that are pruned.
    virtual void clearPrunedResources(const TSizeVec& people, const TSizeVec& attributes) = 0;

    //! Get the person unique identifiers which have a feature value
    //! in the bucketing time interval including \p time.
    template<typename T>
    void currentBucketPersonIds(core_t::TTime time, const T& featureData, TSizeVec& result) const;

    //! Get the value of the \p feature of the person identified
    //! by \p pid for the bucketing interval containing \p time.
    template<typename T>
    const T* featureData(model_t::EFeature feature,
                         std::size_t pid,
                         core_t::TTime time,
                         const std::vector<std::pair<model_t::EFeature, std::vector<std::pair<std::size_t, T>>>>& featureData) const;

    //! Sample the bucket statistics and write the results in to
    //! \p featureData.
    template<typename T, typename FILTER>
    void sampleBucketStatistics(core_t::TTime startTime,
                                core_t::TTime endTime,
                                const FILTER& filter,
                                std::vector<std::pair<model_t::EFeature, T>>& featureData,
                                CResourceMonitor& resourceMonitor);

    //! Add the probability and influences for \p feature and \p pid.
    template<typename PARAMS, typename INFLUENCES>
    bool addProbabilityAndInfluences(std::size_t pid,
                                     PARAMS& params,
                                     const INFLUENCES& influences,
                                     CProbabilityAndInfluenceCalculator& pJoint,
                                     CAnnotatedProbabilityBuilder& builder) const;

    //! Get the weight associated with an update to the prior from an empty bucket
    //! for features which count empty buckets.
    double emptyBucketWeight(model_t::EFeature feature, std::size_t pid, core_t::TTime time) const;

    //! Get the "probability the bucket is empty" to use to correct probabilities
    //! for features which count empty buckets.
    double probabilityBucketEmpty(model_t::EFeature feature, std::size_t pid) const;

    //! Get a read only model corresponding to \p feature of the person \p pid.
    const maths::CModel* model(model_t::EFeature feature, std::size_t pid) const;

    //! Get a writable model corresponding to \p feature of the person \p pid.
    maths::CModel* model(model_t::EFeature feature, std::size_t pid);

    //! Sample the correlate models.
    void sampleCorrelateModels(const maths_t::TWeightStyleVec& weightStyles);

    //! Correct \p baseline with \p corrections for interim results.
    void correctBaselineForInterim(model_t::EFeature feature,
                                   std::size_t pid,
                                   model_t::CResultType type,
                                   const TSizeDoublePr1Vec& correlated,
                                   const TFeatureSizeSizeTripleDouble1VecUMap& corrections,
                                   TDouble1Vec& baseline) const;

    //! Get the first time each person was seen.
    const TTimeVec& firstBucketTimes() const;

    //! Get the last time each persion was seen
    const TTimeVec& lastBucketTimes() const;

    //! Get the amount by which to derate the initial decay rate
    //! and the minimum Winsorisation weight for \p pid at \p time.
    double derate(std::size_t pid, core_t::TTime time) const;

    //! Print the current bucketing interval.
    std::string printCurrentBucket() const;

private:
    //! Get the person counts in the current bucket.
    virtual const TSizeUInt64PrVec& currentBucketPersonCounts() const = 0;

    //! Get writable person counts in the current bucket.
    virtual TSizeUInt64PrVec& currentBucketPersonCounts() = 0;

    //! Get the total number of correlation models.
    std::size_t numberCorrelations() const;

    //! Returns one.
    virtual double attributeFrequency(std::size_t cid) const;

    //! Perform derived class specific operations to accomplish skipping sampling
    virtual void doSkipSampling(core_t::TTime startTime, core_t::TTime endTime);

    //! Get the model memory usage estimator
    virtual CMemoryUsageEstimator* memoryUsageEstimator() const;

private:
    //! The time that each person was first seen.
    TTimeVec m_FirstBucketTimes;

    //! The last time that each person was seen.
    TTimeVec m_LastBucketTimes;

    //! The models of all the correlates for each feature.
    //!
    //! IMPORTANT this must come before m_FeatureModels in the class declaration
    //! so its destructor is called afterwards (12.6.2) because feature models
    //! unregister themselves from correlation models.
    TFeatureCorrelateModelsVec m_FeatureCorrelatesModels;

    //! The individual person models for each feature.
    TFeatureModelsVec m_FeatureModels;

    //! The memory estimator.
    mutable CMemoryUsageEstimator m_MemoryEstimator;
};
}
}

#endif // INCLUDED_ml_model_CIndividualModel_h
