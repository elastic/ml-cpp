/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CCountingModel_h
#define INCLUDED_ml_model_CCountingModel_h

#include <model/CAnomalyDetectorModel.h>

#include <maths/CBasicStatistics.h>

#include <boost/unordered_map.hpp>

#include <memory>

class CCountingModelTest;

namespace ml {
namespace model {

//! \brief A very simple model for counting events in the sampled bucket.
//!
//! DESCRIPTION:\n
//! This performs no analysis and is used by the simple count detector
//! to get the sampling bucket event rate.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This is lightweight as possible and in particular avoids creating
//! any time series models. This avoids a potential pathology where
//! memory limiting can cause us to stop getting counts and also makes
//! interpreting the maths library logging easier.
class MODEL_EXPORT CCountingModel : public CAnomalyDetectorModel {
public:
    using TInterimBucketCorrectorPtr = std::shared_ptr<CInterimBucketCorrector>;

public:
    //! \name Life-cycle.
    //@{
    //! \param[in] params The global configuration parameters.
    //! \param[in] dataGatherer The object that gathers time series data.
    //! \param[in] interimBucketCorrector Calculates corrections for interim
    //! buckets.
    CCountingModel(const SModelParams& params,
                   const TDataGathererPtr& dataGatherer,
                   const TInterimBucketCorrectorPtr& interimBucketCorrector);

    //! Constructor used for restoring persisted models.
    //!
    //! \note The current bucket statistics are left default initialized
    //! and so must be sampled for before this model can be used.
    CCountingModel(const SModelParams& params,
                   const TDataGathererPtr& dataGatherer,
                   const TInterimBucketCorrectorPtr& interimBucketCorrector,
                   core::CStateRestoreTraverser& traverser);

    //! Create a copy that will result in the same persisted state as the
    //! original.  This is effectively a copy constructor that creates a
    //! copy that's only valid for a single purpose.  The boolean flag is
    //! redundant except to create a signature that will not be mistaken for
    //! a general purpose copy constructor.
    CCountingModel(bool isForPersistence, const CCountingModel& other);
    //@}

    //! Returns event rate online.
    model_t::EModelType category() const override;

    //! Returns false.
    bool isPopulation() const override;

    //! Returns false.
    bool isEventRate() const override;

    //! Returns false.
    bool isMetric() const override;

    //! \name Persistence
    //@{
    //! Persist the state of the models only.
    void persistModelsState(core::CStatePersistInserter& /*inserter*/) const override {
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

    //! \name Bucket Statistics
    //!@{
    //! Get the count of the bucketing interval containing \p time
    //! for the person identified by \p pid.
    //!
    //! \param[in] pid The identifier of the person of interest.
    //! \param[in] time The time of interest.
    //! \return The count in the bucketing interval at \p time for the
    //! person identified by \p pid if available and null otherwise.
    TOptionalUInt64 currentBucketCount(std::size_t pid, core_t::TTime time) const override;

    //! Get the mean bucket count or the reference model mean bucket
    //! count if one is defined for the person identified by \p pid.
    //!
    //! \param[in] pid The identifier of the person of interest.
    TOptionalDouble baselineBucketCount(std::size_t pid) const override;

    //! Get the count of the bucketing interval containing \p time
    //! for the person identified by \p pid.
    //!
    //! \param[in] feature Ignored.
    //! \param[in] pid The identifier of the person of interest.
    //! \param[in] cid Ignored.
    //! \param[in] time The time of interest.
    TDouble1Vec currentBucketValue(model_t::EFeature feature,
                                   std::size_t pid,
                                   std::size_t cid,
                                   core_t::TTime time) const override;

    //! Get the mean bucket count or the reference model mean bucket
    //! count if one is defined for the person identified by \p pid.
    //!
    //! \param[in] feature Ignored.
    //! \param[in] pid The identifier of the person of interest.
    //! \param[in] cid Ignored.
    //! \param[in] type Ignored.
    //! \param[in] correlated Ignored.
    //! \param[in] time The time of interest.
    TDouble1Vec baselineBucketMean(model_t::EFeature feature,
                                   std::size_t pid,
                                   std::size_t cid,
                                   model_t::CResultType type,
                                   const TSizeDoublePr1Vec& correlated,
                                   core_t::TTime time) const override;
    //@}

    //! \name Person
    //@{
    //! Get the person unique identifiers which have a feature value
    //! in the bucketing time interval including \p time.
    //!
    //! \param[in] time The time of interest.
    //! \param[out] result Filled in with the person identifiers
    //! in the bucketing time interval of interest.
    void currentBucketPersonIds(core_t::TTime time, TSizeVec& result) const override;
    //@}

    //! \name Update
    //@{
    //! This samples the bucket statistics, in the time interval
    //! [\p startTime, \p endTime]. This is needed by the results
    //! preview.
    //!
    //! \param[in] startTime The start of the time interval to sample.
    //! \param[in] endTime The end of the time interval to sample.
    void sampleBucketStatistics(core_t::TTime startTime,
                                core_t::TTime endTime,
                                CResourceMonitor& resourceMonitor) override;

    //! This samples the bucket statistics, in the time interval
    //! [\p startTime, \p endTime].
    //!
    //! \param[in] startTime The start of the time interval to sample.
    //! \param[in] endTime The end of the time interval to sample.
    //! \param[in] resourceMonitor The resourceMonitor.
    void sample(core_t::TTime startTime, core_t::TTime endTime, CResourceMonitor& resourceMonitor) override;

    //! No-op.
    void prune(std::size_t maximumAge) override;
    //@}

    //! \name Probability
    //@{
    //! Sets \p probability to 1.
    bool computeProbability(std::size_t pid,
                            core_t::TTime startTime,
                            core_t::TTime endTime,
                            CPartitioningFields& partitioningFields,
                            std::size_t numberAttributeProbabilities,
                            SAnnotatedProbability& result) const override;

    //! Sets \p probability to 1.
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

    //! Get the memory used by this model
    void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const override;

    //! Get the memory used by this model
    std::size_t memoryUsage() const override;

    //! Get the static size of this object - used for virtual hierarchies
    std::size_t staticSize() const override;

    //! Returns null.
    CModelDetailsViewPtr details() const override;

    //! Get the descriptions of any occurring scheduled event descriptions for the bucket time
    const TStr1Vec& scheduledEventDescriptions(core_t::TTime time) const override;

protected:
    //! Get the start time of the current bucket.
    core_t::TTime currentBucketStartTime() const override;

    //! Set the start time of the current bucket.
    void currentBucketStartTime(core_t::TTime time) override;

    //! Get the non-estimated value of the the memory used by this model.
    std::size_t computeMemoryUsage() const override;

private:
    //! Get the scheduled events that match at sampleTime.
    SModelParams::TStrDetectionRulePrVec checkScheduledEvents(core_t::TTime sampleTime) const;

    //! Check for scheduled events and append the descriptions of
    //! matched events to the scheduled event descriptions.
    void setMatchedEventsDescriptions(core_t::TTime sampleTime, core_t::TTime bucketStartTime);

    //! Returns one.
    double attributeFrequency(std::size_t cid) const override;

    //! Monitor the resource usage while creating new models.
    void createUpdateNewModels(core_t::TTime, CResourceMonitor& resourceMonitor);

    //! Create the mean counts for "n" newly observed people.
    void createNewModels(std::size_t n, std::size_t m) override;

    //! Update start time and counts for the given bucket.
    void updateCurrentBucketsStats(core_t::TTime time);

    //! Reinitialize the time series models for recycled people.
    void updateRecycledModels() override;

    //! Initialize the time series models for newly observed people.
    void clearPrunedResources(const TSizeVec& people, const TSizeVec& attributes) override;

    //! Get the object which calculates corrections for interim buckets.
    const CInterimBucketCorrector& interimValueCorrector() const override;

    //! Check if bucket statistics are available for the specified time.
    bool bucketStatsAvailable(core_t::TTime time) const override;

    //! Print the current bucketing interval.
    std::string printCurrentBucket() const;

    //! Perform derived class specific operations to accomplish skipping sampling
    void doSkipSampling(core_t::TTime startTime, core_t::TTime endTime) override;

    //! Get the model memory usage estimator
    CMemoryUsageEstimator* memoryUsageEstimator() const override;

private:
    using TSizeUInt64Pr = std::pair<std::size_t, uint64_t>;
    using TSizeUInt64PrVec = std::vector<TSizeUInt64Pr>;
    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;
    using TTimeStr1VecUMap = boost::unordered_map<core_t::TTime, TStr1Vec>;

private:
    //! The start time of the last sampled bucket.
    core_t::TTime m_StartTime;

    //! The current bucket counts.
    TSizeUInt64PrVec m_Counts;

    //! The baseline bucket counts.
    TMeanAccumulatorVec m_MeanCounts;

    //! Map of matched scheduled event descriptions by bucket time.
    TTimeStr1VecUMap m_ScheduledEventDescriptions;

    //! Calculates corrections for interim buckets.
    TInterimBucketCorrectorPtr m_InterimBucketCorrector;

    friend class ::CCountingModelTest;
};
}
}

#endif // INCLUDED_ml_model_CCountingModel_h
