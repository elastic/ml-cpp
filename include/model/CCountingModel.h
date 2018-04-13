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

class CCountingModelTest;

namespace ml
{
namespace model
{

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
class MODEL_EXPORT CCountingModel : public CAnomalyDetectorModel
{
    public:
        //! \name Life-cycle.
        //@{
        //! \param[in] params The global configuration parameters.
        //! \param[in] dataGatherer The object that gathers time series data.
        CCountingModel(const SModelParams &params,
                       const TDataGathererPtr &dataGatherer);

        //! Constructor used for restoring persisted models.
        //!
        //! \note The current bucket statistics are left default initialized
        //! and so must be sampled for before this model can be used.
        CCountingModel(const SModelParams &params,
                       const TDataGathererPtr &dataGatherer,
                       core::CStateRestoreTraverser &traverser);

        //! Create a copy that will result in the same persisted state as the
        //! original.  This is effectively a copy constructor that creates a
        //! copy that's only valid for a single purpose.  The boolean flag is
        //! redundant except to create a signature that will not be mistaken for
        //! a general purpose copy constructor.
        CCountingModel(bool isForPersistence, const CCountingModel &other);
        //@}

        //! Returns event rate online.
        virtual model_t::EModelType category() const;

        //! Returns false.
        virtual bool isPopulation() const;

        //! Returns false.
        virtual bool isEventRate() const;

        //! Returns false.
        virtual bool isMetric() const;

        //! \name Persistence
        //@{
        //! Persist state by passing information to the supplied inserter
        virtual void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

        //! Add to the contents of the object.
        virtual bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

        //! Create a clone of this model that will result in the same persisted
        //! state.  The clone may be incomplete in ways that do not affect the
        //! persisted representation, and must not be used for any other
        //! purpose.
        //! \warning The caller owns the object returned.
        virtual CAnomalyDetectorModel *cloneForPersistence() const;
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
        virtual TOptionalUInt64 currentBucketCount(std::size_t pid,
                                                   core_t::TTime time) const;

        //! Get the mean bucket count or the reference model mean bucket
        //! count if one is defined for the person identified by \p pid.
        //!
        //! \param[in] pid The identifier of the person of interest.
        virtual TOptionalDouble baselineBucketCount(std::size_t pid) const;

        //! Get the count of the bucketing interval containing \p time
        //! for the person identified by \p pid.
        //!
        //! \param[in] feature Ignored.
        //! \param[in] pid The identifier of the person of interest.
        //! \param[in] cid Ignored.
        //! \param[in] time The time of interest.
        virtual TDouble1Vec currentBucketValue(model_t::EFeature feature,
                                               std::size_t pid,
                                               std::size_t cid,
                                               core_t::TTime time) const;

        //! Get the mean bucket count or the reference model mean bucket
        //! count if one is defined for the person identified by \p pid.
        //!
        //! \param[in] feature Ignored.
        //! \param[in] pid The identifier of the person of interest.
        //! \param[in] cid Ignored.
        //! \param[in] type Ignored.
        //! \param[in] correlated Ignored.
        //! \param[in] time The time of interest.
        virtual TDouble1Vec baselineBucketMean(model_t::EFeature feature,
                                               std::size_t pid,
                                               std::size_t cid,
                                               model_t::CResultType type,
                                               const TSizeDoublePr1Vec &correlated,
                                               core_t::TTime time) const;
        //@}

        //! \name Person
        //@{
        //! Get the person unique identifiers which have a feature value
        //! in the bucketing time interval including \p time.
        //!
        //! \param[in] time The time of interest.
        //! \param[out] result Filled in with the person identifiers
        //! in the bucketing time interval of interest.
        virtual void currentBucketPersonIds(core_t::TTime time,
                                            TSizeVec &result) const;
        //@}

        //! \name Update
        //@{
        //! This samples the bucket statistics, in the time interval
        //! [\p startTime, \p endTime]. This is needed by the results
        //! preview.
        //!
        //! \param[in] startTime The start of the time interval to sample.
        //! \param[in] endTime The end of the time interval to sample.
        virtual void sampleBucketStatistics(core_t::TTime startTime,
                                            core_t::TTime endTime,
                                            CResourceMonitor &resourceMonitor);

        //! This samples the bucket statistics, and any state needed
        //! by computeProbablity, in the time interval [\p startTime,
        //! \p endTime], but does not update the model. This is needed
        //! by the results preview.
        //!
        //! \param[in] startTime The start of the time interval to sample.
        //! \param[in] endTime The end of the time interval to sample.
        virtual void sampleOutOfPhase(core_t::TTime startTime,
                                      core_t::TTime endTime,
                                      CResourceMonitor &resourceMonitor);

        //! This samples the bucket statistics, in the time interval
        //! [\p startTime, \p endTime].
        //!
        //! \param[in] startTime The start of the time interval to sample.
        //! \param[in] endTime The end of the time interval to sample.
        //! \param[in] resourceMonitor The resourceMonitor.
        virtual void sample(core_t::TTime startTime,
                            core_t::TTime endTime,
                            CResourceMonitor &resourceMonitor);

        //! No-op.
        virtual void prune(std::size_t maximumAge);
        //@}

        //! \name Probability
        //@{
        //! Sets \p probability to 1.
        virtual bool computeProbability(std::size_t pid,
                                        core_t::TTime startTime,
                                        core_t::TTime endTime,
                                        CPartitioningFields &partitioningFields,
                                        std::size_t numberAttributeProbabilities,
                                        SAnnotatedProbability &result) const;

        //! Sets \p probability to 1.
        virtual bool computeTotalProbability(const std::string &person,
                                             std::size_t numberAttributeProbabilities,
                                             TOptionalDouble &probability,
                                             TAttributeProbability1Vec &attributeProbabilities) const;
        //@}

        //! Get the checksum of this model.
        //!
        //! \param[in] includeCurrentBucketStats If true then include
        //! the current bucket statistics. (This is designed to handle
        //! serialization, for which we don't serialize the current
        //! bucket statistics.)
        virtual uint64_t checksum(bool includeCurrentBucketStats = true) const;

        //! Get the memory used by this model
        virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the memory used by this model
        virtual std::size_t memoryUsage() const;

        //! Get the static size of this object - used for virtual hierarchies
        virtual std::size_t staticSize() const;

        //! Returns null.
        virtual CModelDetailsViewPtr details() const;

        //! Get the descriptions of any occurring scheduled event descriptions for the bucket time
        virtual const TStr1Vec &scheduledEventDescriptions(core_t::TTime time) const;

    public:
        using TSizeUInt64Pr = std::pair<std::size_t, uint64_t>;
        using TSizeUInt64PrVec = std::vector<TSizeUInt64Pr>;
        using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
        using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;

    protected:
        //! Get the start time of the current bucket.
        virtual core_t::TTime currentBucketStartTime() const;

        //! Set the start time of the current bucket.
        virtual void currentBucketStartTime(core_t::TTime time);

        //! Get the non-estimated value of the the memory used by this model.
        virtual std::size_t computeMemoryUsage() const;

    private:
        //! Get the scheduled events that match at sampleTime.
        SModelParams::TStrDetectionRulePrVec
        checkScheduledEvents(core_t::TTime sampleTime) const;

        //! Check for scheduled events and append the descriptions of
        //! matched events to the scheduled event descriptions.
        void setMatchedEventsDescriptions(core_t::TTime sampleTime, core_t::TTime bucketStartTime);

        //! Returns one.
        virtual double attributeFrequency(std::size_t cid) const;

        //! Monitor the resource usage while creating new models.
        void createUpdateNewModels(core_t::TTime,
                                   CResourceMonitor &resourceMonitor);

        //! Create the mean counts for "n" newly observed people.
        virtual void createNewModels(std::size_t n, std::size_t m);

        //! Update start time and counts for the given bucket.
        void updateCurrentBucketsStats(core_t::TTime time);

        //! Reinitialize the time series models for recycled people.
        virtual void updateRecycledModels();

        //! Initialize the time series models for newly observed people.
        virtual void clearPrunedResources(const TSizeVec &people,
                                          const TSizeVec &attributes);

        //! Check if bucket statistics are available for the specified time.
        bool bucketStatsAvailable(core_t::TTime time) const;

        //! Print the current bucketing interval.
        std::string printCurrentBucket() const;

        //! Set the current bucket total count.
        virtual void currentBucketTotalCount(uint64_t totalCount);

        //! Perform derived class specific operations to accomplish skipping sampling
        virtual void doSkipSampling(core_t::TTime startTime, core_t::TTime endTime);

        //! Get the model memory usage estimator
        virtual CMemoryUsageEstimator *memoryUsageEstimator() const;

    private:
        using TTimeStr1VecUMap = boost::unordered_map<core_t::TTime, TStr1Vec>;

    private:
        //! The start time of the last sampled bucket.
        core_t::TTime m_StartTime;

        //! The current bucket counts.
        TSizeUInt64PrVec m_Counts;

        //! The baseline bucket counts.
        TMeanAccumulatorVec m_MeanCounts;

        //! Map of matched scheduled event descriptions by bucket time
        TTimeStr1VecUMap m_ScheduledEventDescriptions;

    friend class ::CCountingModelTest;
};
}
}

#endif // INCLUDED_ml_model_CModel_h
