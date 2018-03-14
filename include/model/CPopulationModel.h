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

#ifndef INCLUDED_ml_model_CPopulationModel_h
#define INCLUDED_ml_model_CPopulationModel_h

#include <core/CMemory.h>
#include <core/CNonCopyable.h>
#include <core/CoreTypes.h>
#include <core/CStatistics.h>
#include <core/CTriple.h>

#include <maths/CBjkstUniqueValues.h>
#include <maths/CCountMinSketch.h>
#include <maths/CMultivariatePrior.h>
#include <maths/COrderings.h>

#include <model/CAnomalyDetectorModel.h>
#include <model/CFeatureData.h>
#include <model/ImportExport.h>
#include <model/ModelTypes.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {
class CPrior;
}
namespace model {

//! \brief The most basic population model interface.
//!
//! DESCRIPTION:\n
//! This defines the interface common to all probabilistic models of the
//! (random) processes which describe person and population's state. It
//! declares core functions used by the anomaly detection code to:
//!   -# Sample the processes in a specified time interval and update
//!      the models.
//!   -# Compute the probability of a person's processes in a specified
//!      time interval.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The population model hierarchy has been abstracted to gather up the
//! implementation which can be shared by the event rate and metric models
//! to avoid unnecessary code duplication.
//!
//! It assumes data are supplied in time order since this means minimal
//! state can be maintained.
class MODEL_EXPORT CPopulationModel : public CAnomalyDetectorModel {
    public:
        typedef std::vector<core_t::TTime>                       TTimeVec;
        typedef std::pair<std::size_t, uint64_t>                 TSizeUInt64Pr;
        typedef std::vector<TSizeUInt64Pr>                       TSizeUInt64PrVec;
        typedef std::vector<maths::CCountMinSketch>              TCountMinSketchVec;
        typedef std::vector<maths::CBjkstUniqueValues>           TBjkstUniqueValuesVec;
        typedef boost::unordered_map<std::size_t, core_t::TTime> TSizeTimeUMap;

        //! Lift the overloads of baselineBucketMean into the class scope.
        using CAnomalyDetectorModel::baselineBucketMean;

        //! Lift the overloads of acceptPersistInserter into the class scope.
        using CAnomalyDetectorModel::acceptPersistInserter;

    public:
        //! \name Life-cycle.
        //@{
        //! \param[in] params The global configuration parameters.
        //! \param[in] dataGatherer The object that gathers time series data.
        //! \param[in] influenceCalculators The influence calculators to use
        //! for each feature.
        CPopulationModel(const SModelParams &params,
                         const TDataGathererPtr &dataGatherer,
                         const TFeatureInfluenceCalculatorCPtrPrVecVec &influenceCalculators);

        //! Create a copy that will result in the same persisted state as the
        //! original.  This is effectively a copy constructor that creates a
        //! copy that's only valid for a single purpose.  The boolean flag is
        //! redundant except to create a signature that will not be mistaken
        //! for a general purpose copy constructor.
        CPopulationModel(bool isForPersistence, const CPopulationModel &other);
        //@}

        //! Returns true.
        virtual bool isPopulation(void) const;

        //! \name Bucket Statistics
        //@{
        //! Get the count of the bucketing interval containing \p time
        //! for the person identified by \p pid.
        virtual TOptionalUInt64 currentBucketCount(std::size_t pid, core_t::TTime time) const;

        //! Returns null.
        virtual TOptionalDouble baselineBucketCount(std::size_t pid) const;

    protected:
        //! Get the index range [begin, end) of the person corresponding to
        //! \p pid in the vector \p data. This relies on the fact that \p data
        //! is sort lexicographically by person then attribute identifier.
        //! This will return an empty range if the person is not present.
        template<typename T>
        static TSizeSizePr personRange(const T &data, std::size_t pid);

        //! Find the person attribute pair identified by \p pid and \p cid,
        //! respectively, in \p data if it exists. Returns the end of the
        //! vector if it doesn't.
        template<typename T>
        static typename T::const_iterator find(const T &data, std::size_t pid, std::size_t cid);

        //! Extract the bucket value for count feature data.
        static inline TDouble1Vec extractValue(model_t::EFeature /*feature*/,
                                               const std::pair<TSizeSizePr, SEventRateFeatureData> &data);
        //! Extract the bucket value for metric feature data.
        static inline TDouble1Vec extractValue(model_t::EFeature feature,
                                               const std::pair<TSizeSizePr, SMetricFeatureData> &data);
    //@}

    public:
        //! \name Person
        //@{
        //! Get the person unique identifiers which are present in the
        //! bucketing time interval including \p time.
        //!
        //! \param[in] time The time of interest.
        //! \param[out] result Filled in with the person identifiers
        //! in the bucketing time interval of interest.
        virtual void currentBucketPersonIds(core_t::TTime time,
                                            TSizeVec &result) const;
        //@}

        //! \name Update
        //@{
        //! Sample any state needed by computeProbablity for the out-
        //! of-phase bucket in the time interval [\p startTime, \p endTime]
        //! but do not update the model.
        //!
        //! \param[in] startTime The start of the time interval to sample.
        //! \param[in] endTime The end of the time interval to sample.
        virtual void sampleOutOfPhase(core_t::TTime startTime,
                                      core_t::TTime endTime,
                                      CResourceMonitor &resourceMonitor);

        //! Update the rates for \p feature and \p people.
        virtual void sample(core_t::TTime startTime,
                            core_t::TTime endTime,
                            CResourceMonitor &resourceMonitor) = 0;
        //@}

        //! Get the checksum of this model.
        //!
        //! \param[in] includeCurrentBucketStats If true then include the
        //! current bucket statistics. (This is designed to handle serialization,
        //! for which we don't serialize the current bucket statistics.)
        virtual uint64_t checksum(bool includeCurrentBucketStats = true) const = 0;

        //! Debug the memory used by this model.
        virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const = 0;

        //! Get the memory used by this model.
        virtual std::size_t memoryUsage(void) const = 0;

        //! Get the static size of this object - used for virtual hierarchies
        virtual std::size_t staticSize(void) const = 0;

        //! Get the non-estimated value of the the memory used by this model.
        virtual std::size_t computeMemoryUsage(void) const = 0;

        //! Get the frequency of the attribute identified by \p cid.
        virtual double attributeFrequency(std::size_t cid) const;

        //! Get the weight for \p feature and the person identified by
        //! \p pid based on their sample rate.
        double sampleRateWeight(std::size_t pid, std::size_t cid) const;

    protected:
        //! \brief A key for the partial bucket corrections map.
        class MODEL_EXPORT CCorrectionKey {
            public:
                CCorrectionKey(model_t::EFeature feature,
                               std::size_t pid,
                               std::size_t cid,
                               std::size_t correlated = 0);
                bool operator==(const CCorrectionKey &rhs) const;
                std::size_t hash(void) const;

            private:
                model_t::EFeature m_Feature;
                std::size_t       m_Pid;
                std::size_t       m_Cid;
                std::size_t       m_Correlate;
        };

        //! \brief A hasher for the partial bucket corrections map key.
        struct MODEL_EXPORT CHashCorrectionKey {
            std::size_t operator()(const CCorrectionKey &key) const {
                return key.hash();
            }
        };
        using TCorrectionKeyDouble1VecUMap =
            boost::unordered_map<CCorrectionKey, TDouble1Vec, CHashCorrectionKey>;

    protected:
        //! Persist state by passing information to the supplied inserter.
        void doAcceptPersistInserter(core::CStatePersistInserter &inserter) const;

        //! Restore the model reading state from the supplied traverser.
        bool doAcceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

        //! Get the current bucket person counts.
        virtual const TSizeUInt64PrVec &personCounts(void) const = 0;

        //! Check if bucket statistics are available for the specified time.
        virtual bool bucketStatsAvailable(core_t::TTime time) const = 0;

        //! Monitor the resource usage while creating new models
        void createUpdateNewModels(core_t::TTime time, CResourceMonitor &resourceMonitor);

        //! Initialize the time series models for "n" newly observed people
        //! and "m" newly observed attributes.
        virtual void createNewModels(std::size_t n, std::size_t m) = 0;

        //! Initialize the time series models for recycled attributes
        //! and/or people.
        virtual void updateRecycledModels(void) = 0;

        //! Update the correlation models.
        virtual void refreshCorrelationModels(std::size_t resourceLimit,
                                              CResourceMonitor &resourceMonitor) = 0;

        //! Clear out large state objects for people/attributes that are pruned.
        virtual void clearPrunedResources(const TSizeVec &people,
                                          const TSizeVec &attributes) = 0;

        //! Correct \p baseline with \p corrections for interim results.
        void correctBaselineForInterim(model_t::EFeature feature,
                                       std::size_t pid,
                                       std::size_t cid,
                                       model_t::CResultType type,
                                       const TSizeDoublePr1Vec &correlated,
                                       const TCorrectionKeyDouble1VecUMap &corrections,
                                       TDouble1Vec &baseline) const;

        //! Get the time by which to propagate the priors on a sample.
        double propagationTime(std::size_t cid, core_t::TTime) const;

        //! Remove heavy hitting people and attributes from the feature
        //! data if necessary.
        template<typename T, typename PERSON_FILTER, typename ATTRIBUTE_FILTER>
        void applyFilters(bool updateStatistics,
                          const PERSON_FILTER &personFilter,
                          const ATTRIBUTE_FILTER &attributeFilter,
                          T &data) const;

        //! Get the first time each attribute was seen.
        const TTimeVec &attributeFirstBucketTimes(void) const;
        //! Get the last time each attribute was seen.
        const TTimeVec &attributeLastBucketTimes(void) const;

        //! Get the people and attributes to remove if any.
        void peopleAndAttributesToRemove(core_t::TTime time,
                                         std::size_t maximumAge,
                                         TSizeVec &peopleToRemove,
                                         TSizeVec &attributesToRemove) const;

        //! Remove the \p people.
        void removePeople(const TSizeVec &peopleToRemove);

        //! Skip sampling the interval \p endTime - \p startTime.
        virtual void doSkipSampling(core_t::TTime startTime, core_t::TTime endTime) = 0;

    private:
        using TOptionalCountMinSketch = boost::optional<maths::CCountMinSketch>;

    private:
        //! The last time each person was seen.
        TTimeVec                  m_PersonLastBucketTimes;

        //! The first time each attribute was seen.
        TTimeVec                  m_AttributeFirstBucketTimes;

        //! The last time each attribute was seen.
        TTimeVec                  m_AttributeLastBucketTimes;

        //! The initial sketch to use for estimating the number of distinct people.
        maths::CBjkstUniqueValues m_NewDistinctPersonCounts;

        //! The number of distinct people generating each attribute.
        TBjkstUniqueValuesVec     m_DistinctPersonCounts;

        //! The initial sketch to use for estimating person bucket counts.
        TOptionalCountMinSketch   m_NewPersonBucketCounts;

        //! The bucket count of each (person, attribute) pair in the exponentially
        //! decaying window with decay rate equal to CAnomalyDetectorModel::m_DecayRate.
        TCountMinSketchVec        m_PersonAttributeBucketCounts;
};

}
}

#endif // INCLUDED_ml_model_CPopulationModel_h
