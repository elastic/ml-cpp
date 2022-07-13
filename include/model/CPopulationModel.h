/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#ifndef INCLUDED_ml_model_CPopulationModel_h
#define INCLUDED_ml_model_CPopulationModel_h

#include <core/CMemory.h>
#include <core/CNonCopyable.h>
#include <core/CTriple.h>
#include <core/CoreTypes.h>

#include <maths/common/CBjkstUniqueValues.h>
#include <maths/common/CMultivariatePrior.h>

#include <maths/time_series/CCountMinSketch.h>

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
namespace common {
class CPrior;
}
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
    using TTimeVec = std::vector<core_t::TTime>;
    using TSizeUInt64Pr = std::pair<std::size_t, uint64_t>;
    using TSizeUInt64PrVec = std::vector<TSizeUInt64Pr>;
    using TCountMinSketchVec = std::vector<maths::time_series::CCountMinSketch>;
    using TBjkstUniqueValuesVec = std::vector<maths::common::CBjkstUniqueValues>;
    using TSizeTimeUMap = boost::unordered_map<std::size_t, core_t::TTime>;

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
    CPopulationModel(const SModelParams& params,
                     const TDataGathererPtr& dataGatherer,
                     const TFeatureInfluenceCalculatorCPtrPrVecVec& influenceCalculators);

    //! Create a copy that will result in the same persisted state as the
    //! original.  This is effectively a copy constructor that creates a
    //! copy that's only valid for a single purpose.  The boolean flag is
    //! redundant except to create a signature that will not be mistaken
    //! for a general purpose copy constructor.
    CPopulationModel(bool isForPersistence, const CPopulationModel& other);
    //@}

    //! Returns true.
    bool isPopulation() const override;

    //! \name Bucket Statistics
    //@{
    //! Get the count of the bucketing interval containing \p time
    //! for the person identified by \p pid.
    TOptionalUInt64 currentBucketCount(std::size_t pid, core_t::TTime time) const override;

    //! Returns null.
    TOptionalDouble baselineBucketCount(std::size_t pid) const override;

protected:
    //! Get the index range [begin, end) of the person corresponding to
    //! \p pid in the vector \p data. This relies on the fact that \p data
    //! is sort lexicographically by person then attribute identifier.
    //! This will return an empty range if the person is not present.
    template<typename T>
    static TSizeSizePr personRange(const T& data, std::size_t pid);

    //! Find the person attribute pair identified by \p pid and \p cid,
    //! respectively, in \p data if it exists. Returns the end of the
    //! vector if it doesn't.
    template<typename T>
    static typename T::const_iterator find(const T& data, std::size_t pid, std::size_t cid);

    //! Extract the bucket value for count feature data.
    static inline TDouble1Vec
    extractValue(model_t::EFeature /*feature*/,
                 const std::pair<TSizeSizePr, SEventRateFeatureData>& data);
    //! Extract the bucket value for metric feature data.
    static inline TDouble1Vec
    extractValue(model_t::EFeature feature,
                 const std::pair<TSizeSizePr, SMetricFeatureData>& data);
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
    void currentBucketPersonIds(core_t::TTime time, TSizeVec& result) const override;
    //@}

    //! \name Update
    //@{
    //! Update the rates for \p feature and \p people.
    void sample(core_t::TTime startTime,
                core_t::TTime endTime,
                CResourceMonitor& resourceMonitor) override = 0;
    //@}

    //! Get the checksum of this model.
    //!
    //! \param[in] includeCurrentBucketStats If true then include the
    //! current bucket statistics. (This is designed to handle serialization,
    //! for which we don't serialize the current bucket statistics.)
    uint64_t checksum(bool includeCurrentBucketStats = true) const override = 0;

    //! Debug the memory used by this model.
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const override = 0;

    //! Get the memory used by this model.
    std::size_t memoryUsage() const override = 0;

    //! Get the static size of this object - used for virtual hierarchies
    std::size_t staticSize() const override = 0;

    //! Get the non-estimated value of the the memory used by this model.
    std::size_t computeMemoryUsage() const override = 0;

    //! Get the frequency of the attribute identified by \p cid.
    double attributeFrequency(std::size_t cid) const override;

    //! Get the first time each attribute was seen.
    const TTimeVec& attributeFirstBucketTimes() const;

    //! Get the last time each attribute was seen.
    const TTimeVec& attributeLastBucketTimes() const;

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
        bool operator==(const CCorrectionKey& rhs) const;
        std::size_t hash() const;

    private:
        model_t::EFeature m_Feature;
        std::size_t m_Pid;
        std::size_t m_Cid;
        std::size_t m_Correlate;
    };

    //! \brief A hasher for the partial bucket corrections map key.
    struct MODEL_EXPORT CHashCorrectionKey {
        std::size_t operator()(const CCorrectionKey& key) const {
            return key.hash();
        }
    };
    using TCorrectionKeyDouble1VecUMap =
        boost::unordered_map<CCorrectionKey, TDouble1Vec, CHashCorrectionKey>;

protected:
    //! Persist state by passing information to the supplied inserter.
    void doAcceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Restore the model reading state from the supplied traverser.
    bool doAcceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Get the current bucket person counts.
    virtual const TSizeUInt64PrVec& personCounts() const = 0;

    //! Check if bucket statistics are available for the specified time.
    bool bucketStatsAvailable(core_t::TTime time) const override = 0;

    //! Monitor the resource usage while creating new models
    void createUpdateNewModels(core_t::TTime time, CResourceMonitor& resourceMonitor);

    //! Initialize the time series models for "n" newly observed people
    //! and "m" newly observed attributes.
    void createNewModels(std::size_t n, std::size_t m) override = 0;

    //! Initialize the time series models for recycled attributes
    //! and/or people.
    void updateRecycledModels() override = 0;

    //! Update the correlation models.
    virtual void refreshCorrelationModels(std::size_t resourceLimit,
                                          CResourceMonitor& resourceMonitor) = 0;

    //! Clear out large state objects for people/attributes that are pruned.
    void clearPrunedResources(const TSizeVec& people, const TSizeVec& attributes) override = 0;

    //! Correct \p baseline with \p corrections for interim results.
    void correctBaselineForInterim(model_t::EFeature feature,
                                   std::size_t pid,
                                   std::size_t cid,
                                   model_t::CResultType type,
                                   const TSizeDoublePr1Vec& correlated,
                                   const TCorrectionKeyDouble1VecUMap& corrections,
                                   TDouble1Vec& baseline) const;

    //! Get the time by which to propagate the priors on a sample.
    double propagationTime(std::size_t cid, core_t::TTime) const;

    //! Remove heavy hitting people and attributes from the feature
    //! data if necessary.
    template<typename T, typename PERSON_FILTER, typename ATTRIBUTE_FILTER>
    void applyFilters(bool updateStatistics,
                      const PERSON_FILTER& personFilter,
                      const ATTRIBUTE_FILTER& attributeFilter,
                      T& data) const;

    //! Get the people and attributes to remove if any.
    void peopleAndAttributesToRemove(core_t::TTime time,
                                     std::size_t maximumAge,
                                     TSizeVec& peopleToRemove,
                                     TSizeVec& attributesToRemove) const;

    //! Remove the \p people.
    void removePeople(const TSizeVec& peopleToRemove);

    //! Skip sampling the interval \p endTime - \p startTime.
    void doSkipSampling(core_t::TTime startTime, core_t::TTime endTime) override = 0;

private:
    using TOptionalCountMinSketch = std::optional<maths::time_series::CCountMinSketch>;

private:
    //! The last time each person was seen.
    TTimeVec m_PersonLastBucketTimes;

    //! The first time each attribute was seen.
    TTimeVec m_AttributeFirstBucketTimes;

    //! The last time each attribute was seen.
    TTimeVec m_AttributeLastBucketTimes;

    //! The initial sketch to use for estimating the number of distinct people.
    maths::common::CBjkstUniqueValues m_NewDistinctPersonCounts;

    //! The number of distinct people generating each attribute.
    TBjkstUniqueValuesVec m_DistinctPersonCounts;

    //! The initial sketch to use for estimating person bucket counts.
    TOptionalCountMinSketch m_NewPersonBucketCounts;

    //! The bucket count of each (person, attribute) pair in the exponentially
    //! decaying window with decay rate equal to CAnomalyDetectorModel::m_DecayRate.
    TCountMinSketchVec m_PersonAttributeBucketCounts;
};
}
}

#endif // INCLUDED_ml_model_CPopulationModel_h
