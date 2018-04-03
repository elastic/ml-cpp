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

#include <model/CEventRateBucketGatherer.h>

#include <core/CCompressUtils.h>
#include <core/CFunctional.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CStatistics.h>
#include <core/Constants.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>
#include <maths/COrderings.h>

#include <model/CDataGatherer.h>
#include <model/CEventData.h>
#include <model/CResourceMonitor.h>
#include <model/CSearchKey.h>
#include <model/CStringStore.h>
#include <model/FunctionTypes.h>

#include <boost/bind.hpp>
#include <boost/make_shared.hpp>
#include <boost/ref.hpp>
#include <boost/unordered_set.hpp>

#include <algorithm>
#include <atomic>
#include <limits>
#include <map>
#include <string>

namespace ml {
namespace model {

namespace {

typedef std::vector<std::size_t> TSizeVec;
typedef std::vector<std::string> TStrVec;
typedef std::map<std::string, uint64_t> TStrUInt64Map;
typedef std::pair<std::size_t, std::size_t> TSizeSizePr;
typedef std::vector<TSizeSizePr> TSizeSizePrVec;
typedef std::vector<uint64_t> TUInt64Vec;
typedef boost::unordered_set<std::size_t> TSizeUSet;
typedef TSizeUSet::const_iterator TSizeUSetCItr;
typedef std::vector<TSizeUSet> TSizeUSetVec;
typedef maths::CBasicStatistics::SSampleMean<double>::TAccumulator TMeanAccumulator;
typedef boost::unordered_map<TSizeSizePr, TMeanAccumulator> TSizeSizePrMeanAccumulatorUMap;
typedef std::map<TSizeSizePr, uint64_t> TSizeSizePrUInt64Map;
typedef CBucketQueue<TSizeSizePrMeanAccumulatorUMap> TSizeSizePrMeanAccumulatorUMapQueue;
typedef CEventRateBucketGatherer::TCategoryAnyMap TCategoryAnyMap;
typedef boost::unordered_map<TSizeSizePr, CUniqueStringFeatureData> TSizeSizePrStrDataUMap;
typedef CBucketQueue<TSizeSizePrStrDataUMap> TSizeSizePrStrDataUMapQueue;
typedef CBucketGatherer::TStoredStringPtrVec TStoredStringPtrVec;

// We use short field names to reduce the state size
const std::string BASE_TAG("a");
const std::string ATTRIBUTE_PEOPLE_TAG("b");
const std::string UNIQUE_VALUES_TAG("c");
const std::string TIMES_OF_DAY_TAG("d");
const std::string EMPTY_STRING;

// Nested tags.
const std::string ATTRIBUTE_TAG("a");
const std::string PERSON_TAG("b");
const std::string STRING_ITEM_TAG("h");
const std::string MEAN_TIMES_TAG("i");

// Unique strings tags.
const std::string INFLUENCER_UNIQUE_STRINGS_TAG("a");
const std::string UNIQUE_STRINGS_TAG("b");

//! \brief Manages persistence of time-of-day feature data maps.
struct STimesBucketSerializer {
    void operator()(const TSizeSizePrMeanAccumulatorUMap& times, core::CStatePersistInserter& inserter) {
        std::vector<TSizeSizePrMeanAccumulatorUMap::const_iterator> ordered;
        ordered.reserve(times.size());
        for (auto i = times.begin(); i != times.end(); ++i) {
            ordered.push_back(i);
        }
        std::sort(ordered.begin(), ordered.end(), core::CFunctional::SDereference<maths::COrderings::SFirstLess>());
        for (std::size_t i = 0u; i < ordered.size(); ++i) {
            inserter.insertValue(PERSON_TAG, CDataGatherer::extractPersonId(*ordered[i]));
            inserter.insertValue(ATTRIBUTE_TAG, CDataGatherer::extractAttributeId(*ordered[i]));
            inserter.insertValue(MEAN_TIMES_TAG, CDataGatherer::extractData(*ordered[i]).toDelimited());
        }
    }

    bool operator()(TSizeSizePrMeanAccumulatorUMap& times, core::CStateRestoreTraverser& traverser) const {
        std::size_t pid = 0;
        std::size_t cid = 0;
        do {
            const std::string& name = traverser.name();
            RESTORE_BUILT_IN(PERSON_TAG, pid)
            RESTORE_BUILT_IN(ATTRIBUTE_TAG, cid)
            RESTORE(MEAN_TIMES_TAG, times[TSizeSizePr(pid, cid)].fromDelimited(traverser.value()))
        } while (traverser.next());

        return true;
    }
};

//! \brief Manages persistence of unique string feature data maps.
struct SStrDataBucketSerializer {
    void operator()(const TSizeSizePrStrDataUMap& strings, core::CStatePersistInserter& inserter) {
        std::vector<TSizeSizePrStrDataUMap::const_iterator> ordered;
        ordered.reserve(strings.size());
        for (auto i = strings.begin(); i != strings.end(); ++i) {
            ordered.push_back(i);
        }
        std::sort(ordered.begin(), ordered.end(), core::CFunctional::SDereference<maths::COrderings::SFirstLess>());
        for (std::size_t i = 0u; i != ordered.size(); ++i) {
            inserter.insertValue(PERSON_TAG, CDataGatherer::extractPersonId(*ordered[i]));
            inserter.insertValue(ATTRIBUTE_TAG, CDataGatherer::extractAttributeId(*ordered[i]));
            inserter.insertLevel(
                STRING_ITEM_TAG,
                boost::bind(&CUniqueStringFeatureData::acceptPersistInserter, boost::cref(CDataGatherer::extractData(*ordered[i])), _1));
        }
    }
    bool operator()(TSizeSizePrStrDataUMap& map, core::CStateRestoreTraverser& traverser) const {
        std::size_t pid = 0;
        std::size_t cid = 0;
        do {
            const std::string& name = traverser.name();
            RESTORE_BUILT_IN(PERSON_TAG, pid)
            RESTORE_BUILT_IN(ATTRIBUTE_TAG, cid)
            RESTORE(STRING_ITEM_TAG,
                    traverser.traverseSubLevel(
                        boost::bind(&CUniqueStringFeatureData::acceptRestoreTraverser, boost::ref(map[TSizeSizePr(pid, cid)]), _1)))
        } while (traverser.next());

        return true;
    }
};

//! Serialize \p data.
void persistAttributePeopleData(const TSizeUSetVec& data, core::CStatePersistInserter& inserter) {
    // Persist the vector in reverse order, because it means we'll
    // find out the correct size more efficiently on restore.
    std::size_t index = data.size();
    while (index > 0) {
        --index;
        inserter.insertValue(ATTRIBUTE_TAG, index);
        const TSizeUSet& people = data[index];

        // Persist the person identifiers in sorted order to make
        // it easier to compare state records.
        TSizeVec orderedPeople(people.begin(), people.end());
        std::sort(orderedPeople.begin(), orderedPeople.end());
        for (std::size_t i = 0u; i < orderedPeople.size(); ++i) {
            inserter.insertValue(PERSON_TAG, orderedPeople[i]);
        }
    }
}

//! Serialize \p featureData.
void persistFeatureData(const TCategoryAnyMap& featureData, core::CStatePersistInserter& inserter) {
    for (const auto& data_ : featureData) {
        model_t::EEventRateCategory category = data_.first;
        const boost::any& data = data_.second;
        try {
            switch (category) {
            case model_t::E_DiurnalTimes:
                inserter.insertLevel(TIMES_OF_DAY_TAG,
                                     boost::bind<void>(TSizeSizePrMeanAccumulatorUMapQueue::CSerializer<STimesBucketSerializer>(),
                                                       boost::cref(boost::any_cast<const TSizeSizePrMeanAccumulatorUMapQueue&>(data)),
                                                       _1));
                break;
            case model_t::E_MeanArrivalTimes:
                // TODO
                break;
            case model_t::E_AttributePeople:
                inserter.insertLevel(ATTRIBUTE_PEOPLE_TAG,
                                     boost::bind(&persistAttributePeopleData, boost::cref(boost::any_cast<const TSizeUSetVec&>(data)), _1));
                break;
            case model_t::E_UniqueValues:
                inserter.insertLevel(UNIQUE_VALUES_TAG,
                                     boost::bind<void>(TSizeSizePrStrDataUMapQueue::CSerializer<SStrDataBucketSerializer>(),
                                                       boost::cref(boost::any_cast<const TSizeSizePrStrDataUMapQueue&>(data)),
                                                       _1));
                break;
            }
        } catch (const std::exception& e) { LOG_ERROR("Failed to serialize data for " << category << ": " << e.what()); }
    }
}

//! Extract \p data from a state document.
bool restoreAttributePeopleData(core::CStateRestoreTraverser& traverser, TSizeUSetVec& data) {
    size_t lastCid = 0;
    bool seenCid = false;

    do {
        const std::string& name = traverser.name();
        if (name == ATTRIBUTE_TAG) {
            if (core::CStringUtils::stringToType(traverser.value(), lastCid) == false) {
                LOG_ERROR("Invalid attribute ID in " << traverser.value());
                return false;
            }
            seenCid = true;
            if (lastCid >= data.size()) {
                data.resize(lastCid + 1);
            }
        } else if (name == PERSON_TAG) {
            if (!seenCid) {
                LOG_ERROR("Incorrect format - person ID before attribute ID in " << traverser.value());
                return false;
            }
            std::size_t pid = 0;
            if (core::CStringUtils::stringToType(traverser.value(), pid) == false) {
                LOG_ERROR("Invalid person ID in " << traverser.value());
                return false;
            }
            data[lastCid].insert(pid);
        }
    } while (traverser.next());

    return true;
}

//! Extract \p featureData from a state document.
bool restoreFeatureData(core::CStateRestoreTraverser& traverser,
                        TCategoryAnyMap& featureData,
                        std::size_t latencyBuckets,
                        core_t::TTime bucketLength,
                        core_t::TTime currentBucketStartTime) {
    const std::string& name = traverser.name();
    if (name == ATTRIBUTE_PEOPLE_TAG) {
        TSizeUSetVec* data{
            boost::unsafe_any_cast<TSizeUSetVec>(&featureData.emplace(model_t::E_AttributePeople, TSizeUSetVec()).first->second)};
        if (traverser.traverseSubLevel(boost::bind(&restoreAttributePeopleData, _1, boost::ref(*data))) == false) {
            LOG_ERROR("Invalid attribute/people mapping in " << traverser.value());
            return false;
        }
    } else if (name == UNIQUE_VALUES_TAG) {
        if (featureData.count(model_t::E_UniqueValues) != 0) {
            featureData.erase(model_t::E_UniqueValues);
        }
        TSizeSizePrStrDataUMapQueue* data{boost::unsafe_any_cast<TSizeSizePrStrDataUMapQueue>(
            &featureData
                 .emplace(model_t::E_UniqueValues,
                          TSizeSizePrStrDataUMapQueue(latencyBuckets, bucketLength, currentBucketStartTime, TSizeSizePrStrDataUMap(1)))
                 .first->second)};
        if (traverser.traverseSubLevel(boost::bind<bool>(
                TSizeSizePrStrDataUMapQueue::CSerializer<SStrDataBucketSerializer>(TSizeSizePrStrDataUMap(1)), boost::ref(*data), _1)) ==
            false) {
            LOG_ERROR("Invalid unique value mapping in " << traverser.value());
            return false;
        }
    } else if (name == TIMES_OF_DAY_TAG) {
        if (featureData.count(model_t::E_DiurnalTimes) == 0) {
            featureData.erase(model_t::E_DiurnalTimes);
        }
        TSizeSizePrMeanAccumulatorUMapQueue* data{boost::unsafe_any_cast<TSizeSizePrMeanAccumulatorUMapQueue>(
            &featureData
                 .emplace(model_t::E_DiurnalTimes,
                          TSizeSizePrMeanAccumulatorUMapQueue(latencyBuckets, bucketLength, currentBucketStartTime))
                 .first->second)};
        if (traverser.traverseSubLevel(boost::bind<bool>(
                TSizeSizePrMeanAccumulatorUMapQueue::CSerializer<STimesBucketSerializer>(), boost::ref(*data), _1)) == false) {
            LOG_ERROR("Invalid times mapping in " << traverser.value());
            return false;
        }
    }
    return true;
}

//! Get the by field name.
const std::string& byField(bool population, const TStrVec& fieldNames) {
    return population ? fieldNames[1] : fieldNames[0];
}

//! Get the over field name.
const std::string& overField(bool population, const TStrVec& fieldNames) {
    return population ? fieldNames[0] : EMPTY_STRING;
}

template<typename ITR, typename T>
struct SMaybeConst {};
template<typename T>
struct SMaybeConst<TCategoryAnyMap::iterator, T> {
    typedef T& TRef;
};
template<typename T>
struct SMaybeConst<TCategoryAnyMap::const_iterator, T> {
    typedef const T& TRef;
};

//! Apply a function \p f to all the data held in [\p begin, \p end).
template<typename ITR, typename F>
void apply(ITR begin, ITR end, const F& f) {
    for (ITR itr = begin; itr != end; ++itr) {
        model_t::EEventRateCategory category = itr->first;
        try {
            switch (category) {
            case model_t::E_DiurnalTimes: {
                f(boost::any_cast<typename SMaybeConst<ITR, TSizeSizePrMeanAccumulatorUMapQueue>::TRef>(itr->second));
                break;
            }
            case model_t::E_MeanArrivalTimes: {
                // TODO
                break;
            }
            case model_t::E_AttributePeople: {
                f(boost::any_cast<typename SMaybeConst<ITR, TSizeUSetVec>::TRef>(itr->second));
                break;
            }
            case model_t::E_UniqueValues:
                f(boost::any_cast<typename SMaybeConst<ITR, TSizeSizePrStrDataUMapQueue>::TRef>(itr->second));
                break;
            }
        } catch (const std::exception& e) { LOG_ERROR("Apply failed for " << category << ": " << e.what()); }
    }
}

//! Apply a function \p f to all the data held in \p featureData.
template<typename T, typename F>
void apply(T& featureData, const F& f) {
    apply(featureData.begin(), featureData.end(), f);
}

//! \brief Removes people from the feature data.
struct SRemovePeople {
    void operator()(TSizeUSetVec& attributePeople, std::size_t lowestPersonToRemove, std::size_t endPeople) const {
        for (std::size_t cid = 0u; cid < attributePeople.size(); ++cid) {
            for (std::size_t pid = lowestPersonToRemove; pid < endPeople; ++pid) {
                attributePeople[cid].erase(pid);
            }
        }
    }
    void operator()(TSizeUSetVec& attributePeople, const TSizeVec& peopleToRemove) const {
        for (std::size_t cid = 0u; cid < attributePeople.size(); ++cid) {
            for (std::size_t i = 0u; i < peopleToRemove.size(); ++i) {
                attributePeople[cid].erase(peopleToRemove[i]);
            }
        }
    }
    void
    operator()(TSizeSizePrStrDataUMapQueue& peopleAttributeUniqueValues, std::size_t lowestPersonToRemove, std::size_t endPeople) const {
        for (auto&& bucket : peopleAttributeUniqueValues) {
            for (auto i = bucket.begin(); i != bucket.end(); /**/) {
                if (CDataGatherer::extractPersonId(*i) >= lowestPersonToRemove && CDataGatherer::extractPersonId(*i) < endPeople) {
                    i = bucket.erase(i);
                } else {
                    ++i;
                }
            }
        }
    }
    void operator()(TSizeSizePrStrDataUMapQueue& peopleAttributeUniqueValues, const TSizeVec& peopleToRemove) const {
        CBucketGatherer::remove(peopleToRemove, CDataGatherer::SExtractPersonId(), peopleAttributeUniqueValues);
    }
    void operator()(TSizeSizePrMeanAccumulatorUMapQueue& arrivalTimes, std::size_t lowestPersonToRemove, std::size_t endPeople) const {
        for (auto&& bucket : arrivalTimes) {
            for (auto i = bucket.begin(); i != bucket.end(); /**/) {
                if (CDataGatherer::extractPersonId(*i) >= lowestPersonToRemove && CDataGatherer::extractPersonId(*i) < endPeople) {
                    i = bucket.erase(i);
                } else {
                    ++i;
                }
            }
        }
    }
    void operator()(TSizeSizePrMeanAccumulatorUMapQueue& arrivalTimes, const TSizeVec& peopleToRemove) const {
        CBucketGatherer::remove(peopleToRemove, CDataGatherer::SExtractPersonId(), arrivalTimes);
    }
};

//! \brief Removes attributes from the feature data.
struct SRemoveAttributes {
    void operator()(TSizeUSetVec& attributePeople, std::size_t lowestAttributeToRemove) const {
        if (lowestAttributeToRemove < attributePeople.size()) {
            attributePeople.erase(attributePeople.begin() + lowestAttributeToRemove, attributePeople.end());
        }
    }
    void operator()(TSizeUSetVec& attributePeople, const TSizeVec& attributesToRemove) const {
        for (std::size_t i = 0u; i < attributesToRemove.size(); ++i) {
            attributePeople[attributesToRemove[i]].clear();
        }
    }
    void operator()(TSizeSizePrStrDataUMapQueue& peopleAttributeUniqueValues, std::size_t lowestAttributeToRemove) const {
        for (auto&& bucket : peopleAttributeUniqueValues) {
            for (auto i = bucket.begin(); i != bucket.end(); /**/) {
                if (CDataGatherer::extractAttributeId(*i) >= lowestAttributeToRemove) {
                    i = bucket.erase(i);
                } else {
                    ++i;
                }
            }
        }
    }
    void operator()(TSizeSizePrStrDataUMapQueue& peopleAttributeUniqueValues, const TSizeVec& attributesToRemove) const {
        CBucketGatherer::remove(attributesToRemove, CDataGatherer::SExtractAttributeId(), peopleAttributeUniqueValues);
    }
    void operator()(TSizeSizePrMeanAccumulatorUMapQueue& arrivalTimes, std::size_t lowestAttributeToRemove) const {
        for (auto&& bucket : arrivalTimes) {
            for (auto i = bucket.begin(); i != bucket.end(); /**/) {
                if (CDataGatherer::extractAttributeId(*i) >= lowestAttributeToRemove) {
                    i = bucket.erase(i);
                } else {
                    ++i;
                }
            }
        }
    }
    void operator()(TSizeSizePrMeanAccumulatorUMapQueue& arrivalTimes, const TSizeVec& attributesToRemove) const {
        CBucketGatherer::remove(attributesToRemove, CDataGatherer::SExtractAttributeId(), arrivalTimes);
    }
};

//! \brief Computes a checksum for the feature data.
struct SChecksum {
    void operator()(const TSizeUSetVec& attributePeople, const CDataGatherer& gatherer, TStrUInt64Map& hashes) const {
        typedef boost::reference_wrapper<const std::string> TStrCRef;
        typedef std::vector<TStrCRef> TStrCRefVec;

        for (std::size_t cid = 0u; cid < attributePeople.size(); ++cid) {
            if (gatherer.isAttributeActive(cid)) {
                TStrCRefVec people;
                people.reserve(attributePeople[cid].size());
                for (const auto& person : attributePeople[cid]) {
                    if (gatherer.isPersonActive(person)) {
                        people.emplace_back(gatherer.personName(person));
                    }
                }
                std::sort(people.begin(), people.end(), maths::COrderings::SReferenceLess());
                uint64_t& hash = hashes[gatherer.attributeName(cid)];
                hash = maths::CChecksum::calculate(hash, people);
            }
        }
    }
    void
    operator()(const TSizeSizePrStrDataUMapQueue& peopleAttributeUniqueValues, const CDataGatherer& gatherer, TStrUInt64Map& hashes) const {
        for (const auto& uniques : peopleAttributeUniqueValues) {
            this->checksum(uniques, gatherer, hashes);
        }
    }
    void operator()(const TSizeSizePrMeanAccumulatorUMapQueue& arrivalTimes, const CDataGatherer& gatherer, TStrUInt64Map& hashes) const {
        for (const auto& time : arrivalTimes) {
            this->checksum(time, gatherer, hashes);
        }
    }
    template<typename DATA>
    void checksum(const boost::unordered_map<TSizeSizePr, DATA>& bucket, const CDataGatherer& gatherer, TStrUInt64Map& hashes) const {
        typedef boost::unordered_map<std::size_t, TUInt64Vec> TSizeUInt64VecUMap;

        TSizeUInt64VecUMap attributeHashes;

        for (const auto& value : bucket) {
            std::size_t pid = CDataGatherer::extractPersonId(value);
            std::size_t cid = CDataGatherer::extractAttributeId(value);
            if (gatherer.isPersonActive(pid) && gatherer.isAttributeActive(cid)) {
                attributeHashes[cid].push_back(maths::CChecksum::calculate(0, value.second));
            }
        }

        for (auto&& hash_ : attributeHashes) {
            std::sort(hash_.second.begin(), hash_.second.end());
            uint64_t& hash = hashes[gatherer.attributeName(hash_.first)];
            hash = maths::CChecksum::calculate(hash, hash_.second);
        }
    }
};

//! \brief Resize the feature data to accommodate a specified
//! person and attribute identifier.
struct SResize {
    void operator()(TSizeUSetVec& attributePeople, std::size_t /*pid*/, std::size_t cid) const {
        if (cid >= attributePeople.size()) {
            attributePeople.resize(cid + 1);
        }
    }
    void operator()(TSizeSizePrStrDataUMapQueue& /*data*/, std::size_t /*pid*/, std::size_t /*cid*/) const {
        // Not needed
    }
    void operator()(const TSizeSizePrMeanAccumulatorUMapQueue& /*arrivalTimes*/, std::size_t /*pid*/, std::size_t /*cid*/) const {
        // Not needed
    }
};

//! \brief Updates the feature data with some aggregated records.
struct SAddValue {
    void operator()(TSizeUSetVec& attributePeople,
                    std::size_t pid,
                    std::size_t cid,
                    core_t::TTime /*time*/,
                    std::size_t /*count*/,
                    const CEventData::TDouble1VecArray& /*values*/,
                    const CEventData::TOptionalStr& /*uniqueStrings*/,
                    const TStoredStringPtrVec& /*influences*/) const {
        attributePeople[cid].insert(pid);
    }
    void operator()(TSizeSizePrStrDataUMapQueue& personAttributeUniqueCounts,
                    std::size_t pid,
                    std::size_t cid,
                    core_t::TTime time,
                    std::size_t /*count*/,
                    const CEventData::TDouble1VecArray& /*values*/,
                    const CEventData::TOptionalStr& uniqueString,
                    const TStoredStringPtrVec& influences) const {
        if (!uniqueString) {
            return;
        }
        if (time > personAttributeUniqueCounts.latestBucketEnd()) {
            LOG_ERROR("No queue item for time " << time);
            personAttributeUniqueCounts.push(TSizeSizePrStrDataUMap(1), time);
        }
        TSizeSizePrStrDataUMap& counts = personAttributeUniqueCounts.get(time);
        counts[{pid, cid}].insert(*uniqueString, influences);
    }
    void operator()(TSizeSizePrMeanAccumulatorUMapQueue& arrivalTimes,
                    std::size_t pid,
                    std::size_t cid,
                    core_t::TTime time,
                    std::size_t count,
                    const CEventData::TDouble1VecArray& values,
                    const CEventData::TOptionalStr& /*uniqueStrings*/,
                    const TStoredStringPtrVec& /*influences*/) const {
        if (time > arrivalTimes.latestBucketEnd()) {
            LOG_ERROR("No queue item for time " << time);
            arrivalTimes.push(TSizeSizePrMeanAccumulatorUMap(1), time);
        }
        TSizeSizePrMeanAccumulatorUMap& times = arrivalTimes.get(time);
        for (std::size_t i = 0; i < count; i++) {
            times[{pid, cid}].add(values[i][0]);
        }
    }
};

//! \brief Updates the feature data for the start of a new bucket.
struct SNewBucket {
    void operator()(TSizeUSetVec& /*attributePeople*/, core_t::TTime /*time*/) const {}
    void operator()(TSizeSizePrStrDataUMapQueue& personAttributeUniqueCounts, core_t::TTime time) const {
        if (time > personAttributeUniqueCounts.latestBucketEnd()) {
            personAttributeUniqueCounts.push(TSizeSizePrStrDataUMap(1), time);
        } else {
            personAttributeUniqueCounts.get(time).clear();
        }
    }
    void operator()(TSizeSizePrMeanAccumulatorUMapQueue& arrivalTimes, core_t::TTime time) const {
        if (time > arrivalTimes.latestBucketEnd()) {
            arrivalTimes.push(TSizeSizePrMeanAccumulatorUMap(1), time);
        } else {
            arrivalTimes.get(time).clear();
        }
    }
};

//! Nested tags.
const std::string DICTIONARY_WORD_TAG("a");
const std::string UNIQUE_WORD_TAG("b");

//! Persist a collection of unique strings.
void persistUniqueStrings(const CUniqueStringFeatureData::TWordStringUMap& map, core::CStatePersistInserter& inserter) {
    typedef std::vector<CUniqueStringFeatureData::TWord> TWordVec;

    if (!map.empty()) {
        // Order the map keys to ensure consistent persistence
        TWordVec keys;
        keys.reserve(map.size());
        for (const auto& value : map) {
            keys.push_back(value.first);
        }
        std::sort(keys.begin(), keys.end());

        for (const auto& key : keys) {
            inserter.insertValue(DICTIONARY_WORD_TAG, key.toDelimited());
            inserter.insertValue(UNIQUE_WORD_TAG, map.at(key));
        }
    }
}

//! Restore a collection of unique strings.
bool restoreUniqueStrings(core::CStateRestoreTraverser& traverser, CUniqueStringFeatureData::TWordStringUMap& map) {
    CUniqueStringFeatureData::TWord word;
    do {
        const std::string& name = traverser.name();
        RESTORE(DICTIONARY_WORD_TAG, word.fromDelimited(traverser.value()))
        RESTORE_NO_ERROR(UNIQUE_WORD_TAG, map[word] = traverser.value())
    } while (traverser.next());
    return true;
}

//! Persist influencer collections of unique strings.
void persistInfluencerUniqueStrings(const CUniqueStringFeatureData::TStoredStringPtrWordSetUMap& map,
                                    core::CStatePersistInserter& inserter) {
    typedef std::vector<core::CStoredStringPtr> TStoredStringPtrVec;

    if (!map.empty()) {
        // Order the map keys to ensure consistent persistence
        TStoredStringPtrVec keys;
        keys.reserve(map.size());
        for (const auto& influence : map) {
            keys.push_back(influence.first);
        }
        std::sort(keys.begin(), keys.end(), maths::COrderings::SLess());

        for (const auto& key : keys) {
            inserter.insertValue(DICTIONARY_WORD_TAG, *key);
            for (const auto& word : map.at(key)) {
                inserter.insertValue(UNIQUE_WORD_TAG, word.toDelimited());
            }
        }
    }
}

//! Restore influencer collections of unique strings.
bool restoreInfluencerUniqueStrings(core::CStateRestoreTraverser& traverser, CUniqueStringFeatureData::TStoredStringPtrWordSetUMap& data) {
    std::string key;
    do {
        const std::string& name = traverser.name();
        if (name == DICTIONARY_WORD_TAG) {
            key = traverser.value();
        } else if (name == UNIQUE_WORD_TAG) {
            CUniqueStringFeatureData::TWord value;
            if (value.fromDelimited(traverser.value()) == false) {
                LOG_ERROR("Failed to restore word " << traverser.value());
                return false;
            }
            auto i = data.begin();
            for (/**/; i != data.end(); ++i) {
                if (*i->first == key) {
                    i->second.insert(value);
                    break;
                }
            }
            if (i == data.end()) {
                data[CStringStore::influencers().get(key)].insert(value);
            }
        }
    } while (traverser.next());

    return true;
}

//! Register the callbacks for computing the size of feature data gatherers
//! with \p visitor.
template<typename VISITOR>
void registerMemoryCallbacks(VISITOR& visitor) {
    visitor.template registerCallback<TSizeUSetVec>();
    visitor.template registerCallback<TSizeSizePrStrDataUMapQueue>();
    visitor.template registerCallback<TSizeSizePrMeanAccumulatorUMapQueue>();
}

//! Register the callbacks for computing the size of feature data gatherers.
void registerMemoryCallbacks(void) {
    static std::atomic_flag once = ATOMIC_FLAG_INIT;
    if (once.test_and_set() == false) {
        registerMemoryCallbacks(core::CMemory::anyVisitor());
        registerMemoryCallbacks(core::CMemoryDebug::anyVisitor());
    }
}

} // unnamed::

CEventRateBucketGatherer::CEventRateBucketGatherer(CDataGatherer& dataGatherer,
                                                   const std::string& summaryCountFieldName,
                                                   const std::string& personFieldName,
                                                   const std::string& attributeFieldName,
                                                   const std::string& valueFieldName,
                                                   const TStrVec& influenceFieldNames,
                                                   core_t::TTime startTime)
    : CBucketGatherer(dataGatherer, startTime), m_BeginInfluencingFields(0), m_BeginValueField(0), m_BeginSummaryFields(0) {
    this->initializeFieldNames(personFieldName, attributeFieldName, valueFieldName, summaryCountFieldName, influenceFieldNames);
    this->initializeFeatureData();
}

CEventRateBucketGatherer::CEventRateBucketGatherer(CDataGatherer& dataGatherer,
                                                   const std::string& summaryCountFieldName,
                                                   const std::string& personFieldName,
                                                   const std::string& attributeFieldName,
                                                   const std::string& valueFieldName,
                                                   const TStrVec& influenceFieldNames,
                                                   core::CStateRestoreTraverser& traverser)
    : CBucketGatherer(dataGatherer, 0), m_BeginInfluencingFields(0), m_BeginValueField(0), m_BeginSummaryFields(0) {
    this->initializeFieldNames(personFieldName, attributeFieldName, valueFieldName, summaryCountFieldName, influenceFieldNames);
    traverser.traverseSubLevel(boost::bind(&CEventRateBucketGatherer::acceptRestoreTraverser, this, _1));
}

CEventRateBucketGatherer::CEventRateBucketGatherer(bool isForPersistence, const CEventRateBucketGatherer& other)
    : CBucketGatherer(isForPersistence, other),
      m_FieldNames(other.m_FieldNames),
      m_BeginInfluencingFields(other.m_BeginInfluencingFields),
      m_BeginValueField(other.m_BeginValueField),
      m_BeginSummaryFields(other.m_BeginSummaryFields),
      m_FeatureData(other.m_FeatureData) {
    if (!isForPersistence) {
        LOG_ABORT("This constructor only creates clones for persistence");
    }
}

bool CEventRateBucketGatherer::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    this->clear();
    do {
        const std::string& name = traverser.name();
        RESTORE(BASE_TAG, traverser.traverseSubLevel(boost::bind(&CBucketGatherer::baseAcceptRestoreTraverser, this, _1)))
        if (restoreFeatureData(
                traverser, m_FeatureData, m_DataGatherer.params().s_LatencyBuckets, this->bucketLength(), this->currentBucketStartTime()) ==
            false) {
            LOG_ERROR("Invalid feature data in " << traverser.value());
            return false;
        }
    } while (traverser.next());

    return true;
}

void CEventRateBucketGatherer::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertLevel(BASE_TAG, boost::bind(&CBucketGatherer::baseAcceptPersistInserter, this, _1));
    persistFeatureData(m_FeatureData, inserter);
}

CBucketGatherer* CEventRateBucketGatherer::cloneForPersistence(void) const {
    return new CEventRateBucketGatherer(true, *this);
}

const std::string& CEventRateBucketGatherer::persistenceTag(void) const {
    return CBucketGatherer::EVENTRATE_BUCKET_GATHERER_TAG;
}

const std::string& CEventRateBucketGatherer::personFieldName(void) const {
    return m_FieldNames[0];
}

const std::string& CEventRateBucketGatherer::attributeFieldName(void) const {
    return m_DataGatherer.isPopulation() ? m_FieldNames[1] : EMPTY_STRING;
}

const std::string& CEventRateBucketGatherer::valueFieldName(void) const {
    return m_BeginValueField != m_BeginSummaryFields ? m_FieldNames[m_BeginValueField] : EMPTY_STRING;
}

CEventRateBucketGatherer::TStrVecCItr CEventRateBucketGatherer::beginInfluencers(void) const {
    return m_FieldNames.begin() + m_BeginInfluencingFields;
}

CEventRateBucketGatherer::TStrVecCItr CEventRateBucketGatherer::endInfluencers(void) const {
    return m_FieldNames.begin() + m_BeginValueField;
}

const CEventRateBucketGatherer::TStrVec& CEventRateBucketGatherer::fieldsOfInterest(void) const {
    return m_FieldNames;
}

std::string CEventRateBucketGatherer::description(void) const {
    return function_t::name(function_t::function(m_DataGatherer.features())) +
           (m_BeginValueField == m_BeginSummaryFields ? "" : (" " + m_FieldNames[m_BeginValueField])) +
           (byField(m_DataGatherer.isPopulation(), m_FieldNames).empty() ? "" : " by ") +
           byField(m_DataGatherer.isPopulation(), m_FieldNames) +
           (overField(m_DataGatherer.isPopulation(), m_FieldNames).empty() ? "" : " over ") +
           overField(m_DataGatherer.isPopulation(), m_FieldNames) + (m_DataGatherer.partitionFieldName().empty() ? "" : " partition=") +
           m_DataGatherer.partitionFieldName();
}

bool CEventRateBucketGatherer::processFields(const TStrCPtrVec& fieldValues, CEventData& result, CResourceMonitor& resourceMonitor) {
    typedef boost::optional<std::size_t> TOptionalSize;
    typedef boost::optional<std::string> TOptionalStr;

    if (fieldValues.size() != m_FieldNames.size()) {
        LOG_ERROR("Unexpected field values: " << core::CContainerPrinter::print(fieldValues)
                                              << ", for field names: " << core::CContainerPrinter::print(m_FieldNames));
        return false;
    }

    const std::string* person = (fieldValues[0] == 0 && m_DataGatherer.useNull()) ? &EMPTY_STRING : fieldValues[0];
    if (person == 0) {
        // Just ignore: the "person" field wasn't present in the
        // record. Note that we don't warn here since we'll permit
        // a small fraction of records to having missing field
        // values.
        return false;
    }

    for (std::size_t i = m_DataGatherer.isPopulation() + 1; i < m_BeginValueField; ++i) {
        result.addInfluence(fieldValues[i] ? TOptionalStr(*fieldValues[i]) : TOptionalStr());
    }

    if (m_BeginValueField != m_BeginSummaryFields) {
        if (const std::string* value = fieldValues[m_BeginValueField]) {
            result.stringValue(*value);
        }
    }

    std::size_t count = 1;
    if (m_DataGatherer.summaryMode() != model_t::E_None) {
        if (m_DataGatherer.extractCountFromField(m_FieldNames[m_BeginSummaryFields], fieldValues[m_BeginSummaryFields], count) == false) {
            result.addValue();
            return true;
        }
    }

    if (count == CDataGatherer::EXPLICIT_NULL_SUMMARY_COUNT) {
        result.setExplicitNull();
    } else {
        model_t::EFeature feature = m_DataGatherer.feature(0);
        if ((feature == model_t::E_IndividualTimeOfDayByBucketAndPerson) ||
            (feature == model_t::E_PopulationTimeOfDayByBucketPersonAndAttribute)) {
            double t = static_cast<double>(result.time() % core::constants::DAY);
            result.addValue(TDouble1Vec(1, t));
        } else if ((feature == model_t::E_IndividualTimeOfWeekByBucketAndPerson) ||
                   (feature == model_t::E_PopulationTimeOfWeekByBucketPersonAndAttribute)) {
            double t = static_cast<double>(result.time() % core::constants::WEEK);
            result.addValue(TDouble1Vec(1, t));
        } else {
            result.addCountStatistic(count);
        }
    }

    bool addedPerson = false;
    std::size_t personId = CDynamicStringIdRegistry::INVALID_ID;
    if (result.isExplicitNull()) {
        m_DataGatherer.personId(*person, personId);
    } else {
        personId = m_DataGatherer.addPerson(*person, resourceMonitor, addedPerson);
    }

    if (personId == CDynamicStringIdRegistry::INVALID_ID) {
        if (!result.isExplicitNull()) {
            LOG_TRACE("Couldn't create a person, over memory limit");
        }
        return false;
    }
    if (addedPerson) {
        resourceMonitor.addExtraMemory(m_DataGatherer.isPopulation() ? CDataGatherer::ESTIMATED_MEM_USAGE_PER_OVER_FIELD
                                                                     : CDataGatherer::ESTIMATED_MEM_USAGE_PER_BY_FIELD);
        (m_DataGatherer.isPopulation() ? core::CStatistics::stat(stat_t::E_NumberOverFields)
                                       : core::CStatistics::stat(stat_t::E_NumberByFields))
            .increment();
    }
    if (!result.person(personId)) {
        LOG_ERROR("Bad by field value: " << *person);
        return false;
    }

    if (m_DataGatherer.isPopulation()) {
        const std::string* attribute = (fieldValues[1] == 0 && m_DataGatherer.useNull()) ? &EMPTY_STRING : fieldValues[1];

        if (attribute == 0) {
            // Just ignore: the "by" field wasn't present in the
            // record. This doesn't necessarily stop us processing
            // the record by other models so we don't return false.
            // Note that we don't warn here since we'll permit a
            // small fraction of records to having missing field
            // values.
            result.addAttribute();
            return true;
        }

        bool addedAttribute = false;
        std::size_t newAttribute = CDynamicStringIdRegistry::INVALID_ID;
        if (result.isExplicitNull()) {
            m_DataGatherer.attributeId(*attribute, newAttribute);
        } else {
            newAttribute = m_DataGatherer.addAttribute(*attribute, resourceMonitor, addedAttribute);
        }
        result.addAttribute(TOptionalSize(newAttribute));

        if (addedAttribute) {
            resourceMonitor.addExtraMemory(CDataGatherer::ESTIMATED_MEM_USAGE_PER_BY_FIELD);
            core::CStatistics::stat(stat_t::E_NumberByFields).increment();
        }
    } else {
        result.addAttribute(std::size_t(0));
    }

    return true;
}

void CEventRateBucketGatherer::recyclePeople(const TSizeVec& peopleToRemove) {
    if (peopleToRemove.empty()) {
        return;
    }

    apply(m_FeatureData, boost::bind<void>(SRemovePeople(), _1, boost::cref(peopleToRemove)));

    this->CBucketGatherer::recyclePeople(peopleToRemove);
}

void CEventRateBucketGatherer::removePeople(std::size_t lowestPersonToRemove) {
    apply(m_FeatureData, boost::bind<void>(SRemovePeople(), _1, lowestPersonToRemove, m_DataGatherer.numberPeople()));
    this->CBucketGatherer::removePeople(lowestPersonToRemove);
}

void CEventRateBucketGatherer::recycleAttributes(const TSizeVec& attributesToRemove) {
    if (attributesToRemove.empty()) {
        return;
    }

    apply(m_FeatureData, boost::bind<void>(SRemoveAttributes(), _1, boost::cref(attributesToRemove)));

    this->CBucketGatherer::recycleAttributes(attributesToRemove);
}

void CEventRateBucketGatherer::removeAttributes(std::size_t lowestAttributeToRemove) {
    apply(m_FeatureData, boost::bind<void>(SRemoveAttributes(), _1, lowestAttributeToRemove));
    this->CBucketGatherer::removeAttributes(lowestAttributeToRemove);
}

uint64_t CEventRateBucketGatherer::checksum(void) const {
    uint64_t seed = this->CBucketGatherer::checksum();

    TStrUInt64Map hashes;
    apply(m_FeatureData, boost::bind<void>(SChecksum(), _1, boost::cref(m_DataGatherer), boost::ref(hashes)));
    LOG_TRACE("seed = " << seed);
    LOG_TRACE("hashes = " << core::CContainerPrinter::print(hashes));
    core::CHashing::CSafeMurmurHash2String64 hasher;
    return core::CHashing::hashCombine(seed, hasher(core::CContainerPrinter::print(hashes)));
}

void CEventRateBucketGatherer::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    registerMemoryCallbacks();
    mem->setName("CPopulationEventRateDataGatherer");
    CBucketGatherer::debugMemoryUsage(mem->addChild());
    core::CMemoryDebug::dynamicSize("m_FieldNames", m_FieldNames, mem);
    core::CMemoryDebug::dynamicSize("m_FeatureData", m_FeatureData, mem);
}

std::size_t CEventRateBucketGatherer::memoryUsage(void) const {
    registerMemoryCallbacks();
    std::size_t mem = CBucketGatherer::memoryUsage();
    mem += core::CMemory::dynamicSize(m_FieldNames);
    mem += core::CMemory::dynamicSize(m_FeatureData);
    return mem;
}

std::size_t CEventRateBucketGatherer::staticSize(void) const {
    return sizeof(*this);
}

void CEventRateBucketGatherer::clear(void) {
    this->CBucketGatherer::clear();
    m_FeatureData.clear();
    this->initializeFeatureData();
}

bool CEventRateBucketGatherer::resetBucket(core_t::TTime bucketStart) {
    return this->CBucketGatherer::resetBucket(bucketStart);
}

void CEventRateBucketGatherer::releaseMemory(core_t::TTime /*samplingCutoffTime*/) {
    // Nothing to release
}

void CEventRateBucketGatherer::sample(core_t::TTime time) {
    // Merge smallest bucket into longer buckets, if they exist
    this->CBucketGatherer::sample(time);
}

void CEventRateBucketGatherer::featureData(core_t::TTime time, core_t::TTime /*bucketLength*/, TFeatureAnyPrVec& result) const {
    result.clear();

    if (!this->dataAvailable(time) || time >= this->currentBucketStartTime() + this->bucketLength()) {
        LOG_DEBUG("No data available at " << time << ", current bucket = " << this->printCurrentBucket());
        return;
    }

    for (std::size_t i = 0u, n = m_DataGatherer.numberFeatures(); i < n; ++i) {
        const model_t::EFeature feature = m_DataGatherer.feature(i);

        switch (feature) {
        case model_t::E_IndividualCountByBucketAndPerson:
            this->personCounts(feature, time, result);
            break;
        case model_t::E_IndividualNonZeroCountByBucketAndPerson:
        case model_t::E_IndividualTotalBucketCountByPerson:
            this->nonZeroPersonCounts(feature, time, result);
            break;
        case model_t::E_IndividualIndicatorOfBucketPerson:
            this->personIndicator(feature, time, result);
            break;
        case model_t::E_IndividualLowCountsByBucketAndPerson:
        case model_t::E_IndividualHighCountsByBucketAndPerson:
            this->personCounts(feature, time, result);
            break;
        case model_t::E_IndividualArrivalTimesByPerson:
        case model_t::E_IndividualLongArrivalTimesByPerson:
        case model_t::E_IndividualShortArrivalTimesByPerson:
            this->personArrivalTimes(feature, time, result);
            break;
        case model_t::E_IndividualLowNonZeroCountByBucketAndPerson:
        case model_t::E_IndividualHighNonZeroCountByBucketAndPerson:
            this->nonZeroPersonCounts(feature, time, result);
            break;
        case model_t::E_IndividualUniqueCountByBucketAndPerson:
        case model_t::E_IndividualLowUniqueCountByBucketAndPerson:
        case model_t::E_IndividualHighUniqueCountByBucketAndPerson:
            this->bucketUniqueValuesPerPerson(feature, time, result);
            break;
        case model_t::E_IndividualInfoContentByBucketAndPerson:
        case model_t::E_IndividualHighInfoContentByBucketAndPerson:
        case model_t::E_IndividualLowInfoContentByBucketAndPerson:
            this->bucketCompressedLengthPerPerson(feature, time, result);
            break;
        case model_t::E_IndividualTimeOfDayByBucketAndPerson:
        case model_t::E_IndividualTimeOfWeekByBucketAndPerson:
            this->bucketMeanTimesPerPerson(feature, time, result);
            break;

        CASE_INDIVIDUAL_METRIC:
            LOG_ERROR("Unexpected feature = " << model_t::print(feature));
            break;

        case model_t::E_PopulationAttributeTotalCountByPerson:
        case model_t::E_PopulationCountByBucketPersonAndAttribute:
            this->nonZeroAttributeCounts(feature, time, result);
            break;
        case model_t::E_PopulationIndicatorOfBucketPersonAndAttribute:
            this->attributeIndicator(feature, time, result);
            break;
        case model_t::E_PopulationUniquePersonCountByAttribute:
            this->peoplePerAttribute(feature, result);
            break;
        case model_t::E_PopulationUniqueCountByBucketPersonAndAttribute:
        case model_t::E_PopulationLowUniqueCountByBucketPersonAndAttribute:
        case model_t::E_PopulationHighUniqueCountByBucketPersonAndAttribute:
            this->bucketUniqueValuesPerPersonAttribute(feature, time, result);
            break;
        case model_t::E_PopulationLowCountsByBucketPersonAndAttribute:
        case model_t::E_PopulationHighCountsByBucketPersonAndAttribute:
            this->nonZeroAttributeCounts(feature, time, result);
            break;
        case model_t::E_PopulationInfoContentByBucketPersonAndAttribute:
        case model_t::E_PopulationLowInfoContentByBucketPersonAndAttribute:
        case model_t::E_PopulationHighInfoContentByBucketPersonAndAttribute:
            this->bucketCompressedLengthPerPersonAttribute(feature, time, result);
            break;
        case model_t::E_PopulationTimeOfDayByBucketPersonAndAttribute:
        case model_t::E_PopulationTimeOfWeekByBucketPersonAndAttribute:
            this->bucketMeanTimesPerPersonAttribute(feature, time, result);
            break;

        CASE_POPULATION_METRIC:
            LOG_ERROR("Unexpected feature = " << model_t::print(feature));
            break;

        case model_t::E_PeersAttributeTotalCountByPerson:
        case model_t::E_PeersCountByBucketPersonAndAttribute:
            this->nonZeroAttributeCounts(feature, time, result);
            break;
        case model_t::E_PeersUniqueCountByBucketPersonAndAttribute:
        case model_t::E_PeersLowUniqueCountByBucketPersonAndAttribute:
        case model_t::E_PeersHighUniqueCountByBucketPersonAndAttribute:
            this->bucketUniqueValuesPerPersonAttribute(feature, time, result);
            break;
        case model_t::E_PeersLowCountsByBucketPersonAndAttribute:
        case model_t::E_PeersHighCountsByBucketPersonAndAttribute:
            this->nonZeroAttributeCounts(feature, time, result);
            break;
        case model_t::E_PeersInfoContentByBucketPersonAndAttribute:
        case model_t::E_PeersLowInfoContentByBucketPersonAndAttribute:
        case model_t::E_PeersHighInfoContentByBucketPersonAndAttribute:
            this->bucketCompressedLengthPerPersonAttribute(feature, time, result);
            break;
        case model_t::E_PeersTimeOfDayByBucketPersonAndAttribute:
        case model_t::E_PeersTimeOfWeekByBucketPersonAndAttribute:
            this->bucketMeanTimesPerPersonAttribute(feature, time, result);
            break;

        CASE_PEERS_METRIC:
            LOG_ERROR("Unexpected feature = " << model_t::print(feature));
            break;
        }
    }
}

void CEventRateBucketGatherer::personCounts(model_t::EFeature feature, core_t::TTime time, TFeatureAnyPrVec& result_) const {
    if (m_DataGatherer.isPopulation()) {
        LOG_ERROR("Function does not support population analysis.");
        return;
    }

    result_.emplace_back(feature, TSizeFeatureDataPrVec());
    auto& result = *boost::unsafe_any_cast<TSizeFeatureDataPrVec>(&result_.back().second);
    result.reserve(m_DataGatherer.numberActivePeople());

    for (std::size_t pid = 0u, n = m_DataGatherer.numberPeople(); pid < n; ++pid) {
        if (!m_DataGatherer.isPersonActive(pid) || this->hasExplicitNullsOnly(time, pid, model_t::INDIVIDUAL_ANALYSIS_ATTRIBUTE_ID)) {
            continue;
        }
        result.emplace_back(pid, 0);
    }

    for (const auto& count_ : this->bucketCounts(time)) {
        uint64_t& count =
            std::lower_bound(result.begin(), result.end(), CDataGatherer::extractPersonId(count_), maths::COrderings::SFirstLess())
                ->second.s_Count;
        count += CDataGatherer::extractData(count_);
    }

    this->addInfluencerCounts(time, result);
}

void CEventRateBucketGatherer::nonZeroPersonCounts(model_t::EFeature feature, core_t::TTime time, TFeatureAnyPrVec& result_) const {
    result_.emplace_back(feature, TSizeFeatureDataPrVec());
    auto& result = *boost::unsafe_any_cast<TSizeFeatureDataPrVec>(&result_.back().second);

    const TSizeSizePrUInt64UMap& personAttributeCounts = this->bucketCounts(time);
    result.reserve(personAttributeCounts.size());
    for (const auto& count : personAttributeCounts) {
        result.emplace_back(CDataGatherer::extractPersonId(count), CDataGatherer::extractData(count));
    }
    std::sort(result.begin(), result.end(), maths::COrderings::SFirstLess());

    this->addInfluencerCounts(time, result);
}

void CEventRateBucketGatherer::personIndicator(model_t::EFeature feature, core_t::TTime time, TFeatureAnyPrVec& result_) const {
    result_.emplace_back(feature, TSizeFeatureDataPrVec());
    auto& result = *boost::unsafe_any_cast<TSizeFeatureDataPrVec>(&result_.back().second);

    const TSizeSizePrUInt64UMap& personAttributeCounts = this->bucketCounts(time);
    result.reserve(personAttributeCounts.size());
    for (const auto& count : personAttributeCounts) {
        result.emplace_back(CDataGatherer::extractPersonId(count), 1);
    }
    std::sort(result.begin(), result.end(), maths::COrderings::SFirstLess());

    this->addInfluencerCounts(time, result);
}

void CEventRateBucketGatherer::personArrivalTimes(model_t::EFeature feature, core_t::TTime /*time*/, TFeatureAnyPrVec& result_) const {
    // TODO
    result_.emplace_back(feature, TSizeFeatureDataPrVec());
}

void CEventRateBucketGatherer::nonZeroAttributeCounts(model_t::EFeature feature, core_t::TTime time, TFeatureAnyPrVec& result_) const {
    result_.emplace_back(feature, TSizeSizePrFeatureDataPrVec());
    auto& result = *boost::unsafe_any_cast<TSizeSizePrFeatureDataPrVec>(&result_.back().second);

    const TSizeSizePrUInt64UMap& personAttributeCounts = this->bucketCounts(time);
    result.reserve(personAttributeCounts.size());
    for (const auto& count : personAttributeCounts) {
        if (CDataGatherer::extractData(count) > 0) {
            result.emplace_back(count.first, CDataGatherer::extractData(count));
        }
    }
    std::sort(result.begin(), result.end(), maths::COrderings::SFirstLess());

    this->addInfluencerCounts(time, result);
}

void CEventRateBucketGatherer::peoplePerAttribute(model_t::EFeature feature, TFeatureAnyPrVec& result_) const {
    result_.emplace_back(feature, TSizeSizePrFeatureDataPrVec());
    auto& result = *boost::unsafe_any_cast<TSizeSizePrFeatureDataPrVec>(&result_.back().second);

    auto i = m_FeatureData.find(model_t::E_AttributePeople);
    if (i == m_FeatureData.end()) {
        return;
    }

    try {
        const TSizeUSetVec& attributePeople = boost::any_cast<const TSizeUSetVec&>(i->second);
        result.reserve(attributePeople.size());
        for (std::size_t cid = 0u; cid < attributePeople.size(); ++cid) {
            if (m_DataGatherer.isAttributeActive(cid)) {
                result.emplace_back(TSizeSizePr(0, cid), attributePeople[cid].size());
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to extract " << model_t::print(model_t::E_PopulationUniquePersonCountByAttribute) << ": " << e.what());
    }
}

void CEventRateBucketGatherer::attributeIndicator(model_t::EFeature feature, core_t::TTime time, TFeatureAnyPrVec& result_) const {
    result_.emplace_back(feature, TSizeSizePrFeatureDataPrVec());
    auto& result = *boost::unsafe_any_cast<TSizeSizePrFeatureDataPrVec>(&result_.back().second);

    const TSizeSizePrUInt64UMap& counts = this->bucketCounts(time);
    result.reserve(counts.size());
    for (const auto& count : counts) {
        if (CDataGatherer::extractData(count) > 0) {
            result.emplace_back(count.first, 1);
        }
    }
    std::sort(result.begin(), result.end(), maths::COrderings::SFirstLess());

    this->addInfluencerCounts(time, result);
    for (std::size_t i = 0u; i < result.size(); ++i) {
        SEventRateFeatureData& data = result[i].second;
        for (std::size_t j = 0u; j < data.s_InfluenceValues.size(); ++j) {
            for (std::size_t k = 0u; k < data.s_InfluenceValues[j].size(); ++k) {
                data.s_InfluenceValues[j][k].second.first = TDoubleVec{1.0};
            }
        }
    }
}

void CEventRateBucketGatherer::bucketUniqueValuesPerPerson(model_t::EFeature feature, core_t::TTime time, TFeatureAnyPrVec& result_) const {
    result_.emplace_back(feature, TSizeFeatureDataPrVec());
    auto& result = *boost::unsafe_any_cast<TSizeFeatureDataPrVec>(&result_.back().second);

    auto i = m_FeatureData.find(model_t::E_UniqueValues);
    if (i == m_FeatureData.end()) {
        return;
    }

    try {
        const auto& personAttributeUniqueValues = boost::any_cast<const TSizeSizePrStrDataUMapQueue&>(i->second).get(time);
        result.reserve(personAttributeUniqueValues.size());
        for (const auto& uniques : personAttributeUniqueValues) {
            result.emplace_back(CDataGatherer::extractPersonId(uniques), 0);
            CDataGatherer::extractData(uniques).populateDistinctCountFeatureData(result.back().second);
        }
        std::sort(result.begin(), result.end(), maths::COrderings::SFirstLess());
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to extract " << model_t::print(model_t::E_IndividualUniqueCountByBucketAndPerson) << ": " << e.what());
    }
}

void CEventRateBucketGatherer::bucketUniqueValuesPerPersonAttribute(model_t::EFeature feature,
                                                                    core_t::TTime time,
                                                                    TFeatureAnyPrVec& result_) const {
    result_.emplace_back(feature, TSizeSizePrFeatureDataPrVec());
    auto& result = *boost::unsafe_any_cast<TSizeSizePrFeatureDataPrVec>(&result_.back().second);

    auto i = m_FeatureData.find(model_t::E_UniqueValues);
    if (i == m_FeatureData.end()) {
        return;
    }

    try {
        const auto& personAttributeUniqueValues = boost::any_cast<const TSizeSizePrStrDataUMapQueue&>(i->second).get(time);
        result.reserve(personAttributeUniqueValues.size());
        for (const auto& uniques : personAttributeUniqueValues) {
            result.emplace_back(uniques.first, 0);
            CDataGatherer::extractData(uniques).populateDistinctCountFeatureData(result.back().second);
        }
        std::sort(result.begin(), result.end(), maths::COrderings::SFirstLess());
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to extract " << model_t::print(model_t::E_PopulationUniqueCountByBucketPersonAndAttribute) << ": " << e.what());
    }
}

void CEventRateBucketGatherer::bucketCompressedLengthPerPerson(model_t::EFeature feature,
                                                               core_t::TTime time,
                                                               TFeatureAnyPrVec& result_) const {
    result_.emplace_back(feature, TSizeFeatureDataPrVec());
    auto& result = *boost::unsafe_any_cast<TSizeFeatureDataPrVec>(&result_.back().second);

    auto i = m_FeatureData.find(model_t::E_UniqueValues);
    if (i == m_FeatureData.end()) {
        return;
    }

    try {
        const auto& personAttributeUniqueValues = boost::any_cast<const TSizeSizePrStrDataUMapQueue&>(i->second).get(time);
        result.reserve(personAttributeUniqueValues.size());
        for (const auto& uniques : personAttributeUniqueValues) {
            result.emplace_back(CDataGatherer::extractPersonId(uniques), 0);
            CDataGatherer::extractData(uniques).populateInfoContentFeatureData(result.back().second);
        }
        std::sort(result.begin(), result.end(), maths::COrderings::SFirstLess());
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to extract " << model_t::print(model_t::E_IndividualInfoContentByBucketAndPerson) << ": " << e.what());
    }
}

void CEventRateBucketGatherer::bucketCompressedLengthPerPersonAttribute(model_t::EFeature feature,
                                                                        core_t::TTime time,
                                                                        TFeatureAnyPrVec& result_) const {
    result_.emplace_back(feature, TSizeSizePrFeatureDataPrVec());
    auto& result = *boost::unsafe_any_cast<TSizeSizePrFeatureDataPrVec>(&result_.back().second);

    auto i = m_FeatureData.find(model_t::E_UniqueValues);
    if (i == m_FeatureData.end()) {
        return;
    }

    try {
        const auto& personAttributeUniqueValues = boost::any_cast<const TSizeSizePrStrDataUMapQueue&>(i->second).get(time);
        result.reserve(personAttributeUniqueValues.size());
        for (const auto& uniques : personAttributeUniqueValues) {
            result.emplace_back(uniques.first, 0);
            CDataGatherer::extractData(uniques).populateInfoContentFeatureData(result.back().second);
        }
        std::sort(result.begin(), result.end(), maths::COrderings::SFirstLess());
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to extract " << model_t::print(model_t::E_PopulationInfoContentByBucketPersonAndAttribute) << ": " << e.what());
    }
}

void CEventRateBucketGatherer::bucketMeanTimesPerPerson(model_t::EFeature feature, core_t::TTime time, TFeatureAnyPrVec& result_) const {
    result_.emplace_back(feature, TSizeFeatureDataPrVec());
    auto& result = *boost::unsafe_any_cast<TSizeFeatureDataPrVec>(&result_.back().second);

    auto i = m_FeatureData.find(model_t::E_DiurnalTimes);
    if (i == m_FeatureData.end()) {
        return;
    }

    try {
        const auto& arrivalTimes = boost::any_cast<const TSizeSizePrMeanAccumulatorUMapQueue&>(i->second).get(time);
        result.reserve(arrivalTimes.size());
        for (const auto& time_ : arrivalTimes) {
            result.emplace_back(CDataGatherer::extractPersonId(time_),
                                static_cast<uint64_t>(maths::CBasicStatistics::mean(CDataGatherer::extractData(time_))));
        }
        std::sort(result.begin(), result.end(), maths::COrderings::SFirstLess());

        // We don't bother to gather the influencer bucket means
        // so the best we can do is use the person and attribute
        // bucket mean.
        this->addInfluencerCounts(time, result);
        for (std::size_t j = 0u; j < result.size(); ++j) {
            SEventRateFeatureData& data = result[j].second;
            for (std::size_t k = 0u; k < data.s_InfluenceValues.size(); ++k) {
                for (std::size_t l = 0u; l < data.s_InfluenceValues[k].size(); ++l) {
                    data.s_InfluenceValues[k][l].second.first = TDouble1Vec{static_cast<double>(data.s_Count)};
                }
            }
        }
    } catch (const std::exception& e) { LOG_ERROR("Failed to extract " << model_t::print(model_t::E_DiurnalTimes) << ": " << e.what()); }
}

void CEventRateBucketGatherer::bucketMeanTimesPerPersonAttribute(model_t::EFeature feature,
                                                                 core_t::TTime time,
                                                                 TFeatureAnyPrVec& result_) const {
    result_.emplace_back(feature, TSizeSizePrFeatureDataPrVec());
    auto& result = *boost::unsafe_any_cast<TSizeSizePrFeatureDataPrVec>(&result_.back().second);

    auto i = m_FeatureData.find(model_t::E_DiurnalTimes);
    if (i == m_FeatureData.end()) {
        return;
    }

    try {
        const auto& arrivalTimes = boost::any_cast<const TSizeSizePrMeanAccumulatorUMapQueue&>(i->second).get(time);
        result.reserve(arrivalTimes.size());
        for (const auto& time_ : arrivalTimes) {
            result.emplace_back(time_.first, static_cast<uint64_t>(maths::CBasicStatistics::mean(CDataGatherer::extractData(time_))));
        }
        std::sort(result.begin(), result.end(), maths::COrderings::SFirstLess());

        // We don't bother to gather the influencer bucket means
        // so the best we can do is use the person and attribute
        // bucket mean.
        this->addInfluencerCounts(time, result);
        for (std::size_t j = 0u; j < result.size(); ++j) {
            SEventRateFeatureData& data = result[j].second;
            for (std::size_t k = 0u; k < data.s_InfluenceValues.size(); ++k) {
                for (std::size_t l = 0u; l < data.s_InfluenceValues[k].size(); ++l) {
                    data.s_InfluenceValues[k][l].second.first = TDouble1Vec{static_cast<double>(data.s_Count)};
                }
            }
        }
    } catch (const std::exception& e) { LOG_ERROR("Failed to extract " << model_t::print(model_t::E_DiurnalTimes) << ": " << e.what()); }
}

void CEventRateBucketGatherer::resize(std::size_t pid, std::size_t cid) {
    apply(m_FeatureData, boost::bind<void>(SResize(), _1, pid, cid));
}

void CEventRateBucketGatherer::addValue(std::size_t pid,
                                        std::size_t cid,
                                        core_t::TTime time,
                                        const CEventData::TDouble1VecArray& values,
                                        std::size_t count,
                                        const CEventData::TOptionalStr& stringValue,
                                        const TStoredStringPtrVec& influences) {
    // Check that we are correctly sized - a person/attribute might have been added
    this->resize(pid, cid);
    apply(
        m_FeatureData,
        boost::bind<void>(SAddValue(), _1, pid, cid, time, count, boost::cref(values), boost::cref(stringValue), boost::cref(influences)));
}

void CEventRateBucketGatherer::startNewBucket(core_t::TTime time, bool /*skipUpdates*/) {
    apply(m_FeatureData, boost::bind<void>(SNewBucket(), _1, time));
}

void CEventRateBucketGatherer::initializeFieldNames(const std::string& personFieldName,
                                                    const std::string& attributeFieldName,
                                                    const std::string& valueFieldName,
                                                    const std::string& summaryCountFieldName,
                                                    const TStrVec& influenceFieldNames) {
    m_FieldNames.push_back(personFieldName);
    if (m_DataGatherer.isPopulation()) {
        m_FieldNames.push_back(attributeFieldName);
    }

    m_BeginInfluencingFields = m_FieldNames.size();
    m_FieldNames.insert(m_FieldNames.end(), influenceFieldNames.begin(), influenceFieldNames.end());

    m_BeginValueField = m_FieldNames.size();
    if (!valueFieldName.empty()) {
        m_FieldNames.push_back(valueFieldName);
    }

    m_BeginSummaryFields = m_FieldNames.size();
    switch (m_DataGatherer.summaryMode()) {
    case model_t::E_None:
        break;
    case model_t::E_Manual:
        m_FieldNames.push_back(summaryCountFieldName);
        break;
    };

    // swap trick to reduce unused capacity
    TStrVec(m_FieldNames).swap(m_FieldNames);
}

void CEventRateBucketGatherer::initializeFeatureData(void) {
    for (std::size_t i = 0u, n = m_DataGatherer.numberFeatures(); i < n; ++i) {
        switch (m_DataGatherer.feature(i)) {
        case model_t::E_IndividualCountByBucketAndPerson:
        case model_t::E_IndividualNonZeroCountByBucketAndPerson:
        case model_t::E_IndividualTotalBucketCountByPerson:
        case model_t::E_IndividualIndicatorOfBucketPerson:
        case model_t::E_IndividualLowCountsByBucketAndPerson:
        case model_t::E_IndividualHighCountsByBucketAndPerson:
            // We always gather person counts.
            break;
        case model_t::E_IndividualArrivalTimesByPerson:
        case model_t::E_IndividualLongArrivalTimesByPerson:
        case model_t::E_IndividualShortArrivalTimesByPerson:
            // TODO
            break;
        case model_t::E_IndividualTimeOfDayByBucketAndPerson:
        case model_t::E_IndividualTimeOfWeekByBucketAndPerson:
            m_FeatureData[model_t::E_DiurnalTimes] = TSizeSizePrMeanAccumulatorUMapQueue(
                m_DataGatherer.params().s_LatencyBuckets, this->bucketLength(), this->currentBucketStartTime());
            break;

        case model_t::E_IndividualLowNonZeroCountByBucketAndPerson:
        case model_t::E_IndividualHighNonZeroCountByBucketAndPerson:
            // We always gather person counts.
            break;
        case model_t::E_IndividualUniqueCountByBucketAndPerson:
        case model_t::E_IndividualLowUniqueCountByBucketAndPerson:
        case model_t::E_IndividualHighUniqueCountByBucketAndPerson:
        case model_t::E_IndividualInfoContentByBucketAndPerson:
        case model_t::E_IndividualHighInfoContentByBucketAndPerson:
        case model_t::E_IndividualLowInfoContentByBucketAndPerson:
            m_FeatureData[model_t::E_UniqueValues] = TSizeSizePrStrDataUMapQueue(
                m_DataGatherer.params().s_LatencyBuckets, this->bucketLength(), this->currentBucketStartTime(), TSizeSizePrStrDataUMap(1));
            break;

        case model_t::E_PopulationAttributeTotalCountByPerson:
        case model_t::E_PopulationCountByBucketPersonAndAttribute:
        case model_t::E_PopulationIndicatorOfBucketPersonAndAttribute:
        case model_t::E_PopulationLowCountsByBucketPersonAndAttribute:
        case model_t::E_PopulationHighCountsByBucketPersonAndAttribute:
            // We always gather person attribute counts.
            break;
        case model_t::E_PopulationUniquePersonCountByAttribute:
            m_FeatureData[model_t::E_AttributePeople] = TSizeUSetVec();
            break;
        case model_t::E_PopulationUniqueCountByBucketPersonAndAttribute:
        case model_t::E_PopulationLowUniqueCountByBucketPersonAndAttribute:
        case model_t::E_PopulationHighUniqueCountByBucketPersonAndAttribute:
        case model_t::E_PopulationInfoContentByBucketPersonAndAttribute:
        case model_t::E_PopulationLowInfoContentByBucketPersonAndAttribute:
        case model_t::E_PopulationHighInfoContentByBucketPersonAndAttribute:
            m_FeatureData[model_t::E_UniqueValues] = TSizeSizePrStrDataUMapQueue(
                m_DataGatherer.params().s_LatencyBuckets, this->bucketLength(), this->currentBucketStartTime(), TSizeSizePrStrDataUMap(1));
            break;
        case model_t::E_PopulationTimeOfDayByBucketPersonAndAttribute:
        case model_t::E_PopulationTimeOfWeekByBucketPersonAndAttribute:
            m_FeatureData[model_t::E_DiurnalTimes] = TSizeSizePrMeanAccumulatorUMapQueue(
                m_DataGatherer.params().s_LatencyBuckets, this->bucketLength(), this->currentBucketStartTime());
            break;

        case model_t::E_PeersAttributeTotalCountByPerson:
        case model_t::E_PeersCountByBucketPersonAndAttribute:
        case model_t::E_PeersLowCountsByBucketPersonAndAttribute:
        case model_t::E_PeersHighCountsByBucketPersonAndAttribute:
            // We always gather person attribute counts.
            break;
        case model_t::E_PeersUniqueCountByBucketPersonAndAttribute:
        case model_t::E_PeersLowUniqueCountByBucketPersonAndAttribute:
        case model_t::E_PeersHighUniqueCountByBucketPersonAndAttribute:
        case model_t::E_PeersInfoContentByBucketPersonAndAttribute:
        case model_t::E_PeersLowInfoContentByBucketPersonAndAttribute:
        case model_t::E_PeersHighInfoContentByBucketPersonAndAttribute:
            m_FeatureData[model_t::E_UniqueValues] = TSizeSizePrStrDataUMapQueue(
                m_DataGatherer.params().s_LatencyBuckets, this->bucketLength(), this->currentBucketStartTime(), TSizeSizePrStrDataUMap(1));
            break;
        case model_t::E_PeersTimeOfDayByBucketPersonAndAttribute:
        case model_t::E_PeersTimeOfWeekByBucketPersonAndAttribute:
            m_FeatureData[model_t::E_DiurnalTimes] = TSizeSizePrMeanAccumulatorUMapQueue(
                m_DataGatherer.params().s_LatencyBuckets, this->bucketLength(), this->currentBucketStartTime());
            break;

        CASE_INDIVIDUAL_METRIC:
        CASE_POPULATION_METRIC:
        CASE_PEERS_METRIC:
            LOG_ERROR("Unexpected feature = " << model_t::print(m_DataGatherer.feature(i)))
            break;
        }
    }
}

void CEventRateBucketGatherer::addInfluencerCounts(core_t::TTime time, TSizeFeatureDataPrVec& result) const {
    const TSizeSizePrStoredStringPtrPrUInt64UMapVec& influencers = this->influencerCounts(time);
    if (influencers.empty()) {
        return;
    }

    for (std::size_t i = 0u; i < result.size(); ++i) {
        result[i].second.s_InfluenceValues.resize(influencers.size());
    }

    for (std::size_t i = 0u; i < influencers.size(); ++i) {
        for (const auto& influence : influencers[i]) {
            std::size_t pid = CDataGatherer::extractPersonId(influence.first);
            auto k = std::lower_bound(result.begin(), result.end(), pid, maths::COrderings::SFirstLess());
            if (k == result.end() || k->first != pid) {
                LOG_ERROR("Missing feature data for person " << m_DataGatherer.personName(pid));
                continue;
            }
            k->second.s_InfluenceValues[i].emplace_back(TStrCRef(*CDataGatherer::extractData(influence.first)),
                                                        TDouble1VecDoublePr(TDouble1Vec{static_cast<double>(influence.second)}, 1.0));
        }
    }
}

void CEventRateBucketGatherer::addInfluencerCounts(core_t::TTime time, TSizeSizePrFeatureDataPrVec& result) const {
    const TSizeSizePrStoredStringPtrPrUInt64UMapVec& influencers = this->influencerCounts(time);
    if (influencers.empty()) {
        return;
    }

    for (std::size_t i = 0u; i < result.size(); ++i) {
        result[i].second.s_InfluenceValues.resize(influencers.size());
    }

    for (std::size_t i = 0u; i < influencers.size(); ++i) {
        for (const auto& influence : influencers[i]) {
            auto k = std::lower_bound(result.begin(), result.end(), influence.first.first, maths::COrderings::SFirstLess());
            if (k == result.end() || k->first != influence.first.first) {
                std::size_t pid = CDataGatherer::extractPersonId(influence.first);
                std::size_t cid = CDataGatherer::extractAttributeId(influence.first);
                LOG_ERROR("Missing feature data for person " << m_DataGatherer.personName(pid) << " and attribute "
                                                             << m_DataGatherer.attributeName(cid));
                continue;
            }
            k->second.s_InfluenceValues[i].emplace_back(TStrCRef(*CDataGatherer::extractData(influence.first)),
                                                        TDouble1VecDoublePr(TDouble1Vec{static_cast<double>(influence.second)}, 1.0));
        }
    }
}

////// CUniqueStringFeatureData //////

void CUniqueStringFeatureData::insert(const std::string& value, const TStoredStringPtrVec& influences) {
    TWord valueHash = m_Dictionary1.word(value);
    m_UniqueStrings.emplace(valueHash, value);
    if (influences.size() > m_InfluencerUniqueStrings.size()) {
        m_InfluencerUniqueStrings.resize(influences.size());
    }
    for (std::size_t i = 0; i < influences.size(); ++i) {
        // The influence strings are optional.
        if (influences[i]) {
            m_InfluencerUniqueStrings[i][influences[i]].insert(valueHash);
        }
    }
}

void CUniqueStringFeatureData::populateDistinctCountFeatureData(SEventRateFeatureData& featureData) const {
    featureData.s_Count = m_UniqueStrings.size();
    featureData.s_InfluenceValues.clear();
    featureData.s_InfluenceValues.resize(m_InfluencerUniqueStrings.size());
    for (std::size_t i = 0u; i < m_InfluencerUniqueStrings.size(); ++i) {
        TStrCRefDouble1VecDoublePrPrVec& data = featureData.s_InfluenceValues[i];
        data.reserve(m_InfluencerUniqueStrings[i].size());
        for (const auto& influence : m_InfluencerUniqueStrings[i]) {
            data.emplace_back(TStrCRef(*influence.first),
                              TDouble1VecDoublePr(TDouble1Vec{static_cast<double>(influence.second.size())}, 1.0));
        }
    }
}

void CUniqueStringFeatureData::populateInfoContentFeatureData(SEventRateFeatureData& featureData) const {
    typedef std::vector<TStrCRef> TStrCRefVec;

    featureData.s_InfluenceValues.clear();
    core::CCompressUtils compressor(true);

    try {
        TStrCRefVec strings;

        strings.reserve(m_UniqueStrings.size());
        for (const auto& string : m_UniqueStrings) {
            strings.emplace_back(string.second);
        }
        std::sort(strings.begin(), strings.end(), maths::COrderings::SLess());
        std::for_each(strings.begin(), strings.end(), [&compressor](const std::string& string) { compressor.addString(string); });

        std::size_t length = 0u;
        if (compressor.compressedLength(true, length) == false) {
            LOG_ERROR("Failed to get compressed length");
            compressor.reset();
        }
        featureData.s_Count = length;

        featureData.s_InfluenceValues.reserve(m_InfluencerUniqueStrings.size());
        for (std::size_t i = 0u; i < m_InfluencerUniqueStrings.size(); ++i) {
            featureData.s_InfluenceValues.push_back(TStrCRefDouble1VecDoublePrPrVec());
            TStrCRefDouble1VecDoublePrPrVec& data = featureData.s_InfluenceValues.back();
            for (const auto& influence : m_InfluencerUniqueStrings[i]) {
                strings.clear();
                strings.reserve(influence.second.size());
                for (const auto& word : influence.second) {
                    strings.emplace_back(m_UniqueStrings.at(word));
                }
                std::sort(strings.begin(), strings.end(), maths::COrderings::SLess());
                std::for_each(strings.begin(), strings.end(), [&compressor](const std::string& string) { compressor.addString(string); });
                length = 0u;
                if (compressor.compressedLength(true, length) == false) {
                    LOG_ERROR("Failed to get compressed length");
                    compressor.reset();
                }
                data.emplace_back(TStrCRef(*influence.first), TDouble1VecDoublePr(TDouble1Vec{static_cast<double>(length)}, 1.0));
            }
        }
    } catch (const std::exception& e) { LOG_ERROR("Failed to get info content: " << e.what()); }
}

void CUniqueStringFeatureData::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertLevel(UNIQUE_STRINGS_TAG, boost::bind(&persistUniqueStrings, boost::cref(m_UniqueStrings), _1));
    for (std::size_t i = 0u; i < m_InfluencerUniqueStrings.size(); ++i) {
        inserter.insertLevel(INFLUENCER_UNIQUE_STRINGS_TAG,
                             boost::bind(&persistInfluencerUniqueStrings, boost::cref(m_InfluencerUniqueStrings[i]), _1));
    }
}

bool CUniqueStringFeatureData::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE(UNIQUE_STRINGS_TAG, traverser.traverseSubLevel(boost::bind(&restoreUniqueStrings, _1, boost::ref(m_UniqueStrings))))
        RESTORE_SETUP_TEARDOWN(
            INFLUENCER_UNIQUE_STRINGS_TAG,
            m_InfluencerUniqueStrings.push_back(TStoredStringPtrWordSetUMap()),
            traverser.traverseSubLevel(boost::bind(&restoreInfluencerUniqueStrings, _1, boost::ref(m_InfluencerUniqueStrings.back()))),
            /**/)
    } while (traverser.next());

    return true;
}

uint64_t CUniqueStringFeatureData::checksum(void) const {
    uint64_t seed = maths::CChecksum::calculate(0, m_UniqueStrings);
    return maths::CChecksum::calculate(seed, m_InfluencerUniqueStrings);
}

void CUniqueStringFeatureData::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CUniqueStringFeatureData", sizeof(*this));
    core::CMemoryDebug::dynamicSize("s_NoInfluenceUniqueStrings", m_UniqueStrings, mem);
    core::CMemoryDebug::dynamicSize("s_InfluenceUniqueStrings", m_InfluencerUniqueStrings, mem);
}

std::size_t CUniqueStringFeatureData::memoryUsage(void) const {
    std::size_t mem = sizeof(*this);
    mem += core::CMemory::dynamicSize(m_UniqueStrings);
    mem += core::CMemory::dynamicSize(m_InfluencerUniqueStrings);
    return mem;
}

std::string CUniqueStringFeatureData::print(void) const {
    return "(" + core::CContainerPrinter::print(m_UniqueStrings) + ", " + core::CContainerPrinter::print(m_InfluencerUniqueStrings) + ")";
}
}
}
