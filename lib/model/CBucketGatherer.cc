/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CBucketGatherer.h>

#include <core/CContainerPrinter.h>
#include <core/CProgramCounters.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CStringUtils.h>
#include <core/RestoreMacros.h>

#include <maths/CChecksum.h>
#include <maths/CIntegerTools.h>
#include <maths/COrderings.h>

#include <model/CDataGatherer.h>
#include <model/CStringStore.h>

#include <algorithm>

namespace ml {
namespace model {

namespace {

// We use short field names to reduce the state size
const std::string BUCKET_START_TAG("b");
const std::string BUCKET_COUNT_TAG("k");
const std::string INFLUENCERS_COUNT_TAG("l");
const std::string BUCKET_EXPLICIT_NULLS_TAG("m");

namespace detail {

using TSizeSizePr = std::pair<std::size_t, std::size_t>;
using TSizeSizePrUInt64Pr = std::pair<TSizeSizePr, uint64_t>;
using TSizeSizePrStoredStringPtrPrUInt64UMap = CBucketGatherer::TSizeSizePrStoredStringPtrPrUInt64UMap;
using TSizeSizePrStoredStringPtrPrUInt64UMapCItr =
    CBucketGatherer::TSizeSizePrStoredStringPtrPrUInt64UMapCItr;

const std::string PERSON_ATTRIBUTE_COUNT_TAG("f");
const std::string PERSON_UID_TAG("a");
const std::string ATTRIBUTE_UID_TAG("b");
const std::string COUNT_TAG("c");
const std::string INFLUENCER_TAG("d");
const std::string INFLUENCE_ITEM_TAG("a");
const std::string INFLUENCE_COUNT_TAG("b");
const std::string EMPTY_MAP_TAG("e");

//! Persist a person, attribute and count tuple.
void insertPersonAttributeCounts(const TSizeSizePrUInt64Pr& tuple,
                                 core::CStatePersistInserter& inserter) {
    inserter.insertValue(PERSON_UID_TAG, CDataGatherer::extractPersonId(tuple));
    inserter.insertValue(ATTRIBUTE_UID_TAG, CDataGatherer::extractAttributeId(tuple));
    inserter.insertValue(COUNT_TAG, CDataGatherer::extractData(tuple));
}

//! Restore a person, attribute and count.
bool restorePersonAttributeCounts(core::CStateRestoreTraverser& traverser,
                                  TSizeSizePr& key,
                                  uint64_t& count) {
    do {
        const std::string& name = traverser.name();
        RESTORE_BUILT_IN(PERSON_UID_TAG, key.first)
        RESTORE_BUILT_IN(ATTRIBUTE_UID_TAG, key.second)
        RESTORE_BUILT_IN(COUNT_TAG, count)
    } while (traverser.next());
    return true;
}

//! Persist a collection of influencer person and attribute counts.
void insertInfluencerPersonAttributeCounts(const TSizeSizePrStoredStringPtrPrUInt64UMap& map,
                                           core::CStatePersistInserter& inserter) {
    std::vector<TSizeSizePrStoredStringPtrPrUInt64UMapCItr> ordered;
    ordered.reserve(map.size());
    for (auto i = map.begin(); i != map.end(); ++i) {
        ordered.push_back(i);
    }
    std::sort(ordered.begin(), ordered.end(),
              [](TSizeSizePrStoredStringPtrPrUInt64UMapCItr lhs,
                 TSizeSizePrStoredStringPtrPrUInt64UMapCItr rhs) {
                  return maths::COrderings::lexicographical_compare(
                      lhs->first.first, *lhs->first.second, lhs->second,
                      rhs->first.first, *rhs->first.second, rhs->second);
              });

    if (ordered.empty()) {
        inserter.insertValue(EMPTY_MAP_TAG, "");
    }
    for (auto& pair : ordered) {
        inserter.insertValue(PERSON_UID_TAG, CDataGatherer::extractPersonId(pair->first));
        inserter.insertValue(ATTRIBUTE_UID_TAG,
                             CDataGatherer::extractAttributeId(pair->first));
        inserter.insertValue(INFLUENCER_TAG, *CDataGatherer::extractData(pair->first));
        inserter.insertValue(COUNT_TAG, pair->second);
    }
}

//! Restore a collection of influencer person and attribute counts.
bool restoreInfluencerPersonAttributeCounts(core::CStateRestoreTraverser& traverser,
                                            TSizeSizePrStoredStringPtrPrUInt64UMap& map) {
    std::size_t person = 0;
    std::size_t attribute = 0;
    std::string influence = "";
    uint64_t count = 0;
    do {
        const std::string name = traverser.name();
        RESTORE_BUILT_IN(PERSON_UID_TAG, person)
        RESTORE_BUILT_IN(ATTRIBUTE_UID_TAG, attribute)
        RESTORE_NO_ERROR(INFLUENCER_TAG, influence = traverser.value())
        if (name == COUNT_TAG) {
            if (core::CStringUtils::stringToType(traverser.value(), count) == false) {
                LOG_ERROR(<< "Failed to restore COUNT_TAG, got " << traverser.value());
                return false;
            }
            map[{{person, attribute}, CStringStore::influencers().get(influence)}] = count;
        }
    } while (traverser.next());
    return true;
}

//! \brief Manages persistence of bucket counts.
struct SBucketCountsPersister {
    using TSizeSizePrUInt64UMap = CBucketGatherer::TSizeSizePrUInt64UMap;

    void operator()(const TSizeSizePrUInt64UMap& bucketCounts,
                    core::CStatePersistInserter& inserter) {
        CBucketGatherer::TSizeSizePrUInt64PrVec personAttributeCounts;
        personAttributeCounts.reserve(bucketCounts.size());
        personAttributeCounts.assign(bucketCounts.begin(), bucketCounts.end());
        std::sort(personAttributeCounts.begin(), personAttributeCounts.end());
        for (std::size_t i = 0; i < personAttributeCounts.size(); ++i) {
            inserter.insertLevel(PERSON_ATTRIBUTE_COUNT_TAG,
                                 std::bind(&insertPersonAttributeCounts,
                                           std::cref(personAttributeCounts[i]),
                                           std::placeholders::_1));
        }
    }

    bool operator()(TSizeSizePrUInt64UMap& bucketCounts,
                    core::CStateRestoreTraverser& traverser) {
        do {
            TSizeSizePr key;
            uint64_t count{0u};
            if (!traverser.hasSubLevel()) {
                continue;
            }
            if (traverser.traverseSubLevel(
                    std::bind(&restorePersonAttributeCounts, std::placeholders::_1,
                              std::ref(key), std::ref(count))) == false) {
                LOG_ERROR(<< "Invalid person attribute count");
                continue;
            }
            bucketCounts[key] = count;
        } while (traverser.next());
        return true;
    }
};

//! \brief Manages persistence influencer bucket counts.
struct SInfluencerCountsPersister {
    using TSizeSizePrStoredStringPtrPrUInt64UMapVec =
        CBucketGatherer::TSizeSizePrStoredStringPtrPrUInt64UMapVec;

    void operator()(const TSizeSizePrStoredStringPtrPrUInt64UMapVec& data,
                    core::CStatePersistInserter& inserter) {
        for (std::size_t i = 0; i < data.size(); ++i) {
            inserter.insertValue(INFLUENCE_COUNT_TAG, i);
            inserter.insertLevel(INFLUENCE_ITEM_TAG,
                                 std::bind(&insertInfluencerPersonAttributeCounts,
                                           std::cref(data[i]), std::placeholders::_1));
        }
    }

    bool operator()(TSizeSizePrStoredStringPtrPrUInt64UMapVec& data,
                    core::CStateRestoreTraverser& traverser) const {
        std::size_t i = 0;
        do {
            const std::string name = traverser.name();
            RESTORE_BUILT_IN(INFLUENCE_COUNT_TAG, i)
            RESTORE_SETUP_TEARDOWN(
                INFLUENCE_ITEM_TAG, data.resize(std::max(data.size(), i + 1)),
                traverser.traverseSubLevel(
                    std::bind(&restoreInfluencerPersonAttributeCounts,
                              std::placeholders::_1, std::ref(data[i]))),
                /**/)
        } while (traverser.next());
        return true;
    }
};

} // detail::
} // unnamed::

const std::string CBucketGatherer::EVENTRATE_BUCKET_GATHERER_TAG("a");
const std::string CBucketGatherer::METRIC_BUCKET_GATHERER_TAG("b");

CBucketGatherer::CBucketGatherer(CDataGatherer& dataGatherer,
                                 core_t::TTime startTime,
                                 std::size_t numberInfluencers)
    : m_DataGatherer(dataGatherer), m_EarliestTime(startTime), m_BucketStart(startTime),
      m_PersonAttributeCounts(dataGatherer.params().s_LatencyBuckets,
                              dataGatherer.params().s_BucketLength,
                              startTime,
                              TSizeSizePrUInt64UMap(1)),
      m_PersonAttributeExplicitNulls(dataGatherer.params().s_LatencyBuckets,
                                     dataGatherer.params().s_BucketLength,
                                     startTime,
                                     TSizeSizePrUSet(1)),
      m_InfluencerCounts(dataGatherer.params().s_LatencyBuckets + 3,
                         dataGatherer.params().s_BucketLength,
                         startTime,
                         TSizeSizePrStoredStringPtrPrUInt64UMapVec(numberInfluencers)) {
}

CBucketGatherer::CBucketGatherer(bool isForPersistence, const CBucketGatherer& other)
    : m_DataGatherer(other.m_DataGatherer),
      m_EarliestTime(other.m_EarliestTime), m_BucketStart(other.m_BucketStart),
      m_PersonAttributeCounts(other.m_PersonAttributeCounts),
      m_PersonAttributeExplicitNulls(other.m_PersonAttributeExplicitNulls),
      m_InfluencerCounts(other.m_InfluencerCounts) {
    if (!isForPersistence) {
        LOG_ABORT(<< "This constructor only creates clones for persistence");
    }
}

bool CBucketGatherer::addEventData(CEventData& data) {
    core_t::TTime time = data.time();

    if (time < this->earliestBucketStartTime()) {
        // Ignore records that are out of the latency window
        // Records in an incomplete first bucket will end up here
        LOG_TRACE(<< "Ignored = " << time);
        return false;
    }

    this->timeNow(time);

    if (!data.personId() || !data.attributeId() || !data.count()) {
        // The record was incomplete.
        return false;
    }

    std::size_t pid = *data.personId();
    std::size_t cid = *data.attributeId();
    std::size_t count = *data.count();
    if ((pid != CDynamicStringIdRegistry::INVALID_ID) &&
        (cid != CDynamicStringIdRegistry::INVALID_ID)) {
        // Has the person/attribute been deleted from the gatherer?
        if (!m_DataGatherer.isPersonActive(pid)) {
            LOG_DEBUG(<< "Not adding value for deleted person " << pid);
            return false;
        }
        if (m_DataGatherer.isPopulation() && !m_DataGatherer.isAttributeActive(cid)) {
            LOG_DEBUG(<< "Not adding value for deleted attribute " << cid);
            return false;
        }

        TSizeSizePr pidCid = std::make_pair(pid, cid);

        // If record is explicit null just note that a null record has been seen
        // for the given (pid, cid) pair.
        if (data.isExplicitNull()) {
            TSizeSizePrUSet& bucketExplicitNulls =
                m_PersonAttributeExplicitNulls.get(time);
            bucketExplicitNulls.insert(pidCid);
            return true;
        }

        TSizeSizePrUInt64UMap& bucketCounts = m_PersonAttributeCounts.get(time);
        if (count > 0) {
            bucketCounts[pidCid] += count;
        }

        const CEventData::TOptionalStrVec& influences = data.influences();
        auto& influencerCounts = m_InfluencerCounts.get(time);
        if (influences.size() != influencerCounts.size()) {
            LOG_ERROR(<< "Unexpected influences: "
                      << core::CContainerPrinter::print(influences) << " expected "
                      << core::CContainerPrinter::print(this->beginInfluencers(),
                                                        this->endInfluencers()));
            return false;
        }

        TStoredStringPtrVec canonicalInfluences(influencerCounts.size());
        for (std::size_t i = 0u; i < influences.size(); ++i) {
            const CEventData::TOptionalStr& influence = influences[i];
            if (influence) {
                const auto& inf = CStringStore::influencers().get(*influence);
                canonicalInfluences[i] = inf;
                if (count > 0) {
                    influencerCounts[i]
                        .emplace(boost::unordered::piecewise_construct,
                                 boost::make_tuple(pidCid, inf),
                                 boost::make_tuple(uint64_t(0)))
                        .first->second += count;
                }
            }
        }

        this->addValue(pid, cid, time, data.values(), count, data.stringValue(),
                       canonicalInfluences);
    }
    return true;
}

void CBucketGatherer::timeNow(core_t::TTime time) {
    this->hiddenTimeNow(time, false);
}

void CBucketGatherer::hiddenTimeNow(core_t::TTime time, bool skipUpdates) {
    m_EarliestTime = std::min(m_EarliestTime, time);
    core_t::TTime n = (time - m_BucketStart) / this->bucketLength();
    if (n <= 0) {
        return;
    }

    core_t::TTime newBucketStart = m_BucketStart;
    for (core_t::TTime i = 0; i < n; ++i) {
        newBucketStart += this->bucketLength();

        // The order here is important. While starting new buckets
        // the gatherers may finalise the earliest bucket within
        // the latency window, thus we push a new count bucket only
        // after startNewBucket has been called.
        std::ptrdiff_t numberInfluences{this->endInfluencers() - this->beginInfluencers()};
        this->startNewBucket(newBucketStart, skipUpdates);
        m_PersonAttributeCounts.push(TSizeSizePrUInt64UMap(1), newBucketStart);
        m_PersonAttributeExplicitNulls.push(TSizeSizePrUSet(1), newBucketStart);
        m_InfluencerCounts.push(TSizeSizePrStoredStringPtrPrUInt64UMapVec(numberInfluences),
                                newBucketStart);
        m_BucketStart = newBucketStart;
    }
}

void CBucketGatherer::sampleNow(core_t::TTime sampleBucketStart) {
    core_t::TTime timeNow =
        sampleBucketStart +
        (m_DataGatherer.params().s_LatencyBuckets + 1) * this->bucketLength() - 1;
    this->timeNow(timeNow);
    this->sample(sampleBucketStart);
}

void CBucketGatherer::skipSampleNow(core_t::TTime sampleBucketStart) {
    core_t::TTime timeNow =
        sampleBucketStart +
        (m_DataGatherer.params().s_LatencyBuckets + 1) * this->bucketLength() - 1;
    this->hiddenTimeNow(timeNow, true);
}

void CBucketGatherer::personNonZeroCounts(core_t::TTime time, TSizeUInt64PrVec& result) const {
    using TSizeUInt64Map = std::map<std::size_t, uint64_t>;

    result.clear();

    if (!this->dataAvailable(time)) {
        LOG_ERROR(<< "No statistics at " << time
                  << ", current bucket = " << this->printCurrentBucket());
        return;
    }

    TSizeUInt64Map personCounts;
    for (const auto& count : this->bucketCounts(time)) {
        personCounts[CDataGatherer::extractPersonId(count)] +=
            CDataGatherer::extractData(count);
    }
    result.reserve(personCounts.size());
    result.assign(personCounts.begin(), personCounts.end());
}

void CBucketGatherer::recyclePeople(const TSizeVec& peopleToRemove) {
    if (!peopleToRemove.empty()) {
        remove(peopleToRemove, CDataGatherer::SExtractPersonId(), m_PersonAttributeCounts);
        remove(peopleToRemove, CDataGatherer::SExtractPersonId(), m_PersonAttributeExplicitNulls);
        remove(peopleToRemove, CDataGatherer::SExtractPersonId(), m_InfluencerCounts);
    }
}

void CBucketGatherer::removePeople(std::size_t lowestPersonToRemove) {
    if (lowestPersonToRemove < m_DataGatherer.numberPeople()) {
        TSizeVec peopleToRemove;
        std::size_t maxPersonId = m_DataGatherer.numberPeople();
        peopleToRemove.reserve(maxPersonId - lowestPersonToRemove);
        for (std::size_t pid = lowestPersonToRemove; pid < maxPersonId; ++pid) {
            peopleToRemove.push_back(pid);
        }
        remove(peopleToRemove, CDataGatherer::SExtractPersonId(), m_PersonAttributeCounts);
        remove(peopleToRemove, CDataGatherer::SExtractPersonId(), m_PersonAttributeExplicitNulls);
        remove(peopleToRemove, CDataGatherer::SExtractPersonId(), m_InfluencerCounts);
    }
}

void CBucketGatherer::recycleAttributes(const TSizeVec& attributesToRemove) {
    if (!attributesToRemove.empty()) {
        remove(attributesToRemove, CDataGatherer::SExtractAttributeId(), m_PersonAttributeCounts);
        remove(attributesToRemove, CDataGatherer::SExtractAttributeId(),
               m_PersonAttributeExplicitNulls);
        remove(attributesToRemove, CDataGatherer::SExtractAttributeId(), m_InfluencerCounts);
    }
}

void CBucketGatherer::removeAttributes(std::size_t lowestAttributeToRemove) {
    if (lowestAttributeToRemove < m_DataGatherer.numberAttributes()) {
        TSizeVec attributesToRemove;
        const std::size_t numAttributes = m_DataGatherer.numberAttributes();
        attributesToRemove.reserve(numAttributes - lowestAttributeToRemove);
        for (std::size_t cid = lowestAttributeToRemove; cid < numAttributes; ++cid) {
            attributesToRemove.push_back(cid);
        }
        remove(attributesToRemove, CDataGatherer::SExtractAttributeId(), m_PersonAttributeCounts);
        remove(attributesToRemove, CDataGatherer::SExtractAttributeId(),
               m_PersonAttributeExplicitNulls);
        remove(attributesToRemove, CDataGatherer::SExtractAttributeId(), m_InfluencerCounts);
    }
}

core_t::TTime CBucketGatherer::currentBucketStartTime() const {
    return m_BucketStart;
}

void CBucketGatherer::currentBucketStartTime(core_t::TTime time) {
    m_BucketStart = time;
}

core_t::TTime CBucketGatherer::earliestBucketStartTime() const {
    return this->currentBucketStartTime() -
           (m_DataGatherer.params().s_LatencyBuckets * this->bucketLength());
}

core_t::TTime CBucketGatherer::bucketLength() const {
    return m_DataGatherer.params().s_BucketLength;
}

bool CBucketGatherer::dataAvailable(core_t::TTime time) const {
    return time >= m_EarliestTime && time >= this->earliestBucketStartTime();
}

bool CBucketGatherer::validateSampleTimes(core_t::TTime& startTime, core_t::TTime endTime) const {
    // Sanity checks:
    //   1) The start and end times are aligned to bucket boundaries.
    //   2) The end time is greater than the start time,
    //   3) The start time is greater than or equal to the start time
    //      of the current bucket of the counter,
    //   4) The start time is greater than or equal to the start time
    //      of the last sampled bucket

    if (!maths::CIntegerTools::aligned(startTime - m_BucketStart, this->bucketLength())) {
        LOG_ERROR(<< "Sample start time " << startTime << " is not bucket aligned");
        return false;
    }
    if (!maths::CIntegerTools::aligned(endTime - m_BucketStart, this->bucketLength())) {
        LOG_ERROR(<< "Sample end time " << endTime << " is not bucket aligned");
        return false;
    }
    if (endTime <= startTime) {
        LOG_ERROR(<< "End time " << endTime
                  << " is not greater than the start time " << startTime);
        return false;
    }
    for (/**/; startTime < endTime; startTime += this->bucketLength()) {
        if (!this->dataAvailable(startTime)) {
            LOG_ERROR(<< "No counts available at " << startTime
                      << ", current bucket = " << this->printCurrentBucket());
            continue;
        }
        return true;
    }

    return false;
}

const CDataGatherer& CBucketGatherer::dataGatherer() const {
    return m_DataGatherer;
}

std::string CBucketGatherer::printCurrentBucket() const {
    std::ostringstream result;
    result << "[" << m_BucketStart << "," << m_BucketStart + this->bucketLength() << ")";
    return result.str();
}

const CBucketGatherer::TSizeSizePrUInt64UMap&
CBucketGatherer::bucketCounts(core_t::TTime time) const {
    return m_PersonAttributeCounts.get(time);
}

const CBucketGatherer::TSizeSizePrStoredStringPtrPrUInt64UMapVec&
CBucketGatherer::influencerCounts(core_t::TTime time) const {
    return m_InfluencerCounts.get(time);
}

bool CBucketGatherer::hasExplicitNullsOnly(core_t::TTime time, std::size_t pid, std::size_t cid) const {
    const TSizeSizePrUSet& bucketExplicitNulls = m_PersonAttributeExplicitNulls.get(time);
    if (bucketExplicitNulls.empty()) {
        return false;
    }
    const TSizeSizePrUInt64UMap& bucketCounts = m_PersonAttributeCounts.get(time);
    TSizeSizePr pidCid = std::make_pair(pid, cid);
    return bucketExplicitNulls.find(pidCid) != bucketExplicitNulls.end() &&
           bucketCounts.find(pidCid) == bucketCounts.end();
}

uint64_t CBucketGatherer::checksum() const {
    using TStrCRef = std::reference_wrapper<const std::string>;
    using TStrCRefStrCRefPr = std::pair<TStrCRef, TStrCRef>;
    using TStrCRefStrCRefPrVec = std::vector<TStrCRefStrCRefPr>;
    using TStrCRefStrCRefPrUInt64Pr = std::pair<TStrCRefStrCRefPr, uint64_t>;
    using TStrCRefStrCRefPrUInt64PrVec = std::vector<TStrCRefStrCRefPrUInt64Pr>;

    uint64_t result = maths::CChecksum::calculate(0, m_BucketStart);

    result = maths::CChecksum::calculate(result, m_PersonAttributeCounts.latestBucketEnd());
    for (const auto& bucketCounts : m_PersonAttributeCounts) {
        TStrCRefStrCRefPrUInt64PrVec personAttributeCounts;
        personAttributeCounts.reserve(bucketCounts.size());
        for (const auto& count : bucketCounts) {
            std::size_t pid = CDataGatherer::extractPersonId(count);
            std::size_t cid = CDataGatherer::extractAttributeId(count);
            TStrCRefStrCRefPr key(TStrCRef(m_DataGatherer.personName(pid)),
                                  TStrCRef(m_DataGatherer.attributeName(cid)));
            personAttributeCounts.emplace_back(key, CDataGatherer::extractData(count));
        }
        std::sort(personAttributeCounts.begin(), personAttributeCounts.end(),
                  maths::COrderings::SLexicographicalCompare());
        result = maths::CChecksum::calculate(result, personAttributeCounts);
    }

    result = maths::CChecksum::calculate(
        result, m_PersonAttributeExplicitNulls.latestBucketEnd());
    for (const auto& bucketExplicitNulls : m_PersonAttributeExplicitNulls) {
        TStrCRefStrCRefPrVec personAttributeExplicitNulls;
        personAttributeExplicitNulls.reserve(bucketExplicitNulls.size());
        for (const auto& nulls : bucketExplicitNulls) {
            std::size_t pid = CDataGatherer::extractPersonId(nulls);
            std::size_t cid = CDataGatherer::extractAttributeId(nulls);
            TStrCRefStrCRefPr key(TStrCRef(m_DataGatherer.personName(pid)),
                                  TStrCRef(m_DataGatherer.attributeName(cid)));
            personAttributeExplicitNulls.push_back(key);
        }
        std::sort(personAttributeExplicitNulls.begin(),
                  personAttributeExplicitNulls.end(),
                  maths::COrderings::SLexicographicalCompare());
        result = maths::CChecksum::calculate(result, personAttributeExplicitNulls);
    }

    LOG_TRACE(<< "checksum = " << result);

    return result;
}

void CBucketGatherer::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CBucketGatherer");
    core::CMemoryDebug::dynamicSize("m_PersonAttributeCounts", m_PersonAttributeCounts, mem);
    core::CMemoryDebug::dynamicSize("m_PersonAttributeExplicitNulls",
                                    m_PersonAttributeExplicitNulls, mem);
    core::CMemoryDebug::dynamicSize("m_Influencers", m_InfluencerCounts, mem);
}

std::size_t CBucketGatherer::memoryUsage() const {
    std::size_t mem = core::CMemory::dynamicSize(m_PersonAttributeCounts);
    mem += core::CMemory::dynamicSize(m_PersonAttributeExplicitNulls);
    mem += core::CMemory::dynamicSize(m_InfluencerCounts);
    return mem;
}

void CBucketGatherer::clear() {
    m_PersonAttributeCounts.clear(TSizeSizePrUInt64UMap(1));
    m_PersonAttributeExplicitNulls.clear(TSizeSizePrUSet(1));
    m_InfluencerCounts.clear(TSizeSizePrStoredStringPtrPrUInt64UMapVec(
        this->endInfluencers() - this->beginInfluencers()));
}

bool CBucketGatherer::resetBucket(core_t::TTime bucketStart) {
    if (!maths::CIntegerTools::aligned(bucketStart, this->bucketLength())) {
        LOG_ERROR(<< "Bucket start time " << bucketStart << " is not bucket aligned");
        return false;
    }

    if (!this->dataAvailable(bucketStart) ||
        bucketStart >= this->currentBucketStartTime() + this->bucketLength()) {
        LOG_WARN(<< "No data available at " << bucketStart
                 << ", current bucket = " << this->printCurrentBucket());
        return false;
    }

    LOG_TRACE(<< "Resetting bucket starting at " << bucketStart);
    std::ptrdiff_t numberInfluences{this->endInfluencers() - this->beginInfluencers()};
    m_PersonAttributeCounts.get(bucketStart).clear();
    m_PersonAttributeExplicitNulls.get(bucketStart).clear();
    m_InfluencerCounts.get(bucketStart) =
        TSizeSizePrStoredStringPtrPrUInt64UMapVec(numberInfluences);
    return true;
}

void CBucketGatherer::baseAcceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(BUCKET_START_TAG, m_BucketStart);
    inserter.insertLevel(
        BUCKET_COUNT_TAG,
        std::bind<void>(TSizeSizePrUInt64UMapQueue::CSerializer<detail::SBucketCountsPersister>(),
                        std::cref(m_PersonAttributeCounts), std::placeholders::_1));
    // Clear any empty collections before persist these are resized on restore.
    TSizeSizePrStoredStringPtrPrUInt64UMapVecQueue influencerCounts{m_InfluencerCounts};
    inserter.insertLevel(
        INFLUENCERS_COUNT_TAG,
        std::bind<void>(TSizeSizePrStoredStringPtrPrUInt64UMapVecQueue::CSerializer<detail::SInfluencerCountsPersister>(),
                        std::cref(m_InfluencerCounts), std::placeholders::_1));
    core::CPersistUtils::persist(BUCKET_EXPLICIT_NULLS_TAG,
                                 m_PersonAttributeExplicitNulls, inserter);
}

bool CBucketGatherer::baseAcceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    this->clear();
    do {
        const std::string& name = traverser.name();
        RESTORE_BUILT_IN(BUCKET_START_TAG, m_BucketStart)
        RESTORE_SETUP_TEARDOWN(
            BUCKET_COUNT_TAG,
            m_PersonAttributeCounts = TSizeSizePrUInt64UMapQueue(
                m_DataGatherer.params().s_LatencyBuckets, this->bucketLength(),
                m_BucketStart, TSizeSizePrUInt64UMap(1)),
            traverser.traverseSubLevel(std::bind<bool>(
                TSizeSizePrUInt64UMapQueue::CSerializer<detail::SBucketCountsPersister>(
                    TSizeSizePrUInt64UMap(1)),
                std::ref(m_PersonAttributeCounts), std::placeholders::_1)),
            /**/)
        RESTORE_SETUP_TEARDOWN(
            INFLUENCERS_COUNT_TAG,
            m_InfluencerCounts = TSizeSizePrStoredStringPtrPrUInt64UMapVecQueue(
                m_DataGatherer.params().s_LatencyBuckets + 3, this->bucketLength(), m_BucketStart),
            traverser.traverseSubLevel(std::bind<bool>(
                TSizeSizePrStoredStringPtrPrUInt64UMapVecQueue::CSerializer<detail::SInfluencerCountsPersister>(),
                std::ref(m_InfluencerCounts), std::placeholders::_1)),
            /**/)
        RESTORE_SETUP_TEARDOWN(
            BUCKET_EXPLICIT_NULLS_TAG,
            m_PersonAttributeExplicitNulls = TSizeSizePrUSetQueue(
                m_DataGatherer.params().s_LatencyBuckets, this->bucketLength(),
                m_BucketStart, TSizeSizePrUSet(1)),
            core::CPersistUtils::restore(BUCKET_EXPLICIT_NULLS_TAG,
                                         m_PersonAttributeExplicitNulls, traverser),
            /**/)
    } while (traverser.next());
    return true;
}
}
}
