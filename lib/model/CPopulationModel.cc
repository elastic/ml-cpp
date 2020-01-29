/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CPopulationModel.h>

#include <core/CAllocationStrategy.h>
#include <core/CContainerPrinter.h>
#include <core/CProgramCounters.h>
#include <core/CStatePersistInserter.h>
#include <core/Constants.h>
#include <core/CoreTypes.h>
#include <core/RestoreMacros.h>

#include <maths/CChecksum.h>
#include <maths/COrderings.h>
#include <maths/CPrior.h>
#include <maths/CPriorStateSerialiser.h>
#include <maths/CTimeSeriesDecomposition.h>
#include <maths/CTimeSeriesDecompositionStateSerialiser.h>

#include <model/CDataGatherer.h>
#include <model/CModelTools.h>
#include <model/CResourceMonitor.h>

#include <algorithm>

namespace ml {
namespace model {

namespace {

using TStrCRef = std::reference_wrapper<const std::string>;
using TStrCRefUInt64Map = std::map<TStrCRef, uint64_t, maths::COrderings::SLess>;

enum EEntity { E_Person, E_Attribute };

const std::string EMPTY;

//! Check if \p entity is active.
bool isActive(EEntity entity, const CDataGatherer& gatherer, std::size_t id) {
    switch (entity) {
    case E_Person:
        return gatherer.isPersonActive(id);
    case E_Attribute:
        return gatherer.isAttributeActive(id);
    }
    return false;
}

//! Get \p entity's name.
const std::string& name(EEntity entity, const CDataGatherer& gatherer, std::size_t id) {
    switch (entity) {
    case E_Person:
        return gatherer.personName(id);
    case E_Attribute:
        return gatherer.attributeName(id);
    }
    return EMPTY;
}

//! Update \p hashes with the hash of the active entities in \p values.
template<typename T>
void hashActive(EEntity entity,
                const CDataGatherer& gatherer,
                const std::vector<T>& values,
                TStrCRefUInt64Map& hashes) {
    for (std::size_t id = 0u; id < values.size(); ++id) {
        if (isActive(entity, gatherer, id)) {
            uint64_t& hash = hashes[std::cref(name(entity, gatherer, id))];
            hash = maths::CChecksum::calculate(hash, values[id]);
        }
    }
}

const std::size_t COUNT_MIN_SKETCH_ROWS = 3u;
const std::size_t COUNT_MIN_SKETCH_COLUMNS = 500u;
const std::size_t BJKST_HASHES = 3u;
const std::size_t BJKST_MAX_SIZE = 100u;
const std::size_t CHUNK_SIZE = 500u;

// We use short field names to reduce the state size
const std::string WINDOW_BUCKET_COUNT_TAG("a");
const std::string PERSON_BUCKET_COUNT_TAG("b");
const std::string PERSON_LAST_BUCKET_TIME_TAG("c");
const std::string ATTRIBUTE_FIRST_BUCKET_TIME_TAG("d");
const std::string ATTRIBUTE_LAST_BUCKET_TIME_TAG("e");
const std::string PERSON_ATTRIBUTE_BUCKET_COUNT_TAG("f");
const std::string DISTINCT_PERSON_COUNT_TAG("g");
// Extra data tag deprecated at model version 34
// TODO remove on next version bump
//const std::string EXTRA_DATA_TAG("h");
//const std::string INTERIM_BUCKET_CORRECTOR_TAG("i");
}

CPopulationModel::CPopulationModel(const SModelParams& params,
                                   const TDataGathererPtr& dataGatherer,
                                   const TFeatureInfluenceCalculatorCPtrPrVecVec& influenceCalculators)
    : CAnomalyDetectorModel(params, dataGatherer, influenceCalculators),
      m_NewDistinctPersonCounts(BJKST_HASHES, BJKST_MAX_SIZE) {
    const model_t::TFeatureVec& features = dataGatherer->features();
    for (std::size_t i = 0u; i < features.size(); ++i) {
        if (!model_t::isCategorical(features[i]) && !model_t::isConstant(features[i])) {
            m_NewPersonBucketCounts.reset(maths::CCountMinSketch(
                COUNT_MIN_SKETCH_ROWS, COUNT_MIN_SKETCH_COLUMNS));
            break;
        }
    }
}

CPopulationModel::CPopulationModel(bool isForPersistence, const CPopulationModel& other)
    : CAnomalyDetectorModel(isForPersistence, other),
      m_PersonLastBucketTimes(other.m_PersonLastBucketTimes),
      m_AttributeFirstBucketTimes(other.m_AttributeFirstBucketTimes),
      m_AttributeLastBucketTimes(other.m_AttributeLastBucketTimes),
      m_NewDistinctPersonCounts(BJKST_HASHES, BJKST_MAX_SIZE),
      m_DistinctPersonCounts(other.m_DistinctPersonCounts),
      m_PersonAttributeBucketCounts(other.m_PersonAttributeBucketCounts) {
    if (!isForPersistence) {
        LOG_ABORT(<< "This constructor only creates clones for persistence");
    }
}

bool CPopulationModel::isPopulation() const {
    return true;
}

CPopulationModel::TOptionalUInt64
CPopulationModel::currentBucketCount(std::size_t pid, core_t::TTime time) const {
    if (!this->bucketStatsAvailable(time)) {
        LOG_ERROR(<< "No statistics at " << time);
        return TOptionalUInt64();
    }

    const TSizeUInt64PrVec& personCounts = this->personCounts();
    auto i = std::lower_bound(personCounts.begin(), personCounts.end(), pid,
                              maths::COrderings::SFirstLess());
    return (i != personCounts.end() && i->first == pid) ? TOptionalUInt64(i->second)
                                                        : TOptionalUInt64();
}

CPopulationModel::TOptionalDouble CPopulationModel::baselineBucketCount(std::size_t /*pid*/) const {
    return TOptionalDouble();
}

void CPopulationModel::currentBucketPersonIds(core_t::TTime time, TSizeVec& result) const {
    result.clear();
    if (!this->bucketStatsAvailable(time)) {
        LOG_ERROR(<< "No statistics at " << time);
        return;
    }

    const TSizeUInt64PrVec& personCounts = this->personCounts();
    result.reserve(personCounts.size());
    for (const auto& count : personCounts) {
        result.push_back(count.first);
    }
}

void CPopulationModel::sample(core_t::TTime startTime,
                              core_t::TTime endTime,
                              CResourceMonitor& resourceMonitor) {
    this->CAnomalyDetectorModel::sample(startTime, endTime, resourceMonitor);

    const CDataGatherer& gatherer = this->dataGatherer();
    const CDataGatherer::TSizeSizePrUInt64UMap& counts = gatherer.bucketCounts(startTime);
    for (const auto& count : counts) {
        std::size_t pid = CDataGatherer::extractPersonId(count);
        std::size_t cid = CDataGatherer::extractAttributeId(count);
        m_PersonLastBucketTimes[pid] = startTime;
        if (CAnomalyDetectorModel::isTimeUnset(m_AttributeFirstBucketTimes[cid])) {
            m_AttributeFirstBucketTimes[cid] = startTime;
        }
        m_AttributeLastBucketTimes[cid] = startTime;
        m_DistinctPersonCounts[cid].add(static_cast<int32_t>(pid));
        if (cid < m_PersonAttributeBucketCounts.size()) {
            m_PersonAttributeBucketCounts[cid].add(static_cast<int32_t>(pid), 1.0);
        }
    }

    double alpha = std::exp(-this->params().s_DecayRate * 1.0);
    for (std::size_t cid = 0u; cid < m_PersonAttributeBucketCounts.size(); ++cid) {
        m_PersonAttributeBucketCounts[cid].age(alpha);
    }
}

uint64_t CPopulationModel::checksum(bool includeCurrentBucketStats) const {
    uint64_t seed = this->CAnomalyDetectorModel::checksum(includeCurrentBucketStats);

    const CDataGatherer& gatherer = this->dataGatherer();
    TStrCRefUInt64Map hashes;
    hashActive(E_Person, gatherer, m_PersonLastBucketTimes, hashes);
    hashActive(E_Attribute, gatherer, m_AttributeFirstBucketTimes, hashes);
    hashActive(E_Attribute, gatherer, m_AttributeLastBucketTimes, hashes);

    LOG_TRACE(<< "seed = " << seed);
    LOG_TRACE(<< "hashes = " << core::CContainerPrinter::print(hashes));

    return maths::CChecksum::calculate(seed, hashes);
}

void CPopulationModel::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CPopulationModel");
    this->CAnomalyDetectorModel::debugMemoryUsage(mem->addChild());
    core::CMemoryDebug::dynamicSize("m_PersonLastBucketTimes", m_PersonLastBucketTimes, mem);
    core::CMemoryDebug::dynamicSize("m_AttributeFirstBucketTimes",
                                    m_AttributeFirstBucketTimes, mem);
    core::CMemoryDebug::dynamicSize("m_AttributeLastBucketTimes",
                                    m_AttributeLastBucketTimes, mem);
    core::CMemoryDebug::dynamicSize("m_NewDistinctPersonCounts",
                                    m_NewDistinctPersonCounts, mem);
    core::CMemoryDebug::dynamicSize("m_DistinctPersonCounts", m_DistinctPersonCounts, mem);
    core::CMemoryDebug::dynamicSize("m_NewPersonBucketCounts", m_NewPersonBucketCounts, mem);
    core::CMemoryDebug::dynamicSize("m_PersonAttributeBucketCounts",
                                    m_PersonAttributeBucketCounts, mem);
}

std::size_t CPopulationModel::memoryUsage() const {
    std::size_t mem = this->CAnomalyDetectorModel::memoryUsage();
    mem += core::CMemory::dynamicSize(m_PersonLastBucketTimes);
    mem += core::CMemory::dynamicSize(m_AttributeFirstBucketTimes);
    mem += core::CMemory::dynamicSize(m_AttributeLastBucketTimes);
    mem += core::CMemory::dynamicSize(m_NewDistinctPersonCounts);
    mem += core::CMemory::dynamicSize(m_DistinctPersonCounts);
    mem += core::CMemory::dynamicSize(m_NewPersonBucketCounts);
    mem += core::CMemory::dynamicSize(m_PersonAttributeBucketCounts);
    return mem;
}

double CPopulationModel::attributeFrequency(std::size_t cid) const {
    std::size_t active = this->dataGatherer().numberActivePeople();
    return active == 0 ? 0.5
                       : static_cast<double>(m_DistinctPersonCounts[cid].number()) /
                             static_cast<double>(active);
}

const CPopulationModel::TTimeVec& CPopulationModel::attributeFirstBucketTimes() const {
    return m_AttributeFirstBucketTimes;
}

const CPopulationModel::TTimeVec& CPopulationModel::attributeLastBucketTimes() const {
    return m_AttributeLastBucketTimes;
}

double CPopulationModel::sampleRateWeight(std::size_t pid, std::size_t cid) const {
    if (cid >= m_PersonAttributeBucketCounts.size() ||
        cid >= m_DistinctPersonCounts.size()) {
        return 1.0;
    }

    const maths::CCountMinSketch& counts = m_PersonAttributeBucketCounts[cid];
    const maths::CBjkstUniqueValues& distinctPeople = m_DistinctPersonCounts[cid];

    double personCount = counts.count(static_cast<uint32_t>(pid)) -
                         counts.oneMinusDeltaError();
    if (personCount <= 0.0) {
        return 1.0;
    }
    LOG_TRACE(<< "personCount = " << personCount);

    double totalCount = counts.totalCount();
    double distinctPeopleCount =
        std::min(static_cast<double>(distinctPeople.number()),
                 static_cast<double>(this->dataGatherer().numberActivePeople()));
    double meanPersonCount = totalCount / distinctPeopleCount;
    LOG_TRACE(<< "meanPersonCount = " << meanPersonCount);

    return std::min(meanPersonCount / personCount, 1.0);
}

void CPopulationModel::doAcceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(WINDOW_BUCKET_COUNT_TAG, this->windowBucketCount(),
                         core::CIEEE754::E_SinglePrecision);
    core::CPersistUtils::persist(PERSON_BUCKET_COUNT_TAG,
                                 this->personBucketCounts(), inserter);
    core::CPersistUtils::persist(PERSON_LAST_BUCKET_TIME_TAG,
                                 m_PersonLastBucketTimes, inserter);
    core::CPersistUtils::persist(ATTRIBUTE_FIRST_BUCKET_TIME_TAG,
                                 m_AttributeFirstBucketTimes, inserter);
    core::CPersistUtils::persist(ATTRIBUTE_LAST_BUCKET_TIME_TAG,
                                 m_AttributeLastBucketTimes, inserter);
    for (std::size_t cid = 0; cid < m_PersonAttributeBucketCounts.size(); ++cid) {
        inserter.insertLevel(PERSON_ATTRIBUTE_BUCKET_COUNT_TAG,
                             std::bind(&maths::CCountMinSketch::acceptPersistInserter,
                                       &m_PersonAttributeBucketCounts[cid],
                                       std::placeholders::_1));
    }
    for (std::size_t cid = 0; cid < m_DistinctPersonCounts.size(); ++cid) {
        inserter.insertLevel(
            DISTINCT_PERSON_COUNT_TAG,
            std::bind(&maths::CBjkstUniqueValues::acceptPersistInserter,
                      &m_DistinctPersonCounts[cid], std::placeholders::_1));
    }
}

bool CPopulationModel::doAcceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE_SETUP_TEARDOWN(WINDOW_BUCKET_COUNT_TAG, double count,
                               core::CStringUtils::stringToType(traverser.value(), count),
                               this->windowBucketCount(count));
        RESTORE(PERSON_BUCKET_COUNT_TAG,
                core::CPersistUtils::restore(name, this->personBucketCounts(), traverser))
        RESTORE(PERSON_LAST_BUCKET_TIME_TAG,
                core::CPersistUtils::restore(name, m_PersonLastBucketTimes, traverser))
        RESTORE(ATTRIBUTE_FIRST_BUCKET_TIME_TAG,
                core::CPersistUtils::restore(name, m_AttributeFirstBucketTimes, traverser))
        RESTORE(ATTRIBUTE_LAST_BUCKET_TIME_TAG,
                core::CPersistUtils::restore(name, m_AttributeLastBucketTimes, traverser))
        if (name == PERSON_ATTRIBUTE_BUCKET_COUNT_TAG) {
            maths::CCountMinSketch sketch(traverser);
            m_PersonAttributeBucketCounts.push_back(maths::CCountMinSketch(0, 0));
            m_PersonAttributeBucketCounts.back().swap(sketch);
            continue;
        }
        if (name == DISTINCT_PERSON_COUNT_TAG) {
            maths::CBjkstUniqueValues sketch(traverser);
            m_DistinctPersonCounts.push_back(maths::CBjkstUniqueValues(0, 0));
            m_DistinctPersonCounts.back().swap(sketch);
            continue;
        }
    } while (traverser.next());

    return true;
}

void CPopulationModel::createUpdateNewModels(core_t::TTime time,
                                             CResourceMonitor& resourceMonitor) {
    this->updateRecycledModels();

    CDataGatherer& gatherer = this->dataGatherer();

    std::size_t numberExistingPeople = m_PersonLastBucketTimes.size();
    std::size_t numberExistingAttributes = m_AttributeLastBucketTimes.size();
    TOptionalSize usageEstimate = this->estimateMemoryUsage(
        std::min(numberExistingPeople, gatherer.numberActivePeople()),
        std::min(numberExistingAttributes, gatherer.numberActiveAttributes()),
        0); // # correlations
    std::size_t ourUsage = usageEstimate ? usageEstimate.get()
                                         : this->computeMemoryUsage();
    std::size_t resourceLimit = ourUsage + resourceMonitor.allocationLimit();
    std::size_t numberNewPeople = gatherer.numberPeople();
    numberNewPeople = numberNewPeople > numberExistingPeople ? numberNewPeople - numberExistingPeople
                                                             : 0;
    std::size_t numberNewAttributes = gatherer.numberAttributes();
    numberNewAttributes = numberNewAttributes > numberExistingAttributes
                              ? numberNewAttributes - numberExistingAttributes
                              : 0;

    while (numberNewPeople > 0 && resourceMonitor.areAllocationsAllowed() &&
           (resourceMonitor.haveNoLimit() || ourUsage < resourceLimit)) {
        // We batch people in CHUNK_SIZE (500) and create models in chunks
        // and test usage after each chunk.
        std::size_t numberToCreate = std::min(numberNewPeople, CHUNK_SIZE);
        LOG_TRACE(<< "Creating batch of " << numberToCreate
                  << " people of remaining " << numberNewPeople << ". "
                  << resourceLimit - ourUsage << " free bytes remaining");
        this->createNewModels(numberToCreate, 0);
        numberExistingPeople += numberToCreate;
        numberNewPeople -= numberToCreate;
        if ((numberNewPeople > 0 || numberNewAttributes > 0) &&
            resourceMonitor.haveNoLimit() == false) {
            ourUsage = this->estimateMemoryUsageOrComputeAndUpdate(
                numberExistingPeople, numberExistingAttributes, 0);
        }
    }

    while (numberNewAttributes > 0 && resourceMonitor.areAllocationsAllowed() &&
           (resourceMonitor.haveNoLimit() || ourUsage < resourceLimit)) {
        // We batch attributes in CHUNK_SIZE (500) and create models in chunks
        // and test usage after each chunk.
        std::size_t numberToCreate = std::min(numberNewAttributes, CHUNK_SIZE);
        LOG_TRACE(<< "Creating batch of " << numberToCreate
                  << " attributes of remaining " << numberNewAttributes << ". "
                  << resourceLimit - ourUsage << " free bytes remaining");
        this->createNewModels(0, numberToCreate);
        numberExistingAttributes += numberToCreate;
        numberNewAttributes -= numberToCreate;
        if (numberNewAttributes > 0 && resourceMonitor.haveNoLimit() == false) {
            ourUsage = this->estimateMemoryUsageOrComputeAndUpdate(
                numberExistingPeople, numberExistingAttributes, 0);
        }
    }

    this->estimateMemoryUsageOrComputeAndUpdate(numberExistingPeople,
                                                numberExistingAttributes, 0);

    if (numberNewPeople > 0) {
        resourceMonitor.acceptAllocationFailureResult(time);
        LOG_DEBUG(<< "Not enough memory to create person models");
        core::CProgramCounters::counter(
            counter_t::E_TSADNumberMemoryLimitModelCreationFailures) += numberNewPeople;
        std::size_t toRemove = gatherer.numberPeople() - numberNewPeople;
        gatherer.removePeople(toRemove);
    }
    if (numberNewAttributes > 0) {
        resourceMonitor.acceptAllocationFailureResult(time);
        LOG_DEBUG(<< "Not enough memory to create attribute models");
        core::CProgramCounters::counter(
            counter_t::E_TSADNumberMemoryLimitModelCreationFailures) += numberNewAttributes;
        std::size_t toRemove = gatherer.numberAttributes() - numberNewAttributes;
        gatherer.removeAttributes(toRemove);
    }

    this->refreshCorrelationModels(resourceLimit, resourceMonitor);
}

void CPopulationModel::createNewModels(std::size_t n, std::size_t m) {
    if (n > 0) {
        core::CAllocationStrategy::resize(m_PersonLastBucketTimes,
                                          n + m_PersonLastBucketTimes.size(),
                                          CAnomalyDetectorModel::TIME_UNSET);
    }

    if (m > 0) {
        std::size_t newM = m + m_AttributeFirstBucketTimes.size();
        core::CAllocationStrategy::resize(m_AttributeFirstBucketTimes, newM,
                                          CAnomalyDetectorModel::TIME_UNSET);
        core::CAllocationStrategy::resize(m_AttributeLastBucketTimes, newM,
                                          CAnomalyDetectorModel::TIME_UNSET);
        core::CAllocationStrategy::resize(m_DistinctPersonCounts, newM, m_NewDistinctPersonCounts);
        if (m_NewPersonBucketCounts) {
            core::CAllocationStrategy::resize(m_PersonAttributeBucketCounts,
                                              newM, *m_NewPersonBucketCounts);
        }
    }

    this->CAnomalyDetectorModel::createNewModels(n, m);
}

void CPopulationModel::updateRecycledModels() {
    CDataGatherer& gatherer = this->dataGatherer();
    for (auto pid : gatherer.recycledPersonIds()) {
        if (pid < m_PersonLastBucketTimes.size()) {
            m_PersonLastBucketTimes[pid] = 0;
        }
    }

    TSizeVec& attributes = gatherer.recycledAttributeIds();
    for (auto cid : attributes) {
        if (cid < m_AttributeFirstBucketTimes.size()) {
            m_AttributeFirstBucketTimes[cid] = CAnomalyDetectorModel::TIME_UNSET;
            m_AttributeLastBucketTimes[cid] = CAnomalyDetectorModel::TIME_UNSET;
            m_DistinctPersonCounts[cid] = m_NewDistinctPersonCounts;
            if (m_NewPersonBucketCounts) {
                m_PersonAttributeBucketCounts[cid] = *m_NewPersonBucketCounts;
            }
        } else {
            LOG_ERROR(<< "Recycled attribute identifier '" << cid << "' out-of-range [0,"
                      << m_AttributeFirstBucketTimes.size() << ")");
        }
    }
    attributes.clear();

    this->CAnomalyDetectorModel::updateRecycledModels();
}

void CPopulationModel::correctBaselineForInterim(model_t::EFeature feature,
                                                 std::size_t pid,
                                                 std::size_t cid,
                                                 model_t::CResultType type,
                                                 const TSizeDoublePr1Vec& correlated,
                                                 const TCorrectionKeyDouble1VecUMap& corrections,
                                                 TDouble1Vec& result) const {
    if (type.isInterim() && model_t::requiresInterimResultAdjustment(feature)) {
        std::size_t correlated_ = 0u;
        switch (type.asConditionalOrUnconditional()) {
        case model_t::CResultType::E_Unconditional:
            break;
        case model_t::CResultType::E_Conditional:
            if (!correlated.empty()) {
                correlated_ = correlated[0].first;
            }
            break;
        }
        auto correction = corrections.find(CCorrectionKey(feature, pid, cid, correlated_));
        if (correction != corrections.end()) {
            result -= (*correction).second;
        }
    }
}

double CPopulationModel::propagationTime(std::size_t cid, core_t::TTime time) const {
    return 1.0 + (this->params().s_InitialDecayRateMultiplier - 1.0) *
                     maths::CTools::truncate(
                         1.0 - static_cast<double>(time - m_AttributeFirstBucketTimes[cid]) /
                                   static_cast<double>(3 * core::constants::WEEK),
                         0.0, 1.0);
}

void CPopulationModel::peopleAndAttributesToRemove(core_t::TTime time,
                                                   std::size_t maximumAge,
                                                   TSizeVec& peopleToRemove,
                                                   TSizeVec& attributesToRemove) const {
    if (time <= 0) {
        return;
    }

    const CDataGatherer& gatherer = this->dataGatherer();

    for (std::size_t pid = 0u; pid < m_PersonLastBucketTimes.size(); ++pid) {
        if ((gatherer.isPersonActive(pid)) &&
            (!CAnomalyDetectorModel::isTimeUnset(m_PersonLastBucketTimes[pid]))) {
            std::size_t bucketsSinceLastEvent = static_cast<std::size_t>(
                (time - m_PersonLastBucketTimes[pid]) / gatherer.bucketLength());
            if (bucketsSinceLastEvent > maximumAge) {
                LOG_TRACE(<< gatherer.personName(pid) << ", bucketsSinceLastEvent = " << bucketsSinceLastEvent
                          << ", maximumAge = " << maximumAge);
                peopleToRemove.push_back(pid);
            }
        }
    }

    for (std::size_t cid = 0u; cid < m_AttributeLastBucketTimes.size(); ++cid) {
        if ((gatherer.isAttributeActive(cid)) &&
            (!CAnomalyDetectorModel::isTimeUnset(m_AttributeLastBucketTimes[cid]))) {
            std::size_t bucketsSinceLastEvent = static_cast<std::size_t>(
                (time - m_AttributeLastBucketTimes[cid]) / gatherer.bucketLength());
            if (bucketsSinceLastEvent > maximumAge) {
                LOG_TRACE(<< gatherer.attributeName(cid)
                          << ", bucketsSinceLastEvent = " << bucketsSinceLastEvent
                          << ", maximumAge = " << maximumAge);
                attributesToRemove.push_back(cid);
            }
        }
    }
}

void CPopulationModel::removePeople(const TSizeVec& peopleToRemove) {
    for (std::size_t i = 0u; i < peopleToRemove.size(); ++i) {
        uint32_t pid = static_cast<uint32_t>(peopleToRemove[i]);
        for (std::size_t cid = 0u; cid < m_PersonAttributeBucketCounts.size(); ++cid) {
            m_PersonAttributeBucketCounts[cid].removeFromMap(pid);
        }
        for (std::size_t cid = 0u; cid < m_DistinctPersonCounts.size(); ++cid) {
            m_DistinctPersonCounts[cid].remove(pid);
        }
    }
}

void CPopulationModel::doSkipSampling(core_t::TTime startTime, core_t::TTime endTime) {
    const CDataGatherer& gatherer = this->dataGatherer();
    core_t::TTime gapDuration = endTime - startTime;

    for (std::size_t pid = 0u; pid < m_PersonLastBucketTimes.size(); ++pid) {
        if (gatherer.isPersonActive(pid) &&
            !CAnomalyDetectorModel::isTimeUnset(m_PersonLastBucketTimes[pid])) {
            m_PersonLastBucketTimes[pid] = m_PersonLastBucketTimes[pid] + gapDuration;
        }
    }

    for (std::size_t cid = 0u; cid < m_AttributeLastBucketTimes.size(); ++cid) {
        if (gatherer.isAttributeActive(cid) &&
            !CAnomalyDetectorModel::isTimeUnset(m_AttributeLastBucketTimes[cid])) {
            m_AttributeLastBucketTimes[cid] = m_AttributeLastBucketTimes[cid] + gapDuration;
        }
    }
}

CPopulationModel::CCorrectionKey::CCorrectionKey(model_t::EFeature feature,
                                                 std::size_t pid,
                                                 std::size_t cid,
                                                 std::size_t correlated)
    : m_Feature(feature), m_Pid(pid), m_Cid(cid), m_Correlate(correlated) {
}

bool CPopulationModel::CCorrectionKey::operator==(const CCorrectionKey& rhs) const {
    return m_Feature == rhs.m_Feature && m_Pid == rhs.m_Pid &&
           m_Cid == rhs.m_Cid && m_Correlate == rhs.m_Correlate;
}

std::size_t CPopulationModel::CCorrectionKey::hash() const {
    uint64_t seed = core::CHashing::hashCombine(static_cast<uint64_t>(m_Feature), m_Pid);
    seed = core::CHashing::hashCombine(seed, m_Cid);
    return static_cast<std::size_t>(core::CHashing::hashCombine(seed, m_Correlate));
}
}
}
