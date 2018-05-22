/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CCountingModel.h>

#include <core/CAllocationStrategy.h>
#include <core/CPersistUtils.h>

#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>

#include <model/CAnnotatedProbabilityBuilder.h>
#include <model/CDataGatherer.h>
#include <model/CInterimBucketCorrector.h>
#include <model/CModelDetailsView.h>

namespace ml {
namespace model {
namespace {
const std::string WINDOW_BUCKET_COUNT_TAG("a");
const std::string PERSON_BUCKET_COUNT_TAG("b");
const std::string MEAN_COUNT_TAG("c");
// Extra data tag deprecated at model version 34
// TODO remove on next version bump
//const std::string EXTRA_DATA_TAG("d");
//const std::string INTERIM_BUCKET_CORRECTOR_TAG("e");

const CCountingModel::TStr1Vec EMPTY_STRING_LIST;
}

CCountingModel::CCountingModel(const SModelParams& params,
                               const TDataGathererPtr& dataGatherer,
                               const TInterimBucketCorrectorPtr& interimBucketCorrector)
    : CAnomalyDetectorModel(params, dataGatherer, {}),
      m_StartTime(CAnomalyDetectorModel::TIME_UNSET),
      m_InterimBucketCorrector(interimBucketCorrector) {
}

CCountingModel::CCountingModel(const SModelParams& params,
                               const TDataGathererPtr& dataGatherer,
                               const TInterimBucketCorrectorPtr& interimBucketCorrector,
                               core::CStateRestoreTraverser& traverser)
    : CAnomalyDetectorModel(params, dataGatherer, {}),
      m_StartTime(CAnomalyDetectorModel::TIME_UNSET),
      m_InterimBucketCorrector(interimBucketCorrector) {
    traverser.traverseSubLevel(
        boost::bind(&CCountingModel::acceptRestoreTraverser, this, _1));
}

CCountingModel::CCountingModel(bool isForPersistence, const CCountingModel& other)
    : CAnomalyDetectorModel(isForPersistence, other), m_StartTime(0),
      m_MeanCounts(other.m_MeanCounts) {
    if (!isForPersistence) {
        LOG_ABORT(<< "This constructor only creates clones for persistence");
    }
}

void CCountingModel::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(WINDOW_BUCKET_COUNT_TAG, this->windowBucketCount(),
                         core::CIEEE754::E_SinglePrecision);
    core::CPersistUtils::persist(PERSON_BUCKET_COUNT_TAG,
                                 this->personBucketCounts(), inserter);
    core::CPersistUtils::persist(MEAN_COUNT_TAG, m_MeanCounts, inserter);
}

bool CCountingModel::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        if (name == WINDOW_BUCKET_COUNT_TAG) {
            double count;
            if (core::CStringUtils::stringToType(traverser.value(), count) == false) {
                LOG_ERROR(<< "Invalid bucket count in " << traverser.value());
                return false;
            }
            this->windowBucketCount(count);
        } else if (name == PERSON_BUCKET_COUNT_TAG) {
            if (core::CPersistUtils::restore(name, this->personBucketCounts(),
                                             traverser) == false) {
                LOG_ERROR(<< "Invalid bucket counts in " << traverser.value());
                return false;
            }
        } else if (name == MEAN_COUNT_TAG) {
            if (core::CPersistUtils::restore(name, m_MeanCounts, traverser) == false) {
                LOG_ERROR(<< "Invalid mean counts");
                return false;
            }
        }
    } while (traverser.next());

    return true;
}

CAnomalyDetectorModel* CCountingModel::cloneForPersistence() const {
    return new CCountingModel(true, *this);
}

model_t::EModelType CCountingModel::category() const {
    return model_t::E_Counting;
}

bool CCountingModel::isPopulation() const {
    return false;
}

bool CCountingModel::isEventRate() const {
    return false;
}

bool CCountingModel::isMetric() const {
    return false;
}

CCountingModel::TOptionalUInt64
CCountingModel::currentBucketCount(std::size_t pid, core_t::TTime time) const {
    if (!this->bucketStatsAvailable(time)) {
        LOG_ERROR(<< "No statistics at " << time
                  << ", current bucket = " << this->printCurrentBucket());
        return {};
    }

    auto result = std::lower_bound(m_Counts.begin(), m_Counts.end(), pid,
                                   maths::COrderings::SFirstLess());

    return result != m_Counts.end() && result->first == pid
               ? result->second
               : static_cast<std::uint64_t>(0);
}

CCountingModel::TOptionalDouble CCountingModel::baselineBucketCount(std::size_t pid) const {
    return pid < m_MeanCounts.size() ? maths::CBasicStatistics::mean(m_MeanCounts[pid]) : 0.0;
}

CCountingModel::TDouble1Vec CCountingModel::currentBucketValue(model_t::EFeature /*feature*/,
                                                               std::size_t pid,
                                                               std::size_t /*cid*/,
                                                               core_t::TTime time) const {
    TOptionalUInt64 count = this->currentBucketCount(pid, time);
    return count ? TDouble1Vec{static_cast<double>(*count)} : TDouble1Vec{};
}

CCountingModel::TDouble1Vec
CCountingModel::baselineBucketMean(model_t::EFeature /*feature*/,
                                   std::size_t pid,
                                   std::size_t /*cid*/,
                                   model_t::CResultType /*type*/,
                                   const TSizeDoublePr1Vec& /*correlated*/,
                                   core_t::TTime /*time*/) const {
    TOptionalDouble count = this->baselineBucketCount(pid);
    return count ? TDouble1Vec{*count} : TDouble1Vec{};
}

void CCountingModel::currentBucketPersonIds(core_t::TTime time, TSizeVec& result) const {
    using TSizeUSet = boost::unordered_set<std::size_t>;

    result.clear();

    if (!this->bucketStatsAvailable(time)) {
        LOG_ERROR(<< "No statistics at " << time
                  << ", current bucket = " << this->printCurrentBucket());
        return;
    }

    TSizeUSet people;
    for (const auto& count : m_Counts) {
        people.insert(count.first);
    }
    result.reserve(people.size());
    result.assign(people.begin(), people.end());
}

void CCountingModel::sampleOutOfPhase(core_t::TTime startTime,
                                      core_t::TTime endTime,
                                      CResourceMonitor& resourceMonitor) {
    this->sampleBucketStatistics(startTime, endTime, resourceMonitor);
}

void CCountingModel::sampleBucketStatistics(core_t::TTime startTime,
                                            core_t::TTime endTime,
                                            CResourceMonitor& /*resourceMonitor*/) {
    CDataGatherer& gatherer = this->dataGatherer();

    m_ScheduledEventDescriptions.clear();

    if (!gatherer.dataAvailable(startTime)) {
        return;
    }

    core_t::TTime bucketLength = gatherer.bucketLength();
    for (core_t::TTime time = startTime; time < endTime; time += bucketLength) {
        gatherer.timeNow(time);
        const auto& counts = gatherer.bucketCounts(time);
        std::uint64_t totalCount{0u};
        for (const auto& count : counts) {
            totalCount += CDataGatherer::extractData(count);
        }
        m_InterimBucketCorrector->currentBucketCount(time, totalCount);

        this->updateCurrentBucketsStats(time);

        // Check for scheduled events
        core_t::TTime sampleTime = model_t::sampleTime(
            model_t::E_IndividualCountByBucketAndPerson, time, bucketLength);
        setMatchedEventsDescriptions(sampleTime, time);
    }
}

void CCountingModel::sample(core_t::TTime startTime,
                            core_t::TTime endTime,
                            CResourceMonitor& resourceMonitor) {
    CDataGatherer& gatherer = this->dataGatherer();

    m_ScheduledEventDescriptions.clear();

    if (!gatherer.validateSampleTimes(startTime, endTime)) {
        return;
    }

    this->createUpdateNewModels(startTime, resourceMonitor);

    core_t::TTime bucketLength = gatherer.bucketLength();
    for (core_t::TTime time = startTime; time < endTime; time += bucketLength) {
        gatherer.sampleNow(time);
        this->CAnomalyDetectorModel::sample(time, time + bucketLength, resourceMonitor);
        this->updateCurrentBucketsStats(time);
        std::uint64_t totalCount{0};
        for (const auto& count : m_Counts) {
            m_MeanCounts[count.first].add(static_cast<double>(count.second));
            totalCount += count.second;
        }
        m_InterimBucketCorrector->finalBucketCount(time, totalCount);

        // Check for scheduled events
        core_t::TTime sampleTime = model_t::sampleTime(
            model_t::E_IndividualCountByBucketAndPerson, time, bucketLength);
        setMatchedEventsDescriptions(sampleTime, time);
    }
}

void CCountingModel::setMatchedEventsDescriptions(core_t::TTime sampleTime,
                                                  core_t::TTime bucketStartTime) {
    SModelParams::TStrDetectionRulePrVec matchedEvents = this->checkScheduledEvents(sampleTime);

    if (matchedEvents.empty() == false) {
        TStr1Vec descriptions;
        for (auto& event : matchedEvents) {
            descriptions.push_back(event.first);
        }
        m_ScheduledEventDescriptions[bucketStartTime] = descriptions;
    }
}

SModelParams::TStrDetectionRulePrVec
CCountingModel::checkScheduledEvents(core_t::TTime sampleTime) const {
    const SModelParams::TStrDetectionRulePrVec& events =
        this->params().s_ScheduledEvents.get();
    SModelParams::TStrDetectionRulePrVec matchedEvents;

    for (auto& event : events) {
        // Note that as the counting model is not aware of partitions
        // scheduled events cannot support partitions as the code stands.
        if (event.second.apply(CDetectionRule::E_SkipSampling, boost::cref(*this),
                               model_t::E_IndividualCountByBucketAndPerson,
                               model_t::CResultType(), model_t::INDIVIDUAL_ANALYSIS_ATTRIBUTE_ID,
                               model_t::INDIVIDUAL_ANALYSIS_ATTRIBUTE_ID, sampleTime)) {
            matchedEvents.push_back(event);
        }
    }

    return matchedEvents;
}

void CCountingModel::doSkipSampling(core_t::TTime /*startTime*/, core_t::TTime /*endTime*/) {
}

void CCountingModel::prune(std::size_t /*maximumAge*/) {
}

bool CCountingModel::computeProbability(std::size_t pid,
                                        core_t::TTime startTime,
                                        core_t::TTime endTime,
                                        CPartitioningFields& /*partitioningFields*/,
                                        std::size_t /*numberAttributeProbabilities*/,
                                        SAnnotatedProbability& result) const {
    result = SAnnotatedProbability(1.0);
    result.s_CurrentBucketCount =
        this->currentBucketCount(pid, (startTime + endTime) / 2 - 1);
    result.s_BaselineBucketCount = this->baselineBucketCount(pid);
    return true;
}

bool CCountingModel::computeTotalProbability(const std::string& /*person*/,
                                             std::size_t /*numberAttributeProbabilities*/,
                                             TOptionalDouble& probability,
                                             TAttributeProbability1Vec& attributeProbabilities) const {
    probability.reset(1.0);
    attributeProbabilities.clear();
    return true;
}

uint64_t CCountingModel::checksum(bool includeCurrentBucketStats) const {
    uint64_t result = this->CAnomalyDetectorModel::checksum(includeCurrentBucketStats);
    result = maths::CChecksum::calculate(result, m_MeanCounts);
    if (includeCurrentBucketStats) {
        result = maths::CChecksum::calculate(result, m_StartTime);
        result = maths::CChecksum::calculate(result, m_Counts);
    }
    return result;
}

void CCountingModel::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CCountingModel");
    this->CAnomalyDetectorModel::debugMemoryUsage(mem->addChild());
    core::CMemoryDebug::dynamicSize("m_Counts", m_Counts, mem);
    core::CMemoryDebug::dynamicSize("m_MeanCounts", m_MeanCounts, mem);
    core::CMemoryDebug::dynamicSize("m_InterimBucketCorrector",
                                    m_InterimBucketCorrector, mem);
}

std::size_t CCountingModel::memoryUsage() const {
    std::size_t mem = this->CAnomalyDetectorModel::memoryUsage();
    mem += core::CMemory::dynamicSize(m_Counts);
    mem += core::CMemory::dynamicSize(m_MeanCounts);
    mem += core::CMemory::dynamicSize(m_InterimBucketCorrector);
    return mem;
}

std::size_t CCountingModel::computeMemoryUsage() const {
    return this->memoryUsage();
}

std::size_t CCountingModel::staticSize() const {
    return sizeof(*this);
}

CCountingModel::CModelDetailsViewPtr CCountingModel::details() const {
    return CModelDetailsViewPtr();
}

core_t::TTime CCountingModel::currentBucketStartTime() const {
    return m_StartTime;
}

void CCountingModel::currentBucketStartTime(core_t::TTime time) {
    m_StartTime = time;
}

const CCountingModel::TStr1Vec&
CCountingModel::scheduledEventDescriptions(core_t::TTime time) const {
    auto it = m_ScheduledEventDescriptions.find(time);
    if (it == m_ScheduledEventDescriptions.end()) {
        return EMPTY_STRING_LIST;
    }
    return it->second;
}

double CCountingModel::attributeFrequency(std::size_t /*cid*/) const {
    return 1.0;
}

void CCountingModel::createUpdateNewModels(core_t::TTime /*time*/,
                                           CResourceMonitor& /*resourceMonitor*/) {
    this->updateRecycledModels();
    CDataGatherer& gatherer = this->dataGatherer();
    std::size_t numberNewPeople = gatherer.numberPeople();
    std::size_t numberExistingPeople = m_MeanCounts.size();
    numberNewPeople = numberNewPeople > numberExistingPeople ? numberNewPeople - numberExistingPeople
                                                             : 0;
    if (numberNewPeople > 0) {
        LOG_TRACE(<< "Creating " << numberNewPeople << " new people");
        this->createNewModels(numberNewPeople, 0);
    }
}

void CCountingModel::createNewModels(std::size_t n, std::size_t m) {
    if (n > 0) {
        core::CAllocationStrategy::resize(m_MeanCounts, m_MeanCounts.size() + n);
    }
    this->CAnomalyDetectorModel::createNewModels(n, m);
}

void CCountingModel::updateCurrentBucketsStats(core_t::TTime time) {
    CDataGatherer& gatherer = this->dataGatherer();

    // Currently, we only remember one bucket.
    m_StartTime = time;
    gatherer.personNonZeroCounts(time, m_Counts);

    // Results are only output if currentBucketPersonIds is
    // not empty. Therefore, we need to explicitly set the
    // count to 0 so that we output results.
    if (m_Counts.empty()) {
        m_Counts.emplace_back(0, 0);
    }
}

void CCountingModel::updateRecycledModels() {
    for (auto pid : this->dataGatherer().recycledPersonIds()) {
        if (pid < m_MeanCounts.size()) {
            m_MeanCounts[pid] = TMeanAccumulator();
        }
    }
    this->CAnomalyDetectorModel::updateRecycledModels();
}

void CCountingModel::clearPrunedResources(const TSizeVec& /*people*/,
                                          const TSizeVec& /*attributes*/) {
}

const CInterimBucketCorrector& CCountingModel::interimValueCorrector() const {
    return *m_InterimBucketCorrector;
}

bool CCountingModel::bucketStatsAvailable(core_t::TTime time) const {
    return time >= m_StartTime && time < m_StartTime + this->bucketLength();
}

std::string CCountingModel::printCurrentBucket() const {
    std::ostringstream result;
    result << "[" << m_StartTime << "," << m_StartTime + this->bucketLength() << ")";
    return result.str();
}

CMemoryUsageEstimator* CCountingModel::memoryUsageEstimator() const {
    return nullptr;
}
}
}
