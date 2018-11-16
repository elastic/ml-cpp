/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CIndividualModel.h>

#include <core/CAllocationStrategy.h>
#include <core/CContainerPrinter.h>
#include <core/CFunctional.h>
#include <core/CLogger.h>
#include <core/CStatistics.h>
#include <core/Constants.h>
#include <core/RestoreMacros.h>

#include <maths/CChecksum.h>
#include <maths/CMultivariatePrior.h>
#include <maths/COrderings.h>
#include <maths/CPrior.h>
#include <maths/CTimeSeriesDecomposition.h>

#include <model/CAnnotatedProbabilityBuilder.h>
#include <model/CDataGatherer.h>
#include <model/CModelDetailsView.h>
#include <model/CModelTools.h>
#include <model/CResourceMonitor.h>
#include <model/FrequencyPredicates.h>

#include <algorithm>
#include <map>

namespace ml {
namespace model {

namespace {

using TStrCRef = boost::reference_wrapper<const std::string>;
using TStrCRefUInt64Map = std::map<TStrCRef, uint64_t, maths::COrderings::SLess>;
using TStrCRefStrCRefPr = std::pair<TStrCRef, TStrCRef>;
using TStrCRefStrCRefPrUInt64Map =
    std::map<TStrCRefStrCRefPr, uint64_t, maths::COrderings::SLess>;

//! Update \p hashes with the hashes of the active people in \p values.
template<typename T>
void hashActive(const CDataGatherer& gatherer,
                const std::vector<T>& values,
                TStrCRefUInt64Map& hashes) {
    for (std::size_t pid = 0u; pid < values.size(); ++pid) {
        if (gatherer.isPersonActive(pid)) {
            uint64_t& hash = hashes[boost::cref(gatherer.personName(pid))];
            hash = maths::CChecksum::calculate(hash, values[pid]);
        }
    }
}

const std::size_t CHUNK_SIZE = 500u;

// We use short field names to reduce the state size
const std::string WINDOW_BUCKET_COUNT_TAG("a");
const std::string PERSON_BUCKET_COUNT_TAG("b");
const std::string FIRST_BUCKET_TIME_TAG("c");
const std::string LAST_BUCKET_TIME_TAG("d");
const std::string FEATURE_MODELS_TAG("e");
const std::string FEATURE_CORRELATE_MODELS_TAG("f");
// Extra data tag deprecated at model version 34
// TODO remove on next version bump
//const std::string EXTRA_DATA_TAG("g");
//const std::string INTERIM_BUCKET_CORRECTOR_TAG("h");
const std::string MEMORY_ESTIMATOR_TAG("i");
}

CIndividualModel::CIndividualModel(const SModelParams& params,
                                   const TDataGathererPtr& dataGatherer,
                                   const TFeatureMathsModelSPtrPrVec& newFeatureModels,
                                   const TFeatureMultivariatePriorSPtrPrVec& newFeatureCorrelateModelPriors,
                                   TFeatureCorrelationsPtrPrVec&& featureCorrelatesModels,
                                   const TFeatureInfluenceCalculatorCPtrPrVecVec& influenceCalculators)
    : CAnomalyDetectorModel(params, dataGatherer, influenceCalculators) {
    m_FeatureModels.reserve(newFeatureModels.size());
    for (const auto& model : newFeatureModels) {
        m_FeatureModels.emplace_back(model.first, model.second);
    }
    std::sort(m_FeatureModels.begin(), m_FeatureModels.end(),
              [](const SFeatureModels& lhs, const SFeatureModels& rhs) {
                  return lhs.s_Feature < rhs.s_Feature;
              });

    if (this->params().s_MultivariateByFields) {
        m_FeatureCorrelatesModels.reserve(featureCorrelatesModels.size());
        for (std::size_t i = 0u; i < featureCorrelatesModels.size(); ++i) {
            m_FeatureCorrelatesModels.emplace_back(
                featureCorrelatesModels[i].first,
                newFeatureCorrelateModelPriors[i].second,
                std::move(featureCorrelatesModels[i].second));
        }
        std::sort(m_FeatureCorrelatesModels.begin(), m_FeatureCorrelatesModels.end(),
                  [](const SFeatureCorrelateModels& lhs, const SFeatureCorrelateModels& rhs) {
                      return lhs.s_Feature < rhs.s_Feature;
                  });
    }
}

CIndividualModel::CIndividualModel(bool isForPersistence, const CIndividualModel& other)
    : CAnomalyDetectorModel(isForPersistence, other),
      m_FirstBucketTimes(other.m_FirstBucketTimes),
      m_LastBucketTimes(other.m_LastBucketTimes),
      m_MemoryEstimator(other.m_MemoryEstimator) {
    if (!isForPersistence) {
        LOG_ABORT(<< "This constructor only creates clones for persistence");
    }

    m_FeatureModels.reserve(m_FeatureModels.size());
    for (const auto& feature : other.m_FeatureModels) {
        m_FeatureModels.emplace_back(feature.s_Feature, feature.s_NewModel);
        m_FeatureModels.back().s_Models.reserve(feature.s_Models.size());
        for (const auto& model : feature.s_Models) {
            m_FeatureModels.back().s_Models.emplace_back(model->cloneForPersistence());
        }
    }

    m_FeatureCorrelatesModels.reserve(other.m_FeatureCorrelatesModels.size());
    for (const auto& feature : other.m_FeatureCorrelatesModels) {
        m_FeatureCorrelatesModels.emplace_back(
            feature.s_Feature, feature.s_ModelPrior,
            TCorrelationsPtr(feature.s_Models->cloneForPersistence()));
    }
}

bool CIndividualModel::isPopulation() const {
    return false;
}

CIndividualModel::TOptionalUInt64
CIndividualModel::currentBucketCount(std::size_t pid, core_t::TTime time) const {
    if (!this->bucketStatsAvailable(time)) {
        LOG_ERROR(<< "No statistics at " << time
                  << ", current bucket = " << this->printCurrentBucket());
        return TOptionalUInt64();
    }

    auto result = std::lower_bound(this->currentBucketPersonCounts().begin(),
                                   this->currentBucketPersonCounts().end(), pid,
                                   maths::COrderings::SFirstLess());

    return result != this->currentBucketPersonCounts().end() && result->first == pid
               ? result->second
               : static_cast<uint64_t>(0);
}

bool CIndividualModel::bucketStatsAvailable(core_t::TTime time) const {
    return time >= this->currentBucketStartTime() &&
           time < this->currentBucketStartTime() + this->bucketLength();
}

void CIndividualModel::sampleBucketStatistics(core_t::TTime startTime,
                                              core_t::TTime endTime,
                                              CResourceMonitor& /*resourceMonitor*/) {
    CDataGatherer& gatherer = this->dataGatherer();

    if (!gatherer.dataAvailable(startTime)) {
        return;
    }

    for (core_t::TTime time = startTime, bucketLength = gatherer.bucketLength();
         time < endTime; time += bucketLength) {
        // Currently, we only remember one bucket.
        this->currentBucketStartTime(time);
        TSizeUInt64PrVec& personCounts = this->currentBucketPersonCounts();
        gatherer.personNonZeroCounts(time, personCounts);
        this->applyFilter(model_t::E_XF_By, false, this->personFilter(), personCounts);
    }
}

void CIndividualModel::sample(core_t::TTime startTime,
                              core_t::TTime endTime,
                              CResourceMonitor& resourceMonitor) {
    const CDataGatherer& gatherer = this->dataGatherer();

    for (core_t::TTime time = startTime, bucketLength = gatherer.bucketLength();
         time < endTime; time += bucketLength) {
        this->CAnomalyDetectorModel::sample(time, time + bucketLength, resourceMonitor);

        this->currentBucketStartTime(time);
        TSizeUInt64PrVec& personCounts = this->currentBucketPersonCounts();
        gatherer.personNonZeroCounts(time, personCounts);
        for (const auto& count : personCounts) {
            std::size_t pid = count.first;
            if (CAnomalyDetectorModel::isTimeUnset(m_FirstBucketTimes[pid])) {
                m_FirstBucketTimes[pid] = time;
            }
            m_LastBucketTimes[pid] = time;
        }
        this->applyFilter(model_t::E_XF_By, true, this->personFilter(), personCounts);
    }
}

void CIndividualModel::prune(std::size_t maximumAge) {
    core_t::TTime time = this->currentBucketStartTime();

    if (time <= 0) {
        return;
    }

    CDataGatherer& gatherer = this->dataGatherer();

    TSizeVec peopleToRemove;
    for (std::size_t pid = 0u; pid < m_LastBucketTimes.size(); ++pid) {
        if (gatherer.isPersonActive(pid) &&
            !CAnomalyDetectorModel::isTimeUnset(m_LastBucketTimes[pid])) {
            std::size_t bucketsSinceLastEvent = static_cast<std::size_t>(
                (time - m_LastBucketTimes[pid]) / gatherer.bucketLength());
            if (bucketsSinceLastEvent > maximumAge) {
                LOG_TRACE(<< gatherer.personName(pid) << ", bucketsSinceLastEvent = " << bucketsSinceLastEvent
                          << ", maximumAge = " << maximumAge);
                peopleToRemove.push_back(pid);
            }
        }
    }

    if (peopleToRemove.empty()) {
        return;
    }

    std::sort(peopleToRemove.begin(), peopleToRemove.end());
    LOG_DEBUG(<< "Removing people {" << this->printPeople(peopleToRemove, 20) << '}');

    // We clear large state objects from removed people's model
    // and reinitialize it when they are recycled.
    this->clearPrunedResources(peopleToRemove, TSizeVec());
}

bool CIndividualModel::computeTotalProbability(const std::string& /*person*/,
                                               std::size_t /*numberAttributeProbabilities*/,
                                               TOptionalDouble& probability,
                                               TAttributeProbability1Vec& attributeProbabilities) const {
    probability = TOptionalDouble();
    attributeProbabilities.clear();
    return true;
}

uint64_t CIndividualModel::checksum(bool includeCurrentBucketStats) const {
    uint64_t seed = this->CAnomalyDetectorModel::checksum(includeCurrentBucketStats);

    TStrCRefUInt64Map hashes1;

    const CDataGatherer& gatherer = this->dataGatherer();
    hashActive(gatherer, m_FirstBucketTimes, hashes1);
    hashActive(gatherer, m_LastBucketTimes, hashes1);
    for (const auto& feature : m_FeatureModels) {
        hashActive(gatherer, feature.s_Models, hashes1);
    }

    TStrCRefStrCRefPrUInt64Map hashes2;

    for (const auto& feature : m_FeatureCorrelatesModels) {
        for (const auto& model : feature.s_Models->correlationModels()) {
            std::size_t pids[]{model.first.first, model.first.second};
            if (gatherer.isPersonActive(pids[0]) && gatherer.isPersonActive(pids[1])) {
                uint64_t& hash = hashes2[{boost::cref(this->personName(pids[0])),
                                          boost::cref(this->personName(pids[1]))}];
                hash = maths::CChecksum::calculate(hash, model.second);
            }
        }
    }

    if (includeCurrentBucketStats) {
        seed = maths::CChecksum::calculate(seed, this->currentBucketStartTime());
        const TSizeUInt64PrVec& personCounts = this->currentBucketPersonCounts();
        for (const auto& count : personCounts) {
            uint64_t& hash = hashes1[boost::cref(this->personName(count.first))];
            hash = maths::CChecksum::calculate(hash, count.second);
        }
    }

    LOG_TRACE(<< "seed = " << seed);
    LOG_TRACE(<< "hashes1 = " << core::CContainerPrinter::print(hashes1));
    LOG_TRACE(<< "hashes2 = " << core::CContainerPrinter::print(hashes2));

    seed = maths::CChecksum::calculate(seed, hashes1);
    return maths::CChecksum::calculate(seed, hashes2);
}

void CIndividualModel::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CIndividualModel");
    this->CAnomalyDetectorModel::debugMemoryUsage(mem->addChild());
    core::CMemoryDebug::dynamicSize("m_FirstBucketTimes", m_FirstBucketTimes, mem);
    core::CMemoryDebug::dynamicSize("m_LastBucketTimes", m_LastBucketTimes, mem);
    core::CMemoryDebug::dynamicSize("m_FeatureModels", m_FeatureModels, mem);
    core::CMemoryDebug::dynamicSize("m_FeatureCorrelatesModels",
                                    m_FeatureCorrelatesModels, mem);
    core::CMemoryDebug::dynamicSize("m_MemoryEstimator", m_MemoryEstimator, mem);
}

std::size_t CIndividualModel::memoryUsage() const {
    const CDataGatherer& gatherer = this->dataGatherer();
    TOptionalSize estimate = this->estimateMemoryUsage(
        gatherer.numberActivePeople(), gatherer.numberActiveAttributes(),
        this->numberCorrelations());
    return estimate ? estimate.get() : this->computeMemoryUsage();
}

std::size_t CIndividualModel::computeMemoryUsage() const {
    std::size_t mem = this->CAnomalyDetectorModel::memoryUsage();
    mem += core::CMemory::dynamicSize(m_FirstBucketTimes);
    mem += core::CMemory::dynamicSize(m_LastBucketTimes);
    mem += core::CMemory::dynamicSize(m_FeatureModels);
    mem += core::CMemory::dynamicSize(m_FeatureCorrelatesModels);
    mem += core::CMemory::dynamicSize(m_MemoryEstimator);
    return mem;
}

CMemoryUsageEstimator* CIndividualModel::memoryUsageEstimator() const {
    return &m_MemoryEstimator;
}

std::size_t CIndividualModel::staticSize() const {
    return sizeof(*this);
}

void CIndividualModel::doAcceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(WINDOW_BUCKET_COUNT_TAG, this->windowBucketCount(),
                         core::CIEEE754::E_SinglePrecision);
    core::CPersistUtils::persist(PERSON_BUCKET_COUNT_TAG,
                                 this->personBucketCounts(), inserter);
    core::CPersistUtils::persist(FIRST_BUCKET_TIME_TAG, m_FirstBucketTimes, inserter);
    core::CPersistUtils::persist(LAST_BUCKET_TIME_TAG, m_LastBucketTimes, inserter);
    for (const auto& feature : m_FeatureModels) {
        inserter.insertLevel(FEATURE_MODELS_TAG, boost::bind(&SFeatureModels::acceptPersistInserter,
                                                             &feature, _1));
    }
    for (const auto& feature : m_FeatureCorrelatesModels) {
        inserter.insertLevel(FEATURE_CORRELATE_MODELS_TAG,
                             boost::bind(&SFeatureCorrelateModels::acceptPersistInserter,
                                         &feature, _1));
    }
    core::CPersistUtils::persist(MEMORY_ESTIMATOR_TAG, m_MemoryEstimator, inserter);
}

bool CIndividualModel::doAcceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    std::size_t i = 0u, j = 0u;
    do {
        const std::string& name = traverser.name();
        RESTORE_SETUP_TEARDOWN(WINDOW_BUCKET_COUNT_TAG, double count,
                               core::CStringUtils::stringToType(traverser.value(), count),
                               this->windowBucketCount(count))
        RESTORE(PERSON_BUCKET_COUNT_TAG,
                core::CPersistUtils::restore(name, this->personBucketCounts(), traverser))
        RESTORE(FIRST_BUCKET_TIME_TAG,
                core::CPersistUtils::restore(name, m_FirstBucketTimes, traverser))
        RESTORE(LAST_BUCKET_TIME_TAG,
                core::CPersistUtils::restore(name, m_LastBucketTimes, traverser))
        RESTORE(FEATURE_MODELS_TAG,
                i == m_FeatureModels.size() ||
                    traverser.traverseSubLevel(boost::bind(
                        &SFeatureModels::acceptRestoreTraverser,
                        &m_FeatureModels[i++], boost::cref(this->params()), _1)))
        RESTORE(FEATURE_CORRELATE_MODELS_TAG,
                j == m_FeatureCorrelatesModels.size() ||
                    traverser.traverseSubLevel(boost::bind(
                        &SFeatureCorrelateModels::acceptRestoreTraverser,
                        &m_FeatureCorrelatesModels[j++], boost::cref(this->params()), _1)))
        RESTORE(MEMORY_ESTIMATOR_TAG,
                core::CPersistUtils::restore(MEMORY_ESTIMATOR_TAG, m_MemoryEstimator, traverser))
    } while (traverser.next());

    for (auto& feature : m_FeatureModels) {
        for (auto& model : feature.s_Models) {
            for (const auto& correlates : m_FeatureCorrelatesModels) {
                if (feature.s_Feature == correlates.s_Feature) {
                    model->modelCorrelations(*correlates.s_Models);
                }
            }
        }
    }

    return true;
}

void CIndividualModel::createUpdateNewModels(core_t::TTime time,
                                             CResourceMonitor& resourceMonitor) {
    this->updateRecycledModels();

    CDataGatherer& gatherer = this->dataGatherer();

    std::size_t numberExistingPeople = m_FirstBucketTimes.size();
    std::size_t numberCorrelations = this->numberCorrelations();

    TOptionalSize usageEstimate = this->estimateMemoryUsage(
        std::min(numberExistingPeople, gatherer.numberActivePeople()),
        0, // # attributes
        numberCorrelations);
    std::size_t ourUsage = usageEstimate ? usageEstimate.get()
                                         : this->computeMemoryUsage();
    std::size_t resourceLimit = ourUsage + resourceMonitor.allocationLimit();
    std::size_t numberNewPeople = gatherer.numberPeople();
    numberNewPeople = numberNewPeople > numberExistingPeople ? numberNewPeople - numberExistingPeople
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
        if (numberNewPeople > 0 && resourceMonitor.haveNoLimit() == false) {
            ourUsage = this->estimateMemoryUsageOrComputeAndUpdate(
                numberExistingPeople, 0, numberCorrelations);
        }
    }
    this->estimateMemoryUsageOrComputeAndUpdate(numberExistingPeople, 0, numberCorrelations);

    if (numberNewPeople > 0) {
        resourceMonitor.acceptAllocationFailureResult(time);
        LOG_DEBUG(<< "Not enough memory to create models");
        core::CStatistics::instance()
            .stat(stat_t::E_NumberMemoryLimitModelCreationFailures)
            .increment(numberNewPeople);
        std::size_t toRemove = gatherer.numberPeople() - numberNewPeople;
        gatherer.removePeople(toRemove);
    }

    this->refreshCorrelationModels(resourceLimit, resourceMonitor);
}

void CIndividualModel::createNewModels(std::size_t n, std::size_t m) {
    if (n > 0) {
        std::size_t newN = m_FirstBucketTimes.size() + n;
        core::CAllocationStrategy::resize(m_FirstBucketTimes, newN,
                                          CAnomalyDetectorModel::TIME_UNSET);
        core::CAllocationStrategy::resize(m_LastBucketTimes, newN,
                                          CAnomalyDetectorModel::TIME_UNSET);
        for (auto& feature : m_FeatureModels) {
            core::CAllocationStrategy::reserve(feature.s_Models, newN);
            for (std::size_t pid = feature.s_Models.size(); pid < newN; ++pid) {
                feature.s_Models.emplace_back(feature.s_NewModel->clone(pid));
                for (const auto& correlates : m_FeatureCorrelatesModels) {
                    if (feature.s_Feature == correlates.s_Feature) {
                        feature.s_Models.back()->modelCorrelations(*correlates.s_Models);
                    }
                }
            }
        }
    }
    this->CAnomalyDetectorModel::createNewModels(n, m);
}

void CIndividualModel::updateRecycledModels() {
    for (auto pid : this->dataGatherer().recycledPersonIds()) {
        if (pid < m_FirstBucketTimes.size()) {
            m_FirstBucketTimes[pid] = CAnomalyDetectorModel::TIME_UNSET;
            m_LastBucketTimes[pid] = CAnomalyDetectorModel::TIME_UNSET;
            for (auto& feature : m_FeatureModels) {
                feature.s_Models[pid].reset(feature.s_NewModel->clone(pid));
                for (const auto& correlates : m_FeatureCorrelatesModels) {
                    if (feature.s_Feature == correlates.s_Feature) {
                        feature.s_Models.back()->modelCorrelations(*correlates.s_Models);
                    }
                }
            }
        }
    }
    this->CAnomalyDetectorModel::updateRecycledModels();
}

void CIndividualModel::refreshCorrelationModels(std::size_t resourceLimit,
                                                CResourceMonitor& resourceMonitor) {
    std::size_t n = this->numberOfPeople();
    double maxNumberCorrelations = this->params().s_CorrelationModelsOverhead *
                                   static_cast<double>(n);
    auto memoryUsage = boost::bind(&CAnomalyDetectorModel::estimateMemoryUsageOrComputeAndUpdate,
                                   this, n, 0, _1);
    CTimeSeriesCorrelateModelAllocator allocator(
        resourceMonitor, memoryUsage, resourceLimit,
        static_cast<std::size_t>(maxNumberCorrelations));
    for (auto& feature : m_FeatureCorrelatesModels) {
        allocator.prototypePrior(feature.s_ModelPrior);
        feature.s_Models->refresh(allocator);
    }
}

void CIndividualModel::clearPrunedResources(const TSizeVec& people,
                                            const TSizeVec& /*attributes*/) {
    for (auto pid : people) {
        for (auto& feature : m_FeatureModels) {
            if (pid < feature.s_Models.size()) {
                feature.s_Models[pid].reset(this->tinyModel());
            }
        }
    }
}

double CIndividualModel::emptyBucketWeight(model_t::EFeature feature,
                                           std::size_t pid,
                                           core_t::TTime time) const {
    double result = 1.0;
    if (model_t::countsEmptyBuckets(feature)) {
        TOptionalUInt64 count = this->currentBucketCount(pid, time);
        if (!count || *count == 0) {
            double frequency = this->personFrequency(pid);
            result = model_t::emptyBucketCountWeight(
                feature, frequency, this->params().s_CutoffToModelEmptyBuckets);
        }
    }
    return result;
}

double CIndividualModel::probabilityBucketEmpty(model_t::EFeature feature,
                                                std::size_t pid) const {
    double result = 0.0;
    if (model_t::countsEmptyBuckets(feature)) {
        double frequency = this->personFrequency(pid);
        double emptyBucketWeight = model_t::emptyBucketCountWeight(
            feature, frequency, this->params().s_CutoffToModelEmptyBuckets);
        result = (1.0 - frequency) * (1.0 - emptyBucketWeight);
    }
    return result;
}

const maths::CModel* CIndividualModel::model(model_t::EFeature feature, std::size_t pid) const {
    return const_cast<CIndividualModel*>(this)->model(feature, pid);
}

maths::CModel* CIndividualModel::model(model_t::EFeature feature, std::size_t pid) {
    auto i = std::find_if(m_FeatureModels.begin(), m_FeatureModels.end(),
                          [feature](const SFeatureModels& model) {
                              return model.s_Feature == feature;
                          });
    return i != m_FeatureModels.end() && pid < i->s_Models.size()
               ? i->s_Models[pid].get()
               : nullptr;
}

void CIndividualModel::sampleCorrelateModels() {
    for (const auto& feature : m_FeatureCorrelatesModels) {
        feature.s_Models->processSamples();
    }
}

void CIndividualModel::correctBaselineForInterim(model_t::EFeature feature,
                                                 std::size_t pid,
                                                 model_t::CResultType type,
                                                 const TSizeDoublePr1Vec& correlated,
                                                 const TFeatureSizeSizeTripleDouble1VecUMap& corrections,
                                                 TDouble1Vec& result) const {
    if (type.isInterim() && model_t::requiresInterimResultAdjustment(feature)) {
        TFeatureSizeSizeTriple key(feature, pid, pid);
        switch (type.asConditionalOrUnconditional()) {
        case model_t::CResultType::E_Unconditional:
            break;
        case model_t::CResultType::E_Conditional:
            if (!correlated.empty()) {
                key.third = correlated[0].first;
            }
            break;
        }
        auto correction = corrections.find(key);
        if (correction != corrections.end()) {
            result -= correction->second;
        }
    }
}

const CIndividualModel::TTimeVec& CIndividualModel::firstBucketTimes() const {
    return m_FirstBucketTimes;
}

const CIndividualModel::TTimeVec& CIndividualModel::lastBucketTimes() const {
    return m_LastBucketTimes;
}

double CIndividualModel::derate(std::size_t pid, core_t::TTime time) const {
    return std::max(1.0 - static_cast<double>(time - m_FirstBucketTimes[pid]) /
                              static_cast<double>(3 * core::constants::WEEK),
                    0.0);
}

std::string CIndividualModel::printCurrentBucket() const {
    std::ostringstream result;
    result << "[" << this->currentBucketStartTime() << ","
           << this->currentBucketStartTime() + this->bucketLength() << ")";
    return result.str();
}

std::size_t CIndividualModel::numberCorrelations() const {
    std::size_t result = 0u;
    for (const auto& feature : m_FeatureCorrelatesModels) {
        result += feature.s_Models->correlationModels().size();
    }
    return result;
}

double CIndividualModel::attributeFrequency(std::size_t /*cid*/) const {
    return 1.0;
}

void CIndividualModel::doSkipSampling(core_t::TTime startTime, core_t::TTime endTime) {
    core_t::TTime gap = endTime - startTime;

    for (auto& time : m_LastBucketTimes) {
        if (!CAnomalyDetectorModel::isTimeUnset(time)) {
            time = time + gap;
        }
    }

    for (auto& feature : m_FeatureModels) {
        for (auto& model : feature.s_Models) {
            model->skipTime(gap);
        }
    }
}
}
}
