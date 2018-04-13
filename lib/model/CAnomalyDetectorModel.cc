/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CAnomalyDetectorModel.h>

#include <core/CAllocationStrategy.h>
#include <core/CFunctional.h>
#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CStatistics.h>
#include <core/RestoreMacros.h>

#include <maths/CChecksum.h>
#include <maths/CMathsFuncs.h>
#include <maths/CModelStateSerialiser.h>
#include <maths/CMultivariatePrior.h>
#include <maths/COrderings.h>
#include <maths/CRestoreParams.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CDataGatherer.h>
#include <model/CDetectionRule.h>
#include <model/CHierarchicalResults.h>
#include <model/CInterimBucketCorrector.h>
#include <model/CProbabilityAndInfluenceCalculator.h>
#include <model/CResourceMonitor.h>
#include <model/FrequencyPredicates.h>

#include <boost/bind.hpp>

#include <algorithm>

namespace ml
{
namespace model
{
namespace
{
const std::string MODEL_TAG{"a"};
const std::string EMPTY;

const model_t::CResultType SKIP_SAMPLING_RESULT_TYPE;

const CAnomalyDetectorModel::TStr1Vec EMPTY_STRING_LIST;

bool checkRules(const SModelParams::TDetectionRuleVec &detectionRules,
                const CAnomalyDetectorModel &model,
                model_t::EFeature feature,
                CDetectionRule::ERuleAction action,
                const model_t::CResultType &resultType,
                std::size_t pid,
                std::size_t cid,
                core_t::TTime time)
{
    bool isIgnored{false};
    for (auto &rule : detectionRules)
    {
        isIgnored = isIgnored || rule.apply(action,
                                            model,
                                            feature,
                                            resultType,
                                            pid, cid, time);
    }
    return isIgnored;
}

bool checkScheduledEvents(const SModelParams::TStrDetectionRulePrVec &scheduledEvents,
                          const CAnomalyDetectorModel &model,
                          model_t::EFeature feature,
                          CDetectionRule::ERuleAction action,
                          const model_t::CResultType &resultType,
                          std::size_t pid,
                          std::size_t cid,
                          core_t::TTime time)
{
    bool isIgnored{false};
    for (auto &event : scheduledEvents)
    {
        isIgnored = isIgnored || event.second.apply(action,
                                                    model,
                                                    feature,
                                                    resultType,
                                                    pid, cid, time);
    }
    return isIgnored;
}

}

CAnomalyDetectorModel::CAnomalyDetectorModel(const SModelParams &params,
                                             const TDataGathererPtr &dataGatherer,
                                             const TFeatureInfluenceCalculatorCPtrPrVecVec &influenceCalculators) :
        m_Params(params),
        m_DataGatherer(dataGatherer),
        m_BucketCount(0.0),
        m_InfluenceCalculators(influenceCalculators),
        m_InterimBucketCorrector(new CInterimBucketCorrector(dataGatherer->bucketLength()))
{
    if (!m_DataGatherer)
    {
        LOG_ABORT("Must provide a data gatherer");
    }
    for (auto &calculators : m_InfluenceCalculators)
    {
        std::sort(calculators.begin(), calculators.end(), maths::COrderings::SFirstLess());
    }
}

CAnomalyDetectorModel::CAnomalyDetectorModel(bool isForPersistence, const CAnomalyDetectorModel &other) :
        // The copy of m_DataGatherer is a shallow copy.  This would be unacceptable
        // if we were going to persist the data gatherer from within this class.
        // We don't, so that's OK, but the next issue is that another thread will be
        // modifying the data gatherer m_DataGatherer points to whilst this object
        // is being persisted.  Therefore, persistence must only call methods on the
        // data gatherer that are invariant.
        m_Params(other.m_Params),
        m_DataGatherer(other.m_DataGatherer),
        m_PersonBucketCounts(other.m_PersonBucketCounts),
        m_BucketCount(other.m_BucketCount),
        m_InfluenceCalculators(),
        m_InterimBucketCorrector(new CInterimBucketCorrector(*other.m_InterimBucketCorrector))
{
    if (!isForPersistence)
    {
        LOG_ABORT("This constructor only creates clones for persistence");
    }
}

std::string CAnomalyDetectorModel::description() const
{
    return m_DataGatherer->description();
}

const std::string &CAnomalyDetectorModel::personName(std::size_t pid) const
{
    return m_DataGatherer->personName(pid, core::CStringUtils::typeToString(pid));
}

const std::string &CAnomalyDetectorModel::personName(std::size_t pid,
                                                     const std::string &fallback) const
{
    return m_DataGatherer->personName(pid, fallback);
}

std::string CAnomalyDetectorModel::printPeople(const TSizeVec &pids, std::size_t limit) const
{
    if (pids.empty())
    {
        return std::string();
    }
    if (limit == 0)
    {
        return core::CStringUtils::typeToString(pids.size()) + " in total";
    }
    std::string result{this->personName(pids[0])};
    for (std::size_t i = 1u; i < std::min(limit, pids.size()); ++i)
    {
        result += ' ';
        result += this->personName(pids[i]);
    }
    if (limit < pids.size())
    {
        result += " and ";
        result += core::CStringUtils::typeToString(pids.size() - limit);
        result += " others";
    }
    return result;
}

std::size_t CAnomalyDetectorModel::numberOfPeople() const
{
    return m_DataGatherer->numberActivePeople();
}

const std::string &CAnomalyDetectorModel::attributeName(std::size_t cid) const
{
    return m_DataGatherer->attributeName(cid, core::CStringUtils::typeToString(cid));
}

const std::string &CAnomalyDetectorModel::attributeName(std::size_t cid,
                                         const std::string &fallback) const
{
    return m_DataGatherer->attributeName(cid, fallback);
}

std::string CAnomalyDetectorModel::printAttributes(const TSizeVec &cids, std::size_t limit) const
{
    if (cids.empty())
    {
        return std::string();
    }
    if (limit == 0)
    {
        return core::CStringUtils::typeToString(cids.size()) + " in total";
    }
    std::string result{this->attributeName(cids[0])};
    for (std::size_t i = 1u; i < std::min(limit, cids.size()); ++i)
    {
        result += ' ';
        result += this->attributeName(cids[i]);
    }
    if (limit < cids.size())
    {
        result += " and ";
        result += core::CStringUtils::typeToString(cids.size() - limit);
        result += " others";
    }
    return result;
}

void CAnomalyDetectorModel::sampleBucketStatistics(core_t::TTime startTime,
                                                core_t::TTime endTime,
                                                CResourceMonitor &/*resourceMonitor*/)
{
    const CDataGatherer &gatherer{this->dataGatherer()};
    core_t::TTime bucketLength{this->bucketLength()};
    for (core_t::TTime time = startTime; time < endTime; time += bucketLength)
    {
        const auto &counts = gatherer.bucketCounts(time);
        std::size_t totalBucketCount{0u};
        for (const auto &count : counts)
        {
            totalBucketCount += CDataGatherer::extractData(count);
        }
        this->currentBucketTotalCount(totalBucketCount);
    }
}

void CAnomalyDetectorModel::sample(core_t::TTime startTime,
                                core_t::TTime endTime,
                                CResourceMonitor &/*resourceMonitor*/)
{
    using TSizeUSet = boost::unordered_set<std::size_t>;

    const CDataGatherer &gatherer{this->dataGatherer()};

    core_t::TTime bucketLength{this->bucketLength()};
    for (core_t::TTime time = startTime; time < endTime; time += bucketLength)
    {
        const auto &counts = gatherer.bucketCounts(time);
        std::size_t totalBucketCount{0u};

        TSizeUSet uniquePeople;
        for (const auto &count : counts)
        {
            std::size_t pid{CDataGatherer::extractPersonId(count)};
            if (uniquePeople.insert(pid).second)
            {
                m_PersonBucketCounts[pid] += 1.0;
            }
            totalBucketCount += CDataGatherer::extractData(count);
        }

        m_InterimBucketCorrector->update(time, totalBucketCount);
        this->currentBucketTotalCount(totalBucketCount);
        m_BucketCount += 1.0;

        double alpha{std::exp(-this->params().s_DecayRate)};
        for (std::size_t pid = 0u; pid < m_PersonBucketCounts.size(); ++pid)
        {
            m_PersonBucketCounts[pid] *= alpha;
        }
        m_BucketCount *= alpha;
    }
}

void CAnomalyDetectorModel::skipSampling(core_t::TTime endTime)
{
    CDataGatherer &gatherer{this->dataGatherer()};
    core_t::TTime startTime{gatherer.earliestBucketStartTime()};

    if (!gatherer.validateSampleTimes(startTime, endTime))
    {
        return;
    }

    gatherer.skipSampleNow(endTime);
    this->doSkipSampling(startTime, endTime);
    this->currentBucketStartTime(endTime - gatherer.bucketLength());
}

bool CAnomalyDetectorModel::addResults(int detector,
                                    core_t::TTime startTime,
                                    core_t::TTime endTime,
                                    std::size_t numberAttributeProbabilities,
                                    CHierarchicalResults &results) const
{
    TSizeVec personIds;
    if (!this->bucketStatsAvailable(startTime))
    {
        LOG_TRACE("No stats available for time " << startTime);
        return false;
    }
    this->currentBucketPersonIds(startTime, personIds);
    LOG_TRACE("Outputting results for " << personIds.size() << " people");

    CPartitioningFields partitioningFields(m_DataGatherer->partitionFieldName(), m_DataGatherer->partitionFieldValue());
    partitioningFields.add(m_DataGatherer->personFieldName(), EMPTY);

    for (auto pid : personIds)
    {
        if (this->category() == model_t::E_Counting)
        {
            SAnnotatedProbability annotatedProbability;
            this->computeProbability(pid, startTime, endTime, partitioningFields,
                                     numberAttributeProbabilities, annotatedProbability);
            results.addSimpleCountResult(annotatedProbability, this, startTime);
        }
        else
        {
            LOG_TRACE("AddResult, for time [" << startTime << "," << endTime << ")");
            partitioningFields.back().second = boost::cref(this->personName(pid));
            std::for_each(m_DataGatherer->beginInfluencers(),
                          m_DataGatherer->endInfluencers(),
                          [&results](const std::string &influencer)
                          { results.addInfluencer(influencer); });
            SAnnotatedProbability annotatedProbability;
            annotatedProbability.s_ResultType = results.resultType();
            if (this->computeProbability(pid, startTime, endTime, partitioningFields,
                                         numberAttributeProbabilities, annotatedProbability))
            {
                function_t::EFunction function{m_DataGatherer->function()};
                results.addModelResult(detector,
                                       this->isPopulation(),
                                       function_t::name(function),
                                       function,
                                       m_DataGatherer->partitionFieldName(),
                                       m_DataGatherer->partitionFieldValue(),
                                       m_DataGatherer->personFieldName(),
                                       this->personName(pid),
                                       m_DataGatherer->valueFieldName(),
                                       annotatedProbability,
                                       this,
                                       startTime);
            }
        }
    }

    return true;
}

std::size_t CAnomalyDetectorModel::defaultPruneWindow() const
{
    // The longest we'll consider keeping priors for is 1M buckets.
    double decayRate{this->params().s_DecayRate};
    double factor{this->params().s_PruneWindowScaleMaximum};
    return (decayRate == 0.0) ?
           MAXIMUM_PERMITTED_AGE :
           std::min(static_cast<std::size_t>(factor / decayRate), MAXIMUM_PERMITTED_AGE);
}

std::size_t CAnomalyDetectorModel::minimumPruneWindow() const
{
    double decayRate{this->params().s_DecayRate};
    double factor{this->params().s_PruneWindowScaleMinimum};
    return (decayRate == 0.0) ?
           MAXIMUM_PERMITTED_AGE :
           std::min(static_cast<std::size_t>(factor / decayRate), MAXIMUM_PERMITTED_AGE);
}

void CAnomalyDetectorModel::prune()
{
    this->prune(this->defaultPruneWindow());
}

uint64_t CAnomalyDetectorModel::checksum(bool /*includeCurrentBucketStats*/) const
{
    using TStrCRefUInt64Map = std::map<TStrCRef, uint64_t, maths::COrderings::SLess>;
    uint64_t seed{m_DataGatherer->checksum()};
    seed = maths::CChecksum::calculate(seed, m_Params);
    seed = maths::CChecksum::calculate(seed, m_BucketCount);
    TStrCRefUInt64Map hashes;
    for (std::size_t pid = 0u; pid < m_PersonBucketCounts.size(); ++pid)
    {
        if (m_DataGatherer->isPersonActive(pid))
        {
            uint64_t &hash{hashes[boost::cref(m_DataGatherer->personName(pid))]};
            hash = maths::CChecksum::calculate(hash, m_PersonBucketCounts[pid]);
        }
    }
    LOG_TRACE("seed = " << seed);
    LOG_TRACE("checksums = " << core::CContainerPrinter::print(hashes));
    return maths::CChecksum::calculate(seed, hashes);
}

void CAnomalyDetectorModel::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    mem->setName("CAnomalyDetectorModel");
    core::CMemoryDebug::dynamicSize("m_DataGatherer", m_DataGatherer, mem);
    core::CMemoryDebug::dynamicSize("m_Params", m_Params, mem);
    core::CMemoryDebug::dynamicSize("m_PersonBucketCounts", m_PersonBucketCounts, mem);
    core::CMemoryDebug::dynamicSize("m_InfluenceCalculators", m_InfluenceCalculators, mem);
    core::CMemoryDebug::dynamicSize("m_InterimBucketCorrector", m_InterimBucketCorrector, mem);
}

std::size_t CAnomalyDetectorModel::memoryUsage() const
{
    std::size_t mem{core::CMemory::dynamicSize(m_Params)};
    mem += core::CMemory::dynamicSize(m_DataGatherer);
    mem += core::CMemory::dynamicSize(m_PersonBucketCounts);
    mem += core::CMemory::dynamicSize(m_InfluenceCalculators);
    mem += core::CMemory::dynamicSize(m_InterimBucketCorrector);
    return mem;
}

CAnomalyDetectorModel::TOptionalSize CAnomalyDetectorModel::estimateMemoryUsage(std::size_t numberPeople,
                                                                              std::size_t numberAttributes,
                                                                              std::size_t numberCorrelations) const
{
    CMemoryUsageEstimator::TSizeArray predictors;
    predictors[CMemoryUsageEstimator::E_People]       = numberPeople;
    predictors[CMemoryUsageEstimator::E_Attributes]   = numberAttributes;
    predictors[CMemoryUsageEstimator::E_Correlations] = numberCorrelations;
    return this->memoryUsageEstimator()->estimate(predictors);
}

std::size_t CAnomalyDetectorModel::estimateMemoryUsageOrComputeAndUpdate(std::size_t numberPeople,
                                                                      std::size_t numberAttributes,
                                                                      std::size_t numberCorrelations)
{
    TOptionalSize estimate{this->estimateMemoryUsage(numberPeople,
                                                     numberAttributes,
                                                     numberCorrelations)};
    if (estimate)
    {
        return estimate.get();
    }

    std::size_t computed{this->computeMemoryUsage()};
    CMemoryUsageEstimator::TSizeArray predictors;
    predictors[CMemoryUsageEstimator::E_People]       = numberPeople;
    predictors[CMemoryUsageEstimator::E_Attributes]   = numberAttributes;
    predictors[CMemoryUsageEstimator::E_Correlations] = numberCorrelations;
    this->memoryUsageEstimator()->addValue(predictors, computed);
    return computed;
}

const CDataGatherer &CAnomalyDetectorModel::dataGatherer() const
{
    return *m_DataGatherer;
}

CDataGatherer &CAnomalyDetectorModel::dataGatherer()
{
    return *m_DataGatherer;
}

core_t::TTime CAnomalyDetectorModel::bucketLength() const
{
    return m_DataGatherer->bucketLength();
}

double CAnomalyDetectorModel::personFrequency(std::size_t pid) const
{
    return m_BucketCount <= 0.0 ? 0.5 : m_PersonBucketCounts[pid] / m_BucketCount;
}

bool CAnomalyDetectorModel::isTimeUnset(core_t::TTime time)
{
    return time == TIME_UNSET;
}

CPersonFrequencyGreaterThan CAnomalyDetectorModel::personFilter() const
{
    return CPersonFrequencyGreaterThan(*this, m_Params.get().s_ExcludePersonFrequency);
}

CAttributeFrequencyGreaterThan CAnomalyDetectorModel::attributeFilter() const
{
    return CAttributeFrequencyGreaterThan(*this, m_Params.get().s_ExcludeAttributeFrequency);
}

const SModelParams &CAnomalyDetectorModel::params() const
{
    return m_Params;
}

double CAnomalyDetectorModel::learnRate(model_t::EFeature feature) const
{
    return model_t::learnRate(feature, m_Params);
}

const CInfluenceCalculator *CAnomalyDetectorModel::influenceCalculator(model_t::EFeature feature,
                                                                        std::size_t iid) const
{
    if (iid >= m_InfluenceCalculators.size())
    {
        LOG_ERROR("Influencer identifier " << iid << " out of range");
        return 0;
    }
    const TFeatureInfluenceCalculatorCPtrPrVec &calculators{m_InfluenceCalculators[iid]};
    auto result = std::lower_bound(calculators.begin(),
                                   calculators.end(),
                                   feature,
                                   maths::COrderings::SFirstLess());
    return result != calculators.end() && result->first == feature ? result->second.get() : 0;
}

const CAnomalyDetectorModel::TDoubleVec &CAnomalyDetectorModel::personBucketCounts() const
{
    return m_PersonBucketCounts;
}

CAnomalyDetectorModel::TDoubleVec &CAnomalyDetectorModel::personBucketCounts()
{
    return m_PersonBucketCounts;
}

void CAnomalyDetectorModel::windowBucketCount(double windowBucketCount)
{
    m_BucketCount = windowBucketCount;
}

double CAnomalyDetectorModel::windowBucketCount() const
{
    return m_BucketCount;
}

void CAnomalyDetectorModel::createNewModels(std::size_t n, std::size_t /*m*/)
{
    if (n > 0)
    {
        n += m_PersonBucketCounts.size();
        core::CAllocationStrategy::resize(m_PersonBucketCounts, n, 0.0);
    }
}

void CAnomalyDetectorModel::updateRecycledModels()
{
    TSizeVec &people{m_DataGatherer->recycledPersonIds()};
    for (auto pid : people)
    {
        m_PersonBucketCounts[pid] = 0.0;
    }
    people.clear();
}

const CInterimBucketCorrector &CAnomalyDetectorModel::interimValueCorrector() const
{
    return *m_InterimBucketCorrector;
}

bool CAnomalyDetectorModel::shouldIgnoreResult(model_t::EFeature feature,
                                            const model_t::CResultType &resultType,
                                            std::size_t pid,
                                            std::size_t cid,
                                            core_t::TTime time) const
{
    bool shouldIgnore = checkScheduledEvents(this->params().s_ScheduledEvents.get(), boost::cref(*this), feature,
                                            CDetectionRule::E_FilterResults, resultType, pid, cid, time) ||
                        checkRules(this->params().s_DetectionRules.get(), boost::cref(*this), feature,
                                  CDetectionRule::E_FilterResults, resultType, pid, cid, time);

    return shouldIgnore;
}

bool CAnomalyDetectorModel::shouldIgnoreSample(model_t::EFeature feature,
                                            std::size_t pid,
                                            std::size_t cid,
                                            core_t::TTime time) const
{
    bool shouldIgnore = checkScheduledEvents(this->params().s_ScheduledEvents.get(), boost::cref(*this), feature,
                                            CDetectionRule::E_SkipSampling, SKIP_SAMPLING_RESULT_TYPE, pid, cid, time) ||
                        checkRules(this->params().s_DetectionRules.get(), boost::cref(*this), feature,
                                   CDetectionRule::E_SkipSampling, SKIP_SAMPLING_RESULT_TYPE, pid, cid, time);

    return shouldIgnore;
}

bool CAnomalyDetectorModel::interimBucketCorrectorAcceptRestoreTraverser(
                                            core::CStateRestoreTraverser &traverser)
{
    if (traverser.traverseSubLevel(boost::bind(&CInterimBucketCorrector::acceptRestoreTraverser,
                                               m_InterimBucketCorrector.get(), _1)) == false)
    {
        LOG_ERROR("Invalid interim bucket corrector");
        return false;
    }
    return true;
}

void CAnomalyDetectorModel::interimBucketCorrectorAcceptPersistInserter(const std::string &tag,
                                                         core::CStatePersistInserter &inserter) const
{
    inserter.insertLevel(tag, boost::bind(&CInterimBucketCorrector::acceptPersistInserter,
                                          m_InterimBucketCorrector.get(), _1));
}

const CAnomalyDetectorModel::TStr1Vec &CAnomalyDetectorModel::scheduledEventDescriptions(core_t::TTime /*time*/) const
{
    return EMPTY_STRING_LIST;
}

maths::CModel *CAnomalyDetectorModel::tinyModel()
{
    return new maths::CModelStub;
}

const std::size_t   CAnomalyDetectorModel::MAXIMUM_PERMITTED_AGE(1000000);
const core_t::TTime CAnomalyDetectorModel::TIME_UNSET(-1);
const std::string   CAnomalyDetectorModel::EMPTY_STRING;


CAnomalyDetectorModel::SFeatureModels::SFeatureModels(model_t::EFeature feature, TMathsModelPtr newModel) :
        s_Feature(feature), s_NewModel(newModel)
{}

bool CAnomalyDetectorModel::SFeatureModels::acceptRestoreTraverser(const SModelParams &params_,
                                                                   core::CStateRestoreTraverser &traverser)
{
    maths_t::EDataType dataType{s_NewModel->dataType()};
    maths::SModelRestoreParams params{s_NewModel->params(),
                                      params_.decompositionRestoreParams(dataType),
                                      params_.distributionRestoreParams(dataType)};
    do
    {
        if (traverser.name() == MODEL_TAG)
        {
            TMathsModelPtr prior;
            if (!traverser.traverseSubLevel(boost::bind<bool>(maths::CModelStateSerialiser(),
                                                              boost::cref(params), boost::ref(prior), _1)))
            {
                return false;
            }
            s_Models.push_back(prior);
        }
    }
    while (traverser.next());
    return true;
}

void CAnomalyDetectorModel::SFeatureModels::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    for (const auto &model : s_Models)
    {
        inserter.insertLevel(MODEL_TAG, boost::bind<void>(maths::CModelStateSerialiser(),
                                                          boost::cref(*model), _1));
    }
}

void CAnomalyDetectorModel::SFeatureModels::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    mem->setName("SFeatureModels");
    core::CMemoryDebug::dynamicSize("s_NewModel", s_NewModel, mem);
    core::CMemoryDebug::dynamicSize("s_Models", s_Models, mem);
}

std::size_t CAnomalyDetectorModel::SFeatureModels::memoryUsage() const
{
    return core::CMemory::dynamicSize(s_NewModel) + core::CMemory::dynamicSize(s_Models);
}


CAnomalyDetectorModel::SFeatureCorrelateModels::SFeatureCorrelateModels(model_t::EFeature feature,
                                                                     TMultivariatePriorPtr modelPrior,
                                                                     TCorrelationsPtr model) :
        s_Feature(feature),
        s_ModelPrior(modelPrior),
        s_Models(model->clone())
{}

bool CAnomalyDetectorModel::SFeatureCorrelateModels::acceptRestoreTraverser(const SModelParams &params_,
                                                                        core::CStateRestoreTraverser &traverser)
{
    maths_t::EDataType dataType{s_ModelPrior->dataType()};
    maths::SDistributionRestoreParams params{params_.distributionRestoreParams(dataType)};
    std::size_t count{0u};
    do
    {
        if (traverser.name() == MODEL_TAG)
        {
            if (  !traverser.traverseSubLevel(boost::bind(&maths::CTimeSeriesCorrelations::acceptRestoreTraverser,
                                                          s_Models.get(), boost::cref(params), _1))
                || count++ > 0)
            {
                return false;
            }
        }
    }
    while (traverser.next());
    return true;
}

void CAnomalyDetectorModel::SFeatureCorrelateModels::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertLevel(MODEL_TAG, boost::bind(&maths::CTimeSeriesCorrelations::acceptPersistInserter,
                                                s_Models.get(), _1));
}

void CAnomalyDetectorModel::SFeatureCorrelateModels::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    mem->setName("SFeatureCorrelateModels");
    core::CMemoryDebug::dynamicSize("s_ModelPrior", s_ModelPrior, mem);
    core::CMemoryDebug::dynamicSize("s_Models", s_Models, mem);
}

std::size_t CAnomalyDetectorModel::SFeatureCorrelateModels::memoryUsage() const
{
    return core::CMemory::dynamicSize(s_ModelPrior) + core::CMemory::dynamicSize(s_Models);
}


CAnomalyDetectorModel::CTimeSeriesCorrelateModelAllocator::CTimeSeriesCorrelateModelAllocator(
                                                                               CResourceMonitor &resourceMonitor,
                                                                               TMemoryUsage memoryUsage,
                                                                               std::size_t resourceLimit,
                                                                               std::size_t maxNumberCorrelations) :
        m_ResourceMonitor(&resourceMonitor),
        m_MemoryUsage(memoryUsage),
        m_ResourceLimit(resourceLimit),
        m_MaxNumberCorrelations(maxNumberCorrelations)
{}

bool CAnomalyDetectorModel::CTimeSeriesCorrelateModelAllocator::areAllocationsAllowed() const
{
    return m_ResourceMonitor->areAllocationsAllowed();
}

bool CAnomalyDetectorModel::CTimeSeriesCorrelateModelAllocator::exceedsLimit(std::size_t correlations) const
{
    return  !m_ResourceMonitor->haveNoLimit()
          && m_MemoryUsage(correlations) >= m_ResourceLimit;
}

std::size_t CAnomalyDetectorModel::CTimeSeriesCorrelateModelAllocator::maxNumberCorrelations() const
{
    return m_MaxNumberCorrelations;
}

std::size_t CAnomalyDetectorModel::CTimeSeriesCorrelateModelAllocator::chunkSize() const
{
    return 500;
}

CAnomalyDetectorModel::TMultivariatePriorPtr
CAnomalyDetectorModel::CTimeSeriesCorrelateModelAllocator::newPrior() const
{
    return TMultivariatePriorPtr(m_PrototypePrior->clone());
}

void CAnomalyDetectorModel::CTimeSeriesCorrelateModelAllocator::prototypePrior(const TMultivariatePriorPtr &prior)
{
    m_PrototypePrior = prior;
}

}
}
