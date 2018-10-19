/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CEventRateModel.h>

#include <core/CContainerPrinter.h>
#include <core/CFunctional.h>
#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CStatistics.h>
#include <core/CoreTypes.h>

#include <maths/CChecksum.h>
#include <maths/CMultivariatePrior.h>
#include <maths/COrderings.h>
#include <maths/CPrior.h>
#include <maths/CRestoreParams.h>
#include <maths/CTools.h>
#include <maths/ProbabilityAggregators.h>

#include <model/CAnnotatedProbabilityBuilder.h>
#include <model/CDataGatherer.h>
#include <model/CIndividualModelDetail.h>
#include <model/CInterimBucketCorrector.h>
#include <model/CModelDetailsView.h>
#include <model/CModelTools.h>
#include <model/CProbabilityAndInfluenceCalculator.h>
#include <model/CResourceMonitor.h>
#include <model/FrequencyPredicates.h>

#include <boost/bind.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/ref.hpp>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <stdint.h>

namespace ml {
namespace model {

namespace {

using TDouble2Vec = core::CSmallVector<double, 2>;
using TDouble2Vec1Vec = core::CSmallVector<TDouble2Vec, 1>;
using TTime2Vec = core::CSmallVector<core_t::TTime, 2>;

// We use short field names to reduce the state size
const std::string INDIVIDUAL_STATE_TAG("a");
const std::string PROBABILITY_PRIOR_TAG("b");
}

CEventRateModel::CEventRateModel(const SModelParams& params,
                                 const TDataGathererPtr& dataGatherer,
                                 const TFeatureMathsModelSPtrPrVec& newFeatureModels,
                                 const TFeatureMultivariatePriorSPtrPrVec& newFeatureCorrelateModelPriors,
                                 TFeatureCorrelationsPtrPrVec&& featureCorrelatesModels,
                                 const maths::CMultinomialConjugate& probabilityPrior,
                                 const TFeatureInfluenceCalculatorCPtrPrVecVec& influenceCalculators,
                                 const TInterimBucketCorrectorCPtr& interimBucketCorrector)
    : CIndividualModel(params,
                       dataGatherer,
                       newFeatureModels,
                       newFeatureCorrelateModelPriors,
                       std::move(featureCorrelatesModels),
                       influenceCalculators),
      m_CurrentBucketStats(CAnomalyDetectorModel::TIME_UNSET),
      m_ProbabilityPrior(probabilityPrior),
      m_InterimBucketCorrector(interimBucketCorrector) {
}

CEventRateModel::CEventRateModel(const SModelParams& params,
                                 const TDataGathererPtr& dataGatherer,
                                 const TFeatureMathsModelSPtrPrVec& newFeatureModels,
                                 const TFeatureMultivariatePriorSPtrPrVec& newFeatureCorrelateModelPriors,
                                 TFeatureCorrelationsPtrPrVec&& featureCorrelatesModels,
                                 const TFeatureInfluenceCalculatorCPtrPrVecVec& influenceCalculators,
                                 const TInterimBucketCorrectorCPtr& interimBucketCorrector,
                                 core::CStateRestoreTraverser& traverser)
    : CIndividualModel(params,
                       dataGatherer,
                       newFeatureModels,
                       newFeatureCorrelateModelPriors,
                       std::move(featureCorrelatesModels),
                       influenceCalculators),
      m_CurrentBucketStats(CAnomalyDetectorModel::TIME_UNSET),
      m_InterimBucketCorrector(interimBucketCorrector) {
    traverser.traverseSubLevel(
        boost::bind(&CEventRateModel::acceptRestoreTraverser, this, _1));
}

CEventRateModel::CEventRateModel(bool isForPersistence, const CEventRateModel& other)
    : CIndividualModel(isForPersistence, other),
      m_CurrentBucketStats(0), // Not needed for persistence so minimally constructed
      m_ProbabilityPrior(other.m_ProbabilityPrior) {
    if (!isForPersistence) {
        LOG_ABORT(<< "This constructor only creates clones for persistence");
    }
}

void CEventRateModel::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertLevel(INDIVIDUAL_STATE_TAG,
                         boost::bind(&CEventRateModel::doAcceptPersistInserter, this, _1));
    inserter.insertLevel(PROBABILITY_PRIOR_TAG,
                         boost::bind(&maths::CMultinomialConjugate::acceptPersistInserter,
                                     &m_ProbabilityPrior, _1));
}

bool CEventRateModel::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        if (name == INDIVIDUAL_STATE_TAG) {
            if (traverser.traverseSubLevel(boost::bind(
                    &CEventRateModel::doAcceptRestoreTraverser, this, _1)) == false) {
                // Logging handled already.
                return false;
            }
        } else if (name == PROBABILITY_PRIOR_TAG) {
            maths::CMultinomialConjugate prior(
                this->params().distributionRestoreParams(maths_t::E_DiscreteData), traverser);
            m_ProbabilityPrior.swap(prior);
        }
    } while (traverser.next());

    return true;
}

CAnomalyDetectorModel* CEventRateModel::cloneForPersistence() const {
    return new CEventRateModel(true, *this);
}

model_t::EModelType CEventRateModel::category() const {
    return model_t::E_EventRateOnline;
}

bool CEventRateModel::isEventRate() const {
    return true;
}

bool CEventRateModel::isMetric() const {
    return false;
}

void CEventRateModel::currentBucketPersonIds(core_t::TTime time, TSizeVec& result) const {
    this->CIndividualModel::currentBucketPersonIds(
        time, m_CurrentBucketStats.s_FeatureData, result);
}

CEventRateModel::TOptionalDouble CEventRateModel::baselineBucketCount(std::size_t /*pid*/) const {
    return TOptionalDouble();
}

CEventRateModel::TDouble1Vec CEventRateModel::currentBucketValue(model_t::EFeature feature,
                                                                 std::size_t pid,
                                                                 std::size_t /*cid*/,
                                                                 core_t::TTime time) const {
    const TFeatureData* data = this->featureData(feature, pid, time);
    if (data) {
        return TDouble1Vec(1, static_cast<double>(data->s_Count));
    }
    return TDouble1Vec();
}

CEventRateModel::TDouble1Vec
CEventRateModel::baselineBucketMean(model_t::EFeature feature,
                                    std::size_t pid,
                                    std::size_t cid,
                                    model_t::CResultType type,
                                    const TSizeDoublePr1Vec& correlated,
                                    core_t::TTime time) const {
    const maths::CModel* model{this->model(feature, pid)};
    if (!model) {
        return TDouble1Vec();
    }

    static const TSizeDoublePr1Vec NO_CORRELATED;
    TDouble2Vec hint;
    if (model_t::isDiurnal(feature)) {
        hint = this->currentBucketValue(feature, pid, cid, time);
    }
    TDouble1Vec result(model->predict(
        time, type.isUnconditional() ? NO_CORRELATED : correlated, hint));

    double probability = 1.0;
    if (model_t::isConstant(feature) && !m_Probabilities.lookup(pid, probability)) {
        probability = 1.0;
    }
    for (auto& coord : result) {
        coord = probability * model_t::inverseOffsetCountToZero(feature, coord);
    }
    this->correctBaselineForInterim(feature, pid, type, correlated,
                                    this->currentBucketInterimCorrections(), result);

    TDouble1VecDouble1VecPr support{model_t::support(feature)};
    return maths::CTools::truncate(result, support.first, support.second);
}

void CEventRateModel::sampleBucketStatistics(core_t::TTime startTime,
                                             core_t::TTime endTime,
                                             CResourceMonitor& resourceMonitor) {
    this->createUpdateNewModels(startTime, resourceMonitor);
    this->currentBucketInterimCorrections().clear();
    this->CIndividualModel::sampleBucketStatistics(
        startTime, endTime, this->personFilter(),
        m_CurrentBucketStats.s_FeatureData, resourceMonitor);
}

void CEventRateModel::sample(core_t::TTime startTime,
                             core_t::TTime endTime,
                             CResourceMonitor& resourceMonitor) {
    CDataGatherer& gatherer = this->dataGatherer();
    core_t::TTime bucketLength = gatherer.bucketLength();

    if (!gatherer.validateSampleTimes(startTime, endTime)) {
        return;
    }

    this->createUpdateNewModels(startTime, resourceMonitor);
    this->currentBucketInterimCorrections().clear();

    for (core_t::TTime time = startTime; time < endTime; time += bucketLength) {
        LOG_TRACE(<< "Sampling [" << time << "," << time + bucketLength << ")");

        gatherer.sampleNow(time);
        gatherer.featureData(time, bucketLength, m_CurrentBucketStats.s_FeatureData);

        const CIndividualModel::TTimeVec& preSampleLastBucketTimes = this->lastBucketTimes();
        CIndividualModel::TSizeTimeUMap lastBucketTimesMap;
        for (const auto& featureData : m_CurrentBucketStats.s_FeatureData) {
            for (const auto& data : featureData.second) {
                std::size_t pid = data.first;
                lastBucketTimesMap[pid] = preSampleLastBucketTimes[pid];
            }
        }

        this->CIndividualModel::sample(time, time + bucketLength, resourceMonitor);

        // Declared outside the loop to minimize the number of times they are created.
        maths::CModel::TTimeDouble2VecSizeTrVec values;
        maths::CModelAddSamplesParams::TDouble2VecWeightsAryVec weights;

        for (auto& featureData : m_CurrentBucketStats.s_FeatureData) {
            model_t::EFeature feature = featureData.first;
            TSizeFeatureDataPrVec& data = featureData.second;
            std::size_t dimension = model_t::dimension(feature);
            LOG_TRACE(<< model_t::print(feature) << ": "
                      << core::CContainerPrinter::print(data));

            if (feature == model_t::E_IndividualTotalBucketCountByPerson) {
                for (const auto& data_ : data) {
                    if (data_.second.s_Count > 0) {
                        LOG_TRACE(<< "person = " << this->personName(data_.first));
                        m_ProbabilityPrior.addSamples({static_cast<double>(data_.first)},
                                                      maths_t::CUnitWeights::SINGLE_UNIT);
                    }
                }
                if (!data.empty()) {
                    m_ProbabilityPrior.propagateForwardsByTime(1.0);
                }
                continue;
            }
            if (model_t::isCategorical(feature)) {
                continue;
            }

            this->applyFilter(model_t::E_XF_By, true, this->personFilter(), data);

            for (const auto& data_ : data) {
                std::size_t pid = data_.first;

                maths::CModel* model = this->model(feature, pid);
                if (!model) {
                    LOG_ERROR(<< "Missing model for " << this->personName(pid));
                    continue;
                }

                core_t::TTime sampleTime = model_t::sampleTime(feature, time, bucketLength);
                if (this->shouldIgnoreSample(feature, pid, model_t::INDIVIDUAL_ANALYSIS_ATTRIBUTE_ID,
                                             sampleTime)) {
                    model->skipTime(sampleTime - lastBucketTimesMap[pid]);
                    continue;
                }

                double emptyBucketWeight = this->emptyBucketWeight(feature, pid, time);
                if (emptyBucketWeight == 0.0) {
                    continue;
                }

                double count = model_t::offsetCountToZero(
                    feature, static_cast<double>(data_.second.s_Count));
                double derate = this->derate(pid, sampleTime);
                double interval =
                    (1.0 + (this->params().s_InitialDecayRateMultiplier - 1.0) * derate) *
                    emptyBucketWeight;
                double ceff = emptyBucketWeight * this->learnRate(feature);

                LOG_TRACE(<< "Bucket = " << this->printCurrentBucket()
                          << ", feature = " << model_t::print(feature) << ", count = "
                          << count << ", person = " << this->personName(pid)
                          << ", empty bucket weight = " << emptyBucketWeight
                          << ", derate = " << derate << ", interval = " << interval);

                model->params().probabilityBucketEmpty(
                    this->probabilityBucketEmpty(feature, pid));

                TDouble2Vec value{count};
                values.assign(1, core::make_triple(sampleTime, value, model_t::INDIVIDUAL_ANALYSIS_ATTRIBUTE_ID));
                weights.resize(1, maths_t::CUnitWeights::unit<TDouble2Vec>(dimension));
                maths_t::setCount(TDouble2Vec(dimension, ceff), weights[0]);
                maths_t::setWinsorisationWeight(
                    model->winsorisationWeight(derate, sampleTime, value), weights[0]);

                maths::CModelAddSamplesParams params;
                params.integer(true)
                    .nonNegative(true)
                    .propagationInterval(interval)
                    .trendWeights(weights)
                    .priorWeights(weights);

                if (model->addSamples(params, values) == maths::CModel::E_Reset) {
                    gatherer.resetSampleCount(pid);
                }
            }
        }

        this->sampleCorrelateModels();
        m_Probabilities = TCategoryProbabilityCache(m_ProbabilityPrior);
    }
}

bool CEventRateModel::computeProbability(std::size_t pid,
                                         core_t::TTime startTime,
                                         core_t::TTime endTime,
                                         CPartitioningFields& partitioningFields,
                                         std::size_t /*numberAttributeProbabilities*/,
                                         SAnnotatedProbability& result) const {
    const CDataGatherer& gatherer = this->dataGatherer();
    core_t::TTime bucketLength = gatherer.bucketLength();

    if (endTime != startTime + bucketLength) {
        LOG_ERROR(<< "Can only compute probability for single bucket");
        return false;
    }

    if (pid >= this->firstBucketTimes().size()) {
        // This is not necessarily an error: the person might have been added
        // only in an out of phase bucket so far
        LOG_TRACE(<< "No first time for person = " << gatherer.personName(pid));
        return false;
    }

    CAnnotatedProbabilityBuilder resultBuilder(
        result,
        1, // # attribute probabilities
        function_t::function(gatherer.features()), gatherer.numberActivePeople());

    CProbabilityAndInfluenceCalculator pJoint(this->params().s_InfluenceCutoff);
    pJoint.addAggregator(maths::CJointProbabilityOfLessLikelySamples());

    CProbabilityAndInfluenceCalculator pFeatures(this->params().s_InfluenceCutoff);
    pFeatures.addAggregator(maths::CJointProbabilityOfLessLikelySamples());
    pFeatures.addAggregator(maths::CProbabilityOfExtremeSample());

    bool addPersonProbability{false};
    bool skippedResults{false};

    for (std::size_t i = 0u, n = gatherer.numberFeatures(); i < n; ++i) {
        model_t::EFeature feature = gatherer.feature(i);
        if (model_t::isCategorical(feature)) {
            continue;
        }
        const TFeatureData* data = this->featureData(feature, pid, startTime);
        if (!data) {
            continue;
        }
        if (this->shouldIgnoreResult(
                feature, result.s_ResultType, pid, model_t::INDIVIDUAL_ANALYSIS_ATTRIBUTE_ID,
                model_t::sampleTime(feature, startTime, bucketLength))) {
            skippedResults = true;
            continue;
        }

        addPersonProbability = true;

        LOG_TRACE(<< "Compute probability for " << data->print());

        if (this->correlates(feature, pid, startTime)) {
            CProbabilityAndInfluenceCalculator::SCorrelateParams params(partitioningFields);
            TStrCRefDouble1VecDouble1VecPrPrVecVecVec influenceValues;
            this->fill(feature, pid, startTime, result.isInterim(), params, influenceValues);
            this->addProbabilityAndInfluences(pid, params, influenceValues,
                                              pFeatures, resultBuilder);
        } else {
            CProbabilityAndInfluenceCalculator::SParams params(partitioningFields);
            this->fill(feature, pid, startTime, result.isInterim(), params);
            this->addProbabilityAndInfluences(pid, params, data->s_InfluenceValues,
                                              pFeatures, resultBuilder);
        }
    }

    TOptionalUInt64 count = this->currentBucketCount(pid, startTime);

    pJoint.add(pFeatures);
    if (addPersonProbability && count && *count != 0) {
        double p;
        if (m_Probabilities.lookup(pid, p)) {
            LOG_TRACE(<< "P(" << gatherer.personName(pid) << ") = " << p);
            pJoint.addProbability(p);
        }
    }

    double p{1.0};
    if (skippedResults && pJoint.empty()) {
        // This means we have skipped results for all features.
        // We set the probability to 1.0 here to ensure the
        // quantiles are updated accordingly.
    } else if (pJoint.empty()) {
        LOG_TRACE(<< "No samples in [" << startTime << "," << endTime << ")");
        return false;
    } else if (!pJoint.calculate(p, result.s_Influences)) {
        LOG_ERROR(<< "Failed to compute probability");
        return false;
    }
    LOG_TRACE(<< "probability(" << this->personName(pid) << ") = " << p);

    resultBuilder.probability(p);
    bool everSeenBefore = this->firstBucketTimes()[pid] != startTime;
    resultBuilder.personFrequency(this->personFrequency(pid), everSeenBefore);
    resultBuilder.build();

    return true;
}

uint64_t CEventRateModel::checksum(bool includeCurrentBucketStats) const {
    using TStrCRefUInt64Map = std::map<TStrCRef, uint64_t, maths::COrderings::SLess>;

    uint64_t seed = this->CIndividualModel::checksum(includeCurrentBucketStats);

    TStrCRefUInt64Map hashes;
    const TDoubleVec& categories = m_ProbabilityPrior.categories();
    const TDoubleVec& concentrations = m_ProbabilityPrior.concentrations();
    for (std::size_t i = 0u; i < categories.size(); ++i) {
        uint64_t& hash =
            hashes[boost::cref(this->personName(static_cast<std::size_t>(categories[i])))];
        hash = maths::CChecksum::calculate(hash, concentrations[i]);
    }
    if (includeCurrentBucketStats) {
        for (const auto& featureData_ : m_CurrentBucketStats.s_FeatureData) {
            for (const auto& data : featureData_.second) {
                uint64_t& hash = hashes[boost::cref(this->personName(data.first))];
                hash = maths::CChecksum::calculate(hash, data.second.s_Count);
            }
        }
    }

    LOG_TRACE(<< "seed = " << seed);
    LOG_TRACE(<< "hashes = " << core::CContainerPrinter::print(hashes));

    return maths::CChecksum::calculate(seed, hashes);
}

void CEventRateModel::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CEventRateModel");
    this->CIndividualModel::debugMemoryUsage(mem->addChild());
    core::CMemoryDebug::dynamicSize("m_CurrentBucketStats.s_PersonCounts",
                                    m_CurrentBucketStats.s_PersonCounts, mem);
    core::CMemoryDebug::dynamicSize("m_CurrentBucketStats.s_FeatureData",
                                    m_CurrentBucketStats.s_FeatureData, mem);
    core::CMemoryDebug::dynamicSize("m_CurrentBucketStats.s_InterimCorrections",
                                    m_CurrentBucketStats.s_InterimCorrections, mem);
    core::CMemoryDebug::dynamicSize("s_Probabilities", m_Probabilities, mem);
    core::CMemoryDebug::dynamicSize("m_ProbabilityPrior", m_ProbabilityPrior, mem);
    core::CMemoryDebug::dynamicSize("m_InterimBucketCorrector",
                                    m_InterimBucketCorrector, mem);
}

std::size_t CEventRateModel::memoryUsage() const {
    return this->CIndividualModel::memoryUsage();
}

std::size_t CEventRateModel::staticSize() const {
    return sizeof(*this);
}

std::size_t CEventRateModel::computeMemoryUsage() const {
    std::size_t mem = this->CIndividualModel::computeMemoryUsage();
    mem += core::CMemory::dynamicSize(m_CurrentBucketStats.s_PersonCounts);
    mem += core::CMemory::dynamicSize(m_CurrentBucketStats.s_FeatureData);
    mem += core::CMemory::dynamicSize(m_CurrentBucketStats.s_InterimCorrections);
    mem += core::CMemory::dynamicSize(m_Probabilities);
    mem += core::CMemory::dynamicSize(m_ProbabilityPrior);
    mem += core::CMemory::dynamicSize(m_InterimBucketCorrector);
    return mem;
}

CEventRateModel::CModelDetailsViewPtr CEventRateModel::details() const {
    return CModelDetailsViewPtr(new CEventRateModelDetailsView(*this));
}

const CEventRateModel::TFeatureData*
CEventRateModel::featureData(model_t::EFeature feature, std::size_t pid, core_t::TTime time) const {
    return this->CIndividualModel::featureData(feature, pid, time,
                                               m_CurrentBucketStats.s_FeatureData);
}

core_t::TTime CEventRateModel::currentBucketStartTime() const {
    return m_CurrentBucketStats.s_StartTime;
}

void CEventRateModel::currentBucketStartTime(core_t::TTime time) {
    m_CurrentBucketStats.s_StartTime = time;
}

const CEventRateModel::TSizeUInt64PrVec& CEventRateModel::currentBucketPersonCounts() const {
    return m_CurrentBucketStats.s_PersonCounts;
}

CEventRateModel::TSizeUInt64PrVec& CEventRateModel::currentBucketPersonCounts() {
    return m_CurrentBucketStats.s_PersonCounts;
}

CIndividualModel::TFeatureSizeSizeTripleDouble1VecUMap&
CEventRateModel::currentBucketInterimCorrections() const {
    return m_CurrentBucketStats.s_InterimCorrections;
}

void CEventRateModel::clearPrunedResources(const TSizeVec& people, const TSizeVec& attributes) {
    CDataGatherer& gatherer = this->dataGatherer();

    // Stop collecting for these people and add them to the free list.
    gatherer.recyclePeople(people);
    if (gatherer.dataAvailable(m_CurrentBucketStats.s_StartTime)) {
        gatherer.featureData(m_CurrentBucketStats.s_StartTime, gatherer.bucketLength(),
                             m_CurrentBucketStats.s_FeatureData);
    }

    TDoubleVec categoriesToRemove;
    categoriesToRemove.reserve(people.size());
    for (std::size_t i = 0u; i < people.size(); ++i) {
        categoriesToRemove.push_back(static_cast<double>(people[i]));
    }
    m_ProbabilityPrior.removeCategories(categoriesToRemove);
    m_Probabilities = TCategoryProbabilityCache(m_ProbabilityPrior);

    this->CIndividualModel::clearPrunedResources(people, attributes);
}

const CInterimBucketCorrector& CEventRateModel::interimValueCorrector() const {
    return *m_InterimBucketCorrector;
}

bool CEventRateModel::correlates(model_t::EFeature feature, std::size_t pid, core_t::TTime time) const {
    if (model_t::dimension(feature) > 1 || !this->params().s_MultivariateByFields) {
        return false;
    }

    const maths::CModel* model{this->model(feature, pid)};
    for (const auto& correlate : model->correlates()) {
        if (this->featureData(
                feature, pid == correlate[0] ? correlate[1] : correlate[0], time)) {
            return true;
        }
    }
    return false;
}

void CEventRateModel::fill(model_t::EFeature feature,
                           std::size_t pid,
                           core_t::TTime bucketTime,
                           bool interim,
                           CProbabilityAndInfluenceCalculator::SParams& params) const {
    const TFeatureData* data{this->featureData(feature, pid, bucketTime)};
    const maths::CModel* model{this->model(feature, pid)};
    core_t::TTime time{model_t::sampleTime(feature, bucketTime, this->bucketLength())};
    TOptionalUInt64 count{this->currentBucketCount(pid, bucketTime)};
    double value{model_t::offsetCountToZero(feature, static_cast<double>(data->s_Count))};
    maths_t::TDouble2VecWeightsAry weight(maths_t::seasonalVarianceScaleWeight(
        model->seasonalWeight(maths::DEFAULT_SEASONAL_CONFIDENCE_INTERVAL, time)));
    bool skipAnomalyModelUpdate = this->shouldIgnoreSample(
        feature, pid, model_t::INDIVIDUAL_ANALYSIS_ATTRIBUTE_ID, time);

    params.s_Feature = feature;
    params.s_Model = model;
    params.s_ElapsedTime = bucketTime - this->firstBucketTimes()[pid];
    params.s_Time.assign(1, TTime2Vec{time});
    params.s_Value.assign(1, TDouble2Vec{value});
    if (interim && model_t::requiresInterimResultAdjustment(feature)) {
        double mode{params.s_Model->mode(time, weight)[0]};
        TDouble2Vec correction{this->interimValueCorrector().corrections(mode, value)};
        params.s_Value[0] += correction;
        this->currentBucketInterimCorrections().emplace(
            core::make_triple(feature, pid, pid), correction);
    }
    params.s_Count = 1.0;
    params.s_ComputeProbabilityParams
        .addCalculation(model_t::probabilityCalculation(feature)) // new line
        .addBucketEmpty({!count || *count == 0})
        .addWeights(weight)
        .skipAnomalyModelUpdate(skipAnomalyModelUpdate);
}

void CEventRateModel::fill(model_t::EFeature feature,
                           std::size_t pid,
                           core_t::TTime bucketTime,
                           bool interim,
                           CProbabilityAndInfluenceCalculator::SCorrelateParams& params,
                           TStrCRefDouble1VecDouble1VecPrPrVecVecVec& influenceValues) const {
    using TStrCRefDouble1VecDoublePrPr = std::pair<TStrCRef, TDouble1VecDoublePr>;

    const CDataGatherer& gatherer{this->dataGatherer()};
    const maths::CModel* model{this->model(feature, pid)};
    const TSize2Vec1Vec& correlates{model->correlates()};
    const TTimeVec& firstBucketTimes{this->firstBucketTimes()};
    core_t::TTime time{model_t::sampleTime(feature, bucketTime, gatherer.bucketLength())};
    bool skipAnomalyModelUpdate = this->shouldIgnoreSample(
        feature, pid, model_t::INDIVIDUAL_ANALYSIS_ATTRIBUTE_ID, time);

    params.s_Feature = feature;
    params.s_Model = model;
    params.s_ElapsedTime = boost::numeric::bounds<core_t::TTime>::highest();
    params.s_Times.resize(correlates.size());
    params.s_Values.resize(correlates.size());
    params.s_Counts.resize(correlates.size());
    params.s_Variables.resize(correlates.size());
    params.s_CorrelatedLabels.resize(correlates.size());
    params.s_Correlated.resize(correlates.size());
    params.s_ComputeProbabilityParams
        .addCalculation(model_t::probabilityCalculation(feature))
        .skipAnomalyModelUpdate(skipAnomalyModelUpdate);

    // These are indexed as follows:
    //   influenceValues["influencer name"]["correlate"]["influence value"]
    // This is because we aren't guaranteed that each influence is present for
    // each feature.
    influenceValues.resize(
        this->featureData(feature, pid, bucketTime)->s_InfluenceValues.size(),
        TStrCRefDouble1VecDouble1VecPrPrVecVec(correlates.size()));

    // Declared outside the loop to minimize the number of times it is created.
    TDouble1VecDouble1VecPr value;

    for (std::size_t i = 0u; i < correlates.size(); ++i) {
        TSize2Vec variables = pid == correlates[i][0] ? TSize2Vec{0, 1}
                                                      : TSize2Vec{1, 0};
        params.s_CorrelatedLabels[i] =
            gatherer.personNamePtr(correlates[i][variables[1]]);
        params.s_Correlated[i] = correlates[i][variables[1]];
        params.s_Variables[i] = variables;
        const maths::CModel* models[]{
            model, this->model(feature, correlates[i][variables[1]])};
        maths_t::TDouble2Vec scale(2);
        scale[variables[0]] = models[0]->seasonalWeight(
            maths::DEFAULT_SEASONAL_CONFIDENCE_INTERVAL, time)[0];
        scale[variables[1]] = models[1]->seasonalWeight(
            maths::DEFAULT_SEASONAL_CONFIDENCE_INTERVAL, time)[0];
        TOptionalUInt64 count[2];
        count[0] = this->currentBucketCount(correlates[i][0], bucketTime);
        count[1] = this->currentBucketCount(correlates[i][1], bucketTime);
        params.s_ComputeProbabilityParams
            .addBucketEmpty({!count[0] || *count[0] == 0, !count[1] || *count[1] == 0}) // new line
            .addWeights(maths_t::seasonalVarianceScaleWeight(scale));

        const TFeatureData* data[2];
        data[0] = this->featureData(feature, correlates[i][0], bucketTime);
        data[1] = this->featureData(feature, correlates[i][1], bucketTime);
        if (data[0] && data[1]) {
            params.s_ElapsedTime = std::min(
                params.s_ElapsedTime, bucketTime - firstBucketTimes[correlates[i][0]]);
            params.s_ElapsedTime = std::min(
                params.s_ElapsedTime, bucketTime - firstBucketTimes[correlates[i][1]]);
            params.s_Times[i] = TTime2Vec(2, time);
            params.s_Values[i] = TDouble2Vec{
                model_t::offsetCountToZero(feature, static_cast<double>(data[0]->s_Count)),
                model_t::offsetCountToZero(feature, static_cast<double>(data[1]->s_Count))};
            for (std::size_t j = 0u; j < data[0]->s_InfluenceValues.size(); ++j) {
                for (const auto& influenceValue : data[0]->s_InfluenceValues[j]) {
                    TStrCRef influence = influenceValue.first;
                    std::size_t match = static_cast<std::size_t>(
                        std::find_if(data[1]->s_InfluenceValues[j].begin(),
                                     data[1]->s_InfluenceValues[j].end(),
                                     [influence](const TStrCRefDouble1VecDoublePrPr& value_) {
                                         return value_.first.get() == influence.get();
                                     }) -
                        data[1]->s_InfluenceValues[j].begin());
                    if (match < data[1]->s_InfluenceValues[j].size()) {
                        const TDouble1VecDoublePr& value0 = influenceValue.second;
                        const TDouble1VecDoublePr& value1 =
                            data[1]->s_InfluenceValues[j][match].second;
                        value.first = TDouble1Vec{value0.first[0], value1.first[0]};
                        value.second = TDouble1Vec{value0.second, value1.second};
                        influenceValues[j][i].emplace_back(influence, value);
                    }
                }
            }
        }
    }
    if (interim && model_t::requiresInterimResultAdjustment(feature)) {
        TDouble2Vec1Vec modes = params.s_Model->correlateModes(
            time, params.s_ComputeProbabilityParams.weights());
        for (std::size_t i = 0u; i < modes.size(); ++i) {
            TDouble2Vec& value_ = params.s_Values[i];
            if (!value_.empty()) {
                TDouble2Vec correction(
                    this->interimValueCorrector().corrections(modes[i], value_));
                value_ += correction;
                this->currentBucketInterimCorrections().emplace(
                    core::make_triple(feature, pid, params.s_Correlated[i]),
                    TDouble1Vec{correction[params.s_Variables[i][0]]});
            }
        }
    }
}

////////// CEventRateModel::SBucketStats Implementation //////////

CEventRateModel::SBucketStats::SBucketStats(core_t::TTime startTime)
    : s_StartTime(startTime), s_InterimCorrections(1) {
}
}
}
