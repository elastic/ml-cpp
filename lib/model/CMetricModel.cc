/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CMetricModel.h>

#include <core/CContainerPrinter.h>
#include <core/CFunctional.h>
#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CoreTypes.h>

#include <maths/CChecksum.h>
#include <maths/CMultivariatePrior.h>
#include <maths/COrderings.h>
#include <maths/CPrior.h>
#include <maths/CTools.h>
#include <maths/ProbabilityAggregators.h>

#include <model/CAnnotatedProbabilityBuilder.h>
#include <model/CAnnotation.h>
#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CDataGatherer.h>
#include <model/CGathererTools.h>
#include <model/CIndividualModelDetail.h>
#include <model/CInterimBucketCorrector.h>
#include <model/CModelDetailsView.h>
#include <model/CModelTools.h>
#include <model/CProbabilityAndInfluenceCalculator.h>
#include <model/CResourceMonitor.h>
#include <model/CSampleGatherer.h>
#include <model/CSearchKey.h>
#include <model/FrequencyPredicates.h>

#include <boost/iterator/counting_iterator.hpp>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace ml {
namespace model {

namespace {

using TTime1Vec = core::CSmallVector<core_t::TTime, 1>;
using TDouble1Vec = core::CSmallVector<double, 1>;
using TDouble2Vec = core::CSmallVector<double, 2>;
using TDouble2Vec1Vec = core::CSmallVector<TDouble2Vec, 1>;

// We use short field names to reduce the state size
const std::string INDIVIDUAL_STATE_TAG("a");
}

CMetricModel::CMetricModel(const SModelParams& params,
                           const TDataGathererPtr& dataGatherer,
                           const TFeatureMathsModelSPtrPrVec& newFeatureModels,
                           const TFeatureMultivariatePriorSPtrPrVec& newFeatureCorrelateModelPriors,
                           TFeatureCorrelationsPtrPrVec&& featureCorrelatesModels,
                           const TFeatureInfluenceCalculatorCPtrPrVecVec& influenceCalculators,
                           const TInterimBucketCorrectorCPtr& interimBucketCorrector)
    : CIndividualModel(params,
                       dataGatherer,
                       newFeatureModels,
                       newFeatureCorrelateModelPriors,
                       std::move(featureCorrelatesModels),
                       influenceCalculators),
      m_CurrentBucketStats(CAnomalyDetectorModel::TIME_UNSET),
      m_InterimBucketCorrector(interimBucketCorrector) {
}

CMetricModel::CMetricModel(const SModelParams& params,
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
    traverser.traverseSubLevel(std::bind(&CMetricModel::acceptRestoreTraverser,
                                         this, std::placeholders::_1));
}

CMetricModel::CMetricModel(bool isForPersistence, const CMetricModel& other)
    : CIndividualModel(isForPersistence, other),
      m_CurrentBucketStats(0) // Not needed for persistence so minimally constructed
{
    if (!isForPersistence) {
        LOG_ABORT(<< "This constructor only creates clones for persistence");
    }
}

void CMetricModel::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertLevel(INDIVIDUAL_STATE_TAG, std::bind(&CMetricModel::doAcceptPersistInserter,
                                                         this, std::placeholders::_1));
}

bool CMetricModel::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE(INDIVIDUAL_STATE_TAG,
                traverser.traverseSubLevel(std::bind(&CMetricModel::doAcceptRestoreTraverser,
                                                     this, std::placeholders::_1)))
    } while (traverser.next());

    return true;
}

CAnomalyDetectorModel* CMetricModel::cloneForPersistence() const {
    return new CMetricModel(true, *this);
}

model_t::EModelType CMetricModel::category() const {
    return model_t::E_MetricOnline;
}

bool CMetricModel::isEventRate() const {
    return false;
}

bool CMetricModel::isMetric() const {
    return true;
}

void CMetricModel::currentBucketPersonIds(core_t::TTime time, TSizeVec& result) const {
    this->CIndividualModel::currentBucketPersonIds(
        time, m_CurrentBucketStats.s_FeatureData, result);
}

CMetricModel::TOptionalDouble CMetricModel::baselineBucketCount(const std::size_t /*pid*/) const {
    return TOptionalDouble();
}

CMetricModel::TDouble1Vec CMetricModel::currentBucketValue(model_t::EFeature feature,
                                                           std::size_t pid,
                                                           std::size_t /*cid*/,
                                                           core_t::TTime time) const {
    const TFeatureData* data = this->featureData(feature, pid, time);
    if (data) {
        const TOptionalSample& value = data->s_BucketValue;
        return value ? value->value(model_t::dimension(feature)) : TDouble1Vec();
    }
    return TDouble1Vec();
}

CMetricModel::TDouble1Vec CMetricModel::baselineBucketMean(model_t::EFeature feature,
                                                           std::size_t pid,
                                                           std::size_t /*cid*/,
                                                           model_t::CResultType type,
                                                           const TSizeDoublePr1Vec& correlated,
                                                           core_t::TTime time) const {
    const maths::CModel* model{this->model(feature, pid)};
    if (!model) {
        return TDouble1Vec();
    }
    static const TSizeDoublePr1Vec NO_CORRELATED;
    TDouble1Vec result(model->predict(time, type.isUnconditional() ? NO_CORRELATED : correlated));
    this->correctBaselineForInterim(feature, pid, type, correlated,
                                    this->currentBucketInterimCorrections(), result);
    TDouble1VecDouble1VecPr support = model_t::support(feature);
    return maths::CTools::truncate(result, support.first, support.second);
}

void CMetricModel::sampleBucketStatistics(core_t::TTime startTime,
                                          core_t::TTime endTime,
                                          CResourceMonitor& resourceMonitor) {
    this->createUpdateNewModels(startTime, resourceMonitor);
    m_CurrentBucketStats.s_InterimCorrections.clear();
    this->CIndividualModel::sampleBucketStatistics(
        startTime, endTime, this->personFilter(),
        m_CurrentBucketStats.s_FeatureData, resourceMonitor);
}

void CMetricModel::sample(core_t::TTime startTime,
                          core_t::TTime endTime,
                          CResourceMonitor& resourceMonitor) {
    CDataGatherer& gatherer = this->dataGatherer();
    core_t::TTime bucketLength = gatherer.bucketLength();

    if (!gatherer.validateSampleTimes(startTime, endTime)) {
        return;
    }

    this->createUpdateNewModels(startTime, resourceMonitor);
    m_CurrentBucketStats.s_InterimCorrections.clear();
    m_CurrentBucketStats.s_Annotations.clear();

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
        maths::CModelAddSamplesParams::TDouble2VecWeightsAryVec trendWeights;
        maths::CModelAddSamplesParams::TDouble2VecWeightsAryVec priorWeights;

        for (auto& featureData : m_CurrentBucketStats.s_FeatureData) {
            model_t::EFeature feature = featureData.first;
            TSizeFeatureDataPrVec& data = featureData.second;
            std::size_t dimension = model_t::dimension(feature);
            LOG_TRACE(<< model_t::print(feature)
                      << " data = " << core::CContainerPrinter::print(data));
            this->applyFilter(model_t::E_XF_By, true, this->personFilter(), data);

            for (const auto& data_ : data) {
                std::size_t pid = data_.first;
                const CGathererTools::TSampleVec& samples = data_.second.s_Samples;

                maths::CModel* model = this->model(feature, pid);
                if (model == nullptr) {
                    LOG_ERROR(<< "Missing model for " << this->personName(pid));
                    continue;
                }

                core_t::TTime sampleTime = model_t::sampleTime(feature, time, bucketLength);
                if (this->shouldIgnoreSample(feature, pid, model_t::INDIVIDUAL_ANALYSIS_ATTRIBUTE_ID,
                                             sampleTime)) {
                    model->skipTime(time - lastBucketTimesMap[pid]);
                    continue;
                }

                const TOptionalSample& bucket = data_.second.s_BucketValue;
                if (model_t::isSampled(feature) && bucket != boost::none) {
                    values.assign(1, core::make_triple(
                                         bucket->time(), TDouble2Vec(bucket->value(dimension)),
                                         model_t::INDIVIDUAL_ANALYSIS_ATTRIBUTE_ID));
                    model->addBucketValue(values);
                }

                // For sparse data we reduce the impact of samples from empty buckets.
                // In effect, we smoothly transition to modeling only values from non-empty
                // buckets as the data becomes sparse.
                double emptyBucketWeight = this->emptyBucketWeight(feature, pid, time);
                if (emptyBucketWeight == 0.0) {
                    continue;
                }

                std::size_t n = samples.size();
                double countWeight =
                    (this->params().s_MaximumUpdatesPerBucket > 0.0 && n > 0
                         ? this->params().s_MaximumUpdatesPerBucket / static_cast<double>(n)
                         : 1.0) *
                    this->learnRate(feature);
                double winsorisationDerate = this->derate(pid, sampleTime);
                // Note we need to scale the amount of data we'll "age out" of the residual
                // model in one bucket by the empty bucket weight so the posterior doesn't
                // end up too flat.
                double scaledInterval = emptyBucketWeight;
                double scaledCountWeight = emptyBucketWeight * countWeight;

                LOG_TRACE(<< "Bucket = " << gatherer.printCurrentBucket()
                          << ", feature = " << model_t::print(feature)
                          << ", samples = " << core::CContainerPrinter::print(samples)
                          << ", isInteger = " << data_.second.s_IsInteger
                          << ", person = " << this->personName(pid)
                          << ", dimension = " << dimension << ", count weight = " << countWeight
                          << ", scaled count weight = " << scaledCountWeight
                          << ", scaled interval = " << scaledInterval);

                values.resize(n);
                trendWeights.resize(n, maths_t::CUnitWeights::unit<TDouble2Vec>(dimension));
                priorWeights.resize(n, maths_t::CUnitWeights::unit<TDouble2Vec>(dimension));
                for (std::size_t i = 0; i < n; ++i) {
                    core_t::TTime ithSampleTime = samples[i].time();
                    TDouble2Vec ithSampleValue(samples[i].value(dimension));
                    double countVarianceScale = samples[i].varianceScale();
                    values[i] = core::make_triple(
                        model_t::sampleTime(feature, time, bucketLength, ithSampleTime),
                        ithSampleValue, model_t::INDIVIDUAL_ANALYSIS_ATTRIBUTE_ID);
                    model->countWeights(ithSampleTime, ithSampleValue,
                                        countWeight, scaledCountWeight,
                                        winsorisationDerate, countVarianceScale,
                                        trendWeights[i], priorWeights[i]);
                }

                auto annotationCallback = [&](const std::string& annotation) {
                    if (this->params().s_AnnotationsEnabled) {
                        m_CurrentBucketStats.s_Annotations.emplace_back(
                            time, CAnnotation::E_ModelChange, annotation,
                            gatherer.searchKey().detectorIndex(),
                            gatherer.searchKey().partitionFieldName(),
                            gatherer.partitionFieldValue(),
                            gatherer.searchKey().overFieldName(), EMPTY_STRING,
                            gatherer.searchKey().byFieldName(), gatherer.personName(pid));
                    }
                };

                maths::CModelAddSamplesParams params;
                params.integer(data_.second.s_IsInteger)
                    .nonNegative(data_.second.s_IsNonNegative)
                    .propagationInterval(scaledInterval)
                    .trendWeights(trendWeights)
                    .priorWeights(priorWeights)
                    .annotationCallback([&](const std::string& annotation) {
                        annotationCallback(annotation);
                    });

                if (model->addSamples(params, values) == maths::CModel::E_Reset) {
                    gatherer.resetSampleCount(pid);
                }
            }
        }

        this->sampleCorrelateModels();
    }
}

bool CMetricModel::computeProbability(const std::size_t pid,
                                      core_t::TTime startTime,
                                      core_t::TTime endTime,
                                      CPartitioningFields& partitioningFields,
                                      const std::size_t /*numberAttributeProbabilities*/,
                                      SAnnotatedProbability& result) const {
    CAnnotatedProbabilityBuilder resultBuilder(result);

    const CDataGatherer& gatherer = this->dataGatherer();
    core_t::TTime bucketLength = gatherer.bucketLength();

    if (endTime != startTime + bucketLength) {
        LOG_ERROR(<< "Can only compute probability for single bucket");
        return false;
    }

    if (pid >= this->firstBucketTimes().size()) {
        LOG_ERROR(<< "No first time for person = " << gatherer.personName(pid));
        return false;
    }

    CProbabilityAndInfluenceCalculator pJoint(this->params().s_InfluenceCutoff);
    pJoint.addAggregator(maths::CJointProbabilityOfLessLikelySamples());
    pJoint.addAggregator(maths::CProbabilityOfExtremeSample());

    bool skippedResults{false};
    for (std::size_t i = 0u, n = gatherer.numberFeatures(); i < n; ++i) {
        model_t::EFeature feature = gatherer.feature(i);
        if (model_t::isCategorical(feature)) {
            continue;
        }
        const TFeatureData* data = this->featureData(feature, pid, startTime);
        if (!data || !data->s_BucketValue) {
            continue;
        }
        const TOptionalSample& bucket = data->s_BucketValue;
        if (this->shouldIgnoreResult(
                feature, result.s_ResultType, pid, model_t::INDIVIDUAL_ANALYSIS_ATTRIBUTE_ID,
                model_t::sampleTime(feature, startTime, bucketLength, bucket->time()))) {
            skippedResults = true;
            continue;
        }

        LOG_TRACE(<< "Compute probability for " << data->print());

        if (this->correlates(feature, pid, startTime)) {
            CProbabilityAndInfluenceCalculator::SCorrelateParams params(partitioningFields);
            TStrCRefDouble1VecDouble1VecPrPrVecVecVec influenceValues;
            this->fill(feature, pid, startTime, result.isInterim(), params, influenceValues);
            this->addProbabilityAndInfluences(pid, params, influenceValues,
                                              pJoint, resultBuilder);
        } else {
            CProbabilityAndInfluenceCalculator::SParams params(partitioningFields);
            if (this->fill(feature, pid, startTime, result.isInterim(), params)) {
                this->addProbabilityAndInfluences(pid, params, data->s_InfluenceValues,
                                                  pJoint, resultBuilder);
            }
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

    double multiBucketImpact{-1.0 * CAnomalyDetectorModelConfig::MAXIMUM_MULTI_BUCKET_IMPACT_MAGNITUDE};
    if (pJoint.calculateMultiBucketImpact(multiBucketImpact)) {
        resultBuilder.multiBucketImpact(multiBucketImpact);
    }

    resultBuilder.build();

    return true;
}

uint64_t CMetricModel::checksum(bool includeCurrentBucketStats) const {
    using TStrCRefUInt64Map = std::map<TStrCRef, uint64_t, maths::COrderings::SLess>;

    uint64_t seed = this->CIndividualModel::checksum(includeCurrentBucketStats);

#define KEY(pid) std::cref(this->personName(pid))

    TStrCRefUInt64Map hashes;
    if (includeCurrentBucketStats) {
        const TFeatureSizeFeatureDataPrVecPrVec& featureData =
            m_CurrentBucketStats.s_FeatureData;
        for (std::size_t i = 0; i < featureData.size(); ++i) {
            for (std::size_t j = 0; j < featureData[i].second.size(); ++j) {
                uint64_t& hash = hashes[KEY(featureData[i].second[j].first)];
                const TFeatureData& data = featureData[i].second[j].second;
                hash = maths::CChecksum::calculate(hash, data.s_BucketValue);
                hash = core::CHashing::hashCombine(
                    hash, static_cast<uint64_t>(data.s_IsInteger));
                hash = maths::CChecksum::calculate(hash, data.s_Samples);
            }
        }
    }

#undef KEY

    LOG_TRACE(<< "seed = " << seed);
    LOG_TRACE(<< "hashes = " << core::CContainerPrinter::print(hashes));

    return maths::CChecksum::calculate(seed, hashes);
}

void CMetricModel::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CMetricModel");
    this->CIndividualModel::debugMemoryUsage(mem->addChild());
    core::CMemoryDebug::dynamicSize("m_CurrentBucketStats.s_PersonCounts",
                                    m_CurrentBucketStats.s_PersonCounts, mem);
    core::CMemoryDebug::dynamicSize("m_CurrentBucketStats.s_FeatureData",
                                    m_CurrentBucketStats.s_FeatureData, mem);
    core::CMemoryDebug::dynamicSize("m_CurrentBucketStats.s_InterimCorrections",
                                    m_CurrentBucketStats.s_InterimCorrections, mem);
    core::CMemoryDebug::dynamicSize("m_CurrentBucketStats.s_Annotations",
                                    m_CurrentBucketStats.s_Annotations, mem);
    core::CMemoryDebug::dynamicSize("m_InterimBucketCorrector",
                                    m_InterimBucketCorrector, mem);
}

std::size_t CMetricModel::memoryUsage() const {
    return this->CIndividualModel::memoryUsage();
}

std::size_t CMetricModel::computeMemoryUsage() const {
    std::size_t mem = this->CIndividualModel::computeMemoryUsage();
    mem += core::CMemory::dynamicSize(m_CurrentBucketStats.s_PersonCounts);
    mem += core::CMemory::dynamicSize(m_CurrentBucketStats.s_FeatureData);
    mem += core::CMemory::dynamicSize(m_CurrentBucketStats.s_InterimCorrections);
    mem += core::CMemory::dynamicSize(m_CurrentBucketStats.s_Annotations);
    mem += core::CMemory::dynamicSize(m_InterimBucketCorrector);
    return mem;
}

std::size_t CMetricModel::staticSize() const {
    return sizeof(*this);
}

CMetricModel::TModelDetailsViewUPtr CMetricModel::details() const {
    return TModelDetailsViewUPtr(new CMetricModelDetailsView(*this));
}

const CMetricModel::TFeatureData*
CMetricModel::featureData(model_t::EFeature feature, std::size_t pid, core_t::TTime time) const {
    return this->CIndividualModel::featureData(feature, pid, time,
                                               m_CurrentBucketStats.s_FeatureData);
}

const CMetricModel::TAnnotationVec& CMetricModel::annotations() const {
    return m_CurrentBucketStats.s_Annotations;
}

core_t::TTime CMetricModel::currentBucketStartTime() const {
    return m_CurrentBucketStats.s_StartTime;
}

void CMetricModel::currentBucketStartTime(core_t::TTime time) {
    m_CurrentBucketStats.s_StartTime = time;
}

CIndividualModel::TFeatureSizeSizeTripleDouble1VecUMap&
CMetricModel::currentBucketInterimCorrections() const {
    return m_CurrentBucketStats.s_InterimCorrections;
}

const CMetricModel::TSizeUInt64PrVec& CMetricModel::currentBucketPersonCounts() const {
    return m_CurrentBucketStats.s_PersonCounts;
}

CMetricModel::TSizeUInt64PrVec& CMetricModel::currentBucketPersonCounts() {
    return m_CurrentBucketStats.s_PersonCounts;
}

void CMetricModel::clearPrunedResources(const TSizeVec& people, const TSizeVec& attributes) {
    CDataGatherer& gatherer = this->dataGatherer();

    // Stop collecting for these people and add them to the free list.
    gatherer.recyclePeople(people);
    if (gatherer.dataAvailable(m_CurrentBucketStats.s_StartTime)) {
        gatherer.featureData(m_CurrentBucketStats.s_StartTime, gatherer.bucketLength(),
                             m_CurrentBucketStats.s_FeatureData);
    }

    this->CIndividualModel::clearPrunedResources(people, attributes);
}

const CInterimBucketCorrector& CMetricModel::interimValueCorrector() const {
    return *m_InterimBucketCorrector;
}

bool CMetricModel::correlates(model_t::EFeature feature, std::size_t pid, core_t::TTime time) const {
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

bool CMetricModel::fill(model_t::EFeature feature,
                        std::size_t pid,
                        core_t::TTime bucketTime,
                        bool interim,
                        CProbabilityAndInfluenceCalculator::SParams& params) const {

    std::size_t dimension{model_t::dimension(feature)};
    const TFeatureData* data{this->featureData(feature, pid, bucketTime)};
    if (data == nullptr) {
        LOG_TRACE(<< "data unexpectedly null");
        return false;
    }
    const TOptionalSample& bucket{data->s_BucketValue};
    const maths::CModel* model{this->model(feature, pid)};
    if (model == nullptr) {
        LOG_TRACE(<< "model unexpectedly null");
        return false;
    }
    core_t::TTime time{model_t::sampleTime(feature, bucketTime,
                                           this->bucketLength(), bucket->time())};
    maths_t::TDouble2VecWeightsAry weights{maths_t::CUnitWeights::unit<TDouble2Vec>(dimension)};
    TDouble2Vec seasonalWeight;
    model->seasonalWeight(maths::DEFAULT_SEASONAL_CONFIDENCE_INTERVAL, time, seasonalWeight);
    maths_t::setSeasonalVarianceScale(seasonalWeight, weights);
    maths_t::setCountVarianceScale(TDouble2Vec(dimension, bucket->varianceScale()), weights);
    bool skipAnomalyModelUpdate = this->shouldIgnoreSample(
        feature, pid, model_t::INDIVIDUAL_ANALYSIS_ATTRIBUTE_ID, time);

    params.s_Feature = feature;
    params.s_Model = model;
    params.s_ElapsedTime = bucketTime - this->firstBucketTimes()[pid];
    params.s_Time.assign(1, TTime2Vec{time});
    params.s_Value.assign(1, bucket->value());
    if (interim && model_t::requiresInterimResultAdjustment(feature)) {
        TDouble2Vec mode(params.s_Model->mode(time, weights));
        TDouble2Vec correction(this->interimValueCorrector().corrections(
            mode, bucket->value(dimension)));
        params.s_Value[0] += correction;
        this->currentBucketInterimCorrections().emplace(
            core::make_triple(feature, pid, pid), correction);
    }
    params.s_Count = bucket->count();
    params.s_ComputeProbabilityParams
        .addCalculation(model_t::probabilityCalculation(feature))
        .addWeights(weights)
        .skipAnomalyModelUpdate(skipAnomalyModelUpdate);

    return true;
}

void CMetricModel::fill(model_t::EFeature feature,
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
    core_t::TTime bucketLength{gatherer.bucketLength()};
    bool skipAnomalyModelUpdate = this->shouldIgnoreSample(
        feature, pid, model_t::INDIVIDUAL_ANALYSIS_ATTRIBUTE_ID,
        model_t::sampleTime(feature, bucketTime, bucketLength));

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
    TDouble2Vec seasonalWeights[2];
    TDouble2Vec weight(2);

    for (std::size_t i = 0; i < correlates.size(); ++i) {
        TSize2Vec variables(pid == correlates[i][0] ? TSize2Vec{0, 1} : TSize2Vec{1, 0});
        params.s_CorrelatedLabels[i] =
            gatherer.personNamePtr(correlates[i][variables[1]]);
        params.s_Correlated[i] = correlates[i][variables[1]];
        params.s_Variables[i] = variables;
        const maths::CModel* models[]{
            model, this->model(feature, correlates[i][variables[1]])};
        maths_t::TDouble2VecWeightsAry weights(maths_t::CUnitWeights::unit<TDouble2Vec>(2));
        models[0]->seasonalWeight(maths::DEFAULT_SEASONAL_CONFIDENCE_INTERVAL,
                                  bucketTime, seasonalWeights[0]);
        models[1]->seasonalWeight(maths::DEFAULT_SEASONAL_CONFIDENCE_INTERVAL,
                                  bucketTime, seasonalWeights[1]);
        weight[variables[0]] = seasonalWeights[0][0];
        weight[variables[1]] = seasonalWeights[1][0];
        maths_t::setSeasonalVarianceScale(weight, weights);

        const TFeatureData* data[2];
        data[0] = this->featureData(feature, correlates[i][0], bucketTime);
        data[1] = this->featureData(feature, correlates[i][1], bucketTime);
        if (data[0] && data[1] && data[0]->s_BucketValue && data[1]->s_BucketValue) {
            const TOptionalSample& bucket0{data[0]->s_BucketValue};
            const TOptionalSample& bucket1{data[1]->s_BucketValue};
            core_t::TTime times[] = {
                model_t::sampleTime(feature, bucketTime, bucketLength, bucket0->time()),
                model_t::sampleTime(feature, bucketTime, bucketLength, bucket1->time())};
            params.s_ElapsedTime = std::min(
                params.s_ElapsedTime, times[0] - firstBucketTimes[correlates[i][0]]);
            params.s_ElapsedTime = std::min(
                params.s_ElapsedTime, times[1] - firstBucketTimes[correlates[i][1]]);
            params.s_Times[i] = TTime2Vec{times[0], times[1]};
            params.s_Values[i].resize(2 * bucket0->value().size());
            for (std::size_t j = 0; j < bucket0->value().size(); ++j) {
                params.s_Values[i][2 * j + 0] = bucket0->value()[j];
                params.s_Values[i][2 * j + 1] = bucket1->value()[j];
            }
            params.s_Counts[i] = TDouble2Vec{bucket0->count(), bucket1->count()};
            weight[variables[0]] = bucket0->varianceScale();
            weight[variables[1]] = bucket1->varianceScale();
            maths_t::setCountVarianceScale(weight, weights);
            for (std::size_t j = 0; j < data[0]->s_InfluenceValues.size(); ++j) {
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
                        value.first.resize(2 * value0.first.size());
                        for (std::size_t k = 0; k < value0.first.size(); ++k) {
                            value.first[2 * k + 0] = value0.first[k];
                            value.first[2 * k + 1] = value1.first[k];
                        }
                        value.second = TDouble1Vec{value0.second, value1.second};
                        influenceValues[j][i].emplace_back(influence, value);
                    }
                }
            }
        }
        params.s_ComputeProbabilityParams.addWeights(weights);
    }
    if (interim && model_t::requiresInterimResultAdjustment(feature)) {
        core_t::TTime time{bucketTime + bucketLength / 2};
        TDouble2Vec1Vec modes(params.s_Model->correlateModes(
            time, params.s_ComputeProbabilityParams.weights()));
        for (std::size_t i = 0; i < modes.size(); ++i) {
            if (!params.s_Values.empty()) {
                TDouble2Vec value_{params.s_Values[i][0], params.s_Values[i][1]};
                TDouble2Vec correction(
                    this->interimValueCorrector().corrections(modes[i], value_));
                for (std::size_t j = 0; j < 2; ++j) {
                    params.s_Values[i][j] += correction[j];
                }
                this->currentBucketInterimCorrections().emplace(
                    core::make_triple(feature, pid, params.s_Correlated[i]),
                    TDouble1Vec{correction[params.s_Variables[i][0]]});
            }
        }
    }
}

////////// CMetricModel::SBucketStats Implementation //////////

CMetricModel::SBucketStats::SBucketStats(core_t::TTime startTime)
    : s_StartTime(startTime), s_InterimCorrections(1) {
}
}
}
