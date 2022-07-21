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

#include <model/CModelFactory.h>

#include <core/CStateRestoreTraverser.h>
#include <core/Constants.h>

#include <maths/common/CModel.h>
#include <maths/common/CMultimodalPrior.h>
#include <maths/common/CMultinomialConjugate.h>
#include <maths/common/CMultivariateMultimodalPriorFactory.h>
#include <maths/common/CMultivariateNormalConjugateFactory.h>
#include <maths/common/CMultivariateOneOfNPriorFactory.h>
#include <maths/common/CNormalMeanPrecConjugate.h>
#include <maths/common/COneOfNPrior.h>
#include <maths/common/CPrior.h>
#include <maths/common/CPriorStateSerialiser.h>
#include <maths/common/CXMeansOnline1d.h>

#include <maths/time_series/CDecayRateController.h>
#include <maths/time_series/CTimeSeriesDecomposition.h>
#include <maths/time_series/CTimeSeriesDecompositionStateSerialiser.h>
#include <maths/time_series/CTimeSeriesDecompositionStub.h>
#include <maths/time_series/CTimeSeriesModel.h>
#include <maths/time_series/CTimeSeriesMultibucketFeatures.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CProbabilityAndInfluenceCalculator.h>

#include <array>
#include <memory>

namespace ml {
namespace model {

const std::string CModelFactory::EMPTY_STRING("");

CModelFactory::CModelFactory(const SModelParams& params,
                             const TInterimBucketCorrectorWPtr& interimBucketCorrector)
    : m_ModelParams(params), m_InterimBucketCorrector(interimBucketCorrector) {
}

const CModelFactory::TFeatureMathsModelPtrPrVec&
CModelFactory::defaultFeatureModels(const TFeatureVec& features,
                                    core_t::TTime bucketLength,
                                    double minimumSeasonalVarianceScale,
                                    bool modelAnomalies) const {
    auto result = m_MathsModelCache.emplace(features, TFeatureMathsModelPtrPrVec());
    if (result.second) {
        result.first->second.reserve(features.size());
        for (auto feature : features) {
            if (model_t::isCategorical(feature)) {
                continue;
            }
            result.first->second.emplace_back(
                feature, this->defaultFeatureModel(feature, bucketLength, minimumSeasonalVarianceScale,
                                                   modelAnomalies));
        }
    }
    return result.first->second;
}

CModelFactory::TMathsModelPtr
CModelFactory::defaultFeatureModel(model_t::EFeature feature,
                                   core_t::TTime bucketLength,
                                   double minimumSeasonalVarianceScale,
                                   bool modelAnomalies) const {
    if (model_t::isCategorical(feature)) {
        return nullptr;
    }

    using TDecayRateController2Ary = std::array<maths::time_series::CDecayRateController, 2>;

    maths::common::CModelParams params{bucketLength,
                                       m_ModelParams.s_LearnRate,
                                       m_ModelParams.s_DecayRate,
                                       minimumSeasonalVarianceScale,
                                       m_ModelParams.s_MinimumTimeToDetectChange,
                                       m_ModelParams.s_MaximumTimeToTestForChange};

    std::size_t dimension{model_t::dimension(feature)};

    bool controlDecayRate{m_ModelParams.s_ControlDecayRate && !model_t::isConstant(feature)};
    TDecayRateController2Ary controllers{
        {maths::time_series::CDecayRateController{
             maths::time_series::CDecayRateController::E_PredictionBias |
                 maths::time_series::CDecayRateController::E_PredictionErrorIncrease,
             dimension},
         maths::time_series::CDecayRateController{
             maths::time_series::CDecayRateController::E_PredictionBias |
                 maths::time_series::CDecayRateController::E_PredictionErrorIncrease |
                 maths::time_series::CDecayRateController::E_PredictionErrorDecrease,
             dimension}}};

    TDecompositionCPtr trend{this->defaultDecomposition(feature, bucketLength)};

    if (dimension == 1) {
        TPriorPtr prior{this->defaultPrior(feature)};
        model_t::TUnivariateMultibucketFeaturePtr multibucketFeature{model_t::univariateMultibucketFeature(
            feature, m_ModelParams.s_MultibucketFeaturesWindowLength)};
        return std::make_unique<maths::time_series::CUnivariateTimeSeriesModel>(
            params, 0 /*identifier (overridden)*/, *trend, *prior,
            controlDecayRate ? &controllers : nullptr, multibucketFeature.get(),
            modelAnomalies && model_t::isConstant(feature) == false);
    }

    TMultivariatePriorUPtr prior{this->defaultMultivariatePrior(feature)};
    model_t::TMultivariateMultibucketFeaturePtr multibucketFeature{model_t::multivariateMultibucketFeature(
        feature, m_ModelParams.s_MultibucketFeaturesWindowLength)};
    return std::make_unique<maths::time_series::CMultivariateTimeSeriesModel>(
        params, *trend, *prior, controlDecayRate ? &controllers : nullptr,
        multibucketFeature.get(), modelAnomalies && model_t::isConstant(feature) == false);
}

const CModelFactory::TFeatureMultivariatePriorSPtrPrVec&
CModelFactory::defaultCorrelatePriors(const TFeatureVec& features) const {
    auto result = m_CorrelatePriorCache.emplace(features, TFeatureMultivariatePriorSPtrPrVec{});
    if (result.second) {
        result.first->second.reserve(features.size());
        for (auto feature : features) {
            if (model_t::isCategorical(feature) || model_t::dimension(feature) > 1) {
                continue;
            }
            result.first->second.emplace_back(feature, this->defaultCorrelatePrior(feature));
        }
    }
    return result.first->second;
}

CModelFactory::TFeatureCorrelationsPtrPrVec
CModelFactory::defaultCorrelates(const TFeatureVec& features) const {
    TFeatureCorrelationsPtrPrVec result;
    result.reserve(features.size());
    for (auto feature : features) {
        if (!model_t::isCategorical(feature) && model_t::dimension(feature) == 1) {
            result.emplace_back(feature, std::make_unique<maths::time_series::CTimeSeriesCorrelations>(
                                             m_ModelParams.s_MinimumSignificantCorrelation,
                                             m_ModelParams.s_DecayRate));
        }
    }
    return result;
}

CModelFactory::TPriorPtr CModelFactory::defaultPrior(model_t::EFeature feature) const {
    return this->defaultPrior(feature, m_ModelParams);
}

CModelFactory::TMultivariatePriorUPtr
CModelFactory::defaultMultivariatePrior(model_t::EFeature feature) const {
    return this->defaultMultivariatePrior(feature, m_ModelParams);
}

CModelFactory::TMultivariatePriorUPtr
CModelFactory::defaultCorrelatePrior(model_t::EFeature feature) const {
    return this->defaultCorrelatePrior(feature, m_ModelParams);
}

maths::common::CMultinomialConjugate CModelFactory::defaultCategoricalPrior() const {
    return maths::common::CMultinomialConjugate::nonInformativePrior(
        std::numeric_limits<std::size_t>::max(), m_ModelParams.s_DecayRate);
}

CModelFactory::TDecompositionCPtr
CModelFactory::defaultDecomposition(model_t::EFeature feature, core_t::TTime bucketLength) const {
    if (model_t::isCategorical(feature)) {
        return nullptr;
    } else if (model_t::isDiurnal(feature) || model_t::isConstant(feature)) {
        return std::make_shared<maths::time_series::CTimeSeriesDecompositionStub>();
    }
    double decayRate = CAnomalyDetectorModelConfig::trendDecayRate(
        m_ModelParams.s_DecayRate, bucketLength);
    return std::make_shared<maths::time_series::CTimeSeriesDecomposition>(
        decayRate, bucketLength, m_ModelParams.s_ComponentSize);
}

const CModelFactory::TFeatureInfluenceCalculatorCPtrPrVec&
CModelFactory::defaultInfluenceCalculators(const std::string& influencerName,
                                           const TFeatureVec& features) const {
    auto& result = m_InfluenceCalculatorCache[{influencerName, features}];

    if (result.empty()) {
        result.reserve(features.size());

        TStrCRefVec partitioningFields = this->partitioningFields();
        std::sort(partitioningFields.begin(), partitioningFields.end(),
                  maths::common::COrderings::SReferenceLess());

        for (auto feature : features) {
            if (model_t::isCategorical(feature)) {
                continue;
            }
            if (std::binary_search(partitioningFields.begin(),
                                   partitioningFields.end(), influencerName,
                                   maths::common::COrderings::SReferenceLess())) {
                result.emplace_back(
                    feature, std::make_shared<CIndicatorInfluenceCalculator>());
            } else {
                result.emplace_back(feature, model_t::influenceCalculator(feature));
            }
        }
    }

    return result;
}

void CModelFactory::sampleCountFactor(std::size_t sampleCountFactor) {
    m_ModelParams.s_SampleCountFactor = sampleCountFactor;
}

void CModelFactory::excludeFrequent(model_t::EExcludeFrequent excludeFrequent) {
    m_ModelParams.s_ExcludeFrequent = excludeFrequent;
}

void CModelFactory::detectionRules(TDetectionRuleVecCRef detectionRules) {
    m_ModelParams.s_DetectionRules = detectionRules;
}

void CModelFactory::scheduledEvents(TStrDetectionRulePrVecCRef scheduledEvents) {
    m_ModelParams.s_ScheduledEvents = scheduledEvents;
}

void CModelFactory::interimBucketCorrector(const TInterimBucketCorrectorWPtr& interimBucketCorrector) {
    m_InterimBucketCorrector = interimBucketCorrector;
}

void CModelFactory::learnRate(double learnRate) {
    m_ModelParams.s_LearnRate = learnRate;
}

void CModelFactory::decayRate(double decayRate) {
    m_ModelParams.s_DecayRate = decayRate;
}

void CModelFactory::initialDecayRateMultiplier(double multiplier) {
    m_ModelParams.s_InitialDecayRateMultiplier = multiplier;
}

void CModelFactory::maximumUpdatesPerBucket(double maximumUpdatesPerBucket) {
    m_ModelParams.s_MaximumUpdatesPerBucket = maximumUpdatesPerBucket;
}

void CModelFactory::pruneWindowScaleMinimum(double factor) {
    m_ModelParams.s_PruneWindowScaleMinimum = factor;
}

void CModelFactory::pruneWindowScaleMaximum(double factor) {
    m_ModelParams.s_PruneWindowScaleMaximum = factor;
}

void CModelFactory::multibucketFeaturesWindowLength(std::size_t length) {
    m_ModelParams.s_MultibucketFeaturesWindowLength = length;
}

void CModelFactory::multivariateByFields(bool enabled) {
    m_ModelParams.s_MultivariateByFields = enabled;
}

void CModelFactory::minimumModeFraction(double minimumModeFraction) {
    m_ModelParams.s_MinimumModeFraction = minimumModeFraction;
}

void CModelFactory::minimumModeCount(double minimumModeCount) {
    m_ModelParams.s_MinimumModeCount = minimumModeCount;
}

void CModelFactory::componentSize(std::size_t componentSize) {
    m_ModelParams.s_ComponentSize = componentSize;
}

double CModelFactory::minimumModeFraction() const {
    return m_ModelParams.s_MinimumModeFraction;
}

double CModelFactory::minimumModeCount() const {
    return m_ModelParams.s_MinimumModeCount;
}

std::size_t CModelFactory::componentSize() const {
    return m_ModelParams.s_ComponentSize;
}

void CModelFactory::updateBucketLength(core_t::TTime length) {
    m_ModelParams.s_BucketLength = length;
}

void CModelFactory::annotationsEnabled(bool enabled) {
    m_ModelParams.s_AnnotationsEnabled = enabled;
}

CModelFactory::TInterimBucketCorrectorPtr CModelFactory::interimBucketCorrector() const {
    TInterimBucketCorrectorPtr result{m_InterimBucketCorrector.lock()};
    if (result == nullptr) {
        // This can never happen if the factory is obtained from the model
        // config object, which ensures the interim bucket corrector is
        // always available. The model objects expect this to be available
        // so failing gracefully here is the best we can do.
        LOG_ABORT(<< "Unable to get interim bucket corrector");
    }
    return result;
}

CModelFactory::TMultivariatePriorUPtr
CModelFactory::multivariateNormalPrior(std::size_t dimension, const SModelParams& params) const {
    return maths::common::CMultivariateNormalConjugateFactory::nonInformative(
        dimension, this->dataType(), params.s_DecayRate);
}

CModelFactory::TMultivariatePriorUPtr
CModelFactory::multivariateMultimodalPrior(std::size_t dimension,
                                           const SModelParams& params,
                                           const maths::common::CMultivariatePrior& modePrior) const {
    return maths::common::CMultivariateMultimodalPriorFactory::nonInformative(
        dimension, this->dataType(), params.s_DecayRate,
        maths_t::E_ClustersFractionWeight, params.s_MinimumModeFraction,
        params.s_MinimumModeCount, params.minimumCategoryCount(), modePrior);
}

CModelFactory::TMultivariatePriorUPtr
CModelFactory::multivariateOneOfNPrior(std::size_t dimension,
                                       const SModelParams& params,
                                       const TMultivariatePriorUPtrVec& models) const {
    return maths::common::CMultivariateOneOfNPriorFactory::nonInformative(
        dimension, this->dataType(), params.s_DecayRate, models);
}

CModelFactory::TPriorPtr CModelFactory::timeOfDayPrior(const SModelParams& params) const {
    maths_t::EDataType dataType = this->dataType();
    maths::common::CNormalMeanPrecConjugate normalPrior =
        maths::common::CNormalMeanPrecConjugate::nonInformativePrior(
            dataType, params.s_DecayRate);

    // Create a multimodal prior with purely normal distributions
    // - don't bother with long-tail distributions

    maths::common::COneOfNPrior::TPriorPtrVec modePriors;
    modePriors.reserve(1);
    modePriors.emplace_back(normalPrior.clone());
    maths::common::COneOfNPrior modePrior(modePriors, dataType, params.s_DecayRate);
    maths::common::CXMeansOnline1d clusterer(
        dataType, maths::common::CAvailableModeDistributions::NORMAL,
        maths_t::E_ClustersFractionWeight, params.s_DecayRate,
        0.03 /*minimumClusterFraction*/, 4 /*minimumClusterCount*/,
        CAnomalyDetectorModelConfig::DEFAULT_CATEGORY_DELETE_FRACTION);

    return std::make_unique<maths::common::CMultimodalPrior>(
        dataType, clusterer, modePrior, params.s_DecayRate);
}

CModelFactory::TMultivariatePriorUPtr
CModelFactory::latLongPrior(const SModelParams& params) const {
    maths_t::EDataType dataType = this->dataType();
    TMultivariatePriorUPtr modePrior =
        maths::common::CMultivariateNormalConjugateFactory::nonInformative(
            2, dataType, params.s_DecayRate);
    return maths::common::CMultivariateMultimodalPriorFactory::nonInformative(
        2 /*dimension*/, dataType, params.s_DecayRate, maths_t::E_ClustersFractionWeight,
        0.03 /*minimumClusterFraction*/, 4 /*minimumClusterCount*/,
        CAnomalyDetectorModelConfig::DEFAULT_CATEGORY_DELETE_FRACTION, *modePrior);
}

const SModelParams& CModelFactory::modelParams() const {
    return m_ModelParams;
}

CModelFactory::SModelInitializationData::SModelInitializationData(const TDataGathererPtr& dataGatherer)
    : s_DataGatherer(dataGatherer) {
}

CModelFactory::SGathererInitializationData::SGathererInitializationData(
    core_t::TTime startTime,
    const std::string& partitionFieldValue,
    unsigned int sampleOverrideCount)
    : s_StartTime(startTime), s_PartitionFieldValue(partitionFieldValue),
      s_SampleOverrideCount(sampleOverrideCount) {
}

CModelFactory::SGathererInitializationData::SGathererInitializationData(core_t::TTime startTime)
    : s_StartTime(startTime), s_PartitionFieldValue(EMPTY_STRING),
      s_SampleOverrideCount(0u) {
}
}
}
