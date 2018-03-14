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

#include <model/CMetricModelFactory.h>

#include <core/CStateRestoreTraverser.h>

#include <maths/CConstantPrior.h>
#include <maths/CGammaRateConjugate.h>
#include <maths/CLogNormalMeanPrecConjugate.h>
#include <maths/CMultimodalPrior.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/COneOfNPrior.h>
#include <maths/CPrior.h>
#include <maths/CXMeansOnline1d.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CDataGatherer.h>
#include <model/CMetricModel.h>

#include <boost/make_shared.hpp>

namespace ml {
namespace model {

CMetricModelFactory::CMetricModelFactory(const SModelParams& params,
                                         model_t::ESummaryMode summaryMode,
                                         const std::string& summaryCountFieldName)
    : CModelFactory(params),
      m_Identifier(),
      m_SummaryMode(summaryMode),
      m_SummaryCountFieldName(summaryCountFieldName),
      m_UseNull(false),
      m_BucketLength(CAnomalyDetectorModelConfig::DEFAULT_BUCKET_LENGTH),
      m_BucketResultsDelay(0) {}

CMetricModelFactory* CMetricModelFactory::clone(void) const {
    return new CMetricModelFactory(*this);
}

CAnomalyDetectorModel*
CMetricModelFactory::makeModel(const SModelInitializationData& initData) const {
    TDataGathererPtr dataGatherer = initData.s_DataGatherer;
    if (!dataGatherer) {
        LOG_ERROR("NULL data gatherer");
        return 0;
    }
    const TFeatureVec& features = dataGatherer->features();

    TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;
    influenceCalculators.reserve(m_InfluenceFieldNames.size());
    for (const auto& name : m_InfluenceFieldNames) {
        influenceCalculators.push_back(this->defaultInfluenceCalculators(name, features));
    }

    return new CMetricModel(this->modelParams(),
                            dataGatherer,
                            this->defaultFeatureModels(features,
                                                       dataGatherer->bucketLength(),
                                                       0.4,
                                                       true),
                            this->defaultCorrelatePriors(features),
                            this->defaultCorrelates(features),
                            influenceCalculators);
}

CAnomalyDetectorModel*
CMetricModelFactory::makeModel(const SModelInitializationData& initData,
                               core::CStateRestoreTraverser& traverser) const {
    TDataGathererPtr dataGatherer = initData.s_DataGatherer;
    if (!dataGatherer) {
        LOG_ERROR("NULL data gatherer");
        return 0;
    }
    const TFeatureVec& features = dataGatherer->features();

    TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;
    influenceCalculators.reserve(m_InfluenceFieldNames.size());
    for (const auto& name : m_InfluenceFieldNames) {
        influenceCalculators.push_back(this->defaultInfluenceCalculators(name, features));
    }

    return new CMetricModel(this->modelParams(),
                            dataGatherer,
                            this->defaultFeatureModels(features,
                                                       dataGatherer->bucketLength(),
                                                       0.4,
                                                       true),
                            this->defaultCorrelatePriors(features),
                            this->defaultCorrelates(features),
                            influenceCalculators,
                            traverser);
}

CDataGatherer*
CMetricModelFactory::makeDataGatherer(const SGathererInitializationData& initData) const {
    return new CDataGatherer(model_t::E_Metric,
                             m_SummaryMode,
                             this->modelParams(),
                             m_SummaryCountFieldName,
                             m_PartitionFieldName,
                             initData.s_PartitionFieldValue,
                             m_PersonFieldName,
                             EMPTY_STRING, // AttributeFieldName
                             m_ValueFieldName,
                             m_InfluenceFieldNames,
                             m_UseNull,
                             this->searchKey(),
                             m_Features,
                             initData.s_StartTime,
                             initData.s_SampleOverrideCount);
}

CDataGatherer*
CMetricModelFactory::makeDataGatherer(const std::string& partitionFieldValue,
                                      core::CStateRestoreTraverser& traverser) const {
    return new CDataGatherer(model_t::E_Metric,
                             m_SummaryMode,
                             this->modelParams(),
                             m_SummaryCountFieldName,
                             m_PartitionFieldName,
                             partitionFieldValue,
                             m_PersonFieldName,
                             EMPTY_STRING, // AttributeFieldName
                             m_ValueFieldName,
                             m_InfluenceFieldNames,
                             m_UseNull,
                             this->searchKey(),
                             traverser);
}

CMetricModelFactory::TPriorPtr CMetricModelFactory::defaultPrior(model_t::EFeature feature,
                                                                 const SModelParams& params) const {
    // Categorical data all use the multinomial prior. The creation
    // of these priors is managed by defaultCategoricalPrior.
    if (model_t::isCategorical(feature)) {
        return TPriorPtr();
    }

    // If the feature data only ever takes a single value we use a
    // special lightweight prior.
    if (model_t::isConstant(feature)) {
        return boost::make_shared<maths::CConstantPrior>();
    }

    using TPriorPtrVec = std::vector<TPriorPtr>;

    // The data will be arbitrary metric values. Metrics with negative values
    // will be handled by adjusting offsets in the gamma and log-normal priors
    // on the fly. We start off with a small non-zero offset for the log-normal
    // prior because the p.d.f. is zero at zero and for the gamma because the
    // p.d.f. is badly behaved at zero (either zero or infinity), and
    //   1) we expect that zero values are relatively common, and
    //   2) we don't want to adjust the offset if we can avoid this since it
    //      is expensive and results in some loss of information.

    maths_t::EDataType dataType = this->dataType();

    maths::CGammaRateConjugate gammaPrior =
        maths::CGammaRateConjugate::nonInformativePrior(dataType, 0.0, params.s_DecayRate);

    maths::CLogNormalMeanPrecConjugate logNormalPrior =
        maths::CLogNormalMeanPrecConjugate::nonInformativePrior(dataType, 0.0, params.s_DecayRate);

    maths::CNormalMeanPrecConjugate normalPrior =
        maths::CNormalMeanPrecConjugate::nonInformativePrior(dataType, params.s_DecayRate);

    // Create the component priors.
    TPriorPtrVec priors;
    priors.reserve(params.s_MinimumModeFraction <= 0.5 ? 4u : 3u);
    priors.emplace_back(gammaPrior.clone());
    priors.emplace_back(logNormalPrior.clone());
    priors.emplace_back(normalPrior.clone());
    if (params.s_MinimumModeFraction <= 0.5) {
        // Create the multimode prior.
        TPriorPtrVec modePriors;
        modePriors.reserve(3u);
        modePriors.emplace_back(gammaPrior.clone());
        modePriors.emplace_back(logNormalPrior.clone());
        modePriors.emplace_back(normalPrior.clone());
        maths::COneOfNPrior modePrior(modePriors, dataType, params.s_DecayRate);
        maths::CXMeansOnline1d clusterer(dataType,
                                         maths::CAvailableModeDistributions::ALL,
                                         maths_t::E_ClustersFractionWeight,
                                         params.s_DecayRate,
                                         params.s_MinimumModeFraction,
                                         params.s_MinimumModeCount,
                                         params.minimumCategoryCount());
        maths::CMultimodalPrior multimodalPrior(dataType, clusterer, modePrior, params.s_DecayRate);
        priors.emplace_back(multimodalPrior.clone());
    }

    return boost::make_shared<maths::COneOfNPrior>(priors, dataType, params.s_DecayRate);
}

CMetricModelFactory::TMultivariatePriorPtr
CMetricModelFactory::defaultMultivariatePrior(model_t::EFeature feature,
                                              const SModelParams& params) const {
    std::size_t dimension = model_t::dimension(feature);

    // Gaussian mixture for modeling (latitude, longitude).
    if (model_t::isLatLong(feature)) {
        return this->latLongPrior(params);
    }

    TMultivariatePriorPtrVec priors;
    priors.reserve(params.s_MinimumModeFraction <= 0.5 ? 2u : 1u);
    TMultivariatePriorPtr multivariateNormal = this->multivariateNormalPrior(dimension, params);
    priors.push_back(multivariateNormal);
    if (params.s_MinimumModeFraction <= 0.5) {
        priors.push_back(this->multivariateMultimodalPrior(dimension, params, *multivariateNormal));
    }

    return this->multivariateOneOfNPrior(dimension, params, priors);
}

CMetricModelFactory::TMultivariatePriorPtr
CMetricModelFactory::defaultCorrelatePrior(model_t::EFeature /*feature*/,
                                           const SModelParams& params) const {
    TMultivariatePriorPtrVec priors;
    priors.reserve(params.s_MinimumModeFraction <= 0.5 ? 2u : 1u);
    TMultivariatePriorPtr multivariateNormal = this->multivariateNormalPrior(2, params);
    priors.push_back(multivariateNormal);
    if (params.s_MinimumModeFraction <= 0.5) {
        priors.push_back(this->multivariateMultimodalPrior(2, params, *multivariateNormal));
    }
    return this->multivariateOneOfNPrior(2, params, priors);
}

const CSearchKey& CMetricModelFactory::searchKey(void) const {
    if (!m_SearchKeyCache) {
        m_SearchKeyCache.reset(CSearchKey(m_Identifier,
                                          function_t::function(m_Features),
                                          m_UseNull,
                                          this->modelParams().s_ExcludeFrequent,
                                          m_ValueFieldName,
                                          m_PersonFieldName,
                                          "",
                                          m_PartitionFieldName,
                                          m_InfluenceFieldNames));
    }
    return *m_SearchKeyCache;
}

bool CMetricModelFactory::isSimpleCount(void) const {
    return false;
}

model_t::ESummaryMode CMetricModelFactory::summaryMode(void) const {
    return m_SummaryMode;
}

maths_t::EDataType CMetricModelFactory::dataType(void) const {
    return maths_t::E_ContinuousData;
}

void CMetricModelFactory::identifier(int identifier) {
    m_Identifier = identifier;
    m_SearchKeyCache.reset();
}

void CMetricModelFactory::fieldNames(const std::string& partitionFieldName,
                                     const std::string& /*overFieldName*/,
                                     const std::string& byFieldName,
                                     const std::string& valueFieldName,
                                     const TStrVec& influenceFieldNames) {
    m_PartitionFieldName = partitionFieldName;
    m_PersonFieldName = byFieldName;
    m_ValueFieldName = valueFieldName;
    m_InfluenceFieldNames = influenceFieldNames;
    m_SearchKeyCache.reset();
}

void CMetricModelFactory::useNull(bool useNull) {
    m_UseNull = useNull;
    m_SearchKeyCache.reset();
}

void CMetricModelFactory::features(const TFeatureVec& features) {
    m_Features = features;
    m_SearchKeyCache.reset();
}

void CMetricModelFactory::bucketLength(core_t::TTime bucketLength) {
    m_BucketLength = bucketLength;
}

void CMetricModelFactory::bucketResultsDelay(std::size_t bucketResultsDelay) {
    m_BucketResultsDelay = bucketResultsDelay;
}

CMetricModelFactory::TStrCRefVec CMetricModelFactory::partitioningFields(void) const {
    TStrCRefVec result;
    result.reserve(2);
    if (!m_PartitionFieldName.empty()) {
        result.emplace_back(m_PartitionFieldName);
    }
    if (!m_PersonFieldName.empty()) {
        result.emplace_back(m_PersonFieldName);
    }
    return result;
}
}
}
