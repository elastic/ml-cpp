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

#include <model/CEventRatePopulationModelFactory.h>

#include <core/CStateRestoreTraverser.h>

#include <maths/CConstantPrior.h>
#include <maths/CGammaRateConjugate.h>
#include <maths/CLogNormalMeanPrecConjugate.h>
#include <maths/CMultimodalPrior.h>
#include <maths/CMultinomialConjugate.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/COneOfNPrior.h>
#include <maths/CPoissonMeanConjugate.h>
#include <maths/CPrior.h>
#include <maths/CTimeSeriesDecomposition.h>
#include <maths/CXMeansOnline1d.h>

#include <model/CDataGatherer.h>
#include <model/CEventRatePopulationModel.h>

#include <boost/make_shared.hpp>

namespace ml {
namespace model {

CEventRatePopulationModelFactory::CEventRatePopulationModelFactory(const SModelParams& params,
                                                                   model_t::ESummaryMode summaryMode,
                                                                   const std::string& summaryCountFieldName)
    : CModelFactory(params),
      m_Identifier(),
      m_SummaryMode(summaryMode),
      m_SummaryCountFieldName(summaryCountFieldName),
      m_UseNull(false),
      m_BucketResultsDelay(0) {
}

CEventRatePopulationModelFactory* CEventRatePopulationModelFactory::clone() const {
    return new CEventRatePopulationModelFactory(*this);
}

CAnomalyDetectorModel* CEventRatePopulationModelFactory::makeModel(const SModelInitializationData& initData) const {
    TDataGathererPtr dataGatherer = initData.s_DataGatherer;
    if (!dataGatherer) {
        LOG_ERROR("NULL data gatherer");
        return nullptr;
    }
    const TFeatureVec& features = dataGatherer->features();

    TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;
    influenceCalculators.reserve(m_InfluenceFieldNames.size());
    for (const auto& name : m_InfluenceFieldNames) {
        influenceCalculators.push_back(this->defaultInfluenceCalculators(name, features));
    }

    return new CEventRatePopulationModel(
        this->modelParams(),
        dataGatherer,
        this->defaultFeatureModels(features, dataGatherer->bucketLength(), this->minimumSeasonalVarianceScale(), false),
        this->defaultCorrelatePriors(features),
        this->defaultCorrelates(features),
        influenceCalculators);
}

CAnomalyDetectorModel* CEventRatePopulationModelFactory::makeModel(const SModelInitializationData& initData,
                                                                   core::CStateRestoreTraverser& traverser) const {
    TDataGathererPtr dataGatherer = initData.s_DataGatherer;
    if (!dataGatherer) {
        LOG_ERROR("NULL data gatherer");
        return nullptr;
    }
    const TFeatureVec& features = dataGatherer->features();

    TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;
    influenceCalculators.reserve(m_InfluenceFieldNames.size());
    for (const auto& name : m_InfluenceFieldNames) {
        influenceCalculators.push_back(this->defaultInfluenceCalculators(name, features));
    }

    return new CEventRatePopulationModel(this->modelParams(),
                                         dataGatherer,
                                         this->defaultFeatureModels(features, dataGatherer->bucketLength(), 1.0, false),
                                         this->defaultCorrelatePriors(features),
                                         this->defaultCorrelates(features),
                                         influenceCalculators,
                                         traverser);
}

CDataGatherer* CEventRatePopulationModelFactory::makeDataGatherer(const SGathererInitializationData& initData) const {
    return new CDataGatherer(model_t::E_PopulationEventRate,
                             m_SummaryMode,
                             this->modelParams(),
                             m_SummaryCountFieldName,
                             m_PartitionFieldName,
                             initData.s_PartitionFieldValue,
                             m_PersonFieldName,
                             m_AttributeFieldName,
                             m_ValueFieldName,
                             m_InfluenceFieldNames,
                             m_UseNull,
                             this->searchKey(),
                             m_Features,
                             initData.s_StartTime,
                             0);
}

CDataGatherer* CEventRatePopulationModelFactory::makeDataGatherer(const std::string& partitionFieldValue,
                                                                  core::CStateRestoreTraverser& traverser) const {
    return new CDataGatherer(model_t::E_PopulationEventRate,
                             m_SummaryMode,
                             this->modelParams(),
                             m_SummaryCountFieldName,
                             m_PartitionFieldName,
                             partitionFieldValue,
                             m_PersonFieldName,
                             m_AttributeFieldName,
                             m_ValueFieldName,
                             m_InfluenceFieldNames,
                             m_UseNull,
                             this->searchKey(),
                             traverser);
}

CEventRatePopulationModelFactory::TPriorPtr CEventRatePopulationModelFactory::defaultPrior(model_t::EFeature feature,
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

    if (model_t::isDiurnal(feature)) {
        return this->timeOfDayPrior(params);
    }

    using TPriorPtrVec = std::vector<TPriorPtr>;

    // The feature data will be counts for the number of events in a
    // specified interval. As such we expect counts to be greater than
    // or equal to zero. We use a small non-zero offset, for the log-
    // normal prior because the p.d.f. is zero at zero and for the
    // gamma because the p.d.f. is badly behaved at zero (either zero
    // or infinity), so they can model counts of zero.

    maths_t::EDataType dataType = this->dataType();

    maths::CGammaRateConjugate gammaPrior = maths::CGammaRateConjugate::nonInformativePrior(dataType, 0.0, params.s_DecayRate);

    maths::CLogNormalMeanPrecConjugate logNormalPrior =
        maths::CLogNormalMeanPrecConjugate::nonInformativePrior(dataType, 0.0, params.s_DecayRate);

    maths::CNormalMeanPrecConjugate normalPrior = maths::CNormalMeanPrecConjugate::nonInformativePrior(dataType, params.s_DecayRate);

    maths::CPoissonMeanConjugate poissonPrior = maths::CPoissonMeanConjugate::nonInformativePrior(0.0, params.s_DecayRate);

    // Create the component priors.
    TPriorPtrVec priors;
    priors.reserve(params.s_MinimumModeFraction <= 0.5 ? 5u : 4u);
    priors.emplace_back(gammaPrior.clone());
    priors.emplace_back(logNormalPrior.clone());
    priors.emplace_back(normalPrior.clone());
    priors.emplace_back(poissonPrior.clone());
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

CEventRatePopulationModelFactory::TMultivariatePriorPtr
CEventRatePopulationModelFactory::defaultMultivariatePrior(model_t::EFeature feature, const SModelParams& params) const {
    std::size_t dimension = model_t::dimension(feature);

    TMultivariatePriorPtrVec priors;
    priors.reserve(params.s_MinimumModeFraction <= 0.5 ? 2u : 1u);
    TMultivariatePriorPtr multivariateNormal = this->multivariateNormalPrior(dimension, params);
    priors.push_back(multivariateNormal);
    if (params.s_MinimumModeFraction <= 0.5) {
        priors.push_back(this->multivariateMultimodalPrior(dimension, params, *multivariateNormal));
    }

    return this->multivariateOneOfNPrior(dimension, params, priors);
}

CEventRatePopulationModelFactory::TMultivariatePriorPtr
CEventRatePopulationModelFactory::defaultCorrelatePrior(model_t::EFeature /*feature*/, const SModelParams& params) const {
    TMultivariatePriorPtrVec priors;
    priors.reserve(params.s_MinimumModeFraction <= 0.5 ? 2u : 1u);
    TMultivariatePriorPtr multivariateNormal = this->multivariateNormalPrior(2, params);
    priors.push_back(multivariateNormal);
    if (params.s_MinimumModeFraction <= 0.5) {
        priors.push_back(this->multivariateMultimodalPrior(2, params, *multivariateNormal));
    }
    return this->multivariateOneOfNPrior(2, params, priors);
}

const CSearchKey& CEventRatePopulationModelFactory::searchKey() const {
    if (!m_SearchKeyCache) {
        m_SearchKeyCache.reset(CSearchKey(m_Identifier,
                                          function_t::function(m_Features),
                                          m_UseNull,
                                          this->modelParams().s_ExcludeFrequent,
                                          m_ValueFieldName,
                                          m_AttributeFieldName,
                                          m_PersonFieldName,
                                          m_PartitionFieldName,
                                          m_InfluenceFieldNames));
    }
    return *m_SearchKeyCache;
}

bool CEventRatePopulationModelFactory::isSimpleCount() const {
    return false;
}

model_t::ESummaryMode CEventRatePopulationModelFactory::summaryMode() const {
    return m_SummaryMode;
}

maths_t::EDataType CEventRatePopulationModelFactory::dataType() const {
    return maths_t::E_IntegerData;
}

void CEventRatePopulationModelFactory::identifier(int identifier) {
    m_Identifier = identifier;
    m_SearchKeyCache.reset();
}

void CEventRatePopulationModelFactory::fieldNames(const std::string& partitionFieldName,
                                                  const std::string& overFieldName,
                                                  const std::string& byFieldName,
                                                  const std::string& valueFieldName,
                                                  const TStrVec& influenceFieldNames) {
    m_PartitionFieldName = partitionFieldName;
    m_PersonFieldName = overFieldName;
    m_AttributeFieldName = byFieldName;
    m_ValueFieldName = valueFieldName;
    m_InfluenceFieldNames = influenceFieldNames;
    m_SearchKeyCache.reset();
}

void CEventRatePopulationModelFactory::useNull(bool useNull) {
    m_UseNull = useNull;
    m_SearchKeyCache.reset();
}

void CEventRatePopulationModelFactory::features(const TFeatureVec& features) {
    m_Features = features;
    m_SearchKeyCache.reset();
}

void CEventRatePopulationModelFactory::bucketResultsDelay(std::size_t bucketResultsDelay) {
    m_BucketResultsDelay = bucketResultsDelay;
}

CEventRatePopulationModelFactory::TStrCRefVec CEventRatePopulationModelFactory::partitioningFields() const {
    TStrCRefVec result;
    result.reserve(3);
    if (!m_PartitionFieldName.empty()) {
        result.emplace_back(m_PartitionFieldName);
    }
    if (!m_PersonFieldName.empty()) {
        result.emplace_back(m_PersonFieldName);
    }
    if (!m_AttributeFieldName.empty()) {
        result.emplace_back(m_AttributeFieldName);
    }
    return result;
}

double CEventRatePopulationModelFactory::minimumSeasonalVarianceScale() const {
    return 1.0;
}
}
}
