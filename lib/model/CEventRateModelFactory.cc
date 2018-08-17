/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CEventRateModelFactory.h>

#include <core/CStateRestoreTraverser.h>

#include <maths/CConstantPrior.h>
#include <maths/CGammaRateConjugate.h>
#include <maths/CLogNormalMeanPrecConjugate.h>
#include <maths/CMultimodalPrior.h>
#include <maths/CMultivariateOneOfNPrior.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/COneOfNPrior.h>
#include <maths/CPoissonMeanConjugate.h>
#include <maths/CPrior.h>
#include <maths/CXMeansOnline1d.h>

#include <model/CDataGatherer.h>
#include <model/CEventRateModel.h>
#include <model/CSearchKey.h>

#include <boost/make_unique.hpp>

#include <memory>

namespace ml {
namespace model {

CEventRateModelFactory::CEventRateModelFactory(const SModelParams& params,
                                               const TInterimBucketCorrectorWPtr& interimBucketCorrector,
                                               model_t::ESummaryMode summaryMode,
                                               const std::string& summaryCountFieldName)
    : CModelFactory(params, interimBucketCorrector), m_SummaryMode(summaryMode),
      m_SummaryCountFieldName(summaryCountFieldName) {
}

CEventRateModelFactory* CEventRateModelFactory::clone() const {
    return new CEventRateModelFactory(*this);
}

CAnomalyDetectorModel*
CEventRateModelFactory::makeModel(const SModelInitializationData& initData) const {
    TDataGathererPtr dataGatherer = initData.s_DataGatherer;
    if (dataGatherer == nullptr) {
        LOG_ERROR(<< "NULL data gatherer");
        return nullptr;
    }
    const TFeatureVec& features = dataGatherer->features();

    TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;
    influenceCalculators.reserve(m_InfluenceFieldNames.size());
    for (const auto& name : m_InfluenceFieldNames) {
        influenceCalculators.push_back(this->defaultInfluenceCalculators(name, features));
    }

    return new CEventRateModel(
        this->modelParams(), dataGatherer,
        this->defaultFeatureModels(features, dataGatherer->bucketLength(),
                                   this->minimumSeasonalVarianceScale(), true),
        this->defaultCorrelatePriors(features),
        this->defaultCorrelates(features), this->defaultCategoricalPrior(),
        influenceCalculators, this->interimBucketCorrector());
}

CAnomalyDetectorModel*
CEventRateModelFactory::makeModel(const SModelInitializationData& initData,
                                  core::CStateRestoreTraverser& traverser) const {
    TDataGathererPtr dataGatherer = initData.s_DataGatherer;
    if (dataGatherer == nullptr) {
        LOG_ERROR(<< "NULL data gatherer");
        return nullptr;
    }
    const TFeatureVec& features = dataGatherer->features();

    TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;
    influenceCalculators.reserve(m_InfluenceFieldNames.size());
    for (const auto& name : m_InfluenceFieldNames) {
        influenceCalculators.push_back(this->defaultInfluenceCalculators(name, features));
    }

    return new CEventRateModel(
        this->modelParams(), dataGatherer,
        this->defaultFeatureModels(features, dataGatherer->bucketLength(),
                                   this->minimumSeasonalVarianceScale(), true),
        this->defaultCorrelatePriors(features), this->defaultCorrelates(features),
        influenceCalculators, this->interimBucketCorrector(), traverser);
}

CDataGatherer*
CEventRateModelFactory::makeDataGatherer(const SGathererInitializationData& initData) const {
    return new CDataGatherer(model_t::E_EventRate, m_SummaryMode,
                             this->modelParams(), m_SummaryCountFieldName,
                             initData.s_PartitionFieldValue, m_PersonFieldName,
                             EMPTY_STRING /*AttributeFieldName*/, m_ValueFieldName,
                             m_InfluenceFieldNames, this->searchKey(), m_Features,
                             initData.s_StartTime, initData.s_SampleOverrideCount);
}

CDataGatherer*
CEventRateModelFactory::makeDataGatherer(const std::string& partitionFieldValue,
                                         core::CStateRestoreTraverser& traverser) const {
    return new CDataGatherer(model_t::E_EventRate, m_SummaryMode,
                             this->modelParams(), m_SummaryCountFieldName,
                             partitionFieldValue, m_PersonFieldName,
                             EMPTY_STRING /*AttributeFieldName*/, m_ValueFieldName,
                             m_InfluenceFieldNames, this->searchKey(), traverser);
}

CEventRateModelFactory::TPriorPtr
CEventRateModelFactory::defaultPrior(model_t::EFeature feature,
                                     const SModelParams& params) const {
    // Categorical data all use the multinomial prior. The creation
    // of these priors is managed by defaultCategoricalPrior.
    if (model_t::isCategorical(feature)) {
        return nullptr;
    }

    // If the feature data only ever takes a single value we use a
    // special lightweight prior.
    if (model_t::isConstant(feature)) {
        return boost::make_unique<maths::CConstantPrior>();
    }

    // Gaussian mixture for modeling time-of-day and time-of-week.
    if (model_t::isDiurnal(feature)) {
        return this->timeOfDayPrior(params);
    }

    // The data will be counts for the number of events in a specified
    // interval. As such we expect counts to be greater than or equal
    // to zero. We use a small non-zero offset, for the log-normal prior
    // because the p.d.f. is zero at zero and for the gamma because the
    // p.d.f. is badly behaved at zero (either zero or infinity), so they
    // can model counts of zero.

    maths_t::EDataType dataType = this->dataType();

    maths::CGammaRateConjugate gammaPrior =
        maths::CGammaRateConjugate::nonInformativePrior(dataType, 0.0, params.s_DecayRate);

    maths::CLogNormalMeanPrecConjugate logNormalPrior =
        maths::CLogNormalMeanPrecConjugate::nonInformativePrior(dataType, 0.0,
                                                                params.s_DecayRate);

    maths::CNormalMeanPrecConjugate normalPrior =
        maths::CNormalMeanPrecConjugate::nonInformativePrior(dataType, params.s_DecayRate);

    maths::CPoissonMeanConjugate poissonPrior =
        maths::CPoissonMeanConjugate::nonInformativePrior(0.0, params.s_DecayRate);

    // Create the component priors.
    maths::COneOfNPrior::TPriorPtrVec priors;
    priors.reserve(params.s_MinimumModeFraction <= 0.5 ? 5u : 4u);
    priors.emplace_back(gammaPrior.clone());
    priors.emplace_back(logNormalPrior.clone());
    priors.emplace_back(normalPrior.clone());
    priors.emplace_back(poissonPrior.clone());
    if (params.s_MinimumModeFraction <= 0.5) {
        // Create the multimode prior.
        maths::COneOfNPrior::TPriorPtrVec modePriors;
        modePriors.reserve(3u);
        modePriors.emplace_back(gammaPrior.clone());
        modePriors.emplace_back(logNormalPrior.clone());
        modePriors.emplace_back(normalPrior.clone());
        maths::COneOfNPrior modePrior(modePriors, dataType, params.s_DecayRate);
        maths::CXMeansOnline1d clusterer(
            dataType, maths::CAvailableModeDistributions::ALL,
            maths_t::E_ClustersFractionWeight, params.s_DecayRate, params.s_MinimumModeFraction,
            params.s_MinimumModeCount, params.minimumCategoryCount());
        maths::CMultimodalPrior multimodalPrior(dataType, clusterer, modePrior,
                                                params.s_DecayRate);
        priors.emplace_back(multimodalPrior.clone());
    }

    return boost::make_unique<maths::COneOfNPrior>(priors, dataType, params.s_DecayRate);
}

CEventRateModelFactory::TMultivariatePriorUPtr
CEventRateModelFactory::defaultMultivariatePrior(model_t::EFeature feature,
                                                 const SModelParams& params) const {
    std::size_t dimension = model_t::dimension(feature);

    TMultivariatePriorUPtrVec priors;
    priors.reserve(params.s_MinimumModeFraction <= 0.5 ? 2u : 1u);
    TMultivariatePriorUPtr normal{this->multivariateNormalPrior(dimension, params)};
    priors.push_back(std::move(normal));
    if (params.s_MinimumModeFraction <= 0.5) {
        priors.push_back(this->multivariateMultimodalPrior(dimension, params,
                                                           *priors.back()));
    }

    return this->multivariateOneOfNPrior(dimension, params, priors);
}

CEventRateModelFactory::TMultivariatePriorUPtr
CEventRateModelFactory::defaultCorrelatePrior(model_t::EFeature /*feature*/,
                                              const SModelParams& params) const {
    TMultivariatePriorUPtrVec priors;
    priors.reserve(params.s_MinimumModeFraction <= 0.5 ? 2u : 1u);
    TMultivariatePriorUPtr normal{this->multivariateNormalPrior(2, params)};
    priors.push_back(std::move(normal));
    if (params.s_MinimumModeFraction <= 0.5) {
        priors.push_back(this->multivariateMultimodalPrior(2, params, *priors.back()));
    }
    return this->multivariateOneOfNPrior(2, params, priors);
}

const CSearchKey& CEventRateModelFactory::searchKey() const {
    if (!m_SearchKeyCache) {
        m_SearchKeyCache.emplace(m_Identifier, function_t::function(m_Features),
                                 m_UseNull, this->modelParams().s_ExcludeFrequent,
                                 m_ValueFieldName, m_PersonFieldName, "",
                                 m_PartitionFieldName, m_InfluenceFieldNames);
    }
    return *m_SearchKeyCache;
}

bool CEventRateModelFactory::isSimpleCount() const {
    return CSearchKey::isSimpleCount(function_t::function(m_Features), m_PersonFieldName);
}

model_t::ESummaryMode CEventRateModelFactory::summaryMode() const {
    return m_SummaryMode;
}

maths_t::EDataType CEventRateModelFactory::dataType() const {
    return maths_t::E_IntegerData;
}

void CEventRateModelFactory::identifier(int identifier) {
    m_Identifier = identifier;
    m_SearchKeyCache.reset();
}

void CEventRateModelFactory::fieldNames(const std::string& partitionFieldName,
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

void CEventRateModelFactory::useNull(bool useNull) {
    m_UseNull = useNull;
    m_SearchKeyCache.reset();
}

void CEventRateModelFactory::features(const TFeatureVec& features) {
    m_Features = features;
    m_SearchKeyCache.reset();
}

void CEventRateModelFactory::bucketResultsDelay(std::size_t bucketResultsDelay) {
    m_BucketResultsDelay = bucketResultsDelay;
}

double CEventRateModelFactory::minimumSeasonalVarianceScale() const {
    return 0.4;
}

CEventRateModelFactory::TStrCRefVec CEventRateModelFactory::partitioningFields() const {
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
