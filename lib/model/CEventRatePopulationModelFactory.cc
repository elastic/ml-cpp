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

#include <model/CEventRatePopulationModelFactory.h>

#include <core/CStateRestoreTraverser.h>

#include <maths/common/CConstantPrior.h>
#include <maths/common/CGammaRateConjugate.h>
#include <maths/common/CLogNormalMeanPrecConjugate.h>
#include <maths/common/CMultimodalPrior.h>
#include <maths/common/CMultinomialConjugate.h>
#include <maths/common/CNormalMeanPrecConjugate.h>
#include <maths/common/COneOfNPrior.h>
#include <maths/common/CPoissonMeanConjugate.h>
#include <maths/common/CPrior.h>
#include <maths/common/CXMeansOnline1d.h>

#include <maths/time_series/CTimeSeriesDecomposition.h>

#include <model/CDataGatherer.h>
#include <model/CEventRatePopulationModel.h>

#include <memory>

namespace ml {
namespace model {

CEventRatePopulationModelFactory::CEventRatePopulationModelFactory(
    const SModelParams& params,
    const TInterimBucketCorrectorWPtr& interimBucketCorrector,
    model_t::ESummaryMode summaryMode,
    const std::string& summaryCountFieldName)
    : CModelFactory(params, interimBucketCorrector), m_SummaryMode(summaryMode),
      m_SummaryCountFieldName(summaryCountFieldName) {
}

CEventRatePopulationModelFactory* CEventRatePopulationModelFactory::clone() const {
    return new CEventRatePopulationModelFactory(*this);
}

CAnomalyDetectorModel*
CEventRatePopulationModelFactory::makeModel(const SModelInitializationData& initData) const {
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

    return new CEventRatePopulationModel(
        this->modelParams(), dataGatherer,
        this->defaultFeatureModels(features, dataGatherer->bucketLength(),
                                   this->minimumSeasonalVarianceScale(), false),
        this->defaultCorrelatePriors(features), this->defaultCorrelates(features),
        influenceCalculators, this->interimBucketCorrector());
}

CAnomalyDetectorModel*
CEventRatePopulationModelFactory::makeModel(const SModelInitializationData& initData,
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

    return new CEventRatePopulationModel(
        this->modelParams(), dataGatherer,
        this->defaultFeatureModels(features, dataGatherer->bucketLength(),
                                   this->minimumSeasonalVarianceScale(), false),
        this->defaultCorrelatePriors(features), this->defaultCorrelates(features),
        influenceCalculators, this->interimBucketCorrector(), traverser);
}

CDataGatherer*
CEventRatePopulationModelFactory::makeDataGatherer(const SGathererInitializationData& initData) const {
    return new CDataGatherer(model_t::E_PopulationEventRate, m_SummaryMode,
                             this->modelParams(), m_SummaryCountFieldName,
                             initData.s_PartitionFieldValue, m_PersonFieldName,
                             m_AttributeFieldName, m_ValueFieldName, m_InfluenceFieldNames,
                             this->searchKey(), m_Features, initData.s_StartTime, 0);
}

CDataGatherer*
CEventRatePopulationModelFactory::makeDataGatherer(const std::string& partitionFieldValue,
                                                   core::CStateRestoreTraverser& traverser) const {
    return new CDataGatherer(model_t::E_PopulationEventRate, m_SummaryMode,
                             this->modelParams(), m_SummaryCountFieldName,
                             partitionFieldValue, m_PersonFieldName,
                             m_AttributeFieldName, m_ValueFieldName,
                             m_InfluenceFieldNames, this->searchKey(), traverser);
}

CEventRatePopulationModelFactory::TPriorPtr
CEventRatePopulationModelFactory::defaultPrior(model_t::EFeature feature,
                                               const SModelParams& params) const {
    // Categorical data all use the multinomial prior. The creation
    // of these priors is managed by defaultCategoricalPrior.
    if (model_t::isCategorical(feature)) {
        return nullptr;
    }

    // If the feature data only ever takes a single value we use a
    // special lightweight prior.
    if (model_t::isConstant(feature)) {
        return std::make_unique<maths::common::CConstantPrior>();
    }

    if (model_t::isDiurnal(feature)) {
        return this->timeOfDayPrior(params);
    }

    // The feature data will be counts for the number of events in a
    // specified interval. As such we expect counts to be greater than
    // or equal to zero. We use a small non-zero offset, for the log-
    // normal prior because the p.d.f. is zero at zero and for the
    // gamma because the p.d.f. is badly behaved at zero (either zero
    // or infinity), so they can model counts of zero.

    maths_t::EDataType dataType = this->dataType();

    maths::common::CGammaRateConjugate gammaPrior =
        maths::common::CGammaRateConjugate::nonInformativePrior(dataType, 0.0,
                                                                params.s_DecayRate);

    maths::common::CLogNormalMeanPrecConjugate logNormalPrior =
        maths::common::CLogNormalMeanPrecConjugate::nonInformativePrior(
            dataType, 0.0, params.s_DecayRate);

    maths::common::CNormalMeanPrecConjugate normalPrior =
        maths::common::CNormalMeanPrecConjugate::nonInformativePrior(
            dataType, params.s_DecayRate);

    maths::common::CPoissonMeanConjugate poissonPrior =
        maths::common::CPoissonMeanConjugate::nonInformativePrior(0.0, params.s_DecayRate);

    // Create the component priors.
    maths::common::COneOfNPrior::TPriorPtrVec priors;
    priors.reserve(params.s_MinimumModeFraction <= 0.5 ? 5 : 4);
    priors.emplace_back(gammaPrior.clone());
    priors.emplace_back(logNormalPrior.clone());
    priors.emplace_back(normalPrior.clone());
    priors.emplace_back(poissonPrior.clone());
    if (params.s_MinimumModeFraction <= 0.5) {
        // Create the multimode prior.
        maths::common::COneOfNPrior::TPriorPtrVec modePriors;
        modePriors.reserve(3);
        modePriors.emplace_back(gammaPrior.clone());
        modePriors.emplace_back(logNormalPrior.clone());
        modePriors.emplace_back(normalPrior.clone());
        maths::common::COneOfNPrior modePrior(modePriors, dataType, params.s_DecayRate);
        maths::common::CXMeansOnline1d clusterer(
            dataType, maths::common::CAvailableModeDistributions::ALL,
            maths_t::E_ClustersFractionWeight, params.s_DecayRate, params.s_MinimumModeFraction,
            params.s_MinimumModeCount, params.minimumCategoryCount());
        maths::common::CMultimodalPrior multimodalPrior(dataType, clusterer, modePrior,
                                                        params.s_DecayRate);
        priors.emplace_back(multimodalPrior.clone());
    }

    return std::make_unique<maths::common::COneOfNPrior>(priors, dataType,
                                                         params.s_DecayRate);
}

CEventRatePopulationModelFactory::TMultivariatePriorUPtr
CEventRatePopulationModelFactory::defaultMultivariatePrior(model_t::EFeature feature,
                                                           const SModelParams& params) const {
    std::size_t dimension = model_t::dimension(feature);

    TMultivariatePriorUPtrVec priors;
    priors.reserve(params.s_MinimumModeFraction <= 0.5 ? 2 : 1);
    TMultivariatePriorUPtr normal{this->multivariateNormalPrior(dimension, params)};
    priors.push_back(std::move(normal));
    if (params.s_MinimumModeFraction <= 0.5) {
        priors.push_back(this->multivariateMultimodalPrior(dimension, params,
                                                           *priors.back()));
    }

    return this->multivariateOneOfNPrior(dimension, params, priors);
}

CEventRatePopulationModelFactory::TMultivariatePriorUPtr
CEventRatePopulationModelFactory::defaultCorrelatePrior(model_t::EFeature /*feature*/,
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

const CSearchKey& CEventRatePopulationModelFactory::searchKey() const {
    if (m_SearchKeyCache == std::nullopt) {
        m_SearchKeyCache.emplace(m_DetectorIndex, function_t::function(m_Features),
                                 m_UseNull, this->modelParams().s_ExcludeFrequent,
                                 m_ValueFieldName, m_AttributeFieldName, m_PersonFieldName,
                                 m_PartitionFieldName, m_InfluenceFieldNames);
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

void CEventRatePopulationModelFactory::detectorIndex(int detectorIndex) {
    m_DetectorIndex = detectorIndex;
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

CEventRatePopulationModelFactory::TStrCRefVec
CEventRatePopulationModelFactory::partitioningFields() const {
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
