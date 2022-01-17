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

#include <model/CMetricPopulationModelFactory.h>

#include <core/CStateRestoreTraverser.h>

#include <maths/common/CConstantPrior.h>
#include <maths/common/CGammaRateConjugate.h>
#include <maths/common/CLogNormalMeanPrecConjugate.h>
#include <maths/common/CMultimodalPrior.h>
#include <maths/common/CNormalMeanPrecConjugate.h>
#include <maths/common/COneOfNPrior.h>
#include <maths/common/CPrior.h>
#include <maths/common/CXMeansOnline1d.h>

#include <maths/time_series/CTimeSeriesDecomposition.h>

#include <model/CDataGatherer.h>
#include <model/CMetricPopulationModel.h>

#include <memory>

namespace ml {
namespace model {

CMetricPopulationModelFactory::CMetricPopulationModelFactory(
    const SModelParams& params,
    const TInterimBucketCorrectorWPtr& interimBucketCorrector,
    model_t::ESummaryMode summaryMode,
    const std::string& summaryCountFieldName)
    : CModelFactory(params, interimBucketCorrector), m_SummaryMode(summaryMode),
      m_SummaryCountFieldName(summaryCountFieldName) {
}

CMetricPopulationModelFactory* CMetricPopulationModelFactory::clone() const {
    return new CMetricPopulationModelFactory(*this);
}

CAnomalyDetectorModel*
CMetricPopulationModelFactory::makeModel(const SModelInitializationData& initData) const {
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

    return new CMetricPopulationModel(
        this->modelParams(), dataGatherer,
        this->defaultFeatureModels(features, dataGatherer->bucketLength(),
                                   this->minimumSeasonalVarianceScale(), false),
        this->defaultCorrelatePriors(features), this->defaultCorrelates(features),
        influenceCalculators, this->interimBucketCorrector());
}

CAnomalyDetectorModel*
CMetricPopulationModelFactory::makeModel(const SModelInitializationData& initData,
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

    return new CMetricPopulationModel(
        this->modelParams(), dataGatherer,
        this->defaultFeatureModels(features, dataGatherer->bucketLength(),
                                   this->minimumSeasonalVarianceScale(), false),
        this->defaultCorrelatePriors(features), this->defaultCorrelates(features),
        influenceCalculators, this->interimBucketCorrector(), traverser);
}

CDataGatherer*
CMetricPopulationModelFactory::makeDataGatherer(const SGathererInitializationData& initData) const {
    return new CDataGatherer(model_t::E_PopulationMetric, m_SummaryMode,
                             this->modelParams(), m_SummaryCountFieldName,
                             initData.s_PartitionFieldValue, m_PersonFieldName,
                             m_AttributeFieldName, m_ValueFieldName,
                             m_InfluenceFieldNames, this->searchKey(), m_Features,
                             initData.s_StartTime, initData.s_SampleOverrideCount);
}

CDataGatherer*
CMetricPopulationModelFactory::makeDataGatherer(const std::string& partitionFieldValue,
                                                core::CStateRestoreTraverser& traverser) const {
    return new CDataGatherer(model_t::E_PopulationMetric, m_SummaryMode,
                             this->modelParams(), m_SummaryCountFieldName,
                             partitionFieldValue, m_PersonFieldName,
                             m_AttributeFieldName, m_ValueFieldName,
                             m_InfluenceFieldNames, this->searchKey(), traverser);
}

CMetricPopulationModelFactory::TPriorPtr
CMetricPopulationModelFactory::defaultPrior(model_t::EFeature feature,
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

    // The data will be arbitrary metric values. Metrics with negative values
    // will be handled by adjusting offsets in the gamma and log-normal priors
    // on the fly. We start off with a small non-zero offset for the log-normal
    // prior because the p.d.f. is zero at zero and for the gamma because the
    // p.d.f. is badly behaved at zero (either zero or infinity), and
    //   1) we expect that zero values are relatively common, and
    //   2) we don't want to adjust the offset if we can avoid this since it
    //      is expensive and results in some loss of information.

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

    // Create the component priors.
    maths::common::COneOfNPrior::TPriorPtrVec priors;
    priors.reserve(params.s_MinimumModeFraction <= 0.5 ? 4u : 3u);
    priors.emplace_back(gammaPrior.clone());
    priors.emplace_back(logNormalPrior.clone());
    priors.emplace_back(normalPrior.clone());
    if (params.s_MinimumModeFraction <= 0.5) {
        // Create the multimode prior.
        maths::common::COneOfNPrior::TPriorPtrVec modePriors;
        modePriors.reserve(3u);
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

CMetricPopulationModelFactory::TMultivariatePriorUPtr
CMetricPopulationModelFactory::defaultMultivariatePrior(model_t::EFeature feature,
                                                        const SModelParams& params) const {
    std::size_t dimension = model_t::dimension(feature);

    // Gaussian mixture for modeling (latitude, longitude).
    if (model_t::isLatLong(feature)) {
        return this->latLongPrior(params);
    }

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

CMetricPopulationModelFactory::TMultivariatePriorUPtr
CMetricPopulationModelFactory::defaultCorrelatePrior(model_t::EFeature /*feature*/,
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

const CSearchKey& CMetricPopulationModelFactory::searchKey() const {
    if (!m_SearchKeyCache) {
        m_SearchKeyCache.emplace(m_DetectorIndex, function_t::function(m_Features),
                                 m_UseNull, this->modelParams().s_ExcludeFrequent,
                                 m_ValueFieldName, m_AttributeFieldName, m_PersonFieldName,
                                 m_PartitionFieldName, m_InfluenceFieldNames);
    }
    return *m_SearchKeyCache;
}

bool CMetricPopulationModelFactory::isSimpleCount() const {
    return false;
}

model_t::ESummaryMode CMetricPopulationModelFactory::summaryMode() const {
    return m_SummaryMode;
}

maths_t::EDataType CMetricPopulationModelFactory::dataType() const {
    return maths_t::E_ContinuousData;
}

void CMetricPopulationModelFactory::detectorIndex(int detectorIndex) {
    m_DetectorIndex = detectorIndex;
    m_SearchKeyCache.reset();
}

void CMetricPopulationModelFactory::fieldNames(const std::string& partitionFieldName,
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

void CMetricPopulationModelFactory::useNull(bool useNull) {
    m_UseNull = useNull;
    m_SearchKeyCache.reset();
}

void CMetricPopulationModelFactory::features(const TFeatureVec& features) {
    m_Features = features;
    m_SearchKeyCache.reset();
}

double CMetricPopulationModelFactory::minimumSeasonalVarianceScale() const {
    return 1.0;
}

CMetricPopulationModelFactory::TStrCRefVec
CMetricPopulationModelFactory::partitioningFields() const {
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
}
}
