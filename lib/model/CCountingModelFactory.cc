/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CCountingModelFactory.h>

#include <core/CStateRestoreTraverser.h>

#include <maths/CConstantPrior.h>
#include <maths/CMultivariateConstantPrior.h>
#include <maths/CTimeSeriesModel.h>

#include <model/CCountingModel.h>
#include <model/CDataGatherer.h>
#include <model/CSearchKey.h>


#include <memory>

namespace ml {
namespace model {

CCountingModelFactory::CCountingModelFactory(const SModelParams& params,
                                             const TInterimBucketCorrectorWPtr& interimBucketCorrector,
                                             model_t::ESummaryMode summaryMode,
                                             const std::string& summaryCountFieldName)
    : CModelFactory(params, interimBucketCorrector), m_Identifier(),
      m_SummaryMode(summaryMode),
      m_SummaryCountFieldName(summaryCountFieldName), m_UseNull(false) {
}

CCountingModelFactory* CCountingModelFactory::clone() const {
    return new CCountingModelFactory(*this);
}

CAnomalyDetectorModel*
CCountingModelFactory::makeModel(const SModelInitializationData& initData) const {
    TDataGathererPtr dataGatherer = initData.s_DataGatherer;
    if (!dataGatherer) {
        LOG_ERROR(<< "NULL data gatherer");
        return nullptr;
    }
    return new CCountingModel(this->modelParams(), dataGatherer,
                              this->interimBucketCorrector());
}

CAnomalyDetectorModel*
CCountingModelFactory::makeModel(const SModelInitializationData& initData,
                                 core::CStateRestoreTraverser& traverser) const {
    TDataGathererPtr dataGatherer = initData.s_DataGatherer;
    if (!dataGatherer) {
        LOG_ERROR(<< "NULL data gatherer");
        return nullptr;
    }
    return new CCountingModel(this->modelParams(), dataGatherer,
                              this->interimBucketCorrector(), traverser);
}

CDataGatherer*
CCountingModelFactory::makeDataGatherer(const SGathererInitializationData& initData) const {
    return new CDataGatherer(model_t::E_EventRate, m_SummaryMode, this->modelParams(),
                             m_SummaryCountFieldName, initData.s_PartitionFieldValue,
                             m_PersonFieldName, EMPTY_STRING, EMPTY_STRING, {},
                             this->searchKey(), m_Features, initData.s_StartTime, 0);
}

CDataGatherer*
CCountingModelFactory::makeDataGatherer(const std::string& partitionFieldValue,
                                        core::CStateRestoreTraverser& traverser) const {
    return new CDataGatherer(model_t::E_EventRate, m_SummaryMode,
                             this->modelParams(), m_SummaryCountFieldName,
                             partitionFieldValue, m_PersonFieldName, EMPTY_STRING,
                             EMPTY_STRING, {}, this->searchKey(), traverser);
}

CCountingModelFactory::TPriorPtr
CCountingModelFactory::defaultPrior(model_t::EFeature /*feature*/,
                                    const SModelParams& /*params*/) const {
    return std::make_unique<maths::CConstantPrior>();
}

CCountingModelFactory::TMultivariatePriorUPtr
CCountingModelFactory::defaultMultivariatePrior(model_t::EFeature feature,
                                                const SModelParams& /*params*/) const {
    return std::make_unique<maths::CMultivariateConstantPrior>(model_t::dimension(feature));
}

CCountingModelFactory::TMultivariatePriorUPtr
CCountingModelFactory::defaultCorrelatePrior(model_t::EFeature /*feature*/,
                                             const SModelParams& /*params*/) const {
    return std::make_unique<maths::CMultivariateConstantPrior>(2);
}

const CSearchKey& CCountingModelFactory::searchKey() const {
    if (!m_SearchKeyCache) {
        m_SearchKeyCache.emplace(m_Identifier, function_t::function(m_Features),
                                 m_UseNull, this->modelParams().s_ExcludeFrequent,
                                 "", m_PersonFieldName, "", m_PartitionFieldName);
    }
    return *m_SearchKeyCache;
}

bool CCountingModelFactory::isSimpleCount() const {
    return CSearchKey::isSimpleCount(function_t::function(m_Features), m_PersonFieldName);
}

model_t::ESummaryMode CCountingModelFactory::summaryMode() const {
    return m_SummaryMode;
}

maths_t::EDataType CCountingModelFactory::dataType() const {
    return maths_t::E_IntegerData;
}

void CCountingModelFactory::identifier(int identifier) {
    m_Identifier = identifier;
    m_SearchKeyCache.reset();
}

void CCountingModelFactory::fieldNames(const std::string& partitionFieldName,
                                       const std::string& /*overFieldName*/,
                                       const std::string& byFieldName,
                                       const std::string& /*valueFieldName*/,
                                       const TStrVec& /*influenceFieldNames*/) {
    m_PartitionFieldName = partitionFieldName;
    m_PersonFieldName = byFieldName;
    m_SearchKeyCache.reset();
}

void CCountingModelFactory::useNull(bool useNull) {
    m_UseNull = useNull;
    m_SearchKeyCache.reset();
}

void CCountingModelFactory::features(const TFeatureVec& features) {
    m_Features = features;
    m_SearchKeyCache.reset();
}

CCountingModelFactory::TStrCRefVec CCountingModelFactory::partitioningFields() const {
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
double CCountingModelFactory::minimumSeasonalVarianceScale() const {
    // unused, return something
    return 0.0;
}
}
}
