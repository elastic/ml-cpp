/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CTestFieldDataCategorizer.h"

CTestFieldDataCategorizer::CTestFieldDataCategorizer(
    const std::string& jobId,
    const ml::api::CAnomalyJobConfig::CAnalysisConfig& config,
    ml::model::CLimits& limits,
    ml::api::CDataProcessor* chainedProcessor,
    ml::core::CJsonOutputStreamWrapper& outputStream,
    ml::api::CPersistenceManager* persistenceManager,
    bool stopCategorizationOnWarnStatus)
    : ml::api::CFieldDataCategorizer{jobId,
                                     config,
                                     limits,
                                     std::string(),
                                     std::string(),
                                     chainedProcessor,
                                     outputStream,
                                     persistenceManager,
                                     stopCategorizationOnWarnStatus} {
}

CTestFieldDataCategorizer::CTestFieldDataCategorizer(
    const std::string& jobId,
    const ml::api::CAnomalyJobConfig::CAnalysisConfig& config,
    ml::model::CLimits& limits,
    const std::string& timeFieldName,
    const std::string& timeFieldFormat,
    ml::api::CDataProcessor* chainedProcessor,
    ml::core::CJsonOutputStreamWrapper& outputStream,
    ml::api::CPersistenceManager* persistenceManager,
    bool stopCategorizationOnWarnStatus)
    : ml::api::CFieldDataCategorizer{jobId,
                                     config,
                                     limits,
                                     timeFieldName,
                                     timeFieldFormat,
                                     chainedProcessor,
                                     outputStream,
                                     persistenceManager,
                                     stopCategorizationOnWarnStatus} {
}
