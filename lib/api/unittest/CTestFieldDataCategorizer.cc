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
