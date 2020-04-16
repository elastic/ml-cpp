/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CTestAnomalyJob.h"

CTestAnomalyJob::CTestAnomalyJob(const std::string& jobId,
                                 ml::model::CLimits& limits,
                                 ml::api::CFieldConfig& fieldConfig,
                                 ml::model::CAnomalyDetectorModelConfig& modelConfig,
                                 ml::core::CJsonOutputStreamWrapper& outputBuffer,
                                 const TPersistCompleteFunc& persistCompleteFunc,
                                 ml::api::CPersistenceManager* persistenceManager,
                                 ml::core_t::TTime maxQuantileInterval,
                                 const std::string& timeFieldName,
                                 const std::string& timeFieldFormat,
                                 std::size_t maxAnomalyRecords)
    : ml::api::CAnomalyJob(jobId,
                           limits,
                           fieldConfig,
                           modelConfig,
                           outputBuffer,
                           persistCompleteFunc,
                           persistenceManager,
                           maxQuantileInterval,
                           timeFieldName,
                           timeFieldFormat,
                           maxAnomalyRecords) {
}
