/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CTestAnomalyJob.h"

#include <api/CAnomalyJobConfig.h>

CTestAnomalyJob::CTestAnomalyJob(const std::string& jobId,
                                 ml::model::CLimits& limits,
                                 ml::api::CAnomalyJobConfig& jobConfig,
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
                           jobConfig,
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

ml::api::CAnomalyJobConfig
CTestAnomalyJob::makeSimpleJobConfig(const std::string& functionName,
                                     const std::string& fieldName,
                                     const std::string& byFieldName,
                                     const std::string& overFieldName,
                                     const std::string& partitionFieldName,
                                     const ml::api::CDataProcessor::TStrVec& influencers,
                                     const std::string& summaryCountFieldName) {
    ml::api::CAnomalyJobConfig jobConfig;
    jobConfig.analysisConfig().addDetector(functionName, fieldName, byFieldName,
                                           overFieldName, partitionFieldName,
                                           influencers, summaryCountFieldName);
    return jobConfig;
}
