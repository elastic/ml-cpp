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
#include "CTestAnomalyJob.h"

#include <api/CAnomalyJobConfig.h>

CTestAnomalyJob::CTestAnomalyJob(const std::string& jobId,
                                 ml::model::CLimits& limits,
                                 ml::api::CAnomalyJobConfig& jobConfig,
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
