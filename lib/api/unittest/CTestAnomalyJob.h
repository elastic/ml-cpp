/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CTestAnomalyJob_h
#define INCLUDED_CTestAnomalyJob_h

#include <api/CAnomalyJob.h>

//! \brief
//! A test convenience wrapper for the ML anomaly  detector.
//!
//! DESCRIPTION:\n
//! Defaults some constructor arguments to make unit tests less
//! verbose.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The base class requires all constructor arguments be provided
//! to avoid accidental defaulting in production code, but for
//! unit tests defaults are often fine.
//!
class CTestAnomalyJob : public ml::api::CAnomalyJob {

public:
    CTestAnomalyJob(const std::string& jobId,
                    ml::model::CLimits& limits,
                    ml::api::CFieldConfig& fieldConfig,
                    ml::model::CAnomalyDetectorModelConfig& modelConfig,
                    ml::core::CJsonOutputStreamWrapper& outputBuffer,
                    const TPersistCompleteFunc& persistCompleteFunc = TPersistCompleteFunc(),
                    ml::api::CPersistenceManager* persistenceManager = nullptr,
                    ml::core_t::TTime maxQuantileInterval = -1,
                    const std::string& timeFieldName = DEFAULT_TIME_FIELD_NAME,
                    const std::string& timeFieldFormat = EMPTY_STRING,
                    std::size_t maxAnomalyRecords = 0u);
};

#endif // INCLUDED_CTestAnomalyJob_h
