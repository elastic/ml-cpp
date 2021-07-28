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
                    ml::api::CAnomalyJobConfig& jobConfig,
                    ml::model::CAnomalyDetectorModelConfig& modelConfig,
                    ml::core::CJsonOutputStreamWrapper& outputBuffer,
                    const TPersistCompleteFunc& persistCompleteFunc = TPersistCompleteFunc(),
                    ml::api::CPersistenceManager* persistenceManager = nullptr,
                    ml::core_t::TTime maxQuantileInterval = -1,
                    const std::string& timeFieldName = DEFAULT_TIME_FIELD_NAME,
                    const std::string& timeFieldFormat = EMPTY_STRING,
                    std::size_t maxAnomalyRecords = 0u);

    //! Bring base class overload of handleRecord() into scope
    using CAnomalyJob::handleRecord;

    bool handleRecord(const TStrStrUMap& dataRowFields) {
        return this->handleRecord(dataRowFields, TOptionalTime{});
    }

    static ml::api::CAnomalyJobConfig
    makeSimpleJobConfig(const std::string& functionName,
                        const std::string& fieldName,
                        const std::string& byFieldName,
                        const std::string& overFieldName,
                        const std::string& partitionFieldName,
                        const TStrVec& influencers = {},
                        const std::string& summaryCountFieldName = "");
};

#endif // INCLUDED_CTestAnomalyJob_h
