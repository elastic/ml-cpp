/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CTestFieldDataCategorizer_h
#define INCLUDED_CTestFieldDataCategorizer_h

#include <api/CFieldDataCategorizer.h>

//! \brief
//! A test convenience wrapper for the ML categorizer.
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
class CTestFieldDataCategorizer : public ml::api::CFieldDataCategorizer {

public:
    CTestFieldDataCategorizer(const std::string& jobId,
                              const ml::api::CFieldConfig& config,
                              ml::model::CLimits& limits,
                              ml::api::CDataProcessor* chainedProcessor,
                              ml::core::CJsonOutputStreamWrapper& outputStream,
                              ml::api::CPersistenceManager* persistenceManager = nullptr,
                              bool stopCategorizationOnWarnStatus = false);

    CTestFieldDataCategorizer(const std::string& jobId,
                              const ml::api::CFieldConfig& config,
                              ml::model::CLimits& limits,
                              const std::string& timeFieldName,
                              const std::string& timeFieldFormat,
                              ml::api::CDataProcessor* chainedProcessor,
                              ml::core::CJsonOutputStreamWrapper& outputStream,
                              ml::api::CPersistenceManager* persistenceManager = nullptr,
                              bool stopCategorizationOnWarnStatus = false);

    //! Bring base class overload of handleRecord() into scope
    using CFieldDataCategorizer::handleRecord;

    bool handleRecord(const TStrStrUMap& dataRowFields) {
        return this->handleRecord(dataRowFields, TOptionalTime{});
    }
};

#endif // INCLUDED_CTestFieldDataCategorizer_h
