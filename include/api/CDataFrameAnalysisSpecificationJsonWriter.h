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

#ifndef INCLUDED_ml_api_CDataFrameAnalysisSpecificationJsonWriter_h
#define INCLUDED_ml_api_CDataFrameAnalysisSpecificationJsonWriter_h

#include "core/CStreamWriter.h"
#include <core/CBoostJsonConcurrentLineWriter.h>
#include <core/CNonInstantiatable.h>

#include <api/ImportExport.h>

#include <string>
#include <vector>

namespace ml {
namespace api {

//! \brief
//! A static utility for writing data frame analysis specification in JSON.
class API_EXPORT CDataFrameAnalysisSpecificationJsonWriter : private core::CNonInstantiatable {
public:
    using TStrVec = std::vector<std::string>;
    using TBoostJsonLineWriter = core::CStreamWriter;

public:
    //! Writes the data frame analysis specification in JSON format.
    static void write(const std::string& jobId,
                      std::size_t rows,
                      std::size_t cols,
                      std::size_t memoryLimit,
                      std::size_t numberThreads,
                      const std::string& temporaryDirectory,
                      const std::string& resultsField,
                      const std::string& missingString,
                      const TStrVec& categoricalFields,
                      bool diskUsageAllowed,
                      const std::string& analysisName,
                      const json::value& analysisParametersDocument,
                      TBoostJsonLineWriter& writer);

    //! Writes the data frame analysis specification in JSON format.
    static void write(const std::string& jobId,
                      std::size_t rows,
                      std::size_t cols,
                      std::size_t memoryLimit,
                      std::size_t numberThreads,
                      const std::string& temporaryDirectory,
                      const std::string& resultsField,
                      const std::string& missingString,
                      const TStrVec& categoricalFields,
                      bool diskUsageAllowed,
                      const std::string& analysisName,
                      const std::string& analysisParameters,
                      TBoostJsonLineWriter& writer);

    //! Returns a string with the data frame analysis specification in JSON format.
    static std::string jsonString(const std::string& jobId,
                                  std::size_t rows,
                                  std::size_t cols,
                                  std::size_t memoryLimit,
                                  std::size_t numberThreads,
                                  const std::string& missingString,
                                  const TStrVec& categoricalFields,
                                  bool diskUsageAllowed,
                                  const std::string& tempDir,
                                  const std::string& resultField,
                                  const std::string& analysisName,
                                  const std::string& analysisParameters);
};
}
}
#endif //INCLUDED_ml_api_CDataFrameAnalysisSpecificationJsonWriter_h
