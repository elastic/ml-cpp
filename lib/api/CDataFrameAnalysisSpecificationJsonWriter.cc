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

#include <api/CDataFrameAnalysisSpecificationJsonWriter.h>

#include <core/CDataFrame.h>

#include <api/CDataFrameAnalysisSpecification.h>

#include <iostream>

namespace ml {
namespace api {

void CDataFrameAnalysisSpecificationJsonWriter::write(const std::string& jobId,
                                                      std::size_t rows,
                                                      std::size_t cols,
                                                      std::size_t memoryLimit,
                                                      std::size_t numberThreads,
                                                      const std::string& temporaryDirectory,
                                                      const std::string& resultsField,
                                                      const std::string& missingFieldValue,
                                                      const TStrVec& categoricalFields,
                                                      bool diskUsageAllowed,
                                                      const std::string& analysisName,
                                                      const std::string& analysisParameters,
                                                      TBoostJsonLineWriter& writer) {
    json::value analysisParametersDoc;
    if (analysisParameters.empty() == false) {
        json::error_code ec;
        json::parser p;
        p.write(analysisParameters, ec);
        if (ec.failed()) {
            HANDLE_FATAL(<< "Input error: analysis parameters " << analysisParameters
                         << " cannot be parsed as json. Please report this problem.");
        }
        analysisParametersDoc = p.release();
    }
    LOG_DEBUG(<< "analysisParametersDoc: " << analysisParametersDoc);

    write(jobId, rows, cols, memoryLimit, numberThreads, temporaryDirectory,
          resultsField, missingFieldValue, categoricalFields, diskUsageAllowed,
          analysisName, analysisParametersDoc, writer);
}

void CDataFrameAnalysisSpecificationJsonWriter::write(const std::string& jobId,
                                                      std::size_t rows,
                                                      std::size_t cols,
                                                      std::size_t memoryLimit,
                                                      std::size_t numberThreads,
                                                      const std::string& temporaryDirectory,
                                                      const std::string& resultsField,
                                                      const std::string& missingFieldValue,
                                                      const TStrVec& categoricalFields,
                                                      bool diskUsageAllowed,
                                                      const std::string& analysisName,
                                                      const json::value& analysisParametersDocument,
                                                      TBoostJsonLineWriter& writer) {
    writer.onObjectBegin();

    writer.onKey(CDataFrameAnalysisSpecification::JOB_ID);
    writer.onString(jobId);

    writer.onKey(CDataFrameAnalysisSpecification::ROWS);
    writer.onUint64(rows);

    writer.onKey(CDataFrameAnalysisSpecification::COLS);
    writer.onUint64(cols);

    writer.onKey(CDataFrameAnalysisSpecification::MEMORY_LIMIT);
    writer.onUint64(memoryLimit);

    writer.onKey(CDataFrameAnalysisSpecification::THREADS);
    writer.onUint64(numberThreads);

    writer.onKey(CDataFrameAnalysisSpecification::TEMPORARY_DIRECTORY);
    writer.onString(temporaryDirectory);

    writer.onKey(CDataFrameAnalysisSpecification::RESULTS_FIELD);
    writer.onString(resultsField);

    if (missingFieldValue != core::CDataFrame::DEFAULT_MISSING_STRING) {
        writer.onKey(CDataFrameAnalysisSpecification::MISSING_FIELD_VALUE);
        writer.onString(missingFieldValue);
    }

    json::array array;
    for (const auto& field : categoricalFields) {
        array.push_back(json::value(field));
    }
    writer.onKey(CDataFrameAnalysisSpecification::CATEGORICAL_FIELD_NAMES);
    writer.write(array);

    writer.onKey(CDataFrameAnalysisSpecification::DISK_USAGE_ALLOWED);
    writer.onBool(diskUsageAllowed);

    writer.onKey(CDataFrameAnalysisSpecification::ANALYSIS);
    writer.onObjectBegin();
    writer.onKey(CDataFrameAnalysisSpecification::NAME);
    writer.onString(analysisName);

    // if no parameters are specified, parameters document has Null as its root element
    if (analysisParametersDocument.is_null() == false) {
        if (analysisParametersDocument.is_object()) {
            writer.onKey(CDataFrameAnalysisSpecification::PARAMETERS);
            writer.write(analysisParametersDocument);
        } else {
            HANDLE_FATAL(<< "Input error: analysis parameters suppose to "
                         << "contain an object as root node.");
        }
    }

    writer.onObjectEnd();
    writer.onObjectEnd();
    writer.flush();
}

std::string CDataFrameAnalysisSpecificationJsonWriter::jsonString(
    const std::string& jobId,
    std::size_t rows,
    std::size_t cols,
    std::size_t memoryLimit,
    std::size_t numberThreads,
    const std::string& missingFieldValue,
    const TStrVec& categoricalFields,
    bool diskUsageAllowed,
    const std::string& tempDir,
    const std::string& resultField,
    const std::string& analysisName,
    const std::string& analysisParameters) {

    std::ostringstream os;
    TBoostJsonLineWriter writer(os);

    write(jobId, rows, cols, memoryLimit, numberThreads, tempDir, resultField,
          missingFieldValue, categoricalFields, diskUsageAllowed, analysisName,
          analysisParameters, writer);

    return os.str();
}
}
}
