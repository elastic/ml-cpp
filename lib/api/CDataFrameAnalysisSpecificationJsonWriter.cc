/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataFrameAnalysisSpecificationJsonWriter.h>

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
                                                      const TStrVec& categoricalFields,
                                                      bool diskUsageAllowed,
                                                      const std::string& analysisName,
                                                      const std::string& analysisParameters,
                                                      TRapidJsonLineWriter& writer) {
    rapidjson::Document analysisParametersDoc;
    if (analysisParameters.empty() == false) {
        analysisParametersDoc.Parse(analysisParameters);
        if (analysisParametersDoc.GetParseError()) {
            HANDLE_FATAL(<< "Input error: analysis parameters " << analysisParameters
                         << " cannot be parsed as json. Please report this problem.")
        }
    }
    write(jobId, rows, cols, memoryLimit, numberThreads, temporaryDirectory,
          resultsField, categoricalFields, diskUsageAllowed, analysisName,
          analysisParametersDoc, writer);
}

void CDataFrameAnalysisSpecificationJsonWriter::write(const std::string& jobId,
                                                      std::size_t rows,
                                                      std::size_t cols,
                                                      std::size_t memoryLimit,
                                                      std::size_t numberThreads,
                                                      const std::string& temporaryDirectory,
                                                      const std::string& resultsField,
                                                      const TStrVec& categoricalFields,
                                                      bool diskUsageAllowed,
                                                      const std::string& analysisName,
                                                      const rapidjson::Document& analysisParametersDocument,
                                                      TRapidJsonLineWriter& writer) {
    writer.StartObject();

    writer.Key(CDataFrameAnalysisSpecification::JOB_ID);
    writer.String(jobId);

    writer.Key(CDataFrameAnalysisSpecification::ROWS);
    writer.Uint64(rows);

    writer.Key(CDataFrameAnalysisSpecification::COLS);
    writer.Uint64(cols);

    writer.Key(CDataFrameAnalysisSpecification::MEMORY_LIMIT);
    writer.Uint64(memoryLimit);

    writer.Key(CDataFrameAnalysisSpecification::THREADS);
    writer.Uint64(numberThreads);

    writer.Key(CDataFrameAnalysisSpecification::TEMPORARY_DIRECTORY);
    writer.String(temporaryDirectory);

    writer.Key(CDataFrameAnalysisSpecification::RESULTS_FIELD);
    writer.String(resultsField);

    rapidjson::Value array(rapidjson::kArrayType);
    for (const auto& field : categoricalFields) {
        array.PushBack(rapidjson::Value(rapidjson::StringRef(field)),
                       writer.getRawAllocator());
    }
    writer.Key(CDataFrameAnalysisSpecification::CATEGORICAL_FIELD_NAMES);
    writer.write(array);

    writer.Key(CDataFrameAnalysisSpecification::DISK_USAGE_ALLOWED);
    writer.Bool(diskUsageAllowed);

    writer.Key(CDataFrameAnalysisSpecification::ANALYSIS);
    writer.StartObject();
    writer.Key(CDataFrameAnalysisSpecification::NAME);
    writer.String(analysisName);

    // if no parameters are specified, parameters document has Null as its root element
    if (analysisParametersDocument.IsNull() == false) {
        if (analysisParametersDocument.IsObject()) {
            writer.Key(CDataFrameAnalysisSpecification::PARAMETERS);
            writer.write(analysisParametersDocument);
        } else {
            HANDLE_FATAL(<< "Input error: analysis parameters suppose to "
                         << "contain an object as root node.")
        }
    }

    writer.EndObject();
    writer.EndObject();
    writer.Flush();
}

std::string
CDataFrameAnalysisSpecificationJsonWriter::jsonString(const std::string& jobId,
                                                      size_t rows,
                                                      size_t cols,
                                                      size_t memoryLimit,
                                                      size_t numberThreads,
                                                      const TStrVec& categoricalFields,
                                                      bool diskUsageAllowed,
                                                      const std::string& tempDir,
                                                      const std::string& resultField,
                                                      const std::string& analysisName,
                                                      const std::string& analysisParameters) {
    rapidjson::StringBuffer stringBuffer;
    api::CDataFrameAnalysisSpecificationJsonWriter::TRapidJsonLineWriter writer;
    writer.Reset(stringBuffer);

    write(jobId, rows, cols, memoryLimit, numberThreads, tempDir, resultField,
          categoricalFields, diskUsageAllowed, analysisName, analysisParameters, writer);

    return stringBuffer.GetString();
}
}
}
