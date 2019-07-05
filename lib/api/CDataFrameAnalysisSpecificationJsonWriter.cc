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

void CDataFrameAnalysisSpecificationJsonWriter::write(std::size_t rows,
                                                      std::size_t cols,
                                                      std::size_t memoryLimit,
                                                      std::size_t numberThreads,
                                                      const std::string& temporaryDirectory,
                                                      const std::string& resultsField,
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
    write(rows, cols, memoryLimit, numberThreads, temporaryDirectory, resultsField,
          diskUsageAllowed, analysisName, analysisParametersDoc, writer);
}

void CDataFrameAnalysisSpecificationJsonWriter::write(std::size_t rows,
                                                      std::size_t cols,
                                                      std::size_t memoryLimit,
                                                      std::size_t numberThreads,
                                                      const std::string& temporaryDirectory,
                                                      const std::string& resultsField,
                                                      bool diskUsageAllowed,
                                                      const std::string& analysisName,
                                                      const rapidjson::Document& analysisParametersDocument,
                                                      TRapidJsonLineWriter& writer) {

    writer.StartObject();

    writer.String(CDataFrameAnalysisSpecification::ROWS);
    writer.Uint64(rows);

    writer.String(CDataFrameAnalysisSpecification::COLS);
    writer.Uint64(cols);

    writer.String(CDataFrameAnalysisSpecification::MEMORY_LIMIT);
    writer.Uint64(memoryLimit);

    writer.String(CDataFrameAnalysisSpecification::THREADS);
    writer.Uint64(numberThreads);

    writer.String(CDataFrameAnalysisSpecification::TEMPORARY_DIRECTORY);
    writer.String(temporaryDirectory);

    writer.String(CDataFrameAnalysisSpecification::RESULTS_FIELD);
    writer.String(resultsField);

    writer.String(CDataFrameAnalysisSpecification::DISK_USAGE_ALLOWED);
    writer.Bool(diskUsageAllowed);

    writer.String(CDataFrameAnalysisSpecification::ANALYSIS);
    writer.StartObject();
    writer.String(CDataFrameAnalysisSpecification::NAME);
    writer.String(analysisName);

    // if no parameters are specified, parameters document has Null as its root element
    if (analysisParametersDocument.IsNull() == false) {
        if (analysisParametersDocument.IsObject()) {
            writer.String(CDataFrameAnalysisSpecification::PARAMETERS);
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
CDataFrameAnalysisSpecificationJsonWriter::jsonString(size_t rows,
                                                      size_t cols,
                                                      size_t memoryLimit,
                                                      size_t numberThreads,
                                                      bool diskUsageAllowed,
                                                      const std::string& tempDir,
                                                      const std::string& resultField,
                                                      const std::string& analysisName,
                                                      const std::string& analysisParameters) {
    rapidjson::StringBuffer stringBuffer;
    api::CDataFrameAnalysisSpecificationJsonWriter::TRapidJsonLineWriter writer;
    writer.Reset(stringBuffer);

    write(rows, cols, memoryLimit, numberThreads, tempDir, resultField,
          diskUsageAllowed, analysisName, analysisParameters, writer);

    return stringBuffer.GetString();
}
}
}