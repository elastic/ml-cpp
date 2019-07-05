/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataFrameAnalysisSpecification.h>
#include <api/CDataFrameAnalysisSpecificationJsonWriter.h>

namespace ml {
namespace api {

// TODO (valeriy) analysis parameter is missing
void CDataFrameAnalysisSpecificationJsonWriter::write(std::size_t rows,
                                                      std::size_t cols,
                                                      std::size_t memoryLimit,
                                                      std::size_t numberThreads,
                                                      const std::string& temporaryDirectory,
                                                      const std::string& resultsField,
                                                      bool diskUsageAllowed,
                                                      const std::string& analysis_name,
                                                      const std::string& analysis_parameters,
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
    writer.String(analysis_name);
    if (analysis_parameters.empty() == false) {
        writer.String(CDataFrameAnalysisSpecification::PARAMETERS);
        writer.String(analysis_parameters);
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
                                                      const std::string& analysis_name,
                                                      const std::string& analysis_parameters) {
    rapidjson::StringBuffer stringBuffer;
    api::CDataFrameAnalysisSpecificationJsonWriter::TRapidJsonLineWriter writer;
    writer.Reset(stringBuffer);

    write(rows, cols, memoryLimit, numberThreads, tempDir, resultField,
          diskUsageAllowed, analysis_name, analysis_parameters, writer);

    return stringBuffer.GetString();
}
}
}