/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_autodetect_CCmdLineParser_h
#define INCLUDED_ml_autodetect_CCmdLineParser_h

#include <core/CoreTypes.h>

#include <string>
#include <vector>

namespace ml {
namespace autodetect {

//! \brief
//! Very simple command line parser.
//!
//! DESCRIPTION:\n
//! Very simple command line parser.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Put in a class rather than main to allow testing.
//! TODO make this generic.
//!
class CCmdLineParser {
public:
    using TStrVec = std::vector<std::string>;

public:
    //! Parse the arguments and return options if appropriate.  Unamed
    //! options are placed in a vector for further processing/validation
    //! later on by the api::CFieldConfig class.
    static bool parse(int argc,
                      const char* const* argv,
                      std::string& limitConfigFile,
                      std::string& modelConfigFile,
                      std::string& fieldConfigFile,
                      std::string& modelPlotConfigFile,
                      std::string& jobId,
                      std::string& logProperties,
                      std::string& logPipe,
                      core_t::TTime& bucketSpan,
                      core_t::TTime& latency,
                      std::string& summaryCountFieldName,
                      char& delimiter,
                      bool& lengthEncodedInput,
                      std::string& timeField,
                      std::string& timeFormat,
                      std::string& quantilesState,
                      bool& deleteStateFiles,
                      core_t::TTime& persistInterval,
                      core_t::TTime& maxQuantileInterval,
                      std::string& inputFileName,
                      bool& isInputFileNamedPipe,
                      std::string& outputFileName,
                      bool& isOutputFileNamedPipe,
                      std::string& restoreFileName,
                      bool& isRestoreFileNamedPipe,
                      std::string& persistFileName,
                      bool& isPersistFileNamedPipe,
                      size_t& maxAnomalyRecords,
                      bool& memoryUsage,
                      std::size_t& bucketResultsDelay,
                      bool& multivariateByFields,
                      std::string& multipleBucketspans,
                      bool& perPartitionNormalization,
                      TStrVec& clauseTokens);

private:
    static const std::string DESCRIPTION;
};
}
}

#endif // INCLUDED_ml_autodetect_CCmdLineParser_h
