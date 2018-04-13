/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_normalize_CCmdLineParser_h
#define INCLUDED_ml_normalize_CCmdLineParser_h

#include <core/CoreTypes.h>

#include <string>
#include <vector>

namespace ml {
namespace normalize {

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
    typedef std::vector<std::string> TStrVec;

public:
    //! Parse the arguments and return options if appropriate.
    static bool parse(int argc,
                      const char* const* argv,
                      std::string& modelConfigFile,
                      std::string& logProperties,
                      std::string& logPipe,
                      core_t::TTime& bucketSpan,
                      bool& lengthEncodedInput,
                      std::string& inputFileName,
                      bool& isInputFileNamedPipe,
                      std::string& outputFileName,
                      bool& isOutputFileNamedPipe,
                      std::string& quantilesState,
                      bool& deleteStateFiles,
                      bool& writeCsv,
                      bool& perPartitionNormalization);

private:
    static const std::string DESCRIPTION;
};
}
}

#endif // INCLUDED_ml_normalize_CCmdLineParser_h
