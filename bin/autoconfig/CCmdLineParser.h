/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_autoconfig_CCmdLineParser_h
#define INCLUDED_ml_autoconfig_CCmdLineParser_h

#include <core/CoreTypes.h>

#include <string>
#include <vector>

namespace ml {
namespace autoconfig {

//! \brief Very simple command line parser.
//!
//! DESCRIPTION:\n
//! Very simple command line parser.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Put in a class rather than main to allow testing.
//!
class CCmdLineParser {
public:
    using TStrVec = std::vector<std::string>;

public:
    //! Parse the arguments and return options if appropriate.
    static bool parse(int argc,
                      const char* const* argv,
                      std::string& logProperties,
                      std::string& logPipe,
                      char& delimiter,
                      bool& lengthEncodedInput,
                      std::string& timeField,
                      std::string& timeFormat,
                      std::string& configFile,
                      std::string& inputFileName,
                      bool& isInputFileNamedPipe,
                      std::string& outputFileName,
                      bool& isOutputFileNamedPipe,
                      bool& verbose,
                      bool& writeDetectorConfigs);

private:
    static const std::string DESCRIPTION;
};
}
}

#endif // INCLUDED_ml_autoconfig_CCmdLineParser_h
