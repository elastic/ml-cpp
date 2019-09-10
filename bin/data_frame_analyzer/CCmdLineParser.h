/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_data_frame_analyzer_CCmdLineParser_h
#define INCLUDED_ml_data_frame_analyzer_CCmdLineParser_h

#include <string>

namespace ml {
namespace data_frame_analyzer {

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
    //! Parse the arguments and return options if appropriate.
    static bool parse(int argc,
                      const char* const* argv,
                      std::string& configFile,
                      std::string& jobId,
                      bool& memoryUsageEstimationOnly,
                      std::string& logProperties,
                      std::string& logPipe,
                      bool& lengthEncodedInput,
                      std::string& inputFileName,
                      bool& isInputFileNamedPipe,
                      std::string& outputFileName,
                      bool& isOutputFileNamedPipe,
                      std::string& restoreFileName,
                      bool& isRestoreFileNamedPipe,
                      std::string& persistFileName,
                      bool& isPersistFileNamedPipe);

private:
    static const std::string DESCRIPTION;
};
}
}

#endif // INCLUDED_ml_data_frame_analyzer_CCmdLineParser_h
