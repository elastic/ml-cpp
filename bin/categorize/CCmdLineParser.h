/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_categorize_CCmdLineParser_h
#define INCLUDED_ml_categorize_CCmdLineParser_h

#include <core/CoreTypes.h>

#include <string>

namespace ml {
namespace categorize {

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
    //! Parse the arguments and return options if appropriate.  Unamed
    //! options are placed in a vector for further processing/validation
    //! later on by the api::CFieldConfig class.
    static bool parse(int argc,
                      const char* const* argv,
                      std::string& limitConfigFile,
                      std::string& jobId,
                      std::string& logProperties,
                      std::string& logPipe,
                      char& delimiter,
                      bool& lengthEncodedInput,
                      core_t::TTime& persistInterval,
                      std::string& inputFileName,
                      bool& isInputFileNamedPipe,
                      std::string& outputFileName,
                      bool& isOutputFileNamedPipe,
                      std::string& restoreFileName,
                      bool& isRestoreFileNamedPipe,
                      std::string& persistFileName,
                      bool& isPersistFileNamedPipe,
                      bool& isPersistInForeground,
                      std::string& categorizationFieldName);

private:
    static const std::string DESCRIPTION;
};
}
}

#endif // INCLUDED_ml_categorize_CCmdLineParser_h
