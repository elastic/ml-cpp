/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2018 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
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
                      const char *const *argv,
                      std::string &limitConfigFile,
                      std::string &jobId,
                      std::string &logProperties,
                      std::string &logPipe,
                      char &delimiter,
                      bool &lengthEncodedInput,
                      core_t::TTime &persistInterval,
                      std::string &inputFileName,
                      bool &isInputFileNamedPipe,
                      std::string &outputFileName,
                      bool &isOutputFileNamedPipe,
                      std::string &restoreFileName,
                      bool &isRestoreFileNamedPipe,
                      std::string &persistFileName,
                      bool &isPersistFileNamedPipe,
                      std::string &categorizationFieldName);

private:
    static const std::string DESCRIPTION;
};
}
}

#endif// INCLUDED_ml_categorize_CCmdLineParser_h
