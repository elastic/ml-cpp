/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
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
#ifndef INCLUDED_ml_controller_CCmdLineParser_h
#define INCLUDED_ml_controller_CCmdLineParser_h

#include <core/CoreTypes.h>

#include <string>
#include <vector>

namespace ml {
namespace controller {

//! \brief
//! Very simple command line parser.
//!
//! DESCRIPTION:\n
//! Very simple command line parser.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Put in a class rather than main to allow testing.
//!
class CCmdLineParser {
public:
    typedef std::vector<std::string> TStrVec;

public:
    //! Parse the arguments and return options if appropriate.
    static bool parse(int argc, const char* const* argv, std::string& jvmPidStr, std::string& logPipe, std::string& commandPipe);

private:
    static const std::string DESCRIPTION;
};
}
}

#endif // INCLUDED_ml_controller_CCmdLineParser_h
