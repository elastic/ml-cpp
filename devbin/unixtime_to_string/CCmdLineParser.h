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
#ifndef INCLUDED_ml_date_time_tester_CCmdLineParser_h
#define INCLUDED_ml_date_time_tester_CCmdLineParser_h

#include <string>

namespace ml {
namespace syslogparsertester {

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
    //! Parse the arguments. ONLY return true if configFile and dateTime
    //! are defined.
    static bool
    parse(int argc, const char *const *argv, std::string &configFile, std::string &syslogLine);

private:
    static const std::string DESCRIPTION;
};
}
}

#endif// INCLUDED_ml_date_time_tester_CCmdLineParser_h
