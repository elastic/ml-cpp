/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
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
    parse(int argc, const char* const* argv, std::string& configFile, std::string& syslogLine);

private:
    static const std::string DESCRIPTION;
};
}
}

#endif // INCLUDED_ml_date_time_tester_CCmdLineParser_h
