/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_test_CBoostTestXmlOutput_h
#define INCLUDED_ml_test_CBoostTestXmlOutput_h

#include <test/ImportExport.h>

#include <fstream>

namespace ml {
namespace test {

//! \brief
//! Add Boost.Test format XML output to default test output.
//!
//! DESCRIPTION:\n
//! A custom Boost.Test init function that unconditionally adds
//! Boost.Text format XML output (to boost_test_results.xml)
//! in addition to the default console output (or whatever this
//! has been overridden to via command line arguments).
//!
//! IMPLEMENTATION DECISIONS:\n
//! Always logs to the same output file, boost_test_results.xml,
//! which makes it clear which file a CI system needs to pick
//! up to render the test results.
//!
//! Always logs at the "test_suite" level, as this is the level
//! required for the Jenkins xUnit plugin to produce good output.
//!
class TEST_EXPORT CBoostTestXmlOutput {
public:
    CBoostTestXmlOutput() = delete;
    CBoostTestXmlOutput(const CBoostTestXmlOutput&) = delete;
    CBoostTestXmlOutput& operator=(const CBoostTestXmlOutput&) = delete;

    static bool init();

private:
    static std::ofstream ms_XmlOutputFile;
};
}
}

#endif // INCLUDED_ml_test_CBoostTestXmlOutput_h
