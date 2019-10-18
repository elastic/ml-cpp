/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_test_CBoostTestJUnitOutput_h
#define INCLUDED_ml_test_CBoostTestJUnitOutput_h

#include <test/ImportExport.h>

#include <fstream>

namespace ml {
namespace test {

//! \brief
//! Add JUnit output to default test output.
//!
//! DESCRIPTION:\n
//! A custom Boost.Test init function that unconditionally
//! adds JUnit output (to junit_results.xml) in addition to
//! the default console output (or whatever this has been
//! overridden to via command line arguments).
//!
//! IMPLEMENTATION DECISIONS:\n
//! Always logs to the same output file, junit_results.xml,
//! which makes it clear which file a CI system needs to
//! pick up to render the test results.
//!
TEST_EXPORT
class CBoostTestJUnitOutput {
public:
    CBoostTestJUnitOutput() = delete;
    CBoostTestJUnitOutput(const CBoostTestJUnitOutput&) = delete;
    CBoostTestJUnitOutput& operator=(const CBoostTestJUnitOutput&) = delete;

    static bool init();

private:
    static std::ofstream ms_JUnitOutputFile;
};
}
}

#endif // INCLUDED_ml_test_CBoostTestJUnitOutput_h
