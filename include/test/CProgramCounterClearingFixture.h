/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */
#ifndef INCLUDED_ml_test_CProgramCounterClearingFixture_h
#define INCLUDED_ml_test_CProgramCounterClearingFixture_h

#include <test/ImportExport.h>

namespace ml {
namespace test {

//! \brief
//! Test fixture that resets all program counters to zero
//! between tests.
//!
//! DESCRIPTION:\n
//! Program counters are implemented as a singleton object,
//! which means their values are remembered across unit tests.
//! Unit tests that care about the values of program counters
//! should use this test fixture to ensure they start from a
//! well known state.  Otherwise test results can vary depending
//! on whether a test is run alone or as part of a suite.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This functionality is required in multiple unit test suites,
//! hence it's in the test library.
//!
class TEST_EXPORT CProgramCounterClearingFixture {
public:
    CProgramCounterClearingFixture();
};
}
}

#endif // INCLUDED_ml_test_CProgramCounterClearingFixture_h
