/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CSystemCallFilterTest.h"

#include <core/CLogger.h>
#include <seccomp/CSystemCallFilter.h>

#include <cstdlib>
#include <string>

bool systemCall() {
    return std::system("hostname") == 0;
}

CppUnit::Test* CSystemCallFilterTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CSystemCallFilterTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CSystemCallFilterTest>(
        "CSystemCallFilterTest::testSystemCallFilter",
        &CSystemCallFilterTest::testSystemCallFilter));

    return suiteOfTests;
}

void CSystemCallFilterTest::testSystemCallFilter() {
    // Ensure actions are not prohibited before the
    // system call filters are applied
    CPPUNIT_ASSERT(systemCall());

    // Install the filter
    ml::seccomp::CSystemCallFilter filter;

    CPPUNIT_ASSERT_ASSERTION_FAIL_MESSAGE("Call std::system", CPPUNIT_ASSERT(systemCall()));
}
