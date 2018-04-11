/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CWindowsErrorTest.h"

#include <core/CLogger.h>
#include <core/CWindowsError.h>

CppUnit::Test* CWindowsErrorTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CWindowsErrorTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CWindowsErrorTest>("CWindowsErrorTest::testErrors", &CWindowsErrorTest::testErrors));

    return suiteOfTests;
}

void CWindowsErrorTest::testErrors() {
    LOG_INFO(<< "Windows error 1 is : " << ml::core::CWindowsError(1));
    LOG_INFO(<< "Windows error 2 is : " << ml::core::CWindowsError(2));
    LOG_INFO(<< "Windows error 3 is : " << ml::core::CWindowsError(3));
    LOG_INFO(<< "Windows error 4 is : " << ml::core::CWindowsError(4));
    LOG_INFO(<< "Windows error 5 is : " << ml::core::CWindowsError(5));
    LOG_INFO(<< "Windows error 6 is : " << ml::core::CWindowsError(6));
}
