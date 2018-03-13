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
#include "CWindowsErrorTest.h"

#include <core/CLogger.h>
#include <core/CWindowsError.h>

CppUnit::Test *CWindowsErrorTest::suite() {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CWindowsErrorTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CWindowsErrorTest>(
        "CWindowsErrorTest::testErrors", &CWindowsErrorTest::testErrors));

    return suiteOfTests;
}

void CWindowsErrorTest::testErrors(void) {
    LOG_INFO("Windows error 1 is : " << ml::core::CWindowsError(1));
    LOG_INFO("Windows error 2 is : " << ml::core::CWindowsError(2));
    LOG_INFO("Windows error 3 is : " << ml::core::CWindowsError(3));
    LOG_INFO("Windows error 4 is : " << ml::core::CWindowsError(4));
    LOG_INFO("Windows error 5 is : " << ml::core::CWindowsError(5));
    LOG_INFO("Windows error 6 is : " << ml::core::CWindowsError(6));
}
