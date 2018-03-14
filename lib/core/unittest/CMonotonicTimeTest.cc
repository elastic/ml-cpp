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
#include "CMonotonicTimeTest.h"

#include <core/CLogger.h>
#include <core/CMonotonicTime.h>
#include <core/CSleep.h>

CppUnit::Test* CMonotonicTimeTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CMonotonicTimeTest");

    suiteOfTests->addTest(
        new CppUnit::TestCaller<CMonotonicTimeTest>("CMonotonicTimeTest::testMilliseconds",
                                                    &CMonotonicTimeTest::testMilliseconds));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CMonotonicTimeTest>("CMonotonicTimeTest::testNanoseconds",
                                                    &CMonotonicTimeTest::testNanoseconds));

    return suiteOfTests;
}

void CMonotonicTimeTest::testMilliseconds(void) {
    ml::core::CMonotonicTime monoTime;

    uint64_t start(monoTime.milliseconds());

    ml::core::CSleep::sleep(1000);

    uint64_t end(monoTime.milliseconds());

    uint64_t diff(end - start);
    LOG_DEBUG("During 1 second the monotonic millisecond timer advanced by " << diff
                                                                             << " milliseconds");

    // Allow 10% margin of error - this is as much for the sleep as the timer
    CPPUNIT_ASSERT(diff > 900);
    CPPUNIT_ASSERT(diff < 1100);
}

void CMonotonicTimeTest::testNanoseconds(void) {
    ml::core::CMonotonicTime monoTime;

    uint64_t start(monoTime.nanoseconds());

    ml::core::CSleep::sleep(1000);

    uint64_t end(monoTime.nanoseconds());

    uint64_t diff(end - start);
    LOG_DEBUG("During 1 second the monotonic nanosecond timer advanced by " << diff
                                                                            << " nanoseconds");

    // Allow 10% margin of error - this is as much for the sleep as the timer
    CPPUNIT_ASSERT(diff > 900000000);
    CPPUNIT_ASSERT(diff < 1100000000);
}
