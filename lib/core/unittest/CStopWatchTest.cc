/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CStopWatchTest.h"

#include <core/CLogger.h>
#include <core/CSleep.h>
#include <core/CStopWatch.h>

#include <stdint.h>

CppUnit::Test* CStopWatchTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CStopWatchTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CStopWatchTest>(
        "CStopWatchTest::testStopWatch", &CStopWatchTest::testStopWatch));

    return suiteOfTests;
}

void CStopWatchTest::testStopWatch() {
    ml::core::CStopWatch stopWatch;

    LOG_DEBUG(<< "About to start stop watch test");

    stopWatch.start();

    ml::core::CSleep::sleep(5500);

    uint64_t elapsed(stopWatch.lap());

    LOG_DEBUG(<< "After a 5.5 second wait, the stop watch reads " << elapsed << " milliseconds");

    // Elapsed time should be between 5.4 and 5.6 seconds
    CPPUNIT_ASSERT(elapsed >= 5400);
    CPPUNIT_ASSERT(elapsed <= 5600);

    ml::core::CSleep::sleep(3500);

    elapsed = stopWatch.stop();

    LOG_DEBUG(<< "After a further 3.5 second wait, the stop watch reads "
              << elapsed << " milliseconds");

    // Elapsed time should be between 8.9 and 9.1 seconds
    CPPUNIT_ASSERT(elapsed >= 8900);
    CPPUNIT_ASSERT(elapsed <= 9100);

    // The stop watch should not count this time, as it's stopped
    ml::core::CSleep::sleep(2000);

    stopWatch.start();

    ml::core::CSleep::sleep(500);

    elapsed = stopWatch.stop();

    LOG_DEBUG(<< "After a further 2 second wait with the stop watch stopped, "
                 "followed by a 0.5 second wait with the stop watch running, "
                 "it "
                 "reads "
              << elapsed << " milliseconds");

    // Elapsed time should be between 9.4 and 9.6 seconds
    CPPUNIT_ASSERT(elapsed >= 9400);
    CPPUNIT_ASSERT(elapsed <= 9600);
}
