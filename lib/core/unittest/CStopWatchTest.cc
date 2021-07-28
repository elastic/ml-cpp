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

#include <core/CLogger.h>
#include <core/CStopWatch.h>

#include <chrono>
#include <cstdint>
#include <thread>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CStopWatchTest)

BOOST_AUTO_TEST_CASE(testStopWatch) {
    ml::core::CStopWatch stopWatch;

    LOG_DEBUG(<< "About to start stop watch test");

    stopWatch.start();

    std::this_thread::sleep_for(std::chrono::milliseconds(5500));

    std::uint64_t elapsed{stopWatch.lap()};

    LOG_DEBUG(<< "After a 5.5 second wait, the stop watch reads " << elapsed << " milliseconds");

    // Elapsed time should be between 5.4 and 5.7 seconds
    BOOST_TEST_REQUIRE(elapsed >= 5400);
    BOOST_TEST_REQUIRE(elapsed <= 5700);
    std::uint64_t previousElapsed{elapsed};

    std::this_thread::sleep_for(std::chrono::milliseconds(3500));

    elapsed = stopWatch.stop();

    LOG_DEBUG(<< "After a further 3.5 second wait, the stop watch reads "
              << elapsed << " milliseconds");

    // Elapsed time should have increased by between 3.4 and 3.7 seconds
    BOOST_TEST_REQUIRE(elapsed >= previousElapsed + 3400);
    BOOST_TEST_REQUIRE(elapsed <= previousElapsed + 3700);
    previousElapsed = elapsed;

    // The stop watch should not count this time, as it's stopped
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    stopWatch.start();

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    elapsed = stopWatch.stop();

    LOG_DEBUG(<< "After a further 2 second wait with the stop watch stopped, "
                 "followed by a 0.5 second wait with the stop watch running, it "
                 "reads "
              << elapsed << " milliseconds");

    // Elapsed time should have increased by between 0.4 and 0.7 seconds
    BOOST_TEST_REQUIRE(elapsed >= previousElapsed + 400);
    BOOST_TEST_REQUIRE(elapsed <= previousElapsed + 700);
}

BOOST_AUTO_TEST_SUITE_END()
