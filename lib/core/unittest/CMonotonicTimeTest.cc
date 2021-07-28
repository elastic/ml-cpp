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
#include <core/CMonotonicTime.h>

#include <boost/test/unit_test.hpp>

#include <chrono>
#include <thread>

BOOST_AUTO_TEST_SUITE(CMonotonicTimeTest)

BOOST_AUTO_TEST_CASE(testMilliseconds) {
    ml::core::CMonotonicTime monoTime;

    uint64_t start(monoTime.milliseconds());

    std::this_thread::sleep_for(std::chrono::seconds(1));

    uint64_t end(monoTime.milliseconds());

    uint64_t diff(end - start);
    LOG_DEBUG(<< "During 1 second the monotonic millisecond timer advanced by "
              << diff << " milliseconds");

    // Allow 10% margin of error - this is as much for the sleep as the timer
    BOOST_TEST_REQUIRE(diff > 900);
    // Allow 20% margin of error - sleep seems to sleep too long under Jenkins
    // on Apple M1
    BOOST_TEST_REQUIRE(diff < 1200);
}

BOOST_AUTO_TEST_CASE(testNanoseconds) {
    ml::core::CMonotonicTime monoTime;

    uint64_t start(monoTime.nanoseconds());

    std::this_thread::sleep_for(std::chrono::seconds(1));

    uint64_t end(monoTime.nanoseconds());

    uint64_t diff(end - start);
    LOG_DEBUG(<< "During 1 second the monotonic nanosecond timer advanced by "
              << diff << " nanoseconds");

    // Allow 10% margin of error - this is as much for the sleep as the timer
    BOOST_TEST_REQUIRE(diff > 900000000);
    // Allow 20% margin of error - sleep seems to sleep too long under Jenkins
    // on Apple M1
    BOOST_TEST_REQUIRE(diff < 1200000000);
}

BOOST_AUTO_TEST_SUITE_END()
