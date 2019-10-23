/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CMonotonicTime.h>
#include <core/CSleep.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CMonotonicTimeTest)

BOOST_AUTO_TEST_CASE(testMilliseconds) {
    ml::core::CMonotonicTime monoTime;

    uint64_t start(monoTime.milliseconds());

    ml::core::CSleep::sleep(1000);

    uint64_t end(monoTime.milliseconds());

    uint64_t diff(end - start);
    LOG_DEBUG(<< "During 1 second the monotonic millisecond timer advanced by "
              << diff << " milliseconds");

    // Allow 10% margin of error - this is as much for the sleep as the timer
    BOOST_TEST_REQUIRE(diff > 900);
    BOOST_TEST_REQUIRE(diff < 1100);
}

BOOST_AUTO_TEST_CASE(testNanoseconds) {
    ml::core::CMonotonicTime monoTime;

    uint64_t start(monoTime.nanoseconds());

    ml::core::CSleep::sleep(1000);

    uint64_t end(monoTime.nanoseconds());

    uint64_t diff(end - start);
    LOG_DEBUG(<< "During 1 second the monotonic nanosecond timer advanced by "
              << diff << " nanoseconds");

    // Allow 10% margin of error - this is as much for the sleep as the timer
    BOOST_TEST_REQUIRE(diff > 900000000);
    BOOST_TEST_REQUIRE(diff < 1100000000);
}

BOOST_AUTO_TEST_SUITE_END()
