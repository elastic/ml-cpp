/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CLoggerThrottler.h>

#include <boost/test/unit_test.hpp>

#include <chrono>
#include <thread>
#include <tuple>

BOOST_AUTO_TEST_SUITE(CLoggerThrottlerTest)

using namespace ml;

BOOST_AUTO_TEST_CASE(testThreadSafety) {

    std::size_t logged1[]{0, 0, 0, 0};
    std::thread t1{[&] {
        for (std::size_t i = 0; i < 100; ++i) {
            logged1[0] += core::CLoggerThrottler::instance().skip("a", 290).second ? 0 : 1;
            logged1[1] += core::CLoggerThrottler::instance().skip("a", 382).second ? 0 : 1;
            logged1[2] += core::CLoggerThrottler::instance().skip("b", 21).second ? 0 : 1;
            logged1[3] += core::CLoggerThrottler::instance().skip("b", 12).second ? 0 : 1;
        }
    }};

    std::size_t logged2[]{0, 0, 0, 0};
    std::thread t2{[&] {
        for (std::size_t i = 0; i < 100; ++i) {
            logged2[0] += core::CLoggerThrottler::instance().skip("a", 290).second ? 0 : 1;
            logged2[1] += core::CLoggerThrottler::instance().skip("a", 382).second ? 0 : 1;
            logged2[2] += core::CLoggerThrottler::instance().skip("b", 21).second ? 0 : 1;
            logged2[3] += core::CLoggerThrottler::instance().skip("b", 12).second ? 0 : 1;
        }
    }};

    t1.join();
    t2.join();

    for (std::size_t i = 0; i < 4; ++i) {
        BOOST_REQUIRE_EQUAL(1, logged1[i] + logged2[i]);
    }
}

BOOST_AUTO_TEST_CASE(testThrottling) {

    core::CLoggerThrottler::instance().minimumLogIntervalMs(1000); // 1s

    std::size_t logged[]{0, 0};
    std::size_t counts[]{0, 0};
    bool skip;
    std::size_t count;

    for (std::size_t i = 0; i < 100; ++i) {
        // Make sure we wait long enough at the end to see all the messages.
        std::this_thread::sleep_for(i == 99 ? std::chrono::milliseconds{1050}
                                            : std::chrono::milliseconds{50});
        std::tie(count, skip) = core::CLoggerThrottler::instance().skip(__FILE__, __LINE__);
        logged[0] += skip ? 0 : 1;
        counts[0] += skip ? 0 : count;
        std::tie(count, skip) = core::CLoggerThrottler::instance().skip(__FILE__, __LINE__);
        logged[1] += skip ? 0 : 1;
        counts[1] += skip ? 0 : count;
    }

    BOOST_REQUIRE(logged[0] >= 4);
    BOOST_REQUIRE(logged[0] >= 4);
    // Allow for long stalls running the tests.
    BOOST_REQUIRE(logged[1] < 10);
    BOOST_REQUIRE(logged[1] < 10);

    BOOST_REQUIRE_EQUAL(100, counts[0]);
    BOOST_REQUIRE_EQUAL(100, counts[1]);
}

BOOST_AUTO_TEST_SUITE_END()
