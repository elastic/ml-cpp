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

BOOST_AUTO_TEST_SUITE(CLoggerThrottlerTest)

using namespace ml;

BOOST_AUTO_TEST_CASE(testThreadSafety) {

    std::size_t logged1[]{0, 0, 0, 0};
    std::thread t1{[&] {
        for (std::size_t i = 0; i < 100; ++i) {
            logged1[0] += core::CLoggerThrottler::instance().skip("a", 290) ? 0 : 1;
            logged1[1] += core::CLoggerThrottler::instance().skip("a", 382) ? 0 : 1;
            logged1[2] += core::CLoggerThrottler::instance().skip("b", 21) ? 0 : 1;
            logged1[3] += core::CLoggerThrottler::instance().skip("b", 12) ? 0 : 1;
        }
    }};

    std::size_t logged2[]{0, 0, 0, 0};
    std::thread t2{[&] {
        for (std::size_t i = 0; i < 100; ++i) {
            logged2[0] += core::CLoggerThrottler::instance().skip("a", 290) ? 0 : 1;
            logged2[1] += core::CLoggerThrottler::instance().skip("a", 382) ? 0 : 1;
            logged2[2] += core::CLoggerThrottler::instance().skip("b", 21) ? 0 : 1;
            logged2[3] += core::CLoggerThrottler::instance().skip("b", 12) ? 0 : 1;
        }
    }};

    t1.join();
    t2.join();

    for (std::size_t i = 0; i < 4; ++i) {
        BOOST_REQUIRE_EQUAL(1, logged1[i] + logged2[i]);
    }
}

BOOST_AUTO_TEST_CASE(testThrottling) {

    core::CLoggerThrottler::instance().minimumLogInterval(1000); // 1s

    std::size_t logged[]{0, 0};

    for (std::size_t i = 0; i < 100; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds{50});
        logged[0] += core::CLoggerThrottler::instance().skip(__FILE__, __LINE__) ? 0 : 1;
        logged[1] += core::CLoggerThrottler::instance().skip(__FILE__, __LINE__) ? 0 : 1;
    }

    LOG_DEBUG(<< core::CContainerPrinter::print(logged));
    BOOST_REQUIRE(logged[0] >= 4);
    BOOST_REQUIRE(logged[0] >= 4);
    BOOST_REQUIRE(logged[1] < 10); // Allow for long stalls running the tests.
    BOOST_REQUIRE(logged[1] < 10);
}

BOOST_AUTO_TEST_SUITE_END()
