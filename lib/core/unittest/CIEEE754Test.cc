/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CIEEE754.h>
#include <core/CLogger.h>

#include <boost/test/unit_test.hpp>

#include <cmath>
#include <iomanip>
#include <sstream>

BOOST_AUTO_TEST_SUITE(CIEEE754Test)

using namespace ml;
using namespace core;

BOOST_AUTO_TEST_CASE(testRound) {
    {
        // Check it matches float precision.
        double test1 = 0.049999998;
        std::ostringstream o1;
        o1 << std::setprecision(10) << static_cast<float>(test1);
        std::ostringstream o2;
        o2 << std::setprecision(10) << CIEEE754::round(test1, CIEEE754::E_SinglePrecision);
        LOG_DEBUG(<< "test1 " << o1.str() << " " << o2.str());
        BOOST_REQUIRE_EQUAL(o1.str(), o2.str());
    }
    {
        // Check it matches float precision.
        double test2 = 0.0499999998;
        std::ostringstream o1;
        o1 << std::setprecision(10) << static_cast<float>(test2);
        std::ostringstream o2;
        o2 << std::setprecision(10) << CIEEE754::round(test2, CIEEE754::E_SinglePrecision);
        LOG_DEBUG(<< "test2 " << o1.str() << " " << o2.str());
        BOOST_REQUIRE_EQUAL(o1.str(), o2.str());
    }
    {
        // Check rounding away from zero.
        double test3 = 0.5 - 0.5 / static_cast<double>(1 << 25);
        std::ostringstream o1;
        o1 << std::setprecision(10) << 0.5;
        std::ostringstream o2;
        o2 << std::setprecision(10) << CIEEE754::round(test3, CIEEE754::E_SinglePrecision);
        LOG_DEBUG(<< "test3 " << o1.str() << " " << o2.str());
        BOOST_REQUIRE_EQUAL(o1.str(), o2.str());
    }
    {
        // Check rounding away from zero.
        double test4 = -(0.5 - 0.5 / static_cast<double>(1 << 25));
        std::ostringstream o1;
        o1 << std::setprecision(10) << -0.5;
        std::ostringstream o2;
        o2 << std::setprecision(10) << CIEEE754::round(test4, CIEEE754::E_SinglePrecision);
        LOG_DEBUG(<< "test4 " << o1.str() << " " << o2.str());
        BOOST_REQUIRE_EQUAL(o1.str(), o2.str());
    }
    {
        // Check rounding for very large numbers.
        double test5 = 0.49999998 * std::pow(2.0, 1023.0);
        std::ostringstream o1;
        o1 << std::setprecision(10) << 0.4999999702 * std::pow(2, 1023.0);
        std::ostringstream o2;
        o2 << std::setprecision(10) << CIEEE754::round(test5, CIEEE754::E_SinglePrecision);
        LOG_DEBUG(<< "test5 " << o1.str() << " " << o2.str());
        BOOST_REQUIRE_EQUAL(o1.str(), o2.str());
    }
    {
        // Check rounding for very large numbers.
        double test6 = 0.499999998 * std::pow(2.0, 1023.0);
        std::ostringstream o1;
        o1 << std::setprecision(10) << std::pow(2.0, 1022.0);
        std::ostringstream o2;
        o2 << std::setprecision(10) << CIEEE754::round(test6, CIEEE754::E_SinglePrecision);
        LOG_DEBUG(<< "test6 " << o1.str() << " " << o2.str());
        BOOST_REQUIRE_EQUAL(o1.str(), o2.str());
    }
}

BOOST_AUTO_TEST_SUITE_END()
