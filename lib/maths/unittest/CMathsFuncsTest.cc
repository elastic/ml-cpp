/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CMathsFuncs.h>

#include <boost/test/unit_test.hpp>

#include <cmath>
#include <limits>
#include <vector>

using TDoubleVec = std::vector<double>;
BOOST_TEST_DONT_PRINT_LOG_VALUE(ml::maths::CMathsFuncs::CFiniteIterator<TDoubleVec::iterator>)

BOOST_AUTO_TEST_SUITE(CMathsFuncsTest)

using namespace ml;

namespace {
double zero() {
    return 0.0;
}
}

BOOST_AUTO_TEST_CASE(testIsNan) {
    BOOST_TEST_REQUIRE(!maths::CMathsFuncs::isNan(0.0));
    BOOST_TEST_REQUIRE(!maths::CMathsFuncs::isNan(1e7));
    BOOST_TEST_REQUIRE(!maths::CMathsFuncs::isNan(-1e17));
    BOOST_TEST_REQUIRE(!maths::CMathsFuncs::isNan(std::numeric_limits<double>::max()));
    BOOST_TEST_REQUIRE(!maths::CMathsFuncs::isNan(-std::numeric_limits<double>::max()));
    BOOST_TEST_REQUIRE(!maths::CMathsFuncs::isNan(std::numeric_limits<double>::min()));
    BOOST_TEST_REQUIRE(!maths::CMathsFuncs::isNan(-std::numeric_limits<double>::min()));
    BOOST_TEST_REQUIRE(maths::CMathsFuncs::isNan(zero() / zero()));
    BOOST_TEST_REQUIRE(maths::CMathsFuncs::isNan(1.0 / zero() - 2.0 / zero()));
}

BOOST_AUTO_TEST_CASE(testIsInf) {
    BOOST_TEST_REQUIRE(!maths::CMathsFuncs::isInf(0.0));
    BOOST_TEST_REQUIRE(!maths::CMathsFuncs::isInf(1.8738e7));
    BOOST_TEST_REQUIRE(!maths::CMathsFuncs::isInf(-1.376e17));
    BOOST_TEST_REQUIRE(!maths::CMathsFuncs::isInf(std::numeric_limits<double>::max()));
    BOOST_TEST_REQUIRE(!maths::CMathsFuncs::isInf(-std::numeric_limits<double>::max()));
    BOOST_TEST_REQUIRE(!maths::CMathsFuncs::isInf(std::numeric_limits<double>::min()));
    BOOST_TEST_REQUIRE(!maths::CMathsFuncs::isInf(-std::numeric_limits<double>::min()));
    BOOST_TEST_REQUIRE(maths::CMathsFuncs::isInf(1.0 / zero()));
    BOOST_TEST_REQUIRE(maths::CMathsFuncs::isInf(2.0 / zero()));
    BOOST_TEST_REQUIRE(maths::CMathsFuncs::isInf(std::log(zero())));
    BOOST_TEST_REQUIRE(maths::CMathsFuncs::isInf(std::exp(1.0 / zero())));
}

BOOST_AUTO_TEST_CASE(testIsFinite) {
    BOOST_TEST_REQUIRE(maths::CMathsFuncs::isFinite(0.0));
    BOOST_TEST_REQUIRE(maths::CMathsFuncs::isFinite(1.3e7));
    BOOST_TEST_REQUIRE(maths::CMathsFuncs::isFinite(-1.5368e17));
    BOOST_TEST_REQUIRE(maths::CMathsFuncs::isFinite(std::numeric_limits<double>::max()));
    BOOST_TEST_REQUIRE(maths::CMathsFuncs::isFinite(-std::numeric_limits<double>::max()));
    BOOST_TEST_REQUIRE(maths::CMathsFuncs::isFinite(std::numeric_limits<double>::min()));
    BOOST_TEST_REQUIRE(maths::CMathsFuncs::isFinite(-std::numeric_limits<double>::min()));
    BOOST_TEST_REQUIRE(!maths::CMathsFuncs::isFinite(1.0 / zero()));
    BOOST_TEST_REQUIRE(!maths::CMathsFuncs::isFinite(2.0 / zero()));
    BOOST_TEST_REQUIRE(!maths::CMathsFuncs::isFinite(std::log(zero())));
    BOOST_TEST_REQUIRE(!maths::CMathsFuncs::isFinite(zero() / zero()));
    BOOST_TEST_REQUIRE(!maths::CMathsFuncs::isFinite(1.0 / zero() - 2.0 / zero()));

    TDoubleVec test1;
    test1.push_back(2.0);
    test1.push_back(25.0);
    test1.push_back(-1e6);
    BOOST_TEST_REQUIRE(std::equal(test1.begin(), test1.end(),
                                  maths::CMathsFuncs::beginFinite(test1)));

    TDoubleVec test2;
    test2.push_back(zero() / zero());
    test2.push_back(2.0);
    test2.push_back(1.0 / zero());
    test2.push_back(zero() / zero());
    test2.push_back(3.0 / zero());
    test2.push_back(25.0);
    test2.push_back(-1e6);
    test2.push_back(zero() / zero());
    BOOST_TEST_REQUIRE(std::equal(test1.begin(), test1.end(),
                                  maths::CMathsFuncs::beginFinite(test2)));

    TDoubleVec test3;
    BOOST_TEST_REQUIRE(maths::CMathsFuncs::beginFinite(test3) ==
                       maths::CMathsFuncs::endFinite(test3));

    TDoubleVec test4;
    test4.push_back(zero() / zero());
    test4.push_back(1.0 / zero());
    test4.push_back(zero() / zero());
    BOOST_TEST_REQUIRE(maths::CMathsFuncs::beginFinite(test4) ==
                       maths::CMathsFuncs::endFinite(test4));
}

BOOST_AUTO_TEST_CASE(testFpStatus) {
    BOOST_REQUIRE_EQUAL(maths_t::E_FpNoErrors, maths::CMathsFuncs::fpStatus(3.8));
    BOOST_REQUIRE_EQUAL(maths_t::E_FpOverflowed,
                        maths::CMathsFuncs::fpStatus(1.0 / zero()));
    BOOST_REQUIRE_EQUAL(maths_t::E_FpFailed,
                        maths::CMathsFuncs::fpStatus(zero() / zero()));
}

BOOST_AUTO_TEST_SUITE_END()
