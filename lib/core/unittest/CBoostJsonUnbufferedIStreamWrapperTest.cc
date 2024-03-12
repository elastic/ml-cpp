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

#include <core/CBoostJsonUnbufferedIStreamWrapper.h>

#include <boost/test/unit_test.hpp>

#include <sstream>

BOOST_AUTO_TEST_SUITE(CBoostJsonUnbufferedIStreamWrapperTest)

BOOST_AUTO_TEST_CASE(testWrap) {

    std::istringstream strm{"abc"};
    ml::core::CBoostJsonUnbufferedIStreamWrapper wrapper{strm};

    BOOST_REQUIRE_EQUAL('a', wrapper.peek());
    BOOST_REQUIRE_EQUAL('a', wrapper.take());
    BOOST_REQUIRE_EQUAL(1, wrapper.tell());
    BOOST_REQUIRE_EQUAL('b', wrapper.take());
    BOOST_REQUIRE_EQUAL(2, wrapper.tell());
    BOOST_REQUIRE_EQUAL('c', wrapper.peek());
    BOOST_REQUIRE_EQUAL('c', wrapper.peek());
    BOOST_REQUIRE_EQUAL('c', wrapper.take());
    BOOST_REQUIRE_EQUAL(3, wrapper.tell());
    BOOST_REQUIRE_EQUAL('\0', wrapper.peek());
    BOOST_REQUIRE_EQUAL('\0', wrapper.take());
    BOOST_REQUIRE_EQUAL(3, wrapper.tell());
}

BOOST_AUTO_TEST_SUITE_END()
