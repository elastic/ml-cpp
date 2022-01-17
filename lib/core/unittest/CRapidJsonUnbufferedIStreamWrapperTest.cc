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

#include <core/CRapidJsonUnbufferedIStreamWrapper.h>

#include <boost/test/unit_test.hpp>

#include <sstream>

BOOST_AUTO_TEST_SUITE(CRapidJsonUnbufferedIStreamWrapperTest)

BOOST_AUTO_TEST_CASE(testWrap) {

    std::istringstream strm{"abc"};
    ml::core::CRapidJsonUnbufferedIStreamWrapper wrapper{strm};

    BOOST_REQUIRE_EQUAL(nullptr, wrapper.Peek4());
    BOOST_REQUIRE_EQUAL('a', wrapper.Peek());
    BOOST_REQUIRE_EQUAL('a', wrapper.Take());
    BOOST_REQUIRE_EQUAL(1, wrapper.Tell());
    BOOST_REQUIRE_EQUAL('b', wrapper.Take());
    BOOST_REQUIRE_EQUAL(2, wrapper.Tell());
    BOOST_REQUIRE_EQUAL('c', wrapper.Peek());
    BOOST_REQUIRE_EQUAL('c', wrapper.Peek());
    BOOST_REQUIRE_EQUAL('c', wrapper.Take());
    BOOST_REQUIRE_EQUAL(3, wrapper.Tell());
    BOOST_REQUIRE_EQUAL('\0', wrapper.Peek());
    BOOST_REQUIRE_EQUAL('\0', wrapper.Take());
    BOOST_REQUIRE_EQUAL(3, wrapper.Tell());
    BOOST_REQUIRE_EQUAL(nullptr, wrapper.Peek4());
}

BOOST_AUTO_TEST_SUITE_END()
