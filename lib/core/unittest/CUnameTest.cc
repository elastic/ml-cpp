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
#include <core/CUname.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CUnameTest)

BOOST_AUTO_TEST_CASE(testUname) {
    LOG_DEBUG(<< ml::core::CUname::sysName());
    BOOST_TEST_REQUIRE(ml::core::CUname::sysName().length() > 0);
    LOG_DEBUG(<< ml::core::CUname::nodeName());
    BOOST_TEST_REQUIRE(ml::core::CUname::nodeName().length() > 0);
    LOG_DEBUG(<< ml::core::CUname::release());
    BOOST_TEST_REQUIRE(ml::core::CUname::release().length() > 0);
    LOG_DEBUG(<< ml::core::CUname::version());
    BOOST_TEST_REQUIRE(ml::core::CUname::version().length() > 0);
    LOG_DEBUG(<< ml::core::CUname::machine());
    BOOST_TEST_REQUIRE(ml::core::CUname::machine().length() > 0);
    LOG_DEBUG(<< ml::core::CUname::all());
    BOOST_TEST_REQUIRE(ml::core::CUname::all().length() > 0);
    LOG_DEBUG(<< ml::core::CUname::mlPlatform());
    BOOST_TEST_REQUIRE(ml::core::CUname::mlPlatform().length() > 0);
    LOG_DEBUG(<< ml::core::CUname::mlOsVer());
    BOOST_TEST_REQUIRE(ml::core::CUname::mlOsVer().length() > 0);
}

BOOST_AUTO_TEST_SUITE_END()
