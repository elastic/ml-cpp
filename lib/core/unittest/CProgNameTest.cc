/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CProgName.h>
#include <core/CRegex.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CProgNameTest)

BOOST_AUTO_TEST_CASE(testProgName) {
    std::string progName(ml::core::CProgName::progName());

    LOG_DEBUG(<< "Current program name is " << progName);

    BOOST_REQUIRE_EQUAL(std::string("ml_test"), progName);
}

BOOST_AUTO_TEST_CASE(testProgDir) {
    std::string progDir(ml::core::CProgName::progDir());

    LOG_DEBUG(<< "Current program directory is " << progDir);

    ml::core::CRegex expectedPathRegex;
    BOOST_TEST_REQUIRE(expectedPathRegex.init(".+[\\\\/]lib[\\\\/]core[\\\\/]unittest$"));
    BOOST_TEST_REQUIRE(expectedPathRegex.matches(progDir));

    // Confirm we've stripped any extended length indicator on Windows
    BOOST_TEST_REQUIRE(progDir.compare(0, 4, "\\\\?\\") != 0);
}

BOOST_AUTO_TEST_SUITE_END()
