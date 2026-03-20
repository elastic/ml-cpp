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
#include <ver/CBuildInfo.h>

#include <core/CLogger.h>
#include <core/CTimeUtils.h>

#include <boost/test/unit_test.hpp>

#include <string>

BOOST_AUTO_TEST_SUITE(CBuildInfoTest)

BOOST_AUTO_TEST_CASE(testFullInfo) {
    std::string fullInfo(ml::ver::CBuildInfo::fullInfo());
    LOG_DEBUG(<< fullInfo);

    std::string currentYear(
        ml::core::CTimeUtils::toIso8601(ml::core::CTimeUtils::now()), 0, 4);
    LOG_DEBUG(<< "Current year is " << currentYear);

    BOOST_TEST_REQUIRE(fullInfo.find("ml_test") != std::string::npos);
    BOOST_TEST_REQUIRE(fullInfo.find("Version") != std::string::npos);
    BOOST_TEST_REQUIRE(fullInfo.find("Build") != std::string::npos);
    BOOST_TEST_REQUIRE(fullInfo.find("Copyright") != std::string::npos);
    BOOST_TEST_REQUIRE(fullInfo.find("Elasticsearch BV") != std::string::npos);
    BOOST_TEST_REQUIRE(fullInfo.find(currentYear) != std::string::npos);
}

BOOST_AUTO_TEST_SUITE_END()
