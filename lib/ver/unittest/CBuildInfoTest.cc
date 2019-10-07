/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
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

    BOOST_TEST(fullInfo.find("ml_test") != std::string::npos);
    BOOST_TEST(fullInfo.find("Version") != std::string::npos);
    BOOST_TEST(fullInfo.find("Build") != std::string::npos);
    BOOST_TEST(fullInfo.find("Copyright") != std::string::npos);
    BOOST_TEST(fullInfo.find("Elasticsearch BV") != std::string::npos);
    BOOST_TEST(fullInfo.find(currentYear) != std::string::npos);
}

BOOST_AUTO_TEST_SUITE_END()
