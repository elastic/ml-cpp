/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CBuildInfoTest.h"

#include <ver/CBuildInfo.h>

#include <core/CLogger.h>
#include <core/CTimeUtils.h>

#include <string>

CppUnit::Test* CBuildInfoTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CBuildInfoTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CBuildInfoTest>("CBuildInfoTest::testFullInfo", &CBuildInfoTest::testFullInfo));

    return suiteOfTests;
}

void CBuildInfoTest::testFullInfo(void) {
    std::string fullInfo(ml::ver::CBuildInfo::fullInfo());
    LOG_DEBUG(<< fullInfo);

    std::string currentYear(ml::core::CTimeUtils::toIso8601(ml::core::CTimeUtils::now()), 0, 4);
    LOG_DEBUG(<< "Current year is " << currentYear);

    CPPUNIT_ASSERT(fullInfo.find("ml_test") != std::string::npos);
    CPPUNIT_ASSERT(fullInfo.find("Version") != std::string::npos);
    CPPUNIT_ASSERT(fullInfo.find("Build") != std::string::npos);
    CPPUNIT_ASSERT(fullInfo.find("Copyright") != std::string::npos);
    CPPUNIT_ASSERT(fullInfo.find("Elasticsearch BV") != std::string::npos);
    CPPUNIT_ASSERT(fullInfo.find(currentYear) != std::string::npos);
}
