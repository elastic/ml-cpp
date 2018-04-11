/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
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
