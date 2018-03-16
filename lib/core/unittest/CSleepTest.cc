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
#include "CSleepTest.h"

#include <core/CLogger.h>
#include <core/CSleep.h>
#include <core/CTimeUtils.h>
#include <core/CoreTypes.h>

CppUnit::Test* CSleepTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CSleepTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CSleepTest>("CSleepTest::testSleep", &CSleepTest::testSleep));

    return suiteOfTests;
}

void CSleepTest::testSleep(void) {
    ml::core_t::TTime start(ml::core::CTimeUtils::now());

    ml::core::CSleep::sleep(7500);

    ml::core_t::TTime end(ml::core::CTimeUtils::now());

    ml::core_t::TTime diff(end - start);
    LOG_DEBUG("During 7.5 second wait, the clock advanced by " << diff << " seconds");

    // Clock time should be 7 or 8 seconds further ahead
    CPPUNIT_ASSERT(diff >= 7);
    CPPUNIT_ASSERT(diff <= 8);
}
