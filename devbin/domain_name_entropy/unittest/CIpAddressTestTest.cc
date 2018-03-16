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
#include "CIpAddressTestTest.h"

#include <core/CLogger.h>

#include "../CIpAddressTest.h"

using namespace ml;
using namespace domain_name_entropy;

CppUnit::Test* CIpAddressTestTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CIpAddressTestTest");

    suiteOfTests->addTest(
        new CppUnit::TestCaller<CIpAddressTestTest>("CIpAddressTestTest::testIpv4", &CIpAddressTestTest::testIpv4));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CIpAddressTestTest>("CIpAddressTestTest::testIpv6", &CIpAddressTestTest::testIpv6));
    return suiteOfTests;
}

void CIpAddressTestTest::testIpv4(void) {
    CIpAddressTest tester;

    CPPUNIT_ASSERT(tester.isIpAddress("127.0.0.1"));
    CPPUNIT_ASSERT(tester.isIpAddress("10.0.0.1"));
    CPPUNIT_ASSERT(tester.isIpAddress("192.168.1.1"));
    CPPUNIT_ASSERT(tester.isIpAddress("0.0.0.0"));
    CPPUNIT_ASSERT(tester.isIpAddress("255.255.255.255"));

    CPPUNIT_ASSERT(!tester.isIpAddress("10002.3.4"));
    CPPUNIT_ASSERT(!tester.isIpAddress("1.2.3"));
    CPPUNIT_ASSERT(!tester.isIpAddress("1.2.3.4.5"));
    CPPUNIT_ASSERT(!tester.isIpAddress("256.0.0.0"));
    CPPUNIT_ASSERT(!tester.isIpAddress("260.0.0.0"));
}

void CIpAddressTestTest::testIpv6(void) {
    CIpAddressTest tester;

    CPPUNIT_ASSERT(tester.isIpAddress("1:2:3:4:5:6:7:8"));
    CPPUNIT_ASSERT(tester.isIpAddress("::ffff:10.0.0.1"));
    CPPUNIT_ASSERT(tester.isIpAddress("::ffff:1.2.3.4"));
    CPPUNIT_ASSERT(tester.isIpAddress("::ffff:0.0.0.0"));
    CPPUNIT_ASSERT(tester.isIpAddress("1:2:3:4:5:6:77:88"));
    CPPUNIT_ASSERT(tester.isIpAddress("::ffff:255.255.255.255"));
    CPPUNIT_ASSERT(tester.isIpAddress("fe08::7:8"));
    CPPUNIT_ASSERT(tester.isIpAddress("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff"));
    CPPUNIT_ASSERT(!tester.isIpAddress("1:2:3:4:5:6:7:8:9"));
    CPPUNIT_ASSERT(!tester.isIpAddress("1:2:3:4:5:6::7:8"));
    CPPUNIT_ASSERT(!tester.isIpAddress(":1:2:3:4:5:6:7:8"));
    CPPUNIT_ASSERT(!tester.isIpAddress("1:2:3:4:5:6:7:8:"));
    CPPUNIT_ASSERT(!tester.isIpAddress("::1:2:3:4:5:6:7:8"));
    CPPUNIT_ASSERT(!tester.isIpAddress("1:2:3:4:5:6:7:8::"));
    CPPUNIT_ASSERT(!tester.isIpAddress("1:2:3:4:5:6:7:88888"));
    CPPUNIT_ASSERT(!tester.isIpAddress("2001:db8:3:4:5::192.0.2.33"));
    CPPUNIT_ASSERT(!tester.isIpAddress("fe08::7:8%"));
    CPPUNIT_ASSERT(!tester.isIpAddress("fe08::7:8i"));
    CPPUNIT_ASSERT(!tester.isIpAddress("fe08::7:8interface"));
}
