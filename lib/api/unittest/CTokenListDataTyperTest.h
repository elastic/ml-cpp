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
#ifndef INCLUDED_CTokenListDataTyperTest_h
#define INCLUDED_CTokenListDataTyperTest_h

#include <cppunit/extensions/HelperMacros.h>


class CTokenListDataTyperTest : public CppUnit::TestFixture
{
    public:
        void testHexData();
        void testRmdsData();
        void testProxyData();
        void testFxData();
        void testApacheData();
        void testBrokerageData();
        void testVmwareData();
        void testBankData();
        void testJavaGcData();
        void testPersist();
        void testLongReverseSearch();
        void testPreTokenised();
        void testPreTokenisedPerformance();

        void setUp();
        void tearDown();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CTokenListDataTyperTest_h

