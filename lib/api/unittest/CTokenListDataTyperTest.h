/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
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

