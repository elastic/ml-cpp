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
        void testHexData(void);
        void testRmdsData(void);
        void testProxyData(void);
        void testFxData(void);
        void testApacheData(void);
        void testBrokerageData(void);
        void testVmwareData(void);
        void testBankData(void);
        void testJavaGcData(void);
        void testPersist(void);
        void testLongReverseSearch(void);
        void testPreTokenised(void);
        void testPreTokenisedPerformance(void);

        void setUp(void);
        void tearDown(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CTokenListDataTyperTest_h

