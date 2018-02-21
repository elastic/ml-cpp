/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CMemoryUsageTest_h
#define INCLUDED_CMemoryUsageTest_h

#include <cppunit/extensions/HelperMacros.h>

class CMemoryUsageTest : public CppUnit::TestFixture
{
    public:
        void testUsage(void);
        void testDebug(void);
        void testDynamicSizeAlwaysZero(void);
        void testCompress(void);
        void testStringBehaviour(void);
        void testStringMemory(void);
        void testStringClear(void);
        void testSharedPointer(void);
        void testRawPointer(void);
        void testSmallVector(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CMemoryUsageTest_h
