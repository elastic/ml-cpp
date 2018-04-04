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
        void testUsage();
        void testDebug();
        void testDynamicSizeAlwaysZero();
        void testCompress();
        void testStringBehaviour();
        void testStringMemory();
        void testStringClear();
        void testSharedPointer();
        void testRawPointer();
        void testSmallVector();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CMemoryUsageTest_h
