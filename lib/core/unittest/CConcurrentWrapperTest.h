/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CConcurrentWrapperTest_h
#define INCLUDED_CConcurrentWrapperTest_h

#include <cppunit/extensions/HelperMacros.h>

class CConcurrentWrapperTest : public CppUnit::TestFixture
{
    public:
        void testBasic(void);
        void testThreads(void);
        void testThreadsSlow(void);
        void testThreadsSlowLowCapacity(void);
        void testThreadsLowCapacity(void);
        void testMemoryDebug(void);

        static CppUnit::Test *suite();
};


#endif /* INCLUDED_CConcurrentWrapperTest_h */
