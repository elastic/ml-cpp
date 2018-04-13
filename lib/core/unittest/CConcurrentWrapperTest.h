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
        void testBasic();
        void testThreads();
        void testThreadsSlow();
        void testThreadsSlowLowCapacity();
        void testThreadsLowCapacity();
        void testMemoryDebug();

        static CppUnit::Test *suite();
};


#endif /* INCLUDED_CConcurrentWrapperTest_h */
