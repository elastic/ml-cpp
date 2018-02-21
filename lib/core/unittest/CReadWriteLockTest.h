/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CReadWriteLockTest_h
#define INCLUDED_CReadWriteLockTest_h

#include <cppunit/extensions/HelperMacros.h>


class CReadWriteLockTest : public CppUnit::TestFixture
{
    public:
        void testReadLock(void);
        void testWriteLock(void);
        void testPerformanceVersusMutex(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CReadWriteLockTest_h

