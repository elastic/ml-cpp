/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CMutexTest_h
#define INCLUDED_CMutexTest_h

#include <cppunit/extensions/HelperMacros.h>

class CMutexTest : public CppUnit::TestFixture
{
    public:
        void    testRecursive(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CMutexTest_h
