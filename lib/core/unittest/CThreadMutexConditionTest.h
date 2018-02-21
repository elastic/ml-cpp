/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CThreadMutexConditionTest_h
#define INCLUDED_CThreadMutexConditionTest_h

#include <cppunit/extensions/HelperMacros.h>

class CThreadMutexConditionTest : public CppUnit::TestFixture
{
    public:
        void    testThread(void);
        void    testThreadCondition(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CThreadMutexConditionTest_h
