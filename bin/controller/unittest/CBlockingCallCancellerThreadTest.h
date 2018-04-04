/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CBlockingCallCancellerThreadTest_h
#define INCLUDED_CBlockingCallCancellerThreadTest_h

#include <cppunit/extensions/HelperMacros.h>

class CBlockingCallCancellerThreadTest : public CppUnit::TestFixture
{
    public:
        void testCancelBlock();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CBlockingCallCancellerThreadTest_h

