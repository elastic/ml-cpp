/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CMessageBufferTest_h
#define INCLUDED_CMessageBufferTest_h

#include <cppunit/extensions/HelperMacros.h>

class CMessageBufferTest : public CppUnit::TestFixture
{
    public:
        void    testAll(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CMessageBufferTest_h
