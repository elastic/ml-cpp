/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CStringStoreTest_h
#define INCLUDED_CStringStoreTest_h

#include <cppunit/extensions/HelperMacros.h>

class CStringStoreTest : public CppUnit::TestFixture
{
    public:
        void setUp(void);

        void testStringStore(void);
        void testMemUsage(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CStringStoreTest_h
