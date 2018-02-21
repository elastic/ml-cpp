/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CHashingTest_h
#define INCLUDED_CHashingTest_h

#include <cppunit/extensions/HelperMacros.h>

class CHashingTest : public CppUnit::TestFixture
{
    public:
        void testUniversalHash(void);
        void testMurmurHash(void);
        void testHashCombine(void);
        void testConstructors(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CHashingTest_h
