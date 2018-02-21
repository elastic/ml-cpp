/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CChecksumTest_h
#define INCLUDED_CChecksumTest_h

#include <cppunit/extensions/HelperMacros.h>

class CChecksumTest : public CppUnit::TestFixture
{
    public:
        void testMemberChecksum(void);
        void testContainers(void);
        void testNullable(void);
        void testAccumulators(void);
        void testPair(void);
        void testArray(void);
        void testCombinations(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CChecksumTest_h
