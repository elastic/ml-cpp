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
        void testMemberChecksum();
        void testContainers();
        void testNullable();
        void testAccumulators();
        void testPair();
        void testArray();
        void testCombinations();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CChecksumTest_h
