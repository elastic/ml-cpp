/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CPackedBitVectorTest_h
#define INCLUDED_CPackedBitVectorTest_h

#include <cppunit/extensions/HelperMacros.h>

class CPackedBitVectorTest : public CppUnit::TestFixture
{
    public:
        void testCreation(void);
        void testExtend(void);
        void testContract(void);
        void testOperators(void);
        void testInner(void);
        void testBitwiseOr(void);
        void testPersist(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CPackedBitVectorTest_h
