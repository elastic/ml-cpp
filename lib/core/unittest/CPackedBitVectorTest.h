/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CPackedBitVectorTest_h
#define INCLUDED_CPackedBitVectorTest_h

#include <cppunit/extensions/HelperMacros.h>

class CPackedBitVectorTest : public CppUnit::TestFixture {
public:
    void testCreation();
    void testExtend();
    void testContract();
    void testComparisonAndLess();
    void testBitwiseComplement();
    void testBitwise();
    void testOneBitIterators();
    void testInnerProductBitwiseAnd();
    void testInnerProductBitwiseOr();
    void testProblemCases();
    void testPersist();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CPackedBitVectorTest_h
