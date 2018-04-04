/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CLinearAlgebraTest_h
#define INCLUDED_CLinearAlgebraTest_h

#include <cppunit/extensions/HelperMacros.h>

class CLinearAlgebraTest : public CppUnit::TestFixture {
public:
    void testSymmetricMatrixNxN();
    void testVectorNx1();
    void testSymmetricMatrix();
    void testVector();
    void testNorms();
    void testUtils();
    void testGaussianLogLikelihood();
    void testSampleGaussian();
    void testLogDeterminant();
    void testProjected();
    void testPersist();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CLinearAlgebraTest_h
