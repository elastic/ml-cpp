/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CLinearAlgebraTest_h
#define INCLUDED_CLinearAlgebraTest_h

#include <cppunit/extensions/HelperMacros.h>

class CLinearAlgebraTest : public CppUnit::TestFixture
{
    public:
        void testSymmetricMatrixNxN(void);
        void testVectorNx1(void);
        void testSymmetricMatrix(void);
        void testVector(void);
        void testNorms(void);
        void testUtils(void);
        void testGaussianLogLikelihood(void);
        void testSampleGaussian(void);
        void testLogDeterminant(void);
        void testProjected(void);
        void testPersist(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CLinearAlgebraTest_h
