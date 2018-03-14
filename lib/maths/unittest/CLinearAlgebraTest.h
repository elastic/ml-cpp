/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */

#ifndef INCLUDED_CLinearAlgebraTest_h
#define INCLUDED_CLinearAlgebraTest_h

#include <cppunit/extensions/HelperMacros.h>

class CLinearAlgebraTest : public CppUnit::TestFixture {
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

    static CppUnit::Test* suite(void);
};

#endif // INCLUDED_CLinearAlgebraTest_h
