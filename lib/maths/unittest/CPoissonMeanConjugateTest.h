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

#ifndef INCLUDED_CPoissonMeanConjugateTest_h
#define INCLUDED_CPoissonMeanConjugateTest_h

#include <cppunit/extensions/HelperMacros.h>

class CPoissonMeanConjugateTest : public CppUnit::TestFixture {
public:
    void testMultipleUpdate(void);
    void testPropagation(void);
    void testMeanEstimation(void);
    void testMarginalLikelihood(void);
    void testMarginalLikelihoodMode(void);
    void testMarginalLikelihoodVariance(void);
    void testSampleMarginalLikelihood(void);
    void testCdf(void);
    void testProbabilityOfLessLikelySamples(void);
    void testAnomalyScore(void);
    void testOffset(void);
    void testPersist(void);
    void testNegativeSample(void);

    static CppUnit::Test *suite(void);
};

#endif// INCLUDED_CPoissonMeanConjugateTest_h
