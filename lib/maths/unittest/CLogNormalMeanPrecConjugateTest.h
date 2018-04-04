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

#ifndef INCLUDED_CLogNormalMeanVarConjugateTest_h
#define INCLUDED_CLogNormalMeanVarConjugateTest_h

#include <cppunit/extensions/HelperMacros.h>

class CLogNormalMeanPrecConjugateTest : public CppUnit::TestFixture {
public:
    void testMultipleUpdate();
    void testPropagation();
    void testMeanEstimation();
    void testPrecisionEstimation();
    void testMarginalLikelihood();
    void testMarginalLikelihoodMean();
    void testMarginalLikelihoodMode();
    void testMarginalLikelihoodVariance();
    void testSampleMarginalLikelihood();
    void testCdf();
    void testProbabilityOfLessLikelySamples();
    void testAnomalyScore();
    void testOffset();
    void testIntegerData();
    void testLowVariationData();
    void testPersist();
    void testVarianceScale();
    void testNegativeSample();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CLogNormalMeanVarConjugateTest_h
