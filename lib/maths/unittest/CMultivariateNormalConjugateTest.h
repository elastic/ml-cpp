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

#ifndef INCLUDED_CMultivariateNormalConjugateTest_h
#define INCLUDED_CMultivariateNormalConjugateTest_h

#include <cppunit/extensions/HelperMacros.h>

class CMultivariateNormalConjugateTest : public CppUnit::TestFixture {
public:
    void testMultipleUpdate();
    void testPropagation();
    void testMeanVectorEstimation();
    void testPrecisionMatrixEstimation();
    void testMarginalLikelihood();
    void testMarginalLikelihoodMode();
    void testSampleMarginalLikelihood();
    void testProbabilityOfLessLikelySamples();
    void testIntegerData();
    void testLowVariationData();
    void testPersist();
    void calibrationExperiment();
    void dataGenerator();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CMultivariateNormalConjugateTest_h
