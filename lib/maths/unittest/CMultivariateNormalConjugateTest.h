/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
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
