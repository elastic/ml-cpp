/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CMultivariateNormalConjugateTest_h
#define INCLUDED_CMultivariateNormalConjugateTest_h

#include <cppunit/extensions/HelperMacros.h>

class CMultivariateNormalConjugateTest : public CppUnit::TestFixture
{
    public:
        void testMultipleUpdate(void);
        void testPropagation(void);
        void testMeanVectorEstimation(void);
        void testPrecisionMatrixEstimation(void);
        void testMarginalLikelihood(void);
        void testMarginalLikelihoodMode(void);
        void testSampleMarginalLikelihood(void);
        void testProbabilityOfLessLikelySamples(void);
        void testIntegerData(void);
        void testLowVariationData(void);
        void testPersist(void);
        void calibrationExperiment(void);
        void dataGenerator(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CMultivariateNormalConjugateTest_h
