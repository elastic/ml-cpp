/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CNormalMeanPrecConjugateTest_h
#define INCLUDED_CNormalMeanPrecConjugateTest_h

#include <cppunit/extensions/HelperMacros.h>


class CNormalMeanPrecConjugateTest : public CppUnit::TestFixture
{
    public:
        void testMultipleUpdate(void);
        void testPropagation(void);
        void testMeanEstimation(void);
        void testPrecisionEstimation(void);
        void testMarginalLikelihood(void);
        void testMarginalLikelihoodMean(void);
        void testMarginalLikelihoodMode(void);
        void testMarginalLikelihoodVariance(void);
        void testSampleMarginalLikelihood(void);
        void testCdf(void);
        void testProbabilityOfLessLikelySamples(void);
        void testAnomalyScore(void);
        void testIntegerData(void);
        void testLowVariationData(void);
        void testPersist(void);
        void testSeasonalVarianceScale(void);
        void testCountVarianceScale(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CNormalMeanPrecConjugateTest_h
