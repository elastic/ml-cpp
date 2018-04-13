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
        void testIntegerData();
        void testLowVariationData();
        void testPersist();
        void testSeasonalVarianceScale();
        void testCountVarianceScale();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CNormalMeanPrecConjugateTest_h
