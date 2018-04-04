/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CLogNormalMeanVarConjugateTest_h
#define INCLUDED_CLogNormalMeanVarConjugateTest_h

#include <cppunit/extensions/HelperMacros.h>


class CLogNormalMeanPrecConjugateTest : public CppUnit::TestFixture
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
        void testOffset();
        void testIntegerData();
        void testLowVariationData();
        void testPersist();
        void testVarianceScale();
        void testNegativeSample();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CLogNormalMeanVarConjugateTest_h
