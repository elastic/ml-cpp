/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CPoissonMeanConjugateTest_h
#define INCLUDED_CPoissonMeanConjugateTest_h

#include <cppunit/extensions/HelperMacros.h>


class CPoissonMeanConjugateTest : public CppUnit::TestFixture
{
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

#endif // INCLUDED_CPoissonMeanConjugateTest_h
