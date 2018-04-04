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
        void testMultipleUpdate();
        void testPropagation();
        void testMeanEstimation();
        void testMarginalLikelihood();
        void testMarginalLikelihoodMode();
        void testMarginalLikelihoodVariance();
        void testSampleMarginalLikelihood();
        void testCdf();
        void testProbabilityOfLessLikelySamples();
        void testAnomalyScore();
        void testOffset();
        void testPersist();
        void testNegativeSample();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CPoissonMeanConjugateTest_h
