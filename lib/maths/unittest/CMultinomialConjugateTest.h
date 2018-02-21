/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CMultinomialConjugateTest_h
#define INCLUDED_CMultinomialConjugateTest_h

#include <cppunit/extensions/HelperMacros.h>

class CMultinomialConjugateTest : public CppUnit::TestFixture
{
    public:
        void testMultipleUpdate(void);
        void testPropagation(void);
        void testProbabilityEstimation(void);
        void testMarginalLikelihood(void);
        void testSampleMarginalLikelihood(void);
        void testProbabilityOfLessLikelySamples(void);
        void testAnomalyScore(void);
        void testRemoveCategories(void);
        void testPersist(void);
        void testOverflow(void);
        void testConcentration(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CMultinomialConjugateTest_h
