/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_COneOfNPriorTest_h
#define INCLUDED_COneOfNPriorTest_h

#include <cppunit/extensions/HelperMacros.h>


class COneOfNPriorTest : public CppUnit::TestFixture
{
    public:
        void testFilter(void);
        void testMultipleUpdate(void);
        void testWeights(void);
        void testModels(void);
        void testModelSelection(void);
        void testMarginalLikelihood(void);
        void testMarginalLikelihoodMean(void);
        void testMarginalLikelihoodMode(void);
        void testMarginalLikelihoodVariance(void);
        void testSampleMarginalLikelihood(void);
        void testCdf(void);
        void testProbabilityOfLessLikelySamples(void);
        void testPersist(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_COneOfNPriorTest_h
