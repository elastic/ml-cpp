/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CMultivariateOneOfNPriorTest_h
#define INCLUDED_CMultivariateOneOfNPriorTest_h

#include <cppunit/extensions/HelperMacros.h>

class CMultivariateOneOfNPriorTest : public CppUnit::TestFixture
{
    public:
        void testMultipleUpdate(void);
        void testPropagation(void);
        void testWeightUpdate(void);
        void testModelUpdate(void);
        void testModelSelection(void);
        void testMarginalLikelihood(void);
        void testMarginalLikelihoodMean(void);
        void testMarginalLikelihoodMode(void);
        void testSampleMarginalLikelihood(void);
        void testProbabilityOfLessLikelySamples(void);
        void testPersist(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CMultivariateOneOfNPriorTest_h
