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
        void testMultipleUpdate();
        void testPropagation();
        void testWeightUpdate();
        void testModelUpdate();
        void testModelSelection();
        void testMarginalLikelihood();
        void testMarginalLikelihoodMean();
        void testMarginalLikelihoodMode();
        void testSampleMarginalLikelihood();
        void testProbabilityOfLessLikelySamples();
        void testPersist();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CMultivariateOneOfNPriorTest_h
