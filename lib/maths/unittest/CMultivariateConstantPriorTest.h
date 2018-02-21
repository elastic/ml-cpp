/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CMultivariateConstantPriorTest_h
#define INCLUDED_CMultivariateConstantPriorTest_h

#include <cppunit/extensions/HelperMacros.h>

class CMultivariateConstantPriorTest : public CppUnit::TestFixture
{
    public:
        void testAddSamples(void);
        void testMarginalLikelihood(void);
        void testMarginalLikelihoodMean(void);
        void testMarginalLikelihoodMode(void);
        void testMarginalLikelihoodCovariance(void);
        void testSampleMarginalLikelihood(void);
        void testProbabilityOfLessLikelySamples(void);
        void testPersist(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CMultivariateConstantPriorTest_h
