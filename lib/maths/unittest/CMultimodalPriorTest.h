/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CMultimodalPriorTest_h
#define INCLUDED_CMultimodalPriorTest_h

#include <cppunit/extensions/HelperMacros.h>

class CMultimodalPriorTest : public CppUnit::TestFixture
{
    public:
        void testMultipleUpdate(void);
        void testPropagation(void);
        void testSingleMode(void);
        void testMultipleModes(void);
        void testMarginalLikelihood(void);
        void testMarginalLikelihoodMode(void);
        void testMarginalLikelihoodConfidenceInterval(void);
        void testSampleMarginalLikelihood(void);
        void testCdf(void);
        void testProbabilityOfLessLikelySamples(void);
        void testSeasonalVarianceScale(void);
        void testLargeValues(void);
        void testPersist(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CMultimodalPriorTest_h
