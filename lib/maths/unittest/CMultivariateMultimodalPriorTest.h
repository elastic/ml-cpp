/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CMultivariateMultimodalPriorTest_h
#define INCLUDED_CMultivariateMultimodalPriorTest_h

#include <cppunit/extensions/HelperMacros.h>

class CMultivariateMultimodalPriorTest : public CppUnit::TestFixture
{
    public:
        void testMultipleUpdate(void);
        void testPropagation(void);
        void testSingleMode(void);
        void testMultipleModes(void);
        void testSplitAndMerge(void);
        void testMarginalLikelihood(void);
        void testMarginalLikelihoodMean(void);
        void testMarginalLikelihoodMode(void);
        void testSampleMarginalLikelihood(void);
        void testProbabilityOfLessLikelySamples(void);
        void testIntegerData(void);
        void testLowVariationData(void);
        void testLatLongData(void);
        void testPersist(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CMultivariateMultimodalPriorTest_h
