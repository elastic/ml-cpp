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
        void testMultipleUpdate();
        void testPropagation();
        void testSingleMode();
        void testMultipleModes();
        void testMarginalLikelihood();
        void testMarginalLikelihoodMode();
        void testMarginalLikelihoodConfidenceInterval();
        void testSampleMarginalLikelihood();
        void testCdf();
        void testProbabilityOfLessLikelySamples();
        void testSeasonalVarianceScale();
        void testLargeValues();
        void testPersist();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CMultimodalPriorTest_h
