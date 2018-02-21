/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CAnnotatedProbabilityBuilderTest_h
#define INCLUDED_CAnnotatedProbabilityBuilderTest_h

#include <cppunit/extensions/HelperMacros.h>

class CAnnotatedProbabilityBuilderTest : public CppUnit::TestFixture
{
    public:
        void testProbability(void);
        void testAddAttributeProbabilityGivenIndividualCount(void);
        void testAddAttributeProbabilityGivenPopulationCount(void);
        void testAddAttributeProbabilityGivenIndividualRare(void);
        void testAddAttributeProbabilityGivenPopulationRare(void);
        void testAddAttributeProbabilityGivenPopulationFreqRare(void);
        void testPersonFrequencyGivenIndividualCount(void);
        void testPersonFrequencyGivenIndividualRare(void);
        void testPersonFrequencyGivenPopulationRare(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CAnnotatedProbabilityBuilderTest_h
