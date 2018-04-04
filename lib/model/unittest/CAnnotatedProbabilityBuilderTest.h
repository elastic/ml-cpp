/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CAnnotatedProbabilityBuilderTest_h
#define INCLUDED_CAnnotatedProbabilityBuilderTest_h

#include <cppunit/extensions/HelperMacros.h>

class CAnnotatedProbabilityBuilderTest : public CppUnit::TestFixture {
public:
    void testProbability();
    void testAddAttributeProbabilityGivenIndividualCount();
    void testAddAttributeProbabilityGivenPopulationCount();
    void testAddAttributeProbabilityGivenIndividualRare();
    void testAddAttributeProbabilityGivenPopulationRare();
    void testAddAttributeProbabilityGivenPopulationFreqRare();
    void testPersonFrequencyGivenIndividualCount();
    void testPersonFrequencyGivenIndividualRare();
    void testPersonFrequencyGivenPopulationRare();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CAnnotatedProbabilityBuilderTest_h
