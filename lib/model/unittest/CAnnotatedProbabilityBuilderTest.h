/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
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
