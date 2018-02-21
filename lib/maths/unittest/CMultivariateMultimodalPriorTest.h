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
