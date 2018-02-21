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
