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

#ifndef INCLUDED_CMultinomialConjugateTest_h
#define INCLUDED_CMultinomialConjugateTest_h

#include <cppunit/extensions/HelperMacros.h>

class CMultinomialConjugateTest : public CppUnit::TestFixture
{
    public:
        void testMultipleUpdate();
        void testPropagation();
        void testProbabilityEstimation();
        void testMarginalLikelihood();
        void testSampleMarginalLikelihood();
        void testProbabilityOfLessLikelySamples();
        void testAnomalyScore();
        void testRemoveCategories();
        void testPersist();
        void testOverflow();
        void testConcentration();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CMultinomialConjugateTest_h
