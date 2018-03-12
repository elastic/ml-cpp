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

#ifndef INCLUDED_CMultivariateNormalConjugateTest_h
#define INCLUDED_CMultivariateNormalConjugateTest_h

#include <cppunit/extensions/HelperMacros.h>

class CMultivariateNormalConjugateTest : public CppUnit::TestFixture {
    public:
        void testMultipleUpdate(void);
        void testPropagation(void);
        void testMeanVectorEstimation(void);
        void testPrecisionMatrixEstimation(void);
        void testMarginalLikelihood(void);
        void testMarginalLikelihoodMode(void);
        void testSampleMarginalLikelihood(void);
        void testProbabilityOfLessLikelySamples(void);
        void testIntegerData(void);
        void testLowVariationData(void);
        void testPersist(void);
        void calibrationExperiment(void);
        void dataGenerator(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CMultivariateNormalConjugateTest_h
