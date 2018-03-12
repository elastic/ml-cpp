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
#ifndef INCLUDED_CStringSimilarityTesterTest_h
#define INCLUDED_CStringSimilarityTesterTest_h

#include <cppunit/extensions/HelperMacros.h>

class CStringSimilarityTesterTest : public CppUnit::TestFixture {
    public:
        void testStringSimilarity(void);
        void testLevensteinDistance(void);
        void testLevensteinDistance2(void);
        void testLevensteinDistanceThroughputDifferent(void);
        void testLevensteinDistanceThroughputSimilar(void);
        void testLevensteinDistanceAlgorithmEquivalence(void);
        void testWeightedEditDistance(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CStringSimilarityTesterTest_h

