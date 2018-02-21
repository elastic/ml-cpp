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

#ifndef INCLUDED_CSeasonalComponentAdaptiveBucketingTest_h
#define INCLUDED_CSeasonalComponentAdaptiveBucketingTest_h

#include <cppunit/extensions/HelperMacros.h>

class CSeasonalComponentAdaptiveBucketingTest : public CppUnit::TestFixture
{
    public:
        void testInitialize(void);
        void testSwap(void);
        void testRefine(void);
        void testPropagateForwardsByTime(void);
        void testMinimumBucketLength(void);
        void testUnintialized(void);
        void testKnots(void);
        void testLongTermTrendKnots(void);
        void testShiftValue(void);
        void testSlope(void);
        void testPersist(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CSeasonalComponentAdaptiveBucketingTest_h
