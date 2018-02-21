/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
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
