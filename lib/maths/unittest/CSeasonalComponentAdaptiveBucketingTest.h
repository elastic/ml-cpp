/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CSeasonalComponentAdaptiveBucketingTest_h
#define INCLUDED_CSeasonalComponentAdaptiveBucketingTest_h

#include <cppunit/extensions/HelperMacros.h>

class CSeasonalComponentAdaptiveBucketingTest : public CppUnit::TestFixture {
public:
    void testInitialize();
    void testSwap();
    void testRefine();
    void testPropagateForwardsByTime();
    void testMinimumBucketLength();
    void testUnintialized();
    void testKnots();
    void testLongTermTrendKnots();
    void testShiftValue();
    void testSlope();
    void testPersist();
    void testUpgrade();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CSeasonalComponentAdaptiveBucketingTest_h
