/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CCalendarComponentAdaptiveBucketingTest_h
#define INCLUDED_CCalendarComponentAdaptiveBucketingTest_h

#include <cppunit/extensions/HelperMacros.h>

class CCalendarComponentAdaptiveBucketingTest : public CppUnit::TestFixture {
public:
    void setUp();
    void tearDown();

    void testInitialize();
    void testSwap();
    void testRefine();
    void testPropagateForwardsByTime();
    void testMinimumBucketLength();
    void testUnintialized();
    void testKnots();
    void testPersist();

    static CppUnit::Test* suite();

private:
    std::string m_Timezone;
};

#endif // INCLUDED_CCalendarComponentAdaptiveBucketingTest_h
