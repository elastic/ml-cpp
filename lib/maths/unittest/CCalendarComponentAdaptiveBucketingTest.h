/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CCalendarComponentAdaptiveBucketingTest_h
#define INCLUDED_CCalendarComponentAdaptiveBucketingTest_h

#include <cppunit/extensions/HelperMacros.h>

class CCalendarComponentAdaptiveBucketingTest : public CppUnit::TestFixture
{
    public:
        void setUp(void);
        void tearDown(void);

        void testInitialize(void);
        void testSwap(void);
        void testRefine(void);
        void testPropagateForwardsByTime(void);
        void testMinimumBucketLength(void);
        void testUnintialized(void);
        void testKnots(void);
        void testPersist(void);

        static CppUnit::Test *suite(void);

    private:
        std::string m_Timezone;
};

#endif // INCLUDED_CCalendarComponentAdaptiveBucketingTest_h
