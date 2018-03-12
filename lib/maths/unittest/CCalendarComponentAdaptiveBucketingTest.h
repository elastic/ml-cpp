/* ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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

#ifndef INCLUDED_CCalendarComponentAdaptiveBucketingTest_h
#define INCLUDED_CCalendarComponentAdaptiveBucketingTest_h

#include <cppunit/extensions/HelperMacros.h>

class CCalendarComponentAdaptiveBucketingTest : public CppUnit::TestFixture {
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
