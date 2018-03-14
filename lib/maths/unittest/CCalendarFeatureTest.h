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

#ifndef INCLUDED_CCalendarFeatureTest_h
#define INCLUDED_CCalendarFeatureTest_h

#include <cppunit/extensions/HelperMacros.h>

class CCalendarFeatureTest : public CppUnit::TestFixture {
    public:
        void testInitialize(void);
        void testComparison(void);
        void testOffset(void);
        void testPersist(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CCalendarFeatureTest_h
