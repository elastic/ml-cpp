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

#ifndef INCLUDED_CQDigestTest_h
#define INCLUDED_CQDigestTest_h

#include <cppunit/extensions/HelperMacros.h>


class CQDigestTest : public CppUnit::TestFixture
{
    public:
        void testAdd();
        void testMerge();
        void testCdf();
        void testSummary();
        void testPropagateForwardByTime();
        void testScale();
        void testPersist();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CQDigestTest_h
