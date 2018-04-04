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
#ifndef INCLUDED_CBucketQueueTest_h
#define INCLUDED_CBucketQueueTest_h

#include <cppunit/extensions/HelperMacros.h>

class CBucketQueueTest : public CppUnit::TestFixture
{
    public:
        void testConstructorFillsQueue();
        void testPushGivenEarlierTime();
        void testGetGivenFullQueueWithNoPop();
        void testGetGivenFullQueueAfterPop();
        void testClear();
        void testIterators();
        void testReverseIterators();
        void testBucketQueueUMap();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CBucketQueueTest_h
