/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CBucketQueueTest_h
#define INCLUDED_CBucketQueueTest_h

#include <cppunit/extensions/HelperMacros.h>

class CBucketQueueTest : public CppUnit::TestFixture
{
    public:
        void testConstructorFillsQueue(void);
        void testPushGivenEarlierTime(void);
        void testGetGivenFullQueueWithNoPop(void);
        void testGetGivenFullQueueAfterPop(void);
        void testClear(void);
        void testIterators(void);
        void testReverseIterators(void);
        void testBucketQueueUMap(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CBucketQueueTest_h
