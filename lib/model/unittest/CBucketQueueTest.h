/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CBucketQueueTest_h
#define INCLUDED_CBucketQueueTest_h

#include <cppunit/extensions/HelperMacros.h>

class CBucketQueueTest : public CppUnit::TestFixture {
public:
    void testConstructorFillsQueue();
    void testPushGivenEarlierTime();
    void testGetGivenFullQueueWithNoPop();
    void testGetGivenFullQueueAfterPop();
    void testClear();
    void testIterators();
    void testReverseIterators();
    void testBucketQueueUMap();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CBucketQueueTest_h
