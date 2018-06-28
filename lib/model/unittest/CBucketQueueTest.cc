/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CBucketQueueTest.h"

#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CLogger.h>
#include <core/CPersistUtils.h>

#include <model/CBucketQueue.h>

#include <boost/bind.hpp>
#include <boost/unordered_map.hpp>

using namespace ml;
using namespace model;

using TSizeSizePr = std::pair<std::size_t, std::size_t>;
using TSizeSizePrUInt64Pr = std::pair<TSizeSizePr, uint64_t>;
using TSizeSizePrUInt64PrVec = std::vector<TSizeSizePrUInt64Pr>;
using TSizeSizePrUInt64UMap = boost::unordered_map<TSizeSizePr, uint64_t>;
using TSizeSizePrUInt64UMapQueue = model::CBucketQueue<TSizeSizePrUInt64UMap>;
using TSizeSizePrUInt64UMapQueueCItr = TSizeSizePrUInt64UMapQueue::const_iterator;

void CBucketQueueTest::testConstructorFillsQueue() {
    CBucketQueue<int> queue(3, 5, 15);

    CPPUNIT_ASSERT_EQUAL(std::size_t(4), queue.size());

    std::set<const int*> values;
    values.insert(&queue.get(0));
    values.insert(&queue.get(5));
    values.insert(&queue.get(10));
    values.insert(&queue.get(15));
    CPPUNIT_ASSERT_EQUAL(std::size_t(4), values.size());
}

void CBucketQueueTest::testPushGivenEarlierTime() {
    CBucketQueue<std::string> queue(1, 5, 0);
    queue.push("a", 5);
    queue.push("b", 10);
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), queue.size());

    queue.push("c", 3);

    CPPUNIT_ASSERT_EQUAL(std::size_t(2), queue.size());
    CPPUNIT_ASSERT_EQUAL(std::string("a"), queue.get(7));
    CPPUNIT_ASSERT_EQUAL(std::string("b"), queue.get(12));
}

void CBucketQueueTest::testGetGivenFullQueueWithNoPop() {
    CBucketQueue<std::string> queue(1, 5, 0);
    queue.push("a", 5);
    queue.push("b", 10);

    CPPUNIT_ASSERT_EQUAL(std::size_t(2), queue.size());
    CPPUNIT_ASSERT_EQUAL(std::string("a"), queue.get(5));
    CPPUNIT_ASSERT_EQUAL(std::string("b"), queue.get(10));
}

void CBucketQueueTest::testGetGivenFullQueueAfterPop() {
    CBucketQueue<std::string> queue(1, 5, 0);
    queue.push("a", 5);
    queue.push("b", 10);
    queue.push("c", 15);

    CPPUNIT_ASSERT_EQUAL(std::size_t(2), queue.size());
    CPPUNIT_ASSERT_EQUAL(std::string("b"), queue.get(11));
    CPPUNIT_ASSERT_EQUAL(std::string("c"), queue.get(19));
}

void CBucketQueueTest::testClear() {
    CBucketQueue<int> queue(2, 5, 0);
    CPPUNIT_ASSERT_EQUAL(std::size_t(3), queue.size());
    queue.push(0, 5);
    queue.push(1, 10);
    queue.push(2, 15);
    CPPUNIT_ASSERT_EQUAL(std::size_t(3), queue.size());

    queue.clear();
    CPPUNIT_ASSERT_EQUAL(int(0), queue.get(5));
    CPPUNIT_ASSERT_EQUAL(int(0), queue.get(10));
    CPPUNIT_ASSERT_EQUAL(int(0), queue.get(15));

    CPPUNIT_ASSERT_EQUAL(std::size_t(3), queue.size());
}

void CBucketQueueTest::testIterators() {
    using TStringQueueItr = CBucketQueue<std::string>::iterator;

    CBucketQueue<std::string> queue(1, 5, 0);
    queue.push("a", 5);
    queue.push("b", 10);

    std::vector<std::string> strings;
    for (TStringQueueItr itr = queue.begin(); itr != queue.end(); ++itr) {
        strings.push_back(*itr);
    }

    CPPUNIT_ASSERT_EQUAL(std::size_t(2), strings.size());
    CPPUNIT_ASSERT_EQUAL(std::string("b"), strings[0]);
    CPPUNIT_ASSERT_EQUAL(std::string("a"), strings[1]);
}

void CBucketQueueTest::testReverseIterators() {
    using TStringQueueCRItr = CBucketQueue<std::string>::const_reverse_iterator;

    CBucketQueue<std::string> queue(1, 5, 0);
    queue.push("a", 5);
    queue.push("b", 10);

    std::vector<std::string> strings;
    for (TStringQueueCRItr itr = queue.rbegin(); itr != queue.rend(); ++itr) {
        strings.push_back(*itr);
    }

    CPPUNIT_ASSERT_EQUAL(std::size_t(2), strings.size());
    CPPUNIT_ASSERT_EQUAL(std::string("a"), strings[0]);
    CPPUNIT_ASSERT_EQUAL(std::string("b"), strings[1]);
}

void CBucketQueueTest::testBucketQueueUMap() {
    // Tests the memory usage of an unordered_map in a bucket queue
    // before and after persistence
    std::size_t usageBefore = 0;
    const std::string queueTag("b");
    std::stringstream ss;
    {
        TSizeSizePrUInt64UMapQueue queue(4, 1000, 5000);

        queue.push(TSizeSizePrUInt64UMap(1));
        queue.latest()[TSizeSizePr(55, 66)] = 99;
        queue.latest()[TSizeSizePr(55, 6)] = 99;
        queue.latest()[TSizeSizePr(55, 7)] = 98;
        queue.latest()[TSizeSizePr(55, 8)] = 99;
        queue.latest()[TSizeSizePr(55, 9)] = 999;
        queue.push(TSizeSizePrUInt64UMap(1));
        queue.latest()[TSizeSizePr(1, 55)] = 88;
        queue.latest()[TSizeSizePr(2, 55)] = 88;
        queue.latest()[TSizeSizePr(3, 55)] = 88;
        queue.latest()[TSizeSizePr(4, 55)] = 88;
        queue.latest()[TSizeSizePr(5, 55)] = 88;
        queue.latest()[TSizeSizePr(6, 55)] = 88;
        queue.latest()[TSizeSizePr(7, 55)] = 88;
        queue.latest()[TSizeSizePr(8, 55)] = 88;
        queue.latest()[TSizeSizePr(9, 55)] = 88;
        queue.push(TSizeSizePrUInt64UMap(1));
        queue.latest()[TSizeSizePr(5, 6)] = 7;
        queue.latest()[TSizeSizePr(5, 67)] = 7;
        queue.latest()[TSizeSizePr(58, 76)] = 7;
        queue.push(TSizeSizePrUInt64UMap(1));
        for (std::size_t i = 0; i < 10000; i += 100) {
            for (std::size_t j = 0; j < 50000; j += 400) {
                queue.latest()[TSizeSizePr(i, j)] = 99 * i * j + 12;
            }
        }
        usageBefore = core::CMemory::dynamicSize(queue);
        core::CJsonStatePersistInserter inserter(ss);
        core::CPersistUtils::persist(queueTag, queue, inserter);
    }
    {
        TSizeSizePrUInt64UMapQueue queue(4, 1000, 5000);
        core::CJsonStateRestoreTraverser traverser(ss);
        core::CPersistUtils::restore(queueTag, queue, traverser);
        std::size_t usageAfter = core::CMemory::dynamicSize(queue);
        CPPUNIT_ASSERT_EQUAL(usageBefore, usageAfter);
    }
}

CppUnit::Test* CBucketQueueTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CBucketQueueTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CBucketQueueTest>(
        "CBucketQueueTest::testConstructorFillsQueue",
        &CBucketQueueTest::testConstructorFillsQueue));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBucketQueueTest>(
        "CBucketQueueTest::testPushGivenEarlierTime", &CBucketQueueTest::testPushGivenEarlierTime));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBucketQueueTest>(
        "CBucketQueueTest::testGetGivenFullQueueWithNoPop",
        &CBucketQueueTest::testGetGivenFullQueueWithNoPop));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBucketQueueTest>(
        "CBucketQueueTest::testGetGivenFullQueueAfterPop",
        &CBucketQueueTest::testGetGivenFullQueueAfterPop));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBucketQueueTest>(
        "CBucketQueueTest::testClear", &CBucketQueueTest::testClear));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBucketQueueTest>(
        "CBucketQueueTest::testIterators", &CBucketQueueTest::testIterators));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBucketQueueTest>(
        "CBucketQueueTest::testReverseIterators", &CBucketQueueTest::testReverseIterators));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBucketQueueTest>(
        "CBucketQueueTest::testBucketQueueUMap", &CBucketQueueTest::testBucketQueueUMap));

    return suiteOfTests;
}
