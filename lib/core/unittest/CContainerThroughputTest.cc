/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CContainerThroughputTest.h"

#include <core/BoostMultiIndex.h>
#include <core/CLogger.h>
#include <core/CTimeUtils.h>

#include <boost/circular_buffer.hpp>

#include <deque>
#include <list>
#include <map>
#include <vector>


const size_t CContainerThroughputTest::FILL_SIZE(2);
const size_t CContainerThroughputTest::TEST_SIZE(10000000);


CppUnit::Test *CContainerThroughputTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CContainerThroughputTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CContainerThroughputTest>(
                                   "CContainerThroughputTest::testVector",
                                   &CContainerThroughputTest::testVector) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CContainerThroughputTest>(
                                   "CContainerThroughputTest::testList",
                                   &CContainerThroughputTest::testList) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CContainerThroughputTest>(
                                   "CContainerThroughputTest::testDeque",
                                   &CContainerThroughputTest::testDeque) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CContainerThroughputTest>(
                                   "CContainerThroughputTest::testMap",
                                   &CContainerThroughputTest::testMap) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CContainerThroughputTest>(
                                   "CContainerThroughputTest::testCircBuf",
                                   &CContainerThroughputTest::testCircBuf) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CContainerThroughputTest>(
                                   "CContainerThroughputTest::testMultiIndex",
                                   &CContainerThroughputTest::testMultiIndex) );

    return suiteOfTests;
}

void CContainerThroughputTest::setUp(void)
{
    CPPUNIT_ASSERT(FILL_SIZE > 0);
    CPPUNIT_ASSERT(TEST_SIZE > FILL_SIZE);
}

void CContainerThroughputTest::testVector(void)
{
    typedef std::vector<SContent> TContentVec;
    TContentVec testVec;
    testVec.reserve(FILL_SIZE);

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO("Starting vector throughput test at " <<
             ml::core::CTimeUtils::toTimeString(start));

    size_t count(0);
    while (count < FILL_SIZE)
    {
        ++count;
        testVec.push_back(SContent(count));
    }

    while (count < TEST_SIZE)
    {
        testVec.erase(testVec.begin());
        ++count;
        testVec.push_back(SContent(count));
    }

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO("Finished vector throughput test at " <<
             ml::core::CTimeUtils::toTimeString(end));

    CPPUNIT_ASSERT_EQUAL(FILL_SIZE, testVec.size());

    LOG_INFO("Vector throughput test with fill size " << FILL_SIZE <<
             " and test size " << TEST_SIZE << " took " << (end - start) <<
             " seconds");
}

void CContainerThroughputTest::testList(void)
{
    typedef std::list<SContent> TContentList;
    TContentList testList;

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO("Starting list throughput test at " <<
             ml::core::CTimeUtils::toTimeString(start));

    size_t count(0);
    while (count < FILL_SIZE)
    {
        ++count;
        testList.push_back(SContent(count));
    }

    while (count < TEST_SIZE)
    {
        testList.pop_front();
        ++count;
        testList.push_back(SContent(count));
    }

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO("Finished list throughput test at " <<
             ml::core::CTimeUtils::toTimeString(end));

    CPPUNIT_ASSERT_EQUAL(FILL_SIZE, testList.size());

    LOG_INFO("List throughput test with fill size " << FILL_SIZE <<
             " and test size " << TEST_SIZE << " took " << (end - start) <<
             " seconds");
}

void CContainerThroughputTest::testDeque(void)
{
    typedef std::deque<SContent> TContentDeque;
    TContentDeque testDeque;

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO("Starting deque throughput test at " <<
             ml::core::CTimeUtils::toTimeString(start));

    size_t count(0);
    while (count < FILL_SIZE)
    {
        ++count;
        testDeque.push_back(SContent(count));
    }

    while (count < TEST_SIZE)
    {
        testDeque.pop_front();
        ++count;
        testDeque.push_back(SContent(count));
    }

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO("Finished deque throughput test at " <<
             ml::core::CTimeUtils::toTimeString(end));

    CPPUNIT_ASSERT_EQUAL(FILL_SIZE, testDeque.size());

    LOG_INFO("Deque throughput test with fill size " << FILL_SIZE <<
             " and test size " << TEST_SIZE << " took " << (end - start) <<
             " seconds");
}

void CContainerThroughputTest::testMap(void)
{
    typedef std::map<size_t, SContent> TSizeContentMap;
    TSizeContentMap testMap;

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO("Starting map throughput test at " <<
             ml::core::CTimeUtils::toTimeString(start));

    size_t count(0);
    while (count < FILL_SIZE)
    {
        ++count;
        testMap.insert(TSizeContentMap::value_type(count, SContent(count)));
    }

    while (count < TEST_SIZE)
    {
        testMap.erase(testMap.begin());
        ++count;
        testMap.insert(TSizeContentMap::value_type(count, SContent(count)));
    }

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO("Finished map throughput test at " <<
             ml::core::CTimeUtils::toTimeString(end));

    CPPUNIT_ASSERT_EQUAL(FILL_SIZE, testMap.size());

    LOG_INFO("Map throughput test with fill size " << FILL_SIZE <<
             " and test size " << TEST_SIZE << " took " << (end - start) <<
             " seconds");
}

void CContainerThroughputTest::testCircBuf(void)
{
    typedef boost::circular_buffer<SContent> TContentCircBuf;
    TContentCircBuf testCircBuf(FILL_SIZE);

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO("Starting circular buffer throughput test at " <<
             ml::core::CTimeUtils::toTimeString(start));

    size_t count(0);
    while (count < FILL_SIZE)
    {
        ++count;
        testCircBuf.push_back(SContent(count));
    }

    while (count < TEST_SIZE)
    {
        testCircBuf.pop_front();
        ++count;
        testCircBuf.push_back(SContent(count));
    }

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO("Finished circular buffer throughput test at " <<
             ml::core::CTimeUtils::toTimeString(end));

    CPPUNIT_ASSERT_EQUAL(FILL_SIZE, testCircBuf.size());

    LOG_INFO("Circular buffer throughput test with fill size " << FILL_SIZE <<
             " and test size " << TEST_SIZE << " took " << (end - start) <<
             " seconds");
}

void CContainerThroughputTest::testMultiIndex(void)
{
    typedef boost::multi_index::multi_index_container<
        SContent,
        boost::multi_index::indexed_by<
            boost::multi_index::hashed_unique<
                BOOST_MULTI_INDEX_MEMBER(SContent, size_t, s_Size)
            >
        >
    > TContentMIndex;
    TContentMIndex testMultiIndex;

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO("Starting multi-index throughput test at " <<
             ml::core::CTimeUtils::toTimeString(start));

    size_t count(0);
    while (count < FILL_SIZE)
    {
        ++count;
        testMultiIndex.insert(SContent(count));
    }

    while (count < TEST_SIZE)
    {
        testMultiIndex.erase(testMultiIndex.begin());
        ++count;
        testMultiIndex.insert(SContent(count));
    }

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO("Finished multi-index throughput test at " <<
             ml::core::CTimeUtils::toTimeString(end));

    CPPUNIT_ASSERT_EQUAL(FILL_SIZE, testMultiIndex.size());

    LOG_INFO("Multi-index throughput test with fill size " << FILL_SIZE <<
             " and test size " << TEST_SIZE << " took " << (end - start) <<
             " seconds");
}

CContainerThroughputTest::SContent::SContent(size_t count)
    : s_Size(count),
      s_Ptr(this),
      s_Double(double(count))
{
}

