/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CMapPopulationTest.h"

#include <core/CLogger.h>
#include <core/CStringUtils.h>
#include <core/CTimeUtils.h>

#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/threadpool.hpp>

#include <algorithm>

#include <stdlib.h>

const size_t CMapPopulationTest::FILL_SIZE(20);
const size_t CMapPopulationTest::TEST_SIZE(200000);

CMapPopulationTest::CMapPopulationTest() : m_TestData(nullptr) {
}

CppUnit::Test* CMapPopulationTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CMapPopulationTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CMapPopulationTest>(
        "CMapPopulationTest::testMapInsertSpeed", &CMapPopulationTest::testMapInsertSpeed));

    return suiteOfTests;
}

void CMapPopulationTest::setUp() {
    // This class gets constructed once for every test, but by making the test
    // data static every test will use the same test data, which is important
    // for a fair comparison
    static const CTestData TEST_DATA(FILL_SIZE);
    m_TestData = &TEST_DATA;
}

CMapPopulationTest::CTestData::CTestData(size_t fillSize)
    // It's essential these vectors don't resize as the char pointers in the
    // last two vectors point into the contents of the strings in the first two,
    // so set the correct size when they're constructed
    : m_StringKeys(fillSize), m_StringVals(fillSize), m_CharPtrKeys(fillSize),
      m_CharPtrVals(fillSize) {
    // Set up test data such that each test uses identical data
    for (size_t index = 0; index < fillSize; ++index) {
        // Keys are 4 to 12 letters followed by a unique number
        for (int count = 4 + (::rand() % 9); count > 0; --count) {
            m_StringKeys[index] += char('a' + ::rand() % 26);
        }
        m_StringKeys[index] += ml::core::CStringUtils::typeToString(index);
        m_CharPtrKeys[index] = m_StringKeys[index].c_str();

        // Values are 16 to 32 printable ASCII characters in length
        for (int count = 16 + (::rand() % 17); count > 0; --count) {
            m_StringVals[index] += char(' ' + ::rand() % 95);
        }
        m_CharPtrVals[index] = m_StringVals[index].c_str();
    }

    for (size_t index = 0; index < fillSize; ++index) {
        LOG_DEBUG(<< "Test entry " << index << ": " << m_CharPtrKeys[index]
                  << " -> " << m_CharPtrVals[index]);
    }
}

const CMapPopulationTest::CTestData::TStrVec& CMapPopulationTest::CTestData::stringKeys() const {
    return m_StringKeys;
}

const CMapPopulationTest::CTestData::TStrVec& CMapPopulationTest::CTestData::stringVals() const {
    return m_StringVals;
}

const CMapPopulationTest::CTestData::TCharPVec&
CMapPopulationTest::CTestData::charPtrKeys() const {
    return m_CharPtrKeys;
}

const CMapPopulationTest::CTestData::TCharPVec&
CMapPopulationTest::CTestData::charPtrVals() const {
    return m_CharPtrVals;
}

void CMapPopulationTest::testMapInsertSpeed() {
    // Schedule all the other tests to be run in a thread pool - the number of
    // threads is chosen to be less than the number of cores so that the results
    // aren't skewed too much if other processes are running on the machine
    unsigned int numCpus(boost::thread::hardware_concurrency());
    boost::threadpool::pool tp(numCpus / 3 + 1);

    // Add tasks for all the tests to the pool
    tp.schedule(boost::bind(&CMapPopulationTest::testMapInsertStr, this));
    tp.schedule(boost::bind(&CMapPopulationTest::testMapInsertCharP, this));
    tp.schedule(boost::bind(&CMapPopulationTest::testMapOpSqBracStr, this));
    tp.schedule(boost::bind(&CMapPopulationTest::testMapOpSqBracCharP, this));
    tp.schedule(boost::bind(&CMapPopulationTest::testUMapInsertStr, this));
    tp.schedule(boost::bind(&CMapPopulationTest::testUMapInsertCharP, this));
    tp.schedule(boost::bind(&CMapPopulationTest::testUMapOpSqBracStr, this));
    tp.schedule(boost::bind(&CMapPopulationTest::testUMapOpSqBracCharP, this));

    //  Wait until all tests are finished
    tp.wait();
}

void CMapPopulationTest::testMapInsertStr() {
    TStrStrMapVec testVec(TEST_SIZE);

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Starting map insert string test at "
             << ml::core::CTimeUtils::toTimeString(start));

    this->addInsert(m_TestData->stringKeys(), m_TestData->stringVals(), testVec);

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Finished map insert string test at "
             << ml::core::CTimeUtils::toTimeString(end));

    LOG_INFO(<< "Map insert string test with fill size " << FILL_SIZE << " and test size "
             << TEST_SIZE << " took " << (end - start) << " seconds");
}

void CMapPopulationTest::testMapInsertCharP() {
    TStrStrMapVec testVec(TEST_SIZE);

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Starting map insert char pointer test at "
             << ml::core::CTimeUtils::toTimeString(start));

    this->addInsert(m_TestData->charPtrKeys(), m_TestData->charPtrVals(), testVec);

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Finished map insert char pointer test at "
             << ml::core::CTimeUtils::toTimeString(end));

    LOG_INFO(<< "Map insert char pointer test with fill size " << FILL_SIZE << " and test size "
             << TEST_SIZE << " took " << (end - start) << " seconds");
}

void CMapPopulationTest::testMapOpSqBracStr() {
    TStrStrMapVec testVec(TEST_SIZE);

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Starting map operator[] string test at "
             << ml::core::CTimeUtils::toTimeString(start));

    this->addOpSqBrac(m_TestData->stringKeys(), m_TestData->stringVals(), testVec);

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Finished map operator[] string test at "
             << ml::core::CTimeUtils::toTimeString(end));

    LOG_INFO(<< "Map operator[] string test with fill size " << FILL_SIZE << " and test size "
             << TEST_SIZE << " took " << (end - start) << " seconds");
}

void CMapPopulationTest::testMapOpSqBracCharP() {
    TStrStrMapVec testVec(TEST_SIZE);

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Starting map operator[] char pointer test at "
             << ml::core::CTimeUtils::toTimeString(start));

    this->addOpSqBrac(m_TestData->charPtrKeys(), m_TestData->charPtrVals(), testVec);

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Finished map operator[] char pointer test at "
             << ml::core::CTimeUtils::toTimeString(end));

    LOG_INFO(<< "Map operator[] char pointer test with fill size " << FILL_SIZE << " and test size "
             << TEST_SIZE << " took " << (end - start) << " seconds");
}

void CMapPopulationTest::testUMapInsertStr() {
    TStrStrUMapVec testVec(TEST_SIZE);

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Starting unordered map insert string test at "
             << ml::core::CTimeUtils::toTimeString(start));

    this->addInsert(m_TestData->stringKeys(), m_TestData->stringVals(), testVec);

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Finished unordered map insert string test at "
             << ml::core::CTimeUtils::toTimeString(end));

    LOG_INFO(<< "Unordered map insert string test with fill size " << FILL_SIZE << " and test size "
             << TEST_SIZE << " took " << (end - start) << " seconds");
}

void CMapPopulationTest::testUMapInsertCharP() {
    TStrStrUMapVec testVec(TEST_SIZE);

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Starting unordered map insert char pointer test at "
             << ml::core::CTimeUtils::toTimeString(start));

    this->addInsert(m_TestData->charPtrKeys(), m_TestData->charPtrVals(), testVec);

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Finished unordered map insert char pointer test at "
             << ml::core::CTimeUtils::toTimeString(end));

    LOG_INFO(<< "Unordered map insert char pointer test with fill size " << FILL_SIZE
             << " and test size " << TEST_SIZE << " took " << (end - start) << " seconds");
}

void CMapPopulationTest::testUMapOpSqBracStr() {
    TStrStrUMapVec testVec(TEST_SIZE);

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Starting unordered map operator[] string test at "
             << ml::core::CTimeUtils::toTimeString(start));

    this->addOpSqBrac(m_TestData->stringKeys(), m_TestData->stringVals(), testVec);

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Finished unordered map operator[] string test at "
             << ml::core::CTimeUtils::toTimeString(end));

    LOG_INFO(<< "Unordered map operator[] string test with fill size " << FILL_SIZE
             << " and test size " << TEST_SIZE << " took " << (end - start) << " seconds");
}

void CMapPopulationTest::testUMapOpSqBracCharP() {
    TStrStrUMapVec testVec(TEST_SIZE);

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Starting unordered map operator[] char pointer test at "
             << ml::core::CTimeUtils::toTimeString(start));

    this->addOpSqBrac(m_TestData->charPtrKeys(), m_TestData->charPtrVals(), testVec);

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Finished unordered map operator[] char pointer test at "
             << ml::core::CTimeUtils::toTimeString(end));

    LOG_INFO(<< "Unordered map operator[] char pointer test with fill size " << FILL_SIZE
             << " and test size " << TEST_SIZE << " took " << (end - start) << " seconds");
}

template<typename INPUT_CONTAINER, typename MAP_CONTAINER>
void CMapPopulationTest::addInsert(const INPUT_CONTAINER& keys,
                                   const INPUT_CONTAINER& values,
                                   MAP_CONTAINER& maps) const {
    for (typename MAP_CONTAINER::iterator iter = maps.begin(); iter != maps.end(); ++iter) {
        typename MAP_CONTAINER::value_type& map = *iter;

        size_t limit(std::min(keys.size(), values.size()));
        for (size_t index = 0; index < limit; ++index) {
            map.insert(typename MAP_CONTAINER::value_type::value_type(
                keys[index], values[index]));
        }

        CPPUNIT_ASSERT_EQUAL(limit, map.size());
    }
}

template<typename INPUT_CONTAINER, typename MAP_CONTAINER>
void CMapPopulationTest::addOpSqBrac(const INPUT_CONTAINER& keys,
                                     const INPUT_CONTAINER& values,
                                     MAP_CONTAINER& maps) const {
    for (typename MAP_CONTAINER::iterator iter = maps.begin(); iter != maps.end(); ++iter) {
        typename MAP_CONTAINER::value_type& map = *iter;

        size_t limit(std::min(keys.size(), values.size()));
        for (size_t index = 0; index < limit; ++index) {
            map[keys[index]] = values[index];
        }

        CPPUNIT_ASSERT_EQUAL(limit, map.size());
    }
}
