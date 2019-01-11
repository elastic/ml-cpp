/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CProgramCountersTest.h"

#include <core/CLogger.h>
#include <core/CProgramCounters.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CRegex.h>
#include <core/CSleep.h>
#include <core/CThread.h>

#include <stdint.h>

namespace {

const int TEST_COUNTER = 0u;

class CProgramCountersTestRunner : public ml::core::CThread {
public:
    CProgramCountersTestRunner() : m_I(0), m_N(0) {}

    void initialise(int i, int n) {
        m_N = n;
        m_I = i;
    }

private:
    virtual void run() {
        if (m_I < 6) {
            ml::core::CProgramCounters::counter(TEST_COUNTER + m_I)++;
        } else {
            ml::core::CProgramCounters::counter(TEST_COUNTER + m_I - m_N)--;
        }
    }

    virtual void shutdown() {}

    int m_I;
    int m_N;
};

} // namespace

CppUnit::Test* CProgramCountersTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CProgramCountersTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CProgramCountersTest>(
        "CProgramCountersTest::testCounters", &CProgramCountersTest::testCounters));
    suiteOfTests->addTest(new CppUnit::TestCaller<CProgramCountersTest>(
        "CProgramCountersTest::testPersist", &CProgramCountersTest::testPersist));
    suiteOfTests->addTest(new CppUnit::TestCaller<CProgramCountersTest>(
        "CProgramCountersTest::testCacheCounters", &CProgramCountersTest::testCacheCounters));

    return suiteOfTests;
}

void CProgramCountersTest::testCounters() {
    ml::core::CProgramCounters& counters = ml::core::CProgramCounters::instance();

    using TCounter = ml::core::CProgramCounters::TCounter;

    static const int N = 6;
    for (int i = 0; i < N; i++) {
        CPPUNIT_ASSERT_EQUAL(TCounter(0), counters.counter(TEST_COUNTER + i));
    }

    counters.counter(TEST_COUNTER)++;
    CPPUNIT_ASSERT_EQUAL(TCounter(1), counters.counter(TEST_COUNTER));
    counters.counter(TEST_COUNTER)++;
    CPPUNIT_ASSERT_EQUAL(TCounter(2), counters.counter(TEST_COUNTER));
    counters.counter(TEST_COUNTER)--;
    CPPUNIT_ASSERT_EQUAL(TCounter(1), counters.counter(TEST_COUNTER));
    counters.counter(TEST_COUNTER)--;
    CPPUNIT_ASSERT_EQUAL(TCounter(0), counters.counter(TEST_COUNTER));

    CProgramCountersTestRunner runners[N * 2];
    for (int i = 0; i < N * 2; i++) {
        runners[i].initialise(i, N);
    }

    for (int i = 0; i < N * 2; i++) {
        runners[i].start();
    }

    for (int i = 0; i < N * 2; i++) {
        runners[i].waitForFinish();
    }

    for (int i = 0; i < N; i++) {
        CPPUNIT_ASSERT_EQUAL(TCounter(0), counters.counter(TEST_COUNTER + i));
    }

    for (int i = 0; i < 0x1000000; i++) {
        counters.counter(TEST_COUNTER)++;
    }
    CPPUNIT_ASSERT_EQUAL(TCounter(0x1000000), counters.counter(TEST_COUNTER));
}

void CProgramCountersTest::testCacheCounters() {
    ml::core::CProgramCounters& counters = ml::core::CProgramCounters::instance();

    // confirm that initially the cache is empty
    CPPUNIT_ASSERT(counters.m_Cache.empty());

    // populate non-zero live counters
    for (int i = 0; i < ml::counter_t::E_LastEnumCounter; i++) {
        counters.counter(i) = (i + 1);
    }

    // copy the live values to the cache
    counters.cacheCounters();

    CPPUNIT_ASSERT(!counters.m_Cache.empty());

    // check that the cached and live counters match and that the values are as expected
    for (int i = 0; i < ml::counter_t::E_LastEnumCounter; i++) {
        CPPUNIT_ASSERT_EQUAL(counters.counter(i).load(), counters.m_Cache[i]);
        CPPUNIT_ASSERT_EQUAL(uint64_t(i + 1), counters.m_Cache[i]);
    }

    // Take a local copy of the cached counters
    ml::core::CProgramCounters::TUInt64Vec origCache = counters.m_Cache;

    // increment the live counters
    for (int i = 0; i < ml::counter_t::E_LastEnumCounter; i++) {
        counters.counter(i)++;
    }

    // compare with the cached values, they should have not changed
    for (int i = 0; i < ml::counter_t::E_LastEnumCounter; i++) {
        CPPUNIT_ASSERT_EQUAL(uint64_t(1), counters.counter(i) - counters.m_Cache[i]);
        CPPUNIT_ASSERT_EQUAL(origCache[i], counters.m_Cache[i]);
    }
}

void CProgramCountersTest::testPersist() {
    // Run the first set of checks without registering a specific subset of counters
	// in order to test the entire set now and in the future
    using TCounter = ml::core::CProgramCounters::TCounter;

    ml::core::CProgramCounters& counters = ml::core::CProgramCounters::instance();

    // Check that a save/restore with all zeros is Ok
    for (int i = 0; i < ml::counter_t::E_LastEnumCounter; i++) {
        counters.counter(i) = 0;
    }

    std::string origStaticsXml;
    {
        counters.cacheCounters();
        ml::core::CRapidXmlStatePersistInserter inserter("root");
        counters.staticsAcceptPersistInserter(inserter);
        inserter.toXml(origStaticsXml);
    }

    {
        ml::core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origStaticsXml));
        ml::core::CRapidXmlStateRestoreTraverser traverser(parser);
        CPPUNIT_ASSERT(traverser.traverseSubLevel(
            &ml::core::CProgramCounters::staticsAcceptRestoreTraverser));
    }

    for (int i = 0; i < ml::counter_t::E_LastEnumCounter; i++) {
        CPPUNIT_ASSERT_EQUAL(TCounter(0), counters.counter(i));
    }

    // Set some other values and check that restore puts all to zero
    for (int i = 0; i < ml::counter_t::E_LastEnumCounter; i++) {
        counters.counter(i) = (567 + (i * 3));
        CPPUNIT_ASSERT_EQUAL(TCounter(567 + (i * 3)), counters.counter(i));
    }

    // Save this state, without first caching the live values
    std::string newStaticsXmlNoCaching;
    {
        ml::core::CRapidXmlStatePersistInserter inserter("root");
        counters.staticsAcceptPersistInserter(inserter);
        inserter.toXml(newStaticsXmlNoCaching);
    }

    // Save the state after updating the cache
    std::string newStaticsXml;
    {
        counters.cacheCounters();
        ml::core::CRapidXmlStatePersistInserter inserter("root");
        counters.staticsAcceptPersistInserter(inserter);
        inserter.toXml(newStaticsXml);
    }

    // we expect the persisted counters without first caching to be the
    // same as those from when we do cache the live values,
    // as persistence uses live values if the cache is not available
    CPPUNIT_ASSERT_EQUAL(newStaticsXml, newStaticsXmlNoCaching);

    // Restore the original all-zero state
    {
        ml::core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origStaticsXml));
        ml::core::CRapidXmlStateRestoreTraverser traverser(parser);
        CPPUNIT_ASSERT(traverser.traverseSubLevel(
            ml::core::CProgramCounters::staticsAcceptRestoreTraverser));
    }

    for (int i = 0; i < ml::counter_t::E_LastEnumCounter; i++) {
        CPPUNIT_ASSERT_EQUAL(TCounter(0), counters.counter(i));
    }

    // Restore the non-zero state
    {
        ml::core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(newStaticsXml));
        ml::core::CRapidXmlStateRestoreTraverser traverser(parser);
        CPPUNIT_ASSERT(traverser.traverseSubLevel(
            &ml::core::CProgramCounters::staticsAcceptRestoreTraverser));
    }

    for (int i = 0; i < ml::counter_t::E_LastEnumCounter; i++) {
        CPPUNIT_ASSERT_EQUAL(TCounter(567 + (i * 3)), counters.counter(i));
    }

    // Restore the zero state to clean up
    {
        ml::core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origStaticsXml));
        ml::core::CRapidXmlStateRestoreTraverser traverser(parser);
        CPPUNIT_ASSERT(traverser.traverseSubLevel(
            &ml::core::CProgramCounters::staticsAcceptRestoreTraverser));
    }

    // Check that the cache is automatically cleaned up
    CPPUNIT_ASSERT(counters.m_Cache.empty());

    for (int i = 0; i < ml::counter_t::E_LastEnumCounter; i++) {
        CPPUNIT_ASSERT_EQUAL(TCounter(0), counters.counter(i));
    }

    // check that the format of the output stream operator is as expected
    std::ostringstream ss;
    ss << counters;
    const std::string output(ss.str());

    ml::core::CRegex::TStrVec tokens;
    {
        ml::core::CRegex regex;
        regex.init("\n");
        regex.split(output, tokens);
    }
    for (ml::core::CRegex::TStrVecCItr i = tokens.begin(); i != (tokens.end() - 1); ++i) {
        ml::core::CRegex regex;
        // Look for "name":"E.*"value": 0}
        regex.init(".*\"name\":\"E.*\"value\":0.*");
        CPPUNIT_ASSERT(regex.matches(*i));
    }

    LOG_DEBUG(<< output);

    // Test persistence/restoration of a subset  of counters
    auto testCounterSubset = [&](const ml::counter_t::TCounterTypeSet& counterSet) {

        // Register interest in a subset of counters
        ml::core::CProgramCounters::registerProgramCounterTypes(counterSet);

        // Check that a save/restore with all zeros is Ok
        for (int i = 0; i < ml::counter_t::E_LastEnumCounter; i++) {
            counters.counter(i) = 0;
        }

        std::string origStaticsXmlSubset;
        {
            counters.cacheCounters();
            ml::core::CRapidXmlStatePersistInserter inserter("root");
            counters.staticsAcceptPersistInserter(inserter);
            inserter.toXml(origStaticsXmlSubset);
        }

        {
            ml::core::CRapidXmlParser parser;
            CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origStaticsXmlSubset));
            ml::core::CRapidXmlStateRestoreTraverser traverser(parser);
            CPPUNIT_ASSERT(traverser.traverseSubLevel(
                &ml::core::CProgramCounters::staticsAcceptRestoreTraverser));
        }

        for (int i = 0; i < ml::counter_t::E_LastEnumCounter; i++) {
            CPPUNIT_ASSERT_EQUAL(TCounter(0), counters.counter(i));
        }

        // set some non-zero values for the subset of counters
        for (const auto& counterType : counterSet) {
            counters.counter(counterType) = (567 + (counterType * 3));
        }

        // Save the state after updating the cache
        std::string newStaticsXmlSubset;
        {
            counters.cacheCounters();
            ml::core::CRapidXmlStatePersistInserter inserter("root");
            counters.staticsAcceptPersistInserter(inserter);
            inserter.toXml(newStaticsXmlSubset);
        }

        // Restore the original all-zero state
        {
            ml::core::CRapidXmlParser parser;
            CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origStaticsXmlSubset));
            ml::core::CRapidXmlStateRestoreTraverser traverser(parser);
            CPPUNIT_ASSERT(traverser.traverseSubLevel(
                ml::core::CProgramCounters::staticsAcceptRestoreTraverser));
        }

        for (int i = 0; i < ml::counter_t::E_LastEnumCounter; i++) {
            CPPUNIT_ASSERT_EQUAL(TCounter(0), counters.counter(i));
        }

        // Restore the non-zero state
        {
            ml::core::CRapidXmlParser parser;
            CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(newStaticsXmlSubset));
            ml::core::CRapidXmlStateRestoreTraverser traverser(parser);
            CPPUNIT_ASSERT(traverser.traverseSubLevel(
                &ml::core::CProgramCounters::staticsAcceptRestoreTraverser));
        }

        // confirm the restored values are as expected
        for (int i = 0; i < ml::counter_t::E_LastEnumCounter; i++) {
            const auto& itr =
                counterSet.find(static_cast<ml::counter_t::ECounterTypes>(i));
            if (itr != counterSet.end()) {
                CPPUNIT_ASSERT_EQUAL(TCounter(567 + (i * 3)), counters.counter(i));
            } else {
                CPPUNIT_ASSERT_EQUAL(TCounter(0), counters.counter(i));
            }
        }

        std::ostringstream counterStream;
        counterStream << counters;

        LOG_DEBUG(<< counterStream.str());

        return counterStream.str();
    };

    const ml::counter_t::TCounterTypeSet counterSetOrder1{
        ml::counter_t::E_TSADNumberNewPeopleNotAllowed,
        ml::counter_t::E_TSADNumberNewPeople,
        ml::counter_t::E_TSADNumberNewPeopleRecycled,
        ml::counter_t::E_TSADNumberApiRecordsHandled,
        ml::counter_t::E_TSADMemoryUsage,
        ml::counter_t::E_TSADNumberMemoryUsageChecks,
        ml::counter_t::E_TSADNumberMemoryUsageEstimates,
        ml::counter_t::E_TSADNumberNewAttributesNotAllowed,
        ml::counter_t::E_TSADNumberNewAttributes,
        ml::counter_t::E_TSADNumberNewAttributesRecycled,
        ml::counter_t::E_TSADNumberByFields,
        ml::counter_t::E_TSADNumberOverFields,
        ml::counter_t::E_TSADNumberMemoryLimitModelCreationFailures,
        ml::counter_t::E_TSADNumberPrunedItems};

    const ml::counter_t::TCounterTypeSet counterSetOrder2{
        ml::counter_t::E_TSADNumberNewPeopleNotAllowed,
        ml::counter_t::E_TSADNumberNewAttributesRecycled,
        ml::counter_t::E_TSADNumberNewPeople,
        ml::counter_t::E_TSADNumberNewPeopleRecycled,
        ml::counter_t::E_TSADNumberApiRecordsHandled,
        ml::counter_t::E_TSADNumberNewAttributesNotAllowed,
        ml::counter_t::E_TSADMemoryUsage,
        ml::counter_t::E_TSADNumberMemoryUsageChecks,
        ml::counter_t::E_TSADNumberMemoryUsageEstimates,
        ml::counter_t::E_TSADNumberNewAttributes,
        ml::counter_t::E_TSADNumberByFields,
        ml::counter_t::E_TSADNumberOverFields,
        ml::counter_t::E_TSADNumberMemoryLimitModelCreationFailures,
        ml::counter_t::E_TSADNumberPrunedItems};

    const std::string outputOrder1 = testCounterSubset(counterSetOrder1);
    const std::string outputOrder2 = testCounterSubset(counterSetOrder2);

    // check that the order in which the counters are given doesn't affect the order in which they are printed
    CPPUNIT_ASSERT_EQUAL(outputOrder1, outputOrder2);
}
