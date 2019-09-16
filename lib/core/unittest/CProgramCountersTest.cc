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

#include <test/CRandomNumbers.h>

#include <cstdint>
#include <thread>

CppUnit::Test* CProgramCountersTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CProgramCountersTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CProgramCountersTest>(
        "CProgramCountersTest::testCounters", &CProgramCountersTest::testCounters));
    suiteOfTests->addTest(new CppUnit::TestCaller<CProgramCountersTest>(
        "CProgramCountersTest::testPersist", &CProgramCountersTest::testPersist));
    suiteOfTests->addTest(new CppUnit::TestCaller<CProgramCountersTest>(
        "CProgramCountersTest::testCacheCounters", &CProgramCountersTest::testCacheCounters));
    suiteOfTests->addTest(new CppUnit::TestCaller<CProgramCountersTest>(
        "CProgramCountersTest::testUnknownCounter", &CProgramCountersTest::testUnknownCounter));
    suiteOfTests->addTest(new CppUnit::TestCaller<CProgramCountersTest>(
        "CProgramCountersTest::testMissingCounter", &CProgramCountersTest::testMissingCounter));
    suiteOfTests->addTest(new CppUnit::TestCaller<CProgramCountersTest>(
        "CProgramCountersTest::testMax", &CProgramCountersTest::testMax));
    return suiteOfTests;
}

void CProgramCountersTest::testCounters() {
    ml::core::CProgramCounters& counters = ml::core::CProgramCounters::instance();

    using TCounter = ml::core::CProgramCounters::TCounter;

    static const int N = 6;
    for (int i = 0; i < N; ++i) {
        CPPUNIT_ASSERT_EQUAL(TCounter(0), counters.counter(TEST_COUNTER + i));
    }

    ++counters.counter(TEST_COUNTER);
    CPPUNIT_ASSERT_EQUAL(TCounter(1), counters.counter(TEST_COUNTER));
    ++counters.counter(TEST_COUNTER);
    CPPUNIT_ASSERT_EQUAL(TCounter(2), counters.counter(TEST_COUNTER));
    --counters.counter(TEST_COUNTER);
    CPPUNIT_ASSERT_EQUAL(TCounter(1), counters.counter(TEST_COUNTER));
    --counters.counter(TEST_COUNTER);
    CPPUNIT_ASSERT_EQUAL(TCounter(0), counters.counter(TEST_COUNTER));

    CProgramCountersTestRunner runners[N * 2];
    for (int i = 0; i < N * 2; ++i) {
        runners[i].initialise(i, N);
    }

    for (int i = 0; i < N * 2; ++i) {
        runners[i].start();
    }

    for (int i = 0; i < N * 2; ++i) {
        runners[i].waitForFinish();
    }

    for (int i = 0; i < N; i++) {
        CPPUNIT_ASSERT_EQUAL(TCounter(0), counters.counter(TEST_COUNTER + i));
    }

    for (int i = 0; i < 0x1000000; ++i) {
        ++counters.counter(TEST_COUNTER);
    }
    CPPUNIT_ASSERT_EQUAL(TCounter(0x1000000), counters.counter(TEST_COUNTER));
}

void CProgramCountersTest::testUnknownCounter() {
    ml::core::CProgramCounters& counters = ml::core::CProgramCounters::instance();

    // Name of the log file to use. It must match the name specified
    // in testfiles/testLogWarnings.boost.log.ini
    const char* logFile = "test.log";

    std::remove(logFile);
    // start logging to a file at level WARN
    CPPUNIT_ASSERT(ml::core::CLogger::instance().reconfigureFromFile(
        "testfiles/testLogWarnings.boost.log.ini"));

    // Attempt to access a counter at an unknown/invalid index
    ++counters.counter(ml::counter_t::E_LastEnumCounter);

    // Revert to the default logger settings
    ml::core::CLogger::instance().reset();

    std::ifstream log(logFile);
    CPPUNIT_ASSERT(log.is_open());
    ml::core::CRegex regex;
    CPPUNIT_ASSERT(regex.init(".*Bad index.*"));
    std::string line;
    while (std::getline(log, line)) {
        LOG_INFO(<< "Got '" << line << "'");
        CPPUNIT_ASSERT(regex.matches(line));
    }
    log.close();
    std::remove(logFile);
}

void CProgramCountersTest::testMissingCounter() {
    // explicitly register interest in a particular set of counters
    const ml::counter_t::TCounterTypeSet counterSet{
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

    ml::core::CProgramCounters::registerProgramCounterTypes(counterSet);

    // Attempt to restore from an XML string that's missing all but 2 of the counters
    const std::string countersXml = "<root><a>0</a><b>618</b><a>18</a><b>621</b></root>";
    this->restore(countersXml);

    using TCounter = ml::core::CProgramCounters::TCounter;
    ml::core::CProgramCounters& counters = ml::core::CProgramCounters::instance();

    // We expect to see the restored counters having expected values and all other should have value 0
    for (size_t i = 0; i < ml::counter_t::NUM_COUNTERS; ++i) {
        if (i == 0) {
            CPPUNIT_ASSERT_EQUAL(TCounter(618), counters.counter(0));
        } else if (i == 18) {
            CPPUNIT_ASSERT_EQUAL(TCounter(621), counters.counter(18));
        } else {
            CPPUNIT_ASSERT_EQUAL(TCounter(0), counters.counter(i));
        }
    }
}

void CProgramCountersTest::testCacheCounters() {
    ml::core::CProgramCounters& counters = ml::core::CProgramCounters::instance();

    // confirm that initially the cache is empty
    CPPUNIT_ASSERT_EQUAL(true, counters.m_Cache.empty());

    // populate non-zero live counters
    for (size_t i = 0; i < ml::counter_t::NUM_COUNTERS; ++i) {
        counters.counter(i) = (i + 1);
    }

    // copy the live values to the cache
    counters.cacheCounters();

    CPPUNIT_ASSERT_EQUAL(false, counters.m_Cache.empty());

    // check that the cached and live counters match and that the values are as expected
    for (size_t i = 0; i < ml::counter_t::NUM_COUNTERS; ++i) {
        CPPUNIT_ASSERT_EQUAL(static_cast<std::uint64_t>(counters.counter(i)),
                             counters.m_Cache[i]);
        CPPUNIT_ASSERT_EQUAL(std::uint64_t(i + 1), counters.m_Cache[i]);
    }

    // Take a local copy of the cached counters
    ml::core::CProgramCounters::TUInt64Vec origCache = counters.m_Cache;

    // increment the live counters
    for (size_t i = 0; i < ml::counter_t::NUM_COUNTERS; ++i) {
        ++counters.counter(i);
    }

    // compare with the cached values, they should have not changed
    for (size_t i = 0; i < ml::counter_t::NUM_COUNTERS; ++i) {
        CPPUNIT_ASSERT_EQUAL(std::uint64_t(1), counters.counter(i) - counters.m_Cache[i]);
        CPPUNIT_ASSERT_EQUAL(origCache[i], counters.m_Cache[i]);
    }

    // Set all counters t 0
    for (size_t i = 0; i < ml::counter_t::NUM_COUNTERS; ++i) {
        counters.counter(i) = 0;
    }
}

void CProgramCountersTest::testPersist() {
    // Run the first set of checks without registering a specific subset of counters
    // in order to test the entire set now and in the future
    using TCounter = ml::core::CProgramCounters::TCounter;

    ml::core::CProgramCounters& counters = ml::core::CProgramCounters::instance();

    // Set some non-zero values
    for (size_t i = 0; i < ml::counter_t::NUM_COUNTERS; ++i) {
        counters.counter(i) = (567 + (i * 3));
        CPPUNIT_ASSERT_EQUAL(TCounter(567 + (i * 3)), counters.counter(i));
    }

    // Save this state, without first caching the live values
    std::string newStaticsXmlNoCaching = this->persist(false);

    // Save the state after updating the cache
    std::string newStaticsXml = this->persist();

    // we expect the persisted counters without first caching to be the
    // same as those from when we do cache the live values,
    // as persistence uses live values if the cache is not available
    CPPUNIT_ASSERT_EQUAL(newStaticsXml, newStaticsXmlNoCaching);

    // Restore the non-zero state
    this->restore(newStaticsXml);

    for (size_t i = 0; i < ml::counter_t::NUM_COUNTERS; ++i) {
        CPPUNIT_ASSERT_EQUAL(TCounter(567 + (i * 3)), counters.counter(i));
    }

    // Check that the cache is automatically cleaned up
    CPPUNIT_ASSERT(counters.m_Cache.empty());

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
    ml::core::CRegex regex;
    // Look for "name":"E.*"value": 0}
    regex.init(".*\"name\":\"E.*\"value\":.*");
    for (size_t i = 0; i < ml::counter_t::NUM_COUNTERS; ++i) {
        CPPUNIT_ASSERT(regex.matches(tokens[i]));
    }

    LOG_DEBUG(<< output);

    // reset counters to zero values
    for (size_t i = 0; i < ml::counter_t::NUM_COUNTERS; ++i) {
        counters.counter(i) = 0;
        CPPUNIT_ASSERT_EQUAL(TCounter(0), counters.counter(i));
    }
    // Test persistence/restoration of a subset  of counters
    auto testCounterSubset = [&](const ml::counter_t::TCounterTypeSet& counterSet) {

        // Register interest in a subset of counters
        ml::core::CProgramCounters::registerProgramCounterTypes(counterSet);

        // set some non-zero values for the subset of counters
        for (const auto& counterType : counterSet) {
            size_t value = static_cast<size_t>(counterType);
            counters.counter(counterType) = (567 + (value * 3));
            CPPUNIT_ASSERT_EQUAL(TCounter(567 + (value * 3)), counters.counter(counterType));
        }

        // Save the state after updating the cache
        std::string newStaticsXmlSubset = this->persist();

        // reset to zero values
        for (size_t i = 0; i < ml::counter_t::NUM_COUNTERS; ++i) {
            counters.counter(i) = 0;
            CPPUNIT_ASSERT_EQUAL(TCounter(0), counters.counter(i));
        }

        // Restore the non-zero state
        this->restore(newStaticsXmlSubset);

        // confirm the restored values are as expected
        for (size_t i = 0; i < ml::counter_t::NUM_COUNTERS; ++i) {
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

void CProgramCountersTest::testMax() {
    ml::test::CRandomNumbers rng;
    std::size_t m1{0}, m2{0};
    std::thread thread1{[&m1, &rng] {
        std::vector<std::size_t> samples;
        rng.generateUniformSamples(0, 100000, 1000, samples);
        for (auto sample : samples) {
            ml::core::CProgramCounters::counter(ml::counter_t::E_DFOEstimatedPeakMemoryUsage)
                .max(sample);
            m1 = std::max(m1, sample);
        }
    }};
    std::thread thread2{[&m2, &rng] {
        std::vector<std::size_t> samples;
        rng.generateUniformSamples(0, 100000, 1000, samples);
        for (auto sample : samples) {
            ml::core::CProgramCounters::counter(ml::counter_t::E_DFOEstimatedPeakMemoryUsage)
                .max(sample);
            m2 = std::max(m2, sample);
        }
    }};

    thread1.join();
    thread2.join();

    std::size_t expected{std::max(m1, m2)};
    std::size_t actual{ml::core::CProgramCounters::counter(
        ml::counter_t::E_DFOEstimatedPeakMemoryUsage)};
    CPPUNIT_ASSERT_EQUAL(expected, actual);
}

std::string CProgramCountersTest::persist(bool doCacheCounters) {
    ml::core::CProgramCounters& counters = ml::core::CProgramCounters::instance();

    if (doCacheCounters == true) {
        counters.cacheCounters();
    }
    ml::core::CRapidXmlStatePersistInserter inserter("root");
    counters.staticsAcceptPersistInserter(inserter);

    std::string staticsXml;
    inserter.toXml(staticsXml);

    return staticsXml;
}

void CProgramCountersTest::restore(const std::string& staticsXml) {
    ml::core::CRapidXmlParser parser;
    CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(staticsXml));
    ml::core::CRapidXmlStateRestoreTraverser traverser(parser);
    CPPUNIT_ASSERT(traverser.traverseSubLevel(
        &ml::core::CProgramCounters::staticsAcceptRestoreTraverser));
}
