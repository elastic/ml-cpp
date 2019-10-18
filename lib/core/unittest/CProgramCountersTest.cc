/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CProgramCounters.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CRegex.h>
#include <core/CSleep.h>
#include <core/CThread.h>

#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

#include <cstdint>
#include <fstream>
#include <thread>

BOOST_AUTO_TEST_SUITE(CProgramCountersTest)

const int TEST_COUNTER{0u};

class CTestFixture {
public:
    CTestFixture() {
        ml::core::CProgramCounters& counters = ml::core::CProgramCounters::instance();

        // Set all counters to 0
        for (std::size_t i = 0; i < ml::counter_t::NUM_COUNTERS; ++i) {
            counters.counter(i) = 0;
        }

        // Clear the cache
        counters.m_Cache.clear();
    }
};

class CProgramCountersTestRunner : public ml::core::CThread {
public:
    static const int N = 6;

public:
    CProgramCountersTestRunner() : m_I(0), m_N(0) {}

    void initialise(int i, int n) {
        m_I = i;
        m_N = n;
    }

private:
    virtual void run() {
        if (m_I < N) {
            ++ml::core::CProgramCounters::counter(TEST_COUNTER + m_I);
        } else {
            --ml::core::CProgramCounters::counter(TEST_COUNTER + m_I - m_N);
        }
    }

    virtual void shutdown() {}

    int m_I;
    int m_N;
};

std::string persist(bool shouldCacheCounters = true) {
    ml::core::CProgramCounters& counters = ml::core::CProgramCounters::instance();

    if (shouldCacheCounters) {
        counters.cacheCounters();
    }
    ml::core::CRapidXmlStatePersistInserter inserter("root");
    counters.staticsAcceptPersistInserter(inserter);

    std::string staticsXml;
    inserter.toXml(staticsXml);

    return staticsXml;
}

void restore(const std::string& staticsXml) {
    ml::core::CRapidXmlParser parser;
    BOOST_TEST(parser.parseStringIgnoreCdata(staticsXml));
    ml::core::CRapidXmlStateRestoreTraverser traverser(parser);
    BOOST_TEST(traverser.traverseSubLevel(&ml::core::CProgramCounters::staticsAcceptRestoreTraverser));
}

BOOST_FIXTURE_TEST_CASE(testCounters, CTestFixture) {
    ml::core::CProgramCounters& counters = ml::core::CProgramCounters::instance();

    using TCounter = ml::core::CProgramCounters::TCounter;

    for (int i = 0; i < CProgramCountersTestRunner::N; ++i) {
        BOOST_CHECK_EQUAL(TCounter(0), counters.counter(TEST_COUNTER + i));
    }

    ++counters.counter(TEST_COUNTER);
    BOOST_CHECK_EQUAL(TCounter(1), counters.counter(TEST_COUNTER));
    ++counters.counter(TEST_COUNTER);
    BOOST_CHECK_EQUAL(TCounter(2), counters.counter(TEST_COUNTER));
    --counters.counter(TEST_COUNTER);
    BOOST_CHECK_EQUAL(TCounter(1), counters.counter(TEST_COUNTER));
    --counters.counter(TEST_COUNTER);
    BOOST_CHECK_EQUAL(TCounter(0), counters.counter(TEST_COUNTER));

    CProgramCountersTestRunner runners[CProgramCountersTestRunner::N * 2];
    for (int i = 0; i < CProgramCountersTestRunner::N * 2; ++i) {
        runners[i].initialise(i, CProgramCountersTestRunner::N);
    }

    for (int i = 0; i < CProgramCountersTestRunner::N * 2; ++i) {
        runners[i].start();
    }

    for (int i = 0; i < CProgramCountersTestRunner::N * 2; ++i) {
        runners[i].waitForFinish();
    }

    for (int i = 0; i < CProgramCountersTestRunner::N; i++) {
        BOOST_CHECK_EQUAL(TCounter(0), counters.counter(TEST_COUNTER + i));
    }

    for (int i = 0; i < 0x1000000; ++i) {
        ++counters.counter(TEST_COUNTER);
    }
    BOOST_CHECK_EQUAL(TCounter(0x1000000), counters.counter(TEST_COUNTER));
}

BOOST_FIXTURE_TEST_CASE(testPersist, CTestFixture) {
    // Run the first set of checks without registering a specific subset of counters
    // in order to test the entire set now and in the future
    using TCounter = ml::core::CProgramCounters::TCounter;

    ml::core::CProgramCounters& counters = ml::core::CProgramCounters::instance();

    // confirm that initially the cache is empty
    BOOST_CHECK_EQUAL(true, counters.m_Cache.empty());

    // Set some non-zero values
    for (size_t i = 0; i < ml::counter_t::NUM_COUNTERS; ++i) {
        counters.counter(i) = (567 + (i * 3));
        BOOST_CHECK_EQUAL(TCounter(567 + (i * 3)), counters.counter(i));
    }

    // Save this state, without first caching the live values
    std::string newStaticsXmlNoCaching = persist(false);

    // Save the state after updating the cache
    std::string newStaticsXml = persist();

    // we expect the persisted counters without first caching to be the
    // same as those from when we do cache the live values,
    // as persistence uses live values if the cache is not available
    BOOST_CHECK_EQUAL(newStaticsXml, newStaticsXmlNoCaching);

    // Restore the non-zero state
    restore(newStaticsXml);

    for (size_t i = 0; i < ml::counter_t::NUM_COUNTERS; ++i) {
        BOOST_CHECK_EQUAL(TCounter(567 + (i * 3)), counters.counter(i));
    }

    // Check that the cache is automatically cleaned up
    BOOST_TEST(counters.m_Cache.empty());

    // check that the format of the output stream operator is as expected
    std::ostringstream ss;
    ss << counters;
    const std::string output(ss.str());
    LOG_INFO(<< output);

    ml::core::CRegex::TStrVec tokens;
    {
        ml::core::CRegex regex;
        regex.init("\n");
        regex.split(output, tokens);
    }
    ml::core::CRegex regex;
    // Look for "name":"E.*"value": 0}
    regex.init(".*\"name\":\"E.*\"value\":.*");
    BOOST_TEST(tokens.size() > ml::counter_t::NUM_COUNTERS);
    for (size_t i = 0; i < ml::counter_t::NUM_COUNTERS; ++i) {
        LOG_INFO(<< "checking line " << i);
        BOOST_TEST(regex.matches(tokens[i]));
    }

    LOG_DEBUG(<< output);

    // reset counters to zero values
    for (size_t i = 0; i < ml::counter_t::NUM_COUNTERS; ++i) {
        counters.counter(i) = 0;
        BOOST_CHECK_EQUAL(TCounter(0), counters.counter(i));
    }
    // Test persistence/restoration of a subset  of counters
    auto testCounterSubset = [&](const ml::counter_t::TCounterTypeSet& counterSet) {

        // Register interest in a subset of counters
        ml::core::CProgramCounters::registerProgramCounterTypes(counterSet);

        // set some non-zero values for the subset of counters
        for (const auto& counterType : counterSet) {
            size_t value = static_cast<size_t>(counterType);
            counters.counter(counterType) = (567 + (value * 3));
            BOOST_CHECK_EQUAL(TCounter(567 + (value * 3)), counters.counter(counterType));
        }

        // Save the state after updating the cache
        std::string newStaticsXmlSubset = persist();

        // reset to zero values
        for (size_t i = 0; i < ml::counter_t::NUM_COUNTERS; ++i) {
            counters.counter(i) = 0;
            BOOST_CHECK_EQUAL(TCounter(0), counters.counter(i));
        }

        // Restore the non-zero state
        restore(newStaticsXmlSubset);

        // confirm the restored values are as expected
        for (size_t i = 0; i < ml::counter_t::NUM_COUNTERS; ++i) {
            const auto& itr =
                counterSet.find(static_cast<ml::counter_t::ECounterTypes>(i));
            if (itr != counterSet.end()) {
                BOOST_CHECK_EQUAL(TCounter(567 + (i * 3)), counters.counter(i));
            } else {
                BOOST_CHECK_EQUAL(TCounter(0), counters.counter(i));
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
    BOOST_CHECK_EQUAL(outputOrder1, outputOrder2);
}

BOOST_FIXTURE_TEST_CASE(testUnknownCounter, CTestFixture) {
    ml::core::CProgramCounters& counters = ml::core::CProgramCounters::instance();

    // Name of the log file to use. It must match the name specified
    // in testfiles/testLogWarnings.boost.log.ini
    const char* logFile = "test.log";

    std::remove(logFile);
    // start logging to a file at level WARN
    BOOST_TEST(ml::core::CLogger::instance().reconfigureFromFile(
        "testfiles/testLogWarnings.boost.log.ini"));

    // Attempt to access a counter at an unknown/invalid index
    ++counters.counter(ml::counter_t::E_LastEnumCounter);

    // Revert to the default logger settings
    ml::core::CLogger::instance().reset();

    std::ifstream log(logFile);
    BOOST_TEST(log.is_open());
    ml::core::CRegex regex;
    BOOST_TEST(regex.init(".*Bad index.*"));
    std::string line;
    while (std::getline(log, line)) {
        LOG_INFO(<< "Got '" << line << "'");
        BOOST_TEST(regex.matches(line));
    }
    log.close();
    std::remove(logFile);
}

BOOST_FIXTURE_TEST_CASE(testMissingCounter, CTestFixture) {
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
    restore(countersXml);

    using TCounter = ml::core::CProgramCounters::TCounter;
    ml::core::CProgramCounters& counters = ml::core::CProgramCounters::instance();

    // We expect to see the restored counters having expected values and all other should have value 0
    for (size_t i = 0; i < ml::counter_t::NUM_COUNTERS; ++i) {
        if (i == 0) {
            BOOST_CHECK_EQUAL(TCounter(618), counters.counter(0));
        } else if (i == 18) {
            BOOST_CHECK_EQUAL(TCounter(621), counters.counter(18));
        } else {
            BOOST_CHECK_EQUAL(TCounter(0), counters.counter(i));
        }
    }
}

BOOST_FIXTURE_TEST_CASE(testCacheCounters, CTestFixture) {
    ml::core::CProgramCounters& counters = ml::core::CProgramCounters::instance();

    // confirm that initially the cache is empty
    BOOST_CHECK_EQUAL(true, counters.m_Cache.empty());

    // populate non-zero live counters
    for (size_t i = 0; i < ml::counter_t::NUM_COUNTERS; ++i) {
        counters.counter(i) = (i + 1);
    }

    // copy the live values to the cache
    counters.cacheCounters();

    BOOST_CHECK_EQUAL(false, counters.m_Cache.empty());

    // check that the cached and live counters match and that the values are as expected
    for (size_t i = 0; i < ml::counter_t::NUM_COUNTERS; ++i) {
        BOOST_CHECK_EQUAL(static_cast<std::uint64_t>(counters.counter(i)),
                          counters.m_Cache[i]);
        BOOST_CHECK_EQUAL(std::uint64_t(i + 1), counters.m_Cache[i]);
    }

    // Take a local copy of the cached counters
    ml::core::CProgramCounters::TUInt64Vec origCache = counters.m_Cache;

    // increment the live counters
    for (size_t i = 0; i < ml::counter_t::NUM_COUNTERS; ++i) {
        ++counters.counter(i);
    }

    // compare with the cached values, they should have not changed
    for (size_t i = 0; i < ml::counter_t::NUM_COUNTERS; ++i) {
        BOOST_CHECK_EQUAL(std::uint64_t(1), counters.counter(i) - counters.m_Cache[i]);
        BOOST_CHECK_EQUAL(origCache[i], counters.m_Cache[i]);
    }
}

BOOST_FIXTURE_TEST_CASE(testMax, CTestFixture) {
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
    BOOST_CHECK_EQUAL(expected, actual);
}

BOOST_AUTO_TEST_SUITE_END()
