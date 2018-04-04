/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CStatisticsTest.h"

#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CRegex.h>
#include <core/CSleep.h>
#include <core/CStatistics.h>
#include <core/CThread.h>

#include <stdint.h>


namespace
{

const int TEST_STAT = 0u;

class CStatisticsTestRunner : public ml::core::CThread
{
public:
    CStatisticsTestRunner() : m_I(0), m_N(0)
    {
    }

    void initialise(int i, int n)
    {
        m_N = n;
        m_I = i;
    }

private:
    virtual void run()
    {
        if (m_I < 6)
        {
            ml::core::CStatistics::stat(TEST_STAT + m_I).increment();
        }
        else
        {
            ml::core::CStatistics::stat(TEST_STAT + m_I - m_N).decrement();
        }
    }

    virtual void shutdown()
    {
    }

    int m_I;
    int m_N;
};

} // namespace

CppUnit::Test *CStatisticsTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CStatisticsTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CStatisticsTest>(
                                   "CStatisticsTest::testStatistics",
                                   &CStatisticsTest::testStatistics) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CStatisticsTest>(
                                   "CStatisticsTest::testPersist",
                                   &CStatisticsTest::testPersist) );

    return suiteOfTests;
}

void CStatisticsTest::testStatistics()
{
    LOG_TRACE("Starting Statistics test");
    ml::core::CStatistics &stats = ml::core::CStatistics::instance();

    static const int N = 6;
    for (int i = 0; i < N; i++)
    {
        CPPUNIT_ASSERT_EQUAL(uint64_t(0), stats.stat(TEST_STAT + i).value());
    }

    stats.stat(TEST_STAT).increment();
    CPPUNIT_ASSERT_EQUAL(uint64_t(1), stats.stat(TEST_STAT).value());
    stats.stat(TEST_STAT).increment();
    CPPUNIT_ASSERT_EQUAL(uint64_t(2), stats.stat(TEST_STAT).value());
    stats.stat(TEST_STAT).decrement();
    CPPUNIT_ASSERT_EQUAL(uint64_t(1), stats.stat(TEST_STAT).value());
    stats.stat(TEST_STAT).decrement();
    CPPUNIT_ASSERT_EQUAL(uint64_t(0), stats.stat(TEST_STAT).value());

    CStatisticsTestRunner runners[N * 2];
    for (int i = 0; i < N * 2; i++)
    {
        runners[i].initialise(i, N);
    }

    for (int i = 0; i < N * 2; i++)
    {
        runners[i].start();
    }

    for (int i = 0; i < N * 2; i++)
    {
        runners[i].waitForFinish();
    }

    for (int i = 0; i < N; i++)
    {
        CPPUNIT_ASSERT_EQUAL(uint64_t(0), stats.stat(TEST_STAT + i).value());
    }

    for (int i = 0; i < 0x1000000; i++)
    {
        stats.stat(TEST_STAT).increment();
    }
    CPPUNIT_ASSERT_EQUAL(uint64_t(0x1000000), stats.stat(TEST_STAT).value());

    LOG_TRACE("Finished Statistics test");
}

void CStatisticsTest::testPersist()
{
    LOG_DEBUG("Starting persist test");
    ml::core::CStatistics &stats = ml::core::CStatistics::instance();

    // Check that a save/restore with all zeros is Ok
    for (int i = 0; i < ml::stat_t::E_LastEnumStat; i++)
    {
        stats.stat(i).set(0);
    }

    std::string origStaticsXml;
    {
        ml::core::CRapidXmlStatePersistInserter inserter("root");
        stats.staticsAcceptPersistInserter(inserter);
        inserter.toXml(origStaticsXml);
    }

    {
        ml::core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origStaticsXml));
        ml::core::CRapidXmlStateRestoreTraverser traverser(parser);
        CPPUNIT_ASSERT(traverser.traverseSubLevel(
            &ml::core::CStatistics::staticsAcceptRestoreTraverser));
    }

    for (int i = 0; i < ml::stat_t::E_LastEnumStat; i++)
    {
        CPPUNIT_ASSERT_EQUAL(uint64_t(0), stats.stat(i).value());
    }

    // Set some other values and check that restore puts all to zero
    for (int i = 0; i < ml::stat_t::E_LastEnumStat; i++)
    {
        stats.stat(i).set(567 + (i * 3));
    }
    for (int i = 0; i < ml::stat_t::E_LastEnumStat; i++)
    {
        CPPUNIT_ASSERT_EQUAL(uint64_t(567 + (i * 3)), stats.stat(i).value());
    }

    // Save this state
    std::string newStaticsXml;
    {
        ml::core::CRapidXmlStatePersistInserter inserter("root");
        stats.staticsAcceptPersistInserter(inserter);
        inserter.toXml(newStaticsXml);
    }

    // Restore the original all-zero state
    {
        ml::core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origStaticsXml));
        ml::core::CRapidXmlStateRestoreTraverser traverser(parser);
        CPPUNIT_ASSERT(traverser.traverseSubLevel(
            &ml::core::CStatistics::staticsAcceptRestoreTraverser));
    }

    for (int i = 0; i < ml::stat_t::E_LastEnumStat; i++)
    {
        CPPUNIT_ASSERT_EQUAL(uint64_t(0), stats.stat(i).value());
    }

    // Restore the non-zero state
    {
        ml::core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(newStaticsXml));
        ml::core::CRapidXmlStateRestoreTraverser traverser(parser);
        CPPUNIT_ASSERT(traverser.traverseSubLevel(
            &ml::core::CStatistics::staticsAcceptRestoreTraverser));
    }

    for (int i = 0; i < ml::stat_t::E_LastEnumStat; i++)
    {
        CPPUNIT_ASSERT_EQUAL(uint64_t(567 + (i * 3)), stats.stat(i).value());
    }

    // Restore the zero state to clean up
    {
        ml::core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origStaticsXml));
        ml::core::CRapidXmlStateRestoreTraverser traverser(parser);
        CPPUNIT_ASSERT(traverser.traverseSubLevel(
            &ml::core::CStatistics::staticsAcceptRestoreTraverser));
    }

    for (int i = 0; i < ml::stat_t::E_LastEnumStat; i++)
    {
        CPPUNIT_ASSERT_EQUAL(uint64_t(0), stats.stat(i).value());
    }

    std::ostringstream ss;
    ss << stats;
    const std::string output(ss.str());
    ml::core::CRegex::TStrVec tokens;
    {
        ml::core::CRegex regex;
        regex.init("\n");
        regex.split(output, tokens);
    }
    for (ml::core::CRegex::TStrVecCItr i = tokens.begin(); i != (tokens.end() - 1); ++i)
    {
        ml::core::CRegex regex;
        // Look for "name":"E.*"value": 0}
        regex.init(".*\"name\":\"E.*\"value\":0.*");
        CPPUNIT_ASSERT(regex.matches(*i));
    }

    LOG_DEBUG(output);
}
