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

#include "CTrendTestsTest.h"

#include <core/CoreTypes.h>
#include <core/CLogger.h>
#include <core/Constants.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CTimezone.h>

#include <maths/CTools.h>
#include <maths/CTrendTests.h>

#include <test/CRandomNumbers.h>
#include <test/CTimeSeriesTestData.h>

#include "TestUtils.h"

#include <boost/bind.hpp>
#include <boost/optional.hpp>
#include <boost/range.hpp>

#include <vector>

using namespace ml;

namespace
{
using TDoubleVec = std::vector<double>;
using TTimeVec = std::vector<core_t::TTime>;
using TTimeDoublePr = std::pair<core_t::TTime, double>;
using TTimeDoublePrVec = std::vector<TTimeDoublePr>;

const core_t::TTime HALF_HOUR = core::constants::HOUR / 2;
const core_t::TTime DAY       = core::constants::DAY;
const core_t::TTime WEEK      = core::constants::WEEK;
}

void CTrendTestsTest::testRandomizedPeriodicity(void)
{
    LOG_DEBUG("+----------------------------------------------+");
    LOG_DEBUG("|  CTrendTestsTest::testRandomizedPeriodicity  |");
    LOG_DEBUG("+----------------------------------------------+");

    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMeanVarAccumulator = maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using TMaxAccumulator = maths::CBasicStatistics::COrderStatisticsStack<double, 1, std::greater<double>>;
    using TFunction = double (*)(core_t::TTime);

    test::CRandomNumbers rng;

    TMeanAccumulator typeI;
    TMeanAccumulator typeII;
    for (std::size_t t = 0u; t < 5; ++t)
    {
        LOG_DEBUG("*** test = " << t << " ***");

        core_t::TTime time = 0;
        core_t::TTime day = 0;

        TDoubleVec samples;
        rng.generateLogNormalSamples(1.0, 4.0, 84000, samples);

        maths::CRandomizedPeriodicityTest::reset();

        maths::CRandomizedPeriodicityTest rtests[8];
        double falsePositives[3] = { 0.0, 0.0, 0.0 };
        double trueNegatives[3]  = { 0.0, 0.0, 0.0 };
        double truePositives[5]  = { 0.0, 0.0, 0.0, 0.0, 0.0 };
        double falseNegatives[5] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
        TMeanVarAccumulator timeToDetectionMoments[5];
        TMaxAccumulator timeToDetectionMax[5];
        core_t::TTime lastTruePositive[5] = { time, time, time, time, time };
        TFunction functions[] =
            {
                &constant,
                &ramp,
                &markov,
                &smoothDaily,
                &smoothWeekly,
                &spikeyDaily,
                &spikeyWeekly,
                &weekends
            };

        for (std::size_t i = 0u; i < samples.size(); ++i)
        {
            for (std::size_t j = 0u; j < boost::size(functions); ++j)
            {
                rtests[j].add(time, 600.0 * (functions[j])(time) + samples[i]);
            }
            if (time >= day + DAY)
            {
                for (std::size_t j = 0u; j < boost::size(rtests); ++j)
                {
                    if (j < 3)
                    {
                        (rtests[j].test() ? falsePositives[j] : trueNegatives[j]) += 1.0;
                    }
                    else
                    {
                        (rtests[j].test() ? truePositives[j - 3] : falseNegatives[j - 3]) += 1.0;
                        if (rtests[j].test())
                        {
                            timeToDetectionMoments[j - 3].add(time - lastTruePositive[j - 3]);
                            timeToDetectionMax[j - 3].add(static_cast<double>(time - lastTruePositive[j - 3]));
                            lastTruePositive[j - 3] = time;
                        }
                    }
                }
                day += DAY;
            }
            time += HALF_HOUR;
        }

        LOG_DEBUG("falsePositives = " << core::CContainerPrinter::print(falsePositives));
        LOG_DEBUG("trueNegatives = " << core::CContainerPrinter::print(trueNegatives));
        for (std::size_t i = 0u; i < boost::size(falsePositives); ++i)
        {
            CPPUNIT_ASSERT(falsePositives[i] / trueNegatives[i] < 0.1);
            typeI.add(falsePositives[i] / trueNegatives[i]);
        }
        LOG_DEBUG("truePositives = " << core::CContainerPrinter::print(truePositives));
        LOG_DEBUG("falseNegatives = " << core::CContainerPrinter::print(falseNegatives));
        for (std::size_t i = 0u; i < boost::size(falsePositives); ++i)
        {
            CPPUNIT_ASSERT(falseNegatives[i] / truePositives[i] < 0.2);
            typeII.add(falseNegatives[i] / truePositives[i]);
        }

        for (std::size_t i = 0u; i < boost::size(timeToDetectionMoments); ++i)
        {
            LOG_DEBUG("time to detect moments = " << timeToDetectionMoments[i]);
            LOG_DEBUG("maximum time to detect = " << timeToDetectionMax[i][0]);
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(timeToDetectionMoments[i]) < 1.5 * DAY);
            CPPUNIT_ASSERT(std::sqrt(maths::CBasicStatistics::variance(timeToDetectionMoments[i])) < 5 * DAY);
            CPPUNIT_ASSERT(timeToDetectionMax[i][0] <= 27 * WEEK);
        }
    }
    LOG_DEBUG("type I  = " << maths::CBasicStatistics::mean(typeI));
    LOG_DEBUG("type II = " << maths::CBasicStatistics::mean(typeII));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(typeI) < 0.015);
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(typeII) < 0.05);
}

void CTrendTestsTest::testCalendarCyclic(void)
{
    LOG_DEBUG("+---------------------------------------+");
    LOG_DEBUG("|  CTrendTestsTest::testCalendarCyclic  |");
    LOG_DEBUG("+---------------------------------------+");

    using TOptionalFeature = maths::CCalendarCyclicTest::TOptionalFeature;

    core::CTimezone::instance().timezoneName("GMT");

    test::CRandomNumbers rng;

    LOG_DEBUG("Day of month");
    {
        // Repeated error on the second day of the month.

        core_t::TTime months[] =
            {
                86400,   // 2nd Jan
                2764800, // 2nd Feb
                5184000, // 2nd Mar
                7862400, // 2nd Apr
                10454400 // 2nd May
            };
        core_t::TTime end = months[boost::size(months) - 1] + 86400;

        maths::CCalendarCyclicTest cyclic(HALF_HOUR);

        TDoubleVec error;
        for (core_t::TTime time = 0; time <= end; time += HALF_HOUR)
        {
            ptrdiff_t i = maths::CTools::truncate(std::lower_bound(boost::begin(months),
                                                                   boost::end(months),
                                                                   time) - boost::begin(months),
                                                  ptrdiff_t(1),
                                                  ptrdiff_t(boost::size(months)));

            rng.generateNormalSamples(0.0, 10.0, 1, error);
            if (time >= months[i - 1] + 30000 && time < months[i - 1] + 50000)
            {
                error[0] *= 5.0;
            }
            cyclic.add(time, error[0]);

            if (time > 121 * DAY && time % DAY == 0)
            {
                TOptionalFeature feature = cyclic.test();
                CPPUNIT_ASSERT_EQUAL(std::string("2nd day of month"),
                                     core::CContainerPrinter::print(feature));
            }
        }
    }

    LOG_DEBUG("Days before end of month");
    {
        // Repeated error on the last day of the month.

        core_t::TTime months[] =
            {
                2592000,  // 31st Jan
                5011200,  // 28th Feb
                7689600,  // 31st Mar
                10281600, // 30th Apr
                12960000  // 31st May
            };
        core_t::TTime end = months[boost::size(months) - 1] + 86400;

        maths::CCalendarCyclicTest cyclic(HALF_HOUR);

        TDoubleVec error;
        for (core_t::TTime time = 0; time <= end; time += HALF_HOUR)
        {
            ptrdiff_t i = maths::CTools::truncate(std::lower_bound(boost::begin(months),
                                                                   boost::end(months),
                                                                   time) - boost::begin(months),
                                                  ptrdiff_t(1),
                                                  ptrdiff_t(boost::size(months)));

            rng.generateNormalSamples(0.0, 10.0, 1, error);
            if (time >= months[i - 1] + 10000 && time < months[i - 1] + 20000)
            {
                error[0] += 12.0;
            }
            cyclic.add(time, error[0]);

            if (time > 121 * DAY && time % DAY == 0)
            {
                TOptionalFeature feature = cyclic.test();
                CPPUNIT_ASSERT_EQUAL(std::string("0 days before end of month"),
                                     core::CContainerPrinter::print(feature));
            }
        }
    }

    LOG_DEBUG("Day of week week of month");
    {
        // Repeated error on first Monday of each month.

        core_t::TTime months[] =
            {
                345600,  // Mon 5th Jan
                2764800, // Mon 2nd Feb
                5184000, // Mon 2nd Mar
                8208000, // Mon 6th Apr
                10627200 // Mon 4th May
            };
        core_t::TTime end = months[boost::size(months) - 1] + 86400;

        maths::CCalendarCyclicTest cyclic(HALF_HOUR);

        TDoubleVec error;
        for (core_t::TTime time = 0; time <= end; time += HALF_HOUR)
        {
            ptrdiff_t i = maths::CTools::truncate(std::lower_bound(boost::begin(months),
                                                                   boost::end(months),
                                                                   time) - boost::begin(months),
                                                  ptrdiff_t(1),
                                                  ptrdiff_t(boost::size(months)));

            rng.generateNormalSamples(0.0, 10.0, 1, error);
            if (time >= months[i - 1] + 45000 && time < months[i - 1] + 60000)
            {
                error[0] += 12.0;
            }
            cyclic.add(time, error[0]);

            if (time > 121 * DAY && time % DAY == 0)
            {
                TOptionalFeature feature = cyclic.test();
                CPPUNIT_ASSERT_EQUAL(std::string("1st Monday of month"),
                                     core::CContainerPrinter::print(feature));
            }
        }
    }

    LOG_DEBUG("Day of week weeks before end of month");
    {
        // Repeated error on last Friday of each month.
        core_t::TTime months[] =
            {
                2505600, // Fri 30th Jan
                4924800, // Fri 27th Feb
                7344000, // Fri 27th Mar
                9763200, // Fri 24th Apr
                12787200 // Fri 29th May
            };
        core_t::TTime end = months[boost::size(months) - 1] + 86400;

        maths::CCalendarCyclicTest cyclic(HALF_HOUR);

        TDoubleVec error;
        for (core_t::TTime time = 0; time <= end; time += HALF_HOUR)
        {
            ptrdiff_t i = maths::CTools::truncate(std::lower_bound(boost::begin(months),
                                                                   boost::end(months),
                                                                   time) - boost::begin(months),
                                                  ptrdiff_t(1),
                                                  ptrdiff_t(boost::size(months)));

            rng.generateNormalSamples(0.0, 10.0, 1, error);
            if (time >= months[i - 1] + 45000 && time < months[i - 1] + 60000)
            {
                error[0] += 12.0;
            }
            cyclic.add(time, error[0]);

            if (time > 121 * DAY && time % DAY == 0)
            {
                TOptionalFeature feature = cyclic.test();
                CPPUNIT_ASSERT_EQUAL(std::string("0 Fridays before end of month"),
                                     core::CContainerPrinter::print(feature));
            }
        }
    }
}

void CTrendTestsTest::testPersist(void)
{
    LOG_DEBUG("+--------------------------------+");
    LOG_DEBUG("|  CTrendTestsTest::testPersist  |");
    LOG_DEBUG("+--------------------------------+");

    // Check that persistence is idempotent.

    LOG_DEBUG("Test CRandomizedPeriodicityTest");
    {
        maths::CRandomizedPeriodicityTest test;
        for (core_t::TTime t = 1400000000; t < 1400050000; t += 5000)
        {
            test.add(t, 0.2);
        }

        std::string origXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            test.acceptPersistInserter(inserter);
            inserter.toXml(origXml);
        }

        std::string origStaticsXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            test.staticsAcceptPersistInserter(inserter);
            inserter.toXml(origStaticsXml);
        }

        // Check that the static state is also preserved
        uint64_t origNextRandom = test.ms_Rng();

        LOG_DEBUG("XML representation:\n" << origXml);

        // Restore the XML into a new test
        maths::CRandomizedPeriodicityTest test2;
        {
            core::CRapidXmlParser parser;
            CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
            core::CRapidXmlStateRestoreTraverser traverser(parser);
            CPPUNIT_ASSERT(traverser.traverseSubLevel(boost::bind(
                    &maths::CRandomizedPeriodicityTest::acceptRestoreTraverser, &test2, _1)));
        }
        std::string newXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            test2.acceptPersistInserter(inserter);
            inserter.toXml(newXml);
        }
        CPPUNIT_ASSERT_EQUAL(origXml, newXml);

        {
            core::CRapidXmlParser parser;
            CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origStaticsXml));
            core::CRapidXmlStateRestoreTraverser traverser(parser);
            CPPUNIT_ASSERT(traverser.traverseSubLevel(
                    &maths::CRandomizedPeriodicityTest::staticsAcceptRestoreTraverser));
        }
        std::string newStaticsXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            test2.staticsAcceptPersistInserter(inserter);
            inserter.toXml(newStaticsXml);
        }
        CPPUNIT_ASSERT_EQUAL(origStaticsXml, newStaticsXml);

        uint64_t newNextRandom = test2.ms_Rng();
        CPPUNIT_ASSERT_EQUAL(origNextRandom, newNextRandom);
    }

    LOG_DEBUG("Test CCalendarCyclicTest");
    {
        test::CRandomNumbers rng;

        maths::CCalendarCyclicTest orig(HALF_HOUR);

        TDoubleVec error;
        for (core_t::TTime time = 0; time <= 12787200; time += HALF_HOUR)
        {
            rng.generateNormalSamples(0.0, 10.0, 1, error);
            orig.add(time, error[0]);
        }

        std::string origXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            orig.acceptPersistInserter(inserter);
            inserter.toXml(origXml);
        }

        LOG_DEBUG("XML representation:\n" << origXml);

        maths::CCalendarCyclicTest restored(HALF_HOUR);
        {
            core::CRapidXmlParser parser;
            CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
            core::CRapidXmlStateRestoreTraverser traverser(parser);
            CPPUNIT_ASSERT(traverser.traverseSubLevel(boost::bind(
                    &maths::CCalendarCyclicTest::acceptRestoreTraverser, &restored, _1)));
        }
        CPPUNIT_ASSERT_EQUAL(orig.checksum(), restored.checksum());

        std::string newXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            restored.acceptPersistInserter(inserter);
            inserter.toXml(newXml);
        }
        CPPUNIT_ASSERT_EQUAL(origXml, newXml);
    }
}

CppUnit::Test *CTrendTestsTest::suite(void)
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CTrendTestsTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CTrendTestsTest>(
                                   "CTrendTestsTest::testRandomizedPeriodicity",
                                   &CTrendTestsTest::testRandomizedPeriodicity) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CTrendTestsTest>(
                                   "CTrendTestsTest::testCalendarCyclic",
                                   &CTrendTestsTest::testCalendarCyclic) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CTrendTestsTest>(
                                   "CTrendTestsTest::testPersist",
                                   &CTrendTestsTest::testPersist) );

    return suiteOfTests;
}
