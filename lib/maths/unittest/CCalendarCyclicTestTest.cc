/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CCalendarCyclicTestTest.h"

#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CTimezone.h>
#include <core/Constants.h>
#include <core/CoreTypes.h>

#include <maths/CCalendarCyclicTest.h>
#include <maths/CTools.h>

#include <test/CRandomNumbers.h>
#include <test/CTimeSeriesTestData.h>

#include "TestUtils.h"

#include <boost/bind.hpp>
#include <boost/optional.hpp>
#include <boost/range.hpp>

#include <vector>

using namespace ml;

namespace {
using TDoubleVec = std::vector<double>;

const core_t::TTime HALF_HOUR{core::constants::HOUR / 2};
const core_t::TTime DAY{core::constants::DAY};
}

void CCalendarCyclicTestTest::testAccuracy() {
    using TOptionalFeature = maths::CCalendarCyclicTest::TOptionalFeature;

    core::CTimezone::instance().timezoneName("GMT");

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "Day of month");
    {
        // Repeated error on the second day of the month.

        core_t::TTime months[] = {
            86400,   // 2nd Jan
            2764800, // 2nd Feb
            5184000, // 2nd Mar
            7862400, // 2nd Apr
            10454400 // 2nd May
        };
        core_t::TTime end = months[boost::size(months) - 1] + 86400;

        maths::CCalendarCyclicTest cyclic(HALF_HOUR);

        TDoubleVec error;
        for (core_t::TTime time = 0; time <= end; time += HALF_HOUR) {
            ptrdiff_t i = maths::CTools::truncate(
                std::lower_bound(boost::begin(months), boost::end(months), time) -
                    boost::begin(months),
                ptrdiff_t(1), ptrdiff_t(boost::size(months)));

            rng.generateNormalSamples(0.0, 10.0, 1, error);
            if (time >= months[i - 1] + 30000 && time < months[i - 1] + 50000) {
                error[0] *= 5.0;
            }
            cyclic.add(time, error[0]);

            if (time > 121 * DAY && time % DAY == 0) {
                TOptionalFeature feature = cyclic.test();
                CPPUNIT_ASSERT_EQUAL(std::string("2nd day of month"),
                                     core::CContainerPrinter::print(feature));
            }
        }
    }

    LOG_DEBUG(<< "Days before end of month");
    {
        // Repeated error on the last day of the month.

        core_t::TTime months[] = {
            2592000,  // 31st Jan
            5011200,  // 28th Feb
            7689600,  // 31st Mar
            10281600, // 30th Apr
            12960000  // 31st May
        };
        core_t::TTime end = months[boost::size(months) - 1] + 86400;

        maths::CCalendarCyclicTest cyclic(HALF_HOUR);

        TDoubleVec error;
        for (core_t::TTime time = 0; time <= end; time += HALF_HOUR) {
            ptrdiff_t i = maths::CTools::truncate(
                std::lower_bound(boost::begin(months), boost::end(months), time) -
                    boost::begin(months),
                ptrdiff_t(1), ptrdiff_t(boost::size(months)));

            rng.generateNormalSamples(0.0, 10.0, 1, error);
            if (time >= months[i - 1] + 10000 && time < months[i - 1] + 20000) {
                error[0] += 12.0;
            }
            cyclic.add(time, error[0]);

            if (time > 121 * DAY && time % DAY == 0) {
                TOptionalFeature feature = cyclic.test();
                CPPUNIT_ASSERT_EQUAL(std::string("0 days before end of month"),
                                     core::CContainerPrinter::print(feature));
            }
        }
    }

    LOG_DEBUG(<< "Day of week week of month");
    {
        // Repeated error on first Monday of each month.

        core_t::TTime months[] = {
            345600,  // Mon 5th Jan
            2764800, // Mon 2nd Feb
            5184000, // Mon 2nd Mar
            8208000, // Mon 6th Apr
            10627200 // Mon 4th May
        };
        core_t::TTime end = months[boost::size(months) - 1] + 86400;

        maths::CCalendarCyclicTest cyclic(HALF_HOUR);

        TDoubleVec error;
        for (core_t::TTime time = 0; time <= end; time += HALF_HOUR) {
            ptrdiff_t i = maths::CTools::truncate(
                std::lower_bound(boost::begin(months), boost::end(months), time) -
                    boost::begin(months),
                ptrdiff_t(1), ptrdiff_t(boost::size(months)));

            rng.generateNormalSamples(0.0, 10.0, 1, error);
            if (time >= months[i - 1] + 45000 && time < months[i - 1] + 60000) {
                error[0] += 12.0;
            }
            cyclic.add(time, error[0]);

            if (time > 121 * DAY && time % DAY == 0) {
                TOptionalFeature feature = cyclic.test();
                CPPUNIT_ASSERT_EQUAL(std::string("1st Monday of month"),
                                     core::CContainerPrinter::print(feature));
            }
        }
    }

    LOG_DEBUG(<< "Day of week weeks before end of month");
    {
        // Repeated error on last Friday of each month.
        core_t::TTime months[] = {
            2505600, // Fri 30th Jan
            4924800, // Fri 27th Feb
            7344000, // Fri 27th Mar
            9763200, // Fri 24th Apr
            12787200 // Fri 29th May
        };
        core_t::TTime end = months[boost::size(months) - 1] + 86400;

        maths::CCalendarCyclicTest cyclic(HALF_HOUR);

        TDoubleVec error;
        for (core_t::TTime time = 0; time <= end; time += HALF_HOUR) {
            ptrdiff_t i = maths::CTools::truncate(
                std::lower_bound(boost::begin(months), boost::end(months), time) -
                    boost::begin(months),
                ptrdiff_t(1), ptrdiff_t(boost::size(months)));

            rng.generateNormalSamples(0.0, 10.0, 1, error);
            if (time >= months[i - 1] + 45000 && time < months[i - 1] + 60000) {
                error[0] += 12.0;
            }
            cyclic.add(time, error[0]);

            if (time > 121 * DAY && time % DAY == 0) {
                TOptionalFeature feature = cyclic.test();
                CPPUNIT_ASSERT_EQUAL(std::string("0 Fridays before end of month"),
                                     core::CContainerPrinter::print(feature));
            }
        }
    }
}

void CCalendarCyclicTestTest::testPersist() {
    // Check that persistence is idempotent.

    test::CRandomNumbers rng;

    maths::CCalendarCyclicTest orig(HALF_HOUR);

    TDoubleVec error;
    for (core_t::TTime time = 0; time <= 12787200; time += HALF_HOUR) {
        rng.generateNormalSamples(0.0, 10.0, 1, error);
        orig.add(time, error[0]);
    }

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        orig.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_TRACE(<< "XML representation:\n" << origXml);
    LOG_TRACE(<< "XML size:" << origXml.size());

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

CppUnit::Test* CCalendarCyclicTestTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CCalendarCyclicTestTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CCalendarCyclicTestTest>(
        "CCalendarCyclicTestTest::testAccuracy", &CCalendarCyclicTestTest::testAccuracy));
    suiteOfTests->addTest(new CppUnit::TestCaller<CCalendarCyclicTestTest>(
        "CCalendarCyclicTestTest::testPersist", &CCalendarCyclicTestTest::testPersist));

    return suiteOfTests;
}
