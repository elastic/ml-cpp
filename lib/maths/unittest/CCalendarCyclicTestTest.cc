/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

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

#include <boost/optional.hpp>
#include <boost/range.hpp>
#include <boost/test/unit_test.hpp>

#include <vector>

BOOST_AUTO_TEST_SUITE(CCalendarCyclicTestTest)

using namespace ml;

namespace {
using TDoubleVec = std::vector<double>;
using TOptionalFeature = maths::CCalendarCyclicTest::TOptionalFeature;

const core_t::TTime HALF_HOUR{core::constants::HOUR / 2};
const core_t::TTime DAY{core::constants::DAY};
const core_t::TTime MONTH{4 * core::constants::WEEK};
const core_t::TTime YEAR{core::constants::YEAR};
}

BOOST_AUTO_TEST_CASE(testTruePositives) {
    // Test the true positive rate for a variety of different features.

    test::CRandomNumbers rng;

    double truePositive{0.0};
    double falsePositive{0.0};
    double falseNegative{0.0};

    LOG_DEBUG(<< "Day of month");

    for (std::size_t t = 0; t < 10; ++t) {
        // Repeated error on the second day of the month.
        core_t::TTime months[]{
            86400,    // 2nd Jan
            2764800,  // 2nd Feb
            5184000,  // 2nd Mar
            7862400,  // 2nd Apr
            10454400, // 2nd May
            13132800, // 2nd June
            15724800, // 2nd July
            18403200, // 2nd Aug
            21081600, // 2nd Sep
            23673600  // 2nd Oct
        };
        core_t::TTime end = months[boost::size(months) - 1] + 86400;

        maths::CCalendarCyclicTest cyclic(HALF_HOUR);

        TDoubleVec error;
        for (core_t::TTime time = 0; time <= end; time += HALF_HOUR) {
            ptrdiff_t i = maths::CTools::truncate(
                std::lower_bound(std::begin(months), std::end(months), time) -
                    std::begin(months),
                ptrdiff_t(1), ptrdiff_t(boost::size(months)));

            rng.generateNormalSamples(0.0, 9.0, 1, error);
            if (time >= months[i - 1] + 30000 && time < months[i - 1] + 50000) {
                TDoubleVec multiplier;
                rng.generateUniformSamples(4.0, 6.0, 1, multiplier);
                error[0] *= multiplier[0];
            }
            cyclic.add(time, error[0]);

            if (time > 121 * DAY && time % DAY == 0) {
                TOptionalFeature feature{cyclic.test()};
                if (feature == boost::none) {
                    falseNegative += 1.0;
                } else {
                    (core::CContainerPrinter::print(feature) == "2nd day of month"
                         ? truePositive
                         : falsePositive) += 1.0;
                }
            }
            BOOST_TEST(core::CMemory::dynamicSize(&cyclic) < 700);
        }
    }
    LOG_DEBUG(<< "true positive = " << truePositive);
    LOG_DEBUG(<< "false negative = " << falseNegative);
    LOG_DEBUG(<< "false positive = " << falsePositive);

    LOG_DEBUG(<< "Days before end of month");
    for (std::size_t t = 0; t < 10; ++t) {
        // Repeated error on the last day of the month.
        core_t::TTime months[]{
            2592000,  // 31st Jan
            5011200,  // 28th Feb
            7689600,  // 31st Mar
            10281600, // 30th Apr
            12960000, // 31st May
            15552000, // 30th June
            18230400  // 31st July
        };
        core_t::TTime end = months[boost::size(months) - 1] + 86400;

        maths::CCalendarCyclicTest cyclic(HALF_HOUR);

        TDoubleVec error;
        for (core_t::TTime time = 0; time <= end; time += HALF_HOUR) {
            ptrdiff_t i = maths::CTools::truncate(
                std::lower_bound(std::begin(months), std::end(months), time) -
                    std::begin(months),
                ptrdiff_t(1), ptrdiff_t(boost::size(months)));

            rng.generateNormalSamples(0.0, 9.0, 1, error);
            if (time >= months[i - 1] + 10000 && time < months[i - 1] + 20000) {
                error[0] -= 15.0;
            }
            cyclic.add(time, error[0]);

            if (time > 121 * DAY && time % DAY == 0) {
                TOptionalFeature feature = cyclic.test();
                if (feature == boost::none) {
                    falseNegative += 1.0;
                } else {
                    (core::CContainerPrinter::print(feature) == "0 days before end of month"
                         ? truePositive
                         : falsePositive) += 1.0;
                }
            }
            BOOST_TEST(core::CMemory::dynamicSize(&cyclic) < 700);
        }
    }
    LOG_DEBUG(<< "true positive = " << truePositive);
    LOG_DEBUG(<< "false negative = " << falseNegative);
    LOG_DEBUG(<< "false positive = " << falsePositive);

    LOG_DEBUG(<< "Day of week week of month");
    for (std::size_t t = 0; t < 10; ++t) {
        // Repeated error on first Monday of each month.
        core_t::TTime months[]{
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
                std::lower_bound(std::begin(months), std::end(months), time) -
                    std::begin(months),
                ptrdiff_t(1), ptrdiff_t(boost::size(months)));

            rng.generateNormalSamples(0.0, 9.0, 1, error);
            if (time >= months[i - 1] + 45000 && time < months[i - 1] + 60000) {
                error[0] += 12.0;
            }
            cyclic.add(time, error[0]);

            if (time > 121 * DAY && time % DAY == 0) {
                TOptionalFeature feature = cyclic.test();
                if (feature == boost::none) {
                    falseNegative += 1.0;
                } else {
                    (core::CContainerPrinter::print(feature) == "1st Monday of month"
                         ? truePositive
                         : falsePositive) += 1.0;
                }
            }
            BOOST_TEST(core::CMemory::dynamicSize(&cyclic) < 700);
        }
    }
    LOG_DEBUG(<< "true positive = " << truePositive);
    LOG_DEBUG(<< "false negative = " << falseNegative);
    LOG_DEBUG(<< "false positive = " << falsePositive);

    LOG_DEBUG(<< "Day of week weeks before end of month");
    for (std::size_t t = 0; t < 10; ++t) {
        // Repeated error on last Friday of each month.
        core_t::TTime months[]{
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
                std::lower_bound(std::begin(months), std::end(months), time) -
                    std::begin(months),
                ptrdiff_t(1), ptrdiff_t(boost::size(months)));

            rng.generateNormalSamples(0.0, 9.0, 1, error);
            if (time >= months[i - 1] + 45000 && time < months[i - 1] + 60000) {
                error[0] -= 12.0;
            }
            cyclic.add(time, error[0]);

            if (time > 121 * DAY && time % DAY == 0) {
                TOptionalFeature feature = cyclic.test();
                if (feature == boost::none) {
                    falseNegative += 1.0;
                } else {
                    (core::CContainerPrinter::print(feature) == "0 Fridays before end of month"
                         ? truePositive
                         : falsePositive) += 1.0;
                }
            }
            BOOST_TEST(core::CMemory::dynamicSize(&cyclic) < 700);
        }
    }
    LOG_DEBUG(<< "true positive = " << truePositive);
    LOG_DEBUG(<< "false negative = " << falseNegative);
    LOG_DEBUG(<< "false positive = " << falsePositive);

    double accuracy{(truePositive / (truePositive + falseNegative + falsePositive))};
    LOG_DEBUG(<< "accuracy = " << accuracy);
    BOOST_TEST(accuracy > 0.9);
}

BOOST_AUTO_TEST_CASE(testFalsePositives) {
    // Test a false positive rates under a variety of noise characteristics.

    test::CRandomNumbers rng;

    double trueNegatives{0.0};
    double falsePositives{0.0};

    LOG_DEBUG(<< "Normal");
    for (std::size_t t = 0; t < 10; ++t) {
        LOG_DEBUG(<< "test = " << t + 1);

        maths::CCalendarCyclicTest cyclic(HALF_HOUR);

        TDoubleVec error;
        for (core_t::TTime time = 0; time <= YEAR; time += HALF_HOUR) {
            rng.generateNormalSamples(0.0, 9.0, 1, error);
            cyclic.add(time, error[0]);

            if (time % MONTH == 0) {
                TOptionalFeature feature{cyclic.test()};
                (feature == boost::none ? trueNegatives : falsePositives) += 1.0;
                if (feature != boost::none) {
                    LOG_DEBUG(<< "Detected = " << feature->print());
                }
                BOOST_TEST(core::CMemory::dynamicSize(&cyclic) < 820);
            }
        }
    }
    LOG_DEBUG(<< "true negatives  = " << trueNegatives);
    LOG_DEBUG(<< "false positives = " << falsePositives);

    LOG_DEBUG(<< "Log-normal");
    for (std::size_t t = 0; t < 10; ++t) {
        LOG_DEBUG(<< "test = " << t + 1);

        maths::CCalendarCyclicTest cyclic(HALF_HOUR);

        TDoubleVec error;
        for (core_t::TTime time = 0; time <= YEAR; time += HALF_HOUR) {
            rng.generateLogNormalSamples(1.0, 2.0, 1, error);
            cyclic.add(time, error[0]);

            if (time % MONTH == 0) {
                TOptionalFeature feature{cyclic.test()};
                (feature == boost::none ? trueNegatives : falsePositives) += 1.0;
                if (feature != boost::none) {
                    LOG_DEBUG(<< "Detected = " << feature->print());
                }
                BOOST_TEST(core::CMemory::dynamicSize(&cyclic) < 830);
            }
        }
    }
    LOG_DEBUG(<< "true negatives  = " << trueNegatives);
    LOG_DEBUG(<< "false positives = " << falsePositives);

    LOG_DEBUG(<< "Mixture");
    for (std::size_t t = 0; t < 10; ++t) {
        LOG_DEBUG(<< "test = " << t + 1);

        maths::CCalendarCyclicTest cyclic(HALF_HOUR);

        TDoubleVec error;
        TDoubleVec uniform01;
        for (core_t::TTime time = 0; time <= YEAR; time += HALF_HOUR) {
            rng.generateUniformSamples(0.0, 1.0, 1, uniform01);
            rng.generateNormalSamples(uniform01[0] < 0.3 ? 0.0 : 15.0, 9.0, 1, error);
            cyclic.add(time, error[0]);

            if (time % MONTH == 0) {
                TOptionalFeature feature{cyclic.test()};
                (feature == boost::none ? trueNegatives : falsePositives) += 1.0;
                if (feature != boost::none) {
                    LOG_DEBUG(<< "Detected = " << feature->print());
                }
                BOOST_TEST(core::CMemory::dynamicSize(&cyclic) < 830);
            }
        }
    }
    LOG_DEBUG(<< "true negatives  = " << trueNegatives);
    LOG_DEBUG(<< "false positives = " << falsePositives);

    double accuracy{trueNegatives / (falsePositives + trueNegatives)};
    LOG_DEBUG(<< "accuracy = " << accuracy);
    BOOST_TEST(accuracy > 0.99);
}

BOOST_AUTO_TEST_CASE(testPersist) {
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
        BOOST_TEST(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);
        BOOST_TEST(traverser.traverseSubLevel(
            std::bind(&maths::CCalendarCyclicTest::acceptRestoreTraverser,
                      &restored, std::placeholders::_1)));
    }
    BOOST_CHECK_EQUAL(orig.checksum(), restored.checksum());

    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restored.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    BOOST_CHECK_EQUAL(origXml, newXml);
}

BOOST_AUTO_TEST_SUITE_END()
