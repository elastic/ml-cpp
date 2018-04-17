/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CPeriodicityHypothesisTestsTest.h"

#include <core/CLogger.h>
#include <core/Constants.h>
#include <core/CoreTypes.h>

#include <maths/CIntegerTools.h>
#include <maths/CPeriodicityHypothesisTests.h>

#include <test/CRandomNumbers.h>
#include <test/CTimeSeriesTestData.h>

#include "TestUtils.h"

#include <boost/math/constants/constants.hpp>

#include <cmath>
#include <vector>

using namespace ml;
using namespace handy_typedefs;

namespace {
using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;
using TTimeVec = std::vector<core_t::TTime>;
using TTimeDoublePr = std::pair<core_t::TTime, double>;
using TTimeDoublePrVec = std::vector<TTimeDoublePr>;
using TStrVec = std::vector<std::string>;

const core_t::TTime TEN_MINS{600};
const core_t::TTime HALF_HOUR{core::constants::HOUR / 2};
const core_t::TTime HOUR{core::constants::HOUR};
const core_t::TTime DAY{core::constants::DAY};
const core_t::TTime WEEK{core::constants::WEEK};
}

void CPeriodicityHypothesisTestsTest::testNonPeriodic() {
    LOG_DEBUG(<< "+----------------------------------------------------+");
    LOG_DEBUG(<< "|  CPeriodicityHypothesisTestsTest::testNonPeriodic  |");
    LOG_DEBUG(<< "+----------------------------------------------------+");

    // Test a variety of synthetic non-periodic signals.

    TTimeVec windows{WEEK, 2 * WEEK, 16 * DAY, 4 * WEEK};
    TTimeVec bucketLengths{TEN_MINS, HALF_HOUR};
    TGeneratorVec generators{constant, ramp, markov};

    test::CRandomNumbers rng;

    TDoubleVec noise;
    TSizeVec index;
    TSizeVec repeats;

    double FP{0.0};
    double TN{0.0};

    for (std::size_t test = 0u; test < 50; ++test) {
        if (test % 10 == 0) {
            LOG_DEBUG(<< "test " << test << " / 50");
        }
        for (auto window : windows) {
            for (auto bucketLength : bucketLengths) {
                switch (test % 3) {
                case 0:
                    rng.generateNormalSamples(0.0, 0.4, window / bucketLength, noise);
                    break;
                case 1:
                    rng.generateGammaSamples(1.0, 5.0, window / bucketLength, noise);
                    break;
                case 2:
                    rng.generateLogNormalSamples(0.2, 0.3, window / bucketLength, noise);
                    break;
                }
                rng.generateUniformSamples(0, generators.size(), 1, index);
                rng.generateUniformSamples(3, 20, 1, repeats);

                maths::CPeriodicityHypothesisTests hypotheses;
                hypotheses.initialize(bucketLength, window,
                                      window / static_cast<core_t::TTime>(repeats[0]));

                for (core_t::TTime time = 10000; time < 10000 + window; time += bucketLength) {
                    hypotheses.add(time, generators[index[0]](time) +
                                             noise[(time - 10000) / bucketLength]);
                }

                maths::CPeriodicityHypothesisTestsResult result{hypotheses.test()};
                if (result.periodic()) {
                    LOG_DEBUG(<< "result = " << result.print());
                }
                FP += result.periodic() ? 1.0 : 0.0;
                TN += result.periodic() ? 0.0 : 1.0;
            }
        }
    }

    LOG_DEBUG(<< "True negative rate = " << TN / (FP + TN));
    CPPUNIT_ASSERT(TN / (FP + TN) > 0.995);
}

void CPeriodicityHypothesisTestsTest::testDiurnal() {
    LOG_DEBUG(<< "+------------------------------------------------+");
    LOG_DEBUG(<< "|  CPeriodicityHypothesisTestsTest::testDiurnal  |");
    LOG_DEBUG(<< "+------------------------------------------------+");

    // Test the recall for a variety of synthetic periodic signals
    // and for a number of real data examples.

    LOG_DEBUG(<< "Random diurnal");
    {
        TTimeVec windows{WEEK, 2 * WEEK, 16 * DAY, 4 * WEEK};
        TTimeVec bucketLengths{TEN_MINS, HALF_HOUR};
        TSizeVec permittedGenerators{2, 4, 4, 5};
        TGeneratorVec generators{smoothDaily, spikeyDaily, smoothWeekly, weekends, spikeyWeekly};
        TStrVec expected{"{ 'daily' }", "{ 'daily' }", "{ 'weekly' }",
                         "{ 'weekend daily' 'weekday daily' 'weekend weekly' 'weekday weekly' }",
                         "{ 'daily' 'weekly' }"};

        test::CRandomNumbers rng;

        TDoubleVec noise;
        TSizeVec index;
        TSizeVec repeats;

        double TP{0.0};
        double FN{0.0};

        for (std::size_t test = 0u; test < 100; ++test) {
            if (test % 10 == 0) {
                LOG_DEBUG(<< "test " << test << " / 100");
            }
            for (std::size_t i = 0u; i < windows.size(); ++i) {
                core_t::TTime window{windows[i]};

                for (auto bucketLength : bucketLengths) {
                    switch (test % 3) {
                    case 0:
                        rng.generateNormalSamples(0.0, 1.0, window / bucketLength, noise);
                        break;
                    case 1:
                        rng.generateGammaSamples(1.0, 1.0, window / bucketLength, noise);
                        break;
                    case 2:
                        rng.generateLogNormalSamples(0.2, 0.3, window / bucketLength, noise);
                        break;
                    }
                    rng.generateUniformSamples(0, permittedGenerators[i], 1, index);
                    rng.generateUniformSamples(3, 20, 1, repeats);

                    maths::CPeriodicityHypothesisTests hypotheses;
                    hypotheses.initialize(bucketLength, window,
                                          window / static_cast<core_t::TTime>(repeats[0]));

                    for (core_t::TTime time = 10000; time < 10000 + window; time += bucketLength) {
                        hypotheses.add(time, 20.0 * generators[index[0]](time) +
                                                 noise[(time - 10000) / bucketLength]);
                    }

                    maths::CPeriodicityHypothesisTestsResult result{hypotheses.test()};
                    if (result.print() != expected[index[0]]) {
                        LOG_DEBUG(<< "result = " << result.print()
                                  << " expected " << expected[index[0]]);
                    }
                    TP += result.print() == expected[index[0]] ? 1.0 : 0.0;
                    FN += result.print() == expected[index[0]] ? 0.0 : 1.0;
                }
            }
        }

        LOG_DEBUG(<< "Recall = " << TP / (TP + FN));
        CPPUNIT_ASSERT(TP / (TP + FN) > 0.99);
    }

    LOG_DEBUG(<< "");
    LOG_DEBUG(<< "*** Spikey: daily ***");
    {
        TTimeDoublePrVec timeseries;
        core_t::TTime startTime;
        core_t::TTime endTime;
        CPPUNIT_ASSERT(test::CTimeSeriesTestData::parse(
            "testfiles/spikey_data.csv", timeseries, startTime, endTime,
            test::CTimeSeriesTestData::CSV_UNIX_REGEX));
        CPPUNIT_ASSERT(!timeseries.empty());

        LOG_DEBUG(<< "timeseries = "
                  << core::CContainerPrinter::print(timeseries.begin(),
                                                    timeseries.begin() + 10)
                  << " ...");

        TTimeVec lastTests{timeseries[0].first, timeseries[0].first};
        TTimeVec windows{4 * DAY, 14 * DAY};

        maths::CPeriodicityHypothesisTests hypotheses[2];
        hypotheses[0].initialize(HOUR, windows[0], DAY);
        hypotheses[1].initialize(HOUR, windows[1], DAY);

        for (std::size_t i = 0u; i < timeseries.size(); ++i) {
            core_t::TTime time{timeseries[i].first};
            for (std::size_t j = 0u; j < 2; ++j) {
                if (time > lastTests[j] + windows[j]) {
                    maths::CPeriodicityHypothesisTestsResult result{
                        hypotheses[j].test()};
                    CPPUNIT_ASSERT_EQUAL(std::string("{ 'daily' }"), result.print());
                    hypotheses[j] = maths::CPeriodicityHypothesisTests();
                    hypotheses[j].initialize(HOUR, windows[j], DAY);
                    lastTests[j] += windows[j];
                }
                hypotheses[j].add(time, timeseries[i].second);
            }
        }
    }

    LOG_DEBUG(<< "");
    LOG_DEBUG(<< "*** Diurnal: daily and weekends ***");
    {
        TTimeDoublePrVec timeseries;
        core_t::TTime startTime;
        core_t::TTime endTime;
        CPPUNIT_ASSERT(test::CTimeSeriesTestData::parse(
            "testfiles/diurnal.csv", timeseries, startTime, endTime,
            test::CTimeSeriesTestData::CSV_UNIX_REGEX));
        CPPUNIT_ASSERT(!timeseries.empty());

        LOG_DEBUG(<< "timeseries = "
                  << core::CContainerPrinter::print(timeseries.begin(),
                                                    timeseries.begin() + 10)
                  << " ...");

        core_t::TTime lastTest{timeseries[0].first};
        core_t::TTime window{14 * DAY};

        maths::CPeriodicityHypothesisTests hypotheses;
        hypotheses.initialize(HOUR, window, DAY);

        for (std::size_t i = 0u; i < timeseries.size(); ++i) {
            core_t::TTime time{timeseries[i].first};
            if (time > lastTest + window) {
                maths::CPeriodicityHypothesisTestsResult result{hypotheses.test()};
                CPPUNIT_ASSERT_EQUAL(std::string("{ 'weekend daily' 'weekday daily' }"),
                                     result.print());
                hypotheses = maths::CPeriodicityHypothesisTests();
                hypotheses.initialize(HOUR, window, DAY);
                lastTest += window;
            }
            hypotheses.add(time, timeseries[i].second);
        }
    }

    LOG_DEBUG(<< "");
    LOG_DEBUG(<< "*** Switching: no periods ***");
    {
        TTimeDoublePrVec timeseries;
        core_t::TTime startTime;
        core_t::TTime endTime;
        CPPUNIT_ASSERT(test::CTimeSeriesTestData::parse(
            "testfiles/no_periods.csv", timeseries, startTime, endTime,
            test::CTimeSeriesTestData::CSV_ISO8601_REGEX,
            test::CTimeSeriesTestData::CSV_ISO8601_DATE_FORMAT));
        CPPUNIT_ASSERT(!timeseries.empty());

        LOG_DEBUG(<< "timeseries = "
                  << core::CContainerPrinter::print(timeseries.begin(),
                                                    timeseries.begin() + 10)
                  << " ...");

        core_t::TTime lastTest{timeseries[0].first};
        core_t::TTime window{14 * DAY};

        maths::CPeriodicityHypothesisTests hypotheses;
        hypotheses.initialize(HOUR, window, DAY);

        for (std::size_t i = 0u; i < timeseries.size(); ++i) {
            core_t::TTime time{timeseries[i].first};
            if (time > lastTest + window) {
                maths::CPeriodicityHypothesisTestsResult result{hypotheses.test()};
                CPPUNIT_ASSERT_EQUAL(std::string("{ }"), result.print());
                hypotheses = maths::CPeriodicityHypothesisTests();
                hypotheses.initialize(HOUR, window, DAY);
                lastTest += window;
            }
            hypotheses.add(time, timeseries[i].second);
        }
    }

    LOG_DEBUG(<< "");
    LOG_DEBUG(<< "*** Diurnal: daily, weekly and weekends ***");
    {
        TTimeDoublePrVec timeseries;
        core_t::TTime startTime;
        core_t::TTime endTime;
        CPPUNIT_ASSERT(test::CTimeSeriesTestData::parse(
            "testfiles/thirty_minute_samples.csv", timeseries, startTime,
            endTime, test::CTimeSeriesTestData::CSV_ISO8601_REGEX,
            test::CTimeSeriesTestData::CSV_ISO8601_DATE_FORMAT));
        CPPUNIT_ASSERT(!timeseries.empty());

        LOG_DEBUG(<< "timeseries = "
                  << core::CContainerPrinter::print(timeseries.begin(),
                                                    timeseries.begin() + 10)
                  << " ...");

        core_t::TTime lastTest{timeseries[0].first};
        core_t::TTime window{14 * DAY};

        maths::CPeriodicityHypothesisTests hypotheses;
        hypotheses.initialize(HOUR, window, DAY);

        for (std::size_t i = 0u; i < timeseries.size(); ++i) {
            core_t::TTime time{timeseries[i].first};
            if (time > lastTest + window) {
                maths::CPeriodicityHypothesisTestsResult result{hypotheses.test()};
                CPPUNIT_ASSERT(result.print() == "{ 'weekend daily' 'weekday daily' }" ||
                               result.print() == "{ 'weekend daily' 'weekday daily' 'weekend weekly' 'weekday weekly' }");
                hypotheses = maths::CPeriodicityHypothesisTests();
                hypotheses.initialize(HOUR, window, DAY);
                lastTest += window;
            }
            hypotheses.add(time, timeseries[i].second);
        }
    }
}

void CPeriodicityHypothesisTestsTest::testNonDiurnal() {
    LOG_DEBUG(<< "+---------------------------------------------------+");
    LOG_DEBUG(<< "|  CPeriodicityHypothesisTestsTest::testNonDiurnal  |");
    LOG_DEBUG(<< "+---------------------------------------------------+");

    // Test the recall for periods in the range [DAY / 5, 5 * DAY].

    TTimeVec windows{WEEK, 2 * WEEK, 16 * DAY, 4 * WEEK};
    TTimeVec bucketLengths{TEN_MINS, HALF_HOUR};
    TGeneratorVec generators{smoothDaily, spikeyDaily};
    TSizeVec permittedGenerators{2, 1};

    test::CRandomNumbers rng;

    TDoubleVec noise;
    TSizeVec index;
    TSizeVec repeats;

    double TP{0.0};
    double FN{0.0};

    for (std::size_t test = 0u; test < 100; ++test) {
        if (test % 10 == 0) {
            LOG_DEBUG(<< "test " << test << " / 100");
        }
        for (std::size_t i = 0u; i < windows.size(); ++i) {
            core_t::TTime window{windows[i]};

            TDoubleVec scaling_;
            rng.generateUniformSamples(1.0, 5.0, 1, scaling_);
            double scaling{test % 2 == 0 ? scaling_[0] : 1.0 / scaling_[0]};

            for (std::size_t j = 0u; j < bucketLengths.size(); ++j) {
                core_t::TTime bucketLength{bucketLengths[j]};
                core_t::TTime period{maths::CIntegerTools::floor(
                    static_cast<core_t::TTime>(static_cast<double>(DAY) / scaling), bucketLength)};
                scaling = static_cast<double>(DAY) / static_cast<double>(period);
                if (scaling == 1.0 || window < 3 * period) {
                    continue;
                }

                maths::CPeriodicityHypothesisTestsResult expected;
                expected.add(core::CStringUtils::typeToString(period), false, 0,
                             period, {0, period});

                switch (test % 3) {
                case 0:
                    rng.generateNormalSamples(0.0, 1.0, window / bucketLength, noise);
                    break;
                case 1:
                    rng.generateGammaSamples(1.0, 1.0, window / bucketLength, noise);
                    break;
                case 2:
                    rng.generateLogNormalSamples(0.2, 0.3, window / bucketLength, noise);
                    break;
                }
                rng.generateUniformSamples(0, permittedGenerators[j], 1, index);
                rng.generateUniformSamples(3, 20, 1, repeats);

                maths::CPeriodicityHypothesisTests hypotheses;
                hypotheses.initialize(bucketLength, window, period);

                for (core_t::TTime time = 10000; time < 10000 + window; time += bucketLength) {
                    hypotheses.add(time, 20.0 * scale(scaling, time, generators[index[0]]) +
                                             noise[(time - 10000) / bucketLength]);
                }

                maths::CPeriodicityHypothesisTestsResult result{hypotheses.test()};
                if (result.print() != expected.print()) {
                    LOG_DEBUG(<< "result = " << result.print() << " expected "
                              << expected.print());
                }
                TP += result.print() == expected.print() ? 1.0 : 0.0;
                FN += result.print() == expected.print() ? 0.0 : 1.0;
            }
        }
    }

    LOG_DEBUG(<< "Recall = " << TP / (TP + FN));
    CPPUNIT_ASSERT(TP / (TP + FN) > 0.99);
}

void CPeriodicityHypothesisTestsTest::testWithSparseData() {
    LOG_DEBUG(<< "+-----------------------------------------------------------+");
    LOG_DEBUG(<< "|  CPeriodicityHypothesisTestsTest::testTestWithSparseData  |");
    LOG_DEBUG(<< "+-----------------------------------------------------------+");

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "Daily Periodic") {
        maths::CPeriodicityHypothesisTests hypotheses;
        hypotheses.initialize(HALF_HOUR, WEEK, DAY);

        core_t::TTime time = 0;
        for (std::size_t t = 0u; t < 7; ++t) {
            for (auto value :
                 {0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 20.0, 18.0, 10.0, 4.0,  4.0, 4.0, 4.0, 5.0,
                  6.0, 8.0, 9.0,  9.0,  10.0, 10.0, 8.0, 4.0, 3.0, 1.0,
                  0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  3.0, 1.0}) {
                if (value > 0.0) {
                    hypotheses.add(time, value);
                }
                time += HALF_HOUR;
            }
            if (t > 3) {
                maths::CPeriodicityHypothesisTestsResult result{hypotheses.test()};
                LOG_DEBUG(<< "result = " << result.print());
                CPPUNIT_ASSERT_EQUAL(std::string("{ 'daily' }"), result.print());
            }
        }
    }

    LOG_DEBUG(<< "Daily Not Periodic") {
        maths::CPeriodicityHypothesisTests hypotheses;
        hypotheses.initialize(HALF_HOUR, WEEK, DAY);

        core_t::TTime time = 0;
        for (std::size_t t = 0u; t < 7; ++t) {
            for (auto value :
                 {0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 20.0, 18.0, 10.0, 4.0,  4.0, 4.0, 4.0, 5.0,
                  6.0, 8.0, 9.0,  9.0,  10.0, 10.0, 8.0, 4.0, 3.0, 1.0,
                  0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  3.0, 1.0}) {
                if (value > 0.0) {
                    TDoubleVec rand;
                    rng.generateUniformSamples(-1.0, 1.0, 1, rand);
                    hypotheses.add(time, rand[0]);
                }
                time += HALF_HOUR;
            }

            maths::CPeriodicityHypothesisTestsResult result{hypotheses.test()};
            LOG_DEBUG(<< "result = " << result.print());
            CPPUNIT_ASSERT_EQUAL(std::string("{ }"), result.print());
        }
    }

    LOG_DEBUG(<< "Weekly") {
        maths::CPeriodicityHypothesisTests hypotheses;
        hypotheses.initialize(HOUR, 2 * WEEK, WEEK);

        core_t::TTime time = 0;
        for (std::size_t t = 0u; t < 4; ++t) {
            for (auto value :
                 {0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                  0.0,  0.0,  10.0, 10.0, 8.0,  4.0,  3.0,  1.0,  1.0,  3.0,
                  0.0,  0.0,  0.0,  0.0,  20.0, 18.0, 10.0, 4.0,  4.0,  4.0,
                  4.0,  5.0,  6.0,  8.0,  9.0,  9.0,  0.0,  0.0,  0.0,  0.0,
                  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                  20.0, 18.0, 10.0, 4.0,  4.0,  4.0,  4.0,  5.0,  6.0,  8.0,
                  9.0,  9.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  20.0, 18.0, 10.0, 4.0,
                  4.0,  4.0,  4.0,  5.0,  6.0,  8.0,  9.0,  9.0,  20.0, 18.0,
                  10.0, 4.0,  4.0,  4.0,  4.0,  5.0,  6.0,  8.0,  9.0,  9.0,
                  10.0, 10.0, 8.0,  4.0,  3.0,  1.0,  1.0,  3.0,  0.0,  0.0,
                  0.0,  0.0,  20.0, 18.0, 10.0, 4.0,  4.0,  4.0,  4.0,  5.0,
                  6.0,  8.0,  9.0,  9.0,  10.0, 10.0, 8.0,  4.0,  3.0,  1.0,
                  1.0,  3.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0}) {
                if (value > 0.0) {
                    hypotheses.add(time, value);
                }
                time += HOUR;
            }

            if (t >= 2) {
                maths::CPeriodicityHypothesisTestsResult result{hypotheses.test()};
                LOG_DEBUG(<< "result = " << result.print());
                CPPUNIT_ASSERT_EQUAL(std::string("{ 'daily' 'weekly' }"), result.print());
            }
        }
    }

    LOG_DEBUG(<< "Weekly Not Periodic") {
        maths::CPeriodicityHypothesisTests hypotheses;
        hypotheses.initialize(HOUR, 4 * WEEK, WEEK);

        core_t::TTime time = 0;
        for (std::size_t t = 0u; t < 4; ++t) {
            for (auto value :
                 {0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                  0.0,  0.0,  10.0, 10.0, 8.0,  4.0,  3.0,  1.0,  1.0,  3.0,
                  0.0,  0.0,  0.0,  0.0,  20.0, 18.0, 10.0, 4.0,  4.0,  4.0,
                  4.0,  5.0,  6.0,  8.0,  9.0,  9.0,  0.0,  0.0,  0.0,  0.0,
                  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                  20.0, 18.0, 10.0, 4.0,  4.0,  4.0,  4.0,  5.0,  6.0,  8.0,
                  9.0,  9.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  20.0, 18.0, 10.0, 4.0,
                  4.0,  4.0,  4.0,  5.0,  6.0,  8.0,  9.0,  9.0,  20.0, 18.0,
                  10.0, 4.0,  4.0,  4.0,  4.0,  5.0,  6.0,  8.0,  9.0,  9.0,
                  10.0, 10.0, 8.0,  4.0,  3.0,  1.0,  1.0,  3.0,  0.0,  0.0,
                  0.0,  0.0,  20.0, 18.0, 10.0, 4.0,  4.0,  4.0,  4.0,  5.0,
                  6.0,  8.0,  9.0,  9.0,  10.0, 10.0, 8.0,  4.0,  3.0,  1.0,
                  1.0,  3.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0}) {
                if (value > 0.0) {
                    TDoubleVec rand;
                    rng.generateUniformSamples(-1.0, 1.0, 1, rand);
                    hypotheses.add(time, rand[0]);
                }
                time += HOUR;
            }

            maths::CPeriodicityHypothesisTestsResult result{hypotheses.test()};
            LOG_DEBUG(<< "result = " << result.print());
            CPPUNIT_ASSERT_EQUAL(std::string("{ }"), result.print());
        }
    }
}

void CPeriodicityHypothesisTestsTest::testTestForPeriods() {
    LOG_DEBUG(<< "+-------------------------------------------------------+");
    LOG_DEBUG(<< "|  CPeriodicityHypothesisTestsTest::testTestForPeriods  |");
    LOG_DEBUG(<< "+-------------------------------------------------------+");

    // Test the ability to correctly find and test for periodic
    // signals without being told the periods to test a-priori.

    TTimeVec windows{WEEK, 2 * WEEK, 16 * DAY, 4 * WEEK};
    TTimeVec bucketLengths{TEN_MINS, HALF_HOUR};
    TGeneratorVec generators{smoothDaily, spikeyDaily};
    TSizeVec permittedGenerators{2, 1};
    core_t::TTime startTime{10000};

    test::CRandomNumbers rng;

    TDoubleVec noise;
    TSizeVec index;
    TSizeVec repeats;

    TDoubleVec TP{0.0, 0.0, 0.0};
    TDoubleVec FN{0.0, 0.0, 0.0};

    for (std::size_t test = 0u; test < 100; ++test) {
        if (test % 10 == 0) {
            LOG_DEBUG(<< "test " << test << " / 100");
        }
        for (std::size_t i = 0u; i < windows.size(); ++i) {
            core_t::TTime window{windows[i]};

            TDoubleVec scaling_;
            rng.generateUniformSamples(1.0, 5.0, 1, scaling_);
            double scaling{test % 2 == 0 ? scaling_[0] : 1.0 / scaling_[0]};

            for (std::size_t j = 0u; j < bucketLengths.size(); ++j) {
                core_t::TTime bucketLength{bucketLengths[j]};
                core_t::TTime period{maths::CIntegerTools::floor(
                    static_cast<core_t::TTime>(static_cast<double>(DAY) / scaling), bucketLength)};
                scaling = static_cast<double>(DAY) / static_cast<double>(period);
                if (scaling == 1.0 || window < 3 * period) {
                    continue;
                }

                maths::CPeriodicityHypothesisTestsResult expected;
                expected.add(core::CStringUtils::typeToString(period), false, 0,
                             period, {0, period});

                switch (test % 3) {
                case 0:
                    rng.generateNormalSamples(0.0, 1.0, window / bucketLength, noise);
                    break;
                case 1:
                    rng.generateGammaSamples(1.0, 1.0, window / bucketLength, noise);
                    break;
                case 2:
                    rng.generateLogNormalSamples(0.2, 0.3, window / bucketLength, noise);
                    break;
                }
                rng.generateUniformSamples(0, permittedGenerators[j], 1, index);
                rng.generateUniformSamples(3, 20, 1, repeats);

                maths::CPeriodicityHypothesisTests hypotheses;
                hypotheses.initialize(bucketLength, window, period);

                maths::TFloatMeanAccumulatorVec values(window / bucketLength);
                for (core_t::TTime time = startTime; time < startTime + window;
                     time += bucketLength) {
                    std::size_t bucket((time - startTime) / bucketLength);
                    double value{20.0 * scale(scaling, time, generators[index[0]]) +
                                 noise[bucket]};
                    values[bucket].add(value);
                }

                maths::CPeriodicityHypothesisTestsConfig config;
                maths::CPeriodicityHypothesisTestsResult result{
                    maths::testForPeriods(config, startTime, bucketLength, values)};
                if (result.print() != expected.print()) {
                    LOG_DEBUG(<< "result = " << result.print() << " expected "
                              << expected.print());
                }

                TP[0] += result.print() == expected.print() ? 1.0 : 0.0;
                FN[0] += result.print() == expected.print() ? 0.0 : 1.0;
                if (result.components().size() == 1) {
                    core_t::TTime modp{result.components()[0].s_Period % period};
                    double error{static_cast<double>(std::min(modp, std::abs(period - modp))) /
                                 static_cast<double>(period)};
                    TP[1] += error < 0.01 ? 1.0 : 0.0;
                    FN[1] += error < 0.01 ? 0.0 : 1.0;
                    TP[2] += error < 0.05 ? 1.0 : 0.0;
                    FN[2] += error < 0.05 ? 0.0 : 1.0;
                } else {
                    FN[0] += 1.0;
                    FN[1] += 1.0;
                    FN[2] += 1.0;
                }
            }
        }
    }

    LOG_DEBUG(<< "Recall at 0% error = " << TP[0] / (TP[0] + FN[0]));
    LOG_DEBUG(<< "Recall at 1% error = " << TP[1] / (TP[1] + FN[1]));
    LOG_DEBUG(<< "Recall at 5% error = " << TP[2] / (TP[2] + FN[2]));
    CPPUNIT_ASSERT(TP[0] / (TP[0] + FN[0]) > 0.91);
    CPPUNIT_ASSERT(TP[1] / (TP[1] + FN[1]) > 0.99);
    CPPUNIT_ASSERT(TP[2] / (TP[2] + FN[2]) > 0.99);
}

CppUnit::Test* CPeriodicityHypothesisTestsTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CPeriodicityHypothesisTestsTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CPeriodicityHypothesisTestsTest>(
        "CPeriodicityHypothesisTestsTest::testNonPeriodic",
        &CPeriodicityHypothesisTestsTest::testNonPeriodic));
    suiteOfTests->addTest(new CppUnit::TestCaller<CPeriodicityHypothesisTestsTest>(
        "CPeriodicityHypothesisTestsTest::testDiurnal",
        &CPeriodicityHypothesisTestsTest::testDiurnal));
    suiteOfTests->addTest(new CppUnit::TestCaller<CPeriodicityHypothesisTestsTest>(
        "CPeriodicityHypothesisTestsTest::testNonDiurnal",
        &CPeriodicityHypothesisTestsTest::testNonDiurnal));
    suiteOfTests->addTest(new CppUnit::TestCaller<CPeriodicityHypothesisTestsTest>(
        "CPeriodicityHypothesisTestsTest::testWithSparseData",
        &CPeriodicityHypothesisTestsTest::testWithSparseData));
    suiteOfTests->addTest(new CppUnit::TestCaller<CPeriodicityHypothesisTestsTest>(
        "CPeriodicityHypothesisTestsTest::testTestForPeriods",
        &CPeriodicityHypothesisTestsTest::testTestForPeriods));

    return suiteOfTests;
}
