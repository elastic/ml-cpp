/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/Constants.h>
#include <core/CoreTypes.h>

#include <maths/CIntegerTools.h>
#include <maths/CPeriodicityHypothesisTests.h>

#include <test/CRandomNumbers.h>
#include <test/CTimeSeriesTestData.h>

#include "TestUtils.h"

#include <boost/math/constants/constants.hpp>
#include <boost/test/unit_test.hpp>

#include <cmath>
#include <vector>

BOOST_AUTO_TEST_SUITE(CPeriodicityHypothesisTestsTest)

using namespace ml;
using namespace handy_typedefs;

namespace {
using TDoubleDoublePr = std::pair<double, double>;
using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;
using TTimeVec = std::vector<core_t::TTime>;
using TTimeDoublePr = std::pair<core_t::TTime, double>;
using TTimeDoublePrVec = std::vector<TTimeDoublePr>;
using TStrVec = std::vector<std::string>;
using TFloatMeanAccumulator =
    maths::CBasicStatistics::SSampleMean<maths::CFloatStorage>::TAccumulator;
using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;

const core_t::TTime TEN_MINS{600};
const core_t::TTime HALF_HOUR{core::constants::HOUR / 2};
const core_t::TTime HOUR{core::constants::HOUR};
const core_t::TTime DAY{core::constants::DAY};
const core_t::TTime WEEK{core::constants::WEEK};
}

BOOST_AUTO_TEST_CASE(testNonPeriodic) {
    // Test a variety of synthetic non-periodic signals.

    TTimeVec windows{WEEK, 2 * WEEK, 16 * DAY, 4 * WEEK};
    TTimeVec bucketLengths{TEN_MINS, HALF_HOUR};
    TGeneratorVec generators{constant, ramp, markov};
    core_t::TTime startTime{10000};

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
            core_t::TTime endTime{startTime + window};

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
                const auto& generator = generators[index[0]];

                maths::CPeriodicityHypothesisTests hypotheses;
                hypotheses.initialize(0 /*startTime*/, bucketLength, window,
                                      window / static_cast<core_t::TTime>(repeats[0]));

                for (core_t::TTime time = startTime; time < endTime; time += bucketLength) {
                    std::size_t bucket((time - startTime) / bucketLength);
                    hypotheses.add(time, generator(time) + noise[bucket]);
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
    BOOST_TEST_REQUIRE(TN / (FP + TN) > 0.995);
}

BOOST_AUTO_TEST_CASE(testDiurnal) {
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
        core_t::TTime startTime{10000};

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
                    core_t::TTime endTime{startTime + window};

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
                    const auto& generator = generators[index[0]];

                    maths::CPeriodicityHypothesisTests hypotheses;
                    hypotheses.initialize(0 /*startTime*/, bucketLength, window,
                                          window / static_cast<core_t::TTime>(repeats[0]));

                    for (core_t::TTime time = startTime; time < endTime; time += bucketLength) {
                        std::size_t bucket((time - startTime) / bucketLength);
                        hypotheses.add(time, 20.0 * generator(time) + noise[bucket]);
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
        BOOST_TEST_REQUIRE(TP / (TP + FN) > 0.99);
    }

    LOG_DEBUG(<< "");
    LOG_DEBUG(<< "*** Spikey: daily ***");
    {
        TTimeDoublePrVec timeseries;
        core_t::TTime startTime;
        core_t::TTime endTime;
        BOOST_TEST_REQUIRE(test::CTimeSeriesTestData::parse(
            "testfiles/spikey_data.csv", timeseries, startTime, endTime,
            test::CTimeSeriesTestData::CSV_UNIX_REGEX));
        BOOST_TEST_REQUIRE(!timeseries.empty());

        LOG_DEBUG(<< "timeseries = "
                  << core::CContainerPrinter::print(timeseries.begin(),
                                                    timeseries.begin() + 10)
                  << " ...");

        TTimeVec lastTests{timeseries[0].first, timeseries[0].first};
        TTimeVec windows{4 * DAY, 14 * DAY};

        maths::CPeriodicityHypothesisTests hypotheses[2];
        hypotheses[0].initialize(0 /*startTime*/, HOUR, windows[0], DAY);
        hypotheses[1].initialize(0 /*startTime*/, HOUR, windows[1], DAY);

        for (std::size_t i = 0u; i < timeseries.size(); ++i) {
            core_t::TTime time{timeseries[i].first};
            for (std::size_t j = 0u; j < 2; ++j) {
                if (time > lastTests[j] + windows[j]) {
                    maths::CPeriodicityHypothesisTestsResult result{
                        hypotheses[j].test()};
                    BOOST_REQUIRE_EQUAL(std::string("{ 'daily' }"), result.print());
                    hypotheses[j] = maths::CPeriodicityHypothesisTests();
                    hypotheses[j].initialize(0 /*startTime*/, HOUR, windows[j], DAY);
                    lastTests[j] += windows[j];
                }
                hypotheses[j].add(time, timeseries[i].second);
            }
        }
    }

    LOG_DEBUG(<< "");
    LOG_DEBUG(<< "*** Diurnal: daily, weekly and weekends + outliers ***");
    {
        TTimeDoublePrVec timeseries;
        core_t::TTime startTime;
        core_t::TTime endTime;
        BOOST_TEST_REQUIRE(test::CTimeSeriesTestData::parse(
            "testfiles/diurnal.csv", timeseries, startTime, endTime,
            test::CTimeSeriesTestData::CSV_UNIX_REGEX));
        BOOST_TEST_REQUIRE(!timeseries.empty());

        LOG_DEBUG(<< "timeseries = "
                  << core::CContainerPrinter::print(timeseries.begin(),
                                                    timeseries.begin() + 10)
                  << " ...");

        core_t::TTime lastTest{timeseries[0].first};
        core_t::TTime window{14 * DAY};

        maths::CPeriodicityHypothesisTests hypotheses;
        hypotheses.initialize(0 /*startTime*/, HOUR, window, DAY);

        for (std::size_t i = 0u; i < timeseries.size(); ++i) {
            core_t::TTime time{timeseries[i].first};
            if (time > lastTest + window) {
                maths::CPeriodicityHypothesisTestsResult result{hypotheses.test()};
                LOG_DEBUG(<< "result = " << result.print());
                BOOST_TEST_REQUIRE(
                    (result.print() == "{ 'weekend daily' 'weekday daily' 'weekend weekly' }" ||
                     result.print() == "{ 'weekend daily' 'weekday daily' 'weekend weekly' 'weekday weekly' }"));
                hypotheses = maths::CPeriodicityHypothesisTests();
                hypotheses.initialize(0 /*startTime*/, HOUR, window, DAY);
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
        BOOST_TEST_REQUIRE(test::CTimeSeriesTestData::parse(
            "testfiles/no_periods.csv", timeseries, startTime, endTime,
            test::CTimeSeriesTestData::CSV_ISO8601_REGEX,
            test::CTimeSeriesTestData::CSV_ISO8601_DATE_FORMAT));
        BOOST_TEST_REQUIRE(!timeseries.empty());

        LOG_DEBUG(<< "timeseries = "
                  << core::CContainerPrinter::print(timeseries.begin(),
                                                    timeseries.begin() + 10)
                  << " ...");

        core_t::TTime lastTest{timeseries[0].first};
        core_t::TTime window{14 * DAY};

        maths::CPeriodicityHypothesisTests hypotheses;
        hypotheses.initialize(0 /*startTime*/, HOUR, window, DAY);

        for (std::size_t i = 0u; i < timeseries.size(); ++i) {
            core_t::TTime time{timeseries[i].first};
            if (time > lastTest + window) {
                maths::CPeriodicityHypothesisTestsResult result{hypotheses.test()};
                BOOST_REQUIRE_EQUAL(std::string("{ }"), result.print());
                hypotheses = maths::CPeriodicityHypothesisTests();
                hypotheses.initialize(0 /*startTime*/, HOUR, window, DAY);
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
        BOOST_TEST_REQUIRE(test::CTimeSeriesTestData::parse(
            "testfiles/thirty_minute_samples.csv", timeseries, startTime,
            endTime, test::CTimeSeriesTestData::CSV_ISO8601_REGEX,
            test::CTimeSeriesTestData::CSV_ISO8601_DATE_FORMAT));
        BOOST_TEST_REQUIRE(!timeseries.empty());

        LOG_DEBUG(<< "timeseries = "
                  << core::CContainerPrinter::print(timeseries.begin(),
                                                    timeseries.begin() + 10)
                  << " ...");

        core_t::TTime lastTest{timeseries[0].first};
        core_t::TTime window{14 * DAY};

        maths::CPeriodicityHypothesisTests hypotheses;
        hypotheses.initialize(0 /*startTime*/, HOUR, window, DAY);

        for (std::size_t i = 0u; i < timeseries.size(); ++i) {
            core_t::TTime time{timeseries[i].first};
            if (time > lastTest + window) {
                maths::CPeriodicityHypothesisTestsResult result{hypotheses.test()};
                const std::string& printedResult{result.print()};
                LOG_DEBUG(<< "result = " << printedResult);
                if (printedResult != "{ 'weekend daily' 'weekday daily' 'weekend weekly' }") {
                    BOOST_TEST_REQUIRE(printedResult == "{ 'weekend daily' 'weekday daily' 'weekend weekly' 'weekday weekly' }");
                }
                hypotheses = maths::CPeriodicityHypothesisTests();
                hypotheses.initialize(0 /*startTime*/, HOUR, window, DAY);
                lastTest += window;
            }
            hypotheses.add(time, timeseries[i].second);
        }
    }
}

BOOST_AUTO_TEST_CASE(testNonDiurnal) {
    // Test the recall for periods in the range [DAY / 5, 5 * DAY].

    TTimeVec windows{WEEK, 2 * WEEK, 16 * DAY, 4 * WEEK};
    TTimeVec bucketLengths{TEN_MINS, HALF_HOUR};
    TGeneratorVec generators{smoothDaily, spikeyDaily};
    TSizeVec permittedGenerators{2, 1};
    core_t::TTime startTime{10000};

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
        for (auto window : windows) {
            core_t::TTime endTime{startTime + window};

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
                expected.add(core::CStringUtils::typeToString(period), false,
                             false, 0, period, {0, period});

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
                auto generator = generators[index[0]];

                maths::CPeriodicityHypothesisTests hypotheses;
                hypotheses.initialize(0 /*startTime*/, bucketLength, window, period);

                for (core_t::TTime time = startTime; time < endTime; time += bucketLength) {
                    std::size_t bucket((time - startTime) / bucketLength);
                    hypotheses.add(time, 20.0 * scale(scaling, time, generator) +
                                             noise[bucket]);
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
    BOOST_TEST_REQUIRE(TP / (TP + FN) > 0.99);
}

BOOST_AUTO_TEST_CASE(testWithSparseData) {
    // Test we correctly identify periodicity if there are periodically
    // missing data.

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "Daily Periodic");
    {
        maths::CPeriodicityHypothesisTests hypotheses;
        hypotheses.initialize(0 /*startTime*/, HALF_HOUR, WEEK, DAY);

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
                BOOST_REQUIRE_EQUAL(std::string("{ 'daily' }"), result.print());
            }
        }
    }

    LOG_DEBUG(<< "Daily Not Periodic");
    {
        maths::CPeriodicityHypothesisTests hypotheses;
        hypotheses.initialize(0 /*startTime*/, HALF_HOUR, WEEK, DAY);

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
            BOOST_REQUIRE_EQUAL(std::string("{ }"), result.print());
        }
    }

    LOG_DEBUG(<< "Weekly");
    {
        maths::CPeriodicityHypothesisTests hypotheses;
        hypotheses.initialize(0 /*startTime*/, HOUR, 2 * WEEK, WEEK);

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
                BOOST_REQUIRE_EQUAL(std::string("{ 'daily' 'weekly' }"), result.print());
            }
        }
    }

    LOG_DEBUG(<< "Weekly Not Periodic");
    {
        maths::CPeriodicityHypothesisTests hypotheses;
        hypotheses.initialize(0 /*startTime*/, HOUR, 4 * WEEK, WEEK);

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
            BOOST_REQUIRE_EQUAL(std::string("{ }"), result.print());
        }
    }
}

BOOST_AUTO_TEST_CASE(testWithOutliers) {
    // Test the we can robustly detect the correct underlying periodic
    // components.

    TTimeVec windows{WEEK, 2 * WEEK, 16 * DAY, 4 * WEEK};
    TTimeVec bucketLengths{TEN_MINS, HALF_HOUR};
    TTimeVec periods{DAY, WEEK};
    TDoubleVec modulation{0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0};
    core_t::TTime startTime{10000};

    test::CRandomNumbers rng;

    TDoubleVec noise;
    TSizeVec outliers;
    TDoubleVec spikeOrTroughSelector;

    LOG_DEBUG(<< "Daily + Weekly");
    for (const auto& period : periods) {
        for (const auto& window : windows) {
            core_t::TTime endTime{startTime + window};

            if (window < 2 * period) {
                continue;
            }

            for (const auto& bucketLength : bucketLengths) {
                core_t::TTime buckets{window / bucketLength};
                std::size_t numberOutliers{static_cast<std::size_t>(0.12 * buckets)};
                rng.generateUniformSamples(0, buckets, numberOutliers, outliers);
                rng.generateUniformSamples(0, 1.0, numberOutliers, spikeOrTroughSelector);
                rng.generateNormalSamples(0.0, 9.0, buckets, noise);
                std::sort(outliers.begin(), outliers.end());

                //std::ofstream file;
                //file.open("results.m");
                //TDoubleVec f;

                TFloatMeanAccumulatorVec values(buckets);
                for (core_t::TTime time = startTime; time < endTime; time += bucketLength) {
                    std::size_t bucket((time - startTime) / bucketLength);
                    auto outlier = std::lower_bound(outliers.begin(), outliers.end(), bucket);
                    if (outlier != outliers.end() && *outlier == bucket) {
                        values[bucket].add(
                            spikeOrTroughSelector[outlier - outliers.begin()] > 0.2 ? 0.0 : 100.0);
                    } else {
                        values[bucket].add(
                            20.0 + 20.0 * std::sin(boost::math::double_constants::two_pi *
                                                   static_cast<double>(time) /
                                                   static_cast<double>(period)));
                    }
                    //f.push_back(maths::CBasicStatistics::mean(values[bucket]));
                }
                //file << "f = " << core::CContainerPrinter::print(f) << ";";

                maths::CPeriodicityHypothesisTestsConfig config;
                maths::CPeriodicityHypothesisTestsResult result{
                    maths::testForPeriods(config, startTime, bucketLength, values)};
                LOG_DEBUG(<< "result = " << result.print());
                if (period == DAY) {
                    BOOST_REQUIRE_EQUAL(std::string{"{ 'daily' }"}, result.print());
                } else if (period == WEEK) {
                    BOOST_REQUIRE_EQUAL(std::string{"{ 'weekly' }"}, result.print());
                }
            }
        }
    }

    LOG_DEBUG(<< "Weekday / Weekend");
    for (const auto& window : windows) {
        core_t::TTime endTime{startTime + window};

        if (window < 2 * WEEK) {
            continue;
        }

        for (const auto& bucketLength : bucketLengths) {
            core_t::TTime buckets{window / bucketLength};
            std::size_t numberOutliers{static_cast<std::size_t>(0.12 * buckets)};
            rng.generateUniformSamples(0, buckets, numberOutliers, outliers);
            rng.generateUniformSamples(0, 1.0, numberOutliers, spikeOrTroughSelector);
            rng.generateNormalSamples(0.0, 9.0, buckets, noise);
            std::sort(outliers.begin(), outliers.end());

            //std::ofstream file;
            //file.open("results.m");
            //TDoubleVec f;

            TFloatMeanAccumulatorVec values(buckets);
            for (core_t::TTime time = startTime; time < endTime; time += bucketLength) {
                std::size_t bucket((time - startTime) / bucketLength);
                auto outlier = std::lower_bound(outliers.begin(), outliers.end(), bucket);
                if (outlier != outliers.end() && *outlier == bucket) {
                    values[bucket].add(
                        spikeOrTroughSelector[outlier - outliers.begin()] > 0.2 ? 0.0 : 100.0);
                } else {
                    values[bucket].add(
                        modulation[((time - startTime) / DAY) % 7] *
                        (20.0 + 20.0 * std::sin(boost::math::double_constants::two_pi *
                                                static_cast<double>(time) /
                                                static_cast<double>(DAY))));
                }
                //f.push_back(maths::CBasicStatistics::mean(values[bucket]));
            }
            //file << "f = " << core::CContainerPrinter::print(f) << ";";

            maths::CPeriodicityHypothesisTestsConfig config;
            maths::CPeriodicityHypothesisTestsResult result{
                maths::testForPeriods(config, startTime, bucketLength, values)};
            LOG_DEBUG(<< "result = " << result.print());
            BOOST_TEST_REQUIRE(result.print() ==
                               std::string("{ 'weekend daily' 'weekday daily' 'weekend weekly' }"));
        }
    }
}

BOOST_AUTO_TEST_CASE(testTestForPeriods) {
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
        for (const auto& window : windows) {
            core_t::TTime endTime{startTime + window};

            TDoubleVec scaling_;
            rng.generateUniformSamples(1.0, 5.0, 1, scaling_);
            double scaling{test % 2 == 0 ? scaling_[0] : 1.0 / scaling_[0]};

            for (std::size_t i = 0u; i < bucketLengths.size(); ++i) {
                core_t::TTime bucketLength{bucketLengths[i]};
                core_t::TTime period{maths::CIntegerTools::floor(
                    static_cast<core_t::TTime>(static_cast<double>(DAY) / scaling), bucketLength)};
                scaling = static_cast<double>(DAY) / static_cast<double>(period);
                if (scaling == 1.0 || window < 3 * period) {
                    continue;
                }

                maths::CPeriodicityHypothesisTestsResult expected;
                expected.add(core::CStringUtils::typeToString(period), false,
                             false, 0, period, {0, period});

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
                const auto& generator = generators[index[0]];

                TFloatMeanAccumulatorVec values(window / bucketLength);
                for (core_t::TTime time = startTime; time < endTime; time += bucketLength) {
                    std::size_t bucket((time - startTime) / bucketLength);
                    double value{20.0 * scale(scaling, time, generator) + noise[bucket]};
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
    BOOST_TEST_REQUIRE(TP[0] / (TP[0] + FN[0]) > 0.91);
    BOOST_TEST_REQUIRE(TP[1] / (TP[1] + FN[1]) > 0.99);
    BOOST_TEST_REQUIRE(TP[2] / (TP[2] + FN[2]) > 0.99);
}

BOOST_AUTO_TEST_CASE(testWithLinearScaling) {
    // Test the ability to correctly find diurnal periodic signals
    // with random linear scaling events in the window.

    TTimeVec windows{3 * WEEK, 4 * WEEK};
    core_t::TTime bucketLength{HALF_HOUR};
    std::size_t segmentSupport[][2]{{100, 200}, {600, 900}};
    double scaleSupport[][2]{{4.0, 6.0}, {0.2, 0.4}};
    TGeneratorVec generators{smoothDaily, spikeyDaily, smoothWeekly};
    TStrVec expected{"{ 'daily' }", "{ 'daily' }", "{ 'weekly' }"};
    core_t::TTime startTime{0};

    test::CRandomNumbers rng;

    TDoubleVec noise;
    TSizeVec index;

    double TP{0.0};
    double FN{0.0};

    for (std::size_t test = 0; test < 100; ++test) {
        if (test % 10 == 0) {
            LOG_DEBUG(<< "test " << test << " / 100");
        }

        for (const auto& window : windows) {
            core_t::TTime endTime{startTime + window};

            TTimeVec segments;
            TDoubleVec scales{1.0};
            for (std::size_t i = 0; i < 2; ++i) {
                TSizeVec segment;
                TDoubleVec scale;
                rng.generateUniformSamples(segmentSupport[i][0],
                                           segmentSupport[i][1], 1, segment);
                rng.generateUniformSamples(scaleSupport[i][0], scaleSupport[i][1], 1, scale);
                segments.push_back(startTime + segment[0] * bucketLength);
                scales.push_back(scale[0]);
            }
            segments.push_back(endTime);

            auto trend = [&](core_t::TTime time) {
                auto i = std::lower_bound(segments.begin(), segments.end(), time);
                return 20.0 * scales[i - segments.begin()] * generators[index[0]](time);
            };

            rng.generateNormalSamples(0.0, 1.0, window / bucketLength, noise);
            rng.generateUniformSamples(0, 2, 1, index);

            TFloatMeanAccumulatorVec values(window / bucketLength);
            for (core_t::TTime time = startTime; time < endTime; time += bucketLength) {
                std::size_t bucket((time - startTime) / bucketLength);
                double value{trend(time) + noise[bucket]};
                values[bucket].add(value);
            }

            maths::CPeriodicityHypothesisTestsConfig config;
            maths::CPeriodicityHypothesisTestsResult result{
                maths::testForPeriods(config, startTime, bucketLength, values)};
            if (result.print() != expected[index[0]]) {
                LOG_DEBUG(<< "result = " << result.print() << " expected "
                          << expected[index[0]]);
            }

            TP += result.print() == expected[index[0]] ? 1.0 : 0.0;
            FN += result.print() == expected[index[0]] ? 0.0 : 1.0;
        }
    }

    LOG_DEBUG(<< "Recall = " << TP / (TP + FN));
    BOOST_TEST_REQUIRE(TP / (TP + FN) > 0.99);

    // Test we don't interpret a weekend (which is a scaled weekday)
    // as a segmented pure periodic component.
    {
        double scale[]{0.3, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0};
        TFloatMeanAccumulatorVec values(2 * WEEK / bucketLength);
        for (core_t::TTime time = 0; time < 2 * WEEK; time += bucketLength) {
            values[time / bucketLength].add(
                scale[(time % WEEK) / DAY] *
                std::sin(boost::math::double_constants::two_pi *
                         static_cast<double>(time) / static_cast<double>(DAY)));
        }

        maths::CPeriodicityHypothesisTestsConfig config;
        maths::CPeriodicityHypothesisTestsResult result{
            maths::testForPeriods(config, startTime, bucketLength, values)};
        LOG_DEBUG(<< "result = " << result.print());
        BOOST_REQUIRE_EQUAL(std::string("{ 'weekend daily' 'weekday daily' }"),
                            result.print());
    }
}

BOOST_AUTO_TEST_CASE(testWithPiecewiseLinearTrend) {
    // Test the ability to correctly find diurnal periodic signals
    // with a random piecewise linear trend.

    using TLinearModel = std::function<double(core_t::TTime)>;
    using TLinearModelVec = std::vector<TLinearModel>;

    TTimeVec windows{3 * WEEK, 4 * WEEK};
    core_t::TTime bucketLength{HALF_HOUR};
    std::size_t segmentSupport[][2]{{100, 200}, {600, 900}};
    double slopeSupport[][2]{{0.5, 1.0}, {-1.0, -0.5}};
    double interceptSupport[][2]{{-10.0, 5.0}, {10.0, 20.0}};
    TGeneratorVec generators{smoothDaily, spikeyDaily};
    TStrVec expected{"{ 'daily' }", "{ 'daily' }"};
    core_t::TTime startTime{0};

    test::CRandomNumbers rng;

    TDoubleVec noise;
    TSizeVec index;

    double TP{0.0};
    double FN{0.0};

    for (std::size_t test = 0; test < 100; ++test) {
        if (test % 10 == 0) {
            LOG_DEBUG(<< "test " << test << " / 100");
        }

        for (const auto& window : windows) {
            core_t::TTime endTime{startTime + window};

            TTimeVec segments;
            TLinearModelVec models{[](core_t::TTime) { return 0.0; }};
            for (std::size_t i = 0; i < 2; ++i) {
                TSizeVec segment;
                TDoubleVec slope;
                TDoubleVec intercept;
                rng.generateUniformSamples(segmentSupport[i][0],
                                           segmentSupport[i][1], 1, segment);
                rng.generateUniformSamples(slopeSupport[i][0], slopeSupport[i][1], 1, slope);
                rng.generateUniformSamples(interceptSupport[i][0],
                                           interceptSupport[i][1], 1, intercept);
                segments.push_back(startTime + segment[0] * bucketLength);
                models.push_back([startTime, slope, intercept](core_t::TTime time) {
                    return slope[0] * static_cast<double>(time - startTime) /
                               static_cast<double>(DAY) +
                           intercept[0];
                });
            }
            segments.push_back(endTime);

            auto trend = [&](core_t::TTime time) {
                auto i = std::lower_bound(segments.begin(), segments.end(), time);
                return 2.0 * (models[i - segments.begin()](time) +
                              5.0 * generators[index[0]](time));
            };

            rng.generateNormalSamples(0.0, 1.0, window / bucketLength, noise);
            rng.generateUniformSamples(0, 2, 1, index);

            TFloatMeanAccumulatorVec values(window / bucketLength);
            for (core_t::TTime time = startTime; time < endTime; time += bucketLength) {
                std::size_t bucket((time - startTime) / bucketLength);
                double value{trend(time) + noise[bucket]};
                values[bucket].add(value);
            }

            maths::CPeriodicityHypothesisTestsConfig config;
            maths::CPeriodicityHypothesisTestsResult result{
                maths::testForPeriods(config, startTime, bucketLength, values)};
            if (result.print() != expected[index[0]]) {
                LOG_DEBUG(<< "result = " << result.print() << " expected "
                          << expected[index[0]]);
            }

            TP += result.print() == expected[index[0]] ? 1.0 : 0.0;
            FN += result.print() == expected[index[0]] ? 0.0 : 1.0;
        }
    }

    LOG_DEBUG(<< "Recall = " << TP / (TP + FN));
    BOOST_TEST_REQUIRE(TP / (TP + FN) > 0.8);
}

BOOST_AUTO_TEST_SUITE_END()
