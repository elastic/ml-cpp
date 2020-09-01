/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/Constants.h>
#include <core/CoreTypes.h>

#include <maths/CBasicStatistics.h>
#include <maths/CIntegerTools.h>
#include <maths/COrderings.h>
#include <maths/CSeasonalTime.h>
#include <maths/CTimeSeriesTestForSeasonality.h>
#include <maths/MathsTypes.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>
#include <test/CTimeSeriesTestData.h>

#include "TestUtils.h"

#include <boost/math/constants/constants.hpp>
#include <boost/test/unit_test.hpp>

#include <cmath>
#include <vector>

BOOST_AUTO_TEST_SUITE(CTimeSeriesTestForSeasonalityTest)

using namespace ml;
using namespace handy_typedefs;

namespace {
using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;
using TTimeVec = std::vector<core_t::TTime>;
using TTimeDoublePr = std::pair<core_t::TTime, double>;
using TTimeDoublePrVec = std::vector<TTimeDoublePr>;
using TStrVec = std::vector<std::string>;
using TStrVecVec = std::vector<TStrVec>;
using TFloatMeanAccumulator =
    maths::CBasicStatistics::SSampleMean<maths::CFloatStorage>::TAccumulator;
using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
using TDiurnalTimeVec = std::vector<maths::CDiurnalTime>;
using TDiurnalTimeVecVec = std::vector<TDiurnalTimeVec>;

void generateNoise(std::size_t type,
                   test::CRandomNumbers& rng,
                   core_t::TTime window,
                   core_t::TTime interval,
                   TDoubleVec& noise) {
    switch (type % 3) {
    case 0:
        rng.generateNormalSamples(0.0, 1.0, window / interval, noise);
        break;
    case 1:
        rng.generateGammaSamples(1.0, 1.0, window / interval, noise);
        break;
    case 2:
        rng.generateLogNormalSamples(0.2, 0.3, window / interval, noise);
        break;
    }
}

const core_t::TTime FIVE_MINS{core::constants::HOUR / 12};
const core_t::TTime HALF_HOUR{core::constants::HOUR / 2};
const core_t::TTime HOUR{core::constants::HOUR};
const core_t::TTime DAY{core::constants::DAY};
const core_t::TTime WEEK{core::constants::WEEK};
}

BOOST_AUTO_TEST_CASE(testSyntheticNoSeasonality) {

    // Test FP % for a variety of synthetic time series with not seasonality.

    TGeneratorVec generators{constant, ramp, markov};
    core_t::TTime startTime{604800};

    test::CRandomNumbers rng;

    TFloatMeanAccumulatorVec values;
    TDoubleVec noise;
    TSizeVec index;
    TSizeVec repeats;

    double FP{0.0};
    double TN{0.0};

    for (std::size_t test = 0; test < 100; ++test) {
        if ((test + 1) % 10 == 0) {
            LOG_DEBUG(<< "test " << test + 1 << " / 100");
        }

        for (auto window : {WEEK, 2 * WEEK, 4 * WEEK}) {
            LOG_TRACE(<< "window = " << window);

            for (auto bucketLength : {HALF_HOUR, HOUR}) {
                LOG_TRACE(<< "bucket length = " << bucketLength);

                generateNoise(test, rng, window, bucketLength, noise);
                rng.generateUniformSamples(0, generators.size(), 1, index);
                rng.generateUniformSamples(3, 20, 1, repeats);
                LOG_TRACE(<< "generator = " << index[0]);
                const auto& generator = generators[index[0]];

                values.assign(window / bucketLength, TFloatMeanAccumulator{});
                for (core_t::TTime time = 0; time < window; time += bucketLength) {
                    std::size_t bucket(time / bucketLength);
                    values[bucket].add(5.0 * generator(startTime + time * bucketLength) +
                                       noise[bucket]);
                }

                maths::CTimeSeriesTestForSeasonality seasonality{
                    startTime, startTime, bucketLength, values};

                auto result = seasonality.decompose();
                bool isSeasonal{result.seasonal().size() > 0};

                if (isSeasonal) {
                    LOG_DEBUG(<< "got " << result.print() << " expected []");
                }
                FP += isSeasonal ? 1.0 : 0.0;
                TN += isSeasonal ? 0.0 : 1.0;
            }
        }
    }

    LOG_DEBUG(<< "True negative rate = " << TN / (FP + TN));
    BOOST_REQUIRE(TN / (FP + TN) > 0.99);
}

BOOST_AUTO_TEST_CASE(testSyntheticDiurnal) {

    // Test accuracy for a variety of synthetic time series with daily and
    // weekly seasonalities.

    TTimeVec windows{4 * DAY, WEEK, 2 * WEEK, 4 * WEEK};
    TSizeVec permittedGenerators{2, 2, 3, 5};
    TGeneratorVec generators{smoothDaily, spikeyDaily, smoothWeekly,
                             spikeyDailyWeekly, weekends};
    TStrVecVec expected{{"86400"},
                        {"86400"},
                        {"604800"},
                        {"86400/(0,172800)", "86400/(172800,604800)", "604800/(0,172800)"},
                        {"86400/(0,172800)", "86400/(172800,604800)",
                         "604800/(0,172800)", "604800/(172800,604800)"}};
    core_t::TTime startTime{10000};

    test::CRandomNumbers rng;

    TFloatMeanAccumulatorVec values;
    TDoubleVec noise;
    TSizeVec index;

    double TP{0.0};
    double FN{0.0};
    double FP{0.0};

    for (std::size_t test = 0; test < 100; ++test) {
        if ((test + 1) % 10 == 0) {
            LOG_DEBUG(<< "test " << test + 1 << " / 100");
        }

        for (std::size_t i = 0; i < windows.size(); ++i) {
            core_t::TTime window{windows[i]};
            LOG_TRACE(<< "window = " << window);

            for (auto bucketLength : {HALF_HOUR, HOUR}) {
                LOG_TRACE(<< "bucket length = " << bucketLength);

                generateNoise(test, rng, window, HALF_HOUR, noise);
                rng.generateUniformSamples(0, permittedGenerators[i], 1, index);
                LOG_TRACE(<< "generator = " << index[0]);
                const auto& generator = generators[index[0]];

                values.assign(window / bucketLength, TFloatMeanAccumulator{});
                for (core_t::TTime time = 0; time < window; time += HALF_HOUR) {
                    values[time / bucketLength].add(20.0 * generator(startTime + time) +
                                                    noise[time / HALF_HOUR]);
                }

                maths::CTimeSeriesTestForSeasonality seasonality{
                    startTime, startTime, bucketLength, values};

                auto result = seasonality.decompose();

                if (result.print() != core::CContainerPrinter::print(expected[index[0]])) {
                    LOG_DEBUG(<< "got " << result.print() << ", expected "
                              << core::CContainerPrinter::print(expected[index[0]]));
                }

                std::size_t found[]{0, 0};
                for (const auto& component : result.seasonal()) {
                    ++found[std::find(expected[index[0]].begin(),
                                      expected[index[0]].end(), component.print()) ==
                            expected[index[0]].end()];
                }
                TP += static_cast<double>(found[0]);
                FN += static_cast<double>(expected[index[0]].size() - found[0]);
                FP += static_cast<double>(found[1]);
            }
        }
    }

    LOG_DEBUG(<< "recall = " << TP / (TP + FN));
    LOG_DEBUG(<< "accuracy = " << TP / (TP + FP));
    BOOST_REQUIRE(TP / (TP + FN) > 0.99);
    BOOST_REQUIRE(TP / (TP + FP) > 0.99);
}

BOOST_AUTO_TEST_CASE(testRealSpikeyDaily) {

    // Test on real spikey data with daily seasonality.

    TTimeDoublePrVec timeseries;
    core_t::TTime startTime;
    core_t::TTime endTime;
    BOOST_REQUIRE(test::CTimeSeriesTestData::parse(
        "testfiles/spikey_data.csv", timeseries, startTime, endTime,
        test::CTimeSeriesTestData::CSV_UNIX_REGEX));
    BOOST_REQUIRE(timeseries.size() > 0);

    LOG_DEBUG(<< "timeseries = "
              << core::CContainerPrinter::print(timeseries.begin(), timeseries.begin() + 10)
              << " ...");

    TTimeVec lastTest{timeseries[0].first, timeseries[0].first};
    TTimeVec windows{3 * DAY, 2 * WEEK};

    TFloatMeanAccumulatorVec values[2]{
        TFloatMeanAccumulatorVec(windows[0] / HOUR, TFloatMeanAccumulator{}),
        TFloatMeanAccumulatorVec(windows[1] / HOUR, TFloatMeanAccumulator{})};

    for (std::size_t i = 0; i < timeseries.size(); ++i) {
        core_t::TTime time;
        double value;
        std::tie(time, value) = timeseries[i];

        for (std::size_t j = 0; j < 2; ++j) {
            values[j][((time - lastTest[j]) % windows[j]) / HOUR].add(value);

            if (time > lastTest[j] + windows[j]) {
                maths::CTimeSeriesTestForSeasonality seasonality{
                    lastTest[j], lastTest[j], HOUR, values[j]};
                auto result = seasonality.decompose();
                BOOST_REQUIRE_EQUAL("[86400]", result.print());
                values[j].assign(windows[j] / HOUR, TFloatMeanAccumulator{});
                lastTest[j] = time;
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testRealTradingDays) {

    // Test on real data with weekdays/weekend.

    TTimeDoublePrVec timeseries;
    core_t::TTime startTime;
    core_t::TTime endTime;
    BOOST_REQUIRE(test::CTimeSeriesTestData::parse(
        "testfiles/thirty_minute_samples.csv", timeseries, startTime, endTime,
        test::CTimeSeriesTestData::CSV_ISO8601_REGEX,
        test::CTimeSeriesTestData::CSV_ISO8601_DATE_FORMAT));
    BOOST_REQUIRE(timeseries.size() > 0);

    LOG_DEBUG(<< "timeseries = "
              << core::CContainerPrinter::print(timeseries.begin(), timeseries.begin() + 10)
              << " ...");

    core_t::TTime lastTest{timeseries[0].first};
    core_t::TTime window{2 * WEEK};

    TFloatMeanAccumulatorVec values(window / HOUR, TFloatMeanAccumulator{});

    for (std::size_t i = 0; i < timeseries.size(); ++i) {
        core_t::TTime time;
        double value;
        std::tie(time, value) = timeseries[i];

        values[((time - lastTest) % window) / HOUR].add(value);

        if (time > lastTest + window) {
            maths::CTimeSeriesTestForSeasonality seasonality{lastTest, lastTest, HOUR, values};
            auto result = seasonality.decompose();
            LOG_DEBUG(<< result.print());
            BOOST_REQUIRE(result.print() == "[86400/(0,172800), 86400/(172800,604800), 604800/(0,172800)]" ||
                          result.print() == "[86400/(172800,604800), 604800/(0,172800), 604800/(172800,604800)]");
            values.assign(window / HOUR, TFloatMeanAccumulator{});
            lastTest = time;
        }
    }
}

BOOST_AUTO_TEST_CASE(testRealTradingDaysPlusOutliers) {

    // Test on real data with weekdays/weekend and outliers.

    TTimeDoublePrVec timeseries;
    core_t::TTime startTime;
    core_t::TTime endTime;
    BOOST_REQUIRE(test::CTimeSeriesTestData::parse(
        "testfiles/diurnal.csv", timeseries, startTime, endTime,
        test::CTimeSeriesTestData::CSV_UNIX_REGEX));
    BOOST_REQUIRE(timeseries.size() > 0);

    LOG_DEBUG(<< "timeseries = "
              << core::CContainerPrinter::print(timeseries.begin(), timeseries.begin() + 10)
              << " ...");

    core_t::TTime lastTest{timeseries[0].first};
    core_t::TTime window{2 * WEEK};

    TFloatMeanAccumulatorVec values(window / HOUR, TFloatMeanAccumulator{});

    for (std::size_t i = 0; i < timeseries.size(); ++i) {
        core_t::TTime time;
        double value;
        std::tie(time, value) = timeseries[i];

        values[((time - lastTest) % window) / HOUR].add(value);

        if (time > lastTest + window) {
            maths::CTimeSeriesTestForSeasonality seasonality{lastTest, lastTest, HOUR, values};
            auto result = seasonality.decompose();
            LOG_DEBUG(<< result.print());
            BOOST_REQUIRE(result.print() == "[86400/(0,172800), 86400/(172800,604800), 604800/(0,172800)]" ||
                          result.print() == "[86400/(0,172800), 86400/(172800,604800), 604800/(0,172800), 604800/(172800,604800)]");
            values.assign(window / HOUR, TFloatMeanAccumulator{});
            lastTest = time;
        }
    }
}

BOOST_AUTO_TEST_CASE(testRealSwitchingNoSeasonality) {

    // Test on real non-periodic switching data.

    TTimeDoublePrVec timeseries;
    core_t::TTime startTime;
    core_t::TTime endTime;
    BOOST_REQUIRE(test::CTimeSeriesTestData::parse(
        "testfiles/no_periods.csv", timeseries, startTime, endTime,
        test::CTimeSeriesTestData::CSV_ISO8601_REGEX,
        test::CTimeSeriesTestData::CSV_ISO8601_DATE_FORMAT));
    BOOST_REQUIRE(timeseries.size() > 0);

    LOG_DEBUG(<< "timeseries = "
              << core::CContainerPrinter::print(timeseries.begin(), timeseries.begin() + 10)
              << " ...");

    core_t::TTime lastTest{timeseries[0].first};
    core_t::TTime window{2 * WEEK};

    TFloatMeanAccumulatorVec values(window / HOUR, TFloatMeanAccumulator{});

    for (std::size_t i = 0; i < timeseries.size(); ++i) {
        core_t::TTime time;
        double value;
        std::tie(time, value) = timeseries[i];

        values[((time - lastTest) % window) / HOUR].add(value);

        if (time > lastTest + window) {
            maths::CTimeSeriesTestForSeasonality seasonality{lastTest, lastTest, HOUR, values};
            auto result = seasonality.decompose();
            BOOST_REQUIRE(result.print() == "[]");
            values.assign(window / HOUR, TFloatMeanAccumulator{});
            lastTest = time;
        }
    }
}

BOOST_AUTO_TEST_CASE(testSyntheticNonDiurnal) {

    // Test the accuracy for non-diurnal seasonal components with periods in the
    // range [DAY / 5, 5 * DAY].

    TGeneratorVec generators{smoothDaily, spikeyDaily};
    core_t::TTime startTime{10000};

    test::CRandomNumbers rng;

    TFloatMeanAccumulatorVec values;
    TDoubleVec noise;
    TSizeVec index;
    TSizeVec repeats;

    double TP[3]{0.0};
    double FN[3]{0.0};
    double FP{0.0};

    for (std::size_t test = 0; test < 100; ++test) {
        if ((test + 1) % 10 == 0) {
            LOG_DEBUG(<< "test " << test + 1 << " / 100");
        }

        for (auto window : {WEEK, 2 * WEEK, 25 * DAY}) {
            LOG_TRACE(<< "window = " << window);

            double periodScale{[&] {
                TDoubleVec result;
                rng.generateUniformSamples(1.0, 5.0, 1, result);
                return test % 2 == 0 ? result[0] : 1.0 / result[0];
            }()};

            for (auto bucketLength : {HALF_HOUR, HOUR}) {
                LOG_TRACE(<< "bucket length = " << bucketLength);

                core_t::TTime period{maths::CIntegerTools::floor(
                    static_cast<core_t::TTime>(static_cast<double>(DAY) / periodScale),
                    bucketLength)};
                periodScale = static_cast<double>(DAY) / static_cast<double>(period);
                if (periodScale == 1.0 || window < 5 * period) {
                    continue;
                }

                generateNoise(test, rng, window, FIVE_MINS, noise);
                rng.generateUniformSamples(0, 2, 1, index);
                rng.generateUniformSamples(3, 20, 1, repeats);
                auto generator = generators[index[0]];

                values.assign(window / bucketLength, TFloatMeanAccumulator{});
                for (core_t::TTime time = 0; time < window; time += FIVE_MINS) {
                    values[time / bucketLength].add(
                        20.0 * scale(periodScale, startTime + time, generator) +
                        noise[time / FIVE_MINS]);
                }

                maths::CTimeSeriesTestForSeasonality seasonality{
                    startTime, startTime, bucketLength, values};
                auto result = seasonality.decompose();

                double found[]{0.0, 0.0, 0.0, 0.0};
                if (result.print() != "[" + std::to_string(period) + "]") {
                    LOG_DEBUG(<< "got " << result.print() << ", expected [" << period << "]");
                }

                for (const auto& component : result.seasonal()) {
                    core_t::TTime componentPeriod{component.seasonalTime()->period()};
                    double error{static_cast<double>(std::min(componentPeriod % period,
                                                              period - componentPeriod % period)) /
                                 static_cast<double>(period)};
                    if (error == 0.0) {
                        found[0] = 1.0;
                    }
                    if (error < 0.01) {
                        found[1] = 1.0;
                    }
                    if (error < 0.05) {
                        found[2] = 1.0;
                    }
                    if (error >= 0.05) {
                        found[3] += 1.0;
                    }
                }
                for (std::size_t i = 0; i < 3; ++i) {
                    TP[i] += found[i];
                    FN[i] += 1.0 - found[i];
                }
                FP += found[3];
            }
        }
    }

    LOG_DEBUG(<< "recall @ 0% error = " << TP[0] / (TP[0] + FN[0]));
    LOG_DEBUG(<< "recall @ 1% error = " << TP[1] / (TP[1] + FN[1]));
    LOG_DEBUG(<< "recall @ 5% error = " << TP[2] / (TP[2] + FN[2]));
    LOG_DEBUG(<< "accuracy @ 0% error = " << TP[0] / (TP[0] + FP));
    LOG_DEBUG(<< "accuracy @ 1% error = " << TP[1] / (TP[1] + FP));
    LOG_DEBUG(<< "accuracy @ 5% error = " << TP[2] / (TP[2] + FP));
    BOOST_REQUIRE(TP[0] / (TP[0] + FN[0]) > 0.98);
    BOOST_REQUIRE(TP[1] / (TP[1] + FN[1]) > 0.99);
    BOOST_REQUIRE(TP[2] / (TP[2] + FN[2]) > 0.99);
    BOOST_REQUIRE(TP[0] / (TP[0] + FP) > 0.94);
    BOOST_REQUIRE(TP[1] / (TP[1] + FP) > 0.94);
    BOOST_REQUIRE(TP[2] / (TP[2] + FP) > 0.94);
}

BOOST_AUTO_TEST_CASE(testSyntheticSparseDaily) {

    // Test a synthetic time series with daily seasonality and periodically
    // missing values.

    TStrVec type{"Daily", "No seasonality"};

    test::CRandomNumbers rng;

    TFloatMeanAccumulatorVec values;
    TDoubleVec noise;

    for (std::size_t test = 0; test < 2; ++test) {
        LOG_DEBUG(<< type[test]);

        values.assign(7 * 48, TFloatMeanAccumulator{});
        for (std::size_t day = 0, i = 0, j = 0; day < 7; ++day, ++j) {
            for (auto value :
                 {0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 20.0, 18.0, 10.0, 4.0,  4.0, 4.0, 4.0, 5.0,
                  6.0, 8.0, 9.0,  9.0,  10.0, 10.0, 8.0, 4.0, 3.0, 1.0,
                  0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  3.0, 1.0}) {
                if (value > 0.0) {
                    rng.generateUniformSamples(-1.0, 1.0, 1, noise);
                    values[i].add(test == 0 ? value : noise[0]);
                }
                ++i;
            }

            if (day > 3) {
                maths::CTimeSeriesTestForSeasonality seasonality{0, 0, HALF_HOUR, values};
                auto result = seasonality.decompose();
                BOOST_REQUIRE(result.print() == (test == 0 ? "[86400]" : "[]"));
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testSyntheticSparseWeekly) {

    // Test a synthetic time series with weekly seasonality and periodically
    // missing values.

    TStrVec type{"Weekly", "No seasonality"};

    test::CRandomNumbers rng;

    TFloatMeanAccumulatorVec values;
    TDoubleVec noise;

    for (std::size_t test = 0; test < 2; ++test) {
        LOG_DEBUG(<< type[test]);

        values.assign(5 * 168, TFloatMeanAccumulator{});

        for (std::size_t week = 0, i = 0, j = 0; week < 5; ++week, ++j) {
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
                    rng.generateUniformSamples(-1.0, 1.0, 1, noise);
                    values[i].add(test == 0 ? value : noise[0]);
                }
                ++i;
            }

            if (week >= 3) {
                maths::CTimeSeriesTestForSeasonality seasonality{0, 0, HOUR, values};
                auto result = seasonality.decompose();
                LOG_DEBUG(<< result.print());
                BOOST_REQUIRE(result.print() == (test == 0 ? "[86400, 604800]" : "[]"));
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testSyntheticWithOutliers) {

    // Test synthetic timeseries data with pepper and salt outliers.

    core_t::TTime startTime{10000};

    test::CRandomNumbers rng;

    TDoubleVec noise;
    TSizeVec outliers;
    TDoubleVec spikeOrTroughSelector;
    TFloatMeanAccumulatorVec values;

    for (auto period : {DAY, WEEK}) {
        LOG_DEBUG(<< "period = " << period);

        for (auto window : {WEEK, 2 * WEEK, 4 * WEEK}) {
            if (window < 2 * period) {
                continue;
            }
            LOG_TRACE(<< "window = " << window);

            for (auto bucketLength : {HALF_HOUR, HOUR}) {
                core_t::TTime buckets{window / bucketLength};
                std::size_t numberOutliers{
                    static_cast<std::size_t>(0.05 * static_cast<double>(buckets))};
                rng.generateUniformSamples(0, buckets, numberOutliers, outliers);
                rng.generateUniformSamples(0, 1.0, numberOutliers, spikeOrTroughSelector);
                rng.generateNormalSamples(0.0, 9.0, buckets, noise);
                std::sort(outliers.begin(), outliers.end());

                values.assign(buckets, TFloatMeanAccumulator{});
                for (core_t::TTime time = startTime; time < startTime + window;
                     time += bucketLength) {
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
                }

                maths::CTimeSeriesTestForSeasonality seasonality{
                    startTime, startTime, bucketLength, values};
                auto result = seasonality.decompose();
                LOG_DEBUG(<< result.print());
                BOOST_REQUIRE(result.print() == "[" + std::to_string(period) + "]");
            }
        }
    }

    LOG_DEBUG(<< "Weekdays/weekend");

    TDoubleVec modulation{0.2, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0};

    for (auto window : {2 * WEEK, 4 * WEEK}) {
        LOG_DEBUG(<< "window length = " << window);

        for (auto bucketLength : {HALF_HOUR, HOUR}) {
            core_t::TTime buckets{window / bucketLength};
            std::size_t numberOutliers{
                static_cast<std::size_t>(0.05 * static_cast<double>(buckets))};
            rng.generateUniformSamples(0, buckets, numberOutliers, outliers);
            rng.generateUniformSamples(0, 1.0, numberOutliers, spikeOrTroughSelector);
            rng.generateNormalSamples(0.0, 9.0, buckets, noise);
            std::sort(outliers.begin(), outliers.end());

            values.assign(buckets, TFloatMeanAccumulator{});
            for (core_t::TTime time = startTime; time < startTime + window; time += bucketLength) {
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
            }

            maths::CTimeSeriesTestForSeasonality seasonality{
                startTime, startTime, bucketLength, std::move(values)};
            auto result = seasonality.decompose();
            LOG_DEBUG(<< result.print());
            BOOST_REQUIRE(result.print() == "[86400/(0,172800), 86400/(172800,604800)]" ||
                          result.print() == "[86400/(0,172800), 86400/(172800,604800), 604800/(172800,604800)]");
        }
    }
}

BOOST_AUTO_TEST_CASE(testSyntheticMixtureOfSeasonalities) {

    // Test the accuracy with which we decompose a synthetic time series into
    // its multiple constitute components.

    TGeneratorVec generators[]{
        {spikeyDaily,
         [](core_t::TTime time) { return scale(16.0 / 24.0, time, spikeyDaily); }},
        {smoothDaily,
         [](core_t::TTime time) {
             return 2.0 * scale(36.0 / 24.0, time, spikeyDaily);
         }},
        {smoothDaily,
         [](core_t::TTime time) { return scale(18.0 / 24.0, time, smoothDaily); }}};
    TStrVecVec expected{{"86400", "129600"}, {"86400", "57600"}, {"86400", "115200"}};

    core_t::TTime startTime{10000};

    test::CRandomNumbers rng;

    TDoubleVec noise;
    TFloatMeanAccumulatorVec values;
    TSizeVec index;

    double TP{0.0};
    double FN{0.0};
    double FP{0.0};

    for (std::size_t test = 0; test < 100; ++test) {
        if ((test + 1) % 10 == 0) {
            LOG_DEBUG(<< "test " << test + 1 << " / 100");
        }

        for (auto window : {16 * DAY, 4 * WEEK}) {
            LOG_TRACE(<< "window = " << window);

            TDoubleVec periodScales;
            rng.generateUniformSamples(1.0, 5.0, 2, periodScales);
            periodScales[0] = 1.0 / periodScales[0];

            for (auto bucketLength : {HALF_HOUR, HOUR}) {
                LOG_TRACE(<< "bucket length = " << bucketLength);

                for (auto periodScale : periodScales) {
                    core_t::TTime period{maths::CIntegerTools::floor(
                        static_cast<core_t::TTime>(static_cast<double>(DAY) / periodScale),
                        bucketLength)};
                    periodScale = static_cast<double>(DAY) / static_cast<double>(period);
                }

                generateNoise(test, rng, window, FIVE_MINS, noise);
                rng.generateUniformSamples(0, 3, 1, index);
                LOG_TRACE(<< "generator = " << index[0]);
                const auto& generator = generators[index[0]];

                values.assign(window / bucketLength, TFloatMeanAccumulator{});
                for (core_t::TTime time = 0; time < window; time += FIVE_MINS) {
                    double value{0.0};
                    for (std::size_t i = 0; i < 2; ++i) {
                        value += 20.0 * generator[i](startTime + time);
                    }
                    values[time / bucketLength].add(value + noise[time / FIVE_MINS]);
                }

                maths::CTimeSeriesTestForSeasonality seasonality{
                    startTime, startTime, bucketLength, values};
                auto result = seasonality.decompose();

                std::size_t found[]{0, 0};
                for (const auto& component : result.seasonal()) {
                    ++found[std::find(expected[index[0]].begin(),
                                      expected[index[0]].end(), component.print()) ==
                            expected[index[0]].end()];
                }
                TP += static_cast<double>(found[0]);
                FN += static_cast<double>(2 - found[0]);
                FP += static_cast<double>(found[1]);
                if (found[0] < 2 || found[1] > 0) {
                    LOG_DEBUG(<< "got " << result.print() << ", expected "
                              << core::CContainerPrinter::print(expected[index[0]]));
                }
            }
        }
    }

    LOG_DEBUG(<< "recall = " << TP / (TP + FN));
    LOG_DEBUG(<< "accuracy = " << TP / (TP + FP));
    BOOST_REQUIRE(TP / (TP + FN) > 0.96);
    BOOST_REQUIRE(TP / (TP + FP) > 0.99);
}

BOOST_AUTO_TEST_CASE(testSyntheticDiurnalWithLinearScaling) {

    // Test the ability to correctly decompose a time series with diurnal seasonal
    // components in the presence of piecewise constant random linear scaling events.

    std::size_t segmentSupport[][2]{{100, 200}, {600, 900}};
    double scaleSupport[][2]{{4.0, 6.0}, {0.2, 0.4}};
    TGeneratorVec generators{smoothDaily, spikeyDaily};
    core_t::TTime startTime{0};

    test::CRandomNumbers rng;

    TDoubleVec noise;
    TSizeVec index;
    TFloatMeanAccumulatorVec values;

    double TP{0.0};
    double FN{0.0};

    for (std::size_t test = 0; test < 100; ++test) {
        if ((test + 1) % 10 == 0) {
            LOG_DEBUG(<< "test " << test + 1 << " / 100");
        }

        for (auto window : {3 * WEEK, 4 * WEEK}) {
            LOG_TRACE(<< "window = " << window);

            core_t::TTime endTime{startTime + window};

            TTimeVec segments;
            TDoubleVec scales{1.0};
            for (std::size_t i = 0; i < 2; ++i) {
                TSizeVec segment;
                TDoubleVec scale;
                rng.generateUniformSamples(segmentSupport[i][0],
                                           segmentSupport[i][1], 1, segment);
                rng.generateUniformSamples(scaleSupport[i][0], scaleSupport[i][1], 1, scale);
                segments.push_back(startTime + segment[0] * HALF_HOUR);
                scales.push_back(scale[0]);
            }
            segments.push_back(endTime);

            auto trend = [&](core_t::TTime time) {
                auto i = std::lower_bound(segments.begin(), segments.end(), time);
                return 20.0 * scales[i - segments.begin()] * generators[index[0]](time);
            };

            rng.generateNormalSamples(0.0, 1.0, window / HALF_HOUR, noise);
            rng.generateUniformSamples(0, 2, 1, index);

            values.assign(window / HALF_HOUR, TFloatMeanAccumulator{});
            for (core_t::TTime time = 0; time < window; time += HALF_HOUR) {
                std::size_t bucket(time / HALF_HOUR);
                values[bucket].add(trend(startTime + time) + noise[bucket]);
            }

            maths::CTimeSeriesTestForSeasonality seasonality{startTime, startTime,
                                                             HALF_HOUR, values};
            auto result = seasonality.decompose();

            if (result.print() != "[86400]") {
                LOG_DEBUG(<< "got " << result.print() << ", expected [86400]");
            }
            TP += result.print() == "[86400]" ? 1.0 : 0.0;
            FN += result.print() == "[86400]" ? 0.0 : 1.0;
        }
    }

    LOG_DEBUG(<< "Recall = " << TP / (TP + FN));
    BOOST_REQUIRE(TP / (TP + FN) > 0.99);
}

BOOST_AUTO_TEST_CASE(testSyntheticNonDiurnalWithLinearTrend) {

    // Test the ability to correctly decompose a time series with seasonal
    // components and a linear trend.

    TGeneratorVec generators{smoothDaily, spikeyDaily};
    core_t::TTime startTime{10000};

    test::CRandomNumbers rng;

    TFloatMeanAccumulatorVec values;
    TDoubleVec noise;
    TSizeVec index;

    double TP[3]{0.0};
    double FN[3]{0.0};
    double FP{0.0};

    for (std::size_t test = 0; test < 100; ++test) {
        if ((test + 1) % 10 == 0) {
            LOG_DEBUG(<< "test " << test + 1 << " / 100");
        }

        for (auto window : {WEEK, 2 * WEEK, 25 * DAY}) {
            LOG_TRACE(<< "window = " << window);

            double periodScale{[&] {
                TDoubleVec result;
                rng.generateUniformSamples(1.0, 5.0, 1, result);
                return test % 2 == 0 ? result[0] : 1.0 / result[0];
            }()};

            for (auto bucketLength : {HALF_HOUR, HOUR}) {
                LOG_TRACE(<< "bucket length = " << bucketLength);

                core_t::TTime period{maths::CIntegerTools::floor(
                    static_cast<core_t::TTime>(static_cast<double>(DAY) / periodScale),
                    bucketLength)};
                periodScale = static_cast<double>(DAY) / static_cast<double>(period);
                if (periodScale == 1.0 || window < 5 * period) {
                    continue;
                }

                generateNoise(test, rng, window, FIVE_MINS, noise);
                rng.generateUniformSamples(0, 2, 1, index);
                auto generator = generators[index[0]];

                values.assign(window / bucketLength, TFloatMeanAccumulator{});
                for (core_t::TTime time = 0; time < window; time += FIVE_MINS) {
                    std::size_t bucket(time / bucketLength);
                    values[bucket].add(0.5 * static_cast<double>(bucket) +
                                       20.0 * scale(periodScale, startTime + time, generator) +
                                       noise[time / FIVE_MINS]);
                }

                maths::CTimeSeriesTestForSeasonality seasonality{
                    startTime, startTime, bucketLength, values};
                auto result = seasonality.decompose();

                double found[]{0.0, 0.0, 0.0, 0.0};
                if (result.print() != "[" + std::to_string(period) + "]") {
                    LOG_DEBUG(<< "got " << result.print() << ", expected [" << period << "]");
                }

                for (const auto& component : result.seasonal()) {
                    core_t::TTime componentPeriod{component.seasonalTime()->period()};
                    double error{static_cast<double>(std::min(componentPeriod % period,
                                                              period - componentPeriod % period)) /
                                 static_cast<double>(period)};
                    if (error == 0.0) {
                        found[0] = 1.0;
                    }
                    if (error < 0.01) {
                        found[1] = std::max(found[1], 1.0);
                    }
                    if (error < 0.05) {
                        found[2] = std::max(found[2], 1.0);
                    }
                    if (error >= 0.05) {
                        found[3] += 1.0;
                    }
                }
                for (std::size_t i = 0; i < 3; ++i) {
                    TP[i] += found[i];
                    FN[i] += 1.0 - found[i];
                }
                FP += found[3];
            }
        }
    }

    LOG_DEBUG(<< "recall @ 0% error = " << TP[0] / (TP[0] + FN[0]));
    LOG_DEBUG(<< "recall @ 1% error = " << TP[1] / (TP[1] + FN[1]));
    LOG_DEBUG(<< "recall @ 5% error = " << TP[2] / (TP[2] + FN[2]));
    LOG_DEBUG(<< "accuracy @ 0% error = " << TP[0] / (TP[0] + FP));
    LOG_DEBUG(<< "accuracy @ 1% error = " << TP[1] / (TP[1] + FP));
    LOG_DEBUG(<< "accuracy @ 5% error = " << TP[2] / (TP[2] + FP));
    BOOST_REQUIRE(TP[0] / (TP[0] + FN[0]) > 0.97);
    BOOST_REQUIRE(TP[1] / (TP[1] + FN[1]) > 0.99);
    BOOST_REQUIRE(TP[2] / (TP[2] + FN[2]) > 0.99);
    BOOST_REQUIRE(TP[0] / (TP[0] + FP) > 0.93);
    BOOST_REQUIRE(TP[1] / (TP[1] + FP) > 0.93);
    BOOST_REQUIRE(TP[2] / (TP[2] + FP) > 0.93);
}

BOOST_AUTO_TEST_CASE(testSyntheticDiurnalWithPiecewiseLinearTrend) {

    // Test the ability to correctly decompose a time series with diurnal seasonal
    // components and a piecewise linear trend.

    using TLinearModel = std::function<double(core_t::TTime)>;
    using TLinearModelVec = std::vector<TLinearModel>;

    std::size_t segmentSupport[][2]{{100, 200}, {600, 900}};
    double slopeSupport[][2]{{0.5, 1.0}, {-1.0, -0.5}};
    double interceptSupport[][2]{{-10.0, 5.0}, {10.0, 20.0}};
    TGeneratorVec generators{smoothDaily, spikeyDaily};
    core_t::TTime startTime{0};

    test::CRandomNumbers rng;

    TFloatMeanAccumulatorVec values;
    TDoubleVec noise;
    TSizeVec index;

    double TP{0.0};
    double FN{0.0};

    for (std::size_t test = 0; test < 20; ++test) {
        if ((test + 1) % 2 == 0) {
            LOG_DEBUG(<< "test " << test + 1 << " / 20");
        }

        for (auto window : {3 * WEEK, 4 * WEEK}) {
            LOG_TRACE(<< "window = " << window);

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
                segments.push_back(startTime + segment[0] * HALF_HOUR);
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

            rng.generateNormalSamples(0.0, 1.0, window / HALF_HOUR, noise);
            rng.generateUniformSamples(0, 2, 1, index);

            values.assign(window / HALF_HOUR, TFloatMeanAccumulator{});
            for (core_t::TTime time = 0; time < window; time += HALF_HOUR) {
                std::size_t bucket(time / HALF_HOUR);
                values[bucket].add(trend(startTime + time) + noise[bucket]);
            }

            maths::CTimeSeriesTestForSeasonality seasonality{startTime, startTime,
                                                             HALF_HOUR, values};
            auto result = seasonality.decompose();

            if (result.print() != "[86400]") {
                LOG_DEBUG(<< "got " << result.print() << ", expected [86400]");
            }
            TP += result.print() == "[86400]" ? 1.0 : 0.0;
            FN += result.print() == "[86400]" ? 0.0 : 1.0;
        }
    }

    LOG_DEBUG(<< "Recall = " << TP / (TP + FN));
    BOOST_REQUIRE(TP / (TP + FN) > 0.99);
}

BOOST_AUTO_TEST_CASE(testModelledSeasonalityWithNoChange) {

    // Check we keep the modelled seasonality if it hasn't changed.

    TGeneratorVec generators{smoothDaily, spikeyDaily, smoothWeekly, weekends};
    TDiurnalTimeVecVec modelledComponents{{{0, 0, WEEK, DAY}},
                                          {{0, 0, WEEK, DAY}},
                                          {{0, 0, WEEK, WEEK}},
                                          {{5 * DAY, 0 * DAY, 2 * DAY, DAY},
                                           {5 * DAY, 0 * DAY, 2 * DAY, WEEK},
                                           {5 * DAY, 2 * DAY, 7 * DAY, DAY},
                                           {5 * DAY, 2 * DAY, 7 * DAY, WEEK}}};

    test::CRandomNumbers rng;

    TFloatMeanAccumulatorVec values;
    TDoubleVec noise;
    TSizeVec index;

    for (std::size_t test = 0; test < 20; ++test) {
        if ((test + 1) % 2 == 0) {
            LOG_DEBUG(<< "test " << test + 1 << " / 20");
        }

        for (auto window : {3 * WEEK, 4 * WEEK}) {
            LOG_TRACE(<< "window = " << window);

            generateNoise(test, rng, window, FIVE_MINS, noise);
            rng.generateUniformSamples(0, 2, 1, index);
            LOG_TRACE(<< "index = " << index[0]);
            auto generator = generators[index[0]];

            values.assign(window / HOUR, TFloatMeanAccumulator{});
            for (core_t::TTime time = 0; time < window; time += FIVE_MINS) {
                values[time / HOUR].add(5.0 * generator(time) + noise[time / FIVE_MINS]);
            }

            maths::CTimeSeriesTestForSeasonality seasonality{0, 0, HOUR, values};
            for (const auto& time : modelledComponents[index[0]]) {
                seasonality.addModelledSeasonality(time, 24);
            }

            auto result = seasonality.decompose();

            // We should only detect the components we have already which we don't
            // bother to return.
            BOOST_REQUIRE(result.seasonal().empty());
        }
    }
}

BOOST_AUTO_TEST_CASE(testModelledSeasonalityWithChange) {

    // Simulate the seasonal components having changed in the test window and
    // check that we correctly identify the new ones.

    TGeneratorVec generators{smoothDaily, weekends};
    TDiurnalTimeVecVec modelledComponents{{{5 * DAY, 0 * DAY, 2 * DAY, DAY},
                                           {5 * DAY, 0 * DAY, 2 * DAY, WEEK},
                                           {5 * DAY, 2 * DAY, 7 * DAY, DAY},
                                           {5 * DAY, 2 * DAY, 7 * DAY, WEEK}},
                                          {{0, 0, WEEK, DAY}}};
    TStrVecVec expected{{"86400"},
                        {"86400/(0,172800)", "86400/(172800,604800)",
                         "604800/(0,172800)", "604800/(172800,604800)"}};

    test::CRandomNumbers rng;

    TFloatMeanAccumulatorVec values;
    TDoubleVec noise;
    TSizeVec index;

    double TP{0.0};
    double FN{0.0};
    double FP{0.0};

    for (std::size_t test = 0; test < 20; ++test) {
        if ((test + 1) % 2 == 0) {
            LOG_DEBUG(<< "test " << test + 1 << " / 20");
        }

        for (auto window : {3 * WEEK, 4 * WEEK}) {
            LOG_TRACE(<< "window = " << window);

            generateNoise(test, rng, window, FIVE_MINS, noise);
            rng.generateUniformSamples(0, 2, 1, index);
            LOG_TRACE(<< "index = " << index[0]);
            auto generator = generators[index[0]];

            values.assign(window / HOUR, TFloatMeanAccumulator{});
            for (core_t::TTime time = 0; time < window; time += FIVE_MINS) {
                values[time / HOUR].add(10.0 * generator(time) + noise[time / FIVE_MINS]);
            }

            maths::CTimeSeriesTestForSeasonality seasonality{0, 0, HOUR, values};
            for (const auto& time : modelledComponents[index[0]]) {
                seasonality.addModelledSeasonality(time, 24);
            }

            auto result = seasonality.decompose();

            if (result.print() != core::CContainerPrinter::print(expected[index[0]])) {
                LOG_DEBUG(<< "got " << result.print() << ", expected "
                          << core::CContainerPrinter::print(expected[index[0]]));
            }

            std::size_t found[]{0, 0};
            for (const auto& component : result.seasonal()) {
                ++found[std::find(expected[index[0]].begin(), expected[index[0]].end(),
                                  component.print()) == expected[index[0]].end()];
            }
            TP += static_cast<double>(found[0]);
            FN += static_cast<double>(expected[index[0]].size() - found[0]);
            FP += static_cast<double>(found[1]);
        }
    }

    LOG_DEBUG(<< "recall = " << TP / (TP + FN));
    LOG_DEBUG(<< "accuracy = " << TP / (TP + FP));
    BOOST_REQUIRE(TP / (TP + FN) > 0.99);
    BOOST_REQUIRE(TP / (TP + FP) > 0.99);
}

BOOST_AUTO_TEST_CASE(testNewComponentInitialValues) {

    // Test that the initial values for the seasonal components identified match
    // the supplied values.

    TGeneratorVec generators{smoothDaily, spikeyDaily, weekends};

    test::CRandomNumbers rng;

    TFloatMeanAccumulatorVec values;
    TDoubleVec predictions;
    TSizeVec startTimes;
    rng.generateUniformSamples(0, 10000000, 10, startTimes);
    TTimeVec expectedWindowStarts{0, 2 * DAY, 0, 2 * DAY};
    TTimeVec expectedWindowEnds{2 * DAY, 7 * DAY, 2 * DAY, 7 * DAY};
    TTimeVec expectedPeriods{DAY, DAY, WEEK, WEEK};

    for (std::size_t test = 0; test < 10; ++test) {
        LOG_DEBUG(<< "test " << test + 1 << " / 10");

        std::size_t index{test % generators.size()};
        auto generator = generators[index];
        core_t::TTime startTime{
            HALF_HOUR * (static_cast<core_t::TTime>(startTimes[test]) / HALF_HOUR)};

        values.assign(3 * WEEK / HALF_HOUR, TFloatMeanAccumulator{});
        for (core_t::TTime time = 0; time < 3 * WEEK; time += HALF_HOUR) {
            values[time / HALF_HOUR].add(10.0 * generator(startTime + time));
        }

        maths::CTimeSeriesTestForSeasonality seasonality{startTime, startTime,
                                                         HALF_HOUR, values};
        auto result = seasonality.decompose();

        // Expect agreement with generator.
        predictions.assign(values.size(), 0.0);
        for (const auto& component : result.seasonal()) {
            BOOST_REQUIRE_EQUAL(startTime, component.initialValuesStartTime());
            BOOST_REQUIRE_EQUAL(startTime + 3 * WEEK, component.initialValuesEndTime());
            for (std::size_t i = 0; i < component.initialValues().size(); ++i) {
                predictions[i] +=
                    maths::CBasicStatistics::mean(component.initialValues()[i]);
            }
        }
        for (core_t::TTime time = 0; time < 3 * WEEK; time += HALF_HOUR) {
            BOOST_REQUIRE_CLOSE_ABSOLUTE(10.0 * generator(startTime + time),
                                         predictions[time / HALF_HOUR], 1e-4);
        }

        // Check the seasonal time is initialized correctly.
        switch (index) {
        case 0:
        case 1:
            for (const auto& component : result.seasonal()) {
                auto time = component.seasonalTime();
                BOOST_REQUIRE_EQUAL(false, time->windowed());
                BOOST_REQUIRE_EQUAL(0, time->windowRepeatStart());
                BOOST_REQUIRE_EQUAL(0, time->windowStart());
                BOOST_REQUIRE_EQUAL(WEEK, time->windowEnd());
                BOOST_REQUIRE_EQUAL(DAY, time->period());
            }
            break;
        default:
            BOOST_REQUIRE_EQUAL(4, result.seasonal().size());
            for (std::size_t i = 0; i < result.seasonal().size(); ++i) {
                auto time = result.seasonal()[i].seasonalTime();
                BOOST_REQUIRE_EQUAL(true, time->windowed());
                BOOST_REQUIRE_EQUAL(5 * DAY / HALF_HOUR, time->windowRepeatStart() / HALF_HOUR);
                BOOST_REQUIRE_EQUAL(expectedWindowStarts[i], time->windowStart());
                BOOST_REQUIRE_EQUAL(expectedWindowEnds[i], time->windowEnd());
                BOOST_REQUIRE_EQUAL(expectedPeriods[i], time->period());
            }
            break;
        }
    }
}

BOOST_AUTO_TEST_CASE(testNewComponentInitialValuesWithPiecewiseLinearScaling) {

    // Test that the initial values for the seasonal components when there
    // are linear scalings in the test values.
}

BOOST_AUTO_TEST_CASE(testNewTrendSummary) {

    // Test we get the trend initial values we expect with either no trend
    // or a linear ramp.

    TGeneratorVec trends{constant, ramp};
    TGeneratorVec seasons{smoothDaily, spikeyDaily, weekends};

    test::CRandomNumbers rng;

    TFloatMeanAccumulatorVec values;
    TDoubleVec predictions;
    TSizeVec startTimes;
    rng.generateUniformSamples(0, 10000000, 10, startTimes);

    for (std::size_t test = 0; test < 10; ++test) {
        LOG_DEBUG(<< "test " << test + 1 << " / 10");

        core_t::TTime startTime{HOUR * (static_cast<core_t::TTime>(startTimes[test]) / HOUR)};
        auto trend = trends[test % trends.size()];
        auto season = seasons[test % seasons.size()];

        values.assign(4 * WEEK / HOUR, TFloatMeanAccumulator{});
        for (core_t::TTime time = 0; time < 4 * WEEK; time += HOUR) {
            values[time / HOUR].add(100.0 * trend(startTime + time) +
                                    10.0 * season(startTime + time));
        }

        maths::CTimeSeriesTestForSeasonality seasonality{startTime, startTime, HOUR, values};
        auto result = seasonality.decompose();

        BOOST_REQUIRE(result.trend() != nullptr);

        predictions.assign(values.size(), 0.0);
        BOOST_REQUIRE_EQUAL(startTime, result.trend()->initialValuesStartTime());
        BOOST_REQUIRE_EQUAL(startTime + 4 * WEEK, result.trend()->initialValuesEndTime());
        for (std::size_t i = 0; i < result.trend()->initialValues().size(); ++i) {
            predictions[i] +=
                maths::CBasicStatistics::mean(result.trend()->initialValues()[i]);
        }

        // Expect slope to match.
        TFloatMeanAccumulator expectedMeanSlope;
        TFloatMeanAccumulator meanSlope;
        for (core_t::TTime time = HOUR; time < 4 * WEEK; time += HOUR) {
            expectedMeanSlope.add(
                100.0 * (trend(startTime + time) - trend(startTime + time - HOUR)));
            meanSlope.add(predictions[time / HOUR] - predictions[time / HOUR - 1]);
        }
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            static_cast<double>(maths::CBasicStatistics::mean(expectedMeanSlope)),
            static_cast<double>(maths::CBasicStatistics::mean(meanSlope)), 2e-3);

        // Expect agreement with generator.
        for (const auto& component : result.seasonal()) {
            for (std::size_t i = 0; i < component.initialValues().size(); ++i) {
                BOOST_REQUIRE_EQUAL(startTime, component.initialValuesStartTime());
                BOOST_REQUIRE_EQUAL(startTime + 4 * WEEK, component.initialValuesEndTime());
                predictions[i] +=
                    maths::CBasicStatistics::mean(component.initialValues()[i]);
            }
        }

        for (core_t::TTime time = 0; time < 4 * WEEK; time += HOUR) {
            BOOST_REQUIRE_CLOSE_ABSOLUTE(100.0 * trend(startTime + time) +
                                             10.0 * season(startTime + time),
                                         predictions[time / HOUR], 10.0);
        }
    }
}

BOOST_AUTO_TEST_CASE(testNewTrendSummaryPiecewiseLinearTrend) {
}

BOOST_AUTO_TEST_CASE(testWithSuppliedPredictor) {

    // Check the initial values in the case that we have a supplied predictor.

    using TBoolVec = std::vector<bool>;

    auto daily = [&](core_t::TTime time) {
        return std::sin(boost::math::double_constants::pi *
                        static_cast<double>(time % DAY) / static_cast<double>(DAY));
    };
    auto weekly = [](core_t::TTime time) {
        double values[]{2.0, 2.1, 2.3, 2.2, 1.8, 1.6, 1.4,
                        1.0, 1.2, 1.8, 1.5, 1.7, 1.8, 1.9};
        return values[2 * (time % WEEK) / DAY];
    };
    core_t::TTime startTime{1000000};

    TFloatMeanAccumulatorVec values(3 * WEEK / HOUR);
    for (core_t::TTime time = 0; time < 3 * WEEK; time += HOUR) {
        values[time / HOUR].add(daily(startTime + time) + weekly(startTime + time));
    }

    maths::CTimeSeriesTestForSeasonality seasonality{startTime, startTime, HOUR, values};
    seasonality.addModelledSeasonality(maths::CDiurnalTime{0, 0, WEEK, DAY}, 24);
    seasonality.modelledSeasonalityPredictor([](core_t::TTime time, const TBoolVec&) {
        return std::sin(boost::math::double_constants::pi *
                        static_cast<double>(time % DAY) / static_cast<double>(DAY));
    });

    auto result = seasonality.decompose();

    for (core_t::TTime time = 0; time < 3 * WEEK; time += HOUR) {
        double prediction{0.0};
        for (const auto& component : result.seasonal()) {
            prediction +=
                maths::CBasicStatistics::mean(component.initialValues()[time / HOUR]);
        }
        BOOST_REQUIRE_CLOSE_ABSOLUTE(prediction, weekly(startTime + time), 1e-4);
    }
}

BOOST_AUTO_TEST_CASE(testMinimumPeriod) {

    // Test with two components only one of which should be accepted based on the
    // minimum period constraint.
}

BOOST_AUTO_TEST_CASE(testMaximumNumberOfComponents) {

    // Test with fewer permitted components than components present: should retain
    // the most useful.
}

BOOST_AUTO_TEST_SUITE_END()
