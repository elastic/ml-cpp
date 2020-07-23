/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/Constants.h>
#include <core/CoreTypes.h>

#include <core/CContainerPrinter.h>
#include <maths/CBasicStatistics.h>
#include <maths/CIntegerTools.h>
#include <maths/CTimeSeriesTestForSeasonality.h>
#include <maths/MathsTypes.h>

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
using TFloatMeanAccumulator =
    maths::CBasicStatistics::SSampleMean<maths::CFloatStorage>::TAccumulator;
using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;

const core_t::TTime TEN_MINS{600};
const core_t::TTime HALF_HOUR{core::constants::HOUR / 2};
const core_t::TTime HOUR{core::constants::HOUR};
const core_t::TTime DAY{core::constants::DAY};
const core_t::TTime WEEK{core::constants::WEEK};
}

BOOST_AUTO_TEST_CASE(testSyntheticNoSeasonality) {

    // Test FP % for a variety of synthetic time series with not seasonality.

    TTimeVec windows{WEEK, 2 * WEEK, 16 * DAY, 4 * WEEK};
    TTimeVec bucketLengths{TEN_MINS, HALF_HOUR};
    TGeneratorVec generators{constant, ramp, markov};
    core_t::TTime startTime{10000};

    test::CRandomNumbers rng;

    TFloatMeanAccumulatorVec values;
    TDoubleVec noise;
    TSizeVec index;
    TSizeVec repeats;

    double FP{0.0};
    double TN{0.0};

    for (std::size_t test = 0; test < 50; ++test) {
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
                const auto& generator = generators[index[0]];

                values.assign(window / bucketLength, TFloatMeanAccumulator{});
                for (core_t::TTime time = startTime; time < startTime + window;
                     time += bucketLength) {
                    std::size_t bucket((time - startTime) / bucketLength);
                    values[bucket].add(generator(time) + noise[bucket]);
                }

                maths::CTimeSeriesTestForSeasonality seasonality{startTime, bucketLength, values};

                auto result = seasonality.decompose();
                bool isSeasonal{result.components().size() > 0};

                if (isSeasonal) {
                    LOG_DEBUG(<< "result = "
                              << core::CContainerPrinter::print(result.components()));
                }
                FP += isSeasonal ? 1.0 : 0.0;
                TN += isSeasonal ? 0.0 : 1.0;
            }
        }
    }

    LOG_DEBUG(<< "True negative rate = " << TN / (FP + TN));
    BOOST_TEST_REQUIRE(TN / (FP + TN) > 0.995);
}

BOOST_AUTO_TEST_CASE(testSyntheticDiurnal) {

    // Test accuracy for a variety of synthetic time series with daily and
    // weekly seasonalities.

    TTimeVec windows{WEEK, 2 * WEEK, 16 * DAY, 4 * WEEK};
    TTimeVec bucketLengths{TEN_MINS, HALF_HOUR};
    TSizeVec permittedGenerators{2, 4, 4, 5};
    TGeneratorVec generators{smoothDaily, spikeyDaily, smoothWeekly, spikeyWeekly, weekends};
    TStrVec expected{"[86400]", "[86400]", "[604800]", "[86400, 604800]",
                     "[86400/(0, 172800), 86400/(172800, 604800),"
                     " 604800/(172800, 604800), 604800/(172800, 604800)]"};
    core_t::TTime startTime{10000};

    test::CRandomNumbers rng;

    TFloatMeanAccumulatorVec values;
    TDoubleVec noise;
    TSizeVec index;
    TSizeVec repeats;

    double TP{0.0};
    double FN{0.0};

    for (std::size_t test = 0; test < 100; ++test) {
        if (test % 10 == 0) {
            LOG_DEBUG(<< "test " << test << " / 100");
        }
        for (std::size_t i = 0; i < windows.size(); ++i) {
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
                const auto& generator = generators[index[0]];

                values.assign(window / bucketLength, TFloatMeanAccumulator{});
                for (core_t::TTime time = startTime; time < startTime + window;
                     time += bucketLength) {
                    std::size_t bucket((time - startTime) / bucketLength);
                    values[bucket].add(20.0 * generator(time) + noise[bucket]);
                }

                maths::CTimeSeriesTestForSeasonality seasonality{startTime, bucketLength, values};

                auto result = seasonality.decompose();

                if (result.print() != expected[index[0]]) {
                    LOG_DEBUG(<< "result = " << result.print() << " expected "
                              << expected[index[0]]);
                }
                TP += result.print() == expected[index[0]] ? 1.0 : 0.0;
                FN += result.print() == expected[index[0]] ? 0.0 : 1.0;
            }
        }
    }

    LOG_DEBUG(<< "Recall = " << TP / (TP + FN));
    BOOST_TEST_REQUIRE(TP / (TP + FN) > 0.98);
}

BOOST_AUTO_TEST_CASE(testRealSpikeyDaily) {

    // Test on real spikey data with daily seasonality.

    TTimeDoublePrVec timeseries;
    core_t::TTime startTime;
    core_t::TTime endTime;
    BOOST_TEST_REQUIRE(test::CTimeSeriesTestData::parse(
        "testfiles/spikey_data.csv", timeseries, startTime, endTime,
        test::CTimeSeriesTestData::CSV_UNIX_REGEX));
    BOOST_TEST_REQUIRE(timeseries.size() > 0);

    LOG_DEBUG(<< "timeseries = "
              << core::CContainerPrinter::print(timeseries.begin(), timeseries.begin() + 10)
              << " ...");

    TTimeVec lastTest{timeseries[0].first, timeseries[0].first};
    TTimeVec windows{3 * DAY, 14 * DAY};
    core_t::TTime bucketLength{HOUR};

    TFloatMeanAccumulatorVec values[2]{
        TFloatMeanAccumulatorVec(windows[0] / bucketLength, TFloatMeanAccumulator{}),
        TFloatMeanAccumulatorVec(windows[1] / bucketLength, TFloatMeanAccumulator{})};

    for (std::size_t i = 0; i < timeseries.size(); ++i) {
        core_t::TTime time;
        double value;
        std::tie(time, value) = timeseries[i];

        for (std::size_t j = 0; j < 2; ++j) {
            values[j][((time - lastTest[j]) % windows[j]) / bucketLength].add(value);

            if (time > lastTest[j] + windows[j]) {
                maths::CTimeSeriesTestForSeasonality seasonality{
                    lastTest[j], bucketLength, values[j]};
                auto result = seasonality.decompose();
                LOG_DEBUG(<< result.print());
                // TODO
                values[j].assign(windows[0] / bucketLength, TFloatMeanAccumulator{});
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
    BOOST_TEST_REQUIRE(test::CTimeSeriesTestData::parse(
        "testfiles/thirty_minute_samples.csv", timeseries, startTime, endTime,
        test::CTimeSeriesTestData::CSV_ISO8601_REGEX,
        test::CTimeSeriesTestData::CSV_ISO8601_DATE_FORMAT));
    BOOST_TEST_REQUIRE(timeseries.size() > 0);

    LOG_DEBUG(<< "timeseries = "
              << core::CContainerPrinter::print(timeseries.begin(), timeseries.begin() + 10)
              << " ...");

    core_t::TTime lastTest{timeseries[0].first};
    core_t::TTime window{14 * DAY};
    core_t::TTime bucketLength{HOUR};

    TFloatMeanAccumulatorVec values(window / bucketLength, TFloatMeanAccumulator{});

    for (std::size_t i = 0; i < timeseries.size(); ++i) {
        core_t::TTime time;
        double value;
        std::tie(time, value) = timeseries[i];

        values[((time - lastTest) % window) / bucketLength].add(value);

        if (time > lastTest + window) {
            maths::CTimeSeriesTestForSeasonality seasonality{lastTest, bucketLength, values};
            auto result = seasonality.decompose();
            LOG_DEBUG(<< result.print());
            // TODO
            values.assign(window / bucketLength, TFloatMeanAccumulator{});
            lastTest = time;
        }
    }
}

BOOST_AUTO_TEST_CASE(testRealTradingDaysPlusOutliers) {

    // Test on real data with weekdays/weekend and outliers.

    TTimeDoublePrVec timeseries;
    core_t::TTime startTime;
    core_t::TTime endTime;
    BOOST_TEST_REQUIRE(test::CTimeSeriesTestData::parse(
        "testfiles/diurnal.csv", timeseries, startTime, endTime,
        test::CTimeSeriesTestData::CSV_UNIX_REGEX));
    BOOST_TEST_REQUIRE(timeseries.size() > 0);

    LOG_DEBUG(<< "timeseries = "
              << core::CContainerPrinter::print(timeseries.begin(), timeseries.begin() + 10)
              << " ...");

    core_t::TTime lastTest{timeseries[0].first};
    core_t::TTime window{14 * DAY};
    core_t::TTime bucketLength{HOUR};

    TFloatMeanAccumulatorVec values(window / bucketLength, TFloatMeanAccumulator{});

    for (std::size_t i = 0; i < timeseries.size(); ++i) {
        core_t::TTime time;
        double value;
        std::tie(time, value) = timeseries[i];

        values[((time - lastTest) % window) / bucketLength].add(value);

        if (time > lastTest + window) {
            maths::CTimeSeriesTestForSeasonality seasonality{lastTest, bucketLength, values};
            auto result = seasonality.decompose();
            LOG_DEBUG(<< result.print());
            // TODO
            values.assign(window / bucketLength, TFloatMeanAccumulator{});
            lastTest = time;
        }
    }
}

BOOST_AUTO_TEST_CASE(testRealSwitchingNotPeriodic) {

    // Test on real non-periodic switching data.

    TTimeDoublePrVec timeseries;
    core_t::TTime startTime;
    core_t::TTime endTime;
    BOOST_TEST_REQUIRE(test::CTimeSeriesTestData::parse(
        "testfiles/no_periods.csv", timeseries, startTime, endTime,
        test::CTimeSeriesTestData::CSV_ISO8601_REGEX,
        test::CTimeSeriesTestData::CSV_ISO8601_DATE_FORMAT));
    BOOST_TEST_REQUIRE(timeseries.size() > 0);

    LOG_DEBUG(<< "timeseries = "
              << core::CContainerPrinter::print(timeseries.begin(), timeseries.begin() + 10)
              << " ...");

    core_t::TTime lastTest{timeseries[0].first};
    core_t::TTime window{14 * DAY};
    core_t::TTime bucketLength{HOUR};

    TFloatMeanAccumulatorVec values(window / bucketLength, TFloatMeanAccumulator{});

    for (std::size_t i = 0; i < timeseries.size(); ++i) {
        core_t::TTime time;
        double value;
        std::tie(time, value) = timeseries[i];

        values[((time - lastTest) % window) / bucketLength].add(value);

        if (time > lastTest + window) {
            maths::CTimeSeriesTestForSeasonality seasonality{lastTest, bucketLength, values};
            auto result = seasonality.decompose();
            LOG_DEBUG(<< result.print());
            // TODO
            values.assign(window / bucketLength, TFloatMeanAccumulator{});
            lastTest = time;
        }
    }
}

BOOST_AUTO_TEST_CASE(testSyntheticNonDiurnal) {

    // Test the accuracy for non-diurnal seasonal components with periods
    // in the range [DAY / 5, 5 * DAY].

    TGeneratorVec generators{smoothDaily, spikeyDaily};
    core_t::TTime startTime{10000};

    test::CRandomNumbers rng;

    TFloatMeanAccumulatorVec values;
    TDoubleVec noise;
    TSizeVec index;
    TSizeVec repeats;

    double TP{0.0};
    double FN{0.0};

    for (std::size_t test = 0; test < 100; ++test) {
        if (test % 10 == 0) {
            LOG_DEBUG(<< "test " << test << " / 100");
        }
        for (auto window : {WEEK, 2 * WEEK, 16 * DAY, 4 * WEEK}) {
            core_t::TTime endTime{startTime + window};

            double periodScale{[&] {
                TDoubleVec result;
                rng.generateUniformSamples(1.0, 5.0, 1, result);
                return test % 2 == 0 ? result[0] : 1.0 / result[0];
            }()};

            for (auto bucketLength : {TEN_MINS, HALF_HOUR}) {

                core_t::TTime period{maths::CIntegerTools::floor(
                    static_cast<core_t::TTime>(static_cast<double>(DAY) / periodScale),
                    bucketLength)};
                periodScale = static_cast<double>(DAY) / static_cast<double>(period);
                if (periodScale == 1.0 || window < 3 * period) {
                    continue;
                }

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
                rng.generateUniformSamples(0, 2, 1, index);
                rng.generateUniformSamples(3, 20, 1, repeats);
                auto generator = generators[index[0]];

                values.assign(window / bucketLength, TFloatMeanAccumulator{});
                for (core_t::TTime time = startTime; time < endTime + window;
                     time += bucketLength) {
                    std::size_t bucket((time - startTime) / bucketLength);
                    values[bucket].add(20.0 * scale(periodScale, time, generator) +
                                       noise[bucket]);
                }

                maths::CTimeSeriesTestForSeasonality seasonality{startTime, bucketLength, values};

                auto result = seasonality.decompose();
                LOG_DEBUG(<< "result = " << result.print());
                // TODO
                /*
                TP += result.print() == expected[index[0]] ? 1.0 : 0.0;
                FN += result.print() == expected[index[0]] ? 0.0 : 1.0;
                */
            }
        }
    }

    LOG_DEBUG(<< "Recall = " << TP / (TP + FN));
    BOOST_TEST_REQUIRE(TP / (TP + FN) > 0.99);
}

BOOST_AUTO_TEST_CASE(testSyntheticSparseDaily) {

    // Test a synthetic time series with daily seasonality and periodically
    // missing values.

    TStrVec type{"Daily", "No seasonality"};
    core_t::TTime bucketLength{HALF_HOUR};

    test::CRandomNumbers rng;

    TFloatMeanAccumulatorVec values;
    TDoubleVec noise;

    for (std::size_t i = 0; i < 2; ++i) {
        LOG_DEBUG(<< type[i]);

        values.assign(7 * 48, TFloatMeanAccumulator{});

        for (std::size_t day = 0, j = 0; day < 7; ++day, ++j) {
            for (auto value :
                 {0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 20.0, 18.0, 10.0, 4.0,  4.0, 4.0, 4.0, 5.0,
                  6.0, 8.0, 9.0,  9.0,  10.0, 10.0, 8.0, 4.0, 3.0, 1.0,
                  0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  3.0, 1.0}) {
                if (value > 0.0) {
                    rng.generateUniformSamples(-1.0, 1.0, 1, noise);
                    values[j].add(i == 0 ? value : noise[0]);
                }
            }

            if (day > 3) {
                maths::CTimeSeriesTestForSeasonality seasonality{0, bucketLength, values};
                auto result = seasonality.decompose();
                LOG_DEBUG(<< "result = " << result.print());
                // TODO
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testSyntheticSparseWeekly) {

    // Test a synthetic time series with weekly seasonality and periodically
    // missing values.

    TStrVec type{"Daily", "No seasonality"};
    core_t::TTime bucketLength{HOUR};

    test::CRandomNumbers rng;

    TFloatMeanAccumulatorVec values;
    TDoubleVec noise;

    for (std::size_t i = 0; i < 2; ++i) {
        LOG_DEBUG(<< type[i]);

        values.assign(4 * 168, TFloatMeanAccumulator{});

        for (std::size_t week = 0, j = 0; week < 4; ++week, ++j) {
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
                    values[j].add(i == 0 ? value : noise[0]);
                }
            }

            if (week >= 2) {
                maths::CTimeSeriesTestForSeasonality seasonality{0, bucketLength, values};
                auto result = seasonality.decompose();
                LOG_DEBUG(<< "result = " << result.print());
                // TODO
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testSyntheticWithOutliers) {

    // Test synthetic timeseries data with pepper and salt outliers.

    TDoubleVec modulation{0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0};
    core_t::TTime startTime{10000};

    test::CRandomNumbers rng;

    TDoubleVec noise;
    TSizeVec outliers;
    TDoubleVec spikeOrTroughSelector;

    for (auto period : {DAY, WEEK}) {
        LOG_DEBUG(<< "period = " << period);

        for (auto window : {WEEK, 2 * WEEK, 16 * DAY, 4 * WEEK}) {
            if (window < 2 * period) {
                continue;
            }
            LOG_DEBUG(<< "window = " << core::CContainerPrinter::print(window));

            for (auto bucketLength : {TEN_MINS, HALF_HOUR}) {
                core_t::TTime buckets{window / bucketLength};
                std::size_t numberOutliers{
                    static_cast<std::size_t>(0.1 * static_cast<double>(buckets))};
                rng.generateUniformSamples(0, buckets, numberOutliers, outliers);
                rng.generateUniformSamples(0, 1.0, numberOutliers, spikeOrTroughSelector);
                rng.generateNormalSamples(0.0, 9.0, buckets, noise);
                std::sort(outliers.begin(), outliers.end());

                TFloatMeanAccumulatorVec values(buckets);
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
                    startTime, bucketLength, std::move(values)};
                auto result = seasonality.decompose();
                LOG_DEBUG(<< "result = " << result.print());
                // TODO
            }
        }
    }

    LOG_DEBUG(<< "Weekdays/weekend");

    for (auto window : {2 * WEEK, 16 * DAY, 4 * WEEK}) {
        LOG_DEBUG(<< "window = " << core::CContainerPrinter::print(window));

        for (auto bucketLength : {TEN_MINS, HALF_HOUR}) {
            core_t::TTime buckets{window / bucketLength};
            std::size_t numberOutliers{
                static_cast<std::size_t>(0.1 * static_cast<double>(buckets))};
            rng.generateUniformSamples(0, buckets, numberOutliers, outliers);
            rng.generateUniformSamples(0, 1.0, numberOutliers, spikeOrTroughSelector);
            rng.generateNormalSamples(0.0, 9.0, buckets, noise);
            std::sort(outliers.begin(), outliers.end());

            TFloatMeanAccumulatorVec values(buckets);
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

            maths::CTimeSeriesTestForSeasonality seasonality{startTime, bucketLength,
                                                             std::move(values)};
            auto result = seasonality.decompose();
            LOG_DEBUG(<< "result = " << result.print());
            // TODO
        }
    }
}

BOOST_AUTO_TEST_CASE(testSyntheticMixtureOfSeasonalities) {

    // Test the accuracy with which we decompose a synthetic time series into
    // its multiple constitute components.

    TGeneratorVec generators{smoothDaily, spikeyDaily};
    core_t::TTime startTime{10000};

    test::CRandomNumbers rng;

    TDoubleVec noise;
    TSizeVec index;
    TSizeVec repeats;

    TDoubleVec TP{0.0, 0.0, 0.0};
    TDoubleVec FN{0.0, 0.0, 0.0};

    for (std::size_t test = 0; test < 100; ++test) {
        if (test % 10 == 0) {
            LOG_DEBUG(<< "test " << test << " / 100");
        }
        for (auto window : {WEEK, 2 * WEEK, 16 * DAY, 4 * WEEK}) {

            TDoubleVec periodScales;
            rng.generateUniformSamples(1.0, 5.0, 2, periodScales);
            periodScales[0] = 1.0 / periodScales[0];

            for (auto bucketLength : {TEN_MINS, HALF_HOUR}) {

                for (auto periodScale : periodScales) {
                    core_t::TTime period{maths::CIntegerTools::floor(
                        static_cast<core_t::TTime>(static_cast<double>(DAY) / periodScale),
                        bucketLength)};
                    periodScale = static_cast<double>(DAY) / static_cast<double>(period);
                }
                if (static_cast<double>(window) < 3.0 * periodScales[1]) {
                    continue;
                }

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

                TFloatMeanAccumulatorVec values(window / bucketLength);
                for (core_t::TTime time = startTime; time < startTime + window;
                     time += bucketLength) {
                    std::size_t bucket((time - startTime) / bucketLength);
                    double value{0.0};
                    for (std::size_t i = 0; i < 2; ++i) {
                        value += 10.0 * static_cast<double>(i + 1) *
                                 scale(periodScales[i], time, generators[i]);
                    }
                    values[bucket].add(value + noise[bucket]);
                }

                maths::CTimeSeriesTestForSeasonality seasonality{
                    startTime, bucketLength, std::move(values)};
                auto result = seasonality.decompose();
                LOG_DEBUG(<< "result = " << result.print());

                // TODO
                /*
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
                */
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

BOOST_AUTO_TEST_CASE(testSyntheticWithLinearScaling) {

    // Test the ability to correctly decompose a time series with diurnal
    // seasonal components in the presence of piecewise constant random
    // linear scaling events.

    core_t::TTime bucketLength{HALF_HOUR};
    std::size_t segmentSupport[][2]{{100, 200}, {600, 900}};
    double scaleSupport[][2]{{4.0, 6.0}, {0.2, 0.4}};
    TGeneratorVec generators{smoothDaily, spikeyDaily, smoothWeekly};
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

        for (auto window : {3 * WEEK, 4 * WEEK}) {
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

            maths::CTimeSeriesTestForSeasonality seasonality{startTime, bucketLength,
                                                             std::move(values)};
            auto result = seasonality.decompose();
            LOG_DEBUG(<< "result = " << result.print());

            /*
            TP += result.print() == expected[index[0]] ? 1.0 : 0.0;
            FN += result.print() == expected[index[0]] ? 0.0 : 1.0;
            */
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

        maths::CTimeSeriesTestForSeasonality seasonality{startTime, bucketLength,
                                                         std::move(values)};
        auto result = seasonality.decompose();
        LOG_DEBUG(<< "result = " << result.print());
    }
}

BOOST_AUTO_TEST_CASE(testSyntheticWithLinearTrend) {

    // Test the ability to correctly decompose a time series with seasonal
    // components and a linear trend.

    TGeneratorVec generators{smoothDaily, spikeyDaily};
    core_t::TTime startTime{10000};

    test::CRandomNumbers rng;

    TFloatMeanAccumulatorVec values;
    TDoubleVec noise;
    TSizeVec index;
    TSizeVec repeats;

    double TP{0.0};
    double FN{0.0};

    for (std::size_t test = 0; test < 100; ++test) {
        if (test % 10 == 0) {
            LOG_DEBUG(<< "test " << test << " / 100");
        }
        for (auto window : {WEEK, 2 * WEEK, 16 * DAY, 4 * WEEK}) {
            core_t::TTime endTime{startTime + window};

            double periodScale{[&] {
                TDoubleVec result;
                rng.generateUniformSamples(1.0, 5.0, 1, result);
                return test % 2 == 0 ? result[0] : 1.0 / result[0];
            }()};

            for (auto bucketLength : {TEN_MINS, HALF_HOUR}) {

                core_t::TTime period{maths::CIntegerTools::floor(
                    static_cast<core_t::TTime>(static_cast<double>(DAY) / periodScale),
                    bucketLength)};
                periodScale = static_cast<double>(DAY) / static_cast<double>(period);
                if (periodScale == 1.0 || window < 3 * period) {
                    continue;
                }

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
                rng.generateUniformSamples(0, 2, 1, index);
                rng.generateUniformSamples(3, 20, 1, repeats);
                auto generator = generators[index[0]];

                values.assign(window / bucketLength, TFloatMeanAccumulator{});
                for (core_t::TTime time = startTime; time < endTime + window;
                     time += bucketLength) {
                    std::size_t bucket((time - startTime) / bucketLength);
                    values[bucket].add(0.5 * static_cast<double>(bucket) +
                                       20.0 * scale(periodScale, time, generator) +
                                       noise[bucket]);
                }

                maths::CTimeSeriesTestForSeasonality seasonality{startTime, bucketLength, values};

                auto result = seasonality.decompose();
                LOG_DEBUG(<< "result = " << result.print());
                // TODO
                /*
                TP += result.print() == expected[index[0]] ? 1.0 : 0.0;
                FN += result.print() == expected[index[0]] ? 0.0 : 1.0;
                */
            }
        }
    }

    LOG_DEBUG(<< "Recall = " << TP / (TP + FN));
    BOOST_TEST_REQUIRE(TP / (TP + FN) > 0.99);
}

BOOST_AUTO_TEST_CASE(testSyntheticWithPiecewiseLinearTrend) {

    // Test the ability to correctly decompose a time series with diurnal
    // seasonal components and a piecewise linear trend.

    using TLinearModel = std::function<double(core_t::TTime)>;
    using TLinearModelVec = std::vector<TLinearModel>;

    core_t::TTime bucketLength{HALF_HOUR};
    std::size_t segmentSupport[][2]{{100, 200}, {600, 900}};
    double slopeSupport[][2]{{0.5, 1.0}, {-1.0, -0.5}};
    double interceptSupport[][2]{{-10.0, 5.0}, {10.0, 20.0}};
    TGeneratorVec generators{smoothDaily, spikeyDaily};
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

        for (const auto& window : {3 * WEEK, 4 * WEEK}) {
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

            maths::CTimeSeriesTestForSeasonality seasonality{startTime, bucketLength,
                                                             std::move(values)};
            auto result = seasonality.decompose();
            LOG_DEBUG(<< "result = " << result.print());

            /*
            TP += result.print() == expected[index[0]] ? 1.0 : 0.0;
            FN += result.print() == expected[index[0]] ? 0.0 : 1.0;
            */
        }
    }

    LOG_DEBUG(<< "Recall = " << TP / (TP + FN));
    BOOST_TEST_REQUIRE(TP / (TP + FN) > 0.87);
}

BOOST_AUTO_TEST_CASE(testComponentInitialValues) {
}

BOOST_AUTO_TEST_SUITE_END()
