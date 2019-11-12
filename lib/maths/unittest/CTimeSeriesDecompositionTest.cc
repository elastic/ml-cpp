/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CTimezone.h>
#include <core/Constants.h>

#include <maths/CDecayRateController.h>
#include <maths/CIntegerTools.h>
#include <maths/CLinearAlgebraFwd.h>
#include <maths/CMathsFuncs.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/CRestoreParams.h>
#include <maths/CSeasonalTime.h>
#include <maths/CTimeSeriesDecomposition.h>
#include <maths/Constants.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>
#include <test/CTimeSeriesTestData.h>

#include <boost/math/constants/constants.hpp>
#include <boost/test/unit_test.hpp>

#include <fstream>
#include <utility>
#include <vector>

BOOST_AUTO_TEST_SUITE(CTimeSeriesDecompositionTest)

using namespace ml;

namespace {

using TDoubleDoublePr = std::pair<double, double>;
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TDouble1Vec = core::CSmallVector<double, 1>;
using TTimeVec = std::vector<core_t::TTime>;
using TTimeDoublePr = std::pair<core_t::TTime, double>;
using TTimeDoublePrVec = std::vector<TTimeDoublePr>;
using TSeasonalComponentVec = maths_t::TSeasonalComponentVec;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TFloatMeanAccumulatorVec =
    std::vector<maths::CBasicStatistics::SSampleMean<maths::CFloatStorage>::TAccumulator>;

double mean(const TDoubleDoublePr& x) {
    return (x.first + x.second) / 2.0;
}

class CDebugGenerator {
public:
    static const bool ENABLED{false};

public:
    CDebugGenerator(const std::string& file = "results.m") : m_File(file) {}

    ~CDebugGenerator() {
        if (ENABLED) {
            std::ofstream file;
            file.open(m_File);
            file << "t = " << core::CContainerPrinter::print(m_ValueTimes) << ";\n";
            file << "f = " << core::CContainerPrinter::print(m_Values) << ";\n";
            file << "te = " << core::CContainerPrinter::print(m_PredictionTimes) << ";\n";
            file << "fe = " << core::CContainerPrinter::print(m_Predictions) << ";\n";
            file << "r = " << core::CContainerPrinter::print(m_Errors) << ";\n";
            file << "figure(1);\n";
            file << "clf;\n";
            file << "hold on;\n";
            file << "plot(t, f);\n";
            file << "plot(te, fe, 'r');\n";
            file << "axis([t(1) t(columns(t)) min(min(f),min(fe)) max(max(f),max(fe))]);\n";
            file << "figure(2);\n";
            file << "clf;\n";
            file << "plot(te, r, 'k');\n";
            file << "axis([t(1) t(columns(t)) min(r) max(r)]);";
        }
    }
    void addValue(core_t::TTime time, double value) {
        if (ENABLED) {
            m_ValueTimes.push_back(time);
            m_Values.push_back(value);
        }
    }
    void addPrediction(core_t::TTime time, double prediction, double error) {
        if (ENABLED) {
            m_PredictionTimes.push_back(time);
            m_Predictions.push_back(prediction);
            m_Errors.push_back(error);
        }
    }

private:
    std::string m_File;
    TTimeVec m_ValueTimes;
    TDoubleVec m_Values;
    TTimeVec m_PredictionTimes;
    TDoubleVec m_Predictions;
    TDoubleVec m_Errors;
};

const core_t::TTime FIVE_MINS = 300;
const core_t::TTime TEN_MINS = 600;
const core_t::TTime HALF_HOUR = core::constants::HOUR / 2;
const core_t::TTime HOUR = core::constants::HOUR;
const core_t::TTime DAY = core::constants::DAY;
const core_t::TTime WEEK = core::constants::WEEK;
const core_t::TTime YEAR = core::constants::YEAR;
}

class CNanInjector {
public:
    // insert a NaN into a seasonal component bucket
    void injectNan(maths::CTimeSeriesDecomposition& decomposition, size_t bucketIndex) {
        firstRegressionStatistic(seasonalComponent(decomposition), bucketIndex) =
            std::numeric_limits<double>::quiet_NaN();
    }

private:
    // helper methods to get access to the state of a seasonal component

    // return the regression statistics from the provided seasonal component
    static maths::CFloatStorage&
    firstRegressionStatistic(maths::CSeasonalComponent& component, size_t bucketIndex) {
        return maths::CBasicStatistics::moment<0>(
            component.m_Bucketing.m_Buckets[bucketIndex].s_Regression.m_S)(0);
    }

    // return the first seasonal component from the provided decomposition
    static maths::CSeasonalComponent&
    seasonalComponent(maths::CTimeSeriesDecomposition& decomposition) {
        return decomposition.m_Components.m_Seasonal->m_Components[0];
    }
};

class CTestFixture {
public:
    CTestFixture() { core::CTimezone::instance().setTimezone("GMT"); }

    ~CTestFixture() { core::CTimezone::instance().setTimezone(""); }
};

BOOST_FIXTURE_TEST_CASE(testSuperpositionOfSines, CTestFixture) {
    TTimeVec times;
    TDoubleVec trend;
    for (core_t::TTime time = 0; time < 100 * WEEK + 1; time += HALF_HOUR) {
        double weekly = 1200.0 + 1000.0 * std::sin(boost::math::double_constants::two_pi *
                                                   static_cast<double>(time) /
                                                   static_cast<double>(WEEK));
        double daily = 5.0 + 5.0 * std::sin(boost::math::double_constants::two_pi *
                                            static_cast<double>(time) /
                                            static_cast<double>(DAY));
        times.push_back(time);
        trend.push_back(weekly * daily);
    }

    test::CRandomNumbers rng;
    TDoubleVec noise;
    rng.generateNormalSamples(0.0, 400.0, times.size(), noise);

    core_t::TTime lastWeek = 0;
    maths::CTimeSeriesDecomposition decomposition(0.01, HALF_HOUR);
    CDebugGenerator debug;

    double totalSumResidual = 0.0;
    double totalMaxResidual = 0.0;
    double totalSumValue = 0.0;
    double totalMaxValue = 0.0;
    double totalPercentileError = 0.0;

    for (std::size_t i = 0u; i < times.size(); ++i) {
        core_t::TTime time = times[i];
        double value = trend[i] + noise[i];

        decomposition.addPoint(time, value);
        debug.addValue(time, value);

        if (time >= lastWeek + WEEK) {
            LOG_TRACE(<< "Processing week");

            double sumResidual = 0.0;
            double maxResidual = 0.0;
            double sumValue = 0.0;
            double maxValue = 0.0;
            double percentileError = 0.0;

            for (core_t::TTime t = lastWeek; t < lastWeek + WEEK; t += HALF_HOUR) {
                TDoubleDoublePr prediction = decomposition.value(t, 70.0);
                double residual = std::fabs(trend[t / HALF_HOUR] - mean(prediction));
                sumResidual += residual;
                maxResidual = std::max(maxResidual, residual);
                sumValue += std::fabs(trend[t / HALF_HOUR]);
                maxValue = std::max(maxValue, std::fabs(trend[t / HALF_HOUR]));
                percentileError +=
                    std::max(std::max(prediction.first - trend[t / HALF_HOUR],
                                      trend[t / HALF_HOUR] - prediction.second),
                             0.0);
                debug.addPrediction(t, mean(prediction), residual);
            }

            LOG_TRACE(<< "'sum residual' / 'sum value' = " << sumResidual / sumValue);
            LOG_TRACE(<< "'max residual' / 'max value' = " << maxResidual / maxValue);
            LOG_TRACE(<< "70% error = " << percentileError / sumValue);

            if (time >= 2 * WEEK) {
                BOOST_TEST_REQUIRE(sumResidual < 0.055 * sumValue);
                BOOST_TEST_REQUIRE(maxResidual < 0.10 * maxValue);
                BOOST_TEST_REQUIRE(percentileError < 0.03 * sumValue);
                totalSumResidual += sumResidual;
                totalMaxResidual += maxResidual;
                totalSumValue += sumValue;
                totalMaxValue += maxValue;
                totalPercentileError += percentileError;
            }

            lastWeek += WEEK;
        }
    }

    LOG_DEBUG(<< "total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
    LOG_DEBUG(<< "total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);
    LOG_DEBUG(<< "total 70% error = " << totalPercentileError / totalSumValue);

    BOOST_TEST_REQUIRE(totalSumResidual < 0.016 * totalSumValue);
    BOOST_TEST_REQUIRE(totalMaxResidual < 0.02 * totalMaxValue);
    BOOST_TEST_REQUIRE(totalPercentileError < 0.01 * totalSumValue);
}

BOOST_FIXTURE_TEST_CASE(testDistortedPeriodic, CTestFixture) {
    TTimeDoublePrVec timeseries;
    core_t::TTime startTime;
    core_t::TTime endTime;
    BOOST_TEST_REQUIRE(test::CTimeSeriesTestData::parse(
        "testfiles/distorted_periodic.csv", timeseries, startTime, endTime,
        "^([0-9]+), ([0-9\\.]+)"));
    BOOST_TEST_REQUIRE(!timeseries.empty());

    LOG_DEBUG(<< "timeseries = "
              << core::CContainerPrinter::print(timeseries.begin(), timeseries.begin() + 10)
              << " ...");

    core_t::TTime lastWeek = startTime;
    const core_t::TTime bucketLength = HOUR;
    maths::CTimeSeriesDecomposition decomposition(0.01, bucketLength);
    CDebugGenerator debug;

    double totalSumResidual = 0.0;
    double totalMaxResidual = 0.0;
    double totalSumValue = 0.0;
    double totalMaxValue = 0.0;
    double totalPercentileError = 0.0;

    for (std::size_t i = 0u; i < timeseries.size(); ++i) {
        core_t::TTime time;
        double value;
        std::tie(time, value) = timeseries[i];
        decomposition.addPoint(time, value);
        debug.addValue(time, value);

        if (time >= lastWeek + WEEK || i == timeseries.size() - 1) {
            LOG_TRACE(<< "Processing week");

            double sumResidual = 0.0;
            double maxResidual = 0.0;
            double sumValue = 0.0;
            double maxValue = 0.0;
            double percentileError = 0.0;

            for (core_t::TTime t = lastWeek;
                 t < lastWeek + WEEK &&
                 static_cast<std::size_t>(t / HOUR) < timeseries.size();
                 t += HOUR) {
                double actual = timeseries[t / HOUR].second;
                TDoubleDoublePr prediction = decomposition.value(t, 70.0);
                double residual = std::fabs(actual - mean(prediction));
                sumResidual += residual;
                maxResidual = std::max(maxResidual, residual);
                sumValue += std::fabs(actual);
                maxValue = std::max(maxValue, std::fabs(actual));
                percentileError += std::max(
                    std::max(prediction.first - actual, actual - prediction.second), 0.0);
                debug.addPrediction(t, mean(prediction), residual);
            }

            LOG_TRACE(<< "'sum residual' / 'sum value' = " << sumResidual / sumValue);
            LOG_TRACE(<< "'max residual' / 'max value' = " << maxResidual / maxValue);
            LOG_TRACE(<< "70% error = " << percentileError / sumValue);

            if (time >= 2 * WEEK) {
                BOOST_TEST_REQUIRE(sumResidual < 0.27 * sumValue);
                BOOST_TEST_REQUIRE(maxResidual < 0.56 * maxValue);
                BOOST_TEST_REQUIRE(percentileError < 0.22 * sumValue);

                totalSumResidual += sumResidual;
                totalMaxResidual += maxResidual;
                totalSumValue += sumValue;
                totalMaxValue += maxValue;
                totalPercentileError += percentileError;
            }

            lastWeek += WEEK;
        }
    }

    LOG_DEBUG(<< "total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
    LOG_DEBUG(<< "total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);
    LOG_DEBUG(<< "total 70% error = " << totalPercentileError / totalSumValue);

    BOOST_TEST_REQUIRE(totalSumResidual < 0.18 * totalSumValue);
    BOOST_TEST_REQUIRE(totalMaxResidual < 0.28 * totalMaxValue);
    BOOST_TEST_REQUIRE(totalPercentileError < 0.1 * totalSumValue);
}

BOOST_FIXTURE_TEST_CASE(testMinimizeLongComponents, CTestFixture) {
    double weights[] = {1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0};

    TTimeVec times;
    TDoubleVec trend;
    for (core_t::TTime time = 0; time < 100 * WEEK; time += HALF_HOUR) {
        double weight = weights[(time / DAY) % 7];
        double daily = 100.0 * std::sin(boost::math::double_constants::two_pi *
                                        static_cast<double>(time) /
                                        static_cast<double>(DAY));
        times.push_back(time);
        trend.push_back(weight * daily);
    }

    test::CRandomNumbers rng;
    TDoubleVec noise;
    rng.generateNormalSamples(0.0, 16.0, times.size(), noise);

    maths::CTimeSeriesDecomposition decomposition(0.01, HALF_HOUR);
    CDebugGenerator debug;

    double totalSumResidual = 0.0;
    double totalMaxResidual = 0.0;
    double totalSumValue = 0.0;
    double totalMaxValue = 0.0;
    double totalPercentileError = 0.0;
    double meanSlope = 0.0;
    double refinements = 0.0;

    core_t::TTime lastWeek = 0;
    for (std::size_t i = 0u; i < times.size(); ++i) {
        core_t::TTime time = times[i];
        double value = trend[i] + noise[i];

        decomposition.addPoint(time, value);
        debug.addValue(time, value);

        if (time >= lastWeek + WEEK) {
            LOG_TRACE(<< "Processing week");

            double sumResidual = 0.0;
            double maxResidual = 0.0;
            double sumValue = 0.0;
            double maxValue = 0.0;
            double percentileError = 0.0;

            for (core_t::TTime t = lastWeek; t < lastWeek + WEEK; t += HALF_HOUR) {
                TDoubleDoublePr prediction = decomposition.value(t, 70.0);
                double residual = std::fabs(trend[t / HALF_HOUR] - mean(prediction));
                sumResidual += residual;
                maxResidual = std::max(maxResidual, residual);
                sumValue += std::fabs(trend[t / HALF_HOUR]);
                maxValue = std::max(maxValue, std::fabs(trend[t / HALF_HOUR]));
                percentileError +=
                    std::max(std::max(prediction.first - trend[t / HALF_HOUR],
                                      trend[t / HALF_HOUR] - prediction.second),
                             0.0);
                debug.addPrediction(t, mean(prediction), residual);
            }

            LOG_TRACE(<< "'sum residual' / 'sum value' = " << sumResidual / sumValue);
            LOG_TRACE(<< "'max residual' / 'max value' = " << maxResidual / maxValue);
            LOG_TRACE(<< "70% error = " << percentileError / sumValue);

            if (time >= 2 * WEEK) {
                BOOST_TEST_REQUIRE(sumResidual < 0.15 * sumValue);
                BOOST_TEST_REQUIRE(maxResidual < 0.33 * maxValue);
                BOOST_TEST_REQUIRE(percentileError < 0.08 * sumValue);

                totalSumResidual += sumResidual;
                totalMaxResidual += maxResidual;
                totalSumValue += sumValue;
                totalMaxValue += maxValue;
                totalPercentileError += percentileError;

                for (const auto& component : decomposition.seasonalComponents()) {
                    if (component.initialized() && component.time().period() == WEEK) {
                        double slope = component.valueSpline().absSlope();
                        meanSlope += slope;
                        LOG_TRACE(<< "weekly |slope| = " << slope);
                        BOOST_TEST_REQUIRE(slope < 0.0014);
                        refinements += 1.0;
                    }
                }
            }

            lastWeek += WEEK;
        }
    }

    LOG_DEBUG(<< "total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
    LOG_DEBUG(<< "total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);
    LOG_DEBUG(<< "total 70% error = " << totalPercentileError / totalSumValue);

    BOOST_TEST_REQUIRE(totalSumResidual < 0.05 * totalSumValue);
    BOOST_TEST_REQUIRE(totalMaxResidual < 0.20 * totalMaxValue);
    BOOST_TEST_REQUIRE(totalPercentileError < 0.02 * totalSumValue);

    meanSlope /= refinements;
    LOG_DEBUG(<< "mean weekly |slope| = " << meanSlope);
    BOOST_TEST_REQUIRE(meanSlope < 0.0013);
}

BOOST_FIXTURE_TEST_CASE(testWeekend, CTestFixture) {
    double weights[] = {0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0};

    for (auto offset : {0 * DAY, 5 * DAY}) {
        TTimeVec times;
        TDoubleVec trend;
        for (core_t::TTime time = 0; time < 100 * WEEK; time += HALF_HOUR) {
            double weight = weights[(time / DAY) % 7];
            double daily = 100.0 * std::sin(boost::math::double_constants::two_pi *
                                            static_cast<double>(time) /
                                            static_cast<double>(DAY));
            times.push_back(time + offset);
            trend.push_back(weight * daily);
        }

        test::CRandomNumbers rng;
        TDoubleVec noise;
        rng.generateNormalSamples(0.0, 20.0, times.size(), noise);

        maths::CTimeSeriesDecomposition decomposition(0.01, HALF_HOUR);
        CDebugGenerator debug;

        double totalSumResidual = 0.0;
        double totalMaxResidual = 0.0;
        double totalSumValue = 0.0;
        double totalMaxValue = 0.0;
        double totalPercentileError = 0.0;

        core_t::TTime lastWeek = offset;
        for (std::size_t i = 0u; i < times.size(); ++i) {
            core_t::TTime time = times[i];
            double value = trend[i] + noise[i];

            decomposition.addPoint(time, value);
            debug.addValue(time, value);

            if (time >= lastWeek + WEEK) {
                LOG_TRACE(<< "Processing week");

                double sumResidual = 0.0;
                double maxResidual = 0.0;
                double sumValue = 0.0;
                double maxValue = 0.0;
                double percentileError = 0.0;

                for (core_t::TTime t = lastWeek; t < lastWeek + WEEK; t += HALF_HOUR) {
                    TDoubleDoublePr prediction = decomposition.value(t, 70.0);
                    double actual = trend[(t - offset) / HALF_HOUR];
                    double residual = std::fabs(actual - mean(prediction));
                    sumResidual += residual;
                    maxResidual = std::max(maxResidual, residual);
                    sumValue += std::fabs(actual);
                    maxValue = std::max(maxValue, std::fabs(actual));
                    percentileError += std::max(
                        std::max(prediction.first - actual, actual - prediction.second), 0.0);
                    debug.addPrediction(t, mean(prediction), residual);
                }

                LOG_TRACE(<< "'sum residual' / 'sum value' = " << sumResidual / sumValue);
                LOG_TRACE(<< "'max residual' / 'max value' = " << maxResidual / maxValue);
                LOG_TRACE(<< "70% error = " << percentileError / sumValue);

                if (time >= 3 * WEEK) {
                    BOOST_TEST_REQUIRE(sumResidual < 0.07 * sumValue);
                    BOOST_TEST_REQUIRE(maxResidual < 0.17 * maxValue);
                    BOOST_TEST_REQUIRE(percentileError < 0.03 * sumValue);

                    totalSumResidual += sumResidual;
                    totalMaxResidual += maxResidual;
                    totalSumValue += sumValue;
                    totalMaxValue += maxValue;
                    totalPercentileError += percentileError;
                }

                lastWeek += WEEK;
            }
        }

        LOG_DEBUG(<< "total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
        LOG_DEBUG(<< "total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);
        LOG_DEBUG(<< "total 70% error = " << totalPercentileError / totalSumValue);

        BOOST_TEST_REQUIRE(totalSumResidual < 0.022 * totalSumValue);
        BOOST_TEST_REQUIRE(totalMaxResidual < 0.055 * totalMaxValue);
        BOOST_TEST_REQUIRE(totalPercentileError < 0.01 * totalSumValue);
    }
}

BOOST_FIXTURE_TEST_CASE(testNanHandling, CTestFixture) {

    TTimeVec times;
    TDoubleVec trend;
    for (core_t::TTime time = 0; time < 10 * WEEK + 1; time += HALF_HOUR) {
        double daily = 100.0 + 100.0 * std::sin(boost::math::double_constants::two_pi *
                                                static_cast<double>(time) /
                                                static_cast<double>(DAY));
        times.push_back(time);
        trend.push_back(daily);
    }

    const double noiseMean = 20.0;
    const double noiseVariance = 16.0;
    test::CRandomNumbers rng;
    TDoubleVec noise;
    rng.generateNormalSamples(noiseMean, noiseVariance, times.size(), noise);

    maths::CTimeSeriesDecomposition decomposition(0.01, HALF_HOUR);

    // run through half of the periodic data
    std::size_t i = 0u;
    for (; i < times.size() / 2; ++i) {
        core_t::TTime time = times[i];
        double value = trend[i] + noise[i];

        decomposition.addPoint(time, value);
    }

    // inject a NaN into one of the seasonal components
    CNanInjector nanInjector;
    nanInjector.injectNan(decomposition, 0L);

    int componentsModified{0};

    // run through the 2nd half of the periodic data set
    for (++i; i < times.size(); ++i) {
        core_t::TTime time = times[i];
        auto value = decomposition.value(time);
        BOOST_TEST_REQUIRE(maths::CMathsFuncs::isFinite(value.first));
        BOOST_TEST_REQUIRE(maths::CMathsFuncs::isFinite(value.second));

        decomposition.addPoint(time, trend[i] + noise[i], maths_t::CUnitWeights::UNIT,
                               [&componentsModified](TFloatMeanAccumulatorVec) {
                                   ++componentsModified;
                               });
    }

    // The call to 'addPoint' that results in the removal of the component
    // containing the NaN value also triggers an immediate re-detection of
    // a daily seasonal component. Hence we only expect it to report that the
    // components have been modified just the once even though two modification
    // event have occurred.
    BOOST_REQUIRE_EQUAL(1, componentsModified);

    // Check that only the daily component has been initialized.
    const TSeasonalComponentVec& components = decomposition.seasonalComponents();
    BOOST_REQUIRE_EQUAL(std::size_t(1), components.size());
    BOOST_REQUIRE_EQUAL(DAY, components[0].time().period());
    BOOST_TEST_REQUIRE(components[0].initialized());
}

BOOST_FIXTURE_TEST_CASE(testSinglePeriodicity, CTestFixture) {
    TTimeVec times;
    TDoubleVec trend;
    for (core_t::TTime time = 0; time < 10 * WEEK + 1; time += HALF_HOUR) {
        double daily = 100.0 + 100.0 * std::sin(boost::math::double_constants::two_pi *
                                                static_cast<double>(time) /
                                                static_cast<double>(DAY));
        times.push_back(time);
        trend.push_back(daily);
    }

    const double noiseMean = 20.0;
    const double noiseVariance = 16.0;
    test::CRandomNumbers rng;
    TDoubleVec noise;
    rng.generateNormalSamples(noiseMean, noiseVariance, times.size(), noise);

    maths::CTimeSeriesDecomposition decomposition(0.01, HALF_HOUR);
    CDebugGenerator debug;

    double totalSumResidual = 0.0;
    double totalMaxResidual = 0.0;
    double totalSumValue = 0.0;
    double totalMaxValue = 0.0;
    double totalPercentileError = 0.0;

    core_t::TTime lastWeek = 0;
    for (std::size_t i = 0u; i < times.size(); ++i) {
        core_t::TTime time = times[i];
        double value = trend[i] + noise[i];

        decomposition.addPoint(time, value);
        debug.addValue(time, value);

        if (time >= lastWeek + WEEK) {
            LOG_TRACE(<< "Processing week");

            double sumResidual = 0.0;
            double maxResidual = 0.0;
            double sumValue = 0.0;
            double maxValue = 0.0;
            double percentileError = 0.0;

            for (core_t::TTime t = lastWeek; t < lastWeek + WEEK; t += HALF_HOUR) {
                TDoubleDoublePr prediction = decomposition.value(t, 70.0);
                double residual =
                    std::fabs(trend[t / HALF_HOUR] + noiseMean - mean(prediction));
                sumResidual += residual;
                maxResidual = std::max(maxResidual, residual);
                sumValue += std::fabs(trend[t / HALF_HOUR]);
                maxValue = std::max(maxValue, std::fabs(trend[t / HALF_HOUR]));
                percentileError += std::max(
                    std::max(prediction.first - (trend[t / HALF_HOUR] + noiseMean),
                             (trend[t / HALF_HOUR] + noiseMean) - prediction.second),
                    0.0);
                debug.addPrediction(t, mean(prediction), residual);
            }

            LOG_TRACE(<< "'sum residual' / 'sum value' = " << sumResidual / sumValue);
            LOG_TRACE(<< "'max residual' / 'max value' = " << maxResidual / maxValue);
            LOG_TRACE(<< "70% error = " << percentileError / sumValue);

            if (time >= 1 * WEEK) {
                BOOST_TEST_REQUIRE(sumResidual < 0.025 * sumValue);
                BOOST_TEST_REQUIRE(maxResidual < 0.035 * maxValue);
                BOOST_TEST_REQUIRE(percentileError < 0.01 * sumValue);

                totalSumResidual += sumResidual;
                totalMaxResidual += maxResidual;
                totalSumValue += sumValue;
                totalMaxValue += maxValue;
                totalPercentileError += percentileError;

                // Check that only the daily component has been initialized.
                const TSeasonalComponentVec& components = decomposition.seasonalComponents();
                BOOST_REQUIRE_EQUAL(std::size_t(1), components.size());
                BOOST_REQUIRE_EQUAL(DAY, components[0].time().period());
                BOOST_TEST_REQUIRE(components[0].initialized());
            }

            lastWeek += WEEK;
        }
    }

    LOG_DEBUG(<< "total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
    LOG_DEBUG(<< "total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);
    LOG_DEBUG(<< "total 70% error = " << totalPercentileError / totalSumValue);
    BOOST_TEST_REQUIRE(totalSumResidual < 0.014 * totalSumValue);
    BOOST_TEST_REQUIRE(totalMaxResidual < 0.022 * totalMaxValue);
    BOOST_TEST_REQUIRE(totalPercentileError < 0.01 * totalSumValue);

    // Check that only the daily component has been initialized.
    const TSeasonalComponentVec& components = decomposition.seasonalComponents();
    BOOST_REQUIRE_EQUAL(std::size_t(1), components.size());
    BOOST_REQUIRE_EQUAL(DAY, components[0].time().period());
    BOOST_TEST_REQUIRE(components[0].initialized());
}

BOOST_FIXTURE_TEST_CASE(testSeasonalOnset, CTestFixture) {
    const double daily[] = {0.0,  0.0,  0.0,  0.0,  5.0,  5.0,  5.0,  40.0,
                            40.0, 40.0, 30.0, 30.0, 35.0, 35.0, 40.0, 50.0,
                            60.0, 80.0, 80.0, 10.0, 5.0,  0.0,  0.0,  0.0};
    const double weekly[] = {0.1, 0.1, 1.2, 1.0, 1.0, 0.9, 1.5};

    TTimeVec times;
    TDoubleVec trend;
    for (core_t::TTime time = 0; time < 150 * WEEK + 1; time += HOUR) {
        double value = 0.0;
        if (time > 10 * WEEK) {
            value += daily[(time % DAY) / HOUR];
            value *= weekly[(time % WEEK) / DAY];
        }
        times.push_back(time);
        trend.push_back(value);
    }

    test::CRandomNumbers rng;
    TDoubleVec noise;
    rng.generateNormalSamples(0.0, 4.0, times.size(), noise);

    maths::CTimeSeriesDecomposition decomposition(0.01, HOUR);
    CDebugGenerator debug;

    double totalSumResidual = 0.0;
    double totalMaxResidual = 0.0;
    double totalSumValue = 0.0;
    double totalMaxValue = 0.0;
    double totalPercentileError = 0.0;

    core_t::TTime lastWeek = 0;
    for (std::size_t i = 0u; i < times.size(); ++i) {
        core_t::TTime time = times[i];
        double value = trend[i] + noise[i];

        decomposition.addPoint(time, value);
        debug.addValue(time, value);

        if (time >= lastWeek + WEEK) {
            LOG_TRACE(<< "Processing week");

            double sumResidual = 0.0;
            double maxResidual = 0.0;
            double sumValue = 0.0;
            double maxValue = 0.0;
            double percentileError = 0.0;
            for (core_t::TTime t = lastWeek; t < lastWeek + WEEK; t += HOUR) {
                TDoubleDoublePr prediction = decomposition.value(t, 70.0);
                double residual = std::fabs(trend[t / HOUR] - mean(prediction));
                sumResidual += residual;
                maxResidual = std::max(maxResidual, residual);
                sumValue += std::fabs(trend[t / HOUR]);
                maxValue = std::max(maxValue, std::fabs(trend[t / HOUR]));
                percentileError +=
                    std::max(std::max(prediction.first - trend[t / HOUR],
                                      trend[t / HOUR] - prediction.second),
                             0.0);
                debug.addPrediction(t, mean(prediction), residual);
            }

            LOG_TRACE(<< "'sum residual' / 'sum value' = "
                      << (sumResidual == 0.0 ? 0.0 : sumResidual / sumValue));
            LOG_TRACE(<< "'max residual' / 'max value' = "
                      << (maxResidual == 0.0 ? 0.0 : maxResidual / maxValue));
            LOG_TRACE(<< "70% error = "
                      << (percentileError == 0.0 ? 0.0 : percentileError / sumValue));

            totalSumResidual += sumResidual;
            totalMaxResidual += maxResidual;
            totalSumValue += sumValue;
            totalMaxValue += maxValue;
            totalPercentileError += percentileError;

            const TSeasonalComponentVec& components = decomposition.seasonalComponents();
            if (time > 12 * WEEK) {
                // Check that there are at two least components.
                BOOST_TEST_REQUIRE(components.size() >= 2);
                BOOST_TEST_REQUIRE(components[0].initialized());
                BOOST_TEST_REQUIRE(components[1].initialized());
            } else if (time > 11 * WEEK) {
                // Check that there is at least one component.
                BOOST_REQUIRE_EQUAL(std::size_t(1), components.size());
                BOOST_TEST_REQUIRE(components[0].initialized());
            } else {
                // Check that there are no components.
                BOOST_TEST_REQUIRE(components.empty());
            }
            lastWeek += WEEK;
        }
    }

    LOG_DEBUG(<< "total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
    LOG_DEBUG(<< "total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);
    LOG_DEBUG(<< "total 70% error = " << totalPercentileError / totalSumValue);
    BOOST_TEST_REQUIRE(totalSumResidual < 0.07 * totalSumValue);
    BOOST_TEST_REQUIRE(totalMaxResidual < 0.08 * totalMaxValue);
    BOOST_TEST_REQUIRE(totalPercentileError < 0.03 * totalSumValue);
}

BOOST_FIXTURE_TEST_CASE(testVarianceScale, CTestFixture) {
    // Test that variance scales are correctly computed.

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "Variance Spike");
    {
        core_t::TTime time = 0;
        maths::CTimeSeriesDecomposition decomposition(0.01, TEN_MINS);

        for (std::size_t i = 0u; i < 50; ++i) {
            for (core_t::TTime t = 0; t < DAY; t += TEN_MINS) {
                double value = 1.0;
                double variance = 1.0;
                if (t >= 3600 && t < 7200) {
                    value = 5.0;
                    variance = 10.0;
                }
                TDoubleVec noise;
                rng.generateNormalSamples(value, variance, 1, noise);
                decomposition.addPoint(time + t, noise[0]);
            }
            time += DAY;
        }

        double meanVariance = (1.0 * 23.0 + 10.0 * 1.0) / 24.0;
        time -= DAY;
        TMeanAccumulator error;
        TMeanAccumulator percentileError;
        TMeanAccumulator meanScale;
        for (core_t::TTime t = 0; t < DAY; t += TEN_MINS) {
            double variance = 1.0;
            if (t >= 3600 && t < 7200) {
                variance = 10.0;
            }
            double expectedScale = variance / meanVariance;
            TDoubleDoublePr interval = decomposition.scale(time + t, meanVariance, 70.0);
            LOG_TRACE(<< "time = " << t << ", expectedScale = " << expectedScale
                      << ", scale = " << core::CContainerPrinter::print(interval));
            double scale = (interval.first + interval.second) / 2.0;
            error.add(std::fabs(scale - expectedScale));
            meanScale.add(scale);
            percentileError.add(std::max(std::max(interval.first - expectedScale,
                                                  expectedScale - interval.second),
                                         0.0));
        }

        LOG_DEBUG(<< "mean error = " << maths::CBasicStatistics::mean(error));
        LOG_DEBUG(<< "mean 70% error = " << maths::CBasicStatistics::mean(percentileError));
        LOG_DEBUG(<< "mean scale = " << maths::CBasicStatistics::mean(meanScale));
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(error) < 0.3);
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(percentileError) < 0.05);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(1.0, maths::CBasicStatistics::mean(meanScale), 0.04);
    }
    LOG_DEBUG(<< "Smoothly Varying Variance");
    {
        core_t::TTime time = 0;
        maths::CTimeSeriesDecomposition decomposition(0.01, TEN_MINS);

        for (std::size_t i = 0u; i < 50; ++i) {
            for (core_t::TTime t = 0; t < DAY; t += TEN_MINS) {
                double value = 5.0 * std::sin(boost::math::double_constants::two_pi *
                                              static_cast<double>(t) /
                                              static_cast<double>(DAY));
                double variance = 1.0;
                if (t >= 3600 && t < 7200) {
                    variance = 10.0;
                }
                TDoubleVec noise;
                rng.generateNormalSamples(0.0, variance, 1, noise);
                decomposition.addPoint(time + t, value + noise[0]);
            }
            time += DAY;
        }

        double meanVariance = (1.0 * 23.0 + 10.0 * 1.0) / 24.0;
        time -= DAY;
        TMeanAccumulator error;
        TMeanAccumulator percentileError;
        TMeanAccumulator meanScale;
        for (core_t::TTime t = 0; t < DAY; t += TEN_MINS) {
            double variance = 1.0;
            if (t >= 3600 && t < 7200) {
                variance = 10.0;
            }
            double expectedScale = variance / meanVariance;
            TDoubleDoublePr interval = decomposition.scale(time + t, meanVariance, 70.0);
            LOG_TRACE(<< "time = " << t << ", expectedScale = " << expectedScale
                      << ", scale = " << core::CContainerPrinter::print(interval));
            double scale = (interval.first + interval.second) / 2.0;
            error.add(std::fabs(scale - expectedScale));
            meanScale.add(scale);
            percentileError.add(std::max(std::max(interval.first - expectedScale,
                                                  expectedScale - interval.second),
                                         0.0));
        }

        LOG_DEBUG(<< "mean error = " << maths::CBasicStatistics::mean(error));
        LOG_DEBUG(<< "mean 70% error = " << maths::CBasicStatistics::mean(percentileError));
        LOG_DEBUG(<< "mean scale = " << maths::CBasicStatistics::mean(meanScale));
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(error) < 0.23);
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(percentileError) < 0.1);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(1.0, maths::CBasicStatistics::mean(meanScale), 0.01);
    }
    LOG_DEBUG(<< "Long Term Trend");
    {
        const core_t::TTime length = 120 * DAY;

        TTimeVec times;
        TDoubleVec trend;
        for (core_t::TTime time = 0; time < length; time += HALF_HOUR) {
            times.push_back(time);
            double x = static_cast<double>(time);
            trend.push_back(150.0 +
                            100.0 * std::sin(boost::math::double_constants::two_pi *
                                             x / static_cast<double>(240 * DAY) /
                                             (1.0 - x / static_cast<double>(2 * length))) +
                            10.0 * std::sin(boost::math::double_constants::two_pi *
                                            x / static_cast<double>(DAY)));
        }

        TDoubleVec noise;
        rng.generateNormalSamples(0.0, 4.0, times.size(), noise);

        maths::CTimeSeriesDecomposition decomposition(0.024, HALF_HOUR);
        for (std::size_t i = 0u; i < times.size(); ++i) {
            decomposition.addPoint(times[i], trend[i] + 0.3 * noise[i]);
        }

        TMeanAccumulator meanScale;
        double meanVariance = decomposition.meanVariance();
        for (core_t::TTime t = 0; t < DAY; t += TEN_MINS) {
            TDoubleDoublePr interval =
                decomposition.scale(times.back() + t, meanVariance, 70.0);
            LOG_TRACE(<< "time = " << t
                      << ", scale = " << core::CContainerPrinter::print(interval));
            double scale = (interval.first + interval.second) / 2.0;
            meanScale.add(scale);
        }

        LOG_DEBUG(<< "mean scale = " << maths::CBasicStatistics::mean(meanScale));
        BOOST_REQUIRE_CLOSE_ABSOLUTE(1.0, maths::CBasicStatistics::mean(meanScale), 0.02);
    }
}

BOOST_FIXTURE_TEST_CASE(testSpikeyDataProblemCase, CTestFixture) {
    TTimeDoublePrVec timeseries;
    core_t::TTime startTime;
    core_t::TTime endTime;
    BOOST_TEST_REQUIRE(test::CTimeSeriesTestData::parse(
        "testfiles/spikey_data.csv", timeseries, startTime, endTime, "^([0-9]+),([0-9\\.]+)"));
    BOOST_TEST_REQUIRE(!timeseries.empty());

    LOG_DEBUG(<< "timeseries = "
              << core::CContainerPrinter::print(timeseries.begin(), timeseries.begin() + 10)
              << " ...");

    double totalSumResidual = 0.0;
    double totalMaxResidual = 0.0;
    double totalSumValue = 0.0;
    double totalMaxValue = 0.0;
    double totalPercentileError = 0.0;

    maths::CTimeSeriesDecomposition decomposition(0.01, FIVE_MINS);
    maths::CNormalMeanPrecConjugate model = maths::CNormalMeanPrecConjugate::nonInformativePrior(
        maths_t::E_ContinuousData, 0.01);
    CDebugGenerator debug;

    core_t::TTime lastWeek = (startTime / WEEK + 1) * WEEK;
    TTimeDoublePrVec lastWeekTimeseries;
    for (std::size_t i = 0u; i < timeseries.size(); ++i) {
        core_t::TTime time = timeseries[i].first;
        double value = timeseries[i].second;

        if (time > lastWeek + WEEK) {
            LOG_TRACE(<< "Processing week");

            double sumResidual = 0.0;
            double maxResidual = 0.0;
            double sumValue = 0.0;
            double maxValue = 0.0;
            double percentileError = 0.0;

            for (std::size_t j = 0u; j < lastWeekTimeseries.size(); ++j) {
                TDoubleDoublePr prediction =
                    decomposition.value(lastWeekTimeseries[j].first, 70.0);
                double residual = std::fabs(lastWeekTimeseries[j].second - mean(prediction));
                sumResidual += residual;
                maxResidual = std::max(maxResidual, residual);
                sumValue += std::fabs(lastWeekTimeseries[j].second);
                maxValue = std::max(maxValue, std::fabs(lastWeekTimeseries[j].second));
                percentileError += std::max(
                    std::max(prediction.first - lastWeekTimeseries[j].second,
                             lastWeekTimeseries[j].second - prediction.second),
                    0.0);
                debug.addPrediction(lastWeekTimeseries[j].first, mean(prediction), residual);
            }

            LOG_TRACE(<< "'sum residual' / 'sum value' = "
                      << (sumResidual == 0.0 ? 0.0 : sumResidual / sumValue));
            LOG_TRACE(<< "'max residual' / 'max value' = "
                      << (maxResidual == 0.0 ? 0.0 : maxResidual / maxValue));
            LOG_TRACE(<< "70% error = " << percentileError / sumValue);

            if (time >= startTime + WEEK) {
                totalSumResidual += sumResidual;
                totalMaxResidual += maxResidual;
                totalSumValue += sumValue;
                totalMaxValue += maxValue;
                totalPercentileError += percentileError;
            }

            lastWeekTimeseries.clear();
            lastWeek += WEEK;
        }
        if (time > lastWeek) {
            lastWeekTimeseries.push_back(timeseries[i]);
        }

        decomposition.addPoint(
            time, value, maths_t::CUnitWeights::UNIT,
            [&model](TFloatMeanAccumulatorVec residuals) {
                model.setToNonInformative(0.0, 0.01);
                for (const auto& residual : residuals) {
                    if (maths::CBasicStatistics::count(residual) > 0.0) {
                        model.addSamples({maths::CBasicStatistics::mean(residual)},
                                         maths_t::CUnitWeights::SINGLE_UNIT);
                    }
                }
            });
        model.addSamples({decomposition.detrend(time, value, 70.0)},
                         maths_t::CUnitWeights::SINGLE_UNIT);
        debug.addValue(time, value);
    }

    LOG_DEBUG(<< "total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
    LOG_DEBUG(<< "total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);
    LOG_DEBUG(<< "total 70% error = " << totalPercentileError / totalSumValue);

    BOOST_TEST_REQUIRE(totalSumResidual < 0.20 * totalSumValue);
    BOOST_TEST_REQUIRE(totalMaxResidual < 0.41 * totalMaxValue);
    BOOST_TEST_REQUIRE(totalPercentileError < 0.16 * totalSumValue);

    //std::ofstream file;
    //file.open("results.m");
    //TTimeVec times;
    //TDoubleVec raw;
    //TDoubleVec values;
    //TDoubleVec scales;
    //TDoubleVec probs;

    double pMinScaled = 1.0;
    double pMinUnscaled = 1.0;
    for (std::size_t i = 0u; timeseries[i].first < startTime + DAY; ++i) {
        core_t::TTime time = timeseries[i].first;
        double value = timeseries[i].second;
        double variance = model.marginalLikelihoodVariance();

        double lb, ub;
        maths_t::ETail tail;
        model.probabilityOfLessLikelySamples(
            maths_t::E_TwoSided, {decomposition.detrend(time, value, 70.0)},
            {maths_t::seasonalVarianceScaleWeight(
                std::max(decomposition.scale(time, variance, 70.0).second, 0.25))},
            lb, ub, tail);
        double pScaled = (lb + ub) / 2.0;
        pMinScaled = std::min(pMinScaled, pScaled);

        //times.push_back(time);
        //raw.push_back(value);
        //values.push_back(mean(decomposition.value(time, 70.0)));
        //scales.push_back(mean(decomposition.scale(time, variance, 70.0)));
        //probs.push_back(-std::log(pScaled));

        model.probabilityOfLessLikelySamples(
            maths_t::E_TwoSided, {decomposition.detrend(time, value, 70.0)},
            maths_t::CUnitWeights::SINGLE_UNIT, lb, ub, tail);
        double pUnscaled = (lb + ub) / 2.0;
        pMinUnscaled = std::min(pMinUnscaled, pUnscaled);
    }

    //file << "hold on;\n";
    //file << "t = " << core::CContainerPrinter::print(times) << ";\n";
    //file << "r = " << core::CContainerPrinter::print(raw) << ";\n";
    //file << "b = " << core::CContainerPrinter::print(values) << ";\n";
    //file << "s = " << core::CContainerPrinter::print(scales) << ";\n";
    //file << "p = " << core::CContainerPrinter::print(probs) << ";\n";
    //file << "subplot(3,1,1); hold on; plot(t, r, 'b'); plot(t, b, 'r');\n";
    //file << "subplot(3,1,2); plot(t, s, 'b');\n";
    //file << "subplot(3,1,3); plot(t, p, 'b');\n";

    LOG_DEBUG(<< "pMinScaled = " << pMinScaled);
    LOG_DEBUG(<< "pMinUnscaled = " << pMinUnscaled);
    BOOST_TEST_REQUIRE(pMinScaled > 1e11 * pMinUnscaled);
}

BOOST_FIXTURE_TEST_CASE(testVeryLargeValuesProblemCase, CTestFixture) {
    TTimeDoublePrVec timeseries;
    core_t::TTime startTime;
    core_t::TTime endTime;
    BOOST_TEST_REQUIRE(test::CTimeSeriesTestData::parse(
        "testfiles/diurnal.csv", timeseries, startTime, endTime, "^([0-9]+),([0-9\\.]+)"));
    BOOST_TEST_REQUIRE(!timeseries.empty());

    LOG_DEBUG(<< "timeseries = "
              << core::CContainerPrinter::print(timeseries.begin(), timeseries.begin() + 10)
              << " ...");

    double totalSumResidual = 0.0;
    double totalMaxResidual = 0.0;
    double totalSumValue = 0.0;
    double totalMaxValue = 0.0;
    double totalPercentileError = 0.0;

    maths::CTimeSeriesDecomposition decomposition(0.01, FIVE_MINS);
    CDebugGenerator debug;

    core_t::TTime lastWeek = (startTime / WEEK + 1) * WEEK;
    TTimeDoublePrVec lastWeekTimeseries;
    for (std::size_t i = 0u; i < timeseries.size(); ++i) {
        core_t::TTime time = timeseries[i].first;
        double value = timeseries[i].second;

        if (time > lastWeek + WEEK) {
            LOG_TRACE(<< "Processing week");

            double sumResidual = 0.0;
            double maxResidual = 0.0;
            double sumValue = 0.0;
            double maxValue = 0.0;
            double percentileError = 0.0;

            for (std::size_t j = 0u; j < lastWeekTimeseries.size(); ++j) {
                TDoubleDoublePr prediction =
                    decomposition.value(lastWeekTimeseries[j].first, 70.0);
                double residual = std::fabs(lastWeekTimeseries[j].second - mean(prediction));
                sumResidual += residual;
                maxResidual = std::max(maxResidual, residual);
                sumValue += std::fabs(lastWeekTimeseries[j].second);
                maxValue = std::max(maxValue, std::fabs(lastWeekTimeseries[j].second));
                percentileError += std::max(
                    std::max(prediction.first - lastWeekTimeseries[j].second,
                             lastWeekTimeseries[j].second - prediction.second),
                    0.0);
                debug.addPrediction(lastWeekTimeseries[j].first, mean(prediction), residual);
            }

            LOG_TRACE(<< "'sum residual' / 'sum value' = " << sumResidual / sumValue);
            LOG_TRACE(<< "'max residual' / 'max value' = " << maxResidual / maxValue);
            LOG_TRACE(<< "70% error = " << percentileError / sumValue);

            if (time >= startTime + 2 * WEEK) {
                totalSumResidual += sumResidual;
                totalMaxResidual += maxResidual;
                totalSumValue += sumValue;
                totalMaxValue += maxValue;
                totalPercentileError += percentileError;
            }

            lastWeekTimeseries.clear();
            lastWeek += WEEK;
        }
        if (time > lastWeek) {
            lastWeekTimeseries.push_back(timeseries[i]);
        }

        decomposition.addPoint(time, value);
        debug.addValue(time, value);
    }

    LOG_DEBUG(<< "total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
    LOG_DEBUG(<< "total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);
    LOG_DEBUG(<< "total 70% error = " << totalPercentileError / totalSumValue);

    BOOST_TEST_REQUIRE(totalSumResidual < 0.35 * totalSumValue);
    BOOST_TEST_REQUIRE(totalMaxResidual < 0.74 * totalMaxValue);
    BOOST_TEST_REQUIRE(totalPercentileError < 0.16 * totalSumValue);

    TMeanAccumulator scale;
    double variance = decomposition.meanVariance();
    core_t::TTime time = maths::CIntegerTools::floor(endTime, DAY);
    for (core_t::TTime t = time; t < time + WEEK; t += TEN_MINS) {
        scale.add(mean(decomposition.scale(t, variance, 70.0)));
    }

    LOG_DEBUG(<< "scale = " << maths::CBasicStatistics::mean(scale));
    BOOST_REQUIRE_CLOSE_ABSOLUTE(1.0, maths::CBasicStatistics::mean(scale), 0.08);
}

BOOST_FIXTURE_TEST_CASE(testMixedSmoothAndSpikeyDataProblemCase, CTestFixture) {
    TTimeDoublePrVec timeseries;
    core_t::TTime startTime;
    core_t::TTime endTime;
    BOOST_TEST_REQUIRE(test::CTimeSeriesTestData::parse(
        "testfiles/thirty_minute_samples.csv", timeseries, startTime, endTime,
        test::CTimeSeriesTestData::CSV_ISO8601_REGEX,
        test::CTimeSeriesTestData::CSV_ISO8601_DATE_FORMAT));
    BOOST_TEST_REQUIRE(!timeseries.empty());

    LOG_DEBUG(<< "timeseries = "
              << core::CContainerPrinter::print(timeseries.begin(), timeseries.begin() + 10)
              << " ...");

    double totalSumResidual = 0.0;
    double totalMaxResidual = 0.0;
    double totalSumValue = 0.0;
    double totalMaxValue = 0.0;
    double totalPercentileError = 0.0;

    maths::CTimeSeriesDecomposition decomposition(0.01, HALF_HOUR);
    CDebugGenerator debug;

    core_t::TTime lastWeek = (startTime / WEEK + 1) * WEEK;
    TTimeDoublePrVec lastWeekTimeseries;
    for (std::size_t i = 0u; i < timeseries.size(); ++i) {
        core_t::TTime time = timeseries[i].first;
        double value = timeseries[i].second;

        if (time > lastWeek + WEEK) {
            LOG_TRACE(<< "Processing week");

            double sumResidual = 0.0;
            double maxResidual = 0.0;
            double sumValue = 0.0;
            double maxValue = 0.0;
            double percentileError = 0.0;

            for (std::size_t j = 0u; j < lastWeekTimeseries.size(); ++j) {
                TDoubleDoublePr prediction =
                    decomposition.value(lastWeekTimeseries[j].first, 70.0);
                double residual = std::fabs(lastWeekTimeseries[j].second - mean(prediction));
                sumResidual += residual;
                maxResidual = std::max(maxResidual, residual);
                sumValue += std::fabs(lastWeekTimeseries[j].second);
                maxValue = std::max(maxValue, std::fabs(lastWeekTimeseries[j].second));
                percentileError += std::max(
                    std::max(prediction.first - lastWeekTimeseries[j].second,
                             lastWeekTimeseries[j].second - prediction.second),
                    0.0);
                debug.addPrediction(lastWeekTimeseries[j].first, mean(prediction), residual);
            }

            LOG_TRACE(<< "'sum residual' / 'sum value' = "
                      << (sumResidual == 0.0 ? 0.0 : sumResidual / sumValue));
            LOG_TRACE(<< "'max residual' / 'max value' = "
                      << (maxResidual == 0.0 ? 0.0 : maxResidual / maxValue));
            LOG_TRACE(<< "70% error = " << percentileError / sumValue);

            if (time >= startTime + 2 * WEEK) {
                totalSumResidual += sumResidual;
                totalMaxResidual += maxResidual;
                totalSumValue += sumValue;
                totalMaxValue += maxValue;
                totalPercentileError += percentileError;
            }

            lastWeekTimeseries.clear();
            lastWeek += WEEK;
        }
        if (time > lastWeek) {
            lastWeekTimeseries.push_back(timeseries[i]);
        }

        decomposition.addPoint(time, value);
        debug.addValue(time, value);
    }

    LOG_DEBUG(<< "total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
    LOG_DEBUG(<< "total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);
    LOG_DEBUG(<< "total 70% error = " << totalPercentileError / totalSumValue);

    BOOST_TEST_REQUIRE(totalSumResidual < 0.2 * totalSumValue);
    BOOST_TEST_REQUIRE(totalMaxResidual < 0.44 * totalMaxValue);
    BOOST_TEST_REQUIRE(totalPercentileError < 0.06 * totalSumValue);
}

BOOST_FIXTURE_TEST_CASE(testDiurnalPeriodicityWithMissingValues, CTestFixture) {
    // Test the accuracy of the modeling when there are periodically missing
    // values.

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "Daily Periodic");
    {
        maths::CTimeSeriesDecomposition decomposition(0.01, HALF_HOUR);
        CDebugGenerator debug("daily.m");

        TMeanAccumulator error;
        core_t::TTime time = 0;
        for (std::size_t t = 0u; t < 50; ++t) {
            for (auto value :
                 {0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 20.0, 18.0, 10.0, 4.0,  4.0, 4.0, 4.0, 5.0,
                  6.0, 8.0, 9.0,  9.0,  10.0, 10.0, 8.0, 4.0, 3.0, 1.0,
                  0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  3.0, 1.0}) {
                if (value > 0.0) {
                    TDoubleVec noise;
                    rng.generateNormalSamples(10.0, 2.0, 1, noise);
                    value += noise[0];
                    decomposition.addPoint(time, value);
                    debug.addValue(time, value);
                    double prediction =
                        maths::CBasicStatistics::mean(decomposition.value(time, 0.0));
                    if (decomposition.initialized()) {
                        error.add(std::fabs(value - prediction) / std::fabs(value));
                    }
                    debug.addPrediction(time, prediction, value - prediction);
                }
                time += HALF_HOUR;
            }
        }

        LOG_DEBUG(<< "mean error = " << maths::CBasicStatistics::mean(error));
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(error) < 0.09);
    }

    LOG_DEBUG(<< "Weekly Periodic");
    {
        maths::CTimeSeriesDecomposition decomposition(0.01, HOUR);
        CDebugGenerator debug("weekly.m");

        TMeanAccumulator error;
        core_t::TTime time = 0;
        for (std::size_t t = 0u; t < 10; ++t) {
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
                    TDoubleVec noise;
                    rng.generateNormalSamples(10.0, 2.0, 1, noise);
                    value += noise[0];
                    decomposition.addPoint(time, value);
                    debug.addValue(time, value);
                    double prediction =
                        maths::CBasicStatistics::mean(decomposition.value(time, 0.0));
                    if (decomposition.initialized()) {
                        error.add(std::fabs(value - prediction) / std::fabs(value));
                    }
                    debug.addPrediction(time, prediction, value - prediction);
                }
                time += HOUR;
            }
        }

        LOG_DEBUG(<< "mean error = " << maths::CBasicStatistics::mean(error));
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(error) < 0.11);
    }
}

BOOST_FIXTURE_TEST_CASE(testLongTermTrend, CTestFixture) {
    // Test a simple linear ramp and non-periodic saw tooth series.

    const core_t::TTime length = 120 * DAY;

    TTimeVec times;
    TDoubleVec trend;

    test::CRandomNumbers rng;
    TDoubleVec noise;
    rng.generateNormalSamples(0.0, 25.0, length / HALF_HOUR, noise);

    LOG_DEBUG(<< "Linear Ramp");
    {
        for (core_t::TTime time = 0; time < length; time += HALF_HOUR) {
            times.push_back(time);
            trend.push_back(5.0 + static_cast<double>(time) / static_cast<double>(DAY));
        }

        maths::CTimeSeriesDecomposition decomposition(0.024, HALF_HOUR);
        CDebugGenerator debug("ramp.m");

        double totalSumResidual = 0.0;
        double totalMaxResidual = 0.0;
        double totalSumValue = 0.0;
        double totalMaxValue = 0.0;
        core_t::TTime lastDay = times[0];

        for (std::size_t i = 0u; i < times.size(); ++i) {
            decomposition.addPoint(times[i], trend[i] + noise[i]);
            debug.addValue(times[i], trend[i] + noise[i]);

            if (times[i] > lastDay + DAY) {
                LOG_TRACE(<< "Processing day " << times[i] / DAY);

                if (decomposition.initialized()) {
                    double sumResidual = 0.0;
                    double maxResidual = 0.0;
                    double sumValue = 0.0;
                    double maxValue = 0.0;

                    for (std::size_t j = i - 48; j < i; ++j) {
                        TDoubleDoublePr prediction = decomposition.value(times[j], 70.0);
                        double residual = std::fabs(trend[j] - mean(prediction));
                        sumResidual += residual;
                        maxResidual = std::max(maxResidual, residual);
                        sumValue += std::fabs(trend[j]);
                        maxValue = std::max(maxValue, std::fabs(trend[j]));
                        debug.addPrediction(times[j], mean(prediction), residual);
                    }

                    LOG_TRACE(<< "'sum residual' / 'sum value' = "
                              << (sumResidual == 0.0 ? 0.0 : sumResidual / sumValue));
                    LOG_TRACE(<< "'max residual' / 'max value' = "
                              << (maxResidual == 0.0 ? 0.0 : maxResidual / maxValue));

                    totalSumResidual += sumResidual;
                    totalMaxResidual += maxResidual;
                    totalSumValue += sumValue;
                    totalMaxValue += maxValue;

                    BOOST_TEST_REQUIRE(sumResidual / sumValue < 0.05);
                    BOOST_TEST_REQUIRE(maxResidual / maxValue < 0.05);
                }
                lastDay += DAY;
            }
        }

        LOG_DEBUG(<< "total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
        LOG_DEBUG(<< "total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);

        BOOST_TEST_REQUIRE(totalSumResidual / totalSumValue < 0.01);
        BOOST_TEST_REQUIRE(totalMaxResidual / totalMaxValue < 0.01);
    }

    LOG_DEBUG(<< "Saw Tooth Not Periodic");
    {
        core_t::TTime drops[] = {0,        30 * DAY,  50 * DAY,  60 * DAY,
                                 85 * DAY, 100 * DAY, 115 * DAY, 120 * DAY};

        times.clear();
        trend.clear();

        {
            std::size_t i = 1;
            for (core_t::TTime time = 0; time < length;
                 time += HALF_HOUR, (time > drops[i] ? ++i : i)) {
                times.push_back(time);
                trend.push_back(25.0 * static_cast<double>(time - drops[i - 1]) /
                                static_cast<double>(drops[i] - drops[i - 1] + 1));
            }
        }

        maths::CTimeSeriesDecomposition decomposition(0.01, HALF_HOUR);
        CDebugGenerator debug("saw_tooth.m");

        double totalSumResidual = 0.0;
        double totalMaxResidual = 0.0;
        double totalSumValue = 0.0;
        double totalMaxValue = 0.0;
        core_t::TTime lastDay = times[0];

        for (std::size_t i = 0u; i < times.size(); ++i) {
            decomposition.addPoint(times[i], trend[i] + 0.3 * noise[i]);
            debug.addValue(times[i], trend[i] + 0.3 * noise[i]);

            if (times[i] > lastDay + DAY) {
                LOG_TRACE(<< "Processing day " << times[i] / DAY);

                if (decomposition.initialized()) {
                    double sumResidual = 0.0;
                    double maxResidual = 0.0;
                    double sumValue = 0.0;
                    double maxValue = 0.0;

                    for (std::size_t j = i - 48; j < i; ++j) {
                        TDoubleDoublePr prediction = decomposition.value(times[j], 70.0);
                        double residual = std::fabs(trend[j] - mean(prediction));
                        sumResidual += residual;
                        maxResidual = std::max(maxResidual, residual);
                        sumValue += std::fabs(trend[j]);
                        maxValue = std::max(maxValue, std::fabs(trend[j]));
                        debug.addPrediction(times[j], mean(prediction), residual);
                    }

                    LOG_TRACE(<< "'sum residual' / 'sum value' = "
                              << (sumResidual == 0.0 ? 0.0 : sumResidual / sumValue));
                    LOG_TRACE(<< "'max residual' / 'max value' = "
                              << (maxResidual == 0.0 ? 0.0 : maxResidual / maxValue));

                    totalSumResidual += sumResidual;
                    totalMaxResidual += maxResidual;
                    totalSumValue += sumValue;
                    totalMaxValue += maxValue;
                }
                lastDay += DAY;
            }
        }

        LOG_DEBUG(<< "total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
        LOG_DEBUG(<< "total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);

        BOOST_TEST_REQUIRE(totalSumResidual / totalSumValue < 0.38);
        BOOST_TEST_REQUIRE(totalMaxResidual / totalMaxValue < 0.41);
    }
}

BOOST_FIXTURE_TEST_CASE(testLongTermTrendAndPeriodicity, CTestFixture) {
    // Test a long term mean reverting component plus daily periodic component.

    TTimeVec times;
    TDoubleVec trend;
    const core_t::TTime length = 120 * DAY;
    for (core_t::TTime time = 0; time < length; time += HALF_HOUR) {
        times.push_back(time);
        double x = static_cast<double>(time);
        trend.push_back(150.0 +
                        100.0 * std::sin(boost::math::double_constants::two_pi *
                                         x / static_cast<double>(240 * DAY) /
                                         (1.0 - x / static_cast<double>(2 * length))) +
                        10.0 * std::sin(boost::math::double_constants::two_pi *
                                        x / static_cast<double>(DAY)));
    }

    test::CRandomNumbers rng;
    TDoubleVec noise;
    rng.generateNormalSamples(0.0, 4.0, times.size(), noise);

    maths::CTimeSeriesDecomposition decomposition(0.024, HALF_HOUR);
    CDebugGenerator debug;

    double totalSumResidual = 0.0;
    double totalMaxResidual = 0.0;
    double totalSumValue = 0.0;
    double totalMaxValue = 0.0;
    core_t::TTime lastDay = times[0];

    for (std::size_t i = 0u; i < times.size(); ++i) {
        decomposition.addPoint(times[i], trend[i] + 0.3 * noise[i]);
        debug.addValue(times[i], trend[i] + 0.3 * noise[i]);

        if (times[i] > lastDay + DAY) {
            LOG_TRACE(<< "Processing day " << times[i] / DAY);

            if (decomposition.initialized()) {
                double sumResidual = 0.0;
                double maxResidual = 0.0;
                double sumValue = 0.0;
                double maxValue = 0.0;

                for (std::size_t j = i - 48; j < i; ++j) {
                    TDoubleDoublePr prediction = decomposition.value(times[j], 70.0);
                    double residual = std::fabs(trend[j] - mean(prediction));
                    sumResidual += residual;
                    maxResidual = std::max(maxResidual, residual);
                    sumValue += std::fabs(trend[j]);
                    maxValue = std::max(maxValue, std::fabs(trend[j]));
                    debug.addPrediction(times[j], mean(prediction), residual);
                }

                LOG_TRACE(<< "'sum residual' / 'sum value' = "
                          << (sumResidual == 0.0 ? 0.0 : sumResidual / sumValue));
                LOG_TRACE(<< "'max residual' / 'max value' = "
                          << (maxResidual == 0.0 ? 0.0 : maxResidual / maxValue));

                totalSumResidual += sumResidual;
                totalMaxResidual += maxResidual;
                totalSumValue += sumValue;
                totalMaxValue += maxValue;

                BOOST_TEST_REQUIRE(sumResidual / sumValue < 0.42);
                BOOST_TEST_REQUIRE(maxResidual / maxValue < 0.46);
            }
            lastDay += DAY;
        }
    }

    LOG_DEBUG(<< "total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
    LOG_DEBUG(<< "total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);

    BOOST_TEST_REQUIRE(totalSumResidual / totalSumValue < 0.04);
    BOOST_TEST_REQUIRE(totalMaxResidual / totalMaxValue < 0.05);
}

BOOST_FIXTURE_TEST_CASE(testNonDiurnal, CTestFixture) {
    // Test the accuracy of the modeling of some non-daily or weekly
    // seasonal components.
    test::CRandomNumbers rng;

    LOG_DEBUG(<< "Hourly");
    for (auto pad : {0 * DAY, 28 * DAY}) {

        double periodic[]{10.0, 1.0, 0.5, 0.5, 1.0, 5.0,
                          2.0,  1.0, 0.5, 0.5, 1.0, 6.0};

        TDoubleVec trend{TDoubleVec(pad / FIVE_MINS, 0.0)};
        TTimeVec times;
        for (core_t::TTime time = 0; time < pad + 21 * DAY; time += FIVE_MINS) {
            times.push_back(time);
            trend.push_back(periodic[(time / FIVE_MINS) % 12]);
        }

        TDoubleVec noise;
        rng.generateNormalSamples(0.0, 1.0, trend.size(), noise);

        core_t::TTime startTesting{pad + 28 * HOUR};

        maths::CTimeSeriesDecomposition decomposition(0.01, FIVE_MINS);
        CDebugGenerator debug("hourly." + core::CStringUtils::typeToString(pad) + ".m");

        double totalSumResidual = 0.0;
        double totalMaxResidual = 0.0;
        double totalSumValue = 0.0;
        double totalMaxValue = 0.0;
        core_t::TTime lastHour = times[0] + 3 * DAY;

        for (std::size_t i = 0u; i < times.size(); ++i) {
            decomposition.addPoint(times[i], trend[i] + noise[i]);
            debug.addValue(times[i], trend[i] + noise[i]);

            if (times[i] > lastHour + HOUR) {
                LOG_TRACE(<< "Processing hour " << times[i] / HOUR);

                if (times[i] > startTesting) {
                    double sumResidual = 0.0;
                    double maxResidual = 0.0;
                    double sumValue = 0.0;
                    double maxValue = 0.0;

                    for (std::size_t j = i - 12; j < i; ++j) {
                        TDoubleDoublePr prediction = decomposition.value(times[j], 70.0);
                        double residual = std::fabs(trend[j] - mean(prediction));
                        sumResidual += residual;
                        maxResidual = std::max(maxResidual, residual);
                        sumValue += std::fabs(trend[j]);
                        maxValue = std::max(maxValue, std::fabs(trend[j]));
                        debug.addPrediction(times[j], mean(prediction), residual);
                    }

                    LOG_TRACE(<< "'sum residual' / 'sum value' = "
                              << (sumResidual == 0.0 ? 0.0 : sumResidual / sumValue));
                    LOG_TRACE(<< "'max residual' / 'max value' = "
                              << (maxResidual == 0.0 ? 0.0 : maxResidual / maxValue));

                    totalSumResidual += sumResidual;
                    totalMaxResidual += maxResidual;
                    totalSumValue += sumValue;
                    totalMaxValue += maxValue;

                    BOOST_TEST_REQUIRE(sumResidual / sumValue < 0.58);
                    BOOST_TEST_REQUIRE(maxResidual / maxValue < 0.58);
                }
                lastHour += HOUR;
            }
        }

        LOG_DEBUG(<< "total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
        LOG_DEBUG(<< "total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);

        BOOST_TEST_REQUIRE(totalSumResidual / totalSumValue < 0.14);
        BOOST_TEST_REQUIRE(totalMaxResidual / totalMaxValue < 0.12);
    }

    LOG_DEBUG(<< "Two daily");
    {
        const core_t::TTime length = 20 * DAY;

        double periodic[] = {10.0, 8.0, 5.5, 2.5, 2.0, 5.0,
                             2.0,  1.0, 1.5, 3.5, 4.0, 7.0};

        TTimeVec times;
        TDoubleVec trend;
        for (core_t::TTime time = 0; time < length; time += TEN_MINS) {
            times.push_back(time);
            trend.push_back(periodic[(time / 4 / HOUR) % 12]);
        }

        TDoubleVec noise;
        rng.generateNormalSamples(0.0, 2.0, times.size(), noise);

        core_t::TTime startTesting{14 * DAY};
        maths::CTimeSeriesDecomposition decomposition(0.01, TEN_MINS);
        CDebugGenerator debug("two_day.m");

        double totalSumResidual = 0.0;
        double totalMaxResidual = 0.0;
        double totalSumValue = 0.0;
        double totalMaxValue = 0.0;
        core_t::TTime lastTwoDay = times[0] + 3 * DAY;

        for (std::size_t i = 0u; i < times.size(); ++i) {
            decomposition.addPoint(times[i], trend[i] + noise[i]);
            debug.addValue(times[i], trend[i] + noise[i]);

            if (times[i] > lastTwoDay + 2 * DAY) {
                LOG_TRACE(<< "Processing two days " << times[i] / 2 * DAY);

                if (times[i] > startTesting) {
                    double sumResidual = 0.0;
                    double maxResidual = 0.0;
                    double sumValue = 0.0;
                    double maxValue = 0.0;

                    for (std::size_t j = i - 288; j < i; ++j) {
                        TDoubleDoublePr prediction = decomposition.value(times[j], 70.0);
                        double residual = std::fabs(trend[j] - mean(prediction));
                        sumResidual += residual;
                        maxResidual = std::max(maxResidual, residual);
                        sumValue += std::fabs(trend[j]);
                        maxValue = std::max(maxValue, std::fabs(trend[j]));
                        debug.addPrediction(times[j], mean(prediction), residual);
                    }

                    LOG_TRACE(<< "'sum residual' / 'sum value' = "
                              << (sumResidual == 0.0 ? 0.0 : sumResidual / sumValue));
                    LOG_TRACE(<< "'max residual' / 'max value' = "
                              << (maxResidual == 0.0 ? 0.0 : maxResidual / maxValue));

                    totalSumResidual += sumResidual;
                    totalMaxResidual += maxResidual;
                    totalSumValue += sumValue;
                    totalMaxValue += maxValue;

                    BOOST_TEST_REQUIRE(sumResidual / sumValue < 0.17);
                    BOOST_TEST_REQUIRE(maxResidual / maxValue < 0.23);
                }
                lastTwoDay += 2 * DAY;
            }
        }

        LOG_DEBUG(<< "total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
        LOG_DEBUG(<< "total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);

        BOOST_TEST_REQUIRE(totalSumResidual / totalSumValue < 0.11);
        BOOST_TEST_REQUIRE(totalMaxResidual / totalMaxValue < 0.20);
    }
}

BOOST_FIXTURE_TEST_CASE(testYearly, CTestFixture) {
    // Test a yearly seasonal component.

    test::CRandomNumbers rng;

    maths::CTimeSeriesDecomposition decomposition(0.012, 4 * HOUR);
    maths::CDecayRateController controller(maths::CDecayRateController::E_PredictionBias |
                                               maths::CDecayRateController::E_PredictionErrorIncrease,
                                           1);
    CDebugGenerator debug;

    TDoubleVec noise;
    core_t::TTime time = 2 * HOUR;
    for (/**/; time < 4 * YEAR; time += 4 * HOUR) {
        double trend =
            15.0 * (2.0 + std::sin(boost::math::double_constants::two_pi *
                                   static_cast<double>(time) / static_cast<double>(YEAR))) +
            7.5 * std::sin(boost::math::double_constants::two_pi *
                           static_cast<double>(time) / static_cast<double>(DAY));
        rng.generateNormalSamples(0.0, 1.0, 1, noise);
        decomposition.addPoint(time, trend + noise[0]);
        if (decomposition.initialized()) {
            TDouble1Vec prediction{decomposition.meanValue(time)};
            TDouble1Vec predictionError{decomposition.detrend(time, trend, 0.0)};
            double multiplier{controller.multiplier(prediction, {predictionError},
                                                    4 * HOUR, 1.0, 0.0005)};
            decomposition.decayRate(multiplier * decomposition.decayRate());
        }
    }

    // Predict over one year and check we get reasonable accuracy.
    TMeanAccumulator meanError;
    for (/**/; time < 5 * YEAR; time += 4 * HOUR) {
        double trend =
            15.0 * (2.0 + std::sin(boost::math::double_constants::two_pi *
                                   static_cast<double>(time) / static_cast<double>(YEAR))) +
            7.5 * std::sin(boost::math::double_constants::two_pi *
                           static_cast<double>(time) / static_cast<double>(DAY));
        double prediction = maths::CBasicStatistics::mean(decomposition.value(time, 0.0));
        double error = std::fabs((prediction - trend) / trend);
        meanError.add(error);
        debug.addValue(time, trend);
        debug.addPrediction(time, prediction, trend - prediction);
        LOG_TRACE(<< "error = " << error);
        BOOST_TEST_REQUIRE(error < 0.19);
    }

    LOG_DEBUG(<< "mean error = " << maths::CBasicStatistics::mean(meanError));

    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanError) < 0.025);
}

BOOST_FIXTURE_TEST_CASE(testWithOutliers, CTestFixture) {
    // Test smooth periodic signal polluted with outliers.

    using TSizeVec = std::vector<std::size_t>;

    test::CRandomNumbers rng;

    TDoubleVec noise;
    TSizeVec outliers;
    TDoubleVec spikeOrTroughSelector;

    core_t::TTime buckets{WEEK / TEN_MINS};
    std::size_t numberOutliers{static_cast<std::size_t>(0.1 * buckets)};
    rng.generateUniformSamples(0, buckets, numberOutliers, outliers);
    rng.generateUniformSamples(0, 1.0, numberOutliers, spikeOrTroughSelector);
    rng.generateNormalSamples(0.0, 9.0, buckets, noise);
    std::sort(outliers.begin(), outliers.end());

    auto trend = [](core_t::TTime time) {
        return 25.0 + 20.0 * std::sin(boost::math::double_constants::two_pi *
                                      static_cast<double>(time) /
                                      static_cast<double>(DAY));
    };

    maths::CTimeSeriesDecomposition decomposition(0.01, TEN_MINS);
    CDebugGenerator debug;

    for (core_t::TTime time = 0; time < WEEK; time += TEN_MINS) {
        std::size_t bucket(time / TEN_MINS);
        auto outlier = std::lower_bound(outliers.begin(), outliers.end(), bucket);
        double value =
            outlier != outliers.end() && *outlier == bucket
                ? (spikeOrTroughSelector[outlier - outliers.begin()] > 0.5 ? 0.0 : 50.0)
                : trend(time);

        bool newComponents{false};
        decomposition.addPoint(
            time, value, maths_t::CUnitWeights::UNIT,
            [&newComponents](TFloatMeanAccumulatorVec) { newComponents = true; });

        if (newComponents) {
            TMeanAccumulator error;
            for (core_t::TTime endTime = time + DAY; time < endTime; time += TEN_MINS) {
                double prediction =
                    maths::CBasicStatistics::mean(decomposition.value(time, 0.0));
                error.add(std::fabs(prediction - trend(time)) / trend(time));
                debug.addValue(time, value);
                debug.addPrediction(time, prediction, trend(time) - prediction);
            }

            LOG_DEBUG(<< "error = " << maths::CBasicStatistics::mean(error));
            BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(error) < 0.05);
            break;
        }
        debug.addValue(time, value);
    }
}

BOOST_FIXTURE_TEST_CASE(testCalendar, CTestFixture) {
    // Test that we significantly reduce the error on the last Friday of each
    // month after estimating the appropriate component.

    TTimeVec months{2505600,  // Fri 30th Jan
                    4924800,  // Fri 27th Feb
                    7344000,  // Fri 27th Mar
                    9763200,  // Fri 24th Apr
                    12787200, // Fri 29th May
                    15206400, // Fri 26th Jun
                    18230400, // Fri 31st Jul
                    18316800};
    core_t::TTime end = months.back();
    TDoubleVec errors{5.0, 15.0, 35.0, 32.0, 25.0, 36.0, 22.0, 12.0, 3.0};

    auto trend = [&months, &errors](core_t::TTime t) {
        double result = 20.0 + 10.0 * std::sin(boost::math::double_constants::two_pi *
                                               static_cast<double>(t) /
                                               static_cast<double>(DAY));
        auto i = std::lower_bound(months.begin(), months.end(), t - DAY);
        if (t >= *i + 7200 &&
            t < *i + 7200 + static_cast<core_t::TTime>(errors.size()) * HALF_HOUR) {
            result += errors[(t - (*i + 7200)) / HALF_HOUR];
        }
        return result;
    };

    test::CRandomNumbers rng;

    maths::CTimeSeriesDecomposition decomposition(0.01, HALF_HOUR);
    CDebugGenerator debug;

    TDoubleVec noise;
    std::size_t count{0};
    for (core_t::TTime time = 0; time < end; time += HALF_HOUR) {
        rng.generateNormalSamples(0.0, 4.0, 1, noise);

        decomposition.addPoint(time, trend(time) + noise[0]);
        debug.addValue(time, trend(time) + noise[0]);

        if (time - DAY == *std::lower_bound(months.begin(), months.end(), time - DAY)) {
            LOG_TRACE(<< "*** time = " << time << " ***");

            std::size_t largeErrorCount = 0u;

            for (core_t::TTime time_ = time - DAY; time_ < time; time_ += TEN_MINS) {
                double prediction =
                    maths::CBasicStatistics::mean(decomposition.value(time_));
                double variance = 4.0 * maths::CBasicStatistics::mean(
                                            decomposition.scale(time_, 4.0, 0.0));
                double actual = trend(time_);
                if (std::fabs(prediction - actual) / std::sqrt(variance) > 3.0) {
                    LOG_TRACE(<< "  prediction = " << prediction);
                    LOG_TRACE(<< "  variance   = " << variance);
                    LOG_TRACE(<< "  trend      = " << trend(time_));
                    ++largeErrorCount;
                }
                debug.addPrediction(time_, prediction, actual - prediction);
            }

            LOG_TRACE(<< "large error count = " << largeErrorCount);
            if (++count <= 4) {
                BOOST_TEST_REQUIRE(largeErrorCount > 15);
            }
            if (count >= 5) {
                BOOST_TEST_REQUIRE(largeErrorCount <= 1);
            }
        }
    }
}

BOOST_FIXTURE_TEST_CASE(testConditionOfTrend, CTestFixture) {
    auto trend = [](core_t::TTime time) {
        return std::pow(static_cast<double>(time) / static_cast<double>(WEEK), 2.0);
    };

    const core_t::TTime bucketLength = 6 * HOUR;

    test::CRandomNumbers rng;

    maths::CTimeSeriesDecomposition decomposition(0.0005, bucketLength);
    TDoubleVec noise;
    for (core_t::TTime time = 0; time < 9 * YEAR; time += 6 * HOUR) {
        rng.generateNormalSamples(0.0, 4.0, 1, noise);
        decomposition.addPoint(time, trend(time) + noise[0]);
        if (time > 10 * WEEK) {
            BOOST_TEST_REQUIRE(std::fabs(decomposition.detrend(time, trend(time), 0.0)) < 3.0);
        }
    }
}

BOOST_FIXTURE_TEST_CASE(testComponentLifecycle, CTestFixture) {
    // Test we adapt to changing seasonality adding and removing components
    // as necessary.

    test::CRandomNumbers rng;

    auto trend = [](core_t::TTime time) {
        return 20.0 + 10.0 * std::sin(boost::math::double_constants::two_pi * time / DAY) +
               3.0 * (time > 4 * WEEK
                          ? std::sin(boost::math::double_constants::two_pi * time / HOUR)
                          : 0.0) -
               3.0 * (time > 9 * WEEK
                          ? std::sin(boost::math::double_constants::two_pi * time / HOUR)
                          : 0.0) +
               8.0 * (time > 16 * WEEK
                          ? std::sin(boost::math::double_constants::two_pi * time / 4 / DAY)
                          : 0.0) -
               8.0 * (time > 21 * WEEK
                          ? std::sin(boost::math::double_constants::two_pi * time / 4 / DAY)
                          : 0.0);
    };

    maths::CTimeSeriesDecomposition decomposition(0.012, FIVE_MINS);
    maths::CDecayRateController controller(maths::CDecayRateController::E_PredictionBias |
                                               maths::CDecayRateController::E_PredictionErrorIncrease,
                                           1);
    CDebugGenerator debug;

    TMeanAccumulator errors[4];
    TDoubleVec noise;
    for (core_t::TTime time = 0; time < 35 * WEEK; time += FIVE_MINS) {
        rng.generateNormalSamples(0.0, 1.0, 1, noise);
        decomposition.addPoint(time, trend(time) + noise[0]);
        debug.addValue(time, trend(time) + noise[0]);

        if (decomposition.initialized()) {
            TDouble1Vec prediction{decomposition.meanValue(time)};
            TDouble1Vec predictionError{
                decomposition.detrend(time, trend(time) + noise[0], 0.0)};
            double multiplier{controller.multiplier(prediction, {predictionError},
                                                    FIVE_MINS, 1.0, 0.0001)};
            decomposition.decayRate(multiplier * decomposition.decayRate());
        }

        double prediction = mean(decomposition.value(time, 0.0));
        if (time > 24 * WEEK) {
            errors[3].add(std::fabs(prediction - trend(time)) / trend(time));
        } else if (time > 18 * WEEK && time < 21 * WEEK) {
            errors[2].add(std::fabs(prediction - trend(time)) / trend(time));
        } else if (time > 11 * WEEK && time < 14 * WEEK) {
            errors[1].add(std::fabs(prediction - trend(time)) / trend(time));
        } else if (time > 6 * WEEK && time < 9 * WEEK) {
            errors[0].add(std::fabs(prediction - trend(time)) / trend(time));
        }

        debug.addPrediction(time, prediction, trend(time) + noise[0] - prediction);
    }

    double bounds[]{0.01, 0.018, 0.025, 0.06};
    for (std::size_t i = 0; i < 4; ++i) {
        double error{maths::CBasicStatistics::mean(errors[i])};
        LOG_DEBUG(<< "error = " << error);
        BOOST_TEST_REQUIRE(error < bounds[i]);
    }
}

BOOST_FIXTURE_TEST_CASE(testSwap, CTestFixture) {
    const double decayRate = 0.01;
    const core_t::TTime bucketLength = HALF_HOUR;

    TTimeVec times;
    TDoubleVec trend1;
    TDoubleVec trend2;
    for (core_t::TTime time = 0; time <= 10 * WEEK; time += HALF_HOUR) {
        double daily = 15.0 + 10.0 * std::sin(boost::math::double_constants::two_pi *
                                              static_cast<double>(time) /
                                              static_cast<double>(DAY));
        times.push_back(time);
        trend1.push_back(daily);
        trend2.push_back(2.0 * daily);
    }

    test::CRandomNumbers rng;
    TDoubleVec noise;
    rng.generateNormalSamples(20.0, 16.0, 2 * times.size(), noise);

    maths::CTimeSeriesDecomposition decomposition1(decayRate, bucketLength);
    maths::CTimeSeriesDecomposition decomposition2(2.0 * decayRate, 2 * bucketLength);

    for (std::size_t i = 0u; i < times.size(); i += 2) {
        decomposition1.addPoint(times[i], trend1[i] + noise[i]);
        decomposition2.addPoint(times[i], trend2[i] + noise[i + 1]);
    }

    uint64_t checksum1 = decomposition1.checksum();
    uint64_t checksum2 = decomposition2.checksum();

    LOG_DEBUG(<< "checksum1 = " << checksum1 << ", checksum2 = " << checksum2);

    decomposition1.swap(decomposition2);

    BOOST_REQUIRE_EQUAL(checksum1, decomposition2.checksum());
    BOOST_REQUIRE_EQUAL(checksum2, decomposition1.checksum());
}

BOOST_FIXTURE_TEST_CASE(testPersist, CTestFixture) {
    // Check that serialization is idempotent.
    const double decayRate = 0.01;
    const core_t::TTime bucketLength = HALF_HOUR;

    TTimeVec times;
    TDoubleVec trend;
    for (core_t::TTime time = 0; time <= 10 * WEEK; time += HALF_HOUR) {
        double daily = 15.0 + 10.0 * std::sin(boost::math::double_constants::two_pi *
                                              static_cast<double>(time) /
                                              static_cast<double>(DAY));
        times.push_back(time);
        trend.push_back(daily);
    }

    test::CRandomNumbers rng;
    TDoubleVec noise;
    rng.generateNormalSamples(20.0, 16.0, times.size(), noise);

    maths::CTimeSeriesDecomposition origDecomposition(decayRate, bucketLength);

    for (std::size_t i = 0u; i < times.size(); ++i) {
        origDecomposition.addPoint(times[i], trend[i] + noise[i]);
    }

    std::string origXml;
    {
        ml::core::CRapidXmlStatePersistInserter inserter("root");
        origDecomposition.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_TRACE(<< "Decomposition XML representation:\n" << origXml);

    // Restore the XML into a new decomposition
    core::CRapidXmlParser parser;
    BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);
    maths::STimeSeriesDecompositionRestoreParams params{
        decayRate + 0.1, bucketLength,
        maths::SDistributionRestoreParams{maths_t::E_ContinuousData, decayRate + 0.1}};

    maths::CTimeSeriesDecomposition restoredDecomposition(params, traverser);

    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredDecomposition.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    BOOST_REQUIRE_EQUAL(origXml, newXml);
}

BOOST_FIXTURE_TEST_CASE(testUpgrade, CTestFixture) {
    // Check we can validly upgrade existing state.

    using TStrVec = std::vector<std::string>;
    using TDouble3Vec = core::CSmallVector<double, 3>;

    auto load = [](const std::string& name, std::string& result) {
        std::ifstream file;
        file.open(name);
        std::stringbuf buf;
        file >> &buf;
        result = buf.str();
    };
    auto stringToPair = [](const std::string& str) {
        double first;
        double second;
        std::size_t n{str.find(",")};
        BOOST_TEST_REQUIRE(n != std::string::npos);
        core::CStringUtils::stringToType(str.substr(0, n), first);
        core::CStringUtils::stringToType(str.substr(n + 1), second);
        return TDoubleDoublePr{first, second};
    };

    maths::STimeSeriesDecompositionRestoreParams params{
        0.1, HALF_HOUR, maths::SDistributionRestoreParams{maths_t::E_ContinuousData, 0.1}};
    std::string empty;

    LOG_DEBUG(<< "**** From 6.2 ****");
    LOG_DEBUG(<< "*** Seasonal and Calendar Components ***");
    {
        std::string xml;
        load("testfiles/CTimeSeriesDecomposition.6.2.seasonal.state.xml", xml);
        LOG_DEBUG(<< "Saved state size = " << xml.size());

        std::string values;
        load("testfiles/CTimeSeriesDecomposition.6.2.seasonal.expected_values.txt", values);
        LOG_DEBUG(<< "Expected values size = " << values.size());
        TStrVec expectedValues;
        core::CStringUtils::tokenise(";", values, expectedValues, empty);

        std::string scales;
        load("testfiles/CTimeSeriesDecomposition.6.2.seasonal.expected_scales.txt", scales);
        LOG_DEBUG(<< "Expected scales size = " << scales.size());
        TStrVec expectedScales;
        core::CStringUtils::tokenise(";", scales, expectedScales, empty);

        BOOST_REQUIRE_EQUAL(expectedValues.size(), expectedScales.size());

        core::CRapidXmlParser parser;
        BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(xml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);

        maths::CTimeSeriesDecomposition decomposition(params, traverser);

        // Check that the decay rates match and the values and variances
        // predictions match the values obtained from 6.2.

        BOOST_REQUIRE_EQUAL(0.01, decomposition.decayRate());

        double meanValue{decomposition.meanValue(60480000)};
        double meanVariance{decomposition.meanVariance()};
        LOG_DEBUG(<< "restored mean value    = " << meanValue);
        LOG_DEBUG(<< "restored mean variance = " << meanVariance);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(5994.36, meanValue, 0.005);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(286374.0, meanVariance, 0.5);

        for (core_t::TTime time = 60480000, i = 0;
             i < static_cast<core_t::TTime>(expectedValues.size());
             time += HALF_HOUR, ++i) {
            TDoubleDoublePr expectedValue{stringToPair(expectedValues[i])};
            TDoubleDoublePr expectedScale{stringToPair(expectedScales[i])};
            TDoubleDoublePr value{decomposition.value(time, 10.0)};
            TDoubleDoublePr scale{decomposition.scale(time, 286374.0, 10.0)};
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedValue.first, value.first,
                                         0.005 * std::fabs(expectedValue.first));
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedValue.second, value.second,
                                         0.005 * std::fabs(expectedValue.second));
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedScale.first, scale.first,
                                         0.005 * expectedScale.first);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedScale.second, scale.second,
                                         0.005 * std::max(expectedScale.second, 0.4));
        }

        // Check some basic operations on the upgraded model.
        decomposition.forecast(60480000, 60480000 + WEEK, HALF_HOUR, 90.0, 1.0,
                               [](core_t::TTime, const TDouble3Vec&) {});
        for (core_t::TTime time = 60480000; time < 60480000 + WEEK; time += HALF_HOUR) {
            decomposition.addPoint(time, 10.0);
        }
    }

    LOG_DEBUG(<< "*** Trend and Seasonal Components ***");
    {
        std::string xml;
        load("testfiles/CTimeSeriesDecomposition.6.2.trend_and_seasonal.state.xml", xml);
        LOG_DEBUG(<< "Saved state size = " << xml.size());

        std::string values;
        load("testfiles/CTimeSeriesDecomposition.6.2.trend_and_seasonal.expected_values.txt",
             values);
        LOG_DEBUG(<< "Expected values size = " << values.size());
        TStrVec expectedValues;
        core::CStringUtils::tokenise(";", values, expectedValues, empty);

        std::string scales;
        load("testfiles/CTimeSeriesDecomposition.6.2.trend_and_seasonal.expected_scales.txt",
             scales);
        LOG_DEBUG(<< "Expected scales size = " << scales.size());
        TStrVec expectedScales;
        core::CStringUtils::tokenise(";", scales, expectedScales, empty);

        BOOST_REQUIRE_EQUAL(expectedValues.size(), expectedScales.size());

        core::CRapidXmlParser parser;
        BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(xml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);

        maths::CTimeSeriesDecomposition decomposition(params, traverser);

        // Check that the decay rates match and the values and variances
        // predictions are close to the values obtained from 6.2. We can't
        // update the state exactly in this case so the tolerances in this
        // test are significantly larger.

        BOOST_REQUIRE_EQUAL(0.024, decomposition.decayRate());

        double meanValue{decomposition.meanValue(10366200)};
        double meanVariance{decomposition.meanVariance()};
        LOG_DEBUG(<< "restored mean value    = " << meanValue);
        LOG_DEBUG(<< "restored mean variance = " << meanVariance);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(133.207, meanValue, 4.0);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(96.1654, meanVariance, 4.0);

        TMeanAccumulator meanValueError;
        TMeanAccumulator meanScaleError;
        for (core_t::TTime time = 10366200, i = 0;
             i < static_cast<core_t::TTime>(expectedValues.size());
             time += HALF_HOUR, ++i) {
            TDoubleDoublePr expectedValue{stringToPair(expectedValues[i])};
            TDoubleDoublePr expectedScale{stringToPair(expectedScales[i])};
            TDoubleDoublePr value{decomposition.value(time, 10.0)};
            TDoubleDoublePr scale{decomposition.scale(time, 96.1654, 10.0)};
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedValue.first, value.first,
                                         0.1 * std::fabs(expectedValue.first));
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedValue.second, value.second,
                                         0.1 * std::fabs(expectedValue.second));
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedScale.first, scale.first,
                                         0.3 * expectedScale.first);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedScale.second, scale.second,
                                         0.3 * expectedScale.second);
            meanValueError.add(std::fabs(expectedValue.first - value.first) /
                               std::fabs(expectedValue.first));
            meanValueError.add(std::fabs(expectedValue.second - value.second) /
                               std::fabs(expectedValue.second));
            meanScaleError.add(std::fabs(expectedScale.first - scale.first) /
                               expectedScale.first);
            meanScaleError.add(std::fabs(expectedScale.second - scale.second) /
                               expectedScale.second);
        }

        LOG_DEBUG(<< "Mean value error = " << maths::CBasicStatistics::mean(meanValueError));
        LOG_DEBUG(<< "Mean scale error = " << maths::CBasicStatistics::mean(meanScaleError));
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanValueError) < 0.06);
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanScaleError) < 0.07);

        // Check some basic operations on the upgraded model.
        decomposition.forecast(10366200, 10366200 + WEEK, HALF_HOUR, 90.0, 1.0,
                               [](core_t::TTime, const TDouble3Vec&) {});
        for (core_t::TTime time = 60480000; time < 60480000 + WEEK; time += HALF_HOUR) {
            decomposition.addPoint(time, 10.0);
        }
    }

    LOG_DEBUG(<< "**** From 5.6 ****");
    LOG_DEBUG(<< "*** Seasonal and Calendar Components ***");
    {
        std::string xml;
        load("testfiles/CTimeSeriesDecomposition.5.6.seasonal.state.xml", xml);
        LOG_DEBUG(<< "Saved state size = " << xml.size());

        core::CRapidXmlParser parser;
        BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(xml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);

        maths::CTimeSeriesDecomposition decomposition(params, traverser);

        // Check that the decay rates match and the values and variances
        // predictions match the values obtained from 6.2.

        BOOST_REQUIRE_EQUAL(0.01, decomposition.decayRate());

        double meanValue{decomposition.meanValue(18316800)};
        double meanVariance{decomposition.meanVariance()};
        LOG_DEBUG(<< "restored mean value    = " << meanValue);
        LOG_DEBUG(<< "restored mean variance = " << meanVariance);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(9.91269, meanValue, 0.005);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(3.99723, meanVariance, 0.5);

        // Check some basic operations on the upgraded model.
        decomposition.forecast(60480000, 60480000 + WEEK, HALF_HOUR, 90.0, 1.0,
                               [](core_t::TTime, const TDouble3Vec&) {});
        for (core_t::TTime time = 60480000; time < 60480000 + WEEK; time += HALF_HOUR) {
            decomposition.addPoint(time, 10.0);
        }
    }

    LOG_DEBUG(<< "*** Trend and Seasonal Components ***");
    {
        std::string xml;
        load("testfiles/CTimeSeriesDecomposition.5.6.trend_and_seasonal.state.xml", xml);
        LOG_DEBUG(<< "Saved state size = " << xml.size());

        core::CRapidXmlParser parser;
        BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(xml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);

        maths::CTimeSeriesDecomposition decomposition(params, traverser);

        // Check that the decay rates match and the values and variances
        // predictions are close to the values obtained from 6.2. We can't
        // update the state exactly in this case so the tolerances in this
        // test are significantly larger.

        BOOST_REQUIRE_EQUAL(0.024, decomposition.decayRate());

        double meanValue{decomposition.meanValue(10366200)};
        double meanVariance{decomposition.meanVariance()};
        LOG_DEBUG(<< "restored mean value    = " << meanValue);
        LOG_DEBUG(<< "restored mean variance = " << meanVariance);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(96.5607, meanValue, 4.0);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(631.094, meanVariance, 7.0);

        // Check some basic operations on the upgraded model.
        decomposition.forecast(10366200, 10366200 + WEEK, HALF_HOUR, 90.0, 1.0,
                               [](core_t::TTime, const TDouble3Vec&) {});
        for (core_t::TTime time = 60480000; time < 60480000 + WEEK; time += HALF_HOUR) {
            decomposition.addPoint(time, 10.0);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
