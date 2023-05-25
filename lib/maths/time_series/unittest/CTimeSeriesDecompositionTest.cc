/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <core/CLogger.h>
#include <core/CMemoryCircuitBreaker.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CTimezone.h>
#include <core/Constants.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CIntegerTools.h>
#include <maths/common/CLinearAlgebraFwd.h>
#include <maths/common/CMathsFuncs.h>
#include <maths/common/CNormalMeanPrecConjugate.h>
#include <maths/common/CRestoreParams.h>
#include <maths/common/MathsTypes.h>

#include <maths/time_series/CDecayRateController.h>
#include <maths/time_series/CTimeSeriesDecomposition.h>
#include <maths/time_series/CTimeSeriesDecompositionDetail.h>
#include <maths/time_series/CTimeSeriesTestForSeasonality.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>
#include <test/CTimeSeriesTestData.h>

#include "TestUtils.h"

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
using TMeanAccumulator = maths::common::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TFloatMeanAccumulatorVec =
    std::vector<maths::common::CBasicStatistics::SSampleMean<maths::common::CFloatStorage>::TAccumulator>;

class CDebugGenerator {
public:
    static const bool ENABLED{false};

public:
    explicit CDebugGenerator(std::string file = "results.py")
        : m_File{std::move(file)} {}

    ~CDebugGenerator() {
        if (ENABLED) {
            std::ofstream file_;
            file_.open(m_File);
            auto file = (file_ << core::CScopePrintContainers{});
            file << "import matplotlib.pyplot as plt;\n";
            file << "import numpy as np;\n";
            file << "t = " << m_ValueTimes << ";\n";
            file << "f = " << m_Values << ";\n";
            file << "te = " << m_PredictionTimes << ";\n";
            file << "fe = " << m_Predictions << ";\n";
            file << "r = " << m_Errors << ";\n";
            file << "plt.figure(1);\n";
            file << "plt.clf();\n";
            file << "plt.plot(t, f);\n";
            file << "plt.plot(te, fe, 'r');\n";
            file << "plt.xlim(t[0], t[-1]);\n";
            file << "plt.ylim(min(np.min(f),np.min(fe)), max(np.max(f),np.max(fe)));\n";
            file << "plt.show();\n";
            file << "plt.figure(2);\n";
            file << "plt.clf();\n";
            file << "plt.plot(te, r, 'k');\n";
            file << "plt.xlim(t[0], t[-1]);\n";
            file << "plt.ylim(np.min(r), np.max(r));\n";
            file << "plt.show();\n";
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

const core_t::TTime ONE_MIN{60};
const core_t::TTime FIVE_MINS{300};
const core_t::TTime TEN_MINS{600};
const core_t::TTime HALF_HOUR{core::constants::HOUR / 2};
const core_t::TTime HOUR{core::constants::HOUR};
const core_t::TTime DAY{core::constants::DAY};
const core_t::TTime WEEK{core::constants::WEEK};
const core_t::TTime YEAR{core::constants::YEAR};
}

class CNanInjector {
public:
    // insert a NaN into a seasonal component bucket
    void injectNan(maths::time_series::CTimeSeriesDecomposition& decomposition,
                   size_t bucketIndex) {
        firstRegressionStatistic(seasonalComponent(decomposition), bucketIndex) =
            std::numeric_limits<double>::quiet_NaN();
    }

private:
    // helper methods to get access to the state of a seasonal component

    // return the regression statistics from the provided seasonal component
    static maths::common::CFloatStorage&
    firstRegressionStatistic(maths::time_series::CSeasonalComponent& component,
                             size_t bucketIndex) {
        return maths::common::CBasicStatistics::moment<0>(
            component.m_Bucketing.m_Buckets[bucketIndex].s_Regression.m_S)(0);
    }

    // return the first seasonal component from the provided decomposition
    static maths::time_series::CSeasonalComponent&
    seasonalComponent(maths::time_series::CTimeSeriesDecomposition& decomposition) {
        return decomposition.m_Components.m_Seasonal->m_Components[0];
    }
};

class CTestFixture {
public:
    CTestFixture() { core::CTimezone::instance().setTimezone("GMT"); }
    ~CTestFixture() { core::CTimezone::instance().setTimezone(""); }
};

class CConfigurableMemoryCircuitBreaker : public core::CMemoryCircuitBreaker {
public:
    //! Constructor
    explicit CConfigurableMemoryCircuitBreaker(bool allowAllocations)
        : m_AllowAllocations(allowAllocations) {}

    //! In hard_limit mode we don't allow any new allocations.
    bool areAllocationsAllowed() const override { return m_AllowAllocations; }

    //! Set hard_limit mode.
    void areAllocationsAllowed(bool allowAllocations) {
        m_AllowAllocations = allowAllocations;
    }

private:
    bool m_AllowAllocations;
};

class CComponentsTest : public CTestFixture {
public:
    using TComponents = ml::maths::time_series::CTimeSeriesDecompositionDetail::CComponents;
    using TSeasonal = TComponents::CSeasonal;
    using TFloatMeanAccumulatorVec =
        ml::maths::time_series::CTimeSeriesDecompositionTypes::TFloatMeanAccumulatorVec;
    using TSeasonalDecomposition = ml::maths::time_series::CSeasonalDecomposition;
    using TSeasonalComponent = ml::maths::time_series::CSeasonalDecomposition::TSeasonalComponent;

public:
    static void testAddSeasonalComponents() {
        // Test that in the hard_limit state we still can add new seasonal components if
        // at the same time we remove old seasonal components of the same total size or larger.

        // Initialise CTimeSeriesDecompositionDetails::CComponents
        double decayRate{0.01};
        core_t::TTime bucketLength{HALF_HOUR};
        std::size_t seasonalComponentSize{4};
        TComponents components{decayRate, bucketLength, seasonalComponentSize};
        components.m_Seasonal = std::make_unique<TSeasonal>();
        TComponents::CScopeAttachComponentChangeCallback attach{
            components, [](TFloatMeanAccumulatorVec) {}, [](const std::string&) {}};

        {
            // initialise CSeasonalDecomposition
            maths::time_series::CSeasonalDecomposition seasonalDecompositionComponents;
            core_t::TTime startTime = 0;
            maths::time_series::CNewTrendSummary::TFloatMeanAccumulatorVec initialValues;
            maths::time_series::CNewTrendSummary trendComponent{
                startTime, bucketLength, initialValues};
            // No component to remove so far
            maths::time_series::CSeasonalDecomposition::TBoolVec componentsToRemove{};
            seasonalDecompositionComponents.add(trendComponent);
            seasonalDecompositionComponents.add(componentsToRemove);

            // Create the first seasonal component and add it to the decomposition
            TSeasonalDecomposition::TSeasonalComponent firstSeasonalComponent;
            TSeasonalDecomposition::TPeriodDescriptor periodDescriptor{
                TSeasonalDecomposition::TPeriodDescriptor::E_Day};
            TSeasonalDecomposition::TOptionalTime startOfWeekTime;
            TSeasonalDecomposition::TFloatMeanAccumulatorVec seasonalValues;
            seasonalDecompositionComponents.add(
                "Test component 1", firstSeasonalComponent, 10.0, periodDescriptor,
                10.0, 10.0, 10.0, startOfWeekTime, seasonalValues);

            CConfigurableMemoryCircuitBreaker allocator{true};

            // add seasonal components
            components.addSeasonalComponents(seasonalDecompositionComponents, allocator);
            BOOST_REQUIRE_EQUAL(1, components.seasonal().size());
            LOG_DEBUG(<< "First add seasonal components finished");
        }
        {
            // initialise CSeasonalDecomposition
            maths::time_series::CSeasonalDecomposition seasonalDecompositionComponents;
            core_t::TTime startTime = 0;
            maths::time_series::CNewTrendSummary::TFloatMeanAccumulatorVec initialValues;
            maths::time_series::CNewTrendSummary trendComponent{
                startTime, bucketLength, initialValues};
            seasonalDecompositionComponents.add(trendComponent);

            // Mark the first seasonal component for removal
            maths::time_series::CSeasonalDecomposition::TBoolVec componentsToRemove{true};
            seasonalDecompositionComponents.add(componentsToRemove);

            // Create the second seasonal component and add it to the decomposition
            TSeasonalDecomposition::TSeasonalComponent secondSeasonalComponent;
            TSeasonalDecomposition::TPeriodDescriptor periodDescriptor{
                TSeasonalDecomposition::TPeriodDescriptor::E_Week};
            TSeasonalDecomposition::TOptionalTime startOfWeekTime;
            TSeasonalDecomposition::TFloatMeanAccumulatorVec seasonalValues;
            seasonalDecompositionComponents.add(
                "Test component 2", secondSeasonalComponent, 0.0,
                periodDescriptor, 0.0, 0.0, 1.0, startOfWeekTime, seasonalValues);

            CConfigurableMemoryCircuitBreaker allocator{false};

            // make sure that when addind the second seasonal component we remove the first one
            auto oldLastSeasonalComponent = components.seasonal().back().checksum();
            components.addSeasonalComponents(seasonalDecompositionComponents, allocator);
            auto newLastSeasonalComponent = components.seasonal().back().checksum();
            BOOST_REQUIRE_EQUAL(1, components.seasonal().size());
            BOOST_TEST_REQUIRE(oldLastSeasonalComponent != newLastSeasonalComponent);
        }
    }
};

BOOST_FIXTURE_TEST_CASE(testSuperpositionOfSines, CTestFixture) {

    // Test mixture of two sine waves.

    TTimeVec times;
    TDoubleVec trend;
    for (core_t::TTime time = 0; time < 50 * WEEK + 1; time += HALF_HOUR) {
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
    maths::time_series::CTimeSeriesDecomposition decomposition(0.01, HALF_HOUR);
    CDebugGenerator debug;

    double totalSumResidual = 0.0;
    double totalMaxResidual = 0.0;
    double totalSumValue = 0.0;
    double totalMaxValue = 0.0;
    double totalPercentileError = 0.0;

    for (std::size_t i = 0; i < times.size(); ++i) {
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
                auto prediction = decomposition.value(t, 70.0, false);
                double residual = std::fabs(trend[t / HALF_HOUR] - prediction.mean());
                sumResidual += residual;
                maxResidual = std::max(maxResidual, residual);
                sumValue += std::fabs(trend[t / HALF_HOUR]);
                maxValue = std::max(maxValue, std::fabs(trend[t / HALF_HOUR]));
                percentileError +=
                    std::max(std::max(prediction(0) - trend[t / HALF_HOUR],
                                      trend[t / HALF_HOUR] - prediction(1)),
                             0.0);
                debug.addPrediction(t, prediction.mean(), residual);
            }

            LOG_TRACE(<< "'sum residual' / 'sum value' = " << sumResidual / sumValue);
            LOG_TRACE(<< "'max residual' / 'max value' = " << maxResidual / maxValue);
            LOG_TRACE(<< "70% error = " << percentileError / sumValue);

            if (time >= 2 * WEEK) {
                BOOST_TEST_REQUIRE(sumResidual < 0.03 * sumValue);
                BOOST_TEST_REQUIRE(maxResidual < 0.03 * maxValue);
                BOOST_TEST_REQUIRE(percentileError < 0.01 * sumValue);
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

    BOOST_TEST_REQUIRE(totalSumResidual < 0.011 * totalSumValue);
    BOOST_TEST_REQUIRE(totalMaxResidual < 0.013 * totalMaxValue);
    BOOST_TEST_REQUIRE(totalPercentileError < 0.01 * totalSumValue);
}

BOOST_FIXTURE_TEST_CASE(testDistortedPeriodicProblemCase, CTestFixture) {

    // Test accuracy on real data set which caused issues historically.

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
    maths::time_series::CTimeSeriesDecomposition decomposition(0.01, bucketLength);
    CDebugGenerator debug;

    double totalSumResidual = 0.0;
    double totalMaxResidual = 0.0;
    double totalSumValue = 0.0;
    double totalMaxValue = 0.0;
    double totalPercentileError = 0.0;

    for (std::size_t i = 0; i < timeseries.size(); ++i) {
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
                auto prediction = decomposition.value(t, 70.0, false);
                double residual = std::fabs(actual - prediction.mean());
                sumResidual += residual;
                maxResidual = std::max(maxResidual, residual);
                sumValue += std::fabs(actual);
                maxValue = std::max(maxValue, std::fabs(actual));
                percentileError += std::max(
                    std::max(prediction(0) - actual, actual - prediction(1)), 0.0);
                debug.addPrediction(t, prediction.mean(), residual);
            }

            LOG_DEBUG(<< "'sum residual' / 'sum value' = " << sumResidual / sumValue);
            LOG_DEBUG(<< "'max residual' / 'max value' = " << maxResidual / maxValue);
            LOG_DEBUG(<< "70% error = " << percentileError / sumValue);

            if (time >= 2 * WEEK) {
                BOOST_TEST_REQUIRE(sumResidual < 0.27 * sumValue);
                BOOST_TEST_REQUIRE(maxResidual < 0.54 * maxValue);
                BOOST_TEST_REQUIRE(percentileError < 0.18 * sumValue);

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

    BOOST_TEST_REQUIRE(totalSumResidual < 0.17 * totalSumValue);
    BOOST_TEST_REQUIRE(totalMaxResidual < 0.23 * totalMaxValue);
    BOOST_TEST_REQUIRE(totalPercentileError < 0.10 * totalSumValue);
}

BOOST_FIXTURE_TEST_CASE(testMinimizeLongComponents, CTestFixture) {

    // Test we make longer components as smooth as possible.

    double weights[]{1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0};

    TTimeVec times;
    TDoubleVec trend;
    for (core_t::TTime time = 0; time < 50 * WEEK; time += HALF_HOUR) {
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

    maths::time_series::CTimeSeriesDecomposition decomposition(0.01, HALF_HOUR);
    CDebugGenerator debug;

    double totalSumResidual = 0.0;
    double totalMaxResidual = 0.0;
    double totalSumValue = 0.0;
    double totalMaxValue = 0.0;
    double totalPercentileError = 0.0;
    double meanSlope = 0.0;
    double refinements = 0.0;

    core_t::TTime lastWeek = 0;
    for (std::size_t i = 0; i < times.size(); ++i) {
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
                auto prediction = decomposition.value(t, 70.0, false);
                double residual = std::fabs(trend[t / HALF_HOUR] - prediction.mean());
                sumResidual += residual;
                maxResidual = std::max(maxResidual, residual);
                sumValue += std::fabs(trend[t / HALF_HOUR]);
                maxValue = std::max(maxValue, std::fabs(trend[t / HALF_HOUR]));
                percentileError +=
                    std::max(std::max(prediction(0) - trend[t / HALF_HOUR],
                                      trend[t / HALF_HOUR] - prediction(1)),
                             0.0);
                debug.addPrediction(t, prediction.mean(), residual);
            }

            LOG_TRACE(<< "'sum residual' / 'sum value' = " << sumResidual / sumValue);
            LOG_TRACE(<< "'max residual' / 'max value' = " << maxResidual / maxValue);
            LOG_TRACE(<< "70% error = " << percentileError / sumValue);

            if (time >= 2 * WEEK) {
                BOOST_TEST_REQUIRE(sumResidual < 0.12 * sumValue);
                BOOST_TEST_REQUIRE(maxResidual < 0.27 * maxValue);
                BOOST_TEST_REQUIRE(percentileError < 0.04 * sumValue);

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
                        BOOST_TEST_REQUIRE(slope < 0.004);
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

    BOOST_TEST_REQUIRE(totalSumResidual < 0.04 * totalSumValue);
    BOOST_TEST_REQUIRE(totalMaxResidual < 0.11 * totalMaxValue);
    BOOST_TEST_REQUIRE(totalPercentileError < 0.01 * totalSumValue);

    meanSlope /= refinements;
    LOG_DEBUG(<< "mean weekly |slope| = " << meanSlope);
    BOOST_TEST_REQUIRE(meanSlope < 0.002);
}

BOOST_FIXTURE_TEST_CASE(testWeekend, CTestFixture) {

    // Test weekday/weekend modulation of daily seasonality.

    double weights[]{0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0};

    for (auto offset : {0 * DAY, 5 * DAY}) {
        LOG_DEBUG(<< "offset = " << offset);

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

        maths::time_series::CTimeSeriesDecomposition decomposition(0.01, HALF_HOUR);
        CDebugGenerator debug;

        double totalSumResidual = 0.0;
        double totalMaxResidual = 0.0;
        double totalSumValue = 0.0;
        double totalMaxValue = 0.0;
        double totalPercentileError = 0.0;

        core_t::TTime lastWeek = offset;
        for (std::size_t i = 0; i < times.size(); ++i) {
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
                    auto prediction = decomposition.value(t, 70.0, false);
                    double actual = trend[(t - offset) / HALF_HOUR];
                    double residual = std::fabs(actual - prediction.mean());
                    sumResidual += residual;
                    maxResidual = std::max(maxResidual, residual);
                    sumValue += std::fabs(actual);
                    maxValue = std::max(maxValue, std::fabs(actual));
                    percentileError += std::max(
                        std::max(prediction(0) - actual, actual - prediction(1)), 0.0);
                    debug.addPrediction(t, prediction.mean(), residual);
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

        BOOST_TEST_REQUIRE(totalSumResidual < 0.021 * totalSumValue);
        BOOST_TEST_REQUIRE(totalMaxResidual < 0.033 * totalMaxValue);
        BOOST_TEST_REQUIRE(totalPercentileError < 0.01 * totalSumValue);
    }
}

BOOST_FIXTURE_TEST_CASE(testNanHandling, CTestFixture) {

    // Test flushing data which contains NaNs.

    TTimeVec times;
    TDoubleVec trend;
    for (core_t::TTime time = 0; time < 10 * WEEK + 1; time += HALF_HOUR) {
        double daily = 120.0 + 100.0 * std::sin(boost::math::double_constants::two_pi *
                                                static_cast<double>(time) /
                                                static_cast<double>(DAY));
        times.push_back(time);
        trend.push_back(daily);
    }
    TDoubleVec noise;
    test::CRandomNumbers rng;
    rng.generateNormalSamples(0.0, 16.0, times.size(), noise);

    maths::time_series::CTimeSeriesDecomposition decomposition(0.01, HALF_HOUR);

    int componentsModifiedBefore{0};

    // Run through half of the periodic data.
    std::size_t i = 0;
    for (; i < times.size() / 2; ++i) {
        decomposition.addPoint(times[i], trend[i] + noise[i],
                               core::CMemoryCircuitBreakerStub::instance(),
                               maths_t::CUnitWeights::UNIT,
                               [&componentsModifiedBefore](TFloatMeanAccumulatorVec) {
                                   ++componentsModifiedBefore;
                               });
    }

    BOOST_REQUIRE_EQUAL(2, componentsModifiedBefore);

    // Inject a NaN into one of the seasonal components.
    CNanInjector nanInjector;
    nanInjector.injectNan(decomposition, 0);

    int componentsModifiedAfter{0};

    // Run through the 2nd half of the periodic data set.
    for (++i; i < times.size(); ++i) {
        core_t::TTime time{times[i]};
        decomposition.addPoint(time, trend[i] + noise[i],
                               core::CMemoryCircuitBreakerStub::instance(),
                               maths_t::CUnitWeights::UNIT,
                               [&componentsModifiedAfter](TFloatMeanAccumulatorVec) {
                                   ++componentsModifiedAfter;
                               });
        auto value = decomposition.value(time, 0.0, false);
        BOOST_TEST_REQUIRE(maths::common::CMathsFuncs::isFinite(value(0)));
        BOOST_TEST_REQUIRE(maths::common::CMathsFuncs::isFinite(value(1)));
    }

    // The call to 'addPoint' that results in the removal of the component
    // containing the NaN value also triggers an immediate re-detection of
    // a daily seasonal component. Hence we only expect it to report that the
    // components have been modified just the once even though two modification
    // event have occurred.
    BOOST_REQUIRE_EQUAL(2, componentsModifiedAfter);

    // Check that only the daily component has been initialized.
    const TSeasonalComponentVec& components = decomposition.seasonalComponents();
    BOOST_REQUIRE_EQUAL(1, components.size());
    BOOST_REQUIRE_EQUAL(DAY, components[0].time().period());
    BOOST_TEST_REQUIRE(components[0].initialized());
}

BOOST_FIXTURE_TEST_CASE(testSinglePeriodicity, CTestFixture) {

    // Test modelling of a single seasonl component.

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

    maths::time_series::CTimeSeriesDecomposition decomposition(0.01, HALF_HOUR);
    CDebugGenerator debug;

    double totalSumResidual = 0.0;
    double totalMaxResidual = 0.0;
    double totalSumValue = 0.0;
    double totalMaxValue = 0.0;
    double totalPercentileError = 0.0;

    core_t::TTime lastWeek = 0;
    for (std::size_t i = 0; i < times.size(); ++i) {
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
                auto prediction = decomposition.value(t, 70.0, false);
                double residual =
                    std::fabs(trend[t / HALF_HOUR] + noiseMean - prediction.mean());
                sumResidual += residual;
                maxResidual = std::max(maxResidual, residual);
                sumValue += std::fabs(trend[t / HALF_HOUR]);
                maxValue = std::max(maxValue, std::fabs(trend[t / HALF_HOUR]));
                percentileError += std::max(
                    std::max(prediction(0) - (trend[t / HALF_HOUR] + noiseMean),
                             (trend[t / HALF_HOUR] + noiseMean) - prediction(1)),
                    0.0);
                debug.addPrediction(t, prediction.mean(), residual);
            }

            LOG_TRACE(<< "'sum residual' / 'sum value' = " << sumResidual / sumValue);
            LOG_TRACE(<< "'max residual' / 'max value' = " << maxResidual / maxValue);
            LOG_TRACE(<< "70% error = " << percentileError / sumValue);

            if (time >= 1 * WEEK) {
                BOOST_TEST_REQUIRE(sumResidual < 0.025 * sumValue);
                BOOST_TEST_REQUIRE(maxResidual < 0.045 * maxValue);
                BOOST_TEST_REQUIRE(percentileError < 0.02 * sumValue);

                totalSumResidual += sumResidual;
                totalMaxResidual += maxResidual;
                totalSumValue += sumValue;
                totalMaxValue += maxValue;
                totalPercentileError += percentileError;

                // Check that only the daily component has been initialized.
                const TSeasonalComponentVec& components = decomposition.seasonalComponents();
                BOOST_REQUIRE_EQUAL(1, components.size());
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
    BOOST_TEST_REQUIRE(totalMaxResidual < 0.021 * totalMaxValue);
    BOOST_TEST_REQUIRE(totalPercentileError < 0.01 * totalSumValue);

    // Check that only the daily component has been initialized.
    const TSeasonalComponentVec& components = decomposition.seasonalComponents();
    BOOST_REQUIRE_EQUAL(1, components.size());
    BOOST_REQUIRE_EQUAL(DAY, components[0].time().period());
    BOOST_TEST_REQUIRE(components[0].initialized());
}

BOOST_FIXTURE_TEST_CASE(testSeasonalOnset, CTestFixture) {

    // Test a signal which only becomes seasonal after some time.

    const double daily[]{0.0,  0.0,  0.0,  0.0,  5.0,  5.0,  5.0,  40.0,
                         40.0, 40.0, 30.0, 30.0, 35.0, 35.0, 40.0, 50.0,
                         60.0, 80.0, 80.0, 10.0, 5.0,  0.0,  0.0,  0.0};
    const double weekly[]{0.1, 0.1, 1.2, 1.0, 1.0, 0.9, 1.5};

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

    maths::time_series::CTimeSeriesDecomposition decomposition(0.01, HOUR);
    CDebugGenerator debug;

    double totalSumResidual = 0.0;
    double totalMaxResidual = 0.0;
    double totalSumValue = 0.0;
    double totalMaxValue = 0.0;
    double totalPercentileError = 0.0;

    core_t::TTime lastWeek = 0;
    for (std::size_t i = 0; i < times.size(); ++i) {
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
                auto prediction = decomposition.value(t, 70.0, false);
                double residual = std::fabs(trend[t / HOUR] - prediction.mean());
                sumResidual += residual;
                maxResidual = std::max(maxResidual, residual);
                sumValue += std::fabs(trend[t / HOUR]);
                maxValue = std::max(maxValue, std::fabs(trend[t / HOUR]));
                percentileError += std::max(std::max(prediction(0) - trend[t / HOUR],
                                                     trend[t / HOUR] - prediction(1)),
                                            0.0);
                debug.addPrediction(t, prediction.mean(), residual);
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
            if (time > 16 * WEEK) {
                // Check that there are at two least components.
                BOOST_TEST_REQUIRE(components.size() >= 2);
                BOOST_TEST_REQUIRE(components[0].initialized());
                BOOST_TEST_REQUIRE(components[1].initialized());
            } else if (time > 11 * WEEK) {
                // Check that there is at least one component.
                BOOST_TEST_REQUIRE(components.size() >= 1);
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
    BOOST_TEST_REQUIRE(totalSumResidual < 0.10 * totalSumValue);
    BOOST_TEST_REQUIRE(totalMaxResidual < 0.11 * totalMaxValue);
    BOOST_TEST_REQUIRE(totalPercentileError < 0.05 * totalSumValue);
}

BOOST_FIXTURE_TEST_CASE(testVarianceScale, CTestFixture) {

    // Test that variance scales are correctly computed.

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "Variance Spike");
    {
        core_t::TTime time = 0;
        maths::time_series::CTimeSeriesDecomposition decomposition(0.01, TEN_MINS);

        for (std::size_t i = 0; i < 50; ++i) {
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
            auto interval = decomposition.varianceScaleWeight(time + t, meanVariance, 70.0);
            LOG_TRACE(<< "time = " << t << ", expectedScale = " << expectedScale
                      << ", scale = " << interval);
            double scale = interval.mean();
            error.add(std::fabs(scale - expectedScale));
            meanScale.add(scale);
            percentileError.add(std::max(
                std::max(interval(0) - expectedScale, expectedScale - interval(1)), 0.0));
        }

        LOG_DEBUG(<< "mean error = " << maths::common::CBasicStatistics::mean(error));
        LOG_DEBUG(<< "mean 70% error = "
                  << maths::common::CBasicStatistics::mean(percentileError));
        LOG_DEBUG(<< "mean scale = " << maths::common::CBasicStatistics::mean(meanScale));
        BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(error) < 0.4);
        BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(percentileError) < 0.1);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            1.0, maths::common::CBasicStatistics::mean(meanScale), 0.02);
    }
    LOG_DEBUG(<< "Smoothly Varying Variance");
    {
        core_t::TTime time = 0;
        maths::time_series::CTimeSeriesDecomposition decomposition(0.01, TEN_MINS);

        for (std::size_t i = 0; i < 50; ++i) {
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
            auto interval = decomposition.varianceScaleWeight(time + t, meanVariance, 70.0);
            LOG_TRACE(<< "time = " << t << ", expectedScale = " << expectedScale
                      << ", scale = " << interval);
            double scale = interval.mean();
            error.add(std::fabs(scale - expectedScale));
            meanScale.add(scale);
            percentileError.add(std::max(
                std::max(interval(0) - expectedScale, expectedScale - interval(1)), 0.0));
        }

        LOG_DEBUG(<< "mean error = " << maths::common::CBasicStatistics::mean(error));
        LOG_DEBUG(<< "mean 70% error = "
                  << maths::common::CBasicStatistics::mean(percentileError));
        LOG_DEBUG(<< "mean scale = " << maths::common::CBasicStatistics::mean(meanScale));
        BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(error) < 0.3);
        BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(percentileError) < 0.15);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            1.0, maths::common::CBasicStatistics::mean(meanScale), 0.02);
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

        maths::time_series::CTimeSeriesDecomposition decomposition(0.048, HALF_HOUR);
        for (std::size_t i = 0; i < times.size(); ++i) {
            decomposition.addPoint(times[i], trend[i] + 0.3 * noise[i]);
        }

        TMeanAccumulator meanScale;
        double meanVariance = decomposition.meanVariance();
        for (core_t::TTime t = 0; t < DAY; t += TEN_MINS) {
            auto interval = decomposition.varianceScaleWeight(times.back() + t,
                                                              meanVariance, 70.0);
            LOG_TRACE(<< "time = " << t << ", scale = " << interval);
            meanScale.add(interval.mean());
        }

        LOG_DEBUG(<< "mean scale = " << maths::common::CBasicStatistics::mean(meanScale));
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            1.0, maths::common::CBasicStatistics::mean(meanScale), 0.06);
    }
}

BOOST_FIXTURE_TEST_CASE(testSpikeyDataProblemCase, CTestFixture) {

    // Test accuracy on real data set which caused issues historically.

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

    maths::time_series::CTimeSeriesDecomposition decomposition(0.01, FIVE_MINS);
    maths::common::CNormalMeanPrecConjugate model =
        maths::common::CNormalMeanPrecConjugate::nonInformativePrior(
            maths_t::E_ContinuousData, 0.01);
    CDebugGenerator debug;

    core_t::TTime lastWeek = (startTime / WEEK + 1) * WEEK;
    TTimeDoublePrVec lastWeekTimeseries;
    for (std::size_t i = 0; i < timeseries.size(); ++i) {
        core_t::TTime time = timeseries[i].first;
        double value = timeseries[i].second;

        if (time > lastWeek + WEEK) {
            LOG_TRACE(<< "Processing week");

            double sumResidual = 0.0;
            double maxResidual = 0.0;
            double sumValue = 0.0;
            double maxValue = 0.0;
            double percentileError = 0.0;

            for (std::size_t j = 0; j < lastWeekTimeseries.size(); ++j) {
                auto prediction =
                    decomposition.value(lastWeekTimeseries[j].first, 70.0, false);
                double residual =
                    std::fabs(lastWeekTimeseries[j].second - prediction.mean());
                sumResidual += residual;
                maxResidual = std::max(maxResidual, residual);
                sumValue += std::fabs(lastWeekTimeseries[j].second);
                maxValue = std::max(maxValue, std::fabs(lastWeekTimeseries[j].second));
                percentileError +=
                    std::max(std::max(prediction(0) - lastWeekTimeseries[j].second,
                                      lastWeekTimeseries[j].second - prediction(1)),
                             0.0);
                debug.addPrediction(lastWeekTimeseries[j].first, prediction.mean(), residual);
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
            time, value, core::CMemoryCircuitBreakerStub::instance(),
            maths_t::CUnitWeights::UNIT, [&model](TFloatMeanAccumulatorVec residuals) {
                model.setToNonInformative(0.0, 0.01);
                for (const auto& residual : residuals) {
                    if (maths::common::CBasicStatistics::count(residual) > 0.0) {
                        model.addSamples({maths::common::CBasicStatistics::mean(residual)},
                                         maths_t::CUnitWeights::SINGLE_UNIT);
                    }
                }
            });
        model.addSamples({decomposition.detrend(time, value, 70.0, false)},
                         maths_t::CUnitWeights::SINGLE_UNIT);
        debug.addValue(time, value);
    }

    LOG_DEBUG(<< "total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
    LOG_DEBUG(<< "total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);
    LOG_DEBUG(<< "total 70% error = " << totalPercentileError / totalSumValue);

    BOOST_TEST_REQUIRE(totalSumResidual < 0.17 * totalSumValue);
    BOOST_TEST_REQUIRE(totalMaxResidual < 0.25 * totalMaxValue);
    BOOST_TEST_REQUIRE(totalPercentileError < 0.12 * totalSumValue);

    double pMinScaled = 1.0;
    double pMinUnscaled = 1.0;
    for (std::size_t i = 0; timeseries[i].first < startTime + DAY; ++i) {
        core_t::TTime time = timeseries[i].first;
        double value = timeseries[i].second;
        double variance = model.marginalLikelihoodVariance();

        double lb;
        double ub;
        maths_t::ETail tail;
        model.probabilityOfLessLikelySamples(
            maths_t::E_TwoSided, {decomposition.detrend(time, value, 0.0, false)},
            {maths_t::seasonalVarianceScaleWeight(std::max(
                decomposition.varianceScaleWeight(time, variance, 70.0)(1), 0.25))},
            lb, ub, tail);
        double pScaled = (lb + ub) / 2.0;
        pMinScaled = std::min(pMinScaled, pScaled);

        model.probabilityOfLessLikelySamples(
            maths_t::E_TwoSided, {decomposition.detrend(time, value, 0.0, false)},
            maths_t::CUnitWeights::SINGLE_UNIT, lb, ub, tail);
        double pUnscaled = (lb + ub) / 2.0;
        pMinUnscaled = std::min(pMinUnscaled, pUnscaled);
    }

    LOG_DEBUG(<< "pMinScaled = " << pMinScaled);
    LOG_DEBUG(<< "pMinUnscaled = " << pMinUnscaled);
    BOOST_TEST_REQUIRE(pMinScaled > 1e3 * pMinUnscaled);
}

BOOST_FIXTURE_TEST_CASE(testVeryLargeValuesProblemCase, CTestFixture) {

    // Test accuracy on real data set which caused issues historically.

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

    maths::time_series::CTimeSeriesDecomposition decomposition(0.01, FIVE_MINS);
    CDebugGenerator debug;

    core_t::TTime lastWeek = (startTime / WEEK + 1) * WEEK;
    TTimeDoublePrVec lastWeekTimeseries;
    for (std::size_t i = 0; i < timeseries.size(); ++i) {
        core_t::TTime time = timeseries[i].first;
        double value = timeseries[i].second;

        if (time > lastWeek + WEEK) {
            LOG_DEBUG(<< "Processing week at " << time);

            double sumResidual = 0.0;
            double maxResidual = 0.0;
            double sumValue = 0.0;
            double maxValue = 0.0;
            double percentileError = 0.0;

            for (std::size_t j = 0; j < lastWeekTimeseries.size(); ++j) {
                auto prediction =
                    decomposition.value(lastWeekTimeseries[j].first, 70.0, false);
                double residual =
                    std::fabs(lastWeekTimeseries[j].second - prediction.mean());
                sumResidual += residual;
                maxResidual = std::max(maxResidual, residual);
                sumValue += std::fabs(lastWeekTimeseries[j].second);
                maxValue = std::max(maxValue, std::fabs(lastWeekTimeseries[j].second));
                percentileError +=
                    std::max(std::max(prediction(0) - lastWeekTimeseries[j].second,
                                      lastWeekTimeseries[j].second - prediction(1)),
                             0.0);
                debug.addPrediction(lastWeekTimeseries[j].first, prediction.mean(), residual);
            }

            LOG_DEBUG(<< "'sum residual' / 'sum value' = " << sumResidual / sumValue);
            LOG_DEBUG(<< "'max residual' / 'max value' = " << maxResidual / maxValue);
            LOG_DEBUG(<< "70% error = " << percentileError / sumValue);

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

    BOOST_TEST_REQUIRE(totalSumResidual < 0.34 * totalSumValue);
    BOOST_TEST_REQUIRE(totalMaxResidual < 0.74 * totalMaxValue);
    BOOST_TEST_REQUIRE(totalPercentileError < 0.24 * totalSumValue);

    TMeanAccumulator scale;
    double variance = decomposition.meanVariance();
    core_t::TTime time = maths::common::CIntegerTools::floor(endTime, DAY);
    for (core_t::TTime t = time; t < time + WEEK; t += TEN_MINS) {
        scale.add(decomposition.varianceScaleWeight(t, variance, 70.0).mean());
    }

    LOG_DEBUG(<< "scale = " << maths::common::CBasicStatistics::mean(scale));
    BOOST_REQUIRE_CLOSE_ABSOLUTE(1.0, maths::common::CBasicStatistics::mean(scale), 0.1);
}

BOOST_FIXTURE_TEST_CASE(testMixedSmoothAndSpikeyDataProblemCase, CTestFixture) {

    // Test accuracy on real data set which caused issues historically.

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

    maths::time_series::CTimeSeriesDecomposition decomposition(0.01, HALF_HOUR);
    CDebugGenerator debug;

    core_t::TTime lastWeek = (startTime / WEEK + 1) * WEEK;
    TTimeDoublePrVec lastWeekTimeseries;
    for (std::size_t i = 0; i < timeseries.size(); ++i) {
        core_t::TTime time = timeseries[i].first;
        double value = timeseries[i].second;

        if (time > lastWeek + WEEK) {
            LOG_DEBUG(<< "Processing week at " << time);

            double sumResidual = 0.0;
            double maxResidual = 0.0;
            double sumValue = 0.0;
            double maxValue = 0.0;
            double percentileError = 0.0;

            for (std::size_t j = 0; j < lastWeekTimeseries.size(); ++j) {
                auto prediction =
                    decomposition.value(lastWeekTimeseries[j].first, 70.0, false);
                double residual =
                    std::fabs(lastWeekTimeseries[j].second - prediction.mean());
                sumResidual += residual;
                maxResidual = std::max(maxResidual, residual);
                sumValue += std::fabs(lastWeekTimeseries[j].second);
                maxValue = std::max(maxValue, std::fabs(lastWeekTimeseries[j].second));
                percentileError +=
                    std::max(std::max(prediction(0) - lastWeekTimeseries[j].second,
                                      lastWeekTimeseries[j].second - prediction(1)),
                             0.0);
                debug.addPrediction(lastWeekTimeseries[j].first, prediction.mean(), residual);
            }

            LOG_DEBUG(<< "'sum residual' / 'sum value' = "
                      << (sumResidual == 0.0 ? 0.0 : sumResidual / sumValue));
            LOG_DEBUG(<< "'max residual' / 'max value' = "
                      << (maxResidual == 0.0 ? 0.0 : maxResidual / maxValue));
            LOG_DEBUG(<< "70% error = " << percentileError / sumValue);

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

    BOOST_TEST_REQUIRE(totalSumResidual < 0.20 * totalSumValue);
    BOOST_TEST_REQUIRE(totalMaxResidual < 0.47 * totalMaxValue);
    BOOST_TEST_REQUIRE(totalPercentileError < 0.09 * totalSumValue);
}

BOOST_FIXTURE_TEST_CASE(testDiurnalPeriodicityWithMissingValues, CTestFixture) {

    // Test the accuracy of the modeling when there are periodically missing
    // values.

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "Daily Periodic");
    {
        maths::time_series::CTimeSeriesDecomposition decomposition(0.01, HALF_HOUR);
        CDebugGenerator debug("daily.py");

        TMeanAccumulator error;
        core_t::TTime time = 0;
        for (std::size_t t = 0; t < 50; ++t) {
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
                    double prediction = decomposition.value(time, 0.0, false).mean();
                    if (decomposition.initialized()) {
                        error.add(std::fabs(value - prediction) / std::fabs(value));
                    }
                    debug.addPrediction(time, prediction, value - prediction);
                }
                time += HALF_HOUR;
            }
        }

        LOG_DEBUG(<< "mean error = " << maths::common::CBasicStatistics::mean(error));
        BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(error) < 0.11);
    }

    LOG_DEBUG(<< "Weekly Periodic");
    {
        maths::time_series::CTimeSeriesDecomposition decomposition(0.01, HOUR);
        CDebugGenerator debug("weekly.py");

        TMeanAccumulator error;
        core_t::TTime time = 0;
        for (std::size_t t = 0; t < 10; ++t) {
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
                    double prediction = decomposition.value(time, 0.0, false).mean();
                    if (decomposition.initialized()) {
                        error.add(std::fabs(value - prediction) / std::fabs(value));
                    }
                    debug.addPrediction(time, prediction, value - prediction);
                }
                time += HOUR;
            }
        }

        LOG_DEBUG(<< "mean error = " << maths::common::CBasicStatistics::mean(error));
        BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(error) < 0.12);
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

        maths::time_series::CTimeSeriesDecomposition decomposition(0.024, HALF_HOUR);
        CDebugGenerator debug("ramp.py");

        double totalSumResidual = 0.0;
        double totalMaxResidual = 0.0;
        double totalSumValue = 0.0;
        double totalMaxValue = 0.0;
        core_t::TTime lastDay = times[0];

        for (std::size_t i = 0; i < times.size(); ++i) {
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
                        auto prediction = decomposition.value(times[j], 70.0, false);
                        double residual = std::fabs(trend[j] - prediction.mean());
                        sumResidual += residual;
                        maxResidual = std::max(maxResidual, residual);
                        sumValue += std::fabs(trend[j]);
                        maxValue = std::max(maxValue, std::fabs(trend[j]));
                        debug.addPrediction(times[j], prediction.mean(), residual);
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
        core_t::TTime drops[]{0,        30 * DAY,  50 * DAY,  60 * DAY,
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

        maths::time_series::CTimeSeriesDecomposition decomposition(0.024, HALF_HOUR);
        CDebugGenerator debug("saw_tooth.py");

        double totalSumResidual = 0.0;
        double totalMaxResidual = 0.0;
        double totalSumValue = 0.0;
        double totalMaxValue = 0.0;
        core_t::TTime lastDay = times[0];

        for (std::size_t i = 0; i < times.size(); ++i) {
            decomposition.addPoint(
                times[i], trend[i] + 0.3 * noise[i],
                core::CMemoryCircuitBreakerStub::instance(),
                maths_t::countWeight(decomposition.countWeight(times[i])));
            debug.addValue(times[i], trend[i] + 0.3 * noise[i]);

            if (times[i] > lastDay + DAY) {
                LOG_TRACE(<< "Processing day " << times[i] / DAY);

                if (decomposition.initialized()) {
                    double sumResidual = 0.0;
                    double maxResidual = 0.0;
                    double sumValue = 0.0;
                    double maxValue = 0.0;

                    for (std::size_t j = i - 48; j < i; ++j) {
                        auto prediction = decomposition.value(times[j], 70.0, false);
                        double residual = std::fabs(trend[j] - prediction.mean());
                        sumResidual += residual;
                        maxResidual = std::max(maxResidual, residual);
                        sumValue += std::fabs(trend[j]);
                        maxValue = std::max(maxValue, std::fabs(trend[j]));
                        debug.addPrediction(times[j], prediction.mean(), residual);
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

        BOOST_TEST_REQUIRE(totalSumResidual / totalSumValue < 0.19);
        BOOST_TEST_REQUIRE(totalMaxResidual / totalMaxValue < 0.20);
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

    maths::time_series::CTimeSeriesDecomposition decomposition(0.072, HALF_HOUR);
    CDebugGenerator debug;

    double totalSumResidual = 0.0;
    double totalMaxResidual = 0.0;
    double totalSumValue = 0.0;
    double totalMaxValue = 0.0;
    core_t::TTime lastDay = times[0];

    for (std::size_t i = 0; i < times.size(); ++i) {
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
                    auto prediction = decomposition.value(times[j], 70.0, false);
                    double residual = std::fabs(trend[j] - prediction.mean());
                    sumResidual += residual;
                    maxResidual = std::max(maxResidual, residual);
                    sumValue += std::fabs(trend[j]);
                    maxValue = std::max(maxValue, std::fabs(trend[j]));
                    debug.addPrediction(times[j], prediction.mean(), residual);
                }

                LOG_TRACE(<< "'sum residual' / 'sum value' = "
                          << (sumResidual == 0.0 ? 0.0 : sumResidual / sumValue));
                LOG_TRACE(<< "'max residual' / 'max value' = "
                          << (maxResidual == 0.0 ? 0.0 : maxResidual / maxValue));

                totalSumResidual += sumResidual;
                totalMaxResidual += maxResidual;
                totalSumValue += sumValue;
                totalMaxValue += maxValue;

                BOOST_TEST_REQUIRE(sumResidual / sumValue < 0.45);
                BOOST_TEST_REQUIRE(maxResidual / maxValue < 0.45);
            }
            lastDay += DAY;
        }
    }

    LOG_DEBUG(<< "total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
    LOG_DEBUG(<< "total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);

    BOOST_TEST_REQUIRE(totalSumResidual / totalSumValue < 0.04);
    BOOST_TEST_REQUIRE(totalMaxResidual / totalMaxValue < 0.04);
}

BOOST_FIXTURE_TEST_CASE(testNonDiurnal, CTestFixture) {

    // Test the accuracy of the modeling of some non-daily or weekly
    // seasonal components.

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "Hourly");
    for (auto pad : {0 * DAY, 28 * DAY}) {

        TDoubleVec periodicPattern{10.0, 1.0, 0.5, 0.5, 1.0, 5.0,
                                   2.0,  1.0, 0.5, 0.5, 1.0, 6.0};

        TDoubleVec trend(pad / FIVE_MINS, 0.0);
        TTimeVec times;
        for (core_t::TTime time = 0; time < pad + 21 * DAY; time += FIVE_MINS) {
            times.push_back(time);
            trend.push_back(periodicPattern[(time / FIVE_MINS) % 12]);
        }

        TDoubleVec noise;
        rng.generateNormalSamples(0.0, 1.0, trend.size(), noise);

        core_t::TTime startTesting{pad + 28 * HOUR};

        maths::time_series::CTimeSeriesDecomposition decomposition(0.01, FIVE_MINS);
        CDebugGenerator debug("hourly." + core::CStringUtils::typeToString(pad) + ".py");

        double totalSumResidual = 0.0;
        double totalMaxResidual = 0.0;
        double totalSumValue = 0.0;
        double totalMaxValue = 0.0;
        core_t::TTime lastHour = times[0] + 3 * DAY;

        for (std::size_t i = 0; i < times.size(); ++i) {
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
                        double prediction =
                            decomposition.value(times[j], 70.0, false).mean();
                        double residual = std::fabs(trend[j] - prediction);
                        sumResidual += residual;
                        maxResidual = std::max(maxResidual, residual);
                        sumValue += std::fabs(trend[j]);
                        maxValue = std::max(maxValue, std::fabs(trend[j]));
                        debug.addPrediction(times[j], prediction, residual);
                    }

                    LOG_TRACE(<< "'sum residual' / 'sum value' = "
                              << (sumResidual == 0.0 ? 0.0 : sumResidual / sumValue));
                    LOG_TRACE(<< "'max residual' / 'max value' = "
                              << (maxResidual == 0.0 ? 0.0 : maxResidual / maxValue));

                    totalSumResidual += sumResidual;
                    totalMaxResidual += maxResidual;
                    totalSumValue += sumValue;
                    totalMaxValue += maxValue;

                    BOOST_TEST_REQUIRE(sumResidual / sumValue < 0.60);
                    BOOST_TEST_REQUIRE(maxResidual / maxValue < 0.60);
                }
                lastHour += HOUR;
            }
        }

        LOG_DEBUG(<< "total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
        LOG_DEBUG(<< "total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);

        BOOST_TEST_REQUIRE(totalSumResidual / totalSumValue < 0.17);
        BOOST_TEST_REQUIRE(totalMaxResidual / totalMaxValue < 0.14);
    }

    LOG_DEBUG(<< "Two daily");
    {
        const core_t::TTime length = 20 * DAY;

        TDoubleVec periodicPattern{10.0, 8.0, 5.5, 2.5, 2.0, 5.0,
                                   2.0,  1.0, 1.5, 3.5, 4.0, 7.0};

        TTimeVec times;
        TDoubleVec trend;
        for (core_t::TTime time = 0; time < length; time += TEN_MINS) {
            times.push_back(time);
            trend.push_back(periodicPattern[(time / 4 / HOUR) % 12]);
        }

        TDoubleVec noise;
        rng.generateNormalSamples(0.0, 2.0, times.size(), noise);

        core_t::TTime startTesting{14 * DAY};
        maths::time_series::CTimeSeriesDecomposition decomposition(0.01, TEN_MINS);
        CDebugGenerator debug("two_day.py");

        double totalSumResidual = 0.0;
        double totalMaxResidual = 0.0;
        double totalSumValue = 0.0;
        double totalMaxValue = 0.0;
        core_t::TTime lastTwoDay = times[0] + 3 * DAY;

        for (std::size_t i = 0; i < times.size(); ++i) {
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
                        double prediction =
                            decomposition.value(times[j], 70.0, false).mean();
                        double residual = std::fabs(trend[j] - prediction);
                        sumResidual += residual;
                        maxResidual = std::max(maxResidual, residual);
                        sumValue += std::fabs(trend[j]);
                        maxValue = std::max(maxValue, std::fabs(trend[j]));
                        debug.addPrediction(times[j], prediction, residual);
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
                    BOOST_TEST_REQUIRE(maxResidual / maxValue < 0.24);
                }
                lastTwoDay += 2 * DAY;
            }
        }

        LOG_DEBUG(<< "total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
        LOG_DEBUG(<< "total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);

        BOOST_TEST_REQUIRE(totalSumResidual / totalSumValue < 0.09);
        BOOST_TEST_REQUIRE(totalMaxResidual / totalMaxValue < 0.20);
    }
}

BOOST_FIXTURE_TEST_CASE(testPrecession, CTestFixture) {

    // Test the case the period is not a multiple of the bucket length.

    test::CRandomNumbers rng;

    TDoubleVec noise{0.0};
    TDoubleVec delta{0.0};
    double period = 3600.0;

    for (std::size_t t = 0; t < 5; ++t) {
        rng.generateUniformSamples(0, 120, 1, delta);
        delta[0] = std::floor(delta[0]);
        LOG_DEBUG(<< "period = " << 3600.0 + delta[0]);

        maths::time_series::CTimeSeriesDecomposition decomposition(0.048, FIVE_MINS);
        CDebugGenerator debug{"period_" + std::to_string(period + delta[0]) + ".py"};

        double sumResidual = 0.0;
        double maxResidual = 0.0;
        double sumValue = 0.0;
        double maxValue = 0.0;

        for (core_t::TTime time = 0; time < WEEK; time += FIVE_MINS) {
            double trend = 2.0 + std::sin(boost::math::double_constants::two_pi *
                                          static_cast<double>(time) /
                                          static_cast<double>(period + delta[0]));
            rng.generateNormalSamples(0.0, 0.1, 1, noise);
            decomposition.addPoint(time, trend + noise[0]);
            if (decomposition.initialized()) {
                double prediction = decomposition.value(time, 0.0, false).mean();
                double residual = decomposition.detrend(time, trend, 0.0, false, FIVE_MINS);
                sumResidual += std::fabs(residual);
                maxResidual = std::max(maxResidual, std::fabs(residual));
                sumValue += std::fabs(trend);
                maxValue = std::max(maxValue, std::fabs(trend));
                debug.addValue(time, trend);
                debug.addPrediction(time, prediction, trend - prediction);
            }
        }

        LOG_DEBUG(<< "'sum residual' / 'sum value' = " << sumResidual / sumValue);
        LOG_DEBUG(<< "'max residual' / 'max value' = " << maxResidual / maxValue);
        BOOST_TEST_REQUIRE(sumResidual / sumValue < 0.01);
        BOOST_TEST_REQUIRE(maxResidual / maxValue < 0.15);
    }
}

BOOST_FIXTURE_TEST_CASE(testRandomShifts, CTestFixture) {

    // Test small sporadic random time shifts.

    test::CRandomNumbers rng;

    TDoubleVec noise{0.0};
    double period = 3600.0;

    double shift = 0.0;
    TDoubleVec u01{0.0};

    auto trend = [&] {
        auto period_ = static_cast<core_t::TTime>(period);
        TDoubleVec periodicPattern{10.0, 1.0, 0.5, 0.5, 1.0, 1.0,
                                   2.0,  1.0, 0.5, 0.5, 1.0, 6.0};
        return [periodicPattern, period_, &shift](core_t::TTime time) {
            auto offset = (time + static_cast<core_t::TTime>(shift)) % period_;
            auto i = (12 * offset) / period_;
            auto j = (i + 1) % 12;

            return maths::common::CTools::linearlyInterpolate(
                0.0, 1.0, periodicPattern[i], periodicPattern[j],
                static_cast<double>(offset % (period_ / 12)));
        };
    }();

    maths::time_series::CTimeSeriesDecomposition decomposition(0.048, FIVE_MINS);
    CDebugGenerator debug{"shifting.py"};

    double sumResidual = 0.0;
    double maxResidual = 0.0;
    double sumValue = 0.0;
    double maxValue = 0.0;

    for (core_t::TTime time = 0; time < 3 * WEEK; time += FIVE_MINS) {
        rng.generateNormalSamples(0.0, 0.1, 1, noise);

        decomposition.addPoint(time, trend(time) + noise[0]);
        if (decomposition.initialized()) {
            double prediction = decomposition.value(time, 0.0, false).mean();
            double residual = decomposition.detrend(time, trend(time), 0.0, false, FIVE_MINS);
            sumResidual += residual;
            maxResidual = std::max(maxResidual, std::fabs(residual));
            sumValue += std::fabs(trend(time));
            maxValue = std::max(maxValue, std::fabs(trend(time)));
            debug.addValue(time, trend(time));
            debug.addPrediction(time, prediction, trend(time) - prediction);
        }

        rng.generateUniformSamples(0.0, 1.0, 1, u01);
        if (u01[0] < 0.001) {
            TDoubleVec shift_;
            rng.generateUniformSamples(0, static_cast<double>(FIVE_MINS), 1, shift_);
            shift += std::floor(shift_[0] + 0.5);
        }
    }

    LOG_DEBUG(<< "'sum residual' / 'sum value' = " << sumResidual / sumValue);
    LOG_DEBUG(<< "'max residual' / 'max value' = " << maxResidual / maxValue);

    BOOST_TEST_REQUIRE(sumResidual / sumValue < 0.025);
    BOOST_TEST_REQUIRE(maxResidual / maxValue < 0.4);
}

BOOST_FIXTURE_TEST_CASE(testYearly, CTestFixture) {

    // Test a yearly seasonal component.

    test::CRandomNumbers rng;

    maths::time_series::CTimeSeriesDecomposition decomposition(0.012, 6 * HOUR);
    maths::time_series::CDecayRateController controller(
        maths::time_series::CDecayRateController::E_PredictionBias |
            maths::time_series::CDecayRateController::E_PredictionErrorIncrease,
        1);
    CDebugGenerator debug;

    TDoubleVec noise;
    core_t::TTime time = 2 * HOUR;
    for (/**/; time < 4 * YEAR; time += 2 * HOUR) {
        double trend =
            15.0 * (2.0 + std::sin(boost::math::double_constants::two_pi *
                                   static_cast<double>(time) / static_cast<double>(YEAR))) +
            7.5 * std::sin(boost::math::double_constants::two_pi *
                           static_cast<double>(time) / static_cast<double>(DAY));
        rng.generateNormalSamples(0.0, 1.0, 1, noise);
        decomposition.addPoint(time, trend + noise[0]);
        if (decomposition.initialized()) {
            TDouble1Vec prediction{decomposition.meanValue(time)};
            TDouble1Vec predictionError{decomposition.detrend(time, trend, 0.0, false)};
            double multiplier{controller.multiplier(prediction, {predictionError},
                                                    2 * HOUR, 1.0, 0.0005)};
            decomposition.decayRate(multiplier * decomposition.decayRate());
        }
    }

    // Predict over one year and check we get reasonable accuracy.
    double maxError{0.0};
    TMeanAccumulator meanError;
    for (/**/; time < 5 * YEAR; time += 2 * HOUR) {
        double trend =
            15.0 * (2.0 + std::sin(boost::math::double_constants::two_pi *
                                   static_cast<double>(time) / static_cast<double>(YEAR))) +
            7.5 * std::sin(boost::math::double_constants::two_pi *
                           static_cast<double>(time) / static_cast<double>(DAY));
        double prediction = decomposition.value(time, 0.0, false).mean();
        double error = std::fabs((prediction - trend) / trend);
        LOG_TRACE(<< "error = " << error);
        maxError = std::max(maxError, error);
        meanError.add(error);
        debug.addValue(time, trend);
        debug.addPrediction(time, prediction, trend - prediction);
    }

    LOG_DEBUG(<< "mean error = " << maths::common::CBasicStatistics::mean(meanError));
    LOG_DEBUG(<< "max error = " << maxError);

    BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(meanError) < 0.022);
    BOOST_TEST_REQUIRE(maxError < 0.08);
}

BOOST_FIXTURE_TEST_CASE(testWithOutliers, CTestFixture) {

    // Test smooth periodic signal polluted with pepper and salt outliers.

    using TSizeVec = std::vector<std::size_t>;

    test::CRandomNumbers rng;

    TDoubleVec noise;
    TSizeVec outliers;
    TDoubleVec spikeOrTroughSelector;

    core_t::TTime buckets{WEEK / TEN_MINS};
    std::size_t numberOutliers{static_cast<std::size_t>(0.1 * static_cast<double>(buckets))};
    rng.generateUniformSamples(0, buckets, numberOutliers, outliers);
    rng.generateUniformSamples(0, 1.0, numberOutliers, spikeOrTroughSelector);
    rng.generateNormalSamples(0.0, 9.0, buckets, noise);
    std::sort(outliers.begin(), outliers.end());

    auto trend = [](core_t::TTime time) {
        return 25.0 + 20.0 * std::sin(boost::math::double_constants::two_pi *
                                      static_cast<double>(time) /
                                      static_cast<double>(DAY));
    };

    maths::time_series::CTimeSeriesDecomposition decomposition(0.01, TEN_MINS);
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
            time, value, core::CMemoryCircuitBreakerStub::instance(),
            maths_t::CUnitWeights::UNIT,
            [&newComponents](TFloatMeanAccumulatorVec) { newComponents = true; });

        if (newComponents) {
            TMeanAccumulator error;
            for (core_t::TTime endTime = time + DAY; time < endTime; time += TEN_MINS) {
                double prediction = decomposition.value(time, 0.0, false).mean();
                error.add(std::fabs(prediction - trend(time)) / trend(time));
                debug.addValue(time, value);
                debug.addPrediction(time, prediction, trend(time) - prediction);
            }

            LOG_DEBUG(<< "error = " << maths::common::CBasicStatistics::mean(error));
            BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(error) < 0.03);
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

    maths::time_series::CTimeSeriesDecomposition decomposition(0.01, HALF_HOUR);
    CDebugGenerator debug;

    TDoubleVec noise;
    std::size_t count{0};
    for (core_t::TTime time = 0; time < end; time += HALF_HOUR) {
        rng.generateNormalSamples(0.0, 4.0, 1, noise);

        decomposition.addPoint(time, trend(time) + noise[0]);
        debug.addValue(time, trend(time) + noise[0]);

        if (time - DAY == *std::lower_bound(months.begin(), months.end(), time - DAY)) {
            LOG_DEBUG(<< "time = " << time);

            std::size_t largeErrorCount = 0;

            for (core_t::TTime time_ = time - DAY; time_ < time; time_ += TEN_MINS) {
                double prediction = decomposition.value(time_, 0.0, false).mean();
                double variance =
                    4.0 * decomposition.varianceScaleWeight(time_, 4.0, 0.0).mean();
                double actual = trend(time_);
                if (std::fabs(prediction - actual) / std::sqrt(variance) > 3.0) {
                    LOG_TRACE(<< "  prediction = " << prediction);
                    LOG_TRACE(<< "  variance   = " << variance);
                    LOG_TRACE(<< "  trend      = " << trend(time_));
                    ++largeErrorCount;
                }
                debug.addPrediction(time_, prediction, actual - prediction);
            }

            LOG_DEBUG(<< "large error count = " << largeErrorCount);
            if (++count <= 5) {
                BOOST_TEST_REQUIRE(largeErrorCount > 15);
            }
            if (count >= 6) {
                BOOST_TEST_REQUIRE(largeErrorCount <= 1);
            }
        }
    }

    // Check that we can detect the calendar component.
    BOOST_REQUIRE_EQUAL(false, decomposition.calendarComponents().empty());
}

BOOST_FIXTURE_TEST_CASE(testConditionOfTrend, CTestFixture) {

    // Test numerical stability of the trend model over very long time spans.

    auto trend = [](core_t::TTime time) {
        return std::pow(static_cast<double>(time) / static_cast<double>(WEEK), 2.0);
    };

    const core_t::TTime bucketLength = 6 * HOUR;

    test::CRandomNumbers rng;

    maths::time_series::CTimeSeriesDecomposition decomposition(0.0005, bucketLength);
    TDoubleVec noise;
    for (core_t::TTime time = 0; time < 9 * YEAR; time += 6 * HOUR) {
        rng.generateNormalSamples(0.0, 4.0, 1, noise);
        decomposition.addPoint(time, trend(time) + noise[0]);
        if (time > 10 * WEEK) {
            BOOST_TEST_REQUIRE(
                std::fabs(decomposition.detrend(time, trend(time), 0.0, false)) < 3.0);
        }
    }
}

BOOST_FIXTURE_TEST_CASE(testComponentLifecycle, CTestFixture) {

    // Test we adapt to changing seasonality adding and removing components
    // as necessary.

    test::CRandomNumbers rng;

    auto trend = [](core_t::TTime time) {
        return 20.0 +
               10.0 * std::sin(boost::math::double_constants::two_pi *
                               static_cast<double>(time) / static_cast<double>(DAY)) +
               3.0 * (time > 4 * WEEK
                          ? std::sin(boost::math::double_constants::two_pi *
                                     static_cast<double>(time) / static_cast<double>(HOUR))
                          : 0.0) -
               3.0 * (time > 9 * WEEK
                          ? std::sin(boost::math::double_constants::two_pi *
                                     static_cast<double>(time) / static_cast<double>(HOUR))
                          : 0.0) +
               8.0 * (time > 16 * WEEK ? std::sin(boost::math::double_constants::two_pi *
                                                  static_cast<double>(time) /
                                                  4.0 / static_cast<double>(DAY))
                                       : 0.0) -
               8.0 * (time > 21 * WEEK ? std::sin(boost::math::double_constants::two_pi *
                                                  static_cast<double>(time) /
                                                  4.0 / static_cast<double>(DAY))
                                       : 0.0);
    };

    maths::time_series::CTimeSeriesDecomposition decomposition(0.012, FIVE_MINS);
    maths::time_series::CDecayRateController controller(
        maths::time_series::CDecayRateController::E_PredictionBias |
            maths::time_series::CDecayRateController::E_PredictionErrorIncrease,
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
                decomposition.detrend(time, trend(time) + noise[0], 0.0, false)};
            double multiplier{controller.multiplier(prediction, {predictionError},
                                                    FIVE_MINS, 1.0, 0.0001)};
            decomposition.decayRate(multiplier * decomposition.decayRate());
        }

        double prediction = decomposition.value(time, 0.0, false).mean();
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

    TDoubleVec bounds{0.013, 0.013, 0.15, 0.02};
    for (std::size_t i = 0; i < 4; ++i) {
        double error{maths::common::CBasicStatistics::mean(errors[i])};
        LOG_DEBUG(<< "error = " << error);
        BOOST_TEST_REQUIRE(error < bounds[i]);
    }
}

BOOST_FIXTURE_TEST_CASE(testStability, CTestFixture) {

    auto trend = [](core_t::TTime time) {
        return 2000.0 + (time < 10 * WEEK ? 100.0 * weekends(time) : -2000.0);
    };

    maths::time_series::CTimeSeriesDecomposition decomposition(0.012, HALF_HOUR);
    maths::time_series::CDecayRateController controller(
        maths::time_series::CDecayRateController::E_PredictionBias |
            maths::time_series::CDecayRateController::E_PredictionErrorIncrease,
        1);
    CDebugGenerator debug;

    for (core_t::TTime time = 0; time < 2 * YEAR; time += HALF_HOUR) {
        decomposition.addPoint(time, trend(time));
        debug.addValue(time, trend(time));

        if (decomposition.initialized()) {
            TDouble1Vec mean{decomposition.meanValue(time)};
            TDouble1Vec predictionError{decomposition.detrend(time, trend(time), 0.0, false)};
            double multiplier{controller.multiplier(mean, {predictionError},
                                                    HALF_HOUR, 1.0, 0.0005)};
            decomposition.decayRate(multiplier * decomposition.decayRate());
        }

        double prediction{decomposition.value(time, 0.0, false).mean()};
        debug.addPrediction(time, prediction, trend(time) - prediction);

        if (time > 20 * WEEK) {
            BOOST_TEST_REQUIRE(std::fabs(trend(time) - prediction) < 5.0);
        }
    }
}

BOOST_FIXTURE_TEST_CASE(testRemoveSeasonal, CTestFixture) {

    // Check we correctly remove all seasonal components.

    test::CRandomNumbers rng;

    auto trend = [](core_t::TTime time) {
        return 20.0 + 10.0 * (time <= 4 * WEEK
                                  ? std::sin(boost::math::double_constants::two_pi *
                                             static_cast<double>(time) /
                                             static_cast<double>(DAY))
                                  : 0.0);
    };

    for (auto noiseVariance : {1.0, 0.0}) {

        maths::time_series::CTimeSeriesDecomposition decomposition(0.012, FIVE_MINS);
        maths::time_series::CDecayRateController controller{
            maths::time_series::CDecayRateController::E_PredictionBias |
                maths::time_series::CDecayRateController::E_PredictionErrorIncrease,
            1};
        CDebugGenerator debug;

        TDoubleVec noise;
        for (core_t::TTime time = 0; time < 20 * WEEK; time += FIVE_MINS) {
            if (noiseVariance > 0.0) {
                rng.generateNormalSamples(0.0, 1.0, 1, noise);
            } else {
                noise.assign(1, 0.0);
            }

            decomposition.addPoint(time, trend(time) + noise[0]);
            debug.addValue(time, trend(time) + noise[0]);

            if (decomposition.initialized()) {
                TDouble1Vec prediction{decomposition.meanValue(time)};
                TDouble1Vec predictionError{
                    decomposition.detrend(time, trend(time) + noise[0], 0.0, false)};
                double multiplier{controller.multiplier(
                    prediction, {predictionError}, FIVE_MINS, 1.0, 0.0001)};
                decomposition.decayRate(multiplier * decomposition.decayRate());
            }

            double prediction{decomposition.value(time, 0.0, false).mean()};
            debug.addPrediction(time, prediction, trend(time) + noise[0] - prediction);
        }

        // We shouldn't have any components left at this point.
        BOOST_REQUIRE_EQUAL(0, decomposition.seasonalComponents().size());
    }
}

BOOST_FIXTURE_TEST_CASE(testFastAndSlowSeasonality, CTestFixture) {

    // Test we have good modelling of the fast component after detecting a slow
    // periodic component.

    test::CRandomNumbers rng;

    auto trend = [] {
        TDoubleVec fast{0.0, 7.0, 10.0, 7.0, 0.0, 0.0};
        return [=](core_t::TTime time) {
            return 2.0 +
                   std::sin(boost::math::double_constants::two_pi *
                            static_cast<double>(time) / static_cast<double>(DAY)) +
                   fast[static_cast<std::size_t>(time / FIVE_MINS) % fast.size()];
        };
    }();

    maths::time_series::CTimeSeriesDecomposition decomposition(0.012, ONE_MIN);
    maths::time_series::CDecayRateController controller{
        maths::time_series::CDecayRateController::E_PredictionBias |
            maths::time_series::CDecayRateController::E_PredictionErrorIncrease,
        1};
    CDebugGenerator debug;

    TMeanAccumulator meanError;

    TDoubleVec noise;
    for (core_t::TTime time = 0; time < WEEK; time += ONE_MIN) {
        rng.generateNormalSamples(0.0, 0.2, 1, noise);

        decomposition.addPoint(time, trend(time) + noise[0]);
        debug.addValue(time, trend(time) + noise[0]);

        if (decomposition.initialized()) {
            TDouble1Vec prediction{decomposition.meanValue(time)};
            TDouble1Vec predictionError{
                decomposition.detrend(time, trend(time) + noise[0], 0.0, false)};
            double multiplier{controller.multiplier(prediction, {predictionError},
                                                    FIVE_MINS, 1.0, 0.0001)};
            decomposition.decayRate(multiplier * decomposition.decayRate());
        }

        double prediction{decomposition.value(time, 0.0, false).mean()};
        debug.addPrediction(time, prediction, trend(time) + noise[0] - prediction);
        if (time > 4 * DAY) {
            double error{(std::fabs(decomposition.detrend(time, trend(time), 0.0, false, FIVE_MINS))) /
                         std::fabs(trend(time))};
            BOOST_TEST_REQUIRE(error < 0.25);
            meanError.add(error);
        }
    }

    BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(meanError) < 0.01);

    // We should be modelling both seasonalities.
    BOOST_TEST_REQUIRE(2, decomposition.seasonalComponents().size());
}

BOOST_FIXTURE_TEST_CASE(testNonNegative, CTestFixture) {

    // Test if we specify the time series is non-negative then we never predict
    // a negative value for it.

    test::CRandomNumbers rng;

    auto trend = [](core_t::TTime time) {
        return std::max(15.0 - 0.5 * static_cast<double>(time) / static_cast<double>(DAY), 1.0) +
               std::sin(boost::math::double_constants::two_pi *
                        static_cast<double>(time) / static_cast<double>(DAY));
    };

    maths::time_series::CTimeSeriesDecomposition decomposition(0.012, FIVE_MINS);
    CDebugGenerator debug;

    TMeanAccumulator meanError;

    TDoubleVec noise;
    for (core_t::TTime time = 0; time < 6 * WEEK; time += FIVE_MINS) {
        rng.generateNormalSamples(0.0, 0.1, 1, noise);

        decomposition.addPoint(time, trend(time) + noise[0]);
        debug.addValue(time, trend(time) + noise[0]);

        auto prediction = decomposition.value(time, 0.0, true);
        BOOST_TEST_REQUIRE(prediction(0) >= 0.0);
        BOOST_TEST_REQUIRE(prediction(1) >= 0.0);
        debug.addPrediction(time, prediction.mean(),
                            trend(time) + noise[0] - prediction.mean());

        if (time > 4 * DAY && trend(time) > 1.0) {
            double error{(std::fabs(decomposition.detrend(time, trend(time), 0.0, true, FIVE_MINS))) /
                         std::fabs(trend(time))};
            BOOST_TEST_REQUIRE(error < 0.8);
            meanError.add(error);
        }
    }

    BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(meanError) < 0.1);
}

BOOST_FIXTURE_TEST_CASE(testSwap, CTestFixture) {

    // Test that swapping preserves checksums.

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

    maths::time_series::CTimeSeriesDecomposition decomposition1(decayRate, bucketLength);
    maths::time_series::CTimeSeriesDecomposition decomposition2(2.0 * decayRate,
                                                                2 * bucketLength);

    for (std::size_t i = 0; i < times.size(); i += 2) {
        decomposition1.addPoint(times[i], trend1[i] + noise[i]);
        decomposition2.addPoint(times[i], trend2[i] + noise[i + 1]);
    }

    std::uint64_t checksum1 = decomposition1.checksum();
    std::uint64_t checksum2 = decomposition2.checksum();

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

    maths::time_series::CTimeSeriesDecomposition origDecomposition(decayRate, bucketLength);

    for (std::size_t i = 0; i < times.size(); ++i) {
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
    maths::common::STimeSeriesDecompositionRestoreParams params{
        decayRate + 0.1, bucketLength,
        maths::common::SDistributionRestoreParams{maths_t::E_ContinuousData, decayRate + 0.1}};

    maths::time_series::CTimeSeriesDecomposition restoredDecomposition(params, traverser);

    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredDecomposition.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    BOOST_REQUIRE_EQUAL(origXml, newXml);
}

BOOST_FIXTURE_TEST_CASE(testNoAllocationsAllowed, CTestFixture) {
    // Test that when CTimeSeriesDecompositionAllocator::areAllocationsAllowed returns false,
    // the call of addPoint() does not lead to creation of new seasonal or calendar components
    // in the decomposition.

    TTimeVec months{2505600,   // Fri 30th Jan
                    4924800,   // Fri 27th Feb
                    7344000,   // Fri 27th Mar
                    9763200,   // Fri 24th Apr
                    12787200,  // Fri 29th May
                    15206400,  // Fri 26th Jun
                    18230400,  // Fri 31st Jul
                    18316800}; // Sat 1st Aug
    core_t::TTime end = months.back();
    TDoubleVec errors{5.0, 15.0, 35.0, 32.0, 25.0, 36.0, 22.0, 12.0, 3.0};
    double decayRate{0.01};

    auto trend = [](core_t::TTime t) {
        double weekly = 1200.0 + 1000.0 * std::sin(boost::math::double_constants::two_pi *
                                                   static_cast<double>(t) /
                                                   static_cast<double>(WEEK));
        double daily = 5.0 + 5.0 * std::sin(boost::math::double_constants::two_pi *
                                            static_cast<double>(t) /
                                            static_cast<double>(DAY));
        double result = weekly + daily;
        return result;
    };

    test::CRandomNumbers rng;
    {
        CConfigurableMemoryCircuitBreaker allocator{false};

        maths::time_series::CTimeSeriesDecomposition decomposition(decayRate, HALF_HOUR);

        TDoubleVec noise;
        for (core_t::TTime time = 0; time < end; time += HALF_HOUR) {
            rng.generateNormalSamples(0.0, 4.0, 1, noise);
            decomposition.addPoint(time, trend(time) + noise[0], allocator);
        }

        // Check that we don't have any seasonal components.
        BOOST_REQUIRE_EQUAL(true, decomposition.seasonalComponents().empty());

        // Check that we don't have any calendar components.
        BOOST_REQUIRE_EQUAL(true, decomposition.calendarComponents().empty());
    }
}

BOOST_FIXTURE_TEST_CASE(testAddSeasonalComponentsNoAllocations, CComponentsTest) {
    // Test that in the hard_limit state we still can add new seasonal components if
    // at the same time we remove old seasonal components of the same total size or larger.
    this->testAddSeasonalComponents();
}

BOOST_AUTO_TEST_SUITE_END()
