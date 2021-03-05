/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/Constants.h>

#include <maths/CIntegerTools.h>
#include <maths/CLeastSquaresOnlineRegression.h>
#include <maths/CLeastSquaresOnlineRegressionDetail.h>
#include <maths/COrderings.h>
#include <maths/CSeasonalComponent.h>
#include <maths/CSeasonalTime.h>
#include <maths/CTools.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

#include <utility>
#include <vector>

BOOST_AUTO_TEST_SUITE(CSeasonalComponentTest)

using namespace ml;

namespace {

using TDoubleDoublePr = std::pair<double, double>;
using TDoubleVec = std::vector<double>;
using TTimeVec = std::vector<core_t::TTime>;
using TTimeDoublePr = std::pair<core_t::TTime, double>;
using TTimeDoublePrVec = std::vector<TTimeDoublePr>;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

class CTestSeasonalComponent : public maths::CSeasonalComponent {
public:
    // Bring base class method hidden by the signature above into scope.
    using maths::CSeasonalComponent::initialize;

public:
    CTestSeasonalComponent(core_t::TTime period,
                           std::size_t space,
                           double decayRate = 0.0,
                           double minimumBucketLength = 0.0)
        : maths::CSeasonalComponent(maths::CGeneralPeriodTime{period}, space, decayRate, minimumBucketLength) {
        this->initialize();
    }
    CTestSeasonalComponent(const maths::CSeasonalTime& time,
                           std::size_t space,
                           double decayRate = 0.0,
                           double minimumBucketLength = 0.0)
        : maths::CSeasonalComponent(time, space, decayRate, minimumBucketLength) {
        this->initialize();
    }

    void addPoint(core_t::TTime time, double value, double weight = 1.0) {
        if (this->shouldInterpolate(time)) {
            this->interpolate(time);
        }
        this->maths::CSeasonalComponent::add(time, value, weight);
    }
};

void generateSeasonalValues(test::CRandomNumbers& rng,
                            const TTimeDoublePrVec& function,
                            core_t::TTime startTime,
                            core_t::TTime endTime,
                            std::size_t numberSamples,
                            TTimeDoublePrVec& samples) {
    using TSizeVec = std::vector<std::size_t>;

    // Generate time uniformly at random in the interval [startTime, endTime).

    core_t::TTime period = function[function.size() - 1].first;

    TSizeVec times;
    rng.generateUniformSamples(static_cast<std::size_t>(startTime),
                               static_cast<std::size_t>(endTime), numberSamples, times);
    std::sort(times.begin(), times.end());
    for (std::size_t i = 0; i < times.size(); ++i) {
        core_t::TTime offset = static_cast<core_t::TTime>(times[i] % period);
        std::size_t b = std::lower_bound(function.begin(), function.end(), offset,
                                         maths::COrderings::SFirstLess()) -
                        function.begin();
        b = maths::CTools::truncate(b, std::size_t(1), std::size_t(function.size() - 1));
        std::size_t a = b - 1;
        double m = (function[b].second - function[a].second) /
                   static_cast<double>(function[b].first - function[a].first);
        samples.emplace_back(
            times[i], function[a].second +
                          m * static_cast<double>(offset - function[a].first));
    }
}

double mean(const TDoubleDoublePr& x) {
    return (x.first + x.second) / 2.0;
}
}

BOOST_AUTO_TEST_CASE(testSwap) {

    // Test swap preserves object checksums.

    TTimeDoublePrVec function;
    for (std::size_t i = 0; i < 25; ++i) {
        function.emplace_back((i * core::constants::DAY) / 24,
                              0.1 * static_cast<double>(i));
    }

    test::CRandomNumbers rng;

    std::size_t n{100};

    TTimeDoublePrVec samples;
    generateSeasonalValues(rng, function, 0, 3 * core::constants::DAY, n, samples);

    CTestSeasonalComponent seasonal1(core::constants::DAY, 24);
    CTestSeasonalComponent seasonal2(core::constants::DAY, 24);
    for (std::size_t i = 0; i < n; ++i) {
        seasonal1.addPoint(samples[i].first, samples[i].second);
        seasonal2.addPoint(samples[i].first,
                           0.01 * static_cast<double>(samples[i].first) +
                               samples[i].second);
    }
    seasonal1.interpolate(3 * core::constants::DAY);
    seasonal2.interpolate(3 * core::constants::DAY);

    std::uint64_t checksum1{seasonal1.checksum()};
    std::uint64_t checksum2{seasonal2.checksum()};

    seasonal1.swap(seasonal2);

    BOOST_REQUIRE_EQUAL(checksum1, seasonal2.checksum());
    BOOST_REQUIRE_EQUAL(checksum2, seasonal1.checksum());
}

BOOST_AUTO_TEST_CASE(testInitialize) {
    // Test we correctly initialize a component when supplying values.

    maths::CSeasonalComponent::TFloatMeanAccumulatorVec initialValues(100);
    for (std::size_t i = 0; i < 20; ++i) {
        initialValues[i].add(static_cast<double>(i % 10));
    }

    maths::CSeasonalComponent component{maths::CGeneralPeriodTime{20}, 10, 0.0, 0};
    component.initialize(0, 200, initialValues);

    BOOST_REQUIRE_EQUAL(false, component.shouldInterpolate(59));
    BOOST_REQUIRE_EQUAL(true, component.shouldInterpolate(60));

    TMeanAccumulator meanError;
    for (std::size_t i = 20; i < 30; ++i) {
        meanError.add(std::fabs(maths::CBasicStatistics::mean(component.value(
                                    2 * static_cast<core_t::TTime>(i), 0.0)) -
                                static_cast<double>(i % 10)));
    }
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanError) < 0.3);
}

BOOST_AUTO_TEST_CASE(testShouldInterpolate) {
    // Test should interpolate is true at the times we expect.

    // Vanilla...
    for (core_t::TTime period : {core::constants::DAY, core::constants::WEEK}) {
        LOG_DEBUG(<< "period = " << period);

        maths::CGeneralPeriodTime time{period};
        maths::CSeasonalComponent component{time, 36};
        BOOST_REQUIRE(component.shouldInterpolate(0));
        component.interpolate(0);

        for (core_t::TTime startTime = 0; startTime < 2 * period; startTime += period) {
            core_t::TTime t{startTime};
            for (/**/; t < startTime + period; t += core::constants::HOUR) {
                if (component.shouldInterpolate(t))
                    LOG_DEBUG(<< t);
                BOOST_REQUIRE_EQUAL(false, component.shouldInterpolate(t));
            }
            BOOST_REQUIRE(component.shouldInterpolate(t));
            component.interpolate(t);
        }
        // Gap...
        BOOST_REQUIRE(component.shouldInterpolate(5 * core::constants::WEEK));
    }

    // Time windowed...
    for (core_t::TTime startOfWeek : {0, 10800}) {
        LOG_DEBUG(<< "start of week = " << startOfWeek);
        for (core_t::TTime period : {core::constants::DAY, core::constants::WEEK}) {
            LOG_DEBUG(<< "period = " << period);

            maths::CDiurnalTime weekendTime{startOfWeek, 0, core::constants::WEEKEND, period};
            maths::CSeasonalComponent weekendComponent{weekendTime, 36};
            BOOST_REQUIRE(weekendComponent.shouldInterpolate(0));
            weekendComponent.interpolate(startOfWeek);
            for (core_t::TTime startTime = 0; startTime < 2 * period; startTime += period) {
                core_t::TTime time{startTime + startOfWeek};
                for (/**/; time < startTime + startOfWeek + period;
                     time += core::constants::HOUR) {
                    BOOST_REQUIRE_EQUAL(false, weekendComponent.shouldInterpolate(time));
                }
                BOOST_REQUIRE(weekendComponent.shouldInterpolate(time));
                weekendComponent.interpolate(time);
            }
            // Gap...
            BOOST_REQUIRE(weekendComponent.shouldInterpolate(5 * core::constants::WEEK));

            maths::CDiurnalTime weekdayTime{startOfWeek, core::constants::WEEKEND,
                                            core::constants::WEEK, period};
            maths::CSeasonalComponent weekdayComponent{weekdayTime, 36};
            BOOST_REQUIRE(weekdayComponent.shouldInterpolate(0));
            weekdayComponent.interpolate(startOfWeek + core::constants::WEEKEND);
            for (core_t::TTime startTime = 0; startTime < 2 * period; startTime += period) {
                core_t::TTime time{startTime + startOfWeek + core::constants::WEEKEND};
                for (/**/; time < startTime + startOfWeek + core::constants::WEEKEND + period;
                     time += core::constants::HOUR) {
                    BOOST_REQUIRE_EQUAL(false, weekdayComponent.shouldInterpolate(time));
                }
                BOOST_REQUIRE(weekdayComponent.shouldInterpolate(time));
                weekdayComponent.interpolate(time);
            }
            // Gap...
            BOOST_REQUIRE(weekdayComponent.shouldInterpolate(5 * core::constants::WEEK));
        }
    }
}

BOOST_AUTO_TEST_CASE(testConstant) {
    // Test that the seasonal component tends to the correct constant.

    const core_t::TTime startTime = 1354492800;

    TTimeDoublePrVec function;
    for (std::size_t i = 0; i < 25; ++i) {
        function.emplace_back((i * core::constants::DAY) / 24, 0.0);
    }

    test::CRandomNumbers rng;

    std::size_t n = 5000;

    TTimeDoublePrVec samples;
    generateSeasonalValues(rng, function, startTime,
                           startTime + 31 * core::constants::DAY, n, samples);

    TDoubleVec residuals;
    rng.generateGammaSamples(10.0, 1.2, n, residuals);
    double residualMean = maths::CBasicStatistics::mean(residuals);

    CTestSeasonalComponent seasonal(core::constants::DAY, 24);

    double totalError1 = 0.0;
    double totalError2 = 0.0;
    core_t::TTime time = startTime;
    for (std::size_t i = 0u, d = 0; i < n; ++i) {
        seasonal.addPoint(samples[i].first, samples[i].second + residuals[i]);

        if (samples[i].first >= time + core::constants::DAY) {
            LOG_TRACE(<< "Processing day = " << ++d);

            time += core::constants::DAY;

            double error1 = 0.0;
            double error2 = 0.0;
            for (std::size_t j = 0; j < function.size(); ++j) {
                TDoubleDoublePr interval = seasonal.value(time + function[j].first, 70.0);
                double f = function[j].second + residualMean;

                double e = mean(interval) - f;
                error1 += std::fabs(e);
                error2 += std::max(std::max(interval.first - f, f - interval.second), 0.0);
            }

            if (d > 1) {
                LOG_TRACE(<< "f(0) = " << mean(seasonal.value(time, 0.0)) << ", f(T) = "
                          << mean(seasonal.value(time + core::constants::DAY - 1, 0.0)));
                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    mean(seasonal.value(time, 0.0)),
                    mean(seasonal.value(time + core::constants::DAY - 1, 0.0)), 0.1);
            }
            error1 /= static_cast<double>(function.size());
            error2 /= static_cast<double>(function.size());
            LOG_TRACE(<< "error1 = " << error1);
            LOG_TRACE(<< "error2 = " << error2);
            BOOST_TEST_REQUIRE(error1 < 1.4);
            BOOST_TEST_REQUIRE(error2 < 0.35);
            totalError1 += error1;
            totalError2 += error2;
        }
    }

    totalError1 /= 30.0;
    totalError2 /= 30.0;
    LOG_DEBUG(<< "totalError1 = " << totalError1);
    LOG_DEBUG(<< "totalError2 = " << totalError2);
    BOOST_TEST_REQUIRE(totalError1 < 0.6);
    BOOST_TEST_REQUIRE(totalError2 < 0.15);
}

BOOST_AUTO_TEST_CASE(testStablePeriodic) {
    const core_t::TTime startTime = 1354492800;

    test::CRandomNumbers rng;

    // Test smooth...
    {
        LOG_DEBUG(<< "*** sin(2 * pi * t / 24 hrs) ***");

        TTimeDoublePrVec function;
        for (core_t::TTime i = 0; i < 49; ++i) {
            core_t::TTime t = (i * core::constants::DAY) / 48;
            double ft = 100.0 + 40.0 * std::sin(boost::math::double_constants::two_pi *
                                                static_cast<double>(i) / 48.0);
            function.emplace_back(t, ft);
        }

        std::size_t n = 5000;

        TTimeDoublePrVec samples;
        generateSeasonalValues(rng, function, startTime,
                               startTime + 31 * core::constants::DAY, n, samples);

        TDoubleVec residuals;
        rng.generateGammaSamples(10.0, 1.2, n, residuals);
        double residualMean = maths::CBasicStatistics::mean(residuals);

        CTestSeasonalComponent seasonal(core::constants::DAY, 24, 0.01);

        double totalError1 = 0.0;
        double totalError2 = 0.0;
        core_t::TTime time = startTime;
        for (std::size_t i = 0u, d = 0; i < n; ++i) {
            seasonal.addPoint(samples[i].first, samples[i].second + residuals[i]);

            if (samples[i].first >= time + core::constants::DAY) {
                LOG_TRACE(<< "Processing day = " << ++d);

                time += core::constants::DAY;

                double error1 = 0.0;
                double error2 = 0.0;
                for (std::size_t j = 0; j < function.size(); ++j) {
                    TDoubleDoublePr interval = seasonal.value(time + function[j].first, 70.0);
                    double f = residualMean + function[j].second;
                    double e = mean(interval) - f;
                    error1 += std::fabs(e);
                    error2 += std::max(std::max(interval.first - f, f - interval.second), 0.0);
                }

                if (d > 1) {
                    LOG_TRACE(<< "f(0) = " << mean(seasonal.value(time, 0.0)) << ", f(T) = "
                              << mean(seasonal.value(time + core::constants::DAY - 1, 0.0)));
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(
                        mean(seasonal.value(time, 0.0)),
                        mean(seasonal.value(time + core::constants::DAY - 1, 0.0)), 0.1);
                }

                error1 /= static_cast<double>(function.size());
                error2 /= static_cast<double>(function.size());
                LOG_TRACE(<< "error1 = " << error1);
                LOG_TRACE(<< "error2 = " << error2);
                BOOST_TEST_REQUIRE(error1 < 1.7);
                BOOST_TEST_REQUIRE(error2 < 0.6);
                totalError1 += error1;
                totalError2 += error2;

                seasonal.propagateForwardsByTime(1.0);
            }
        }

        totalError1 /= 30.0;
        totalError2 /= 30.0;
        LOG_DEBUG(<< "totalError1 = " << totalError1);
        LOG_DEBUG(<< "totalError2 = " << totalError2);
        BOOST_TEST_REQUIRE(totalError1 < 0.46);
        BOOST_TEST_REQUIRE(totalError2 < 0.01);
    }

    // Test non-smooth...
    {
        LOG_DEBUG(<< "*** piecewise linear ***");

        TTimeDoublePrVec function{
            TTimeDoublePr(0, 1.0),       TTimeDoublePr(1800, 1.0),
            TTimeDoublePr(3600, 2.0),    TTimeDoublePr(5400, 3.0),
            TTimeDoublePr(7200, 5.0),    TTimeDoublePr(9000, 5.0),
            TTimeDoublePr(10800, 10.0),  TTimeDoublePr(12600, 10.0),
            TTimeDoublePr(14400, 12.0),  TTimeDoublePr(16200, 12.0),
            TTimeDoublePr(18000, 14.0),  TTimeDoublePr(19800, 12.0),
            TTimeDoublePr(21600, 10.0),  TTimeDoublePr(23400, 14.0),
            TTimeDoublePr(25200, 16.0),  TTimeDoublePr(27000, 50.0),
            TTimeDoublePr(28800, 300.0), TTimeDoublePr(30600, 330.0),
            TTimeDoublePr(32400, 310.0), TTimeDoublePr(34200, 290.0),
            TTimeDoublePr(36000, 280.0), TTimeDoublePr(37800, 260.0),
            TTimeDoublePr(39600, 250.0), TTimeDoublePr(41400, 230.0),
            TTimeDoublePr(43200, 230.0), TTimeDoublePr(45000, 220.0),
            TTimeDoublePr(46800, 240.0), TTimeDoublePr(48600, 220.0),
            TTimeDoublePr(50400, 260.0), TTimeDoublePr(52200, 250.0),
            TTimeDoublePr(54000, 260.0), TTimeDoublePr(55800, 270.0),
            TTimeDoublePr(57600, 280.0), TTimeDoublePr(59400, 290.0),
            TTimeDoublePr(61200, 290.0), TTimeDoublePr(63000, 60.0),
            TTimeDoublePr(64800, 20.0),  TTimeDoublePr(66600, 18.0),
            TTimeDoublePr(68400, 19.0),  TTimeDoublePr(70200, 10.0),
            TTimeDoublePr(72000, 10.0),  TTimeDoublePr(73800, 5.0),
            TTimeDoublePr(75600, 5.0),   TTimeDoublePr(77400, 10.0),
            TTimeDoublePr(79200, 5.0),   TTimeDoublePr(81000, 3.0),
            TTimeDoublePr(82800, 1.0),   TTimeDoublePr(84600, 1.0),
            TTimeDoublePr(86400, 1.0)};

        std::size_t n = 6000;

        TTimeDoublePrVec samples;
        generateSeasonalValues(rng, function, startTime,
                               startTime + 41 * core::constants::DAY, n, samples);

        TDoubleVec residuals;
        rng.generateGammaSamples(10.0, 1.2, n, residuals);
        double residualMean = maths::CBasicStatistics::mean(residuals);

        CTestSeasonalComponent seasonal(core::constants::DAY, 24, 0.01);

        double totalError1 = 0.0;
        double totalError2 = 0.0;
        core_t::TTime time = startTime;
        for (std::size_t i = 0u, d = 0; i < n; ++i) {
            seasonal.addPoint(samples[i].first, samples[i].second + residuals[i]);

            if (samples[i].first >= time + core::constants::DAY) {
                LOG_TRACE(<< "Processing day = " << ++d);

                time += core::constants::DAY;
                double error1 = 0.0;
                double error2 = 0.0;
                for (std::size_t j = 0; j < function.size(); ++j) {
                    TDoubleDoublePr interval = seasonal.value(time + function[j].first, 70.0);
                    double f = residualMean + function[j].second;

                    double e = mean(interval) - f;
                    error1 += std::fabs(e);
                    error2 += std::max(std::max(interval.first - f, f - interval.second), 0.0);
                }

                if (d > 1) {
                    LOG_TRACE(<< "f(0) = " << mean(seasonal.value(time, 0.0)) << ", f(T) = "
                              << mean(seasonal.value(time + core::constants::DAY - 1, 0.0)));
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(
                        mean(seasonal.value(time, 0.0)),
                        mean(seasonal.value(time + core::constants::DAY - 1, 0.0)), 0.1);
                }

                error1 /= static_cast<double>(function.size());
                error2 /= static_cast<double>(function.size());
                LOG_TRACE(<< "error1 = " << error1);
                LOG_TRACE(<< "error2 = " << error2);
                BOOST_TEST_REQUIRE(error1 < 11.0);
                BOOST_TEST_REQUIRE(error2 < 4.7);
                totalError1 += error1;
                totalError2 += error2;

                seasonal.propagateForwardsByTime(1.0);
            }
        }

        totalError1 /= 40.0;
        totalError2 /= 40.0;
        LOG_DEBUG(<< "totalError1 = " << totalError1);
        LOG_DEBUG(<< "totalError2 = " << totalError2);
        BOOST_TEST_REQUIRE(totalError1 < 7.5);
        BOOST_TEST_REQUIRE(totalError2 < 4.2);
    }
}

BOOST_AUTO_TEST_CASE(testTimeVaryingPeriodic) {
    // Test a signal with periodicity which changes slowly over time.

    core_t::TTime startTime = 0;

    TTimeDoublePrVec function{
        TTimeDoublePr(0, 1.0),       TTimeDoublePr(1800, 1.0),
        TTimeDoublePr(3600, 2.0),    TTimeDoublePr(5400, 3.0),
        TTimeDoublePr(7200, 5.0),    TTimeDoublePr(9000, 5.0),
        TTimeDoublePr(10800, 10.0),  TTimeDoublePr(12600, 10.0),
        TTimeDoublePr(14400, 12.0),  TTimeDoublePr(16200, 12.0),
        TTimeDoublePr(18000, 14.0),  TTimeDoublePr(19800, 12.0),
        TTimeDoublePr(21600, 10.0),  TTimeDoublePr(23400, 14.0),
        TTimeDoublePr(25200, 16.0),  TTimeDoublePr(27000, 50.0),
        TTimeDoublePr(28800, 300.0), TTimeDoublePr(30600, 330.0),
        TTimeDoublePr(32400, 310.0), TTimeDoublePr(34200, 290.0),
        TTimeDoublePr(36000, 280.0), TTimeDoublePr(37800, 260.0),
        TTimeDoublePr(39600, 250.0), TTimeDoublePr(41400, 230.0),
        TTimeDoublePr(43200, 230.0), TTimeDoublePr(45000, 220.0),
        TTimeDoublePr(46800, 240.0), TTimeDoublePr(48600, 220.0),
        TTimeDoublePr(50400, 260.0), TTimeDoublePr(52200, 250.0),
        TTimeDoublePr(54000, 260.0), TTimeDoublePr(55800, 270.0),
        TTimeDoublePr(57600, 280.0), TTimeDoublePr(59400, 290.0),
        TTimeDoublePr(61200, 290.0), TTimeDoublePr(63000, 60.0),
        TTimeDoublePr(64800, 20.0),  TTimeDoublePr(66600, 18.0),
        TTimeDoublePr(68400, 19.0),  TTimeDoublePr(70200, 10.0),
        TTimeDoublePr(72000, 10.0),  TTimeDoublePr(73800, 5.0),
        TTimeDoublePr(75600, 5.0),   TTimeDoublePr(77400, 10.0),
        TTimeDoublePr(79200, 5.0),   TTimeDoublePr(81000, 3.0),
        TTimeDoublePr(82800, 1.0),   TTimeDoublePr(84600, 1.0),
        TTimeDoublePr(86400, 1.0)};

    test::CRandomNumbers rng;

    CTestSeasonalComponent seasonal(core::constants::DAY, 24, 0.048);

    core_t::TTime time = startTime;

    double totalError1 = 0.0;
    double totalError2 = 0.0;
    double numberErrors = 0.0;

    for (std::size_t d = 0; d < 365; ++d) {
        double scale = 2.0 + 2.0 * std::sin(3.14159265358979 * static_cast<double>(d) / 365.0);

        TTimeDoublePrVec samples;
        generateSeasonalValues(rng, function, time, time + core::constants::DAY, 100, samples);

        TDoubleVec residuals;
        rng.generateGammaSamples(10.0, 1.2, 100, residuals);
        double residualMean = maths::CBasicStatistics::mean(residuals);

        for (std::size_t i = 0; i < 100; ++i) {
            seasonal.addPoint(samples[i].first, scale * samples[i].second + residuals[i]);
        }

        LOG_TRACE(<< "Processing day = " << d);

        time += core::constants::DAY;

        seasonal.interpolate(time);

        if (seasonal.initialized()) {
            double error1 = 0.0;
            double error2 = 0.0;
            for (std::size_t j = 0; j < function.size(); ++j) {
                TDoubleDoublePr interval = seasonal.value(time + function[j].first, 70.0);
                double f = residualMean + scale * function[j].second;

                double e = mean(interval) - f;
                error1 += std::fabs(e);
                error2 += std::max(std::max(interval.first - f, f - interval.second), 0.0);
            }

            if (d > 1) {
                LOG_TRACE(<< "f(0) = " << mean(seasonal.value(time, 0.0)) << ", f(T) = "
                          << mean(seasonal.value(time + core::constants::DAY - 1, 0.0)));
                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    mean(seasonal.value(time, 0.0)),
                    mean(seasonal.value(time + core::constants::DAY - 1, 0.0)), 0.4);
            }

            error1 /= static_cast<double>(function.size());
            error2 /= static_cast<double>(function.size());
            LOG_TRACE(<< "error1 = " << error1);
            LOG_TRACE(<< "error2 = " << error2);
            BOOST_TEST_REQUIRE(error1 < 27.0);
            BOOST_TEST_REQUIRE(error2 < 19.0);
            totalError1 += error1;
            totalError2 += error2;
            numberErrors += 1.0;
        }

        seasonal.propagateForwardsByTime(1.0);
    }

    LOG_DEBUG(<< "mean error 1 = " << totalError1 / numberErrors);
    LOG_DEBUG(<< "mean error 2 = " << totalError2 / numberErrors);
    BOOST_TEST_REQUIRE(totalError1 / numberErrors < 19.0);
    BOOST_TEST_REQUIRE(totalError2 / numberErrors < 14.0);
}

BOOST_AUTO_TEST_CASE(testWindowed) {
    // Test time windowed components.

    auto trend = [](core_t::TTime time) {
        return 0.0001 * static_cast<double>(time);
    };
    auto seasonality = [](core_t::TTime time) {
        return 50.0 * std::sin(boost::math::double_constants::two_pi *
                               static_cast<double>(time) /
                               static_cast<double>(core::constants::DAY));
    };

    core_t::TTime bucketLength{core::constants::HOUR / 2};

    maths::CDiurnalTime weekendTime{0, 0, core::constants::WEEKEND, core::constants::DAY};
    maths::CDiurnalTime weekdayTime{0, core::constants::WEEKEND,
                                    core::constants::WEEK, core::constants::DAY};
    CTestSeasonalComponent weekendComponent{weekendTime, 36, 0.048,
                                            static_cast<double>(bucketLength)};
    CTestSeasonalComponent weekdayComponent{weekdayTime, 36, 0.048,
                                            static_cast<double>(bucketLength)};

    core_t::TTime time{0};
    for (/**/; time < 4 * core::constants::WEEK; time += bucketLength) {
        if (weekendTime.inWindow(time)) {
            weekendComponent.addPoint(time, trend(time) + 0.3 * seasonality(time));
            weekendComponent.propagateForwardsByTime(1.0 / 24.0);
        }
        if (weekdayTime.inWindow(time)) {
            weekdayComponent.addPoint(time, trend(time) + seasonality(time));
            weekdayComponent.propagateForwardsByTime(1.0 / 24.0);
        }
    }
    for (/**/; time < 6 * core::constants::WEEK; time += bucketLength) {
        if (weekendTime.inWindow(time)) {
            if (weekendComponent.shouldInterpolate(time)) {
                weekendComponent.interpolate(time);
                double error{0.0};
                for (core_t::TTime t = 0; t < core::constants::DAY; t += bucketLength) {
                    double rt{weekendComponent.time().regression(time + t)};
                    error += std::fabs(
                        maths::CBasicStatistics::mean(weekendComponent.value(time + t, 0.0)) -
                        weekendComponent.bucketing().regression(time + t)->predict(rt));
                }
                error /= 48.0;
                LOG_DEBUG(<< "error = " << error);
                BOOST_TEST_REQUIRE(error < 2.5);
            }
            weekendComponent.add(time, trend(time) + 0.3 * seasonality(time));
            weekendComponent.propagateForwardsByTime(1.0 / 24.0);
        }

        if (weekdayTime.inWindow(time)) {
            if (weekdayComponent.shouldInterpolate(time)) {
                weekdayComponent.interpolate(time);
                double error{0.0};
                for (core_t::TTime t = 0; t < core::constants::DAY; t += bucketLength) {
                    double rt{weekdayComponent.time().regression(time + t)};
                    error += std::fabs(
                        maths::CBasicStatistics::mean(weekdayComponent.value(time + t, 0.0)) -
                        weekdayComponent.bucketing().regression(time + t)->predict(rt));
                }
                error /= 48.0;
                LOG_DEBUG(<< "error = " << error);
                BOOST_TEST_REQUIRE(error < 2.5);
            }
            weekdayComponent.add(time, trend(time) + seasonality(time));
            weekdayComponent.propagateForwardsByTime(1.0 / 24.0);
        }
    }
}

BOOST_AUTO_TEST_CASE(testVeryLowVariation) {
    // Test we very accurately fit low variation data.

    const core_t::TTime startTime = 1354492800;

    TTimeDoublePrVec function;
    for (std::size_t i = 0; i < 25; ++i) {
        function.emplace_back((i * core::constants::DAY) / 24, 50.0);
    }

    test::CRandomNumbers rng;

    std::size_t n = 5000;

    TTimeDoublePrVec samples;
    generateSeasonalValues(rng, function, startTime,
                           startTime + 31 * core::constants::DAY, n, samples);

    TDoubleVec residuals;
    rng.generateNormalSamples(0.0, 1e-3, n, residuals);
    double residualMean = maths::CBasicStatistics::mean(residuals);

    double deviation = std::sqrt(1e-3);

    CTestSeasonalComponent seasonal(core::constants::DAY, 24);

    double totalError1 = 0.0;
    double totalError2 = 0.0;
    core_t::TTime time = startTime;
    for (std::size_t i = 0u, d = 0; i < n; ++i) {
        seasonal.addPoint(samples[i].first, samples[i].second + residuals[i]);

        if (samples[i].first >= time + core::constants::DAY) {
            LOG_TRACE(<< "Processing day = " << ++d);

            time += core::constants::DAY;

            double error1 = 0.0;
            double error2 = 0.0;
            for (std::size_t j = 0; j < function.size(); ++j) {
                TDoubleDoublePr interval = seasonal.value(time + function[j].first, 70.0);
                double f = residualMean + function[j].second;

                double e = mean(interval) - f;
                error1 += std::fabs(e);
                error2 += std::max(std::max(interval.first - f, f - interval.second), 0.0);
            }

            if (d > 1) {
                LOG_TRACE(<< "f(0) = " << mean(seasonal.value(time, 0.0)) << ", f(T) = "
                          << mean(seasonal.value(time + core::constants::DAY - 1, 0.0)));
                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    mean(seasonal.value(time, 0.0)),
                    mean(seasonal.value(time + core::constants::DAY - 1, 0.0)), 0.1);
            }
            error1 /= static_cast<double>(function.size());
            error2 /= static_cast<double>(function.size());
            LOG_TRACE(<< "deviation = " << deviation);
            LOG_TRACE(<< "error1 = " << error1 << ", error2 = " << error2);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(0.0, error1, 1.0 * deviation);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(0.0, error2, 0.1 * deviation);
            totalError1 += error1;
            totalError2 += error2;
        }
    }

    totalError1 /= 30.0;
    totalError2 /= 30.0;
    LOG_DEBUG(<< "deviation = " << deviation);
    LOG_DEBUG(<< "totalError1 = " << totalError1 << ", totalError2 = " << totalError2);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(totalError1, 0.0, 0.20 * deviation);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(totalError2, 0.0, 0.04 * deviation);
}

BOOST_AUTO_TEST_CASE(testVariance) {
    // Check that we estimate a periodic variance.

    test::CRandomNumbers rng;

    TTimeDoublePrVec function;
    for (core_t::TTime i = 0; i < 481; ++i) {
        core_t::TTime t = (i * core::constants::DAY) / 48;
        double vt = 80.0 + 20.0 * std::sin(boost::math::double_constants::two_pi *
                                           static_cast<double>(i % 48) / 48.0);
        TDoubleVec sample;
        rng.generateNormalSamples(0.0, vt, 10, sample);
        for (std::size_t j = 0; j < sample.size(); ++j) {
            function.emplace_back(t, sample[j]);
        }
    }

    CTestSeasonalComponent seasonal(core::constants::DAY, 24);

    for (std::size_t i = 0; i < function.size(); ++i) {
        seasonal.addPoint(function[i].first, function[i].second);
    }

    TMeanAccumulator error;
    for (core_t::TTime i = 0; i < 48; ++i) {
        core_t::TTime t = (i * core::constants::DAY) / 48;
        double v_ = 80.0 + 20.0 * std::sin(boost::math::double_constants::two_pi *
                                           static_cast<double>(i) / 48.0);
        TDoubleDoublePr vv = seasonal.variance(t, 98.0);
        double v = (vv.first + vv.second) / 2.0;
        LOG_TRACE(<< "v_ = " << v_ << ", v = " << core::CContainerPrinter::print(vv)
                  << ", relative error = " << std::fabs(v - v_) / v_);

        BOOST_REQUIRE_CLOSE_ABSOLUTE(v_, v, 0.5 * v_);
        BOOST_TEST_REQUIRE(v_ > vv.first);
        BOOST_TEST_REQUIRE(v_ < vv.second);
        error.add(std::fabs(v - v_) / v_);
    }

    LOG_DEBUG(<< "mean relative error = " << maths::CBasicStatistics::mean(error));
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(error) < 0.12);
}

BOOST_AUTO_TEST_CASE(testPersist) {
    // Check that persistence is idempotent.

    const core_t::TTime startTime = 1354492800;
    const core_t::TTime minute = 60;
    const double decayRate = 0.001;
    const double minimumBucketLength = 0.0;

    test::CRandomNumbers rng;

    TTimeDoublePrVec function;
    for (core_t::TTime i = 0; i < 49; ++i) {
        core_t::TTime t = (i * core::constants::DAY) / 48;
        double ft = 100.0 + 40.0 * std::sin(boost::math::double_constants::two_pi *
                                            static_cast<double>(i) / 48.0);
        function.emplace_back(t, ft);
    }

    std::size_t n = 3300;

    TTimeDoublePrVec samples;
    generateSeasonalValues(rng, function, startTime,
                           startTime + 31 * core::constants::DAY, n, samples);

    TDoubleVec residuals;
    rng.generateGammaSamples(10.0, 1.2, n, residuals);

    CTestSeasonalComponent origSeasonal(core::constants::DAY, 24, decayRate);

    for (std::size_t i = 0; i < n; ++i) {
        origSeasonal.addPoint(samples[i].first, samples[i].second + residuals[i]);
    }

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origSeasonal.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_DEBUG(<< "seasonal component XML representation:\n" << origXml);

    // Restore the XML into a new filter
    core::CRapidXmlParser parser;
    BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    maths::CSeasonalComponent restoredSeasonal(decayRate, minimumBucketLength, traverser);

    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredSeasonal.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    BOOST_REQUIRE_EQUAL(origXml, newXml);

    // Test that the values and variances of the original and
    // restored components are similar.
    for (core_t::TTime time = 0; time < core::constants::DAY; time += minute) {
        TDoubleDoublePr xo = origSeasonal.value(time, 80.0);
        TDoubleDoublePr xn = restoredSeasonal.value(time, 80.0);
        if (time % (15 * minute) == 0) {
            LOG_DEBUG(<< "xo = " << core::CContainerPrinter::print(xo)
                      << ", xn = " << core::CContainerPrinter::print(xn));
        }
        BOOST_REQUIRE_CLOSE_ABSOLUTE(xo.first, xn.first, 0.3);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(xo.second, xn.second, 0.3);
        TDoubleDoublePr vo = origSeasonal.variance(time, 80.0);
        TDoubleDoublePr vn = origSeasonal.variance(time, 80.0);
        if (time % (15 * minute) == 0) {
            LOG_DEBUG(<< "vo = " << core::CContainerPrinter::print(vo)
                      << ", vn = " << core::CContainerPrinter::print(vn));
        }
        BOOST_REQUIRE_CLOSE_ABSOLUTE(vo.first, vn.first, 1e-3);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(vo.second, vn.second, 1e-3);
    }
}

BOOST_AUTO_TEST_SUITE_END()
