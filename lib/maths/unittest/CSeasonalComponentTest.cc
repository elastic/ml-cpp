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

class CTestSeasonalComponent : public maths::CSeasonalComponent {
public:
    // Bring base class method hidden by the signature above into scope
    using maths::CSeasonalComponent::initialize;

public:
    CTestSeasonalComponent(
        core_t::TTime startTime,
        core_t::TTime period,
        std::size_t space,
        double decayRate = 0.0,
        double minimumBucketLength = 0.0,
        maths::CSplineTypes::EBoundaryCondition boundaryCondition = maths::CSplineTypes::E_Periodic,
        maths::CSplineTypes::EType valueInterpolationType = maths::CSplineTypes::E_Cubic,
        maths::CSplineTypes::EType varianceInterpolationType = maths::CSplineTypes::E_Linear)
        : maths::CSeasonalComponent(maths::CDiurnalTime(0, 0, core::constants::WEEK, period),
                                    space,
                                    decayRate,
                                    minimumBucketLength,
                                    boundaryCondition,
                                    valueInterpolationType,
                                    varianceInterpolationType),
          m_StartTime(startTime) {}

    void addPoint(core_t::TTime time, double value, double weight = 1.0) {
        core_t::TTime period = this->time().period();
        if (time > m_StartTime + period) {
            this->updateStartOfCurrentPeriodAndInterpolate(time);
        }
        this->maths::CSeasonalComponent::add(time, value, weight);
    }

    void updateStartOfCurrentPeriodAndInterpolate(core_t::TTime time) {
        core_t::TTime period = this->time().period();
        this->interpolate(maths::CIntegerTools::floor(time, period));
        m_StartTime = maths::CIntegerTools::floor(time, period);
    }

private:
    core_t::TTime m_StartTime;
};

void generateSeasonalValues(test::CRandomNumbers& rng,
                            const TTimeDoublePrVec& function,
                            core_t::TTime startTime,
                            core_t::TTime endTime,
                            std::size_t numberSamples,
                            TTimeDoublePrVec& samples) {
    using TSizeVec = std::vector<std::size_t>;

    // Generate time uniformly at random in the interval
    // [startTime, endTime).

    core_t::TTime period = function[function.size() - 1].first;

    TSizeVec times;
    rng.generateUniformSamples(static_cast<std::size_t>(startTime),
                               static_cast<std::size_t>(endTime), numberSamples, times);
    std::sort(times.begin(), times.end());
    for (std::size_t i = 0u; i < times.size(); ++i) {
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

BOOST_AUTO_TEST_CASE(testNoPeriodicity) {
    const core_t::TTime startTime = 1354492800;

    TTimeDoublePrVec function;
    for (std::size_t i = 0; i < 25; ++i) {
        function.emplace_back((i * core::constants::DAY) / 24, 0.0);
    }

    test::CRandomNumbers rng;

    // Test that the seasonal component tends to flat if the
    // values have no seasonal component.
    std::size_t n = 5000u;

    TTimeDoublePrVec samples;
    generateSeasonalValues(rng, function, startTime,
                           startTime + 31 * core::constants::DAY, n, samples);

    TDoubleVec residuals;
    rng.generateGammaSamples(10.0, 1.2, n, residuals);
    double residualMean = maths::CBasicStatistics::mean(residuals);

    CTestSeasonalComponent seasonal(startTime, core::constants::DAY, 24);
    seasonal.initialize();

    //std::ofstream file;
    //file.open("results.m");
    //file << "hold on;\n";

    double totalError1 = 0.0;
    double totalError2 = 0.0;
    core_t::TTime time = startTime;
    for (std::size_t i = 0u, d = 0u; i < n; ++i) {
        seasonal.addPoint(samples[i].first, samples[i].second + residuals[i]);

        if (samples[i].first >= time + core::constants::DAY) {
            LOG_TRACE(<< "Processing day = " << ++d);

            time += core::constants::DAY;

            //std::ostringstream t;
            //std::ostringstream ft;
            //t << "t = [";
            //ft << "ft = [";
            double error1 = 0.0;
            double error2 = 0.0;
            for (std::size_t j = 0u; j < function.size(); ++j) {
                //t << time + function[j].first << " ";
                //ft << function[j].second << " ";
                TDoubleDoublePr interval = seasonal.value(time + function[j].first, 70.0);
                double f = function[j].second + residualMean;

                double e = mean(interval) - f;
                error1 += std::fabs(e);
                error2 += std::max(std::max(interval.first - f, f - interval.second), 0.0);
            }
            //t << "];\n";
            //ft << "];\n";

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

            //file << seasonal.print() << "\n";
            //file << t.str();
            //file << ft.str();
            //file << "plot(t, ft, 'r');\n";
            //file << "input(\"Hit any key for next period\");\n\n";
        }
    }

    totalError1 /= 30.0;
    totalError2 /= 30.0;
    LOG_DEBUG(<< "totalError1 = " << totalError1);
    LOG_DEBUG(<< "totalError2 = " << totalError2);
    BOOST_TEST_REQUIRE(totalError1 < 0.6);
    BOOST_TEST_REQUIRE(totalError2 < 0.15);
}

BOOST_AUTO_TEST_CASE(testConstantPeriodic) {
    const core_t::TTime startTime = 1354492800;

    test::CRandomNumbers rng;

    // Test smooth.
    {
        LOG_DEBUG(<< "*** sin(2 * pi * t / 24 hrs) ***");

        TTimeDoublePrVec function;
        for (core_t::TTime i = 0u; i < 49; ++i) {
            core_t::TTime t = (i * core::constants::DAY) / 48;
            double ft = 100.0 + 40.0 * std::sin(boost::math::double_constants::two_pi *
                                                static_cast<double>(i) / 48.0);
            function.emplace_back(t, ft);
        }

        std::size_t n = 5000u;

        TTimeDoublePrVec samples;
        generateSeasonalValues(rng, function, startTime,
                               startTime + 31 * core::constants::DAY, n, samples);

        TDoubleVec residuals;
        rng.generateGammaSamples(10.0, 1.2, n, residuals);
        double residualMean = maths::CBasicStatistics::mean(residuals);

        CTestSeasonalComponent seasonal(startTime, core::constants::DAY, 24, 0.01);
        seasonal.initialize();

        //std::ofstream file;
        //file.open("results.m");
        //file << "hold on;\n";

        double totalError1 = 0.0;
        double totalError2 = 0.0;
        core_t::TTime time = startTime;
        for (std::size_t i = 0u, d = 0u; i < n; ++i) {
            seasonal.addPoint(samples[i].first, samples[i].second + residuals[i]);

            if (samples[i].first >= time + core::constants::DAY) {
                LOG_TRACE(<< "Processing day = " << ++d);

                time += core::constants::DAY;

                //std::ostringstream t;
                //std::ostringstream ft;
                //t << "t = [";
                //ft << "ft = [";
                double error1 = 0.0;
                double error2 = 0.0;
                for (std::size_t j = 0u; j < function.size(); ++j) {
                    //t << time + function[j].first << " ";
                    //ft << function[j].second << " ";
                    TDoubleDoublePr interval = seasonal.value(time + function[j].first, 70.0);
                    double f = residualMean + function[j].second;
                    double e = mean(interval) - f;
                    error1 += std::fabs(e);
                    error2 += std::max(std::max(interval.first - f, f - interval.second), 0.0);
                }
                //t << "];\n";
                //ft << "];\n";

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

                //file << seasonal.print() << "\n";
                //file << t.str();
                //file << ft.str();
                //file << "plot(t, ft, 'r');\n";
                //file << "input(\"Hit any key for next period\");\n\n";

                seasonal.propagateForwardsByTime(1.0);
            }
        }

        totalError1 /= 30.0;
        totalError2 /= 30.0;
        LOG_DEBUG(<< "totalError1 = " << totalError1);
        LOG_DEBUG(<< "totalError2 = " << totalError2);
        BOOST_TEST_REQUIRE(totalError1 < 0.42);
        BOOST_TEST_REQUIRE(totalError2 < 0.01);
    }

    // Test high slope.
    {
        LOG_DEBUG(<< "*** piecewise linear ***");

        TTimeDoublePr knotPoints[] = {
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

        TTimeDoublePrVec function(std::begin(knotPoints), std::end(knotPoints));

        std::size_t n = 6000u;

        TTimeDoublePrVec samples;
        generateSeasonalValues(rng, function, startTime,
                               startTime + 41 * core::constants::DAY, n, samples);

        TDoubleVec residuals;
        rng.generateGammaSamples(10.0, 1.2, n, residuals);
        double residualMean = maths::CBasicStatistics::mean(residuals);

        CTestSeasonalComponent seasonal(startTime, core::constants::DAY, 24, 0.01);
        seasonal.initialize();

        //std::ofstream file;
        //file.open("results.m");
        //file << "hold on;\n";

        double totalError1 = 0.0;
        double totalError2 = 0.0;
        core_t::TTime time = startTime;
        for (std::size_t i = 0u, d = 0u; i < n; ++i) {
            seasonal.addPoint(samples[i].first, samples[i].second + residuals[i]);

            if (samples[i].first >= time + core::constants::DAY) {
                LOG_TRACE(<< "Processing day = " << ++d);

                time += core::constants::DAY;

                //std::ostringstream t;
                //std::ostringstream ft;
                //t << "t = [";
                //ft << "ft = [";
                double error1 = 0.0;
                double error2 = 0.0;
                for (std::size_t j = 0u; j < function.size(); ++j) {
                    //t << time + function[j].first << " ";
                    //ft << function[j].second << " ";
                    TDoubleDoublePr interval = seasonal.value(time + function[j].first, 70.0);
                    double f = residualMean + function[j].second;

                    double e = mean(interval) - f;
                    error1 += std::fabs(e);
                    error2 += std::max(std::max(interval.first - f, f - interval.second), 0.0);
                }
                //t << "];\n";
                //ft << "];\n";

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

                //file << seasonal.print() << "\n";
                //file << t.str();
                //file << ft.str();
                //file << "plot(t, ft, 'r');\n";
                //file << "input(\"Hit any key for next period\");\n\n";

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
    // Test a signal with periodicity which changes slowly
    // over time.

    core_t::TTime startTime = 0;

    TTimeDoublePr knotPoints[] = {
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

    TTimeDoublePrVec function(std::begin(knotPoints), std::end(knotPoints));

    test::CRandomNumbers rng;

    CTestSeasonalComponent seasonal(startTime, core::constants::DAY, 24, 0.048);
    seasonal.initialize();

    core_t::TTime time = startTime;

    double totalError1 = 0.0;
    double totalError2 = 0.0;
    double numberErrors = 0.0;

    for (std::size_t d = 0u; d < 365; ++d) {
        double scale = 2.0 + 2.0 * std::sin(3.14159265358979 * static_cast<double>(d) / 365.0);

        TTimeDoublePrVec samples;
        generateSeasonalValues(rng, function, time, time + core::constants::DAY, 100, samples);

        TDoubleVec residuals;
        rng.generateGammaSamples(10.0, 1.2, 100, residuals);
        double residualMean = maths::CBasicStatistics::mean(residuals);

        for (std::size_t i = 0u; i < 100; ++i) {
            seasonal.addPoint(samples[i].first, scale * samples[i].second + residuals[i]);
        }

        LOG_TRACE(<< "Processing day = " << d);

        time += core::constants::DAY;

        seasonal.updateStartOfCurrentPeriodAndInterpolate(time);

        if (seasonal.initialized()) {
            //std::ostringstream t;
            //std::ostringstream ft;
            //t << "t = [";
            //ft << "ft = [";
            double error1 = 0.0;
            double error2 = 0.0;
            for (std::size_t j = 0u; j < function.size(); ++j) {
                //t << time + function[j].first << " ";
                //ft << function[j].second << " ";
                TDoubleDoublePr interval = seasonal.value(time + function[j].first, 70.0);
                double f = residualMean + scale * function[j].second;

                double e = mean(interval) - f;
                error1 += std::fabs(e);
                error2 += std::max(std::max(interval.first - f, f - interval.second), 0.0);
            }
            //t << "];\n";
            //ft << "];\n";

            if (d > 1) {
                LOG_TRACE(<< "f(0) = " << mean(seasonal.value(time, 0.0)) << ", f(T) = "
                          << mean(seasonal.value(time + core::constants::DAY - 1, 0.0)));
                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    mean(seasonal.value(time, 0.0)),
                    mean(seasonal.value(time + core::constants::DAY - 1, 0.0)), 0.2);
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

            //file << seasonal.print() << "\n";
            //file << t.str();
            //file << ft.str();
            //file << "plot(t, ft, 'r');\n";
            //file << "input(\"Hit any key for next period\");\n\n";
        }

        seasonal.propagateForwardsByTime(1.0);
    }

    LOG_DEBUG(<< "mean error 1 = " << totalError1 / numberErrors);
    LOG_DEBUG(<< "mean error 2 = " << totalError2 / numberErrors);
    BOOST_TEST_REQUIRE(totalError1 / numberErrors < 18.0);
    BOOST_TEST_REQUIRE(totalError2 / numberErrors < 14.0);
}

BOOST_AUTO_TEST_CASE(testVeryLowVariation) {
    // Test we very accurately fit low variation data.

    const core_t::TTime startTime = 1354492800;

    TTimeDoublePrVec function;
    for (std::size_t i = 0u; i < 25; ++i) {
        function.emplace_back((i * core::constants::DAY) / 24, 50.0);
    }

    test::CRandomNumbers rng;

    std::size_t n = 5000u;

    TTimeDoublePrVec samples;
    generateSeasonalValues(rng, function, startTime,
                           startTime + 31 * core::constants::DAY, n, samples);

    TDoubleVec residuals;
    rng.generateNormalSamples(0.0, 1e-3, n, residuals);
    double residualMean = maths::CBasicStatistics::mean(residuals);

    double deviation = std::sqrt(1e-3);

    CTestSeasonalComponent seasonal(startTime, core::constants::DAY, 24);
    seasonal.initialize(startTime);

    //std::ofstream file;
    //file.open("results.m");
    //file << "hold on;\n";

    double totalError1 = 0.0;
    double totalError2 = 0.0;
    core_t::TTime time = startTime;
    for (std::size_t i = 0u, d = 0u; i < n; ++i) {
        seasonal.addPoint(samples[i].first, samples[i].second + residuals[i]);

        if (samples[i].first >= time + core::constants::DAY) {
            LOG_TRACE(<< "Processing day = " << ++d);

            time += core::constants::DAY;

            //std::ostringstream t;
            //std::ostringstream ft;
            //t << "t = [";
            //ft << "ft = [";
            double error1 = 0.0;
            double error2 = 0.0;
            for (std::size_t j = 0u; j < function.size(); ++j) {
                //t << time + function[j].first << " ";
                //ft << function[j].second << " ";
                TDoubleDoublePr interval = seasonal.value(time + function[j].first, 70.0);
                double f = residualMean + function[j].second;

                double e = mean(interval) - f;
                error1 += std::fabs(e);
                error2 += std::max(std::max(interval.first - f, f - interval.second), 0.0);
            }
            //t << "];\n";
            //ft << "];\n";

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

            //file << seasonal.print() << "\n";
            //file << t.str();
            //file << ft.str();
            //file << "plot(t, ft, 'r');\n";
            //file << "input(\"Hit any key for next period\");\n\n";
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
    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

    // Check that we estimate a periodic variance.

    test::CRandomNumbers rng;

    TTimeDoublePrVec function;
    for (core_t::TTime i = 0u; i < 481; ++i) {
        core_t::TTime t = (i * core::constants::DAY) / 48;
        double vt = 80.0 + 20.0 * std::sin(boost::math::double_constants::two_pi *
                                           static_cast<double>(i % 48) / 48.0);
        TDoubleVec sample;
        rng.generateNormalSamples(0.0, vt, 10, sample);
        for (std::size_t j = 0u; j < sample.size(); ++j) {
            function.emplace_back(t, sample[j]);
        }
    }

    CTestSeasonalComponent seasonal(0, core::constants::DAY, 24);
    seasonal.initialize(0);

    for (std::size_t i = 0u; i < function.size(); ++i) {
        seasonal.addPoint(function[i].first, function[i].second);
    }

    TMeanAccumulator error;
    for (core_t::TTime i = 0u; i < 48; ++i) {
        core_t::TTime t = (i * core::constants::DAY) / 48;
        double v_ = 80.0 + 20.0 * std::sin(boost::math::double_constants::two_pi *
                                           static_cast<double>(i) / 48.0);
        TDoubleDoublePr vv = seasonal.variance(t, 98.0);
        double v = (vv.first + vv.second) / 2.0;
        LOG_TRACE(<< "v_ = " << v_ << ", v = " << core::CContainerPrinter::print(vv)
                  << ", relative error = " << std::fabs(v - v_) / v_);

        BOOST_REQUIRE_CLOSE_ABSOLUTE(v_, v, 0.4 * v_);
        BOOST_TEST_REQUIRE(v_ > vv.first);
        BOOST_TEST_REQUIRE(v_ < vv.second);
        error.add(std::fabs(v - v_) / v_);
    }

    LOG_DEBUG(<< "mean relative error = " << maths::CBasicStatistics::mean(error));
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(error) < 0.11);
}

BOOST_AUTO_TEST_CASE(testPersist) {
    // Check that persistence is idempotent.

    const core_t::TTime startTime = 1354492800;
    const core_t::TTime minute = 60;
    const double decayRate = 0.001;
    const double minimumBucketLength = 0.0;

    test::CRandomNumbers rng;

    TTimeDoublePrVec function;
    for (core_t::TTime i = 0u; i < 49; ++i) {
        core_t::TTime t = (i * core::constants::DAY) / 48;
        double ft = 100.0 + 40.0 * std::sin(boost::math::double_constants::two_pi *
                                            static_cast<double>(i) / 48.0);
        function.emplace_back(t, ft);
    }

    std::size_t n = 3300u;

    TTimeDoublePrVec samples;
    generateSeasonalValues(rng, function, startTime,
                           startTime + 31 * core::constants::DAY, n, samples);

    TDoubleVec residuals;
    rng.generateGammaSamples(10.0, 1.2, n, residuals);

    CTestSeasonalComponent origSeasonal(startTime, core::constants::DAY, 24, decayRate);
    origSeasonal.initialize(startTime);

    for (std::size_t i = 0u; i < n; ++i) {
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
