/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CStringUtils.h>
#include <core/Constants.h>
#include <core/CoreTypes.h>

#include <maths/CTimeSeriesSegmentation.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include "TestUtils.h"

#include <boost/test/unit_test.hpp>

#include <fstream>
#include <vector>

BOOST_AUTO_TEST_SUITE(CTimeSeriesSegmentationTest)

using namespace ml;

using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;
using TFloatMeanAccumulator =
    maths::CBasicStatistics::SSampleMean<maths::CFloatStorage>::TAccumulator;
using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
using TMeanVarAccumulator = maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
using TSegmentation = maths::CTimeSeriesSegmentation;

namespace {
class CDebugGenerator {
public:
    static const bool ENABLED{false};

public:
    CDebugGenerator(const std::string& file = "results.m") : m_File(file) {}
    ~CDebugGenerator() {
        if (ENABLED) {
            std::ofstream file;
            file.open(m_File);
            file << "f = " << core::CContainerPrinter::print(m_Values) << ";\n";
            file << "r = " << core::CContainerPrinter::print(m_Residuals) << ";\n";
            file << "figure(1);\n";
            file << "clf;\n";
            file << "hold on;\n";
            file << "plot(f);\n";
            file << "axis([1 length(f) min(f) max(f)]);\n";
            file << "figure(2);\n";
            file << "clf;\n";
            file << "plot(r, 'k');\n";
            file << "axis([1 length(r) min(r) max(r)]);";
        }
    }
    void addValue(double value) {
        if (ENABLED) {
            m_Values.push_back(value);
        }
    }
    void addResidual(double residual) {
        if (ENABLED) {
            m_Residuals.push_back(residual);
        }
    }

private:
    std::string m_File;
    TDoubleVec m_Values;
    TDoubleVec m_Residuals;
};

std::size_t distance(const TSizeVec& lhs, const TSizeVec& rhs) {
    std::size_t distance{0};
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        distance += std::max(lhs[i], rhs[i]) - std::min(lhs[i], rhs[i]);
    }
    return distance;
}
}

BOOST_AUTO_TEST_CASE(testPiecewiseLinear) {

    core_t::TTime halfHour{core::constants::HOUR / 2};
    core_t::TTime week{core::constants::WEEK};
    core_t::TTime range{5 * week};

    test::CRandomNumbers rng;

    TFloatMeanAccumulatorVec values(range / halfHour);
    TDoubleVec noise;

    LOG_DEBUG(<< "Basic");
    for (auto outlierFraction : {0.0, 0.1}) {
        CDebugGenerator debug(
            "results.m." + core::CStringUtils::typeToStringPretty(outlierFraction));

        values.assign(range / halfHour, TFloatMeanAccumulator{});
        TMeanVarAccumulator noiseMoments;
        for (core_t::TTime time = 0; time < range; time += halfHour) {
            rng.generateNormalSamples(0.0, 3.0, 1, noise);
            noiseMoments.add(noise[0]);
            if (time < 2 * week) {
                values[time / halfHour].add(3.0 + 200.0 * ramp(time) + noise[0]);
            } else if (time < 3 * week) {
                values[time / halfHour].add(20.0 - 100.0 * ramp(time) + noise[0]);
            } else {
                values[time / halfHour].add(50.0 * ramp(time) - 25.0 + noise[0]);
            }
            debug.addValue(maths::CBasicStatistics::mean(values[time / halfHour]));
        }
        TSizeVec trueSegmentation{0, static_cast<std::size_t>(2 * week / halfHour),
                                  static_cast<std::size_t>(3 * week / halfHour),
                                  values.size()};

        TSizeVec segmentation(
            TSegmentation::piecewiseLinear(values, 0.01, outlierFraction, 0.1));
        LOG_DEBUG(<< "true segmentation = "
                  << core::CContainerPrinter::print(trueSegmentation));
        LOG_DEBUG(<< "segmentation      = " << core::CContainerPrinter::print(segmentation));

        TFloatMeanAccumulatorVec residuals{TSegmentation::removePiecewiseLinear(
            values, segmentation, outlierFraction, 0.1)};
        TMeanVarAccumulator residualMoments;
        for (const auto& residual : residuals) {
            residualMoments.add(maths::CBasicStatistics::mean(residual));
            debug.addResidual(maths::CBasicStatistics::mean(residual));
        }
        LOG_DEBUG(<< "noise moments    = " << noiseMoments);
        LOG_DEBUG(<< "residual moments = " << residualMoments);

        // No false positives.
        BOOST_REQUIRE_EQUAL(trueSegmentation.size(), segmentation.size());

        // Distance in index space is small.
        BOOST_TEST_REQUIRE(distance(trueSegmentation, segmentation) < 35);

        // Not biased.
        BOOST_TEST_REQUIRE(
            std::fabs(maths::CBasicStatistics::mean(residualMoments) -
                      maths::CBasicStatistics::mean(noiseMoments)) <
            2.0 * std::sqrt(maths::CBasicStatistics::variance(noiseMoments) /
                            maths::CBasicStatistics::count(noiseMoments)));

        // We've explained nearly all the variance.
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::variance(residualMoments) <
                           1.4 * maths::CBasicStatistics::variance(noiseMoments));
    }

    LOG_DEBUG(<< "With Outliers");

    // Same again but with 5% salt-and-pepper outliers.

    CDebugGenerator debug("results.m.outliers");

    values.assign(range / halfHour, TFloatMeanAccumulator{});
    TDoubleVec u01;
    TSizeVec inliers;
    TMeanVarAccumulator noiseMoments;
    for (core_t::TTime time = 0; time < range; time += halfHour) {
        rng.generateUniformSamples(0.0, 1.0, 1, u01);
        if (u01[0] < 0.05) {
            values[time / halfHour].add(u01[0] < 0.025 ? 0.0 : 50.0);
        } else {
            rng.generateNormalSamples(0.0, 3.0, 1, noise);
            noiseMoments.add(noise[0]);
            if (time < 2 * week) {
                values[time / halfHour].add(3.0 + 200.0 * ramp(time) + noise[0]);
            } else if (time < 3 * week) {
                values[time / halfHour].add(20.0 - 100.0 * ramp(time) + noise[0]);
            } else {
                values[time / halfHour].add(50.0 * ramp(time) - 25.0 + noise[0]);
            }
            inliers.push_back(time / halfHour);
        }
        debug.addValue(maths::CBasicStatistics::mean(values[time / halfHour]));
    }
    TSizeVec trueSegmentation{0, static_cast<std::size_t>(2 * week / halfHour),
                              static_cast<std::size_t>(3 * week / halfHour),
                              values.size()};

    TSizeVec segmentation(TSegmentation::piecewiseLinear(values, 0.01, 0.05, 0.1));
    LOG_DEBUG(<< "true segmentation = " << core::CContainerPrinter::print(trueSegmentation));
    LOG_DEBUG(<< "segmentation      = " << core::CContainerPrinter::print(segmentation));

    TFloatMeanAccumulatorVec residuals{
        TSegmentation::removePiecewiseLinear(values, segmentation, 0.05, 0.1)};

    // Project onto inliers.
    TMeanVarAccumulator residualMoments;
    for (auto i : inliers) {
        residualMoments.add(maths::CBasicStatistics::mean(residuals[i]));
        debug.addResidual(maths::CBasicStatistics::mean(residuals[i]));
    }
    LOG_DEBUG(<< "noise moments    = " << noiseMoments);
    LOG_DEBUG(<< "residual moments = " << residualMoments);

    // No false positives
    BOOST_REQUIRE_EQUAL(trueSegmentation.size(), segmentation.size());

    // Distance in index space is small.
    BOOST_TEST_REQUIRE(distance(trueSegmentation, segmentation) < 35);

    // Not biased.
    BOOST_TEST_REQUIRE(std::fabs(maths::CBasicStatistics::mean(residualMoments) -
                                 maths::CBasicStatistics::mean(noiseMoments)) <
                       3.0 * std::sqrt(maths::CBasicStatistics::variance(noiseMoments) /
                                       maths::CBasicStatistics::count(noiseMoments)));

    // We've explained nearly all the variance.
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::variance(residualMoments) <
                       1.4 * maths::CBasicStatistics::variance(noiseMoments));
}

BOOST_AUTO_TEST_CASE(testPiecewiseLinearScaledPeriodic) {

    core_t::TTime halfHour{core::constants::HOUR / 2};
    core_t::TTime week{core::constants::WEEK};
    core_t::TTime range{5 * week};

    test::CRandomNumbers rng;

    std::size_t period{48};
    std::string periods[]{"smooth", "spikey"};
    TFloatMeanAccumulatorVec values(range / halfHour);
    TDoubleVec noise;

    LOG_DEBUG(<< "Basic");
    for (auto outlierFraction : {0.0, 0.1}) {
        std::size_t j{0};
        for (auto periodic : {smoothDaily, spikeyDaily}) {
            LOG_DEBUG(<< periods[j]);
            CDebugGenerator debug("results.m." +
                                  core::CStringUtils::typeToStringPretty(outlierFraction) +
                                  "." + periods[j++]);

            values.assign(range / halfHour, TFloatMeanAccumulator{});
            TMeanVarAccumulator noiseMoments;
            for (core_t::TTime time = 0; time < range; time += halfHour) {
                rng.generateNormalSamples(0.0, 3.0, 1, noise);
                noiseMoments.add(noise[0]);
                if (time < 3 * week / 2) {
                    values[time / halfHour].add(100.0 * periodic(time) + noise[0]);
                } else if (time < 2 * week) {
                    values[time / halfHour].add(50.0 * periodic(time) + noise[0]);
                } else {
                    values[time / halfHour].add(300.0 * periodic(time) + noise[0]);
                }
                debug.addValue(maths::CBasicStatistics::mean(values[time / halfHour]));
            }
            TSizeVec trueSegmentation{
                0, static_cast<std::size_t>(3 * week / halfHour / 2),
                static_cast<std::size_t>(2 * week / halfHour), values.size()};

            TSizeVec segmentation(TSegmentation::piecewiseLinearScaledPeriodic(
                values, period, 0.01, outlierFraction, 0.1));
            LOG_DEBUG(<< "true segmentation = "
                      << core::CContainerPrinter::print(trueSegmentation));
            LOG_DEBUG(<< "segmentation      = "
                      << core::CContainerPrinter::print(segmentation));

            TFloatMeanAccumulatorVec residuals{TSegmentation::removePiecewiseLinearScaledPeriodic(
                values, period, segmentation, outlierFraction, 0.1)};
            TMeanVarAccumulator residualMoments;
            for (const auto& residual : residuals) {
                residualMoments.add(maths::CBasicStatistics::mean(residual));
                debug.addResidual(maths::CBasicStatistics::mean(residual));
            }
            LOG_DEBUG(<< "noise moments    = " << noiseMoments);
            LOG_DEBUG(<< "residual moments = " << residualMoments);

            // No false positives.
            BOOST_REQUIRE_EQUAL(trueSegmentation.size(), segmentation.size());

            // Distance in index space is small.
            BOOST_TEST_REQUIRE(distance(trueSegmentation, segmentation) < 5);

            // Not biased.
            BOOST_TEST_REQUIRE(
                std::fabs(maths::CBasicStatistics::mean(residualMoments) -
                          maths::CBasicStatistics::mean(noiseMoments)) <
                3.0 * std::sqrt(maths::CBasicStatistics::variance(noiseMoments) /
                                maths::CBasicStatistics::count(noiseMoments)));

            // We've explained nearly all the variance.
            BOOST_TEST_REQUIRE(maths::CBasicStatistics::variance(residualMoments) <
                               1.4 * maths::CBasicStatistics::variance(noiseMoments));
        }
    }

    LOG_DEBUG(<< "With Outliers");

    // Same again but with 5% salt-and-pepper outliers.

    std::size_t j{0};
    for (auto periodic : {smoothDaily, spikeyDaily}) {
        LOG_DEBUG(<< periods[j]);
        CDebugGenerator debug("results.m.outliers." + periods[j++]);
        values.assign(range / halfHour, TFloatMeanAccumulator{});
        TDoubleVec u01;
        TSizeVec inliers;
        TMeanVarAccumulator noiseMoments;
        for (core_t::TTime time = 0; time < range; time += halfHour) {
            rng.generateUniformSamples(0.0, 1.0, 1, u01);
            if (u01[0] < 0.05) {
                values[time / halfHour].add(u01[0] < 0.025 ? -300.0 : 300.0);
            } else {
                rng.generateNormalSamples(0.0, 3.0, 1, noise);
                noiseMoments.add(noise[0]);
                if (time < 3 * week / 2) {
                    values[time / halfHour].add(100.0 * periodic(time) + noise[0]);
                } else if (time < 2 * week) {
                    values[time / halfHour].add(50.0 * periodic(time) + noise[0]);
                } else {
                    values[time / halfHour].add(300.0 * periodic(time) + noise[0]);
                }
                inliers.push_back(time / halfHour);
            }
            debug.addValue(maths::CBasicStatistics::mean(values[time / halfHour]));
        }
        TSizeVec trueSegmentation{0, static_cast<std::size_t>(3 * week / halfHour / 2),
                                  static_cast<std::size_t>(2 * week / halfHour),
                                  values.size()};

        TSizeVec segmentation(TSegmentation::piecewiseLinearScaledPeriodic(
            values, period, 0.01, 0.05, 0.01));
        LOG_DEBUG(<< "true segmentation = "
                  << core::CContainerPrinter::print(trueSegmentation));
        LOG_DEBUG(<< "segmentation      = " << core::CContainerPrinter::print(segmentation));

        TFloatMeanAccumulatorVec residuals{TSegmentation::removePiecewiseLinearScaledPeriodic(
            values, period, trueSegmentation, 0.05, 0.01)};

        // Project onto inliers.
        TMeanVarAccumulator residualMoments;
        for (auto i : inliers) {
            residualMoments.add(maths::CBasicStatistics::mean(residuals[i]));
            debug.addResidual(maths::CBasicStatistics::mean(residuals[i]));
        }
        LOG_DEBUG(<< "noise moments    = " << noiseMoments);
        LOG_DEBUG(<< "residual moments = " << residualMoments);

        // No false positives
        BOOST_REQUIRE_EQUAL(trueSegmentation.size(), segmentation.size());

        // Distance in index space is small.
        BOOST_TEST_REQUIRE(distance(trueSegmentation, segmentation) < 20);

        // Not biased.
        BOOST_TEST_REQUIRE(
            std::fabs(maths::CBasicStatistics::mean(residualMoments) -
                      maths::CBasicStatistics::mean(noiseMoments)) <
            3.0 * std::sqrt(maths::CBasicStatistics::variance(noiseMoments) /
                            maths::CBasicStatistics::count(noiseMoments)));

        // We've explained nearly all the variance.
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::variance(residualMoments) <
                           1.4 * maths::CBasicStatistics::variance(noiseMoments));
    }
}

BOOST_AUTO_TEST_CASE(testRemovePiecewiseLinearDiscontinuities) {

    std::size_t length{300};

    TFloatMeanAccumulatorVec values(length);

    for (std::size_t i = 0; i < length; ++i) {
        if (i < 50) {
            values[i].add(10.0);
        } else if (i < 200) {
            values[i].add(25.0);
        } else {
            values[i].add(3.0);
        }
    }

    TSizeVec segmentation(TSegmentation::piecewiseLinear(values));
    LOG_DEBUG(<< "segmentation = " << core::CContainerPrinter::print(segmentation));

    values = TSegmentation::removePiecewiseLinearDiscontinuities(values, segmentation);

    for (const auto& value : values) {
        BOOST_REQUIRE_EQUAL(3.0, static_cast<double>(maths::CBasicStatistics::mean(value)));
    }

    values.assign(length, TFloatMeanAccumulator{});
    for (std::size_t i = 0; i < length; ++i) {
        if (i < 50) {
            values[i].add(0.1 * static_cast<double>(i) + 3.0);
        } else if (i < 200) {
            values[i].add(-0.1 * static_cast<double>(i) + 7.0);
        } else {
            values[i].add(0.2 * static_cast<double>(i) - 2.0);
        }
    }

    segmentation = TSegmentation::piecewiseLinear(values);
    LOG_DEBUG(<< "segmentation = " << core::CContainerPrinter::print(segmentation));

    values = TSegmentation::removePiecewiseLinearDiscontinuities(values, segmentation);

    // Test
    //   1) We don't have any jump discontinuities,
    //   2) The slopes are preserved,
    //   3) The values are unchanged in the range [200, 300).

    for (std::size_t i = 1; i < values.size(); ++i) {
        BOOST_TEST_REQUIRE(std::fabs(maths::CBasicStatistics::mean(values[i]) -
                                     maths::CBasicStatistics::mean(values[i - 1])) < 0.25);
    }
    for (std::size_t i = 1; i < length; ++i) {
        if (i < 50) {
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                0.1,
                static_cast<double>(maths::CBasicStatistics::mean(values[i]) -
                                    maths::CBasicStatistics::mean(values[i - 1])),
                1e-4);
        } else if (i > 50 && i < 200) {
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                -0.1,
                static_cast<double>(maths::CBasicStatistics::mean(values[i]) -
                                    maths::CBasicStatistics::mean(values[i - 1])),
                1e-4);
        } else if (i > 200) {
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                0.2 * static_cast<double>(i) - 2.0,
                static_cast<double>(maths::CBasicStatistics::mean(values[i])), 1e-4);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
