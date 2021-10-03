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

#include <core/CContainerPrinter.h>
#include <core/CStringUtils.h>
#include <core/Constants.h>
#include <core/CoreTypes.h>

#include <maths/CBasicStatistics.h>
#include <maths/CSignal.h>
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
    CDebugGenerator(const std::string& file = "results.py") : m_File(file) {}
    ~CDebugGenerator() {
        if (ENABLED) {
            std::ofstream file;
            file.open(m_File);
            file << "import matplotlib.pyplot as plt;\n";
            file << "f = " << core::CContainerPrinter::print(m_Values) << ";\n";
            file << "r = " << core::CContainerPrinter::print(m_Residuals) << ";\n";
            file << "plt.figure(1);\n";
            file << "plt.clf();\n";
            file << "plt.plot(f);\n";
            file << "plt.show();\n";
            file << "plt.figure(2);\n";
            file << "plt.clf();\n";
            file << "plt.plot(r, 'k');\n";
            file << "plt.show();\n";
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

    // Test we identify trend knot points.

    core_t::TTime halfHour{core::constants::HOUR / 2};
    core_t::TTime week{core::constants::WEEK};
    core_t::TTime range{5 * week};

    test::CRandomNumbers rng;

    TFloatMeanAccumulatorVec values(range / halfHour);
    TDoubleVec noise;

    LOG_DEBUG(<< "Basic");
    for (auto outlierFraction : {0.0, 0.1}) {
        CDebugGenerator debug(
            "results." + core::CStringUtils::typeToStringPretty(outlierFraction) + ".py");

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

        TSizeVec segmentation{TSegmentation::piecewiseLinear(values, 1e-5, outlierFraction)};
        LOG_DEBUG(<< "true segmentation = "
                  << core::CContainerPrinter::print(trueSegmentation));
        LOG_DEBUG(<< "segmentation      = " << core::CContainerPrinter::print(segmentation));

        TFloatMeanAccumulatorVec residuals{TSegmentation::removePiecewiseLinear(
            values, segmentation, outlierFraction)};
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

    // Same again but with 5% salt-and-pepper outliers.

    LOG_DEBUG(<< "With Outliers");
    CDebugGenerator debug("results.outliers.py");
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

    TSizeVec segmentation(TSegmentation::piecewiseLinear(values, 0.01, 0.05));
    LOG_DEBUG(<< "true segmentation = " << core::CContainerPrinter::print(trueSegmentation));
    LOG_DEBUG(<< "segmentation      = " << core::CContainerPrinter::print(segmentation));

    TFloatMeanAccumulatorVec residuals{
        TSegmentation::removePiecewiseLinear(values, segmentation, 0.05)};

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
                       3.5 * std::sqrt(maths::CBasicStatistics::variance(noiseMoments) /
                                       maths::CBasicStatistics::count(noiseMoments)));

    // We've explained nearly all the variance.
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::variance(residualMoments) <
                       1.4 * maths::CBasicStatistics::variance(noiseMoments));
}

BOOST_AUTO_TEST_CASE(testPiecewiseLinearScaledSeasonal) {

    // Test we identify scale change points.

    core_t::TTime halfHour{core::constants::HOUR / 2};
    core_t::TTime week{core::constants::WEEK};
    core_t::TTime range{5 * week};

    test::CRandomNumbers rng;

    std::string descriptions[]{"smooth", "weekends", "spikey"};
    maths::CSignal::TSeasonalComponentVec periods[]{
        {maths::CSignal::seasonalComponentSummary(48)},
        {maths::CSignal::SSeasonalComponentSummary{48, 0, 336, {0, 240}},
         maths::CSignal::SSeasonalComponentSummary{336, 0, 336, {0, 240}},
         maths::CSignal::SSeasonalComponentSummary{48, 0, 336, {240, 336}},
         maths::CSignal::SSeasonalComponentSummary{336, 0, 336, {240, 336}}},
        {maths::CSignal::seasonalComponentSummary(48)}};

    TFloatMeanAccumulatorVec values(range / halfHour);
    TDoubleVec noise;

    LOG_DEBUG(<< "Basic");
    std::size_t i{0};
    for (auto seasonal : {smoothDaily, weekends, spikeyDaily}) {
        LOG_DEBUG(<< descriptions[i]);
        CDebugGenerator debug{"results." + descriptions[i] + ".py"};

        values.assign(range / halfHour, TFloatMeanAccumulator{});
        TMeanVarAccumulator noiseMoments;
        for (core_t::TTime time = 0; time < range; time += halfHour) {
            rng.generateNormalSamples(0.0, 3.0, 1, noise);
            noiseMoments.add(noise[0]);
            if (time < 3 * week / 2) {
                values[time / halfHour].add(100.0 * seasonal(time) + noise[0]);
            } else if (time < 2 * week) {
                values[time / halfHour].add(50.0 * seasonal(time) + noise[0]);
            } else {
                values[time / halfHour].add(300.0 * seasonal(time) + noise[0]);
            }
            debug.addValue(maths::CBasicStatistics::mean(values[time / halfHour]));
        }
        TSizeVec trueSegmentation{0, static_cast<std::size_t>(3 * week / halfHour / 2),
                                  static_cast<std::size_t>(2 * week / halfHour),
                                  values.size()};

        TSizeVec segmentation{TSegmentation::piecewiseLinearScaledSeasonal(
            values, [&](std::size_t j) { return seasonal(halfHour * j); }, 0.001)};
        LOG_DEBUG(<< "true segmentation = "
                  << core::CContainerPrinter::print(trueSegmentation));
        LOG_DEBUG(<< "segmentation      = " << core::CContainerPrinter::print(segmentation));

        // No false positives.
        BOOST_REQUIRE_EQUAL(trueSegmentation.size(), segmentation.size());

        // Distance in index space is small.
        BOOST_TEST_REQUIRE(distance(trueSegmentation, segmentation) < 5);
    }

    // Same again but with 5% salt-and-pepper outliers.

    LOG_DEBUG(<< "With Outliers");
    i = 0;
    for (auto seasonal : {smoothDaily, weekends}) {
        LOG_DEBUG(<< descriptions[i]);
        CDebugGenerator debug("results.outliers." + descriptions[i] + ".py");
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
                    values[time / halfHour].add(100.0 * seasonal(time) + noise[0]);
                } else if (time < 2 * week) {
                    values[time / halfHour].add(50.0 * seasonal(time) + noise[0]);
                } else {
                    values[time / halfHour].add(300.0 * seasonal(time) + noise[0]);
                }
                inliers.push_back(time / halfHour);
            }
            debug.addValue(maths::CBasicStatistics::mean(values[time / halfHour]));
        }
        TSizeVec trueSegmentation{0, static_cast<std::size_t>(3 * week / halfHour / 2),
                                  static_cast<std::size_t>(2 * week / halfHour),
                                  values.size()};

        maths::CSignal::TMeanAccumulatorVecVec components;
        maths::CSignal::fitSeasonalComponentsRobust(periods[i], 0.05, values, components);
        TSizeVec segmentation(TSegmentation::piecewiseLinearScaledSeasonal(
            values,
            [&](std::size_t j) {
                double result{0.0};
                for (std::size_t k = 0; k < periods[i].size(); ++k) {
                    if (periods[i][k].contains(j)) {
                        result += maths::CBasicStatistics::mean(
                            components[k][periods[i][k].offset(j)]);
                    }
                }
                return result;
            },
            0.01));
        ++i;
        LOG_DEBUG(<< "true segmentation = "
                  << core::CContainerPrinter::print(trueSegmentation));
        LOG_DEBUG(<< "segmentation      = " << core::CContainerPrinter::print(segmentation));

        // No false positives
        BOOST_REQUIRE_EQUAL(trueSegmentation.size(), segmentation.size());

        // Distance in index space is small.
        BOOST_TEST_REQUIRE(distance(trueSegmentation, segmentation) < 20);
    }
}

BOOST_AUTO_TEST_CASE(testRemovePiecewiseLinearScaledSeasonal) {

    // Test we get the residual distribution we expect after removing a piecewise
    // linear scaled seasonal component.

    core_t::TTime halfHour{core::constants::HOUR / 2};
    core_t::TTime week{core::constants::WEEK};
    core_t::TTime range{5 * week};

    test::CRandomNumbers rng;

    std::string descriptions[]{"smooth", "weekends", "spikey"};
    maths::CSignal::TSeasonalComponentVec periods[]{
        {maths::CSignal::seasonalComponentSummary(48)},
        {maths::CSignal::SSeasonalComponentSummary{48, 0, 336, {0, 240}},
         maths::CSignal::SSeasonalComponentSummary{336, 0, 336, {0, 240}},
         maths::CSignal::SSeasonalComponentSummary{48, 0, 336, {240, 336}},
         maths::CSignal::SSeasonalComponentSummary{336, 0, 336, {240, 336}}},
        {maths::CSignal::seasonalComponentSummary(48)}};

    TFloatMeanAccumulatorVec values(range / halfHour);
    TDoubleVec noise;

    LOG_DEBUG(<< "Basic");
    std::size_t i{0};
    for (auto seasonal : {smoothDaily, weekends, spikeyDaily}) {
        LOG_DEBUG(<< descriptions[i]);
        CDebugGenerator debug{"results." + descriptions[i] + ".py"};

        values.assign(range / halfHour, TFloatMeanAccumulator{});
        TMeanVarAccumulator noiseMoments;
        for (core_t::TTime time = 0; time < range; time += halfHour) {
            rng.generateNormalSamples(0.0, 3.0, 1, noise);
            noiseMoments.add(noise[0]);
            if (time < 3 * week / 2) {
                values[time / halfHour].add(100.0 * seasonal(time) + noise[0]);
            } else if (time < 2 * week) {
                values[time / halfHour].add(50.0 * seasonal(time) + noise[0]);
            } else {
                values[time / halfHour].add(300.0 * seasonal(time) + noise[0]);
            }
            debug.addValue(maths::CBasicStatistics::mean(values[time / halfHour]));
        }
        TSizeVec segmentation{0, static_cast<std::size_t>(3 * week / halfHour / 2),
                              static_cast<std::size_t>(2 * week / halfHour),
                              values.size()};

        TFloatMeanAccumulatorVec residuals{TSegmentation::removePiecewiseLinearScaledSeasonal(
            values, [&](std::size_t j) { return seasonal(halfHour * j); }, segmentation, 0.1)};

        TMeanVarAccumulator residualMoments;
        for (const auto& residual : residuals) {
            residualMoments.add(maths::CBasicStatistics::mean(residual));
            debug.addResidual(maths::CBasicStatistics::mean(residual));
        }
        LOG_DEBUG(<< "noise moments    = " << noiseMoments);
        LOG_DEBUG(<< "residual moments = " << residualMoments);

        // Not biased.
        BOOST_TEST_REQUIRE(
            std::fabs(maths::CBasicStatistics::mean(residualMoments) -
                      maths::CBasicStatistics::mean(noiseMoments)) <
            3.0 * std::sqrt(maths::CBasicStatistics::variance(noiseMoments) /
                            maths::CBasicStatistics::count(noiseMoments)));

        // We've explained nearly all the variance.
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::variance(residualMoments) <
                           1.1 * maths::CBasicStatistics::variance(noiseMoments));
    }
}

BOOST_AUTO_TEST_CASE(testRemovePiecewiseLinearDiscontinuities) {

    // Test we correctly remove step discontinuities.

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

    TSizeVec segmentation{TSegmentation::piecewiseLinear(values, 0.001, 0.1)};
    LOG_DEBUG(<< "segmentation = " << core::CContainerPrinter::print(segmentation));

    values = TSegmentation::removePiecewiseLinearDiscontinuities(values, segmentation, 0.1);

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

    segmentation = TSegmentation::piecewiseLinear(values, 0.001, 0.1);
    LOG_DEBUG(<< "segmentation = " << core::CContainerPrinter::print(segmentation));

    values = TSegmentation::removePiecewiseLinearDiscontinuities(values, segmentation, 0.1);

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

BOOST_AUTO_TEST_CASE(testMeanScalePiecewiseLinearScaledSeasonal) {

    // Test that mean scaling a component produces the expected values.

    auto predictor = [](const maths::CSignal::TSeasonalComponentVec& periods,
                        const maths::CSignal::TMeanAccumulatorVecVec& components,
                        std::size_t j) {
        double result{0.0};
        for (std::size_t i = 0; i < components.size(); ++i) {
            if (periods[i].contains(j)) {
                result += maths::CBasicStatistics::mean(components[i][periods[i].offset(j)]);
            }
        }
        return result;
    };
    auto meanScale = [](const TSizeVec& segmentation, const TDoubleVec& scales) {
        return TSegmentation::meanScale(segmentation, scales);
    };

    test::CRandomNumbers rng;

    core_t::TTime hour{core::constants::HOUR};
    maths::CSignal::TSeasonalComponentVec periods[]{
        {maths::CSignal::seasonalComponentSummary(24)},
        {maths::CSignal::SSeasonalComponentSummary{24, 0, 168, {0, 120}},
         maths::CSignal::SSeasonalComponentSummary{168, 0, 168, {0, 120}},
         maths::CSignal::SSeasonalComponentSummary{24, 0, 168, {120, 168}},
         maths::CSignal::SSeasonalComponentSummary{168, 0, 168, {120, 168}}}};
    TSizeVec segmentation{0, 220, 380, 600};
    TSizeVec repeats{24, 168};
    TDoubleVec scales{0.2, 1.3, 0.5};
    double expectedMeanScale{(11.0 * 0.2 + 8.0 * 1.3 + 11.0 * 0.5) / 30.0};
    TDoubleVec noise;
    TFloatMeanAccumulatorVec values;

    maths::CSignal::TMeanAccumulatorVecVec components;

    std::size_t i{0};
    for (const auto& seasonal : {smoothDaily, weekends}) {

        TMeanVarAccumulator overallErrorMoments;
        TSegmentation::TMeanAccumulatorVecVec scaledModels;
        TDoubleVec modelScales;

        for (std::size_t test = 0; test < 10; ++test) {

            rng.generateNormalSamples(0.0, 1.0, segmentation.back(), noise);

            values.assign(segmentation.back(), TFloatMeanAccumulator{});
            for (std::size_t j = 1; j < segmentation.size(); ++j) {
                for (std::size_t k = segmentation[j - 1]; k < segmentation[j]; ++k) {
                    values[k].add(scales[j - 1] * 10.0 * seasonal(hour * k) + noise[k]);
                }
            }

            values = TSegmentation::constantScalePiecewiseLinearScaledSeasonal(
                std::move(values), periods[i], segmentation, meanScale, 0.05,
                scaledModels, modelScales);
            BOOST_TEST_REQUIRE(values.size() > 0);

            maths::CSignal::fitSeasonalComponents(periods[i], values, components);

            TMeanVarAccumulator errorMoments;
            for (std::size_t j = 0; j < repeats[i]; ++j) {
                double actual{expectedMeanScale * 10.0 * seasonal(hour * j)};
                double prediction{predictor(periods[i], components, j)};
                BOOST_REQUIRE_CLOSE_ABSOLUTE(
                    actual, prediction,
                    7.0 / std::sqrt(static_cast<double>(values.size()) /
                                    static_cast<double>(repeats[i])));
                errorMoments.add(actual - prediction);
            }
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                1.0 / (static_cast<double>(values.size()) / static_cast<double>(repeats[i])),
                maths::CBasicStatistics::variance(errorMoments), 0.3);

            overallErrorMoments += errorMoments;
        }

        BOOST_REQUIRE_CLOSE_ABSOLUTE(0.0, maths::CBasicStatistics::mean(overallErrorMoments),
                                     std::sqrt(static_cast<double>(values.size()) /
                                               static_cast<double>(repeats[i]) / 10.0));
        BOOST_REQUIRE_CLOSE(1.0 / (static_cast<double>(values.size()) /
                                   static_cast<double>(repeats[i])),
                            maths::CBasicStatistics::variance(overallErrorMoments),
                            40.0); // 40%
        ++i;
    }
}

BOOST_AUTO_TEST_CASE(testPiecewiseTimeShifted) {

    // Test that we identify time shift points.

    core_t::TTime hour{core::constants::HOUR};

    TSegmentation::TModel models[]{
        [](core_t::TTime time) { return 10.0 * smoothDaily(86400 + time); },
        [](core_t::TTime time) { return 10.0 * spikeyDaily(86400 + time); }};

    test::CRandomNumbers rng;

    TSizeVec shift;
    core_t::TTime shifts[3]{0};
    TFloatMeanAccumulatorVec values;
    TDoubleVec noise;

    TSizeVec segmentation{0, 80, 200, 240};

    TSizeVec estimatedSegmentation;
    TSegmentation::TTimeVec estimatedShifts;
    TMeanVarAccumulator meanError;

    for (std::size_t i = 0; i < 10; ++i) {
        rng.generateUniformSamples(-5 * hour, -hour, 1, shift);
        shifts[1] = hour * (static_cast<core_t::TTime>(shift[0]) / hour);
        rng.generateUniformSamples(hour, 5 * hour, 1, shift);
        shifts[2] = hour * (static_cast<core_t::TTime>(shift[0]) / hour);
        LOG_DEBUG(<< "shifts = " << core::CContainerPrinter::print(shifts));

        values.assign(240, TFloatMeanAccumulator{});
        rng.generateNormalSamples(0.0, 1.0, 240, noise);

        core_t::TTime time{0};
        for (std::size_t j = 0; j < values.size(); ++j, time += hour / 2) {
            core_t::TTime shiftedTime{
                time + (j < 80 ? shifts[0] : (j < 200 ? shifts[1] : shifts[2]))};
            values[j].add(models[i % 2](shiftedTime) + noise[j]);
        }

        estimatedSegmentation = TSegmentation::piecewiseTimeShifted(
            values, hour / 2,
            {-4 * hour, -3 * hour, -2 * hour, -1 * hour, 1 * hour, 2 * hour,
             3 * hour, 4 * hour},
            models[i % 2], 0.001, 3, &estimatedShifts);

        BOOST_REQUIRE_EQUAL(segmentation.size(), estimatedSegmentation.size());
        int error{0};
        for (std::size_t j = 0; j < 4; ++j) {
            error += std::abs(static_cast<int>(estimatedSegmentation[j]) -
                              static_cast<int>(segmentation[j]));
        }
        BOOST_TEST_REQUIRE(error < 10);
        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(shifts),
                            core::CContainerPrinter::print(estimatedShifts));
        meanError.add(static_cast<double>(error));
    }

    LOG_DEBUG(<< "mean error = " << maths::CBasicStatistics::mean(meanError));
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanError) < 2.7);
}

BOOST_AUTO_TEST_CASE(testMeanScale) {

    // Test the mean scales matches what we would expect.

    BOOST_REQUIRE_CLOSE((2.0 + 2.0) / 3.0,
                        TSegmentation::meanScale({0, 20, 30}, {1.0, 2.0},
                                                 [](double) { return 1.0; }),
                        1e-4);
    BOOST_REQUIRE_CLOSE((4.0 + 1.0) / 3.0,
                        TSegmentation::meanScale({0, 20, 30}, {2.0, 1.0},
                                                 [](double) { return 1.0; }),
                        1e-4);
    BOOST_REQUIRE_CLOSE(
        1.5,
        TSegmentation::meanScale({0, 20, 30}, {1.0, 2.0},
                                 [](double i) { return i < 20 ? 0.5 : 1.0; }),
        1e-4);
    BOOST_REQUIRE_CLOSE(
        1.5,
        TSegmentation::meanScale({0, 20, 30}, {2.0, 1.0},
                                 [](double i) { return i >= 20 ? 2.0 : 1.0; }),
        1e-4);
}

BOOST_AUTO_TEST_SUITE_END()
