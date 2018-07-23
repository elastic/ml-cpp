/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CTimeSeriesSegmentationTest.h"

#include <core/CStringUtils.h>
#include <core/Constants.h>
#include <core/CoreTypes.h>

#include <maths/CTimeSeriesSegmentation.h>

#include <test/CRandomNumbers.h>

#include "TestUtils.h"

#include <vector>

using namespace ml;

using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;
using TFloatMeanAccumulator =
    maths::CBasicStatistics::SSampleMean<maths::CFloatStorage>::TAccumulator;
using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
using TMeanVarAccumulator = maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

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

void CTimeSeriesSegmentationTest::testTopDownPiecewiseLinear() {

    core_t::TTime halfHour{core::constants::HOUR / 2};
    core_t::TTime week{core::constants::WEEK};
    core_t::TTime range{5 * week};

    test::CRandomNumbers rng;

    TFloatMeanAccumulatorVec values(range / halfHour);
    TDoubleVec noise;

    LOG_DEBUG(<< "Basic");
    for (auto outlierFraction : {0.0, 0.1}) {
        CDebugGenerator debug("results.m." + core::CStringUtils::typeToStringPretty(outlierFraction));

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

        TSizeVec segmentation(maths::CTimeSeriesSegmentation::topDownPiecewiseLinear(
            values, 0.01, outlierFraction, 0.1));
        LOG_DEBUG(<< "true segmentation = "
                  << core::CContainerPrinter::print(trueSegmentation));
        LOG_DEBUG(<< "segmentation      = " << core::CContainerPrinter::print(segmentation));

        TFloatMeanAccumulatorVec residuals{maths::CTimeSeriesSegmentation::removePredictionsOfPiecewiseLinear(
            values, segmentation, outlierFraction, 0.1)};
        TMeanVarAccumulator residualMoments;
        for (const auto& residual : residuals) {
            residualMoments.add(maths::CBasicStatistics::mean(residual));
            debug.addResidual(maths::CBasicStatistics::mean(residual));
        }
        LOG_DEBUG(<< "noise moments    = " << noiseMoments);
        LOG_DEBUG(<< "residual moments = " << residualMoments);

        // No false positives.
        CPPUNIT_ASSERT_EQUAL(trueSegmentation.size(), segmentation.size());

        // Distance in index space is small.
        CPPUNIT_ASSERT(distance(trueSegmentation, segmentation) < 35);

        // Not biased.
        CPPUNIT_ASSERT(std::fabs(maths::CBasicStatistics::mean(residualMoments) -
                                 maths::CBasicStatistics::mean(noiseMoments)) <
                       2.0 * std::sqrt(maths::CBasicStatistics::variance(noiseMoments) /
                                       maths::CBasicStatistics::count(noiseMoments)));

        // We've explained nearly all the variance.
        CPPUNIT_ASSERT(maths::CBasicStatistics::variance(residualMoments) <
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

    TSizeVec segmentation(
        maths::CTimeSeriesSegmentation::topDownPiecewiseLinear(values, 0.01, 0.05, 0.1));
    LOG_DEBUG(<< "true segmentation = " << core::CContainerPrinter::print(trueSegmentation));
    LOG_DEBUG(<< "segmentation      = " << core::CContainerPrinter::print(segmentation));

    TFloatMeanAccumulatorVec residuals{maths::CTimeSeriesSegmentation::removePredictionsOfPiecewiseLinear(
        values, segmentation, 0.05, 0.1)};

    // Project onto inliers.
    TMeanVarAccumulator residualMoments;
    for (auto i : inliers) {
        residualMoments.add(maths::CBasicStatistics::mean(residuals[i]));
        debug.addResidual(maths::CBasicStatistics::mean(residuals[i]));
    }
    LOG_DEBUG(<< "noise moments    = " << noiseMoments);
    LOG_DEBUG(<< "residual moments = " << residualMoments);

    // No false positives
    CPPUNIT_ASSERT_EQUAL(trueSegmentation.size(), segmentation.size());

    // Distance in index space is small.
    CPPUNIT_ASSERT(distance(trueSegmentation, segmentation) < 35);

    // Not biased.
    CPPUNIT_ASSERT(std::fabs(maths::CBasicStatistics::mean(residualMoments) -
                             maths::CBasicStatistics::mean(noiseMoments)) <
                   3.0 * std::sqrt(maths::CBasicStatistics::variance(noiseMoments) /
                                   maths::CBasicStatistics::count(noiseMoments)));

    // We've explained nearly all the variance.
    CPPUNIT_ASSERT(maths::CBasicStatistics::variance(residualMoments) <
                   1.4 * maths::CBasicStatistics::variance(noiseMoments));
}

void CTimeSeriesSegmentationTest::testTopDownPeriodicPiecewiseLinearScaling() {

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
            CDebugGenerator debug("results.m." + core::CStringUtils::typeToStringPretty(outlierFraction) + "." + periods[j++]);

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

            TSizeVec segmentation(maths::CTimeSeriesSegmentation::topDownPeriodicPiecewiseLinearScaled(
                values, period, 0.01, outlierFraction));
            LOG_DEBUG(<< "true segmentation = "
                      << core::CContainerPrinter::print(trueSegmentation));
            LOG_DEBUG(<< "segmentation      = "
                      << core::CContainerPrinter::print(segmentation));

            TFloatMeanAccumulatorVec residuals{maths::CTimeSeriesSegmentation::removePredictionsOfPiecewiseLinearScaled(
                values, period, segmentation, outlierFraction, 0.1)};
            TMeanVarAccumulator residualMoments;
            for (const auto& residual : residuals) {
                residualMoments.add(maths::CBasicStatistics::mean(residual));
                debug.addResidual(maths::CBasicStatistics::mean(residual));
            }
            LOG_DEBUG(<< "noise moments    = " << noiseMoments);
            LOG_DEBUG(<< "residual moments = " << residualMoments);

            // No false positives.
            CPPUNIT_ASSERT_EQUAL(trueSegmentation.size(), segmentation.size());

            // Distance in index space is small.
            CPPUNIT_ASSERT(distance(trueSegmentation, segmentation) < 5);

            // Not biased.
            CPPUNIT_ASSERT(std::fabs(maths::CBasicStatistics::mean(residualMoments) -
                                     maths::CBasicStatistics::mean(noiseMoments)) <
                           3.0 * std::sqrt(maths::CBasicStatistics::variance(noiseMoments) /
                                           maths::CBasicStatistics::count(noiseMoments)));

            // We've explained nearly all the variance.
            CPPUNIT_ASSERT(maths::CBasicStatistics::variance(residualMoments) <
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
        TSizeVec trueSegmentation{
            0, static_cast<std::size_t>(3 * week / halfHour / 2),
            static_cast<std::size_t>(2 * week / halfHour), values.size()};

        TSizeVec segmentation(
            maths::CTimeSeriesSegmentation::topDownPeriodicPiecewiseLinearScaled(values, period, 0.01, 0.05, 0.01));
        LOG_DEBUG(<< "true segmentation = " << core::CContainerPrinter::print(trueSegmentation));
        LOG_DEBUG(<< "segmentation      = " << core::CContainerPrinter::print(segmentation));

        TFloatMeanAccumulatorVec residuals{maths::CTimeSeriesSegmentation::removePredictionsOfPiecewiseLinearScaled(
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
        CPPUNIT_ASSERT_EQUAL(trueSegmentation.size(), segmentation.size());

        // Distance in index space is small.
        CPPUNIT_ASSERT(distance(trueSegmentation, segmentation) < 20);

        // Not biased.
        CPPUNIT_ASSERT(std::fabs(maths::CBasicStatistics::mean(residualMoments) -
                                 maths::CBasicStatistics::mean(noiseMoments)) <
                       3.0 * std::sqrt(maths::CBasicStatistics::variance(noiseMoments) /
                                        maths::CBasicStatistics::count(noiseMoments)));

        // We've explained nearly all the variance.
        CPPUNIT_ASSERT(maths::CBasicStatistics::variance(residualMoments) <
                       2.5 * maths::CBasicStatistics::variance(noiseMoments));
    }
}

CppUnit::Test* CTimeSeriesSegmentationTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CTimeSeriesSegmentationTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CTimeSeriesSegmentationTest>(
        "CTimeSeriesSegmentationTest::testTopDownPiecewiseLinear",
        &CTimeSeriesSegmentationTest::testTopDownPiecewiseLinear));
    suiteOfTests->addTest(new CppUnit::TestCaller<CTimeSeriesSegmentationTest>(
        "CTimeSeriesSegmentationTest::testTopDownPeriodicPiecewiseLinearScaling",
        &CTimeSeriesSegmentationTest::testTopDownPeriodicPiecewiseLinearScaling));

    return suiteOfTests;
}
