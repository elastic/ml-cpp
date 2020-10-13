/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CoreTypes.h>

#include <maths/CBasicStatistics.h>
#include <maths/CSignal.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <boost/math/constants/constants.hpp>
#include <boost/test/unit_test.hpp>

#include <string>

BOOST_AUTO_TEST_SUITE(CSignalTest)

using namespace ml;

namespace {

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TSizeVec = std::vector<std::size_t>;
using TSizeVecVec = std::vector<TSizeVec>;
using TPredictorVec = std::vector<maths::CSignal::TPredictor>;
using TPredictorVecVec = std::vector<TPredictorVec>;
using TMeanVarAccumulator = maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

std::string print(const maths::CSignal::TComplexVec& f) {
    std::ostringstream result;
    for (std::size_t i = 0; i < f.size(); ++i) {
        LOG_DEBUG(<< f[i].real() << " + " << f[i].imag() << 'i');
    }
    return result.str();
}

void bruteForceDft(maths::CSignal::TComplexVec& f, double sign) {
    maths::CSignal::TComplexVec result(f.size(), maths::CSignal::TComplex(0.0, 0.0));
    for (std::size_t k = 0; k < f.size(); ++k) {
        for (std::size_t n = 0; n < f.size(); ++n) {
            double t{-sign * boost::math::double_constants::two_pi *
                     static_cast<double>(k * n) / static_cast<double>(f.size())};
            result[k] += maths::CSignal::TComplex(std::cos(t), std::sin(t)) * f[n];
        }
        if (sign < 0.0) {
            result[k] /= static_cast<double>(f.size());
        }
    }
    f.swap(result);
}

maths::CSignal::TSeasonalComponentVec seasonalComponentSummary(TSizeVec periods) {
    maths::CSignal::TSeasonalComponentVec result;
    result.reserve(periods.size());
    for (auto period : periods) {
        result.emplace_back(period, 0, period,
                            std::pair<std::size_t, std::size_t>{0, period});
    }
    return result;
}
}

BOOST_AUTO_TEST_CASE(testFFTVersusOctave) {
    // Test versus values calculated using octave fft.

    double x[][20] = {
        {2555.33, 1451.79,  465.60,   4394.83, -1553.24, -2772.07, -3977.73,
         2249.31, -2006.04, 3540.84,  4271.63, 4648.81,  -727.90,  2285.24,
         3129.56, -3596.79, -1968.66, 3795.18, 1627.84,  228.40},
        {4473.77,  -4815.63, -818.38, -1953.72, -2323.39, -3007.25, 4444.24,
         435.21,   3613.32,  3471.37, -1735.72, 2560.82,  -2383.29, -2370.23,
         -4921.04, -541.25,  1516.69, -2028.42, 3981.02,  3156.88}};

    maths::CSignal::TComplexVec fx;
    for (std::size_t i = 0; i < 20; ++i) {
        fx.emplace_back(x[0][i], x[1][i]);
    }

    LOG_DEBUG(<< "*** Power of 2 Length ***");
    {
        double expected[][2]{// length 2
                             {4007.1, -341.9},
                             {1103.5, 9289.4},
                             // length 4
                             {8867.5, -3114.0},
                             {-772.18, 8235.2},
                             {-2825.7, 10425.0},
                             {4951.6, 2349.1},
                             // length 8
                             {2813.8, -3565.1},
                             {-2652.4, -1739.5},
                             {-1790.1, 6488.9},
                             {4933.6, 6326.1},
                             {-7833.9, 15118.0},
                             {344.29, 6447.2},
                             {10819.0, -9439.9},
                             {13809.0, 16155.0},
                             // length 16
                             {14359.0, -5871.2},
                             {1176.5, -3143.9},
                             {636.25, -1666.2},
                             {-2819.0, 8259.4},
                             {-12844.0, 9601.7},
                             {-2292.3, -5598.5},
                             {11737.0, 4809.3},
                             {-2499.2, -143.95},
                             {-10045.0, 6570.2},
                             {27277.0, 10002.0},
                             {870.01, 16083.0},
                             {21695.0, 19192.0},
                             {1601.9, 3220.9},
                             {-7675.7, 5483.5},
                             {-1921.5, 31949.0},
                             {1629.0, -27167.0}};

        for (std::size_t i = 0, l = 2; l < fx.size(); i += l, l <<= 1) {
            LOG_DEBUG(<< "Testing length " << l);

            maths::CSignal::TComplexVec actual(fx.begin(), fx.begin() + l);
            maths::CSignal::fft(actual);
            LOG_DEBUG(<< print(actual));

            double error{0.0};
            for (std::size_t j = 0; j < l; ++j) {
                error += std::abs(actual[j] -
                                  maths::CSignal::TComplex(expected[i + j][0],
                                                           expected[i + j][1]));
            }
            error /= static_cast<double>(l);
            LOG_DEBUG(<< "error = " << error);
            BOOST_TEST_REQUIRE(error < 0.2);
        }
    }

    LOG_DEBUG(<< "*** Arbitrary Length ***");
    {
        double expected[][2]{
            {18042.0, 755.0},    {961.0, 5635.6},     {-5261.8, 7542.2},
            {-12814.0, 2250.2},  {-8248.5, 6620.5},   {-21626.0, 3570.6},
            {6551.5, -12732.0},  {6009.5, 10622.0},   {9954.0, -1224.2},
            {-2871.5, 7073.6},   {-14409.0, 10939.0}, {13682.0, 25304.0},
            {-10468.0, -6338.5}, {6506.0, 6283.3},    {32665.0, 5127.7},
            {3190.7, 4323.4},    {-6988.7, -3865.0},  {-3881.4, 4360.8},
            {46434.0, 20556.0},  {-6319.6, -7329.0}};

        maths::CSignal::TComplexVec actual(fx.begin(), fx.end());
        maths::CSignal::fft(actual);
        double error{0.0};
        for (std::size_t j = 0; j < actual.size(); ++j) {
            error += std::abs(actual[j] - maths::CSignal::TComplex(expected[j][0],
                                                                   expected[j][1]));
        }
        error /= static_cast<double>(actual.size());
        LOG_DEBUG(<< "error = " << error);
        BOOST_TEST_REQUIRE(error < 0.2);
    }
}

BOOST_AUTO_TEST_CASE(testIFFTVersusOctave) {
    // Test versus values calculated using octave ifft.

    double x[][20]{
        {2555.33, 1451.79,  465.60,   4394.83, -1553.24, -2772.07, -3977.73,
         2249.31, -2006.04, 3540.84,  4271.63, 4648.81,  -727.90,  2285.24,
         3129.56, -3596.79, -1968.66, 3795.18, 1627.84,  228.40},
        {4473.77,  -4815.63, -818.38, -1953.72, -2323.39, -3007.25, 4444.24,
         435.21,   3613.32,  3471.37, -1735.72, 2560.82,  -2383.29, -2370.23,
         -4921.04, -541.25,  1516.69, -2028.42, 3981.02,  3156.88}};

    maths::CSignal::TComplexVec fx;
    for (std::size_t i = 0; i < 20; ++i) {
        fx.emplace_back(x[0][i], x[1][i]);
    }

    LOG_DEBUG(<< "*** Powers of 2 Length ***");
    {
        double expected[][2]{// length 2
                             {2003.56, -170.93},
                             {551.77, 4644.70},
                             // length 4
                             {2216.89, -778.49},
                             {1237.91, 587.28},
                             {-706.42, 2606.19},
                             {-193.04, 2058.80},
                             {351.73, -445.64},
                             // length 8
                             {1726.09, 2019.35},
                             {1352.32, -1179.99},
                             {43.04, 805.89},
                             {-979.24, 1889.70},
                             {616.70, 790.77},
                             {-223.77, 811.12},
                             {-331.55, -217.44},
                             {897.45, -366.95},
                             // length 16
                             {101.81, -1697.92},
                             {-120.10, 1996.81},
                             {-479.73, 342.72},
                             {100.12, 201.31},
                             {1355.94, 1199.49},
                             {54.38, 1005.18},
                             {1704.78, 625.13},
                             {-627.80, 410.64},
                             {-156.20, -9.00},
                             {733.56, 300.58},
                             {-143.27, -349.91},
                             {-802.73, 600.10},
                             {-176.19, 516.21},
                             {39.77, -104.14},
                             {73.53, -196.49}};

        for (std::size_t i = 0, l = 2; l < fx.size(); i += l, l <<= 1) {
            LOG_DEBUG(<< "Testing length " << l);

            maths::CSignal::TComplexVec actual(fx.begin(), fx.begin() + l);
            maths::CSignal::ifft(actual);
            LOG_DEBUG(<< print(actual));

            double error{0.0};
            for (std::size_t j = 0; j < l; ++j) {
                error += std::abs(actual[j] -
                                  maths::CSignal::TComplex(expected[i + j][0],
                                                           expected[i + j][1]));
            }
            error /= static_cast<double>(l);
            LOG_DEBUG(<< "error = " << error);
            BOOST_TEST_REQUIRE(error < 0.01);
        }
    }
}

BOOST_AUTO_TEST_CASE(testFFTRandomized) {

    // Test on randomized input versus brute force.

    test::CRandomNumbers rng;

    TDoubleVec components;
    rng.generateUniformSamples(-100000.0, 100000.0, 20000, components);

    TSizeVec lengths;
    rng.generateUniformSamples(2, 100, 1000, lengths);

    for (std::size_t i = 0, j = 0;
         i < lengths.size() && j + 2 * lengths[i] < components.size();
         ++i, j += 2 * lengths[i]) {
        maths::CSignal::TComplexVec expected;
        for (std::size_t k = 0; k < lengths[i]; ++k) {
            expected.emplace_back(components[j + 2 * k], components[j + 2 * k + 1]);
        }
        maths::CSignal::TComplexVec actual(expected);

        bruteForceDft(expected, +1.0);
        maths::CSignal::fft(actual);

        double error{0.0};
        for (std::size_t k = 0; k < actual.size(); ++k) {
            error += std::abs(actual[k] - expected[k]);
        }

        if (i % 5 == 0 || error >= 1e-5) {
            LOG_DEBUG(<< "length = " << lengths[i] << ", error  = " << error);
        }
        BOOST_TEST_REQUIRE(error < 1e-5);
    }
}

BOOST_AUTO_TEST_CASE(testIFFTRandomized) {
    // Test on randomized input versus brute force.

    test::CRandomNumbers rng;

    TDoubleVec components;
    rng.generateUniformSamples(-100000.0, 100000.0, 20000, components);

    TSizeVec lengths;
    rng.generateUniformSamples(2, 100, 1000, lengths);

    for (std::size_t i = 0, j = 0;
         i < lengths.size() && j + 2 * lengths[i] < components.size();
         ++i, j += 2 * lengths[i]) {
        maths::CSignal::TComplexVec expected;
        for (std::size_t k = 0; k < lengths[i]; ++k) {
            expected.emplace_back(components[j + 2 * k], components[j + 2 * k + 1]);
        }
        maths::CSignal::TComplexVec actual(expected);

        bruteForceDft(expected, -1.0);
        maths::CSignal::ifft(actual);

        double error = 0.0;
        for (std::size_t k = 0; k < actual.size(); ++k) {
            error += std::abs(actual[k] - expected[k]);
        }

        if (i % 5 == 0 || error >= 1e-5) {
            LOG_DEBUG(<< "length = " << lengths[i] << ", error  = " << error);
        }
        BOOST_TEST_REQUIRE(error < 1e-5);
    }
}

BOOST_AUTO_TEST_CASE(testFFTIFFTIdempotency) {
    // Test on randomized input that x = F(F^-1(x)).

    test::CRandomNumbers rng;

    TDoubleVec components;
    rng.generateUniformSamples(-100000.0, 100000.0, 20000, components);

    TSizeVec lengths;
    rng.generateUniformSamples(2, 100, 1000, lengths);

    for (std::size_t i = 0, j = 0;
         i < lengths.size() && j + 2 * lengths[i] < components.size();
         ++i, j += 2 * lengths[i]) {
        maths::CSignal::TComplexVec expected;
        for (std::size_t k = 0; k < lengths[i]; ++k) {
            expected.emplace_back(components[j + 2 * k], components[j + 2 * k + 1]);
        }

        maths::CSignal::TComplexVec actual(expected);
        maths::CSignal::fft(actual);
        maths::CSignal::ifft(actual);

        double error = 0.0;
        for (std::size_t k = 0; k < actual.size(); ++k) {
            error += std::abs(actual[k] - expected[k]);
        }

        if (i % 5 == 0 || error >= 1e-5) {
            LOG_DEBUG(<< "length = " << lengths[i] << ", error  = " << error);
        }
        BOOST_TEST_REQUIRE(error < 1e-5);
    }
}

BOOST_AUTO_TEST_CASE(testAutocorrelations) {
    test::CRandomNumbers rng;

    TSizeVec sizes;
    rng.generateUniformSamples(10, 30, 100, sizes);

    for (std::size_t t = 0; t < sizes.size(); ++t) {
        TDoubleVec values_;
        rng.generateUniformSamples(-10.0, 10.0, sizes[t], values_);

        maths::CSignal::TFloatMeanAccumulatorVec values(sizes[t]);
        for (std::size_t i = 0; i < values_.size(); ++i) {
            values[i].add(values_[i]);
        }

        TDoubleVec expected;
        for (std::size_t offset = 1; offset < values.size(); ++offset) {
            expected.push_back(maths::CSignal::cyclicAutocorrelation(
                maths::CSignal::seasonalComponentSummary(offset), values));
        }

        TDoubleVec actual;
        maths::CSignal::autocorrelations(values, actual);

        if (t % 10 == 0) {
            LOG_DEBUG(<< "expected = " << core::CContainerPrinter::print(expected));
            LOG_DEBUG(<< "actual   = " << core::CContainerPrinter::print(actual));
        }
        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expected),
                            core::CContainerPrinter::print(actual));
    }
}

BOOST_AUTO_TEST_CASE(testSeasonalComponentSummary) {

    using TSizeSizePrVec = std::vector<std::pair<std::size_t, std::size_t>>;

    TSizeSizePrVec expectedWindows;
    {
        maths::CSignal::SSeasonalComponentSummary period{10, 0, 10, {0, 10}};
        BOOST_REQUIRE_EQUAL(10, period.period());
        BOOST_REQUIRE_EQUAL(false, period.windowed());
        BOOST_REQUIRE(period == period);
        BOOST_REQUIRE((period < period) == false);
        for (std::size_t i = 0; i < 100; ++i) {
            BOOST_REQUIRE_EQUAL(true, period.contains(i));
            BOOST_REQUIRE_EQUAL(i % 10, period.offset(i));
            expectedWindows.assign(1, {0, i});
            BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedWindows),
                                core::CContainerPrinter::print(period.windows(i)));
        }
    }
    {
        maths::CSignal::SSeasonalComponentSummary period{10, 5, 15, {0, 5}};
        BOOST_REQUIRE_EQUAL(5, period.period());
        BOOST_REQUIRE_EQUAL(true, period.windowed());
        BOOST_REQUIRE(period == period);
        BOOST_REQUIRE((period < period) == false);
        TSizeSizePrVec windows{{5, 10}, {20, 25}, {35, 40}, {50, 55}};
        for (std::size_t i = 0; i < 50; ++i) {
            std::size_t expectedOffset{[&] {
                for (const auto& window : windows) {
                    if (i >= window.first && i < window.second) {
                        return i - window.first;
                    }
                }
                return std::size_t{5};
            }()};
            BOOST_REQUIRE_EQUAL(expectedOffset < 5, period.contains(i));
            if (expectedOffset < 5) {
                BOOST_REQUIRE_EQUAL(expectedOffset, period.offset(i));
            }
            BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(
                                    windows.begin(), windows.begin() + (i + 14) / 15),
                                core::CContainerPrinter::print(period.windows(i)));
        }
    }
}

BOOST_AUTO_TEST_CASE(testCountNotMissing) {

    maths::CSignal::TFloatMeanAccumulatorVec values;
    BOOST_REQUIRE_EQUAL(0, maths::CSignal::countNotMissing(values));

    values.resize(5);
    BOOST_REQUIRE_EQUAL(0, maths::CSignal::countNotMissing(values));

    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i].add(1.0);
        BOOST_REQUIRE_EQUAL(i + 1, maths::CSignal::countNotMissing(values));
    }
}

BOOST_AUTO_TEST_CASE(testRestrictTo) {

    maths::CSignal::TFloatMeanAccumulatorVec values(200);
    for (std::size_t i = 0; i < 200; ++i) {
        values[i].add(static_cast<double>(i));
    }

    maths::CSignal::TSizeSizePr2Vec windows;
    maths::CSignal::TFloatMeanAccumulatorVec restricted;

    // Exactly half the values are in windows.
    restricted = values;
    windows = {{10, 20}, {50, 120}, {190, 210}};
    maths::CSignal::restrictTo(windows, restricted);
    BOOST_REQUIRE_EQUAL(100, restricted.size());
    std::size_t j{0};
    for (const auto& window : windows) {
        for (std::size_t i = window.first; i < window.second; ++i, ++j) {
            BOOST_REQUIRE_EQUAL(static_cast<double>(i % 200),
                                maths::CBasicStatistics::mean(restricted[j]));
        }
    }

    test::CRandomNumbers rng;

    std::size_t tests{0};
    TSizeVec endpoints;

    for (std::size_t test = 0; test < 1000; ++test) {

        rng.generateUniformSamples(100, 300, 10, endpoints);
        std::sort(endpoints.begin(), endpoints.end());
        endpoints.erase(std::unique(endpoints.begin(), endpoints.end()),
                        endpoints.end());

        if (endpoints.size() % 2 == 0) {
            std::size_t size{0};
            windows.resize(endpoints.size() / 2);
            for (std::size_t i = 0; i < endpoints.size(); i += 2) {
                size += endpoints[i + 1] - endpoints[i];
                windows[i / 2] = std::make_pair(endpoints[i], endpoints[i + 1]);
            }
            LOG_TRACE(<< "windows = " << core::CContainerPrinter::print(windows));

            restricted = values;
            maths::CSignal::restrictTo(windows, restricted);

            j = 0;
            BOOST_REQUIRE_EQUAL(size, restricted.size());
            for (const auto& window : windows) {
                for (std::size_t i = window.first; i < window.second; ++i, ++j) {
                    BOOST_REQUIRE_EQUAL(static_cast<double>(i % 200),
                                        maths::CBasicStatistics::mean(restricted[j]));
                }
            }
            ++tests;
        }
    }

    BOOST_TEST_REQUIRE(tests > 0);
}

BOOST_AUTO_TEST_CASE(testReweightOutliers) {

    // Check that we pickout pepper and salt outliers for a variety of components.

    TPredictorVec components{
        [](std::size_t) { return 10.0; },
        [](std::size_t index) { return static_cast<double>(index) / 5.0; },
        [](std::size_t index) {
            return 10.0 + 10.0 * std::sin(boost::math::double_constants::two_pi *
                                          static_cast<double>(index) / 50.0);
        }};

    test::CRandomNumbers rng;

    maths::CSignal::TFloatMeanAccumulatorVec values;
    TDoubleVec noise;
    TDoubleVec u01;

    for (std::size_t test = 0; test < 100; ++test) {

        const auto& component = components[test % components.size()];

        values.assign(values.size(), maths::CSignal::TFloatMeanAccumulator{});
        rng.generateUniformSamples(0.0, 1.0, values.size(), u01);
        rng.generateNormalSamples(0.0, 4.0, values.size(), noise);

        for (std::size_t i = 0; i < noise.size(); ++i) {
            if (u01[i] < 0.02) {
                values[i].add(-10.0);
            } else if (u01[i] > 0.98) {
                values[i].add(30.0);
            } else {
                values[i].add(component(i) + noise[i]);
            }
        }

        maths::CSignal::reweightOutliers(component, 0.1, values);

        for (std::size_t i = 0; i < values.size(); ++i) {
            if (u01[i] < 0.02 || u01[i] > 0.98) {
                BOOST_TEST_REQUIRE(maths::CBasicStatistics::count(values[i]) < 1.0);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testFitSingleSeasonalComponent) {

    // Test the accuracy with which we estimate one seasonal component.

    TSizeVec periods{5, 20, 10};
    TPredictorVec expectedComponents{
        [](std::size_t i) {
            double values[]{10.0, 5.0, 6.0, 15.0, 18.0};
            return values[i % 5];
        },
        [](std::size_t i) {
            return 10.0 * std::sin(boost::math::double_constants::two_pi *
                                   static_cast<double>(i) / 20.0);
        },
        [](std::size_t i) { return i % 10 == 3 ? 10.0 : 0.0; }};

    test::CRandomNumbers rng;

    maths::CSignal::TFloatMeanAccumulatorVec values;
    maths::CSignal::TMeanAccumulatorVecVec actuals;
    TDoubleVec noise;

    for (std::size_t test = 0; test < 100; ++test) {

        auto expected = expectedComponents[test % expectedComponents.size()];
        std::size_t period{periods[test % periods.size()]};

        values.assign(100, maths::CSignal::TFloatMeanAccumulator{});
        rng.generateNormalSamples(0.0, 4.0, values.size(), noise);
        for (std::size_t i = 0; i < noise.size(); ++i) {
            values[i].add(expected(i) + noise[i]);
        }

        maths::CSignal::fitSeasonalComponents(seasonalComponentSummary({period}),
                                              values, actuals);

        BOOST_REQUIRE_EQUAL(1, actuals.size());
        BOOST_REQUIRE_EQUAL(period, actuals[0].size());

        TMeanVarAccumulator meanError;
        double sigma{std::sqrt(4.0 / (static_cast<double>(values.size()) / period))};
        for (std::size_t i = 0; i < actuals[0].size(); ++i) {
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                expected(i), maths::CBasicStatistics::mean(actuals[0][i]), 4.0 * sigma);
            meanError(std::fabs(expected(i) - maths::CBasicStatistics::mean(actuals[0][i])) / sigma);
        }
        BOOST_REQUIRE(maths::CBasicStatistics::mean(meanError) < 1.5);
    }
}

BOOST_AUTO_TEST_CASE(testFitMultipleSeasonalComponents) {

    // Test the accuracy with which we estimate a mixture of two seasonal components.

    TPredictorVecVec expectedComponents{
        {[](std::size_t i) {
             double values[]{10.0, 5.0, 6.0, 15.0, 18.0};
             return values[i % 5];
         },
         [](std::size_t i) {
             return 10.0 * std::sin(boost::math::double_constants::two_pi *
                                    static_cast<double>(i) / 12.0);
         }},
        {[](std::size_t i) {
             double values[]{10.0, 5.0, 6.0, 15.0, 18.0};
             return values[i % 5];
         },
         [](std::size_t i) { return i % 10 == 3 ? 10.0 : 0.0; }},
        {[](std::size_t i) { return i % 10 == 3 ? 10.0 : 0.0; },
         [](std::size_t i) {
             return 10.0 * std::sin(boost::math::double_constants::two_pi *
                                    static_cast<double>(i) / 15.0);
         }}};

    test::CRandomNumbers rng;

    TSizeVec lengths{72, 43, 95};
    TSizeVecVec periods{{5, 12}, {5, 10}, {10, 15}};
    maths::CSignal::TFloatMeanAccumulatorVec values;
    maths::CSignal::TMeanAccumulatorVecVec actuals;
    TDoubleVec noise;

    TMeanVarAccumulator overallMeanError;

    for (std::size_t test = 0; test < 100; ++test) {

        const auto& expected = expectedComponents[test % expectedComponents.size()];
        const auto& period = periods[test % periods.size()];

        values.assign(lengths[test % lengths.size()],
                      maths::CSignal::TFloatMeanAccumulator{});
        rng.generateNormalSamples(0.0, 4.0, values.size(), noise);
        for (std::size_t i = 0; i < noise.size(); ++i) {
            double sum{0.0};
            for (std::size_t j = 0; j < expected.size(); ++j) {
                sum += expected[j](i);
            }
            values[i].add(sum + noise[i]);
        }

        maths::CSignal::fitSeasonalComponents(seasonalComponentSummary(period),
                                              values, actuals);

        BOOST_REQUIRE_EQUAL(period.size(), actuals.size());
        for (std::size_t i = 0; i < period.size(); ++i) {
            BOOST_REQUIRE_EQUAL(period[i], actuals[i].size());
        }

        TMeanVarAccumulator meanError;
        for (std::size_t i = 0; i < actuals.size(); ++i) {
            double sigma{std::sqrt(4.0 / (static_cast<double>(values.size()) /
                                          static_cast<double>(period[i])))};
            for (std::size_t j = 0; j < actuals[i].size(); ++j) {
                meanError(std::fabs(expected[i](j) -
                                    maths::CBasicStatistics::mean(actuals[i][j])) /
                          sigma);
            }
        }
        BOOST_REQUIRE(maths::CBasicStatistics::mean(meanError) < 2.5);
        overallMeanError += meanError;

        // We can only really guaranty individual component values up to additive
        // constants which sum to zero. Therefore, we check their sum is close to
        // the expected value w.r.t. the standard deviation of the noise.
        double sigma{std::sqrt(4.0 / (static_cast<double>(values.size()) /
                                      static_cast<double>(*std::max_element(
                                          period.begin(), period.end()))))};
        for (std::size_t j = 0; j < period[0] * period[1]; ++j) {
            double expectedSum{0.0};
            double actualSum{0.0};
            for (std::size_t i = 0; i < actuals.size(); ++i) {
                expectedSum += expected[i](j);
                actualSum += maths::CBasicStatistics::mean(actuals[i][j % period[i]]);
            }
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedSum, actualSum, 5.0 * sigma);
        }
    }

    LOG_DEBUG(<< "Overall mean error = " << maths::CBasicStatistics::mean(overallMeanError));
    BOOST_REQUIRE(maths::CBasicStatistics::mean(overallMeanError) < 1.25);
}

BOOST_AUTO_TEST_CASE(testFitTradingDaySeasonalComponents) {

    // The idea of this test is to test we correctly identify weekday/weekend
    // modulation of a daily seasonality. We randomize over the start offset
    // of the weekend to simulate different sliding window starts and check
    // we always partition where we should.

    test::CRandomNumbers rng;

    TDoubleVecVec amplitude{{0.2, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0},
                            {0.2, 0.4, 1.3, 1.2, 1.0, 1.0, 1.4}};
    maths::CSignal::TSeasonalComponentVec periods;

    maths::CSignal::TFloatMeanAccumulatorVec values;
    maths::CSignal::TMeanAccumulatorVecVec actuals;
    TSizeVec offset;

    for (std::size_t test = 0; test < 10; ++test) {
        rng.generateUniformSamples(0, 168, 1, offset);

        auto expectedComponent = [&](std::size_t i) {
            i = (168 + i - offset[0]) % 168;
            return 10.0 * amplitude[test % 2][(i % 168) / 24] *
                   std::sin(boost::math::double_constants::pi *
                            static_cast<double>(i % 24) / 24.0);
        };

        if (test % 2 == 0) {
            periods.assign({{24, offset[0], 168, {0, 48}}, {24, offset[0], 168, {48, 168}}});
        } else {
            periods.assign({{24, offset[0], 168, {0, 48}},
                            {24, offset[0], 168, {48, 168}},
                            {168, offset[0], 168, {0, 48}},
                            {168, offset[0], 168, {48, 168}}});
        }

        values.assign(336, maths::CSignal::TFloatMeanAccumulator{});
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i].add(expectedComponent(i));
        }

        maths::CSignal::fitSeasonalComponents(periods, values, actuals);

        for (std::size_t i = 0; i < values.size(); ++i) {
            double prediction{0.0};
            for (std::size_t j = 0; j < periods.size(); ++j) {
                if (periods[j].contains(i)) {
                    prediction += maths::CBasicStatistics::mean(
                        actuals[j][periods[j].offset(i)]);
                }
            }
            BOOST_REQUIRE_CLOSE(maths::CBasicStatistics::mean(values[i]), prediction, 1e-4);
        }
    }
}

BOOST_AUTO_TEST_CASE(testFitSingleSeasonalComponentRobust) {

    // Test the improvement we get using the robust approach with pepper and salt
    // outliers for a single seasonal component.

    std::size_t period{10};
    auto expected = [](std::size_t i) {
        double values[]{10.0, 11.0, 7.0,  5.0,  6.0,
                        15.0, 18.0, 19.0, 17.0, 14.0};
        return values[i % 10];
    };

    test::CRandomNumbers rng;

    maths::CSignal::TFloatMeanAccumulatorVec values;
    maths::CSignal::TMeanAccumulatorVecVec actuals;
    maths::CSignal::TMeanAccumulatorVecVec actualsRobust;
    TDoubleVec noise;
    TDoubleVec u01;

    TMeanVarAccumulator overallImprovement;

    for (std::size_t test = 0; test < 100; ++test) {

        values.assign(50, maths::CSignal::TFloatMeanAccumulator{});
        rng.generateNormalSamples(0.0, 1.0, values.size(), noise);
        rng.generateUniformSamples(0.0, 1.0, values.size(), u01);
        for (std::size_t i = 0; i < noise.size(); ++i) {
            if (u01[i] < 0.05) {
                values[i].add(0.0);
            } else if (u01[i] > 0.95) {
                values[i].add(30.0);
            } else {
                values[i].add(expected(i) + noise[i]);
            }
        }

        maths::CSignal::fitSeasonalComponents(seasonalComponentSummary({period}),
                                              values, actuals);
        maths::CSignal::fitSeasonalComponentsRobust(
            seasonalComponentSummary({period}), 0.1, values, actualsRobust);

        BOOST_REQUIRE_EQUAL(1, actuals.size());
        BOOST_REQUIRE_EQUAL(period, actuals[0].size());
        BOOST_REQUIRE_EQUAL(1, actualsRobust.size());
        BOOST_REQUIRE_EQUAL(period, actualsRobust[0].size());

        TMeanVarAccumulator meanError;
        TMeanVarAccumulator meanErrorRobust;
        double sigma{1.0 / std::sqrt(static_cast<double>(values.size()) /
                                     static_cast<double>(actualsRobust[0].size()))};
        for (std::size_t i = 0; i < period; ++i) {
            meanError.add(
                std::fabs(maths::CBasicStatistics::mean(actuals[0][i]) - expected(i)) / sigma);
            meanErrorRobust.add(
                std::fabs(maths::CBasicStatistics::mean(actualsRobust[0][i]) - expected(i)) / sigma);
        }

        overallImprovement.add(maths::CBasicStatistics::mean(meanError) -
                               maths::CBasicStatistics::mean(meanErrorRobust));
    }

    LOG_DEBUG(<< "Overall improvement = "
              << maths::CBasicStatistics::mean(overallImprovement));
    BOOST_REQUIRE(maths::CBasicStatistics::mean(overallImprovement) > 1.75);
}

BOOST_AUTO_TEST_CASE(testFitMultipleSeasonalComponentsRobust) {

    // Test the improvement we get using the robust approach with pepper and salt
    // outliers for a mixture of two seasonal components.

    TSizeVec period{10, 15};
    TPredictorVec expected{
        [](std::size_t i) {
            double values[]{10.0, 11.0, 7.0,  5.0,  6.0,
                            15.0, 18.0, 19.0, 17.0, 14.0};
            return values[i % 10];
        },
        [](std::size_t i) {
            return 10.0 + 10.0 * std::sin(boost::math::double_constants::two_pi *
                                          static_cast<double>(i) / 15.0);
        }};

    test::CRandomNumbers rng;

    maths::CSignal::TFloatMeanAccumulatorVec values;
    maths::CSignal::TMeanAccumulatorVecVec actuals;
    maths::CSignal::TMeanAccumulatorVecVec actualsRobust;
    TDoubleVec noise;
    TDoubleVec u01;

    TMeanVarAccumulator overallImprovement;

    for (std::size_t test = 0; test < 100; ++test) {

        values.assign(175, maths::CSignal::TFloatMeanAccumulator{});
        rng.generateNormalSamples(0.0, 1.0, values.size(), noise);
        rng.generateUniformSamples(0.0, 1.0, values.size(), u01);
        for (std::size_t i = 0; i < noise.size(); ++i) {
            if (u01[i] < 0.05) {
                values[i].add(0.0);
            } else if (u01[i] > 0.95) {
                values[i].add(40.0);
            } else {
                double sum{0.0};
                for (const auto& component : expected) {
                    sum += component(i);
                }
                values[i].add(sum + noise[i]);
            }
        }

        maths::CSignal::fitSeasonalComponents(seasonalComponentSummary(period),
                                              values, actuals);
        maths::CSignal::fitSeasonalComponentsRobust(seasonalComponentSummary(period),
                                                    0.1, values, actualsRobust);

        BOOST_REQUIRE_EQUAL(period.size(), actuals.size());
        BOOST_REQUIRE_EQUAL(period.size(), actualsRobust.size());
        for (std::size_t i = 0; i < period.size(); ++i) {
            BOOST_REQUIRE_EQUAL(period[i], actuals[i].size());
            BOOST_REQUIRE_EQUAL(period[i], actualsRobust[i].size());
        }

        TMeanVarAccumulator meanError;
        TMeanVarAccumulator meanErrorRobust;
        double sigma{1.0 / std::sqrt(static_cast<double>(values.size()) / 15.0)};
        for (std::size_t j = 0; j < 60; ++j) {
            double expectedSum{0.0};
            double actualSum{0.0};
            double actualRobustSum{0.0};
            for (std::size_t i = 0; i < period.size(); ++i) {
                expectedSum += expected[i](j);
                actualSum += maths::CBasicStatistics::mean(actuals[i][j % period[i]]);
                actualRobustSum +=
                    maths::CBasicStatistics::mean(actualsRobust[i][j % period[i]]);
            }
            meanError.add(std::fabs(actualSum - expectedSum) / sigma);
            meanErrorRobust.add(std::fabs(actualRobustSum - expectedSum) / sigma);
        }

        overallImprovement.add(maths::CBasicStatistics::mean(meanError) -
                               maths::CBasicStatistics::mean(meanErrorRobust));
    }

    LOG_DEBUG(<< "Overall improvement = "
              << maths::CBasicStatistics::mean(overallImprovement));
    BOOST_REQUIRE(maths::CBasicStatistics::mean(overallImprovement) > 4.0);
}

BOOST_AUTO_TEST_CASE(testSingleComponentSeasonalDecomposition) {

    // Test that we reliably find a single seasonal component.

    TSizeVec periods{7, 20, 10};
    TPredictorVec expectedComponents{
        [](std::size_t i) {
            double values[]{10.0, 5.0, 6.0, 15.0, 18.0, 17.0, 14.0};
            return values[i % 7];
        },
        [](std::size_t i) {
            return 10.0 * std::sin(boost::math::double_constants::two_pi *
                                   static_cast<double>(i) / 20.0);
        },
        [](std::size_t i) { return i % 10 == 3 ? 15.0 : 0.0; }};

    test::CRandomNumbers rng;

    maths::CSignal::TFloatMeanAccumulatorVec values;
    maths::CSignal::TMeanAccumulatorVecVec actuals;
    TDoubleVec noise;

    for (std::size_t test = 0; test < 100; ++test) {

        auto expected = expectedComponents[test % expectedComponents.size()];
        std::size_t period{periods[test % periods.size()]};

        values.assign(100, maths::CSignal::TFloatMeanAccumulator{});
        rng.generateNormalSamples(0.0, 2.0, values.size(), noise);
        for (std::size_t i = 0; i < noise.size(); ++i) {
            values[i].add(expected(i) + noise[i]);
        }

        auto decomposition = maths::CSignal::seasonalDecomposition(
            values, 0.1, {24, 7 * 24, 365 * 24});

        // We can detect additional components but must detect the real
        // component first.
        BOOST_REQUIRE(decomposition.size() >= 1);

        decomposition.resize(1);

        BOOST_REQUIRE_EQUAL(period, decomposition[0].period());
    }
}

BOOST_AUTO_TEST_CASE(testMultipleSeasonalDecomposition) {

    // Test that we reliably find a mixture of two seasonal component.

    TSizeVecVec periods{{7, 12}, {20, 13}, {7, 10}};
    TPredictorVecVec expectedComponents{
        {[](std::size_t i) {
             double values[]{10.0, 5.0, 6.0, 15.0, 18.0, 17.0, 14.0};
             return values[i % 7];
         },
         [](std::size_t i) {
             return 10.0 * std::sin(boost::math::double_constants::two_pi *
                                    static_cast<double>(i) / 12.0);
         }},
        {[](std::size_t i) {
             return 10.0 * std::sin(boost::math::double_constants::two_pi *
                                    static_cast<double>(i) / 20.0);
         },
         [](std::size_t i) {
             return 7.0 * std::sin(boost::math::double_constants::two_pi *
                                   static_cast<double>(i) / 13.0);
         }},
        {[](std::size_t i) {
             double values[]{10.0, 5.0, 6.0, 15.0, 18.0, 17.0, 14.0};
             return values[i % 7];
         },
         [](std::size_t i) { return i % 10 == 3 ? 15.0 : 0.0; }}};

    test::CRandomNumbers rng;

    maths::CSignal::TFloatMeanAccumulatorVec values;
    maths::CSignal::TMeanAccumulatorVecVec actuals;
    TDoubleVec noise;

    TMeanVarAccumulator meanError;

    for (std::size_t test = 0; test < 100; ++test) {

        auto expected = expectedComponents[test % expectedComponents.size()];
        auto period = periods[test % periods.size()];

        values.assign(100, maths::CSignal::TFloatMeanAccumulator{});
        rng.generateNormalSamples(0.0, 2.0, values.size(), noise);
        for (std::size_t i = 0; i < noise.size(); ++i) {
            double sum{0.0};
            for (const auto& component : expected) {
                sum += component(i);
            }
            values[i].add(sum + noise[i]);
        }

        auto decomposition = maths::CSignal::seasonalDecomposition(
            values, 0.1, {24, 7 * 24, 365 * 24});

        // We can detect additional components but must detect the two real
        // components first.
        BOOST_REQUIRE(decomposition.size() >= 2);

        decomposition.resize(2);
        std::sort(period.begin(), period.end());
        std::sort(decomposition.begin(), decomposition.end());
        for (std::size_t i = 0; i < 2; ++i) {
            meanError.add(std::fabs(
                static_cast<double>(decomposition[i].period() - period[i])));
        }
    }

    LOG_DEBUG(<< "mean error = " << maths::CBasicStatistics::mean(meanError));
    BOOST_REQUIRE(maths::CBasicStatistics::mean(meanError) < 0.01);
}

BOOST_AUTO_TEST_CASE(testMultipleDiurnalSeasonalDecomposition) {

    // Test seasonal decomposition with weekdays/weekend.

    test::CRandomNumbers rng;

    maths::CSignal::TFloatMeanAccumulatorVec values;
    maths::CSignal::TMeanAccumulatorVecVec actuals;
    TDoubleVec noise;
    TSizeVec offset;
    TDoubleVecVec amplitudes{{0.3, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0},
                             {0.3, 0.3, 1.3, 1.0, 1.0, 1.3, 1.2},
                             {0.3, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0}};

    for (std::size_t test = 0; test < 100; ++test) {

        rng.generateUniformSamples(0, 168, 1, offset);
        const auto& amplitude = amplitudes[test % amplitudes.size()];

        std::string expectedDecomposition{
            "[24/" + std::to_string(offset[0]) + "/168/(0, 48)," + " 24/" +
            std::to_string(offset[0]) + "/168/(48, 168)," + " 168/" +
            std::to_string(offset[0]) + "/168/(0, 48)," + " 168/" +
            std::to_string(offset[0]) + "/168/(48, 168)]"};

        auto component = [&](std::size_t i) {
            i = (168 + i - offset[0]) % 168;
            return 20.0 * amplitude[i / 24] *
                   (1.0 + std::sin(boost::math::double_constants::two_pi *
                                   static_cast<double>(i) / 24.0));
        };

        values.assign(336, maths::CSignal::TFloatMeanAccumulator{});
        rng.generateNormalSamples(0.0, 1.0, values.size(), noise);
        for (std::size_t i = 0; i < noise.size(); ++i) {
            values[i].add(component(i) + noise[i]);
        }

        auto decomposition = maths::CSignal::seasonalDecomposition(
            values, 0.0, {24, 7 * 24, 365 * 24}, {}, 1e-6);

        BOOST_REQUIRE_EQUAL(expectedDecomposition,
                            core::CContainerPrinter::print(decomposition));
    }
}

BOOST_AUTO_TEST_CASE(testTradingDayDecomposition) {

    // Test decomposing into weekdays/weekend with and without and override.

    test::CRandomNumbers rng;

    maths::CSignal::TFloatMeanAccumulatorVec values;
    maths::CSignal::TMeanAccumulatorVecVec actuals;
    TDoubleVec noise;
    TSizeVec offset;
    TDoubleVecVec modulations{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                              {0.3, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0},
                              {0.3, 0.3, 1.3, 1.0, 1.0, 1.3, 1.2},
                              {0.3, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0}};

    for (std::size_t test = 0; test < 100; ++test) {

        rng.generateUniformSamples(0, 168, 1, offset);
        const auto& modulation = modulations[test % modulations.size()];

        std::string expectedDecomposition{
            "[24/" + std::to_string(offset[0]) + "/168/(0, 48)," + " 24/" +
            std::to_string(offset[0]) + "/168/(48, 168)," + " 168/" +
            std::to_string(offset[0]) + "/168/(0, 48)," + " 168/" +
            std::to_string(offset[0]) + "/168/(48, 168)]"};

        auto component = [&](std::size_t i) {
            i = (168 + i - offset[0]) % 168;
            return 20.0 * modulation[i / 24] *
                   (1.0 + std::sin(boost::math::double_constants::two_pi *
                                   static_cast<double>(i) / 24.0));
        };

        rng.generateNormalSamples(0.0, 1.0, values.size(), noise);

        for (auto startOfWeekOverride : {maths::CSignal::TOptionalSize{},
                                         maths::CSignal::TOptionalSize{offset[0]}}) {
            values.assign(336, maths::CSignal::TFloatMeanAccumulator{});
            for (std::size_t i = 0; i < noise.size(); ++i) {
                values[i].add(component(i) + noise[i]);
            }

            auto decomposition = maths::CSignal::tradingDayDecomposition(
                values, 0.0, 168, startOfWeekOverride, 1e-6);
            if (test % 4 == 0) {
                BOOST_REQUIRE(decomposition.empty());
            } else {
                BOOST_REQUIRE_EQUAL(expectedDecomposition,
                                    core::CContainerPrinter::print(decomposition));
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testMeanNumberRepeatedValues) {

    // Test we correctly count the mean number of repeated values.

    test::CRandomNumbers rng;

    std::size_t period{20};

    maths::CSignal::TFloatMeanAccumulatorVec values;

    // Edge cases: all missing and no missing.
    values.assign(100, maths::CSignal::TFloatMeanAccumulator{});
    BOOST_REQUIRE_EQUAL(0.0, maths::CSignal::meanNumberRepeatedValues(
                                 values, maths::CSignal::seasonalComponentSummary(period)));
    for (auto& value : values) {
        value.add(1.0);
    }
    BOOST_REQUIRE_CLOSE(100.0 / static_cast<double>(period),
                        maths::CSignal::meanNumberRepeatedValues(
                            values, maths::CSignal::seasonalComponentSummary(period)),
                        1e-4);

    TDoubleVec repeats;
    TDoubleVec u01;

    for (double fraction : {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}) {

        for (const auto& time :
             {maths::CSignal::SSeasonalComponentSummary{period, 0, period, {0, period}},
              maths::CSignal::SSeasonalComponentSummary{period, 5, 2 * period, {0, period}}}) {
            values.assign(100, maths::CSignal::TFloatMeanAccumulator{});
            repeats.assign(period, 0.0);

            rng.generateUniformSamples(0.0, 1.0, values.size(), u01);

            for (std::size_t i = 0; i < values.size(); ++i) {
                if (u01[i] < fraction) {
                    if (time.contains(i)) {
                        repeats[i % period] += 1.0;
                    }
                    values[i].add(1.0);
                }
            }
            TMeanVarAccumulator expectedMeanRepeats;
            for (auto repeat : repeats) {
                if (repeat > 0.0) {
                    expectedMeanRepeats.add(repeat);
                }
            }

            BOOST_REQUIRE_CLOSE(maths::CBasicStatistics::mean(expectedMeanRepeats),
                                maths::CSignal::meanNumberRepeatedValues(values, time), 1e-4);
        }
    }
}

BOOST_AUTO_TEST_CASE(testResidualVariance) {

    // Test we get the residual variance we expect.

    test::CRandomNumbers rng;

    std::size_t period{20};

    auto component = [=](std::size_t index) {
        return 10.0 * std::sin(boost::math::double_constants::pi *
                               static_cast<double>(index % period) /
                               static_cast<double>(period));
    };

    maths::CSignal::TFloatMeanAccumulatorVec values;
    TDoubleVec noise;
    maths::CSignal::TMeanAccumulatorVecVec actualComponent(1);

    for (std::size_t test = 0; test < 10; ++test) {

        values.assign(100, maths::CSignal::TFloatMeanAccumulator{});
        actualComponent[0].assign(20, maths::CSignal::TMeanAccumulator{});
        TMeanVarAccumulator moments;

        rng.generateNormalSamples(0.0, 1.0, values.size(), noise);
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i].add(component(i) + noise[i]);
            actualComponent[0][i % period].add(component(i));
            moments.add(noise[i]);
        }

        double residualVariance{maths::CSignal::residualVariance(
            values, {maths::CSignal::seasonalComponentSummary(20)}, actualComponent)};

        BOOST_REQUIRE_CLOSE(maths::CBasicStatistics::maximumLikelihoodVariance(moments),
                            residualVariance, 1e-4);
    }
}

BOOST_AUTO_TEST_CASE(testSelectComponentSize) {

    // Test that the selected component decreases monotonically with increasing
    // noise. This is a straight variance/bias tradeoff.

    auto component = [](std::size_t i) {
        return 10.0 * std::sin(boost::math::double_constants::two_pi *
                               static_cast<double>(i) / 24.0);
    };

    test::CRandomNumbers rng;

    maths::CSignal::TFloatMeanAccumulatorVec values;
    maths::CSignal::TMeanAccumulatorVecVec actuals;
    TDoubleVec noise;

    maths::CSignal::TMeanAccumulatorVec sizes(5);

    for (std::size_t test = 0; test < 50; ++test) {
        for (std::size_t i = 0; i < 5; ++i) {

            values.assign(168, maths::CSignal::TFloatMeanAccumulator{});
            rng.generateNormalSamples(0, 4.0 * static_cast<double>(i), 168, noise);
            for (std::size_t j = 0; j < noise.size(); ++j) {
                values[j].add(component(j) + noise[j]);
            }

            std::size_t size{maths::CSignal::selectComponentSize(values, 24)};
            sizes[i].add(size);
        }
    }

    LOG_DEBUG(<< "sizes = " << core::CContainerPrinter::print(sizes));
    for (std::size_t i = 1; i < sizes.size(); ++i) {
        BOOST_TEST_REQUIRE(sizes[i] < sizes[i - 1]);
    }
}

BOOST_AUTO_TEST_SUITE_END()
