/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */

#include "CSignalTest.h"

#include <core/CLogger.h>
#include <core/CoreTypes.h>

#include <maths/CSignal.h>

#include <test/CRandomNumbers.h>

#include <boost/math/constants/constants.hpp>

using namespace ml;

namespace {

using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;

std::string print(const maths::CSignal::TComplexVec& f) {
    std::ostringstream result;
    for (std::size_t i = 0u; i < f.size(); ++i) {
        LOG_DEBUG(f[i].real() << " + " << f[i].imag() << 'i');
    }
    return result.str();
}

void bruteForceDft(maths::CSignal::TComplexVec& f, double sign) {
    maths::CSignal::TComplexVec result(f.size(), maths::CSignal::TComplex(0.0, 0.0));
    for (std::size_t k = 0u; k < f.size(); ++k) {
        for (std::size_t n = 0u; n < f.size(); ++n) {
            double t = -sign * boost::math::double_constants::two_pi * static_cast<double>(k * n) / static_cast<double>(f.size());
            result[k] += maths::CSignal::TComplex(std::cos(t), std::sin(t)) * f[n];
        }
        if (sign < 0.0) {
            result[k] /= static_cast<double>(f.size());
        }
    }
    f.swap(result);
}
}

void CSignalTest::testFFTVersusOctave() {
    LOG_DEBUG("+------------------------------------+");
    LOG_DEBUG("|  CSignalTest::testFFTVersusOctave  |");
    LOG_DEBUG("+------------------------------------+");

    // Test versus values calculated using octave fft.

    double x[][20] = {{2555.33, 1451.79, 465.60,  4394.83, -1553.24, -2772.07, -3977.73, 2249.31, -2006.04, 3540.84,
                       4271.63, 4648.81, -727.90, 2285.24, 3129.56,  -3596.79, -1968.66, 3795.18, 1627.84,  228.40},
                      {4473.77,  -4815.63, -818.38,  -1953.72, -2323.39, -3007.25, 4444.24, 435.21,   3613.32, 3471.37,
                       -1735.72, 2560.82,  -2383.29, -2370.23, -4921.04, -541.25,  1516.69, -2028.42, 3981.02, 3156.88}};

    maths::CSignal::TComplexVec fx;
    for (std::size_t i = 0u; i < 20; ++i) {
        fx.push_back(maths::CSignal::TComplex(x[0][i], x[1][i]));
    }

    LOG_DEBUG("*** Power of 2 Length ***");
    {
        double expected[][2] = {// length 2
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

        for (std::size_t i = 0u, l = 2u; l < fx.size(); i += l, l <<= 1) {
            LOG_DEBUG("Testing length " << l);

            maths::CSignal::TComplexVec actual(fx.begin(), fx.begin() + l);
            maths::CSignal::fft(actual);
            LOG_DEBUG(print(actual));

            double error = 0.0;
            for (std::size_t j = 0u; j < l; ++j) {
                error += std::abs(actual[j] - maths::CSignal::TComplex(expected[i + j][0], expected[i + j][1]));
            }
            error /= static_cast<double>(l);
            LOG_DEBUG("error = " << error);
            CPPUNIT_ASSERT(error < 0.2);
        }
    }

    LOG_DEBUG("*** Arbitrary Length ***");
    {
        double expected[][2] = {{18042.0, 755.0},    {961.0, 5635.6},    {-5261.8, 7542.2},   {-12814.0, 2250.2}, {-8248.5, 6620.5},
                                {-21626.0, 3570.6},  {6551.5, -12732.0}, {6009.5, 10622.0},   {9954.0, -1224.2},  {-2871.5, 7073.6},
                                {-14409.0, 10939.0}, {13682.0, 25304.0}, {-10468.0, -6338.5}, {6506.0, 6283.3},   {32665.0, 5127.7},
                                {3190.7, 4323.4},    {-6988.7, -3865.0}, {-3881.4, 4360.8},   {46434.0, 20556.0}, {-6319.6, -7329.0}};

        maths::CSignal::TComplexVec actual(fx.begin(), fx.end());
        maths::CSignal::fft(actual);
        double error = 0.0;
        for (std::size_t j = 0u; j < actual.size(); ++j) {
            error += std::abs(actual[j] - maths::CSignal::TComplex(expected[j][0], expected[j][1]));
        }
        error /= static_cast<double>(actual.size());
        LOG_DEBUG("error = " << error);
        CPPUNIT_ASSERT(error < 0.2);
    }
}

void CSignalTest::testIFFTVersusOctave() {
    LOG_DEBUG("+-------------------------------------+");
    LOG_DEBUG("|  CSignalTest::testIFFTVersusOctave  |");
    LOG_DEBUG("+-------------------------------------+");

    // Test versus values calculated using octave ifft.

    double x[][20] = {{2555.33, 1451.79, 465.60,  4394.83, -1553.24, -2772.07, -3977.73, 2249.31, -2006.04, 3540.84,
                       4271.63, 4648.81, -727.90, 2285.24, 3129.56,  -3596.79, -1968.66, 3795.18, 1627.84,  228.40},
                      {4473.77,  -4815.63, -818.38,  -1953.72, -2323.39, -3007.25, 4444.24, 435.21,   3613.32, 3471.37,
                       -1735.72, 2560.82,  -2383.29, -2370.23, -4921.04, -541.25,  1516.69, -2028.42, 3981.02, 3156.88}};

    maths::CSignal::TComplexVec fx;
    for (std::size_t i = 0u; i < 20; ++i) {
        fx.push_back(maths::CSignal::TComplex(x[0][i], x[1][i]));
    }

    LOG_DEBUG("*** Powers of 2 Length ***");
    {
        double expected[][2] = {// length 2
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

        for (std::size_t i = 0u, l = 2u; l < fx.size(); i += l, l <<= 1) {
            LOG_DEBUG("Testing length " << l);

            maths::CSignal::TComplexVec actual(fx.begin(), fx.begin() + l);
            maths::CSignal::ifft(actual);
            LOG_DEBUG(print(actual));

            double error = 0.0;
            for (std::size_t j = 0u; j < l; ++j) {
                error += std::abs(actual[j] - maths::CSignal::TComplex(expected[i + j][0], expected[i + j][1]));
            }
            error /= static_cast<double>(l);
            LOG_DEBUG("error = " << error);
            CPPUNIT_ASSERT(error < 0.01);
        }
    }
}

void CSignalTest::testFFTRandomized() {
    LOG_DEBUG("+----------------------------------+");
    LOG_DEBUG("|  CSignalTest::testFFTRandomized  |");
    LOG_DEBUG("+----------------------------------+");

    // Test on randomized input versus brute force.

    test::CRandomNumbers rng;

    TDoubleVec components;
    rng.generateUniformSamples(-100000.0, 100000.0, 20000, components);

    TSizeVec lengths;
    rng.generateUniformSamples(2, 100, 1000, lengths);

    for (std::size_t i = 0u, j = 0u; i < lengths.size() && j + 2 * lengths[i] < components.size(); ++i, j += 2 * lengths[i]) {
        maths::CSignal::TComplexVec expected;
        for (std::size_t k = 0u; k < lengths[i]; ++k) {
            expected.push_back(maths::CSignal::TComplex(components[j + 2 * k], components[j + 2 * k + 1]));
        }
        maths::CSignal::TComplexVec actual(expected);

        bruteForceDft(expected, +1.0);
        maths::CSignal::fft(actual);

        double error = 0.0;
        for (std::size_t k = 0u; k < actual.size(); ++k) {
            error += std::abs(actual[k] - expected[k]);
        }

        if (i % 5 == 0 || error >= 1e-5) {
            LOG_DEBUG("length = " << lengths[i] << ", error  = " << error);
        }
        CPPUNIT_ASSERT(error < 1e-5);
    }
}

void CSignalTest::testIFFTRandomized() {
    LOG_DEBUG("+-----------------------------------+");
    LOG_DEBUG("|  CSignalTest::testIFFTRandomized  |");
    LOG_DEBUG("+-----------------------------------+");

    // Test on randomized input versus brute force.

    test::CRandomNumbers rng;

    TDoubleVec components;
    rng.generateUniformSamples(-100000.0, 100000.0, 20000, components);

    TSizeVec lengths;
    rng.generateUniformSamples(2, 100, 1000, lengths);

    for (std::size_t i = 0u, j = 0u; i < lengths.size() && j + 2 * lengths[i] < components.size(); ++i, j += 2 * lengths[i]) {
        maths::CSignal::TComplexVec expected;
        for (std::size_t k = 0u; k < lengths[i]; ++k) {
            expected.push_back(maths::CSignal::TComplex(components[j + 2 * k], components[j + 2 * k + 1]));
        }
        maths::CSignal::TComplexVec actual(expected);

        bruteForceDft(expected, -1.0);
        maths::CSignal::ifft(actual);

        double error = 0.0;
        for (std::size_t k = 0u; k < actual.size(); ++k) {
            error += std::abs(actual[k] - expected[k]);
        }

        if (i % 5 == 0 || error >= 1e-5) {
            LOG_DEBUG("length = " << lengths[i] << ", error  = " << error);
        }
        CPPUNIT_ASSERT(error < 1e-5);
    }
}

void CSignalTest::testFFTIFFTIdempotency() {
    LOG_DEBUG("+---------------------------------------+");
    LOG_DEBUG("|  CSignalTest::testFFTIFFTIdempotency  |");
    LOG_DEBUG("+---------------------------------------+");

    // Test on randomized input that x = F(F^-1(x)).

    test::CRandomNumbers rng;

    TDoubleVec components;
    rng.generateUniformSamples(-100000.0, 100000.0, 20000, components);

    TSizeVec lengths;
    rng.generateUniformSamples(2, 100, 1000, lengths);

    for (std::size_t i = 0u, j = 0u; i < lengths.size() && j + 2 * lengths[i] < components.size(); ++i, j += 2 * lengths[i]) {
        maths::CSignal::TComplexVec expected;
        for (std::size_t k = 0u; k < lengths[i]; ++k) {
            expected.push_back(maths::CSignal::TComplex(components[j + 2 * k], components[j + 2 * k + 1]));
        }

        maths::CSignal::TComplexVec actual(expected);
        maths::CSignal::fft(actual);
        maths::CSignal::ifft(actual);

        double error = 0.0;
        for (std::size_t k = 0u; k < actual.size(); ++k) {
            error += std::abs(actual[k] - expected[k]);
        }

        if (i % 5 == 0 || error >= 1e-5) {
            LOG_DEBUG("length = " << lengths[i] << ", error  = " << error);
        }
        CPPUNIT_ASSERT(error < 1e-5);
    }
}

void CSignalTest::testAutocorrelations() {
    LOG_DEBUG("+-------------------------------------+");
    LOG_DEBUG("|  CSignalTest::testAutocorrelations  |");
    LOG_DEBUG("+-------------------------------------+");

    test::CRandomNumbers rng;

    TSizeVec sizes;
    rng.generateUniformSamples(10, 30, 100, sizes);

    for (std::size_t t = 0u; t < sizes.size(); ++t) {
        TDoubleVec values_;
        rng.generateUniformSamples(-10.0, 10.0, sizes[t], values_);

        maths::CSignal::TFloatMeanAccumulatorVec values(sizes[t]);
        for (std::size_t i = 0u; i < values_.size(); ++i) {
            values[i].add(values_[i]);
        }

        TDoubleVec expected;
        for (std::size_t offset = 1; offset < values.size(); ++offset) {
            expected.push_back(maths::CSignal::autocorrelation(offset, values));
        }

        TDoubleVec actual;
        maths::CSignal::autocorrelations(values, actual);

        if (t % 10 == 0) {
            LOG_DEBUG("expected = " << core::CContainerPrinter::print(expected));
            LOG_DEBUG("actual   = " << core::CContainerPrinter::print(actual));
        }
        CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expected), core::CContainerPrinter::print(actual));
    }
}

CppUnit::Test* CSignalTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CSignalTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CSignalTest>("CSignalTest::testFFTVersusOctave", &CSignalTest::testFFTVersusOctave));
    suiteOfTests->addTest(new CppUnit::TestCaller<CSignalTest>("CSignalTest::testIFFTVersusOctave", &CSignalTest::testIFFTVersusOctave));
    suiteOfTests->addTest(new CppUnit::TestCaller<CSignalTest>("CSignalTest::testFFTRandomized", &CSignalTest::testFFTRandomized));
    suiteOfTests->addTest(new CppUnit::TestCaller<CSignalTest>("CSignalTest::testIFFTRandomized", &CSignalTest::testIFFTRandomized));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CSignalTest>("CSignalTest::testFFTIFFTIdempotency", &CSignalTest::testFFTIFFTIdempotency));
    suiteOfTests->addTest(new CppUnit::TestCaller<CSignalTest>("CSignalTest::testAutocorrelations", &CSignalTest::testAutocorrelations));

    return suiteOfTests;
}
