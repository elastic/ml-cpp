/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CMicTest.h"

#include <maths/CBasicStatistics.h>
#include <maths/CMic.h>
#include <maths/CTools.h>

#include <test/CRandomNumbers.h>

#include <numeric>
#include <vector>

using namespace ml;

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TSizeVec = std::vector<std::size_t>;
using TSizeVecVec = std::vector<TSizeVec>;

namespace {
TSizeVec setUnion(TSizeVec set, std::size_t element) {
    set.push_back(element);
    return set;
}

TDoubleVecVec allSubsetsOfSizeLessThan(std::size_t m, const TDoubleVec& pi) {

    std::size_t n{pi.size()};

    TDoubleVecVec result;

    TSizeVecVec boundary;
    for (std::size_t i = 0; i + 1 < n; ++i) {
        boundary.push_back({i});
        result.push_back({pi[i], pi.back()});
    }

    for (std::size_t i = 1; i + 1 < m; ++i) {

        // Generate all subsets of size i + 1.
        std::size_t size{boundary.size()};
        for (std::size_t j = 0; j < size; ++j) {
            for (std::size_t k = boundary[j].back() + 1; k + 1 < n; ++k) {
                boundary.push_back(setUnion(boundary[j], k));
            }
        }
        boundary.erase(boundary.begin(), boundary.begin() + size);

        // Add the corresponding partition to the result.
        for (const auto& set : boundary) {
            result.emplace_back();
            for (auto j : set) {
                result.back().push_back(pi[j]);
            }
            result.back().push_back(pi.back());
        }
    }
    return result;
}
}

void CMicTest::testOptimizeXAxis() {

    // Test that the dynamic program matches the brute force calculation of mutual
    // information over subsets of a master partition.

    test::CRandomNumbers rng;

    TDoubleVec samples;
    TDoubleVec noise;
    rng.generateUniformSamples(-1.0, 1.0, 2000, samples);
    rng.generateUniformSamples(-0.2, 0.2, 2000, noise);

    maths::CMic mic;
    mic.reserve(samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i) {
        mic.add(samples[i], 2.0 * samples[i] + 1.0 + noise[i]);
    }
    mic.setup();

    std::size_t k{5};
    std::size_t ck{static_cast<std::size_t>(k * mic.c())};

    TDoubleVec pi(mic.equipartitionAxis(0, ck));
    TDoubleVec q(mic.equipartitionAxis(1, k));

    TDoubleVec expected;

    for (std::size_t t = 2; t <= k; ++t) {

        double Z{static_cast<double>(samples.size())};

        double mimax{-std::numeric_limits<double>::max()};

        for (const auto& pi_ : allSubsetsOfSizeLessThan(t, pi)) {

            TDoubleVecVec p(pi_.size(), TDoubleVec(q.size(), 0.0));
            TDoubleVec px(pi_.size(), 0.0);
            TDoubleVec py(q.size(), 0.0);
            for (const auto& sample : mic.m_Samples) {
                std::ptrdiff_t i{std::upper_bound(pi_.begin(), pi_.end(), sample(0)) -
                                 pi_.begin()};
                std::ptrdiff_t j{std::upper_bound(q.begin(), q.end(), sample(1)) -
                                 q.begin()};
                p[i][j] += 1.0 / Z;
                px[i] += 1.0 / Z;
                py[j] += 1.0 / Z;
            }

            double mi{0.0};
            for (std::size_t i = 0; i < p.size(); ++i) {
                for (std::size_t j = 0; j < p[i].size(); ++j) {
                    mi += p[i][j] * maths::CTools::fastLog(p[i][j] / px[i] / py[j]);
                }
            }
            mimax = std::max(mimax, mi);
        }

        expected.push_back(mimax);
    }

    TDoubleVec actual(mic.optimizeXAxis(q, q.size(), 5));

    LOG_DEBUG(<< "MI expected = " << core::CContainerPrinter::print(expected));
    LOG_DEBUG(<< "MI actual   = " << core::CContainerPrinter::print(actual));
    for (std::size_t i = 0; i < expected.size(); ++i) {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected[i], actual[i], 1e-4);
    }
}

void CMicTest::testInvariants() {

    // Test the MIC doesn't change for shifts, scales and reflections.

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateUniformSamples(0, 6.28, 2000, samples);
    TDoubleVec noise;
    rng.generateUniformSamples(-0.1, 0.1, 2000, noise);

    double expected;
    {
        maths::CMic mic;
        mic.reserve(samples.size());
        for (std::size_t i = 0; i < samples.size(); ++i) {
            mic.add(samples[i], std::sin(samples[i]) + noise[i]);
        }
        expected = mic.compute();
        LOG_DEBUG(<< "original MICe = " << expected);
    }
    {
        maths::CMic mic;
        mic.reserve(samples.size());
        for (std::size_t i = 0; i < samples.size(); ++i) {
            mic.add(5.0 + samples[i], std::sin(samples[i]) + noise[i]);
        }
        double actual{mic.compute()};
        LOG_DEBUG(<< "shifted MICe = " << actual);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, 1e-3);
    }
    {
        maths::CMic mic;
        mic.reserve(samples.size());
        for (std::size_t i = 0; i < samples.size(); ++i) {
            mic.add(samples[i], 2.0 * (std::sin(samples[i]) + noise[i]));
        }
        double actual{mic.compute()};
        LOG_DEBUG(<< "scaled MICe = " << actual);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, 1e-3);
    }
    {
        maths::CMic mic;
        mic.reserve(samples.size());
        for (std::size_t i = 0; i < samples.size(); ++i) {
            mic.add(samples[i], 1.0 - std::sin(samples[i]) - noise[i]);
        }
        double actual{mic.compute()};
        LOG_DEBUG(<< "reflected MICe = " << actual);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, 2e-3);
    }
}

void CMicTest::testIndependent() {

    // Test for independent random variable MICe is close to zero.

    test::CRandomNumbers rng;

    TSizeVec counts{100, 500, 1000, 5000, 10000};
    TDoubleVec maximumMic{0.26, 0.14, 0.09, 0.03, 0.015};

    TDoubleVec samples;

    LOG_DEBUG(<< "Independent uniform");

    for (std::size_t t = 0; t < counts.size(); ++t) {

        rng.generateUniformSamples(0.0, 1.0, 2 * counts[t], samples);

        maths::CMic mic;
        mic.reserve(counts[t]);
        for (std::size_t i = 0; i < 2 * counts[t]; i += 2) {
            mic.add(samples[i], samples[i + 1]);
        }

        double mic_{mic.compute()};
        LOG_DEBUG(<< "MICe = " << mic_);
        CPPUNIT_ASSERT(mic_ < maximumMic[t]);
    }

    LOG_DEBUG(<< "Independent normal");

    for (std::size_t t = 0; t < counts.size(); ++t) {
        rng.generateNormalSamples(0.0, 4.0, 2 * counts[t], samples);

        maths::CMic mic;
        mic.reserve(counts[t]);
        for (std::size_t i = 0; i < 2 * counts[t]; i += 2) {
            mic.add(samples[i], samples[i + 1]);
        }

        double mic_{mic.compute()};
        LOG_DEBUG(<< "MICe = " << mic_);
        CPPUNIT_ASSERT(mic_ < maximumMic[t]);
    }
}

void CMicTest::testOneToOne() {

    // Test for one-to-one function MICe is close to 1.

    test::CRandomNumbers rng;

    TSizeVec counts{100, 500, 1000, 2000, 6000};

    TDoubleVec samples;

    LOG_DEBUG(<< "Test linear");

    for (std::size_t t = 0; t < 5; ++t) {

        TDoubleVec intercept;
        TDoubleVec slope;
        rng.generateUniformSamples(-2.0, 2.0, 1, intercept);
        rng.generateUniformSamples(-5.0, 5.0, 1, slope);
        rng.generateUniformSamples(-1.0, 1.0, counts[t], samples);

        maths::CMic mic;
        mic.reserve(counts[t]);
        for (std::size_t i = 0; i < counts[t]; ++i) {
            mic.add(samples[i], intercept[0] + slope[0] * samples[i]);
        }

        double mic_{mic.compute()};
        LOG_DEBUG(<< "MICe = " << mic_);
        CPPUNIT_ASSERT(mic_ > 0.99);
    }

    LOG_DEBUG(<< "Test sine");

    for (std::size_t t = 0; t < 5; ++t) {

        TDoubleVec period;
        rng.generateUniformSamples(0.5, 2.0, 1, period);
        rng.generateUniformSamples(0, 6.28, counts[t], samples);

        maths::CMic mic;
        mic.reserve(counts[t]);
        for (std::size_t i = 0; i < counts[t]; ++i) {
            mic.add(samples[i], std::sin(samples[i] / period[0]));
        }

        double mic_{mic.compute()};
        LOG_DEBUG(<< "MICe = " << mic_);
        CPPUNIT_ASSERT(mic_ > 0.96);
    }
}

void CMicTest::testCorrelated() {

    // Test MICe monotonically reduces with increasing noise.

    test::CRandomNumbers rng;

    TDoubleVec samples;
    TDoubleVec noise;

    for (double t = 1.0, last = 1.0; t <= 5.0; t += 1.0) {

        rng.generateUniformSamples(-1.0, 1.0, 1000, samples);
        rng.generateUniformSamples(-0.2 * t, 0.2 * t, 1000, noise);

        maths::CMic mic;
        mic.reserve(1000);
        for (std::size_t i = 0; i < 1000; i += 2) {
            mic.add(samples[i], 1.0 + 2.0 * samples[i] + noise[i]);
        }

        double current{mic.compute()};
        LOG_DEBUG(<< "MICe = " << current);
        CPPUNIT_ASSERT(current < last);
        last = current;
    }

    for (double t = 1.0, last = 1.0; t <= 5.0; t += 1.0) {

        rng.generateUniformSamples(0.0, 6.28, 1000, samples);
        rng.generateUniformSamples(-0.2 * t, 0.2 * t, 1000, noise);

        maths::CMic mic;
        mic.reserve(1000);
        for (std::size_t i = 0; i < 1000; i += 2) {
            mic.add(samples[i], 2.0 * std::sin(samples[i]) + noise[i]);
        }

        double current{mic.compute()};
        LOG_DEBUG(<< "MICe = " << current);
        CPPUNIT_ASSERT(current < last);
        last = current;
    }
}

void CMicTest::testVsMutualInformation() {

    // Test against some relationships where we can calculate the expected maximum
    // normalised mutual information.

    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

    test::CRandomNumbers rng;

    TDoubleVec samples;

    LOG_DEBUG(<< "Test grids");

    TMeanAccumulator error;

    for (std::size_t t = 0; t < 10; ++t) {
        TDoubleVec nx_;
        TDoubleVec ny_;
        rng.generateUniformSamples(3.0, 6.0, 1, nx_);
        rng.generateUniformSamples(3.0, 6.0, 1, ny_);
        std::size_t nx{static_cast<std::size_t>(nx_[0])};
        std::size_t ny{static_cast<std::size_t>(ny_[0])};

        TDoubleVec p(nx);
        rng.generateLogNormalSamples(1.0, 2.0, nx * ny, p);
        double Z{std::accumulate(p.begin(), p.end(), 0.0)};
        for (std::size_t i = 0; i < p.size(); ++i) {
            p[i] /= Z;
        }
        LOG_TRACE(<< "p = " << core::CContainerPrinter::print(p));

        double expected{0.0};
        TDoubleVec px(nx, 0.0);
        TDoubleVec py(ny, 0.0);
        for (std::size_t i = 0; i < nx; ++i) {
            for (std::size_t j = 0; j < ny; ++j) {
                px[i] += p[i * ny + j];
                py[j] += p[i * ny + j];
            }
        }
        LOG_TRACE(<< "p(x) = " << core::CContainerPrinter::print(px));
        LOG_TRACE(<< "p(y) = " << core::CContainerPrinter::print(py));

        for (std::size_t i = 0; i < nx; ++i) {
            for (std::size_t j = 0; j < ny; ++j) {
                if (p[i * ny + j] > 0.0) {
                    expected += p[i * ny + j] * std::log(p[i * ny + j] / px[i] / py[j]);
                }
            }
        }
        expected /= std::log(static_cast<double>(std::min(nx, ny)));

        maths::CMic mic;

        TDoubleVec selector;
        rng.generateUniformSamples(0.0, 1.0, 5000, selector);

        for (std::size_t i = 1; i < p.size(); ++i) {
            p[i] += p[i - 1];
        }

        TDoubleVec x;
        TDoubleVec y;
        for (std::size_t i = 0; i < 5000; ++i) {
            std::ptrdiff_t bucket{
                std::lower_bound(p.begin(), p.end(), selector[i]) - p.begin()};
            double bx{static_cast<double>(bucket / ny)};
            double by{static_cast<double>(bucket % nx)};
            rng.generateUniformSamples(bx, bx + 1.0, 1, x);
            rng.generateUniformSamples(by, by + 1.0, 1, y);
            mic.add(x[0], y[0]);
        }

        double actual{mic.compute()};
        LOG_DEBUG(<< "expected = " << expected << " actual = " << actual);

        error.add(std::fabs(actual - expected));
    }
    LOG_DEBUG(<< "Error = " << maths::CBasicStatistics::mean(error));

    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(error) < 0.12);

    LOG_DEBUG(<< "Test circle");

    TSizeVec counts{100, 500, 1000, 2000, 6000};

    TDoubleVec noise;

    for (std::size_t t = 0; t < 5; ++t) {
        TDoubleVec centre;
        TDoubleVec radius;

        rng.generateUniformSamples(-2.0, 2.0, 2, centre);
        rng.generateUniformSamples(1.0, 5.0, 1, radius);
        rng.generateUniformSamples(-0.05 * radius[0], 0.05 * radius[0], counts[t], noise);
        rng.generateUniformSamples(0, 6.28, counts[t], samples);

        maths::CMic mic;
        mic.reserve(counts[t]);
        for (std::size_t i = 0; i < counts[t]; i += 2) {
            mic.add(centre[0] + (radius[0] + noise[i]) * std::sin(samples[i]),
                    centre[1] + (radius[0] + noise[i]) * std::cos(samples[i]));
        }

        double mic_{mic.compute()};
        LOG_DEBUG(<< "MICe = " << mic_);
        CPPUNIT_ASSERT(std::fabs(mic_ - 0.6) < 0.05);
    }
}

void CMicTest::testEdgeCases() {

    // Test small number samples and constant.

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "Test small n");

    for (std::size_t t = 1; t < 10; ++t) {

        TDoubleVec samples;
        rng.generateUniformSamples(0.0, 1.0, 2 * t, samples);

        maths::CMic mic;
        mic.reserve(t);
        for (std::size_t i = 0; i < 2 * t; i += 2) {
            mic.add(samples[i], samples[i + 1]);
        }

        double mic_{mic.compute()};
        LOG_DEBUG(<< "MICe = " << mic_);
        CPPUNIT_ASSERT(mic_ < 0.65);
    }

    LOG_DEBUG(<< "Test constant");

    maths::CMic mic;
    mic.reserve(100);
    for (std::size_t i = 0; i < 100; ++i) {
        mic.add(1.0, 5.0);
    }
    double mic_{mic.compute()};
    LOG_DEBUG(<< "MICe = " << mic_);
    CPPUNIT_ASSERT_EQUAL(0.0, mic_);
}

CppUnit::Test* CMicTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CMicTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CMicTest>(
        "CMicTest::testOptimizeXAxis", &CMicTest::testOptimizeXAxis));
    suiteOfTests->addTest(new CppUnit::TestCaller<CMicTest>(
        "CMicTest::testInvariants", &CMicTest::testInvariants));
    suiteOfTests->addTest(new CppUnit::TestCaller<CMicTest>(
        "CMicTest::testIndependent", &CMicTest::testIndependent));
    suiteOfTests->addTest(new CppUnit::TestCaller<CMicTest>(
        "CMicTest::testOneToOne", &CMicTest::testOneToOne));
    suiteOfTests->addTest(new CppUnit::TestCaller<CMicTest>(
        "CMicTest::testCorrelated", &CMicTest::testCorrelated));
    suiteOfTests->addTest(new CppUnit::TestCaller<CMicTest>(
        "CMicTest::testVsMutualInformation", &CMicTest::testVsMutualInformation));
    suiteOfTests->addTest(new CppUnit::TestCaller<CMicTest>(
        "CMicTest::testEdgeCases", &CMicTest::testEdgeCases));

    return suiteOfTests;
}
