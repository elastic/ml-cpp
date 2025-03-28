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

#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CLogger.h>
#include <core/CMemoryDef.h>
#include <core/CStopWatch.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CBasicStatisticsPersist.h>
#include <maths/common/CQuantileSketch.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <numeric>

BOOST_AUTO_TEST_SUITE(CQuantileSketchTest)

using namespace ml;

namespace {

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TMeanAccumulator = maths::common::CBasicStatistics::SSampleMean<double>::TAccumulator;

template<typename SKETCH>
void testSketch(SKETCH sketch,
                TDoubleVec& samples,
                double maxBias,
                double maxError,
                TMeanAccumulator& meanBias,
                TMeanAccumulator& meanError) {
    sketch = std::for_each(samples.begin(), samples.end(), sketch);

    std::size_t N = samples.size();
    std::sort(samples.begin(), samples.end());

    TMeanAccumulator bias;
    TMeanAccumulator error;
    for (std::size_t i = 1; i < 20; ++i) {
        double q = static_cast<double>(i) / 20.0;
        double xq = samples[static_cast<std::size_t>(static_cast<double>(N) * q)];
        double sq;
        BOOST_TEST_REQUIRE(sketch.quantile(100.0 * q, sq));
        bias.add(xq - sq);
        error.add(std::fabs(xq - sq));
    }

    double min;
    double max;
    sketch.quantile(0.0, min);
    sketch.quantile(100.0, max);
    double scale = max - min;

    LOG_TRACE(<< "bias = " << maths::common::CBasicStatistics::mean(bias)
              << ", error " << maths::common::CBasicStatistics::mean(error));
    BOOST_TEST_REQUIRE(std::fabs(maths::common::CBasicStatistics::mean(bias)) < maxBias);
    BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(error) < maxError);

    meanBias += maths::common::CBasicStatistics::momentsAccumulator(
        maths::common::CBasicStatistics::count(bias),
        maths::common::CBasicStatistics::mean(bias) / scale);
    meanError += maths::common::CBasicStatistics::momentsAccumulator(
        maths::common::CBasicStatistics::count(error),
        maths::common::CBasicStatistics::mean(error) / scale);
}
}

BOOST_AUTO_TEST_CASE(testOrderAndDeduplicate) {

    maths::common::CQuantileSketch sketch{15};
    sketch.add(1.0);
    sketch.add(3.0);
    sketch.add(1.0);
    sketch.add(1.0);
    sketch.add(5.0);
    sketch.add(4.0);
    sketch.add(5.0);
    sketch.add(4.0);
    sketch.order();
    sketch.deduplicate(sketch.m_Knots.begin(), sketch.m_Knots.end());
    LOG_DEBUG(<< "sketch = " << sketch);
    BOOST_REQUIRE_EQUAL("[(1, 3), (3, 1), (4, 2), (5, 2)]", sketch.print());

    sketch.add(3.0);
    sketch.add(1.0);
    sketch.add(0.0);
    sketch.add(9.0);
    sketch.order();
    sketch.deduplicate(sketch.m_Knots.begin(), sketch.m_Knots.end());
    LOG_DEBUG(<< "sketch = " << sketch);
    BOOST_REQUIRE_EQUAL("[(0, 1), (1, 4), (3, 2), (4, 2), (5, 2), (9, 1)]",
                        sketch.print());

    sketch.add(9.0);
    sketch.add(9.0);
    sketch.add(9.0);
    sketch.order();
    sketch.deduplicate(sketch.m_Knots.begin(), sketch.m_Knots.end());
    LOG_DEBUG(<< "sketch = " << sketch);
    BOOST_REQUIRE_EQUAL("[(0, 1), (1, 4), (3, 2), (4, 2), (5, 2), (9, 4)]",
                        sketch.print());
}

BOOST_AUTO_TEST_CASE(testAdd) {
    maths::common::CQuantileSketch sketch{5};

    // Test adding a point.
    sketch.add(1.2);
    BOOST_TEST_REQUIRE(sketch.checkInvariants());

    // Test adding a weighted point.
    sketch.add(0.9, 3.0);
    BOOST_TEST_REQUIRE(sketch.checkInvariants());

    // Test add via operator().
    TDoubleVec x{1.8, 2.1};
    sketch = std::for_each(x.begin(), x.end(), sketch);
    BOOST_TEST_REQUIRE(sketch.checkInvariants());

    LOG_DEBUG(<< "sketch = " << sketch);
    BOOST_REQUIRE_EQUAL(6.0, sketch.count());
    BOOST_REQUIRE_EQUAL("[(1.2, 1), (0.9, 3), (1.8, 1), (2.1, 1)]", sketch.print());
}

BOOST_AUTO_TEST_CASE(testReduce) {
    {
        maths::common::CQuantileSketch sketch{6};

        // Test duplicate points.

        TDoubleVecVec points{{5.0, 1.0}, {0.4, 2.0}, {0.4, 1.0}, {1.0, 1.0},
                             {1.2, 2.0}, {1.2, 1.5}, {5.0, 1.0}};
        for (const auto& point : points) {
            sketch.add(point[0], point[1]);
            BOOST_TEST_REQUIRE(sketch.checkInvariants());
        }

        LOG_DEBUG(<< "sketch = " << sketch);
        BOOST_REQUIRE_EQUAL("[(0.4, 3), (1, 1), (1.2, 3.5), (5, 2)]", sketch.print());

        // Regular compress (merging two point).

        sketch.add(0.1);
        sketch.add(0.2);
        sketch.add(0.0);
        LOG_DEBUG(<< "sketch = " << sketch);
        BOOST_REQUIRE_EQUAL("[(0, 1), (0.15, 2), (0.4, 3), (1, 1), (1.2, 3.5), (5, 2)]",
                            sketch.print());
    }
    {
        // Multiple points compressed at once.

        maths::common::CQuantileSketch sketch{30};

        for (std::size_t i = 0; i <= 30; ++i) {
            sketch.add(static_cast<double>(i));
            BOOST_TEST_REQUIRE(sketch.checkInvariants());
        }
        LOG_DEBUG(<< "sketch = " << sketch);
        BOOST_REQUIRE_EQUAL("[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1),"
                            " (5.5, 2), (7, 1), (8, 1), (9, 1), (10, 1),"
                            " (11, 1), (12, 1), (13.5, 2), (15, 1), (16, 1),"
                            " (17, 1), (18, 1), (19, 1), (20, 1), (21, 1),"
                            " (22.5, 2), (24, 1), (25, 1), (26, 1), (27, 1),"
                            " (28, 1), (29, 1), (30, 1)]",
                            sketch.print());
    }
    {
        // Test the quantiles are reasonable at a compression ratio of 2:1.

        TDoubleVec points{1.0,  2.0,  40.0, 13.0, 5.0,  6.0,  4.0,
                          7.0,  15.0, 17.0, 19.0, 44.0, 42.0, 3.0,
                          46.0, 48.0, 50.0, 21.0, 23.0, 52.0};
        TDoubleVec cdf{5.0,  10.0, 15.0, 20.0, 25.0, 30.0, 35.0,
                       40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0,
                       75.0, 80.0, 85.0, 90.0, 95.0, 100.0};

        maths::common::CQuantileSketch sketch{10};
        for (std::size_t i = 0; i < points.size(); ++i) {
            sketch.add(points[i]);
            BOOST_TEST_REQUIRE(sketch.checkInvariants());
            if ((i + 1) % 5 == 0) {
                LOG_DEBUG(<< "sketch = " << sketch);
            }
        }

        std::sort(points.begin(), points.end());
        TMeanAccumulator error;
        for (std::size_t i = 0; i < cdf.size(); ++i) {
            double x;
            BOOST_TEST_REQUIRE(sketch.quantile(cdf[i], x));
            LOG_DEBUG(<< "expected quantile = " << points[i] << ", actual quantile = " << x);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(points[i], x, 10.0);
            error.add(std::fabs(points[i] - x));
        }
        LOG_DEBUG(<< "error = " << maths::common::CBasicStatistics::mean(error));
        BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(error) < 1.5);
    }
}

BOOST_AUTO_TEST_CASE(testMerge) {
    {
        // Simple merge no reduction.

        maths::common::CQuantileSketch sketch1{20};
        maths::common::CQuantileSketch sketch2{10};

        sketch1.add(2.0);
        sketch1.add(1.0);
        sketch1.add(3.1, 2.0);
        sketch1.add(1.1);
        sketch1.add(1.0, 1.5);
        sketch2.add(3.0);
        sketch2.add(5.1);
        sketch2.add(1.0, 1.1);
        sketch2.add(5.1);

        sketch1 += sketch2;
        LOG_DEBUG(<< "merged sketch = " << sketch1);
        BOOST_REQUIRE_EQUAL("[(1, 3.6), (1.1, 1), (2, 1), (3, 1), (3.1, 2), (5.1, 2)]",
                            sketch1.print());
    }

    {
        // Test the quantiles are reasonable at a compression ratio of 2:1.

        TDoubleVec points{1.0,  2.0,  40.0, 13.0, 5.0,  6.0,  4.0,
                          7.0,  15.0, 17.0, 19.0, 44.0, 42.0, 3.0,
                          46.0, 48.0, 50.0, 21.0, 23.0, 52.0};
        TDoubleVec cdf{5.0,  10.0, 15.0, 20.0, 25.0, 30.0, 35.0,
                       40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0,
                       75.0, 80.0, 85.0, 90.0, 95.0, 100.0};

        maths::common::CQuantileSketch sketch1{10};
        maths::common::CQuantileSketch sketch2{10};
        for (std::size_t i = 0; i < points.size(); i += 2) {
            sketch1.add(points[i]);
            sketch2.add(points[i + 1]);
        }
        LOG_DEBUG(<< "sketch 1 = " << sketch1);
        LOG_DEBUG(<< "sketch 2 = " << sketch2);

        maths::common::CQuantileSketch sketch3 = sketch1 + sketch2;
        LOG_DEBUG(<< "merged sketch = " << sketch3);

        std::sort(points.begin(), points.end());
        TMeanAccumulator error;
        for (std::size_t i = 0; i < cdf.size(); ++i) {
            double x;
            BOOST_TEST_REQUIRE(sketch3.quantile(cdf[i], x));
            LOG_DEBUG(<< "expected quantile = " << points[i] << ", actual quantile = " << x);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(points[i], x, 10.0);
            error.add(std::fabs(points[i] - x));
        }
        LOG_DEBUG(<< "error = " << maths::common::CBasicStatistics::mean(error));
        BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(error) < 1.8);
    }
}

BOOST_AUTO_TEST_CASE(testDuplicates) {

    // Test we never have duplicate values in the sketch even for a data set
    // with many duplicates.

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateUniformSamples(0.0, 30.0, 100000, samples);

    auto testSketch = [&](auto& sketch) {
        for (auto sample : samples) {
            sketch.add(std::floor(sample));
            BOOST_TEST_REQUIRE(sketch.checkInvariants());
        }
    };

    LOG_DEBUG(<< "sketch");
    maths::common::CQuantileSketch sketch{40};
    testSketch(sketch);

    LOG_DEBUG(<< "fast sketch");
    maths::common::CFastQuantileSketch fastSketch{40};
    testSketch(sketch);
}

BOOST_AUTO_TEST_CASE(testMedian) {

    LOG_DEBUG(<< "**** Exact ****");

    // Test that the quantiles have zero error when the number of distinct
    // values is less than the sketch size.

    {
        maths::common::CQuantileSketch sketch{10};

        double median;
        BOOST_TEST_REQUIRE(!sketch.quantile(50.0, median));

        sketch.add(1.0);
        BOOST_TEST_REQUIRE(sketch.quantile(50.0, median));
        BOOST_REQUIRE_EQUAL(1.0, median);

        // [1.0, 2.0]
        sketch.add(2.0);
        BOOST_TEST_REQUIRE(sketch.quantile(50.0, median));
        BOOST_REQUIRE_EQUAL(1.5, median);

        // [1.0, 2.0, 3.0]
        sketch.add(3.0);
        BOOST_TEST_REQUIRE(sketch.quantile(50.0, median));
        BOOST_REQUIRE_EQUAL(2.0, median);

        // [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        sketch.add(8.0);
        sketch.add(4.0);
        sketch.add(7.0);
        sketch.add(6.0);
        sketch.add(9.0);
        sketch.add(5.0);
        BOOST_TEST_REQUIRE(sketch.quantile(50.0, median));
        BOOST_REQUIRE_EQUAL(5.0, median);
    }

    test::CRandomNumbers rng;

    TDoubleVec samples;

    for (std::size_t n = 1; n <= 200; ++n) {
        maths::common::CQuantileSketch sketch{220};

        rng.generateLogNormalSamples(0.0, 3.0, n, samples);

        for (auto sample : samples) {
            sketch.add(sample);
        }

        double expectedMedian{maths::common::CBasicStatistics::median(samples)};
        double actualMedian;
        sketch.quantile(50.0, actualMedian);
        if (n % 10 == 0) {
            LOG_DEBUG(<< "expectedMedian = " << expectedMedian
                      << ", actualMedian = " << actualMedian);
        }

        BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedMedian, actualMedian, 1e-6);
    }

    LOG_DEBUG(<< "**** Approximate ****");
    {
        TMeanAccumulator bias;
        TMeanAccumulator error;
        for (std::size_t t = 0; t < 500; ++t) {
            rng.generateUniformSamples(0.0, 100.0, 501, samples);
            maths::common::CQuantileSketch sketch{30};
            sketch = std::for_each(samples.begin(), samples.end(), sketch);
            std::sort(samples.begin(), samples.end());
            double expectedMedian = samples[250];
            double actualMedian;
            sketch.quantile(50.0, actualMedian);
            BOOST_TEST_REQUIRE(std::fabs(actualMedian - expectedMedian) < 6.7);
            bias.add(actualMedian - expectedMedian);
            error.add(std::fabs(actualMedian - expectedMedian));
        }

        LOG_DEBUG(<< "bias  = " << maths::common::CBasicStatistics::mean(bias));
        LOG_DEBUG(<< "error = " << maths::common::CBasicStatistics::mean(error));
        BOOST_TEST_REQUIRE(std::fabs(maths::common::CBasicStatistics::mean(bias)) < 0.025);
        BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(error) < 0.25);
    }
}

BOOST_AUTO_TEST_CASE(testMad) {

    // Check some edge cases and also accuracy verses exact values
    // some random data.

    test::CRandomNumbers rng;

    double mad = 0.0;

    {
        maths::common::CQuantileSketch sketch{10};

        BOOST_TEST_REQUIRE(!sketch.mad(mad));

        sketch.add(1.0);
        BOOST_TEST_REQUIRE(sketch.mad(mad));
        LOG_DEBUG(<< "MAD = " << mad);
        BOOST_REQUIRE_EQUAL(0.0, mad);

        sketch.add(2.0);
        BOOST_TEST_REQUIRE(sketch.mad(mad));
        LOG_DEBUG(<< "MAD = " << mad);
        BOOST_REQUIRE_EQUAL(0.5, mad);
    }
    {
        TDoubleVec samples;
        TMeanAccumulator error;

        for (std::size_t t = 0; t < 500; ++t) {
            rng.generateNormalSamples(10.0, 10.0, 101, samples);

            maths::common::CQuantileSketch sketch{20};

            for (auto sample : samples) {
                sketch.add(sample);
            }
            BOOST_TEST_REQUIRE(sketch.mad(mad));

            std::nth_element(samples.begin(), samples.begin() + 50, samples.end());
            double median = samples[50];
            for (auto& sample : samples) {
                sample = std::fabs(sample - median);
            }
            std::nth_element(samples.begin(), samples.begin() + 50, samples.end());
            double expectedMad = samples[50];

            if (t % 50 == 0) {
                LOG_DEBUG(<< "expected MAD = " << expectedMad << " actual MAD = " << mad);
            }

            error.add(std::fabs(mad - expectedMad));
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedMad, mad, 0.2 * expectedMad);
        }

        LOG_DEBUG(<< "error = " << maths::common::CBasicStatistics::mean(error));
        BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(error) < 0.082);
    }
}

BOOST_AUTO_TEST_CASE(testPropagateForwardByTime) {

    // Check that the count is reduced and the invariants still hold.

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateUniformSamples(0.0, 20.0, 100, samples);

    maths::common::CQuantileSketch sketch{20};
    sketch = std::for_each(samples.begin(), samples.end(), sketch);

    sketch.age(0.9);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(90.0, sketch.count(), 1e-6);
    BOOST_TEST_REQUIRE(sketch.checkInvariants());
}

BOOST_AUTO_TEST_CASE(testQuantileAccuracy) {

    // Test on a variety of random data sets versus the corresponding
    // quantile in the raw data.

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "**** Uniform ****");
    {
        auto testUniform = [rng](const auto& sketch, double maxMaxBias, double maxMeanBias,
                                 double maxMaxError, double maxMeanError) mutable {
            TMeanAccumulator meanBias;
            TMeanAccumulator meanError;
            for (double t = 1.0; t <= 50.0; t += 1.0) {
                TDoubleVec samples;
                rng.generateUniformSamples(0.0, 20.0 * t, 1000, samples);
                testSketch(sketch, samples, maxMaxBias * t, maxMaxError * t,
                           meanBias, meanError);
            }
            LOG_DEBUG(<< "mean bias = " << maths::common::CBasicStatistics::mean(meanBias)
                      << ", mean error = " << maths::common::CBasicStatistics::mean(meanError));
            BOOST_TEST_REQUIRE(std::fabs(maths::common::CBasicStatistics::mean(meanBias)) <
                               maxMeanBias);
            BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(meanError) < maxMeanError);
        };

        maths::common::CQuantileSketch sketch{20};
        maths::common::CFastQuantileSketch fastSketch{20};
        LOG_DEBUG(<< "** sketch **");
        testUniform(sketch, 0.1, 0.0005, 0.12, 0.003);
        LOG_DEBUG(<< "** fast sketch **");
        testUniform(fastSketch, 0.1, 0.0005, 0.4, 0.01);
    }

    LOG_DEBUG(<< "**** Normal ****");
    {
        auto testNormal = [rng](const auto& sketch, double maxMaxBias, double maxMeanBias,
                                double maxMaxError, double maxMeanError) mutable {
            TMeanAccumulator meanBias;
            TMeanAccumulator meanError;
            for (double t = 1.0; t <= 50.0; t += 1.0) {
                TDoubleVec samples;
                rng.generateNormalSamples(20.0 * (t - 1.0), 20.0 * t, 1000, samples);
                testSketch(sketch, samples, maxMaxBias * t, maxMaxError * t,
                           meanBias, meanError);
            }
            LOG_DEBUG(<< "mean bias = " << maths::common::CBasicStatistics::mean(meanBias)
                      << ", mean error = " << maths::common::CBasicStatistics::mean(meanError));
            BOOST_TEST_REQUIRE(std::fabs(maths::common::CBasicStatistics::mean(meanBias)) <
                               maxMeanBias);
            BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(meanError) < maxMeanError);
        };

        maths::common::CQuantileSketch sketch{20};
        maths::common::CFastQuantileSketch fastSketch{20};
        LOG_DEBUG(<< "** sketch **");
        testNormal(sketch, 0.06, 0.0005, 0.1, 0.002);
        LOG_DEBUG(<< "** fast sketch **");
        testNormal(fastSketch, 0.07, 0.0005, 0.4, 0.012);
    }

    LOG_DEBUG(<< "**** Log-Normal ****");
    {
        auto testLogNormal = [rng](const auto& sketch, double maxMaxBias, double maxMeanBias,
                                   double maxMaxError, double maxMeanError) mutable {
            TMeanAccumulator meanBias;
            TMeanAccumulator meanError;
            for (double t = 1.0; t <= 50.0; t += 1.0) {
                TDoubleVec samples;
                rng.generateLogNormalSamples(0.03 * (t - 1.0), 0.12 * t, 1000, samples);
                testSketch(sketch, samples, maxMaxBias * t, maxMaxError * t,
                           meanBias, meanError);
            }
            LOG_DEBUG(<< "mean bias = " << maths::common::CBasicStatistics::mean(meanBias)
                      << ", mean error = " << maths::common::CBasicStatistics::mean(meanError));
            BOOST_TEST_REQUIRE(std::fabs(maths::common::CBasicStatistics::mean(meanBias)) <
                               maxMeanBias);
            BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(meanError) < maxMeanError);
        };

        maths::common::CQuantileSketch sketch{20};
        maths::common::CFastQuantileSketch fastSketch{20};
        LOG_DEBUG(<< "** sketch **");
        testLogNormal(sketch, 0.05, 0.0002, 0.1, 0.0004);
        LOG_DEBUG(<< "** fast sketch **");
        testLogNormal(fastSketch, 1.0, 0.002, 1.1, 0.005);
    }

    LOG_DEBUG(<< "**** Mixture ****");
    {
        auto testMixture = [rng](const auto& sketch, double maxBias, double maxError,
                                 double maxMeanBias, double maxMeanError) mutable {
            TMeanAccumulator meanBias;
            TMeanAccumulator meanError;
            for (double t = 1.0; t < 50.0; ++t) {
                TDoubleVecVec samples_(4);
                rng.generateNormalSamples(10.0 * (t - 1.0), 20.0 * t, 400, samples_[0]);
                rng.generateNormalSamples(20.0 * (t - 1.0), 20.0 * t, 600, samples_[1]);
                rng.generateNormalSamples(100.0 * (t - 1.0), 40.0 * t, 400, samples_[2]);
                rng.generateUniformSamples(500.0 * (t - 1.0), 550.0 * t, 600, samples_[3]);
                TDoubleVec samples;
                for (std::size_t i = 0; i < 4; ++i) {
                    samples.insert(samples.end(), samples_[i].begin(),
                                   samples_[i].end());
                }
                rng.random_shuffle(samples.begin(), samples.end());
                testSketch(sketch, samples, maxBias * t, maxError * t, meanBias, meanError);
            }
            LOG_DEBUG(<< "mean bias = " << maths::common::CBasicStatistics::mean(meanBias)
                      << ", mean error = " << maths::common::CBasicStatistics::mean(meanError));
            BOOST_TEST_REQUIRE(std::fabs(maths::common::CBasicStatistics::mean(meanBias)) <
                               maxMeanBias);
            BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(meanError) < maxMeanError);
        };

        maths::common::CQuantileSketch sketch{40};
        maths::common::CFastQuantileSketch fastSketch{40};

        LOG_DEBUG(<< "** sketch **");
        testMixture(sketch, 60, 62, 0.021, 0.021);
        LOG_DEBUG(<< "** fast linear sketch **");
        testMixture(fastSketch, 60, 62, 0.021, 0.026);
    }
}

BOOST_AUTO_TEST_CASE(testCdf) {

    // Test the quantile and c.d.f. functions are mutual inverses.

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "**** Exact ****");
    {
        TDoubleVec values{1.3, 5.2, 0.3, 0.7, 6.9, 10.3, 0.1, -2.9, 9.3, 0.0};

        {
            maths::common::CQuantileSketch sketch{10};
            sketch = std::for_each(values.begin(), values.end(), sketch);
            for (std::size_t i = 0; i < 10; ++i) {
                double x;
                sketch.quantile(10.0 * static_cast<double>(i) + 5.0, x);
                double f;
                sketch.cdf(x, f);
                LOG_DEBUG(<< "x = " << x
                          << ", f(exact) = " << static_cast<double>(i) / 10.0 + 0.05
                          << ", f(actual) = " << f);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(static_cast<double>(i) / 10.0 + 0.05, f, 1e-6);
            }

            double x;
            sketch.quantile(99.0, x);
            double f;
            sketch.cdf(x, f);
            LOG_DEBUG(<< "x = " << x << ", f(exact) = 0.99, f(actual) = " << f);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(0.99, f, 1e-6);
        }
    }

    // Test the estimation error vs the generating distribution.

    LOG_DEBUG(<< "**** Error vs Generating Distribution ****");
    auto exactCdf = [](const TDoubleVec& samples, double x) {
        return static_cast<double>((std::upper_bound(samples.begin(), samples.end(), x) -
                                    samples.begin())) /
               static_cast<double>(samples.size());
    };
    TMeanAccumulator meanBias;
    TMeanAccumulator meanError;
    for (std::size_t t = 0; t < 5; ++t) {
        LOG_DEBUG(<< "test " << t + 1);
        TDoubleVec samples;
        rng.generateUniformSamples(0.0, 20.0 * static_cast<double>(t + 1), 1000, samples);
        maths::common::CQuantileSketch sketch{20};
        sketch = std::for_each(samples.begin(), samples.end(), sketch);
        std::sort(samples.begin(), samples.end());
        for (std::size_t i = 0; i <= 100; ++i) {
            double x;
            sketch.quantile(static_cast<double>(i), x);
            double f;
            sketch.cdf(x, f);
            if (i % 10 == 0) {
                LOG_DEBUG(<< "  U[0, " << 20.0 * static_cast<double>(t + 1) << "]"
                          << ", x = " << x << ", f(expected) = "
                          << static_cast<double>(i) / 100.0 << ", f(actual) = " << f
                          << ", f(exact) = " << exactCdf(samples, x));
            }
            BOOST_REQUIRE_CLOSE_ABSOLUTE(static_cast<double>(i) / 100.0, f, 1e-6);
            meanBias.add(f - exactCdf(samples, x));
            meanError.add(std::fabs(f - exactCdf(samples, x)));
        }
    }

    LOG_DEBUG(<< "mean bias  = " << maths::common::CBasicStatistics::mean(meanBias));
    LOG_DEBUG(<< "mean error = " << maths::common::CBasicStatistics::mean(meanError));
    BOOST_TEST_REQUIRE(std::fabs(maths::common::CBasicStatistics::mean(meanBias)) < 0.0002);
    BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(meanError) < 0.0025);
}

BOOST_AUTO_TEST_CASE(testFastSketchComputeMergeCosts) {

    // Test the fast sketch calculation of all merge costs produces the
    // same results as the standard sketch.

    test::CRandomNumbers generator;
    TDoubleVec samples;
    generator.generateUniformSamples(0.0, 5000.0, 10000, samples);

    TDoubleVec expectedCosts;

    for (std::size_t i = 0; i < samples.size(); /**/) {
        maths::common::CFastQuantileSketch fastSketch{100};
        for (std::size_t j = 0; i < samples.size() && j < 599; ++i, ++j) {
            fastSketch.add(samples[i]);
        }
        fastSketch.order();

        auto knots = fastSketch.knots();

        expectedCosts.clear();
        for (std::size_t j = 0; j + 3 < knots.size(); ++j) {
            expectedCosts.push_back(maths::common::CFastQuantileSketch::mergeCost(
                knots[j + 1], knots[j + 2]));
        }

        fastSketch.computeMergeCosts(knots);

        BOOST_REQUIRE_EQUAL(expectedCosts.size(), fastSketch.m_MergeCosts.size());
        for (std::size_t j = 0; j + 3 < expectedCosts.size(); ++j) {
            BOOST_REQUIRE_CLOSE(expectedCosts[j], fastSketch.m_MergeCosts[j], 1e-3);
        }
    }
}

BOOST_AUTO_TEST_CASE(testFastSketchFastReduce) {

    // Check that we hit the target size, every non-merged knot is preserved,
    // and the count and average of merged knots and class invariants are
    // preserved by fastReduce.

    test::CRandomNumbers rng;
    TDoubleVec samples;
    rng.generateUniformSamples(0.0, 5000.0, 10000, samples);

    for (std::size_t i = 0; i < 1; /**/) {
        maths::common::CFastQuantileSketch fastSketch{100};
        for (std::size_t j = 0; j < 119; ++i, ++j) {
            fastSketch.add(samples[i]);
        }

        auto knots = fastSketch.knots();
        std::sort(knots.begin(), knots.end());

        fastSketch.fastReduce();

        BOOST_TEST_REQUIRE(fastSketch.fastReduceTargetSize(), fastSketch.knots().size());

        double actualMergeMean{0.0};
        double actualMergeCount{0.0};
        for (const auto& knot : fastSketch.knots()) {
            if (knot.second == 1.0) {
                bool found{*std::lower_bound(knots.begin(), knots.end(), knot) == knot};
                BOOST_TEST_REQUIRE(found);
            } else {
                actualMergeMean += knot.second * knot.first;
                actualMergeCount += knot.second;
            }
        }
        actualMergeMean /= actualMergeCount;

        double expectedMergeMean{0.0};
        double expectedMergeCount{0.0};
        for (const auto& knot : knots) {
            if (*std::lower_bound(fastSketch.knots().begin(),
                                  fastSketch.knots().end(), knot) != knot) {
                expectedMergeMean += knot.second * knot.first;
                expectedMergeCount += knot.second;
            }
        }
        expectedMergeMean /= expectedMergeCount;

        BOOST_REQUIRE_CLOSE(expectedMergeMean, actualMergeMean, 1e-3);
        BOOST_REQUIRE_CLOSE(expectedMergeCount, actualMergeCount, 1e-3);
        BOOST_TEST_REQUIRE(fastSketch.checkInvariants());
    }
}

BOOST_AUTO_TEST_CASE(testFastSketchPerformance) {

    // Check the fast sketch relative size.

    test::CRandomNumbers rng;
    TDoubleVec samples;
    rng.generateUniformSamples(0.0, 5000.0, 1500000, samples);

    maths::common::CQuantileSketch sketch{75};
    maths::common::CFastQuantileSketch fastSketch{75};

    core::CStopWatch watch{true};
    std::for_each(samples.begin(), samples.end(), sketch);
    auto lap = watch.lap();
    LOG_DEBUG(<< "sketch duration = " << lap);
    std::for_each(samples.begin(), samples.end(), fastSketch);
    LOG_DEBUG(<< "fast sketch duration = " << watch.lap() - lap);

    LOG_DEBUG(<< "sketch memory usage = " << core::memory::dynamicSize(&sketch));
    LOG_DEBUG(<< "fast sketch memory usage = " << core::memory::dynamicSize(&fastSketch));
    BOOST_TEST_REQUIRE(4 * core::memory::dynamicSize(&sketch) >
                       core::memory::dynamicSize(&fastSketch));
}

BOOST_AUTO_TEST_CASE(testPersist) {

    // Test that persist then restore is idempotent.

    test::CRandomNumbers generator;
    TDoubleVec samples;
    generator.generateUniformSamples(0.0, 5000.0, 500, samples);

    maths::common::CQuantileSketch origSketch{100};
    for (const auto& sample : samples) {
        origSketch.add(sample);
    }

    std::ostringstream origJson;
    core::CJsonStatePersistInserter::persist(
        origJson, std::bind_front(&maths::common::CQuantileSketch::acceptPersistInserter,
                                  &origSketch));

    LOG_DEBUG(<< "quantile sketch JSON representation:\n" << origJson.str());

    maths::common::CQuantileSketch restoredSketch{100};
    {
        std::istringstream origJsonStrm{"{\"topLevel\" : " + origJson.str() + "}"};
        core::CJsonStateRestoreTraverser traverser(origJsonStrm);
        BOOST_TEST_REQUIRE(traverser.traverseSubLevel([&](auto& traverser_) {
            return restoredSketch.acceptRestoreTraverser(traverser_);
        }));
    }

    // Checksums should agree.
    BOOST_REQUIRE_EQUAL(origSketch.checksum(), restoredSketch.checksum());

    // The persist then restore should be idempotent.
    std::ostringstream newJson;
    core::CJsonStatePersistInserter::persist(
        newJson, std::bind_front(&maths::common::CQuantileSketch::acceptPersistInserter,
                                 &restoredSketch));
    BOOST_REQUIRE_EQUAL(origJson.str(), newJson.str());
}

BOOST_AUTO_TEST_SUITE_END()
