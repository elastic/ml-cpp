/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CQuantileSketchTest.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/CBasicStatistics.h>
#include <maths/CQuantileSketch.h>

#include <test/CRandomNumbers.h>

#include <boost/range.hpp>

#include <algorithm>

using namespace ml;

namespace {

using TDoubleVec = std::vector<double>;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

void testSketch(maths::CQuantileSketch::EInterpolation interpolation,
                std::size_t n,
                TDoubleVec& samples,
                double maxBias,
                double maxError,
                TMeanAccumulator& meanBias,
                TMeanAccumulator& meanError) {
    maths::CQuantileSketch sketch(interpolation, n);
    sketch = std::for_each(samples.begin(), samples.end(), sketch);

    LOG_DEBUG(<< "sketch = " << core::CContainerPrinter::print(sketch.knots()));

    std::size_t N = samples.size();
    std::sort(samples.begin(), samples.end());

    TMeanAccumulator bias;
    TMeanAccumulator error;
    for (std::size_t i = 1u; i < 20; ++i) {
        double q = static_cast<double>(i) / 20.0;
        double xq = samples[static_cast<std::size_t>(static_cast<double>(N) * q)];
        double sq;
        CPPUNIT_ASSERT(sketch.quantile(100.0 * q, sq));
        bias.add(xq - sq);
        error.add(std::fabs(xq - sq));
    }

    double min, max;
    sketch.quantile(0.0, min);
    sketch.quantile(100.0, max);
    double scale = max - min;

    LOG_DEBUG(<< "bias = " << maths::CBasicStatistics::mean(bias) << ", error " << maths::CBasicStatistics::mean(error));
    CPPUNIT_ASSERT(std::fabs(maths::CBasicStatistics::mean(bias)) < maxBias);
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(error) < maxError);

    meanBias += maths::CBasicStatistics::accumulator(maths::CBasicStatistics::count(bias), maths::CBasicStatistics::mean(bias) / scale);
    meanError += maths::CBasicStatistics::accumulator(maths::CBasicStatistics::count(error), maths::CBasicStatistics::mean(error) / scale);
}
}

void CQuantileSketchTest::testAdd() {
    LOG_DEBUG(<< "+--------------------------------+");
    LOG_DEBUG(<< "|  CQuantileSketchTest::testAdd  |");
    LOG_DEBUG(<< "+--------------------------------+");

    maths::CQuantileSketch sketch(maths::CQuantileSketch::E_Linear, 5);

    // Test adding a point.
    sketch.add(1.2);
    CPPUNIT_ASSERT(sketch.checkInvariants());

    // Test adding a weighted point.
    sketch.add(0.9, 3.0);
    CPPUNIT_ASSERT(sketch.checkInvariants());

    // Test add via operator().
    double x[] = {1.8, 2.1};
    sketch = std::for_each(x, x + 2, sketch);
    CPPUNIT_ASSERT(sketch.checkInvariants());

    LOG_DEBUG(<< "sketch = " << core::CContainerPrinter::print(sketch.knots()));
    CPPUNIT_ASSERT_EQUAL(6.0, sketch.count());
    CPPUNIT_ASSERT_EQUAL(std::string("[(1.2, 1), (0.9, 3), (1.8, 1), (2.1, 1)]"), core::CContainerPrinter::print(sketch.knots()));
}

void CQuantileSketchTest::testReduce() {
    LOG_DEBUG(<< "+-----------------------------------+");
    LOG_DEBUG(<< "|  CQuantileSketchTest::testReduce  |");
    LOG_DEBUG(<< "+-----------------------------------+");

    LOG_DEBUG(<< "*** Linear ***");
    {
        maths::CQuantileSketch sketch(maths::CQuantileSketch::E_Linear, 6);

        // Test duplicate points.

        double points[][2] = {{5.0, 1.0}, {0.4, 2.0}, {0.4, 1.0}, {1.0, 1.0}, {1.2, 2.0}, {1.2, 1.5}, {5.0, 1.0}};
        for (std::size_t i = 0u; i < boost::size(points); ++i) {
            sketch.add(points[i][0], points[i][1]);
            CPPUNIT_ASSERT(sketch.checkInvariants());
        }

        LOG_DEBUG(<< "sketch = " << core::CContainerPrinter::print(sketch.knots()));
        CPPUNIT_ASSERT_EQUAL(std::string("[(0.4, 3), (1, 1), (1.2, 3.5), (5, 2)]"), core::CContainerPrinter::print(sketch.knots()));

        // Regular compress (merging two point).

        sketch.add(0.1);
        sketch.add(0.2);
        sketch.add(0.0);
        LOG_DEBUG(<< "sketch = " << core::CContainerPrinter::print(sketch.knots()));
        CPPUNIT_ASSERT_EQUAL(std::string("[(0, 1), (0.15, 2), (0.4, 3), (1, 1), (1.2, 3.5), (5, 2)]"),
                             core::CContainerPrinter::print(sketch.knots()));
    }
    {
        // Multiple points compressed at once.

        maths::CQuantileSketch sketch(maths::CQuantileSketch::E_Linear, 30);

        for (std::size_t i = 0u; i <= 30; ++i) {
            sketch.add(static_cast<double>(i));
            CPPUNIT_ASSERT(sketch.checkInvariants());
        }
        LOG_DEBUG(<< "sketch = " << core::CContainerPrinter::print(sketch.knots()));
        CPPUNIT_ASSERT_EQUAL(std::string("[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1),"
                                         " (5.5, 2), (7, 1), (8, 1), (9, 1), (10, 1),"
                                         " (11, 1), (12, 1), (13.5, 2), (15, 1), (16, 1),"
                                         " (17, 1), (18, 1), (19, 1), (20, 1), (21, 1),"
                                         " (22.5, 2), (24, 1), (25, 1), (26, 1), (27, 1),"
                                         " (28, 1), (29, 1), (30, 1)]"),
                             core::CContainerPrinter::print(sketch.knots()));
    }
    {
        // Test the quantiles are reasonable at a compression ratio of 2:1.

        double points[] = {1.0, 2.0, 40.0, 13.0, 5.0, 6.0, 4.0, 7.0, 15.0, 17.0, 19.0, 44.0, 42.0, 3.0, 46.0, 48.0, 50.0, 21.0, 23.0, 52.0};
        double cdf[] = {5.0,  10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,
                        55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0};

        maths::CQuantileSketch sketch(maths::CQuantileSketch::E_Linear, 10);
        for (std::size_t i = 0u; i < boost::size(points); ++i) {
            sketch.add(points[i]);
            CPPUNIT_ASSERT(sketch.checkInvariants());
            if ((i + 1) % 5 == 0) {
                LOG_DEBUG(<< "sketch = " << core::CContainerPrinter::print(sketch.knots()));
            }
        }

        std::sort(boost::begin(points), boost::end(points));
        TMeanAccumulator error;
        for (std::size_t i = 0u; i < boost::size(cdf); ++i) {
            double x;
            CPPUNIT_ASSERT(sketch.quantile(cdf[i], x));
            LOG_DEBUG(<< "expected quantile = " << points[i] << ", actual quantile = " << x);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(points[i], x, 10.0);
            error.add(std::fabs(points[i] - x));
        }
        LOG_DEBUG(<< "error = " << maths::CBasicStatistics::mean(error));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(error) < 1.5);
    }

    LOG_DEBUG(<< "*** Piecewise Constant ***");
    {
        maths::CQuantileSketch sketch(maths::CQuantileSketch::E_PiecewiseConstant, 6);

        // Test duplicate points.

        double points[][2] = {{5.0, 1.0}, {0.4, 2.0}, {0.4, 1.0}, {1.0, 1.0}, {1.2, 2.0}, {1.2, 1.5}, {5.0, 1.0}};
        for (std::size_t i = 0u; i < boost::size(points); ++i) {
            sketch.add(points[i][0], points[i][1]);
            CPPUNIT_ASSERT(sketch.checkInvariants());
        }

        LOG_DEBUG(<< "sketch = " << core::CContainerPrinter::print(sketch.knots()));
        CPPUNIT_ASSERT_EQUAL(std::string("[(0.4, 3), (1, 1), (1.2, 3.5), (5, 2)]"), core::CContainerPrinter::print(sketch.knots()));

        // Regular compress (merging two point).

        sketch.add(0.1);
        sketch.add(0.2);
        sketch.add(0.0);
        LOG_DEBUG(<< "sketch = " << core::CContainerPrinter::print(sketch.knots()));
        CPPUNIT_ASSERT_EQUAL(std::string("[(0, 1), (0.2, 2), (0.4, 3), (1, 1), (1.2, 3.5), (5, 2)]"),
                             core::CContainerPrinter::print(sketch.knots()));
    }
    {
        // Multiple points compressed at once.

        maths::CQuantileSketch sketch(maths::CQuantileSketch::E_PiecewiseConstant, 30);

        for (std::size_t i = 0u; i <= 30; ++i) {
            sketch.add(static_cast<double>(i));
            CPPUNIT_ASSERT(sketch.checkInvariants());
        }
        LOG_DEBUG(<< "sketch = " << core::CContainerPrinter::print(sketch.knots()));
        CPPUNIT_ASSERT_EQUAL(std::string("[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1),"
                                         " (6, 2), (7, 1), (8, 1), (9, 1), (10, 1),"
                                         " (11, 1), (12, 1), (13, 1), (14, 1), (15, 1),"
                                         " (16, 1), (17, 1), (18, 1), (19, 1), (20, 1),"
                                         " (21, 1), (23, 3), (25, 1), (26, 1), (27, 1),"
                                         " (28, 1), (29, 1), (30, 1)]"),
                             core::CContainerPrinter::print(sketch.knots()));
    }
    {
        // Test the quantiles are reasonable at a compression ratio of 2:1.

        double points[] = {1.0, 2.0, 40.0, 13.0, 5.0, 6.0, 4.0, 7.0, 15.0, 17.0, 19.0, 44.0, 42.0, 3.0, 46.0, 48.0, 50.0, 21.0, 23.0, 52.0};
        double cdf[] = {5.0,  10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,
                        55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0};

        maths::CQuantileSketch sketch(maths::CQuantileSketch::E_PiecewiseConstant, 10);
        for (std::size_t i = 0u; i < boost::size(points); ++i) {
            sketch.add(points[i]);
            CPPUNIT_ASSERT(sketch.checkInvariants());
            if ((i + 1) % 5 == 0) {
                LOG_DEBUG(<< "sketch = " << core::CContainerPrinter::print(sketch.knots()));
            }
        }

        std::sort(boost::begin(points), boost::end(points));
        TMeanAccumulator error;
        for (std::size_t i = 0u; i < boost::size(cdf); ++i) {
            double x;
            CPPUNIT_ASSERT(sketch.quantile(cdf[i], x));
            LOG_DEBUG(<< "expected quantile = " << points[i] << ", actual quantile = " << x);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(points[i], x, 10.0);
            error.add(std::fabs(points[i] - x));
        }
        LOG_DEBUG(<< "error = " << maths::CBasicStatistics::mean(error));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(error) < 1.8);
    }
}

void CQuantileSketchTest::testMerge() {
    LOG_DEBUG(<< "+----------------------------------+");
    LOG_DEBUG(<< "|  CQuantileSketchTest::testMerge  |");
    LOG_DEBUG(<< "+----------------------------------+");

    {
        // Simple merge no reduction.

        maths::CQuantileSketch sketch1(maths::CQuantileSketch::E_Linear, 20);
        maths::CQuantileSketch sketch2(maths::CQuantileSketch::E_Linear, 10);

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
        LOG_DEBUG(<< "merged sketch = " << core::CContainerPrinter::print(sketch1.knots()));
        CPPUNIT_ASSERT_EQUAL(std::string("[(1, 3.6), (1.1, 1), (2, 1), (3, 1), (3.1, 2), (5.1, 2)]"),
                             core::CContainerPrinter::print(sketch1.knots()));
    }

    {
        // Test the quantiles are reasonable at a compression ratio of 2:1.

        double points[] = {1.0, 2.0, 40.0, 13.0, 5.0, 6.0, 4.0, 7.0, 15.0, 17.0, 19.0, 44.0, 42.0, 3.0, 46.0, 48.0, 50.0, 21.0, 23.0, 52.0};
        double cdf[] = {5.0,  10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,
                        55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0};

        maths::CQuantileSketch sketch1(maths::CQuantileSketch::E_Linear, 10);
        maths::CQuantileSketch sketch2(maths::CQuantileSketch::E_Linear, 10);
        for (std::size_t i = 0u; i < boost::size(points); i += 2) {
            sketch1.add(points[i]);
            sketch2.add(points[i + 1]);
        }
        LOG_DEBUG(<< "sketch 1 = " << core::CContainerPrinter::print(sketch1.knots()));
        LOG_DEBUG(<< "sketch 2 = " << core::CContainerPrinter::print(sketch2.knots()));

        maths::CQuantileSketch sketch3 = sketch1 + sketch2;
        LOG_DEBUG(<< "merged sketch = " << core::CContainerPrinter::print(sketch3.knots()));

        std::sort(boost::begin(points), boost::end(points));
        TMeanAccumulator error;
        for (std::size_t i = 0u; i < boost::size(cdf); ++i) {
            double x;
            CPPUNIT_ASSERT(sketch3.quantile(cdf[i], x));
            LOG_DEBUG(<< "expected quantile = " << points[i] << ", actual quantile = " << x);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(points[i], x, 10.0);
            error.add(std::fabs(points[i] - x));
        }
        LOG_DEBUG(<< "error = " << maths::CBasicStatistics::mean(error));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(error) < 1.8);
    }
}

void CQuantileSketchTest::testMedian() {
    LOG_DEBUG(<< "+-----------------------------------+");
    LOG_DEBUG(<< "|  CQuantileSketchTest::testMedian  |");
    LOG_DEBUG(<< "+-----------------------------------+");

    LOG_DEBUG(<< "*** Exact ***");
    {
        maths::CQuantileSketch sketch(maths::CQuantileSketch::E_PiecewiseConstant, 10);

        double median;
        CPPUNIT_ASSERT(!sketch.quantile(50.0, median));

        sketch.add(1.0);
        CPPUNIT_ASSERT(sketch.quantile(50.0, median));
        CPPUNIT_ASSERT_EQUAL(1.0, median);

        // [1.0, 2.0]
        sketch.add(2.0);
        CPPUNIT_ASSERT(sketch.quantile(50.0, median));
        CPPUNIT_ASSERT_EQUAL(1.5, median);

        // [1.0, 2.0, 3.0]
        sketch.add(3.0);
        CPPUNIT_ASSERT(sketch.quantile(50.0, median));
        CPPUNIT_ASSERT_EQUAL(2.0, median);

        // [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        sketch.add(8.0);
        sketch.add(4.0);
        sketch.add(7.0);
        sketch.add(6.0);
        sketch.add(9.0);
        sketch.add(5.0);
        CPPUNIT_ASSERT(sketch.quantile(50.0, median));
        CPPUNIT_ASSERT_EQUAL(5.0, median);
    }

    LOG_DEBUG(<< "*** Approximate ***");

    test::CRandomNumbers rng;

    TMeanAccumulator bias;
    TMeanAccumulator error;
    for (std::size_t t = 0u; t < 500; ++t) {
        TDoubleVec samples;
        rng.generateUniformSamples(0.0, 100.0, 501, samples);
        maths::CQuantileSketch sketch(maths::CQuantileSketch::E_PiecewiseConstant, 20);
        sketch = std::for_each(samples.begin(), samples.end(), sketch);
        std::sort(samples.begin(), samples.end());
        double expectedMedian = samples[250];
        double actualMedian;
        sketch.quantile(50.0, actualMedian);
        CPPUNIT_ASSERT(std::fabs(actualMedian - expectedMedian) < 6.7);
        bias.add(actualMedian - expectedMedian);
        error.add(std::fabs(actualMedian - expectedMedian));
    }

    LOG_DEBUG(<< "bias  = " << maths::CBasicStatistics::mean(bias));
    LOG_DEBUG(<< "error = " << maths::CBasicStatistics::mean(error));
    CPPUNIT_ASSERT(std::fabs(maths::CBasicStatistics::mean(bias)) < 0.2);
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(error) < 1.6);
}

void CQuantileSketchTest::testPropagateForwardByTime() {
    LOG_DEBUG(<< "+---------------------------------------------------+");
    LOG_DEBUG(<< "|  CQuantileSketchTest::testPropagateForwardByTime  |");
    LOG_DEBUG(<< "+---------------------------------------------------+");

    // Check that the count is reduced and the invariants still hold.

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateUniformSamples(0.0, 20.0, 100, samples);

    maths::CQuantileSketch sketch(maths::CQuantileSketch::E_PiecewiseConstant, 20);
    sketch = std::for_each(samples.begin(), samples.end(), sketch);

    sketch.age(0.9);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(90.0, sketch.count(), 1e-6);
    CPPUNIT_ASSERT(sketch.checkInvariants());
}

void CQuantileSketchTest::testQuantileAccuracy() {
    LOG_DEBUG(<< "+---------------------------------------------+");
    LOG_DEBUG(<< "|  CQuantileSketchTest::testQuantileAccuracy  |");
    LOG_DEBUG(<< "+---------------------------------------------+");

    // Test on a variety of random data sets versus the corresponding
    // quantile in the raw data.

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "*** Uniform ***");
    {
        TMeanAccumulator meanBias;
        TMeanAccumulator meanError;
        for (std::size_t t = 0u; t < 5; ++t) {
            TDoubleVec samples;
            rng.generateUniformSamples(0.0, 20.0 * static_cast<double>(t + 1), 1000, samples);
            testSketch(maths::CQuantileSketch::E_Linear, 20, samples, 0.15, 0.3, meanBias, meanError);
        }
        LOG_DEBUG(<< "mean bias = " << std::fabs(maths::CBasicStatistics::mean(meanBias)) << ", mean error "
                  << maths::CBasicStatistics::mean(meanError));
        CPPUNIT_ASSERT(std::fabs(maths::CBasicStatistics::mean(meanBias)) < 0.0007);
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanError) < 0.003);
    }

    LOG_DEBUG(<< "*** Normal ***");
    {
        TMeanAccumulator meanBias;
        TMeanAccumulator meanError;
        for (std::size_t t = 0u; t < 5; ++t) {
            TDoubleVec samples;
            rng.generateNormalSamples(20.0 * static_cast<double>(t), 20.0 * static_cast<double>(t + 1), 1000, samples);
            testSketch(maths::CQuantileSketch::E_Linear, 20, samples, 0.16, 0.2, meanBias, meanError);
        }
        LOG_DEBUG(<< "mean bias = " << maths::CBasicStatistics::mean(meanBias) << ", mean error "
                  << maths::CBasicStatistics::mean(meanError));
        CPPUNIT_ASSERT(std::fabs(maths::CBasicStatistics::mean(meanBias)) < 0.002);
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanError) < 0.003);
    }

    LOG_DEBUG(<< "*** Log-Normal ***");
    {
        TMeanAccumulator meanBias;
        TMeanAccumulator meanError;
        for (std::size_t t = 0u; t < 5; ++t) {
            TDoubleVec samples;
            rng.generateLogNormalSamples(0.1 * static_cast<double>(t), 0.4 * static_cast<double>(t + 1), 1000, samples);
            testSketch(maths::CQuantileSketch::E_Linear, 20, samples, 0.11, 0.12, meanBias, meanError);
        }
        LOG_DEBUG(<< "mean bias = " << maths::CBasicStatistics::mean(meanBias) << ", mean error "
                  << maths::CBasicStatistics::mean(meanError));
        CPPUNIT_ASSERT(std::fabs(maths::CBasicStatistics::mean(meanBias)) < 0.0006);
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanError) < 0.0009);
    }
    LOG_DEBUG(<< "*** Mixture ***");
    {
        TMeanAccumulator meanBiasLinear;
        TMeanAccumulator meanErrorLinear;
        TMeanAccumulator meanBiasPiecewise;
        TMeanAccumulator meanErrorPiecewise;
        for (std::size_t t = 0u; t < 5; ++t) {
            TDoubleVec samples_[4] = {};
            rng.generateNormalSamples(10.0 * static_cast<double>(t), 20.0 * static_cast<double>(t + 1), 400, samples_[0]);
            rng.generateNormalSamples(20.0 * static_cast<double>(t), 20.0 * static_cast<double>(t + 1), 600, samples_[1]);
            rng.generateNormalSamples(100.0 * static_cast<double>(t), 40.0 * static_cast<double>(t + 1), 400, samples_[2]);
            rng.generateUniformSamples(500.0 * static_cast<double>(t), 550.0 * static_cast<double>(t + 1), 600, samples_[3]);
            TDoubleVec samples;
            for (std::size_t i = 0u; i < 4; ++i) {
                samples.insert(samples.end(), samples_[i].begin(), samples_[i].end());
            }
            rng.random_shuffle(samples.begin(), samples.end());
            testSketch(maths::CQuantileSketch::E_Linear, 40, samples, 49, 50, meanBiasLinear, meanErrorLinear);
            testSketch(maths::CQuantileSketch::E_PiecewiseConstant, 40, samples, 55, 56, meanBiasPiecewise, meanErrorPiecewise);
        }
        LOG_DEBUG(<< "linear mean bias = " << maths::CBasicStatistics::mean(meanBiasLinear) << ", mean error "
                  << maths::CBasicStatistics::mean(meanErrorLinear));
        CPPUNIT_ASSERT(std::fabs(maths::CBasicStatistics::mean(meanBiasLinear)) < 0.012);
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanErrorLinear) < 0.013);
        LOG_DEBUG(<< "piecewise mean bias = " << maths::CBasicStatistics::mean(meanBiasPiecewise) << ", mean error "
                  << maths::CBasicStatistics::mean(meanErrorPiecewise));
        CPPUNIT_ASSERT(std::fabs(maths::CBasicStatistics::mean(meanBiasPiecewise)) < 0.015);
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanErrorPiecewise) < 0.015);
    }
}

void CQuantileSketchTest::testCdf() {
    LOG_DEBUG(<< "+--------------------------------+");
    LOG_DEBUG(<< "|  CQuantileSketchTest::testCdf  |");
    LOG_DEBUG(<< "+--------------------------------+");

    // Test that quantile and c.d.f. are idempotent.

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "*** Exact ***");
    {
        double values[] = {1.3, 5.2, 0.3, 0.7, 6.9, 10.3, 0.1, -2.9, 9.3, 0.0};

        {
            maths::CQuantileSketch sketch(maths::CQuantileSketch::E_PiecewiseConstant, 10);
            sketch = std::for_each(boost::begin(values), boost::end(values), sketch);
            for (std::size_t i = 0u; i < 10; ++i) {
                double x;
                sketch.quantile(10.0 * static_cast<double>(i) + 5.0, x);
                double f;
                sketch.cdf(x, f);
                LOG_DEBUG(<< "x = " << x << ", f(exact) = " << static_cast<double>(i) / 10.0 + 0.05 << ", f(actual) = " << f);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(static_cast<double>(i) / 10.0 + 0.05, f, 1e-6);
            }
        }
        {
            maths::CQuantileSketch sketch(maths::CQuantileSketch::E_Linear, 10);
            sketch = std::for_each(boost::begin(values), boost::end(values), sketch);
            for (std::size_t i = 0u; i < 10; ++i) {
                double x;
                sketch.quantile(10.0 * static_cast<double>(i) + 5.0, x);
                double f;
                sketch.cdf(x, f);
                LOG_DEBUG(<< "x = " << x << ", f(exact) = " << static_cast<double>(i) / 10.0 + 0.05 << ", f(actual) = " << f);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(static_cast<double>(i) / 10.0 + 0.05, f, 1e-6);
            }

            double x;
            sketch.quantile(99.0, x);
            double f;
            sketch.cdf(x, f);
            LOG_DEBUG(<< "x = " << x << ", f(exact) = 0.99, f(actual) = " << f);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.99, f, 1e-6);
        }
    }

    LOG_DEBUG(<< "*** Uniform ***");
    {
        TMeanAccumulator meanBias;
        TMeanAccumulator meanError;
        for (std::size_t t = 0u; t < 5; ++t) {
            LOG_DEBUG(<< "test " << t + 1);
            TDoubleVec samples;
            rng.generateUniformSamples(0.0, 20.0 * static_cast<double>(t + 1), 1000, samples);
            {
                maths::CQuantileSketch sketch(maths::CQuantileSketch::E_Linear, 20);
                sketch = std::for_each(samples.begin(), samples.end(), sketch);
                for (std::size_t i = 0u; i <= 100; ++i) {
                    double x;
                    sketch.quantile(static_cast<double>(i), x);
                    double f;
                    sketch.cdf(x, f);
                    if (i % 10 == 0) {
                        LOG_DEBUG(<< "  x = " << x << ", f(exact) = " << static_cast<double>(i) / 100.0 << ", f(actual) = " << f);
                    }
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(static_cast<double>(i) / 100.0, f, 1e-6);
                }
            }
        }
    }
}

void CQuantileSketchTest::testPersist() {
    LOG_DEBUG(<< "+------------------------------------+");
    LOG_DEBUG(<< "|  CQuantileSketchTest::testPersist  |");
    LOG_DEBUG(<< "+------------------------------------+");

    test::CRandomNumbers generator;
    TDoubleVec samples;
    generator.generateUniformSamples(0.0, 5000.0, 500u, samples);

    maths::CQuantileSketch origSketch(maths::CQuantileSketch::E_Linear, 100u);
    for (std::size_t i = 0u; i < samples.size(); ++i) {
        origSketch.add(samples[i]);
    }

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origSketch.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_DEBUG(<< "quantile sketch XML representation:\n" << origXml);

    maths::CQuantileSketch restoredSketch(maths::CQuantileSketch::E_Linear, 100u);
    {
        core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);
        CPPUNIT_ASSERT(traverser.traverseSubLevel(boost::bind(&maths::CQuantileSketch::acceptRestoreTraverser, &restoredSketch, _1)));
    }

    // Checksums should agree.
    CPPUNIT_ASSERT_EQUAL(origSketch.checksum(), restoredSketch.checksum());

    // The persist and restore should be idempotent.
    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredSketch.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    CPPUNIT_ASSERT_EQUAL(origXml, newXml);
}

CppUnit::Test* CQuantileSketchTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CQuantileSketchTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CQuantileSketchTest>("CQuantileSketchTest::testAdd", &CQuantileSketchTest::testAdd));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CQuantileSketchTest>("CQuantileSketchTest::testReduce", &CQuantileSketchTest::testReduce));
    suiteOfTests->addTest(new CppUnit::TestCaller<CQuantileSketchTest>("CQuantileSketchTest::testMerge", &CQuantileSketchTest::testMerge));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CQuantileSketchTest>("CQuantileSketchTest::testMedian", &CQuantileSketchTest::testMedian));
    suiteOfTests->addTest(new CppUnit::TestCaller<CQuantileSketchTest>("CQuantileSketchTest::testPropagateForwardByTime",
                                                                       &CQuantileSketchTest::testPropagateForwardByTime));
    suiteOfTests->addTest(new CppUnit::TestCaller<CQuantileSketchTest>("CQuantileSketchTest::testQuantileAccuracy",
                                                                       &CQuantileSketchTest::testQuantileAccuracy));
    suiteOfTests->addTest(new CppUnit::TestCaller<CQuantileSketchTest>("CQuantileSketchTest::testCdf", &CQuantileSketchTest::testCdf));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CQuantileSketchTest>("CQuantileSketchTest::testPersist", &CQuantileSketchTest::testPersist));

    return suiteOfTests;
}
