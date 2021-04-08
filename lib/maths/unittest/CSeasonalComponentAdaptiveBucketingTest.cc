/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/Constants.h>

#include <maths/CSeasonalComponentAdaptiveBucketing.h>
#include <maths/CSeasonalTime.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <boost/math/constants/constants.hpp>
#include <boost/range.hpp>
#include <boost/test/unit_test.hpp>

#include <fstream>
#include <numeric>
#include <string>
#include <vector>

BOOST_AUTO_TEST_SUITE(CSeasonalComponentAdaptiveBucketingTest)

using namespace ml;

namespace {
using TDoubleVec = std::vector<double>;
using TFloatVec = std::vector<maths::CFloatStorage>;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanVarAccumulator = maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
using TMinAccumulator = maths::CBasicStatistics::SMin<double>::TAccumulator;
using TMaxAccumulator = maths::CBasicStatistics::SMax<double>::TAccumulator;
}

BOOST_AUTO_TEST_CASE(testInitialize) {
    maths::CGeneralPeriodTime time(100);
    maths::CSeasonalComponentAdaptiveBucketing bucketing(time);

    BOOST_TEST_REQUIRE(!bucketing.initialize(0));

    const std::string expectedEndpoints("[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]");
    const std::string expectedKnots("[0, 5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 100]");
    const std::string expectedValues("[50, 5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 50]");

    BOOST_TEST_REQUIRE(bucketing.initialize(10));
    const TFloatVec& endpoints = bucketing.endpoints();
    BOOST_REQUIRE_EQUAL(expectedEndpoints, core::CContainerPrinter::print(endpoints));

    for (core_t::TTime t = 5; t < 100; t += 10) {
        double v = static_cast<double>(t);
        bucketing.add(t, v, v);
    }
    TDoubleVec knots;
    TDoubleVec values;
    TDoubleVec variances;
    bucketing.knots(1, maths::CSplineTypes::E_Periodic, knots, values, variances);
    BOOST_REQUIRE_EQUAL(expectedKnots, core::CContainerPrinter::print(knots));
    BOOST_REQUIRE_EQUAL(expectedValues, core::CContainerPrinter::print(values));
}

BOOST_AUTO_TEST_CASE(testSwap) {
    maths::CGeneralPeriodTime time1(100);
    maths::CSeasonalComponentAdaptiveBucketing bucketing1(time1, 0.05);

    test::CRandomNumbers rng;

    bucketing1.initialize(10);
    for (std::size_t p = 0; p < 50; ++p) {
        TDoubleVec noise;
        rng.generateNormalSamples(0.0, 2.0, 100, noise);

        for (std::size_t i = 0; i < 100; ++i) {
            core_t::TTime x = static_cast<core_t::TTime>(100 * p + i);
            double y = 0.02 * (static_cast<double>(i) - 50.0) *
                       (static_cast<double>(i) - 50.0);
            bucketing1.add(x, y + noise[i], y);
        }
        bucketing1.refine(static_cast<core_t::TTime>(100 * (p + 1)));
        bucketing1.propagateForwardsByTime(1.0);
    }

    maths::CGeneralPeriodTime time2(120);
    maths::CSeasonalComponentAdaptiveBucketing bucketing2(time2, 0.1);

    uint64_t checksum1 = bucketing1.checksum();
    uint64_t checksum2 = bucketing2.checksum();

    bucketing1.swap(bucketing2);

    LOG_DEBUG(<< "checksum 1 = " << checksum1);
    LOG_DEBUG(<< "checksum 2 = " << checksum2);

    BOOST_REQUIRE_EQUAL(checksum1, bucketing2.checksum());
    BOOST_REQUIRE_EQUAL(checksum2, bucketing1.checksum());
}

BOOST_AUTO_TEST_CASE(testRefine) {
    test::CRandomNumbers rng;

    {
        // Test that refine reduces the function approximation error.

        core_t::TTime times[] = {-1,    3600,  10800, 18000, 25200,
                                 32400, 39600, 46800, 54000, 61200,
                                 68400, 75600, 82800, 86400};
        double function[] = {10, 10,  10, 10, 100, 90, 80,
                             90, 100, 20, 10, 10,  10, 10};

        maths::CDiurnalTime time(0, 0, core::constants::WEEK, core::constants::DAY);
        maths::CSeasonalComponentAdaptiveBucketing bucketing1(time);
        maths::CSeasonalComponentAdaptiveBucketing bucketing2(time);

        bucketing1.initialize(12);
        bucketing2.initialize(12);

        for (std::size_t p = 0; p < 20; ++p) {
            TDoubleVec noise;
            rng.generateNormalSamples(0.0, 4.0, 48, noise);

            core_t::TTime start = static_cast<core_t::TTime>(86400 * p);
            for (core_t::TTime t = 0; t < 86400; t += 1800) {
                core_t::TTime x = start + t;

                ptrdiff_t i = std::lower_bound(std::begin(times), std::end(times), t) -
                              std::begin(times);

                double x0 = static_cast<double>(times[i - 1]);
                double x1 = static_cast<double>(times[i]);
                double y0 = function[i - 1];
                double y1 = function[i];
                double y = y0 + (y1 - y0) * (static_cast<double>(t) - x0) / (x1 - x0);

                bucketing1.add(x, y + noise[t / 1800], y);
                bucketing2.add(x, y + noise[t / 1800], y);
            }
            bucketing2.refine(static_cast<core_t::TTime>(86400 * (p + 1)));
        }

        TMeanAccumulator meanError1;
        TMaxAccumulator maxError1;
        const TFloatVec& endpoints1 = bucketing1.endpoints();
        TDoubleVec values1 = bucketing1.values(20 * 86400);
        for (std::size_t i = 1; i < endpoints1.size(); ++i) {
            core_t::TTime t = static_cast<core_t::TTime>(
                0.5 * (endpoints1[i] + endpoints1[i - 1] + 1.0));
            ptrdiff_t j = std::lower_bound(std::begin(times), std::end(times), t) -
                          std::begin(times);
            double x0 = static_cast<double>(times[j - 1]);
            double x1 = static_cast<double>(times[j]);
            double y0 = function[j - 1];
            double y1 = function[j];
            double y = y0 + (y1 - y0) * (static_cast<double>(t) - x0) / (x1 - x0);
            meanError1.add(std::fabs(values1[i - 1] - y));
            maxError1.add(std::fabs(values1[i - 1] - y));
        }

        TMeanAccumulator meanError2;
        TMaxAccumulator maxError2;
        const TFloatVec& endpoints2 = bucketing2.endpoints();
        TDoubleVec values2 = bucketing2.values(20 * 86400);
        for (std::size_t i = 1; i < endpoints1.size(); ++i) {
            core_t::TTime t = static_cast<core_t::TTime>(
                0.5 * (endpoints2[i] + endpoints2[i - 1] + 1.0));
            ptrdiff_t j = std::lower_bound(std::begin(times), std::end(times), t) -
                          std::begin(times);
            double x0 = static_cast<double>(times[j - 1]);
            double x1 = static_cast<double>(times[j]);
            double y0 = function[j - 1];
            double y1 = function[j];
            double y = y0 + (y1 - y0) * (static_cast<double>(t) - x0) / (x1 - x0);
            meanError2.add(std::fabs(values2[i - 1] - y));
            maxError2.add(std::fabs(values2[i - 1] - y));
        }

        LOG_DEBUG(<< "mean error         = " << maths::CBasicStatistics::mean(meanError1));
        LOG_DEBUG(<< "max error          = " << maxError1[0]);
        LOG_DEBUG(<< "refined mean error = " << maths::CBasicStatistics::mean(meanError2));
        LOG_DEBUG(<< "refined max error  = " << maxError2[0]);
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanError2) <
                           0.85 * maths::CBasicStatistics::mean(meanError1));
        BOOST_TEST_REQUIRE(maxError2[0] < 0.7 * maxError1[0]);
    }

    {
        // Test that the variance in each bucket is approximately equal.
        // The test function is y = (x - 50)^2 / 50.

        maths::CGeneralPeriodTime time(100);
        maths::CSeasonalComponentAdaptiveBucketing bucketing(time, 0.05);

        bucketing.initialize(10);
        for (std::size_t p = 0; p < 50; ++p) {
            TDoubleVec noise;
            rng.generateNormalSamples(0.0, 9.0, 100, noise);

            for (std::size_t i = 0; i < 100; ++i) {
                core_t::TTime x = static_cast<core_t::TTime>(100 * p + i);
                double y = 0.02 * (static_cast<double>(i) - 50.0) *
                           (static_cast<double>(i) - 50.0);
                bucketing.add(x, y + noise[i], y);
            }
            bucketing.refine(static_cast<core_t::TTime>(100 * (p + 1)));
            bucketing.propagateForwardsByTime(1.0);
        }

        const TFloatVec& endpoints = bucketing.endpoints();
        TDoubleVec values = bucketing.values(5100);
        TDoubleVec variances = bucketing.variances();
        LOG_DEBUG(<< "endpoints = " << core::CContainerPrinter::print(endpoints));
        LOG_DEBUG(<< "values    = " << core::CContainerPrinter::print(values));
        LOG_DEBUG(<< "variances = " << core::CContainerPrinter::print(variances));

        TMeanAccumulator meanError;
        TMeanAccumulator varianceError;
        TMeanVarAccumulator avgError;

        for (std::size_t i = 1; i < endpoints.size(); ++i) {
            double a = endpoints[i - 1];
            double b = endpoints[i];
            LOG_DEBUG(<< "bucket = [" << a << "," << b << "]");

            a -= 50.0;
            b -= 50.0;

            double m = values[i - 1];
            double v = variances[i - 1];

            // Function mean and variance.
            double m_ = std::fabs(a) < std::fabs(b)
                            ? 0.02 / 3.0 * std::pow(b, 3.0) *
                                  (1.0 - std::pow(a / b, 3.0)) / (b - a)
                            : 0.02 / 3.0 * std::pow(a, 3.0) *
                                  (std::pow(b / a, 3.0) - 1.0) / (b - a);
            double v_ = 9.0;
            LOG_DEBUG(<< "m = " << m << ", m_ = " << m_
                      << ", absolute error = " << std::fabs(m - m_));
            LOG_DEBUG(<< "v = " << v << ", v_ = " << v_
                      << ", relative error = " << std::fabs(v - v_) / v_);

            BOOST_REQUIRE_CLOSE_ABSOLUTE(m_, m, 0.7);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(v_, v, 0.5 * v_);
            meanError.add(std::fabs(m_ - m) / m_);
            varianceError.add(std::fabs(v_ - v) / v_);

            if (i == 1 || i == endpoints.size() - 1) {
                continue;
            }
            if ((b * b / 50.0 - m) * (a * a / 50.0 - m) < 0.0) {
                // Root.
                double c = b < 0.0 ? -std::sqrt(50.0 * m) : +::sqrt(50.0 * m);
                // Left and right partial averaging errors.
                double l = std::fabs(c) < std::fabs(a)
                               ? 0.02 / 3.0 * a * a * a *
                                         ((c / a) * (c / a) * (c / a) - 1.0) -
                                     m * (c - a)
                               : 0.02 / 3.0 * c * c * c *
                                         (1.0 - (a / c) * (a / c) * (a / c)) -
                                     m * (c - a);
                double r = std::fabs(c) < std::fabs(b)
                               ? 0.02 / 3.0 * b * b * b *
                                         (1.0 - (c / b) * (c / b) * (c / b)) -
                                     m * (b - c)
                               : 0.02 / 3.0 * c * c * c *
                                         ((b / c) * (b / c) * (b / c) - 1.0) -
                                     m * (b - c);
                LOG_DEBUG(<< "c = " << c << ", l = " << l << " r = " << r
                          << ", error = " << std::fabs(l) + std::fabs(r));
                avgError.add(std::fabs(l) + std::fabs(r));
            } else {
                avgError.add(std::fabs((m_ - m) * (b - a)));
            }
        }

        double meanError_ = maths::CBasicStatistics::mean(meanError);
        double varianceError_ = maths::CBasicStatistics::mean(varianceError);
        LOG_DEBUG(<< "meanError = " << meanError_ << ", varianceError = " << varianceError_);

        BOOST_TEST_REQUIRE(meanError_ < 0.10);
        BOOST_TEST_REQUIRE(varianceError_ < 0.22);

        double avgErrorMean = maths::CBasicStatistics::mean(avgError);
        double avgErrorStd = std::sqrt(maths::CBasicStatistics::variance(avgError));
        LOG_DEBUG(<< "avgErrorMean = " << avgErrorMean << ", avgErrorStd = " << avgErrorStd);

        BOOST_TEST_REQUIRE(avgErrorStd / avgErrorMean < 0.5);
    }
}

BOOST_AUTO_TEST_CASE(testPropagateForwardsByTime) {
    // Check no error is introduced by the aging process to
    // the bucket values and that the rate at which the total
    // count is reduced uniformly.

    maths::CGeneralPeriodTime time(100);
    maths::CSeasonalComponentAdaptiveBucketing bucketing(time, 0.2);

    bucketing.initialize(10);
    for (std::size_t p = 0; p < 10; ++p) {
        for (std::size_t i = 0; i < 100; ++i) {
            core_t::TTime x = static_cast<core_t::TTime>(100 * p + i);
            double y = 0.02 * (static_cast<double>(i) - 50.0) *
                       (static_cast<double>(i) - 50.0);
            bucketing.add(x, y, y);
        }
        bucketing.refine(static_cast<core_t::TTime>(100 * (p + 1)));
        bucketing.propagateForwardsByTime(1.0);
    }

    double lastCount = bucketing.count();
    for (std::size_t i = 0; i < 20; ++i) {
        bucketing.propagateForwardsByTime(1.0);
        double count = bucketing.count();
        LOG_DEBUG(<< "count = " << count << ", lastCount = " << lastCount
                  << " count/lastCount = " << count / lastCount);
        BOOST_TEST_REQUIRE(count < lastCount);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(0.81873, count / lastCount, 5e-6);
        lastCount = count;
    }
}

BOOST_AUTO_TEST_CASE(testMinimumBucketLength) {
    using TSizeVec = std::vector<std::size_t>;

    const double bucketLength{3600.0};
    const double function[]{0.0,  0.0, 10.0, 12.0, 11.0, 16.0,
                            15.0, 1.0, 0.0,  0.0,  0.0,  0.0};
    std::size_t n = boost::size(function);

    test::CRandomNumbers rng;

    core_t::TTime period = static_cast<core_t::TTime>(n) *
                           static_cast<core_t::TTime>(bucketLength);
    maths::CGeneralPeriodTime time(period);
    maths::CSeasonalComponentAdaptiveBucketing bucketing1(time, 0.0, 0.0);
    maths::CSeasonalComponentAdaptiveBucketing bucketing2(time, 0.0, 3000.0);
    bucketing1.initialize(n);
    bucketing2.initialize(n);

    TSizeVec indices(bucketing1.endpoints().size());
    std::iota(indices.begin(), indices.end(), 0);

    for (std::size_t i = 0; i < 3; ++i) {

        for (std::size_t j = 0; j < n; ++j) {
            TDoubleVec values;
            rng.generateNormalSamples(function[j], 1.0, 5, values);

            TDoubleVec times;
            rng.generateUniformSamples(0.0, bucketLength, values.size(), times);
            std::sort(times.begin(), times.end());

            for (std::size_t k = 0; k < times.size(); ++k) {
                core_t::TTime t{static_cast<core_t::TTime>(i) * period +
                                static_cast<core_t::TTime>(static_cast<double>(j) * bucketLength) +
                                static_cast<core_t::TTime>(times[k])};
                bucketing1.add(t, values[k], function[j]);
                bucketing2.add(t, values[k], function[j]);
            }
        }
        bucketing1.refine(static_cast<core_t::TTime>(i) * period);
        bucketing2.refine(static_cast<core_t::TTime>(i) * period);

        const TFloatVec& endpoints1{bucketing1.endpoints()};
        TFloatVec endpoints2{bucketing2.endpoints()};

        BOOST_REQUIRE_EQUAL(endpoints1.size(), endpoints2.size());

        // Check the separation constraint is satisfied.

        TMinAccumulator minimumBucketLength1;
        TMinAccumulator minimumBucketLength2;
        for (std::size_t j = 1; j < endpoints1.size(); ++j) {
            minimumBucketLength1.add(endpoints1[j] - endpoints1[j - 1]);
            minimumBucketLength2.add(endpoints2[j] - endpoints2[j - 1]);
        }
        LOG_DEBUG(<< "minimumBucketLength1 = " << minimumBucketLength1);
        LOG_DEBUG(<< "minimumBucketLength2 = " << minimumBucketLength2);
        BOOST_TEST_REQUIRE(minimumBucketLength2[0] >= 3000.0);

        double difference{std::accumulate(
            indices.begin(), indices.end(), 0.0, [&](double result, std::size_t index) {
                return result + std::fabs(endpoints2[index] - endpoints1[index]);
            })};

        // Check that perturbations of the endpoints which preserve the separation
        // constraint increase the difference from desired positions, i.e. that we
        // have minimized displacement to achieve the required separation constraint.
        for (std::size_t trial = 0; trial < 20; ++trial) {
            endpoints2 = bucketing2.endpoints();
            for (std::size_t j = 1; j + 1 < endpoints2.size(); ++j) {
                double a{endpoints2[j - 1] + 3000.0};
                double b{endpoints2[j + 1] - 3000.0};
                if (a < b) {
                    TDoubleVec shift;
                    rng.generateUniformSamples(a, b, 1, shift);
                    endpoints2[j] = 0.95 * endpoints2[j] + 0.05 * shift[0];
                }
            }
            for (std::size_t j = 1; j < endpoints1.size(); ++j) {
                minimumBucketLength2.add(endpoints2[j] - endpoints2[j - 1]);
            }
            BOOST_TEST_REQUIRE(minimumBucketLength2[0] >= 3000.0);

            double shiftedDifference{std::accumulate(
                indices.begin(), indices.end(), 0.0, [&](double result, std::size_t index) {
                    return result + std::fabs(endpoints2[index] - endpoints1[index]);
                })};

            LOG_TRACE(<< "difference         = " << difference);
            LOG_TRACE(<< "shifted difference = " << shiftedDifference);
            BOOST_TEST_REQUIRE(difference < shiftedDifference);
        }
    }
}

BOOST_AUTO_TEST_CASE(testUnintialized) {
    // Check that all the functions work and return the expected
    // values on an uninitialized bucketing.

    maths::CGeneralPeriodTime time(10);
    maths::CSeasonalComponentAdaptiveBucketing bucketing(time, 0.1);

    bucketing.add(0, 1.0, 1.0);
    bucketing.add(1, 2.0, 2.0);

    BOOST_TEST_REQUIRE(!bucketing.initialized());
    bucketing.propagateForwardsByTime(1.0);
    bucketing.refine(10);
    TDoubleVec knots;
    TDoubleVec values;
    TDoubleVec variances;
    bucketing.knots(10, maths::CSplineTypes::E_Periodic, knots, values, variances);
    BOOST_TEST_REQUIRE(knots.empty());
    BOOST_TEST_REQUIRE(values.empty());
    BOOST_TEST_REQUIRE(variances.empty());
    BOOST_REQUIRE_EQUAL(0.0, bucketing.count());
    BOOST_TEST_REQUIRE(bucketing.endpoints().empty());
    BOOST_TEST_REQUIRE(bucketing.values(100).empty());
    BOOST_TEST_REQUIRE(bucketing.variances().empty());

    bucketing.initialize(10);
    BOOST_TEST_REQUIRE(bucketing.initialized());
    for (std::size_t i = 0; i < 10; ++i) {
        core_t::TTime t = static_cast<core_t::TTime>(i);
        double v = static_cast<double>(t * t);
        bucketing.add(t, v, v);
    }
    bucketing.clear();
    BOOST_TEST_REQUIRE(!bucketing.initialized());
    bucketing.knots(10, maths::CSplineTypes::E_Periodic, knots, values, variances);
    BOOST_TEST_REQUIRE(knots.empty());
    BOOST_TEST_REQUIRE(values.empty());
    BOOST_TEST_REQUIRE(variances.empty());
    BOOST_REQUIRE_EQUAL(0.0, bucketing.count());
    BOOST_TEST_REQUIRE(bucketing.endpoints().empty());
    BOOST_TEST_REQUIRE(bucketing.values(100).empty());
    BOOST_TEST_REQUIRE(bucketing.variances().empty());
}

BOOST_AUTO_TEST_CASE(testKnots) {
    // Check prediction errors in values and variances.

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "*** Values ***");
    {
        maths::CDiurnalTime time(0, 0, core::constants::WEEK, core::constants::DAY);
        maths::CSeasonalComponentAdaptiveBucketing bucketing(time, 0.1, 864.0);

        bucketing.initialize(20);
        for (std::size_t p = 0; p < 5; ++p) {
            TDoubleVec noise;
            rng.generateNormalSamples(0.0, 4.0, 100, noise);

            for (std::size_t i = 0; i < 100; ++i) {
                core_t::TTime x = static_cast<core_t::TTime>(p * 86400 + 864 * i);
                double y = 0.02 * (static_cast<double>(i) - 50.0) *
                           (static_cast<double>(i) - 50.0);
                bucketing.add(x, y + noise[i], y);
            }
            bucketing.refine(static_cast<core_t::TTime>(86400 * (p + 1)));

            TDoubleVec knots;
            TDoubleVec values;
            TDoubleVec variances;
            bucketing.knots(static_cast<core_t::TTime>(86400 * (p + 1)),
                            maths::CSplineTypes::E_Periodic, knots, values, variances);
            LOG_DEBUG(<< "knots  = " << core::CContainerPrinter::print(knots));
            LOG_DEBUG(<< "values = " << core::CContainerPrinter::print(values));

            TMeanAccumulator meanError;
            TMeanAccumulator meanValue;
            for (std::size_t i = 0; i < knots.size(); ++i) {
                double x = knots[i] / 864.0;
                double expectedValue = 0.02 * (x - 50.0) * (x - 50.0);
                LOG_DEBUG(<< "expected = " << expectedValue << ", value = " << values[i]);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedValue, values[i], 15.0);
                meanError.add(std::fabs(values[i] - expectedValue));
                meanValue.add(std::fabs(expectedValue));
            }
            LOG_DEBUG(<< "meanError = " << maths::CBasicStatistics::mean(meanError));
            LOG_DEBUG(<< "meanValue = " << maths::CBasicStatistics::mean(meanValue));
            BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanError) /
                                   maths::CBasicStatistics::mean(meanValue) <
                               0.1 / std::sqrt(static_cast<double>(p + 1)));
        }
    }
    LOG_DEBUG(<< "*** Variances ***");
    {
        maths::CDiurnalTime time(0, 0, core::constants::WEEK, core::constants::DAY);
        maths::CSeasonalComponentAdaptiveBucketing bucketing(time, 0.1, 864.0);

        bucketing.initialize(20);
        for (std::size_t p = 0; p < 50; ++p) {
            TDoubleVec noise;

            for (std::size_t i = 0; i < 100; ++i) {
                core_t::TTime x = static_cast<core_t::TTime>(p * 86400 + 864 * i);
                double v = 0.01 * (static_cast<double>(i) - 50.0) *
                           (static_cast<double>(i) - 50.0);
                rng.generateNormalSamples(0.0, v, 1, noise);
                bucketing.add(x, noise[0], 0.0);
            }
            bucketing.refine(static_cast<core_t::TTime>(86400 * (p + 1)));

            if ((p + 1) % 10 == 0) {
                TDoubleVec knots;
                TDoubleVec values;
                TDoubleVec variances;
                bucketing.knots(static_cast<core_t::TTime>(86400 * (p + 1)),
                                maths::CSplineTypes::E_Periodic, knots, values, variances);
                LOG_DEBUG(<< "knots     = " << core::CContainerPrinter::print(knots));
                LOG_DEBUG(<< "variances = " << core::CContainerPrinter::print(variances));

                TMeanAccumulator meanError;
                TMeanAccumulator meanVariance;
                for (std::size_t i = 0; i < knots.size(); ++i) {
                    double x = knots[i] / 864.0;
                    double expectedVariance = 0.01 * (x - 50.0) * (x - 50.0);
                    LOG_DEBUG(<< "expected = " << expectedVariance
                              << ", variance = " << variances[i]);
                    BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedVariance, variances[i], 15.0);
                    meanError.add(std::fabs(variances[i] - expectedVariance));
                    meanVariance.add(std::fabs(expectedVariance));
                }
                LOG_DEBUG(<< "meanError    = " << maths::CBasicStatistics::mean(meanError));
                LOG_DEBUG(<< "meanVariance = " << maths::CBasicStatistics::mean(meanVariance));
                BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanError) /
                                       maths::CBasicStatistics::mean(meanVariance) <
                                   0.2);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testLongTermTrendKnots) {
    // Check prediction errors in values.

    test::CRandomNumbers rng;

    maths::CDiurnalTime time(0, 0, core::constants::WEEK, core::constants::DAY);
    maths::CSeasonalComponentAdaptiveBucketing bucketing(time, 0.1, 864.0);
    maths::CSeasonalComponentAdaptiveBucketing::TFloatMeanAccumulatorVec empty;

    bucketing.initialize(20);
    bucketing.initialValues(0, 0, empty);

    for (std::size_t p = 0; p < 100; ++p) {
        TDoubleVec noise;
        rng.generateNormalSamples(0.0, 100.0, 144, noise);

        for (std::size_t i = 0; i < 144; ++i) {
            double x = static_cast<double>(i) / 144.0;
            double y = 10.0 * (std::min(static_cast<double>(p + 1) + x, 50.0) -
                               std::max(static_cast<double>(p + 1) + x - 50.0, 0.0) +
                               10.0 * std::sin(boost::math::double_constants::two_pi * x));
            bucketing.add(static_cast<core_t::TTime>(86400 * p + 600 * i),
                          y + noise[i], y);
        }
        bucketing.refine(static_cast<core_t::TTime>(86400 * (p + 1)));
        bucketing.propagateForwardsByTime(1.0);

        if (p > 14 && (p + 1) % 5 == 0) {
            TDoubleVec knots;
            TDoubleVec values;
            TDoubleVec variances;
            bucketing.knots(static_cast<core_t::TTime>(86400 * (p + 1)),
                            maths::CSplineTypes::E_Periodic, knots, values, variances);
            LOG_DEBUG(<< "knots     = " << core::CContainerPrinter::print(knots));
            LOG_DEBUG(<< "values = " << core::CContainerPrinter::print(values));
            LOG_DEBUG(<< "variances = " << core::CContainerPrinter::print(variances));

            TMeanAccumulator meanError;
            TMeanAccumulator meanValue;
            for (std::size_t i = 0; i < knots.size(); ++i) {
                double x = knots[i] / 86400.0;
                double expectedValue =
                    10.0 * (std::min(static_cast<double>(p + 1) + x, 50.0) -
                            std::max(static_cast<double>(p + 1) + x - 50.0, 0.0) +
                            10.0 * std::sin(boost::math::double_constants::two_pi * x));
                LOG_DEBUG(<< "expected = " << expectedValue << ", value = " << values[i]);
                BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedValue, values[i], 70.0);
                meanError.add(std::fabs(values[i] - expectedValue));
                meanValue.add(std::fabs(expectedValue));
            }
            LOG_DEBUG(<< "meanError = " << maths::CBasicStatistics::mean(meanError));
            LOG_DEBUG(<< "meanValue = " << maths::CBasicStatistics::mean(meanValue));
            BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanError) /
                                   maths::CBasicStatistics::mean(meanValue) <
                               0.15);
        }
    }
}

BOOST_AUTO_TEST_CASE(testShiftValue) {
    // Test that applying a shift translates the predicted values
    // but doesn't alter the slope or predicted variances.

    maths::CDiurnalTime time(0, 0, core::constants::WEEK, core::constants::DAY);
    maths::CSeasonalComponentAdaptiveBucketing bucketing(time, 0.1, 600.0);
    maths::CSeasonalComponentAdaptiveBucketing::TFloatMeanAccumulatorVec empty;

    bucketing.initialize(20);
    bucketing.initialValues(0, 0, empty);

    core_t::TTime t = 0;
    for (/**/; t < 40 * 86400; t += 600) {
        double x = static_cast<double>(t) / 86400.0;
        double y = x + 20.0 + 20.0 * std::sin(boost::math::double_constants::two_pi * x);
        bucketing.add(t, y, y);
        if (t % 86400 == 0) {
            bucketing.refine(t);
            bucketing.propagateForwardsByTime(1.0);
        }
    }

    TDoubleVec knots1;
    TDoubleVec values1;
    TDoubleVec variances1;
    bucketing.knots(t + 7 * 86400, maths::CSplineTypes::E_Natural, knots1, values1, variances1);

    bucketing.shiftLevel(20.0);

    TDoubleVec knots2;
    TDoubleVec values2;
    TDoubleVec variances2;
    bucketing.knots(t + 7 * 86400, maths::CSplineTypes::E_Natural, knots2, values2, variances2);

    BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(knots1),
                        core::CContainerPrinter::print(knots2));
    BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(variances1),
                        core::CContainerPrinter::print(variances2));

    for (std::size_t i = 0; i < values1.size(); ++i) {
        LOG_DEBUG(<< "values = " << values1[i] << " vs " << values2[i]);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(20.0 + values1[i], values2[i], 1e-6 * values1[i]);
    }
}

BOOST_AUTO_TEST_CASE(testSlope) {
    // Test that the slope increases by the shift.

    maths::CDiurnalTime time(0, 0, core::constants::WEEK, core::constants::DAY);
    maths::CSeasonalComponentAdaptiveBucketing bucketing(time, 0.1, 600.0);
    maths::CSeasonalComponentAdaptiveBucketing::TFloatMeanAccumulatorVec empty;

    bucketing.initialize(20);
    bucketing.initialValues(0, 0, empty);

    core_t::TTime t = 0;
    for (/**/; t < 60 * 86400; t += 600) {
        double x = static_cast<double>(t) / 86400.0;
        double y = x + 20.0 + 20.0 * std::sin(boost::math::double_constants::two_pi * x);
        bucketing.add(t, y, y);
        if (t % 86400 == 0) {
            bucketing.refine(t);
            bucketing.propagateForwardsByTime(1.0);
        }
    }

    double slopeBefore = bucketing.slope();

    LOG_DEBUG(<< "slope = " << slopeBefore);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(7.0, slopeBefore, 0.25);

    bucketing.shiftSlope(t, 10.0);

    double slopeAfter = bucketing.slope();

    LOG_DEBUG(<< "slope = " << slopeAfter);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(slopeBefore + 10.0, slopeAfter, 1e-4);
}

BOOST_AUTO_TEST_CASE(testPersist) {
    // Check that serialization is idempotent.

    double decayRate = 0.1;
    double minimumBucketLength = 1.0;

    maths::CDiurnalTime time(0, 0, core::constants::WEEK, core::constants::DAY);
    maths::CSeasonalComponentAdaptiveBucketing origBucketing(time, decayRate, minimumBucketLength);

    origBucketing.initialize(10);
    for (std::size_t p = 0; p < 10; ++p) {
        for (std::size_t i = 0; i < 100; ++i) {
            core_t::TTime x = static_cast<core_t::TTime>(p * 86400 + 864 * i);
            double y = 0.02 * (static_cast<double>(i) - 50.0) *
                       (static_cast<double>(i) - 50.0);
            origBucketing.add(x, y, y);
        }
        origBucketing.refine(static_cast<core_t::TTime>(86400 * (p + 1)));
    }

    uint64_t checksum = origBucketing.checksum();

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origBucketing.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_DEBUG(<< "Bucketing XML representation:\n" << origXml);

    core::CRapidXmlParser parser;
    BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    // Restore the XML into a new bucketing.
    maths::CSeasonalComponentAdaptiveBucketing restoredBucketing(
        decayRate + 0.1, minimumBucketLength, traverser);

    LOG_DEBUG(<< "orig checksum = " << checksum
              << " restored checksum = " << restoredBucketing.checksum());
    BOOST_REQUIRE_EQUAL(checksum, restoredBucketing.checksum());

    // The XML representation of the new bucketing should be the
    // same as the original.
    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredBucketing.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    BOOST_REQUIRE_EQUAL(origXml, newXml);
}

BOOST_AUTO_TEST_CASE(testUpgrade) {
    // Check we can validly upgrade existing state.

    double decayRate = 0.1;
    double minimumBucketLength = 1.0;

    maths::CDiurnalTime time(0, 0, core::constants::WEEK, core::constants::DAY);
    maths::CSeasonalComponentAdaptiveBucketing expectedBucketing(time, decayRate, minimumBucketLength);

    expectedBucketing.initialize(10);
    for (std::size_t p = 0; p < 10; ++p) {
        for (std::size_t i = 0; i < 100; ++i) {
            core_t::TTime x = static_cast<core_t::TTime>(p * 86400 + 864 * i);
            double y = 0.02 * (static_cast<double>(i) - 50.0) *
                       (static_cast<double>(i) - 50.0);
            expectedBucketing.add(x, y, y);
        }
        expectedBucketing.refine(static_cast<core_t::TTime>(86400 * (p + 1)));
    }

    std::ifstream file;
    file.open("testfiles/CSeasonalComponentAdaptiveBucketing.6.2.state.xml");
    std::stringbuf buf;
    file >> &buf;
    std::string xml{buf.str()};
    LOG_DEBUG(<< "Saved state size = " << xml.size());

    core::CRapidXmlParser parser;
    BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(xml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    // Restore the XML into a new bucketing.
    maths::CSeasonalComponentAdaptiveBucketing restoredBucketing(
        decayRate + 0.1, minimumBucketLength, traverser);

    // Check that the knots points we get back are very nearly
    // those we expect.

    TDoubleVec expectedKnots;
    TDoubleVec expectedValues;
    TDoubleVec expectedVariances;
    expectedBucketing.knots(863136, maths::CSplineTypes::E_Periodic,
                            expectedKnots, expectedValues, expectedVariances);

    TDoubleVec restoredKnots;
    TDoubleVec restoredValues;
    TDoubleVec restoredVariances;
    expectedBucketing.knots(863136, maths::CSplineTypes::E_Periodic,
                            restoredKnots, restoredValues, restoredVariances);

    BOOST_REQUIRE_EQUAL(expectedBucketing.decayRate(), restoredBucketing.decayRate());

    LOG_DEBUG(<< "expected knots = " << core::CContainerPrinter::print(expectedKnots));
    LOG_DEBUG(<< "restored knots = " << core::CContainerPrinter::print(restoredKnots));
    BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedKnots),
                        core::CContainerPrinter::print(restoredKnots));

    LOG_DEBUG(<< "expected values = " << core::CContainerPrinter::print(expectedValues));
    LOG_DEBUG(<< "restored values = " << core::CContainerPrinter::print(restoredValues));
    BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedValues),
                        core::CContainerPrinter::print(restoredValues));

    LOG_DEBUG(<< "expected variances = " << core::CContainerPrinter::print(expectedVariances));
    LOG_DEBUG(<< "restored variances = " << core::CContainerPrinter::print(restoredVariances));
    BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedVariances),
                        core::CContainerPrinter::print(restoredVariances));
}

BOOST_AUTO_TEST_CASE(testName) {
    double decayRate{0.1};
    double minimumBucketLength{1.0};

    maths::CDiurnalTime time(0, 0, core::constants::WEEK, core::constants::DAY);
    maths::CSeasonalComponentAdaptiveBucketing bucketing(time, decayRate, minimumBucketLength);

    BOOST_REQUIRE_EQUAL(std::string("Seasonal[") + std::to_string(decayRate) +
                            "," + std::to_string(minimumBucketLength) + "]",
                        bucketing.name());
}

BOOST_AUTO_TEST_SUITE_END()
