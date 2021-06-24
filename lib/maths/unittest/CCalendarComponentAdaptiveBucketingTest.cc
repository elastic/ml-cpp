/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CTimeUtils.h>
#include <core/CTimezone.h>
#include <core/Constants.h>

#include <maths/CCalendarComponentAdaptiveBucketing.h>
#include <maths/CCalendarFeature.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

#include <numeric>
#include <vector>

BOOST_AUTO_TEST_SUITE(CCalendarComponentAdaptiveBucketingTest)

using namespace ml;

namespace {
using TDoubleVec = std::vector<double>;
using TFloatVec = std::vector<maths::CFloatStorage>;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMinAccumulator = maths::CBasicStatistics::SMin<double>::TAccumulator;
using TMaxAccumulator = maths::CBasicStatistics::SMax<double>::TAccumulator;
}

class CTestFixture {
public:
    CTestFixture()
        : m_OrigTimezone{core::CTimezone::instance().timezoneName()} {
        core::CTimezone::instance().setTimezone("GMT");
    }

    ~CTestFixture() { core::CTimezone::instance().setTimezone(m_OrigTimezone); }

private:
    std::string m_OrigTimezone;
};

BOOST_FIXTURE_TEST_CASE(testInitialize, CTestFixture) {
    maths::CCalendarFeature feature{maths::CCalendarFeature::DAYS_SINCE_START_OF_MONTH, 86400};
    maths::CCalendarComponentAdaptiveBucketing bucketing{feature};

    BOOST_TEST_REQUIRE(!bucketing.initialize(0));

    const std::string expectedEndpoints{
        "[0, 7200, 14400, 21600, 28800, 36000, 43200, 50400, 57600, 64800, 72000, 79200, 86400]"};
    const std::string expectedKnots{
        "[0, 3600, 10800, 18000, 25200, 32400, 39600, 46800, 54000, 61200, 68400, 75600, 82800, 86400]"};
    const std::string expectedValues{
        "[129600, 90000, 97200, 104400, 111600, 118800, 126000, 133200, 140400, 147600, 154800, 162000, 169200, 129600]"};

    BOOST_TEST_REQUIRE(bucketing.initialize(12));
    const TFloatVec& endpoints{bucketing.endpoints()};
    BOOST_REQUIRE_EQUAL(expectedEndpoints, core::CContainerPrinter::print(endpoints));

    for (core_t::TTime t = 86400 + 3600; t < 172800; t += 7200) {
        bucketing.add(t, static_cast<double>(t));
    }
    TDoubleVec knots;
    TDoubleVec values;
    TDoubleVec variances;
    bucketing.knots(1, maths::CSplineTypes::E_Periodic, knots, values, variances);
    BOOST_REQUIRE_EQUAL(expectedKnots, core::CContainerPrinter::print(knots));
    BOOST_REQUIRE_EQUAL(expectedValues, core::CContainerPrinter::print(values));
}

BOOST_FIXTURE_TEST_CASE(testSwap, CTestFixture) {
    core_t::TTime now{core::CTimeUtils::now()};

    maths::CCalendarFeature feature1{maths::CCalendarFeature::DAYS_SINCE_START_OF_MONTH, now};
    maths::CCalendarComponentAdaptiveBucketing bucketing1{feature1, 0.05};

    test::CRandomNumbers rng;

    bucketing1.initialize(10);
    for (std::size_t p = 0; p < 50; ++p) {
        TDoubleVec noise;
        rng.generateNormalSamples(0.0, 2.0, 100, noise);

        core_t::TTime start{now + static_cast<core_t::TTime>(86400 * p)};
        for (std::size_t i = 0; i < 100; ++i) {
            core_t::TTime t{start + static_cast<core_t::TTime>(864 * i)};
            if (bucketing1.feature().inWindow(t)) {
                double y{0.02 * (static_cast<double>(i) - 50.0) *
                         (static_cast<double>(i) - 50.0)};
                bucketing1.add(t, y + noise[i]);
            }
        }
        bucketing1.refine(start + 86400);
        bucketing1.propagateForwardsByTime(1.0);
    }

    maths::CCalendarFeature feature2{maths::CCalendarFeature::DAYS_BEFORE_END_OF_MONTH,
                                     now - core::constants::WEEK};
    maths::CCalendarComponentAdaptiveBucketing bucketing2{feature2, 0.1};

    uint64_t checksum1{bucketing1.checksum()};
    uint64_t checksum2{bucketing2.checksum()};

    bucketing1.swap(bucketing2);

    LOG_DEBUG(<< "checksum 1 = " << checksum1);
    LOG_DEBUG(<< "checksum 2 = " << checksum2);

    BOOST_REQUIRE_EQUAL(checksum1, bucketing2.checksum());
    BOOST_REQUIRE_EQUAL(checksum2, bucketing1.checksum());
}

BOOST_FIXTURE_TEST_CASE(testRefine, CTestFixture) {
    // Test that refine reduces the function approximation error.

    core_t::TTime times[] = {-1,    3600,  10800, 18000, 25200, 32400, 39600,
                             46800, 54000, 61200, 68400, 75600, 82800, 86400};
    double function[] = {10, 10,  10, 10, 100, 90, 80,
                         90, 100, 20, 10, 10,  10, 10};

    maths::CCalendarFeature feature{maths::CCalendarFeature::DAYS_SINCE_START_OF_MONTH, 0};
    maths::CCalendarComponentAdaptiveBucketing bucketing1{feature};
    maths::CCalendarComponentAdaptiveBucketing bucketing2{feature};

    bucketing1.initialize(12);
    bucketing2.initialize(12);

    test::CRandomNumbers rng;

    bool lastInWindow{true};
    for (core_t::TTime t = 0; t < 31536000; t += 1800) {
        bool inWindow{bucketing1.feature().inWindow(t)};
        if (inWindow) {
            core_t::TTime x{bucketing1.feature().offset(t)};
            ptrdiff_t i{std::lower_bound(std::begin(times), std::end(times), x) -
                        std::begin(times)};
            double x0{static_cast<double>(times[i - 1])};
            double x1{static_cast<double>(times[i])};
            double y0{function[i - 1]};
            double y1{function[i]};
            double y{y0 + (y1 - y0) * (static_cast<double>(x) - x0) / (x1 - x0)};
            TDoubleVec noise;
            rng.generateNormalSamples(0.0, 4.0, 1, noise);
            bucketing1.add(t, y + noise[0]);
            bucketing2.add(t, y + noise[0]);
        } else if (lastInWindow && !inWindow) {
            bucketing2.refine(t);
        }
        lastInWindow = inWindow;
    }

    TMeanAccumulator meanError1;
    TMaxAccumulator maxError1;
    const TFloatVec& endpoints1{bucketing1.endpoints()};
    TDoubleVec values1{bucketing1.values(20 * 86400)};
    for (std::size_t i = 1; i < endpoints1.size(); ++i) {
        core_t::TTime t{static_cast<core_t::TTime>(
            0.5 * (endpoints1[i] + endpoints1[i - 1] + 1.0))};
        ptrdiff_t j{std::lower_bound(std::begin(times), std::end(times), t) -
                    std::begin(times)};
        double x0{static_cast<double>(times[j - 1])};
        double x1{static_cast<double>(times[j])};
        double y0{function[j - 1]};
        double y1{function[j]};
        double y{y0 + (y1 - y0) * (static_cast<double>(t) - x0) / (x1 - x0)};
        meanError1.add(std::fabs(values1[i - 1] - y));
        maxError1.add(std::fabs(values1[i - 1] - y));
    }

    TMeanAccumulator meanError2;
    TMaxAccumulator maxError2;
    const TFloatVec& endpoints2{bucketing2.endpoints()};
    TDoubleVec values2{bucketing2.values(20 * 86400)};
    for (std::size_t i = 1; i < endpoints1.size(); ++i) {
        core_t::TTime t{static_cast<core_t::TTime>(
            0.5 * (endpoints2[i] + endpoints2[i - 1] + 1.0))};
        ptrdiff_t j{std::lower_bound(std::begin(times), std::end(times), t) -
                    std::begin(times)};
        double x0{static_cast<double>(times[j - 1])};
        double x1{static_cast<double>(times[j])};
        double y0{function[j - 1]};
        double y1{function[j]};
        double y{y0 + (y1 - y0) * (static_cast<double>(t) - x0) / (x1 - x0)};
        meanError2.add(std::fabs(values2[i - 1] - y));
        maxError2.add(std::fabs(values2[i - 1] - y));
    }

    LOG_DEBUG(<< "mean error         = " << maths::CBasicStatistics::mean(meanError1));
    LOG_DEBUG(<< "max error          = " << maxError1[0]);
    LOG_DEBUG(<< "refined mean error = " << maths::CBasicStatistics::mean(meanError2));
    LOG_DEBUG(<< "refined max error  = " << maxError2[0]);
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanError2) <
                       0.7 * maths::CBasicStatistics::mean(meanError1));
    BOOST_TEST_REQUIRE(maxError2[0] < 0.65 * maxError1[0]);
}

BOOST_FIXTURE_TEST_CASE(testPropagateForwardsByTime, CTestFixture) {
    // Check no error is introduced by the aging process to
    // the bucket values and that the rate at which the total
    // count is reduced uniformly.

    maths::CCalendarFeature feature{maths::CCalendarFeature::DAYS_SINCE_START_OF_MONTH, 0};
    maths::CCalendarComponentAdaptiveBucketing bucketing{feature, 0.2};

    bucketing.initialize(10);
    for (core_t::TTime t = 0; t < 86400; t += 1800) {
        double y = 10.0 * (static_cast<double>(t) - 43200.0) / 43200.0 *
                   (static_cast<double>(t) - 43200.0) / 43200.0;
        bucketing.add(t, y);
    }
    bucketing.refine(86400);
    bucketing.propagateForwardsByTime(1.0);

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

BOOST_FIXTURE_TEST_CASE(testMinimumBucketLength, CTestFixture) {
    using TSizeVec = std::vector<std::size_t>;

    double function[]{0.0, 0.0, 10.0, 12.0, 11.0, 16.0, 15.0, 1.0,
                      0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                      0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0};
    std::size_t n{boost::size(function)};

    test::CRandomNumbers rng;

    maths::CCalendarFeature feature{maths::CCalendarFeature::DAYS_SINCE_START_OF_MONTH, 0};
    maths::CCalendarComponentAdaptiveBucketing bucketing1{feature, 0.0, 0.0};
    maths::CCalendarComponentAdaptiveBucketing bucketing2{feature, 0.0, 1500.0};
    bucketing1.initialize(n);
    bucketing2.initialize(n);

    TSizeVec indices(bucketing1.endpoints().size());
    std::iota(indices.begin(), indices.end(), 0);

    for (std::size_t i = 0; i < 7; ++i) {
        for (core_t::TTime t = 0; t < 86400; t += 3600) {
            TDoubleVec values;
            rng.generateNormalSamples(function[t / 3600], 1.0, 6, values);

            for (core_t::TTime dt = 0; dt < 3600; dt += 600) {
                bucketing1.add(t + dt, values[dt / 600]);
                bucketing2.add(t + dt, values[dt / 600]);
            }
        }
        bucketing1.refine(86400);
        bucketing2.refine(86400);

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
        BOOST_TEST_REQUIRE(minimumBucketLength2[0] >= 1500.0);

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
                double a{endpoints2[j - 1] + 1500.0};
                double b{endpoints2[j + 1] - 1500.0};
                if (a < b) {
                    TDoubleVec shift;
                    rng.generateUniformSamples(a, b, 1, shift);
                    endpoints2[j] = 0.95 * endpoints2[j] + 0.05 * shift[0];
                }
            }
            for (std::size_t j = 1; j < endpoints1.size(); ++j) {
                minimumBucketLength2.add(endpoints2[j] - endpoints2[j - 1]);
            }
            BOOST_TEST_REQUIRE(minimumBucketLength2[0] >= 1500.0);

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

BOOST_FIXTURE_TEST_CASE(testUnintialized, CTestFixture) {
    // Check that all the functions work and return the expected
    // values on an uninitialized bucketing.

    maths::CCalendarFeature feature{maths::CCalendarFeature::DAYS_SINCE_START_OF_MONTH, 0};
    maths::CCalendarComponentAdaptiveBucketing bucketing{feature, 0.1};

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
    for (core_t::TTime t = 0; t < 86400; t += 8640) {
        bucketing.add(t, static_cast<double>(t * t));
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

BOOST_FIXTURE_TEST_CASE(testKnots, CTestFixture) {
    // Check prediction errors in values and variances.

    test::CRandomNumbers rng;

    maths::CCalendarFeature feature{maths::CCalendarFeature::DAYS_SINCE_START_OF_MONTH, 0};

    LOG_DEBUG(<< "*** Values ***");
    {
        maths::CCalendarComponentAdaptiveBucketing bucketing{feature, 0.0, 600.0};

        bucketing.initialize(24);

        for (core_t::TTime t = 0; t < 86400; t += 600) {
            double y{0.0002 * (static_cast<double>(t) - 43800.0) *
                     (static_cast<double>(t) - 43800.0)};
            TDoubleVec noise;
            rng.generateNormalSamples(0.0, 4.0, 1, noise);
            bucketing.add(t, y + noise[0]);
        }
        bucketing.refine(86400);

        TDoubleVec knots;
        TDoubleVec values;
        TDoubleVec variances;
        bucketing.knots(86400, maths::CSplineTypes::E_Periodic, knots, values, variances);
        LOG_DEBUG(<< "knots  = " << core::CContainerPrinter::print(knots));
        LOG_DEBUG(<< "values = " << core::CContainerPrinter::print(values));

        TMeanAccumulator meanError;
        TMeanAccumulator meanValue;
        for (std::size_t i = 0; i < knots.size(); ++i) {
            double expectedValue{0.0002 * (knots[i] - 43800.0) * (knots[i] - 43800.0)};
            LOG_DEBUG(<< "expected = " << expectedValue << ", value = " << values[i]);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedValue, values[i], 50000.0);
            meanError.add(std::fabs(values[i] - expectedValue));
            meanValue.add(std::fabs(expectedValue));
        }
        LOG_DEBUG(<< "meanError = " << maths::CBasicStatistics::mean(meanError));
        LOG_DEBUG(<< "meanValue = " << maths::CBasicStatistics::mean(meanValue));
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanError) /
                               maths::CBasicStatistics::mean(meanValue) <
                           0.02);
    }

    LOG_DEBUG(<< "*** Variances ***");
    {
        maths::CCalendarComponentAdaptiveBucketing bucketing{feature, 0.0, 600.0};

        bucketing.initialize(24);

        bool lastInWindow{true};
        for (core_t::TTime t = 0; t < 15638400; t += 600) {
            bool inWindow{bucketing.feature().inWindow(t)};
            if (inWindow) {
                double x = static_cast<double>(bucketing.feature().offset(t));
                double v{0.001 * (x - 43800.0) * (x - 43800.0) / 86400};
                TDoubleVec noise;
                rng.generateNormalSamples(0.0, v, 1, noise);
                bucketing.add(t, noise[0]);
            } else if (lastInWindow && !inWindow) {
                bucketing.refine(t);
            }
            lastInWindow = inWindow;
        }

        TDoubleVec knots;
        TDoubleVec values;
        TDoubleVec variances;
        bucketing.knots(13996800, maths::CSplineTypes::E_Periodic, knots, values, variances);
        LOG_DEBUG(<< "knots     = " << core::CContainerPrinter::print(knots));
        LOG_DEBUG(<< "variances = " << core::CContainerPrinter::print(variances));

        TMeanAccumulator meanError;
        TMeanAccumulator meanVariance;
        for (std::size_t i = 0; i < knots.size(); ++i) {
            double expectedVariance{0.001 * (static_cast<double>(knots[i]) - 43800.0) *
                                    (static_cast<double>(knots[i]) - 43800.0) / 86400};
            LOG_DEBUG(<< "expected = " << expectedVariance
                      << ", variance = " << variances[i]);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedVariance, variances[i], 5.0);
            meanError.add(std::fabs(variances[i] - expectedVariance));
            meanVariance.add(std::fabs(expectedVariance));
        }
        LOG_DEBUG(<< "meanError    = " << maths::CBasicStatistics::mean(meanError));
        LOG_DEBUG(<< "meanVariance = " << maths::CBasicStatistics::mean(meanVariance));
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanError) /
                               maths::CBasicStatistics::mean(meanVariance) <
                           0.16);
    }
}

BOOST_FIXTURE_TEST_CASE(testPersist, CTestFixture) {
    // Check that serialization is idempotent.

    double decayRate{0.1};
    double minimumBucketLength{1.0};

    maths::CCalendarFeature feature{maths::CCalendarFeature::DAYS_SINCE_START_OF_MONTH, 0};
    maths::CCalendarComponentAdaptiveBucketing bucketing{feature, decayRate, minimumBucketLength};

    bucketing.initialize(10);
    for (std::size_t p = 0; p < 10; ++p) {
        for (std::size_t i = 0; i < 100; ++i) {
            core_t::TTime t{static_cast<core_t::TTime>(p * 86400 + 864 * i)};
            if (bucketing.feature().inWindow(t)) {
                double y{0.02 * (static_cast<double>(i) - 50.0) *
                         (static_cast<double>(i) - 50.0)};
                bucketing.add(t, y);
            }
        }
        bucketing.refine(static_cast<core_t::TTime>(86400 * (p + 1)));
    }

    uint64_t checksum{bucketing.checksum()};

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter{"root"};
        bucketing.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_DEBUG(<< "Bucketing XML representation:\n" << origXml);

    core::CRapidXmlParser parser;
    BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    // Restore the XML into a new bucketing.
    maths::CCalendarComponentAdaptiveBucketing restoredBucketing{
        decayRate + 0.1, minimumBucketLength, traverser};

    LOG_DEBUG(<< "orig checksum = " << checksum
              << " restored checksum = " << restoredBucketing.checksum());
    BOOST_REQUIRE_EQUAL(checksum, restoredBucketing.checksum());

    // The XML representation of the new bucketing should be the
    // same as the original.
    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter{"root"};
        restoredBucketing.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    BOOST_REQUIRE_EQUAL(origXml, newXml);
}

BOOST_FIXTURE_TEST_CASE(testName, CTestFixture) {
    double decayRate{0.1};
    double minimumBucketLength{1.0};

    maths::CCalendarFeature feature{maths::CCalendarFeature::DAYS_SINCE_START_OF_MONTH, 0};
    maths::CCalendarComponentAdaptiveBucketing bucketing{feature, decayRate, minimumBucketLength};

    BOOST_REQUIRE_EQUAL(std::string("Calendar[") + std::to_string(decayRate) +
                            "," + std::to_string(minimumBucketLength) + "]",
                        bucketing.name());
}

BOOST_AUTO_TEST_SUITE_END()
