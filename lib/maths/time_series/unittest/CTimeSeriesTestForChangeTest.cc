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
#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/Constants.h>

#include <maths/common/CBasicStatistics.h>

#include <maths/time_series/CTimeSeriesTestForChange.h>

#include <test/CRandomNumbers.h>

#include "TestUtils.h"

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <functional>
#include <memory>
#include <utility>

BOOST_AUTO_TEST_SUITE(CTimeSeriesTestForChangeTest)

using namespace ml;

namespace {

using TDoubleVec = std::vector<double>;
using TGenerator = std::function<double(core_t::TTime)>;
using TGeneratorVec = std::vector<TGenerator>;
using TChangePointUPtr = std::unique_ptr<maths::time_series::CChangePoint>;
using TChange = std::function<double(TGenerator generator, core_t::TTime)>;
using TMeanAccumulator = maths::common::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TFloatMeanAccumulatorVec = maths::time_series::CTimeSeriesTestForChange::TFloatMeanAccumulatorVec;

core_t::TTime BUCKET_LENGTH{core::constants::HOUR / 2};

void testChange(const TGeneratorVec& trends,
                TChange applyChange,
                const std::string& expectedChangeType,
                double expectedChange,
                std::size_t changeTime = 50) {

    test::CRandomNumbers rng;

    core_t::TTime startTime{100000};
    std::size_t numberTests{100};

    double truePositives{0.0};
    TMeanAccumulator meanError;
    TMeanAccumulator meanTimeError;

    TDoubleVec samples;
    for (std::size_t test = 0; test < numberTests; ++test) {
        if (test % 10 == 0) {
            LOG_DEBUG(<< test << "%");
        }

        const auto& trend = trends[test % trends.size()];

        rng.generateNormalSamples(0.0, 1.0, 100, samples);

        TFloatMeanAccumulatorVec values(samples.size());
        core_t::TTime time{startTime};
        for (std::size_t i = 0; i < changeTime; ++i, time += BUCKET_LENGTH) {
            values[i].add(10.0 * trend(time) + samples[i]);
        }
        for (std::size_t i = changeTime; i < samples.size(); ++i, time += BUCKET_LENGTH) {
            values[i].add(10.0 * applyChange(trend, time) + samples[i]);
        }
        auto predictor = [&](core_t::TTime time_) { return 10.0 * trend(time_); };

        maths::time_series::CTimeSeriesTestForChange testForChange(
            maths::time_series::CTimeSeriesTestForChange::E_All, startTime,
            startTime, BUCKET_LENGTH, BUCKET_LENGTH, predictor, std::move(values));

        auto change = testForChange.test();

        std::string actualChangeType{(change == nullptr ? "null" : change->type())};
        truePositives += actualChangeType == expectedChangeType ? 1.0 : 0.0;

        if (change == nullptr) {
            continue;
        }

        LOG_TRACE(<< change->print() << " at " << change->time());
        if (actualChangeType == expectedChangeType) {
            auto expectedChangeTime =
                startTime + static_cast<core_t::TTime>(changeTime) * BUCKET_LENGTH;
            meanError.add(std::fabs(expectedChange - change->value()) / expectedChange);
            meanTimeError(
                static_cast<double>(std::abs(expectedChangeTime - change->time())));
        }
    }

    LOG_DEBUG(<< "true positives = " << truePositives);
    LOG_DEBUG(<< "mean error = " << maths::common::CBasicStatistics::mean(meanError));
    LOG_DEBUG(<< "mean time error = " << maths::common::CBasicStatistics::mean(meanTimeError));

    truePositives /= static_cast<double>(numberTests);

    BOOST_TEST_REQUIRE(truePositives >= 0.98);
    BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(meanError) < 0.04);
    BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(meanTimeError) < 5000);
}
}

BOOST_AUTO_TEST_CASE(testNoChange) {

    // Test we don't identify a change when none is occurring.

    test::CRandomNumbers rng;

    TDoubleVec variances{1.0, 10.0, 20.0, 30.0, 100.0, 1000.0};
    TDoubleVec scales{0.1, 1.0, 2.0, 3.0, 5.0, 8.0};

    double trueNegatives{0.0};
    std::size_t numberTests{100};

    TDoubleVec noise;
    for (std::size_t test = 0; test < numberTests; ++test) {
        if (test % 10 == 0) {
            LOG_DEBUG(<< test << "%");
        }

        switch (test % 3) {
        case 0:
            rng.generateNormalSamples(
                10.0, variances[(test / 3) % variances.size()], 100, noise);
            break;
        case 1:
            rng.generateLogNormalSamples(1.0, scales[(test / 3) % scales.size()], 100, noise);
            break;
        case 2:
            rng.generateGammaSamples(
                10.0, 10.0 * scales[(test / 3) % scales.size()], 100, noise);
            break;
        }

        TMeanAccumulator mean;
        TFloatMeanAccumulatorVec values(noise.size());
        for (std::size_t i = 0; i < noise.size(); ++i) {
            values[i].add(noise[i]);
            mean.add(noise[i]);
        }
        auto predictor = [&](core_t::TTime) {
            return maths::common::CBasicStatistics::mean(mean);
        };

        maths::time_series::CTimeSeriesTestForChange testForChange(
            maths::time_series::CTimeSeriesTestForChange::E_All, 10000, 10000,
            BUCKET_LENGTH, BUCKET_LENGTH, predictor, std::move(values));

        auto change = testForChange.test();

        if (change != nullptr) {
            LOG_DEBUG(<< change->print());
        }
        trueNegatives += change == nullptr ? 1.0 : 0.0;
    }

    trueNegatives /= static_cast<double>(numberTests);

    BOOST_TEST_REQUIRE(trueNegatives >= 0.99);
}

BOOST_AUTO_TEST_CASE(testLevelShift) {
    TGeneratorVec trends{smoothDaily, weekends, spikeyDaily};
    testChange(
        trends,
        [](TGenerator trend, core_t::TTime time) { return trend(time) + 0.7; },
        maths::time_series::CLevelShift::TYPE, 7.0);
}

BOOST_AUTO_TEST_CASE(testScale) {
    TGeneratorVec trends{smoothDaily, spikeyDaily};
    testChange(
        trends,
        [](TGenerator trend, core_t::TTime time) { return 3.0 * trend(time); },
        maths::time_series::CScale::TYPE, 3.0);
}

BOOST_AUTO_TEST_CASE(testTimeShift) {
    TGeneratorVec trends{
        [](core_t::TTime time) { return 2.0 * smoothDaily(time); },
        [](core_t::TTime time) { return 2.0 * spikeyDaily(time); }};
    testChange(trends,
               [](TGenerator trend, core_t::TTime time) {
                   return trend(time - core::constants::HOUR);
               },
               maths::time_series::CTimeShift::TYPE,
               -static_cast<double>(core::constants::HOUR));
    testChange(trends,
               [](TGenerator trend, core_t::TTime time) {
                   return trend(time + core::constants::HOUR);
               },
               maths::time_series::CTimeShift::TYPE,
               +static_cast<double>(core::constants::HOUR));
}

BOOST_AUTO_TEST_CASE(testTimeShiftFullWindow) {

    // Test the case we only get to observe a shifted window, i.e. we have no
    // visible change in the window. This can happen for example if the time
    // series is flat in parts.

    TGeneratorVec trends{
        [](core_t::TTime time) { return 2.0 * smoothDaily(time); }};
    testChange(trends,
               [](TGenerator trend, core_t::TTime time) {
                   return trend(time - core::constants::HOUR);
               },
               maths::time_series::CTimeShift::TYPE,
               -static_cast<double>(core::constants::HOUR), 0 /*change time*/);
    testChange(trends,
               [](TGenerator trend, core_t::TTime time) {
                   return trend(time + core::constants::HOUR);
               },
               maths::time_series::CTimeShift::TYPE,
               +static_cast<double>(core::constants::HOUR), 0 /*change time*/);
}

BOOST_AUTO_TEST_CASE(testWithReversion) {

    // Test we handle temporary changes correctly.

    using TTransform = std::function<double(double)>;
    using TTransformVec = std::vector<TTransform>;

    test::CRandomNumbers rng;

    core_t::TTime startTime{100000};
    double noiseVariance{1.0};
    TGeneratorVec trends{smoothDaily, weekends, spikeyDaily};
    TTransformVec transforms{[](double x) { return x + 10.0; },
                             [](double x) { return 3.0 * x; }};

    TDoubleVec samples;
    for (std::size_t test = 0; test < 100; ++test) {
        if (test % 10 == 0) {
            LOG_DEBUG(<< test << "%");
        }

        const auto& trend = trends[test % trends.size()];
        const auto& transform = transforms[test % transforms.size()];
        auto predictor = [&](core_t::TTime time_) { return 10.0 * trend(time_); };

        rng.generateNormalSamples(0.0, noiseVariance, 100, samples);

        TFloatMeanAccumulatorVec values(samples.size());
        core_t::TTime time{startTime};
        for (std::size_t i = 0; i < 60; ++i, time += BUCKET_LENGTH) {
            values[i].add(predictor(time) + samples[i]);
        }
        for (std::size_t i = 60; i < 80; ++i, time += BUCKET_LENGTH) {
            values[i].add(transform(predictor(time)) + samples[i]);
        }
        for (std::size_t i = 80; i < samples.size(); ++i, time += BUCKET_LENGTH) {
            values[i].add(predictor(time) + samples[i]);
        }

        maths::time_series::CTimeSeriesTestForChange testForChange(
            maths::time_series::CTimeSeriesTestForChange::E_All, startTime,
            startTime, BUCKET_LENGTH, BUCKET_LENGTH, predictor, std::move(values));

        auto change = testForChange.test();

        LOG_TRACE(<< (change == nullptr ? "null" : change->print()));
        if (change != nullptr) {
            BOOST_TEST_REQUIRE(change->largeEnough(3.0 * std::sqrt(noiseVariance)) == false);
        }
    }
}

BOOST_AUTO_TEST_CASE(testPersist) {

    // Test we persist and restore change points correctly.

    maths::time_series::CUndoableChangePointStateSerializer serializer;

    TChangePointUPtr origChangePoint{
        std::make_unique<maths::time_series::CTimeShift>(500, -1800, 0.01, 10.0)};

    std::ostringstream origJson;
    {
        core::CJsonStatePersistInserter inserter{origJson};
        serializer(*origChangePoint, inserter);
    }
    LOG_DEBUG(<< "original sketch JSON = " << origJson.str());

    TChangePointUPtr restoredChangePoint;
    {
        std::istringstream origJsonStrm{"{\"topLevel\" : " + origJson.str() + "}"};
        core::CJsonStateRestoreTraverser traverser{origJsonStrm};
        BOOST_TEST_REQUIRE(traverser.traverseSubLevel(std::bind(
            serializer, std::ref(restoredChangePoint), std::placeholders::_1)));
    }

    BOOST_REQUIRE_EQUAL(origChangePoint->checksum(), restoredChangePoint->checksum());
}

BOOST_AUTO_TEST_SUITE_END()
