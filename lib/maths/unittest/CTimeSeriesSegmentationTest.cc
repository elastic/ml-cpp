/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CTimeSeriesSegmentationTest.h"

#include <core/CoreTypes.h>
#include <core/Constants.h>

#include <maths/CTimeSeriesSegmentation.h>

#include <test/CRandomNumbers.h>

#include "TestUtils.h"

#include <vector>

using namespace ml;

using TDoubleVec = std::vector<double>;
using TFloatMeanAccumulator = maths::CBasicStatistics::SSampleMean<maths::CFloatStorage>::TAccumulator;
using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;

void CTimeSeriesSegmentationTest::testTopDownPiecewiseLinear() {

    core_t::TTime halfHour{core::constants::HOUR / 2};
    core_t::TTime week{core::constants::WEEK};
    core_t::TTime range{5 * week};

    test::CRandomNumbers rng;

    TFloatMeanAccumulatorVec values(range / halfHour);

    LOG_DEBUG(<< "Basic");

    TDoubleVec noise;
    for (core_t::TTime time = 0; time < range; time += halfHour) {
        rng.generateNormalSamples(0.0, 3.0, 1, noise);
        if (time < 2 * week) {
            values[time / halfHour].add(3.0 + 200.0 * ramp(time) + noise[0]);
        } else if (time < 3 * week) {
            values[time / halfHour].add(20.0 - 100.0 * ramp(time) + noise[0]);
        } else {
            values[time / halfHour].add(50.0 * ramp(time) - 25.0 + noise[0]);
        }
    }

    maths::CTimeSeriesSegmentation::topDownPiecewiseLinear(values);

    LOG_DEBUG(<< "With Outliers");
}

void CTimeSeriesSegmentationTest::testTopDownPeriodicPiecewiseLinearScaling() {

    core_t::TTime halfHour{core::constants::HOUR / 2};
    core_t::TTime week{core::constants::WEEK};
    core_t::TTime range{5 * week};

    test::CRandomNumbers rng;

    TFloatMeanAccumulatorVec values(range / halfHour);
    TDoubleVec noise;

    LOG_DEBUG(<< "Basic");
    for (auto periodic : {smoothDaily, spikeyDaily}) {
        for (core_t::TTime time = 0; time < range; time += halfHour) {
            rng.generateNormalSamples(0.0, 3.0, 1, noise);
            if (time < 3 * week / 2) {
                values[time / halfHour].add(100.0 * periodic(time) + noise[0]);
            } else if (time < 2 * week) {
                values[time / halfHour].add(50.0 * periodic(time) + noise[0]);
            } else {
                values[time / halfHour].add(300.0 * periodic(time) + noise[0]);
            }
        }

        TFloatMeanAccumulatorVec valuesMinusTrend(
                maths::CTimeSeriesSegmentation::topDownPeriodicPiecewiseLinearScaling(values, 48));
    }

    LOG_DEBUG(<< "With Outliers");
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
