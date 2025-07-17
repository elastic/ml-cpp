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

#include <core/CLogger.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CIntegerTools.h>
#include <maths/common/CLinearAlgebra.h>

#include <maths/time_series/CTimeSeriesSmoother.h>

#include <test/CTestTmpDir.h>

#include <boost/test/unit_test.hpp>

#include <cmath>
#include <vector>

BOOST_AUTO_TEST_SUITE(CTimeSeriesSmootherTest)

using namespace ml;

// Test that the smoother correctly smooths discontinuities
BOOST_AUTO_TEST_CASE(testSmoothing) {
    // Setup a test function with a discontinuity
    auto discontinuousFunction = [](core_t::TTime time) {
        // Simple step function: 0 for time < 0, 1 for time >= 0
        return time < 0 ? 0.0 : 1.0;
    };

    // Create a smoother with a smoothing interval
    const core_t::TTime smoothingInterval = 10;
    maths::time_series::CTimeSeriesSmoother smoother(smoothingInterval);
    
    // Test points before, at, and after the discontinuity
    std::vector<core_t::TTime> testTimes = {-smoothingInterval, -smoothingInterval/2, 
                                           0, smoothingInterval/2, smoothingInterval};
    
    // Check that points far from the discontinuity are unaffected
    BOOST_REQUIRE_CLOSE(discontinuousFunction(-2 * smoothingInterval), 
                        smoother.smooth(discontinuousFunction, -2 * smoothingInterval), 1e-10);
    BOOST_REQUIRE_CLOSE(discontinuousFunction(2 * smoothingInterval),
                        smoother.smooth(discontinuousFunction, 2 * smoothingInterval), 1e-10);
    
    // Check points in the smoothing interval
    for (auto time : testTimes) {
        double smoothed = smoother.smooth(discontinuousFunction, time);
        LOG_DEBUG(<< "Time: " << time << ", Original: " << discontinuousFunction(time)
                 << ", Smoothed: " << smoothed);
        
        // Smoothed value should be between 0 and 1
        BOOST_REQUIRE_GE(smoothed, 0.0);
        BOOST_REQUIRE_LE(smoothed, 1.0);
        
        // For negative times, smoothed should be >= raw value (which is 0)
        // For positive times, smoothed should be <= raw value (which is 1)
        if (time < 0) {
            BOOST_REQUIRE_GE(smoothed, discontinuousFunction(time));
        } else if (time > 0) {
            BOOST_REQUIRE_LE(smoothed, discontinuousFunction(time));
        }
    }
    
    // Verify smoothing continuity - the function should change gradually
    std::vector<double> smoothedValues;
    for (int i = -smoothingInterval; i <= smoothingInterval; ++i) {
        smoothedValues.push_back(smoother.smooth(discontinuousFunction, i));
    }
    
    for (std::size_t i = 1; i < smoothedValues.size(); ++i) {
        // Check that changes between adjacent points are relatively small
        BOOST_REQUIRE_LE(std::fabs(smoothedValues[i] - smoothedValues[i-1]), 0.3);
    }
}

// Test that the smoother handles multiple discontinuities correctly
BOOST_AUTO_TEST_CASE(testMultipleDiscontinuities) {
    // Setup a test function with multiple discontinuities
    auto periodicFunction = [](core_t::TTime time) {
        // A function that's 1 for even intervals and 0 for odd intervals
        return (time / 100) % 2 == 0 ? 1.0 : 0.0;
    };
    
    // Create a smoother with a smoothing interval
    const core_t::TTime smoothingInterval = 20;
    maths::time_series::CTimeSeriesSmoother smoother(smoothingInterval);
    
    // Check smoothing around multiple discontinuities
    for (core_t::TTime time = 50; time <= 350; time += 10) {
        double smoothed = smoother.smooth(periodicFunction, time);
        LOG_DEBUG(<< "Time: " << time << ", Original: " << periodicFunction(time)
                 << ", Smoothed: " << smoothed);
        
        // Verify smoothed value is between 0 and 1
        BOOST_REQUIRE_GE(smoothed, 0.0);
        BOOST_REQUIRE_LE(smoothed, 1.0);
        
        // At discontinuities, the smoothed value should be between the before and after values
        if (std::abs(time % 100) <= smoothingInterval) {
            double before = periodicFunction(time - (time % 100) - 1);
            double after = periodicFunction(time - (time % 100) + 1);
            BOOST_REQUIRE_GE(smoothed, std::min(before, after));
            BOOST_REQUIRE_LE(smoothed, std::max(before, after));
        }
    }
    
    // Verify no smoothing occurs for points far from discontinuities
    for (core_t::TTime time = 50; time <= 350; time += 100) {
        BOOST_REQUIRE_CLOSE(periodicFunction(time), 
                          smoother.smooth(periodicFunction, time), 1e-10);
    }
}

// Test smoothing with vector values
BOOST_AUTO_TEST_CASE(testVectorSmoothing) {
    using TVector2 = maths::common::CVectorNx1<double, 2>;
    
    // Setup a vector-valued function with a discontinuity
    auto vectorFunction = [](core_t::TTime time) -> TVector2 {
        TVector2 result;
        if (time < 0) {
            result(0) = 0.0;
            result(1) = 1.0;
        } else {
            result(0) = 1.0;
            result(1) = 0.0;
        }
        return result;
    };
    
    // Create a smoother with a smoothing interval
    const core_t::TTime smoothingInterval = 10;
    maths::time_series::CTimeSeriesSmoother smoother(smoothingInterval);
    
    // Test points at the discontinuity
    TVector2 smoothedAtZero = smoother.smooth(vectorFunction, 0);
    LOG_DEBUG(<< "Smoothed at t=0: [" << smoothedAtZero(0) << ", " << smoothedAtZero(1) << "]");
    
    // Verify both components are smoothed (should be around 0.5 for both)
    BOOST_REQUIRE_GT(smoothedAtZero(0), 0.0);
    BOOST_REQUIRE_LT(smoothedAtZero(0), 1.0);
    BOOST_REQUIRE_GT(smoothedAtZero(1), 0.0);
    BOOST_REQUIRE_LT(smoothedAtZero(1), 1.0);
    
    // Values should sum close to 1
    BOOST_REQUIRE_CLOSE(smoothedAtZero(0) + smoothedAtZero(1), 1.0, 1e-10);
}

// Test setting the smoothing interval
BOOST_AUTO_TEST_CASE(testSmoothingInterval) {
    maths::time_series::CTimeSeriesSmoother smoother;
    
    // Default smoothing interval
    core_t::TTime defaultInterval = smoother.smoothingInterval();
    BOOST_REQUIRE_GT(defaultInterval, 0);
    
    // Set a custom interval
    const core_t::TTime customInterval = 42;
    smoother.smoothingInterval(customInterval);
    BOOST_REQUIRE_EQUAL(smoother.smoothingInterval(), customInterval);
    
    // Test that the new interval is being used for smoothing
    auto stepFunction = [](core_t::TTime time) {
        return time < 0 ? 0.0 : 1.0;
    };
    
    // Points just outside the smoothing interval shouldn't be affected
    BOOST_REQUIRE_CLOSE(stepFunction(-customInterval - 1), 
                      smoother.smooth(stepFunction, -customInterval - 1), 1e-10);
    BOOST_REQUIRE_CLOSE(stepFunction(customInterval + 1),
                      smoother.smooth(stepFunction, customInterval + 1), 1e-10);
    
    // Points just inside should be affected
    BOOST_REQUIRE_NE(stepFunction(-customInterval + 1), 
                    smoother.smooth(stepFunction, -customInterval + 1));
    BOOST_REQUIRE_NE(stepFunction(customInterval - 1),
                    smoother.smooth(stepFunction, customInterval - 1));
}

BOOST_AUTO_TEST_SUITE_END()
