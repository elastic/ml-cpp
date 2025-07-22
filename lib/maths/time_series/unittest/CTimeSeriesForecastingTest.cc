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

#include <maths/time_series/CTimeSeriesDecomposition.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CTimeSeriesForecastingTest)

// Simple test case to verify integration with CTimeSeriesDecomposition
BOOST_AUTO_TEST_CASE(testBasicForecastDelegation) {
    // Create a decomposition object which will internally use CTimeSeriesForecasting
    ml::maths::time_series::CTimeSeriesDecomposition decomp(0.01, 24 * 3600);
    
    // Verify that the object is properly initialized
    BOOST_REQUIRE_EQUAL(decomp.initialized(), false);
    
    // Check that maximumForecastInterval returns a valid value
    // This calls through to CTimeSeriesForecasting
    ml::core_t::TTime interval = decomp.maximumForecastInterval();
    
    // Simple verification that the interval is reasonable
    BOOST_REQUIRE_GT(interval, 0);
    
    LOG_DEBUG(<< "Maximum forecast interval: " << interval);
}

BOOST_AUTO_TEST_CASE(testForecastMethod) {
    // This test verifies that the forecast method can be called without errors
    ml::maths::time_series::CTimeSeriesDecomposition decomp(0.01, 24 * 3600);
    
    // Add a simple data point
    decomp.addPoint(0, 10.0);
    
    // Define simple forecast parameters
    ml::core_t::TTime startTime = 3600;
    ml::core_t::TTime endTime = 7200;
    ml::core_t::TTime step = 3600;
    
    // Create a simple writer that just counts points
    int forecastPointCount = 0;
    auto writer = [&forecastPointCount](ml::core_t::TTime, 
                                      const ml::maths::time_series::CTimeSeriesDecomposition::TDouble3Vec&) {
        ++forecastPointCount;
    };
    
    // Call the forecast method
    decomp.forecast(startTime, endTime, step, 95.0, 0.1, false, writer);
    
    // We should get at least one forecast point
    BOOST_REQUIRE_GT(forecastPointCount, 0);
    
    LOG_DEBUG(<< "Generated " << forecastPointCount << " forecast points");
}

BOOST_AUTO_TEST_SUITE_END()
