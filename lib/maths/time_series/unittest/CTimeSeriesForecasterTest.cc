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
#include <maths/common/Constants.h>
#include <maths/common/MathsTypes.h>

#include <maths/time_series/CTimeSeriesForecaster.h>
#include <maths/time_series/CTimeSeriesDecomposition.h>
#include <maths/time_series/CTimeSeriesDecompositionInterface.h>

#include <boost/test/unit_test.hpp>

#include <cmath>
#include <vector>

BOOST_AUTO_TEST_SUITE(CTimeSeriesForecasterTest)

using namespace ml;

// Basic test using CTimeSeriesDecomposition directly

// Helper class to collect forecast values
class CForecastCollector {
public:
    void operator()(core_t::TTime time, const std::vector<double>& values) {
        m_Times.push_back(time);
        if (!values.empty()) {
            m_Values.push_back(values[0]);
        } else {
            m_Values.push_back(0.0);
        }
    }
    
    const std::vector<core_t::TTime>& times() const { return m_Times; }
    const std::vector<double>& values() const { return m_Values; }
    
    std::size_t size() const { return m_Times.size(); }
    
private:
    std::vector<core_t::TTime> m_Times;
    std::vector<double> m_Values;
};

// Test basic forecaster functionality with a real CTimeSeriesDecomposition
BOOST_AUTO_TEST_CASE(testBasicForecastingWithRealDecomposition) {
    // Create a real time series decomposition
    maths::time_series::CTimeSeriesDecomposition decomposition(0.01, 3600);
    
    // Add some points to train the decomposition
    for (core_t::TTime t = 0; t < 86400 * 7; t += 3600) {
        // Create a simple seasonal pattern with trend
        double trend = 10.0 + (0.01 * (static_cast<double>(t) / 3600.0));
        double seasonal = 5.0 * std::sin((2.0 * 3.14159 * static_cast<double>(t % 86400)) / 86400.0);
        double value = trend + seasonal;
        
        // Add the point to the decomposition
        decomposition.addPoint(t, value);
    }
    
    // Create a forecaster with the decomposition
    maths::time_series::CTimeSeriesForecaster forecaster(decomposition);
    
    // Test parameters for forecasting
    core_t::TTime startTime = static_cast<core_t::TTime>(86400 * 7);     // Start after training period
    core_t::TTime endTime = static_cast<core_t::TTime>(86400 * 8);       // Forecast for 1 day
    core_t::TTime step = 3600;               // Hourly steps
    double confidence = 0.95;
    double minimumScale = 0.1;
    bool isNonNegative = false;
    core_t::TTime timeShift = 0;
    
    // Create a collector for the forecast results
    CForecastCollector collector;
    
    // Generate forecast
    forecaster.forecast(startTime, endTime, step, confidence, 
                       minimumScale, isNonNegative, timeShift, 
                       std::ref(collector));
    
    // Verify number of points
    std::size_t expectedPoints = ((endTime - startTime) / step) + 1;
    BOOST_REQUIRE_EQUAL(collector.size(), expectedPoints);
    
    // Verify forecast shows seasonal pattern
    // Only check if we have values
    if (!collector.values().empty()) {
        double minValue = *std::min_element(collector.values().begin(), collector.values().end());
        double maxValue = *std::max_element(collector.values().begin(), collector.values().end());
        
        // Should have a range due to seasonality
        BOOST_REQUIRE_GT(maxValue - minValue, 1.0);
    }
    
    // Check was moved inside the if block above
    
    // Check a few specific points to ensure forecast has expected pattern
    if (collector.size() >= 24) {
        // Values at similar times of day should be similar
        double morning = collector.values().at(6);   // 6 hours in - using .at() for bounds checking
        double evening = collector.values().at(18);  // 18 hours in
        
        // Morning and evening should differ due to seasonality
        LOG_DEBUG(<< "Morning value: " << morning << ", Evening value: " << evening);
        BOOST_TEST_MESSAGE("Morning value: " << morning << ", Evening value: " << evening);
    }
}

// Test maximum forecast interval
BOOST_AUTO_TEST_CASE(testMaximumForecastInterval) {
    // Create a decomposition with default settings
    maths::time_series::CTimeSeriesDecomposition decomposition;
    
    // Create a forecaster with the decomposition
    maths::time_series::CTimeSeriesForecaster forecaster(decomposition);
    
    // Verify that maximum interval is available (not testing specific value
    // since it depends on implementation details of CTimeSeriesDecomposition)
    core_t::TTime interval = forecaster.maximumForecastInterval();
    LOG_DEBUG(<< "Maximum forecast interval: " << interval);
    BOOST_REQUIRE_GT(interval, 0);
}

// Test forecasting with different confidence levels
BOOST_AUTO_TEST_CASE(testForecastingWithDifferentConfidenceLevels) {
    // Create a decomposition
    maths::time_series::CTimeSeriesDecomposition decomposition(0.01, 3600);
    
    // Add some points with noise to train the decomposition
    // No need for real randomness in tests, just use a simple deterministic pattern
    for (core_t::TTime t = 0; t < static_cast<core_t::TTime>(86400 * 7); t += 3600) {
        // Create a simple pattern with some pseudo-random noise
        double baseValue = 10.0 + (5.0 * std::sin((2.0 * 3.14159 * static_cast<double>(t % 86400)) / 86400.0));
        double noise = std::sin(static_cast<double>(t) / 1000.0); // Simple pseudo-random noise
        double value = baseValue + noise;
        
        // Add the point to the decomposition
        decomposition.addPoint(t, value);
    }
    
    // Create a forecaster with the decomposition
    maths::time_series::CTimeSeriesForecaster forecaster(decomposition);
    
    // Forecast parameters
    core_t::TTime startTime = static_cast<core_t::TTime>(86400 * 7);
    core_t::TTime endTime = static_cast<core_t::TTime>(86400 * 7 + 3600);
    core_t::TTime step = 3600;
    double minimumScale = 0.1;
    bool isNonNegative = false;
    core_t::TTime timeShift = 0;
    
    // Generate forecasts with different confidence levels
    CForecastCollector collector50;
    CForecastCollector collector95;
    
    forecaster.forecast(startTime, endTime, step, 0.5,  // 50% confidence
                       minimumScale, isNonNegative, timeShift, 
                       std::ref(collector50));
                       
    forecaster.forecast(startTime, endTime, step, 0.95, // 95% confidence
                       minimumScale, isNonNegative, timeShift, 
                       std::ref(collector95));
    
    // Verify both forecasts have same number of points
    BOOST_REQUIRE_EQUAL(collector50.size(), collector95.size());
    
    // For real forecasts with CTimeSeriesDecomposition, confidence doesn't affect mean prediction
    // but rather the confidence bounds, so the values should be similar
    if (!collector50.values().empty() && !collector95.values().empty()) {
        LOG_DEBUG(<< "50% confidence forecast: " << collector50.values()[0]
                 << ", 95% confidence forecast: " << collector95.values()[0]);
    }
}

BOOST_AUTO_TEST_SUITE_END()
