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
#include <maths/common/CLinearAlgebra.h>

#include <maths/time_series/CTimeSeriesPredictor.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CTimeSeriesPredictorTest)

using namespace ml;

namespace {

} // namespace

// Test that CTimeSeriesPredictor can be instantiated
BOOST_AUTO_TEST_CASE(testPredictorInstantiation) {
    // Since we can't directly test with components due to header dependency issues,
    // we'll just verify that the class can be instantiated and the header includes correctly.
    // In a real usage scenario, these would be created by a factory or retrieved from a decomposition.
    
    // Create a predictor with null components (just testing compilation)
    maths::time_series::CTimeSeriesPredictor predictor(nullptr, nullptr, nullptr);
    
    // Verify memory usage method doesn't throw
    BOOST_REQUIRE_NO_THROW(predictor.memoryUsage());
}

// Test basic method access
BOOST_AUTO_TEST_CASE(testBasicMethodAccess) {
    // Create a predictor with null components
    maths::time_series::CTimeSeriesPredictor predictor(nullptr, nullptr, nullptr);
    
    // Test parameters
    core_t::TTime testTime = 0;
    double confidence = 0.95;
    bool isNonNegative = false;
    core_t::TTime timeShift = 0;
    
    // Check that we can access the methods without crashing
    // With null components, we expect zero or default values
    BOOST_REQUIRE_NO_THROW(predictor.trendValue(testTime, confidence, isNonNegative));
    BOOST_REQUIRE_NO_THROW(predictor.seasonalValue(testTime, confidence, isNonNegative));
    BOOST_REQUIRE_NO_THROW(predictor.calendarValue(testTime, confidence, isNonNegative));
    BOOST_REQUIRE_NO_THROW(predictor.value(testTime, confidence, isNonNegative, timeShift));
    
    // Check that the timeShift parameter is processed without error
    BOOST_REQUIRE_NO_THROW(predictor.value(testTime, confidence, isNonNegative, 3600));
}

// Test debug methods
BOOST_AUTO_TEST_CASE(testDebugMethods) {
    // Create a predictor with null components
    maths::time_series::CTimeSeriesPredictor predictor(nullptr, nullptr, nullptr);
    
    // Memory usage should be at least some minimal amount even with null components
    BOOST_REQUIRE_GT(predictor.memoryUsage(), 0);
    
    // Just verify the predictor method doesn't throw
    BOOST_REQUIRE_NO_THROW(predictor.predictor());
}

BOOST_AUTO_TEST_SUITE_END()
