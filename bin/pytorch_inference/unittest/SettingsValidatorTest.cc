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

#include "../SettingsValidator.h"

#include <boost/test/unit_test.hpp>
#include <cstdint>

BOOST_AUTO_TEST_SUITE(SettingsValidatorTest)

BOOST_AUTO_TEST_CASE(testValidationNoChanges) {
    std::int32_t modelThreads{4};
    std::int32_t inferenceThreads{4};
    ml::torch::validateThreadingParameters(16, inferenceThreads, modelThreads);
    BOOST_REQUIRE_EQUAL(4, modelThreads);
    BOOST_REQUIRE_EQUAL(4, inferenceThreads);
}

BOOST_AUTO_TEST_CASE(testValidationValuesAreCapped) {
    std::int32_t modelThreads{1};
    std::int32_t inferenceThreads{32};
    ml::torch::validateThreadingParameters(16, inferenceThreads, modelThreads);
    BOOST_REQUIRE_EQUAL(1, modelThreads);
    BOOST_REQUIRE_EQUAL(16, inferenceThreads);

    modelThreads = 32;
    inferenceThreads = 1;
    ml::torch::validateThreadingParameters(16, inferenceThreads, modelThreads);
    BOOST_REQUIRE_EQUAL(15, modelThreads);
    BOOST_REQUIRE_EQUAL(1, inferenceThreads);
}

BOOST_AUTO_TEST_CASE(testValidationNegativeValues) {
    std::int32_t modelThreads{-1};
    std::int32_t inferenceThreads{-2};
    ml::torch::validateThreadingParameters(16, inferenceThreads, modelThreads);
    BOOST_REQUIRE_EQUAL(1, modelThreads);
    BOOST_REQUIRE_EQUAL(1, inferenceThreads);
}

BOOST_AUTO_TEST_CASE(testValidationMaxThreadsUnknown) {
    std::int32_t modelThreads{4};
    std::int32_t inferenceThreads{4};
    // 0 == maxThreads is not known
    ml::torch::validateThreadingParameters(0, inferenceThreads, modelThreads);
    BOOST_REQUIRE_EQUAL(1, modelThreads);
    BOOST_REQUIRE_EQUAL(1, inferenceThreads);
}

BOOST_AUTO_TEST_CASE(testValidationTotalGreaterThanMaxThreads) {
    {
        std::int32_t modelThreads{10};
        std::int32_t inferenceThreads{10};
        ml::torch::validateThreadingParameters(16, inferenceThreads, modelThreads);
        BOOST_REQUIRE_EQUAL(10, modelThreads);
        BOOST_REQUIRE_EQUAL(6, inferenceThreads);
    }
    {
        std::int32_t modelThreads{1};
        std::int32_t inferenceThreads{32};
        ml::torch::validateThreadingParameters(16, inferenceThreads, modelThreads);
        BOOST_REQUIRE_EQUAL(1, modelThreads);
        BOOST_REQUIRE_EQUAL(16, inferenceThreads);
    }
    {
        std::int32_t modelThreads{4};
        std::int32_t inferenceThreads{1};
        ml::torch::validateThreadingParameters(4, inferenceThreads, modelThreads);
        BOOST_REQUIRE_EQUAL(3, modelThreads);
        BOOST_REQUIRE_EQUAL(1, inferenceThreads);
    }
    {
        std::int32_t modelThreads{1};
        std::int32_t inferenceThreads{4};
        ml::torch::validateThreadingParameters(4, inferenceThreads, modelThreads);
        BOOST_REQUIRE_EQUAL(1, modelThreads);
        BOOST_REQUIRE_EQUAL(4, inferenceThreads);
    }
    {
        std::int32_t modelThreads{2};
        std::int32_t inferenceThreads{4};
        ml::torch::validateThreadingParameters(4, inferenceThreads, modelThreads);
        BOOST_REQUIRE_EQUAL(2, modelThreads);
        BOOST_REQUIRE_EQUAL(2, inferenceThreads);
    }
}

BOOST_AUTO_TEST_SUITE_END()
