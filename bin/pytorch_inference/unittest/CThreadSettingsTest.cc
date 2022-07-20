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

#include "../CThreadSettings.h"

#include <boost/test/unit_test.hpp>
#include <cstdint>

BOOST_AUTO_TEST_SUITE(CThreadSettingsTest)

BOOST_AUTO_TEST_CASE(testValidationNoChanges) {
    std::int32_t maxThreads{16};
    std::int32_t numAllocations{4};
    std::int32_t numThreadsPerAllocation{4};
    ml::torch::CThreadSettings::validateThreadingParameters(
        maxThreads, numThreadsPerAllocation, numAllocations);
    BOOST_REQUIRE_EQUAL(4, numAllocations);
    BOOST_REQUIRE_EQUAL(4, numThreadsPerAllocation);
}

BOOST_AUTO_TEST_CASE(testValidationValuesAreCapped) {
    std::int32_t maxThreads{16};
    std::int32_t numAllocations{1};
    std::int32_t numThreadsPerAllocation{32};
    ml::torch::CThreadSettings::validateThreadingParameters(
        maxThreads, numThreadsPerAllocation, numAllocations);
    BOOST_REQUIRE_EQUAL(1, numAllocations);
    BOOST_REQUIRE_EQUAL(16, numThreadsPerAllocation);

    numAllocations = 32;
    numThreadsPerAllocation = 1;
    ml::torch::CThreadSettings::validateThreadingParameters(
        maxThreads, numThreadsPerAllocation, numAllocations);
    BOOST_REQUIRE_EQUAL(16, numAllocations);
    BOOST_REQUIRE_EQUAL(1, numThreadsPerAllocation);
}

BOOST_AUTO_TEST_CASE(testValidationNegativeValues) {
    std::int32_t maxThreads{16};
    std::int32_t numAllocations{-1};
    std::int32_t numThreadsPerAllocation{-2};
    ml::torch::CThreadSettings::validateThreadingParameters(
        maxThreads, numThreadsPerAllocation, numAllocations);
    BOOST_REQUIRE_EQUAL(1, numAllocations);
    BOOST_REQUIRE_EQUAL(1, numThreadsPerAllocation);
}

BOOST_AUTO_TEST_CASE(testValidationMaxThreadsUnknown) {
    // 0 == maxThreads is not known
    std::int32_t maxThreads{0};
    std::int32_t numAllocations{4};
    std::int32_t numThreadsPerAllocation{4};
    ml::torch::CThreadSettings::validateThreadingParameters(
        maxThreads, numThreadsPerAllocation, numAllocations);
    BOOST_REQUIRE_EQUAL(1, numAllocations);
    BOOST_REQUIRE_EQUAL(1, numThreadsPerAllocation);
}

BOOST_AUTO_TEST_CASE(testValidationTotalGreaterThanMaxThreads) {
    std::int32_t maxThreads{16};
    {
        std::int32_t numAllocations{10};
        std::int32_t numThreadsPerAllocation{10};
        ml::torch::CThreadSettings::validateThreadingParameters(
            maxThreads, numThreadsPerAllocation, numAllocations);
        BOOST_REQUIRE_EQUAL(10, numAllocations);
        BOOST_REQUIRE_EQUAL(2, numThreadsPerAllocation);
    }
    {
        std::int32_t numAllocations{1};
        std::int32_t numThreadsPerAllocation{32};
        ml::torch::CThreadSettings::validateThreadingParameters(
            maxThreads, numThreadsPerAllocation, numAllocations);
        BOOST_REQUIRE_EQUAL(1, numAllocations);
        BOOST_REQUIRE_EQUAL(16, numThreadsPerAllocation);
    }
    maxThreads = 4;
    {
        std::int32_t numAllocations{4};
        std::int32_t numThreadsPerAllocation{1};
        ml::torch::CThreadSettings::validateThreadingParameters(
            maxThreads, numThreadsPerAllocation, numAllocations);
        BOOST_REQUIRE_EQUAL(4, numAllocations);
        BOOST_REQUIRE_EQUAL(1, numThreadsPerAllocation);
    }
    {
        std::int32_t numAllocations{1};
        std::int32_t numThreadsPerAllocation{4};
        ml::torch::CThreadSettings::validateThreadingParameters(
            maxThreads, numThreadsPerAllocation, numAllocations);
        BOOST_REQUIRE_EQUAL(1, numAllocations);
        BOOST_REQUIRE_EQUAL(4, numThreadsPerAllocation);
    }
    {
        std::int32_t numAllocations{2};
        std::int32_t numThreadsPerAllocation{4};
        ml::torch::CThreadSettings::validateThreadingParameters(
            maxThreads, numThreadsPerAllocation, numAllocations);
        BOOST_REQUIRE_EQUAL(2, numAllocations);
        BOOST_REQUIRE_EQUAL(2, numThreadsPerAllocation);
    }
}

BOOST_AUTO_TEST_SUITE_END()
