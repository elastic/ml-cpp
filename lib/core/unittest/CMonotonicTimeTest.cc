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
#include <core/CMonotonicTime.h>

#include <boost/test/unit_test.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <thread>

BOOST_AUTO_TEST_SUITE(CMonotonicTimeTest)

namespace {
// Tolerance for how closely the monotonic timer must track an independently
// measured reference interval. This is deliberately small because we are
// comparing two measurements of the *same* elapsed period rather than relying
// on the accuracy of sleep_for - the two clocks should agree regardless of how
// long the (possibly oversubscribed CI) machine actually slept for.
const double MONOTONIC_TIMER_TOLERANCE{0.05};
}

BOOST_AUTO_TEST_CASE(testMilliseconds) {
    ml::core::CMonotonicTime monoTime;

    // Measure the elapsed interval with both the class under test and an
    // independent reference clock. sleep_for is unreliable on loaded CI
    // machines (it can massively overshoot), so we do not assert against its
    // nominal duration - instead we assert that the monotonic timer agrees
    // with however much real time actually elapsed.
    auto referenceStart = std::chrono::steady_clock::now();
    std::uint64_t start(monoTime.milliseconds());

    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::uint64_t end(monoTime.milliseconds());
    auto referenceEnd = std::chrono::steady_clock::now();

    std::uint64_t diff(end - start);
    std::uint64_t reference(static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(referenceEnd - referenceStart)
            .count()));
    LOG_DEBUG(<< "The monotonic millisecond timer advanced by " << diff << " milliseconds; reference clock advanced by "
              << reference << " milliseconds");

    // The monotonic timer must never run backwards or stand still over a real
    // elapsed interval.
    BOOST_TEST_REQUIRE(diff > 0U);
    // Both clocks measured the same real interval, so they must agree closely.
    double allowedError{static_cast<double>(reference) * MONOTONIC_TIMER_TOLERANCE};
    BOOST_TEST_REQUIRE(std::abs(static_cast<double>(diff) -
                                static_cast<double>(reference)) < allowedError);
}

BOOST_AUTO_TEST_CASE(testNanoseconds) {
    ml::core::CMonotonicTime monoTime;

    auto referenceStart = std::chrono::steady_clock::now();
    std::uint64_t start(monoTime.nanoseconds());

    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::uint64_t end(monoTime.nanoseconds());
    auto referenceEnd = std::chrono::steady_clock::now();

    std::uint64_t diff(end - start);
    std::uint64_t reference(static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(referenceEnd - referenceStart)
            .count()));
    LOG_DEBUG(<< "The monotonic nanosecond timer advanced by " << diff << " nanoseconds; reference clock advanced by "
              << reference << " nanoseconds");

    // The monotonic timer must never run backwards or stand still over a real
    // elapsed interval.
    BOOST_TEST_REQUIRE(diff > 0U);
    // Both clocks measured the same real interval, so they must agree closely.
    double allowedError{static_cast<double>(reference) * MONOTONIC_TIMER_TOLERANCE};
    BOOST_TEST_REQUIRE(std::abs(static_cast<double>(diff) -
                                static_cast<double>(reference)) < allowedError);
}

BOOST_AUTO_TEST_SUITE_END()
