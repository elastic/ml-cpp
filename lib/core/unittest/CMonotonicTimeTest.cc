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

// These tests sleep for a nominal one second and then make two independent
// checks on what CMonotonicTime measured over that interval:
//
//   1. A cross-domain lower bound. We assert the timer advanced by at least
//      (close to) the requested sleep duration. std::this_thread::sleep_for
//      guarantees it sleeps for *at least* the requested time, so a reading
//      well below one second means the monotonic clock is running too slowly
//      or has stalled. This is our only genuinely independent check that the
//      clock ticks at roughly real-time rate, because it compares against a
//      different clock domain (the scheduler's sleep timer) rather than
//      against another reading of the same underlying counter.
//
//   2. A scaling/consistency check. We measure the same interval with
//      std::chrono::steady_clock and assert CMonotonicTime agrees with it to
//      within a small tolerance. On every platform steady_clock and
//      CMonotonicTime ultimately derive from the same hardware counter, so
//      this does not re-check the clock's real-time fidelity (that is covered
//      by check 1); what it validates is CMonotonicTime's own unit-scaling
//      arithmetic (the mach_timebase / QueryPerformanceFrequency / timespec
//      conversions), which is the part of this class we can actually break.
//
// Crucially there is NO upper bound on the elapsed time relative to the
// nominal sleep duration. sleep_for only promises a *minimum* sleep and can
// overshoot arbitrarily when the machine is loaded or oversubscribed - on
// shared CI hosts (notably the macOS Orka VMs) the thread simply is not
// rescheduled promptly. Such overshoot is a property of the scheduler, not a
// timer defect, so any assertion of the form "diff < someConstant" is
// inherently flaky. A previous version of this test asserted diff < 1200ms and
// failed intermittently for exactly this reason (the timer correctly reported
// ~1293ms because that much wall-clock time had genuinely elapsed). Check 2
// still bounds diff from above, but only against the *actual* elapsed time
// measured over the same window, so overshoot cannot cause a failure.

namespace {
// Tolerance for how closely the monotonic timer must track the independently
// measured reference interval (check 2 above). This is deliberately small
// because we are comparing two measurements of the *same* elapsed period; it
// is sized to comfortably exceed the coarsest platform timer's granularity
// (Windows GetTickCount64 is only accurate to ~15ms, i.e. ~1.5% of a one
// second interval), so the nominal one second interval must stay large
// relative to that granularity.
const double MONOTONIC_TIMER_TOLERANCE{0.05};
}

BOOST_AUTO_TEST_CASE(testMilliseconds) {
    ml::core::CMonotonicTime monoTime;

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

    // Check 1: cross-domain lower bound (allow a 10% margin below the one
    // second sleep). Deliberately no upper bound - see the note above.
    BOOST_TEST_REQUIRE(diff > 900U);

    // Check 2: agreement with the independent reference over the same interval.
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

    // Check 1: cross-domain lower bound (allow a 10% margin below the one
    // second sleep). Deliberately no upper bound - see the note above.
    BOOST_TEST_REQUIRE(diff > 900000000U);

    // Check 2: agreement with the independent reference over the same interval.
    double allowedError{static_cast<double>(reference) * MONOTONIC_TIMER_TOLERANCE};
    BOOST_TEST_REQUIRE(std::abs(static_cast<double>(diff) -
                                static_cast<double>(reference)) < allowedError);
}

BOOST_AUTO_TEST_SUITE_END()
