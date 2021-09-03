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
#include <core/CMonotonicTime.h>

#include <core/CLogger.h>

#include <time.h>

namespace ml {
namespace core {

CMonotonicTime::CMonotonicTime()
    // Scaling factors never vary for clock_gettime()
    : m_ScalingFactor1(0), m_ScalingFactor2(0), m_ScalingFactor3(0) {
}

uint64_t CMonotonicTime::milliseconds() const {
    struct timespec ts;

    int rc(-1);

    // For milliseconds, use the coarse timers if available, as millisecond
    // granularity is good enough
#if defined(CLOCK_MONOTONIC_COARSE)
    rc = ::clock_gettime(CLOCK_MONOTONIC_COARSE, &ts);
#elif defined(CLOCK_MONOTONIC)
    rc = ::clock_gettime(CLOCK_MONOTONIC, &ts);
#elif defined(CLOCK_REALTIME_COARSE)
#warn "Monotonic clock not available - using CLOCK_REALTIME_COARSE"
    rc = ::clock_gettime(CLOCK_REALTIME_COARSE, &ts);
#else
#warn "Monotonic clock not available - using CLOCK_REALTIME"
    rc = ::clock_gettime(CLOCK_REALTIME, &ts);
#endif

    if (rc < 0) {
        LOG_ERROR(<< "Failed to get reading from hi-res clock");

        // Return a very approximate time
        return ::time(0) * 1000ULL;
    }

    uint64_t result(static_cast<uint64_t>(ts.tv_sec) * 1000ULL);
    result += static_cast<uint64_t>(ts.tv_nsec) / 1000000ULL;

    return result;
}

uint64_t CMonotonicTime::nanoseconds() const {
    struct timespec ts;

    int rc(-1);

    // Don't use the coarse timers here, as they only provide around millisecond
    // granularity
#if defined(CLOCK_MONOTONIC)
    rc = ::clock_gettime(CLOCK_MONOTONIC, &ts);
#else
#warn "Monotonic clock not available - using CLOCK_REALTIME"
    rc = ::clock_gettime(CLOCK_REALTIME, &ts);
#endif

    if (rc < 0) {
        LOG_ERROR(<< "Failed to get reading from hi-res clock");

        // Return a very approximate time
        return ::time(0) * 1000000000ULL;
    }

    uint64_t result(static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL);
    result += static_cast<uint64_t>(ts.tv_nsec);

    return result;
}
}
}
