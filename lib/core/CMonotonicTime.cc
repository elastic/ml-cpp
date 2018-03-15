/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */
#include <core/CMonotonicTime.h>

#include <core/CLogger.h>

#include <time.h>

namespace ml {
namespace core {

CMonotonicTime::CMonotonicTime(void)
    // Scaling factors never vary for clock_gettime()
    : m_ScalingFactor1(0), m_ScalingFactor2(0), m_ScalingFactor3(0) {
}

uint64_t CMonotonicTime::milliseconds(void) const {
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
        LOG_ERROR("Failed to get reading from hi-res clock");

        // Return a very approximate time
        return ::time(0) * 1000ULL;
    }

    uint64_t result(static_cast<uint64_t>(ts.tv_sec) * 1000ULL);
    result += static_cast<uint64_t>(ts.tv_nsec) / 1000000ULL;

    return result;
}

uint64_t CMonotonicTime::nanoseconds(void) const {
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
        LOG_ERROR("Failed to get reading from hi-res clock");

        // Return a very approximate time
        return ::time(0) * 1000000000ULL;
    }

    uint64_t result(static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL);
    result += static_cast<uint64_t>(ts.tv_nsec);

    return result;
}
}
}
