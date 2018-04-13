/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CMonotonicTime.h>

#include <core/CLogger.h>
#include <core/WindowsSafe.h>

namespace ml {
namespace core {

CMonotonicTime::CMonotonicTime()
    : m_ScalingFactor1(0),
      // Only one variable scaling factor is needed on Windows
      m_ScalingFactor2(0), m_ScalingFactor3(0) {
    LARGE_INTEGER largeInt;
    if (QueryPerformanceFrequency(&largeInt) == FALSE) {
        LOG_WARN(<< "High frequency performance counters not available");
    } else {
        // The high frequency counter ticks this many times per second
        m_ScalingFactor1 = static_cast<uint64_t>(largeInt.QuadPart);
    }
}

uint64_t CMonotonicTime::milliseconds() const {
    // This is only accurate to about 15 milliseconds
    return GetTickCount64();
}

uint64_t CMonotonicTime::nanoseconds() const {
    if (m_ScalingFactor1 == 0) {
        // High frequency performance counters are not available, so return an
        // approximation
        return GetTickCount64() * 1000000ULL;
    }

    LARGE_INTEGER largeInt;

    // This function call is slow
    if (QueryPerformanceCounter(&largeInt) == FALSE) {
        // Failed to obtain high frequency performance counter reading, so
        // return an approximation
        return GetTickCount64() * 1000000ULL;
    }

    // To get nanoseconds, we need to divide by the counter frequency and then
    // multiply by 1 billion, but literally doing it this way will lose accuracy
    // due to integer division.  To avoid losing accuracy we need to multiply
    // first and then divide, but that may overflow.  To avoid both problems
    // we divide then multiply for the bulk of the result, but also add on an
    // adjustment for the remainder of the initial division.

    // Doing the division first here truncates the result to the number of
    // nanoseconds in a number of full seconds
    uint64_t fullSecondNanoseconds(
        (static_cast<uint64_t>(largeInt.QuadPart) / m_ScalingFactor1) * 1000000000ULL);

    // This is the number of ticks over and above the last full second
    uint64_t remainder(static_cast<uint64_t>(largeInt.QuadPart) % m_ScalingFactor1);

    // Assuming the counter ticks less than 18.4 billion times per second, this
    // won't overflow when we do the multiplication first (and on Windows 2008
    // it ticks about 3.75 million times per second, so there's a fair amount
    // of leeway here)
    uint64_t extraNanoseconds((remainder * 1000000000ULL) / m_ScalingFactor1);

    return fullSecondNanoseconds + extraNanoseconds;
}
}
}
