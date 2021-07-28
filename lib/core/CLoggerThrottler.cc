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

#include <core/CLoggerThrottler.h>

#include <core/CTimeUtils.h>

#include <limits>

namespace ml {
namespace core {

CLoggerThrottler::CLoggerThrottler() : m_MinimumLogIntervalMs{3600 * 1000} {
}

void CLoggerThrottler::minimumLogIntervalMs(std::int64_t minimumLogIntervalMs) {
    m_MinimumLogIntervalMs = minimumLogIntervalMs;
}

std::pair<std::size_t, bool> CLoggerThrottler::skip(const char* file, int line) {

    auto key = std::make_pair(file, line);
    auto now = CTimeUtils::nowMs();
    auto value = std::make_pair(std::numeric_limits<std::int64_t>::min(), 0);

    // We have to hold the lock while updating the map entry because the same log
    // line may be triggered from different threads and we don't try and update
    // the last time and count atomically.

    std::unique_lock<std::mutex> lock{m_Mutex};
    auto& valueRef = m_LastLogTimesAndCounts.emplace(key, value).first->second;
    std::size_t count{valueRef.second + 1};

    if (now >= valueRef.first + m_MinimumLogIntervalMs) {
        valueRef = std::make_pair(now, 0);
        return std::make_pair(count, false);
    }
    ++valueRef.second;
    return std::make_pair(count, true);
}
}
}
