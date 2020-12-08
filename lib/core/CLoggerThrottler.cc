/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLoggerThrottler.h>

#include <core/CTimeUtils.h>

#include <iostream>
#include <limits>

namespace ml {
namespace core {

CLoggerThrottler& CLoggerThrottler::instance() {
    return ms_Instance;
}

void CLoggerThrottler::minimumLogInterval(std::int64_t minimumLogInterval) {
    m_MinimumLogInterval = minimumLogInterval;
}

std::pair<std::size_t, bool> CLoggerThrottler::skip(const char* file, int line) {

    auto key = std::make_pair(file, line);
    auto value = std::make_pair(std::numeric_limits<std::int64_t>::min(), 0);
    auto now = CTimeUtils::nowMs();

    std::unique_lock<std::mutex> lock{m_Mutex};
    auto lastTime = m_LastLogTimesAndCounts.emplace(key, value).first;
    auto count = lastTime->second.second + 1;

    if (now > lastTime->second.first + m_MinimumLogInterval) {
        lastTime->second = std::make_pair(now, 0);
        return std::make_pair(count, false);
    }
    ++lastTime->second.second;
    return std::make_pair(count, true);
}

CLoggerThrottler::CLoggerThrottler() : m_MinimumLogInterval{3600 * 1000} {
}

CLoggerThrottler CLoggerThrottler::ms_Instance;
}
}
