/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLoggerThrottler.h>

#include <core/CTimeUtils.h>

#include <limits>
#include <iostream>

namespace ml {
namespace core {

CLoggerThrottler& CLoggerThrottler::instance() {
    return ms_Instance;
}

void CLoggerThrottler::minimumLogInterval(std::int64_t minimumLogInterval) {
    m_MinimumLogInterval = minimumLogInterval;
}

bool CLoggerThrottler::skip(const char* file, int line) {

    auto key = std::make_pair(file, line);
    auto now = CTimeUtils::nowMs();

    std::unique_lock<std::mutex> lock{m_Mutex};
    auto lastTime = m_LastLogTimes.emplace(key, std::numeric_limits<std::int64_t>::min()).first;

    if (now > lastTime->second + m_MinimumLogInterval) {
        lastTime->second = now;
        return false;
    }
    return true;
}

CLoggerThrottler::CLoggerThrottler() : m_MinimumLogInterval{3600 * 1000} {}

CLoggerThrottler CLoggerThrottler::ms_Instance;
}
}
