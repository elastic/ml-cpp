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
    auto now = CTimeUtils::nowMs();

    // The following makes use of the fact that unordered_map insertions (even
    // those triggering a rehash) do not invalidate references.

    auto& value = this->lookup(key);
    std::size_t count{value.second + 1};

    if (now > value.first + m_MinimumLogInterval) {
        value = std::make_pair(now, 0);
        return std::make_pair(count, false);
    }
    ++value.second;
    return std::make_pair(count, true);
}

CLoggerThrottler::CLoggerThrottler() : m_MinimumLogInterval{3600 * 1000} {
}

CLoggerThrottler::TInt64SizePr& CLoggerThrottler::lookup(const TConstCharPtrIntPr& key) {
    auto value = std::make_pair(std::numeric_limits<std::int64_t>::min(), 0);
    std::unique_lock<std::mutex> lock{m_Mutex};
    return m_LastLogTimesAndCounts.emplace(key, value).first->second;
}

CLoggerThrottler CLoggerThrottler::ms_Instance;
}
}
