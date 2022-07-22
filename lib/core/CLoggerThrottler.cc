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
#include <core/Constants.h>

#include <boost/unordered_map.hpp>

#include <atomic>
#include <cstdint>
#include <limits>
#include <mutex>

namespace ml {
namespace core {

class CLoggerThrottler::CImpl {
public:
    using TConstCharPtrIntPr = std::pair<const char*, int>;
    using TInt64SizePr = std::pair<std::int64_t, std::size_t>;

public:
    void minimumLogIntervalMs(std::int64_t minimumLogIntervalMs) {
        m_MinimumLogIntervalMs = minimumLogIntervalMs;
    }

    std::pair<std::size_t, bool> skip(const char* file, int line) {

        auto key = TConstCharPtrIntPr{file, line};
        auto now = CTimeUtils::nowMs();
        auto value = TInt64SizePr{std::numeric_limits<std::int64_t>::min(), 0};

        // We have to hold the lock while updating the map entry because the same log
        // line may be triggered from different threads and we don't try and update
        // the last time and count atomically.

        std::unique_lock<std::mutex> lock{m_Mutex};
        auto& valueRef = m_LastLogTimesAndCounts.emplace(key, value).first->second;
        std::size_t count{valueRef.second + 1};

        if (now >= valueRef.first + m_MinimumLogIntervalMs) {
            valueRef = std::make_pair(now, 0);
            return {count, false};
        }
        ++valueRef.second;
        return {count, true};
    }

private:
    using TConstCharPtrIntPrInt64SizePrUMap =
        boost::unordered_map<TConstCharPtrIntPr, TInt64SizePr>;

private:
    std::int64_t m_MinimumLogIntervalMs{core::constants::HOUR * 1000};
    std::mutex m_Mutex;
    TConstCharPtrIntPrInt64SizePrUMap m_LastLogTimesAndCounts;
};

CLoggerThrottler::CLoggerThrottler() : m_Impl{new CImpl} {
}

CLoggerThrottler::~CLoggerThrottler() {
    delete m_Impl;
}

void CLoggerThrottler::minimumLogIntervalMs(std::int64_t minimumLogIntervalMs) {
    m_Impl->minimumLogIntervalMs(minimumLogIntervalMs);
}

std::pair<std::size_t, bool> CLoggerThrottler::skip(const char* file, int line) {
    return m_Impl->skip(file, line);
}
}
}
