/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_core_CLoggerThrottler_h
#define INCLUDED_ml_core_CLoggerThrottler_h

#include <core/CoreTypes.h>
#include <core/ImportExport.h>

#include <boost/unordered_map.hpp>

#include <atomic>
#include <cstdint>
#include <mutex>

namespace ml {
namespace core {

//! \brief Implements per log line throttling.
//!
//! DESCRIPTION:\n
//! This implements global throttling for a program per line log line. By default
//! the same log line will only be emitted once per hour.
//!
//! IMPLEMENTATION:\n
//! This is thread safe but uses a very simple strategy: all accesses to a single
//! hash map are sychronised. We assume that log throttling is only applied to
//! messages which normally occur infrequently; for example, this is only currently
//! applied to WARN and ERROR level logging (see LogMacros.h). So there will be
//! little contention. Furthermore, the overhead of locking and unlocking the mutex
//! should be neglible compared to the work done if the log line were actually
//! emitted. So this should actually give a significant performance improvement
//! if a log line is spamming.
class CORE_EXPORT CLoggerThrottler {
public:
    CLoggerThrottler(const CLoggerThrottler&) = delete;
    CLoggerThrottler& operator=(const CLoggerThrottler&) = delete;

    //! Get the unique log throttler.
    static CLoggerThrottler& instance();

    //! Set the minimum interval between repeated log messages.
    void minimumLogInterval(std::int64_t minimumLogInterval);

    //! Should we skip logging of \p line in \p file?
    //!
    //! \param[in] file The file containing the log line.
    //! \param[in] line The log line.
    //! \return A pair comprising the count of log lines since the log line was
    //! last output (skip returned false) and whether to output the line or not.
    //! \note This is expected to be used in conjuction with __FILE__ and __LINE__
    //! macros although any unique identifiers which are a string literal and an
    //! integer are permitted.
    std::pair<std::size_t, bool> skip(const char* file, int line);

private:
    using TConstCharPtrIntPr = std::pair<const char*, int>;
    using TInt64SizePr = std::pair<std::int64_t, std::size_t>;
    using TConstCharPtrIntPrInt64SizePrUMap =
        boost::unordered_map<TConstCharPtrIntPr, TInt64SizePr>;

private:
    CLoggerThrottler();
    TInt64SizePr& lookup(const TConstCharPtrIntPr& key);

private:
    static CLoggerThrottler ms_Instance;
    std::int64_t m_MinimumLogInterval;
    std::mutex m_Mutex;
    TConstCharPtrIntPrInt64SizePrUMap m_LastLogTimesAndCounts;
};
}
}

#endif // INCLUDED_ml_core_CLoggerThrottler_h
