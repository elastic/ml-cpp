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
#include <core/CTimeUtils.h>

#include <core/CoreTypes.h>
#include <core/CLogger.h>
#include <core/CScopedFastLock.h>
#include <core/CStrFTime.h>
#include <core/CStrPTime.h>
#include <core/CTimezone.h>

#include <errno.h>
#include <string.h>


namespace ml {
namespace core {


// Initialise class static data
const core_t::TTime CTimeUtils::MAX_CLOCK_DISCREPANCY(300);


core_t::TTime CTimeUtils::now(void) {
    return ::time(0);
}

std::string CTimeUtils::toIso8601(core_t::TTime t) {
    std::string result;
    CTimeUtils::toStringCommon(t, "%Y-%m-%dT%H:%M:%S%z", result);
    return result;
}

std::string CTimeUtils::toLocalString(core_t::TTime t) {
    std::string result;
    CTimeUtils::toStringCommon(t, "%c", result);
    return result;
}

std::string CTimeUtils::toTimeString(core_t::TTime t) {
    std::string result;
    CTimeUtils::toStringCommon(t, "%H:%M:%S", result);
    return result;
}

bool CTimeUtils::strptime(const std::string &format,
                          const std::string &dateTime,
                          core_t::TTime &preTime) {
    if (CTimeUtils::strptimeSilent(format, dateTime, preTime) == false) {
        LOG_ERROR("Unable to convert " << dateTime << " to " << format);
        return false;
    }

    return true;
}

bool CTimeUtils::strptimeSilent(const std::string &format,
                                const std::string &dateTime,
                                core_t::TTime &preTime) {
    struct tm t;
    ::memset(&t, 0, sizeof(struct tm));

    const char *ret(CStrPTime::strPTime(dateTime.c_str(), format.c_str(), &t));
    if (ret == 0) {
        return false;
    }

    t.tm_isdst = -1;

    CTimezone &tz = CTimezone::instance();

    // Some date formats don't give the year, so we might need to guess it
    // We'll assume that the year is the current year, unless that results
    // in a time in the future, in which case we'll guess the date refers to
    // last year
    struct tm copy;
    ::memset(&copy, 0, sizeof(struct tm));
    bool guessedYear(false);
    if (t.tm_year == 0) {
        struct tm now;
        ::memset(&now, 0, sizeof(struct tm));

        if (tz.utcToLocal(CTimeUtils::now(), now) == false) {
            return false;
        }

        t.tm_year = now.tm_year;

        guessedYear = true;
        copy = t;
    }

    // Use CTimezone to allow for timezone
    preTime = tz.localToUtc(t);

    // If we guessed the year and the time is in the future then decrease the
    // year and recalculate
    // Use a tolerance of 5 minutes in case of slight clock discrepancies
    // between different machines at the customer location
    if (guessedYear && preTime > CTimeUtils::now() + MAX_CLOCK_DISCREPANCY) {
        // Recalculate using a copy since mktime changes the contents of the
        // struct
        copy.tm_year -= 1;

        preTime = tz.localToUtc(copy);
    }

    return true;
}

void CTimeUtils::toStringCommon(core_t::TTime t,
                                const std::string &format,
                                std::string &result) {
    // core_t::TTime holds an epoch time (UTC)
    struct tm out;

    CTimezone &tz = CTimezone::instance();
    if (tz.utcToLocal(t, out) == false) {
        LOG_ERROR("Cannot convert time " << t << " : " << ::strerror(errno));
        result.clear();
        return;
    }

    static const size_t SIZE(256);
    char buf[SIZE] = { '\0' };

    size_t ret(CStrFTime::strFTime(buf, SIZE, format.c_str(), &out));
    if (ret == 0) {
        LOG_ERROR("Cannot convert time " << t << " : " << ::strerror(errno));
        result.clear();
        return;
    }

    result = buf;
}

bool CTimeUtils::isDateWord(const std::string &word) {
    return CDateWordCache::instance().isDateWord(word);
}


// Initialise statics for the inner class CDateWordCache
CFastMutex CTimeUtils::CDateWordCache::ms_InitMutex;
volatile CTimeUtils::CDateWordCache *CTimeUtils::CDateWordCache::ms_Instance(0);

const CTimeUtils::CDateWordCache &CTimeUtils::CDateWordCache::instance(void) {
    if (ms_Instance == 0) {
        CScopedFastLock lock(ms_InitMutex);

        // Even if we get into this code block in more than one thread, whatever
        // measures the compiler is taking to ensure this variable is only
        // constructed once should be fine given that the block is protected by
        // a mutex.
        static volatile CDateWordCache instance;

        ms_Instance = &instance;
    }

    // Need to explicitly cast away volatility
    return *const_cast<const CDateWordCache *>(ms_Instance);
}

bool CTimeUtils::CDateWordCache::isDateWord(const std::string &word) const
{
    return m_DateWords.find(word) != m_DateWords.end();
}

CTimeUtils::CDateWordCache::CDateWordCache(void) {
    static const size_t SIZE(256);
    char buf[SIZE] = { '\0' };

    struct tm workTime;
    ::memset(&workTime, 0, sizeof(struct tm));

    // Start on Saturday 1st January 1994 (not a leap year)
    workTime.tm_sec = 0;
    workTime.tm_min = 0;
    workTime.tm_hour = 12;
    workTime.tm_mday = 1;
    workTime.tm_mon = 0;
    workTime.tm_year = 94;
    workTime.tm_wday = 6;
    workTime.tm_yday = 1;
    workTime.tm_isdst = -1;

    // Populate day-of-week names and abbreviations
    for (int dayOfWeek = 0; dayOfWeek < 7; ++dayOfWeek) {
        ++workTime.tm_mday;
        workTime.tm_wday = dayOfWeek;
        ++workTime.tm_yday;

        if (::strftime(buf, SIZE, "%a", &workTime) > 0) {
            m_DateWords.insert(buf);
        }
        if (::strftime(buf, SIZE, "%A", &workTime) > 0) {
            m_DateWords.insert(buf);
        }
    }

    // Populate month names and abbreviations - first January
    if (::strftime(buf, SIZE, "%b", &workTime) > 0) {
        m_DateWords.insert(buf);
    }
    if (::strftime(buf, SIZE, "%B", &workTime) > 0) {
        m_DateWords.insert(buf);
    }

    static const int DAYS_PER_MONTH[] = { 31, 28, 31, 30, 31, 30,
                                          31, 31, 30, 31, 30, 31 };

    // Populate other month names and abbreviations
    for (int month = 1; month < 12; ++month) {
        int prevMonthDays(DAYS_PER_MONTH[month - 1]);
        workTime.tm_mon = month;
        workTime.tm_wday += prevMonthDays;
        workTime.tm_wday %= 7;
        workTime.tm_yday += prevMonthDays;

        if (::strftime(buf, SIZE, "%b", &workTime) > 0) {
            m_DateWords.insert(buf);
        }

        if (::strftime(buf, SIZE, "%B", &workTime) > 0) {
            m_DateWords.insert(buf);
        }
    }

    // These timezones may be in use anywhere
    m_DateWords.insert("GMT");
    m_DateWords.insert("UTC");

    // Finally, add the current timezone (if available)
    CTimezone &tz = CTimezone::instance();
    const std::string &stdAbbrev = tz.stdAbbrev();
    if (!stdAbbrev.empty()) {
        m_DateWords.insert(stdAbbrev);
    }
    const std::string &dstAbbrev = tz.dstAbbrev();
    if (!dstAbbrev.empty()) {
        m_DateWords.insert(dstAbbrev);
    }
}

CTimeUtils::CDateWordCache::~CDateWordCache(void) {
    ms_Instance = 0;
}


}
}

