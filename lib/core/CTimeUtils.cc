/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CTimeUtils.h>

#include <core/CLogger.h>
#include <core/CRegex.h>
#include <core/CScopedFastLock.h>
#include <core/CStrFTime.h>
#include <core/CStrPTime.h>
#include <core/CStringUtils.h>
#include <core/CTimezone.h>
#include <core/Constants.h>
#include <core/CoreTypes.h>

#include <chrono>

#include <errno.h>
#include <string.h>

namespace {

// Constants for parsing & converting time duration strings in standard ES format
const std::string TIME_DURATION_FORMAT{"([\\d]+)(d|h|m|s|ms|micros|nanos)"};
const std::string DAY{"d"};
const std::string HOUR{"h"};
const std::string MINUTE{"m"};
const std::string SECOND{"s"};
const std::string MILLI_SECOND{"ms"};
const std::string MICRO_SECOND{"micros"};
const std::string NANO_SECOND{"nanos"};
const ml::core_t::TTime SECONDS_IN_DAY{24 * 60 * 60};
const ml::core_t::TTime SECONDS_IN_HOUR{60 * 60};
const ml::core_t::TTime SECONDS_IN_MINUTE{60};
const ml::core_t::TTime MILLI_SECONDS_IN_SECOND{1000};
const ml::core_t::TTime MICRO_SECONDS_IN_SECOND{1000 * 1000};
const ml::core_t::TTime NANO_SECONDS_IN_SECOND{1000 * 1000 * 1000};
}

namespace ml {
namespace core {

// Initialise class static data
const core_t::TTime CTimeUtils::MAX_CLOCK_DISCREPANCY(300);

core_t::TTime CTimeUtils::now() {
    return ::time(nullptr);
}

std::int64_t CTimeUtils::nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
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

int64_t CTimeUtils::toEpochMs(core_t::TTime t) {
    return static_cast<int64_t>(t) * 1000;
}

bool CTimeUtils::strptime(const std::string& format,
                          const std::string& dateTime,
                          core_t::TTime& preTime) {
    if (CTimeUtils::strptimeSilent(format, dateTime, preTime) == false) {
        LOG_ERROR(<< "Unable to convert " << dateTime << " to " << format);
        return false;
    }

    return true;
}

bool CTimeUtils::strptimeSilent(const std::string& format,
                                const std::string& dateTime,
                                core_t::TTime& preTime) {
    struct tm t;
    ::memset(&t, 0, sizeof(struct tm));

    const char* ret(CStrPTime::strPTime(dateTime.c_str(), format.c_str(), &t));
    if (ret == nullptr) {
        return false;
    }

    t.tm_isdst = -1;

    CTimezone& tz = CTimezone::instance();

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

void CTimeUtils::toStringCommon(core_t::TTime t, const std::string& format, std::string& result) {
    // core_t::TTime holds an epoch time (UTC)
    struct tm out;

    CTimezone& tz = CTimezone::instance();
    if (tz.utcToLocal(t, out) == false) {
        LOG_ERROR(<< "Cannot convert time " << t << " : " << ::strerror(errno));
        result.clear();
        return;
    }

    static const size_t SIZE(256);
    char buf[SIZE] = {'\0'};

    size_t ret(CStrFTime::strFTime(buf, SIZE, format.c_str(), &out));
    if (ret == 0) {
        LOG_ERROR(<< "Cannot convert time " << t << " : " << ::strerror(errno));
        result.clear();
        return;
    }

    result = buf;
}

bool CTimeUtils::isDateWord(const std::string& word) {
    return CDateWordCache::instance().isDateWord(word);
}

std::string CTimeUtils::durationToString(core_t::TTime duration) {
    core_t::TTime days = duration / constants::DAY;
    duration -= days * constants::DAY;
    core_t::TTime hours = duration / constants::HOUR;
    duration -= hours * constants::HOUR;
    core_t::TTime minutes = duration / constants::MINUTE;
    duration -= minutes * constants::MINUTE;
    std::string res;
    if (days > 0) {
        res += std::to_string(days) + "d";
    }
    if (hours > 0) {
        res += std::to_string(hours) + "h";
    }
    if (minutes > 0) {
        res += std::to_string(minutes) + "m";
    }
    if ((duration > 0) || res.empty()) {
        res += std::to_string(duration) + "s";
    }
    return res;
}

CTimeUtils::TTimeBoolPr
CTimeUtils::timeDurationStringToSeconds(const std::string& timeDurationString,
                                        core_t::TTime defaultValue) {
    if (timeDurationString.empty()) {
        LOG_ERROR(<< "Unable to parse empty time duration string");
        return {defaultValue, false};
    }

    CRegex regex;

    if (regex.init(TIME_DURATION_FORMAT) == false) {
        LOG_ERROR(<< "Unable to init regex " << TIME_DURATION_FORMAT);
        return {defaultValue, false};
    }

    CRegex::TStrVec tokens;
    const std::string lowerDurationString{CStringUtils::toLower(timeDurationString)};
    if (regex.tokenise(lowerDurationString, tokens) == false) {
        LOG_ERROR(<< "Unable to parse a time duration from " << timeDurationString);
        return {defaultValue, false};
    }

    if (tokens.size() != 2) {
        LOG_INFO(<< "Got wrong number of tokens:: " << tokens.size());
        return {defaultValue, false};
    }

    const std::string& spanStr = tokens[0];
    const std::string& multiplierStr = tokens[1];

    core_t::TTime span{0};
    if (CStringUtils::stringToType(spanStr, span) == false) {
        LOG_ERROR(<< "Could not convert " << spanStr << " to type long.");
        return {defaultValue, false};
    }

    // The assumption here is that the bucket span string has already been validated
    // in the sense that it is convertible to a whole number of seconds with a minimum
    // value of 1s.
    if (multiplierStr == DAY) {
        span *= SECONDS_IN_DAY;
    } else if (multiplierStr == HOUR) {
        span *= SECONDS_IN_HOUR;
    } else if (multiplierStr == MINUTE) {
        span *= SECONDS_IN_MINUTE;
    } else if (multiplierStr == SECOND) {
        // no-op
    } else if (multiplierStr == MILLI_SECOND) {
        span /= MILLI_SECONDS_IN_SECOND;
    } else if (multiplierStr == MICRO_SECOND) {
        span /= MICRO_SECONDS_IN_SECOND;
    } else if (multiplierStr == NANO_SECOND) {
        span /= NANO_SECONDS_IN_SECOND;
    }

    return {span, true};
}

// Initialise statics for the inner class CDateWordCache
CTimeUtils::CDateWordCache* CTimeUtils::CDateWordCache::ms_Instance{nullptr};

const CTimeUtils::CDateWordCache& CTimeUtils::CDateWordCache::instance() {
    if (ms_Instance == nullptr) {
        // This initialisation is thread safe due to the "magic statics" feature
        // introduced in C++11.  This is implemented in Visual Studio 2015 and
        // above.
        static CDateWordCache instance;

        ms_Instance = &instance;
    }

    return *ms_Instance;
}

bool CTimeUtils::CDateWordCache::isDateWord(const std::string& word) const {
    return m_DateWords.find(word) != m_DateWords.end();
}

CTimeUtils::CDateWordCache::CDateWordCache() {
    static const size_t SIZE(256);
    char buf[SIZE] = {'\0'};

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

    static const int DAYS_PER_MONTH[] = {31, 28, 31, 30, 31, 30,
                                         31, 31, 30, 31, 30, 31};

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
    CTimezone& tz = CTimezone::instance();
    const std::string& stdAbbrev = tz.stdAbbrev();
    if (!stdAbbrev.empty()) {
        m_DateWords.insert(stdAbbrev);
    }
    const std::string& dstAbbrev = tz.dstAbbrev();
    if (!dstAbbrev.empty()) {
        m_DateWords.insert(dstAbbrev);
    }
}

CTimeUtils::CDateWordCache::~CDateWordCache() {
    ms_Instance = nullptr;
}
}
}
