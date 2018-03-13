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
#include <core/CTimezone.h>

#include <core/CLogger.h>
#include <core/CResourceLocator.h>
#include <core/CScopedFastLock.h>
#include <core/CSetEnv.h>

#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <exception>

#include <ctype.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

namespace {
// To ensure the singleton is constructed before multiple threads may require it
// call instance() during the static initialisation phase of the program.  Of
// course, the instance may already be constructed before this if another static
// object has used it.
const ml::core::CTimezone &DO_NOT_USE_THIS_VARIABLE = ml::core::CTimezone::instance();
}

namespace ml {
namespace core {

CTimezone::CTimezone(void) {
    CScopedFastLock lock(m_Mutex);

    // We never want to use the Visual C++ runtime library's timezone switching
    // functionality, as it's appallingly bad.  Therefore, make sure the TZ
    // environment variable is unset, so that the operating system settings are
    // obtained and used by the C runtime.  Timezones other than the current
    // operating system timezone will be dealt with using Boost.
    if (::getenv("TZ") != 0) {
        ::_putenv_s("TZ", "");
    }

    ::_tzset();

    // Try to load the Boost timezone database
    std::string path(CResourceLocator::resourceDir());
    path += "/date_time_zonespec.csv";
    try {
        m_TimezoneDb.load_from_file(path);
    } catch (std::exception &ex) {
        LOG_ERROR("Failed to load Boost timezone database from " << path << " : " << ex.what());
    }
}

CTimezone::~CTimezone(void) {}

CTimezone &CTimezone::instance(void) {
    static CTimezone instance;
    return instance;
}

const std::string &CTimezone::timezoneName(void) const {
    CScopedFastLock lock(m_Mutex);

    return m_Name;
}

bool CTimezone::timezoneName(const std::string &name) {
    CScopedFastLock lock(m_Mutex);

    if (name.empty()) {
        m_Timezone.reset();
        m_Name.clear();
        return true;
    }

    m_Timezone = m_TimezoneDb.time_zone_from_region(name);
    if (m_Timezone == 0) {
        LOG_ERROR("Unable to set timezone to "
                  << name << " - operating system timezone settings will be used instead");
        m_Name.clear();

        return false;
    }

    m_Name = name;

    return true;
}

bool CTimezone::setTimezone(const std::string &timezone) {
    return CTimezone::instance().timezoneName(timezone);
}

std::string CTimezone::stdAbbrev(void) const {
    CScopedFastLock lock(m_Mutex);

    if (m_Timezone == 0) {
        return _tzname[0];
    }

    return m_Timezone->std_zone_abbrev();
}

std::string CTimezone::dstAbbrev(void) const {
    CScopedFastLock lock(m_Mutex);

    if (m_Timezone == 0) {
        return _tzname[1];
    }

    return m_Timezone->has_dst() ? m_Timezone->dst_zone_abbrev() : m_Timezone->std_zone_abbrev();
}

core_t::TTime CTimezone::localToUtc(struct tm &localTime) const {
    CScopedFastLock lock(m_Mutex);

    if (m_Timezone == 0) {
        // We're using operating system timezone settings, so use the C
        // runtime's result
        return ::mktime(&localTime);
    }

    // The timezone for this program has been explicitly set, and might not
    // be the same as the operating system timezone, so use Boost

    static const boost::posix_time::ptime EPOCH(boost::gregorian::date(1970, 1, 1));

    boost::gregorian::date dateIn(boost::gregorian::date_from_tm(localTime));
    boost::posix_time::time_duration timeIn(
        static_cast<boost::posix_time::time_duration::hour_type>(localTime.tm_hour),
        static_cast<boost::posix_time::time_duration::min_type>(localTime.tm_min),
        static_cast<boost::posix_time::time_duration::sec_type>(localTime.tm_sec));

    boost::posix_time::time_duration diff;
    try {
        boost::local_time::local_date_time boostLocal(
            dateIn, timeIn, m_Timezone, boost::local_time::local_date_time::EXCEPTION_ON_ERROR);
        diff = boostLocal.utc_time() - EPOCH;
        localTime.tm_isdst = (boostLocal.is_dst() ? 1 : 0);
    } catch (boost::local_time::ambiguous_result &) {
        // If we get an ambiguous time, assume it's standard, not daylight
        // savings
        boost::local_time::local_date_time boostLocal(dateIn, timeIn, m_Timezone, false);
        diff = boostLocal.utc_time() - EPOCH;
        localTime.tm_isdst = 0;
    } catch (std::exception &ex) {
        // Any other exception represents an error in the input
        LOG_ERROR("Error converting local time to UTC : " << ex.what());
        errno = EINVAL;
        return 0;
    }

    return diff.total_seconds();
}

bool CTimezone::utcToLocal(core_t::TTime utcTime, struct tm &localTime) const {
    CScopedFastLock lock(m_Mutex);

    if (m_Timezone == 0) {
        // We're using operating system timezone settings, so use the C runtime
        if (::localtime_s(&localTime, &utcTime) != 0) {
            return false;
        }
        return true;
    }

    // The timezone for this program has been explicitly set, and might not
    // be the same as the operating system timezone, so use Boost

    boost::posix_time::ptime boostUtc(boost::posix_time::from_time_t(utcTime));
    boost::local_time::local_date_time boostLocal(boostUtc, m_Timezone);
    localTime = boost::local_time::to_tm(boostLocal);
    return true;
}

bool CTimezone::dateFields(core_t::TTime utcTime,
                           int &daysSinceSunday,
                           int &dayOfMonth,
                           int &daysSinceJanuary1st,
                           int &monthsSinceJanuary,
                           int &yearsSince1900,
                           int &secondsSinceMidnight) const {
    daysSinceSunday = -1;
    dayOfMonth = -1;
    daysSinceJanuary1st = -1;
    monthsSinceJanuary = -1;
    yearsSince1900 = -1;
    secondsSinceMidnight = -1;

    struct tm result;

    // core_t::TTime holds an epoch time (UTC)
    if (this->utcToLocal(utcTime, result)) {
        daysSinceSunday = result.tm_wday;
        dayOfMonth = result.tm_mday;
        monthsSinceJanuary = result.tm_mon;
        daysSinceJanuary1st = result.tm_yday;
        yearsSince1900 = result.tm_year;
        secondsSinceMidnight = 3600 * result.tm_hour + 60 * result.tm_min + result.tm_sec;
        return true;
    }

    return false;
}
}
}
