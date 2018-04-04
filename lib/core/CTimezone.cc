/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CTimezone.h>

#include <core/CLogger.h>
#include <core/CScopedFastLock.h>
#include <core/CSetEnv.h>

#include <errno.h>
#include <string.h>

namespace {
// To ensure the singleton is constructed before multiple threads may require it
// call instance() during the static initialisation phase of the program.  Of
// course, the instance may already be constructed before this if another static
// object has used it.
const ml::core::CTimezone& DO_NOT_USE_THIS_VARIABLE = ml::core::CTimezone::instance();
}

namespace ml {
namespace core {

CTimezone::CTimezone() {
}

CTimezone::~CTimezone() {
}

CTimezone& CTimezone::instance() {
    static CTimezone instance;
    return instance;
}

const std::string& CTimezone::timezoneName() const {
    CScopedFastLock lock(m_Mutex);

    return m_Name;
}

bool CTimezone::timezoneName(const std::string& name) {
    CScopedFastLock lock(m_Mutex);

    if (CSetEnv::setEnv("TZ", name.c_str(), 1) != 0) {
        LOG_ERROR("Unable to set TZ environment variable to " << name << " : " << ::strerror(errno));

        return false;
    }

    ::tzset();

    m_Name = name;

    return true;
}

bool CTimezone::setTimezone(const std::string& timezone) {
    return CTimezone::instance().timezoneName(timezone);
}

std::string CTimezone::stdAbbrev() const {
    CScopedFastLock lock(m_Mutex);

    return ::tzname[0];
}

std::string CTimezone::dstAbbrev() const {
    CScopedFastLock lock(m_Mutex);

    return ::tzname[1];
}

core_t::TTime CTimezone::localToUtc(struct tm& localTime) const {
    return ::mktime(&localTime);
}

bool CTimezone::utcToLocal(core_t::TTime utcTime, struct tm& localTime) const {
    if (::localtime_r(&utcTime, &localTime) == 0) {
        return false;
    }
    return true;
}

bool CTimezone::dateFields(core_t::TTime utcTime,
                           int& daysSinceSunday,
                           int& dayOfMonth,
                           int& daysSinceJanuary1st,
                           int& monthsSinceJanuary,
                           int& yearsSince1900,
                           int& secondsSinceMidnight) const {
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
