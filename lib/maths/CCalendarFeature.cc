/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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

#include <maths/CCalendarFeature.h>

#include <core/CLogger.h>
#include <core/Constants.h>
#include <core/CPersistUtils.h>
#include <core/CTimezone.h>

#include <maths/CChecksum.h>
#include <maths/CIntegerTools.h>

#include <boost/numeric/conversion/bounds.hpp>

namespace ml {
namespace maths {

namespace {
const int LAST_DAY_IN_MONTH[] = {
    30, 27, 30, 29, 30, 29, 30, 30, 29, 30, 29, 30
};
const std::string DAYS[] = {
    "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"
};

const int DAY  = core::constants::DAY;

//! Check if \p year (years since 1900) is a leap year.
bool isLeapYear(int year) {
    year += 1900;
    return (year % 4 == 0 && year % 100 == 0) || year % 400 == 0;
}

//! Get the number of days in \p month of \p year.
int lastDayInMonth(int year, int month) {
    return LAST_DAY_IN_MONTH[month] + (month == 1 && isLeapYear(year) ? 1 : 0);
}

//! Compute the day of week of the first of the month if the
//! \p dayOfMonth is a \p dayOfWeek.
int dayOfFirst(int dayOfMonth, int dayOfWeek) {
    return (CIntegerTools::ceil(dayOfMonth, 7) - dayOfMonth + dayOfWeek) % 7;
}

//! Print the day or week count.
std::string print_(int count, bool suffix) {
    static const std::string suffix_[] = { "th", "st", "nd", "rd", "th" };
    return  core::CStringUtils::typeToString(count)
            + (suffix ? suffix_[count < 20 ? std::min(count, 4) :
                                      std::min(count % 10, 4)] : "");
}

}

CCalendarFeature::CCalendarFeature(void) : m_Feature(INVALID), m_Value(INVALID) {}

CCalendarFeature::CCalendarFeature(uint16_t feature, core_t::TTime time) :
    m_Feature(INVALID), m_Value(INVALID) {
    int dayOfWeek{};
    int dayOfMonth{};
    int dayOfYear{};
    int month{};
    int year{};
    int secondsSinceMidnight{};
    if (core::CTimezone::instance().dateFields(time,
                                               dayOfWeek, dayOfMonth, dayOfYear,
                                               month, year, secondsSinceMidnight)) {
        dayOfMonth -= 1;
        this->initialize(feature, dayOfWeek, dayOfMonth, month, year);
    } else {
        LOG_ERROR("Invalid time: " << time);
    }
}

CCalendarFeature::TCalendarFeature4Ary CCalendarFeature::features(core_t::TTime time) {
    TCalendarFeature4Ary result;
    int dayOfWeek{};
    int dayOfMonth{};
    int dayOfYear{};
    int month{};
    int year{};
    int secondsSinceMidnight{};
    if (core::CTimezone::instance().dateFields(time,
                                               dayOfWeek, dayOfMonth, dayOfYear,
                                               month, year, secondsSinceMidnight)) {
        dayOfMonth -= 1;
        auto i = result.begin();
        for (uint16_t feature = BEGIN_FEATURES; feature < END_FEATURES; ++feature, ++i) {
            i->initialize(feature, dayOfWeek, dayOfMonth, month, year);
        }
    } else {
        LOG_ERROR("Invalid time: " << time);
    }
    return result;
}

void CCalendarFeature::initialize(uint16_t feature,
                                  int dayOfWeek,
                                  int dayOfMonth,
                                  int month,
                                  int year) {
    switch (feature) {
        case DAYS_SINCE_START_OF_MONTH:
            m_Feature = feature;
            m_Value = static_cast<uint16_t>(dayOfMonth);
            break;
        case DAYS_BEFORE_END_OF_MONTH:
            m_Feature = feature;
            m_Value = static_cast<uint16_t>(lastDayInMonth(year, month) - dayOfMonth);
            break;
        case DAY_OF_WEEK_AND_WEEKS_SINCE_START_OF_MONTH:
            m_Feature = feature;
            m_Value = static_cast<uint16_t>(8 * (dayOfMonth / 7) + dayOfWeek);
            break;
        case DAY_OF_WEEK_AND_WEEKS_BEFORE_END_OF_MONTH:
            m_Feature = feature;
            m_Value = static_cast<uint16_t>(
                          8 * ((lastDayInMonth(year, month) - dayOfMonth) / 7) + dayOfWeek);
            break;
        default:
            LOG_ERROR("Invalid feature: " << feature);
            break;
    }
}

bool CCalendarFeature::fromDelimited(const std::string &value) {
    int state[2];
    if (core::CPersistUtils::fromString(value, boost::begin(state), boost::end(state))) {
        m_Feature = static_cast<uint16_t>(state[0]);
        m_Value   = static_cast<uint16_t>(state[1]);
        return true;
    }
    return false;
}

std::string CCalendarFeature::toDelimited(void) const {
    int state[2] = {
        static_cast<int>(m_Feature),
        static_cast<int>(m_Value)
    };
    const int *begin = boost::begin(state);
    const int *end   = boost::end(state);
    return core::CPersistUtils::toString(begin, end);
}

bool CCalendarFeature::operator==(CCalendarFeature rhs) const {
    return m_Feature == rhs.m_Feature && m_Value == rhs.m_Value;
}

bool CCalendarFeature::operator<(CCalendarFeature rhs) const {
    return COrderings::lexicographical_compare(m_Feature, m_Value, rhs.m_Feature, rhs.m_Value);
}

core_t::TTime CCalendarFeature::offset(core_t::TTime time) const {
    int dayOfWeek{};
    int dayOfMonth{};
    int dayOfYear{};
    int month{};
    int year{};
    int secondsSinceMidnight{};
    if (core::CTimezone::instance().dateFields(time,
                                               dayOfWeek, dayOfMonth, dayOfYear,
                                               month, year, secondsSinceMidnight)) {
        dayOfMonth -= 1;
        switch (m_Feature) {
            case DAYS_SINCE_START_OF_MONTH:
                return  DAY * (dayOfMonth - static_cast<int>(m_Value)) + secondsSinceMidnight;
            case DAYS_BEFORE_END_OF_MONTH:
                return  DAY * (dayOfMonth - (lastDayInMonth(year, month) - static_cast<int>(m_Value)))
                        + secondsSinceMidnight;
            case DAY_OF_WEEK_AND_WEEKS_SINCE_START_OF_MONTH: {
                int dayOfFirst_  = dayOfFirst(dayOfMonth, dayOfWeek);
                int dayOfWeek_   = static_cast<int>(m_Value) % 8;
                int weekOfMonth_ = static_cast<int>(m_Value) / 8;
                int dayOfMonth_  = 7 * weekOfMonth_ + (7 + dayOfWeek_ - dayOfFirst_) % 7;
                return DAY * (dayOfMonth - dayOfMonth_) + secondsSinceMidnight;
            }
            case DAY_OF_WEEK_AND_WEEKS_BEFORE_END_OF_MONTH: {
                int lastDayInMonth_    = lastDayInMonth(year, month);
                int dayOfLast_         = (lastDayInMonth_ + dayOfFirst(dayOfMonth, dayOfWeek)) % 7;
                int dayOfWeek_         = static_cast<int>(m_Value) % 8;
                int weeksToEndOfMonth_ = static_cast<int>(m_Value) / 8;
                int dayOfMonth_        = lastDayInMonth_ - (7 * weeksToEndOfMonth_ + (7 + dayOfLast_ - dayOfWeek_) % 7);
                return DAY * (dayOfMonth - dayOfMonth_) + secondsSinceMidnight;
            }
            default:
                LOG_ERROR("Invalid feature: '" << m_Feature << "'");
                break;
        }
    } else {
        LOG_ERROR("Invalid time: '" << time << "'");
    }
    return 0;
}

bool CCalendarFeature::inWindow(core_t::TTime time) const {
    core_t::TTime offset = this->offset(time);
    return offset >= 0 && offset < this->window();
}

core_t::TTime CCalendarFeature::window(void) const {
    return core::constants::DAY;
}

uint64_t CCalendarFeature::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_Feature);
    return CChecksum::calculate(seed, m_Value);
}

std::string CCalendarFeature::print(void) const {
    switch (m_Feature) {
        case DAYS_SINCE_START_OF_MONTH:
            return print_(static_cast<int>(m_Value) + 1, true) + " day of month";
        case DAYS_BEFORE_END_OF_MONTH:
            return print_(static_cast<int>(m_Value), false) + " days before end of month";
        case DAY_OF_WEEK_AND_WEEKS_SINCE_START_OF_MONTH: {
            int dayOfWeek_   = static_cast<int>(m_Value) % 8;
            int weekOfMonth_ = static_cast<int>(m_Value) / 8;
            return print_(weekOfMonth_ + 1, true) + " " + DAYS[dayOfWeek_] + " of month";
        }
        case DAY_OF_WEEK_AND_WEEKS_BEFORE_END_OF_MONTH: {
            int dayOfWeek_         = static_cast<int>(m_Value) % 8;
            int weeksToEndOfMonth_ = static_cast<int>(m_Value) / 8;
            return print_(weeksToEndOfMonth_, false) + " " + DAYS[dayOfWeek_] + "s before end of month";
        }
    }
    return "-";
}

const uint16_t CCalendarFeature::INVALID(boost::numeric::bounds<uint16_t>::highest());

}
}
