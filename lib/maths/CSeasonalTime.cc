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

#include <maths/CSeasonalTime.h>

#include <core/CLogger.h>
#include <core/CPersistUtils.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/Constants.h>

#include <maths/CChecksum.h>
#include <maths/CIntegerTools.h>

#include <boost/array.hpp>
#include <boost/numeric/conversion/bounds.hpp>

#include <cstddef>
#include <string>

namespace ml {
namespace maths {
namespace {
// DO NOT change the existing tags if new sub-classes are added.
const std::string DIURNAL_TIME_TAG("a");
const std::string ARBITRARY_PERIOD_TIME_TAG("b");
}

//////// CSeasonalTime ////////

CSeasonalTime::CSeasonalTime()
    : m_Period(0), m_RegressionOrigin(0), m_Precedence(0) {
}

CSeasonalTime::CSeasonalTime(core_t::TTime period, double precedence)
    : m_Period(period), m_RegressionOrigin(0), m_Precedence(precedence) {
}

bool CSeasonalTime::operator<(const CSeasonalTime& rhs) const {
    return COrderings::lexicographical_compare(m_Period, -m_Precedence,
                                               rhs.m_Period, -rhs.m_Precedence);
}

double CSeasonalTime::periodic(core_t::TTime time) const {
    return static_cast<double>((time - this->startOfWindow(time)) % m_Period);
}

double CSeasonalTime::regression(core_t::TTime time) const {
    return static_cast<double>(time - m_RegressionOrigin) /
           static_cast<double>(this->regressionTimeScale());
}

double CSeasonalTime::regressionInterval(core_t::TTime start, core_t::TTime end) const {
    return static_cast<double>(end - start) /
           static_cast<double>(this->regressionTimeScale());
}

core_t::TTime CSeasonalTime::startOfWindowRepeat(core_t::TTime time) const {
    return this->startOfWindowRepeat(this->windowRepeatStart(), time);
}

core_t::TTime CSeasonalTime::startOfWindow(core_t::TTime time) const {
    return this->startOfWindowRepeat(this->windowRepeatStart() + this->windowStart(), time);
}

bool CSeasonalTime::inWindow(core_t::TTime time) const {
    time = time - this->startOfWindowRepeat(time);
    return time >= this->windowStart() && time < this->windowEnd();
}

core_t::TTime CSeasonalTime::period() const {
    return m_Period;
}

void CSeasonalTime::period(core_t::TTime period) {
    m_Period = period;
}

core_t::TTime CSeasonalTime::regressionOrigin() const {
    return m_RegressionOrigin;
}

void CSeasonalTime::regressionOrigin(core_t::TTime origin) {
    m_RegressionOrigin = origin;
}

CSeasonalTime::TTimeTimePr CSeasonalTime::window() const {
    return {this->windowStart(), this->windowEnd()};
}

core_t::TTime CSeasonalTime::windowLength() const {
    return this->windowEnd() - this->windowStart();
}

bool CSeasonalTime::windowed() const {
    return this->windowLength() < this->windowRepeat();
}

double CSeasonalTime::fractionInWindow() const {
    return static_cast<double>(std::max(this->period(), this->windowLength())) /
           static_cast<double>(this->windowRepeat());
}

bool CSeasonalTime::excludes(const CSeasonalTime& other) const {
    return std::abs(other.m_Period - m_Period) < std::max(other.m_Period, m_Period) / 20 &&
           m_Precedence >= other.m_Precedence;
}

core_t::TTime CSeasonalTime::startOfWindowRepeat(core_t::TTime offset,
                                                 core_t::TTime time) const {
    return offset + CIntegerTools::floor(time - offset, this->windowRepeat());
}

//////// CDiurnalTime ////////

CDiurnalTime::CDiurnalTime()
    : m_StartOfWeek(0), m_WindowStart(0), m_WindowEnd(0) {
}

CDiurnalTime::CDiurnalTime(core_t::TTime startOfWeek,
                           core_t::TTime windowStart,
                           core_t::TTime windowEnd,
                           core_t::TTime period,
                           double precedence)
    : CSeasonalTime(period, precedence), m_StartOfWeek(startOfWeek),
      m_WindowStart(windowStart), m_WindowEnd(windowEnd) {
}

CDiurnalTime* CDiurnalTime::clone() const {
    return new CDiurnalTime(*this);
}

bool CDiurnalTime::fromString(const std::string& value) {
    boost::array<core_t::TTime, 5> times;
    if (core::CPersistUtils::fromString(value, times)) {
        m_StartOfWeek = times[0];
        m_WindowStart = times[1];
        m_WindowEnd = times[2];
        this->period(times[3]);
        this->regressionOrigin(times[4]);
        return true;
    }
    return false;
}

std::string CDiurnalTime::toString() const {
    boost::array<core_t::TTime, 5> times;
    times[0] = m_StartOfWeek;
    times[1] = m_WindowStart;
    times[2] = m_WindowEnd;
    times[3] = this->period();
    times[4] = this->regressionOrigin();
    return core::CPersistUtils::toString(times);
}

core_t::TTime CDiurnalTime::windowRepeat() const {
    return core::constants::WEEK;
}

core_t::TTime CDiurnalTime::windowRepeatStart() const {
    return m_StartOfWeek;
}

core_t::TTime CDiurnalTime::windowStart() const {
    return m_WindowStart;
}

core_t::TTime CDiurnalTime::windowEnd() const {
    return m_WindowEnd;
}

bool CDiurnalTime::hasWeekend() const {
    return this->windowLength() == core::constants::WEEKEND ||
           this->windowLength() == core::constants::WEEKDAYS;
}

uint64_t CDiurnalTime::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_StartOfWeek);
    seed = CChecksum::calculate(seed, m_WindowStart);
    seed = CChecksum::calculate(seed, m_WindowEnd);
    return CChecksum::calculate(seed, this->period());
}

core_t::TTime CDiurnalTime::regressionTimeScale() const {
    return core::constants::WEEK;
}

//////// CGeneralPeriodTime ////////

CGeneralPeriodTime::CGeneralPeriodTime(core_t::TTime period, double precedence)
    : CSeasonalTime(period, precedence) {
}

CGeneralPeriodTime* CGeneralPeriodTime::clone() const {
    return new CGeneralPeriodTime(*this);
}

bool CGeneralPeriodTime::fromString(const std::string& value) {
    boost::array<core_t::TTime, 2> times;
    if (core::CPersistUtils::fromString(value, times)) {
        this->period(times[0]);
        this->regressionOrigin(times[1]);
        return true;
    }
    return false;
}

std::string CGeneralPeriodTime::toString() const {
    boost::array<core_t::TTime, 2> times;
    times[0] = this->period();
    times[1] = this->regressionOrigin();
    return core::CPersistUtils::toString(times);
}

core_t::TTime CGeneralPeriodTime::windowRepeat() const {
    return this->period();
}

core_t::TTime CGeneralPeriodTime::windowRepeatStart() const {
    return 0;
}

core_t::TTime CGeneralPeriodTime::windowStart() const {
    return 0;
}

core_t::TTime CGeneralPeriodTime::windowEnd() const {
    return this->period();
}

bool CGeneralPeriodTime::hasWeekend() const {
    return false;
}

uint64_t CGeneralPeriodTime::checksum(uint64_t seed) const {
    return CChecksum::calculate(seed, this->period());
}

core_t::TTime CGeneralPeriodTime::regressionTimeScale() const {
    return std::max(core::constants::WEEK, this->period());
}

//////// CSeasonalTimeStateSerializer ////////

bool CSeasonalTimeStateSerializer::acceptRestoreTraverser(TSeasonalTimePtr& result,
                                                          core::CStateRestoreTraverser& traverser) {
    std::size_t numResults = 0;

    do {
        const std::string& name = traverser.name();
        if (name == DIURNAL_TIME_TAG) {
            result.reset(new CDiurnalTime);
            result->fromString(traverser.value());
            ++numResults;
        } else if (name == ARBITRARY_PERIOD_TIME_TAG) {
            result.reset(new CGeneralPeriodTime);
            result->fromString(traverser.value());
            ++numResults;
        } else {
            LOG_ERROR(<< "No seasonal time corresponds to name " << traverser.name());
            return false;
        }
    } while (traverser.next());

    if (numResults != 1) {
        LOG_ERROR(<< "Expected 1 (got " << numResults << ") seasonal time tags");
        result.reset();
        return false;
    }

    return true;
}

void CSeasonalTimeStateSerializer::acceptPersistInserter(const CSeasonalTime& time,
                                                         core::CStatePersistInserter& inserter) {
    if (dynamic_cast<const CDiurnalTime*>(&time) != nullptr) {
        inserter.insertValue(DIURNAL_TIME_TAG, time.toString());
    } else if (dynamic_cast<const CGeneralPeriodTime*>(&time) != nullptr) {
        inserter.insertValue(ARBITRARY_PERIOD_TIME_TAG, time.toString());
    } else {
        LOG_ERROR(<< "Seasonal time with type " << typeid(time).name() << " has no defined name");
    }
}
}
}
