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

#include <maths/time_series/CTimeSeriesPredictor.h>

#include <core/CMemoryDef.h>

#include <maths/common/CMathsFuncs.h>
#include <maths/common/CMathsFuncsForMatrixAndVectorTypes.h>

#include <maths/time_series/CTrendDecomposition.h>
#include <maths/time_series/CSeasonalDecomposition.h>
#include <maths/time_series/CCalendarDecomposition.h>

namespace ml {
namespace maths {
namespace time_series {

CTimeSeriesPredictor::CTimeSeriesPredictor(const CTrendDecomposition* trendDecomposition,
                                         const CSeasonalDecomposition* seasonalDecomposition,
                                         const CCalendarDecomposition* calendarDecomposition)
    : m_TrendDecomposition(trendDecomposition),
      m_SeasonalDecomposition(seasonalDecomposition),
      m_CalendarDecomposition(calendarDecomposition) {
}

CTimeSeriesPredictor::TVector2x1
CTimeSeriesPredictor::value(core_t::TTime time, 
                           double confidence, 
                           bool isNonNegative,
                           core_t::TTime timeShift) const {
    time += timeShift;
    
    // Combine predictions from all components
    TVector2x1 result{0.0};
    
    // Add trend prediction if available
    if (m_TrendDecomposition && m_TrendDecomposition->initialized()) {
        result += m_TrendDecomposition->value(time - timeShift, confidence, false);
    }
    
    // Add seasonal prediction if available
    if (m_SeasonalDecomposition && m_SeasonalDecomposition->initialized()) {
        result += m_SeasonalDecomposition->value(time - timeShift, confidence, false);
    }
    
    // Add calendar prediction if available
    if (m_CalendarDecomposition && m_CalendarDecomposition->initialized()) {
        result += m_CalendarDecomposition->value(time - timeShift, confidence, false);
    }
    
    return isNonNegative ? max(result, 0.0) : result;
}

CTimeSeriesPredictor::TFilteredPredictor
CTimeSeriesPredictor::predictor() const {
    // Get component predictors
    TFilteredPredictor trendPredictor;
    if (m_TrendDecomposition) {
        trendPredictor = m_TrendDecomposition->predictor();
    } else {
        trendPredictor = [](core_t::TTime, const TBoolVec&) { return 0.0; };
    }
    
    TFilteredPredictor seasonalPredictor;
    if (m_SeasonalDecomposition) {
        seasonalPredictor = m_SeasonalDecomposition->predictor();
    } else {
        seasonalPredictor = [](core_t::TTime, const TBoolVec&) { return 0.0; };
    }
    
    TFilteredPredictor calendarPredictor;
    if (m_CalendarDecomposition) {
        calendarPredictor = m_CalendarDecomposition->predictor();
    } else {
        calendarPredictor = [](core_t::TTime, const TBoolVec&) { return 0.0; };
    }
    
    // Combine predictors
    return [trendPredictor, seasonalPredictor, calendarPredictor](core_t::TTime time, const TBoolVec& mask) {
        return trendPredictor(time, mask) + seasonalPredictor(time, mask) + calendarPredictor(time, mask);
    };
}

CTimeSeriesPredictor::TVector2x1 
CTimeSeriesPredictor::trendValue(core_t::TTime time, double confidence, bool isNonNegative) const {
    if (!m_TrendDecomposition) {
        return {0.0, 0.0};
    }
    
    TVector2x1 result = m_TrendDecomposition->value(time, confidence, false);
    return isNonNegative ? max(result, 0.0) : result;
}

CTimeSeriesPredictor::TVector2x1 
CTimeSeriesPredictor::seasonalValue(core_t::TTime time, double confidence, bool isNonNegative) const {
    if (!m_SeasonalDecomposition) {
        return {0.0, 0.0};
    }
    
    TVector2x1 result = m_SeasonalDecomposition->value(time, confidence, false);
    return isNonNegative ? max(result, 0.0) : result;
}

CTimeSeriesPredictor::TVector2x1 
CTimeSeriesPredictor::calendarValue(core_t::TTime time, double confidence, bool isNonNegative) const {
    if (!m_CalendarDecomposition) {
        return {0.0, 0.0};
    }
    
    TVector2x1 result = m_CalendarDecomposition->value(time, confidence, false);
    return isNonNegative ? max(result, 0.0) : result;
}

void CTimeSeriesPredictor::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CTimeSeriesPredictor");
}

std::size_t CTimeSeriesPredictor::memoryUsage() const {
    return sizeof(*this);
}

}
}
}
