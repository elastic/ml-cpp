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

#include <maths/time_series/CTimeSeriesForecaster.h>

#include <core/CLogger.h>
#include <core/CMemoryDef.h>

#include <maths/common/CChecksum.h>
#include <maths/common/CIntegerTools.h>
#include <maths/common/CMathsFuncs.h>
#include <maths/common/CMathsFuncsForMatrixAndVectorTypes.h>

#include <maths/time_series/CTimeSeriesDecompositionInterface.h>
#include <maths/time_series/CTrendDecomposition.h>
#include <maths/time_series/CSeasonalDecomposition.h>
#include <maths/time_series/CCalendarDecomposition.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace ml {
namespace maths {
namespace time_series {

CTimeSeriesForecaster::CTimeSeriesForecaster(const CTimeSeriesDecompositionInterface& decomposition)
    : m_Decomposition(&decomposition) {
}

CTimeSeriesForecaster::CTimeSeriesForecaster(const CTrendDecomposition* trendDecomposition,
                                           const CSeasonalDecomposition* seasonalDecomposition,
                                           const CCalendarDecomposition* calendarDecomposition)
    : m_TrendDecomposition(trendDecomposition),
      m_SeasonalDecomposition(seasonalDecomposition),
      m_CalendarDecomposition(calendarDecomposition) {
}

core_t::TTime CTimeSeriesForecaster::maximumForecastInterval() const {
    if (m_Decomposition) {
        return m_Decomposition->maximumForecastInterval();
    }
    
    // If using individual components, use the trend component interval
    if (m_TrendDecomposition) {
        return m_TrendDecomposition->maximumForecastInterval();
    }
    
    // Default forecast interval if no components available
    return 3 * core::constants::WEEK;
}

void CTimeSeriesForecaster::forecast(core_t::TTime startTime,
                                    core_t::TTime endTime,
                                    core_t::TTime step,
                                    double confidence,
                                    double minimumScale,
                                    bool isNonNegative,
                                    core_t::TTime timeShift,
                                    const TWriteForecastResult& writer) const {
    if (endTime < startTime) {
        LOG_ERROR(<< "Bad forecast range: [" << startTime << "," << endTime << "]");
        return;
    }
    if (confidence < 0.0 || confidence >= 100.0) {
        LOG_ERROR(<< "Bad confidence interval: " << confidence << "%");
        return;
    }

    // If we have a full decomposition, use it directly
    if (m_Decomposition) {
        m_Decomposition->forecast(startTime, endTime, step, confidence, 
                                 minimumScale, isNonNegative, writer);
        return;
    }

    startTime += timeShift;
    endTime += timeShift;
    endTime = startTime + common::CIntegerTools::ceil(endTime - startTime, step);
    
    // Otherwise, calculate forecasts from individual components
    for (core_t::TTime time = startTime; time < endTime; time += step) {
        TDouble3Vec result = this->calculateForecastWithBounds(
            time - timeShift, confidence, minimumScale, isNonNegative, timeShift);
            
        writer(time - timeShift, result);
    }
}

CTimeSeriesForecaster::TDouble3Vec 
CTimeSeriesForecaster::calculateForecastWithBounds(core_t::TTime time,
                                                 double confidence,
                                                 double minimumScale,
                                                 bool isNonNegative,
                                                 core_t::TTime timeShift) const {
    // If using full decomposition, delegate to it
    if (m_Decomposition) {
        TDouble3Vec result(3);
        m_Decomposition->forecast(time, time + 1, 1, confidence, minimumScale,
                                 isNonNegative, [&result](core_t::TTime, const TDouble3Vec& forecast) {
            result = forecast;
        });
        return result;
    }

    // Otherwise, combine individual component forecasts
    TDouble3Vec trendResult{0.0, 0.0, 0.0};
    TDouble3Vec seasonalResult{0.0, 0.0, 0.0};
    TDouble3Vec calendarResult{0.0, 0.0, 0.0};
    
    // Get trend forecast if available
    if (m_TrendDecomposition) {
        trendResult = this->calculateTrendForecastWithConfidenceInterval(
            time + timeShift, confidence, minimumScale);
    }
    
    // Get seasonal forecast if available
    if (m_SeasonalDecomposition) {
        seasonalResult = this->calculateSeasonalForecastWithConfidenceInterval(
            time + timeShift, confidence, minimumScale);
    }
    
    // Get calendar forecast if available
    if (m_CalendarDecomposition) {
        calendarResult = this->calculateCalendarForecastWithConfidenceInterval(
            time + timeShift, confidence, minimumScale);
    }
    
    // Combine forecasts
    TDouble3Vec combinedForecast = {
        trendResult[0] + seasonalResult[0] + calendarResult[0],
        trendResult[1] + seasonalResult[1] + calendarResult[1],
        trendResult[2] + seasonalResult[2] + calendarResult[2]
    };
    
    if (isNonNegative) {
        combinedForecast[0] = std::max(0.0, combinedForecast[0]);
        combinedForecast[1] = std::max(0.0, combinedForecast[1]);
        combinedForecast[2] = std::max(0.0, combinedForecast[2]);
    }
    
    return combinedForecast;
}

CTimeSeriesForecaster::TDouble3Vec 
CTimeSeriesForecaster::calculateTrendForecastWithConfidenceInterval(core_t::TTime time,
                                                                  double confidence,
                                                                  double minimumScale) const {
    if (!m_TrendDecomposition) {
        return {0.0, 0.0, 0.0};
    }
    
    TDouble3Vec result{0.0, 0.0, 0.0};
    
    // Use a lambda to capture the forecast result
    auto writer = [&result](core_t::TTime, const TDouble3Vec& forecast) {
        result = forecast;
    };
    
    // Generate the trend forecast
    m_TrendDecomposition->forecast(time, time + 1, 1, confidence, minimumScale, false, writer);
    
    return result;
}

CTimeSeriesForecaster::TDouble3Vec 
CTimeSeriesForecaster::calculateSeasonalForecastWithConfidenceInterval(core_t::TTime time,
                                                                     double confidence,
                                                                     double minimumScale) const {
    if (!m_SeasonalDecomposition) {
        return {0.0, 0.0, 0.0};
    }
    
    TDouble3Vec result{0.0, 0.0, 0.0};
    
    // Use a lambda to capture the forecast result
    auto writer = [&result](core_t::TTime, const TDouble3Vec& forecast) {
        result = forecast;
    };
    
    // Generate the seasonal forecast
    m_SeasonalDecomposition->forecast(time, time + 1, 1, confidence, minimumScale, false, writer);
    
    return result;
}

CTimeSeriesForecaster::TDouble3Vec 
CTimeSeriesForecaster::calculateCalendarForecastWithConfidenceInterval(core_t::TTime time,
                                                                     double confidence,
                                                                     double minimumScale) const {
    if (!m_CalendarDecomposition) {
        return {0.0, 0.0, 0.0};
    }
    
    TDouble3Vec result{0.0, 0.0, 0.0};
    
    // Use a lambda to capture the forecast result
    auto writer = [&result](core_t::TTime, const TDouble3Vec& forecast) {
        result = forecast;
    };
    
    // Generate the calendar forecast
    m_CalendarDecomposition->forecast(time, time + 1, 1, confidence, minimumScale, false, writer);
    
    return result;
}

void CTimeSeriesForecaster::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CTimeSeriesForecaster");
}

std::size_t CTimeSeriesForecaster::memoryUsage() const {
    return sizeof(*this);
}

}
}
}
