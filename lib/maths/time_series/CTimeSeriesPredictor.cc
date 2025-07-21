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

#include <core/CLogger.h>

#include <maths/common/CMathsFuncs.h>
#include <maths/common/CTools.h>

namespace ml {
namespace maths {
namespace time_series {

// CTimeSeriesPredictor::CTimeSeriesPredictor(
//     core_t::TTime timeShift,
//     const CTimeSeriesSmoothing& smoother,
//     std::function<TVector2x1(core_t::TTime, double)> trendValueFunc,
//     std::function<std::function<double(core_t::TTime)>()> trendPredictorFunc,
//     std::function<bool()> usingTrendForPredictionFunc,
//     std::function<const maths_t::TSeasonalComponentVec&()> seasonalComponentsFunc,
//     std::function<const maths_t::TCalendarComponentVec&()> calendarComponentsFunc)
//     : m_TimeShift(timeShift), 
//       m_Smoother(smoother),
//       m_TrendValueFunc(std::move(trendValueFunc)),
//       m_TrendPredictorFunc(std::move(trendPredictorFunc)),
//       m_UsingTrendForPredictionFunc(std::move(usingTrendForPredictionFunc)),
//       m_SeasonalComponentsFunc(std::move(seasonalComponentsFunc)),
//       m_CalendarComponentsFunc(std::move(calendarComponentsFunc)) {
// }

CTimeSeriesPredictor::CTimeSeriesPredictor(
    core_t::TTime timeShift,
    const CTimeSeriesSmoothing& smoother,
    const CTimeSeriesDecompositionDetail::CComponents& components
) : m_TimeShift(timeShift), m_Smoother(smoother), m_Components(components) {
}

CTimeSeriesPredictor::TVector2x1
CTimeSeriesPredictor::value(core_t::TTime time, double confidence, int components, bool smooth) const {
    TVector2x1 result{0.0};

    time += m_TimeShift;

    if ((components & CTimeSeriesDecompositionInterface::E_TrendForced) != 0) {
        result += m_Components.trend().value(time, confidence);
    } else if ((components & CTimeSeriesDecompositionInterface::E_Trend) != 0) {
        if (m_Components.usingTrendForPrediction()) {
            result += m_Components.trend().value(time, confidence);
        }
    }

    if ((components & CTimeSeriesDecompositionInterface::E_Seasonal) != 0) {
        for (const auto& component : m_Components.seasonal()) {
            if (component.initialized() && component.time().inWindow(time)) {
                result += component.value(time, confidence);
            }
        }
    }

    if ((components & CTimeSeriesDecompositionInterface::E_Calendar) != 0) {
        for (const auto& component : m_Components.calendar()) {
            if (component.initialized() && component.feature().inWindow(time)) {
                result += component.value(time, confidence);
            }
        }
    }

    if (smooth) {
        result += this->smooth(
            [&](core_t::TTime time_) {
                return this->value(time_ - m_TimeShift, confidence,
                                 components & CTimeSeriesDecompositionInterface::E_Seasonal, false);
            },
            time, components);
    }

    return result;
}

CTimeSeriesPredictor::TFilteredPredictor
CTimeSeriesPredictor::predictor(int components) const {

    auto trend_ = (((components & CTimeSeriesDecompositionInterface::E_TrendForced) != 0) || 
                  ((components & CTimeSeriesDecompositionInterface::E_Trend) != 0))
                      ? m_Components.trend().predictor()
                      : [](core_t::TTime) { return 0.0; };

    return [components, trend = std::move(trend_), this](core_t::TTime time, const TBoolVec& removedSeasonalMask) {

        double result{0.0};

        time += m_TimeShift;

        if ((components & CTimeSeriesDecompositionInterface::E_TrendForced) != 0) {
            result += trend(time);
        } else if ((components & CTimeSeriesDecompositionInterface::E_Trend) != 0) {
            if (m_Components.usingTrendForPrediction()) {
                result += trend(time);
            }
        }

        if ((components & CTimeSeriesDecompositionInterface::E_Seasonal) != 0) {
            const auto& seasonal = m_Components.seasonal();
            for (std::size_t i = 0; i < seasonal.size(); ++i) {
                if (seasonal[i].initialized() &&
                    (removedSeasonalMask.empty() || removedSeasonalMask[i] == false) &&
                    seasonal[i].time().inWindow(time)) {
                    result += seasonal[i].value(time, 0.0).mean();
                }
            }
        }

        if ((components & CTimeSeriesDecompositionInterface::E_Calendar) != 0) {
            for (const auto& component : m_Components.calendar()) {
                if (component.initialized() && component.feature().inWindow(time)) {
                    result += component.value(time, 0.0).mean();
                }
            }
        }

        return result;
    };
}

template<typename F>
CTimeSeriesPredictor::TVector2x1
CTimeSeriesPredictor::smooth(const F& f, core_t::TTime time, int components) const {
    TVector2x1 result{0.0};
    
    if ((components & CTimeSeriesDecompositionInterface::E_Seasonal) != 0) {
        const auto& seasonal = m_Components.seasonal();
        
        // Apply the smoother
        result(0) = m_Smoother.smooth(
            [&f](core_t::TTime time_) { return f(time_)(0); },
            time, components, seasonal);
    }
    
    return result;
}

}  // namespace time_series
}  // namespace maths
}  // namespace ml
