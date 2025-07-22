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

#include <cmath>

// Explicit instantiation of the isFinite template function for CVectorNx1<double, 2>
namespace ml {
namespace maths {
namespace common {

template<>
bool CMathsFuncs::isFinite<2UL>(const CVectorNx1<double, 2UL>& val) {
    return std::isfinite(val(0)) && std::isfinite(val(1));
}

} // namespace common
} // namespace maths
} // namespace ml

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
    const CTimeSeriesDecompositionDetail::CComponents& components,
    double meanVarianceScale
) : m_TimeShift(timeShift), m_Smoother(smoother), m_Components(components), 
    m_MeanVarianceScale(meanVarianceScale) {
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

core_t::TTime CTimeSeriesPredictor::maximumForecastInterval() const {
    return m_Components.trend().maximumForecastInterval();
}

void CTimeSeriesPredictor::forecast(core_t::TTime startTime,
                                   core_t::TTime endTime,
                                   core_t::TTime step,
                                   double confidence,
                                   double minimumScale,
                                   bool isNonNegative,
                                   const TWriteForecastResult& writer) const {
    using namespace common;
    
    if (endTime < startTime) {
        LOG_ERROR(<< "Bad forecast range: [" << startTime << "," << endTime << "]");
        return;
    }
    if (confidence < 0.0 || confidence >= 100.0) {
        LOG_ERROR(<< "Bad confidence interval: " << confidence << "%");
        return;
    }

    auto seasonal = [this, confidence](core_t::TTime time) -> TVector2x1 {
        TVector2x1 prediction{0.0};
        for (const auto& component : m_Components.seasonal()) {
            if (component.initialized() && component.time().inWindow(time)) {
                prediction += component.value(time, confidence);
            }
        }
        for (const auto& component : m_Components.calendar()) {
            if (component.initialized() && component.feature().inWindow(time)) {
                prediction += component.value(time, confidence);
            }
        }
        return prediction;
    };

    startTime += m_TimeShift;
    endTime += m_TimeShift;
    endTime = startTime + CIntegerTools::ceil(endTime - startTime, step);

    auto forecastSeasonal = [&](core_t::TTime time) -> TDouble3Vec {
        TVector2x1 bounds{seasonal(time)};

        // Decompose the smoothing into shift plus stretch and ensure that the
        // smoothed interval between the prediction bounds remains positive length.
        TVector2x1 smoothing{this->smooth(seasonal, time, CTimeSeriesDecompositionInterface::E_Seasonal)};
        double shift{smoothing.mean()};
        double stretch{std::max(smoothing(1) - smoothing(0), bounds(0) - bounds(1))};
        bounds += TVector2x1{{shift - (stretch / 2.0), shift + (stretch / 2.0)}};

        double variance{this->meanVariance()};
        double boundsScale{std::sqrt(std::max(
            minimumScale, this->varianceScaleWeight(time, variance, 0.0, false).mean()))};
        double prediction{(bounds(0) + bounds(1)) / 2.0};
        double interval{boundsScale * (bounds(1) - bounds(0))};

        return {prediction - (interval / 2.0), prediction, prediction + (interval / 2.0)};
    };

    m_Components.trend().forecast(startTime, endTime, step, confidence,
                               isNonNegative, forecastSeasonal, writer);
}

double CTimeSeriesPredictor::meanVariance() const {
    return m_MeanVarianceScale * m_Components.meanVariance();
}

CTimeSeriesPredictor::TVector2x1
CTimeSeriesPredictor::varianceScaleWeight(core_t::TTime time,
                                       double variance,
                                       double confidence,
                                       bool smooth) const {
    if (variance <= 0.0) {
        LOG_ERROR(<< "Supplied variance is " << variance << ".");
        return TVector2x1{1.0};
    }
    double mean{this->meanVariance()};
    if (mean <= 0.0 || variance <= 0.0) {
        return TVector2x1{1.0};
    }

    time += m_TimeShift;

    double components{0.0};
    TVector2x1 scale(0.0);
    if (m_Components.usingTrendForPrediction()) {
        scale += m_Components.trend().variance(confidence);
    }
    for (const auto& component : m_Components.seasonal()) {
        if (component.initialized() && component.time().inWindow(time)) {
            scale += component.variance(time, confidence);
            components += 1.0;
        }
    }
    for (const auto& component : m_Components.calendar()) {
        if (component.initialized() && component.feature().inWindow(time)) {
            scale += component.variance(time, confidence);
            components += 1.0;
        }
    }

    double bias{std::min(2.0 * mean / variance, 1.0)};
    if (m_Components.usingTrendForPrediction()) {
        bias *= (components + 1.0) / std::max(components, 1.0);
    }
    LOG_TRACE(<< "mean = " << mean << " variance = " << variance
              << " bias = " << bias << " scale = " << scale);

    scale *= m_MeanVarianceScale / mean;
    scale = common::max(TVector2x1{1.0} + bias * (scale - TVector2x1{1.0}), TVector2x1{0.0});

    if (smooth) {
        scale += this->smooth(
            [&](core_t::TTime time_) {
                return this->varianceScaleWeight(time_ - m_TimeShift, variance,
                                               confidence, false);
            },
            time, CTimeSeriesDecompositionInterface::E_All);
    }

    // If anything overflowed just bail and don't scale.
    return common::CMathsFuncs::isFinite(scale) ? scale : TVector2x1{1.0};
}

}  // namespace time_series
}  // namespace maths
}  // namespace ml
