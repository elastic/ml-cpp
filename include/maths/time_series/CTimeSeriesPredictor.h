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

#ifndef INCLUDED_ml_maths_time_series_CTimeSeriesPredictor_h
#define INCLUDED_ml_maths_time_series_CTimeSeriesPredictor_h

#include "core/CoreTypes.h"
#include <core/CMemoryUsage.h>
#include <core/WindowsSafe.h>

#include <maths/common/CLinearAlgebra.h>
#include <maths/common/MathsTypes.h>
#include <maths/time_series/CSeasonalComponent.h>
#include <maths/time_series/CCalendarComponent.h>

#include <maths/time_series/ImportExport.h>
#include <maths/time_series/CTimeSeriesDecompositionInterface.h>
#include <maths/time_series/CTimeSeriesSmoothing.h>
#include <maths/time_series/CTimeSeriesDecompositionDetail.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace ml {
namespace maths {
namespace time_series {

//! \brief Makes predictions based on time series decomposition components.
//!
//! DESCRIPTION:\n
//! This class encapsulates the prediction functionality for time series decomposition.
//! It works with various components (trend, seasonal, calendar) to generate predictions
//! for time series data. It also handles component smoothing at discontinuities
//! to provide more accurate forecasts.
//!
//! The predictor supports different modes of operation by allowing selection of which
//! components to include in predictions via flags. It can also apply smoothing at
//! seasonal boundaries to avoid discontinuities in the forecast.
//!
//! The main functionalities are:
//! - Predicting time series values based on decomposition components
//! - Creating predictor functions for efficient repeated predictions
//! - Smoothing seasonal component boundaries to avoid discontinuities
//! - Selectively including/excluding components in predictions
//!
//! This class is responsible for making predictions using the components
//! from a time series decomposition. It separates the prediction logic from
//! the decomposition implementation, allowing for a cleaner separation of concerns.
//! The class operates on functions that provide access to trend, seasonal, and
//! calendar components, rather than accessing these components directly.
//!
//! The main functionality provided includes:
//! - Getting predicted values at a given time point with the value() method
//! - Creating efficient predictor functions for repeated predictions with the predictor() method
//! - Optional smoothing at periodic boundaries to reduce discontinuities
//!
//! USAGE:\n
//! The class is designed to work with a function-based interface where component
//! access is provided through function objects. This allows greater flexibility
//! and decoupling from the specific implementation details of components.
//!
//! Typically, this class is created and managed by CTimeSeriesDecomposition, which
//! provides the necessary function objects for accessing its internal components.
//! The class can also be used independently with custom component providers.
class MATHS_TIME_SERIES_EXPORT CTimeSeriesPredictor {
public:
    using TVector2x1 = common::CVectorNx1<double, 2>;
    using TFloatMeanAccumulatorVec = std::vector<common::CFloatStorage>;
    using TBoolVec = std::vector<bool>;
    using TFilteredPredictor = std::function<double(core_t::TTime, const TBoolVec&)>;
    
public:
    //! Constructor for prediction functionality.
    //! \param[in] timeShift The time shift to apply.
    //! \param[in] smoother The smoothing object to use at boundaries.
    //! \param[in] trendValueFunc Function to get trend component value.
    //! \param[in] trendPredictorFunc Function to get trend predictor.
    //! \param[in] usingTrendForPredictionFunc Function to check if using trend for prediction.
    //! \param[in] seasonalComponentsFunc Function to get seasonal components.
    //! \param[in] calendarComponentsFunc Function to get calendar components.
    // CTimeSeriesPredictor(
    //     core_t::TTime timeShift,
    //     const CTimeSeriesSmoothing& smoother,
    //     std::function<TVector2x1(core_t::TTime, double)> trendValueFunc,
    //     std::function<std::function<double(core_t::TTime)>()> trendPredictorFunc,
    //     std::function<bool()> usingTrendForPredictionFunc,
    //     std::function<const maths_t::TSeasonalComponentVec&()> seasonalComponentsFunc,
    //     std::function<const maths_t::TCalendarComponentVec&()> calendarComponentsFunc);

    //! Constructor for prediction functionality using CComponents reference.
    //! This approach provides direct access to components rather than using function objects.
    //!
    //! \param[in] timeShift The time shift to apply to predictions (used to align components).
    //! \param[in] smoother The smoothing object to use at boundaries to reduce discontinuities.
    //! \param[in] components Reference to the decomposition components used for predictions.
    CTimeSeriesPredictor(
        core_t::TTime timeShift,
        const CTimeSeriesSmoothing& smoother,
        const CTimeSeriesDecompositionDetail::CComponents& components
    );

    //! Get the predicted value of the time series at \p time.
    //!
    //! This method calculates the predicted value at the specified time point
    //! by combining the selected components (trend, seasonal, calendar) according
    //! to the components flag. It can optionally apply smoothing at periodic boundaries.
    //!
    //! \param[in] time The time point for which to make a prediction.
    //! \param[in] confidence The symmetric confidence interval for the prediction
    //!                     as a percentage (used for variance calculations).
    //! \param[in] components Flags indicating which components to include in the prediction
    //!                      (see CTimeSeriesDecompositionInterface::EComponents).
    //! \param[in] smooth Whether to apply smoothing at periodic boundaries to reduce discontinuities.
    //! \return A 2D vector containing the predicted value and variance.
    TVector2x1 value(core_t::TTime time, double confidence, int components, bool smooth) const;

    //! Get a function which returns the decomposition value as a function of time.
    //!
    //! This method creates a function object that can efficiently generate predictions
    //! at different time points. It caches the expensive part of the calculation,
    //! making it much faster than repeatedly calling the value() method.
    //!
    //! The returned function takes a time parameter and an optional vector of booleans
    //! that can be used to selectively exclude individual seasonal components.
    //!
    //! \param[in] components Flags indicating which components to include in the prediction
    //!                      (see CTimeSeriesDecompositionInterface::EComponents).
    //! \return A function object that generates predictions for given time points.
    //! \warning This can only be used as long as the component models aren't updated.
    TFilteredPredictor predictor(int components) const;

private:
    //! Apply smoothing to the prediction at periodic boundaries.
    //!
    //! This method uses the provided smoother to reduce discontinuities at the boundaries
    //! of periodic components. It works by identifying seasonal components that are near
    //! their boundaries and applying a smoothing correction.
    //!
    //! \param[in] f Function that produces a basic prediction without smoothing.
    //! \param[in] time The time point at which to evaluate the prediction.
    //! \param[in] components Flags indicating which components to include in the prediction.
    //! \return A smoothed prediction vector (value and variance).
    template<typename F>
    TVector2x1 smooth(const F& f, core_t::TTime time, int components) const;

private:
    //! The time shift to apply
    core_t::TTime m_TimeShift;

    //! The smoothing object
    const CTimeSeriesSmoothing& m_Smoother;

    const CTimeSeriesDecompositionDetail::CComponents& m_Components;
    
    // //! Function to get trend component value
    // std::function<TVector2x1(core_t::TTime, double)> m_TrendValueFunc;
    
    // //! Function to get trend predictor
    // std::function<std::function<double(core_t::TTime)>()> m_TrendPredictorFunc;
    
    // //! Function to check if using trend for prediction
    // std::function<bool()> m_UsingTrendForPredictionFunc;
    
    // //! Function to get seasonal components
    // std::function<const maths_t::TSeasonalComponentVec&()> m_SeasonalComponentsFunc;
    
    // //! Function to get calendar components
    // std::function<const maths_t::TCalendarComponentVec&()> m_CalendarComponentsFunc;
};

}  // namespace time_series
}  // namespace maths
}  // namespace ml

#endif  // INCLUDED_ml_maths_time_series_CTimeSeriesPredictor_h
