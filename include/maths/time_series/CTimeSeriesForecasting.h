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

#ifndef INCLUDED_ml_maths_time_series_CTimeSeriesForecasting_h
#define INCLUDED_ml_maths_time_series_CTimeSeriesForecasting_h

#include <core/CoreTypes.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CLinearAlgebra.h>

#include <maths/time_series/ImportExport.h>
#include <maths/time_series/CTimeSeriesDecompositionInterface.h>
#include <maths/time_series/CTimeSeriesDecompositionDetail.h>
#include <maths/time_series/CTimeSeriesSmoothing.h>

#include <functional>
#include <memory>
#include <vector>

namespace ml {
namespace maths {
namespace time_series {

//! \brief Handles forecasting for time series decomposition.
//!
//! DESCRIPTION:\n
//! This class encapsulates the forecasting functionality for time series decomposition.
//! It provides methods to generate forecasts from time series components (trend, seasonal,
//! calendar), handling confidence intervals, smoothing at boundaries, and prediction
//! variance scaling.
//!
//! The forecasting process combines predictions from each of the time series components:
//! - Trend component: provides the baseline prediction
//! - Seasonal components: add periodic patterns to the prediction
//! - Calendar components: add special calendar-based effects
//!
//! At each forecast point, the class computes a prediction with confidence bounds
//! and applies appropriate smoothing at periodic boundaries to ensure continuity.
//! It also handles variance scaling based on the historical variability of the data.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This class was extracted from CTimeSeriesDecomposition to improve separation of concerns.
//! It focuses exclusively on forecast-related functionality while delegating other
//! aspects of time series analysis to more specialized classes.
//! The class is responsible for coordinating forecasts across different time series
//! components, combining their predictions, and applying appropriate scaling and
//! smoothing to ensure realistic and consistent forecasts.
//!
//! USAGE:\n
//! This class is typically created and managed by CTimeSeriesDecomposition, which
//! provides it with access to the necessary components for generating forecasts.
//! The class can also be used independently with appropriate component references.
class MATHS_TIME_SERIES_EXPORT CTimeSeriesForecasting {
public:
    using TVector2x1 = common::CVectorNx1<double, 2>;
    using TDouble3Vec = core::CSmallVector<double, 3>;
    using TWriteForecastResult = std::function<void(core_t::TTime, const TDouble3Vec&)>;

public:
    //! Constructor for forecasting functionality.
    //!
    //! \param[in] timeShift The time shift to apply to forecasts.
    //! \param[in] smoother The smoothing object to use at boundaries.
    //! \param[in] components Reference to the decomposition components used for forecasts.
    //! \param[in] meanVarianceScale The scale factor for the mean variance.
    CTimeSeriesForecasting(
        core_t::TTime timeShift,
        const CTimeSeriesSmoothing& smoother,
        const CTimeSeriesDecompositionDetail::CComponents& components,
        double meanVarianceScale
    );

    //! Get the maximum interval for which the time series can be forecast.
    //!
    //! \return The maximum forecast interval in seconds.
    core_t::TTime maximumForecastInterval() const;

    //! Forecast from \p start to \p end at \p dt intervals.
    //!
    //! This method generates a forecast for the time series from the specified
    //! start time to end time, at regular intervals defined by the step parameter.
    //! It combines predictions from all relevant components (trend, seasonal, calendar)
    //! and applies appropriate scaling and smoothing.
    //!
    //! \param[in] startTime The start of the forecast period.
    //! \param[in] endTime The end of the forecast period.
    //! \param[in] step The time increment between forecast points.
    //! \param[in] confidence The forecast confidence interval as a percentage.
    //! \param[in] minimumScale The minimum permitted seasonal scale factor.
    //! \param[in] isNonNegative True if the data being modelled are known to be non-negative.
    //! \param[in] writer Callback function to receive forecast results.
    void forecast(core_t::TTime startTime,
                  core_t::TTime endTime,
                  core_t::TTime step,
                  double confidence,
                  double minimumScale,
                  bool isNonNegative,
                  const TWriteForecastResult& writer);

private:
    //! Apply smoothing to the prediction at periodic boundaries.
    //!
    //! This method handles the discontinuities that can occur at periodic boundaries
    //! by applying appropriate smoothing to ensure forecast continuity. It uses the
    //! smoothing object to calculate corrections at these boundaries.
    //!
    //! \param[in] f A function that returns a prediction for a given time.
    //! \param[in] time The time at which to compute the smoothed prediction.
    //! \param[in] components Bitmask indicating which components to include.
    //! \return A vector containing the smoothing correction to apply.
    template<typename F>
    TVector2x1 smooth(const F& f, core_t::TTime time, int components) const;

    //! Calculate the variance scale weight for predictions.
    //!
    //! This method computes the scaling factor to apply to prediction variances,
    //! which helps determine the width of confidence intervals.
    //!
    //! \param[in] time The time for which to compute the scale weight.
    //! \param[in] variance The baseline variance value to scale.
    //! \param[in] confidence The confidence level as a percentage.
    //! \param[in] smooth Whether to apply smoothing to the scale weights.
    //! \return A 2D vector containing the lower and upper scale weights.
    TVector2x1 varianceScaleWeight(core_t::TTime time,
                                   double variance,
                                   double confidence,
                                   bool smooth) const;

    //! Get the mean variance of the baseline.
    //!
    //! This method returns the mean variance value used for scaling predictions.
    //! It represents the typical variability in the time series.
    //!
    //! \return The mean variance value for the time series.
    double meanVariance() const;

private:
    //! The time shift to apply to forecasts
    core_t::TTime m_TimeShift;

    //! The smoothing object for handling discontinuities
    const CTimeSeriesSmoothing& m_Smoother;

    //! Reference to the decomposition components
    const CTimeSeriesDecompositionDetail::CComponents& m_Components;

    //! Scale factor for the mean variance
    double m_MeanVarianceScale;
};

} // namespace time_series
} // namespace maths
} // namespace ml

#endif // INCLUDED_ml_maths_time_series_CTimeSeriesForecasting_h
