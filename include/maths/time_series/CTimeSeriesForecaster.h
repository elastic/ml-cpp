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

#ifndef INCLUDED_ml_maths_time_series_CTimeSeriesForecaster_h
#define INCLUDED_ml_maths_time_series_CTimeSeriesForecaster_h

#include <core/CoreTypes.h>
#include <core/CMemoryUsage.h>

#include <maths/common/CLinearAlgebraFwd.h>
#include <maths/common/MathsTypes.h>

#include <maths/time_series/ImportExport.h>

#include <functional>
#include <vector>

namespace ml {
namespace maths {
namespace time_series {

// Forward declarations
class CTimeSeriesDecompositionInterface;
class CTrendDecomposition;
class CSeasonalDecomposition;
class CCalendarDecomposition;

//! \brief Specialized class for time series forecasting functionality
//!
//! DESCRIPTION:\n
//! This class handles all forecasting operations for time series decomposition.
//! It takes decomposition components and generates forecasts with confidence
//! intervals, separating this responsibility from the decomposition classes.
//!
//! IMPLEMENTATION:\n
//! The forecaster uses the provided decomposition components to generate
//! predictions. It handles the combination of component forecasts and
//! calculation of confidence intervals.
class MATHS_TIME_SERIES_EXPORT CTimeSeriesForecaster {
public:
    using TDouble3Vec = std::vector<double>;
    using TVector2x1 = common::CVectorNx1<double, 2>;
    using TWriteForecastResult = std::function<void(core_t::TTime, const TDouble3Vec&)>;

public:
    CTimeSeriesForecaster() = default;

    //! Constructor for a complete decomposition
    explicit CTimeSeriesForecaster(const CTimeSeriesDecompositionInterface& decomposition);

    //! Constructor for separate components
    CTimeSeriesForecaster(const CTrendDecomposition* trendDecomposition,
                          const CSeasonalDecomposition* seasonalDecomposition,
                          const CCalendarDecomposition* calendarDecomposition);

    //! Get the maximum interval for which the time series can be forecast.
    core_t::TTime maximumForecastInterval() const;

    //! Forecast from \p start to \p end at \p dt intervals.
    //!
    //! \param[in] startTime The start of the forecast.
    //! \param[in] endTime The end of the forecast.
    //! \param[in] step The time increment.
    //! \param[in] confidence The forecast confidence interval.
    //! \param[in] minimumScale The minimum permitted seasonal scale.
    //! \param[in] isNonNegative True if the data being modelled are known to be
    //! non-negative.
    //! \param[in] timeShift Any time shift to apply to the forecast times.
    //! \param[in] writer Forecast results are passed to this callback.
    void forecast(core_t::TTime startTime,
                  core_t::TTime endTime,
                  core_t::TTime step,
                  double confidence,
                  double minimumScale,
                  bool isNonNegative,
                  core_t::TTime timeShift,
                  const TWriteForecastResult& writer) const;

    //! Calculate forecast with confidence bounds at a single point in time
    //! 
    //! \param[in] time The time to forecast.
    //! \param[in] confidence The forecast confidence interval.
    //! \param[in] minimumScale The minimum permitted scale.
    //! \param[in] isNonNegative True if the data being modelled are known to be non-negative.
    //! \param[in] timeShift Any time shift to apply to the forecast time.
    //! \return A vector with [lower bound, prediction, upper bound]
    TDouble3Vec calculateForecastWithBounds(core_t::TTime time,
                                           double confidence,
                                           double minimumScale,
                                           bool isNonNegative,
                                           core_t::TTime timeShift) const;

    //! Debug the memory used by this object.
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const;

    //! Get the memory used by this object.
    std::size_t memoryUsage() const;

private:
    //! Calculate seasonal forecast with confidence interval
    TDouble3Vec calculateSeasonalForecastWithConfidenceInterval(core_t::TTime time, 
                                                              double confidence,
                                                              double minimumScale) const;

    //! Calculate calendar forecast with confidence interval
    TDouble3Vec calculateCalendarForecastWithConfidenceInterval(core_t::TTime time,
                                                              double confidence,
                                                              double minimumScale) const;

    //! Calculate trend forecast with confidence interval
    TDouble3Vec calculateTrendForecastWithConfidenceInterval(core_t::TTime time,
                                                           double confidence,
                                                           double minimumScale) const;

private:
    //! Pointer to the trend decomposition (might be null)
    const CTrendDecomposition* m_TrendDecomposition{nullptr};

    //! Pointer to the seasonal decomposition (might be null)
    const CSeasonalDecomposition* m_SeasonalDecomposition{nullptr};

    //! Pointer to the calendar decomposition (might be null)
    const CCalendarDecomposition* m_CalendarDecomposition{nullptr};

    //! Pointer to the full decomposition interface (might be null)
    const CTimeSeriesDecompositionInterface* m_Decomposition{nullptr};
};

}
}
}

#endif // INCLUDED_ml_maths_time_series_CTimeSeriesForecaster_h
