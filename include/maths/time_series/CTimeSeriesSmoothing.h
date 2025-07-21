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

#ifndef INCLUDED_ml_maths_time_series_CTimeSeriesSmoothing_h
#define INCLUDED_ml_maths_time_series_CTimeSeriesSmoothing_h

#include <core/CoreTypes.h>

#include <maths/time_series/CTimeSeriesDecompositionInterface.h>
#include <maths/time_series/CSeasonalTime.h>

#include <cmath>

namespace ml {
namespace maths {
namespace time_series {

//! \brief Class for smoothing discontinuities between periodic repeats
//! and partitions in time series data.
//!
//! DESCRIPTION:\n
//! This class provides functionality to apply smoothing to discontinuities
//! in time series data, particularly at the boundaries of periodic windows.
//! It calculates correction values to ensure smooth transitions between
//! different segments of time series data, which helps prevent abrupt jumps
//! or discontinuities in predictions that can occur at window boundaries.
//!
//! The smoothing is implemented by calculating a weighted correction based on the
//! difference between values just before and just after a discontinuity, and the
//! proximity of the current time point to that discontinuity. The correction is
//! then applied to the predicted value to create a smooth transition.
//!
//! USAGE:\n
//! The smooth() method takes a function object that provides values at specific times,
//! along with the time point, component flags, and a collection of seasonal components.
//! It returns a correction value that should be applied to create a smooth transition.
//!
//! This class is typically used by the CTimeSeriesPredictor to smooth predictions
//! at boundaries between periodic windows. The smoothingInterval parameter controls
//! how far from a discontinuity the smoothing effect extends.
class CTimeSeriesSmoothing {
public:

    //! Initialize with the specified smoothing interval.
    //!
    //! The smoothing interval determines how far from a discontinuity the smoothing effect extends.
    //! Larger values will create wider, more gradual transitions at discontinuities,
    //! while smaller values will create narrower, more abrupt transitions.
    //!
    //! \param[in] smoothingInterval The time interval over which to apply smoothing (in seconds).
    //!                             Defaults to the predefined SMOOTHING_INTERVAL constant.
    explicit CTimeSeriesSmoothing(core_t::TTime smoothingInterval = SMOOTHING_INTERVAL);

    //! Calculate a correction value to produce a smooth join between periodic
    //! repeats and partitions in time series data.
    //!
    //! This method examines the collection of seasonal components to identify any
    //! boundaries or discontinuities near the specified time point. If a discontinuity
    //! is found within the smoothing interval, it calculates a weighted correction
    //! based on the proximity to the discontinuity and the difference in values
    //! just before and after the boundary.
    //!
    //! The correction diminishes linearly with distance from the discontinuity, reaching
    //! zero at the edge of the smoothing interval. This creates a gradual transition
    //! rather than an abrupt jump at seasonal boundaries.
    //!
    //! \param[in] f A function object that provides values at specific time points.
    //!             Must accept a core_t::TTime parameter and return a numeric value.
    //! \param[in] time The time point for which to calculate the smoothing correction.
    //! \param[in] components Flags indicating which components to include in the smoothing.
    //! \param[in] seasonalComponents Collection of seasonal components to check for discontinuities.
    //! \return A correction value of the same type as returned by the function object.
    template<typename F, typename TSeasonalComponentVec>
    auto smooth(const F& f,
                core_t::TTime time,
                int components,
                const TSeasonalComponentVec& seasonalComponents) const
        -> decltype(f(time)) {
        
        using TResultType = decltype(f(time));

        // E_Seasonal is defined in CTimeSeriesDecomposition.h, so we need to check
        // if the components include seasonal components
        if ((components & CTimeSeriesDecompositionInterface::E_Seasonal) != CTimeSeriesDecompositionInterface::E_Seasonal) {
            return TResultType{0.0};
        }

        auto offset = [&f, time, this](core_t::TTime discontinuity) {
            auto baselineMinusEps = f(discontinuity - 1);
            auto baselinePlusEps = f(discontinuity + 1);
            double weight = std::max((1.0 - (static_cast<double>(std::abs(time - discontinuity)) /
                                    static_cast<double>(m_SmoothingInterval))),
                             0.0);
            return 0.5 * weight * (baselinePlusEps - baselineMinusEps);
        };

        for (const auto& component : seasonalComponents) {
            if (component.initialized() == false ||
                component.time().windowRepeat() <= m_SmoothingInterval) {
                continue;
            }

            const CSeasonalTime& times{component.time()};

            bool timeInWindow{times.inWindow(time)};
            bool inWindowBefore{times.inWindow(time - m_SmoothingInterval)};
            bool inWindowAfter{times.inWindow(time + m_SmoothingInterval)};
            if (timeInWindow == false && inWindowBefore) {
                core_t::TTime discontinuity{times.startOfWindow(time - m_SmoothingInterval) +
                                            times.windowLength()};
                return -offset(discontinuity);
            }
            if (timeInWindow == false && inWindowAfter) {
                core_t::TTime discontinuity{component.time().startOfWindow(time + m_SmoothingInterval)};
                return offset(discontinuity);
            }
        }

        return TResultType{0.0};
    }

    //! Get the current smoothing interval.
    //!
    //! \return The time interval (in seconds) over which smoothing is applied.
    const core_t::TTime& smoothingInterval() const;

private:
    //! The default time interval over which discontinuities between different
    //! time periods (like weekdays/weekends or day/night transitions) are smoothed out.
    //! 
    //! This constant defines the default range around a discontinuity where
    //! smoothing is applied. The value is 14400 seconds (4 hours).
    static const core_t::TTime SMOOTHING_INTERVAL;

private:
    //! The configured time interval over which discontinuities between
    //! different time windows are smoothed out.
    //!
    //! This is the actual interval used for smoothing calculations, which may
    //! be different from the default if specified in the constructor.
    core_t::TTime m_SmoothingInterval;
};

}
}
}

#endif // INCLUDED_ml_maths_time_series_CTimeSeriesSmoothing_h
