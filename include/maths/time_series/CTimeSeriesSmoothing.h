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
//! different segments of time series data.
class CTimeSeriesSmoothing {
public:

    //! Initialize with smoothing interval
    explicit CTimeSeriesSmoothing(core_t::TTime smoothingInterval = SMOOTHING_INTERVAL);

    //! The correction to produce a smooth join between periodic
    //! repeats and partitions.
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

    const core_t::TTime& smoothingInterval() const;

private:
    //! The time over which discontinuities between weekdays
    //! and weekends are smoothed out.
    static const core_t::TTime SMOOTHING_INTERVAL;

private:
    //! The time over which discontinuities between windows
    //! are smoothed out.
    core_t::TTime m_SmoothingInterval;
};

}
}
}

#endif // INCLUDED_ml_maths_time_series_CTimeSeriesSmoothing_h
