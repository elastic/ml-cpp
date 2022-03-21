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

#ifndef INCLUDED_ml_maths_time_series_CCalendarComponent_h
#define INCLUDED_ml_maths_time_series_CCalendarComponent_h

#include <core/CMemory.h>
#include <core/CoreTypes.h>

#include <maths/time_series/CCalendarComponentAdaptiveBucketing.h>
#include <maths/time_series/CDecompositionComponent.h>
#include <maths/time_series/ImportExport.h>

#include <cstddef>
#include <string>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {
namespace time_series {

//! \brief Estimates a calendar component of a time series.
//!
//! DESCRIPTION:\n
//! This uses an adaptive bucketing strategy to compute the mean value and variance of a
//! function in various subintervals of its calendar feature window.
//!
//! The intervals are adjusted to minimize the maximum averaging error in any bucket (see
//! CCalendarComponentAdaptiveBucketing for more details). Estimates of the true function
//! values are obtained by interpolating the bucket values (using cubic spline).
//!
//! The bucketing is aged by relaxing it back towards uniform and aging the counts of the
//! mean value for each bucket as usual.
class MATHS_TIME_SERIES_EXPORT CCalendarComponent : private CDecompositionComponent {
public:
    //! \param[in] feature The calendar feature.
    //! \param[in] maxSize The maximum number of component buckets.
    //! \param[in] decayRate Controls the rate at which information is lost from
    //! its adaptive bucketing.
    //! \param[in] minBucketLength The minimum bucket length permitted in the
    //! adaptive bucketing.
    //! \param[in] boundaryCondition The boundary condition to use for the splines.
    //! \param[in] valueInterpolationType The style of interpolation to use for
    //! computing values.
    //! \param[in] varianceInterpolationType The style of interpolation to use for
    //! computing variances.
    CCalendarComponent(
        const CCalendarFeature& feature,
        core_t::TTime timeZoneOffset,
        std::size_t maxSize,
        double decayRate = 0.0,
        double minBucketLength = 0.0,
        common::CSplineTypes::EBoundaryCondition boundaryCondition = common::CSplineTypes::E_Periodic,
        common::CSplineTypes::EType valueInterpolationType = common::CSplineTypes::E_Cubic,
        common::CSplineTypes::EType varianceInterpolationType = common::CSplineTypes::E_Linear);

    //! Construct by traversing part of an state document.
    CCalendarComponent(double decayRate,
                       double minBucketLength,
                       core::CStateRestoreTraverser& traverser,
                       common::CSplineTypes::EType valueInterpolationType = common::CSplineTypes::E_Cubic,
                       common::CSplineTypes::EType varianceInterpolationType = common::CSplineTypes::E_Linear);

    //! An efficient swap of the contents of two components.
    void swap(CCalendarComponent& other);

    //! Persist state by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Check if the component has been estimated.
    bool initialized() const;

    //! Initialize the adaptive bucketing.
    void initialize();

    //! Get the size of this component.
    std::size_t size() const;

    //! Clear all data.
    void clear();

    //! Linearly scale the component's by \p scale.
    void linearScale(core_t::TTime time, double scale);

    //! Adds a value \f$(t, f(t))\f$ to this component.
    //!
    //! \param[in] time The time of the point.
    //! \param[in] value The value at \p time.
    //! \param[in] weight The weight of \p value. The smaller this is the
    //! less influence it has on the component.
    void add(core_t::TTime time, double value, double weight = 1.0);

    //! Check whether to reinterpolate the component predictions.
    bool shouldInterpolate(core_t::TTime time) const;

    //! Update the interpolation of the bucket values.
    //!
    //! \param[in] time The time at which to interpolate.
    //! \param[in] refine If false disable refining the bucketing.
    void interpolate(core_t::TTime time, bool refine = true);

    //! Get the rate at which the seasonal component loses information.
    double decayRate() const;

    //! Set the rate at which the seasonal component loses information.
    void decayRate(double decayRate);

    //! Age out old data to account for elapsed \p time.
    void propagateForwardsByTime(double time);

    //! Get the calendar feature.
    CCalendarFeatureAndTZ feature() const;

    //! Interpolate the component at \p time.
    //!
    //! \param[in] time The time of interest.
    //! \param[in] confidence The symmetric confidence interval for the variance
    //! as a percentage.
    TDoubleDoublePr value(core_t::TTime time, double confidence) const;

    //! Get the mean value of the component.
    double meanValue() const;

    //! Get the variance of the residual about the prediction at \p time.
    //!
    //! \param[in] time The time of interest.
    //! \param[in] confidence The symmetric confidence interval for the
    //! variance as a percentage.
    TDoubleDoublePr variance(core_t::TTime time, double confidence) const;

    //! Get the mean variance of the component residuals.
    double meanVariance() const;

    //! Get a checksum for this object.
    uint64_t checksum(uint64_t seed = 0) const;

    //! Debug the memory used by this component.
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const;

    //! Get the memory used by this component.
    std::size_t memoryUsage() const;

    //! Check that the state is valid.
    bool isBad() const;

private:
    //! Create by traversing a state document.
    bool acceptRestoreTraverser(double decayRate,
                                double minimumBucketLength,
                                core::CStateRestoreTraverser& traverser);

private:
    //! The mean and variance in collection of buckets covering the period.
    CCalendarComponentAdaptiveBucketing m_Bucketing;

    //! The last interpolation time.
    core_t::TTime m_LastInterpolationTime;
};

//! Create a free function which will be found by Koenig lookup.
inline void swap(CCalendarComponent& lhs, CCalendarComponent& rhs) {
    lhs.swap(rhs);
}
}
}
}

#endif // INCLUDED_ml_maths_time_series_CCalendarComponent_h
