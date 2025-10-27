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

#include <maths/time_series/CTimeSeriesSmoother.h>

#include <core/CMemoryDef.h>

#include <maths/common/CLinearAlgebra.h>
#include <maths/common/CMathsFuncsForMatrixAndVectorTypes.h>
#include <maths/time_series/CTimeSeriesDecompositionDetail.h>
#include <maths/time_series/CSeasonalTime.h>

#include <cmath>

namespace ml {
namespace maths {
namespace time_series {

namespace {
// Default smoothing interval (4 hours = 14400 seconds)
const core_t::TTime DEFAULT_SMOOTHING_INTERVAL{14400};
}

CTimeSeriesSmoother::CTimeSeriesSmoother()
    : m_SmoothingInterval{DEFAULT_SMOOTHING_INTERVAL} {
}

CTimeSeriesSmoother::CTimeSeriesSmoother(core_t::TTime smoothingInterval)
    : m_SmoothingInterval{smoothingInterval} {
}

core_t::TTime CTimeSeriesSmoother::smoothingInterval() const {
    return m_SmoothingInterval;
}

void CTimeSeriesSmoother::smoothingInterval(core_t::TTime interval) {
    m_SmoothingInterval = interval;
}

bool CTimeSeriesSmoother::shouldSmooth(core_t::TTime time) const {
    // Check if we're within the smoothing interval of a weekend/weekday boundary
    return CTimeSeriesDecompositionDetail::CSeasonalTime::isWithinBoundary(
        time, m_SmoothingInterval);
}

double CTimeSeriesSmoother::smoothingWeight(core_t::TTime time) const {
    if (!this->shouldSmooth(time)) {
        return 1.0;
    }
    
    core_t::TTime boundary{CTimeSeriesDecompositionDetail::CSeasonalTime::boundaryTime(time)};
    core_t::TTime dt{std::abs(time - boundary)};
    
    // Calculate linear weight based on distance from boundary
    return static_cast<double>(dt) / static_cast<double>(m_SmoothingInterval);
}

template<typename F>
auto CTimeSeriesSmoother::smooth(const F& f, core_t::TTime time) const -> decltype(f(time)) {
    using TResult = decltype(f(time));
    
    // If we're not near a boundary, no smoothing needed
    if (!this->shouldSmooth(time)) {
        return TResult{0};
    }
    
    // Calculate boundary time and distance
    core_t::TTime boundary{CTimeSeriesDecompositionDetail::CSeasonalTime::boundaryTime(time)};
    core_t::TTime dt{std::abs(time - boundary)};
    double weight{static_cast<double>(dt) / static_cast<double>(m_SmoothingInterval)};
    
    // Get values at current time and reflected time
    TResult forTime{f(time)};
    
    // Reflect across boundary
    core_t::TTime reflect{(2 * boundary) - time};
    TResult forReflect{f(reflect)};

    // Linear interpolation for smooth transition
    return (weight * forTime) + ((1.0 - weight) * forReflect);
}

void ml::maths::time_series::CTimeSeriesSmoother::debugMemoryUsage(const ml::core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    // This method doesn't use any instance members, but we're keeping it as a const method for API consistency
    mem->setName("CTimeSeriesSmoother");
}

std::size_t CTimeSeriesSmoother::memoryUsage() const {
    return sizeof(*this);
}

// Explicit template instantiations for the types we need
template CTimeSeriesSmoother::TVector2x1 CTimeSeriesSmoother::smooth<CTimeSeriesSmoother::TPredictionFunc>(
    const TPredictionFunc&, core_t::TTime) const;
template double CTimeSeriesSmoother::smooth<CTimeSeriesSmoother::TDoubleFunc>(
    const TDoubleFunc&, core_t::TTime) const;

}
}
}
