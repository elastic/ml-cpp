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

#include <maths/time_series/CTimeSeriesDecompositionStub.h>

#include <core/CMemoryCircuitBreaker.h>

#include <maths/time_series/CCalendarComponent.h>
#include <maths/time_series/CSeasonalComponent.h>
#include <maths/time_series/CSeasonalTime.h>

namespace ml {
namespace maths {
namespace time_series {
namespace {
const maths_t::TSeasonalComponentVec NO_SEASONAL_COMPONENTS;
const maths_t::TCalendarComponentVec NO_CALENDAR_COMPONENTS;
}

CTimeSeriesDecompositionStub* CTimeSeriesDecompositionStub::clone(bool /*isForForecast*/) const {
    return new CTimeSeriesDecompositionStub(*this);
}

void CTimeSeriesDecompositionStub::dataType(maths_t::EDataType /*dataType*/) {
}

void CTimeSeriesDecompositionStub::decayRate(double /*decayRate*/) {
}

double CTimeSeriesDecompositionStub::decayRate() const {
    return 0.0;
}

bool CTimeSeriesDecompositionStub::initialized() const {
    return false;
}

void CTimeSeriesDecompositionStub::addPoint(
    core_t::TTime /*time*/,
    double /*value*/,
    const core::CMemoryCircuitBreaker& /*allocator*/,
    const maths_t::TDoubleWeightsAry& /*weights*/,
    const TComponentChangeCallback& /*componentChangeCallback*/,
    const maths_t::TModelAnnotationCallback& /*modelAnnotationCallback*/,
    double /*occupancy*/,
    core_t::TTime /*firstValueTime*/) {
}

void CTimeSeriesDecompositionStub::shiftTime(core_t::TTime /*time*/, core_t::TTime /*shift*/) {
}

void CTimeSeriesDecompositionStub::propagateForwardsTo(core_t::TTime /*time*/) {
}

double CTimeSeriesDecompositionStub::meanValue(core_t::TTime /*time*/) const {
    return 0.0;
}

CTimeSeriesDecompositionStub::TVector2x1
CTimeSeriesDecompositionStub::value(core_t::TTime /*time*/,
                                    double /*confidence*/,
                                    bool /*isNonNegative*/) const {
    return TVector2x1{0.0};
}

core_t::TTime CTimeSeriesDecompositionStub::maximumForecastInterval() const {
    return 0;
}

void CTimeSeriesDecompositionStub::forecast(core_t::TTime /*startTime*/,
                                            core_t::TTime /*endTime*/,
                                            core_t::TTime /*step*/,
                                            double /*confidence*/,
                                            double /*minimumScale*/,
                                            bool /*isNonNegative*/,
                                            const TWriteForecastResult& /*writer*/) {
}

double CTimeSeriesDecompositionStub::detrend(core_t::TTime /*time*/,
                                             double value,
                                             double /*confidence*/,
                                             bool /*isNonNegative*/,
                                             core_t::TTime /*maximumTimeShift*/) const {
    return value;
}

double CTimeSeriesDecompositionStub::meanVariance() const {
    return 0.0;
}

CTimeSeriesDecompositionStub::TVector2x1
CTimeSeriesDecompositionStub::varianceScaleWeight(core_t::TTime /*time*/,
                                                  double /*variance*/,
                                                  double /*confidence*/) const {
    return TVector2x1{1.0};
}

double CTimeSeriesDecompositionStub::countWeight(core_t::TTime /*time*/) const {
    return 1.0;
}

double CTimeSeriesDecompositionStub::outlierWeightDerate(core_t::TTime /*time*/,
                                                         double /*error*/) const {
    return 0.0;
}

CTimeSeriesDecompositionStub::TFloatMeanAccumulatorVec
CTimeSeriesDecompositionStub::residuals(bool /*isNonNegative*/) const {
    return {};
}

void CTimeSeriesDecompositionStub::skipTime(core_t::TTime /*skipInterval*/) {
}

std::uint64_t CTimeSeriesDecompositionStub::checksum(std::uint64_t seed) const {
    return seed;
}

void CTimeSeriesDecompositionStub::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CTimeSeriesDecompositionStub");
}

std::size_t CTimeSeriesDecompositionStub::memoryUsage() const {
    return 0;
}

std::size_t CTimeSeriesDecompositionStub::staticSize() const {
    return sizeof(*this);
}

core_t::TTime CTimeSeriesDecompositionStub::timeShift() const {
    return 0;
}

const maths_t::TSeasonalComponentVec& CTimeSeriesDecompositionStub::seasonalComponents() const {
    return NO_SEASONAL_COMPONENTS;
}

const maths_t::TCalendarComponentVec& CTimeSeriesDecompositionStub::calendarComponents() const {
    return NO_CALENDAR_COMPONENTS;
}
}
}
}
