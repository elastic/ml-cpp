/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CTimeSeriesDecompositionStub.h>

#include <maths/CSeasonalComponent.h>
#include <maths/CSeasonalTime.h>

namespace ml {
namespace maths {
namespace {
const maths_t::TSeasonalComponentVec NO_COMPONENTS;
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

bool CTimeSeriesDecompositionStub::addPoint(core_t::TTime /*time*/,
                                            double /*value*/,
                                            const maths_t::TDoubleWeightsAry& /*weights*/) {
    return false;
}

bool CTimeSeriesDecompositionStub::applyChange(core_t::TTime /*time*/,
                                               double /*value*/,
                                               const SChangeDescription& /*change*/) {
    return false;
}

void CTimeSeriesDecompositionStub::propagateForwardsTo(core_t::TTime /*time*/) {
}

double CTimeSeriesDecompositionStub::meanValue(core_t::TTime /*time*/) const {
    return 0.0;
}

maths_t::TDoubleDoublePr CTimeSeriesDecompositionStub::value(core_t::TTime /*time*/,
                                                             double /*confidence*/,
                                                             int /*components*/,
                                                             bool /*smooth*/) const {
    return {0.0, 0.0};
}

void CTimeSeriesDecompositionStub::forecast(core_t::TTime /*startTime*/,
                                            core_t::TTime /*endTime*/,
                                            core_t::TTime /*step*/,
                                            double /*confidence*/,
                                            double /*minimumScale*/,
                                            const TWriteForecastResult& /*writer*/) {
}

double CTimeSeriesDecompositionStub::detrend(core_t::TTime /*time*/,
                                             double value,
                                             double /*confidence*/,
                                             int /*components*/) const {
    return value;
}

double CTimeSeriesDecompositionStub::meanVariance() const {
    return 0.0;
}

maths_t::TDoubleDoublePr CTimeSeriesDecompositionStub::scale(core_t::TTime /*time*/,
                                                             double /*variance*/,
                                                             double /*confidence*/,
                                                             bool /*smooth*/) const {
    return {1.0, 1.0};
}

bool CTimeSeriesDecompositionStub::mightAddComponents(core_t::TTime /*time*/) const {
    return false;
}

CTimeSeriesDecompositionStub::TTimeDoublePrVec
CTimeSeriesDecompositionStub::windowValues() const {
    return {};
}

void CTimeSeriesDecompositionStub::skipTime(core_t::TTime /*skipInterval*/) {
}

uint64_t CTimeSeriesDecompositionStub::checksum(uint64_t seed) const {
    return seed;
}

void CTimeSeriesDecompositionStub::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
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
    return NO_COMPONENTS;
}

core_t::TTime CTimeSeriesDecompositionStub::lastValueTime() const {
    return 0;
}
}
}
