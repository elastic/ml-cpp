/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CTimeSeriesDecompositionStub.h>

#include <maths/CSeasonalComponent.h>

namespace ml
{
namespace maths
{
namespace
{
const maths_t::TSeasonalComponentVec NO_COMPONENTS;
}

CTimeSeriesDecompositionStub *CTimeSeriesDecompositionStub::clone(void) const
{
    return new CTimeSeriesDecompositionStub(*this);
}

void CTimeSeriesDecompositionStub::decayRate(double /*decayRate*/)
{
}

double CTimeSeriesDecompositionStub::decayRate(void) const
{
    return 0.0;
}

bool CTimeSeriesDecompositionStub::initialized(void) const
{
    return false;
}

bool CTimeSeriesDecompositionStub::addPoint(core_t::TTime /*time*/,
                                            double /*value*/,
                                            const maths_t::TWeightStyleVec &/*weightStyles*/,
                                            const maths_t::TDouble4Vec &/*weights*/)
{
    return false;
}

void CTimeSeriesDecompositionStub::propagateForwardsTo(core_t::TTime /*time*/)
{
}

double CTimeSeriesDecompositionStub::meanValue(core_t::TTime /*time*/) const
{
    return 0.0;
}

maths_t::TDoubleDoublePr CTimeSeriesDecompositionStub::value(core_t::TTime /*time*/,
                                                             double /*confidence*/,
                                                             int /*components*/,
                                                             bool /*smooth*/) const
{
    return {0.0, 0.0};
}

void CTimeSeriesDecompositionStub::forecast(core_t::TTime /*startTime*/,
                                            core_t::TTime /*endTime*/,
                                            core_t::TTime /*step*/,
                                            double /*confidence*/,
                                            double /*minimumScale*/,
                                            TDouble3VecVec &result)
{
    result.clear();
}

double CTimeSeriesDecompositionStub::detrend(core_t::TTime /*time*/,
                                             double value,
                                             double /*confidence*/) const
{
    return value;
}

double CTimeSeriesDecompositionStub::meanVariance(void) const
{
    return 0.0;
}

maths_t::TDoubleDoublePr CTimeSeriesDecompositionStub::scale(core_t::TTime /*time*/,
                                                             double /*variance*/,
                                                             double /*confidence*/,
                                                             bool /*smooth*/) const
{
    return {1.0, 1.0};
}

void CTimeSeriesDecompositionStub::skipTime(core_t::TTime /*skipInterval*/)
{
}

uint64_t CTimeSeriesDecompositionStub::checksum(uint64_t seed) const
{
    return seed;
}

void CTimeSeriesDecompositionStub::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    mem->setName("CTimeSeriesDecompositionStub");
}

std::size_t CTimeSeriesDecompositionStub::memoryUsage(void) const
{
    return 0;
}

std::size_t CTimeSeriesDecompositionStub::staticSize(void) const
{
    return sizeof(*this);
}

const maths_t::TSeasonalComponentVec &CTimeSeriesDecompositionStub::seasonalComponents(void) const
{
    return NO_COMPONENTS;
}

core_t::TTime CTimeSeriesDecompositionStub::lastValueTime(void) const
{
    return 0;
}

}
}
