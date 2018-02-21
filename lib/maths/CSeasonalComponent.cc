/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CSeasonalComponent.h>

#include <core/CLogger.h>
#include <core/Constants.h>
#include <core/CPersistUtils.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/CChecksum.h>
#include <maths/CIntegerTools.h>
#include <maths/CRegressionDetail.h>
#include <maths/CSampling.h>
#include <maths/CSeasonalTime.h>

#include <boost/bind.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/normal.hpp>

#include <ios>
#include <vector>

namespace ml
{
namespace maths
{
namespace
{

typedef maths_t::TDoubleDoublePr TDoubleDoublePr;

const std::string DECOMPOSITION_COMPONENT_TAG{"a"};
const std::string RNG_TAG{"b"};
const std::string BUCKETING_TAG{"c"};
const std::string EMPTY_STRING;

}

CSeasonalComponent::CSeasonalComponent(const CSeasonalTime &time,
                                       std::size_t maxSize,
                                       double decayRate,
                                       double minimumBucketLength,
                                       CSplineTypes::EBoundaryCondition boundaryCondition,
                                       CSplineTypes::EType valueInterpolationType,
                                       CSplineTypes::EType varianceInterpolationType) :
        CDecompositionComponent{maxSize, boundaryCondition, valueInterpolationType, varianceInterpolationType},
        m_Bucketing{time, decayRate, minimumBucketLength}
{}

CSeasonalComponent::CSeasonalComponent(double decayRate,
                                       double minimumBucketLength,
                                       core::CStateRestoreTraverser &traverser,
                                       CSplineTypes::EType valueInterpolationType,
                                       CSplineTypes::EType varianceInterpolationType) :
        CDecompositionComponent{0, CSplineTypes::E_Periodic, valueInterpolationType, varianceInterpolationType}
{
    traverser.traverseSubLevel(boost::bind(&CSeasonalComponent::acceptRestoreTraverser,
                                           this, decayRate, minimumBucketLength, _1));
}

void CSeasonalComponent::swap(CSeasonalComponent &other)
{
    this->CDecompositionComponent::swap(other);
    std::swap(m_Rng, other.m_Rng);
    m_Bucketing.swap(other.m_Bucketing);
}

bool CSeasonalComponent::acceptRestoreTraverser(double decayRate,
                                                double minimumBucketLength,
                                                core::CStateRestoreTraverser &traverser)
{
    do
    {
        const std::string &name{traverser.name()};
        RESTORE(DECOMPOSITION_COMPONENT_TAG,
                traverser.traverseSubLevel(boost::bind(&CDecompositionComponent::acceptRestoreTraverser,
                                                       static_cast<CDecompositionComponent*>(this), _1)))
        RESTORE(RNG_TAG, m_Rng.fromString(traverser.value()))
        RESTORE_SETUP_TEARDOWN(BUCKETING_TAG,
                               CSeasonalComponentAdaptiveBucketing bucketing(decayRate, minimumBucketLength, traverser),
                               true,
                               m_Bucketing.swap(bucketing))
    }
    while (traverser.next());

    return true;
}

void CSeasonalComponent::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertLevel(DECOMPOSITION_COMPONENT_TAG,
                         boost::bind(&CDecompositionComponent::acceptPersistInserter,
                                     static_cast<const CDecompositionComponent*>(this), _1));
    inserter.insertValue(RNG_TAG, m_Rng.toString());
    inserter.insertLevel(BUCKETING_TAG, boost::bind(
                             &CSeasonalComponentAdaptiveBucketing::acceptPersistInserter, &m_Bucketing, _1));
}

bool CSeasonalComponent::initialized(void) const
{
    return this->CDecompositionComponent::initialized();
}

bool CSeasonalComponent::initialize(core_t::TTime startTime,
                                    core_t::TTime endTime,
                                    const TTimeTimePrMeanVarPrVec &values)
{
    this->clear();

    if (!m_Bucketing.initialize(this->maxSize()))
    {
        LOG_ERROR("Bad input size: " << this->maxSize());
        return false;
    }

    m_Bucketing.initialValues(startTime, endTime, values);

    return true;
}

std::size_t CSeasonalComponent::size(void) const
{
    return m_Bucketing.size();
}

void CSeasonalComponent::clear(void)
{
    this->CDecompositionComponent::clear();
    if (m_Bucketing.initialized())
    {
        m_Bucketing.clear();
    }
}

void CSeasonalComponent::shiftOrigin(core_t::TTime time)
{
    m_Bucketing.shiftOrigin(time);
}

void CSeasonalComponent::shiftLevel(double shift)
{
    this->CDecompositionComponent::shiftLevel(shift);
    m_Bucketing.shiftLevel(shift);
}

void CSeasonalComponent::shiftSlope(double shift)
{
    m_Bucketing.shiftSlope(shift);
}

void CSeasonalComponent::add(core_t::TTime time, double value, double weight)
{
    double predicted{CBasicStatistics::mean(this->value(this->jitter(time), 0.0))};
    m_Bucketing.add(time, value, predicted, weight);
}

void CSeasonalComponent::interpolate(core_t::TTime time, bool refine)
{
    if (refine)
    {
        m_Bucketing.refine(time);
    }

    TDoubleVec knots;
    TDoubleVec values;
    TDoubleVec variances;
    if (m_Bucketing.knots(time, this->boundaryCondition(), knots, values, variances))
    {
        this->CDecompositionComponent::interpolate(knots, values, variances);
    }
}

double CSeasonalComponent::decayRate(void) const
{
    return m_Bucketing.decayRate();
}

void CSeasonalComponent::decayRate(double decayRate)
{
    return m_Bucketing.decayRate(decayRate);
}

void CSeasonalComponent::propagateForwardsByTime(double time, bool meanRevert)
{
    m_Bucketing.propagateForwardsByTime(time, meanRevert);
}

const CSeasonalTime &CSeasonalComponent::time(void) const
{
    return m_Bucketing.time();
}

TDoubleDoublePr CSeasonalComponent::value(core_t::TTime time, double confidence) const
{
    double offset{this->time().periodic(time)};
    double n{m_Bucketing.count(time)};
    return this->CDecompositionComponent::value(offset, n, confidence);
}

double CSeasonalComponent::meanValue(void) const
{
    return this->CDecompositionComponent::meanValue();
}

double CSeasonalComponent::differenceFromMean(core_t::TTime time, core_t::TTime shortPeriod) const
{
    const CSeasonalTime &time_{this->time()};
    core_t::TTime longPeriod{time_.period()};

    if (longPeriod > shortPeriod && longPeriod % shortPeriod == 0)
    {
        CBasicStatistics::CMinMax<double> minmax;
        double mean = this->CDecompositionComponent::meanValue();
        for (core_t::TTime t = time; t < time + longPeriod; t += shortPeriod)
        {
            if (time_.inWindow(t))
            {
                double difference{CBasicStatistics::mean(this->value(t, 0.0)) - mean};
                minmax.add(difference);
            }
        }
        return minmax.signMargin();
    }

    return 0.0;
}

TDoubleDoublePr CSeasonalComponent::variance(core_t::TTime time, double confidence) const
{
    double offset{this->time().periodic(time)};
    double n{m_Bucketing.count(time)};
    return this->CDecompositionComponent::variance(offset, n, confidence);
}

double CSeasonalComponent::meanVariance(void) const
{
    return this->CDecompositionComponent::meanVariance();
}

double CSeasonalComponent::heteroscedasticity(void) const
{
    return this->CDecompositionComponent::heteroscedasticity();
}

double CSeasonalComponent::varianceDueToParameterDrift(core_t::TTime time) const
{
    return this->initialized() ? m_Bucketing.varianceDueToParameterDrift(time) : 0.0;
}

bool CSeasonalComponent::covariances(core_t::TTime time, TMatrix &result) const
{
    result = TMatrix(0.0);

    if (!this->initialized())
    {
        return false;
    }

    if (auto r = m_Bucketing.regression(time))
    {
        double variance{CBasicStatistics::mean(this->variance(time, 0.0))};
        return r->covariances(variance, result);
    }

    return false;
}

CSeasonalComponent::TSplineCRef CSeasonalComponent::valueSpline(void) const
{
    return this->CDecompositionComponent::valueSpline();
}

double CSeasonalComponent::slope(void) const
{
    return m_Bucketing.slope();
}

bool CSeasonalComponent::sufficientHistoryToPredict(core_t::TTime time) const
{
    return m_Bucketing.sufficientHistoryToPredict(time);
}

uint64_t CSeasonalComponent::checksum(uint64_t seed) const
{
    seed = this->CDecompositionComponent::checksum(seed);
    return CChecksum::calculate(seed, m_Bucketing);
}

void CSeasonalComponent::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    mem->setName("CSeasonalComponent");
    core::CMemoryDebug::dynamicSize("m_Bucketing", m_Bucketing, mem);
    core::CMemoryDebug::dynamicSize("m_Splines", this->splines(), mem);
}

std::size_t CSeasonalComponent::memoryUsage(void) const
{
    return core::CMemory::dynamicSize(m_Bucketing) + core::CMemory::dynamicSize(this->splines());
}

core_t::TTime CSeasonalComponent::jitter(core_t::TTime time)
{
    core_t::TTime result{time};
    if (m_Bucketing.minimumBucketLength() > 0.0)
    {
        const CSeasonalTime &time_{this->time()};
        double f{CSampling::uniformSample(m_Rng, 0.0, 1.0)};
        core_t::TTime a{time_.startOfWindow(time)};
        core_t::TTime b{a + time_.windowLength() - 1};
        double jitter{0.5 * m_Bucketing.minimumBucketLength()
                          * (f <= 0.5 ? ::sqrt(2.0 * f) - 1.0 : ::sqrt(2.0 * (f - 0.5)))};
        result = CTools::truncate(result + static_cast<core_t::TTime>(jitter + 0.5), a, b);
    }
    return result;
}

}
}
