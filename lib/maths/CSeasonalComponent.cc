/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CSeasonalComponent.h>

#include <core/CLogger.h>
#include <core/CPersistUtils.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/Constants.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CChecksum.h>
#include <maths/CIntegerTools.h>
#include <maths/CLeastSquaresOnlineRegressionDetail.h>
#include <maths/CSampling.h>
#include <maths/CSeasonalTime.h>

#include <cmath>
#include <limits>
#include <vector>

namespace ml {
namespace maths {
namespace {

using TDoubleDoublePr = maths_t::TDoubleDoublePr;

const core::TPersistenceTag DECOMPOSITION_COMPONENT_TAG{"a", "decomposition_component"};
const core::TPersistenceTag RNG_TAG{"b", "rng"};
const core::TPersistenceTag BUCKETING_TAG{"c", "bucketing"};
const core::TPersistenceTag LAST_INTERPOLATION_TAG{"d", "last_interpolation_time"};
const std::string EMPTY_STRING;
}

CSeasonalComponent::CSeasonalComponent(const CSeasonalTime& time,
                                       std::size_t maxSize,
                                       double decayRate,
                                       double minimumBucketLength,
                                       CSplineTypes::EBoundaryCondition boundaryCondition,
                                       CSplineTypes::EType valueInterpolationType,
                                       CSplineTypes::EType varianceInterpolationType)
    : CDecompositionComponent{maxSize, boundaryCondition,
                              valueInterpolationType, varianceInterpolationType},
      m_Bucketing{time, decayRate, minimumBucketLength},
      m_LastInterpolationTime{2 * (std::numeric_limits<core_t::TTime>::min() / 3)} {
}

CSeasonalComponent::CSeasonalComponent(double decayRate,
                                       double minimumBucketLength,
                                       core::CStateRestoreTraverser& traverser,
                                       CSplineTypes::EType valueInterpolationType,
                                       CSplineTypes::EType varianceInterpolationType)
    : CDecompositionComponent{0, CSplineTypes::E_Periodic,
                              valueInterpolationType, varianceInterpolationType},
      m_LastInterpolationTime{2 * (std::numeric_limits<core_t::TTime>::min() / 3)} {
    traverser.traverseSubLevel(std::bind(&CSeasonalComponent::acceptRestoreTraverser,
                                         this, decayRate, minimumBucketLength,
                                         std::placeholders::_1));
}

void CSeasonalComponent::swap(CSeasonalComponent& other) {
    this->CDecompositionComponent::swap(other);
    std::swap(m_Rng, other.m_Rng);
    m_Bucketing.swap(other.m_Bucketing);
    std::swap(m_LastInterpolationTime, other.m_LastInterpolationTime);
}

bool CSeasonalComponent::acceptRestoreTraverser(double decayRate,
                                                double minimumBucketLength,
                                                core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name{traverser.name()};
        RESTORE(DECOMPOSITION_COMPONENT_TAG,
                traverser.traverseSubLevel(std::bind(
                    &CDecompositionComponent::acceptRestoreTraverser,
                    static_cast<CDecompositionComponent*>(this), std::placeholders::_1)))
        RESTORE(RNG_TAG, m_Rng.fromString(traverser.value()))
        RESTORE_SETUP_TEARDOWN(BUCKETING_TAG,
                               CSeasonalComponentAdaptiveBucketing bucketing(
                                   decayRate, minimumBucketLength, traverser),
                               true, m_Bucketing.swap(bucketing))
        RESTORE_BUILT_IN(LAST_INTERPOLATION_TAG, m_LastInterpolationTime)
    } while (traverser.next());

    return true;
}

void CSeasonalComponent::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertLevel(DECOMPOSITION_COMPONENT_TAG,
                         std::bind(&CDecompositionComponent::acceptPersistInserter,
                                   static_cast<const CDecompositionComponent*>(this),
                                   std::placeholders::_1));
    inserter.insertValue(RNG_TAG, m_Rng.toString());
    inserter.insertLevel(BUCKETING_TAG,
                         std::bind(&CSeasonalComponentAdaptiveBucketing::acceptPersistInserter,
                                   &m_Bucketing, std::placeholders::_1));
    inserter.insertValue(LAST_INTERPOLATION_TAG, m_LastInterpolationTime);
}

bool CSeasonalComponent::initialized() const {
    return this->CDecompositionComponent::initialized();
}

bool CSeasonalComponent::initialize(core_t::TTime startTime,
                                    core_t::TTime endTime,
                                    const TFloatMeanAccumulatorVec& values) {
    this->clear();

    if (m_Bucketing.initialize(this->maxSize()) == false) {
        LOG_ERROR(<< "Bad input size: " << this->maxSize());
        return false;
    }

    m_Bucketing.initialValues(startTime, endTime, values);
    auto last = std::find_if(values.rbegin(), values.rend(),
                             [](const auto& value) {
                                 return CBasicStatistics::count(value) > 0.0;
                             })
                    .base();
    if (last != values.begin()) {
        this->interpolate(startTime + (static_cast<core_t::TTime>(last - values.begin()) *
                                       (endTime - startTime)) /
                                          static_cast<core_t::TTime>(values.size()));
    }

    return true;
}

std::size_t CSeasonalComponent::size() const {
    return m_Bucketing.size();
}

void CSeasonalComponent::clear() {
    this->CDecompositionComponent::clear();
    if (m_Bucketing.initialized()) {
        m_Bucketing.clear();
    }
}

void CSeasonalComponent::shiftOrigin(core_t::TTime time) {
    m_Bucketing.shiftOrigin(time);
}

void CSeasonalComponent::shiftLevel(double shift) {
    this->CDecompositionComponent::shiftLevel(shift);
    m_Bucketing.shiftLevel(shift);
}

void CSeasonalComponent::shiftSlope(core_t::TTime time, double shift) {
    m_Bucketing.shiftSlope(time, shift);
}

void CSeasonalComponent::linearScale(core_t::TTime time, double scale) {
    const auto& time_ = m_Bucketing.time();
    core_t::TTime startOfWindow{time_.startOfWindow(time) +
                                (time_.inWindow(time) ? 0 : time_.windowRepeat())};
    time = time <= startOfWindow ? startOfWindow : time_.startOfPeriod(time);
    m_Bucketing.linearScale(time, scale);
    this->interpolate(time, false);
}

void CSeasonalComponent::add(core_t::TTime time, double value, double weight, double gradientLearnRate) {
    double predicted{CBasicStatistics::mean(this->value(this->jitter(time), 0.0))};
    m_Bucketing.add(time, value, predicted, weight, gradientLearnRate);
}

bool CSeasonalComponent::shouldInterpolate(core_t::TTime time) const {
    const auto& time_ = m_Bucketing.time();
    return time_.startOfPeriod(time) > time_.startOfPeriod(m_LastInterpolationTime);
}

void CSeasonalComponent::interpolate(core_t::TTime time, bool refine) {
    if (refine) {
        m_Bucketing.refine(time);
    }

    const auto& time_ = m_Bucketing.time();
    core_t::TTime startOfWindow{time_.startOfWindow(time) +
                                (time_.inWindow(time) ? 0 : time_.windowRepeat())};

    TDoubleVec knots;
    TDoubleVec values;
    TDoubleVec variances;
    if (m_Bucketing.knots(time <= startOfWindow ? startOfWindow : time_.startOfPeriod(time),
                          this->boundaryCondition(), knots, values, variances)) {
        this->CDecompositionComponent::interpolate(knots, values, variances);
    }
    m_LastInterpolationTime = time_.startOfPeriod(time);
    LOG_TRACE(<< "last interpolation time = " << m_LastInterpolationTime);
}

double CSeasonalComponent::decayRate() const {
    return m_Bucketing.decayRate();
}

void CSeasonalComponent::decayRate(double decayRate) {
    return m_Bucketing.decayRate(decayRate);
}

void CSeasonalComponent::propagateForwardsByTime(double time, double meanRevertFactor) {
    m_Bucketing.propagateForwardsByTime(time, meanRevertFactor);
}

const CSeasonalTime& CSeasonalComponent::time() const {
    return m_Bucketing.time();
}

const CSeasonalComponentAdaptiveBucketing& CSeasonalComponent::bucketing() const {
    return m_Bucketing;
}

TDoubleDoublePr CSeasonalComponent::value(core_t::TTime time, double confidence) const {
    double offset{this->time().periodic(time)};
    double n{m_Bucketing.count(time)};
    return this->CDecompositionComponent::value(offset, n, confidence);
}

double CSeasonalComponent::meanValue() const {
    return this->CDecompositionComponent::meanValue();
}

double CSeasonalComponent::delta(core_t::TTime time,
                                 core_t::TTime shortPeriod,
                                 double shortDifference) const {
    // This is used to adjust how periodic patterns in the trend are
    // represented in the case that we have two periodic components
    // one of which is a divisor of the other. We are interested in
    // two situations:
    //   1) The long component has a bias at this time, w.r.t. its
    //      mean, for all repeats of short component,
    //   2) The long and short components partially cancel at the
    //      specified time.
    // In the first case we can represent the bias using the short
    // seasonal component; we prefer to do this since the resolution
    // is better. In the second case we have a bad decomposition of
    // periodic features at the long period into terms which cancel
    // out or reinforce. In this case we want to just represent the
    // periodic features in long component. We can achieve this by
    // reducing the value in the short seasonal component.

    using TMinMaxAccumulator = CBasicStatistics::CMinMax<double>;

    const CSeasonalTime& time_{this->time()};
    core_t::TTime longPeriod{time_.period()};

    if (longPeriod > shortPeriod && longPeriod % shortPeriod == 0) {
        TMinMaxAccumulator bias;
        double amplitude{0.0};
        double margin{std::fabs(shortDifference)};
        double cancelling{0.0};
        double mean{this->CDecompositionComponent::meanValue()};
        for (core_t::TTime t = time; t < time + longPeriod; t += shortPeriod) {
            if (time_.inWindow(t)) {
                double difference{CBasicStatistics::mean(this->value(t, 0.0)) - mean};
                bias.add(difference);
                amplitude = std::max(amplitude, std::fabs(difference));
                if (shortDifference * difference < 0.0) {
                    margin = std::min(margin, std::fabs(difference));
                    cancelling += 1.0;
                } else {
                    cancelling -= 1.0;
                }
            }
        }
        return bias.signMargin() != 0.0 ? bias.signMargin()
                                        : (cancelling > 0.0 && margin > 0.2 * amplitude
                                               ? std::copysign(margin, -shortDifference)
                                               : 0.0);
    }

    return 0.0;
}

TDoubleDoublePr CSeasonalComponent::variance(core_t::TTime time, double confidence) const {
    double offset{this->time().periodic(time)};
    double n{m_Bucketing.count(time)};
    return this->CDecompositionComponent::variance(offset, n, confidence);
}

double CSeasonalComponent::meanVariance() const {
    return this->CDecompositionComponent::meanVariance();
}

bool CSeasonalComponent::covariances(core_t::TTime time, TMatrix& result) const {
    result = TMatrix(0.0);

    if (this->initialized() == false) {
        return false;
    }

    if (auto r = m_Bucketing.regression(time)) {
        double variance{CBasicStatistics::mean(this->variance(time, 0.0))};
        return r->covariances(variance, result);
    }

    return false;
}

CSeasonalComponent::TSplineCRef CSeasonalComponent::valueSpline() const {
    return this->CDecompositionComponent::valueSpline();
}

double CSeasonalComponent::slope() const {
    return m_Bucketing.slope();
}

bool CSeasonalComponent::slopeAccurate(core_t::TTime time) const {
    return m_Bucketing.slopeAccurate(time);
}

std::uint64_t CSeasonalComponent::checksum(std::uint64_t seed) const {
    seed = this->CDecompositionComponent::checksum(seed);
    seed = CChecksum::calculate(seed, m_Bucketing);
    return CChecksum::calculate(seed, m_LastInterpolationTime);
}

void CSeasonalComponent::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CSeasonalComponent");
    core::CMemoryDebug::dynamicSize("m_Bucketing", m_Bucketing, mem);
    core::CMemoryDebug::dynamicSize("m_Splines", this->splines(), mem);
}

std::size_t CSeasonalComponent::memoryUsage() const {
    return core::CMemory::dynamicSize(m_Bucketing) +
           core::CMemory::dynamicSize(this->splines());
}

core_t::TTime CSeasonalComponent::jitter(core_t::TTime time) {
    core_t::TTime result{time};
    if (m_Bucketing.minimumBucketLength() > 0.0) {
        const CSeasonalTime& time_{this->time()};
        double f{CSampling::uniformSample(m_Rng, 0.0, 1.0)};
        core_t::TTime a{time_.startOfWindow(time)};
        core_t::TTime b{a + time_.windowLength() - 1};
        double jitter{0.5 * m_Bucketing.minimumBucketLength() *
                      (f <= 0.5 ? std::sqrt(2.0 * f) - 1.0 : std::sqrt(2.0 * (f - 0.5)))};
        result = CTools::truncate(result + static_cast<core_t::TTime>(jitter + 0.5), a, b);
    }
    return result;
}
}
}
