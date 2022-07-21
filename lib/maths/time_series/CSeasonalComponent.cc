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

#include <maths/time_series/CSeasonalComponent.h>

#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/Constants.h>
#include <core/RestoreMacros.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CChecksum.h>
#include <maths/common/CLeastSquaresOnlineRegressionDetail.h>
#include <maths/common/CLinearAlgebra.h>
#include <maths/common/CSampling.h>
#include <maths/common/CSolvers.h>

#include <maths/time_series/CSeasonalTime.h>

#include <cmath>
#include <vector>

namespace ml {
namespace maths {
namespace time_series {
namespace {
const core::TPersistenceTag DECOMPOSITION_COMPONENT_TAG{"a", "decomposition_component"};
const core::TPersistenceTag RNG_TAG{"b", "rng"};
const core::TPersistenceTag BUCKETING_TAG{"c", "bucketing"};
const core::TPersistenceTag LAST_INTERPOLATION_TAG{"d", "last_interpolation_time"};
const core::TPersistenceTag TOTAL_SHIFT_TAG{"e", "total_shift"};
const core::TPersistenceTag CURRENT_MEAN_SHIFT_TAG{"f", "current_mean"};
const core::TPersistenceTag MAX_TIME_SHIFT_PER_PERIOD_TAG{"g", "max_time_shift_per_period"};
const std::string EMPTY_STRING;
}

CSeasonalComponent::CSeasonalComponent(const CSeasonalTime& time,
                                       std::size_t maxSize,
                                       double decayRate,
                                       double minBucketLength,
                                       core_t::TTime maxTimeShiftPerPeriod,
                                       common::CSplineTypes::EBoundaryCondition boundaryCondition,
                                       common::CSplineTypes::EType valueInterpolationType,
                                       common::CSplineTypes::EType varianceInterpolationType)
    : CDecompositionComponent{maxSize, boundaryCondition,
                              valueInterpolationType, varianceInterpolationType},
      m_Bucketing{time, decayRate, minBucketLength},
      m_MaxTimeShiftPerPeriod{common::CBasicStatistics::min(
          maxTimeShiftPerPeriod,
          static_cast<core_t::TTime>(minBucketLength / 2.0 + 0.5),
          static_cast<core_t::TTime>(0.1 * static_cast<double>(time.period()) + 0.5))} {
}

CSeasonalComponent::CSeasonalComponent(double decayRate,
                                       double minBucketLength,
                                       core::CStateRestoreTraverser& traverser,
                                       common::CSplineTypes::EType valueInterpolationType,
                                       common::CSplineTypes::EType varianceInterpolationType)
    : CDecompositionComponent{0, common::CSplineTypes::E_Periodic,
                              valueInterpolationType, varianceInterpolationType} {
    if (traverser.traverseSubLevel([&](auto& traverser_) {
            return this->acceptRestoreTraverser(decayRate, minBucketLength, traverser_);
        }) == false) {
        traverser.setBadState();
    }
}

void CSeasonalComponent::swap(CSeasonalComponent& other) {
    this->CDecompositionComponent::swap(other);
    std::swap(m_Rng, other.m_Rng);
    m_Bucketing.swap(other.m_Bucketing);
    std::swap(m_LastInterpolationTime, other.m_LastInterpolationTime);
    std::swap(m_MaxTimeShiftPerPeriod, other.m_MaxTimeShiftPerPeriod);
    std::swap(m_TotalShift, other.m_TotalShift);
    std::swap(m_CurrentMeanShift, other.m_CurrentMeanShift);
}

bool CSeasonalComponent::acceptRestoreTraverser(double decayRate,
                                                double minBucketLength,
                                                core::CStateRestoreTraverser& traverser) {
    bool restoredBucketing{false};
    do {
        const std::string& name{traverser.name()};
        RESTORE(DECOMPOSITION_COMPONENT_TAG, traverser.traverseSubLevel([this](auto& traverser_) {
            return this->CDecompositionComponent::acceptRestoreTraverser(traverser_);
        }))
        RESTORE(RNG_TAG, m_Rng.fromString(traverser.value()))
        RESTORE_SETUP_TEARDOWN(BUCKETING_TAG,
                               CSeasonalComponentAdaptiveBucketing bucketing(
                                   decayRate, minBucketLength, traverser),
                               restoredBucketing = (traverser.haveBadState() == false),
                               m_Bucketing.swap(bucketing))
        RESTORE_BUILT_IN(LAST_INTERPOLATION_TAG, m_LastInterpolationTime)
        RESTORE_BUILT_IN(MAX_TIME_SHIFT_PER_PERIOD_TAG, m_MaxTimeShiftPerPeriod)
        RESTORE_BUILT_IN(TOTAL_SHIFT_TAG, m_TotalShift)
        RESTORE(CURRENT_MEAN_SHIFT_TAG,
                m_CurrentMeanShift.fromDelimited(traverser.value()))
    } while (traverser.next());

    if (restoredBucketing == false) {
        LOG_ERROR(<< "Did not restore seasonal component adaptive bucketing");
        return false;
    }

    return true;
}

void CSeasonalComponent::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertLevel(DECOMPOSITION_COMPONENT_TAG, [this](auto& inserter_) {
        this->CDecompositionComponent::acceptPersistInserter(inserter_);
    });
    inserter.insertValue(RNG_TAG, m_Rng.toString());
    inserter.insertLevel(BUCKETING_TAG, [this](auto& inserter_) {
        m_Bucketing.acceptPersistInserter(inserter_);
    });
    inserter.insertValue(LAST_INTERPOLATION_TAG, m_LastInterpolationTime);
    inserter.insertValue(MAX_TIME_SHIFT_PER_PERIOD_TAG, m_MaxTimeShiftPerPeriod);
    inserter.insertValue(TOTAL_SHIFT_TAG, m_TotalShift);
    inserter.insertValue(CURRENT_MEAN_SHIFT_TAG, m_CurrentMeanShift.toDelimited());
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
                                 return common::CBasicStatistics::count(value) > 0.0;
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
    core_t::TTime shift;
    double shiftWeight;
    std::tie(shift, shiftWeight) = this->likelyShift(time, value);
    m_CurrentMeanShift.add(static_cast<double>(shift), weight * shiftWeight);
    double prediction{this->value(this->jitter(time), 0.0).mean()};
    m_Bucketing.add(time + m_TotalShift, value, prediction, weight, gradientLearnRate);
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
    m_TotalShift += static_cast<core_t::TTime>(
        common::CBasicStatistics::mean(m_CurrentMeanShift) + 0.5);
    m_TotalShift = m_TotalShift % this->time().period();
    m_CurrentMeanShift = TFloatMeanAccumulator{};
    LOG_TRACE(<< "total shift = " << m_TotalShift);
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

CSeasonalComponent::TVector2x1 CSeasonalComponent::value(core_t::TTime time,
                                                         double confidence) const {
    time += m_TotalShift;
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

    using TMinMaxAccumulator = common::CBasicStatistics::CMinMax<double>;

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
                double difference{this->value(t, 0.0).mean() - mean};
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

CSeasonalComponent::TVector2x1
CSeasonalComponent::variance(core_t::TTime time, double confidence) const {
    time += m_TotalShift;
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

    time += m_TotalShift;
    if (const auto* r = m_Bucketing.regression(time)) {
        double variance{this->variance(time, 0.0).mean()};
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
    seed = common::CChecksum::calculate(seed, m_Bucketing);
    seed = common::CChecksum::calculate(seed, m_LastInterpolationTime);
    seed = common::CChecksum::calculate(seed, m_MaxTimeShiftPerPeriod);
    seed = common::CChecksum::calculate(seed, m_TotalShift);
    return common::CChecksum::calculate(seed, m_CurrentMeanShift);
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

CSeasonalComponent::TTimeDoublePr CSeasonalComponent::likelyShift(core_t::TTime maxTimeShift,
                                                                  core_t::TTime time,
                                                                  const TLossFunc& loss) {
    if (maxTimeShift == 0) {
        return {0, 0.0};
    }

    std::array<double, 7> times;
    double range{2 * static_cast<double>(maxTimeShift)};
    double step{range / static_cast<double>(times.size() - 1)};
    times[0] = static_cast<double>(time) - range / 2.0;
    for (std::size_t i = 1; i < times.size(); ++i) {
        times[i] = times[i - 1] + step;
    }

    double shiftedTime;
    double lossAtShiftedTime;
    double lossStandardDeviation;
    common::CSolvers::globalMinimize(times, loss, shiftedTime,
                                     lossAtShiftedTime, lossStandardDeviation);
    LOG_TRACE(<< "shift = " << static_cast<core_t::TTime>(shiftedTime + 0.5) - time
              << ", loss(shift) = " << lossAtShiftedTime
              << ", sd(loss) = " << lossStandardDeviation);

    return {static_cast<core_t::TTime>(shiftedTime + 0.5), lossStandardDeviation};
}

CSeasonalComponent::TTimeDoublePr
CSeasonalComponent::likelyShift(core_t::TTime time, double value) const {

    double range{2 * static_cast<double>(m_MaxTimeShiftPerPeriod)};

    // If the change due to the shift is small compared to the prediction
    // error force it to zero.
    double noise{0.2 * std::sqrt(this->meanVariance()) / range};
    auto loss = [&](double shift) {
        auto shift_ = static_cast<core_t::TTime>(shift + 0.5);
        return std::fabs(this->value(time + shift_, 0.0).mean() - value) +
               noise * std::fabs(shift);
    };

    return likelyShift(m_MaxTimeShiftPerPeriod, 0, loss);
}

core_t::TTime CSeasonalComponent::jitter(core_t::TTime time) {
    core_t::TTime result{time};
    if (m_Bucketing.minimumBucketLength() > 0.0) {
        const CSeasonalTime& time_{this->time()};
        double f{common::CSampling::uniformSample(m_Rng, 0.0, 1.0)};
        core_t::TTime a{time_.startOfWindow(time)};
        core_t::TTime b{a + time_.windowLength() - 1};
        double jitter{0.5 * m_Bucketing.minimumBucketLength() *
                      (f <= 0.5 ? std::sqrt(2.0 * f) - 1.0 : std::sqrt(2.0 * (f - 0.5)))};
        result = common::CTools::truncate(
            result + static_cast<core_t::TTime>(jitter + 0.5), a, b);
    }
    return result;
}
}
}
}
