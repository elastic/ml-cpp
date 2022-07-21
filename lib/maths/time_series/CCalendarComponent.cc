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

#include <maths/time_series/CCalendarComponent.h>

#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/Constants.h>
#include <core/RestoreMacros.h>

#include <maths/common/CChecksum.h>
#include <maths/common/CIntegerTools.h>
#include <maths/common/CLinearAlgebra.h>
#include <maths/common/CSampling.h>

#include <maths/time_series/CSeasonalTime.h>

#include <ios>
#include <limits>
#include <vector>

namespace ml {
namespace maths {
namespace time_series {
namespace {
const core::TPersistenceTag DECOMPOSITION_COMPONENT_TAG{"a", "decomposition_component"};
const core::TPersistenceTag BUCKETING_TAG{"b", "bucketing"};
const core::TPersistenceTag LAST_INTERPOLATION_TAG{"c", "last_interpolation_time"};
const std::string EMPTY_STRING;
}

CCalendarComponent::CCalendarComponent(const CCalendarFeature& feature,
                                       core_t::TTime timeZoneOffset,
                                       std::size_t maxSize,
                                       double decayRate,
                                       double minBucketLength,
                                       common::CSplineTypes::EBoundaryCondition boundaryCondition,
                                       common::CSplineTypes::EType valueInterpolationType,
                                       common::CSplineTypes::EType varianceInterpolationType)
    : CDecompositionComponent{maxSize, boundaryCondition,
                              valueInterpolationType, varianceInterpolationType},
      m_Bucketing{feature, timeZoneOffset, decayRate, minBucketLength},
      m_LastInterpolationTime{2 * (std::numeric_limits<core_t::TTime>::min() / 3)} {
}

CCalendarComponent::CCalendarComponent(double decayRate,
                                       double minBucketLength,
                                       core::CStateRestoreTraverser& traverser,
                                       common::CSplineTypes::EType valueInterpolationType,
                                       common::CSplineTypes::EType varianceInterpolationType)
    : CDecompositionComponent{0, common::CSplineTypes::E_Periodic,
                              valueInterpolationType, varianceInterpolationType},
      m_LastInterpolationTime{2 * (std::numeric_limits<core_t::TTime>::min() / 3)} {
    if (traverser.traverseSubLevel([&](auto& traverser_) {
            return this->acceptRestoreTraverser(decayRate, minBucketLength, traverser_);
        }) == false) {
        traverser.setBadState();
    }
}

void CCalendarComponent::swap(CCalendarComponent& other) {
    this->CDecompositionComponent::swap(other);
    m_Bucketing.swap(other.m_Bucketing);
    std::swap(m_LastInterpolationTime, other.m_LastInterpolationTime);
}

bool CCalendarComponent::acceptRestoreTraverser(double decayRate,
                                                double minimumBucketLength,
                                                core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name{traverser.name()};
        RESTORE(DECOMPOSITION_COMPONENT_TAG, traverser.traverseSubLevel([this](auto& traverser_) {
            return this->CDecompositionComponent::acceptRestoreTraverser(traverser_);
        }))
        RESTORE_SETUP_TEARDOWN(BUCKETING_TAG,
                               CCalendarComponentAdaptiveBucketing bucketing(
                                   decayRate, minimumBucketLength, traverser),
                               true, m_Bucketing.swap(bucketing))
        RESTORE_BUILT_IN(LAST_INTERPOLATION_TAG, m_LastInterpolationTime)
    } while (traverser.next());
    return true;
}

void CCalendarComponent::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertLevel(DECOMPOSITION_COMPONENT_TAG, [this](auto& inserter_) {
        this->CDecompositionComponent::acceptPersistInserter(inserter_);
    });
    inserter.insertLevel(BUCKETING_TAG, [this](auto& inserter_) {
        m_Bucketing.acceptPersistInserter(inserter_);
    });
    inserter.insertValue(LAST_INTERPOLATION_TAG, m_LastInterpolationTime);
}

bool CCalendarComponent::initialized() const {
    return this->CDecompositionComponent::initialized();
}

void CCalendarComponent::initialize() {
    this->clear();
    m_Bucketing.initialize(this->maxSize());
}

std::size_t CCalendarComponent::size() const {
    return m_Bucketing.size();
}

void CCalendarComponent::clear() {
    this->CDecompositionComponent::clear();
    if (m_Bucketing.initialized()) {
        m_Bucketing.clear();
    }
}

void CCalendarComponent::linearScale(core_t::TTime time, double scale) {
    m_Bucketing.linearScale(scale);
    this->interpolate(time, false);
}

void CCalendarComponent::add(core_t::TTime time, double value, double weight) {
    m_Bucketing.add(time, value, weight);
}

bool CCalendarComponent::shouldInterpolate(core_t::TTime time) const {
    auto feature = this->feature();
    core_t::TTime offset{feature.offset(time)};
    return offset > feature.window() && time > m_LastInterpolationTime + offset;
}

void CCalendarComponent::interpolate(core_t::TTime time, bool refine) {
    if (refine) {
        m_Bucketing.refine(time);
    }
    TDoubleVec knots;
    TDoubleVec values;
    TDoubleVec variances;
    if (m_Bucketing.knots(time - this->feature().offset(time),
                          this->boundaryCondition(), knots, values, variances)) {
        this->CDecompositionComponent::interpolate(knots, values, variances);
    }
    m_LastInterpolationTime = time;
    LOG_TRACE(<< "last interpolation time = " << m_LastInterpolationTime);
}

double CCalendarComponent::decayRate() const {
    return m_Bucketing.decayRate();
}

void CCalendarComponent::decayRate(double decayRate) {
    return m_Bucketing.decayRate(decayRate);
}

void CCalendarComponent::propagateForwardsByTime(double time) {
    m_Bucketing.propagateForwardsByTime(time);
}

CCalendarFeatureAndTZ CCalendarComponent::feature() const {
    return m_Bucketing.feature();
}

CCalendarComponent::TVector2x1 CCalendarComponent::value(core_t::TTime time,
                                                         double confidence) const {
    double offset{static_cast<double>(this->feature().offset(time))};
    double n{m_Bucketing.count(time)};
    return this->CDecompositionComponent::value(offset, n, confidence);
}

double CCalendarComponent::meanValue() const {
    return this->CDecompositionComponent::meanValue();
}

CCalendarComponent::TVector2x1
CCalendarComponent::variance(core_t::TTime time, double confidence) const {
    double offset{static_cast<double>(this->feature().offset(time))};
    double n{m_Bucketing.count(time)};
    return this->CDecompositionComponent::variance(offset, n, confidence);
}

double CCalendarComponent::meanVariance() const {
    return this->CDecompositionComponent::meanVariance();
}

std::uint64_t CCalendarComponent::checksum(std::uint64_t seed) const {
    seed = this->CDecompositionComponent::checksum(seed);
    seed = common::CChecksum::calculate(seed, m_Bucketing);
    return common::CChecksum::calculate(seed, m_LastInterpolationTime);
}

void CCalendarComponent::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CCalendarComponent");
    core::CMemoryDebug::dynamicSize("m_Bucketing", m_Bucketing, mem);
    core::CMemoryDebug::dynamicSize("m_Splines", this->splines(), mem);
}

std::size_t CCalendarComponent::memoryUsage() const {
    return core::CMemory::dynamicSize(m_Bucketing) +
           core::CMemory::dynamicSize(this->splines());
}

bool CCalendarComponent::isBad() const {
    return m_Bucketing.isBad();
}
}
}
}
