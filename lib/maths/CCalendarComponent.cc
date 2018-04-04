/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */

#include <maths/CCalendarComponent.h>

#include <core/CLogger.h>
#include <core/CPersistUtils.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/Constants.h>
#include <core/RestoreMacros.h>

#include <maths/CChecksum.h>
#include <maths/CIntegerTools.h>
#include <maths/CSampling.h>
#include <maths/CSeasonalTime.h>

#include <boost/bind.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/normal.hpp>

#include <ios>
#include <vector>

namespace ml {
namespace maths {
namespace {
using TDoubleDoublePr = maths_t::TDoubleDoublePr;
const std::string DECOMPOSITION_COMPONENT_TAG{"a"};
const std::string BUCKETING_TAG{"b"};
const std::string EMPTY_STRING;
}

CCalendarComponent::CCalendarComponent(const CCalendarFeature& feature,
                                       std::size_t maxSize,
                                       double decayRate,
                                       double minimumBucketLength,
                                       CSplineTypes::EBoundaryCondition boundaryCondition,
                                       CSplineTypes::EType valueInterpolationType,
                                       CSplineTypes::EType varianceInterpolationType)
    : CDecompositionComponent{maxSize, boundaryCondition, valueInterpolationType, varianceInterpolationType},
      m_Bucketing{feature, decayRate, minimumBucketLength} {
}

CCalendarComponent::CCalendarComponent(double decayRate,
                                       double minimumBucketLength,
                                       core::CStateRestoreTraverser& traverser,
                                       CSplineTypes::EType valueInterpolationType,
                                       CSplineTypes::EType varianceInterpolationType)
    : CDecompositionComponent{0, CSplineTypes::E_Periodic, valueInterpolationType, varianceInterpolationType} {
    traverser.traverseSubLevel(boost::bind(&CCalendarComponent::acceptRestoreTraverser, this, decayRate, minimumBucketLength, _1));
}

void CCalendarComponent::swap(CCalendarComponent& other) {
    this->CDecompositionComponent::swap(other);
    m_Bucketing.swap(other.m_Bucketing);
}

bool CCalendarComponent::acceptRestoreTraverser(double decayRate, double minimumBucketLength, core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name{traverser.name()};
        RESTORE(DECOMPOSITION_COMPONENT_TAG,
                traverser.traverseSubLevel(
                    boost::bind(&CDecompositionComponent::acceptRestoreTraverser, static_cast<CDecompositionComponent*>(this), _1)))
        RESTORE_SETUP_TEARDOWN(BUCKETING_TAG,
                               CCalendarComponentAdaptiveBucketing bucketing(decayRate, minimumBucketLength, traverser),
                               true,
                               m_Bucketing.swap(bucketing))
    } while (traverser.next());

    return true;
}

void CCalendarComponent::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertLevel(
        DECOMPOSITION_COMPONENT_TAG,
        boost::bind(&CDecompositionComponent::acceptPersistInserter, static_cast<const CDecompositionComponent*>(this), _1));
    inserter.insertLevel(BUCKETING_TAG, boost::bind(&CCalendarComponentAdaptiveBucketing::acceptPersistInserter, &m_Bucketing, _1));
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

void CCalendarComponent::add(core_t::TTime time, double value, double weight) {
    m_Bucketing.add(time, value, weight);
}

void CCalendarComponent::interpolate(core_t::TTime time, bool refine) {
    if (refine) {
        m_Bucketing.refine(time);
    }

    TDoubleVec knots;
    TDoubleVec values;
    TDoubleVec variances;
    if (m_Bucketing.knots(time, this->boundaryCondition(), knots, values, variances)) {
        this->CDecompositionComponent::interpolate(knots, values, variances);
    }
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

CCalendarFeature CCalendarComponent::feature() const {
    return m_Bucketing.feature();
}

TDoubleDoublePr CCalendarComponent::value(core_t::TTime time, double confidence) const {
    double offset{static_cast<double>(this->feature().offset(time))};
    double n{m_Bucketing.count(time)};
    return this->CDecompositionComponent::value(offset, n, confidence);
}

double CCalendarComponent::meanValue() const {
    return this->CDecompositionComponent::meanValue();
}

TDoubleDoublePr CCalendarComponent::variance(core_t::TTime time, double confidence) const {
    double offset{static_cast<double>(this->feature().offset(time))};
    double n{m_Bucketing.count(time)};
    return this->CDecompositionComponent::variance(offset, n, confidence);
}

double CCalendarComponent::meanVariance() const {
    return this->CDecompositionComponent::meanVariance();
}

double CCalendarComponent::heteroscedasticity() const {
    return this->CDecompositionComponent::heteroscedasticity();
}

uint64_t CCalendarComponent::checksum(uint64_t seed) const {
    seed = this->CDecompositionComponent::checksum(seed);
    return CChecksum::calculate(seed, m_Bucketing);
}

void CCalendarComponent::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CCalendarComponent");
    core::CMemoryDebug::dynamicSize("m_Bucketing", m_Bucketing, mem);
    core::CMemoryDebug::dynamicSize("m_Splines", this->splines(), mem);
}

std::size_t CCalendarComponent::memoryUsage() const {
    return core::CMemory::dynamicSize(m_Bucketing) + core::CMemory::dynamicSize(this->splines());
}
}
}
