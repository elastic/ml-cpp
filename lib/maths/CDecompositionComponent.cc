/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CDecompositionComponent.h>

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

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/normal.hpp>

#include <ios>
#include <vector>

namespace ml {
namespace maths {
namespace {

using TDoubleDoublePr = maths_t::TDoubleDoublePr;

const core::TPersistenceTag MAX_SIZE_TAG{"a", "max_size"};
const core::TPersistenceTag RNG_TAG{"b", "rng"};
const core::TPersistenceTag BOUNDARY_CONDITION_TAG{"c", "boundary_condition"};
const core::TPersistenceTag BUCKETING_TAG{"d", "bucketing"};
const core::TPersistenceTag SPLINES_TAG{"e", "splines"};

// Nested tags
const core::TPersistenceTag ESTIMATED_TAG{"a", "estimated"};
const core::TPersistenceTag KNOTS_TAG{"b", "knots"};
const core::TPersistenceTag VALUES_TAG{"c", "values"};
const core::TPersistenceTag VARIANCES_TAG{"d", "variances"};

const std::string EMPTY_STRING;
}

CDecompositionComponent::CDecompositionComponent(std::size_t maxSize,
                                                 CSplineTypes::EBoundaryCondition boundaryCondition,
                                                 CSplineTypes::EType valueInterpolationType,
                                                 CSplineTypes::EType varianceInterpolationType)
    : m_MaxSize{maxSize}, m_BoundaryCondition{boundaryCondition}, m_Splines{valueInterpolationType,
                                                                            varianceInterpolationType},
      m_MeanValue{0.0}, m_MeanVariance{0.0} {
}

bool CDecompositionComponent::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name{traverser.name()};
        RESTORE_BUILT_IN(MAX_SIZE_TAG, m_MaxSize)
        RESTORE_SETUP_TEARDOWN(
            BOUNDARY_CONDITION_TAG, int boundaryCondition,
            core::CStringUtils::stringToType(traverser.value(), boundaryCondition),
            m_BoundaryCondition = static_cast<CSplineTypes::EBoundaryCondition>(boundaryCondition))
        RESTORE(SPLINES_TAG, traverser.traverseSubLevel(std::bind(
                                 &CPackedSplines::acceptRestoreTraverser, &m_Splines,
                                 m_BoundaryCondition, std::placeholders::_1)))
    } while (traverser.next());

    if (this->initialized()) {
        m_MeanValue = this->valueSpline().mean();
        m_MeanVariance = this->varianceSpline().mean();
    }

    return true;
}

void CDecompositionComponent::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(MAX_SIZE_TAG, m_MaxSize);
    inserter.insertValue(BOUNDARY_CONDITION_TAG, static_cast<int>(m_BoundaryCondition));
    inserter.insertLevel(SPLINES_TAG, std::bind(&CPackedSplines::acceptPersistInserter,
                                                &m_Splines, std::placeholders::_1));
}

void CDecompositionComponent::swap(CDecompositionComponent& other) {
    std::swap(m_MaxSize, other.m_MaxSize);
    std::swap(m_BoundaryCondition, other.m_BoundaryCondition);
    std::swap(m_MeanValue, other.m_MeanValue);
    std::swap(m_MeanVariance, other.m_MeanVariance);
    m_Splines.swap(other.m_Splines);
}

bool CDecompositionComponent::initialized() const {
    return m_Splines.initialized();
}

void CDecompositionComponent::clear() {
    if (m_Splines.initialized()) {
        m_Splines.clear();
    }
    m_MeanValue = 0.0;
    m_MeanVariance = 0.0;
}

void CDecompositionComponent::interpolate(const TDoubleVec& knots,
                                          const TDoubleVec& values,
                                          const TDoubleVec& variances) {
    m_Splines.interpolate(knots, values, variances, m_BoundaryCondition);

    m_MeanValue = this->valueSpline().mean();
    m_MeanVariance = this->varianceSpline().mean();
}

void CDecompositionComponent::shiftLevel(double shift) {
    m_Splines.shift(CPackedSplines::E_Value, shift);
    m_MeanValue += shift;
}

TDoubleDoublePr CDecompositionComponent::value(double offset, double n, double confidence) const {
    // In order to compute a confidence interval we need to know
    // the distribution of the samples. In practice, as long as
    // they are independent, then the sample mean will be
    // asymptotically normal with mean equal to the sample mean
    // and variance equal to the sample variance divided by root
    // of the number of samples.
    if (this->initialized()) {
        double m{this->valueSpline().value(offset)};

        if (confidence == 0.0) {
            return {m, m};
        }

        n = std::max(n, 1.0);
        double sd{::sqrt(std::max(this->varianceSpline().value(offset), 0.0) / n)};
        if (sd == 0.0) {
            return {m, m};
        }

        try {
            boost::math::normal normal{m, sd};
            double ql{boost::math::quantile(normal, (100.0 - confidence) / 200.0)};
            double qu{boost::math::quantile(normal, (100.0 + confidence) / 200.0)};
            return {ql, qu};
        } catch (const std::exception& e) {
            LOG_ERROR(<< "Failed calculating confidence interval: " << e.what()
                      << ", n = " << n << ", m = " << m << ", sd = " << sd
                      << ", confidence = " << confidence);
        }
        return {m, m};
    }

    return {m_MeanValue, m_MeanValue};
}

double CDecompositionComponent::meanValue() const {
    return m_MeanValue;
}

TDoubleDoublePr CDecompositionComponent::variance(double offset, double n, double confidence) const {
    // In order to compute a confidence interval we need to know
    // the distribution of the samples. In practice, as long as
    // they are independent, then the sample variance will be
    // asymptotically chi-squared with number of samples minus
    // one degrees of freedom.

    if (this->initialized()) {
        n = std::max(n, 2.0);
        double v{this->varianceSpline().value(offset)};
        if (confidence == 0.0) {
            return {v, v};
        }
        try {
            boost::math::chi_squared chi{n - 1.0};
            double ql{boost::math::quantile(chi, (100.0 - confidence) / 200.0)};
            double qu{boost::math::quantile(chi, (100.0 + confidence) / 200.0)};
            return std::make_pair(ql * v / (n - 1.0), qu * v / (n - 1.0));
        } catch (const std::exception& e) {
            LOG_ERROR(<< "Failed calculating confidence interval: " << e.what()
                      << ", n = " << n << ", confidence = " << confidence);
        }
        return {v, v};
    }
    return {m_MeanVariance, m_MeanVariance};
}

double CDecompositionComponent::meanVariance() const {
    return m_MeanVariance;
}

std::size_t CDecompositionComponent::maxSize() const {
    return std::max(m_MaxSize, MIN_MAX_SIZE);
}

CSplineTypes::EBoundaryCondition CDecompositionComponent::boundaryCondition() const {
    return m_BoundaryCondition;
}

CDecompositionComponent::TSplineCRef CDecompositionComponent::valueSpline() const {
    return m_Splines.spline(CPackedSplines::E_Value);
}

CDecompositionComponent::TSplineCRef CDecompositionComponent::varianceSpline() const {
    return m_Splines.spline(CPackedSplines::E_Variance);
}

uint64_t CDecompositionComponent::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_MaxSize);
    seed = CChecksum::calculate(seed, m_BoundaryCondition);
    seed = CChecksum::calculate(seed, m_Splines);
    seed = CChecksum::calculate(seed, m_MeanValue);
    return CChecksum::calculate(seed, m_MeanVariance);
}

const CDecompositionComponent::CPackedSplines& CDecompositionComponent::splines() const {
    return m_Splines;
}

const std::size_t CDecompositionComponent::MIN_MAX_SIZE{1u};

////// CDecompositionComponent::CPackedSplines //////

CDecompositionComponent::CPackedSplines::CPackedSplines(CSplineTypes::EType valueInterpolationType,
                                                        CSplineTypes::EType varianceInterpolationType) {
    m_Types[static_cast<std::size_t>(E_Value)] = valueInterpolationType;
    m_Types[static_cast<std::size_t>(E_Variance)] = varianceInterpolationType;
}

bool CDecompositionComponent::CPackedSplines::acceptRestoreTraverser(
    CSplineTypes::EBoundaryCondition boundary,
    core::CStateRestoreTraverser& traverser) {
    int estimated{0};
    TDoubleVec knots;
    TDoubleVec values;
    TDoubleVec variances;

    do {
        const std::string& name{traverser.name()};
        RESTORE_BUILT_IN(ESTIMATED_TAG, estimated)
        RESTORE(KNOTS_TAG, core::CPersistUtils::fromString(traverser.value(), knots))
        RESTORE(VALUES_TAG, core::CPersistUtils::fromString(traverser.value(), values))
        RESTORE(VARIANCES_TAG, core::CPersistUtils::fromString(traverser.value(), variances))
    } while (traverser.next());

    if (estimated == 1) {
        this->interpolate(knots, values, variances, boundary);
    }

    this->checkRestoredInvariants();

    return true;
}

void CDecompositionComponent::CPackedSplines::checkRestoredInvariants() const {
    VIOLATES_INVARIANT(m_Knots.size(), !=, m_Values[0].size());
    VIOLATES_INVARIANT(m_Values[0].size(), !=, m_Values[1].size());
}

void CDecompositionComponent::CPackedSplines::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(ESTIMATED_TAG, static_cast<int>(this->initialized()));
    if (this->initialized()) {
        inserter.insertValue(KNOTS_TAG, core::CPersistUtils::toString(m_Knots));
        inserter.insertValue(VALUES_TAG, core::CPersistUtils::toString(m_Values[0]));
        inserter.insertValue(VARIANCES_TAG, core::CPersistUtils::toString(m_Values[1]));
    }
}

void CDecompositionComponent::CPackedSplines::swap(CPackedSplines& other) noexcept {
    std::swap(m_Types, other.m_Types);
    m_Knots.swap(other.m_Knots);
    m_Values[0].swap(other.m_Values[0]);
    m_Values[1].swap(other.m_Values[1]);
    m_Curvatures[0].swap(other.m_Curvatures[0]);
    m_Curvatures[1].swap(other.m_Curvatures[1]);
}

bool CDecompositionComponent::CPackedSplines::initialized() const {
    return m_Knots.size() > 0;
}

void CDecompositionComponent::CPackedSplines::clear() {
    this->spline(E_Value).clear();
    this->spline(E_Variance).clear();
}

void CDecompositionComponent::CPackedSplines::shift(ESpline spline, double shift) {
    for (auto& value : m_Values[static_cast<std::size_t>(spline)]) {
        value += shift;
    }
}

CDecompositionComponent::TSplineCRef
CDecompositionComponent::CPackedSplines::spline(ESpline spline) const {
    return TSplineCRef(m_Types[static_cast<std::size_t>(spline)], std::cref(m_Knots),
                       std::cref(m_Values[static_cast<std::size_t>(spline)]),
                       std::cref(m_Curvatures[static_cast<std::size_t>(spline)]));
}

CDecompositionComponent::TSplineRef
CDecompositionComponent::CPackedSplines::spline(ESpline spline) {
    return TSplineRef(m_Types[static_cast<std::size_t>(spline)], std::ref(m_Knots),
                      std::ref(m_Values[static_cast<std::size_t>(spline)]),
                      std::ref(m_Curvatures[static_cast<std::size_t>(spline)]));
}

const CDecompositionComponent::TFloatVec& CDecompositionComponent::CPackedSplines::knots() const {
    return m_Knots;
}

void CDecompositionComponent::CPackedSplines::interpolate(const TDoubleVec& knots,
                                                          const TDoubleVec& values,
                                                          const TDoubleVec& variances,
                                                          CSplineTypes::EBoundaryCondition boundary) {
    CPackedSplines oldSpline{m_Types[0], m_Types[1]};
    this->swap(oldSpline);
    TSplineRef valueSpline{this->spline(E_Value)};
    TSplineRef varianceSpline{this->spline(E_Variance)};
    if (!valueSpline.interpolate(knots, values, boundary)) {
        this->swap(oldSpline);
    } else if (!varianceSpline.interpolate(knots, variances, boundary)) {
        this->swap(oldSpline);
    }
    LOG_TRACE(<< "types = " << core::CContainerPrinter::print(m_Types));
    LOG_TRACE(<< "knots = " << core::CContainerPrinter::print(m_Knots));
    LOG_TRACE(<< "values = " << core::CContainerPrinter::print(m_Values));
    LOG_TRACE(<< "curvatures = " << core::CContainerPrinter::print(m_Curvatures));
}

uint64_t CDecompositionComponent::CPackedSplines::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_Types);
    seed = CChecksum::calculate(seed, m_Knots);
    seed = CChecksum::calculate(seed, m_Values);
    return CChecksum::calculate(seed, m_Curvatures);
}

void CDecompositionComponent::CPackedSplines::debugMemoryUsage(
    const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CPackedSplines");
    core::CMemoryDebug::dynamicSize("m_Knots", m_Knots, mem);
    core::CMemoryDebug::dynamicSize("m_Values[0]", m_Values[0], mem);
    core::CMemoryDebug::dynamicSize("m_Values[1]", m_Values[1], mem);
    core::CMemoryDebug::dynamicSize("m_Curvatures[0]", m_Curvatures[0], mem);
    core::CMemoryDebug::dynamicSize("m_Curvatures[1]", m_Curvatures[1], mem);
}

std::size_t CDecompositionComponent::CPackedSplines::memoryUsage() const {
    std::size_t mem{core::CMemory::dynamicSize(m_Knots)};
    mem += core::CMemory::dynamicSize(m_Values[0]);
    mem += core::CMemory::dynamicSize(m_Values[1]);
    mem += core::CMemory::dynamicSize(m_Curvatures[0]);
    mem += core::CMemory::dynamicSize(m_Curvatures[1]);
    return mem;
}
}
}
