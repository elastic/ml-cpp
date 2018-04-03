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

#include <boost/bind.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/normal.hpp>

#include <ios>
#include <vector>

namespace ml {
namespace maths {
namespace {

typedef maths_t::TDoubleDoublePr TDoubleDoublePr;

const std::string MAX_SIZE_TAG{"a"};
const std::string RNG_TAG{"b"};
const std::string BOUNDARY_CONDITION_TAG{"c"};
const std::string BUCKETING_TAG{"d"};
const std::string SPLINES_TAG{"e"};

// Nested tags
const std::string ESTIMATED_TAG{"a"};
const std::string KNOTS_TAG{"b"};
const std::string VALUES_TAG{"c"};
const std::string VARIANCES_TAG{"d"};

const std::string EMPTY_STRING;
}

CDecompositionComponent::CDecompositionComponent(std::size_t maxSize,
                                                 CSplineTypes::EBoundaryCondition boundaryCondition,
                                                 CSplineTypes::EType valueInterpolationType,
                                                 CSplineTypes::EType varianceInterpolationType)
    : m_MaxSize{maxSize},
      m_BoundaryCondition{boundaryCondition},
      m_Splines{valueInterpolationType, varianceInterpolationType},
      m_MeanValue{0.0},
      m_MeanVariance{0.0} {
}

bool CDecompositionComponent::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name{traverser.name()};
        RESTORE_BUILT_IN(MAX_SIZE_TAG, m_MaxSize)
        RESTORE_SETUP_TEARDOWN(BOUNDARY_CONDITION_TAG,
                               int boundaryCondition,
                               core::CStringUtils::stringToType(traverser.value(), boundaryCondition),
                               m_BoundaryCondition = static_cast<CSplineTypes::EBoundaryCondition>(boundaryCondition))
        RESTORE(SPLINES_TAG,
                traverser.traverseSubLevel(boost::bind(&CPackedSplines::acceptRestoreTraverser, &m_Splines, m_BoundaryCondition, _1)))
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
    inserter.insertLevel(SPLINES_TAG, boost::bind(&CPackedSplines::acceptPersistInserter, &m_Splines, _1));
}

void CDecompositionComponent::swap(CDecompositionComponent& other) {
    std::swap(m_MaxSize, other.m_MaxSize);
    std::swap(m_BoundaryCondition, other.m_BoundaryCondition);
    std::swap(m_MeanValue, other.m_MeanValue);
    std::swap(m_MeanVariance, other.m_MeanVariance);
    m_Splines.swap(other.m_Splines);
}

bool CDecompositionComponent::initialized(void) const {
    return m_Splines.initialized();
}

void CDecompositionComponent::clear(void) {
    if (m_Splines.initialized()) {
        m_Splines.clear();
    }
    m_MeanValue = 0.0;
    m_MeanVariance = 0.0;
}

void CDecompositionComponent::interpolate(const TDoubleVec& knots, const TDoubleVec& values, const TDoubleVec& variances) {
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
            boost::math::normal_distribution<> normal{m, sd};
            double ql{boost::math::quantile(normal, (100.0 - confidence) / 200.0)};
            double qu{boost::math::quantile(normal, (100.0 + confidence) / 200.0)};
            return {ql, qu};
        } catch (const std::exception& e) {
            LOG_ERROR("Failed calculating confidence interval: " << e.what() << ", n = " << n << ", m = " << m << ", sd = " << sd
                                                                 << ", confidence = " << confidence);
        }
        return {m, m};
    }

    return {m_MeanValue, m_MeanValue};
}

double CDecompositionComponent::meanValue(void) const {
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
            boost::math::chi_squared_distribution<> chi{n - 1.0};
            double ql{boost::math::quantile(chi, (100.0 - confidence) / 200.0)};
            double qu{boost::math::quantile(chi, (100.0 + confidence) / 200.0)};
            return std::make_pair(ql * v / (n - 1.0), qu * v / (n - 1.0));
        } catch (const std::exception& e) {
            LOG_ERROR("Failed calculating confidence interval: " << e.what() << ", n = " << n << ", confidence = " << confidence);
        }
        return {v, v};
    }
    return {m_MeanVariance, m_MeanVariance};
}

double CDecompositionComponent::meanVariance(void) const {
    return m_MeanVariance;
}

double CDecompositionComponent::heteroscedasticity(void) const {
    if (m_MeanVariance == 0.0) {
        return 0.0;
    }

    typedef CBasicStatistics::SMax<double>::TAccumulator TMaxAccumulator;

    TMaxAccumulator result;

    TSplineCRef spline = this->varianceSpline();
    for (const auto& value : spline.values()) {
        result.add(value / m_MeanVariance);
    }

    return result.count() > 0 ? result[0] : 0.0;
}

std::size_t CDecompositionComponent::maxSize(void) const {
    return std::max(m_MaxSize, MIN_MAX_SIZE);
}

CSplineTypes::EBoundaryCondition CDecompositionComponent::boundaryCondition(void) const {
    return m_BoundaryCondition;
}

CDecompositionComponent::TSplineCRef CDecompositionComponent::valueSpline(void) const {
    return m_Splines.spline(CPackedSplines::E_Value);
}

CDecompositionComponent::TSplineCRef CDecompositionComponent::varianceSpline(void) const {
    return m_Splines.spline(CPackedSplines::E_Variance);
}

uint64_t CDecompositionComponent::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_MaxSize);
    seed = CChecksum::calculate(seed, m_BoundaryCondition);
    seed = CChecksum::calculate(seed, m_Splines);
    seed = CChecksum::calculate(seed, m_MeanValue);
    return CChecksum::calculate(seed, m_MeanVariance);
}

const CDecompositionComponent::CPackedSplines& CDecompositionComponent::splines(void) const {
    return m_Splines;
}

const std::size_t CDecompositionComponent::MIN_MAX_SIZE{1u};

////// CDecompositionComponent::CPackedSplines //////

CDecompositionComponent::CPackedSplines::CPackedSplines(CSplineTypes::EType valueInterpolationType,
                                                        CSplineTypes::EType varianceInterpolationType) {
    m_Types[static_cast<std::size_t>(E_Value)] = valueInterpolationType;
    m_Types[static_cast<std::size_t>(E_Variance)] = varianceInterpolationType;
}

bool CDecompositionComponent::CPackedSplines::acceptRestoreTraverser(CSplineTypes::EBoundaryCondition boundary,
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

    return true;
}

void CDecompositionComponent::CPackedSplines::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(ESTIMATED_TAG, static_cast<int>(this->initialized()));
    if (this->initialized()) {
        inserter.insertValue(KNOTS_TAG, core::CPersistUtils::toString(m_Knots));
        inserter.insertValue(VALUES_TAG, core::CPersistUtils::toString(m_Values[0]));
        inserter.insertValue(VARIANCES_TAG, core::CPersistUtils::toString(m_Values[1]));
    }
}

void CDecompositionComponent::CPackedSplines::swap(CPackedSplines& other) {
    std::swap(m_Types, other.m_Types);
    m_Knots.swap(other.m_Knots);
    m_Values[0].swap(other.m_Values[0]);
    m_Values[1].swap(other.m_Values[1]);
    m_Curvatures[0].swap(other.m_Curvatures[0]);
    m_Curvatures[1].swap(other.m_Curvatures[1]);
}

bool CDecompositionComponent::CPackedSplines::initialized(void) const {
    return m_Knots.size() > 0;
}

void CDecompositionComponent::CPackedSplines::clear(void) {
    this->spline(E_Value).clear();
    this->spline(E_Variance).clear();
}

void CDecompositionComponent::CPackedSplines::shift(ESpline spline, double shift) {
    for (auto&& value : m_Values[static_cast<std::size_t>(spline)]) {
        value += shift;
    }
}

CDecompositionComponent::TSplineCRef CDecompositionComponent::CPackedSplines::spline(ESpline spline) const {
    return TSplineCRef(m_Types[static_cast<std::size_t>(spline)],
                       boost::cref(m_Knots),
                       boost::cref(m_Values[static_cast<std::size_t>(spline)]),
                       boost::cref(m_Curvatures[static_cast<std::size_t>(spline)]));
}

CDecompositionComponent::TSplineRef CDecompositionComponent::CPackedSplines::spline(ESpline spline) {
    return TSplineRef(m_Types[static_cast<std::size_t>(spline)],
                      boost::ref(m_Knots),
                      boost::ref(m_Values[static_cast<std::size_t>(spline)]),
                      boost::ref(m_Curvatures[static_cast<std::size_t>(spline)]));
}

const CDecompositionComponent::TFloatVec& CDecompositionComponent::CPackedSplines::knots(void) const {
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
    LOG_TRACE("types = " << core::CContainerPrinter::print(m_Types));
    LOG_TRACE("knots = " << core::CContainerPrinter::print(m_Knots));
    LOG_TRACE("values = " << core::CContainerPrinter::print(m_Values));
    LOG_TRACE("curvatures = " << core::CContainerPrinter::print(m_Curvatures));
}

uint64_t CDecompositionComponent::CPackedSplines::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_Types);
    seed = CChecksum::calculate(seed, m_Knots);
    seed = CChecksum::calculate(seed, m_Values);
    return CChecksum::calculate(seed, m_Curvatures);
}

void CDecompositionComponent::CPackedSplines::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CPackedSplines");
    core::CMemoryDebug::dynamicSize("m_Knots", m_Knots, mem);
    core::CMemoryDebug::dynamicSize("m_Values[0]", m_Values[0], mem);
    core::CMemoryDebug::dynamicSize("m_Values[1]", m_Values[1], mem);
    core::CMemoryDebug::dynamicSize("m_Curvatures[0]", m_Curvatures[0], mem);
    core::CMemoryDebug::dynamicSize("m_Curvatures[1]", m_Curvatures[1], mem);
}

std::size_t CDecompositionComponent::CPackedSplines::memoryUsage(void) const {
    std::size_t mem{core::CMemory::dynamicSize(m_Knots)};
    mem += core::CMemory::dynamicSize(m_Values[0]);
    mem += core::CMemory::dynamicSize(m_Values[1]);
    mem += core::CMemory::dynamicSize(m_Curvatures[0]);
    mem += core::CMemory::dynamicSize(m_Curvatures[1]);
    return mem;
}
}
}
