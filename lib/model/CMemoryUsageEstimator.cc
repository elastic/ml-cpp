/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CMemoryUsageEstimator.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CPersistUtils.h>
#include <core/CProgramCounters.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>

#include <boost/numeric/conversion/bounds.hpp>

#include <iostream>

namespace ml {
namespace model {

namespace {
const std::size_t MAXIMUM_ESTIMATES_BEFORE_NEW_VALUE(10);
const std::string VALUES_TAG("a");
}

CMemoryUsageEstimator::CMemoryUsageEstimator()
    : m_Values(2 * E_NumberPredictors),
      // Initialise this so that the first estimate triggers a calculation
      m_NumEstimatesSinceValue(MAXIMUM_ESTIMATES_BEFORE_NEW_VALUE - 1) {
}

CMemoryUsageEstimator::TOptionalSize
CMemoryUsageEstimator::estimate(const TSizeArray& predictors) {
    using TDoubleArray = boost::array<double, E_NumberPredictors>;

    if (m_Values.size() < E_NumberPredictors) {
        return TOptionalSize();
    }

    if (m_NumEstimatesSinceValue >= MAXIMUM_ESTIMATES_BEFORE_NEW_VALUE) {
        return TOptionalSize();
    }

    std::size_t last = m_Values.size() - 1;
    TDoubleArray x0;
    for (std::size_t i = 0u; i < m_Values[last].first.size(); ++i) {
        x0[i] = static_cast<double>(m_Values[last].first[i]);
    }
    double c0 = static_cast<double>(m_Values[last].second);

    bool origin = true;
    for (std::size_t i = 0u; i < predictors.size(); ++i) {
        origin &= (predictors[i] == 0);
        if (predictors[i] - static_cast<size_t>(x0[i]) >
            this->maximumExtrapolation(static_cast<EComponent>(i))) {
            LOG_TRACE(<< "Sample too big for variance of predictor(" << i
                      << "): " << predictors[i] << " > "
                      << this->maximumExtrapolation(static_cast<EComponent>(i)));
            return TOptionalSize();
        }
    }
    if (origin) {
        return TOptionalSize();
    }

    // Ideally we'd use NNLS here, but the optimization problem
    // requires more work to solve.

    Eigen::MatrixXd X(m_Values.size(), static_cast<std::size_t>(E_NumberPredictors));
    Eigen::VectorXd y(m_Values.size());
    for (std::size_t i = 0u; i < m_Values.size(); i++) {
        for (std::size_t j = 0u; j < E_NumberPredictors; ++j) {
            X(i, j) = static_cast<double>(m_Values[i].first[j]) - x0[j];
        }
        y(i) = static_cast<double>(m_Values[i].second) - c0;
    }
    Eigen::MatrixXd theta =
        X.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);

    double predicted = c0;
    for (std::size_t i = 0u; i < E_NumberPredictors; ++i) {
        predicted += std::max(theta(i), 0.0) *
                     (static_cast<double>(predictors[i]) - x0[i]);
    }
    std::size_t mem = static_cast<std::size_t>(predicted + 0.5);
    ++m_NumEstimatesSinceValue;

    ++core::CProgramCounters::counter(counter_t::E_TSADNumberMemoryUsageEstimates);
    return TOptionalSize(mem);
}

void CMemoryUsageEstimator::addValue(const TSizeArray& predictors, std::size_t memory) {
    LOG_TRACE(<< "Add Value for " << core::CContainerPrinter::print(predictors)
              << ": " << memory);

    m_NumEstimatesSinceValue = 0;

    if (m_Values.size() == m_Values.capacity()) {
        // Replace closest.
        std::size_t closest = 0u;
        std::size_t closestDistance = boost::numeric::bounds<std::size_t>::highest();
        for (std::size_t i = 0u; closestDistance > 0 && i < m_Values.size(); ++i) {
            std::size_t distance = 0u;
            for (std::size_t j = 0u; j < predictors.size(); ++j) {
                distance += std::max(m_Values[i].first[j], predictors[j]) -
                            std::min(m_Values[i].first[j], predictors[j]);
            }
            if (distance < closestDistance) {
                closest = i;
                closestDistance = distance;
            }
        }
        m_Values.erase(m_Values.begin() + closest);
    }
    m_Values.push_back(TSizeArraySizePr(predictors, memory));
    ++core::CProgramCounters::counter(counter_t::E_TSADNumberMemoryUsageChecks);
}

void CMemoryUsageEstimator::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CMemoryUsageEstimator");
    core::CMemoryDebug::dynamicSize("m_Values", m_Values, mem);
}

std::size_t CMemoryUsageEstimator::memoryUsage() const {
    return core::CMemory::dynamicSize(m_Values);
}

void CMemoryUsageEstimator::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    core::CPersistUtils::persist(VALUES_TAG, m_Values, inserter);
}

bool CMemoryUsageEstimator::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        if (name == VALUES_TAG) {
            if (!core::CPersistUtils::restore(VALUES_TAG, m_Values, traverser)) {
                LOG_ERROR(<< "Failed to restore values");
                return false;
            }
        }
    } while (traverser.next());

    return true;
}

std::size_t CMemoryUsageEstimator::maximumExtrapolation(EComponent component) const {
    std::size_t min = boost::numeric::bounds<std::size_t>::highest();
    std::size_t max = boost::numeric::bounds<std::size_t>::lowest();
    for (std::size_t i = 0u; i < m_Values.size(); ++i) {
        min = std::max(min, m_Values[i].first[component]);
        max = std::max(max, m_Values[i].first[component]);
    }
    return 2 * (max - min);
}

} // model
} // ml
