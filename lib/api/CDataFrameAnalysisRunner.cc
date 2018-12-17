/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataFrameAnalysisRunner.h>

#include <core/CDataFrame.h>
#include <core/CLogger.h>
#include <core/CScopedFastLock.h>

#include <api/CDataFrameAnalysisSpecification.h>

#include <boost/iterator/counting_iterator.hpp>

#include <algorithm>

namespace ml {
namespace api {
namespace {
std::size_t memoryLimitWithMargin(const CDataFrameAnalysisSpecification& spec) {
    return static_cast<std::size_t>(0.8 * static_cast<double>(spec.memoryLimit()) + 0.5);
}
}

CDataFrameAnalysisRunner::CDataFrameAnalysisRunner(const CDataFrameAnalysisSpecification& spec)
    : m_Spec{spec}, m_Finished{false}, m_FractionalProgress{0.0} {
}

CDataFrameAnalysisRunner::~CDataFrameAnalysisRunner() {
    this->waitToFinish();
}

void CDataFrameAnalysisRunner::computeAndSaveExecutionStrategy() {

    std::size_t numberRows{m_Spec.numberRows()};
    std::size_t numberColumns{m_Spec.numberColumns() + this->numberExtraColumns()};
    std::size_t memoryLimit{memoryLimitWithMargin(m_Spec)};

    // Find the smallest number of partitions such that the size per partition
    // is less than the memory limit.

    for (m_NumberPartitions = 1; m_NumberPartitions < numberRows; ++m_NumberPartitions) {
        std::size_t partitionNumberRows{numberRows / m_NumberPartitions};
        if (this->estimateMemoryUsage(partitionNumberRows, numberColumns) < memoryLimit) {
            break;
        }
    }

    if (m_NumberPartitions == numberRows) {
        m_Bad = true;
    } else if (m_NumberPartitions > 1) {
        // The maximum number of rows is found by binary search in the interval
        // [numberRows / m_NumberPartitions, numberRows / (m_NumberPartitions - 1)).

        m_MaximumNumberRowsPerPartition = *std::lower_bound(
            boost::make_counting_iterator(numberRows / m_NumberPartitions),
            boost::make_counting_iterator(numberRows / (m_NumberPartitions - 1)),
            memoryLimit, [&](std::size_t partitionNumberRows, std::size_t limit) {
                return this->estimateMemoryUsage(partitionNumberRows, numberColumns) < limit;
            });
    }
}

bool CDataFrameAnalysisRunner::storeDataFrameInMainMemory() const {
    return m_NumberPartitions > 1;
}

std::size_t CDataFrameAnalysisRunner::numberPartitions() const {
    return m_NumberPartitions;
}

std::size_t CDataFrameAnalysisRunner::maximumNumberRowsPerPartition() const {
    return m_MaximumNumberRowsPerPartition;
}

void CDataFrameAnalysisRunner::run(core::CDataFrame& frame) {
    if (m_Runner.joinable()) {
        LOG_INFO(<< "Already running analysis");
    } else if (m_Spec.bad()) {
        LOG_ERROR(<< "Bad specification: not running analysis");
        m_FractionalProgress.store(1.0);
        m_Finished.store(true);
    } else {
        m_FractionalProgress.store(0.0);
        m_Finished.store(false);
        m_Runner = std::thread([this, &frame]() { this->runImpl(frame); });
    }
}

void CDataFrameAnalysisRunner::waitToFinish() {
    if (m_Runner.joinable()) {
        m_Runner.join();
    }
}

bool CDataFrameAnalysisRunner::bad() const {
    return m_Bad;
}

bool CDataFrameAnalysisRunner::finished() const {
    return m_Finished.load();
}

double CDataFrameAnalysisRunner::progress() const {
    return m_FractionalProgress.load();
}

const CDataFrameAnalysisSpecification& CDataFrameAnalysisRunner::spec() const {
    return m_Spec;
}

CDataFrameAnalysisRunner::TStrVec CDataFrameAnalysisRunner::errors() const {
    core::CScopedFastLock lock(m_Mutex);
    auto result = m_Errors;
    return result;
}

void CDataFrameAnalysisRunner::setToBad() {
    m_Bad = true;
}

void CDataFrameAnalysisRunner::setToFinished() {
    m_Finished.store(true);
    m_FractionalProgress.store(1.0);
}

void CDataFrameAnalysisRunner::updateProgress(double fractionalProgress) {
    m_FractionalProgress.store(fractionalProgress);
}

void CDataFrameAnalysisRunner::addError(const std::string& error) {
    core::CScopedFastLock lock(m_Mutex);
    m_Errors.push_back(error);
}

std::size_t CDataFrameAnalysisRunner::estimateMemoryUsage(std::size_t numberRows,
                                                          std::size_t numberColumns) const {
    return core::CDataFrame::estimateMemoryUsage(m_NumberPartitions == 1,
                                                 numberRows, numberColumns) +
           this->estimateBookkeepingMemoryUsage(m_NumberPartitions, numberRows, numberColumns);
}

CDataFrameAnalysisRunnerFactory::TRunnerUPtr
CDataFrameAnalysisRunnerFactory::make(const CDataFrameAnalysisSpecification& spec) const {
    auto result = this->makeImpl(spec);
    result->computeAndSaveExecutionStrategy();
    return result;
}

CDataFrameAnalysisRunnerFactory::TRunnerUPtr
CDataFrameAnalysisRunnerFactory::make(const CDataFrameAnalysisSpecification& spec,
                                      const rapidjson::Value& params) const {
    auto result = this->makeImpl(spec, params);
    result->computeAndSaveExecutionStrategy();
    return result;
}
}
}
