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
std::size_t memoryLimitWithSafetyMargin(const CDataFrameAnalysisSpecification& spec) {
    return static_cast<std::size_t>(0.9 * static_cast<double>(spec.memoryLimit()) + 0.5);
}

const double MAXIMUM_FRACTIONAL_PROGRESS{1024.0};
}

CDataFrameAnalysisRunner::CDataFrameAnalysisRunner(const CDataFrameAnalysisSpecification& spec,
                                                   const TErrorHandler& errorHandler)
    : m_Spec{spec}, m_Finished{false}, m_FractionalProgress{0}, m_ErrorHandler{errorHandler} {
}

CDataFrameAnalysisRunner::~CDataFrameAnalysisRunner() {
    this->waitToFinish();
}

void CDataFrameAnalysisRunner::computeAndSaveExecutionStrategy() {

    std::size_t numberRows{m_Spec.numberRows()};
    std::size_t numberColumns{m_Spec.numberColumns() + this->numberExtraColumns()};
    std::size_t memoryLimit{memoryLimitWithSafetyMargin(m_Spec)};

    LOG_TRACE(<< "memory limit = " << memoryLimit);

    // Find the smallest number of partitions such that the size per partition
    // is less than the memory limit.

    for (m_NumberPartitions = 1; m_NumberPartitions < numberRows; ++m_NumberPartitions) {
        std::size_t partitionNumberRows{numberRows / m_NumberPartitions};
        std::size_t memoryUsage{this->estimateMemoryUsage(partitionNumberRows, numberColumns)};
        LOG_TRACE(<< "partition number rows = " << partitionNumberRows);
        LOG_TRACE(<< "memory usage = " << memoryUsage);
        if (memoryUsage <= memoryLimit) {
            break;
        }
    }

    LOG_TRACE(<< "number partitions = " << m_NumberPartitions);

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

        LOG_TRACE(<< "maximum rows per partition = " << m_MaximumNumberRowsPerPartition);
    } else {
        m_MaximumNumberRowsPerPartition = numberRows;
    }
}

bool CDataFrameAnalysisRunner::storeDataFrameInMainMemory() const {
    return m_NumberPartitions == 1;
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
    return static_cast<double>(std::min(m_FractionalProgress.load(),
                                        static_cast<int>(MAXIMUM_FRACTIONAL_PROGRESS))) /
           MAXIMUM_FRACTIONAL_PROGRESS;
}

const CDataFrameAnalysisSpecification& CDataFrameAnalysisRunner::spec() const {
    return m_Spec;
}

void CDataFrameAnalysisRunner::setToBad() {
    m_Bad = true;
}

void CDataFrameAnalysisRunner::setToFinished() {
    m_Finished.store(true);
    m_FractionalProgress.store(static_cast<int>(MAXIMUM_FRACTIONAL_PROGRESS));
}

CDataFrameAnalysisRunner::TProgressRecorder CDataFrameAnalysisRunner::progressRecorder() {
    return [this](double fractionalProgress) {
        this->recordProgress(fractionalProgress);
    };
}

const CDataFrameAnalysisRunner::TErrorHandler& CDataFrameAnalysisRunner::errorHandler() const {
    return m_ErrorHandler;
}

std::size_t CDataFrameAnalysisRunner::estimateMemoryUsage(std::size_t numberRows,
                                                          std::size_t numberColumns) const {
    return core::CDataFrame::estimateMemoryUsage(this->storeDataFrameInMainMemory(),
                                                 numberRows, numberColumns) +
           this->estimateBookkeepingMemoryUsage(m_NumberPartitions, numberRows, numberColumns);
}

void CDataFrameAnalysisRunner::recordProgress(double fractionalProgress) {
    m_FractionalProgress.fetch_add(std::max(
        static_cast<int>(MAXIMUM_FRACTIONAL_PROGRESS * fractionalProgress + 0.5), 1));
}

CDataFrameAnalysisRunnerFactory::TRunnerUPtr
CDataFrameAnalysisRunnerFactory::make(const CDataFrameAnalysisSpecification& spec,
                                      const TErrorHandler& errorHandler) const {
    auto result = this->makeImpl(spec, errorHandler);
    result->computeAndSaveExecutionStrategy();
    return result;
}

CDataFrameAnalysisRunnerFactory::TRunnerUPtr
CDataFrameAnalysisRunnerFactory::make(const CDataFrameAnalysisSpecification& spec,
                                      const rapidjson::Value& params,
                                      const TErrorHandler& errorHandler) const {
    auto result = this->makeImpl(spec, params, errorHandler);
    result->computeAndSaveExecutionStrategy();
    return result;
}

void CDataFrameAnalysisRunnerFactory::defaultErrorHandler(const std::string&) {
    // No op since logging is handled where the error is emitted.
}
}
}
