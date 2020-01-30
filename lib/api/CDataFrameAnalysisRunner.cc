/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataFrameAnalysisRunner.h>

#include <core/CDataFrame.h>
#include <core/CJsonStatePersistInserter.h>
#include <core/CLogger.h>
#include <core/CStateCompressor.h>

#include <api/CDataFrameAnalysisSpecification.h>
#include <api/CMemoryUsageEstimationResultJsonWriter.h>
#include <api/CSingleStreamDataAdder.h>
#include <api/ElasticsearchStateIndex.h>

#include <boost/iterator/counting_iterator.hpp>

#include <algorithm>
#include <cstddef>

namespace ml {
namespace api {
namespace {
using TBoolVec = std::vector<bool>;

std::size_t maximumNumberPartitions(const CDataFrameAnalysisSpecification& spec) {
    // We limit the maximum number of partitions to rows^(1/2) because very
    // large numbers of partitions are going to be slow and it is better to tell
    // user to allocate more resources for the job in this case.
    return static_cast<std::size_t>(std::sqrt(static_cast<double>(spec.numberRows())) + 0.5);
}
}

CDataFrameAnalysisRunner::CDataFrameAnalysisRunner(const CDataFrameAnalysisSpecification& spec)
    : m_Spec{spec} {
}

CDataFrameAnalysisRunner::~CDataFrameAnalysisRunner() {
    this->waitToFinish();
}

TBoolVec CDataFrameAnalysisRunner::columnsForWhichEmptyIsMissing(const TStrVec& fieldNames) const {
    return TBoolVec(fieldNames.size(), false);
}

void CDataFrameAnalysisRunner::estimateMemoryUsage(CMemoryUsageEstimationResultJsonWriter& writer) const {
    std::size_t numberRows{m_Spec.numberRows()};
    std::size_t numberColumns{m_Spec.numberColumns() + this->numberExtraColumns()};
    std::size_t maxNumberPartitions{maximumNumberPartitions(m_Spec)};
    if (maxNumberPartitions == 0) {
        writer.write("0", "0");
        return;
    }
    std::size_t expectedMemoryWithoutDisk{
        this->estimateMemoryUsage(numberRows, numberRows, numberColumns)};
    std::size_t expectedMemoryWithDisk{this->estimateMemoryUsage(
        numberRows, numberRows / maxNumberPartitions, numberColumns)};
    auto roundUpToNearestKilobyte = [](std::size_t bytes) {
        return std::to_string((bytes + 1024 - 1) / 1024) + "kB";
    };
    writer.write(roundUpToNearestKilobyte(expectedMemoryWithoutDisk),
                 roundUpToNearestKilobyte(expectedMemoryWithDisk));
}

void CDataFrameAnalysisRunner::computeAndSaveExecutionStrategy() {

    std::size_t numberRows{m_Spec.numberRows()};
    std::size_t numberColumns{m_Spec.numberColumns() + this->numberExtraColumns()};
    std::size_t memoryLimit{m_Spec.memoryLimit()};

    LOG_TRACE(<< "memory limit = " << memoryLimit);

    // Find the smallest number of partitions such that the size per partition
    // is less than the memory limit.

    std::size_t maxNumberPartitions{maximumNumberPartitions(m_Spec)};
    std::size_t memoryUsage{0};

    for (m_NumberPartitions = 1; m_NumberPartitions < maxNumberPartitions; ++m_NumberPartitions) {
        std::size_t partitionNumberRows{numberRows / m_NumberPartitions};
        memoryUsage = this->estimateMemoryUsage(numberRows, partitionNumberRows, numberColumns);
        LOG_TRACE(<< "partition number rows = " << partitionNumberRows);
        LOG_TRACE(<< "memory usage = " << memoryUsage);
        if (memoryUsage <= memoryLimit) {
            break;
        }
        // If we are not allowed to spill over to disk then only one partition
        // is possible.
        if (m_Spec.diskUsageAllowed() == false) {
            LOG_TRACE(<< "stop partition number computation since disk usage is turned off");
            break;
        }
    }

    LOG_TRACE(<< "number partitions = " << m_NumberPartitions);

    if (memoryUsage > memoryLimit) {
        auto roundMb = [](std::size_t memory) {
            return 0.01 * static_cast<double>((100 * memory) / (1024 * 1024));
        };

        // Report rounded up to the nearest MB.
        HANDLE_FATAL(<< "Input error: memory limit " << roundMb(memoryLimit)
                     << "MB is too low to perform analysis. You need to give the process"
                     << " at least " << std::ceil(roundMb(memoryUsage))
                     << "MB, but preferably more.");

    } else if (m_NumberPartitions > 1) {
        // The maximum number of rows is found by binary search in the interval
        // [numberRows / m_NumberPartitions, numberRows / (m_NumberPartitions - 1)).

        m_MaximumNumberRowsPerPartition = *std::lower_bound(
            boost::make_counting_iterator(numberRows / m_NumberPartitions),
            boost::make_counting_iterator(numberRows / (m_NumberPartitions - 1)),
            memoryLimit, [&](std::size_t partitionNumberRows, std::size_t limit) {
                return this->estimateMemoryUsage(numberRows, partitionNumberRows,
                                                 numberColumns) < limit;
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
    } else {
        this->instrumentation().resetProgress();
        m_Runner = std::thread([&frame, this]() {
            this->runImpl(frame);
            this->instrumentation().setToFinished();
        });
    }
}

void CDataFrameAnalysisRunner::waitToFinish() {
    if (m_Runner.joinable()) {
        m_Runner.join();
    }
}

const CDataFrameAnalysisSpecification& CDataFrameAnalysisRunner::spec() const {
    return m_Spec;
}

std::size_t CDataFrameAnalysisRunner::estimateMemoryUsage(std::size_t totalNumberRows,
                                                          std::size_t partitionNumberRows,
                                                          std::size_t numberColumns) const {
    return core::CDataFrame::estimateMemoryUsage(this->storeDataFrameInMainMemory(),
                                                 totalNumberRows, numberColumns) +
           this->estimateBookkeepingMemoryUsage(m_NumberPartitions, totalNumberRows,
                                                partitionNumberRows, numberColumns);
}

CDataFrameAnalysisRunner::TStatePersister CDataFrameAnalysisRunner::statePersister() {
    return [this](std::function<void(core::CStatePersistInserter&)> persistFunction) -> void {
        auto persister = m_Spec.persister();
        if (persister != nullptr) {
            core::CStateCompressor compressor(*persister);
            auto persistStream = compressor.addStreamed(
                ML_STATE_INDEX, getStateId(m_Spec.jobId(), m_Spec.analysisName()));
            {
                core::CJsonStatePersistInserter inserter{*persistStream};
                persistFunction(inserter);
            }
            if (compressor.streamComplete(persistStream, true) == false ||
                persistStream->bad()) {
                LOG_ERROR(<< "Failed to complete last persistence stream");
            }
        }
    };
}

CDataFrameAnalysisRunner::TInferenceModelDefinitionUPtr
CDataFrameAnalysisRunner::inferenceModelDefinition(const TStrVec& /*fieldNames*/,
                                                   const TStrVecVec& /*categoryNames*/) const {
    return TInferenceModelDefinitionUPtr();
}

CDataFrameAnalysisRunnerFactory::TRunnerUPtr
CDataFrameAnalysisRunnerFactory::make(const CDataFrameAnalysisSpecification& spec) const {
    auto result = this->makeImpl(spec);
    result->computeAndSaveExecutionStrategy();
    return result;
}

CDataFrameAnalysisRunnerFactory::TRunnerUPtr
CDataFrameAnalysisRunnerFactory::make(const CDataFrameAnalysisSpecification& spec,
                                      const rapidjson::Value& jsonParameters) const {
    auto result = this->makeImpl(spec, jsonParameters);
    result->computeAndSaveExecutionStrategy();
    return result;
}
}
}
