/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CDataFrameAnalysisInstrumentation.h>

#include <boost/iostreams/filter/zlib.hpp>
#include <core/CTimeUtils.h>

namespace ml {
namespace api {

namespace {
const std::string STEP_TAG{"step"};
const std::string PROGRESS_TAG{"progress"};
const std::string PEAK_MEMORY_USAGE_TAG{"peak_usage_bytes"};
const std::string TYPE_TAG{"type"};
const std::string JOB_ID_TAG{"job_id"};
const std::string TIMESTAMP_TAG{"timestamp"};
const std::string MEMORY_TYPE_TAG{"analytics_memory_usage"};
const std::string ANALYSIS_TYPE_TAG{"analysis_stats"};
const std::string REGRESSION_STATS_TAG{"regression_stats"};
const std::string ITERATION_TAG{"iteration"};
const std::string HYPERPARAMETERS_TAG{"hyperparameters"};
const std::string VALIDATION_LOSS_TAG{"validation_loss"};
const std::string TIMING_STATS_TAG{"timing_stats"};
const std::string VALIDATION_LOSS_TYPE_TAG{"loss_type"};
const std::string VALIDATION_LOSS_VALUES_TAG{"values"};
const std::string VALIDATION_NUM_FOLDS_TAG{"num_folds"};
const std::string TIMING_ELAPSED_TIME_TAG{"elapsed_time"};
const std::string TIMING_ITERATION_TIME_TAG{"iteration_time"};

const std::size_t MAXIMUM_FRACTIONAL_PROGRESS{std::size_t{1}
                                              << ((sizeof(std::size_t) - 2) * 8)};
}

void CDataFrameAnalysisInstrumentation::updateMemoryUsage(std::int64_t delta) {
    std::int64_t memory{m_Memory.fetch_add(delta)};
    if (memory >= 0) {
        core::CProgramCounters::counter(this->memoryCounterType()).max(memory);
    } else {
        // Something has gone wrong with memory estimation. Trap this case
        // to avoid underflowing the peak memory usage statistic.
        LOG_WARN(<< "Memory estimate " << memory << " is negative!");
    }
}

void CDataFrameAnalysisInstrumentation::updateProgress(double fractionalProgress) {
    m_FractionalProgress.fetch_add(static_cast<std::size_t>(std::max(
        static_cast<double>(MAXIMUM_FRACTIONAL_PROGRESS) * fractionalProgress + 0.5, 1.0)));
}

void CDataFrameAnalysisInstrumentation::setToFinished() {
    m_Finished.store(true);
    m_FractionalProgress.store(MAXIMUM_FRACTIONAL_PROGRESS);
}

bool CDataFrameAnalysisInstrumentation::finished() const {
    return m_Finished.load();
}

double CDataFrameAnalysisInstrumentation::progress() const {
    return this->finished()
               ? 1.0
               : static_cast<double>(std::min(m_FractionalProgress.load(),
                                              MAXIMUM_FRACTIONAL_PROGRESS - 1)) /
                     static_cast<double>(MAXIMUM_FRACTIONAL_PROGRESS);
}

CDataFrameAnalysisInstrumentation::CDataFrameAnalysisInstrumentation(const std::string& jobId)
    : m_Finished{false}, m_FractionalProgress{0}, m_Memory{0}, m_Writer{nullptr}, m_JobId{jobId} {
}

void CDataFrameAnalysisInstrumentation::resetProgress() {
    m_FractionalProgress.store(0.0);
    m_Finished.store(false);
}

void CDataFrameAnalysisInstrumentation::writer(core::CRapidJsonConcurrentLineWriter* writer) {
    m_Writer = writer;
}

void CDataFrameAnalysisInstrumentation::nextStep(std::uint32_t step) {
    this->writeState(step);
}

void CDataFrameAnalysisInstrumentation::writeState(std::uint32_t step) {
    // this->writeProgress(step);
    std::int64_t timestamp{core::CTimeUtils::toEpochMs(core::CTimeUtils::now())};
    this->writeMemory(timestamp);
}

std::int64_t CDataFrameAnalysisInstrumentation::memory() const {
    return m_Memory.load();
}

void CDataFrameAnalysisInstrumentation::writeProgress(std::uint32_t step) {
    if (m_Writer != nullptr) {
        m_Writer->StartObject();
        m_Writer->Key(STEP_TAG);
        m_Writer->Uint(step);
        m_Writer->Key(PROGRESS_TAG);
        m_Writer->Double(this->progress());
        m_Writer->EndObject();
    }
}

void CDataFrameAnalysisInstrumentation::writeMemory(std::int64_t timestamp) {
    if (m_Writer != nullptr) {
        m_Writer->StartObject();
        m_Writer->Key(TYPE_TAG);
        m_Writer->String(MEMORY_TYPE_TAG);
        m_Writer->Key(JOB_ID_TAG);
        m_Writer->String(m_JobId);
        m_Writer->Key(TIMESTAMP_TAG);
        m_Writer->Int64(timestamp);
        m_Writer->Key(PEAK_MEMORY_USAGE_TAG);
        m_Writer->Int64(m_Memory.load());
        m_Writer->EndObject();
    }
}

const std::string& CDataFrameAnalysisInstrumentation::jobId() const {
    return m_JobId;
}

core::CRapidJsonConcurrentLineWriter* CDataFrameAnalysisInstrumentation::writer() {
    return m_Writer;
}

counter_t::ECounterTypes CDataFrameOutliersInstrumentation::memoryCounterType() {
    return counter_t::E_DFOPeakMemoryUsage;
}

counter_t::ECounterTypes CDataFrameTrainBoostedTreeInstrumentation::memoryCounterType() {
    return counter_t::E_DFTPMPeakMemoryUsage;
}

void CDataFrameOutliersInstrumentation::writeAnalysisStats(std::int64_t timestamp,
                                                           std::uint32_t /*step*/) {
    auto* writer{this->writer()};
    if (writer != nullptr) {
        writer->StartObject();
        writer->Key(JOB_ID_TAG);
        writer->String(this->jobId());
        writer->Key(TIMESTAMP_TAG);
        writer->Int64(timestamp);
        writer->EndObject();
    }
}

void CDataFrameTrainBoostedTreeInstrumentation::writeAnalysisStats(std::int64_t timestamp,
                                                                   std::uint32_t step) {
    auto* writer{this->writer()};
    if (writer != nullptr) {
        writer->StartObject();
        writer->Key(JOB_ID_TAG);
        writer->String(this->jobId());
        writer->Key(TIMESTAMP_TAG);
        writer->Int64(timestamp);
        writer->EndObject();
    }
}
}
}
