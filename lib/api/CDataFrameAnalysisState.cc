/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataFrameAnalysisState.h>

namespace ml {
namespace api {

namespace {
const std::size_t MAXIMUM_FRACTIONAL_PROGRESS{std::size_t{1}
                                              << ((sizeof(std::size_t) - 2) * 8)};
}

void CDataFrameAnalysisState::updateMemoryUsage(std::int64_t delta) {
    std::int64_t memory{m_InternalState.s_Memory.fetch_add(delta)};
    if (memory >= 0) {
        core::CProgramCounters::counter(this->memoryCounterType()).max(memory);
    } else {
        // Something has gone wrong with memory estimation. Trap this case
        // to avoid underflowing the peak memory usage statistic.
        LOG_WARN(<< "Memory estimate " << memory << " is negative!");
    }
}

void CDataFrameAnalysisState::updateProgress(double fractionalProgress) {
    m_InternalState.s_FractionalProgress.fetch_add(static_cast<std::size_t>(std::max(
        static_cast<double>(MAXIMUM_FRACTIONAL_PROGRESS) * fractionalProgress + 0.5, 1.0)));
}

void CDataFrameAnalysisState::SInternalState::setToFinished() {
    s_Finished.store(true);
    s_FractionalProgress.store(MAXIMUM_FRACTIONAL_PROGRESS);
}

bool CDataFrameAnalysisState::SInternalState::finished() const {
    return s_Finished.load();
}

void CDataFrameAnalysisState::setToFinished() {
    this->m_InternalState.setToFinished();
}

bool CDataFrameAnalysisState::finished() const {
    return this->m_InternalState.finished();
}

double CDataFrameAnalysisState::progress() const {
    return this->m_InternalState.progress();
}

double CDataFrameAnalysisState::SInternalState::progress() const {
    return this->finished()
               ? 1.0
               : static_cast<double>(std::min(s_FractionalProgress.load(),
                                              MAXIMUM_FRACTIONAL_PROGRESS - 1)) /
                     static_cast<double>(MAXIMUM_FRACTIONAL_PROGRESS);
}

CDataFrameAnalysisState::CDataFrameAnalysisState()
    : m_InternalState(), m_Writer{nullptr} {
}

void CDataFrameAnalysisState::resetProgress() {
    m_InternalState.s_FractionalProgress.store(0.0);
    m_InternalState.s_Finished.store(false);
}

void CDataFrameAnalysisState::writer(core::CRapidJsonConcurrentLineWriter* writer) {
    m_Writer = writer;
}

void CDataFrameAnalysisState::nextStep(std::size_t step) {
    //    CDataFrameAnalysisStateInterface::nextStep(size);
    m_StateQueue.tryPush(SInternalState(m_InternalState));
    if (m_Writer != nullptr) {
        while (m_StateQueue.size() > 0) {
            this->writeState(step, m_StateQueue.pop());
        }
    }
}

void CDataFrameAnalysisState::writeState(std::size_t step,
                                         CDataFrameAnalysisState::SInternalState&& state) {
    state.writeProgress(step, *m_Writer);
    state.writeMemory(step, *m_Writer);
}

static const char* const STEP_TAG = "step";

static const char* const PROGRESS_TAG = "progress";

void CDataFrameAnalysisState::SInternalState::writeProgress(std::uint32_t step,
                                                            core::CRapidJsonConcurrentLineWriter& writer) {
    writer.StartObject();
    writer.Key(STEP_TAG);
    writer.Uint(step);
    writer.Key(PROGRESS_TAG);
    writer.Double(this->progress());
    writer.EndObject();
}

static const char* const PEAK_MEMORY_USAGE_TAG = "peak_memory_usage";

void CDataFrameAnalysisState::SInternalState::writeMemory(std::uint32_t step,
                                                          core::CRapidJsonConcurrentLineWriter& writer) {
    writer.StartObject();
    writer.Key(STEP_TAG);
    writer.Uint(step);
    writer.Key(PEAK_MEMORY_USAGE_TAG);
    writer.Double(s_Memory.load());
    writer.EndObject();
}

CDataFrameAnalysisState::SInternalState::SInternalState(const CDataFrameAnalysisState::SInternalState& other)
    : s_Memory(other.s_Memory.load()),
      s_FractionalProgress(other.s_FractionalProgress.load()),
      s_Finished(other.s_Finished.load()) {
}

CDataFrameAnalysisState::SInternalState& CDataFrameAnalysisState::SInternalState::
operator=(const CDataFrameAnalysisState::SInternalState& other) {
    s_Memory.store(other.s_Memory.load());
    s_FractionalProgress.store(other.s_FractionalProgress.load());
    s_Finished.store(other.s_Finished.load());
    return *this;
}
}
}
