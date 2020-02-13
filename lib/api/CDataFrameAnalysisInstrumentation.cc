/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataFrameAnalysisInstrumentation.h>

namespace ml {
namespace api {

namespace {
const std::string STEP_TAG{"step"};
const std::string PROGRESS_TAG{"progress"};
const std::string PEAK_MEMORY_USAGE_TAG{"peak_memory_usage"};

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

CDataFrameAnalysisInstrumentation::CDataFrameAnalysisInstrumentation()
    : m_Finished{false}, m_FractionalProgress{0}, m_Memory{0}, m_Writer{nullptr} {
}

void CDataFrameAnalysisInstrumentation::resetProgress() {
    m_FractionalProgress.store(0.0);
    m_Finished.store(false);
}

void CDataFrameAnalysisInstrumentation::writer(core::CRapidJsonConcurrentLineWriter* writer) {
    m_Writer = writer;
}

void CDataFrameAnalysisInstrumentation::nextStep(std::uint32_t /*step*/) {
    // TODO reactivate state writing, once the Java backend can accept it
    //    this->writeState(step);
}

void CDataFrameAnalysisInstrumentation::writeState(std::uint32_t step) {
    this->writeProgress(step);
    this->writeMemory(step);
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

void CDataFrameAnalysisInstrumentation::writeMemory(std::uint32_t step) {
    if (m_Writer != nullptr) {
        m_Writer->StartObject();
        m_Writer->Key(STEP_TAG);
        m_Writer->Uint(step);
        m_Writer->Key(PEAK_MEMORY_USAGE_TAG);
        m_Writer->Int64(m_Memory.load());
        m_Writer->EndObject();
    }
}

counter_t::ECounterTypes CDataFrameOutliersInstrumentation::memoryCounterType() {
    return counter_t::E_DFOPeakMemoryUsage;
}

counter_t::ECounterTypes CDataFrameTrainBoostedTreeInstrumentation::memoryCounterType() {
    return counter_t::E_DFTPMPeakMemoryUsage;
}
}
}
