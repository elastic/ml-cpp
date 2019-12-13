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
    std::int64_t memory{m_Memory.fetch_add(delta)};
    if (memory >= 0) {
        core::CProgramCounters::counter(this->memoryCounterType()).max(memory);
    } else {
        // Something has gone wrong with memory estimation. Trap this case
        // to avoid underflowing the peak memory usage statistic.
        LOG_WARN(<< "Memory estimate " << memory << " is negative!");
    }
}

void CDataFrameAnalysisState::updateProgress(double fractionalProgress) {
    m_FractionalProgress.fetch_add(static_cast<std::size_t>(std::max(
        static_cast<double>(MAXIMUM_FRACTIONAL_PROGRESS) * fractionalProgress + 0.5, 1.0)));
}

void CDataFrameAnalysisState::setToFinished() {
    m_Finished.store(true);
    m_FractionalProgress.store(MAXIMUM_FRACTIONAL_PROGRESS);
}

bool CDataFrameAnalysisState::finished() const {
    return m_Finished.load();
}

double CDataFrameAnalysisState::progress() const {
    return this->finished()
               ? 1.0
               : static_cast<double>(std::min(m_FractionalProgress.load(),
                                              MAXIMUM_FRACTIONAL_PROGRESS - 1)) /
                     static_cast<double>(MAXIMUM_FRACTIONAL_PROGRESS);
}

CDataFrameAnalysisState::CDataFrameAnalysisState()
    : m_Memory(0), m_FractionalProgress(0), m_Finished(false) {
}

void CDataFrameAnalysisState::resetProgress() {
    m_FractionalProgress.store(0.0);
    m_Finished.store(false);
}
}
}
