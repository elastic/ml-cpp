/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLoopProgress.h>

#include <core/CHashing.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <algorithm>
#include <functional>

namespace ml {
namespace core {
namespace {
const std::size_t STEPS{32};
const std::string LOOP_SIZE_TAG{"loop_size_tag"};
const std::string PROGRESS_STEPS_TAG{"progress_steps_tag"};
const std::string CURRENT_STEP_PROGRESS_TAG{"current_step_progress_tag"};
const std::string LOOP_POS_TAG{"loop_pos_tag"};
const std::hash<std::string> stringHasher;
}

CLoopProgress::CLoopProgress()
    : m_Size{std::numeric_limits<std::size_t>::max()}, m_Steps{1},
      m_StepProgress{1.0}, m_RecordProgress{noop} {
}

CLoopProgress::CLoopProgress(std::size_t size, const TProgressCallback& recordProgress, double scale)
    : m_Size{size}, m_Steps{std::min(size, STEPS)},
      m_StepProgress{scale / static_cast<double>(m_Steps)}, m_RecordProgress{recordProgress} {
}

void CLoopProgress::attach(const TProgressCallback& recordProgress) {
    m_RecordProgress = recordProgress;
}

void CLoopProgress::increment(std::size_t i) {
    m_Pos += i;

    if (m_Steps * m_Pos + 1 > (m_LastProgress + 1) * m_Size) {
        // Account for the fact that if i is large we may have jumped several steps.
        std::size_t stride{m_Steps * std::min(m_Pos, m_Size) / m_Size - m_LastProgress};

        m_RecordProgress(static_cast<double>(stride) * m_StepProgress);
        m_LastProgress += stride;
    }
}

void CLoopProgress::resumeRestored() {
    this->increment(0);
}

std::uint64_t CLoopProgress::checksum() const {
    std::uint64_t seed{core::CHashing::hashCombine(
        static_cast<std::uint64_t>(m_Size), static_cast<std::uint64_t>(m_Steps))};
    seed = core::CHashing::hashCombine(
        seed, stringHasher(core::CStringUtils::typeToStringPrecise(
                  m_StepProgress, core::CIEEE754::E_DoublePrecision)));
    return core::CHashing::hashCombine(seed, static_cast<std::uint64_t>(m_Pos));
}

void CLoopProgress::acceptPersistInserter(CStatePersistInserter& inserter) const {
    inserter.insertValue(LOOP_SIZE_TAG, m_Size);
    inserter.insertValue(PROGRESS_STEPS_TAG, m_Steps);
    inserter.insertValue(CURRENT_STEP_PROGRESS_TAG, m_StepProgress,
                         core::CIEEE754::E_DoublePrecision);
    inserter.insertValue(LOOP_POS_TAG, m_Pos);
}

bool CLoopProgress::acceptRestoreTraverser(CStateRestoreTraverser& traverser) {
    do {
        const std::string& name{traverser.name()};
        RESTORE_BUILT_IN(LOOP_SIZE_TAG, m_Size)
        RESTORE_BUILT_IN(PROGRESS_STEPS_TAG, m_Steps)
        RESTORE_BUILT_IN(CURRENT_STEP_PROGRESS_TAG, m_StepProgress)
        RESTORE_BUILT_IN(LOOP_POS_TAG, m_Pos)
    } while (traverser.next());
    return true;
}

void CLoopProgress::noop(double) {
}
}
}
