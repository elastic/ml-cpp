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
const std::string LOOP_RANGE_TAG{"loop_size_tag"};
const std::string PROGRESS_STEPS_TAG{"progress_steps_tag"};
const std::string CURRENT_STEP_PROGRESS_TAG{"current_step_progress_tag"};
const std::string LOOP_POS_TAG{"loop_pos_tag"};
}

CLoopProgress::CLoopProgress()
    : m_Range{std::numeric_limits<std::size_t>::max()}, m_Steps{1},
      m_StepProgress{1.0}, m_RecordProgress{noop} {
}

CLoopProgress::CLoopProgress(std::size_t range,
                             const TProgressCallback& recordProgress,
                             double scale,
                             std::size_t steps)
    : m_Range{range}, m_Steps{std::min(range, steps)},
      m_StepProgress{scale / static_cast<double>(m_Steps)}, m_RecordProgress{recordProgress} {
}

void CLoopProgress::progressCallback(const TProgressCallback& recordProgress) {
    m_RecordProgress = recordProgress;
}

void CLoopProgress::increment(std::size_t i) {
    m_Pos += i;
    if (m_Steps * m_Pos + 1 > (m_LastProgress + 1) * m_Range) {
        // Account for the fact that if i is large we may have jumped several steps.
        std::size_t stride{m_Steps * std::min(m_Pos, m_Range) / m_Range - m_LastProgress};
        m_RecordProgress(static_cast<double>(stride) * m_StepProgress);
        m_LastProgress += stride;
    }
}

void CLoopProgress::incrementRange(int i) {
    // This function deals with the case that the number of iterations of the "loop"
    // are changed after the CLoopProgress object is initialized. The main task we
    // need to perform is to record progress and update last progress point. We treat
    // progress as monotonic, so if the range is increased we'll simply stick at the
    // current progress until some number of iterations have passed. However, if the
    // range is reduced our fractional progress is now m_Pos / m_Range. If this is
    // larger than the next step at which we need to output progress, i.e. larger than
    // (m_LastProgress + 1) / m_Steps, then we output the new progress and update the
    // last progress to the corresponding proportion of steps. We also always cap the
    // number of progress steps by the loop range so must reduce this if necessary.

    std::size_t steps{m_Steps};
    m_Range += i;
    m_Steps = std::min(m_Range, m_Steps);
    // Check if we need to advance progress to reflect a lower range.
    if (steps * m_Pos + 1 > (m_LastProgress + 1) * m_Range) {
        std::size_t stride{steps * std::min(m_Pos, m_Range) / m_Range - m_LastProgress};
        m_RecordProgress(static_cast<double>(stride) * m_StepProgress);
        m_LastProgress = m_Steps * std::min(m_Pos, m_Range) / m_Range;
    }
    m_StepProgress *= static_cast<double>(steps) / static_cast<double>(m_Steps);
}

void CLoopProgress::resumeRestored() {
    // This outputs progress and updates m_LastProgress to the correct value.
    this->increment(0);
}

std::uint64_t CLoopProgress::checksum() const {
    std::uint64_t seed{core::CHashing::hashCombine(
        static_cast<std::uint64_t>(m_Range), static_cast<std::uint64_t>(m_Steps))};
    std::hash<std::string> stringHasher;
    seed = core::CHashing::hashCombine(
        seed, stringHasher(core::CStringUtils::typeToStringPrecise(
                  m_StepProgress, core::CIEEE754::E_DoublePrecision)));
    return core::CHashing::hashCombine(seed, static_cast<std::uint64_t>(m_Pos));
}

void CLoopProgress::acceptPersistInserter(CStatePersistInserter& inserter) const {
    inserter.insertValue(LOOP_RANGE_TAG, m_Range);
    inserter.insertValue(PROGRESS_STEPS_TAG, m_Steps);
    inserter.insertValue(CURRENT_STEP_PROGRESS_TAG, m_StepProgress,
                         core::CIEEE754::E_DoublePrecision);
    inserter.insertValue(LOOP_POS_TAG, m_Pos);
    // m_LastProgress is not persisted because when restoring we will have never
    // recorded progress.
}

bool CLoopProgress::acceptRestoreTraverser(CStateRestoreTraverser& traverser) {
    do {
        const std::string& name{traverser.name()};
        RESTORE_BUILT_IN(LOOP_RANGE_TAG, m_Range)
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
