/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLoopProgress.h>

#include <algorithm>

namespace ml {
namespace core {

CLoopProgress::CLoopProgress(std::size_t size, const TProgressCallback& recordProgress, double scale)
    : m_Size{size}, m_Steps{std::min(size, STEPS)},
      m_StepProgress{scale / static_cast<double>(m_Steps)}, m_RecordProgress{recordProgress} {
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
}
}
