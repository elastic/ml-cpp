/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */
#include <core/CBufferFlushTimer.h>

#include <core/CTimeUtils.h>

#include <limits>

namespace ml {
namespace core {

CBufferFlushTimer::CBufferFlushTimer() : m_LastMaxTime(0), m_LastFlushTime(0) {
}

core_t::TTime CBufferFlushTimer::flushTime(core_t::TTime bufferDelay, core_t::TTime bufferMaxTime) {
    core_t::TTime now(CTimeUtils::now());

    if (bufferMaxTime == 0) {
        // If we get here then there's no evidence to be flushed.
        // However, downstream components might still be relying on
        // us to return a sensible time, so pretend the buffer time
        // is the same as last time.
        bufferMaxTime = m_LastMaxTime;
    }

    if (m_LastMaxTime == bufferMaxTime) {
        // Same max time
        core_t::TTime ahead(now - m_LastFlushTime);

        // If max time has been the same for bufferDelay seconds
        // flush based on elapsed real time
        if (ahead > bufferDelay) {
            // Defend against wrap
            if (bufferMaxTime - bufferDelay >= std::numeric_limits<core_t::TTime>::max() - ahead) {
                return std::numeric_limits<core_t::TTime>::max();
            }

            return bufferMaxTime - bufferDelay + ahead;
        }

        // No new flush
        return bufferMaxTime - bufferDelay;
    }

    m_LastFlushTime = now;
    m_LastMaxTime = bufferMaxTime;

    return bufferMaxTime - bufferDelay;
}
}
}
