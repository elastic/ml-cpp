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
#include "CBlockingCallCancellerThread.h"

#include <core/CLogger.h>

#include <istream>

namespace ml {
namespace controller {

CBlockingCallCancellerThread::CBlockingCallCancellerThread(core::CThread::TThreadId potentiallyBlockedThreadId,
                                                           std::istream& monitorStream)
    : m_PotentiallyBlockedThreadId(potentiallyBlockedThreadId), m_MonitorStream(monitorStream), m_Shutdown(false) {
}

void CBlockingCallCancellerThread::run(void) {
    char c;
    while (m_MonitorStream >> c) {
        if (m_Shutdown) {
            return;
        }
    }

    if (core::CThread::cancelBlockedIo(m_PotentiallyBlockedThreadId) == false) {
        LOG_WARN("Failed to cancel blocked IO in thread " << m_PotentiallyBlockedThreadId);
    }
}

void CBlockingCallCancellerThread::shutdown(void) {
    m_Shutdown = true;

    // This is to wake up the stream reading in the run() method of this object.
    // If this has an effect then the assumption is that the program is exiting
    // due to a reason other than the stream this object is monitoring ending.
    if (this->cancelBlockedIo() == false) {
        LOG_WARN("Failed to cancel blocked IO in thread " << this->currentThreadId());
    }
}
}
}
