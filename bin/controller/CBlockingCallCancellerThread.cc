/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CBlockingCallCancellerThread.h"

#include <core/CLogger.h>

#include <istream>

namespace ml {
namespace controller {

CBlockingCallCancellerThread::CBlockingCallCancellerThread(core::CThread::TThreadId potentiallyBlockedThreadId, std::istream& monitorStream)
    : m_PotentiallyBlockedThreadId(potentiallyBlockedThreadId), m_MonitorStream(monitorStream), m_Shutdown(false) {
}

void CBlockingCallCancellerThread::run() {
    char c;
    while (m_MonitorStream >> c) {
        if (m_Shutdown) {
            return;
        }
    }

    if (core::CThread::cancelBlockedIo(m_PotentiallyBlockedThreadId) == false) {
        LOG_WARN(<< "Failed to cancel blocked IO in thread " << m_PotentiallyBlockedThreadId);
    }
}

void CBlockingCallCancellerThread::shutdown() {
    m_Shutdown = true;

    // This is to wake up the stream reading in the run() method of this object.
    // If this has an effect then the assumption is that the program is exiting
    // due to a reason other than the stream this object is monitoring ending.
    if (this->cancelBlockedIo() == false) {
        LOG_WARN(<< "Failed to cancel blocked IO in thread " << this->currentThreadId());
    }
}
}
}
