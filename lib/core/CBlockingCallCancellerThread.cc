/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CBlockingCallCancellerThread.h>

#include <core/CLogger.h>

namespace ml {
namespace core {

CBlockingCallCancellerThread::CBlockingCallCancellerThread(CThread::TThreadId potentiallyBlockedThreadId)
    : m_PotentiallyBlockedThreadId{potentiallyBlockedThreadId},
      m_Shutdown{false}, m_HasCancelledBlockingCall{false} {
}

const std::atomic_bool& CBlockingCallCancellerThread::hasCancelledBlockingCall() const {
    return m_HasCancelledBlockingCall;
}

void CBlockingCallCancellerThread::run() {
    this->waitForCondition();

    if (this->isShutdown() == false) {
        m_HasCancelledBlockingCall.store(true);
        if (CThread::cancelBlockedIo(m_PotentiallyBlockedThreadId) == false) {
            LOG_WARN(<< "Failed to cancel blocked IO in thread " << m_PotentiallyBlockedThreadId);
        }
    }
}

void CBlockingCallCancellerThread::shutdown() {
    m_Shutdown.store(true);
    this->stopWaitForCondition();
}

bool CBlockingCallCancellerThread::isShutdown() {
    return m_Shutdown.load();
}

bool CBlockingCallCancellerThread::reset() {
    if (this->isStarted()) {
        LOG_ERROR(<< "Programmatic error - cannot reset canceller thread while running");
        return false;
    }
    m_Shutdown.store(false);
    m_HasCancelledBlockingCall.store(false);
    return true;
}
}
}
