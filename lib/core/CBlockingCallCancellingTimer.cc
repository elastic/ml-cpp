/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CBlockingCallCancellingTimer.h>

namespace ml {
namespace core {

const core_t::TTime CBlockingCallCancellingTimer::DEFAULT_TIMEOUT_SECONDS{300};

CBlockingCallCancellingTimer::CBlockingCallCancellingTimer(CThread::TThreadId potentiallyBlockedThreadId,
                                                           std::chrono::seconds timeoutSeconds)
    : CBlockingCallCancellerThread{potentiallyBlockedThreadId}, m_TimeoutSeconds{
                                                                    std::move(timeoutSeconds)} {
}

void CBlockingCallCancellingTimer::waitForCondition() {
    std::chrono::steady_clock::time_point cancellationTime{
        std::chrono::steady_clock::now() + m_TimeoutSeconds};
    TMutexUniqueLock lock{m_Mutex};
    m_Condition.wait_until(lock, cancellationTime,
                           [this]() { return this->isShutdown(); });
}

void CBlockingCallCancellingTimer::stopWaitForCondition() {
    TMutexUniqueLock lock{m_Mutex};
    m_Condition.notify_one();
}
}
}
