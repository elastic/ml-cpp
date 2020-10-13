/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CBlockingCallCancellingStreamMonitor.h"

#include <core/CLogger.h>

#include <istream>

namespace ml {
namespace controller {

CBlockingCallCancellingStreamMonitor::CBlockingCallCancellingStreamMonitor(
    core::CThread::TThreadId potentiallyBlockedThreadId,
    std::istream& monitorStream)
    : CBlockingCallCancellerThread{potentiallyBlockedThreadId}, m_MonitorStream{monitorStream} {
}

void CBlockingCallCancellingStreamMonitor::waitForCondition() {
    char c;
    while (m_MonitorStream >> c) {
        if (this->isShutdown()) {
            return;
        }
    }
}

void CBlockingCallCancellingStreamMonitor::stopWaitForCondition() {
    // This is to wake up the stream reading in the waitForCondition() method of
    // this object.  If this has an effect then the assumption is that the
    // program is exiting due to a reason other than the stream this object is
    // monitoring ending.
    if (this->cancelBlockedIo() == false) {
        LOG_WARN(<< "Failed to cancel blocked IO in thread " << this->currentThreadId());
    }
}
}
}
