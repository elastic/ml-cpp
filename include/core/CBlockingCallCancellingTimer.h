/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CBlockingCallCancellingTimer_h
#define INCLUDED_ml_core_CBlockingCallCancellingTimer_h

#include <core/CBlockingCallCancellerThread.h>
#include <core/CoreTypes.h>
#include <core/ImportExport.h>

#include <chrono>
#include <condition_variable>
#include <mutex>

namespace ml {
namespace core {

//! \brief
//! Cancels a blocking IO call after a supplied timeout has elapsed.
//!
//! DESCRIPTION:\n
//! A thread that, when started, waits for the specified number of
//! seconds, then cancels any in-progress blocking IO call in the
//! other thread specified on construction.
//!
//! If this thread is shut down before the timeout occurs then no
//! blocking IO call cancellation takes place.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Designed to handle the following sequence of events:
//! 1) Java process spawns this process
//! 2) This process creates named pipes and waits for Java to connect
//! 3) Java never connects for some reason
//!
//! We want to unblock this process when this happens so that it can
//! delete the named pipe and exit.
//!
class CORE_EXPORT CBlockingCallCancellingTimer : public CBlockingCallCancellerThread {
public:
    //! Default timeout if not specified.
    static const ml::core_t::TTime DEFAULT_TIMEOUT_SECONDS;

public:
    explicit CBlockingCallCancellingTimer(CThread::TThreadId potentiallyBlockedThreadId,
                                          std::chrono::seconds timeoutSeconds = std::chrono::seconds{
                                              DEFAULT_TIMEOUT_SECONDS});

protected:
    //! Derived classes must implement this such that it waits for the
    //! appropriate indication that the blocking call should be cancelled,
    //! or until it is told to stop waiting by stopWaitForCondition().
    virtual void waitForCondition();

    //! Derived classes must implement this such that when called it causes
    //! the waitForCondition() method to return false immediately.
    virtual void stopWaitForCondition();

private:
    using TMutexUniqueLock = std::unique_lock<std::mutex>;

private:
    //! Mutex required by the condition variable.
    std::mutex m_Mutex;

    //! Condition variable to provide a reliably interruptible sleep.
    std::condition_variable m_Condition;

    //! How many seconds to wait before cancelling the blocking call.
    std::chrono::seconds m_TimeoutSeconds;
};
}
}

#endif // INCLUDED_ml_core_CBlockingCallCancellingTimer_h
