/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CBlockingCallCancellerThread_h
#define INCLUDED_ml_core_CBlockingCallCancellerThread_h

#include <core/CThread.h>
#include <core/ImportExport.h>

#include <atomic>

namespace ml {
namespace core {

//! \brief
//! Abstract base for classes that cancels blocking IO on some event.
//!
//! DESCRIPTION:\n
//! Waits for some condition defined in the concrete derived class, then
//! cancels blocking IO calls in another thread.
//!
//! If this thread is shut down before the condition occurs then the
//! shutdown function cancels the wait for the condition.
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
class CORE_EXPORT CBlockingCallCancellerThread : public CThread {
public:
    CBlockingCallCancellerThread(CThread::TThreadId potentiallyBlockedThreadId);

    //! Has this object cancelled the blocking call?
    const std::atomic_bool& hasCancelledBlockingCall() const;

    //! Reset the object so that it can be reused.  The thread must be
    //! stopped when this method is called.
    bool reset();

protected:
    //! Called when the thread is started.
    void run() override;

    //! Called when the thread is stopped.
    void shutdown() override;

    //! Has the thread been stopped?
    bool isShutdown();

    //! Derived classes must implement this such that it waits for the
    //! appropriate indication that the blocking call should be cancelled,
    //! or until it is told to stop waiting by stopWaitForCondition().
    virtual void waitForCondition() = 0;

    //! Derived classes must implement this such that when called it causes
    //! the waitForCondition() method to return false immediately.
    virtual void stopWaitForCondition() = 0;

private:
    //! Thread ID of the thread that this object will cancel blocking IO in
    //! if it detects end-of-file on its input stream.
    CThread::TThreadId m_PotentiallyBlockedThreadId;

    //! Flag to indicate the monitoring thread should shut down.
    std::atomic_bool m_Shutdown;

    //! Flag to indicate that an attempt to cancel blocking calls in the
    //! monitored thread has been made.
    std::atomic_bool m_HasCancelledBlockingCall;
};
}
}

#endif // INCLUDED_ml_core_CBlockingCallCancellerThread_h
