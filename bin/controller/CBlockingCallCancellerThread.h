/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_controller_CBlockingCallCancellerThread_h
#define INCLUDED_ml_controller_CBlockingCallCancellerThread_h

#include <core/CThread.h>

#include <iosfwd>

namespace ml {
namespace controller {

//! \brief
//! Cancels blocking IO in one thread if a stream reaches end-of-file.
//!
//! DESCRIPTION:\n
//! Reads from the supplied stream until end-of-file is reached.  Then
//! cancels blocking IO calls in another thread.
//!
//! If this thread is shut down before end-of-file is reached on the
//! supplied stream then the shutdown function cancels reading of the
//! supplied stream.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Designed to handle the following sequence of events:
//! 1) Java process spawns this process
//! 2) This process creates named pipes for logging and to receive commands
//! 3) Before connecting the named pipes, the Java process dies
//!
//! In this situation this process will receive end-of-file on its STDIN,
//! but will be blocked opening one of the named pipes.  The blocking call
//! needs to be cancelled to allow this process to exit gracefully.
//!
class CBlockingCallCancellerThread : public core::CThread {
public:
    CBlockingCallCancellerThread(core::CThread::TThreadId potentiallyBlockedThreadId,
                                 std::istream& monitorStream);

protected:
    //! Called when the thread is started.
    virtual void run();

    //! Called when the thread is stopped.
    virtual void shutdown();

private:
    //! Thread ID of the thread that this object will cancel blocking IO in
    //! if it detects end-of-file on its input stream.
    core::CThread::TThreadId m_PotentiallyBlockedThreadId;

    //! Stream to monitor for end-of-file.
    std::istream& m_MonitorStream;

    //! Flag to indicate the thread should shut down
    volatile bool m_Shutdown;
};
}
}

#endif // INCLUDED_ml_controller_CBlockingCallCancellerThread_h
