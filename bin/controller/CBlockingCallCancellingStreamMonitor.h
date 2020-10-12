/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_controller_CBlockingCallCancellingStreamMonitor_h
#define INCLUDED_ml_controller_CBlockingCallCancellingStreamMonitor_h

#include <core/CBlockingCallCancellerThread.h>

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
class CBlockingCallCancellingStreamMonitor : public core::CBlockingCallCancellerThread {
public:
    CBlockingCallCancellingStreamMonitor(core::CThread::TThreadId potentiallyBlockedThreadId,
                                         std::istream& monitorStream);

protected:
    //! Waits for end-of-file on the stream being monitored.
    void waitForCondition() override;

    //! Interrupts the wait for end-of-file.
    void stopWaitForCondition() override;

private:
    std::istream& m_MonitorStream;
};
}
}

#endif // INCLUDED_ml_controller_CBlockingCallCancellingStreamMonitor_h
