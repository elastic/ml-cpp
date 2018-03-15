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
#include <core/CThread.h>

#include <core/CLogger.h>
#include <core/CScopedLock.h>

#include <errno.h>
#include <signal.h>
#include <string.h>


namespace {

//! Handler for signals that does nothing but will, unlike ignoring a signal,
//! cause calls to be interrupted with EINTR.  The idea is that an open() or
//! read() call that's blocking in our IO code will be interrupted to execute
//! this function, and when control returns to open() or read() it will
//! return a failure code and set errno to EINTR.  This gives the code
//! surrounding the blocking call a chance to not call it again but instead
//! do something different.
void noOpHandler(int /*sig*/) {}

//! Use SIGIO for waking up blocking calls.  The same handler will be used in
//! all threads, so there's an assumption here that having some other sort of
//! handling for SIGIO is not important to any other thread in the process.
//! That will be true of Ml code.  If a 3rd party library relied on SIGIO
//! handling then we could change the signal we use in this class to another
//! (maybe SIGURG).  However, it's bad practice for reusable libraries to
//! unconditionally install signal handlers, so unlikely to be a problem.
bool installNoOpSigIoHandler(void) {
    struct sigaction sa;
    sigemptyset(&sa.sa_mask);
    sa.sa_handler = &noOpHandler;
    sa.sa_flags = 0;
    return ::sigaction(SIGIO, &sa, 0) == 0;
}

const bool SIGIO_HANDLER_INSTALLED(installNoOpSigIoHandler());

}

namespace ml {
namespace core {


CThread::CThread(void)
    : m_ThreadId(0) {}

CThread::~CThread(void) {
    CScopedLock lock(m_IdMutex);

    if (m_ThreadId != 0) {
        LOG_ERROR("Trying to destroy a running thread. Call 'stop' before destroying");
    }
}

bool CThread::start(void) {
    TThreadId dummy(0);

    return this->start(dummy);
}

bool CThread::start(TThreadId &threadId) {
    CScopedLock lock(m_IdMutex);

    if (m_ThreadId != 0) {
        LOG_ERROR("Thread already running");
        threadId = m_ThreadId;
        return false;
    }

    int ret = pthread_create(&m_ThreadId, 0, &CThread::threadFunc, this);
    if (ret != 0) {
        LOG_ERROR("Cannot create thread: " << ::strerror(ret));
        threadId = 0;
        return false;
    }

    threadId = m_ThreadId;

    return true;
}

bool CThread::stop(void) {
    CScopedLock lock(m_IdMutex);

    if (m_ThreadId == 0) {
        LOG_ERROR("Thread not running");
        return false;
    }

    if (pthread_self() == m_ThreadId) {
        LOG_ERROR("Can't stop own thread");
        return false;
    }

    // Signal to running thread to shutdown
    this->shutdown();

    int ret = pthread_join(m_ThreadId, 0);
    if (ret != 0) {
        LOG_ERROR("Error joining thread: " << ::strerror(ret));
    }

    m_ThreadId = 0;

    return true;
}

bool CThread::waitForFinish(void) {
    CScopedLock lock(m_IdMutex);

    if (m_ThreadId == 0) {
        LOG_ERROR("Thread not running");
        return false;
    }

    if (pthread_self() == m_ThreadId) {
        LOG_ERROR("Can't stop own thread");
        return false;
    }

    int ret = pthread_join(m_ThreadId, 0);
    if (ret != 0) {
        LOG_ERROR("Error joining thread: " << ::strerror(ret));
    }

    m_ThreadId = 0;

    return true;
}

bool CThread::isStarted(void) const {
    CScopedLock lock(m_IdMutex);

    return (m_ThreadId != 0);
}

bool CThread::cancelBlockedIo(void) {
    CScopedLock lock(m_IdMutex);

    if (m_ThreadId == 0) {
        LOG_ERROR("Thread not running");
        return false;
    }

    if (pthread_self() == m_ThreadId) {
        LOG_ERROR("Can't cancel blocked IO in own thread");
        return false;
    }

    // Deliver the signal using pthread_kill() rather than raise() to ensure it
    // is delivered to the correct thread.
    int ret = pthread_kill(m_ThreadId, SIGIO);
    if (ret != 0) {
        // Don't report an error if the thread has already exited
        if (ret != ESRCH) {
            LOG_ERROR("Error cancelling blocked IO in thread: " <<
                      ::strerror(ret));
            return false;
        }
    }

    return true;
}

bool CThread::cancelBlockedIo(TThreadId threadId) {
    if (pthread_self() == threadId) {
        LOG_ERROR("Can't cancel blocked IO in own thread");
        return false;
    }

    // Deliver the signal using pthread_kill() rather than raise() to ensure it
    // is delivered to the correct thread.
    int ret = pthread_kill(threadId, SIGIO);
    if (ret != 0) {
        // Don't report an error if the thread has already exited
        if (ret != ESRCH) {
            LOG_ERROR("Error cancelling blocked IO in thread " <<
                      threadId << ": " << ::strerror(ret));
            return false;
        }
    }

    return true;
}

CThread::TThreadId CThread::currentThreadId(void) {
    return pthread_self();
}

CThread::TThreadRet STDCALL CThread::threadFunc(void *obj) {
    CThread *instance = static_cast<CThread *>(obj);

    instance->run();

    return 0;
}


}
}

