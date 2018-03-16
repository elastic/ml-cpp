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
#include <core/CWindowsError.h>

#include <errno.h>
#include <process.h>
#include <string.h>

namespace ml {
namespace core {

CThread::CThread(void) : m_ThreadId(0), m_ThreadHandle(INVALID_HANDLE_VALUE) {
}

CThread::~CThread(void) {
    CScopedLock lock(m_IdMutex);

    if (m_ThreadHandle != INVALID_HANDLE_VALUE) {
        LOG_ERROR("Trying to destroy a running thread. Call 'stop' before destroying");
    }
}

bool CThread::start(void) {
    TThreadId dummy(0);

    return this->start(dummy);
}

bool CThread::start(TThreadId& threadId) {
    CScopedLock lock(m_IdMutex);

    if (m_ThreadHandle != INVALID_HANDLE_VALUE) {
        LOG_ERROR("Thread already running");
        threadId = m_ThreadId;
        return false;
    }

    // Must use _beginthread() or _beginthreadex() from process.h in order for
    // the C runtime library to be available in the new thread.  (If we used
    // a Windows API function to start the thread then we wouldn't be able to
    // use any C or C++ standard library functionality within it.)  It is safer
    // to use _beginthreadex() than _beginthread(), because if the thread
    // generated by _beginthread() exits quickly, the handle returned to the
    // caller of _beginthread() might be invalid or, worse, point to another
    // thread.  However, the handle returned by _beginthreadex() has to be
    // closed by the caller of _beginthreadex(), so it is guaranteed to be a
    // valid handle if _beginthreadex() did not return an error.
    uintptr_t handle(_beginthreadex(0, 0, &CThread::threadFunc, this, 0, 0));
    if (handle == 0) {
        LOG_ERROR("Cannot create thread: " << ::strerror(errno));
        threadId = 0;
        return false;
    }

    m_ThreadHandle = reinterpret_cast<HANDLE>(handle);
    m_ThreadId = GetThreadId(m_ThreadHandle);
    threadId = m_ThreadId;

    return true;
}

bool CThread::stop(void) {
    CScopedLock lock(m_IdMutex);

    if (m_ThreadHandle == INVALID_HANDLE_VALUE) {
        LOG_ERROR("Thread not running");
        return false;
    }

    if (GetCurrentThreadId() == m_ThreadId) {
        LOG_ERROR("Can't stop own thread");
        return false;
    }

    // Signal to running thread to shutdown
    this->shutdown();

    if (WaitForSingleObject(m_ThreadHandle, INFINITE) != 0) {
        DWORD errCode(GetLastError());

        // To match pthread behaviour, we won't report an error for joining a
        // thread that's already stopped
        if (errCode != ERROR_INVALID_HANDLE) {
            LOG_ERROR("Error joining thread: " << CWindowsError(errCode));
        }
    }

    CloseHandle(m_ThreadHandle);
    m_ThreadHandle = INVALID_HANDLE_VALUE;
    m_ThreadId = 0;

    return true;
}

bool CThread::waitForFinish(void) {
    CScopedLock lock(m_IdMutex);

    if (m_ThreadHandle == INVALID_HANDLE_VALUE) {
        LOG_ERROR("Thread not running");
        return false;
    }

    if (GetCurrentThreadId() == m_ThreadId) {
        LOG_ERROR("Can't stop own thread");
        return false;
    }

    if (WaitForSingleObject(m_ThreadHandle, INFINITE) != 0) {
        DWORD errCode(GetLastError());

        // To match pthread behaviour, we won't report an error for joining a
        // thread that's already stopped
        if (errCode != ERROR_INVALID_HANDLE) {
            LOG_ERROR("Error joining thread: " << CWindowsError(errCode));
        }
    }

    CloseHandle(m_ThreadHandle);
    m_ThreadHandle = INVALID_HANDLE_VALUE;
    m_ThreadId = 0;

    return true;
}

bool CThread::isStarted(void) const {
    CScopedLock lock(m_IdMutex);

    return (m_ThreadHandle != INVALID_HANDLE_VALUE);
}

bool CThread::cancelBlockedIo(void) {
    CScopedLock lock(m_IdMutex);

    if (m_ThreadHandle == INVALID_HANDLE_VALUE) {
        LOG_ERROR("Thread not running");
        return false;
    }

    if (GetCurrentThreadId() == m_ThreadId) {
        LOG_ERROR("Can't cancel blocked IO in own thread");
        return false;
    }

    if (CancelSynchronousIo(m_ThreadHandle) == FALSE) {
        DWORD errCode(GetLastError());

        // Don't report an error if there is no blocking call to cancel
        if (errCode != ERROR_NOT_FOUND) {
            LOG_ERROR("Error cancelling blocked IO in thread: " << CWindowsError(errCode));
            return false;
        }
    }

    return true;
}

bool CThread::cancelBlockedIo(TThreadId threadId) {
    if (GetCurrentThreadId() == threadId) {
        LOG_ERROR("Can't cancel blocked IO in own thread");
        return false;
    }

    HANDLE threadHandle = OpenThread(THREAD_TERMINATE, FALSE, threadId);
    // Note inconsistency in Win32 thread function return codes here - the error
    // return is NULL rather than INVALID_HANDLE_VALUE!
    if (threadHandle == 0) {
        LOG_ERROR("Error cancelling blocked IO in thread " << threadId << ": " << CWindowsError());
        return false;
    }

    if (CancelSynchronousIo(threadHandle) == FALSE) {
        DWORD errCode(GetLastError());

        // Don't report an error if there is no blocking call to cancel
        if (errCode != ERROR_NOT_FOUND) {
            LOG_ERROR("Error cancelling blocked IO in thread " << threadId << ": " << CWindowsError(errCode));
            CloseHandle(threadHandle);
            return false;
        }
    }

    CloseHandle(threadHandle);

    return true;
}

CThread::TThreadId CThread::currentThreadId(void) {
    return GetCurrentThreadId();
}

CThread::TThreadRet STDCALL CThread::threadFunc(void* obj) {
    CThread* instance = static_cast<CThread*>(obj);

    instance->run();

    // No need to call _endthreadex(), as returning from the thread cleans up
    // the C runtime library data structures associated with it
    return 0;
}
}
}
