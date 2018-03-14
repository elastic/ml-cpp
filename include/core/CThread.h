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
#ifndef INCLUDED_ml_core_CThread_h
#define INCLUDED_ml_core_CThread_h

#include <core/CMutex.h>
#include <core/CNonCopyable.h>
#include <core/ImportExport.h>
#include <core/WindowsSafe.h>

#ifndef Windows
#include <pthread.h>
#endif


namespace ml {
namespace core {

//! \brief
//! Basic wrapper class around pthread_create etc.
//!
//! DESCRIPTION:\n
//! Basic wrapper class around pthread_create etc.
//!
//! IMPLEMENTATION DECISIONS:\n
//! A mutex protects the thread ID variable to prevent race
//! conditions when a thread is started and stopped in quick
//! succession.
//!
class CORE_EXPORT CThread : private CNonCopyable {
    public:
        //! Thread ID type
#ifdef Windows
        typedef DWORD        TThreadId;
        typedef unsigned int TThreadRet;
#else
        typedef pthread_t      TThreadId;
        typedef void          *TThreadRet;
#endif

    public:
        CThread(void);
        virtual ~CThread(void);

        //! Start the thread.  It's an error to call this if the thread is
        //! already running.
        bool                      start(void);

        //! Start the thread, retrieving the thread ID.  It's an error to call
        //! this if the thread is already running.
        bool                      start(TThreadId &threadId);

        //! Stop the thread.  It's an error to call this if the thread is
        //! already stopped.  Only call one of stop() and waitForFinish(); do
        //! NOT call both.
        bool                      stop(void);

        //! This method blocks and waits for the thread to finish.
        //! It differs from 'stop' as it doesn't call shutdown.
        //! BE AWARE THIS MAY BLOCK INDEFINITELY.  Only call one of stop()
        //! and waitForFinish(); do NOT call both.
        bool                      waitForFinish(void);

        //! Has the thread been started?
        bool                      isStarted(void) const;

        //! Wake up any blocking IO calls in this thread, such as reads to named
        //! pipes where nothing has connected to the other end of the pipe.
        bool                      cancelBlockedIo(void);

        //! Wake up any blocking IO calls in the specified thread, such as reads
        //! to named pipes where nothing has connected to the other end of the
        //! pipe.
        static bool               cancelBlockedIo(TThreadId threadId);

        //! Static method to get the ID of the currently running thread
        static TThreadId          currentThreadId(void);

    protected:
        //! The run() method should only be called from threadFunc()
        virtual void              run(void) = 0;
        virtual void              shutdown(void) = 0;

    private:
        //! This method is used as a thread start function, hence it must be
        //! static so that we can take its address like a free function
        static TThreadRet STDCALL threadFunc(void *obj);

    private:
        //! ID of the most recently started thread
        TThreadId      m_ThreadId;

#ifdef Windows
        //! Windows needs a thread handle as well as a thread ID
        HANDLE         m_ThreadHandle;
#endif

        //! Mutex to protect access to m_ThreadId
        mutable CMutex m_IdMutex;
};


}
}

#endif // INCLUDED_ml_core_CThread_h

