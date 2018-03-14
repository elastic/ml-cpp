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
#ifndef INCLUDED_ml_core_CCondition_h
#define INCLUDED_ml_core_CCondition_h

#include <core/CNonCopyable.h>
#include <core/ImportExport.h>
#include <core/WindowsSafe.h>

#ifndef Windows
#include <pthread.h>
#endif
#include <stdint.h>


namespace ml {
namespace core {
class CMutex;


//! \brief
//! Wrapper around pthread_cond_wait.
//!
//! DESCRIPTION:\n
//! Wrapper around pthread_cond_wait.
//!
//! IMPLEMENTATION DECISIONS:\n
//! On Windows, condition variables are preferred to events, as
//! they do not consume system handles, and match the semantics
//! of pthread condition variables.
//!
class CORE_EXPORT CCondition : private CNonCopyable {
    public:
        CCondition(CMutex &);
        ~CCondition(void);

        //! Wait in current thread for signal - blocks.  The wait may be
        //! spuriously interrupted by a signal, so the caller must check a
        //! condition that will detect spurious wakeups, and wait again if
        //! necessary.
        bool wait(void);

        //! Timed wait in current thread for millisecs - blocks.  The wait may
        //! be spuriously interrupted by a signal, so the caller must check a
        //! condition that will detect spurious wakeups, and wait again if
        //! necessary.
        bool wait(uint32_t t);

        //! Wake up a single thread that is blocked in wait
        void signal(void);

        //! Wake up all threads that are blocked in wait
        void broadcast(void);

    private:
#ifndef Windows
        //! Convert milliseconds to timespec
        static bool        convert(uint32_t, timespec &);
#endif

    private:
        //! Reference to associated mutex
        CMutex             &m_Mutex;

        //! The condition variable
#ifdef Windows
        CONDITION_VARIABLE m_Condition;
#else
        pthread_cond_t     m_Condition;
#endif
};


}
}

#endif // INCLUDED_ml_core_CCondition_h

