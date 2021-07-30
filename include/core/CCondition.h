/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
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
    CCondition(CMutex&);
    ~CCondition();

    //! Wait in current thread for signal - blocks.  The wait may be
    //! spuriously interrupted by a signal, so the caller must check a
    //! condition that will detect spurious wakeups, and wait again if
    //! necessary.
    bool wait();

    //! Timed wait in current thread for millisecs - blocks.  The wait may
    //! be spuriously interrupted by a signal, so the caller must check a
    //! condition that will detect spurious wakeups, and wait again if
    //! necessary.
    bool wait(uint32_t t);

    //! Wake up a single thread that is blocked in wait
    void signal();

    //! Wake up all threads that are blocked in wait
    void broadcast();

private:
#ifndef Windows
    //! Convert milliseconds to timespec
    static bool convert(uint32_t, timespec&);
#endif

private:
    //! Reference to associated mutex
    CMutex& m_Mutex;

    //! The condition variable
#ifdef Windows
    CONDITION_VARIABLE m_Condition;
#else
    pthread_cond_t m_Condition;
#endif
};
}
}

#endif // INCLUDED_ml_core_CCondition_h
