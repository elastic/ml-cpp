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
#ifndef INCLUDED_ml_core_CMutex_h
#define INCLUDED_ml_core_CMutex_h

#include <core/CNonCopyable.h>
#include <core/ImportExport.h>
#include <core/WindowsSafe.h>

#ifndef Windows
#include <pthread.h>
#endif

namespace ml {
namespace core {
class CCondition;

//! \brief
//! Wrapper class around pthread mutex.
//!
//! DESCRIPTION:\n
//! Wrapper class around pthread mutex.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Recursive locking allowed.
//!
//! Constructor makes the mutex, destructor destroys.
//!
//! All errors are just warnings - no action taken.
//!
//! On Windows, critical sections are preferred to mutexes, as
//! they do not consume system handles.
//!
class CORE_EXPORT CMutex : private CNonCopyable {
public:
    CMutex();
    ~CMutex();

    void lock();
    void unlock();

private:
#ifdef Windows
    CRITICAL_SECTION m_Mutex;
#else
    pthread_mutex_t m_Mutex;
#endif

    // Allow CCondition access to internals
    friend class CCondition;
};
}
}

#endif // INCLUDED_ml_core_CMutex_h
