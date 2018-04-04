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
