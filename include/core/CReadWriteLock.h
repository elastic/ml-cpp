/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CReadWriteLock_h
#define INCLUDED_ml_core_CReadWriteLock_h

#include <core/CNonCopyable.h>
#include <core/ImportExport.h>
#include <core/WindowsSafe.h>

#ifndef Windows
#include <pthread.h>
#endif


namespace ml
{
namespace core
{


//! \brief
//! Wrapper class around pthread rw lock.
//!
//! DESCRIPTION:\n
//! Wrapper class around pthread rw lock.
//!
//! Note that the efficiency of read/write locks varies massively
//! between systems.  Therefore, careful thought is required
//! before choosing to use this class in preference to a standard
//! mutex.  There needs to be a big benefit in distinguishing
//! read/write access before it's worth it.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Recursive locking is NOT allowed.  The results of trying
//! are operating system specific and may result in deadlocks.
//!
//! Constructor makes the lock, destructor destroys.
//!
//! All errors are just warnings - no action taken.
//!
class CORE_EXPORT CReadWriteLock : private CNonCopyable
{
    public:
        CReadWriteLock();
        ~CReadWriteLock();

        void readLock();
        void readUnlock();

        void writeLock();
        void writeUnlock();

    private:
#ifdef Windows
        SRWLOCK          m_ReadWriteLock;
#else
        pthread_rwlock_t m_ReadWriteLock;
#endif
};


}
}

#endif // INCLUDED_ml_core_CReadWriteLock_h

