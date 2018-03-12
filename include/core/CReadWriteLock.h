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
#ifndef INCLUDED_ml_core_CReadWriteLock_h
#define INCLUDED_ml_core_CReadWriteLock_h

#include <core/CNonCopyable.h>
#include <core/ImportExport.h>
#include <core/WindowsSafe.h>

#ifndef Windows
#include <pthread.h>
#endif


namespace ml {
namespace core {


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
class CORE_EXPORT CReadWriteLock : private CNonCopyable {
    public:
        CReadWriteLock(void);
        ~CReadWriteLock(void);

        void readLock(void);
        void readUnlock(void);

        void writeLock(void);
        void writeUnlock(void);

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

