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
#ifndef INCLUDED_ml_core_CFastMutex_h
#define INCLUDED_ml_core_CFastMutex_h

#include <core/CNonCopyable.h>
#include <core/ImportExport.h>
#include <core/WindowsSafe.h>

#ifndef Windows
#ifdef MacOSX
#include <os/lock.h>
#else
#include <pthread.h>
#endif
#endif

namespace ml {
namespace core {

//! \brief
//! Wrapper class around a fast simple mutex.
//!
//! DESCRIPTION:\n
//! The fastest simple mutex implementation varies between operating
//! systems.  For example, on Windows the slim read/write lock is
//! faster than a critical section, whereas the pthreads read/write
//! lock can be very slow.
//!
//! For situations where a very simple mutex is required, this class
//! wraps up the fastest way of achieving it.
//!
//! This class CANNOT be used in conjunction with a condition variable.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Recursive locking is NOT allowed.
//!
//! All errors are just warnings - no action taken.
//!
//! On Mac OS X, spin locks used to be used, but OSSpinLock has now
//! been tagged as deprecated as of macOS 10.12. Instead, Apple have
//! provided os_unfair_lock as a replacement (see the documentation here
//! https://developer.apple.com/documentation/os/1646466-os_unfair_lock_lock)
//! The other reason is that pthread mutexes are appallingly slow on
//! Mac OS X so it's easy to massively outperform them
//!
//! On Linux, standard non-recursive mutexes are used, on the
//! assumption that Linux may have relatively few available CPU
//! cores (say 2-8), and because they're pretty fast.
//!
//! On Windows, slim read/write locks are used (always taking the
//! write lock).  These are faster than critical sections, presumably
//! because critical sections are recursive.
//!
class CORE_EXPORT CFastMutex : private CNonCopyable {
public:
    CFastMutex();
    ~CFastMutex();

    void lock();
    void unlock();

private:
#ifdef Windows
    SRWLOCK m_Mutex;
#elif defined(MacOSX)
    os_unfair_lock m_Mutex;
#else
    pthread_mutex_t m_Mutex;
#endif
};
}
}

#endif // INCLUDED_ml_core_CFastMutex_h
