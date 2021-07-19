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
#include <libkern/OSAtomic.h>
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
//! On Mac OS X, spin locks are also used, which might seem strange
//! given that Mac OS X will have relatively few available CPU cores
//! (say 2-4).  However, Mac OS X spinlocks are actually quite clever
//! about yielding if they don't quickly get a lock, and are hence
//! more akin to adaptive mutexes than pure spinlocks.  The source
//! code at confirms this:
//! http://www.opensource.apple.com/source/Libc/Libc-825.25/x86_64/sys/spinlocks_asm.s
//! The other reason is that pthread mutexes are appallingly slow on
//! Mac OS X so it's easy to massively outperform them - e.g. see:
//! http://www.mr-edd.co.uk/blog/sad_state_of_osx_pthread_mutex_t
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
    OSSpinLock m_Mutex;
#else
    pthread_mutex_t m_Mutex;
#endif
};
}
}

#endif // INCLUDED_ml_core_CFastMutex_h
