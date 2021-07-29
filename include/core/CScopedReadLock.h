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
#ifndef INCLUDED_ml_core_CScopedReadLock_h
#define INCLUDED_ml_core_CScopedReadLock_h

#include <core/CNonCopyable.h>
#include <core/ImportExport.h>

namespace ml {
namespace core {
class CReadWriteLock;

//! \brief
//! Implementation of Scoped Locking idiom/pattern.
//!
//! DESCRIPTION:\n
//! Implementation of Scoped Locking idiom/pattern.
//!
//! IMPLEMENTATION DECISIONS:\n
//! See Schmidt etc. for details.
//!
class CORE_EXPORT CScopedReadLock : private CNonCopyable {
public:
    //! Read lock specified read/write lock
    CScopedReadLock(CReadWriteLock& readWriteLock);

    //! Unlock specified read/write lock
    ~CScopedReadLock();

private:
    CReadWriteLock& m_ReadWriteLock;
};
}
}

#endif // INCLUDED_ml_core_CScopedReadLock_h
