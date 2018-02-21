/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CScopedReadLock_h
#define INCLUDED_ml_core_CScopedReadLock_h

#include <core/CNonCopyable.h>
#include <core/ImportExport.h>


namespace ml
{
namespace core
{
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
class CORE_EXPORT CScopedReadLock : private CNonCopyable
{
    public:
        //! Read lock specified read/write lock
        CScopedReadLock(CReadWriteLock &readWriteLock);

        //! Unlock specified read/write lock
        ~CScopedReadLock(void);

    private:
        CReadWriteLock &m_ReadWriteLock;
};


}
}

#endif // INCLUDED_ml_core_CScopedReadLock_h

