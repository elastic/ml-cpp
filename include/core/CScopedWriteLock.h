/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CScopedWriteLock_h
#define INCLUDED_ml_core_CScopedWriteLock_h

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
class CORE_EXPORT CScopedWriteLock : private CNonCopyable
{
    public:
        //! Write lock specified read/write lock
        CScopedWriteLock(CReadWriteLock &readWriteLock);

        //! Unlock specified read/write lock
        ~CScopedWriteLock(void);

    private:
        CReadWriteLock &m_ReadWriteLock;
};


}
}

#endif // INCLUDED_ml_core_CScopedWriteLock_h

