/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CScopedLock_h
#define INCLUDED_ml_core_CScopedLock_h

#include <core/CNonCopyable.h>
#include <core/ImportExport.h>


namespace ml
{
namespace core
{
class CMutex;

//! \brief
//! Implementation of Scoped Locking idiom/pattern.
//!
//! DESCRIPTION:\n
//! Implementation of Scoped Locking idiom/pattern.
//!
//! IMPLEMENTATION DECISIONS:\n
//! See Schmidt etc. for details.
//!
class CORE_EXPORT CScopedLock : private CNonCopyable
{
    public:
        //! Lock specified mutex
        CScopedLock(CMutex &mutex);

        //! Unlock specified mutex
        ~CScopedLock();

    private:
        CMutex &m_Mutex;
};


}
}

#endif // INCLUDED_ml_core_CScopedLock_h

