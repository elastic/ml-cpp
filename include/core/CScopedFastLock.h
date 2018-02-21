/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CScopedFastLock_h
#define INCLUDED_ml_core_CScopedFastLock_h

#include <core/CNonCopyable.h>
#include <core/ImportExport.h>


namespace ml
{
namespace core
{
class CFastMutex;

//! \brief
//! Implementation of Scoped Locking idiom/pattern.
//!
//! DESCRIPTION:\n
//! Implementation of Scoped Locking idiom/pattern.
//!
//! IMPLEMENTATION DECISIONS:\n
//! See Schmidt etc. for details.
//!
class CORE_EXPORT CScopedFastLock : private CNonCopyable
{
    public:
        //! Lock specified mutex
        CScopedFastLock(CFastMutex &mutex);

        //! Unlock specified mutex
        ~CScopedFastLock(void);

    private:
        CFastMutex &m_Mutex;
};


}
}

#endif // INCLUDED_ml_core_CScopedFastLock_h

