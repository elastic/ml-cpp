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
        CScopedReadLock(CReadWriteLock &readWriteLock);

        //! Unlock specified read/write lock
        ~CScopedReadLock(void);

    private:
        CReadWriteLock &m_ReadWriteLock;
};


}
}

#endif // INCLUDED_ml_core_CScopedReadLock_h

