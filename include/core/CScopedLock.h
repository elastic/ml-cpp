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
#ifndef INCLUDED_ml_core_CScopedLock_h
#define INCLUDED_ml_core_CScopedLock_h

#include <core/CNonCopyable.h>
#include <core/ImportExport.h>


namespace ml {
namespace core {
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
class CORE_EXPORT CScopedLock : private CNonCopyable {
    public:
        //! Lock specified mutex
        CScopedLock(CMutex &mutex);

        //! Unlock specified mutex
        ~CScopedLock(void);

    private:
        CMutex &m_Mutex;
};


}
}

#endif // INCLUDED_ml_core_CScopedLock_h

