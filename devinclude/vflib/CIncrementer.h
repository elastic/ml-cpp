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
#ifndef INCLUDED_ml_vflib_CIncrementer_h
#define INCLUDED_ml_vflib_CIncrementer_h

#include <vflib/ImportExport.h>

#include <stddef.h>

namespace ml {
namespace vflib {

//! \brief
//! Class for measuring function call overhead.
//!
//! DESCRIPTION:\n
//! Class for measuring function call overhead within a shared
//! library.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The aim is that the functions do very little - with an ABI
//! that passes arguments in registers no memory access should
//! be required.
//!
class VFLIB_EXPORT CIncrementer {
public:
    //! Best practice, though not really required in this case
    virtual ~CIncrementer(void);

    //! Inlined incrementer
    size_t inlinedIncrement(size_t val) { return val + 1; }

    //! Non-virtual incrementer
    size_t nonVirtualIncrement(size_t val);

    //! Virtual incrementer
    virtual size_t virtualIncrement(size_t val);
};
}
}

#endif// INCLUDED_ml_vflib_CIncrementer_h
