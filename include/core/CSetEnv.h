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
#ifndef INCLUDED_ml_core_CSetEnv_h
#define INCLUDED_ml_core_CSetEnv_h

#include <core/CNonInstantiatable.h>
#include <core/ImportExport.h>

namespace ml {
namespace core {

//! \brief
//! Portable wrapper for the setenv() function.
//!
//! DESCRIPTION:\n
//! Portable wrapper for the setenv() function.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This has been broken into a class of its own because Windows has a
//! _putenv_s() function with slightly different semantics to Unix's
//! setenv().
//!
class CORE_EXPORT CSetEnv : private CNonInstantiatable {
public:
    static int setEnv(const char* name, const char* value, int overwrite);
};
}
}

#endif // INCLUDED_ml_core_CSetEnv_h
