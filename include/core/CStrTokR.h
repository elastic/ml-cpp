/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CStrTokR_h
#define INCLUDED_ml_core_CStrTokR_h

#include <core/CNonInstantiatable.h>
#include <core/ImportExport.h>

namespace ml {
namespace core {

//! \brief
//! Portable wrapper for the strtok_r() function.
//!
//! DESCRIPTION:\n
//! Portable wrapper for the strtok_r() function.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This has been broken into a class of its own because Windows has a
//! strtok_s() function rather than Unix's strtok_r().
//!
class CORE_EXPORT CStrTokR : private CNonInstantiatable {
public:
    static char* strTokR(char* str, const char* sep, char** lasts);
};
}
}

#endif // INCLUDED_ml_core_CStrTokR_h
