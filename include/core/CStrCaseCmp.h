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
#ifndef INCLUDED_ml_core_CStrCaseCmp_h
#define INCLUDED_ml_core_CStrCaseCmp_h

#include <core/CNonInstantiatable.h>
#include <core/ImportExport.h>

#include <stddef.h>

namespace ml {
namespace core {

//! \brief
//! Portable wrapper for the strcasecmp() function.
//!
//! DESCRIPTION:\n
//! Portable wrapper for the strcasecmp() function and the closely
//! related strncasecmp() function.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This has been broken into a class of its own because Windows has a
//! _stricmp() function whilst Unix has strcasecmp().
//!
class CORE_EXPORT CStrCaseCmp : private CNonInstantiatable {
public:
    static int strCaseCmp(const char* s1, const char* s2);
    static int strNCaseCmp(const char* s1, const char* s2, size_t n);
};
}
}

#endif // INCLUDED_ml_core_CStrCaseCmp_h
