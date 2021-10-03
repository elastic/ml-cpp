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
#ifndef INCLUDED_ml_core_CCTimeR_h
#define INCLUDED_ml_core_CCTimeR_h

#include <core/CNonInstantiatable.h>
#include <core/ImportExport.h>

#include <time.h>

namespace ml {
namespace core {

//! \brief
//! Portable wrapper for the ctime_r() function.
//!
//! DESCRIPTION:\n
//! Portable wrapper for the ctime_r() function.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This has been broken into a class of its own because Windows has a
//! ctime_s() function with slightly different semantics to Unix's
//! ctime_r().
//!
class CORE_EXPORT CCTimeR : private CNonInstantiatable {
public:
    static char* cTimeR(const time_t* clock, char* result);
};
}
}

#endif // INCLUDED_ml_core_CCTimeR_h
