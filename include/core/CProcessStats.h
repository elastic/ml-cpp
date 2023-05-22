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
#ifndef INCLUDED_ml_core_CProcessStats_h
#define INCLUDED_ml_core_CProcessStats_h

#include <core/CNonInstantiatable.h>
#include <core/ImportExport.h>

#include <cstddef>

namespace ml {
namespace core {

//! \brief
//! Functions related to getting process statistics, e.g. memory usage.
//!
//! DESCRIPTION:\n
//! Methods of this class are useful to get process statistics of the
//! running process, e.g. memory usage(RSS).
//!
//! IMPLEMENTATION DECISIONS:\n
//! This is a static class - it's not possible to construct an instance of it.
//!
//! The implementation is OS specific, not all methods might be available
//! for every OS.
//!
class CORE_EXPORT CProcessStats : private CNonInstantiatable {
public:
    //! return the RSS of this process in bytes
    static std::size_t residentSetSize();

    //! return the maximum RSS of this process in bytes
    static std::size_t maxResidentSetSize();
};
}
}

#endif // INCLUDED_ml_core_CProcessStats_h
