/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CProcessPriority_h
#define INCLUDED_ml_core_CProcessPriority_h

#include <core/CNonInstantiatable.h>
#include <core/ImportExport.h>

namespace ml {
namespace core {

//! \brief
//! Functions related to adjusting process priority
//!
//! DESCRIPTION:\n
//! Methods of this class should be called to lower the priority of machine
//! learning processes in whatever ways are appropriate on a given operating
//! system to be a good neighbour to the Elasticsearch JVM.
//!
//! We don't ever want an ML C++ process to hog resources to the extent that
//! the Elasticsearch JVM cannot operate effectively.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This is a static class - it's not possible to construct an instance of it.
//!
//! The basic implementation does nothing.  Platform-specific implementations
//! exist for platforms where we have decided to do something.
//!
//! Currently the only platform-specific implementation is for Linux, and it
//! attempts to increase the OOM killer adjustment for the process such that
//! it is more likely to be killed than other processes when the Linux kernel
//! decides that there isn't enough free memory.
//!
class CORE_EXPORT CProcessPriority : private CNonInstantiatable {
public:
    //! Reduce whatever priority measures are deemed appropriate for the
    //! current OS.
    static void reducePriority();
};
}
}

#endif // INCLUDED_ml_core_CProcessPriority_h
