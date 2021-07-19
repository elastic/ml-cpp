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
//! On Linux and macOS we reduce the CPU scheduling priority using "nice".
//! This should be done after connecting named pipes to the JVM, because
//! named pipe connection can be time-sensitive and failures are hard to
//! diagnose.
//!
//! On Linux we also attempt to increase the OOM killer adjustment for the
//! process such that it is more likely to be killed than other processes
//! when the Linux kernel decides that there isn't enough free memory.
//!
class CORE_EXPORT CProcessPriority : private CNonInstantiatable {
public:
    //! Reduce priority for memory if deemed appropriate for the
    //! current OS.
    static void reduceMemoryPriority();

    //! Reduce priority for CPU if deemed appropriate for the
    //! current OS.
    static void reduceCpuPriority();
};
}
}

#endif // INCLUDED_ml_core_CProcessPriority_h
