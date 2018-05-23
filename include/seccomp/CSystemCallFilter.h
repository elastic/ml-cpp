/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_seccomp_CSystemCallFilter_h
#define INCLUDED_ml_seccomp_CSystemCallFilter_h

#include <core/CNonInstantiatable.h>

namespace ml {
namespace seccomp {

//! \brief
//! Installs secure computing modes for Linux, macOs and Windows
//!
//! DESCRIPTION:\n
//! ML processes require a subset of system calls to function correctly.
//! These are create a named pipe, connect to a named pipe, read and write
//! no other system calls are necessary and should be resticted to prevent
//! malicious actions.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Implementations are platform specific more details can be found in the
//! particular .cc files.
//!
//! Linux:
//! Seccomp BPF is used to restrict system calls on kernels since 3.5.
//!
//! macOs:
//! The sandbox facility is used to restict access to system resources.
//!
//! Windows:
//! Job Objects prevent the process spawning another.
//!
class CSystemCallFilter : private core::CNonInstantiatable {
public:
    static void installSystemCallFilter();
};
}
}

#endif // INCLUDED_ml_seccomp_CSystemCallFilter_h
