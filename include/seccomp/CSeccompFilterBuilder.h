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
#ifndef INCLUDED_ml_seccomp_CSeccompFilterBuilder_h
#define INCLUDED_ml_seccomp_CSeccompFilterBuilder_h

#ifdef __linux__

#include <linux/filter.h>

#include <vector>

namespace ml {
namespace seccomp {

//! Builds a seccomp BPF program that allows exactly allowedSyscalls, on the
//! native architecture only, and denies everything else with EACCES.
//!
//! The caller supplies allowedSyscalls in any order: every generated jump
//! offset is derived from the vector's size and the row's own index, so
//! adding, removing or reordering a syscall never requires updating any
//! other row. This is the mechanism that lets CSystemCallFilter_Linux.cc
//! apply CPytorchInferenceSyscallAllowlist.h's declaration directly, instead
//! of maintaining a second, hand-written BPF program with manual jump
//! offsets that can silently drift from the declaration.
std::vector<sock_filter> buildSyscallAllowlistProgram(const std::vector<int>& allowedSyscalls);
}
}

#endif // __linux__

#endif // INCLUDED_ml_seccomp_CSeccompFilterBuilder_h
