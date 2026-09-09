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
#ifndef INCLUDED_ml_seccomp_CSystemCallFilter_h
#define INCLUDED_ml_seccomp_CSystemCallFilter_h

#include <core/CNonInstantiatable.h>

#include <string>

namespace ml {
namespace seccomp {

//! \brief
//! Installs secure computing modes for Linux, macOS and Windows
//!
//! DESCRIPTION:\n
//! ML processes require a subset of system calls to function correctly:
//! creating a named pipe, connecting to a named pipe, and reading and
//! writing.  No other system calls are necessary, so the rest should be
//! restricted to prevent malicious actions.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Implementations are platform specific more details can be found in the
//! particular .cc files.
//!
//! Linux:
//! Seccomp BPF is used to restrict system calls on kernels since 3.5.
//! The filter first requires seccomp_data.arch to match the native ABI
//! (rejecting compat ABIs such as i386 int 0x80 on x86_64, which would
//! otherwise collide with allowlisted syscall numbers).
//!
//! macOS:
//! The sandbox facility is used to restrict access to system resources.
//!
//! Windows:
//! Job Objects prevent the process spawning another.
//!
enum class ESystemCallFilterInstallOutcome {
    E_Installed,
    //! The platform mechanism itself is unavailable (e.g. kernel not built
    //! with CONFIG_SECCOMP_FILTER).
    E_MechanismUnavailable,
    //! The mechanism is available but a required privilege-restriction step
    //! failed (e.g. PR_SET_NO_NEW_PRIVS on Linux).
    E_PrivilegeRestrictionFailed,
    //! The mechanism is available but installing the filter/profile itself
    //! failed.
    E_FilterInstallFailed
};

//! Human-readable description of an install outcome, for diagnostics only;
//! not a stable machine-parsed value.
inline const char* describe(ESystemCallFilterInstallOutcome outcome) {
    switch (outcome) {
    case ESystemCallFilterInstallOutcome::E_Installed:
        return "installed";
    case ESystemCallFilterInstallOutcome::E_MechanismUnavailable:
        return "mechanism unavailable";
    case ESystemCallFilterInstallOutcome::E_PrivilegeRestrictionFailed:
        return "privilege restriction failed";
    case ESystemCallFilterInstallOutcome::E_FilterInstallFailed:
        return "filter install failed";
    }
    return "unknown";
}

//! What a caller should do, given an install outcome and whether hard
//! termination is currently enabled at that call site.
enum class EDegradedModeAction {
    E_ContinueDespiteFailure,
    E_TerminateBeforeIo
};

//! Pure decision function: does this install outcome require terminating
//! before untrusted IO/model processing?
//!
//! terminateOnFailure is an internal switch, not an operator setting. Every
//! degraded-mode seccomp failure should eventually terminate before
//! processing, but flipping that on for every call site before the
//! ml-cpp/Elasticsearch controller protocol can guarantee a degraded-mode
//! launch was a deliberate operator choice would fail every launch on a
//! host lacking seccomp BPF, with no operator fallback setting to select
//! instead. Callers pass false today; a later change wires the real route
//! decision through this parameter once that guarantee exists.
inline EDegradedModeAction decideDegradedModeAction(ESystemCallFilterInstallOutcome outcome,
                                                    bool terminateOnFailure) {
    if (outcome == ESystemCallFilterInstallOutcome::E_Installed || !terminateOnFailure) {
        return EDegradedModeAction::E_ContinueDespiteFailure;
    }
    return EDegradedModeAction::E_TerminateBeforeIo;
}

//! Structured signal a controller/Elasticsearch observer asserts to confirm
//! that a legacy/degraded-mode pytorch_inference launch actually installed
//! its in-process seccomp filter before processing untrusted model input.
//! Replaces attesting readiness by inference — "no fatal log line appeared
//! before initIo() ran" — with an explicit signal a test or observer can
//! assert on directly. Returns empty when
//! installation did not succeed: a failed degraded launch already exits
//! before initIo() (see decideDegradedModeAction()) and must never emit
//! this marker, since doing so would falsely attest a filter that isn't
//! there. Logged over the existing per-process log pipe; this is not a new
//! startup channel.
inline std::string degradedModeAttestationMarker(ESystemCallFilterInstallOutcome outcome) {
    if (outcome != ESystemCallFilterInstallOutcome::E_Installed) {
        return std::string();
    }
    return "{\"ml_sandbox2_route\":\"legacy\",\"event\":\"seccomp_installed\"}";
}

class CSystemCallFilter : private core::CNonInstantiatable {
public:
    //! Installs the platform syscall filter. Returns the typed outcome so a
    //! caller can decide whether to continue or terminate; callers must not
    //! silently discard the result (see decideDegradedModeAction()).
    [[nodiscard]] static ESystemCallFilterInstallOutcome installSystemCallFilter();
};
}
}

#endif // INCLUDED_ml_seccomp_CSystemCallFilter_h
