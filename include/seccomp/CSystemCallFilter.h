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

#include <cstdlib>
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
//! instead. It is only safe to pass true where a degraded-mode launch is
//! guaranteed to be a deliberate route decision rather than the production
//! default. bin/controller's CProcessSpawnerRouter provides half of that
//! guarantee (it never retries a failed Sandbox2 spawn through the legacy
//! spawner), but while CCommandProcessor's no-token case still always
//! routes to legacy, and no caller is yet guaranteed to always send an
//! explicit --disableSandbox/--requireSandbox token, an ordinary launch
//! *is* a degraded-route launch, so bin/pytorch_inference/Main.cc passes
//! false. See the comment at TERMINATE_ON_DEGRADED_SECCOMP_FAILURE there
//! for when it flips.
//! This decision only ever
//! applies to a launch that installs its own in-process filter at all - see
//! sandbox2LaunchedChild() and applyInProcessSeccompFilter() below.
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

//! Pure form of the "was this process launched by the Sandbox2 executor?"
//! test, taking the raw ML_SANDBOXED environment value (nullptr when unset)
//! so it is testable on every platform without mutating the environment.
//!
//! pytorch_inference skips in-process seccomp only when ML_SANDBOXED is
//! *exactly* "1", the
//! value CSandboxedProcessSpawner_Linux.cc sets on a Sandbox2-launched
//! child. It is stripped from every legacy-route child's environment by
//! lib/core/CDetachedProcessSpawner.cc (detail::buildChildEnvironment(),
//! declared in include/core/CDetachedProcessSpawner.h), so an inherited or
//! injected ML_SANDBOXED in the controller's own environment can never
//! suppress a legacy-route child's mandatory in-process filter. Any other
//! value - unset, "", "0", "true", "10" -
//! is a legacy/non-sandboxed launch that must install its own filter.
inline bool sandbox2LaunchedChild(const char* mlSandboxedEnv) {
    return mlSandboxedEnv != nullptr && std::string{mlSandboxedEnv} == "1";
}

//! \return true if this process is a Sandbox2-launched sandboxee, per
//! sandbox2LaunchedChild(const char*) applied to the live environment.
inline bool sandbox2LaunchedChild() {
    return sandbox2LaunchedChild(std::getenv("ML_SANDBOXED"));
}

//! Everything one launch's in-process seccomp startup step decided, so a
//! caller has no way to attest or terminate on a step that never ran.
struct SInProcessFilterResult {
    //! False iff the filter installation was skipped because this process
    //! is a Sandbox2 sandboxee (the executor's own policy is already the
    //! security boundary). When false, every other field is the inert
    //! "nothing happened" value.
    bool s_Attempted{false};
    //! What the caller must do before untrusted IO/model processing.
    EDegradedModeAction s_Action{EDegradedModeAction::E_ContinueDespiteFailure};
    //! Outcome of the installation attempt; meaningless when
    //! s_Attempted == false.
    ESystemCallFilterInstallOutcome s_Outcome{ESystemCallFilterInstallOutcome::E_Installed};
    //! degradedModeAttestationMarker() for s_Outcome, or empty when nothing
    //! is attested. Always empty when s_Attempted == false: that marker
    //! describes the *legacy* route's own filter installation, so emitting
    //! it on a Sandbox2-route launch would both attest a filter that was
    //! never installed and contradict the sandbox2_launch signal's
    //! "route":"sandbox2" for the same launch.
    std::string s_AttestationMarker;
};

//! Pure driver for the in-process seccomp startup step of a single launch.
//!
//! \param sandbox2Launched typically sandbox2LaunchedChild(); when true the
//!        filter installation is skipped *entirely* - \p installer is never
//!        invoked, no degraded-mode action is derived and no attestation
//!        marker is produced, regardless of what an installation attempt
//!        would have returned. Installing an in-process filter from inside
//!        an already-sandboxed environment can fail (which would kill every
//!        enforced-route launch once TERMINATE_ON_DEGRADED_SECCOMP_FAILURE is activated) or
//!        succeed and mislabel the launch as legacy.
//! \param terminateOnFailure passed through to decideDegradedModeAction().
//! \param installer invoked at most once; normally
//!        CSystemCallFilter::installSystemCallFilter.
template<typename INSTALLER>
SInProcessFilterResult applyInProcessSeccompFilter(bool sandbox2Launched,
                                                   bool terminateOnFailure,
                                                   INSTALLER installer) {
    SInProcessFilterResult result;
    if (sandbox2Launched) {
        return result;
    }
    result.s_Attempted = true;
    result.s_Outcome = installer();
    result.s_Action = decideDegradedModeAction(result.s_Outcome, terminateOnFailure);
    result.s_AttestationMarker = degradedModeAttestationMarker(result.s_Outcome);
    return result;
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
