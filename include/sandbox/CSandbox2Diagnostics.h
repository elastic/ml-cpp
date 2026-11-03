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
#ifndef INCLUDED_ml_sandbox_CSandbox2Diagnostics_h
#define INCLUDED_ml_sandbox_CSandbox2Diagnostics_h

#include <string>

namespace ml {
namespace sandbox {

//! Which Sandbox2 startup prerequisite this host denies, if any.
//!
//! Sandbox2's forkserver builds its initial namespaces in a fixed order, and
//! a host that denies any one step fails the whole launch with the same
//! opaque SETUP_ERROR/FAILED_SUBPROCESS result (see
//! CSandboxedProcessSpawner_Linux.cc's RunAsync() failure path). The
//! sandboxee's own stderr carries the real reason, but it dies with the
//! sandboxee, and the controller's stderr is redirected onto the ML log pipe
//! where Elasticsearch discards anything that is not framed as a JSON log
//! message. Naming the denied step here is therefore the only way an
//! operator can tell a seccomp policy that blocks user namespaces from a
//! masked /proc - two different asks of whoever owns the container runtime.
enum class ESandbox2Capability {
    //! Every prerequisite succeeded: this host can run an enforced sandbox.
    E_Available,
    //! unshare(CLONE_NEWUSER) was denied - typically a container runtime
    //! seccomp profile that blocks the flag, or
    //! kernel.unprivileged_userns_clone=0.
    E_UserNamespaceDenied,
    //! The user namespace was created but its uid_map/gid_map could not be
    //! written, so nothing inside it can gain the capabilities mounts need.
    E_IdMapWriteDenied,
    //! unshare(CLONE_NEWNS|CLONE_NEWPID) was denied after the user namespace
    //! was already established.
    E_MountOrPidNamespaceDenied,
    //! Mounting a tmpfs inside the new namespaces was denied - typically an
    //! LSM (AppArmor/SELinux) mount rule rather than seccomp, since the
    //! namespace itself was created successfully.
    E_TmpfsMountDenied,
    //! Mounting a fresh procfs was denied - typically because the runtime has
    //! bind-mounted over part of /proc (Docker's masked paths), which makes
    //! the kernel refuse a new procfs mount that would unmask them.
    E_ProcMountDenied,
    //! The probe itself could not run (fork failed, or a temporary directory
    //! could not be created). Says nothing about the host's capabilities.
    E_ProbeFailed,
    //! Not a Linux build, or built without Sandbox2 support, so there is no
    //! prerequisite to probe.
    E_ProbeUnsupported
};

//! Human-readable one-line form of \p capability, suitable for a log message.
std::string describe(ESandbox2Capability capability);

//! Actively probe whether this host permits the namespace and mount
//! operations Sandbox2's forkserver performs before it can launch anything.
//!
//! Mirrors the forkserver's own sequence (see sandboxed_api
//! forkserver.cc::CreateInitialNamespaces and
//! namespace.cc::InitializeInitialNamespaces) rather than reading sysctls:
//! kernel.unprivileged_userns_clone and user.max_user_namespaces are
//! host-global and are inherited unchanged by a container whose seccomp or
//! LSM policy nonetheless denies the operation, so a passive check reports a
//! healthy environment on exactly the hosts where the sandbox cannot start.
//!
//! Runs entirely in forked children and never mutates this process: the
//! unshare()/mount() calls happen after fork(), so the caller's namespaces
//! and mount table are untouched whatever the outcome. Safe to call from a
//! multi-threaded process - unshare(CLONE_NEWUSER) requires a single-threaded
//! caller, which is why the probe forks first rather than unsharing inline.
ESandbox2Capability probeSandbox2Capability();

//! probeSandbox2Capability(), run at most once per process and cached.
//!
//! Every consumer of the verdict - the startup self-check and the spawn-time
//! routing decision - must use this rather than probing independently, so
//! the logged verdict and the route actually taken can never disagree. The
//! first call pays for the probe (one short-lived forked child); the result
//! is fixed for the life of the controller, which matches reality: whether
//! the host permits user namespaces does not change under a running process.
ESandbox2Capability sandbox2Capability();

//! The strongest confinement this host can give a launch that Elasticsearch
//! asked to be sandboxed (--requireSandbox). The controller walks this ladder
//! top-down and never silently skips a rung: each step down is logged with
//! the reason and what an administrator would have to change.
enum class EConfinementLevel {
    //! Full Sandbox2 isolation: private mount, PID and network namespaces,
    //! a minimal pivoted root filesystem, and the Sandbox2 syscall policy.
    E_Sandbox2,
    //! Sandbox2 is impossible on this host, but Landlock is available:
    //! filesystem access is confined by a Landlock ruleset, stacked with the
    //! in-process seccomp filter. No process, mount or network isolation.
    E_Landlock,
    //! Neither is possible. A --requireSandbox launch is refused outright -
    //! running untrusted models unconfined while the operator asked for a
    //! sandbox would be a silent downgrade.
    E_Unavailable
};

//! Everything the controller knows about this host's confinement options,
//! gathered once. The passive sysctl values are carried alongside the active
//! probe results because they are what tell an administrator which knob to
//! turn, even though they are never used to decide the level.
struct SHostConfinement {
    EConfinementLevel s_Level{EConfinementLevel::E_Unavailable};
    ESandbox2Capability s_Sandbox2{ESandbox2Capability::E_ProbeUnsupported};
    //! As returned by seccomp::landlockAbiVersion(): >= 1 the ABI version,
    //! 0 unsupported by the kernel, -1 denied by a seccomp filter or LSM.
    int s_LandlockAbi{0};
    //! /proc/sys/kernel/unprivileged_userns_clone, or "absent".
    std::string s_UnprivilegedUsernsClone{"absent"};
    //! /proc/sys/user/max_user_namespaces, or "absent".
    std::string s_MaxUserNamespaces{"absent"};
};

//! The ladder itself, as a pure function of the two probe results so that it
//! can be tested without a host that has (or lacks) each capability.
EConfinementLevel decideConfinement(ESandbox2Capability sandbox2, int landlockAbi);

//! This host's confinement, probed at most once per process and cached - the
//! single source of truth for both the startup self-check and every routing
//! decision, for the same reason as sandbox2Capability().
const SHostConfinement& hostConfinement();

//! Human-readable form of a landlockAbiVersion() result.
std::string describeLandlock(int landlockAbi);

//! What a system administrator must change for full Sandbox2 isolation on
//! this host, as one or two sentences. Derived from the diagnosed cause, not
//! a generic hint: kernel.unprivileged_userns_clone=0 and a container runtime
//! that blocks CLONE_NEWUSER look identical to the probe but need different
//! fixes, and only the sysctl value tells them apart. Empty when Sandbox2 is
//! already available.
std::string fullSandboxRemedy(const SHostConfinement& host);

//! The INFO-level explanation logged when a launch takes the Landlock rung.
std::string landlockFallbackMessage(const SHostConfinement& host,
                                    const std::string& processPath);

//! The ERROR-level explanation when neither rung is available, which is also
//! returned to Elasticsearch as the failure reason for the launch. Tells the
//! user to deactivate xpack.ml.trained_models.sandbox_enabled, because on
//! such a host that is the only way to run models at all.
std::string noConfinementMessage(const SHostConfinement& host, const std::string& processPath);

//! Log a one-time Sandbox2 environment self-check at INFO level, combining
//! the active capability probe above with the passive host facts that help
//! interpret it. No-op after the first call, and on platforms without
//! Sandbox2 support.
void logSandbox2EnvironmentSelfCheck();

} // namespace sandbox
} // namespace ml

#endif // INCLUDED_ml_sandbox_CSandbox2Diagnostics_h
