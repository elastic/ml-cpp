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

//! Log a one-time Sandbox2 environment self-check at INFO level, combining
//! the active capability probe above with the passive host facts that help
//! interpret it. No-op after the first call, and on platforms without
//! Sandbox2 support.
void logSandbox2EnvironmentSelfCheck();

} // namespace sandbox
} // namespace ml

#endif // INCLUDED_ml_sandbox_CSandbox2Diagnostics_h
