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
#ifndef INCLUDED_ml_seccomp_CLandlockFilesystemPolicy_h
#define INCLUDED_ml_seccomp_CLandlockFilesystemPolicy_h

#include <string>
#include <vector>

namespace ml {
namespace seccomp {

//! \brief
//! Filesystem confinement that does not require user namespaces.
//!
//! DESCRIPTION:\n
//! seccomp-BPF cannot restrict which *paths* a process opens - its filter
//! program sees only register values, never the pointed-to path string - so
//! the legacy in-process filter leaves a sandboxee able to read and write
//! anything its uid can reach. Sandbox2 closes that gap with a mount
//! namespace and pivot_root, but creating one needs an unprivileged user
//! namespace, which some hosts forbid outright
//! (kernel.unprivileged_userns_clone=0, or a container runtime seccomp
//! profile that denies CLONE_NEWUSER - see
//! sandbox::probeSandbox2Capability()).
//!
//! Landlock is the kernel's answer to exactly that case: an unprivileged,
//! self-applied filesystem access-control ruleset needing no capabilities and
//! no namespaces. It gives the path confinement half of what the Sandbox2
//! rootfs gives, and composes with the existing seccomp filter.
//!
//! IMPLEMENTATION DECISIONS:\n
//! What Landlock does NOT provide, and must not be claimed for it: no process
//! table isolation (the sandboxee still sees host PIDs through /proc unless a
//! rule denies it), no private mount view (denied paths remain visible and
//! enumerable, they just cannot be opened), no network namespace, and no
//! effect on file descriptors that were already open when the ruleset was
//! applied. It is strictly weaker than the Sandbox2 route and is a fallback
//! for hosts that cannot run it, never a replacement.
//!
//! The Landlock UAPI headers are absent from the CI build image
//! (docker.elastic.co/ml-dev/ml-linux-build is CentOS7-based), so the
//! syscall numbers and structures are declared locally in the .cc, the same
//! way CMlLegacyBpfSyscallAllowlist.h falls back to raw numbers for statx,
//! rseq and clone3. Runtime ABI negotiation then decides which access rights
//! this kernel understands, so a binary built anywhere runs correctly on any
//! kernel.
struct SLandlockPaths {
    //! Directories and files the sandboxee may read, and nothing else - never
    //! write, and never execute. EXECUTE is deliberately withheld everywhere:
    //! dlopen() opens and maps a library without it (only execve() needs it),
    //! so withholding it makes Landlock alone refuse execve() even if the
    //! seccomp filter that normally denies it failed to install.
    std::vector<std::string> s_ReadOnly;

    //! Directories that may contain nothing but this process's own named
    //! pipes: it may create a FIFO, open it for reading or writing, and unlink
    //! it (CNamedPipeFactory unlinks each FIFO once connected). Creating a
    //! regular file, directory, symlink or socket is denied, so the directory
    //! cannot be used to fill the disk or stage data.
    std::vector<std::string> s_PipeDirectories;
};

//! Outcome of applyLandlockFilesystemPolicy().
enum class ELandlockOutcome {
    //! The ruleset was applied; the process is now confined.
    E_Applied,
    //! This kernel has no Landlock support (the syscall returned ENOSYS, or
    //! the LSM is not enabled in the bootloader's lsm= list).
    E_Unsupported,
    //! Landlock exists but the ruleset could not be built or applied.
    E_Failed
};

//! Human-readable one-line form of \p outcome.
std::string describe(ELandlockOutcome outcome);

//! Query, without applying anything, whether this process could use Landlock.
//!
//! \return the Landlock ABI version (>= 1) the kernel supports; 0 if the
//! kernel has no Landlock (ENOSYS: older than 5.13 or compiled out;
//! EOPNOTSUPP: built in but absent from the bootloader's lsm= list); or -1
//! if the kernel supports it but a seccomp filter or LSM denied the query.
//! The three cases have different remedies, so they are never collapsed.
//! Safe to call from any process: asking for the ABI version creates no
//! ruleset and restricts nothing.
int landlockAbiVersion();

//! The per-child IPC directory that holds \p logPipePath, i.e. the path with
//! its last component removed, provided that directory has the shape
//! .../ml-child-ipc/<deployment-id>. Empty for anything else - in particular
//! the legacy flat layout, where the pipes sit directly in the shared
//! $TMPDIR. The Landlock pipe-directory grant includes unlinking, so it may
//! only ever be given to a directory that holds nothing but this process's
//! own pipes; callers must refuse to confine (and must not run) otherwise.
inline std::string perChildIpcDirectory(const std::string& logPipePath) {
    static const std::string PER_CHILD_PARENT{"ml-child-ipc"};
    // Absolute only: a relative path would make the Landlock rule depend on
    // the process's working directory. Elasticsearch always sends absolute
    // pipe paths.
    if (logPipePath.empty() || logPipePath[0] != '/') {
        return std::string{};
    }
    const std::size_t fileSlash{logPipePath.rfind('/')};
    if (fileSlash == std::string::npos || fileSlash == 0) {
        return std::string{};
    }
    const std::string directory{logPipePath.substr(0, fileSlash)};
    const std::size_t idSlash{directory.rfind('/')};
    if (idSlash == std::string::npos || idSlash + 1 == directory.size()) {
        return std::string{};
    }
    const std::size_t parentSlash{directory.rfind('/', idSlash - 1)};
    const std::size_t parentStart{parentSlash == std::string::npos ? 0 : parentSlash + 1};
    if (idSlash == 0 ||
        directory.compare(parentStart, idSlash - parentStart, PER_CHILD_PARENT) != 0 ||
        idSlash - parentStart != PER_CHILD_PARENT.size()) {
        return std::string{};
    }
    return directory;
}

//! The paths pytorch_inference needs, derived from its own resolved binary
//! location and the directory its IPC pipes live in.
//!
//! Derived from a trace of every Landlock-mediated operation pytorch_inference
//! performs after the ruleset is applied - startup, model load and inference
//! of the quantized ELSER model with two threads - plus the exceptions noted
//! at each entry. Sensitive trees (/proc, /etc) are granted as exact files;
//! whole directories are granted only where the contents are not sensitive
//! and the exact set varies by CPU (the bundled library directory, from which
//! oneMKL dlopen()s CPU-specific kernels, and the CPU topology in sysfs).
//!
//! \param ipcDirectory the per-child IPC directory
//! $TMPDIR/ml-child-ipc/<deployment-id> holding the --input/--output/
//! --restore/--logPipe pipes. Callers must not pass a shared directory: its
//! pipe-directory rights include unlinking, which in a directory shared
//! between deployments would let one sandboxee delete another's pipes.
SLandlockPaths pytorchInferenceLandlockPaths(const std::string& ipcDirectory);

//! Apply \p paths as a Landlock ruleset to the calling process, denying every
//! filesystem access the ABI can describe that the rules do not grant.
//!
//! Irreversible for the lifetime of the process, and inherited by children.
//! Sets PR_SET_NO_NEW_PRIVS, which Landlock requires of an unprivileged
//! caller. Must be called before any untrusted input is processed.
ELandlockOutcome applyLandlockFilesystemPolicy(const SLandlockPaths& paths);

} // namespace seccomp
} // namespace ml

#endif // INCLUDED_ml_seccomp_CLandlockFilesystemPolicy_h
