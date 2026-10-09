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
#ifndef INCLUDED_ml_sandbox_CPytorchInferenceSandboxPolicy_h
#define INCLUDED_ml_sandbox_CPytorchInferenceSandboxPolicy_h

#include <string>
#include <vector>

#ifdef SANDBOX2_AVAILABLE
#include "absl/status/statusor.h"
#include <sandboxed_api/sandbox2/policybuilder.h>
#endif

namespace ml {
namespace sandbox {

//! Reasons a path-bearing launch argument fails typed validation against the
//! pinned child-root contract (see validateChildIpcLaunchSpec below). Every
//! value here must fail *before* a policy is constructed; none of them widen
//! a mount to recover.
enum class EChildIpcPathRejection {
    E_NotAbsolute,    //!< value does not start with '/'.
    E_RootLevelPath,  //!< value has no mountable parent directory below '/'.
    E_ContainsDotDot, //!< value has a ".." path component.
    E_CanonicalizationFailed, //!< the trusted base or the value's parent directory could not be
    //!< resolved (realpath() on POSIX, _fullpath() on Windows).
    E_OutsideTrustedBase, //!< canonical parent is not beneath the trusted $TMPDIR.
    E_WrongDepth, //!< canonical parent is not exactly $TMPDIR/ml-child-ipc/<child-id>.
    E_ChildIdMismatch, //!< two path options resolved to a different <child-id>.
    E_MutableSymlinkOrAlias, //!< the literal and canonical parent directories diverge.
    E_Duplicate //!< the same literal argument was supplied more than once.
};

//! One rejected path-bearing argument and why.
struct SRejectedChildIpcPath {
    std::string s_Arg;
    EChildIpcPathRejection s_Reason;
};

//! A typed, validated launch specification for a single sandboxed
//! pytorch_inference child, derived from its path-bearing launch options
//! (input, output, restore, logPipe). Replaces raw argument-directory
//! inference: every accepted path is provably beneath the one pinned
//! per-child IPC root, never inferred from arbitrary argv content.
struct SChildIpcLaunchSpec {
    //! <child-id> path component shared by every accepted path option.
    //! Empty whenever the overall result is not s_Ok - either no
    //! recognized path option was present, or at least one was rejected
    //! (SChildIpcValidationResult clears the whole spec on any rejection).
    std::string s_ChildId;
    //! Canonical $TMPDIR/ml-child-ipc/<child-id> - the directory the native
    //! controller creates (mode 0700) before policy construction, and the
    //! only host directory CSandboxedProcessSpawner mounts into the sandbox
    //! (at this same path - see buildPytorchInferenceFilesystemPolicy).
    //! Empty iff s_ChildId is empty.
    std::string s_ChildIpcRoot;
    //! Canonical paths of every accepted path-bearing argument, always
    //! s_ChildIpcRoot plus exactly one leaf component.
    std::vector<std::string> s_PipePaths;
};

//! Result of validating a pytorch_inference launch command line against the
//! pinned child-root contract.
struct SChildIpcValidationResult {
    //! True only when at least one path option was present and every
    //! path option that was present was accepted. False means the caller
    //! must fail the spawn - never fall back to a partially-built policy.
    bool s_Ok = false;
    SChildIpcLaunchSpec s_Spec;
    std::vector<SRejectedChildIpcPath> s_Rejected;
};

//! Validate every input/output/restore/logPipe argument in args against the
//! pinned child-root contract: each must canonicalize to a parent directory
//! of exactly trustedTmpDir/ml-child-ipc/<child-id>, for one consistent
//! <child-id>, with no ".."; no relative, root, or out-of-root path; no
//! divergent literal/canonical parent; and no duplicate literal argument.
//! Scalar (non path-bearing) options are never inspected as candidate paths.
//! trustedTmpDir must already be the canonical form of the operator's
//! Environment.tmpDir(); this function does not itself decide what counts
//! as trusted.
//!
//! realpath() (POSIX) / _fullpath() (Windows) require their target to
//! already exist, so this can only succeed for a <child-id> whose
//! $TMPDIR/ml-child-ipc/<child-id> directory has already been created - see
//! ensureChildIpcDirectory() below, which every caller must run first.
SChildIpcValidationResult validateChildIpcLaunchSpec(const std::string& trustedTmpDir,
                                                     const std::vector<std::string>& args);

//! Outcome of ensureChildIpcDirectory().
enum class EChildIpcDirectoryOutcome {
    E_Ready, //!< $TMPDIR/ml-child-ipc/<child-id> exists now - freshly
             //!< created, or already present (a retry/restart reusing the
             //!< same child-id).
    E_NoPathOptions, //!< no path-bearing launch option had the expected
                     //!< $TMPDIR/ml-child-ipc/<child-id> literal shape, so
                     //!< there was no directory to create.
                     //!< validateChildIpcLaunchSpec() still runs and reports
                     //!< the precise rejection reason for such an argument.
    E_CreationFailed //!< mkdir() failed, or an existing path at the target
                     //!< is not an owner-only mode-0700 directory (regular
                     //!< file, symlink, looser permissions, wrong owner, ...).
};

//! Create $TMPDIR/ml-child-ipc/<child-id> (mode 0700) for the single
//! <child-id> implied by args' path-bearing launch options, *before*
//! validateChildIpcLaunchSpec() ever calls realpath()/canonicalize() on it.
//! This is the "native controller creates the per-child IPC directory" half
//! of the contract: Elasticsearch only ever constructs the path *strings*
//! it passes as --input=/--output=/--restore=/--logPipe= arguments; the
//! controller is responsible for making the directory those paths live in
//! exist (and be mode 0700) before anything tries to resolve or mount it.
//! Both production call sites that eventually reach
//! validateChildIpcLaunchSpec() - CSandboxedProcessSpawner_Linux.cc's
//! spawn() and CProcessSpawnerRouter::spawn() (via deriveDeploymentId(), for
//! the sandbox2_launch signal, which runs even on the legacy route) - must
//! call this first.
//!
//! The per-child directory is not removed here. Once pytorch_inference has
//! unlinked its FIFOs the directory is usually empty; CChildIpcDirectoryReaper
//! removes it when the controller observes that child exit (or after a failed
//! spawn that never bound a live pid). The parent $TMPDIR/ml-child-ipc
//! directory is never removed here.
//!
//! Idempotent: an already-existing directory is E_Ready, not an error, so a
//! retry/restart that reuses the same child-id never fails here. Uses only
//! a *literal* (pre-canonicalization) structural match of trustedTmpDir
//! against args - it is deliberately not a security gate. The real
//! canonical-base/symlink-alias/depth checks still run afterwards, in
//! validateChildIpcLaunchSpec(), against whatever directory this function
//! creates or finds already there.
EChildIpcDirectoryOutcome ensureChildIpcDirectory(const std::string& trustedTmpDir,
                                                  const std::vector<std::string>& args);

//! Returns the canonical per-child IPC root when validation succeeds, or the
//! literal $TMPDIR/ml-child-ipc/<child-id> path implied by args when a path
//! option matches the expected shape (used to reap directories left behind by
//! a failed spawn after ensureChildIpcDirectory()).
std::string perChildIpcRootFromArgs(const std::string& trustedTmpDir,
                                    const std::vector<std::string>& args);

#ifdef SANDBOX2_AVAILABLE

//! Builds the filesystem and network-shape portion of the pytorch_inference
//! Sandbox2 policy: minimized fixed mounts (fixedMountDecisions,
//! allowlistedEtcFiles - a read-only directory decision is mounted only if
//! its source actually exists on this host, since Sandbox2 fails the whole
//! spawn on a missing source), a private bounded tmpfs at /tmp, the one per-child
//! IPC root mounted at the same path inside and outside the sandbox (so
//! Elasticsearch's host-path argv still resolves), and the syscall allowlist shared
//! with the legacy BPF filter (seccomp::legacyBpfAllowedSyscalls and
//! seccomp::sandbox2ExplicitSyscalls, kept in sync per those headers' own
//! comments). Does not call TryBuild() - the caller owns final policy
//! construction so tests can inspect the builder before commit. Returns an
//! error when validated.s_Ok is false or s_ChildIpcRoot is not a canonical
//! $TMPDIR/ml-child-ipc/<child-id> directory.
absl::StatusOr<sandbox2::PolicyBuilder>
buildPytorchInferenceFilesystemPolicy(const std::string& binDir,
                                      const std::string& libDir,
                                      const SChildIpcValidationResult& validated,
                                      std::size_t tmpfsSizeBytes);

#endif // SANDBOX2_AVAILABLE

} // namespace sandbox
} // namespace ml

#endif // INCLUDED_ml_sandbox_CPytorchInferenceSandboxPolicy_h
