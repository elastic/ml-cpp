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
    //! only host directory CSandboxedProcessSpawner maps to
    //! /run/elastic/ml-ipc. Empty iff s_ChildId is empty.
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
SChildIpcValidationResult validateChildIpcLaunchSpec(const std::string& trustedTmpDir,
                                                     const std::vector<std::string>& args);

#ifdef SANDBOX2_AVAILABLE

//! What buildPytorchInferenceFilesystemPolicy does with one of the seven
//! historically bulk-mounted fixed directories
//! (/lib /lib64 /usr/lib /usr/lib64 /etc /proc /sys). Mounting whole /etc or
//! binding the host's /proc or /sys directly is non-conformant.
enum class EFixedMountAction {
    E_MountReadOnlyDirectory, //!< the whole directory is demonstrated necessary read-only.
    E_MountNamespacedProcfs, //!< Sandbox2 supplies this inside the sandbox's own PID/mount namespace; never bind the host directory.
    E_Skip //!< not mapped at all; narrower entries (files) are added separately.
};

//! One fixed-mount decision plus the reason it is scoped that way.
struct SFixedMountDecision {
    std::string s_Path;
    EFixedMountAction s_Action;
    std::string s_Reason;
};

//! The minimization decision applied to each of the seven historically
//! bulk-mounted fixed directories, with its justification. /etc is Skip
//! (see allowlistedEtcFiles() for the narrower replacement); /proc and /sys
//! are the Sandbox2-namespaced procfs/sysfs, never a host bind (the
//! ml_sandbox_probe mechanism test asserts this held for a real launch, via
//! its pid_namespace check). /lib, /lib64, /usr/lib, /usr/lib64 remain whole
//! read-only directories: the dynamic loader resolves libtorch/glibc shared
//! objects from them at runtime from an unbounded, platform-dependent set,
//! so per-file allowlisting would duplicate the loader's own search logic.
const std::vector<SFixedMountDecision>& fixedMountDecisions();

//! Individual /etc files pytorch_inference/libtorch are demonstrated to
//! need, replacing a whole-/etc bind. Extend only with a named consumer.
const std::vector<std::string>& allowlistedEtcFiles();

//! Builds the filesystem and network-shape portion of the pytorch_inference
//! Sandbox2 policy: minimized fixed mounts (fixedMountDecisions,
//! allowlistedEtcFiles - a read-only directory decision is mounted only if
//! its source actually exists on this host, since Sandbox2 fails the whole
//! spawn on a missing source), a private bounded tmpfs at /tmp, the one per-child
//! IPC root mapped to /run/elastic/ml-ipc, and the syscall allowlist shared
//! with the legacy BPF filter
//! (seccomp::pytorch_inference::legacyBpfAllowedSyscalls, kept in sync per
//! that header's own comment). Does not call TryBuild() - the caller owns
//! final policy construction so tests can inspect the builder before
//! commit. spec must already be s_Ok from validateChildIpcLaunchSpec; this
//! function does not re-validate it.
sandbox2::PolicyBuilder
buildPytorchInferenceFilesystemPolicy(const std::string& binDir,
                                      const std::string& libDir,
                                      const SChildIpcLaunchSpec& spec,
                                      std::size_t tmpfsSizeBytes);

#endif // SANDBOX2_AVAILABLE

} // namespace sandbox
} // namespace ml

#endif // INCLUDED_ml_sandbox_CPytorchInferenceSandboxPolicy_h
