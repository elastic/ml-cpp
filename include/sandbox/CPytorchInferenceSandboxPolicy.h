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

#include <set>
#include <string>
#include <vector>

#ifdef SANDBOX2_AVAILABLE
#include <sandboxed_api/sandbox2/policybuilder.h>
#include <sandboxed_api/sandbox2/result.h>
#endif

namespace ml {
namespace sandbox {

struct SArgDirExtraction {
    std::set<std::string> m_ArgDirs;
    std::vector<std::string> m_RejectedPipeArgs;
    std::vector<std::string> m_PipeDirAliasMappings;
};

//! Extract mountable directories from named-pipe path arguments.
SArgDirExtraction extractArgDirs(const std::vector<std::string>& args);

#ifdef SANDBOX2_AVAILABLE

void logSandbox2SpawnContext(const std::string& absPath,
                             const std::string& binDir,
                             const std::string& libDir,
                             const SArgDirExtraction& argDirInfo,
                             const std::string& originalTmpdir,
                             bool tmpdirOverridden,
                             const std::string& sandboxeeTmpdir);

std::string sandboxPlatformArch();

std::string formatSandbox2Result(const sandbox2::Result& result);

//! Builds the syscall and filesystem policy for a sandboxed pytorch_inference.
//! Keep the syscall allowlist in sync with lib/seccomp/CSystemCallFilter_Linux.cc.
sandbox2::PolicyBuilder buildPytorchInferencePolicy(const std::string& binDir,
                                                    const std::string& libDir,
                                                    const std::set<std::string>& argDirs);

#endif // SANDBOX2_AVAILABLE

} // namespace sandbox
} // namespace ml

#endif // INCLUDED_ml_sandbox_CPytorchInferenceSandboxPolicy_h
