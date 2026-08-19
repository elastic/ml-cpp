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
#include <sandbox/CSandboxedProcessSpawner.h>

#include <core/CLogger.h>
#include <sandbox/CPytorchInferenceSandboxPolicy.h>
#include <sandbox/CSandbox2Diagnostics.h>

#include <memory>
#include <sstream>
#include <thread>
#include <utility>

#include <errno.h>
#include <limits.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

extern char** environ;

#ifdef SANDBOX2_AVAILABLE
#include <absl/time/time.h>

#include <sandboxed_api/sandbox2/executor.h>
#include <sandboxed_api/sandbox2/sandbox2.h>
#endif // SANDBOX2_AVAILABLE

namespace {

const std::string SANDBOX2_DISABLE_HINT{
    " If this host cannot support Sandbox2, an operator can disable sandboxing"
    " by setting xpack.ml.trained_models.sandbox_enabled: false, which runs"
    " pytorch_inference on the legacy path with the in-process seccomp filter."};

void assignFailureReason(std::string* failureReason, const std::string& reason) {
    if (failureReason != nullptr) {
        *failureReason = reason;
    }
}

} // namespace

namespace ml {
namespace sandbox {

CSandboxedProcessSpawner::CSandboxedProcessSpawner() = default;

CSandboxedProcessSpawner::~CSandboxedProcessSpawner() = default;

bool CSandboxedProcessSpawner::spawn(const std::string& processPath,
                                     const TStrVec& args,
                                     core::CProcess::TPid& childPid,
                                     std::string* failureReason) {
#ifdef SANDBOX2_AVAILABLE
    logSandbox2EnvironmentSelfCheck();

    // Resolve to absolute path - Sandbox2 requires absolute paths
    char resolvedPath[PATH_MAX];
    if (::realpath(processPath.c_str(), resolvedPath) == nullptr) {
        const std::string reason{"Cannot resolve path " + processPath + ": " +
                                 ::strerror(errno)};
        LOG_ERROR(<< reason);
        assignFailureReason(failureReason, reason);
        return false;
    }
    std::string absPath(resolvedPath);

    // Verify binary exists and is accessible
    struct stat binaryStat;
    if (::stat(absPath.c_str(), &binaryStat) != 0) {
        const std::string reason{"Cannot stat " + absPath + ": " + ::strerror(errno)};
        LOG_ERROR(<< reason);
        assignFailureReason(failureReason, reason);
        return false;
    }

    // Build argument vector
    std::vector<std::string> fullArgs;
    fullArgs.reserve(args.size() + 1);
    fullArgs.push_back(processPath);
    for (const auto& arg : args) {
        fullArgs.push_back(arg);
    }

    // Get binary and library directories
    std::string binDir = absPath.substr(0, absPath.rfind('/'));
    std::string libDir = binDir.substr(0, binDir.rfind('/')) + "/lib";

    // Extract directories from command-line arguments for pipe paths
    const SArgDirExtraction argDirInfo{extractArgDirs(args)};
    const std::set<std::string>& argDirs{argDirInfo.m_ArgDirs};

    // Build sandbox policy
    sandbox2::PolicyBuilder policyBuilder{buildPytorchInferencePolicy(binDir, libDir, argDirs)};

    auto policyResult = policyBuilder.TryBuild();
    if (!policyResult.ok()) {
        std::ostringstream statusMessage;
        statusMessage << policyResult.status();
        const std::string reason{"Failed to build Sandbox2 policy: " +
                                 statusMessage.str() + SANDBOX2_DISABLE_HINT};
        LOG_ERROR(<< reason);
        assignFailureReason(failureReason, reason);
        return false;
    }

    // Create executor with a sandbox marker.
    std::vector<std::string> customEnv;
    bool sandboxMarkerSet{false};
    for (char** env = environ; *env != nullptr; ++env) {
        std::string envVar(*env);
        if (envVar.find("ML_SANDBOXED=") == 0) {
            customEnv.push_back("ML_SANDBOXED=1");
            sandboxMarkerSet = true;
        } else {
            customEnv.push_back(envVar);
        }
    }
    if (!sandboxMarkerSet) {
        customEnv.push_back("ML_SANDBOXED=1");
    }

    logSandbox2SpawnContext(absPath, binDir, libDir, argDirInfo);

    std::unique_ptr<sandbox2::Executor> executor =
        std::make_unique<sandbox2::Executor>(absPath, fullArgs, customEnv);

    // Apply sandbox before exec since pytorch_inference doesn't use Sandbox2 client library
    executor->set_enable_sandbox_before_exec(true);
    executor->set_cwd(binDir);
    // pytorch_inference is a long-lived daemon that stays up for the whole
    // lifetime of a deployed model, not a run-to-completion sandboxee.
    // Sandbox2 defaults to a 120s wall-time limit and a 1024s CPU-time
    // limit, either of which would kill a healthy inference process (and
    // did, with Result::TIMEOUT, on the QA clusters). Disarm both.
    executor->limits()->set_walltime_limit(absl::ZeroDuration());
    executor->limits()->set_rlimit_cpu(RLIM64_INFINITY);
    // Sandbox2 defaults to rlimit_nofile=1024; libtorch thread pools and pipe I/O
    // under concurrent inference can approach that on QA clusters.
    executor->limits()->set_rlimit_nofile(65536);

    auto sandboxPtr = std::make_unique<sandbox2::Sandbox2>(std::move(executor),
                                                           std::move(*policyResult));

    if (!sandboxPtr->RunAsync()) {
        sandbox2::Result result{sandboxPtr->AwaitResult()};
        const std::string reason{
            "Sandbox2 failed to start pytorch_inference: " + formatSandbox2Result(result) +
            " - check that unprivileged user namespaces are enabled" + SANDBOX2_DISABLE_HINT};
        LOG_ERROR(<< reason);
        assignFailureReason(failureReason, reason);
        return false;
    }

    childPid = sandboxPtr->pid();
    if (childPid <= 0) {
        sandbox2::Result result{sandboxPtr->AwaitResult()};
        const std::string reason{"Sandbox2 returned invalid PID: " +
                                 formatSandbox2Result(result) + SANDBOX2_DISABLE_HINT};
        LOG_ERROR(<< reason);
        assignFailureReason(failureReason, reason);
        return false;
    }

    LOG_INFO(<< "Spawned sandboxed pytorch_inference with PID " << childPid);

    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        m_Pids.insert(childPid);
    }

    // The sandboxee is a child of the Sandbox2 forkserver rather than of the
    // controller, so waitpid() never sees it. Own the sandbox instance on a
    // dedicated thread that keeps it alive for the lifetime of
    // pytorch_inference, waits for its result, logs termination, and removes
    // the PID from the tracker so PID reuse cannot make terminateChild()
    // signal an unrelated process.
    {
        core::CProcess::TPid sandboxPid{childPid};
        CSandboxedProcessSpawner* self = this;
        std::thread(
            [ sandboxPid, self, sbx = std::move(sandboxPtr) ]() mutable {
                sandbox2::Result result{sbx->AwaitResult()};
                switch (result.final_status()) {
                case sandbox2::Result::OK:
                    if (result.reason_code() == 0) {
                        LOG_DEBUG(<< "Sandboxed pytorch_inference (PID "
                                  << sandboxPid << ") has exited");
                    } else {
                        LOG_WARN(<< "Sandboxed pytorch_inference (PID "
                                 << sandboxPid << ") has exited with exit code "
                                 << result.reason_code());
                    }
                    break;
                case sandbox2::Result::SIGNALED:
                    if (result.reason_code() == SIGTERM) {
                        LOG_INFO(<< "Sandboxed pytorch_inference (PID " << sandboxPid
                                 << ") was terminated by signal " << SIGTERM);
                    } else if (result.reason_code() == SIGKILL) {
                        LOG_ERROR(<< "Sandboxed pytorch_inference (PID " << sandboxPid
                                  << ") was terminated by signal 9 (SIGKILL)."
                                  << " This is likely due to the OOM killer.");
                    } else {
                        LOG_ERROR(<< "Sandboxed pytorch_inference (PID "
                                  << sandboxPid << ") was terminated by signal "
                                  << result.reason_code());
                    }
                    break;
                default: {
                    const std::string details{formatSandbox2Result(result)};
                    LOG_ERROR(<< "Sandboxed pytorch_inference (PID " << sandboxPid
                              << ") terminated abnormally: " << details);
                    if (result.final_status() == sandbox2::Result::VIOLATION) {
                        LOG_ERROR(<< "Sandboxed pytorch_inference (PID " << sandboxPid
                                  << ") seccomp violation: syscall=" << result.reason_code()
                                  << " arch=" << sandboxPlatformArch());
                    }
                    break;
                }
                }
                std::lock_guard<std::mutex> lock(self->m_Mutex);
                self->m_Pids.erase(sandboxPid);
            })
            .detach();
    }

    return true;
#else
    const std::string reason{"pytorch_inference built without Sandbox2 support - cannot spawn "
                             "securely" +
                             SANDBOX2_DISABLE_HINT};
    LOG_ERROR(<< reason);
    assignFailureReason(failureReason, reason);
    return false;
#endif // SANDBOX2_AVAILABLE
}

bool CSandboxedProcessSpawner::terminateChild(core::CProcess::TPid pid) {
    std::lock_guard<std::mutex> lock(m_Mutex);
    if (m_Pids.find(pid) == m_Pids.end()) {
        LOG_WARN(<< "Will not attempt to kill sandboxed process " << pid
                 << ": not a child process");
        return false;
    }

    if (::kill(pid, SIGTERM) == -1) {
        if (errno != ESRCH) {
            LOG_ERROR(<< "Failed to kill sandboxed process " << pid << ": "
                      << ::strerror(errno));
        }
        return false;
    }

    return true;
}

bool CSandboxedProcessSpawner::hasChild(core::CProcess::TPid pid) const {
    if (pid <= 0) {
        return false;
    }

    std::lock_guard<std::mutex> lock(m_Mutex);
    return m_Pids.find(pid) != m_Pids.end();
}

} // namespace sandbox
} // namespace ml
