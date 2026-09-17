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

#include <exception>
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
#include <sys/syscall.h>
#include <unistd.h>

// The CentOS 7 based CI build image has kernel headers that predate pidfd, so
// __NR_pidfd_open / __NR_pidfd_send_signal may be undefined at build time even
// though the runtime kernel supports them. Both syscalls are 434/424 on every
// architecture we build for (x86_64 and aarch64), so fall back to those
// literals to keep the controller independent of the build image's header
// version.
#ifdef __NR_pidfd_open
#define ML_NR_pidfd_open __NR_pidfd_open
#else
#define ML_NR_pidfd_open 434
#endif
#ifdef __NR_pidfd_send_signal
#define ML_NR_pidfd_send_signal __NR_pidfd_send_signal
#else
#define ML_NR_pidfd_send_signal 424
#endif

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

#ifdef SANDBOX2_AVAILABLE
void closePidFdIfOpen(int pidFd) {
    if (pidFd >= 0) {
        ::close(pidFd);
    }
}

//! RAII owner for a pidfd between pidfd_open() and the registry insertion that
//! takes over its lifetime. Closes the descriptor on destruction unless
//! release() has handed ownership to the registry entry, so an exception (e.g.
//! bad_alloc from the map node allocation) thrown before registration cannot
//! leak the fd.
class CScopedPidFd {
public:
    explicit CScopedPidFd(int pidFd) : m_PidFd{pidFd} {}
    ~CScopedPidFd() { closePidFdIfOpen(m_PidFd); }
    CScopedPidFd(const CScopedPidFd&) = delete;
    CScopedPidFd& operator=(const CScopedPidFd&) = delete;
    int get() const { return m_PidFd; }
    //! Relinquish ownership: the caller (the registry entry) is now responsible
    //! for closing the descriptor.
    void release() { m_PidFd = -1; }

private:
    int m_PidFd;
};
#endif // SANDBOX2_AVAILABLE

} // namespace

namespace ml {
namespace sandbox {
namespace {

#ifdef SANDBOX2_AVAILABLE

//! The sandboxee's environment: the caller's, with ML_SANDBOXED=1 set exactly
//! once so pytorch_inference skips its in-process seccomp filter and relies on
//! the Sandbox2 policy instead.
std::vector<std::string> buildSandboxeeEnvironment() {
    std::vector<std::string> sandboxeeEnv;
    bool markerSet{false};
    for (char** env = environ; *env != nullptr; ++env) {
        std::string envVar{*env};
        if (envVar.find("ML_SANDBOXED=") == 0) {
            sandboxeeEnv.push_back("ML_SANDBOXED=1");
            markerSet = true;
        } else {
            sandboxeeEnv.push_back(std::move(envVar));
        }
    }
    if (markerSet == false) {
        sandboxeeEnv.push_back("ML_SANDBOXED=1");
    }
    return sandboxeeEnv;
}

//! An executor configured for a long-lived daemon sandboxee.
std::unique_ptr<sandbox2::Executor>
makeConfiguredExecutor(const std::string& absPath,
                       const std::vector<std::string>& fullArgs,
                       const std::string& binDir) {
    auto executor = std::make_unique<sandbox2::Executor>(
        absPath, fullArgs, buildSandboxeeEnvironment());

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
    return executor;
}

//! Log how a sandboxed pytorch_inference terminated.
//!
//! Runs on the monitor thread that owns the sandbox instance, so it deliberately
//! takes no spawner state - the caller does the PID bookkeeping under the lock.
void logSandboxeeTermination(core::CProcess::TPid sandboxPid, const sandbox2::Result& result) {
    switch (result.final_status()) {
    case sandbox2::Result::OK:
        if (result.reason_code() == 0) {
            LOG_DEBUG(<< "Sandboxed pytorch_inference (PID " << sandboxPid << ") has exited");
        } else {
            LOG_WARN(<< "Sandboxed pytorch_inference (PID " << sandboxPid
                     << ") has exited with exit code " << result.reason_code());
        }
        break;
    case sandbox2::Result::SIGNALED:
        if (result.reason_code() == SIGTERM) {
            LOG_INFO(<< "Sandboxed pytorch_inference (PID " << sandboxPid
                     << ") was terminated by signal " << SIGTERM);
        } else if (result.reason_code() == SIGKILL) {
            LOG_ERROR(<< "Sandboxed pytorch_inference (PID " << sandboxPid << ") was terminated by signal 9 (SIGKILL)."
                      << " This is likely due to the OOM killer.");
        } else {
            LOG_ERROR(<< "Sandboxed pytorch_inference (PID " << sandboxPid
                      << ") was terminated by signal " << result.reason_code());
        }
        break;
    default:
        LOG_ERROR(<< "Sandboxed pytorch_inference (PID " << sandboxPid
                  << ") terminated abnormally: " << formatSandbox2Result(result));
        if (result.final_status() == sandbox2::Result::VIOLATION) {
            LOG_ERROR(<< "Sandboxed pytorch_inference (PID " << sandboxPid
                      << ") seccomp violation: syscall=" << result.reason_code()
                      << " arch=" << sandboxPlatformArch());
        }
        break;
    }
}

#endif // SANDBOX2_AVAILABLE

} // namespace

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
    for (const std::string& arg : args) {
        fullArgs.push_back(arg);
    }

    // Binary and library directories to bind-mount. Note that libDir is the
    // SIBLING of binDir, not a child of it: the ML distribution lays out
    // <install>/bin/pytorch_inference alongside <install>/lib, so this strips
    // "bin" off binDir before appending "lib" rather than appending to binDir.
    const std::string binDir{absPath.substr(0, absPath.rfind('/'))};
    const std::string libDir{binDir.substr(0, binDir.rfind('/')) + "/lib"};

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

    logSandbox2SpawnContext(absPath, binDir, libDir, argDirInfo);

    auto sandboxPtr = std::make_unique<sandbox2::Sandbox2>(
        makeConfiguredExecutor(absPath, fullArgs, binDir), std::move(*policyResult));

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

    auto sandbox = std::shared_ptr<sandbox2::Sandbox2>(std::move(sandboxPtr));
    CScopedPidFd pidFdGuard{static_cast<int>(
        ::syscall(ML_NR_pidfd_open, static_cast<pid_t>(childPid), 0u))};

    const core::CProcess::TPid sandboxPid{childPid};
    std::uint64_t generation{0};

    // The sandboxee is a child of the Sandbox2 forkserver rather than of the
    // controller, so waitpid() never sees it. Own the sandbox instance on a
    // dedicated thread that keeps it alive for the lifetime of
    // pytorch_inference, waits for its result, and removes the registry entry
    // before logging termination.
    //
    // Each entry carries a monotonic generation so a stale monitor cannot erase
    // a re-registered PID after AwaitResult() reaps the process. terminateChild()
    // signals via pidfd when available so it targets the exact process even if
    // the numeric PID has been reused; on kernels <5.3 where pidfd_open returns
    // -1/ENOSYS the ::kill fallback retains a bounded residual window where a
    // recycled PID could be signalled.
    //
    // The thread co-owns the registry and the Sandbox2 shared_ptr rather than
    // capturing this: it can still be waiting on a live sandboxee when the
    // spawner is destroyed, and a raw pointer would be dangling by the time the
    // sandboxee exits.
    //
    // Registration and monitor-thread creation are the only operations that can
    // throw after the sandboxee is live (bad_alloc from the map node, or a
    // resource error from thread construction / detach()). All three share one
    // recovery path that reaps the orphaned sandboxee. The monitor is a named
    // thread rather than a temporary so that a detach() failure - which leaves
    // the thread joinable and running - can be joined during cleanup instead of
    // destroying a joinable thread and calling std::terminate().
    std::thread monitor;
    try {
        {
            std::lock_guard<std::mutex> lock(m_PidRegistry->s_Mutex);
            generation = ++m_PidRegistry->s_NextGeneration;
            const auto existing = m_PidRegistry->s_Children.find(sandboxPid);
            if (existing != m_PidRegistry->s_Children.end()) {
                closePidFdIfOpen(existing->second.s_PidFd);
                LOG_DEBUG(<< "Replacing stale registry entry for sandboxed pytorch_inference PID "
                          << sandboxPid << " before registering generation " << generation);
            }
            m_PidRegistry->s_Children[sandboxPid] = {generation, sandbox,
                                                     pidFdGuard.get()};
            // The registry entry now owns the pidfd; do not double-close it via
            // the guard's destructor on the happy path.
            pidFdGuard.release();
        }

        monitor = std::thread(
            [ sandboxPid, registry = m_PidRegistry, sandbox, generation ]() {
                const sandbox2::Result result{sandbox->AwaitResult()};
                {
                    std::lock_guard<std::mutex> lock(registry->s_Mutex);
                    const auto it = registry->s_Children.find(sandboxPid);
                    if (it != registry->s_Children.end() &&
                        it->second.s_Generation == generation) {
                        closePidFdIfOpen(it->second.s_PidFd);
                        registry->s_Children.erase(it);
                    }
                }
                logSandboxeeTermination(sandboxPid, result);
            });
        monitor.detach();
    } catch (const std::exception& e) {
        // Reap the sandboxee first so a running-but-not-yet-detached monitor
        // thread's AwaitResult() returns.
        sandbox->Kill();
        if (monitor.joinable()) {
            // detach() threw: the monitor thread is running and owns the registry
            // cleanup and its own AwaitResult(); just join it so it is not
            // destroyed joinable.
            monitor.join();
        } else {
            // Registration or thread construction threw: no monitor ran, so this
            // frame owns the reap and drops any registry entry it inserted.
            {
                std::lock_guard<std::mutex> lock(m_PidRegistry->s_Mutex);
                const auto it = m_PidRegistry->s_Children.find(sandboxPid);
                if (it != m_PidRegistry->s_Children.end() &&
                    it->second.s_Generation == generation) {
                    closePidFdIfOpen(it->second.s_PidFd);
                    m_PidRegistry->s_Children.erase(it);
                }
            }
            sandbox->AwaitResult();
        }
        // pidFdGuard closes the fd if registration never reached release().
        const std::string reason{"Failed to register or monitor sandboxed pytorch_inference: " +
                                 std::string{e.what()} + SANDBOX2_DISABLE_HINT};
        LOG_ERROR(<< reason);
        assignFailureReason(failureReason, reason);
        childPid = 0;
        return false;
    }

    LOG_INFO(<< "Spawned sandboxed pytorch_inference with PID " << childPid);

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
    std::lock_guard<std::mutex> lock(m_PidRegistry->s_Mutex);
    const auto it = m_PidRegistry->s_Children.find(pid);
    if (it == m_PidRegistry->s_Children.end()) {
        LOG_WARN(<< "Will not attempt to kill sandboxed process " << pid
                 << ": not a child process");
        return false;
    }

    const int pidFd = it->second.s_PidFd;
    if (pidFd >= 0) {
        if (::syscall(ML_NR_pidfd_send_signal, pidFd, SIGTERM, nullptr, 0u) == -1) {
            if (errno != ESRCH) {
                LOG_ERROR(<< "Failed to kill sandboxed process " << pid << ": "
                          << ::strerror(errno));
            }
            return false;
        }
        return true;
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

    std::lock_guard<std::mutex> lock(m_PidRegistry->s_Mutex);
    return m_PidRegistry->s_Children.find(pid) != m_PidRegistry->s_Children.end();
}

} // namespace sandbox
} // namespace ml
