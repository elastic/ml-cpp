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

#include <cerrno>
#include <exception>
#include <sstream>
#include <thread>
#include <utility>

// classifyPidFdOutcome (design.md gate V9) is a pure function with no
// syscalls or Sandbox2 types in its signature, so - unlike the rest of this
// file - it is defined below outside the SANDBOX2_AVAILABLE-gated block: it
// must compile, and be unit-testable, on every platform, matching this TU's
// own "compiled unconditionally" contract (see the comment above the
// SANDBOX2_AVAILABLE block). <cerrno> (for ENOSYS) is therefore included
// unconditionally too, rather than inside that block alongside <errno.h>.

// This translation unit is compiled unconditionally (see lib/sandbox/CMakeLists.txt
// - it is added to SRCS the same way lib/core/CMakeLists.txt unconditionally
// builds CDetachedProcessSpawner.cc), so every symbol outside the
// SANDBOX2_AVAILABLE-gated block below must compile with no Sandbox2/Linux
// headers available at all. The real spawn() logic - and everything that
// needs sandbox2:: types or Linux-only syscalls - lives inside that block;
// non-Linux/no-Sandbox2 configures fall through to the "not built with
// Sandbox2 support" stub path at the bottom of spawn(), matching
// CPytorchInferenceSandboxPolicy.cc's split.
#ifdef SANDBOX2_AVAILABLE

#include <errno.h>
#include <limits.h>
#include <signal.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <absl/time/time.h>

#include <sandboxed_api/sandbox2/executor.h>
#include <sandboxed_api/sandbox2/sandbox2.h>

// environ is a global variable from the C runtime library.
extern char** environ;

// The CentOS 7 based CI build image has kernel headers that predate pidfd, so
// __NR_pidfd_open may be undefined at build time even though the runtime
// kernel supports it. pidfd_open is syscall number 434 on every architecture
// ml-cpp builds for (x86_64 and aarch64); fall back to that literal so the
// spawner does not depend on the build image's header version. Task 3 owns
// classifying what a failed acquisition means (ENOSYS vs. a resource error);
// this task only needs the raw syscall wrapped behind the injectable seam.
#ifdef __NR_pidfd_open
#define ML_NR_pidfd_open __NR_pidfd_open
#else
#define ML_NR_pidfd_open 434
#endif

// Same rationale as ML_NR_pidfd_open above: pidfd_send_signal is syscall
// number 424 on every architecture ml-cpp builds for (x86_64 and aarch64),
// so fall back to that literal when the build image's kernel headers
// predate it. Used by terminateChild()'s E_Acquired path (SIGTERM request
// via the held pidfd) - the only place this file sends a signal to a
// sandboxee by identity-bound handle rather than by recycled numeric PID.
#ifdef __NR_pidfd_send_signal
#define ML_NR_pidfd_send_signal __NR_pidfd_send_signal
#else
#define ML_NR_pidfd_send_signal 424
#endif

#endif // SANDBOX2_AVAILABLE

namespace ml {
namespace sandbox {

// Defined outside the SANDBOX2_AVAILABLE-gated block below (unlike
// everything else in this file): a pure function with no syscalls, no
// Sandbox2 types, and no platform-specific behaviour, so it must compile -
// and be unit-testable - on every configure, matching this TU's
// "compiled unconditionally" contract (see the file-level comment above).
CSandboxedProcessSpawner::EPidFdOutcome CSandboxedProcessSpawner::classifyPidFdOutcome(
    const CSandboxedProcessSpawner::SPidFdAcquisitionResult& result) {
    if (result.s_Fd >= 0) {
        return EPidFdOutcome::E_Acquired;
    }
    if (result.s_Errno == ENOSYS) {
        return EPidFdOutcome::E_KernelUnsupported;
    }
    return EPidFdOutcome::E_Failed;
}

#ifdef SANDBOX2_AVAILABLE

namespace {

//! RAII owner for a pidfd between acquisition and the registry insertion
//! that takes over its lifetime. Closes the descriptor on destruction unless
//! release() has handed ownership to the registry entry, so an exception
//! (e.g. std::bad_alloc from the map node allocation, injected via the
//! registry-allocation seam) thrown before registration cannot leak the fd.
class CScopedPidFd {
public:
    explicit CScopedPidFd(int pidFd) : m_PidFd{pidFd} {}
    ~CScopedPidFd() {
        if (m_PidFd >= 0) {
            ::close(m_PidFd);
        }
    }
    CScopedPidFd(const CScopedPidFd&) = delete;
    CScopedPidFd& operator=(const CScopedPidFd&) = delete;
    int get() const { return m_PidFd; }
    //! Relinquish ownership: the caller (the registry entry) is now
    //! responsible for closing the descriptor.
    void release() { m_PidFd = -1; }

private:
    int m_PidFd;
};

//! Close a pidfd that a registry entry owns, tolerating an already-released
//! (-1) value.
void closePidFdIfOpen(int pidFd) {
    if (pidFd >= 0) {
        ::close(pidFd);
    }
}

//! Non-throwing kill-and-reap guard (design.md LI1, closing MG3). Armed
//! immediately when RunAsync() succeeds and pid() is captured
//! (E_IdentityCaptured) - before any potentially-throwing operation
//! (registry insertion, monitor-thread construction, detach()) - and
//! disarmed only after registry insertion AND monitor handoff both succeed
//! (E_Monitoring, LI2). Every early return on the path between those two
//! points goes through this guard's destructor rather than a hand-written
//! duplicate cleanup block (design.md LI3), so there is exactly one cleanup
//! owner for "launched but not yet fully handed off".
//!
//! Holds its own shared_ptr<Sandbox2> copy (not a raw, non-owning pointer)
//! so its lifetime is entirely self-sufficient: it does not matter what
//! order this guard is declared in relative to other shared_ptr-holding
//! locals in spawn() (e.g. `sandbox`, `child.s_Sandbox`), nor what order
//! those locals get destroyed in during stack unwinding on a failure path.
//! A raw pointer previously used here relied on some other local staying
//! alive for the guard's own destructor to run safely against; if that
//! local's declaration (and therefore destruction) order ever changed, or
//! if the object's last owning shared_ptr was destroyed before this guard
//! during unwinding, the guard's destructor would call Kill() on a dangling
//! pointer. Holding an owning copy makes that structurally impossible: this
//! guard is always one of the owners, so the object cannot be freed before
//! this guard's own destructor has run.
//!
//! The destructor must not throw: it runs during stack unwinding on the
//! failure paths this guard exists to cover, and a second exception there
//! would call std::terminate. Kill() and the AwaitResult seam are wrapped in
//! a catch-all for that reason; this task does not classify what Kill()
//! itself can fail with (Task 3 scope), only ensures a throw from it cannot
//! escape a destructor.
class CKillAndReapGuard {
public:
    CKillAndReapGuard(std::shared_ptr<sandbox2::Sandbox2> sandbox,
                       CSandboxedProcessSpawner::TAwaitResultFn awaitResultFn)
        : m_Sandbox{std::move(sandbox)}, m_AwaitResultFn{std::move(awaitResultFn)} {}

    ~CKillAndReapGuard() {
        if (m_Armed && m_Sandbox) {
            try {
                m_Sandbox->Kill();
                if (m_AwaitResultFn) {
                    m_AwaitResultFn(*m_Sandbox);
                } else {
                    m_Sandbox->AwaitResult();
                }
            } catch (...) {
                // Never let an exception escape a destructor; this guard's
                // whole purpose is bounded, best-effort cleanup on a
                // failure path that is already unwinding.
            }
        }
    }

    CKillAndReapGuard(const CKillAndReapGuard&) = delete;
    CKillAndReapGuard& operator=(const CKillAndReapGuard&) = delete;

    //! Called once registry insertion AND monitor handoff have both
    //! succeeded (E_Monitoring, LI2). After this, the monitor thread owns
    //! calling the (possibly injected) AwaitResult seam exactly once.
    void disarm() { m_Armed = false; }

private:
    std::shared_ptr<sandbox2::Sandbox2> m_Sandbox;
    CSandboxedProcessSpawner::TAwaitResultFn m_AwaitResultFn;
    bool m_Armed{true};
};

//! The sandboxee's environment: the caller's, with ML_SANDBOXED=1 set
//! exactly once so pytorch_inference skips its in-process seccomp filter and
//! relies on the Sandbox2 policy instead.
std::vector<std::string> buildSandboxeeEnvironment() {
    std::vector<std::string> sandboxeeEnv;
    bool markerSet{false};
    for (char** env = ::environ; *env != nullptr; ++env) {
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

//! An executor configured for a long-lived daemon sandboxee, matching the
//! frozen pre-rebuild reference's timeout/rlimit relaxations (a run-to-
//! completion default would kill a healthy, long-lived pytorch_inference).
std::unique_ptr<sandbox2::Executor> makeConfiguredExecutor(const std::string& absPath,
                                                             const std::vector<std::string>& fullArgs,
                                                             const std::string& binDir) {
    auto executor =
        std::make_unique<sandbox2::Executor>(absPath, fullArgs, buildSandboxeeEnvironment());
    executor->set_enable_sandbox_before_exec(true);
    executor->set_cwd(binDir);
    executor->limits()->set_walltime_limit(absl::ZeroDuration());
    executor->limits()->set_rlimit_cpu(RLIM64_INFINITY);
    executor->limits()->set_rlimit_nofile(65536);
    return executor;
}

//! Production default for the pidfd-acquisition seam: the raw pidfd_open
//! syscall. classifyPidFdOutcome() (defined below, outside this
//! SANDBOX2_AVAILABLE block) turns this raw fd/errno pair into the
//! Acquired/KernelUnsupported/Failed classification spawn() acts on. No
//! numeric-kill(pid) fallback is introduced anywhere by this file.
CSandboxedProcessSpawner::SPidFdAcquisitionResult
defaultPidFdOpen(core::CProcess::TPid pid) {
    CSandboxedProcessSpawner::SPidFdAcquisitionResult result;
    result.s_Fd = static_cast<int>(::syscall(ML_NR_pidfd_open, static_cast<pid_t>(pid), 0u));
    result.s_Errno = (result.s_Fd < 0) ? errno : 0;
    return result;
}

//! Production default for the monitor-thread creation/detach seam:
//! construct a std::thread running monitorBody and detach it, converting
//! any std::system_error from either step into a false return (LI8) instead
//! of letting it propagate as an exception - the caller (spawn()) treats a
//! false return the same way regardless of which step failed.
bool defaultMonitorLaunch(std::function<void()> monitorBody) {
    try {
        std::thread monitor{std::move(monitorBody)};
        monitor.detach();
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

//! Log how a sandboxed pytorch_inference terminated. Runs on the monitor
//! thread that owns the sandbox instance, so it deliberately takes no
//! spawner state - the caller does the registry bookkeeping under the lock.
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
        LOG_INFO(<< "Sandboxed pytorch_inference (PID " << sandboxPid
                 << ") was terminated by signal " << result.reason_code());
        break;
    default:
        LOG_ERROR(<< "Sandboxed pytorch_inference (PID " << sandboxPid
                  << ") terminated abnormally, final_status=" << result.final_status());
        break;
    }
}

//! Production default for the registry-allocation seam: lock, allocate the
//! next generation, replace any stale entry for the same PID (closing its
//! pidfd first), insert, and return the new generation. A test overriding
//! this seam can throw (e.g. std::bad_alloc) or return a colliding
//! generation to exercise LI8 deterministically.
std::uint64_t
defaultRegistryInsert(CSandboxedProcessSpawner::SPidRegistry& registry,
                       core::CProcess::TPid pid,
                       CSandboxedProcessSpawner::SSandboxedChild child) {
    std::lock_guard<std::mutex> lock(registry.s_Mutex);
    const std::uint64_t generation{++registry.s_NextGeneration};
    const auto existing = registry.s_Children.find(pid);
    if (existing != registry.s_Children.end()) {
        closePidFdIfOpen(existing->second.s_PidFd);
        LOG_DEBUG(<< "Replacing stale registry entry for sandboxed pytorch_inference PID " << pid
                  << " before registering generation " << generation);
    }
    child.s_Generation = generation;
    child.s_State = CSandboxedProcessSpawner::EChildLifecycleState::E_Registered;
    registry.s_Children[pid] = std::move(child);
    return generation;
}

} // namespace

#endif // SANDBOX2_AVAILABLE

CSandboxedProcessSpawner::CSandboxedProcessSpawner() = default;

CSandboxedProcessSpawner::CSandboxedProcessSpawner(TPidFdOpenFn pidFdOpenFn,
                                                     TRegistryInsertFn registryInsertFn,
                                                     TMonitorLaunchFn monitorLaunchFn
#ifdef SANDBOX2_AVAILABLE
                                                     ,
                                                     TAwaitResultFn awaitResultFn
#endif
                                                     )
    : m_PidFdOpenFn{std::move(pidFdOpenFn)}, m_RegistryInsertFn{std::move(registryInsertFn)},
      m_MonitorLaunchFn{std::move(monitorLaunchFn)}
#ifdef SANDBOX2_AVAILABLE
      ,
      m_AwaitResultFn{std::move(awaitResultFn)}
#endif
{
}

CSandboxedProcessSpawner::~CSandboxedProcessSpawner() = default;

bool CSandboxedProcessSpawner::spawn(const std::string& processPath,
                                      const TStrVec& args,
                                      core::CProcess::TPid& childPid) {
    childPid = 0;

#ifdef SANDBOX2_AVAILABLE

    // Resolve to absolute path - Sandbox2 requires absolute paths.
    char resolvedPath[PATH_MAX];
    if (::realpath(processPath.c_str(), resolvedPath) == nullptr) {
        LOG_ERROR(<< "Cannot resolve path " << processPath << ": " << ::strerror(errno));
        return false;
    }
    const std::string absPath(resolvedPath);

    struct stat binaryStat;
    if (::stat(absPath.c_str(), &binaryStat) != 0) {
        LOG_ERROR(<< "Cannot stat " << absPath << ": " << ::strerror(errno));
        return false;
    }

    TStrVec fullArgs;
    fullArgs.reserve(args.size() + 1);
    fullArgs.push_back(processPath);
    for (const std::string& arg : args) {
        fullArgs.push_back(arg);
    }

    // PR C interlock (design.md V16): validate every path-bearing launch
    // argument against the pinned child-root contract *before* a policy is
    // ever constructed. s_Ok == false must fail the spawn outright - never
    // fall back to a partially-built policy.
    const char* tmpDirEnv{::getenv("TMPDIR")};
    const std::string trustedTmpDir{tmpDirEnv != nullptr ? tmpDirEnv : "/tmp"};
    const SChildIpcValidationResult validated{validateChildIpcLaunchSpec(trustedTmpDir, args)};
    if (validated.s_Ok == false) {
        std::ostringstream rejected;
        for (const SRejectedChildIpcPath& r : validated.s_Rejected) {
            rejected << " [" << r.s_Arg << ": reason=" << static_cast<int>(r.s_Reason) << ']';
        }
        LOG_ERROR(<< "Rejected pytorch_inference child-IPC launch spec for " << processPath
                  << ':' << rejected.str());
        return false;
    }

    // Binary and library directories to bind-mount. libDir is the SIBLING of
    // binDir, not a child of it: the ML distribution lays out
    // <install>/bin/pytorch_inference alongside <install>/lib, so this
    // strips "bin" off binDir before appending "lib" rather than appending
    // to binDir. Derived from processPath rather than added as a
    // CSandboxedProcessSpawner constructor parameter: spawn()'s signature is
    // pinned by the plan and every known caller launches pytorch_inference
    // from that fixed distribution layout, so there is nothing a caller-
    // supplied binDir/libDir would let a test or caller express that
    // deriving from absPath does not already cover.
    const std::string binDir{absPath.substr(0, absPath.rfind('/'))};
    const std::string libDir{binDir.substr(0, binDir.rfind('/')) + "/lib"};

    // A private, bounded tmpfs at /tmp inside the sandbox - never the host's
    // shared /tmp. 16 MiB matches the size the PR C mechanism test already
    // exercises end-to-end (CPytorchInferenceSandboxPolicyMechanismTest_Linux.cc);
    // revisit if a real pytorch_inference workload needs more scratch space.
    const std::size_t tmpfsSizeBytes{16 * 1024 * 1024};

    sandbox2::PolicyBuilder policyBuilder{
        buildPytorchInferenceFilesystemPolicy(binDir, libDir, validated.s_Spec, tmpfsSizeBytes)};

    auto policyResult = policyBuilder.TryBuild();
    if (!policyResult.ok()) {
        LOG_ERROR(<< "Failed to build Sandbox2 policy for " << processPath);
        return false;
    }

    auto sandboxPtr = std::make_unique<sandbox2::Sandbox2>(
        makeConfiguredExecutor(absPath, fullArgs, binDir), std::move(*policyResult));

    // Take shared ownership immediately, before RunAsync() ever launches
    // anything - not after pid() is captured. This conversion can itself
    // throw (MG3's control-block allocation failure), but nothing has been
    // launched yet at this point, so sandboxPtr's own (plain) destructor is
    // sufficient cleanup on that failure; no Kill()/AwaitResult() is needed
    // for a sandboxee that was never started. Doing this early - rather
    // than arming CKillAndReapGuard on a raw, non-owning pointer into the
    // still-unique_ptr-owned object and converting to shared_ptr afterward
    // - means the guard constructed below always holds a genuine owning
    // shared_ptr copy, making its cleanup self-sufficient regardless of
    // declaration/destruction order among the other shared_ptr-holding
    // locals later in this function (`sandbox` itself, `child.s_Sandbox`).
    std::shared_ptr<sandbox2::Sandbox2> sandbox;
    try {
        sandbox = std::shared_ptr<sandbox2::Sandbox2>(std::move(sandboxPtr));
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to take shared ownership of a sandboxee for " << processPath << ": "
                  << e.what());
        return false;
    }

    // E_Launched.
    if (!sandbox->RunAsync()) {
        sandbox->AwaitResult();
        LOG_ERROR(<< "Sandbox2 failed to start " << processPath);
        return false;
    }

    childPid = sandbox->pid();
    if (childPid <= 0) {
        sandbox->AwaitResult();
        childPid = 0;
        LOG_ERROR(<< "Sandbox2 returned an invalid PID for " << processPath);
        return false;
    }

    const core::CProcess::TPid sandboxPid{childPid};

    // E_IdentityCaptured (LI1): arm the kill-and-reap guard now that the
    // sandboxee is actually running. The guard takes its own shared_ptr
    // copy of `sandbox` (see CKillAndReapGuard's comment), so it remains
    // valid through every early return below - registry-insert throw,
    // monitor-launch-span throw, monitor-launch-seam false - independent of
    // when `sandbox`/`child.s_Sandbox` themselves get destroyed during
    // stack unwinding.
    CKillAndReapGuard killAndReapGuard{sandbox, m_AwaitResultFn};

    const SPidFdAcquisitionResult pidFdResult{m_PidFdOpenFn ? m_PidFdOpenFn(sandboxPid)
                                                             : defaultPidFdOpen(sandboxPid)};
    CScopedPidFd pidFdGuard{pidFdResult.s_Fd};
    const EPidFdOutcome pidFdOutcome{classifyPidFdOutcome(pidFdResult)};

    // MG2/LI8: an errno other than ENOSYS (ESRCH, EMFILE, ENFILE, ...) is a
    // resource/identity error, not "no kernel support" for pidfd - it must
    // never be treated the same as E_KernelUnsupported. Fail registration
    // outright rather than register a child whose termination would need an
    // undefined fallback. pidFdGuard closes any fd this path somehow still
    // holds; killAndReapGuard (still armed) Kill()s/awaits the sandboxee.
    if (pidFdOutcome == EPidFdOutcome::E_Failed) {
        LOG_ERROR(<< "pidfd_open failed for sandboxed process " << processPath << " (PID "
                  << sandboxPid << ") with errno " << pidFdResult.s_Errno << " ("
                  << ::strerror(pidFdResult.s_Errno)
                  << "); refusing to register a child with an undefined termination fallback");
        childPid = 0;
        return false; // killAndReapGuard fires here; pidFdGuard closes any fd on unwind.
    }

    SSandboxedChild child;
    child.s_State = EChildLifecycleState::E_IdentityCaptured;
    child.s_Sandbox = sandbox;
    child.s_PidFd = pidFdGuard.get();
    child.s_PidFdOutcome = pidFdOutcome;
    child.s_Outcome = std::make_shared<CCasOutcomeLatch>();

    std::uint64_t generation{0};
    try {
        generation = m_RegistryInsertFn ? m_RegistryInsertFn(*m_PidRegistry, sandboxPid, child)
                                         : defaultRegistryInsert(*m_PidRegistry, sandboxPid, child);
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to register sandboxed process " << processPath << " (PID "
                  << sandboxPid << "): " << e.what());
        childPid = 0;
        return false; // killAndReapGuard fires here; pidFdGuard still owns the fd.
    }
    // E_Registered. The registry entry now owns the pidfd; do not double-
    // close it via pidFdGuard's destructor on this path.
    pidFdGuard.release();

    // Erase the registry entry this call just inserted, matching by
    // generation (in case a racing call already replaced it). Shared by
    // every failure path between a successful registry insertion and a
    // successful monitor handoff, since no monitor thread exists on any of
    // those paths to ever perform that erase itself.
    const auto eraseRegistryEntry = [this, sandboxPid, generation]() {
        std::lock_guard<std::mutex> lock(m_PidRegistry->s_Mutex);
        const auto it = m_PidRegistry->s_Children.find(sandboxPid);
        if (it != m_PidRegistry->s_Children.end() && it->second.s_Generation == generation) {
            closePidFdIfOpen(it->second.s_PidFd);
            m_PidRegistry->s_Children.erase(it);
        }
    };

    // The sandboxee is a child of the Sandbox2 forkserver rather than of the
    // controller, so waitpid() never sees it. Own the sandbox instance on a
    // dedicated monitor thread that keeps it alive for the lifetime of
    // pytorch_inference, waits for its result (via the injectable
    // AwaitResult seam), and removes the registry entry before logging
    // termination. The thread co-owns the registry and the Sandbox2
    // shared_ptr rather than capturing this: it can still be waiting on a
    // live sandboxee when the spawner is destroyed (LI9), and a raw pointer
    // back to the spawner would be dangling by then.
    //
    // Everything from copying m_PidRegistry/m_AwaitResultFn through
    // launching the monitor thread runs inside a try/catch: those copies
    // and constructing monitorBody's capture list can themselves throw
    // (e.g. std::bad_alloc copying a std::function), and left unguarded
    // that exception would otherwise escape spawn() uncaught, leaking the
    // just-inserted registry entry. Catching here ensures every throw in
    // this span still erases the registry entry and returns false with
    // childPid == 0 (LI3); killAndReapGuard's destructor performs the
    // Kill()/await half of cleanup on unwind either way.
    bool monitorStarted{false};
    try {
        const TPidRegistryPtr registry{m_PidRegistry};
        const TAwaitResultFn awaitResultFn{m_AwaitResultFn};
        auto monitorBody = [sandboxPid, registry, sandbox, generation, awaitResultFn]() {
            const sandbox2::Result result{awaitResultFn ? awaitResultFn(*sandbox)
                                                          : sandbox->AwaitResult()};
            // MG4/V11: this thread's completion and a (currently unwired -
            // Task 3 scope stops at this call site; no external timeout
            // caller exists yet) timeout path both race to decide who
            // performs cleanup for the same child. Route that decision
            // through exactly one tryResolve() call on the child's own CAS
            // latch rather than an ad-hoc boolean - if a timeout caller
            // resolves the latch to E_TimedOut first, this call loses the
            // race and must not also erase the registry entry or log
            // termination (the timeout path owns that instead).
            bool completionWonRace{true};
            {
                std::lock_guard<std::mutex> lock(registry->s_Mutex);
                const auto it = registry->s_Children.find(sandboxPid);
                if (it != registry->s_Children.end() && it->second.s_Generation == generation) {
                    if (it->second.s_Outcome) {
                        EOutcomeState desired{EOutcomeState::E_Completed};
                        completionWonRace = it->second.s_Outcome->tryResolve(desired);
                    }
                    if (completionWonRace) {
                        closePidFdIfOpen(it->second.s_PidFd);
                        registry->s_Children.erase(it);
                    }
                }
            }
            if (completionWonRace) {
                logSandboxeeTermination(sandboxPid, result);
            }
        };

        monitorStarted = m_MonitorLaunchFn ? m_MonitorLaunchFn(std::move(monitorBody))
                                            : defaultMonitorLaunch(std::move(monitorBody));
    } catch (const std::exception& e) {
        eraseRegistryEntry();
        LOG_ERROR(<< "Failed to launch monitor thread for sandboxed process " << processPath
                  << " (PID " << sandboxPid << "): " << e.what());
        childPid = 0;
        return false; // killAndReapGuard fires here.
    }

    if (monitorStarted == false) {
        // Monitor handoff failed (LI8): no thread is running to ever erase
        // this registry entry or call AwaitResult(), so this frame owns
        // both. killAndReapGuard's destructor Kill()s/awaits the sandboxee.
        eraseRegistryEntry();
        LOG_ERROR(<< "Failed to start monitor thread for sandboxed process " << processPath
                  << " (PID " << sandboxPid << ")");
        childPid = 0;
        return false; // killAndReapGuard fires here.
    }

    // E_Monitoring (LI2): registry insertion and monitor handoff both
    // succeeded, so the monitor thread now owns calling AwaitResult() and
    // removing the registry entry. Disarm - the guard must not also reap.
    killAndReapGuard.disarm();

    LOG_INFO(<< "Spawned sandboxed process " << processPath << " with PID " << childPid);

    return true;

#else // !SANDBOX2_AVAILABLE

    LOG_ERROR(<< "Cannot spawn " << processPath
              << ": ml-cpp was built without Sandbox2 support");
    return false;

#endif // SANDBOX2_AVAILABLE
}

#ifdef SANDBOX2_AVAILABLE

bool CSandboxedProcessSpawner::terminateChild(core::CProcess::TPid pid) {
    // Two mechanisms only, selected by the classification recorded on the
    // registry entry at *registration* time (never re-derived here by
    // re-calling pidfd_open, per the task brief): pidfd_send_signal(SIGTERM)
    // - a graceful termination *request* - for E_Acquired, or Sandbox2::Kill()
    // (SIGKILL via the owned monitor) for E_KernelUnsupported. No numeric
    // kill(pid) fallback exists anywhere in this file.
    int pidFdToSignal{-1};
    std::shared_ptr<sandbox2::Sandbox2> sandboxToKill;
    EChildLifecycleState previousState{EChildLifecycleState::E_Failed};
    {
        std::lock_guard<std::mutex> lock(m_PidRegistry->s_Mutex);
        const auto it = m_PidRegistry->s_Children.find(pid);
        if (it == m_PidRegistry->s_Children.end() ||
            it->second.s_State == EChildLifecycleState::E_Reaped ||
            it->second.s_State == EChildLifecycleState::E_Failed) {
            return false;
        }
        SSandboxedChild& child{it->second};
        previousState = child.s_State;
        switch (child.s_PidFdOutcome) {
        case EPidFdOutcome::E_Acquired:
            if (child.s_PidFd < 0) {
                // Logic error (should be structurally unreachable given
                // spawn()'s fail-closed registration in this task): a
                // registry entry classified E_Acquired must hold a real
                // pidfd. Do not silently no-op - log loudly and refuse.
                LOG_ERROR(<< "Logic error: sandboxed child PID " << pid
                          << " classified E_Acquired but holds no pidfd");
                return false;
            }
            pidFdToSignal = child.s_PidFd;
            break;
        case EPidFdOutcome::E_KernelUnsupported:
            if (!child.s_Sandbox) {
                // Same reasoning as above: E_KernelUnsupported without a
                // Sandbox2 handle to Kill() is a logic error, not a
                // silent no-op.
                LOG_ERROR(<< "Logic error: sandboxed child PID " << pid
                          << " classified E_KernelUnsupported but holds no Sandbox2 handle");
                return false;
            }
            sandboxToKill = child.s_Sandbox;
            break;
        case EPidFdOutcome::E_Failed:
        default:
            // Structurally unreachable: spawn() never registers an
            // E_Failed child (see the pidFdOutcome check above it). Assert
            // in debug builds and refuse rather than silently no-op if it
            // somehow happened anyway.
            LOG_ERROR(<< "Logic error: sandboxed child PID " << pid
                      << " registered with an undefined termination fallback (classification="
                      << static_cast<int>(child.s_PidFdOutcome) << ')');
            return false;
        }
        child.s_State = EChildLifecycleState::E_TerminationRequested;
    }

    // Rolls the registry entry's s_State back to what it was before this
    // call optimistically set it to E_TerminationRequested, but only if
    // nothing else has moved the state on in the meantime (e.g. a
    // concurrent reap). Called on the failure paths below, after the actual
    // pidfd_send_signal()/Kill() call - re-acquires the lock briefly; the
    // call itself still happens outside the lock, unchanged.
    const auto rollBackState = [this, pid, previousState]() {
        std::lock_guard<std::mutex> lock(m_PidRegistry->s_Mutex);
        const auto it = m_PidRegistry->s_Children.find(pid);
        if (it != m_PidRegistry->s_Children.end() &&
            it->second.s_State == EChildLifecycleState::E_TerminationRequested) {
            it->second.s_State = previousState;
        }
    };

    if (pidFdToSignal >= 0) {
        if (::syscall(ML_NR_pidfd_send_signal, pidFdToSignal, SIGTERM, nullptr, 0u) != 0) {
            LOG_ERROR(<< "pidfd_send_signal(SIGTERM) failed for sandboxed child PID " << pid
                      << ": " << ::strerror(errno));
            rollBackState();
            return false;
        }
        return true;
    }

    // sandboxToKill is only ever set on the E_KernelUnsupported branch
    // above; pidFdToSignal >= 0 is only ever set on the E_Acquired branch.
    // Exactly one of the two is populated by the switch, so reaching here
    // with neither would itself be a logic error - defensively refuse
    // rather than silently no-op.
    if (!sandboxToKill) {
        LOG_ERROR(<< "Logic error: terminateChild() for PID " << pid
                  << " resolved neither a pidfd nor a Sandbox2 handle to kill");
        rollBackState();
        return false;
    }

    try {
        // Locked design decision (design.md): MonitorBase::Kill() takes no
        // signal parameter and hard-codes SIGKILL - this is the ENOSYS
        // forced-kill fallback, never a SIGTERM-via-monitor path.
        sandboxToKill->Kill();
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Sandbox2::Kill() failed for sandboxed child PID " << pid << ": "
                  << e.what());
        rollBackState();
        return false;
    }
    return true;
}

#else // !SANDBOX2_AVAILABLE

bool CSandboxedProcessSpawner::terminateChild(core::CProcess::TPid /* pid */) {
    return false;
}

#endif // SANDBOX2_AVAILABLE

bool CSandboxedProcessSpawner::hasChild(core::CProcess::TPid pid) const {
    std::lock_guard<std::mutex> lock(m_PidRegistry->s_Mutex);
    const auto it = m_PidRegistry->s_Children.find(pid);
    return it != m_PidRegistry->s_Children.end() &&
           it->second.s_State != EChildLifecycleState::E_Reaped &&
           it->second.s_State != EChildLifecycleState::E_Failed;
}

} // namespace sandbox
} // namespace ml
