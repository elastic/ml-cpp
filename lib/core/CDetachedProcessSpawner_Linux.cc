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
#include <core/CDetachedProcessSpawner.h>

#include <core/CCondition.h>
#include <core/CLogger.h>
#include <core/CMutex.h>
#include <core/CScopedLock.h>
#include <core/CThread.h>

#include <algorithm>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <linux/futex.h>
#include <signal.h>
#include <spawn.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

#ifdef SANDBOX2_AVAILABLE
#include <absl/time/time.h>

#include <sandboxed_api/sandbox2/executor.h>
#include <sandboxed_api/sandbox2/policybuilder.h>
#include <sandboxed_api/sandbox2/result.h>
#include <sandboxed_api/sandbox2/sandbox2.h>
#include <sys/syscall.h>

// The CentOS 7 based CI build image has kernel headers that predate clone3, so
// __NR_clone3 may be undefined at build time even though the runtime glibc uses
// clone3. clone3 is syscall 435 on every architecture we build for (x86_64 and
// aarch64), so fall back to that literal to keep the sandbox policy independent
// of the build image's header version.
#ifdef __NR_clone3
#define ML_NR_clone3 __NR_clone3
#else
#define ML_NR_clone3 435
#endif
#endif

namespace {

const int MAX_NEW_OPEN_FILES{10};

//! Minimal RAII scope guard that runs a callable when it goes out of scope.
//! Used to guarantee the controller's global TMPDIR is restored on every exit
//! path of the sandboxed spawn.
template<typename FUNC>
class CScopeExit {
public:
    explicit CScopeExit(FUNC func) : m_Func(std::move(func)) {}
    ~CScopeExit() { m_Func(); }
    CScopeExit(const CScopeExit&) = delete;
    CScopeExit& operator=(const CScopeExit&) = delete;

private:
    FUNC m_Func;
};

template<typename FUNC>
CScopeExit<FUNC> makeScopeExit(FUNC func) {
    return CScopeExit<FUNC>{std::move(func)};
}

//! Attempt to close all file descriptors except the standard ones. The
//! standard file descriptors will be reopened on /dev/null in the spawned
//! process. Returns false if the actions cannot be initialised.
bool setupFileActions(posix_spawn_file_actions_t* fileActions, int& maxFdHint) {
    if (::posix_spawn_file_actions_init(fileActions) != 0) {
        return false;
    }

    struct rlimit rlim;
    ::memset(&rlim, 0, sizeof(struct rlimit));
    if (::getrlimit(RLIMIT_NOFILE, &rlim) != 0) {
        rlim.rlim_cur = 36;
    }

    int maxFdToTest{std::min(static_cast<int>(rlim.rlim_cur), maxFdHint + MAX_NEW_OPEN_FILES)};
    for (int fd = 0; fd <= maxFdToTest; ++fd) {
        if (fd == STDIN_FILENO) {
            ::posix_spawn_file_actions_addopen(fileActions, fd, "/dev/null", O_RDONLY, S_IRUSR);
            maxFdHint = fd;
        } else if (fd == STDOUT_FILENO || fd == STDERR_FILENO) {
            ::posix_spawn_file_actions_addopen(fileActions, fd, "/dev/null", O_WRONLY, S_IWUSR);
            maxFdHint = fd;
        } else {
            if (::fcntl(fd, F_GETFL) != -1) {
                ::posix_spawn_file_actions_addclose(fileActions, fd);
                maxFdHint = fd;
            }
        }
    }

    return true;
}
}

namespace ml {
namespace core {

namespace detail {

class CTrackerThread : public CThread {
public:
    using TPidSet = std::set<CProcess::TPid>;

    CTrackerThread() : m_Shutdown(false), m_Condition(m_Mutex) {}

    CMutex& mutex() { return m_Mutex; }

    void addPid(CProcess::TPid pid) {
        CScopedLock lock(m_Mutex);
        m_Pids.insert(pid);
        m_Condition.signal();
    }

    void addSandboxPid(CProcess::TPid pid) {
        CScopedLock lock(m_Mutex);
        m_SandboxPids.insert(pid);
        m_Condition.signal();
    }

    void removePid(CProcess::TPid pid) {
        CScopedLock lock(m_Mutex);
        m_Pids.erase(pid);
        m_SandboxPids.erase(pid);
    }

    bool terminatePid(CProcess::TPid pid) {
        if (!this->havePid(pid)) {
            LOG_ERROR(<< "Will not attempt to kill process " << pid << ": not a child process");
            return false;
        }

        if (::kill(pid, SIGTERM) == -1) {
            if (errno != ESRCH) {
                LOG_ERROR(<< "Failed to kill process " << pid << ": " << ::strerror(errno));
            }
            return false;
        }

        return true;
    }

    bool havePid(CProcess::TPid pid) const {
        if (pid <= 0) {
            return false;
        }

        CScopedLock lock(m_Mutex);
        const_cast<CTrackerThread*>(this)->checkForDeadChildren();
        return m_Pids.find(pid) != m_Pids.end() ||
               m_SandboxPids.find(pid) != m_SandboxPids.end();
    }

protected:
    void run() override {
        CScopedLock lock(m_Mutex);

        while (!m_Shutdown) {
            if (m_Pids.empty()) {
                m_Condition.wait();
            } else {
                m_Condition.wait(50);
            }

            this->checkForDeadChildren();
        }
    }

    void shutdown() override {
        CScopedLock lock(m_Mutex);
        m_Shutdown = true;
        m_Condition.signal();
    }

private:
    void checkForDeadChildren() {
        TPidSet pidsCopy{m_Pids};
        for (CProcess::TPid pid : pidsCopy) {
            int status = 0;
            CProcess::TPid waited = ::waitpid(pid, &status, WNOHANG);
            if (waited == 0) {
                continue;
            }
            if (waited == -1) {
                if (errno == EINTR) {
                    continue;
                }
                if (errno == ECHILD) {
                    m_Pids.erase(pid);
                }
                continue;
            }
            if (WIFSIGNALED(status)) {
                int signal = WTERMSIG(status);
                if (signal == SIGTERM) {
                    LOG_INFO(<< "Child process with PID " << pid
                             << " was terminated by signal " << signal);
                } else if (signal == SIGKILL) {
                    LOG_ERROR(<< "Child process with PID " << pid << " was terminated by signal 9 (SIGKILL)."
                              << " This is likely due to the OOM killer.");
                } else {
                    LOG_ERROR(<< "Child process with PID " << pid
                              << " was terminated by signal " << signal);
                }
            } else {
                int exitCode = WEXITSTATUS(status);
                if (exitCode == 0) {
                    LOG_DEBUG(<< "Child process with PID " << pid << " has exited");
                } else {
                    LOG_WARN(<< "Child process with PID " << pid
                             << " has exited with exit code " << exitCode);
                }
            }
            m_Pids.erase(pid);
        }
    }

private:
    bool m_Shutdown;
    TPidSet m_Pids;
    TPidSet m_SandboxPids;
    mutable CMutex m_Mutex;
    CCondition m_Condition;
};
}

CDetachedProcessSpawner::CDetachedProcessSpawner(const TStrVec& permittedProcessPaths)
    : m_PermittedProcessPaths(permittedProcessPaths),
      m_TrackerThread(std::make_shared<detail::CTrackerThread>()) {
    if (m_TrackerThread->start() == false) {
        LOG_ERROR(<< "Failed to start spawned process tracker thread");
    }
}

CDetachedProcessSpawner::~CDetachedProcessSpawner() {
    if (m_TrackerThread->stop() == false) {
        LOG_ERROR(<< "Failed to stop spawned process tracker thread");
    }
}

bool CDetachedProcessSpawner::spawn(const std::string& processPath, const TStrVec& args) {
    CProcess::TPid dummy(0);
    return this->spawn(processPath, args, dummy);
}

bool CDetachedProcessSpawner::spawn(const std::string& processPath,
                                    const TStrVec& args,
                                    CProcess::TPid& childPid) {
    // Authorization gate: only exact permitted paths may be spawned, whether or
    // not they are routed through Sandbox2. Must run before any dispatch so a
    // path merely containing "pytorch_inference" cannot bypass the allowlist.
    if (std::find(m_PermittedProcessPaths.begin(), m_PermittedProcessPaths.end(),
                  processPath) == m_PermittedProcessPaths.end()) {
        LOG_ERROR(<< "Spawning process '" << processPath << "' is not permitted");
        return false;
    }

#ifdef __linux__
    // Use Sandbox2 for pytorch_inference to provide security isolation
    if (processPath.find("pytorch_inference") != std::string::npos) {
#ifdef SANDBOX2_AVAILABLE
        // Save original TMPDIR to restore for the sandboxed process
        std::string originalTmpdir;
        const char* tmpdir = ::getenv("TMPDIR");
        if (tmpdir != nullptr) {
            originalTmpdir = tmpdir;
        }

        // Sandbox2 forkserver uses Unix sockets with 108 char path limit
        bool tmpdirOverridden{false};
        if (tmpdir != nullptr && ::strlen(tmpdir) > 80) {
            LOG_WARN(<< "TMPDIR path too long, temporarily overriding to /tmp for forkserver");
            ::setenv("TMPDIR", "/tmp", 1);
            tmpdirOverridden = true;
        }

        // Restore the controller's global TMPDIR on every exit path. The
        // override above must remain in effect until the Sandbox2 forkserver has
        // started (it derives its Unix socket path from TMPDIR), but leaving the
        // controller's environment mutated corrupts subsequent spawns: they would
        // observe the short /tmp value, skip the per-process TMPDIR restoration
        // below, and launch pytorch_inference with the wrong TMPDIR.
        auto tmpdirRestorer = makeScopeExit([tmpdirOverridden, &originalTmpdir]() {
            if (tmpdirOverridden) {
                ::setenv("TMPDIR", originalTmpdir.c_str(), 1);
            }
        });

        // Resolve to absolute path - Sandbox2 requires absolute paths
        char resolvedPath[PATH_MAX];
        if (::realpath(processPath.c_str(), resolvedPath) == nullptr) {
            LOG_ERROR(<< "Cannot resolve path " << processPath << ": " << ::strerror(errno));
            return false;
        }
        std::string absPath(resolvedPath);

        // Verify binary exists and is accessible
        struct stat binaryStat;
        if (::stat(absPath.c_str(), &binaryStat) != 0) {
            LOG_ERROR(<< "Cannot stat " << absPath << ": " << ::strerror(errno));
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
        std::set<std::string> argDirs;
        for (const auto& arg : args) {
            size_t eqPos = arg.find('=');
            if (eqPos != std::string::npos && eqPos + 1 < arg.size() &&
                arg[eqPos + 1] == '/') {
                std::string path = arg.substr(eqPos + 1);
                size_t lastSlash = path.rfind('/');
                if (lastSlash != std::string::npos && lastSlash > 0) {
                    std::string dir = path.substr(0, lastSlash);
                    char resolved[PATH_MAX];
                    std::string canonical =
                        ::realpath(dir.c_str(), resolved) != nullptr ? resolved : dir;
                    // Bind-mount the canonical directory, and also the literal
                    // path the sandboxee actually passes to mkfifo()/open() if it
                    // differs (e.g. a symlinked path component). If only the
                    // canonical path is mounted, a mkfifo() against the literal
                    // path inside the mount namespace can land in the sandbox's
                    // throw-away rootfs instead of the host directory that
                    // Elasticsearch is watching, so the FIFO never becomes visible
                    // and the connection times out.
                    argDirs.insert(canonical);
                    struct stat dirStat;
                    if (dir != canonical && ::stat(dir.c_str(), &dirStat) == 0) {
                        argDirs.insert(dir);
                    }
                }
            }
        }

        // Build sandbox policy
        sandbox2::PolicyBuilder policyBuilder;
        policyBuilder.AllowDynamicStartup()
            .AllowOpen()
            .AllowRead()
            .AllowWrite()
            .AllowExit()
            .AllowStat()
            .AllowGetPIDs()
            .AllowGetRandom()
            .AllowHandleSignals()
            .AllowTcMalloc()
            .AllowMmap()
            // glibc/libtorch use futex for mutexes and condition variables. Only
            // FUTEX_WAIT and FUTEX_WAKE are insufficient under sustained concurrent
            // load: timed waits use FUTEX_WAIT_BITSET and some broadcast/requeue
            // paths use FUTEX_CMP_REQUEUE/FUTEX_WAKE_OP. Denying those ops is a
            // Sandbox2 policy VIOLATION (SIGSYS), which kills pytorch_inference and
            // surfaces as "Unexpected end of file" in Elasticsearch. PI futex ops
            // are deliberately excluded — libtorch uses ordinary mutexes only.
            .AllowFutexOp(FUTEX_WAIT)
            .AllowFutexOp(FUTEX_WAKE)
            .AllowFutexOp(FUTEX_WAIT_BITSET)
            .AllowFutexOp(FUTEX_WAKE_BITSET)
            .AllowFutexOp(FUTEX_REQUEUE)
            .AllowFutexOp(FUTEX_CMP_REQUEUE)
            .AllowFutexOp(FUTEX_WAKE_OP)
            // Threading and scheduling
            .AllowSyscall(__NR_sched_yield)
            .AllowSyscall(__NR_sched_getaffinity)
            .AllowSyscall(__NR_sched_setaffinity)
            .AllowSyscall(__NR_sched_getparam)
            .AllowSyscall(__NR_sched_getscheduler)
            .AllowSyscall(__NR_clone)
            // clone3 (syscall 435 on both x86_64 and aarch64) must be allowed by
            // number rather than via __NR_clone3: the CentOS 7 based CI build
            // image has kernel headers that predate clone3 and therefore leave
            // __NR_clone3 undefined, yet the newer glibc on the CI runtime uses
            // clone3 for thread creation. Without this, pytorch_inference is
            // killed by a seccomp violation the moment libtorch spawns a thread,
            // long before it can create its log FIFO.
            .AllowSyscall(ML_NR_clone3)
            .AllowSyscall(__NR_set_tid_address)
            .AllowSyscall(__NR_set_robust_list)
#ifdef __NR_rseq
            .AllowSyscall(__NR_rseq)
#endif
            // Time operations
            .AllowSyscall(__NR_clock_gettime)
            .AllowSyscall(__NR_clock_getres)
            .AllowSyscall(__NR_clock_nanosleep)
            .AllowSyscall(__NR_gettimeofday)
            .AllowSyscall(__NR_nanosleep)
            .AllowSyscall(__NR_times)
            // I/O multiplexing
            .AllowSyscall(__NR_epoll_create1)
            .AllowSyscall(__NR_epoll_ctl)
            .AllowSyscall(__NR_epoll_pwait)
            .AllowSyscall(__NR_eventfd2)
            .AllowSyscall(__NR_ppoll)
            .AllowSyscall(__NR_pselect6)
            // File operations
            .AllowSyscall(__NR_ioctl)
            .AllowSyscall(__NR_fcntl)
            .AllowSyscall(__NR_pipe2)
            .AllowSyscall(__NR_dup)
            .AllowSyscall(__NR_dup3)
            .AllowSyscall(__NR_lseek)
            .AllowSyscall(__NR_ftruncate)
            .AllowSyscall(__NR_readlinkat)
            .AllowSyscall(__NR_faccessat)
            .AllowSyscall(__NR_getdents64)
            .AllowSyscall(__NR_getcwd)
            .AllowSyscall(__NR_unlinkat)
            .AllowSyscall(__NR_renameat)
            .AllowSyscall(__NR_mkdirat)
            .AllowSyscall(__NR_mknodat)
        // On some architectures (notably x86_64) glibc's file-system
        // wrappers issue the legacy syscalls rather than their *at
        // equivalents, e.g. mkfifo()->mknod, remove()/unlink()->unlink,
        // mkdir()->mkdir. pytorch_inference creates and tears down its
        // named pipes via these wrappers, so the legacy syscalls must be
        // permitted too or the process is killed with SIGSYS the moment it
        // touches a pipe. These syscalls do not exist on aarch64 (which is
        // *at-only), hence the guards. They are exact equivalents of the
        // *at syscalls already permitted above, so allowing them does not
        // widen the policy.
#ifdef __NR_mknod
            .AllowSyscall(__NR_mknod)
#endif
#ifdef __NR_unlink
            .AllowSyscall(__NR_unlink)
#endif
#ifdef __NR_rmdir
            .AllowSyscall(__NR_rmdir)
#endif
#ifdef __NR_mkdir
            .AllowSyscall(__NR_mkdir)
#endif
#ifdef __NR_rename
            .AllowSyscall(__NR_rename)
#endif
#ifdef __NR_readlink
            .AllowSyscall(__NR_readlink)
#endif
#ifdef __NR_access
            .AllowSyscall(__NR_access)
#endif
#ifdef __NR_dup2
            .AllowSyscall(__NR_dup2)
#endif
            // Memory management
            .AllowSyscall(__NR_mprotect)
            .AllowSyscall(__NR_mremap)
            .AllowSyscall(__NR_madvise)
            .AllowSyscall(__NR_munmap)
            .AllowSyscall(__NR_brk)
            // System info
            .AllowSyscall(__NR_sysinfo)
            .AllowSyscall(__NR_uname)
            .AllowSyscall(__NR_prlimit64)
            .AllowSyscall(__NR_getrusage)
            // Process control
            .AllowSyscall(__NR_prctl)
#ifdef __NR_arch_prctl
            .AllowSyscall(__NR_arch_prctl)
#endif
            .AllowSyscall(__NR_wait4)
            .AllowSyscall(__NR_exit)
            // User/group IDs
            .AllowSyscall(__NR_getuid)
            .AllowSyscall(__NR_getgid)
            .AllowSyscall(__NR_geteuid)
            .AllowSyscall(__NR_getegid)
            // Process priority: pytorch_inference lowers its own nice value.
            .AllowSyscall(__NR_setpriority)
            .AllowSyscall(__NR_getpriority)
            // Crash handler uses tgkill to re-raise fatal signals.
            .AllowSyscall(__NR_tgkill)
            // Misc runtime syscalls exercised by pytorch_inference / libtorch.
            // These mirror the legacy CSystemCallFilter allowlist that ran the
            // same binary successfully.
            .AllowSyscall(__NR_statfs)
            .AllowSyscall(__NR_connect)
#ifdef __NR_time
            .AllowSyscall(__NR_time)
#endif
#ifdef __NR_getdents
            .AllowSyscall(__NR_getdents)
#endif
            // Filesystem mounts
            .AddDirectory(binDir, /*is_ro=*/true)
            .AddDirectory(libDir, /*is_ro=*/true)
            .AddDirectory("/lib", /*is_ro=*/true)
            .AddDirectory("/lib64", /*is_ro=*/true)
            .AddDirectory("/usr/lib", /*is_ro=*/true)
            .AddDirectory("/usr/lib64", /*is_ro=*/true)
            .AddDirectory("/etc", /*is_ro=*/true)
            .AddDirectory("/proc", /*is_ro=*/true)
            .AddDirectory("/sys", /*is_ro=*/true)
            .AddFile("/dev/null", /*is_ro=*/false)
            .AddFile("/dev/urandom", /*is_ro=*/true)
            .AddFile("/dev/random", /*is_ro=*/true)
            .AddDirectory("/tmp", /*is_ro=*/false);

        // Add directories from command-line arguments (pipe paths)
        for (const auto& dir : argDirs) {
            policyBuilder.AddDirectory(dir, /*is_ro=*/false);
        }

        auto policy_result = policyBuilder.TryBuild();
        if (!policy_result.ok()) {
            LOG_ERROR(<< "Failed to build Sandbox2 policy: " << policy_result.status());
            return false;
        }

        // Create executor with a sandbox marker and restored TMPDIR when needed.
        std::vector<std::string> customEnv;
        bool sandboxMarkerSet{false};
        const char* currentTmpdir = ::getenv("TMPDIR");
        const bool restoreTmpdir = !originalTmpdir.empty() &&
                                   (currentTmpdir == nullptr || originalTmpdir != currentTmpdir);
        for (char** env = environ; *env != nullptr; ++env) {
            std::string envVar(*env);
            if (restoreTmpdir && envVar.find("TMPDIR=") == 0) {
                customEnv.push_back("TMPDIR=" + originalTmpdir);
            } else if (envVar.find("ML_SANDBOXED=") == 0) {
                customEnv.push_back("ML_SANDBOXED=1");
                sandboxMarkerSet = true;
            } else {
                customEnv.push_back(envVar);
            }
        }
        if (!sandboxMarkerSet) {
            customEnv.push_back("ML_SANDBOXED=1");
        }
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

        auto sandboxPtr = std::make_unique<sandbox2::Sandbox2>(
            std::move(executor), std::move(*policy_result));

        if (!sandboxPtr->RunAsync()) {
            LOG_ERROR(<< "Sandbox2 failed to start pytorch_inference");
            return false;
        }

        childPid = sandboxPtr->pid();
        if (childPid <= 0) {
            LOG_ERROR(<< "Sandbox2 returned invalid PID");
            sandbox2::Result result = sandboxPtr->AwaitResult();
            LOG_ERROR(<< "Sandbox2 Result: " << result.ToString());
            return false;
        }

        LOG_INFO(<< "Spawned sandboxed pytorch_inference with PID " << childPid);

        m_TrackerThread->addSandboxPid(childPid);

        // The sandboxee is a child of the Sandbox2 forkserver rather than of the
        // controller, so the tracker's waitpid() never sees it. Own the sandbox
        // instance on a dedicated thread that keeps it alive for the lifetime of
        // pytorch_inference, waits for its result, logs termination the same way
        // checkForDeadChildren() does for regular children, and removes the PID
        // from the tracker so PID reuse cannot make terminateChild() signal an
        // unrelated process.
        {
            CProcess::TPid sandboxPid{childPid};
            std::thread([
                sandboxPid, tracker = m_TrackerThread, sbx = std::move(sandboxPtr)
            ]() mutable {
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
                default:
                    LOG_ERROR(<< "Sandboxed pytorch_inference (PID " << sandboxPid
                              << ") terminated abnormally: " << result.ToString() << " [status="
                              << sandbox2::Result::StatusEnumToString(result.final_status())
                              << ", reason_code=" << result.reason_code() << "]");
                    break;
                }
                tracker->removePid(sandboxPid);
            })
                .detach();
        }

        return true;
#else
        LOG_ERROR(<< "Sandbox2 not available - cannot spawn pytorch_inference securely");
        return false;
#endif
    }
#endif

    // Standard spawn for other processes (not pytorch_inference)
    if (::access(processPath.c_str(), X_OK) != 0) {
        LOG_ERROR(<< "Cannot execute '" << processPath << "': " << ::strerror(errno));
        return false;
    }

    using TCharPVec = std::vector<char*>;
    TCharPVec argv;
    argv.reserve(args.size() + 2);

    argv.push_back(const_cast<char*>(processPath.c_str()));
    for (size_t index = 0; index < args.size(); ++index) {
        argv.push_back(const_cast<char*>(args[index].c_str()));
    }
    argv.push_back(static_cast<char*>(nullptr));

    posix_spawn_file_actions_t fileActions;
    if (setupFileActions(&fileActions, m_MaxObservedFd) == false) {
        LOG_ERROR(<< "Failed to set up file actions: " << ::strerror(errno));
        return false;
    }
    posix_spawnattr_t spawnAttributes;
    if (::posix_spawnattr_init(&spawnAttributes) != 0) {
        LOG_ERROR(<< "Failed to set up spawn attributes: " << ::strerror(errno));
        return false;
    }
    ::posix_spawnattr_setflags(&spawnAttributes, POSIX_SPAWN_SETPGROUP);

    {
        CScopedLock lock(m_TrackerThread->mutex());

        int err(::posix_spawn(&childPid, processPath.c_str(), &fileActions,
                              &spawnAttributes, argv.data(), environ));

        ::posix_spawn_file_actions_destroy(&fileActions);
        ::posix_spawnattr_destroy(&spawnAttributes);

        if (err != 0) {
            LOG_ERROR(<< "Failed to spawn '" << processPath << "': " << ::strerror(err));
            return false;
        }

        m_TrackerThread->addPid(childPid);
    }

    LOG_DEBUG(<< "Spawned '" << processPath << "' with PID " << childPid);

    return true;
}

bool CDetachedProcessSpawner::terminateChild(CProcess::TPid pid) {
    return m_TrackerThread->terminatePid(pid);
}

bool CDetachedProcessSpawner::hasChild(CProcess::TPid pid) const {
    return m_TrackerThread->havePid(pid);
}

} // namespace core
} // namespace ml
