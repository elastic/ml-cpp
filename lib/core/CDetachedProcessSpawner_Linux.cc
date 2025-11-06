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

#include <errno.h>
#include <fcntl.h>
#include <grp.h>
#include <pwd.h>
#include <signal.h>
#include <spawn.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

// environ is a global variable from the C runtime library
extern char** environ;

// Sandbox2 integration - use conditional compilation
#ifdef SANDBOX2_AVAILABLE
#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <sandboxed_api/sandbox2/policy.h>
#include <sandboxed_api/sandbox2/policybuilder.h>
#include <sandboxed_api/sandbox2/result.h>
#include <sandboxed_api/sandbox2/sandbox2.h>
#include <sandboxed_api/sandbox2/util/bpf_helper.h>

// Define syscall numbers for x86_64
#ifndef __NR_mount
#define __NR_mount 165
#endif
#ifndef __NR_umount
#define __NR_umount 166
#endif
#ifndef __NR_umount2
#define __NR_umount2 166
#endif
#ifndef __NR_connect
#define __NR_connect 42
#endif
#ifdef __x86_64__
#ifndef __NR_mkdir
#define __NR_mkdir 83
#endif
#ifndef __NR_rmdir
#define __NR_rmdir 84
#endif
#ifndef __NR_unlink
#define __NR_unlink 87
#endif
#ifndef __NR_mknod
#define __NR_mknod 133
#endif
#ifndef __NR_getdents
#define __NR_getdents 78
#endif
#endif // __x86_64__
#ifndef __NR_mkdirat
#define __NR_mkdirat 258
#endif
#ifndef __NR_unlinkat
#define __NR_unlinkat 263
#endif
#ifndef __NR_mknodat
#define __NR_mknodat 259
#endif
#ifndef __NR_getdents64
#define __NR_getdents64 217
#endif
#endif // SANDBOX2_AVAILABLE

namespace {

//! Maximum number of newly opened files between calls to setupFileActions().
const int MAX_NEW_OPEN_FILES{10};

//! Attempt to close all file descriptors except the standard ones.  The
//! standard file descriptors will be reopened on /dev/null in the spawned
//! process.  Returns false and sets errno if the actions cannot be initialised
//! at all, but other errors are ignored.
bool setupFileActions(posix_spawn_file_actions_t* fileActions, int& maxFdHint) {
    if (::posix_spawn_file_actions_init(fileActions) != 0) {
        return false;
    }

    struct rlimit rlim;
    ::memset(&rlim, 0, sizeof(struct rlimit));
    if (::getrlimit(RLIMIT_NOFILE, &rlim) != 0) {
        rlim.rlim_cur = 36; // POSIX default
    }

    // Assume only a handful of new files have been opened since the last time
    // this function was called. Doing this means we learn the practical limit
    // on the number of open files, which will be a lot less than the enforced
    // limit, and avoids making masses of expensive fcntl() calls.
    int maxFdToTest{std::min(static_cast<int>(rlim.rlim_cur), maxFdHint + MAX_NEW_OPEN_FILES)};
    for (int fd = 0; fd <= maxFdToTest; ++fd) {
        if (fd == STDIN_FILENO) {
            ::posix_spawn_file_actions_addopen(fileActions, fd, "/dev/null", O_RDONLY, S_IRUSR);
            maxFdHint = fd;
        } else if (fd == STDOUT_FILENO || fd == STDERR_FILENO) {
            ::posix_spawn_file_actions_addopen(fileActions, fd, "/dev/null", O_WRONLY, S_IWUSR);
            maxFdHint = fd;
        } else {
            // Close other files that are open.  There is a race condition here,
            // in that files could be opened or closed between this code running
            // and the posix_spawn() function being called.  However, this would
            // violate the restrictions stated in the contract detailed in the
            // Doxygen description of this class.
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

// Sandbox2 helper functions and structures
#ifdef SANDBOX2_AVAILABLE

//! Structure to hold process paths for Sandbox2 policy
struct ProcessPaths {
    std::string pytorchLibDir;
    std::string modelPath;
    std::string inputPipe;
    std::string outputPipe;
    std::string logPipe;
    std::string logProperties;
};

//! Parse command line arguments to extract file paths
ProcessPaths parseProcessPaths(const std::vector<std::string>& args) {
    ProcessPaths paths;
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg.find("--input=") == 0) {
            paths.inputPipe = arg.substr(8);
        } else if (arg.find("--output=") == 0) {
            paths.outputPipe = arg.substr(9);
        } else if (arg.find("--restore=") == 0) {
            paths.modelPath = arg.substr(10);
        } else if (arg.find("--logPipe=") == 0) {
            paths.logPipe = arg.substr(10);
        } else if (arg.find("--logProperties=") == 0) {
            paths.logProperties = arg.substr(16);
        }
    }
    return paths;
}

//! Calculate PyTorch library directory from executable path
std::string calculatePytorchLibDir(const std::string& processPath) {
    size_t lastSlash = processPath.find_last_of('/');
    if (lastSlash == std::string::npos) return "";
    
    std::string exeDir = processPath.substr(0, lastSlash);
    size_t lastDirSlash = exeDir.find_last_of('/');
    if (lastDirSlash == std::string::npos) return "";
    
    return exeDir.substr(0, lastDirSlash) + "/lib";
}

//! Look up UID/GID for nobody user
bool lookupNobodyUser(uid_t& uid, gid_t& gid) {
    struct passwd* pwd = getpwnam("nobody");
    if (!pwd) {
        LOG_ERROR(<< "Failed to lookup nobody user");
        return false;
    }
    uid = pwd->pw_uid;

    struct group* grp = getgrnam("nogroup");
    if (!grp) {
        LOG_ERROR(<< "Failed to lookup nogroup");
        return false;
    }
    gid = grp->gr_gid;
    return true;
}

//! Build Sandbox2 policy for pytorch_inference
std::unique_ptr<sandbox2::Policy> buildSandboxPolicy(const ProcessPaths& paths) {
    auto builder = sandbox2::PolicyBuilder()
                       .AddDirectoryAt("/lib", "/lib", true)
                       .AddDirectoryAt("/usr/lib", "/usr/lib", true)
                       .AddDirectoryAt("/lib64", "/lib64", true)
                       .AddDirectoryAt("/usr/lib64", "/usr/lib64", true)
                       .AddTmpfs("/tmp", 64 * 1024 * 1024);

    // Block dangerous syscalls
    builder.BlockSyscallWithErrno(__NR_mount, EPERM)
        .BlockSyscallWithErrno(__NR_umount, EPERM)
        .BlockSyscallWithErrno(__NR_umount2, EPERM)
        .BlockSyscallWithErrno(__NR_connect, EPERM);

#ifdef __x86_64__
    builder.BlockSyscallWithErrno(__NR_mkdir, EPERM)
        .BlockSyscallWithErrno(__NR_rmdir, EPERM)
        .BlockSyscallWithErrno(__NR_unlink, EPERM)
        .BlockSyscallWithErrno(__NR_mknod, EPERM)
        .BlockSyscallWithErrno(__NR_getdents, EPERM);
#endif
    builder.BlockSyscallWithErrno(__NR_mkdirat, EPERM)
        .BlockSyscallWithErrno(__NR_unlinkat, EPERM)
        .BlockSyscallWithErrno(__NR_mknodat, EPERM)
        .BlockSyscallWithErrno(__NR_getdents64, EPERM);

    // Allow PyTorch libraries
    if (!paths.pytorchLibDir.empty()) {
        builder.AddDirectoryAt(paths.pytorchLibDir, paths.pytorchLibDir, true);
    }

    // Allow model file and pipes
    if (!paths.modelPath.empty()) {
        builder.AddFileAt(paths.modelPath, paths.modelPath, true);
    }
    if (!paths.inputPipe.empty()) {
        builder.AddFileAt(paths.inputPipe, paths.inputPipe, false);
    }
    if (!paths.outputPipe.empty()) {
        builder.AddFileAt(paths.outputPipe, paths.outputPipe, false);
    }
    if (!paths.logPipe.empty()) {
        builder.AddFileAt(paths.logPipe, paths.logPipe, false);
    }
    if (!paths.logProperties.empty()) {
        builder.AddFileAt(paths.logProperties, paths.logProperties, true);
    }

    return builder.BuildOrDie();
}

#endif // SANDBOX2_AVAILABLE



class CTrackerThread : public CThread {
public:
    using TPidSet = std::set<CProcess::TPid>;

public:
    CTrackerThread() : m_Shutdown(false), m_Condition(m_Mutex) {}

    //! Mutex is accessible so the code outside the class can avoid race
    //! conditions.
    CMutex& mutex() { return m_Mutex; }

    //! Add a PID to track.
    void addPid(CProcess::TPid pid) {
        CScopedLock lock(m_Mutex);
        m_Pids.insert(pid);
        m_Condition.signal();
    }

    bool terminatePid(CProcess::TPid pid) {
        if (!this->havePid(pid)) {
            LOG_ERROR(<< "Will not attempt to kill process " << pid << ": not a child process");
            return false;
        }

        if (::kill(pid, SIGTERM) == -1) {
            // Don't log an error if the process exited normally in between
            // checking whether it was our child process and killing it
            if (errno != ESRCH) {
                LOG_ERROR(<< "Failed to kill process " << pid << ": " << ::strerror(errno));
            } else {
                // But log at debug in case there's a bug in this area
                LOG_DEBUG(<< "No such process while trying to kill PID " << pid);
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
        // Do an extra cycle of waiting for zombies, so we give the most
        // up-to-date answer possible
        const_cast<CTrackerThread*>(this)->checkForDeadChildren();
        return m_Pids.find(pid) != m_Pids.end();
    }

protected:
    void run() override {
        CScopedLock lock(m_Mutex);

        while (!m_Shutdown) {
            // Reap zombies every 50ms if child processes are running,
            // otherwise wait for a child process to start.
            if (m_Pids.empty()) {
                m_Condition.wait();
            } else {
                m_Condition.wait(50);
            }

            this->checkForDeadChildren();
        }
    }

    void shutdown() override {
        LOG_DEBUG(<< "Shutting down spawned process tracker thread");
        CScopedLock lock(m_Mutex);
        m_Shutdown = true;
        m_Condition.signal();
    }

private:
    //! Reap zombie child processes and adjust the set of live child PIDs
    //! accordingly.  MUST be called with m_Mutex locked.
    void checkForDeadChildren() {
        int status = 0;
        for (;;) {
            CProcess::TPid pid = ::waitpid(-1, &status, WNOHANG);
            // 0 means there are child processes but none have died
            if (pid == 0) {
                break;
            }
            // -1 means error
            if (pid == -1) {
                if (errno != EINTR) {
                    break;
                }
            } else {
                if (WIFSIGNALED(status)) {
                    int signal = WTERMSIG(status);
                    if (signal == SIGTERM) {
                        // We expect this when a job is force-closed, so log
                        // at a lower level
                        LOG_INFO(<< "Child process with PID " << pid
                                 << " was terminated by signal " << signal);
                    } else if (signal == SIGKILL) {
                        // This should never happen if the system is working
                        // normally - possible reasons are the Linux OOM
                        // killer or manual intervention. The latter is highly unlikely
                        // if running in the cloud.
                        LOG_ERROR(<< "Child process with PID " << pid << " was terminated by signal 9 (SIGKILL)."
                                  << " This is likely due to the OOM killer."
                                  << " Please check system logs for more details.");
                    } else {
                        // This should never happen if the system is working
                        // normally - possible reasons are bugs that cause
                        // access violations or manual intervention. The latter is highly unlikely
                        // if running in the cloud.
                        LOG_ERROR(<< "Child process with PID " << pid
                                  << " was terminated by signal " << signal
                                  << " Please check system logs for more details.");
                    }
                } else {
                    int exitCode = WEXITSTATUS(status);
                    if (exitCode == 0) {
                        // This is the happy case
                        LOG_DEBUG(<< "Child process with PID " << pid << " has exited");
                    } else {
                        LOG_WARN(<< "Child process with PID " << pid
                                 << " has exited with exit code " << exitCode);
                    }
                }
                m_Pids.erase(pid);
            }
        }
    }

private:
    bool m_Shutdown;
    TPidSet m_Pids;
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
#ifdef __linux__
    // Check if this is pytorch_inference and use Sandbox2 if available
    // This is done here to avoid having to override spawn() in the Linux file
    // and deal with CTrackerThread access issues
    if (processPath.find("pytorch_inference") != std::string::npos) {
#ifndef SANDBOX2_DISABLED
#ifdef SANDBOX2_AVAILABLE
        // Forward declaration of Linux-specific Sandbox2 spawn function
        // Function is in ml::core namespace
        bool ml_core_spawnWithSandbox2Linux(const std::string& processPath,
                                            const std::vector<std::string>& args,
                                            CProcess::TPid& childPid);
        if (ml_core_spawnWithSandbox2Linux(processPath, args, childPid)) {
            // Add PID to tracker - we can access m_TrackerThread since we're in the same class
            // and CTrackerThread's mutex() and addPid() methods are accessible through the
            // forward declaration in the header (they're public methods)
            {
                CScopedLock lock(m_TrackerThread->mutex());
                m_TrackerThread->addPid(childPid);
            }
            return true;
        }
        HANDLE_FATAL(<< "Failed to spawn pytorch_inference with Sandbox2: " << processPath);
        return false;
#else
        HANDLE_FATAL(<< "Sandbox2 is not available but required for pytorch_inference process: "
                     << processPath);
        return false;
#endif
#else
        HANDLE_FATAL(<< "Sandbox2 is disabled but required for pytorch_inference process: "
                     << processPath);
        return false;
#endif
    }
#endif // __linux__

    if (std::find(m_PermittedProcessPaths.begin(), m_PermittedProcessPaths.end(),
                  processPath) == m_PermittedProcessPaths.end()) {
        LOG_ERROR(<< "Spawning process '" << processPath << "' is not permitted");
        return false;
    }

    if (::access(processPath.c_str(), X_OK) != 0) {
        LOG_ERROR(<< "Cannot execute '" << processPath << "': " << ::strerror(errno));
        return false;
    }

    using TCharPVec = std::vector<char*>;
    // Size of argv is two bigger than the number of arguments because:
    // 1) We add the program name at the beginning
    // 2) The list of arguments must be terminated by a NULL pointer
    TCharPVec argv;
    argv.reserve(args.size() + 2);

    // These const_casts may cause const data to get modified BUT only in the
    // child post-fork, so this won't corrupt parent process data
    argv.push_back(const_cast<char*>(processPath.c_str()));
    for (size_t index = 0; index < args.size(); ++index) {
        argv.push_back(const_cast<char*>(args[index].c_str()));
    }
    argv.push_back(static_cast<char*>(nullptr));

    posix_spawn_file_actions_t fileActions;
    if (setupFileActions(&fileActions, m_MaxObservedFd) == false) {
        LOG_ERROR(<< "Failed to set up file actions prior to spawn of '"
                  << processPath << "': " << ::strerror(errno));
        return false;
    }
    posix_spawnattr_t spawnAttributes;
    if (::posix_spawnattr_init(&spawnAttributes) != 0) {
        LOG_ERROR(<< "Failed to set up spawn attributes prior to spawn of '"
                  << processPath << "': " << ::strerror(errno));
        return false;
    }
    ::posix_spawnattr_setflags(&spawnAttributes, POSIX_SPAWN_SETPGROUP);

    {
        // Hold the tracker thread mutex until the PID is added to the tracker
        // to avoid a race condition if the process is started but dies really
        // quickly
        CScopedLock lock(m_TrackerThread->mutex());

        int err(::posix_spawn(&childPid, processPath.c_str(), &fileActions,
                              &spawnAttributes, &argv[0], environ));

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

// Sandbox2 spawn function - called from base CDetachedProcessSpawner.cc
bool ml_core_spawnWithSandbox2Linux(const std::string& processPath,
                                    const std::vector<std::string>& args,
                                    CProcess::TPid& childPid) {
#ifndef SANDBOX2_DISABLED
#ifdef SANDBOX2_AVAILABLE
    // Parse command line arguments
    detail::ProcessPaths paths = detail::parseProcessPaths(args);
    paths.pytorchLibDir = detail::calculatePytorchLibDir(processPath);

    // Build Sandbox2 policy
    auto policy = detail::buildSandboxPolicy(paths);

    // Create executor and sandbox
    auto executor = std::make_unique<sandbox2::Executor>(processPath, args);
    sandbox2::Sandbox2 sandbox(std::move(executor), std::move(policy));

    // Launch sandboxed process asynchronously
    sandbox.RunAsync();
    
    // TODO: Extract PID from Sandbox2 - current API limitation
    childPid = 0;
    
    LOG_DEBUG(<< "Spawned sandboxed '" << processPath << "'");
    return true;
#else
    return false;
#endif
#else
    return false;
#endif
}

} // namespace core
} // namespace ml
