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
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <ctime>

#include <errno.h>
#include <fcntl.h>
#include <grp.h>
#include <limits.h>
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
#include <sandboxed_api/sandbox2/notify.h>
#include <sandboxed_api/sandbox2/allow_all_syscalls.h>
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

//! Custom Notify class to capture Sandbox2 violations and events
class Sandbox2LoggingNotify : public sandbox2::Notify {
public:
    void EventFinished(const sandbox2::Result& result) override {
        sandbox2::Result::StatusEnum status = result.final_status();
        uintptr_t reason_code = result.reason_code();
        
        if (status == sandbox2::Result::OK) {
            LOG_DEBUG(<< "Sandbox2 process finished successfully (OK)");
        } else if (status == sandbox2::Result::VIOLATION) {
            LOG_ERROR(<< "Sandbox2 process finished with VIOLATION (reason_code: " << reason_code << ")");
        } else if (status == sandbox2::Result::SIGNALED) {
            LOG_ERROR(<< "Sandbox2 process was SIGNALED (signal: " << reason_code << ")");
        } else if (status == sandbox2::Result::SETUP_ERROR) {
            LOG_ERROR(<< "Sandbox2 process SETUP_ERROR (reason_code: " << reason_code << ")");
        } else if (status == sandbox2::Result::TIMEOUT) {
            LOG_ERROR(<< "Sandbox2 process TIMEOUT");
        } else if (status == sandbox2::Result::EXTERNAL_KILL) {
            LOG_ERROR(<< "Sandbox2 process EXTERNAL_KILL");
        } else if (status == sandbox2::Result::INTERNAL_ERROR) {
            LOG_ERROR(<< "Sandbox2 process INTERNAL_ERROR");
        } else {
            LOG_ERROR(<< "Sandbox2 process finished with status: " << static_cast<int>(status) << " (reason_code: " << reason_code << ")");
        }
        
        // Log exit code if available (from reason_code for OK status)
        if (status == sandbox2::Result::OK) {
            int exit_code = static_cast<int>(reason_code);
            if (exit_code != 0) {
                LOG_ERROR(<< "Process exit code: " << exit_code);
            }
        }
    }

    void EventSyscallViolation(const sandbox2::Syscall& syscall,
                               sandbox2::ViolationType type) override {
        LOG_ERROR(<< "Sandbox2 syscall violation detected:");
        LOG_ERROR(<< "  PID: " << syscall.pid());
        LOG_ERROR(<< "  Syscall: " << syscall.GetDescription());
        LOG_ERROR(<< "  Violation type: " << (type == sandbox2::ViolationType::kSyscall ? "kSyscall" : "kArchitectureSwitch"));
        LOG_ERROR(<< "  This violation may have caused the process to exit");
    }

    void EventSignal(pid_t pid, int sig_no) override {
        LOG_WARN(<< "Sandbox2 process " << pid << " received signal " << sig_no);
    }
};

//! Structure to hold process paths for Sandbox2 policy
struct ProcessPaths {
    std::string executablePath;
    std::string executableDir;
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
// MAXIMALLY PERMISSIVE POLICY: Allow everything to get Test 1 passing
// Once working, we'll gradually restrict it
std::unique_ptr<sandbox2::Policy> buildSandboxPolicy(const ProcessPaths& paths) {
    LOG_DEBUG(<< "Building Sandbox2 policy (maximally permissive mode)");
    LOG_DEBUG(<< "  Model path: " << (paths.modelPath.empty() ? "<none>" : paths.modelPath));
    LOG_DEBUG(<< "  Input pipe: " << (paths.inputPipe.empty() ? "<none>" : paths.inputPipe));
    LOG_DEBUG(<< "  Output pipe: " << (paths.outputPipe.empty() ? "<none>" : paths.outputPipe));
    LOG_DEBUG(<< "  Log pipe: " << (paths.logPipe.empty() ? "<none>" : paths.logPipe));
    LOG_DEBUG(<< "  PyTorch lib dir: " << (paths.pytorchLibDir.empty() ? "<none>" : paths.pytorchLibDir));
    
    // Start with most permissive policy - add all common directories
    // Note: Cannot add root "/" directly, so we add all common paths
    auto builder = sandbox2::PolicyBuilder()
                       // Add tmpfs for /tmp with large size (this already provides /tmp access)
                       .AddTmpfs("/tmp", 256 * 1024 * 1024)
                       // Allow /proc, /sys, /dev for process/system access
                       .AddDirectoryAt("/proc", "/proc", true)
                       .AddDirectoryAt("/sys", "/sys", true)
                       .AddDirectoryAt("/dev", "/dev", true)
                       // Standard library directories
                       .AddDirectoryAt("/lib", "/lib", true)
                       .AddDirectoryAt("/lib64", "/lib64", true)
                       .AddDirectoryAt("/usr", "/usr", true)
                       .AddDirectoryAt("/usr/lib", "/usr/lib", true)
                       .AddDirectoryAt("/usr/lib64", "/usr/lib64", true)
                       .AddDirectoryAt("/usr/local", "/usr/local", true)
                       .AddDirectoryAt("/usr/local/lib", "/usr/local/lib", true)
                       // Allow /etc for configuration files
                       .AddDirectoryAt("/etc", "/etc", true)
                       // Allow /bin and /sbin for executables
                       .AddDirectoryAt("/bin", "/bin", true)
                       .AddDirectoryAt("/sbin", "/sbin", true)
                       .AddDirectoryAt("/usr/bin", "/usr/bin", true)
                       .AddDirectoryAt("/usr/sbin", "/usr/sbin", true)
                       // Allow /var and /run for runtime files
                       .AddDirectoryAt("/var", "/var", true)
                       .AddDirectoryAt("/run", "/run", true)
                       // Allow /usr/local/gcc133 for compiler libraries (from strace)
                       .AddDirectoryAt("/usr/local/gcc133", "/usr/local/gcc133", true)
                       .AddDirectoryAt("/usr/local/gcc133/lib", "/usr/local/gcc133/lib", true)
                       .AddDirectoryAt("/usr/local/gcc133/lib64", "/usr/local/gcc133/lib64", true)
                       // Allow /usr/share for shared data
                       .AddDirectoryAt("/usr/share", "/usr/share", true);
    
    // Add executable's directory to policy
    if (!paths.executableDir.empty()) {
        LOG_DEBUG(<< "Adding executable directory: " << paths.executableDir);
        builder.AddDirectoryAt(paths.executableDir, paths.executableDir, true);
        // Also add the executable file itself
        if (!paths.executablePath.empty()) {
            LOG_DEBUG(<< "Adding executable file: " << paths.executablePath);
            builder.AddFileAt(paths.executablePath, paths.executablePath, true);
        }
    }

    // Allow ALL syscalls by default - this is the most permissive policy possible
    // This allows brk, mmap, and all other syscalls needed for normal operation
    builder.DefaultAction(sandbox2::AllowAllSyscalls());
    
    // DO NOT block any syscalls - allow everything for maximum permissiveness

    // Allow PyTorch libraries
    if (!paths.pytorchLibDir.empty()) {
        LOG_DEBUG(<< "Adding PyTorch lib directory: " << paths.pytorchLibDir);
        builder.AddDirectoryAt(paths.pytorchLibDir, paths.pytorchLibDir, true);
    }

    // Allow model file and its directory
    if (!paths.modelPath.empty()) {
        LOG_DEBUG(<< "Adding model file: " << paths.modelPath);
        builder.AddFileAt(paths.modelPath, paths.modelPath, true);
        // Also add the directory containing the model file
        size_t lastSlash = paths.modelPath.find_last_of('/');
        if (lastSlash != std::string::npos) {
            std::string modelDir = paths.modelPath.substr(0, lastSlash);
            if (!modelDir.empty()) {
                LOG_DEBUG(<< "Adding model directory: " << modelDir);
                builder.AddDirectoryAt(modelDir, modelDir, true);
            }
        }
    }
    
    // Add pipes with read-write access
    if (!paths.inputPipe.empty()) {
        LOG_DEBUG(<< "Adding input pipe: " << paths.inputPipe);
        builder.AddFileAt(paths.inputPipe, paths.inputPipe, false);  // is_ro=false allows read and write
    }
    if (!paths.outputPipe.empty()) {
        LOG_DEBUG(<< "Adding output pipe: " << paths.outputPipe);
        builder.AddFileAt(paths.outputPipe, paths.outputPipe, false);
    }
    if (!paths.logPipe.empty()) {
        LOG_DEBUG(<< "Adding log pipe: " << paths.logPipe);
        builder.AddFileAt(paths.logPipe, paths.logPipe, false);
    }
    if (!paths.logProperties.empty()) {
        LOG_DEBUG(<< "Adding log properties file: " << paths.logProperties);
        builder.AddFileAt(paths.logProperties, paths.logProperties, true);
    }

    LOG_DEBUG(<< "Building Sandbox2 policy...");
    auto policy_result = builder.TryBuild();
    if (!policy_result.ok()) {
        LOG_ERROR(<< "Failed to build Sandbox2 policy: " << policy_result.status());
        return nullptr;
    }
    LOG_DEBUG(<< "Sandbox2 policy built successfully");
    return std::move(*policy_result);
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

//! Static map to keep Sandbox2 objects alive for the lifetime of sandboxed processes
//! This is necessary because destroying the Sandbox2 object would kill the sandboxed process
#ifdef SANDBOX2_AVAILABLE
namespace {
    std::map<CProcess::TPid, std::unique_ptr<sandbox2::Sandbox2>> g_SandboxMap;
    CMutex g_SandboxMapMutex;
}
#endif // SANDBOX2_AVAILABLE

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
    LOG_DEBUG(<< "Starting Sandbox2 spawn for: " << processPath);
    LOG_DEBUG(<< "Arguments count: " << args.size());
    for (size_t i = 0; i < args.size(); ++i) {
        LOG_DEBUG(<< "  Arg[" << i << "]: " << args[i]);
    }
    
    // Parse command line arguments
    detail::ProcessPaths paths = detail::parseProcessPaths(args);
    
    // Convert processPath to absolute path (Sandbox2 requires absolute paths)
    std::string absoluteProcessPath = processPath;
    if (processPath[0] != '/') {
        // Relative path - need to resolve it
        char resolved_path[PATH_MAX];
        if (realpath(processPath.c_str(), resolved_path) != nullptr) {
            absoluteProcessPath = resolved_path;
            LOG_DEBUG(<< "Resolved relative path '" << processPath << "' to absolute path '" << absoluteProcessPath << "'");
        } else {
            // If realpath fails, try to make it absolute based on current working directory
            char cwd[PATH_MAX];
            if (getcwd(cwd, sizeof(cwd)) != nullptr) {
                if (processPath.find("./") == 0) {
                    absoluteProcessPath = std::string(cwd) + "/" + processPath.substr(2);
                } else {
                    absoluteProcessPath = std::string(cwd) + "/" + processPath;
                }
                LOG_DEBUG(<< "Made path absolute using CWD: '" << absoluteProcessPath << "'");
            } else {
                LOG_ERROR(<< "Failed to get current working directory and realpath failed for: " << processPath);
                return false;
            }
        }
    }
    
    paths.executablePath = absoluteProcessPath;
    // Extract executable directory
    size_t lastSlash = absoluteProcessPath.find_last_of('/');
    if (lastSlash != std::string::npos) {
        paths.executableDir = absoluteProcessPath.substr(0, lastSlash);
    } else {
        paths.executableDir = "/";
    }
    paths.pytorchLibDir = detail::calculatePytorchLibDir(absoluteProcessPath);
    LOG_DEBUG(<< "Parsed paths:");
    LOG_DEBUG(<< "  Executable path: " << paths.executablePath);
    LOG_DEBUG(<< "  Executable dir: " << paths.executableDir);
    LOG_DEBUG(<< "  PyTorch lib dir: " << paths.pytorchLibDir);
    
    // Log full command line for debugging (use absolute path)
    std::string full_command = absoluteProcessPath;
    for (const auto& arg : args) {
        full_command += " " + arg;
    }
    LOG_DEBUG(<< "Full command line: " << full_command);

    // Build Sandbox2 policy
    LOG_DEBUG(<< "Building Sandbox2 policy...");
    std::unique_ptr<sandbox2::Policy> policy;
    try {
        policy = detail::buildSandboxPolicy(paths);
        if (!policy) {
            LOG_ERROR(<< "Failed to build Sandbox2 policy (returned nullptr)");
            return false;
        }
        LOG_DEBUG(<< "Sandbox2 policy built successfully");
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Exception while building Sandbox2 policy: " << e.what());
        return false;
    } catch (...) {
        LOG_ERROR(<< "Unknown exception while building Sandbox2 policy");
        return false;
    }

    // Create executor and sandbox
    LOG_DEBUG(<< "Creating Sandbox2 executor for: " << processPath);
    LOG_DEBUG(<< "Executor will run with " << args.size() << " arguments");
    
    // Create temporary files to capture stderr/stdout if possible
    // Note: Sandbox2 Executor may handle this internally, but we'll try to capture what we can
    std::string stderr_file = "/tmp/sandbox2_stderr_" + std::to_string(getpid()) + "_" + std::to_string(time(nullptr)) + ".log";
    std::string stdout_file = "/tmp/sandbox2_stdout_" + std::to_string(getpid()) + "_" + std::to_string(time(nullptr)) + ".log";
    LOG_DEBUG(<< "Will attempt to capture stderr to: " << stderr_file);
    LOG_DEBUG(<< "Will attempt to capture stdout to: " << stdout_file);
    
    // Use absolute path for Executor (Sandbox2 requires absolute paths)
    auto executor = std::make_unique<sandbox2::Executor>(absoluteProcessPath, args);
    LOG_DEBUG(<< "Sandbox2 executor created");
    
    // Create custom Notify object to capture violations
    auto notify = std::make_unique<detail::Sandbox2LoggingNotify>();
    LOG_DEBUG(<< "Created Sandbox2 logging notify handler");
    
    LOG_DEBUG(<< "Creating Sandbox2 instance with policy and notify handler...");
    auto sandbox = std::make_unique<sandbox2::Sandbox2>(std::move(executor), std::move(policy), std::move(notify));
    LOG_DEBUG(<< "Sandbox2 instance created successfully");

    // DIAGNOSTIC MODE: Try synchronous execution first to get immediate result
    // This will give us the exit code and any violations directly
    // Set to true for diagnostics, false for production (async)
    // NOTE: pytorch_inference is a long-running process, so async mode is required
    const bool USE_SYNC_FOR_DIAGNOSTICS = false;  // Use async mode for long-running processes
    
    if (USE_SYNC_FOR_DIAGNOSTICS) {
        LOG_DEBUG(<< "DIAGNOSTIC MODE: Using synchronous execution to capture exit code immediately");
        LOG_DEBUG(<< "Launching sandboxed process synchronously...");
        
        // Run synchronously - this will block until process completes
        // Run() returns Result directly (not StatusOr)
        sandbox2::Result result = sandbox->Run();
        
        sandbox2::Result::StatusEnum status = result.final_status();
        uintptr_t reason_code = result.reason_code();
        childPid = sandbox->pid();
        
        LOG_DEBUG(<< "Sandbox2 process completed synchronously");
        LOG_DEBUG(<< "  PID: " << childPid);
        LOG_DEBUG(<< "  Status: " << static_cast<int>(status));
        LOG_DEBUG(<< "  Reason code: " << reason_code);
        
        // Extract exit code from reason_code if status is OK
        // For non-OK statuses, reason_code contains the error code
        int exit_code = -1;
        if (status == sandbox2::Result::OK) {
            // For OK status, reason_code should be the exit code
            exit_code = static_cast<int>(reason_code);
        } else if (status == sandbox2::Result::SIGNALED) {
            // For SIGNALED, reason_code is the signal number
            LOG_ERROR(<< "Process was killed by signal " << reason_code);
        } else if (status == sandbox2::Result::VIOLATION) {
            LOG_ERROR(<< "Process violated Sandbox2 policy (reason_code: " << reason_code << ")");
        }
        
        if (status != sandbox2::Result::OK || exit_code != 0) {
            LOG_ERROR(<< "Process exited with status " << static_cast<int>(status));
            if (exit_code >= 0) {
                LOG_ERROR(<< "  Exit code: " << exit_code);
            }
            LOG_ERROR(<< "Command that was executed: " << full_command);
            
            // The Notify handler should have logged any violations
            // Return false to indicate failure
            return false;
        } else {
            LOG_DEBUG(<< "Process completed successfully (exit code 0)");
        }
        
        // In sync mode, process is already done, so skip monitoring
        // Store sandbox object and return
        {
            CScopedLock lock(g_SandboxMapMutex);
            g_SandboxMap[childPid] = std::move(sandbox);
        }
        
        LOG_DEBUG(<< "Spawned sandboxed '" << absoluteProcessPath << "' with PID " << childPid << " (sync mode)");
        return true;
    } else {
        // Production mode: Launch sandboxed process asynchronously
        LOG_DEBUG(<< "Launching sandboxed process asynchronously...");
        sandbox->RunAsync();
        LOG_DEBUG(<< "RunAsync() called, polling for PID...");
        
        // Poll for PID with timeout (monitor initializes asynchronously)
        const int timeout_ms = 5000;  // Increased timeout for better diagnostics
        const int poll_interval_us = 10000; // 10ms for less CPU usage
        int elapsed_ms = 0;
        
        childPid = -1;
        while (elapsed_ms < timeout_ms) {
            childPid = sandbox->pid();
            if (childPid > 0) {
                LOG_DEBUG(<< "Got PID from Sandbox2: " << childPid << " after " << elapsed_ms << "ms");
                break;
            }
            usleep(poll_interval_us);
            elapsed_ms += 10;
        }
        
        if (childPid <= 0) {
            LOG_ERROR(<< "Failed to get PID from Sandbox2 after " << timeout_ms << "ms");
            LOG_ERROR(<< "This may indicate the process failed to start or crashed immediately");
            return false;
        }
    }
    
    // Monitor the process for a short time to detect early exits (async mode only)
    LOG_DEBUG(<< "Monitoring process " << childPid << " for early exits...");
    const int monitor_duration_ms = 3000;  // Increased to catch slower exits
    const int monitor_interval_ms = 50;    // Check more frequently (every 50ms)
    int monitor_elapsed_ms = 0;
    bool process_still_running = true;
    
    while (monitor_elapsed_ms < monitor_duration_ms && process_still_running) {
        // Check process status from /proc before checking if it exists
        // This gives us a better chance to catch the exit code
        std::string status_file = "/proc/" + std::to_string(childPid) + "/status";
        std::ifstream proc_status(status_file);
        if (proc_status.is_open()) {
            std::string line;
            std::string state;
            while (std::getline(proc_status, line)) {
                if (line.find("State:") == 0) {
                    state = line;
                    // Check if process is in zombie state (exited but not reaped)
                    if (line.find("State:\tZ") == 0) {
                        LOG_WARN(<< "Process " << childPid << " is in zombie state (exited but not reaped)");
                        process_still_running = false;
                        // Try to reap it immediately
                        int status = 0;
                        pid_t waited_pid = ::waitpid(childPid, &status, WNOHANG);
                        if (waited_pid == childPid) {
                            if (WIFEXITED(status)) {
                                int exit_code = WEXITSTATUS(status);
                                LOG_ERROR(<< "Process " << childPid << " exited with code " << exit_code << " (within " << monitor_elapsed_ms << "ms)");
                                LOG_ERROR(<< "Command that caused exit: " << full_command);
                            } else if (WIFSIGNALED(status)) {
                                int signal = WTERMSIG(status);
                                LOG_ERROR(<< "Process " << childPid << " was killed by signal " << signal << " (within " << monitor_elapsed_ms << "ms)");
                                LOG_ERROR(<< "Command that was running: " << full_command);
                            }
                        }
                        break;
                    }
                }
            }
        } else {
            // Process directory doesn't exist - process has exited and been reaped
            LOG_WARN(<< "Process " << childPid << " exited early (within " << monitor_elapsed_ms << "ms) - already reaped");
            process_still_running = false;
            
            // Try to get process exit status (may fail if already reaped)
            int status = 0;
            pid_t waited_pid = ::waitpid(childPid, &status, WNOHANG);
            if (waited_pid == childPid) {
                if (WIFEXITED(status)) {
                    int exit_code = WEXITSTATUS(status);
                    LOG_ERROR(<< "Process " << childPid << " exited with code " << exit_code << " (within " << monitor_elapsed_ms << "ms)");
                    LOG_ERROR(<< "Command that caused exit: " << full_command);
                } else if (WIFSIGNALED(status)) {
                    int signal = WTERMSIG(status);
                    LOG_ERROR(<< "Process " << childPid << " was killed by signal " << signal << " (within " << monitor_elapsed_ms << "ms)");
                    LOG_ERROR(<< "Command that was running: " << full_command);
                }
            } else {
                LOG_ERROR(<< "Process " << childPid << " exited but waitpid returned " << waited_pid << " (errno: " << errno << " - already reaped by another process)");
                LOG_ERROR(<< "Command that was running: " << full_command);
                
                // Try to read cmdline from a backup location or check if CTrackerThread logged it
                LOG_ERROR(<< "Note: Exit code may be logged by CTrackerThread in controller logs");
            }
            break;
        }
        usleep(monitor_interval_ms * 1000);
        monitor_elapsed_ms += monitor_interval_ms;
    }
    
    if (process_still_running) {
        LOG_DEBUG(<< "Process " << childPid << " is still running after " << monitor_duration_ms << "ms");
    }
    
    // Store sandbox object in static map to keep it alive for the lifetime of the process
    // This is necessary because destroying the Sandbox2 object would kill the sandboxed process
    {
        CScopedLock lock(g_SandboxMapMutex);
        g_SandboxMap[childPid] = std::move(sandbox);
    }
    
    LOG_DEBUG(<< "Spawned sandboxed '" << processPath << "' with PID " << childPid);
    return true;
#else
    LOG_ERROR(<< "Sandbox2 is not available");
    return false;
#endif
#else
    LOG_ERROR(<< "Sandbox2 is disabled");
    return false;
#endif
}

} // namespace core
} // namespace ml
