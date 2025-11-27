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
#include <atomic>
#include <ctime>
#include <fstream>
#include <map>
#include <set>
#include <string>

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
#include <sandboxed_api/sandbox2/allow_all_syscalls.h>
#include <sandboxed_api/sandbox2/notify.h>
#include <sandboxed_api/sandbox2/policy.h>
#include <sandboxed_api/sandbox2/policybuilder.h>
#include <sandboxed_api/sandbox2/result.h>
#include <sandboxed_api/sandbox2/sandbox2.h>
#include <sandboxed_api/sandbox2/util/bpf_helper.h>
#include <sys/syscall.h>

// Fallback definitions for newer syscalls that may not be in RHEL8 headers
// These are only defined if not already present in sys/syscall.h
#ifndef __NR_statx
#ifdef __x86_64__
#define __NR_statx 332
#elif defined(__aarch64__)
#define __NR_statx 291
#endif
#endif

#ifndef __NR_rseq
#ifdef __x86_64__
#define __NR_rseq 334
#elif defined(__aarch64__)
#define __NR_rseq 293
#endif
#endif

#ifndef __NR_clone3
#define __NR_clone3 435
#endif

#ifndef __NR_execve
#ifdef __x86_64__
#define __NR_execve 59
#elif defined(__aarch64__)
#define __NR_execve 221
#endif
#endif

#ifndef __NR_execveat
#ifdef __x86_64__
#define __NR_execveat 322
#elif defined(__aarch64__)
#define __NR_execveat 281
#endif
#endif

#ifndef __NR_futex_waitv
#ifdef __x86_64__
#define __NR_futex_waitv 449
#elif defined(__aarch64__)
#define __NR_futex_waitv 449
#endif
#endif

#ifndef __NR_dup2
// dup2 syscall number is 33 on both x86_64 and aarch64
#define __NR_dup2 33
#endif

#ifndef __NR_dup3
// dup3 syscall number is 24 on both x86_64 and aarch64
#define __NR_dup3 24
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
    Sandbox2LoggingNotify() : m_Condition(m_Mutex), m_Failed(false), m_Status(sandbox2::Result::OK), m_ReasonCode(0) {}

    void EventFinished(const sandbox2::Result& result) override {
        sandbox2::Result::StatusEnum status = result.final_status();
        uintptr_t reason_code = result.reason_code();

        // Store failure state for main thread to check
        {
            core::CScopedLock lock(m_Mutex);
            m_Status = status;
            m_ReasonCode = reason_code;
            if (status != sandbox2::Result::OK) {
                m_Failed = true;
            }
        }
        m_Condition.broadcast();

        if (status == sandbox2::Result::OK) {
            LOG_DEBUG(<< "Sandbox2 process finished successfully (OK)");
        } else if (status == sandbox2::Result::VIOLATION) {
            LOG_ERROR(<< "Sandbox2 process finished with VIOLATION (reason_code: " << reason_code
                      << ")");
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
            LOG_ERROR(<< "Sandbox2 process finished with status: " << static_cast<int>(status)
                      << " (reason_code: " << reason_code << ")");
        }

        // Log exit code if available (from reason_code for OK status)
        if (status == sandbox2::Result::OK) {
            int exit_code = static_cast<int>(reason_code);
            if (exit_code != 0) {
                LOG_ERROR(<< "Process exit code: " << exit_code);
            }
        }
    }

    // Check if process has failed (non-blocking)
    bool hasFailed() const {
        core::CScopedLock lock(m_Mutex);
        return m_Failed;
    }

    // Get failure status and reason code
    void getFailureInfo(sandbox2::Result::StatusEnum& status, uintptr_t& reasonCode) const {
        core::CScopedLock lock(m_Mutex);
        status = m_Status;
        reasonCode = m_ReasonCode;
    }

    // Wait for failure or success with timeout (returns true if failure detected, false on timeout)
    bool waitForFailure(std::uint32_t timeoutMs) {
        core::CScopedLock lock(m_Mutex);
        if (m_Failed) {
            return true;
        }
        // Wait for failure notification or timeout
        // wait() unlocks mutex, waits, then re-locks mutex
        m_Condition.wait(timeoutMs);
        return m_Failed;
    }

    void EventSyscallViolation(const sandbox2::Syscall& syscall,
                               sandbox2::ViolationType type) override {
        LOG_ERROR(<< "Sandbox2 syscall violation detected:");
        LOG_ERROR(<< "  PID: " << syscall.pid());
        LOG_ERROR(<< "  Syscall: " << syscall.GetDescription());
        LOG_ERROR(<< "  Violation type: "
                  << (type == sandbox2::ViolationType::kSyscall ? "kSyscall" : "kArchitectureSwitch"));
        LOG_ERROR(<< "  This violation may have caused the process to exit");
    }

    void EventSignal(pid_t pid, int sig_no) override {
        LOG_WARN(<< "Sandbox2 process " << pid << " received signal " << sig_no);
    }

private:
    mutable core::CMutex m_Mutex;
    core::CCondition m_Condition;
    std::atomic<bool> m_Failed;
    sandbox2::Result::StatusEnum m_Status;
    uintptr_t m_ReasonCode;
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
    std::string restorePipe;
    std::string logProperties;
    bool isRestorePipe = false;
};

//! Parse command line arguments to extract file paths
ProcessPaths parseProcessPaths(const std::vector<std::string>& args) {
    ProcessPaths paths;
    // First pass: find --restoreIsPipe flag
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--restoreIsPipe") {
            paths.isRestorePipe = true;
            LOG_DEBUG(<< "Found --restoreIsPipe flag at position " << i);
            break;
        }
    }
    // Second pass: extract paths
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg.find("--input=") == 0) {
            paths.inputPipe = arg.substr(8);
        } else if (arg.find("--output=") == 0) {
            paths.outputPipe = arg.substr(9);
        } else if (arg.find("--restore=") == 0) {
            paths.modelPath = arg.substr(10);
            LOG_DEBUG(<< "Found --restore= argument: " << paths.modelPath << ", isRestorePipe=" << paths.isRestorePipe);
            if (paths.isRestorePipe) {
                paths.restorePipe = paths.modelPath;
                LOG_DEBUG(<< "Set restorePipe to: " << paths.restorePipe);
            }
        } else if (arg.find("--logPipe=") == 0) {
            paths.logPipe = arg.substr(10);
        } else if (arg.find("--logProperties=") == 0) {
            paths.logProperties = arg.substr(16);
        }
    }
    LOG_DEBUG(<< "parseProcessPaths result: isRestorePipe=" << paths.isRestorePipe 
              << ", restorePipe=" << (paths.restorePipe.empty() ? "<empty>" : paths.restorePipe)
              << ", modelPath=" << (paths.modelPath.empty() ? "<empty>" : paths.modelPath));
    return paths;
}

//! Calculate PyTorch library directory from executable path
std::string calculatePytorchLibDir(const std::string& processPath) {
    size_t lastSlash = processPath.find_last_of('/');
    if (lastSlash == std::string::npos)
        return "";

    std::string exeDir = processPath.substr(0, lastSlash);
    size_t lastDirSlash = exeDir.find_last_of('/');
    if (lastDirSlash == std::string::npos)
        return "";

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
    LOG_DEBUG(<< "  Output pipe: "
              << (paths.outputPipe.empty() ? "<none>" : paths.outputPipe));
    LOG_DEBUG(<< "  Log pipe: " << (paths.logPipe.empty() ? "<none>" : paths.logPipe));
    LOG_DEBUG(<< "  Restore pipe: " << (paths.restorePipe.empty() ? "<none>" : paths.restorePipe));
    LOG_DEBUG(<< "  Is restore pipe: " << (paths.isRestorePipe ? "true" : "false"));
    LOG_DEBUG(<< "  PyTorch lib dir: "
              << (paths.pytorchLibDir.empty() ? "<none>" : paths.pytorchLibDir));

    // Start with most permissive policy - add all common directories
    // Note: Cannot add root "/" directly, so we add all common paths
    auto builder = sandbox2::PolicyBuilder()
                       // Add tmpfs for /tmp with large size (this already provides /tmp access)
                       .AddTmpfs("/tmp", 256 * 1024 * 1024)
                       // Allow /proc, /sys, /dev for process/system access (read-only)
                       .AddDirectoryAt("/proc", "/proc", true) // read-only
                       .AddDirectoryAt("/sys", "/sys", true)   // read-only
                       .AddDirectoryAt("/dev", "/dev", true)   // read-only
                       // Standard library directories (read-only)
                       .AddDirectoryAt("/lib", "/lib", true)     // read-only
                       .AddDirectoryAt("/lib64", "/lib64", true) // read-only
                       .AddDirectoryAt("/usr", "/usr", true)     // read-only
                       .AddDirectoryAt("/usr/lib", "/usr/lib", true) // read-only
                       .AddDirectoryAt("/usr/lib64", "/usr/lib64", true) // read-only
                       .AddDirectoryAt("/usr/local", "/usr/local", true) // read-only
                       .AddDirectoryAt("/usr/local/lib", "/usr/local/lib", true) // read-only
                       // Allow /etc for configuration files (read-only)
                       .AddDirectoryAt("/etc", "/etc", true) // read-only
                       // Allow /bin and /sbin for executables (read-only)
                       .AddDirectoryAt("/bin", "/bin", true)   // read-only
                       .AddDirectoryAt("/sbin", "/sbin", true) // read-only
                       .AddDirectoryAt("/usr/bin", "/usr/bin", true) // read-only
                       .AddDirectoryAt("/usr/sbin", "/usr/sbin", true); // read-only
    // Note: /tmp is writable via AddTmpfs above
    // Note: Removed /var, /run, /usr/local/gcc133, /usr/share as they may not be needed
    // If test fails, we'll add them back one by one

    // Add executable's directory and all parent directories to policy
    // Sandbox2 requires all parent directories in the path to be accessible
    if (!paths.executableDir.empty()) {
        // Add all parent directories up to root
        std::string currentPath = paths.executableDir;
        while (!currentPath.empty() && currentPath != "/") {
            LOG_DEBUG(<< "Adding parent directory: " << currentPath);
            builder.AddDirectoryAt(currentPath, currentPath, true);
            size_t lastSlash = currentPath.find_last_of('/');
            if (lastSlash == 0) {
                // Reached root
                break;
            } else if (lastSlash != std::string::npos) {
                currentPath = currentPath.substr(0, lastSlash);
            } else {
                break;
            }
        }
        // Also add the executable directory itself (explicitly, in case it wasn't added above)
        LOG_DEBUG(<< "Adding executable directory: " << paths.executableDir);
        builder.AddDirectoryAt(paths.executableDir, paths.executableDir, true);
        // Also add the executable file itself
        // Note: false means read-write, but for executables we need execute permission
        // Sandbox2 will respect the file's actual permissions, so we use false to allow execution
        if (!paths.executablePath.empty()) {
            LOG_DEBUG(<< "Adding executable file: " << paths.executablePath);
            builder.AddFileAt(paths.executablePath, paths.executablePath, false);
        }
    }

    // Replace AllowAllSyscalls() with explicit syscall allowlist matching seccomp filter
    // This provides the same security level as the seccomp filter while using Sandbox2

    // Basic process control
    builder.AllowSyscall(__NR_exit);
    builder.AllowSyscall(__NR_exit_group);
    builder.AllowSyscall(__NR_brk);
    builder.AllowSyscall(__NR_getuid);
    builder.AllowSyscall(__NR_getpid);
    builder.AllowSyscall(__NR_getrusage);
    builder.AllowSyscall(__NR_getpriority);
    builder.AllowSyscall(__NR_setpriority);
    builder.AllowSyscall(__NR_prctl);
    builder.AllowSyscall(__NR_prlimit64);
    builder.AllowSyscall(__NR_uname);

    // CPU/scheduling operations
    builder.AllowSyscall(__NR_sched_getaffinity);
    builder.AllowSyscall(__NR_sched_setaffinity);
    builder.AllowSyscall(__NR_getcpu);

    // Directory operations
    builder.AllowSyscall(__NR_getcwd);

    // Memory management
    builder.AllowSyscall(__NR_mmap);
    builder.AllowSyscall(__NR_munmap);
    builder.AllowSyscall(__NR_mremap);
    builder.AllowSyscall(__NR_mprotect);
    builder.AllowSyscall(__NR_madvise);

    // File operations - basic
    builder.AllowSyscall(__NR_read);
    builder.AllowSyscall(__NR_write);
    builder.AllowSyscall(__NR_writev);
    builder.AllowSyscall(__NR_pread64);
    builder.AllowSyscall(__NR_pwrite64);
    builder.AllowSyscall(__NR_lseek);
    builder.AllowSyscall(__NR_close);
    builder.AllowSyscall(__NR_fcntl);
    builder.AllowSyscall(__NR_fstat);
    builder.AllowSyscall(__NR_statfs);

    // File operations - x86_64 specific
#ifdef __x86_64__
    builder.AllowSyscall(__NR_access);
    builder.AllowSyscall(__NR_open);
    builder.AllowSyscall(__NR_stat);
    builder.AllowSyscall(__NR_lstat);
    builder.AllowSyscall(__NR_readlink);
    builder.AllowSyscall(__NR_unlink);
    builder.AllowSyscall(__NR_mkdir);
    builder.AllowSyscall(__NR_rmdir);
    builder.AllowSyscall(__NR_mknod);
    builder.AllowSyscall(__NR_getdents);
    builder.AllowSyscall(__NR_time);
#elif defined(__aarch64__)
    // ARM64 uses faccessat instead of access
    builder.AllowSyscall(__NR_faccessat);
#endif

    // File operations - modern (all architectures)
    builder.AllowSyscall(__NR_openat);
    builder.AllowSyscall(__NR_newfstatat);
    builder.AllowSyscall(__NR_readlinkat);
    builder.AllowSyscall(__NR_mkdirat);
    builder.AllowSyscall(__NR_unlinkat);
    builder.AllowSyscall(__NR_mknodat);
    builder.AllowSyscall(__NR_getdents64);
    builder.AllowSyscall(__NR_statx);

    // File descriptor operations
    builder.AllowSyscall(__NR_dup);
    builder.AllowSyscall(__NR_dup2);
    builder.AllowSyscall(__NR_dup3);

    // Time operations
    builder.AllowSyscall(__NR_clock_gettime);
    builder.AllowSyscall(__NR_gettimeofday);
    builder.AllowSyscall(__NR_nanosleep);

    // Process/thread operations
    builder.AllowSyscall(__NR_clone);
    builder.AllowSyscall(__NR_clone3);
    builder.AllowSyscall(__NR_execve);
    builder.AllowSyscall(__NR_execveat);
    builder.AllowSyscall(__NR_futex);
    builder.AllowSyscall(__NR_futex_waitv);
    builder.AllowSyscall(__NR_set_robust_list);
    builder.AllowSyscall(__NR_set_tid_address);
    builder.AllowSyscall(__NR_rseq);
#ifdef __x86_64__
    // x86_64-specific: arch_prctl for thread-local storage
    builder.AllowSyscall(__NR_arch_prctl);
#endif

    // Signal operations
    builder.AllowSyscall(__NR_rt_sigaction);
    builder.AllowSyscall(__NR_rt_sigreturn);
    builder.AllowSyscall(__NR_rt_sigprocmask);
    builder.AllowSyscall(__NR_tgkill);

    // Random number generation
    builder.AllowSyscall(__NR_getrandom);

    // Network operations (for named pipes)
    builder.AllowSyscall(__NR_connect);

    // Allow PyTorch libraries (and all parent directories)
    if (!paths.pytorchLibDir.empty()) {
        // Add all parent directories up to root
        std::string currentPath = paths.pytorchLibDir;
        while (!currentPath.empty() && currentPath != "/") {
            LOG_DEBUG(<< "Adding PyTorch lib parent directory: " << currentPath);
            builder.AddDirectoryAt(currentPath, currentPath, true);
            size_t lastSlash = currentPath.find_last_of('/');
            if (lastSlash == 0) {
                // Reached root
                break;
            } else if (lastSlash != std::string::npos) {
                currentPath = currentPath.substr(0, lastSlash);
            } else {
                break;
            }
        }
        // Also add the PyTorch lib directory itself explicitly
        LOG_DEBUG(<< "Adding PyTorch lib directory: " << paths.pytorchLibDir);
        builder.AddDirectoryAt(paths.pytorchLibDir, paths.pytorchLibDir, true);
    }

    // Allow model file and its directory (and all parent directories)
    // Skip if it's a restore pipe (handled separately above)
    if (!paths.modelPath.empty() && !paths.isRestorePipe) {
        LOG_DEBUG(<< "Adding model file: " << paths.modelPath);
        builder.AddFileAt(paths.modelPath, paths.modelPath, true);
        // Also add the directory containing the model file and all parent directories
        size_t lastSlash = paths.modelPath.find_last_of('/');
        if (lastSlash != std::string::npos) {
            std::string modelDir = paths.modelPath.substr(0, lastSlash);
            if (!modelDir.empty()) {
                // Add all parent directories up to root (Sandbox2 requires all parent directories)
                std::string currentPath = modelDir;
                while (!currentPath.empty() && currentPath != "/") {
                    // Skip /tmp since it's already added as tmpfs
                    if (currentPath == "/tmp") {
                        LOG_DEBUG(<< "Skipping /tmp (already added as tmpfs)");
                        break;
                    }
                    LOG_DEBUG(<< "Adding model parent directory: " << currentPath);
                    builder.AddDirectoryAt(currentPath, currentPath, true);
                    size_t dirLastSlash = currentPath.find_last_of('/');
                    if (dirLastSlash == 0) {
                        // Reached root
                        break;
                    } else if (dirLastSlash != std::string::npos) {
                        currentPath = currentPath.substr(0, dirLastSlash);
                    } else {
                        break;
                    }
                }
                // Also add the model directory itself explicitly (unless it's /tmp)
                if (modelDir != "/tmp") {
                    LOG_DEBUG(<< "Adding model directory: " << modelDir);
                    builder.AddDirectoryAt(modelDir, modelDir, true);
                } else {
                    LOG_DEBUG(<< "Skipping /tmp directory (already added as tmpfs)");
                }
            }
        }
    }

    // Add pipes with read-write access
    // Helper lambda to add a file and its parent directories
    // For named pipes, we need to allow both read and write access to the file
    // even if we only use it in one direction, because the open() syscall needs
    // to be able to access the file
    auto addFileWithParents = [&builder](const std::string& filePath, bool readOnly) {
        LOG_DEBUG(<< "Adding file: " << filePath << " (readOnly=" << readOnly << ")");
        // For named pipes, always allow read-write access to the file itself
        // The readOnly parameter is just for documentation - named pipes need
        // both read and write access to be opened
        builder.AddFileAt(filePath, filePath, false); // false = read-write access
        // Also add parent directories
        size_t lastSlash = filePath.find_last_of('/');
        if (lastSlash != std::string::npos) {
            std::string fileDir = filePath.substr(0, lastSlash);
            if (!fileDir.empty()) {
                // Add all parent directories up to root
                std::string currentPath = fileDir;
                while (!currentPath.empty() && currentPath != "/") {
                    // Skip /tmp since it's already added as tmpfs
                    if (currentPath == "/tmp") {
                        LOG_DEBUG(<< "Skipping /tmp (already added as tmpfs)");
                        break;
                    }
                    LOG_DEBUG(<< "Adding parent directory: " << currentPath);
                    // Directories need read access for traversal and stat operations
                    builder.AddDirectoryAt(currentPath, currentPath, true);
                    size_t dirLastSlash = currentPath.find_last_of('/');
                    if (dirLastSlash == 0) {
                        break;
                    } else if (dirLastSlash != std::string::npos) {
                        currentPath = currentPath.substr(0, dirLastSlash);
                    } else {
                        break;
                    }
                }
                // Also add the file directory itself explicitly (unless it's /tmp)
                if (fileDir != "/tmp") {
                    LOG_DEBUG(<< "Adding file directory: " << fileDir);
                    builder.AddDirectoryAt(fileDir, fileDir, true);
                } else {
                    LOG_DEBUG(<< "Skipping /tmp directory (already added as tmpfs)");
                }
            }
        }
    };
    
    // Helper function to add a pipe directory (for pipes that will be created by the process)
    // This is needed because Sandbox2 validates file paths during policy building,
    // but pipes created with mkfifo() don't exist yet at that time.
    auto addPipeDirectory = [&builder](const std::string& pipePath) {
        if (pipePath.empty()) {
            return;
        }
        
        size_t lastSlash = pipePath.find_last_of('/');
        if (lastSlash == std::string::npos) {
            LOG_WARN(<< "Pipe path has no directory component: " << pipePath);
            return;
        }
        
        std::string pipeDir = pipePath.substr(0, lastSlash);
        if (pipeDir.empty()) {
            LOG_WARN(<< "Pipe directory is empty for path: " << pipePath);
            return;
        }
        
        // Check if directory exists and is accessible (for debugging)
        struct stat dirStat;
        int statResult = ::stat(pipeDir.c_str(), &dirStat);
        if (statResult == 0) {
            LOG_DEBUG(<< "Pipe directory exists: " << pipeDir);
            if (::access(pipeDir.c_str(), W_OK) == 0) {
                LOG_DEBUG(<< "Pipe directory is writable: " << pipeDir);
            } else {
                LOG_WARN(<< "Pipe directory is NOT writable (errno: " << errno << "): " << pipeDir);
            }
        } else {
            LOG_DEBUG(<< "Pipe directory does not exist yet (will be created): " << pipeDir << " (errno: " << errno << ")");
        }
        
        LOG_DEBUG(<< "Adding writable pipe directory: " << pipeDir << " for pipe: " << pipePath);
        
        // CRITICAL: Add the pipe directory as writable so the process can create the pipe
        builder.AddDirectoryAt(pipeDir, pipeDir, false); // false = writable
        
        // Add parent directories (read-only for traversal) but limit depth and skip system directories
        // This avoids validation issues with excessive parent directory additions
        std::string currentPath = pipeDir;
        int depth = 0;
        const int MAX_PARENT_DEPTH = 5; // Limit to 5 levels up to avoid adding too many system dirs
        
        while (!currentPath.empty() && currentPath != "/" && depth < MAX_PARENT_DEPTH) {
            // Skip /tmp since it's already added as tmpfs (which is writable)
            if (currentPath == "/tmp") {
                LOG_DEBUG(<< "Skipping /tmp (already added as writable tmpfs)");
                break;
            }
            
            // Skip system directories that are already in the policy
            // These are typically read-only system directories that don't need to be added again
            if (currentPath == "/home" || currentPath == "/usr" || currentPath == "/lib" || 
                currentPath == "/lib64" || currentPath == "/bin" || currentPath == "/sbin" ||
                currentPath == "/etc" || currentPath == "/proc" || currentPath == "/sys" ||
                currentPath == "/dev") {
                LOG_DEBUG(<< "Skipping system directory (already in policy): " << currentPath);
                break;
            }
            
            LOG_DEBUG(<< "Adding pipe directory parent: " << currentPath);
            builder.AddDirectoryAt(currentPath, currentPath, true); // true = read-only
            
            size_t dirLastSlash = currentPath.find_last_of('/');
            if (dirLastSlash == 0) {
                break;
            } else if (dirLastSlash != std::string::npos) {
                currentPath = currentPath.substr(0, dirLastSlash);
                depth++;
            } else {
                break;
            }
        }
    };
    
    // For pipes that may be created by the process, add the directory as writable
    // Also add the pipe file path for when it exists (for opening)
    if (!paths.inputPipe.empty()) {
        addPipeDirectory(paths.inputPipe);
        addFileWithParents(paths.inputPipe, true);
    }
    if (!paths.outputPipe.empty()) {
        addPipeDirectory(paths.outputPipe);
        addFileWithParents(paths.outputPipe, false);
    }
    if (!paths.logPipe.empty()) {
        addPipeDirectory(paths.logPipe);
        addFileWithParents(paths.logPipe, false);
    }
    // Handle restore pipe separately when it's a pipe (not a regular file)
    // For restore pipes that don't exist yet, we need to allow the directory to be writable
    // so the process can create the pipe using mkfifo()
    if (paths.isRestorePipe && paths.restorePipe.empty() == false) {
        LOG_INFO(<< "Adding restore pipe directory to Sandbox2 policy (pipe will be created by process): " << paths.restorePipe);
        
        // Add the pipe directory as writable so the process can create the pipe
        addPipeDirectory(paths.restorePipe);
        
        // Note: We don't add the pipe file itself because it doesn't exist yet.
        // The process will create it using mkfifo(), which is already allowed via __NR_mknod/__NR_mknodat.
        // Once created, the directory permissions will allow access to the pipe.
        LOG_INFO(<< "Restore pipe directory added to policy successfully");
    } else if (paths.isRestorePipe) {
        LOG_ERROR(<< "Restore pipe flag is set but restore pipe path is empty! Model path: " << paths.modelPath);
    } else if (!paths.restorePipe.empty()) {
        LOG_WARN(<< "Restore pipe path is set but isRestorePipe flag is false: " << paths.restorePipe);
    }
    if (!paths.logProperties.empty()) {
        LOG_DEBUG(<< "Adding log properties file: " << paths.logProperties);
        builder.AddFileAt(paths.logProperties, paths.logProperties, true);
        // Also add parent directories for log properties file
        size_t lastSlash = paths.logProperties.find_last_of('/');
        if (lastSlash != std::string::npos) {
            std::string logPropsDir = paths.logProperties.substr(0, lastSlash);
            if (!logPropsDir.empty()) {
                // Add all parent directories up to root
                std::string currentPath = logPropsDir;
                while (!currentPath.empty() && currentPath != "/") {
                    // Skip /tmp since it's already added as tmpfs
                    if (currentPath == "/tmp") {
                        LOG_DEBUG(<< "Skipping /tmp (already added as tmpfs)");
                        break;
                    }
                    LOG_DEBUG(<< "Adding log properties parent directory: " << currentPath);
                    builder.AddDirectoryAt(currentPath, currentPath, true);
                    size_t dirLastSlash = currentPath.find_last_of('/');
                    if (dirLastSlash == 0) {
                        break;
                    } else if (dirLastSlash != std::string::npos) {
                        currentPath = currentPath.substr(0, dirLastSlash);
                    } else {
                        break;
                    }
                }
                // Also add the log properties directory itself explicitly (unless it's /tmp)
                if (logPropsDir != "/tmp") {
                    LOG_DEBUG(<< "Adding log properties directory: " << logPropsDir);
                    builder.AddDirectoryAt(logPropsDir, logPropsDir, true);
                } else {
                    LOG_DEBUG(<< "Skipping /tmp directory (already added as tmpfs)");
                }
            }
        }
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
std::map<core::CProcess::TPid, std::unique_ptr<sandbox2::Sandbox2>> g_SandboxMap;
core::CMutex g_SandboxMapMutex;
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
            LOG_DEBUG(<< "Resolved relative path '" << processPath
                      << "' to absolute path '" << absoluteProcessPath << "'");
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
                LOG_ERROR(<< "Failed to get current working directory and realpath failed for: "
                          << processPath);
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
    std::string stderr_file = "/tmp/sandbox2_stderr_" + std::to_string(getpid()) +
                              "_" + std::to_string(time(nullptr)) + ".log";
    std::string stdout_file = "/tmp/sandbox2_stdout_" + std::to_string(getpid()) +
                              "_" + std::to_string(time(nullptr)) + ".log";
    LOG_DEBUG(<< "Will attempt to capture stderr to: " << stderr_file);
    LOG_DEBUG(<< "Will attempt to capture stdout to: " << stdout_file);

    // Use absolute path for Executor (Sandbox2 requires absolute paths)
    auto executor = std::make_unique<sandbox2::Executor>(absoluteProcessPath, args);
    LOG_DEBUG(<< "Sandbox2 executor created");

    // Create custom Notify object to capture violations
    auto notify = std::make_unique<detail::Sandbox2LoggingNotify>();
    // Keep raw pointer to check for failures (notify is moved into sandbox but remains valid)
    detail::Sandbox2LoggingNotify* notifyPtr = notify.get();
    LOG_DEBUG(<< "Created Sandbox2 logging notify handler");

    LOG_DEBUG(<< "Creating Sandbox2 instance with policy and notify handler...");
    auto sandbox = std::make_unique<sandbox2::Sandbox2>(
        std::move(executor), std::move(policy), std::move(notify));
    LOG_DEBUG(<< "Sandbox2 instance created successfully");

    // DIAGNOSTIC MODE: Try synchronous execution first to get immediate result
    // This will give us the exit code and any violations directly
    // Set to true for diagnostics, false for production (async)
    // NOTE: pytorch_inference is a long-running process, so async mode is required
    const bool USE_SYNC_FOR_DIAGNOSTICS = false; // Use async mode for long-running processes

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
            LOG_ERROR(<< "Process violated Sandbox2 policy (reason_code: " << reason_code
                      << ")");
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
            core::CScopedLock lock(g_SandboxMapMutex);
            g_SandboxMap[childPid] = std::move(sandbox);
        }

        LOG_DEBUG(<< "Spawned sandboxed '" << absoluteProcessPath
                  << "' with PID " << childPid << " (sync mode)");
        return true;
    } else {
        // Production mode: Launch sandboxed process asynchronously
        LOG_DEBUG(<< "Launching sandboxed process asynchronously...");
        sandbox->RunAsync();
        LOG_DEBUG(<< "RunAsync() called, polling for PID and checking for failures...");

        // Poll for PID with timeout (monitor initializes asynchronously)
        // Also check for Sandbox2 failures during polling
        const int timeout_ms = 5000; // Increased timeout for better diagnostics
        const int poll_interval_us = 10000; // 10ms for less CPU usage
        int elapsed_ms = 0;

        childPid = -1;
        while (elapsed_ms < timeout_ms) {
            // Check for Sandbox2 failure first (non-blocking check)
            if (notifyPtr->hasFailed()) {
                sandbox2::Result::StatusEnum status;
                uintptr_t reason_code;
                notifyPtr->getFailureInfo(status, reason_code);
                
                std::string statusStr;
                switch (status) {
                    case sandbox2::Result::OK:
                        statusStr = "OK";
                        break;
                    case sandbox2::Result::SETUP_ERROR:
                        statusStr = "SETUP_ERROR";
                        break;
                    case sandbox2::Result::VIOLATION:
                        statusStr = "VIOLATION";
                        break;
                    case sandbox2::Result::SIGNALED:
                        statusStr = "SIGNALED";
                        break;
                    case sandbox2::Result::TIMEOUT:
                        statusStr = "TIMEOUT";
                        break;
                    case sandbox2::Result::EXTERNAL_KILL:
                        statusStr = "EXTERNAL_KILL";
                        break;
                    case sandbox2::Result::INTERNAL_ERROR:
                        statusStr = "INTERNAL_ERROR";
                        break;
                    default:
                        statusStr = "UNKNOWN(" + std::to_string(static_cast<int>(status)) + ")";
                        break;
                }
                
                LOG_ERROR(<< "Sandbox2 process failed to start with status: " << statusStr
                          << " (reason_code: " << reason_code << ")");
                LOG_ERROR(<< "Command that failed: " << full_command);
                if (status == sandbox2::Result::SETUP_ERROR) {
                    LOG_ERROR(<< "SETUP_ERROR typically indicates a policy violation or resource issue during process setup");
                }
                return false;
            }
            
            childPid = sandbox->pid();
            if (childPid > 0) {
                LOG_DEBUG(<< "Got PID from Sandbox2: " << childPid << " after "
                          << elapsed_ms << "ms");
                break;
            }
            usleep(poll_interval_us);
            elapsed_ms += 10;
        }

        // Check for failure one more time after timeout
        if (childPid <= 0) {
            if (notifyPtr->hasFailed()) {
                sandbox2::Result::StatusEnum status;
                uintptr_t reason_code;
                notifyPtr->getFailureInfo(status, reason_code);
                
                std::string statusStr;
                switch (status) {
                    case sandbox2::Result::OK:
                        statusStr = "OK";
                        break;
                    case sandbox2::Result::SETUP_ERROR:
                        statusStr = "SETUP_ERROR";
                        break;
                    case sandbox2::Result::VIOLATION:
                        statusStr = "VIOLATION";
                        break;
                    case sandbox2::Result::SIGNALED:
                        statusStr = "SIGNALED";
                        break;
                    case sandbox2::Result::TIMEOUT:
                        statusStr = "TIMEOUT";
                        break;
                    case sandbox2::Result::EXTERNAL_KILL:
                        statusStr = "EXTERNAL_KILL";
                        break;
                    case sandbox2::Result::INTERNAL_ERROR:
                        statusStr = "INTERNAL_ERROR";
                        break;
                    default:
                        statusStr = "UNKNOWN(" + std::to_string(static_cast<int>(status)) + ")";
                        break;
                }
                
                LOG_ERROR(<< "Failed to get PID from Sandbox2 after " << timeout_ms << "ms");
                LOG_ERROR(<< "Sandbox2 process failed with status: " << statusStr
                          << " (reason_code: " << reason_code << ")");
                LOG_ERROR(<< "Command that failed: " << full_command);
                if (status == sandbox2::Result::SETUP_ERROR) {
                    LOG_ERROR(<< "SETUP_ERROR typically indicates a policy violation or resource issue during process setup");
                }
            } else {
                LOG_ERROR(<< "Failed to get PID from Sandbox2 after " << timeout_ms << "ms");
                LOG_ERROR(<< "This may indicate the process failed to start or crashed immediately");
                LOG_ERROR(<< "Command that was attempted: " << full_command);
            }
            return false;
        }
    }

    // Monitor the process for a short time to detect early exits (async mode only)
    LOG_DEBUG(<< "Monitoring process " << childPid << " for early exits...");
    const int monitor_duration_ms = 3000; // Increased to catch slower exits
    const int monitor_interval_ms = 50;   // Check more frequently (every 50ms)
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
                        LOG_WARN(<< "Process " << childPid
                                 << " is in zombie state (exited but not reaped)");
                        process_still_running = false;
                        // Try to reap it immediately
                        int status = 0;
                        pid_t waited_pid = ::waitpid(childPid, &status, WNOHANG);
                        if (waited_pid == childPid) {
                            if (WIFEXITED(status)) {
                                int exit_code = WEXITSTATUS(status);
                                LOG_ERROR(<< "Process " << childPid << " exited with code "
                                          << exit_code << " (within "
                                          << monitor_elapsed_ms << "ms)");
                                LOG_ERROR(<< "Command that caused exit: " << full_command);
                            } else if (WIFSIGNALED(status)) {
                                int signal = WTERMSIG(status);
                                LOG_ERROR(<< "Process " << childPid << " was killed by signal "
                                          << signal << " (within "
                                          << monitor_elapsed_ms << "ms)");
                                LOG_ERROR(<< "Command that was running: " << full_command);
                            }
                        }
                        break;
                    }
                }
            }
        } else {
            // Process directory doesn't exist - process has exited and been reaped
            LOG_WARN(<< "Process " << childPid << " exited early (within "
                     << monitor_elapsed_ms << "ms) - already reaped");
            process_still_running = false;

            // Try to get process exit status (may fail if already reaped)
            int status = 0;
            pid_t waited_pid = ::waitpid(childPid, &status, WNOHANG);
            if (waited_pid == childPid) {
                if (WIFEXITED(status)) {
                    int exit_code = WEXITSTATUS(status);
                    LOG_ERROR(<< "Process " << childPid << " exited with code " << exit_code
                              << " (within " << monitor_elapsed_ms << "ms)");
                    LOG_ERROR(<< "Command that caused exit: " << full_command);
                } else if (WIFSIGNALED(status)) {
                    int signal = WTERMSIG(status);
                    LOG_ERROR(<< "Process " << childPid << " was killed by signal "
                              << signal << " (within " << monitor_elapsed_ms << "ms)");
                    LOG_ERROR(<< "Command that was running: " << full_command);
                }
            } else {
                LOG_ERROR(<< "Process " << childPid << " exited but waitpid returned "
                          << waited_pid << " (errno: " << errno
                          << " - already reaped by another process)");
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
        LOG_DEBUG(<< "Process " << childPid << " is still running after "
                  << monitor_duration_ms << "ms");
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
