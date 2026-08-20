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
#include <sandbox/CPytorchInferenceSandboxPolicy.h>

#include <core/CLogger.h>
#include <core/CStringUtils.h>

#include <seccomp/CPytorchInferenceSyscallAllowlist.h>

#include <limits.h>
#include <linux/futex.h>
#include <sstream>
#include <sys/stat.h>

#ifdef SANDBOX2_AVAILABLE
#include <sandboxed_api/sandbox2/policybuilder.h>
#include <sandboxed_api/sandbox2/result.h>
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
#endif // SANDBOX2_AVAILABLE

namespace ml {
namespace sandbox {

SArgDirExtraction extractArgDirs(const std::vector<std::string>& args) {
    SArgDirExtraction extraction;
    for (const auto& arg : args) {
        size_t eqPos = arg.find('=');
        if (eqPos == std::string::npos || eqPos + 1 >= arg.size()) {
            continue;
        }

        if (arg[eqPos + 1] != '/') {
            extraction.m_RejectedPipeArgs.push_back(arg + " (path not absolute)");
            continue;
        }

        std::string path = arg.substr(eqPos + 1);
        size_t lastSlash = path.rfind('/');
        if (lastSlash == 0) {
            extraction.m_RejectedPipeArgs.push_back(arg + " (no mountable directory)");
            continue;
        }

        std::string dir = path.substr(0, lastSlash);
        char resolved[PATH_MAX];
        std::string canonical = ::realpath(dir.c_str(), resolved) != nullptr ? resolved : dir;
        extraction.m_ArgDirs.insert(canonical);
        struct stat dirStat;
        if (dir != canonical && ::stat(dir.c_str(), &dirStat) == 0) {
            extraction.m_ArgDirs.insert(dir);
            extraction.m_PipeDirAliasMappings.push_back(dir + "->" + canonical);
        }
    }
    return extraction;
}

namespace {

template<typename Container>
std::string joinForLog(const Container& values) {
    const std::string joined{core::CStringUtils::join(values, ",")};
    return joined.empty() ? "(none)" : joined;
}

} // namespace

#ifdef SANDBOX2_AVAILABLE

void logSandbox2SpawnContext(const std::string& absPath,
                             const std::string& binDir,
                             const std::string& libDir,
                             const SArgDirExtraction& argDirInfo) {
    std::ostringstream fixedMounts;
    fixedMounts << binDir << ',' << libDir;
    for (const auto& mountPath : seccomp::pytorch_inference::fixedSandboxMountDirectories()) {
        fixedMounts << ',' << mountPath;
    }
    fixedMounts << ",/tmp";
    for (const auto& mountPath : seccomp::pytorch_inference::fixedSandboxMountFiles()) {
        fixedMounts << ',' << mountPath;
    }

    LOG_INFO(<< "Sandbox2 pytorch_inference spawn context:"
             << " binary=" << absPath << " fixedMounts=[" << fixedMounts.str() << "]"
             << " pipeDirs=[" << joinForLog(argDirInfo.m_ArgDirs) << "]"
             << " pipeDirAliases=[" << joinForLog(argDirInfo.m_PipeDirAliasMappings) << "]"
             << " rejectedPipeArgs=[" << joinForLog(argDirInfo.m_RejectedPipeArgs) << "]"
             << " rlimit_nofile=65536 walltime_limit=disabled cpu_limit=disabled");
}

std::string sandboxPlatformArch() {
#ifdef __x86_64__
    return "x86_64";
#elif defined(__aarch64__)
    return "aarch64";
#else
    return "unknown";
#endif
}

std::string formatSandbox2Result(const sandbox2::Result& result) {
    std::ostringstream formatted;
    formatted << result.ToString() << " [status="
              << sandbox2::Result::StatusEnumToString(result.final_status())
              << ", reason_code=" << result.reason_code() << ']';
    if (result.final_status() == sandbox2::Result::VIOLATION) {
        formatted << " [syscall=" << result.reason_code()
                  << " arch=" << sandboxPlatformArch() << ']';
    }
    return formatted.str();
}

sandbox2::PolicyBuilder buildPytorchInferencePolicy(const std::string& binDir,
                                                    const std::string& libDir,
                                                    const std::set<std::string>& argDirs) {
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
        // Sandbox2 isolates pytorch_inference in a dedicated network namespace
        // with no external routes; __NR_connect is required for libtorch
        // internal socket setup but cannot reach the host network.
        .AllowSyscall(__NR_connect)
#ifdef __NR_time
        .AllowSyscall(__NR_time)
#endif
#ifdef __NR_getdents
        .AllowSyscall(__NR_getdents)
#endif
        // Filesystem mounts
        .AddDirectory(binDir, /*is_ro=*/true)
        .AddDirectory(libDir, /*is_ro=*/true);

    for (const auto& mountPath : seccomp::pytorch_inference::fixedSandboxMountDirectories()) {
        policyBuilder.AddDirectory(mountPath, /*is_ro=*/true);
    }
    for (const auto& mountPath : seccomp::pytorch_inference::fixedSandboxMountFiles()) {
        policyBuilder.AddFile(mountPath, mountPath == "/dev/null" ? false : true);
    }
    policyBuilder.AddDirectory("/tmp", /*is_ro=*/false);

    for (const auto& dir : argDirs) {
        policyBuilder.AddDirectory(dir, /*is_ro=*/false);
    }

    return policyBuilder;
}

#endif // SANDBOX2_AVAILABLE

} // namespace sandbox
} // namespace ml
