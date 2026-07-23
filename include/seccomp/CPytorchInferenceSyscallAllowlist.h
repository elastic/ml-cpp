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
#ifndef INCLUDED_ml_seccomp_CPytorchInferenceSyscallAllowlist_h
#define INCLUDED_ml_seccomp_CPytorchInferenceSyscallAllowlist_h

#include <set>
#include <string>
#include <vector>

namespace ml {
namespace seccomp {
namespace pytorch_inference {

//! Paths bind-mounted read-only for every sandboxed pytorch_inference process,
//! excluding the per-install binDir and libDir. Keep in sync with
//! buildPytorchInferencePolicy() in CDetachedProcessSpawner_Linux.cc.
inline const std::vector<std::string>& fixedSandboxMountDirectories() {
    static const std::vector<std::string> PATHS{"/lib", "/lib64", "/usr/lib",
                                                "/usr/lib64", "/etc", "/proc", "/sys"};
    return PATHS;
}

//! Device files bind-mounted for every sandboxed pytorch_inference process.
inline const std::vector<std::string>& fixedSandboxMountFiles() {
    static const std::vector<std::string> PATHS{"/dev/null", "/dev/urandom", "/dev/random"};
    return PATHS;
}

#ifdef __linux__

#include <sys/syscall.h>

namespace detail {

#ifndef ML_NR_clone3
#ifdef __NR_clone3
#define ML_NR_clone3 __NR_clone3
#else
#define ML_NR_clone3 435
#endif
#endif

inline void appendCommonLegacySyscalls(std::vector<int>& syscalls) {
    syscalls.insert(syscalls.end(),
                    {__NR_fcntl,
                     __NR_getrusage,
                     __NR_getpid,
#ifdef __NR_statx
                     __NR_statx,
#endif
                     __NR_getrandom,
                     __NR_mknodat,
                     __NR_newfstatat,
                     __NR_readlinkat,
                     __NR_dup3,
                     __NR_dup,
                     __NR_getpriority,
                     __NR_setpriority,
                     __NR_read,
                     __NR_write,
                     __NR_writev,
                     __NR_lseek,
                     __NR_clock_gettime,
                     __NR_gettimeofday,
                     __NR_fstat,
                     __NR_close,
                     __NR_connect,
                     ML_NR_clone3,
                     __NR_clone,
                     __NR_statfs,
                     __NR_mkdirat,
                     __NR_unlinkat,
                     __NR_getdents64,
                     __NR_openat,
                     __NR_tgkill,
                     __NR_rt_sigaction,
                     __NR_rt_sigreturn,
                     __NR_rt_sigprocmask,
#ifdef __NR_rseq
                     __NR_rseq,
#endif
                     __NR_futex,
                     __NR_madvise,
                     __NR_nanosleep,
                     __NR_set_robust_list,
                     __NR_mprotect,
                     __NR_mremap,
                     __NR_munmap,
                     __NR_mmap,
                     __NR_getuid,
                     __NR_exit_group,
                     __NR_brk,
                     __NR_exit});
}

inline void appendArchSpecificLegacySyscalls(std::vector<int>& syscalls) {
#ifdef __x86_64__
    syscalls.insert(syscalls.end(),
                    {__NR_access,
                     __NR_open,
                     __NR_dup2,
                     __NR_unlink,
                     __NR_stat,
                     __NR_lstat,
#ifdef __NR_time
                     __NR_time,
#endif
                     __NR_readlink,
#ifdef __NR_getdents
                     __NR_getdents,
#endif
                     __NR_rmdir,
                     __NR_mkdir,
#ifdef __NR_mknod
                     __NR_mknod,
#endif
                     __NR_faccessat});
#elif defined(__aarch64__)
    syscalls.push_back(__NR_faccessat);
#endif
}

inline void appendSandbox2ExplicitSyscalls(std::vector<int>& syscalls) {
    syscalls.insert(syscalls.end(),
                    {__NR_sched_yield,
                     __NR_sched_getaffinity,
                     __NR_sched_setaffinity,
                     __NR_sched_getparam,
                     __NR_sched_getscheduler,
                     __NR_clone,
                     ML_NR_clone3,
                     __NR_set_tid_address,
                     __NR_set_robust_list,
#ifdef __NR_rseq
                     __NR_rseq,
#endif
                     __NR_clock_gettime,
                     __NR_clock_getres,
                     __NR_clock_nanosleep,
                     __NR_gettimeofday,
                     __NR_nanosleep,
                     __NR_times,
                     __NR_epoll_create1,
                     __NR_epoll_ctl,
                     __NR_epoll_pwait,
                     __NR_eventfd2,
                     __NR_ppoll,
                     __NR_pselect6,
                     __NR_ioctl,
                     __NR_fcntl,
                     __NR_pipe2,
                     __NR_dup,
                     __NR_dup3,
                     __NR_lseek,
                     __NR_ftruncate,
                     __NR_readlinkat,
                     __NR_faccessat,
                     __NR_getdents64,
                     __NR_getcwd,
                     __NR_unlinkat,
                     __NR_renameat,
                     __NR_mkdirat,
                     __NR_mknodat,
#ifdef __NR_mknod
                     __NR_mknod,
#endif
#ifdef __NR_unlink
                     __NR_unlink,
#endif
#ifdef __NR_rmdir
                     __NR_rmdir,
#endif
#ifdef __NR_mkdir
                     __NR_mkdir,
#endif
#ifdef __NR_rename
                     __NR_rename,
#endif
#ifdef __NR_readlink
                     __NR_readlink,
#endif
#ifdef __NR_access
                     __NR_access,
#endif
#ifdef __NR_dup2
                     __NR_dup2,
#endif
                     __NR_mprotect,
                     __NR_mremap,
                     __NR_madvise,
                     __NR_munmap,
                     __NR_brk,
                     __NR_sysinfo,
                     __NR_uname,
                     __NR_prlimit64,
                     __NR_getrusage,
                     __NR_prctl,
#ifdef __NR_arch_prctl
                     __NR_arch_prctl,
#endif
                     __NR_wait4,
                     __NR_exit,
                     __NR_getuid,
                     __NR_getgid,
                     __NR_geteuid,
                     __NR_getegid,
                     __NR_setpriority,
                     __NR_getpriority,
                     __NR_tgkill,
                     __NR_statfs,
                     __NR_connect,
#ifdef __NR_time
                     __NR_time,
#endif
#ifdef __NR_getdents
                     __NR_getdents,
#endif
                     });
}

inline void appendSandbox2HelperCoveredSyscalls(std::vector<int>& syscalls) {
    syscalls.insert(syscalls.end(),
                    {__NR_read,
                     __NR_write,
                     __NR_writev,
                     __NR_openat,
#ifdef __NR_open
                     __NR_open,
#endif
#ifdef __NR_stat
                     __NR_stat,
#endif
#ifdef __NR_lstat
                     __NR_lstat,
#endif
                     __NR_close,
                     __NR_mmap,
                     __NR_munmap,
                     __NR_mprotect,
                     __NR_mremap,
                     __NR_madvise,
                     __NR_brk,
                     __NR_futex,
                     __NR_clone,
                     ML_NR_clone3,
                     __NR_set_robust_list,
#ifdef __NR_rseq
                     __NR_rseq,
#endif
                     __NR_rt_sigaction,
                     __NR_rt_sigreturn,
                     __NR_rt_sigprocmask,
                     __NR_getpid,
                     __NR_getrandom,
                     __NR_exit,
                     __NR_exit_group,
                     __NR_newfstatat,
                     __NR_fstat,
                     __NR_getuid,
                     __NR_getgid,
                     __NR_geteuid,
                     __NR_getegid,
#ifdef __NR_statx
                     __NR_statx,
#endif
                     });
}

} // namespace detail

//! Syscalls permitted by the legacy BPF filter in CSystemCallFilter_Linux.cc.
inline std::vector<int> legacyBpfAllowedSyscalls() {
    std::vector<int> syscalls;
    detail::appendArchSpecificLegacySyscalls(syscalls);
    detail::appendCommonLegacySyscalls(syscalls);
    return syscalls;
}

//! Syscalls explicitly granted via AllowSyscall() in buildPytorchInferencePolicy().
inline std::vector<int> sandbox2ExplicitSyscalls() {
    std::vector<int> syscalls;
    detail::appendSandbox2ExplicitSyscalls(syscalls);
    return syscalls;
}

//! Syscalls covered by Sandbox2 PolicyBuilder helpers that are also listed in
//! the legacy BPF filter.
inline std::vector<int> sandbox2HelperCoveredSyscalls() {
    std::vector<int> syscalls;
    detail::appendSandbox2HelperCoveredSyscalls(syscalls);
    return syscalls;
}

//! Returns true when every legacy BPF syscall is allowed by the Sandbox2 policy.
inline bool sandbox2AllowsAllLegacySyscalls() {
    std::set<int> allowed;
    for (int nr : sandbox2ExplicitSyscalls()) {
        allowed.insert(nr);
    }
    for (int nr : sandbox2HelperCoveredSyscalls()) {
        allowed.insert(nr);
    }

    for (int nr : legacyBpfAllowedSyscalls()) {
        if (allowed.find(nr) == allowed.end()) {
            return false;
        }
    }
    return true;
}

#endif // __linux__

} // namespace pytorch_inference
} // namespace seccomp
} // namespace ml

#endif // INCLUDED_ml_seccomp_CPytorchInferenceSyscallAllowlist_h
