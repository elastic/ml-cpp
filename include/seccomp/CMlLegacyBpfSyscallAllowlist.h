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
#ifndef INCLUDED_ml_seccomp_CMlLegacyBpfSyscallAllowlist_h
#define INCLUDED_ml_seccomp_CMlLegacyBpfSyscallAllowlist_h

#include <cstddef>
#include <vector>

#ifdef __linux__
#include <sys/syscall.h>
#endif

namespace ml {
namespace seccomp {

#ifdef __linux__

// statx, rseq and clone3 won't be defined on a RHEL/CentOS 7 build machine,
// but might exist on the kernel we run on, so fall back to the raw numbers.
#if defined(__x86_64__)
#ifndef __NR_statx
#define ML_NR_statx 332
#else
#define ML_NR_statx __NR_statx
#endif
#ifndef __NR_rseq
#define ML_NR_rseq 334
#else
#define ML_NR_rseq __NR_rseq
#endif
#elif defined(__aarch64__)
#ifndef __NR_statx
#define ML_NR_statx 291
#else
#define ML_NR_statx __NR_statx
#endif
#ifndef __NR_rseq
#define ML_NR_rseq 293
#else
#define ML_NR_rseq __NR_rseq
#endif
#else
#error "ML seccomp syscall allowlists support only x86_64 and aarch64 Linux builds"
#endif
#ifndef __NR_clone3
#define ML_NR_clone3 435
#else
#define ML_NR_clone3 __NR_clone3
#endif

//! Syscalls permitted by the legacy in-process BPF filter
//! (CSystemCallFilter_Linux.cc) for every process that installs it, currently
//! shared by pytorch_inference, autodetect, categorize, normalize and
//! data_frame_analyzer. This is the single machine-readable declaration that
//! the applied BPF program is generated from: CSystemCallFilter_Linux.cc
//! contains no independent syscall list and no manually maintained jump
//! offsets. A future Sandbox2 policy is expected to consume the same
//! declaration for its explicit grants, so both mechanisms stay in sync.
//!
//! Carry-forward note: PR #2873 fixed several pytorch_inference/libtorch
//! compatibility gaps the hard way, and this declaration is a rewrite from
//! scratch rather than a copy of that work, so it deliberately keeps two of
//! them. ML_NR_clone3 (see 57f00ed1b) and __NR_prlimit64 (see 03b1ee4a) are
//! carried into this shared declaration so a future Sandbox2 policy
//! inherits them automatically instead of rediscovering them the same way;
//! CSeccompFilterBuilderTest.cc asserts both stay present. The x86_64
//! legacy filesystem syscalls below (see ec7d3ed85) were already part of
//! this filter's syscall set prior to this declaration and remain
//! unchanged. PR #2873's futex-op broadening (see d9a856d5f) and CI
//! link-order/test-bundle packaging fixes (see 730933db, f8b0a534) apply to
//! the Sandbox2 policy and its Buildkite pipeline respectively, not to this
//! file — carry those forward when that code is written instead of
//! rediscovering them.
inline constexpr int kLegacyBpfAllowedSyscalls[] {
#if defined(__x86_64__)
    __NR_access, __NR_open, __NR_dup2, __NR_unlink, __NR_stat, __NR_lstat,
        __NR_time, __NR_readlink, __NR_getdents, // for forecast temp storage
        __NR_rmdir,                              // for forecast temp storage
        __NR_mkdir,                              // for forecast temp storage
        __NR_mknod,
#elif defined(__aarch64__)
    __NR_faccessat,
#endif
        __NR_fcntl, // for fdopendir
        __NR_getrusage,
        __NR_getpid,    // for pthread_kill
        ML_NR_statx,    // for create_directories
        __NR_getrandom, // for unique_path
        __NR_mknodat, __NR_newfstatat, __NR_readlinkat, __NR_dup3,
        __NR_getpriority, // for nice
        __NR_setpriority, // for nice
        __NR_read, __NR_write, __NR_writev, __NR_lseek, __NR_clock_gettime,
        __NR_gettimeofday, __NR_fstat, __NR_close, __NR_connect, ML_NR_clone3,
        __NR_clone, __NR_statfs,
        __NR_mkdirat,      // for forecast temp storage
        __NR_unlinkat,     // for forecast temp storage
        __NR_getdents64,   // for forecast temp storage
        __NR_openat,       // for forecast temp storage
        __NR_tgkill,       // for the crash handler
        __NR_rt_sigaction, // for the crash handler
        __NR_rt_sigreturn,
        __NR_rt_sigprocmask, // for recent pthread_create
        ML_NR_rseq,          // for recent pthread_create
        __NR_futex, __NR_madvise, __NR_nanosleep, __NR_set_robust_list,
        __NR_mprotect, // for malloc arenas and pthread stacks
        __NR_mremap,   // for malloc arenas
        __NR_munmap,   // for malloc arenas
        __NR_mmap,     // for malloc arenas
        __NR_getuid, __NR_exit_group, __NR_brk, __NR_exit,
        __NR_prlimit64, // libtorch/Sandbox2-monitor query rlimits under load (03b1ee4a)
};

static_assert(std::size(kLegacyBpfAllowedSyscalls) <= 255,
              "legacy BPF allowlist exceeds classic BPF jt (8-bit)");

inline std::vector<int> legacyBpfAllowedSyscalls() {
    return {kLegacyBpfAllowedSyscalls,
            kLegacyBpfAllowedSyscalls + std::size(kLegacyBpfAllowedSyscalls)};
}

//! Syscalls that must be explicitly granted (via AllowSyscall()) in
//! buildPytorchInferenceFilesystemPolicy(), in addition to the
//! legacyBpfAllowedSyscalls() loop there. Sandbox2's namespace and
//! threading setup exercises syscalls (scheduling, epoll, pipes,
//! directory/file management for forecast temp storage) that the simpler
//! legacy in-process BPF filter never needed a grant for. Carried forward
//! from PR #2873; see CSeccompFilterBuilderTest.cc.
inline std::vector<int> sandbox2ExplicitSyscalls() {
    std::vector<int> syscalls{
        __NR_sched_yield,
        __NR_sched_getaffinity,
        __NR_sched_setaffinity,
        __NR_sched_getparam,
        __NR_sched_getscheduler,
        __NR_clone,
        ML_NR_clone3,
        __NR_set_tid_address,
        __NR_set_robust_list,
        ML_NR_rseq,
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
        __NR_mknodat, // mkfifo() for named pipes (CNamedPipeFactory)
#ifdef __NR_mknod
        __NR_mknod, // mkfifo() on x86_64 glibc paths
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
        __NR_connect, // AF_UNIX connect() while opening named pipes for I/O
#ifdef __NR_time
        __NR_time,
#endif
#ifdef __NR_getdents
        __NR_getdents,
#endif
    };
    return syscalls;
}

#endif // __linux__

} // namespace seccomp
} // namespace ml

#endif // INCLUDED_ml_seccomp_CMlLegacyBpfSyscallAllowlist_h
