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
#include <seccomp/CSystemCallFilter.h>

#include <core/CLogger.h>

#include <cerrno>
#include <cstdint>
#include <cstring>

#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/prctl.h>
#include <sys/syscall.h>

namespace ml {
namespace seccomp {

namespace {
// The old x32 ABI always has bit 30 set in the sys call numbers.
// The x64 ABI should fail these calls
const std::uint32_t UPPER_NR_LIMIT = 0x3FFFFFFF;

// Offset to the nr field in struct seccomp_data
const std::uint32_t SECCOMP_DATA_NR_OFFSET = 0x00;

const struct sock_filter FILTER[] = {
    // Load the system call number into accumulator
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS, SECCOMP_DATA_NR_OFFSET),

#ifdef __x86_64__
// The statx, rseq and clone3 syscalls won't be defined on a RHEL/CentOS 7 build
// machine, but might exist on the kernel we run on
#ifndef __NR_statx
#define __NR_statx 332
#endif
#ifndef __NR_rseq
#define __NR_rseq 334
#endif
#ifndef __NR_clone3
#define __NR_clone3 435
#endif
    // Only applies to x86_64 arch. Jump to disallow for calls using the x32 ABI
    BPF_JUMP(BPF_JMP | BPF_JGT | BPF_K, UPPER_NR_LIMIT, 56, 0),
    // If any sys call filters are added or removed then the jump
    // destination for each statement including the one above must
    // be updated accordingly

    // Allowed architecture-specific sys calls, jump to return allow on match
    // Some of these are not used in latest glibc, and not supported in Linux
    // kernels for recent architectures, but in a few cases different sys calls
    // are used on different architectures
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_access, 56, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_open, 55, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_dup2, 54, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_unlink, 53, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_stat, 52, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_lstat, 51, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_time, 50, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_readlink, 49, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_getdents, 48, 0), // for forecast temp storage
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_rmdir, 47, 0), // for forecast temp storage
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_mkdir, 46, 0), // for forecast temp storage
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_mknod, 45, 0),
#elif defined(__aarch64__)
// The statx, rseq and clone3 syscalls won't be defined on a RHEL/CentOS 7 build
// machine, but might exist on the kernel we run on
#ifndef __NR_statx
#define __NR_statx 291
#endif
#ifndef __NR_rseq
#define __NR_rseq 293
#endif
#ifndef __NR_clone3
#define __NR_clone3 435
#endif
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_faccessat, 45, 0),
#else
#error Unsupported hardware architecture
#endif

    // Allowed sys calls for all architectures, jump to return allow on match
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_fcntl, 44, 0), // for fdopendir
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_getrusage, 43, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_getpid, 42, 0), // for pthread_kill
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_statx, 41, 0), // for create_directories
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_getrandom, 40, 0), // for unique_path
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_mknodat, 39, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_newfstatat, 38, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_readlinkat, 37, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_dup3, 36, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_getpriority, 35, 0), // for nice
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_setpriority, 34, 0), // for nice
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_read, 33, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_write, 32, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_writev, 31, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_lseek, 30, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_clock_gettime, 29, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_gettimeofday, 28, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_fstat, 27, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_close, 26, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_connect, 25, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_clone3, 24, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_clone, 23, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_statfs, 22, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_mkdirat, 21, 0), // for forecast temp storage
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_unlinkat, 20, 0), // for forecast temp storage
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_getdents64, 19, 0), // for forecast temp storage
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_openat, 18, 0), // for forecast temp storage
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_tgkill, 17, 0), // for the crash handler
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_rt_sigaction, 16, 0), // for the crash handler
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_rt_sigreturn, 15, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_rt_sigprocmask, 14, 0), // for recent pthread_create
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_rseq, 13, 0), // for recent pthread_create
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_futex, 12, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_madvise, 11, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_nanosleep, 10, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_set_robust_list, 9, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_mprotect, 8, 0), // for malloc arenas and pthread stacks
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_mremap, 7, 0), // for malloc arenas
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_munmap, 6, 0), // for malloc arenas
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_mmap, 5, 0),   // for malloc arenas
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_getuid, 4, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_exit_group, 3, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_brk, 2, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_exit, 1, 0),
    // Disallow call with error code EACCES
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | (EACCES & SECCOMP_RET_DATA)),
    // Allow call
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW)};

bool canUseSeccompBpf() {
    // This call is expected to fail due to the nullptr argument
    // but the failure mode informs us if the kernel was configured
    // with CONFIG_SECCOMP_FILTER
    // http://man7.org/linux/man-pages/man2/prctl.2.html
    int result = prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, nullptr);
    int configError = errno;
    if (result != -1) {
        LOG_ERROR(<< "prctl set seccomp with null argument should have failed");
        return false;
    }

    // If the kernel is not configured with CONFIG_SECCOMP_FILTER
    // or CONFIG_SECCOMP the error is EINVAL. EFAULT indicates the
    // seccomp filters are enabled but the 3rd argument (nullptr)
    // was invalid.
    return configError == EFAULT;
}
}

void CSystemCallFilter::installSystemCallFilter() {
    if (canUseSeccompBpf()) {
        LOG_DEBUG(<< "Seccomp BPF filters available");

        // Ensure more permissive privileges cannot be set in future.
        // This must be set before installing the filter.
        // PR_SET_NO_NEW_PRIVS was aded in kernel 3.5
        if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)) {
            LOG_ERROR(<< "prctl PR_SET_NO_NEW_PRIVS failed: " << std::strerror(errno));
            return;
        }

        struct sock_fprog prog = {
            .len = static_cast<unsigned short>(sizeof(FILTER) / sizeof(FILTER[0])),
            .filter = const_cast<sock_filter*>(FILTER)};

        // Install the filter.
        // prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, filter) was introduced
        // in kernel 3.5. This is functionally equivalent to
        // seccomp(SECCOMP_SET_MODE_FILTER, 0, filter) which was added in
        // kernel 3.17. We choose the older more compatible function.
        // Note this precludes the use of calling seccomp() with the
        // SECCOMP_FILTER_FLAG_TSYNC which is acceptable if the filter
        // is installed by the main thread before any other threads are
        // spawned.
        if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog)) {
            LOG_ERROR(<< "Unable to install Seccomp BPF: " << std::strerror(errno));
        } else {
            LOG_DEBUG(<< "Seccomp BPF installed");
        }

    } else {
        LOG_DEBUG(<< "Seccomp BPF not available");
    }
}
}
}
