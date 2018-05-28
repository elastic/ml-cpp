/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "seccomp/CSystemCallFilter.h"

#include <core/CLogger.h>

#include <linux/audit.h>
#include <linux/filter.h>
#include <sys/prctl.h>
#include <sys/syscall.h>

#include <cerrno>
#include <cstdint>
#include <cstring>

namespace ml {
namespace seccomp {

namespace {
// The old x32 ABI always has bit 30 set in the sys call numbers.
// The x64 architecture should fail these calls
const std::uint32_t UPPER_NR_LIMIT = 0x3FFFFFFF;

// Offset to the nr field in struct seccomp_data
const std::uint32_t SECCOMP_DATA_NR_OFFSET = 0x00;
// Offset to the arch field in struct seccomp_data
const std::uint32_t SECCOMP_DATA_ARCH_OFFSET = 0x04;

// Copied from seccomp.h
// seccomp.h cannot be included as it was added in Linux kernel 3.17
// and this must build on older versions.
// TODO: remove on the minumum build kernel version supports seccomp
#define SECCOMP_MODE_FILTER 2
#define SECCOMP_RET_ERRNO 0x00050000U
#define SECCOMP_RET_ALLOW 0x7fff0000U
#define SECCOMP_RET_DATA 0x0000ffffU

// Added in Linux 3.5
#ifndef PR_SET_NO_NEW_PRIVS
#define PR_SET_NO_NEW_PRIVS 38
#endif

const struct sock_filter FILTER[] = {
    // Load architecture from 'seccomp_data' buffer into accumulator
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS, SECCOMP_DATA_ARCH_OFFSET),
    // Jump to disallow if architecture is not X86_64
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 0, 5),
    // Load the system call number into accumulator
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS, SECCOMP_DATA_NR_OFFSET),
    // Only applies to X86_64 arch. Jump to disallow for calls using the x32 ABI
    BPF_JUMP(BPF_JMP | BPF_JGT | BPF_K, UPPER_NR_LIMIT, 34, 0),
    // Allowed sys calls, jump to return allow on match
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_read, 34, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_write, 33, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_writev, 32, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_lseek, 31, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_lstat, 30, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_readlink, 29, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_stat, 28, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_fstat, 27, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_open, 26, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_close, 25, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_connect, 24, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_clone, 23, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_statfs, 22, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_dup2, 21, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_rmdir, 20, 0), // for forecast temp storage
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_getdents, 19, 0), // for forecast temp storage
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_openat, 18, 0), // for forecast temp storage
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_tgkill, 17, 0), // for the crash handler
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_rt_sigaction, 16, 0), // for the crash handler
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_rt_sigreturn, 15, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_futex, 14, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_madvise, 13, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_unlink, 12, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_mknod, 11, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_nanosleep, 10, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_set_robust_list, 9, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_mprotect, 8, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_munmap, 7, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_mmap, 6, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_getuid, 5, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_exit_group, 4, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_access, 3, 0),
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
