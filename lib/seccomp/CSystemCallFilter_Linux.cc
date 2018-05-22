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


namespace ml {
namespace seccomp {

// The old x32 ABI always has bit 30 set in the sys call numbers.
// The x64 architecture should fail these calls
unsigned int UPPER_NR_LIMIT = 0x3FFFFFFF;

// Offset to the nr field in struct seccomp_data
unsigned int SECCOMP_DATA_NR_OFFSET   = 0x00;
// Offset to the arch field in struct seccomp_data
unsigned int SECCOMP_DATA_ARCH_OFFSET = 0x04;

// Copied from seccomp.h
// seccomp.h cannot be included as it was added in Linux kernel 3.17
// and this must build on older versions.
#define SECCOMP_MODE_FILTER 2
#define SECCOMP_RET_ERRNO   0x00050000U
#define SECCOMP_RET_ALLOW   0x7fff0000U
#define SECCOMP_RET_DATA    0x0000ffffU

// Added in Linux 3.5
#ifndef PR_SET_NO_NEW_PRIVS
#define PR_SET_NO_NEW_PRIVS 38
#endif

struct sock_filter filter[] = {
       /* Load architecture from 'seccomp_data' buffer into accumulator */
       BPF_STMT(BPF_LD | BPF_W | BPF_ABS, SECCOMP_DATA_ARCH_OFFSET),
       /* Jump to disallow if architecture is not X86_64 */
       BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 0, 5),
       /* Load the system call number into accumulator */
       BPF_STMT(BPF_LD | BPF_W | BPF_ABS, SECCOMP_DATA_NR_OFFSET),
       /* Only applies to X86_64 arch. Fail calls for the x32 ABI  */
       BPF_JUMP(BPF_JMP | BPF_JGT | BPF_K, UPPER_NR_LIMIT, 4, 0),
       BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_fork, 3, 0),
       BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_vfork, 2, 0),
       BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_execve, 1, 0),
       /* Allow call */
       BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
       /* Disallow call with error code EACCES */
       BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | (EACCES & SECCOMP_RET_DATA))
   };

bool canUseSeccompBpf() {
    // This call is expected to fail due to the nullptr argument
    // but the failure mode informs us if the kernel was configured
    // with CONFIG_SECCOMP_FILTER
    // http://man7.org/linux/man-pages/man2/prctl.2.html
    int result = prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, nullptr);
    int configError = errno;
    if (result != -1) {
        LOG_ERROR(<< "prctl set seccomp should have failed");
        return false;
    }

    // If the kernel is not configured with CONFIG_SECCOMP_FILTER
    // or CONFIG_SECCOMP the error is EINVAL. EFAULT indicates the
    // seccomp filters are enabled but the 3rd argument (nullptr)
    // was invalid.
    return configError == EFAULT;
}

CSystemCallFilter::CSystemCallFilter() {
    if (canUseSeccompBpf()) {
        LOG_DEBUG(<< "Seccomp BPF filters available");

        // Ensure more permissive privileges cannot be set in future.
        // This must be set before installing the filter.
        // PR_SET_NO_NEW_PRIVS was aded in kernel 3.5
        prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);

        struct sock_fprog prog = {
           .len = static_cast<unsigned short>(sizeof(filter) / sizeof(filter[0])),
           .filter = filter,
        };

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
            LOG_ERROR("Unable to install Seccomp BPF");
        } else {
            LOG_DEBUG("Seccomp BPF installed");
        }

    } else {
        LOG_DEBUG(<< "Seccomp BPF not available");
    }
}
}
}

