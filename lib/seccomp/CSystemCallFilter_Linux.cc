/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "seccomp/CSystemCallFilter.h"

#include <core/CLogger.h>

#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/prctl.h>
#include <sys/syscall.h>


namespace ml {
namespace seccomp {

unsigned int upper_nr_limit = 0x3FFFFFFF;

struct sock_filter filter[] = {
       /* Load architecture from 'seccomp_data' buffer into accumulator */
       BPF_STMT(BPF_LD | BPF_W | BPF_ABS, (offsetof(struct seccomp_data, arch))),
       /* Jump to disallow if architecture is not X86_64 */
       BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 0, 5),
       /* Load the system call number into accumulator */
       BPF_STMT(BPF_LD | BPF_W | BPF_ABS, (offsetof(struct seccomp_data, nr))),
       /* [3] Check ABI - only needed for x86-64 in blacklist use
                cases.  Use BPF_JGT instead of checking against the bit
                              mask to avoid having to reload the syscall number. */
       BPF_JUMP(BPF_JMP | BPF_JGT | BPF_K, upper_nr_limit, 4, 0),
       BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_fork, 3, 0),
       BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_vfork, 2, 0),
       BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_execve, 1, 0),
       /* Allow call */
       BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
       /* Disallow call with error code EACCES */
       BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | (EACCES & SECCOMP_RET_DATA))
   };

bool canUseSeccompBpf() {
    // This call is expected to fail due to the nullptr argument
    // but the failure mode informs us if the kernal was configured
    // with CONFIG_SECCOMP_FILTER
    // http://man7.org/linux/man-pages/man2/prctl.2.html
    int result = prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, nullptr);
    int configError = errno;
    if (result != -1) {
        LOG_ERROR(<< "prctl set seccomp should have failed");
        return false;
    }

    // If the kernal is not configured with CONFIG_SECCOMP_FILTER
    // or CONFIG_SECCOMP the error is EINVAL. EFAULT indicates the
    // seccomp filters are enabled but the 3rd argument (nullptr)
    // was invalid.
    return configError == EFAULT;
}

CSystemCallFilter::CSystemCallFilter() {
    if (canUseSeccompBpf()) {
        LOG_DEBUG(<< "Seccomp BPF filters available");

        // Ensure more permissive privileges cannot be set in future.
        // This must be set before installing the filter
        prctl(PR_SET_NO_NEW_PRIVS, 1);


        struct sock_fprog prog = {
           .len = static_cast<unsigned short>(sizeof(filter) / sizeof(filter[0])),
           .filter = filter,
        };

        // Install the filter.
        // prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ...) was introduced
        // in kernal 3.5.
        // The alternative seccomp(SECCOMP_SET_MODE_FILTER, SECCOMP_FILTER_FLAG_TSYNC, ...)
        // was introduced in 3.17 and similar functionality we choose the older more
        // compatible function
        if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog)) {
        // if (seccomp(SECCOMP_SET_MODE_FILTER, SECCOMP_FILTER_FLAG_TSYNC, &prog)) {
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

