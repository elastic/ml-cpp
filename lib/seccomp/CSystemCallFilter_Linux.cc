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

/*
 * NOTE: This seccomp filter is being gradually replaced by Sandbox2 policies
 * for processes that are spawned via CDetachedProcessSpawner. The allowed
 * syscall set lives in CPytorchInferenceSyscallAllowlist.h, the single
 * machine-readable declaration this filter is generated from; a future
 * Sandbox2 policy is expected to consume the same declaration for its
 * explicit grants.
 */
#include <seccomp/CSystemCallFilter.h>

#include <core/CLogger.h>

#include <seccomp/CPytorchInferenceSyscallAllowlist.h>
#include <seccomp/CSeccompFilterBuilder.h>

#include <cerrno>
#include <cstddef>
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
}

std::vector<sock_filter> buildSyscallAllowlistProgram(const std::vector<int>& allowedSyscalls) {
    const auto numSyscalls = static_cast<std::uint32_t>(allowedSyscalls.size());

    std::vector<sock_filter> program;
    program.reserve(numSyscalls + 6);

    // Reject non-native ABIs before matching syscall numbers.  Without this,
    // an x86_64 process can issue int 0x80 (i386) and hit number collisions —
    // e.g. i386 socketcall (102) matches an allowlisted x86_64 syscall with
    // the same number. Hardening in response to a privately reported ML
    // seccomp-bypass finding. This prefix is self-contained (immediate RET
    // on mismatch), so it never affects the jump offsets below.
    program.push_back(
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, arch)));
#ifdef __x86_64__
    program.push_back(BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 1, 0));
#elif defined(__aarch64__)
    program.push_back(BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_AARCH64, 1, 0));
#else
#error Unsupported hardware architecture
#endif
    program.push_back(BPF_STMT(BPF_RET | BPF_K,
                               SECCOMP_RET_ERRNO | (EACCES & SECCOMP_RET_DATA)));

    // Load the system call number into accumulator
    program.push_back(BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)));

#ifdef __x86_64__
    // Jump to the deny row (immediately after the last syscall row below,
    // i.e. numSyscalls rows ahead) for calls using the x32 ABI, without
    // checking any allowlisted syscall.
    program.push_back(BPF_JUMP(BPF_JMP | BPF_JGT | BPF_K, UPPER_NR_LIMIT,
                               static_cast<std::uint8_t>(numSyscalls), 0));
#endif

    // Every syscall row jumps to the terminal SECCOMP_RET_ALLOW row on match.
    // The jump distance is derived from the row's own index and the total
    // count, so adding, removing or reordering an entry in allowedSyscalls
    // never requires touching any other row.
    for (std::uint32_t i = 0; i < numSyscalls; ++i) {
        const auto jumpToAllow = static_cast<std::uint8_t>(numSyscalls - i);
        program.push_back(BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K,
                                   static_cast<std::uint32_t>(allowedSyscalls[i]),
                                   jumpToAllow, 0));
    }

    // Disallow call with error code EACCES
    program.push_back(BPF_STMT(BPF_RET | BPF_K,
                               SECCOMP_RET_ERRNO | (EACCES & SECCOMP_RET_DATA)));
    // Allow call
    program.push_back(BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW));

    return program;
}

namespace {

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

ESystemCallFilterInstallOutcome CSystemCallFilter::installSystemCallFilter() {
    if (canUseSeccompBpf() == false) {
        LOG_DEBUG(<< "Seccomp BPF not available");
        return ESystemCallFilterInstallOutcome::E_MechanismUnavailable;
    }
    LOG_DEBUG(<< "Seccomp BPF filters available");

    // Ensure more permissive privileges cannot be set in future.
    // This must be set before installing the filter.
    // PR_SET_NO_NEW_PRIVS was added in kernel 3.5
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)) {
        LOG_ERROR(<< "prctl PR_SET_NO_NEW_PRIVS failed: " << std::strerror(errno));
        return ESystemCallFilterInstallOutcome::E_PrivilegeRestrictionFailed;
    }

    const std::vector<sock_filter> program{
        buildSyscallAllowlistProgram(pytorch_inference::legacyBpfAllowedSyscalls())};

    struct sock_fprog prog = {.len = static_cast<unsigned short>(program.size()),
                              .filter = const_cast<sock_filter*>(program.data())};

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
        return ESystemCallFilterInstallOutcome::E_FilterInstallFailed;
    }

    LOG_DEBUG(<< "Seccomp BPF installed");
    LOG_INFO(<< "ml.seccomp.installed");
    return ESystemCallFilterInstallOutcome::E_Installed;
}
}
}
