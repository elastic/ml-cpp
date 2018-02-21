/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CCrashHandler.h>

#include <cstring>

#include <dlfcn.h>
#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>

namespace ml
{
namespace core
{

//! get useful information for debugging
void crashHandler(int sig, siginfo_t *info, void *context)
{
    // reset all handlers
    signal(SIGILL, SIG_DFL);
    signal(SIGABRT, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    signal(SIGFPE, SIG_DFL);
    signal(SIGSEGV, SIG_DFL);
    signal(SIGSTKFLT, SIG_DFL);

    // note: Not using backtrace(...) as it does only contain information for the main thread,
    // but the segfault could have happened on a different thread.
    ucontext_t *uContext = static_cast<ucontext_t*>(context);
    void *errorAddress = 0;

// various platform specifics, although we do not need all of them
#ifdef REG_RIP // x86_64
    errorAddress = reinterpret_cast<void*>(uContext->uc_mcontext.gregs[REG_RIP]);
#elif defined(REG_EIP) // x86_32
    errorAddress = reinterpret_cast<void*>(uContext->uc_mcontext.gregs[REG_EIP]);
#elif defined(__arm__)
    errorAddress = reinterpret_cast<void*>(uContext->uc_mcontext.arm_pc);
#elif defined(__aarch64__)
    errorAddress = reinterpret_cast<void*>(uContext->uc_mcontext.pc);
#elif defined(__ppc__) || defined(__powerpc) || defined(__powerpc__) || defined(__POWERPC__)
    errorAddress = reinterpret_cast<void*>(uContext->uc_mcontext.regs->nip);
#else
#   error ":/ sorry, ain't know no nothing none not of your architecture!"
#endif

    Dl_info symbolInfo;
    dladdr(errorAddress, &symbolInfo);

    fprintf(stderr, "si_signo %d, si_code: %d, si_errno: %d, address: %p, library: %s, base: %p, normalized address: %p\n",
            info->si_signo,
            info->si_code,
            info->si_errno,
            errorAddress, symbolInfo.dli_fname, symbolInfo.dli_fbase,
            reinterpret_cast<void*>(
            reinterpret_cast<intptr_t>(errorAddress) -
            reinterpret_cast<intptr_t>(symbolInfo.dli_fbase)));

    // Still generate a core dump,
    // see http://www.alexonlinux.com/how-to-handle-sigsegv-but-also-generate-core-dump
    raise(sig);
}

void CCrashHandler::installCrashHandler(void)
{
    struct sigaction actionOnCrash;
    std::memset(&actionOnCrash, 0, sizeof actionOnCrash);
    actionOnCrash.sa_flags = (SA_SIGINFO | SA_ONSTACK | SA_NODEFER);
    sigemptyset(&actionOnCrash.sa_mask);
    actionOnCrash.sa_sigaction = &crashHandler;

    sigaction(SIGILL, &actionOnCrash, nullptr);
    sigaction(SIGABRT, &actionOnCrash, nullptr);
    sigaction(SIGBUS, &actionOnCrash, nullptr);
    sigaction(SIGFPE, &actionOnCrash, nullptr);
    sigaction(SIGSEGV, &actionOnCrash, nullptr);
    sigaction(SIGSTKFLT, &actionOnCrash, nullptr);
}

}
}

