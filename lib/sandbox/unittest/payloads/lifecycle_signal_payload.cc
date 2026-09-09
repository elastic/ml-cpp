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

// Deliberately dependency-free sandboxee for
// CSandboxedProcessSpawnerLifecycleTest_Linux (Task 4). Unlike
// sandbox_smoke_payload.cc (exits immediately) this payload stays alive
// indefinitely so the lifecycle test can drive CSandboxedProcessSpawner::
// terminateChild() against a genuinely live child and distinguish its two
// termination mechanisms by observable effect:
//
//  - pidfd_send_signal(SIGTERM) (the E_Acquired branch) is a *request*: this
//    payload installs a SIGTERM handler that does nothing and returns, so
//    the process stays alive and the test can observe "still running".
//  - Sandbox2::Kill() (the E_KernelUnsupported branch) hard-codes SIGKILL,
//    which cannot be caught or ignored, so the process actually exits.
//
// Only syscalls in seccomp::pytorch_inference::legacyBpfAllowedSyscalls()
// are available under the real spawn() policy - notably __NR_pause is NOT
// in that allowlist, so this cannot simply call pause() in a loop. Blocking
// on FUTEX_WAIT against a private, never-signalled word uses only
// __NR_futex (allowed) and is interrupted (EINTR) by the caught SIGTERM,
// after which the loop just re-enters the wait; the only way to actually
// terminate this process is an uncatchable signal (SIGKILL).
//
// No ml-cpp library dependencies, no policy of its own - same rationale as
// sandbox_smoke_payload.cc and ml_sandbox_probe.cc.

#include <atomic>
#include <csignal>
#include <cstdint>
#include <sys/syscall.h>
#include <unistd.h>

namespace {

std::atomic<int> gFutexWord{0};

void ignoreSigterm(int /* signum */) {
    // Deliberately empty: catching (rather than ignoring via SIG_IGN) means
    // the blocking futex(2) call below observes EINTR and this handler
    // itself is proof the process is still alive and processing signals
    // normally - SIG_IGN would make that indistinguishable from "never
    // received the signal at all".
}

} // namespace

int main() {
    struct sigaction sa {};
    sa.sa_handler = ignoreSigterm;
    ::sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    ::sigaction(SIGTERM, &sa, nullptr);

    for (;;) {
        // FUTEX_WAIT (0): block while *reinterpret_cast<int*>(&gFutexWord) ==
        // 0, which it always is - nothing ever calls FUTEX_WAKE on this
        // word. Returns on a spurious wake, a real wake (never happens
        // here), or EINTR from the caught SIGTERM; any of those just loops
        // back into another wait.
        ::syscall(SYS_futex, reinterpret_cast<int*>(&gFutexWord), 0, 0, nullptr);
    }
    return 0;
}
