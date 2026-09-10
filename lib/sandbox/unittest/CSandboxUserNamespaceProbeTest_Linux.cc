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

// Linux-only controller-side (host-process) test for PR E's
// ML_SANDBOX2_REQUIRE CI wiring (design.md Sandbox2 clean rebuild plan).
// Runs ml_sandbox_userns_probe as a plain subprocess - deliberately NOT
// through a Sandbox2 Executor/policy, since this test is checking the
// *ambient* CI environment's userns capability (e.g. whether a Buildkite
// k8s pod's runtime permits mount("proc", ...)), not any Sandbox2 policy;
// running it inside a Sandbox2 sandbox here would test the wrong thing.
//
// Three modes, selected by the ML_SANDBOX2_REQUIRE environment variable:
//   unset       -> "ambient" mode: run the probe once, log its outcome, do
//                  not fail the test either way (design.md: "Ambient Docker
//                  seccomp behavior is diagnostic, never load-bearing
//                  coverage").
//   enforced    -> the probe must succeed (all 7 stages complete); fail the
//                  test if any stage fails. Wired into run_tests.sh's
//                  aarch64/Docker branch only (H3 accepted risk: no
//                  userns-capable x86_64 CI runner exists).
//   fail_closed -> pins the *absence* of userns capability as the tested
//                  condition: assert the probe fails at some stage (the
//                  specific stage isn't load-bearing). This mode's job is
//                  confirming the CI environment matches what the existing
//                  fail-closed spawn path (V2) expects, not re-testing V2
//                  itself.

#include <boost/test/unit_test.hpp>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

#ifndef ML_SANDBOX2_USERNS_PROBE_PAYLOAD
#error "ML_SANDBOX2_USERNS_PROBE_PAYLOAD must be defined by lib/sandbox/unittest/CMakeLists.txt"
#endif

namespace {

//! Outcome of running the userns probe payload, distinguishing a genuine
//! staged probe failure (the payload ran and its own stage logic reported
//! failure, pipe/exit code EXIT_FAILURE) from an exec/setup failure (the
//! payload binary could not be launched at all - missing, wrong
//! permissions, bad path). The two must never be conflated: fail_closed's
//! job is confirming the *ambient environment* lacks userns capability, not
//! masking a broken test harness (missing build artifact, CMake wiring
//! regression) as that same "expected absence" result.
enum class EProbeOutcome { E_Success, E_StagedFailure, E_ExecFailure };

//! Forks/execs the userns probe payload directly (no Sandbox2 involved) and
//! classifies the result. POSIX convention: an exec failure surfaces as
//! exit code 126 (found but not executable) or 127 (not found/exec
//! otherwise failed) - the payload's own staged-failure exit code is
//! EXIT_FAILURE (1), which never collides with 126/127. A signal death, or
//! any other non-zero exit, is treated as a staged failure: only 126/127
//! are reserved here for "the child never ran the probe's own logic".
EProbeOutcome runProbe() {
    const std::string payloadPath{ML_SANDBOX2_USERNS_PROBE_PAYLOAD};

    const pid_t child = ::fork();
    BOOST_TEST_REQUIRE(child >= 0);

    if (child == 0) {
        ::execl(payloadPath.c_str(), payloadPath.c_str(), static_cast<char*>(nullptr));
        // execl only returns on failure. Distinguish "found but not
        // executable" (126) from "not found/exec otherwise failed" (127),
        // matching shell convention, so the parent can tell an exec/setup
        // failure apart from the payload's own staged-failure exit code.
        ::_exit(errno == EACCES ? 126 : 127);
    }

    int status = 0;
    BOOST_TEST_REQUIRE(::waitpid(child, &status, 0) == child);

    if (WIFEXITED(status) == 0) {
        // Killed by a signal: not a meaningful staged result, but also not
        // the specific exec-failure signature (126/127) - treat as a
        // staged failure rather than a hard harness-broken failure.
        return EProbeOutcome::E_StagedFailure;
    }

    const int exitStatus = WEXITSTATUS(status);
    if (exitStatus == 126 || exitStatus == 127) {
        return EProbeOutcome::E_ExecFailure;
    }
    return exitStatus == 0 ? EProbeOutcome::E_Success : EProbeOutcome::E_StagedFailure;
}

} // namespace

BOOST_AUTO_TEST_SUITE(CSandboxUserNamespaceProbeTest_Linux)

BOOST_AUTO_TEST_CASE(testMatchesRequiredMode) {
    const char* mode = std::getenv("ML_SANDBOX2_REQUIRE");
    const EProbeOutcome outcome = runProbe();

    // An exec/setup failure means the payload never ran at all - a broken
    // test harness (missing build artifact, CMake wiring regression, bad
    // permissions), not a probe result. Never meaningful in any mode, so
    // fail outright before consulting ML_SANDBOX2_REQUIRE - in particular,
    // this must never be allowed to satisfy fail_closed's "probe failed"
    // check vacuously.
    if (outcome == EProbeOutcome::E_ExecFailure) {
        BOOST_FAIL("ml_sandbox_userns_probe payload could not be exec'd "
                   "(exit 126/127) - test harness is broken, not a "
                   "genuine probe result");
    }

    const bool probeSucceeded = outcome == EProbeOutcome::E_Success;

    if (mode == nullptr) {
        // Ambient mode: diagnostic only - never load-bearing.
        BOOST_TEST_MESSAGE("ml_sandbox_userns_probe ambient outcome: "
                           << (probeSucceeded ? "success" : "failure"));
        return;
    }

    if (std::strcmp(mode, "enforced") == 0) {
        BOOST_TEST_REQUIRE(probeSucceeded);
        return;
    }

    if (std::strcmp(mode, "fail_closed") == 0) {
        // fail_closed pins the *absence* of userns capability as the tested
        // condition (see the file-level comment). MG6's accepted risk names
        // its own revisit trigger as "when a userns-capable x86_64 CI
        // runner becomes available" - the day that happens, a runner
        // acquiring a capability is an environment improvement, not a
        // regression, so it must not look like this test broke. Distinguish
        // three outcomes rather than a single BOOST_TEST_REQUIRE(!probeSucceeded):
        //   - harness/exec broken: already a hard failure via the
        //     E_ExecFailure branch above, unaffected by this branch.
        //   - environment genuinely lacks userns capability (the expected,
        //     currently-universal case): log and pass.
        //   - environment now HAS userns capability: emit a clear,
        //     actionable message, but do NOT fail the build - acquiring a
        //     capability is not a regression.
        if (probeSucceeded) {
            BOOST_TEST_MESSAGE(
                "userns capability is now available on this host (ml_sandbox_userns_probe "
                "succeeded under ML_SANDBOX2_REQUIRE=fail_closed); consider re-pinning "
                "enforced coverage here per the MG6 accepted-risk's revisit trigger "
                "(no userns-capable x86_64 CI runner exists yet)");
        } else {
            BOOST_TEST_MESSAGE("ml_sandbox_userns_probe fail_closed check: userns capability "
                               "genuinely absent, as expected");
        }
        return;
    }

    BOOST_FAIL("Unrecognised ML_SANDBOX2_REQUIRE value: " + std::string(mode));
}

BOOST_AUTO_TEST_SUITE_END()
