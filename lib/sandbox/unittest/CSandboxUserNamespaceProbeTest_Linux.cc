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

#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

#ifndef ML_SANDBOX2_USERNS_PROBE_PAYLOAD
#error "ML_SANDBOX2_USERNS_PROBE_PAYLOAD must be defined by lib/sandbox/unittest/CMakeLists.txt"
#endif

namespace {

//! Forks/execs the userns probe payload directly (no Sandbox2 involved) and
//! reports whether it exited 0 (all 7 stages succeeded).
bool runProbe() {
    const std::string payloadPath{ML_SANDBOX2_USERNS_PROBE_PAYLOAD};

    const pid_t child = ::fork();
    BOOST_TEST_REQUIRE(child >= 0);

    if (child == 0) {
        ::execl(payloadPath.c_str(), payloadPath.c_str(), static_cast<char*>(nullptr));
        // execl only returns on failure.
        ::_exit(127);
    }

    int status = 0;
    BOOST_TEST_REQUIRE(::waitpid(child, &status, 0) == child);
    return WIFEXITED(status) != 0 && WEXITSTATUS(status) == 0;
}

} // namespace

BOOST_AUTO_TEST_SUITE(CSandboxUserNamespaceProbeTest_Linux)

BOOST_AUTO_TEST_CASE(testMatchesRequiredMode) {
    const char* mode = std::getenv("ML_SANDBOX2_REQUIRE");
    const bool probeSucceeded = runProbe();

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
        BOOST_TEST_REQUIRE(!probeSucceeded);
        return;
    }

    BOOST_FAIL("Unrecognised ML_SANDBOX2_REQUIRE value: " + std::string(mode));
}

BOOST_AUTO_TEST_SUITE_END()
