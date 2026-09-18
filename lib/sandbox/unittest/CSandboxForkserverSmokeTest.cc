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

// Linux-only forkserver runtime smoke test for the dormant MlSandbox
// dependency foundation. This is NOT a security test: it uses
// PolicyBuilder::DangerDefaultAllowAll(), which imposes no seccomp
// restriction. Its only purpose is to prove that the vendored Sandbox2
// forkserver - built via the checked-in patches under
// 3rd_party/patches/sandboxed-api/ - can actually fork, exec, and reap a
// child process end-to-end. Typed launch policy and syscall filtering are
// out of scope here and land in follow-up PRs.
//
// The payload is dynamically linked, so AddLibrariesForBinary() mounts its
// shared-library dependencies into the sandbox namespace; without it,
// Sandbox2's forkserver fails execveat with ENOENT. A static-linked payload
// (matching upstream sandboxed-api's own examples/static/static_bin.cc, to
// sidestep AddLibrariesForBinary entirely) was tried first, but this CI's
// build image has no static libc/libm archives (`ld: cannot find -lm/-lc`).

#include <sandbox/CMlSandboxAvailability.h>

#include <boost/test/unit_test.hpp>

#include <string>

#ifndef ML_SANDBOX2_SMOKE_PAYLOAD
#error "ML_SANDBOX2_SMOKE_PAYLOAD must be defined by lib/sandbox/unittest/CMakeLists.txt"
#endif

#include "absl/time/time.h"
#include "sandboxed_api/sandbox2/executor.h"
#include "sandboxed_api/sandbox2/policybuilder.h"
#include "sandboxed_api/sandbox2/result.h"
#include "sandboxed_api/sandbox2/sandbox2.h"

#include <memory>
#include <vector>

BOOST_AUTO_TEST_SUITE(CSandboxForkserverSmokeTest_Linux)

BOOST_AUTO_TEST_CASE(testForkserverRunsPayloadToCompletion) {
    BOOST_TEST_REQUIRE(ml::sandbox::CMlSandboxAvailability::isCompiledIn());

    const std::string payloadPath{ML_SANDBOX2_SMOKE_PAYLOAD};
    std::vector<std::string> args{payloadPath};

    auto executor = std::make_unique<sandbox2::Executor>(payloadPath, args);
    executor->limits()->set_rlimit_cpu(10).set_walltime_limit(absl::Seconds(10));

    // DangerDefaultAllowAll is deliberately permissive: this test exercises
    // the forkserver plumbing only, not the (not-yet-implemented) sandbox
    // policy. Do not copy this policy into production or security-relevant
    // test code. AddLibrariesForBinary mounts the payload's shared-library
    // dependencies (ldd-derived) so the dynamic loader can find them inside
    // the sandbox namespace.
    auto policy = sandbox2::PolicyBuilder()
                      .DangerDefaultAllowAll()
                      .AddLibrariesForBinary(payloadPath)
                      .BuildOrDie();

    sandbox2::Sandbox2 s2(std::move(executor), std::move(policy));
    sandbox2::Result result = s2.Run();

    BOOST_TEST_REQUIRE(result.final_status() == sandbox2::Result::OK);
    BOOST_TEST_REQUIRE(result.reason_code() == 0);
}

BOOST_AUTO_TEST_SUITE_END()
