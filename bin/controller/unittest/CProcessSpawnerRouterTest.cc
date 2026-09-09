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

#include <core/CProcess.h>

#include "../CProcessSpawnerRouter.h"

#include <boost/test/unit_test.hpp>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>

// This file follows CCommandProcessorTest.cc's convention of testing spawn
// dispatch without a spawner spy: it drives real (non-Linux) dispatch to
// core::CDetachedProcessSpawner and observes side effects / hasChild(), and
// gates anything that would actually reach CSandboxedProcessSpawner behind
// SANDBOX2_AVAILABLE - the same macro CProcessSpawnerRouter::spawn() itself
// branches on - rather than the coarser `Linux`.

BOOST_AUTO_TEST_SUITE(CProcessSpawnerRouterTest)

namespace {
#ifdef Windows
// Unlike Windows NT system calls, copy's command line cannot cope with
// forward slash path separators
const std::string INPUT_FILE{"testfiles\\slogan1.txt"};
const char* winDir{std::getenv("windir")};
const std::string PROCESS_PATH{winDir != nullptr ? std::string{winDir} + "\\System32\\cmd"
                                                  : std::string{"C:\\Windows\\System32\\cmd"}};
std::string copyArgsScript(const std::string& outputFile) {
    return "copy " + INPUT_FILE + " " + outputFile;
}
const std::string SHELL_FLAG{"/C"};
#else
const std::string INPUT_FILE{"testfiles/slogan1.txt"};
const std::string PROCESS_PATH{"/bin/sh"};
std::string copyArgsScript(const std::string& outputFile) {
    return "cp " + INPUT_FILE + " " + outputFile;
}
const std::string SHELL_FLAG{"-c"};
#endif
const std::string SLOGAN1{"Elastic is great!"};

//! Run \p router's spawn() for a shell command that copies INPUT_FILE to
//! \p outputFile, and assert the copy actually happened - proof the call
//! was dispatched to a working spawner backend, not just that spawn()
//! returned true.
void assertDispatchCopiesFile(ml::controller::CProcessSpawnerRouter& router,
                               ml::controller::CProcessSpawnerRouter::ERoute route,
                               const std::string& outputFile) {
    std::remove(outputFile.c_str());

    ml::controller::CProcessSpawnerRouter::TStrVec args{SHELL_FLAG, copyArgsScript(outputFile)};
    ml::core::CProcess::TPid childPid{0};
    BOOST_TEST_REQUIRE(router.spawn(route, PROCESS_PATH, args, childPid));
    BOOST_TEST_REQUIRE(childPid != 0);

    // Expect the copy to complete well inside 1 second, matching
    // CCommandProcessorTest.cc's own timing assumption for the same kind of
    // command.
    std::this_thread::sleep_for(std::chrono::seconds{1});

    std::ifstream ifs{outputFile};
    BOOST_TEST_REQUIRE(ifs.is_open());
    std::string content;
    std::getline(ifs, content);
    ifs.close();
    BOOST_REQUIRE_EQUAL(SLOGAN1, content);

    std::remove(outputFile.c_str());
}
}

BOOST_AUTO_TEST_CASE(testSandbox2RouteDispatchesLegacyForUnsandboxedPath) {
    // processPath is permitted but not listed as sandboxed: an E_Sandbox2
    // route must still land on the legacy spawner, exactly like today's
    // CDetachedProcessSpawner-only paths for autodetect/categorize/etc.
    ml::controller::CProcessSpawnerRouter::TStrVec permittedPaths{PROCESS_PATH};
    ml::controller::CProcessSpawnerRouter::TStrVec sandboxedPaths; // empty
    ml::controller::CProcessSpawnerRouter router{permittedPaths, sandboxedPaths};

    assertDispatchCopiesFile(router, ml::controller::CProcessSpawnerRouter::ERoute::E_Sandbox2,
                              "router_test_never_sandboxed.txt");
}

BOOST_AUTO_TEST_CASE(testLegacyRouteDispatchesLegacyForSandboxedPath) {
    // processPath IS listed as sandboxed, but the caller has already
    // decided E_Legacy (operator kill switch, validated upstream): the
    // router must still dispatch to the legacy spawner and never consult
    // Sandbox2 availability for this route.
    ml::controller::CProcessSpawnerRouter::TStrVec permittedPaths{PROCESS_PATH};
    ml::controller::CProcessSpawnerRouter::TStrVec sandboxedPaths{PROCESS_PATH};
    ml::controller::CProcessSpawnerRouter router{permittedPaths, sandboxedPaths};

    assertDispatchCopiesFile(router, ml::controller::CProcessSpawnerRouter::ERoute::E_Legacy,
                              "router_test_legacy_route.txt");
}

BOOST_AUTO_TEST_CASE(testTerminateAndHasChildCoverBothBackends) {
    // A PID this router never spawned is owned by neither backend.
    ml::controller::CProcessSpawnerRouter::TStrVec permittedPaths{PROCESS_PATH};
    ml::controller::CProcessSpawnerRouter::TStrVec sandboxedPaths{PROCESS_PATH};
    ml::controller::CProcessSpawnerRouter router{permittedPaths, sandboxedPaths};

    BOOST_REQUIRE_EQUAL(false, router.hasChild(0));
    BOOST_REQUIRE_EQUAL(false, router.terminateChild(0));
}

#ifndef SANDBOX2_AVAILABLE
BOOST_AUTO_TEST_CASE(testSandbox2RouteFailsClosedWithoutSandbox2Support) {
    // Build/deployment contradiction case (design doc): processPath is
    // configured as sandboxed, but this build has no Sandbox2 support.
    // spawn() must fail closed - never fall through to the legacy spawner,
    // and never touch either spawner's live-child bookkeeping for the pid
    // it would have used.
    ml::controller::CProcessSpawnerRouter::TStrVec permittedPaths{PROCESS_PATH};
    ml::controller::CProcessSpawnerRouter::TStrVec sandboxedPaths{PROCESS_PATH};
    ml::controller::CProcessSpawnerRouter router{permittedPaths, sandboxedPaths};

    ml::controller::CProcessSpawnerRouter::TStrVec args{SHELL_FLAG, copyArgsScript("router_test_should_not_run.txt")};
    ml::core::CProcess::TPid childPid{0};
    BOOST_REQUIRE_EQUAL(
        false, router.spawn(ml::controller::CProcessSpawnerRouter::ERoute::E_Sandbox2,
                             PROCESS_PATH, args, childPid));

    // No child was ever registered with either backend for this attempt.
    BOOST_REQUIRE_EQUAL(false, router.hasChild(childPid));

    // The legacy spawner was never reached either: the output file the
    // copy command would have produced must not exist.
    std::ifstream ifs{"router_test_should_not_run.txt"};
    BOOST_REQUIRE_EQUAL(false, ifs.is_open());
}
#endif // !SANDBOX2_AVAILABLE

// Buildkite-deferred (Linux + Sandbox2 only, design.md V2): asserting that
// an E_Sandbox2 route for a sandboxedProcessPaths entry reaches
// CSandboxedProcessSpawner::spawn(), and that a failure there returns false
// without any retry through the legacy spawner, needs a real Sandbox2
// launch target. That requires the payload-executable + filesystem-policy
// scaffolding lib/sandbox/unittest/CMakeLists.txt builds for
// CSandboxedProcessSpawnerLifecycleTest_Linux (payloads/, sandbox2::sandbox2
// link, Linux-only CMake block) - none of which bin/controller/unittest
// currently has. This host (macOS) cannot build or run that scaffolding, so
// this assertion is intentionally not implemented here; it belongs either
// in a future Linux-gated addition to this file once bin/controller/unittest
// grows the same payload machinery, or as a lib/sandbox-level test that
// exercises CProcessSpawnerRouter directly.

BOOST_AUTO_TEST_SUITE_END()
