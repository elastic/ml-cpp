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

#include <core/CLogger.h>
#include <core/CProcess.h>
#include <core/CSetEnv.h>
#include <core/CUnSetEnv.h>

#include "../CProcessSpawnerRouter.h"

#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>
#include <boost/test/unit_test.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

// This file follows CCommandProcessorTest.cc's convention of testing spawn
// dispatch without a spawner spy: it drives real (non-Linux) dispatch to
// core::CDetachedProcessSpawner and observes side effects / hasChild(), and
// gates anything that would actually reach CSandboxedProcessSpawner behind
// SANDBOX2_AVAILABLE - the same macro CProcessSpawnerRouter::spawn() itself
// branches on - rather than the coarser `Linux`.
//
// H4 (PR E Task 4) signal assertions redirect ml::core::CLogger to an
// in-memory stream (the same technique CBoostedTreeTest.cc uses for its own
// LOG_ERROR assertions) and inspect the emitted JSON line as a substring
// match per field, rather than parsing JSON - this avoids pulling in a JSON
// parser dependency for a handful of flat string/bool fields.

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

//! Redirect ml::core::CLogger to an in-memory stream for the duration of
//! \p fn, then reset() it back to its default configuration before
//! returning - callers must not leak the redirect into later test cases.
//! \return everything logged while \p fn ran, so the caller can search for
//! the H4 signal's JSON line as a substring.
template<typename FN>
std::string captureLogged(FN&& fn) {
    auto stream = boost::make_shared<std::ostringstream>();
    BOOST_TEST_REQUIRE(ml::core::CLogger::instance().reconfigure(stream));
    fn();
    ml::core::CLogger::instance().reset();
    return stream->str();
}

#ifndef Windows
//! Creates a canonical, existing $TMPDIR/ml-child-ipc/<child-id> directory
//! and points TMPDIR at that trusted base for the duration of a scope, so
//! sandbox::validateChildIpcLaunchSpec() (which does live ::realpath() calls
//! and requires the parent directory to exist) can derive a real
//! deployment_id. Restores the previous TMPDIR and removes the tree on
//! destruction.
class CScopedChildIpcRoot {
public:
    explicit CScopedChildIpcRoot(const std::string& childId) : m_ChildId{childId} {
        const char* previous{std::getenv("TMPDIR")};
        m_HadPreviousTmpDir = previous != nullptr;
        if (m_HadPreviousTmpDir) {
            m_PreviousTmpDir.assign(previous);
        }

        // boost::filesystem::canonical() so the base itself is already
        // canonical - validateChildIpcLaunchSpec() compares the literal and
        // canonical parents and rejects any difference, and on macOS the
        // system temporary directories are reached through symlinks.
        m_TrustedTmpDir =
            (boost::filesystem::canonical(boost::filesystem::current_path()) /
             ("router_h4_tmp_" + childId))
                .string();
        m_ChildIpcRoot = m_TrustedTmpDir + "/ml-child-ipc/" + childId;
        boost::filesystem::create_directories(m_ChildIpcRoot);

        BOOST_REQUIRE_EQUAL(0, ml::core::CSetEnv::setEnv("TMPDIR", m_TrustedTmpDir.c_str(), 1));
    }

    ~CScopedChildIpcRoot() {
        if (m_HadPreviousTmpDir) {
            ml::core::CSetEnv::setEnv("TMPDIR", m_PreviousTmpDir.c_str(), 1);
        } else {
            ml::core::CUnSetEnv::unSetEnv("TMPDIR");
        }
        boost::system::error_code ignored;
        boost::filesystem::remove_all(m_TrustedTmpDir, ignored);
    }

    //! An --input=<path> argument inside this child's IPC root, i.e. one
    //! validateChildIpcLaunchSpec() accepts and derives m_ChildId from.
    std::string inputArg() const { return "--input=" + m_ChildIpcRoot + "/input"; }

    CScopedChildIpcRoot(const CScopedChildIpcRoot&) = delete;
    CScopedChildIpcRoot& operator=(const CScopedChildIpcRoot&) = delete;

private:
    std::string m_ChildId;
    std::string m_TrustedTmpDir;
    std::string m_ChildIpcRoot;
    std::string m_PreviousTmpDir;
    bool m_HadPreviousTmpDir{false};
};
#endif // !Windows
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

BOOST_AUTO_TEST_CASE(testH4SignalFailClosedWithoutSandbox2Support) {
    // Reuses the exact non-Linux fail-closed vector above (route ==
    // E_Sandbox2 for a sandboxedProcessPaths entry, no SANDBOX2_AVAILABLE)
    // to assert the H4 signal itself: mode == "fail_closed",
    // sandbox2_established == false (a JSON boolean, not the string
    // "false"), route == "sandbox2", and the signal fires even though
    // spawn() returns false - it must not be gated behind a success check.
    ml::controller::CProcessSpawnerRouter::TStrVec permittedPaths{PROCESS_PATH};
    ml::controller::CProcessSpawnerRouter::TStrVec sandboxedPaths{PROCESS_PATH};
    ml::controller::CProcessSpawnerRouter router{permittedPaths, sandboxedPaths};

    ml::controller::CProcessSpawnerRouter::TStrVec args{"--modelid=deploy-fail-closed"};
    ml::core::CProcess::TPid childPid{0};
    std::string logged{captureLogged([&] {
        BOOST_REQUIRE_EQUAL(
            false, router.spawn(ml::controller::CProcessSpawnerRouter::ERoute::E_Sandbox2,
                                 PROCESS_PATH, args, childPid));
    })};

    BOOST_REQUIRE(logged.find("\"event\":\"sandbox2_launch\"") != std::string::npos);
    BOOST_REQUIRE(logged.find("\"route\":\"sandbox2\"") != std::string::npos);
    BOOST_REQUIRE(logged.find("\"mode\":\"fail_closed\"") != std::string::npos);
    BOOST_REQUIRE(logged.find("\"sandbox2_established\":false") != std::string::npos);
    BOOST_REQUIRE(logged.find("\"model_id\":\"deploy-fail-closed\"") != std::string::npos);
    // No path-bearing (input/output/restore/logPipe) option was present in
    // args *at all*, which is the only case that still yields an empty
    // deployment_id - it must be the explicit empty string, not omitted.
    // When such an option is present, deployment_id is populated in this
    // same fail_closed mode: see
    // testH4SignalDeploymentIdPopulatedOnFailClosed below.
    BOOST_REQUIRE(logged.find("\"deployment_id\":\"\"") != std::string::npos);
}

#ifndef Windows
BOOST_AUTO_TEST_CASE(testH4SignalDeploymentIdPopulatedOnFailClosed) {
    // deployment_id is derived once, before dispatch, so it is populated on
    // the fail_closed mode too - previously the derivation ran after
    // spawn() had already failed, and reported "" on exactly the modes this
    // signal exists to make debuggable.
    const std::string childId{"deployfailclosed"};
    CScopedChildIpcRoot childIpcRoot{childId};

    ml::controller::CProcessSpawnerRouter::TStrVec permittedPaths{PROCESS_PATH};
    ml::controller::CProcessSpawnerRouter::TStrVec sandboxedPaths{PROCESS_PATH};
    ml::controller::CProcessSpawnerRouter router{permittedPaths, sandboxedPaths};

    ml::controller::CProcessSpawnerRouter::TStrVec args{childIpcRoot.inputArg()};
    ml::core::CProcess::TPid childPid{0};
    std::string logged{captureLogged([&] {
        BOOST_REQUIRE_EQUAL(
            false, router.spawn(ml::controller::CProcessSpawnerRouter::ERoute::E_Sandbox2,
                                 PROCESS_PATH, args, childPid));
    })};

    BOOST_REQUIRE(logged.find("\"mode\":\"fail_closed\"") != std::string::npos);
    BOOST_REQUIRE(logged.find("\"deployment_id\":\"" + childId + "\"") != std::string::npos);
}
#endif // !Windows
#endif // !SANDBOX2_AVAILABLE

#ifndef Windows
BOOST_AUTO_TEST_CASE(testH4SignalDeploymentIdPopulatedOnDegradedRoute) {
    // Same single-derivation guarantee on the degraded (legacy-route) mode,
    // which never reaches CSandboxedProcessSpawner's own validation call at
    // all - and here the legacy spawn itself also fails (PROCESS_PATH is
    // deliberately not permitted), so this covers the worst case for the
    // old post-spawn derivation.
    const std::string childId{"deploydegraded"};
    CScopedChildIpcRoot childIpcRoot{childId};

    ml::controller::CProcessSpawnerRouter::TStrVec permittedPaths; // deliberately empty
    ml::controller::CProcessSpawnerRouter::TStrVec sandboxedPaths{PROCESS_PATH};
    ml::controller::CProcessSpawnerRouter router{permittedPaths, sandboxedPaths};

    ml::controller::CProcessSpawnerRouter::TStrVec args{childIpcRoot.inputArg()};
    ml::core::CProcess::TPid childPid{0};
    std::string logged{captureLogged([&] {
        BOOST_REQUIRE_EQUAL(
            false, router.spawn(ml::controller::CProcessSpawnerRouter::ERoute::E_Legacy,
                                 PROCESS_PATH, args, childPid));
    })};

    BOOST_REQUIRE(logged.find("\"mode\":\"degraded\"") != std::string::npos);
    BOOST_REQUIRE(logged.find("\"deployment_id\":\"" + childId + "\"") != std::string::npos);
}
#endif // !Windows

#ifndef Windows
BOOST_AUTO_TEST_CASE(testH4SignalEscapesControlCharactersInDeploymentId) {
    // deployment_id is a filesystem path component, so a raw control
    // character in it would otherwise split what must stay a single-line
    // JSON object.
    const std::string childId{"deploy\nid\tx"};
    CScopedChildIpcRoot childIpcRoot{childId};

    ml::controller::CProcessSpawnerRouter::TStrVec permittedPaths; // deliberately empty
    ml::controller::CProcessSpawnerRouter::TStrVec sandboxedPaths{PROCESS_PATH};
    ml::controller::CProcessSpawnerRouter router{permittedPaths, sandboxedPaths};

    ml::controller::CProcessSpawnerRouter::TStrVec args{childIpcRoot.inputArg()};
    ml::core::CProcess::TPid childPid{0};
    std::string logged{captureLogged([&] {
        BOOST_REQUIRE_EQUAL(
            false, router.spawn(ml::controller::CProcessSpawnerRouter::ERoute::E_Legacy,
                                 PROCESS_PATH, args, childPid));
    })};

    BOOST_REQUIRE(logged.find("\"deployment_id\":\"deploy\\nid\\tx\"") != std::string::npos);
    // ...and the raw control characters are gone from the emitted line.
    const std::size_t signalStart{logged.find("{\"event\":\"sandbox2_launch\"")};
    BOOST_TEST_REQUIRE(signalStart != std::string::npos);
    const std::size_t signalEnd{logged.find("\"mode\":\"degraded\"}", signalStart)};
    BOOST_TEST_REQUIRE(signalEnd != std::string::npos);
    BOOST_REQUIRE(logged.find('\n', signalStart) > signalEnd);
}
#endif // !Windows

BOOST_AUTO_TEST_CASE(testNoH4SignalForUnsandboxedProcessPath) {
    // Negative assertion: a process path that is not configured as sandboxed
    // (autodetect, categorize, and every other permitted process) must
    // produce no sandbox2_launch line at all - not one with route "legacy",
    // not one with an empty deployment_id, none.
    ml::controller::CProcessSpawnerRouter::TStrVec permittedPaths{PROCESS_PATH};
    ml::controller::CProcessSpawnerRouter::TStrVec sandboxedPaths; // empty
    ml::controller::CProcessSpawnerRouter router{permittedPaths, sandboxedPaths};

    const std::string outputFile{"router_test_no_h4_signal.txt"};
    std::remove(outputFile.c_str());
    ml::controller::CProcessSpawnerRouter::TStrVec args{
        SHELL_FLAG, copyArgsScript(outputFile), "--modelid=deploy-not-sandboxed"};
    ml::core::CProcess::TPid childPid{0};
    std::string logged{captureLogged([&] {
        BOOST_REQUIRE_EQUAL(
            true, router.spawn(ml::controller::CProcessSpawnerRouter::ERoute::E_Sandbox2,
                                PROCESS_PATH, args, childPid));
    })};
    std::this_thread::sleep_for(std::chrono::seconds{1});
    std::remove(outputFile.c_str());

    BOOST_REQUIRE(logged.find("sandbox2_launch") == std::string::npos);
    BOOST_REQUIRE(logged.find("deploy-not-sandboxed") == std::string::npos);
}

BOOST_AUTO_TEST_CASE(testH4SignalDegradedOnLegacyRouteSuccess) {
    // Token-present route: mode must be "degraded" and sandbox2_established
    // false regardless of the legacy spawn's own outcome. This case is the
    // successful-spawn half of that "regardless" - see
    // testH4SignalDegradedOnLegacyRouteFailure for the failed-spawn half.
    ml::controller::CProcessSpawnerRouter::TStrVec permittedPaths{PROCESS_PATH};
    ml::controller::CProcessSpawnerRouter::TStrVec sandboxedPaths{PROCESS_PATH};
    ml::controller::CProcessSpawnerRouter router{permittedPaths, sandboxedPaths};

    const std::string outputFile{"router_test_h4_degraded_success.txt"};
    std::remove(outputFile.c_str());
    ml::controller::CProcessSpawnerRouter::TStrVec args{
        SHELL_FLAG, copyArgsScript(outputFile), "--modelid=deploy-degraded-ok"};
    ml::core::CProcess::TPid childPid{0};
    std::string logged{captureLogged([&] {
        BOOST_REQUIRE_EQUAL(
            true, router.spawn(ml::controller::CProcessSpawnerRouter::ERoute::E_Legacy,
                                PROCESS_PATH, args, childPid));
    })};
    // The copy runs in the detached child asynchronously - give it the same
    // grace period assertDispatchCopiesFile above uses before cleaning up,
    // so this test doesn't race the shell command and leave debris behind.
    std::this_thread::sleep_for(std::chrono::seconds{1});
    std::remove(outputFile.c_str());

    BOOST_REQUIRE(logged.find("\"event\":\"sandbox2_launch\"") != std::string::npos);
    BOOST_REQUIRE(logged.find("\"route\":\"legacy\"") != std::string::npos);
    BOOST_REQUIRE(logged.find("\"mode\":\"degraded\"") != std::string::npos);
    BOOST_REQUIRE(logged.find("\"sandbox2_established\":false") != std::string::npos);
    BOOST_REQUIRE(logged.find("\"model_id\":\"deploy-degraded-ok\"") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(testH4SignalDegradedOnLegacyRouteFailure) {
    // Same route (E_Legacy) but the legacy spawn itself fails
    // deterministically, without touching the filesystem or the real
    // Sandbox2 backend: PROCESS_PATH is listed as sandboxed (so the signal
    // is eligible to fire) but deliberately left out of permittedPaths, so
    // core::CDetachedProcessSpawner::spawn() rejects it up front
    // ("is not permitted") before any fork/exec attempt. Confirms mode ==
    // "degraded" (not "fail_closed" - that mode is reserved for the
    // no-token Sandbox2 route) even though the underlying spawn failed.
    ml::controller::CProcessSpawnerRouter::TStrVec permittedPaths; // PROCESS_PATH deliberately absent
    ml::controller::CProcessSpawnerRouter::TStrVec sandboxedPaths{PROCESS_PATH};
    ml::controller::CProcessSpawnerRouter router{permittedPaths, sandboxedPaths};

    ml::controller::CProcessSpawnerRouter::TStrVec args{"--modelid=deploy-degraded-fail"};
    ml::core::CProcess::TPid childPid{0};
    std::string logged{captureLogged([&] {
        BOOST_REQUIRE_EQUAL(
            false, router.spawn(ml::controller::CProcessSpawnerRouter::ERoute::E_Legacy,
                                 PROCESS_PATH, args, childPid));
    })};

    BOOST_REQUIRE(logged.find("\"event\":\"sandbox2_launch\"") != std::string::npos);
    BOOST_REQUIRE(logged.find("\"route\":\"legacy\"") != std::string::npos);
    BOOST_REQUIRE(logged.find("\"mode\":\"degraded\"") != std::string::npos);
    BOOST_REQUIRE(logged.find("\"sandbox2_established\":false") != std::string::npos);
    BOOST_REQUIRE(logged.find("\"model_id\":\"deploy-degraded-fail\"") != std::string::npos);
}

// Buildkite-deferred (Linux + Sandbox2 only): the mode == "enforced" /
// sandbox2_established == true case requires a real successful Sandbox2
// launch (route == E_Sandbox2, a sandboxedProcessPaths entry, spawn()
// returning true) - on a build without SANDBOX2_AVAILABLE that combination
// is unreachable, since CProcessSpawnerRouter::spawn() unconditionally
// fails closed for it (see testH4SignalFailClosedWithoutSandbox2Support
// immediately above). This is the same platform limitation the pre-existing
// Buildkite-deferred note below documents for the router's own Sandbox2
// dispatch; the H4 "enforced" case needs the identical Linux + Sandbox2
// scaffolding once a Sandbox2-aware controller unittest target exists.

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
