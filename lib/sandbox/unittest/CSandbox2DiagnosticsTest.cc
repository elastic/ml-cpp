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

#include <sandbox/CMlSandboxAvailability.h>
#include <sandbox/CSandbox2Diagnostics.h>

#include <boost/test/unit_test.hpp>

#include <set>
#include <string>
#include <vector>

#ifdef Linux
#include <glob.h>
#include <sched.h>
#include <stdlib.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

BOOST_AUTO_TEST_SUITE(CSandbox2DiagnosticsTest)

BOOST_AUTO_TEST_CASE(testDescribeCoversEveryCapability) {
    // Every enumerator must map to a distinct, non-empty sentence: the whole
    // point of the vocabulary is that an operator can tell the denied steps
    // apart, so two enumerators sharing a description would silently defeat
    // it.
    const std::vector<ml::sandbox::ESandbox2Capability> ALL{
        ml::sandbox::ESandbox2Capability::E_Available,
        ml::sandbox::ESandbox2Capability::E_UserNamespaceDenied,
        ml::sandbox::ESandbox2Capability::E_IdMapWriteDenied,
        ml::sandbox::ESandbox2Capability::E_MountOrPidNamespaceDenied,
        ml::sandbox::ESandbox2Capability::E_TmpfsMountDenied,
        ml::sandbox::ESandbox2Capability::E_ProcMountDenied,
        ml::sandbox::ESandbox2Capability::E_ProbeFailed,
        ml::sandbox::ESandbox2Capability::E_ProbeUnsupported};

    std::set<std::string> descriptions;
    for (const auto capability : ALL) {
        const std::string description{ml::sandbox::describe(capability)};
        BOOST_TEST_REQUIRE(description.empty() == false);
        BOOST_TEST_REQUIRE(description != "unrecognized capability value");
        descriptions.insert(description);
    }
    BOOST_REQUIRE_EQUAL(descriptions.size(), ALL.size());
}

BOOST_AUTO_TEST_CASE(testProbeReportsUnsupportedWithoutSandbox2) {
    // On a build with no Sandbox2 support the probe must say so explicitly
    // rather than reporting a denial that was never actually attempted -
    // "not applicable" and "this host forbids user namespaces" are very
    // different operational conclusions.
    if (ml::sandbox::CMlSandboxAvailability::isCompiledIn() == false) {
        BOOST_REQUIRE(ml::sandbox::probeSandbox2Capability() ==
                      ml::sandbox::ESandbox2Capability::E_ProbeUnsupported);
    }
}

#ifdef Linux

BOOST_AUTO_TEST_CASE(testProbeAgreesWithAnIndependentUnshareAttempt) {
    if (ml::sandbox::CMlSandboxAvailability::isCompiledIn() == false) {
        return;
    }

    // Independently establish whether this host permits user namespaces at
    // all, using a plain fork+unshare that shares no code with the probe.
    // The probe's verdict must be consistent with it: if unsharing is
    // denied here the probe has to report E_UserNamespaceDenied, and if it
    // is permitted the probe must report something past that first step.
    // This deliberately asserts a relationship rather than a fixed value,
    // because the answer is a property of the machine the test runs on -
    // it differs between a CI container and a developer's host, and both
    // are legitimate.
    bool usernsPermitted{false};
    const pid_t child{::fork()};
    BOOST_TEST_REQUIRE(child >= 0);
    if (child == 0) {
        ::_exit(::unshare(CLONE_NEWUSER) == 0 ? 0 : 1);
    }
    int status{0};
    BOOST_TEST_REQUIRE(::waitpid(child, &status, 0) == child);
    BOOST_TEST_REQUIRE(WIFEXITED(status));
    usernsPermitted = (WEXITSTATUS(status) == 0);

    const ml::sandbox::ESandbox2Capability capability{ml::sandbox::probeSandbox2Capability()};
    BOOST_TEST_MESSAGE("Sandbox2 capability on this host: " << ml::sandbox::describe(capability));

    if (usernsPermitted) {
        BOOST_REQUIRE(capability != ml::sandbox::ESandbox2Capability::E_UserNamespaceDenied);
    } else {
        BOOST_REQUIRE(capability == ml::sandbox::ESandbox2Capability::E_UserNamespaceDenied);
    }
}

BOOST_AUTO_TEST_CASE(testProbeLeavesTheCallersNamespacesUntouched) {
    if (ml::sandbox::CMlSandboxAvailability::isCompiledIn() == false) {
        return;
    }

    // The probe unshares and mounts, but only ever inside forked children.
    // If any of that leaked into this process the controller would be
    // running in a namespace it never asked for, so pin the caller's own
    // user/mount namespace identity across the call.
    struct stat userNsBefore {};
    struct stat mountNsBefore {};
    BOOST_TEST_REQUIRE(::stat("/proc/self/ns/user", &userNsBefore) == 0);
    BOOST_TEST_REQUIRE(::stat("/proc/self/ns/mnt", &mountNsBefore) == 0);

    ml::sandbox::probeSandbox2Capability();

    struct stat userNsAfter {};
    struct stat mountNsAfter {};
    BOOST_TEST_REQUIRE(::stat("/proc/self/ns/user", &userNsAfter) == 0);
    BOOST_TEST_REQUIRE(::stat("/proc/self/ns/mnt", &mountNsAfter) == 0);

    BOOST_REQUIRE_EQUAL(userNsBefore.st_ino, userNsAfter.st_ino);
    BOOST_REQUIRE_EQUAL(mountNsBefore.st_ino, mountNsAfter.st_ino);
}

BOOST_AUTO_TEST_CASE(testProbeIsRepeatableAndLeaksNoScratchDirectories) {
    if (ml::sandbox::CMlSandboxAvailability::isCompiledIn() == false) {
        return;
    }

    // Two calls must agree - the probe reads no cached state - and neither
    // may leave its mkdtemp() scratch directory behind in TMPDIR.
    const ml::sandbox::ESandbox2Capability first{ml::sandbox::probeSandbox2Capability()};
    const ml::sandbox::ESandbox2Capability second{ml::sandbox::probeSandbox2Capability()};
    BOOST_REQUIRE(first == second);

    const char* tmpDirEnv{::getenv("TMPDIR")};
    const std::string base{tmpDirEnv != nullptr ? tmpDirEnv : "/tmp"};
    // Any leftover would be named ml-sandbox2-probe-* directly under TMPDIR.
    const std::string pattern{base + "/ml-sandbox2-probe-*"};
    ::glob_t globResult;
    const int globStatus{::glob(pattern.c_str(), 0, nullptr, &globResult)};
    const std::size_t leftovers{globStatus == 0 ? globResult.gl_pathc : 0};
    ::globfree(&globResult);
    BOOST_REQUIRE_EQUAL(leftovers, 0);
}

BOOST_AUTO_TEST_CASE(testProbeIsUnaffectedByANonDumpableCaller) {
    if (ml::sandbox::CMlSandboxAvailability::isCompiledIn() == false) {
        return;
    }

    // The controller calls PR_SET_DUMPABLE=0 on itself early in startup, and
    // a non-dumpable process cannot open its own /proc/self/uid_map. The
    // probe must therefore give the same verdict whether its caller is
    // dumpable or not - otherwise a routing decision taken after startup
    // would disagree with the self-check logged before it, which is exactly
    // how a Sandbox2-capable host once got silently downgraded.
    const ml::sandbox::ESandbox2Capability dumpableVerdict{
        ml::sandbox::probeSandbox2Capability()};

    const pid_t child{::fork()};
    BOOST_TEST_REQUIRE(child >= 0);
    if (child == 0) {
        ::prctl(PR_SET_DUMPABLE, 0, 0, 0, 0);
        ::_exit(static_cast<int>(ml::sandbox::probeSandbox2Capability()));
    }
    int status{0};
    BOOST_TEST_REQUIRE(::waitpid(child, &status, 0) == child);
    BOOST_TEST_REQUIRE(WIFEXITED(status));

    BOOST_TEST_MESSAGE("dumpable caller: " << ml::sandbox::describe(dumpableVerdict));
    BOOST_REQUIRE_EQUAL(WEXITSTATUS(status), static_cast<int>(dumpableVerdict));
}

#endif // Linux

BOOST_AUTO_TEST_SUITE_END()
