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

#include <core/CDetachedProcessSpawner.h>
#include <core/COsFileFuncs.h>
#include <core/CStringUtils.h>

#include <boost/test/unit_test.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

BOOST_AUTO_TEST_SUITE(CDetachedProcessSpawnerTest)

namespace {
const std::string OUTPUT_FILE("withNs.xml");
const std::string INPUT_FILE("testfiles/withNs.xml");
const size_t EXPECTED_FILE_SIZE(563);
const std::string PROCESS_PATH1("/bin/dd");
const std::string PROCESS_ARGS1[] = {
    "if=" + INPUT_FILE, "of=" + OUTPUT_FILE, "bs=1",
    "count=" + ml::core::CStringUtils::typeToString(EXPECTED_FILE_SIZE)};
const std::string PROCESS_PATH2("/bin/sleep");
const std::string PROCESS_ARGS2[] = {"10"};

bool sandbox2RuntimeSupported() {
    return ::getuid() == 0 ||
           ::access("/proc/sys/kernel/unprivileged_userns_clone", F_OK) == 0;
}

bool requireSandbox2() {
    return ::getenv("ML_REQUIRE_SANDBOX2") != nullptr;
}

std::string findPytorchInferenceBinary() {
    const char* candidates[] = {
        "build/distribution/platform/linux-x86_64/bin/pytorch_inference",
        "build/distribution/platform/linux-aarch64/bin/pytorch_inference",
        "../build/distribution/platform/linux-x86_64/bin/pytorch_inference",
        "../build/distribution/platform/linux-aarch64/bin/pytorch_inference",
    };
    for (const char* candidate : candidates) {
        char resolved[PATH_MAX];
        if (::realpath(candidate, resolved) != nullptr && ::access(resolved, X_OK) == 0) {
            return resolved;
        }
    }
    return {};
}
}

BOOST_AUTO_TEST_CASE(testSpawn) {
    std::remove(OUTPUT_FILE.c_str());

    ml::core::CDetachedProcessSpawner::TStrVec permittedPaths(1, PROCESS_PATH1);
    ml::core::CDetachedProcessSpawner spawner(permittedPaths);

    ml::core::CDetachedProcessSpawner::TStrVec args(
        PROCESS_ARGS1, PROCESS_ARGS1 + std::size(PROCESS_ARGS1));

    BOOST_TEST_REQUIRE(spawner.spawn(PROCESS_PATH1, args));

    std::this_thread::sleep_for(std::chrono::seconds(1));

    ml::core::COsFileFuncs::TStat statBuf;
    BOOST_REQUIRE_EQUAL(0, ml::core::COsFileFuncs::stat(OUTPUT_FILE.c_str(), &statBuf));
    BOOST_REQUIRE_EQUAL(EXPECTED_FILE_SIZE, static_cast<size_t>(statBuf.st_size));

    BOOST_REQUIRE_EQUAL(0, std::remove(OUTPUT_FILE.c_str()));
}

BOOST_AUTO_TEST_CASE(testKill) {
    ml::core::CDetachedProcessSpawner::TStrVec permittedPaths(1, PROCESS_PATH2);
    ml::core::CDetachedProcessSpawner spawner(permittedPaths);

    ml::core::CDetachedProcessSpawner::TStrVec args(
        PROCESS_ARGS2, PROCESS_ARGS2 + std::size(PROCESS_ARGS2));

    ml::core::CProcess::TPid childPid = 0;
    BOOST_TEST_REQUIRE(spawner.spawn(PROCESS_PATH2, args, childPid));

    BOOST_TEST_REQUIRE(spawner.hasChild(childPid));
    BOOST_TEST_REQUIRE(spawner.terminateChild(childPid));

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    BOOST_TEST_REQUIRE(!spawner.hasChild(childPid));
    BOOST_TEST_REQUIRE(!spawner.terminateChild(childPid));
    BOOST_TEST_REQUIRE(!spawner.terminateChild(1));
    BOOST_TEST_REQUIRE(!spawner.terminateChild(0));
    BOOST_TEST_REQUIRE(!spawner.terminateChild(static_cast<ml::core::CProcess::TPid>(-1)));
}

BOOST_AUTO_TEST_CASE(testPermitted) {
    ml::core::CDetachedProcessSpawner::TStrVec permittedPaths(1, PROCESS_PATH1);
    ml::core::CDetachedProcessSpawner spawner(permittedPaths);

    BOOST_TEST_REQUIRE(
        !spawner.spawn("./ml_test", ml::core::CDetachedProcessSpawner::TStrVec()));
}

BOOST_AUTO_TEST_CASE(testPytorchInferenceSubstringNotPermitted) {
    ml::core::CDetachedProcessSpawner::TStrVec permittedPaths(1, PROCESS_PATH1);
    ml::core::CDetachedProcessSpawner spawner(permittedPaths);

    BOOST_TEST_REQUIRE(!spawner.spawn("./evil_pytorch_inference",
                                      ml::core::CDetachedProcessSpawner::TStrVec()));
}

BOOST_AUTO_TEST_CASE(testNonExistent) {
    ml::core::CDetachedProcessSpawner::TStrVec permittedPaths(1, "./does_not_exist");
    ml::core::CDetachedProcessSpawner spawner(permittedPaths);

    BOOST_TEST_REQUIRE(!spawner.spawn(
        "./does_not_exist", ml::core::CDetachedProcessSpawner::TStrVec()));
}

BOOST_AUTO_TEST_CASE(testDisableSandboxUsesLegacyPathAndStripsFlag) {
    // Unique per-process temp dir (see STYLEGUIDE: use PID, not random).
    const std::string dir{"/tmp/ml_spawn_test_" +
                          ml::core::CStringUtils::typeToString(::getpid())};
    ::mkdir(dir.c_str(), 0700);
    const std::string linkPath{dir + "/pytorch_inference"};
    std::remove(linkPath.c_str());
    BOOST_TEST_REQUIRE(::symlink("/bin/dd", linkPath.c_str()) == 0);

    const std::string outputFile{dir + "/out.xml"};
    std::remove(outputFile.c_str());

    ml::core::CDetachedProcessSpawner::TStrVec permittedPaths(1, linkPath);
    ml::core::CDetachedProcessSpawner spawner(permittedPaths);

    ml::core::CDetachedProcessSpawner::TStrVec args{
        "--disableSandbox",
        "if=" + INPUT_FILE,
        "of=" + outputFile,
        "bs=1",
        "count=" + ml::core::CStringUtils::typeToString(EXPECTED_FILE_SIZE)};

    BOOST_TEST_REQUIRE(spawner.spawn(linkPath, args));
    std::this_thread::sleep_for(std::chrono::seconds(1));

    ml::core::COsFileFuncs::TStat statBuf;
    BOOST_REQUIRE_EQUAL(0, ml::core::COsFileFuncs::stat(outputFile.c_str(), &statBuf));
    BOOST_REQUIRE_EQUAL(EXPECTED_FILE_SIZE, static_cast<size_t>(statBuf.st_size));

    std::remove(outputFile.c_str());
    std::remove(linkPath.c_str());
    ::rmdir(dir.c_str());
}

#ifdef SANDBOX2_AVAILABLE

BOOST_AUTO_TEST_CASE(testSandbox2PytorchInferenceRequiresExactAllowlist) {
    const std::string pytorchPath{"/tmp/pytorch_inference"};
    ml::core::CDetachedProcessSpawner::TStrVec wrongAllowlist(1, PROCESS_PATH1);
    ml::core::CDetachedProcessSpawner spawner(wrongAllowlist);

    BOOST_TEST_REQUIRE(
        !spawner.spawn(pytorchPath, ml::core::CDetachedProcessSpawner::TStrVec()));
}

BOOST_AUTO_TEST_CASE(testSandbox2PytorchInferenceSpawnStartsAndTerminates) {
    if (!sandbox2RuntimeSupported()) {
        if (requireSandbox2()) {
            BOOST_FAIL("user namespaces not available but ML_REQUIRE_SANDBOX2 is set");
        }
        BOOST_TEST_MESSAGE("Skipping: user namespaces not available on this host");
        return;
    }

    const std::string pytorchPath = findPytorchInferenceBinary();
    if (pytorchPath.empty()) {
        if (requireSandbox2()) {
            BOOST_FAIL("pytorch_inference binary not found but ML_REQUIRE_SANDBOX2 is set");
        }
        BOOST_TEST_MESSAGE("Skipping: pytorch_inference binary not found in build tree");
        return;
    }

    ml::core::CDetachedProcessSpawner::TStrVec permittedPaths(1, pytorchPath);
    ml::core::CDetachedProcessSpawner spawner(permittedPaths);

    ml::core::CDetachedProcessSpawner::TStrVec args{
        "--validElasticLicenseKeyConfirmed",
        "--namedPipeConnectTimeout=1",
    };

    ml::core::CProcess::TPid childPid = 0;
    const bool spawned = spawner.spawn(pytorchPath, args, childPid);
    if (!spawned) {
        if (requireSandbox2()) {
            BOOST_FAIL("sandboxed pytorch_inference did not start but ML_REQUIRE_SANDBOX2 is set");
        }
        BOOST_TEST_MESSAGE("Skipping: sandboxed pytorch_inference did not start in this environment");
        return;
    }

    BOOST_TEST_REQUIRE(childPid > 0);
    BOOST_TEST_REQUIRE(spawner.hasChild(childPid));

    BOOST_TEST_REQUIRE(spawner.terminateChild(childPid));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    BOOST_TEST_REQUIRE(!spawner.hasChild(childPid));
}

#else

BOOST_AUTO_TEST_CASE(testSandbox2NotAvailable) {
    ml::core::CDetachedProcessSpawner::TStrVec permittedPaths(1, "/tmp/pytorch_inference");
    ml::core::CDetachedProcessSpawner spawner(permittedPaths);

    BOOST_TEST_REQUIRE(!spawner.spawn("/tmp/pytorch_inference",
                                      ml::core::CDetachedProcessSpawner::TStrVec()));
}

#endif

BOOST_AUTO_TEST_SUITE_END()
