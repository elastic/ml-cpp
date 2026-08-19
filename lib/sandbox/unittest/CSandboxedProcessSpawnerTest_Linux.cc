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
#include <core/CLogger.h>
#include <core/COsFileFuncs.h>
#include <core/CProcess.h>
#include <core/CResourceLocator.h>
#include <core/CStringUtils.h>

#include <sandbox/CSandboxedProcessSpawner.h>

#include <seccomp/CPytorchInferenceSyscallAllowlist.h>

#include <boost/test/unit_test.hpp>

#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits.h>
#include <sched.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

BOOST_AUTO_TEST_SUITE(CSandboxedProcessSpawnerTest)

namespace {

enum class ESandbox2Expect { ENFORCED, FAIL_CLOSED };

enum class EUserNamespacesProbe { AVAILABLE, UNAVAILABLE };

const char* const KILL_SWITCH_HINT{"xpack.ml.trained_models.sandbox_enabled: false"};

ESandbox2Expect sandbox2Expect() {
    const char* expectEnv = ::getenv("ML_SANDBOX2_EXPECT");
    if (expectEnv == nullptr || expectEnv[0] == '\0') {
        BOOST_FAIL("ML_SANDBOX2_EXPECT must be set to 'enforced' or 'fail_closed'");
    }
    if (::strcmp(expectEnv, "enforced") == 0) {
        return ESandbox2Expect::ENFORCED;
    }
    if (::strcmp(expectEnv, "fail_closed") == 0) {
        return ESandbox2Expect::FAIL_CLOSED;
    }
    BOOST_FAIL("ML_SANDBOX2_EXPECT must be 'enforced' or 'fail_closed', got '" +
               std::string{expectEnv} + "'");
    return ESandbox2Expect::ENFORCED;
}

EUserNamespacesProbe probeUserNamespaces() {
    const ml::core::CProcess::TPid probePid = ::fork();
    if (probePid == 0) {
        // Must capture host ids before unshare(): inside a fresh user namespace
        // getuid()/getgid() return the overflow id (65534) until uid_map is written.
        const uid_t uid = ::getuid();
        const gid_t gid = ::getgid();
        if (::unshare(CLONE_NEWUSER) != 0) {
            ::_exit(1);
        }
        // unshare() alone is insufficient: Sandbox2 must write uid_map. Hosts that
        // allow CLONE_NEWUSER but deny id mapping (common on GCP dev VMs) pass a
        // naive unshare probe yet hang inside Sandbox2's forkserver.
        const std::string uidMapping{"0 " + ml::core::CStringUtils::typeToString(uid) + " 1\n"};
        const int uidMapFd = ::open("/proc/self/uid_map", O_WRONLY);
        if (uidMapFd < 0) {
            ::_exit(1);
        }
        const ssize_t uidWritten = ::write(uidMapFd, uidMapping.c_str(), uidMapping.size());
        ::close(uidMapFd);
        if (uidWritten != static_cast<ssize_t>(uidMapping.size())) {
            ::_exit(1);
        }

        const int setgroupsFd = ::open("/proc/self/setgroups", O_WRONLY);
        if (setgroupsFd >= 0) {
            static const char deny[]{"deny"};
            const ssize_t denyWritten = ::write(setgroupsFd, deny, sizeof(deny) - 1);
            ::close(setgroupsFd);
            if (denyWritten != static_cast<ssize_t>(sizeof(deny) - 1)) {
                ::_exit(1);
            }
        }

        const std::string gidMapping{"0 " + ml::core::CStringUtils::typeToString(gid) + " 1\n"};
        const int gidMapFd = ::open("/proc/self/gid_map", O_WRONLY);
        if (gidMapFd < 0) {
            ::_exit(1);
        }
        const ssize_t gidWritten = ::write(gidMapFd, gidMapping.c_str(), gidMapping.size());
        ::close(gidMapFd);
        ::_exit(gidWritten == static_cast<ssize_t>(gidMapping.size()) ? 0 : 1);
    }
    if (probePid < 0) {
        LOG_WARN(<< "Sandbox2 user-namespace probe fork failed: " << ::strerror(errno));
        return EUserNamespacesProbe::UNAVAILABLE;
    }

    int status = 0;
    if (::waitpid(probePid, &status, 0) != probePid) {
        LOG_WARN(<< "Sandbox2 user-namespace probe waitpid failed: " << ::strerror(errno));
        return EUserNamespacesProbe::UNAVAILABLE;
    }

    const bool available =
        WIFEXITED(status) != 0 && WEXITSTATUS(status) == 0;
    LOG_INFO(<< "Sandbox2 user-namespace probe: "
             << (available ? "available" : "unavailable"));
    return available ? EUserNamespacesProbe::AVAILABLE : EUserNamespacesProbe::UNAVAILABLE;
}

void validateSandbox2ExpectMatchesProbe() {
    const ESandbox2Expect expect = sandbox2Expect();
    const EUserNamespacesProbe probe = probeUserNamespaces();

    if (expect == ESandbox2Expect::ENFORCED &&
        probe == EUserNamespacesProbe::UNAVAILABLE) {
        BOOST_FAIL("ML_SANDBOX2_EXPECT=enforced but user namespaces are unavailable");
    }
    if (expect == ESandbox2Expect::FAIL_CLOSED &&
        probe == EUserNamespacesProbe::AVAILABLE) {
        BOOST_FAIL("ML_SANDBOX2_EXPECT=fail_closed but user namespaces are available");
    }
}

std::string findPytorchInferenceBinary() {
    const std::string cppRoot{ml::core::CResourceLocator::cppRootDir()};
    const char* const architectures[] = {"linux-x86_64", "linux-aarch64"};

    for (const char* architecture : architectures) {
        const std::string candidate{cppRoot + "/build/distribution/platform/" + architecture +
                                    "/bin/pytorch_inference"};
        char resolved[PATH_MAX];
        if (::realpath(candidate.c_str(), resolved) != nullptr &&
            ::access(resolved, X_OK) == 0) {
            return resolved;
        }
    }

    BOOST_FAIL("pytorch_inference binary not found under " + cppRoot +
               "/build/distribution/platform/linux-{x86_64,aarch64}/bin/");
    return {};
}

bool waitForSandboxChildExit(ml::sandbox::CSandboxedProcessSpawner& spawner,
                             ml::core::CProcess::TPid childPid,
                             std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (!spawner.hasChild(childPid)) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return !spawner.hasChild(childPid);
}

std::string makeTestDir() {
    return "/tmp/ml_sandbox_test_" +
           ml::core::CStringUtils::typeToString(::getpid());
}

bool spawnSandboxedWithTimeout(ml::sandbox::CSandboxedProcessSpawner& spawner,
                               const std::string& processPath,
                               const ml::sandbox::CSandboxedProcessSpawner::TStrVec& args,
                               ml::core::CProcess::TPid& childPid,
                               std::string& failureReason,
                               std::chrono::seconds timeout) {
    // Do not use std::async: if wait_for() times out, ~std::future still joins the
    // stuck spawn thread and the test appears to hang indefinitely.
    bool spawnResult{false};
    std::atomic<bool> spawnFinished{false};
    std::thread spawnThread([&]() {
        spawnResult = spawner.spawn(processPath, args, childPid, &failureReason);
        spawnFinished.store(true, std::memory_order_release);
    });

    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!spawnFinished.load(std::memory_order_acquire)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            spawnThread.detach();
            failureReason = "Timed out waiting for Sandbox2 spawn after " +
                            ml::core::CStringUtils::typeToString(timeout.count()) + " seconds";
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    spawnThread.join();
    return spawnResult;
}

} // namespace

#ifdef SANDBOX2_AVAILABLE

BOOST_AUTO_TEST_CASE(testSandbox2PytorchInferenceRequiresExactAllowlist) {
    validateSandbox2ExpectMatchesProbe();
    BOOST_TEST_REQUIRE(ml::seccomp::pytorch_inference::sandbox2AllowsAllLegacySyscalls());
}

BOOST_AUTO_TEST_CASE(testSandbox2PytorchInferenceSpawnStartsAndTerminates) {
    validateSandbox2ExpectMatchesProbe();
    if (sandbox2Expect() != ESandbox2Expect::ENFORCED) {
        return;
    }

    const std::string pytorchPath = findPytorchInferenceBinary();

    ml::sandbox::CSandboxedProcessSpawner spawner;
    ml::sandbox::CSandboxedProcessSpawner::TStrVec args{
        "--validElasticLicenseKeyConfirmed",
        "--namedPipeConnectTimeout=1",
    };

    std::string failureReason;
    ml::core::CProcess::TPid childPid = 0;
    BOOST_TEST_REQUIRE(spawnSandboxedWithTimeout(spawner, pytorchPath, args, childPid,
                                                 failureReason, std::chrono::seconds(30)));
    BOOST_TEST_REQUIRE(failureReason.empty());
    BOOST_TEST_REQUIRE(childPid > 0);
    BOOST_TEST_REQUIRE(spawner.hasChild(childPid));

    BOOST_TEST_REQUIRE(spawner.terminateChild(childPid));
    BOOST_TEST_REQUIRE(waitForSandboxChildExit(spawner, childPid, std::chrono::seconds(5)));
    BOOST_TEST_REQUIRE(!spawner.hasChild(childPid));
}

BOOST_AUTO_TEST_CASE(testFailClosedWhenUserNamespacesUnavailable) {
    validateSandbox2ExpectMatchesProbe();
    if (sandbox2Expect() != ESandbox2Expect::FAIL_CLOSED) {
        return;
    }

    const std::string pytorchPath = findPytorchInferenceBinary();

    ml::sandbox::CSandboxedProcessSpawner spawner;
    ml::sandbox::CSandboxedProcessSpawner::TStrVec args{
        "--validElasticLicenseKeyConfirmed",
        "--namedPipeConnectTimeout=1",
    };

    std::string failureReason;
    ml::core::CProcess::TPid childPid = 0;
    const bool spawned = spawnSandboxedWithTimeout(spawner, pytorchPath, args, childPid,
                                                   failureReason, std::chrono::seconds(30));

    BOOST_TEST_REQUIRE(!spawned);
    BOOST_TEST_REQUIRE(childPid <= 0);
    BOOST_TEST_REQUIRE(!spawner.hasChild(childPid));
    BOOST_TEST_REQUIRE(failureReason.find(KILL_SWITCH_HINT) != std::string::npos);
}

BOOST_AUTO_TEST_CASE(testPolicyViolationDifferential) {
    validateSandbox2ExpectMatchesProbe();
    if (sandbox2Expect() != ESandbox2Expect::ENFORCED) {
        return;
    }

    const std::string testDir = makeTestDir();
    ::mkdir(testDir.c_str(), 0700);

    // Use a path outside every Sandbox2 mount. $HOME is never bind-mounted for
    // pytorch_inference, while /tmp and the fixed system paths are.
    const char* homeDir = ::getenv("HOME");
    BOOST_TEST_REQUIRE(homeDir != nullptr);
    const std::string forbiddenPath{std::string{homeDir} + "/.ml_sandbox_violation_" +
                                    ml::core::CStringUtils::typeToString(::getpid())};
    std::remove(forbiddenPath.c_str());

    const std::string shellCommand{"touch " + forbiddenPath};

    // Positive control: unsandboxed /bin/sh can write outside sandbox mounts.
    {
        ml::core::CDetachedProcessSpawner::TStrVec permittedPaths{"/bin/sh"};
        ml::core::CDetachedProcessSpawner legacySpawner{permittedPaths};
        ml::core::CDetachedProcessSpawner::TStrVec legacyArgs{"-c", shellCommand};
        BOOST_TEST_REQUIRE(legacySpawner.spawn("/bin/sh", legacyArgs));

        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        ml::core::COsFileFuncs::TStat statBuf;
        BOOST_REQUIRE_EQUAL(0, ml::core::COsFileFuncs::stat(forbiddenPath.c_str(), &statBuf));
        std::remove(forbiddenPath.c_str());
    }

    // Sandboxed path: same write attempt must be blocked by filesystem policy.
    const std::string linkPath{testDir + "/pytorch_inference"};
    std::remove(linkPath.c_str());
    BOOST_TEST_REQUIRE(::symlink("/bin/sh", linkPath.c_str()) == 0);

    ml::sandbox::CSandboxedProcessSpawner sandboxSpawner;
    ml::sandbox::CSandboxedProcessSpawner::TStrVec sandboxArgs{"-c", shellCommand};

    std::string failureReason;
    ml::core::CProcess::TPid childPid = 0;
    const bool spawned = spawnSandboxedWithTimeout(sandboxSpawner, linkPath, sandboxArgs,
                                                   childPid, failureReason,
                                                   std::chrono::seconds(30));
    BOOST_TEST_REQUIRE(spawned);
    BOOST_TEST_REQUIRE(failureReason.empty());
    BOOST_TEST_REQUIRE(childPid > 0);

    BOOST_TEST_REQUIRE(
        waitForSandboxChildExit(sandboxSpawner, childPid, std::chrono::seconds(10)));

    ml::core::COsFileFuncs::TStat statBuf;
    BOOST_REQUIRE_NE(0, ml::core::COsFileFuncs::stat(forbiddenPath.c_str(), &statBuf));

    std::remove(linkPath.c_str());
    ::rmdir(testDir.c_str());
}

#else

BOOST_AUTO_TEST_CASE(testSandbox2NotAvailable) {
    ml::sandbox::CSandboxedProcessSpawner spawner;

    std::string failureReason;
    ml::core::CProcess::TPid childPid = 0;
    BOOST_TEST_REQUIRE(!spawner.spawn("/tmp/pytorch_inference",
                                      ml::sandbox::CSandboxedProcessSpawner::TStrVec(),
                                      childPid,
                                      &failureReason));
    BOOST_TEST_REQUIRE(failureReason.find(KILL_SWITCH_HINT) != std::string::npos);
}

#endif

BOOST_AUTO_TEST_SUITE_END()
