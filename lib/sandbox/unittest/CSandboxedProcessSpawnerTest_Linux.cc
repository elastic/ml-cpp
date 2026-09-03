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
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits.h>
#include <sched.h>
#include <sstream>
#include <sys/mount.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

BOOST_AUTO_TEST_SUITE(CSandboxedProcessSpawnerTest)

namespace {

enum class ESandbox2Mode { ENFORCED, FAIL_CLOSED };

// Ordered stages of the user-namespace probe. Reported by name and errno so a
// failure says which syscall the host denied instead of only "unavailable".
enum class EProbeStage {
    E_Success = 0,
    E_Pipe,
    E_Fork,
    E_UnshareUser,
    E_OpenUidMap,
    E_WriteUidMap,
    E_OpenSetgroups,
    E_WriteSetgroups,
    E_OpenGidMap,
    E_WriteGidMap,
    E_UnshareMount,
    E_MountRootPrivate,
    E_MountProc,
    E_ChildLost
};

const char* const KILL_SWITCH_HINT{"xpack.ml.trained_models.sandbox_enabled: false"};

const char* probeStageName(EProbeStage stage) {
    switch (stage) {
    case EProbeStage::E_Success:
        return "success";
    case EProbeStage::E_Pipe:
        return "pipe";
    case EProbeStage::E_Fork:
        return "fork";
    case EProbeStage::E_UnshareUser:
        return "unshare(CLONE_NEWUSER)";
    case EProbeStage::E_OpenUidMap:
        return "open(/proc/self/uid_map)";
    case EProbeStage::E_WriteUidMap:
        return "write(/proc/self/uid_map)";
    case EProbeStage::E_OpenSetgroups:
        return "open(/proc/self/setgroups)";
    case EProbeStage::E_WriteSetgroups:
        return "write(/proc/self/setgroups)";
    case EProbeStage::E_OpenGidMap:
        return "open(/proc/self/gid_map)";
    case EProbeStage::E_WriteGidMap:
        return "write(/proc/self/gid_map)";
    case EProbeStage::E_UnshareMount:
        return "unshare(CLONE_NEWNS)";
    case EProbeStage::E_MountRootPrivate:
        return "mount(/, MS_REC|MS_PRIVATE)";
    case EProbeStage::E_MountProc:
        return "mount(proc, /proc, proc)";
    case EProbeStage::E_ChildLost:
        return "probe child exited without reporting";
    }
    return "unknown";
}

struct SUserNamespaceProbe {
    bool s_Available{false};
    EProbeStage s_Stage{EProbeStage::E_Success};
    int s_Errno{0};

    std::string diagnosis() const {
        if (s_Available) {
            return "available";
        }
        std::string reason{"unavailable: "};
        reason += probeStageName(s_Stage);
        if (s_Errno != 0) {
            reason += " failed with errno " +
                      ml::core::CStringUtils::typeToString(s_Errno) + " (" +
                      ::strerror(s_Errno) + ")";
        }
        return reason;
    }
};

std::string readSysctl(const std::string& path) {
    std::ifstream file{path};
    if (file.is_open() == false) {
        return "<absent>";
    }
    std::string value;
    std::getline(file, value);
    return value.empty() ? "<empty>" : value;
}

// Mounts covering part of /proc make the kernel refuse mount(proc) inside a new
// user namespace even when CLONE_NEWUSER itself succeeds, so list them.
std::string mountsCoveringProc() {
    std::ifstream mountInfo{"/proc/self/mountinfo"};
    std::string covering;
    std::string line;
    while (std::getline(mountInfo, line)) {
        // Fields are: id parentId major:minor root mountPoint options...
        std::istringstream fields{line};
        std::string ignored;
        std::string mountPoint;
        if (static_cast<bool>(fields >> ignored >> ignored >> ignored >>
                              ignored >> mountPoint) &&
            mountPoint.compare(0, 6, "/proc/") == 0) {
            if (covering.empty() == false) {
                covering += ",";
            }
            covering += mountPoint;
        }
    }
    return covering.empty() ? "none" : covering;
}

// Records the host properties that decide whether Sandbox2 can start, in the
// same shape as the controller's boot-time self-check so a CI log and a
// production log can be compared directly.
void logSandbox2Environment() {
    LOG_INFO(<< "Sandbox2 environment self-check: uid=" << ::getuid() << " euid="
             << ::geteuid() << " user.max_user_namespaces="
             << readSysctl("/proc/sys/user/max_user_namespaces")
             << " kernel.unprivileged_userns_clone="
             << readSysctl("/proc/sys/kernel/unprivileged_userns_clone")
             << " selinux.enforce=" << readSysctl("/sys/fs/selinux/enforce")
             << " mountsCoveringProc=" << mountsCoveringProc());
}

SUserNamespaceProbe probeUserNamespaces() {
    // unshare(CLONE_NEWUSER) alone is insufficient: Sandbox2 must write
    // uid_map and remount /proc in a new mount namespace. Hosts that allow
    // CLONE_NEWUSER but deny id mapping (common on GCP dev VMs) or deny
    // mount(proc) (Buildkite k8s agents) must report unavailable.
    int reportFds[2];
    if (::pipe(reportFds) != 0) {
        return {false, EProbeStage::E_Pipe, errno};
    }

    const ml::core::CProcess::TPid probePid = ::fork();
    if (probePid == 0) {
        ::close(reportFds[0]);

        // Reports the reached stage plus errno to the parent, then exits. errno
        // is sampled before the write so the write cannot overwrite it.
        const auto reportAndExit = [&reportFds](EProbeStage stage) {
            const int report[2]{static_cast<int>(stage), errno};
            const ssize_t ignored = ::write(reportFds[1], report, sizeof(report));
            static_cast<void>(ignored);
            ::close(reportFds[1]);
            ::_exit(stage == EProbeStage::E_Success ? 0 : 1);
        };

        const uid_t uid = ::getuid();
        const gid_t gid = ::getgid();
        if (::unshare(CLONE_NEWUSER) != 0) {
            reportAndExit(EProbeStage::E_UnshareUser);
        }

        const std::string uidMapping{"0 " + ml::core::CStringUtils::typeToString(uid) + " 1\n"};
        const int uidMapFd = ::open("/proc/self/uid_map", O_WRONLY);
        if (uidMapFd < 0) {
            reportAndExit(EProbeStage::E_OpenUidMap);
        }
        const ssize_t uidWritten =
            ::write(uidMapFd, uidMapping.c_str(), uidMapping.size());
        ::close(uidMapFd);
        if (uidWritten != static_cast<ssize_t>(uidMapping.size())) {
            reportAndExit(EProbeStage::E_WriteUidMap);
        }

        const int setgroupsFd = ::open("/proc/self/setgroups", O_WRONLY);
        if (setgroupsFd >= 0) {
            static const char deny[]{"deny"};
            const ssize_t denyWritten = ::write(setgroupsFd, deny, sizeof(deny) - 1);
            ::close(setgroupsFd);
            if (denyWritten != static_cast<ssize_t>(sizeof(deny) - 1)) {
                reportAndExit(EProbeStage::E_WriteSetgroups);
            }
        }

        const std::string gidMapping{"0 " + ml::core::CStringUtils::typeToString(gid) + " 1\n"};
        const int gidMapFd = ::open("/proc/self/gid_map", O_WRONLY);
        if (gidMapFd < 0) {
            reportAndExit(EProbeStage::E_OpenGidMap);
        }
        const ssize_t gidWritten =
            ::write(gidMapFd, gidMapping.c_str(), gidMapping.size());
        ::close(gidMapFd);
        if (gidWritten != static_cast<ssize_t>(gidMapping.size())) {
            reportAndExit(EProbeStage::E_WriteGidMap);
        }

        // Sandbox2 also needs a new mount namespace and a fresh /proc. Buildkite
        // k8s agents allow CLONE_NEWUSER + uid_map but reject mount(proc) with
        // EPERM - the stage name in the report distinguishes the two.
        if (::unshare(CLONE_NEWNS) != 0) {
            reportAndExit(EProbeStage::E_UnshareMount);
        }
        // Make "/" private so the proc mount stays in this namespace only.
        if (::mount(nullptr, "/", nullptr, MS_REC | MS_PRIVATE, nullptr) != 0) {
            reportAndExit(EProbeStage::E_MountRootPrivate);
        }
        if (::mount("proc", "/proc", "proc", MS_NODEV | MS_NOEXEC | MS_NOSUID, nullptr) != 0) {
            reportAndExit(EProbeStage::E_MountProc);
        }
        reportAndExit(EProbeStage::E_Success);
    }

    const int forkErrno = errno;
    ::close(reportFds[1]);
    if (probePid < 0) {
        ::close(reportFds[0]);
        return {false, EProbeStage::E_Fork, forkErrno};
    }

    int report[2]{static_cast<int>(EProbeStage::E_ChildLost), 0};
    const ssize_t reportBytes = ::read(reportFds[0], report, sizeof(report));
    ::close(reportFds[0]);

    int status = 0;
    ::waitpid(probePid, &status, 0);

    if (reportBytes != static_cast<ssize_t>(sizeof(report))) {
        return {false, EProbeStage::E_ChildLost, 0};
    }

    const EProbeStage stage = static_cast<EProbeStage>(report[0]);
    const bool exitedCleanly = WIFEXITED(status) != 0 && WEXITSTATUS(status) == 0;
    if (stage == EProbeStage::E_Success && exitedCleanly) {
        return {true, EProbeStage::E_Success, 0};
    }
    return {false, stage, report[1]};
}

const SUserNamespaceProbe& userNamespaceProbe() {
    static const SUserNamespaceProbe PROBE{[]() {
        logSandbox2Environment();
        const SUserNamespaceProbe probe = probeUserNamespaces();
        LOG_INFO(<< "Sandbox2 user-namespace probe: " << probe.diagnosis());
        return probe;
    }()};
    return PROBE;
}

const char* modeName(ESandbox2Mode mode) {
    return mode == ESandbox2Mode::ENFORCED ? "enforced" : "fail_closed";
}

// The mode this host can actually deliver, as opposed to the mode we want it to.
ESandbox2Mode hostSandbox2Mode() {
    return userNamespaceProbe().s_Available ? ESandbox2Mode::ENFORCED
                                            : ESandbox2Mode::FAIL_CLOSED;
}

// ML_SANDBOX2_REQUIRE pins the mode a runner must achieve, so the environments
// we rely on for enforced coverage cannot silently degrade to fail-closed
// coverage. Leaving it unset means "cover whichever mode this host supports",
// which is what developer machines and not-yet-characterised agents want.
void requireModeIfPinned() {
    const char* const required = ::getenv("ML_SANDBOX2_REQUIRE");
    if (required == nullptr || required[0] == '\0') {
        return;
    }
    if (::strcmp(required, "enforced") != 0 && ::strcmp(required, "fail_closed") != 0) {
        BOOST_FAIL("ML_SANDBOX2_REQUIRE must be 'enforced' or 'fail_closed', got '" +
                   std::string{required} + "'");
        return;
    }
    const ESandbox2Mode hostMode = hostSandbox2Mode();
    if (::strcmp(required, modeName(hostMode)) != 0) {
        BOOST_FAIL("ML_SANDBOX2_REQUIRE=" + std::string{required} +
                   " but this host supports only " + modeName(hostMode) +
                   "; user namespaces " + userNamespaceProbe().diagnosis());
    }
}

// True when this host exercises mode, i.e. the calling test body should run.
// The complementary mode is covered by the other test cases on a host that
// supports it; ML_SANDBOX2_REQUIRE is what prevents both from being skipped
// everywhere at once.
bool sandbox2ModeActive(ESandbox2Mode mode) {
    requireModeIfPinned();
    const ESandbox2Mode hostMode = hostSandbox2Mode();
    if (mode == hostMode) {
        return true;
    }
    LOG_WARN(<< "Skipping " << modeName(mode) << " coverage: this host supports "
             << modeName(hostMode) << " only; user namespaces "
             << userNamespaceProbe().diagnosis());
    return false;
}

std::string findPytorchInferenceBinary() {
    const std::string cppRoot{ml::core::CResourceLocator::cppRootDir()};
    const char* const architectures[] = {"linux-x86_64", "linux-aarch64"};

    for (const char* architecture : architectures) {
        const std::string candidate{cppRoot + "/build/distribution/platform/" +
                                    architecture + "/bin/pytorch_inference"};
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
    return "/tmp/ml_sandbox_test_" + ml::core::CStringUtils::typeToString(::getpid());
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
    // Mode independent: the allowlist comparison is pure policy data. Touching
    // the probe puts the environment self-check in every run's log.
    static_cast<void>(userNamespaceProbe());
    requireModeIfPinned();
    BOOST_TEST_REQUIRE(ml::seccomp::pytorch_inference::sandbox2AllowsAllLegacySyscalls());
}

BOOST_AUTO_TEST_CASE(testSandbox2PytorchInferenceSpawnStartsAndTerminates) {
    if (sandbox2ModeActive(ESandbox2Mode::ENFORCED) == false) {
        return;
    }

    const std::string pytorchPath = findPytorchInferenceBinary();

    ml::sandbox::CSandboxedProcessSpawner spawner;
    ml::sandbox::CSandboxedProcessSpawner::TStrVec args{
        "--validElasticLicenseKeyConfirmed=true",
        "--namedPipeConnectTimeout=1",
    };

    std::string failureReason;
    ml::core::CProcess::TPid childPid = 0;
    BOOST_TEST_REQUIRE(spawnSandboxedWithTimeout(
        spawner, pytorchPath, args, childPid, failureReason, std::chrono::seconds(30)));
    BOOST_TEST_REQUIRE(failureReason.empty());
    BOOST_TEST_REQUIRE(childPid > 0);
    BOOST_TEST_REQUIRE(spawner.hasChild(childPid));

    BOOST_TEST_REQUIRE(spawner.terminateChild(childPid));
    BOOST_TEST_REQUIRE(waitForSandboxChildExit(spawner, childPid, std::chrono::seconds(5)));
    BOOST_TEST_REQUIRE(!spawner.hasChild(childPid));
}

BOOST_AUTO_TEST_CASE(testFailClosedWhenUserNamespacesUnavailable) {
    if (sandbox2ModeActive(ESandbox2Mode::FAIL_CLOSED) == false) {
        return;
    }

    const std::string pytorchPath = findPytorchInferenceBinary();

    ml::sandbox::CSandboxedProcessSpawner spawner;
    ml::sandbox::CSandboxedProcessSpawner::TStrVec args{
        "--validElasticLicenseKeyConfirmed=true",
        "--namedPipeConnectTimeout=1",
    };

    std::string failureReason;
    ml::core::CProcess::TPid childPid = 0;
    const bool spawned = spawnSandboxedWithTimeout(
        spawner, pytorchPath, args, childPid, failureReason, std::chrono::seconds(30));

    BOOST_TEST_REQUIRE(!spawned);
    BOOST_TEST_REQUIRE(childPid <= 0);
    BOOST_TEST_REQUIRE(!spawner.hasChild(childPid));
    BOOST_TEST_REQUIRE(failureReason.find(KILL_SWITCH_HINT) != std::string::npos);
}

BOOST_AUTO_TEST_CASE(testPolicyViolationDifferential) {
    if (sandbox2ModeActive(ESandbox2Mode::ENFORCED) == false) {
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
    const bool spawned = spawnSandboxedWithTimeout(sandboxSpawner, linkPath,
                                                   sandboxArgs, childPid, failureReason,
                                                   std::chrono::seconds(30));
    BOOST_TEST_REQUIRE(spawned);
    BOOST_TEST_REQUIRE(failureReason.empty());
    BOOST_TEST_REQUIRE(childPid > 0);

    BOOST_TEST_REQUIRE(waitForSandboxChildExit(sandboxSpawner, childPid,
                                               std::chrono::seconds(10)));

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
                                      childPid, &failureReason));
    BOOST_TEST_REQUIRE(failureReason.find(KILL_SWITCH_HINT) != std::string::npos);
}

#endif

BOOST_AUTO_TEST_SUITE_END()
