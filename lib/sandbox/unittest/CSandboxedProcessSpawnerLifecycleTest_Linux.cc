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

// Linux-only lifecycle test for CSandboxedProcessSpawner (PR D, Task 4 of
// docs/projects/mlcpp-sandbox2-pr2873/pr-d-lifecycle.plan.md). Drives the
// spawner's four injectable seams (TPidFdOpenFn, TRegistryInsertFn,
// TMonitorLaunchFn, TAwaitResultFn - see CSandboxedProcessSpawner.h) to
// exercise fault-injection and race scenarios deterministically, but there
// is no seam that bypasses Sandbox2::RunAsync() itself: every test case
// below performs one genuine spawn() of a real, minimal, dependency-free
// payload (lifecycle_signal_payload.cc) under the real filesystem policy
// spawn() builds. The registry-insert seam receives a mutable reference to
// the spawner's *actual* internal SPidRegistry (not a copy), which every
// test below uses after that one real spawn() to fabricate/mutate further
// registry state directly - this is the only externally reachable handle to
// that private registry, since the spawner has no accessor for it and no
// constructor overload accepts a caller-supplied one.
//
// No sleep()/wall-clock polling anywhere in this file. Timing-sensitive
// races are driven either by directly exercising
// CSandboxedProcessSpawner::CCasOutcomeLatch (a pure, thread-safe type, see
// gate 5), by manually invoking a *captured* monitor-body callable on the
// calling thread instead of ever starting a background thread for it, or -
// where a genuine background thread is required (gates 7 and 8) - by a
// std::promise/future gate the test controls explicitly. The one bounded
// wait that has no other synchronisation primitive available (observing
// "still alive" - the absence of an event) uses a single poll() call with a
// timeout, never a sleep-and-recheck loop.
//
// NOT YET RUN: like CPytorchInferenceSandboxPolicyMechanismTest_Linux, this
// file has not executed on a real Linux+Sandbox2 host in this session (the
// authoring host is macOS, and the Sandbox2 headers are fetched at CMake
// configure time, not vendored in this checkout - see task-4-report.md for
// what could and could not be verified here, and for open concerns this
// design accepts).

#include <sandbox/CSandboxedProcessSpawner.h>

#include <boost/test/unit_test.hpp>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <dirent.h>
#include <fcntl.h>
#include <ftw.h>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <new>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#ifndef ML_SANDBOX2_LIFECYCLE_PAYLOAD
#error "ML_SANDBOX2_LIFECYCLE_PAYLOAD must be defined by lib/sandbox/unittest/CMakeLists.txt"
#endif

#include "sandboxed_api/sandbox2/result.h"
#include "sandboxed_api/sandbox2/sandbox2.h"

// Same rationale as CSandboxedProcessSpawner_Linux.cc's identical fallback:
// the CentOS 7 CI build image's kernel headers may predate pidfd_open, but
// this is the same syscall number (434) on every architecture ml-cpp
// builds for. Used here purely for *test-owned observation* pidfds (poll()
// for exit, never a signal) - never to signal a spawner-owned child.
#ifdef __NR_pidfd_open
#define ML_TEST_NR_pidfd_open __NR_pidfd_open
#else
#define ML_TEST_NR_pidfd_open 434
#endif

namespace {

using ml::sandbox::CSandboxedProcessSpawner;
using TSpawner = CSandboxedProcessSpawner;
using TPid = ml::core::CProcess::TPid;

// ---------------------------------------------------------------------
// Descriptor-count baseline helpers (gate 6).
// ---------------------------------------------------------------------

//! Number of open file descriptors this process currently holds, via
//! /proc/self/fd (Linux-only, fine - this whole translation unit is
//! Linux-gated). Excludes "." and "..", includes the directory fd opendir()
//! itself just opened (consistently, on both the "before" and "after"
//! snapshot, so it cancels out).
std::size_t openFdCount() {
    DIR* dir = ::opendir("/proc/self/fd");
    BOOST_TEST_REQUIRE(dir != nullptr);
    std::size_t count{0};
    struct dirent* entry{nullptr};
    while ((entry = ::readdir(dir)) != nullptr) {
        const std::string name{entry->d_name};
        if (name != "." && name != "..") {
            ++count;
        }
    }
    ::closedir(dir);
    return count;
}

//! Applied to the whole suite: every test case must leave the process with
//! exactly the descriptors it started with (LI4, V10's cleanup assertions).
struct SFdBaselineFixture {
    SFdBaselineFixture() : s_Baseline(openFdCount()) {}
    ~SFdBaselineFixture() { BOOST_CHECK_EQUAL(openFdCount(), s_Baseline); }
    std::size_t s_Baseline;
};

// ---------------------------------------------------------------------
// $TMPDIR / child-IPC-root scaffolding, matching the PR C interlock
// (validateChildIpcLaunchSpec) spawn() enforces before building a policy -
// see CPytorchInferenceSandboxPolicyMechanismTest_Linux.cc for the same
// pattern used directly against the validation function.
// ---------------------------------------------------------------------

int removeEntryBestEffort(const char* fpath, const struct stat*, int typeflag, struct FTW*) {
    if (typeflag == FTW_DP) {
        ::rmdir(fpath);
    } else {
        ::unlink(fpath);
    }
    return 0;
}

void removeTreeBestEffort(const std::string& path) {
    ::nftw(path.c_str(), removeEntryBestEffort, 16, FTW_DEPTH | FTW_PHYS);
}

//! RAII: creates a fresh, private tmp directory and points $TMPDIR at it for
//! the lifetime of this object, so spawn()'s own trustedTmpDir
//! (getenv("TMPDIR") or "/tmp") matches exactly the root this test builds
//! its child-IPC directories under - giving each test case an isolated
//! ml-child-ipc root instead of colliding on a shared /tmp/ml-child-ipc.
class CScopedTmpDirEnv {
public:
    CScopedTmpDirEnv() {
        char tmpl[] = "/tmp/ml_sandbox_lifecycle_XXXXXX";
        char* dir = ::mkdtemp(tmpl);
        BOOST_TEST_REQUIRE(dir != nullptr);
        m_Dir = dir;
        const char* previous = ::getenv("TMPDIR");
        if (previous != nullptr) {
            m_PreviousTmpDir = previous;
            m_HadPrevious = true;
        }
        ::setenv("TMPDIR", m_Dir.c_str(), 1);
    }
    ~CScopedTmpDirEnv() {
        if (m_HadPrevious) {
            ::setenv("TMPDIR", m_PreviousTmpDir.c_str(), 1);
        } else {
            ::unsetenv("TMPDIR");
        }
        removeTreeBestEffort(m_Dir);
    }
    CScopedTmpDirEnv(const CScopedTmpDirEnv&) = delete;
    CScopedTmpDirEnv& operator=(const CScopedTmpDirEnv&) = delete;
    const std::string& dir() const { return m_Dir; }

private:
    std::string m_Dir;
    std::string m_PreviousTmpDir;
    bool m_HadPrevious{false};
};

//! Creates $TMPDIR/ml-child-ipc/<childId> (mode 0700), matching the layout
//! the native controller is responsible for per design.md, and returns its
//! path.
std::string makeChildIpcRoot(const std::string& trustedTmpDir, const std::string& childId) {
    const std::string mlChildIpc{trustedTmpDir + "/ml-child-ipc"};
    ::mkdir(mlChildIpc.c_str(), 0700); // may already exist from an earlier case in this dir; ignore.
    const std::string childRoot{mlChildIpc + "/" + childId};
    BOOST_TEST_REQUIRE(::mkdir(childRoot.c_str(), 0700) == 0);
    return childRoot;
}

//! One recognized path-bearing launch option is enough to satisfy
//! validateChildIpcLaunchSpec's s_Ok requirement (at least one present and
//! accepted) - the leaf file need not exist on disk (only its parent
//! directory is canonicalized).
std::vector<std::string> childIpcArgs(const std::string& childRoot) {
    return {"--input=" + childRoot + "/input.fifo"};
}

// ---------------------------------------------------------------------
// Test-owned pidfd observation (never signalling) - used only to answer
// "did this PID exit yet", never to terminate a spawner-owned child by
// numeric PID.
// ---------------------------------------------------------------------

int testPidfdOpen(pid_t pid) {
    return static_cast<int>(::syscall(ML_TEST_NR_pidfd_open, pid, 0u));
}

//! Single bounded poll() call (not a sleep/recheck loop): returns true if
//! the pidfd became readable (the process exited) within timeoutMs, false
//! on timeout (process presumably still running).
bool pidfdReadableWithin(int pidfd, int timeoutMs) {
    struct pollfd pfd {};
    pfd.fd = pidfd;
    pfd.events = POLLIN;
    const int rc = ::poll(&pfd, 1, timeoutMs);
    return rc > 0 && (pfd.revents & POLLIN) != 0;
}

// ---------------------------------------------------------------------
// Seam factories. Each mirrors just enough of the corresponding production
// default (see CSandboxedProcessSpawner_Linux.cc's defaultRegistryInsert
// etc.) to keep spawn() on its normal success path, while also handing the
// test a way to observe or control what happened.
// ---------------------------------------------------------------------

//! Registry-insert seam that behaves like the production default (lock,
//! allocate the next generation, insert) and additionally captures a
//! non-owning pointer to the live SPidRegistry plus copies of the inserted
//! entry's pid/pidfd/Sandbox2 handle. Any output parameter may be nullptr
//! if the caller does not need it. The captured SPidRegistry* stays valid
//! for as long as something keeps the underlying shared_ptr<SPidRegistry>
//! alive - normally the owning spawner, and after the owning spawner is
//! destroyed, only the monitor thread's own shared_ptr<SPidRegistry> copy
//! (co-owned per design, LI9) - see the V8 test's own comment for the one
//! place this matters.
TSpawner::TRegistryInsertFn
capturingRegistryInsert(TSpawner::SPidRegistry** capturedRegistry,
                         TPid* capturedPid,
                         int* capturedPidFd,
                         std::shared_ptr<sandbox2::Sandbox2>* capturedSandbox) {
    return [=](TSpawner::SPidRegistry& registry, TPid pid,
               TSpawner::SSandboxedChild child) -> std::uint64_t {
        if (capturedRegistry != nullptr) {
            *capturedRegistry = &registry;
        }
        if (capturedPid != nullptr) {
            *capturedPid = pid;
        }
        if (capturedPidFd != nullptr) {
            *capturedPidFd = child.s_PidFd;
        }
        if (capturedSandbox != nullptr) {
            *capturedSandbox = child.s_Sandbox;
        }
        std::lock_guard<std::mutex> lock(registry.s_Mutex);
        const std::uint64_t generation{++registry.s_NextGeneration};
        child.s_Generation = generation;
        child.s_State = TSpawner::EChildLifecycleState::E_Registered;
        registry.s_Children[pid] = std::move(child);
        return generation;
    };
}

//! pidfd-acquisition seam that ignores the real pidfd_open syscall entirely
//! and returns a fixed, caller-chosen SPidFdAcquisitionResult - used to
//! force ENOSYS/ESRCH/EMFILE/ENFILE/"other" classifications deterministically
//! (V9), independent of what the real, presumably-modern, CI kernel would
//! actually report.
TSpawner::TPidFdOpenFn forcedPidFdOutcome(TSpawner::SPidFdAcquisitionResult toReturn,
                                           TPid* capturedPid = nullptr) {
    return [=](TPid pid) -> TSpawner::SPidFdAcquisitionResult {
        if (capturedPid != nullptr) {
            *capturedPid = pid;
        }
        return toReturn;
    };
}

//! Monitor-launch seam that never starts a thread: it just hands the real
//! monitorBody callable spawn() built (complete with its captured
//! registry/sandbox/generation/awaitResultFn closure) back to the test via
//! capturedBody, and reports success. The test then decides exactly when -
//! or whether - to invoke it, on whatever thread it chooses (usually the
//! test's own calling thread), which is what makes gates 1, 3, 4 and 6
//! below fully deterministic without ever starting a background thread.
TSpawner::TMonitorLaunchFn captureMonitorBodyWithoutRunning(std::function<void()>* capturedBody) {
    return [capturedBody](std::function<void()> body) -> bool {
        *capturedBody = std::move(body);
        return true;
    };
}

//! Monitor-launch seam that DOES start a genuine background thread (like
//! the production default), but additionally signals donePromise once the
//! monitor body - including its registry cleanup - has fully returned, so
//! a test can block deterministically until that has happened without
//! polling or joining the (deliberately detached, per LI9) thread itself.
TSpawner::TMonitorLaunchFn realMonitorLaunchWithCompletionSignal(std::promise<void>* donePromise) {
    return [donePromise](std::function<void()> body) -> bool {
        std::thread([body = std::move(body), donePromise]() mutable {
            body();
            donePromise->set_value();
        }).detach();
        return true;
    };
}

//! AwaitResult seam that always delegates to the real
//! sandbox2::Sandbox2::AwaitResult() (never fabricates a sandbox2::Result -
//! its constructor is not part of any header available in this checkout,
//! see task-4-report.md) and additionally stashes a copy for the test to
//! inspect afterward, since production code only ever uses the result for
//! logging and never exposes it.
TSpawner::TAwaitResultFn capturingAwaitResult(std::shared_ptr<sandbox2::Result>* capturedResult) {
    return [capturedResult](sandbox2::Sandbox2& sandbox) -> sandbox2::Result {
        sandbox2::Result result{sandbox.AwaitResult()};
        if (capturedResult != nullptr) {
            *capturedResult = std::make_shared<sandbox2::Result>(result);
        }
        return result;
    };
}

} // namespace

BOOST_FIXTURE_TEST_SUITE(CSandboxedProcessSpawnerLifecycleTest_Linux, SFdBaselineFixture)

// =====================================================================
// Gate 1 (V9): every pidfd classification.
// =====================================================================

//! classifyPidFdOutcome() is pure and platform-independent (no syscalls, no
//! Sandbox2 types) - exercised exhaustively here with no spawn() at all,
//! covering every classification and a representative errno for each of
//! the two failure buckets, including one genuinely "other" errno (EPERM)
//! that is neither ENOSYS nor one of the two resource-exhaustion examples
//! the brief names (ESRCH/EMFILE/ENFILE all also asserted explicitly).
BOOST_AUTO_TEST_CASE(testClassifyPidFdOutcomeExhaustive) {
    using EOutcome = TSpawner::EPidFdOutcome;
    BOOST_CHECK(TSpawner::classifyPidFdOutcome({3, 0}) == EOutcome::E_Acquired);
    BOOST_CHECK(TSpawner::classifyPidFdOutcome({0, 0}) == EOutcome::E_Acquired);
    BOOST_CHECK(TSpawner::classifyPidFdOutcome({-1, ENOSYS}) == EOutcome::E_KernelUnsupported);
    BOOST_CHECK(TSpawner::classifyPidFdOutcome({-1, ESRCH}) == EOutcome::E_Failed);
    BOOST_CHECK(TSpawner::classifyPidFdOutcome({-1, EMFILE}) == EOutcome::E_Failed);
    BOOST_CHECK(TSpawner::classifyPidFdOutcome({-1, ENFILE}) == EOutcome::E_Failed);
    BOOST_CHECK(TSpawner::classifyPidFdOutcome({-1, EPERM}) == EOutcome::E_Failed); // "other"
}

//! Every non-success, non-ENOSYS classification must fail spawn() outright
//! (MG2/LI8) rather than register a child with an undefined termination
//! fallback. Runs each of ESRCH/EMFILE/ENFILE/EPERM through the real
//! spawn() path via the pidfd seam.
BOOST_AUTO_TEST_CASE(testSpawnFailsClosedOnEveryNonKernelUnsupportedPidfdFailure) {
    const int errnosToTry[] = {ESRCH, EMFILE, ENFILE, EPERM};
    for (int forcedErrno : errnosToTry) {
        CScopedTmpDirEnv tmpEnv;
        const std::string childRoot{
            makeChildIpcRoot(tmpEnv.dir(), std::string("case1-failed-") + std::to_string(forcedErrno))};

        TPid capturedPid{0};
        TSpawner::TPidFdOpenFn pidFdOpen = forcedPidFdOutcome({-1, forcedErrno}, &capturedPid);
        TSpawner spawner{pidFdOpen, TSpawner::TRegistryInsertFn{}, TSpawner::TMonitorLaunchFn{},
                          TSpawner::TAwaitResultFn{}};

        TPid childPid{0};
        const bool spawned =
            spawner.spawn(ML_SANDBOX2_LIFECYCLE_PAYLOAD, childIpcArgs(childRoot), childPid);

        BOOST_TEST_REQUIRE(spawned == false); // negative assertion
        BOOST_CHECK_EQUAL(childPid, 0);
        BOOST_TEST_REQUIRE(capturedPid > 0); // reached marker: the seam was invoked with a real pid
        BOOST_CHECK(spawner.hasChild(capturedPid) == false);

        // Mechanism assertion: no registry entry exists to terminate, so
        // there is nothing to call terminateChild() against, and no pidfd
        // seam was ever consulted a second time. Cleanup assertion: the
        // kill-and-reap guard ran synchronously during spawn()'s stack
        // unwind (before spawn() returned), so the real sandboxee should
        // already be gone - confirm via a test-owned observer pidfd,
        // never a signal.
        const int observerPidFd{testPidfdOpen(capturedPid)};
        BOOST_TEST_REQUIRE(observerPidFd >= 0);
        BOOST_CHECK(pidfdReadableWithin(observerPidFd, 3000));
        ::close(observerPidFd);
    }
}

//! ENOSYS classification: terminateChild() must fall back to
//! Sandbox2::Kill() (hard-coded SIGKILL, uncatchable). There is no seam
//! around Kill() itself (unlike AwaitResult()), so this cannot be verified
//! via a spy on the call - see task-4-report.md. Instead this asserts the
//! only externally observable effect Kill()/SIGKILL and
//! pidfd_send_signal()/SIGTERM can be told apart by: the payload installs a
//! SIGTERM handler that does nothing and keeps running, so only an
//! uncatchable signal can end it - if it dies, SIGKILL (via Kill()) must
//! have been what ended it.
BOOST_AUTO_TEST_CASE(testTerminateChildFallsBackToKillWhenKernelUnsupportsPidfd) {
    CScopedTmpDirEnv tmpEnv;
    const std::string childRoot{makeChildIpcRoot(tmpEnv.dir(), "case1-enosys")};

    TSpawner::SPidRegistry* registry{nullptr};
    std::shared_ptr<sandbox2::Result> capturedResult;
    std::function<void()> monitorBody;

    TSpawner::TPidFdOpenFn pidFdOpen = forcedPidFdOutcome({-1, ENOSYS});
    TSpawner::TRegistryInsertFn insertFn =
        capturingRegistryInsert(&registry, nullptr, nullptr, nullptr);
    TSpawner::TMonitorLaunchFn monitorLaunch = captureMonitorBodyWithoutRunning(&monitorBody);
    TSpawner::TAwaitResultFn awaitResultFn = capturingAwaitResult(&capturedResult);

    TSpawner spawner{pidFdOpen, insertFn, monitorLaunch, awaitResultFn};
    TPid childPid{0};
    BOOST_TEST_REQUIRE(
        spawner.spawn(ML_SANDBOX2_LIFECYCLE_PAYLOAD, childIpcArgs(childRoot), childPid));
    BOOST_TEST_REQUIRE(childPid > 0);
    BOOST_CHECK(spawner.hasChild(childPid)); // positive control / reached marker

    BOOST_TEST_REQUIRE(spawner.terminateChild(childPid));

    BOOST_TEST_REQUIRE(static_cast<bool>(monitorBody));
    monitorBody(); // real cleanup path: calls the (injected) AwaitResult exactly once.

    BOOST_TEST_REQUIRE(capturedResult != nullptr);
    BOOST_CHECK(capturedResult->final_status() == sandbox2::Result::SIGNALED); // mechanism assertion
    BOOST_CHECK(registry->s_Children.count(childPid) == 0); // cleanup assertion
}

//! E_Acquired classification (the un-forced, real-kernel path on any modern
//! CI host): terminateChild() must use pidfd_send_signal(SIGTERM), which
//! the payload's handler catches and survives - the negative assertion
//! (never Kill()/SIGKILL) is that the process is demonstrably still alive
//! afterward.
BOOST_AUTO_TEST_CASE(testTerminateChildUsesPidfdSignalWhenAcquiredAndChildSurvives) {
    CScopedTmpDirEnv tmpEnv;
    const std::string childRoot{makeChildIpcRoot(tmpEnv.dir(), "case1-acquired")};

    TSpawner::SPidRegistry* registry{nullptr};
    int capturedPidFd{-1};
    std::shared_ptr<sandbox2::Sandbox2> capturedSandbox;
    std::shared_ptr<sandbox2::Result> capturedResult;
    std::function<void()> monitorBody;

    TSpawner::TRegistryInsertFn insertFn =
        capturingRegistryInsert(&registry, nullptr, &capturedPidFd, &capturedSandbox);
    TSpawner::TMonitorLaunchFn monitorLaunch = captureMonitorBodyWithoutRunning(&monitorBody);
    TSpawner::TAwaitResultFn awaitResultFn = capturingAwaitResult(&capturedResult);

    // Left as the default (empty) seam: the real kernel's pidfd_open() is
    // expected to succeed (E_Acquired) on any CI host new enough to build
    // Sandbox2 at all - this is the natural, un-forced positive control.
    TSpawner spawner{TSpawner::TPidFdOpenFn{}, insertFn, monitorLaunch, awaitResultFn};
    TPid childPid{0};
    BOOST_TEST_REQUIRE(
        spawner.spawn(ML_SANDBOX2_LIFECYCLE_PAYLOAD, childIpcArgs(childRoot), childPid));
    BOOST_TEST_REQUIRE(childPid > 0);
    BOOST_TEST_REQUIRE(capturedPidFd >= 0); // confirms the real kernel classified E_Acquired

    BOOST_TEST_REQUIRE(spawner.terminateChild(childPid)); // positive control

    // Negative + mechanism assertion, single bounded poll(), not a
    // sleep/recheck loop: the process must still be alive.
    const int observerPidFd{testPidfdOpen(childPid)};
    BOOST_TEST_REQUIRE(observerPidFd >= 0);
    BOOST_CHECK(pidfdReadableWithin(observerPidFd, 1500) == false);
    ::close(observerPidFd);

    // Cleanup: the payload never exits on its own; reap it via the
    // identity-bound Sandbox2 handle (never a numeric ::kill()) and run
    // the real cleanup path.
    BOOST_TEST_REQUIRE(capturedSandbox != nullptr);
    capturedSandbox->Kill();
    BOOST_TEST_REQUIRE(static_cast<bool>(monitorBody));
    monitorBody();
    BOOST_TEST_REQUIRE(capturedResult != nullptr);
    BOOST_CHECK(registry->s_Children.count(childPid) == 0);
}

// =====================================================================
// Gate 2 (V10, LI1, LI8): allocation/resource failure.
// =====================================================================

BOOST_AUTO_TEST_CASE(testRegistryInsertBadAllocKillsAndReapsCleanly) {
    CScopedTmpDirEnv tmpEnv;
    const std::string childRoot{makeChildIpcRoot(tmpEnv.dir(), "case2a")};

    TPid capturedPid{0};
    TSpawner::TPidFdOpenFn pidFdOpen = forcedPidFdOutcome({-1, ENOSYS}, &capturedPid);
    TSpawner::TRegistryInsertFn throwingInsert =
        [](TSpawner::SPidRegistry&, TPid, TSpawner::SSandboxedChild) -> std::uint64_t {
        throw std::bad_alloc();
    };

    TSpawner spawner{pidFdOpen, throwingInsert, TSpawner::TMonitorLaunchFn{},
                      TSpawner::TAwaitResultFn{}};
    TPid childPid{0};
    const bool spawned =
        spawner.spawn(ML_SANDBOX2_LIFECYCLE_PAYLOAD, childIpcArgs(childRoot), childPid);

    BOOST_TEST_REQUIRE(spawned == false);
    BOOST_CHECK_EQUAL(childPid, 0); // LI3
    BOOST_TEST_REQUIRE(capturedPid > 0);
    BOOST_CHECK(spawner.hasChild(capturedPid) == false); // no registry entry

    const int observerPidFd{testPidfdOpen(capturedPid)};
    BOOST_TEST_REQUIRE(observerPidFd >= 0);
    BOOST_CHECK(pidfdReadableWithin(observerPidFd, 3000)); // guard's Kill()+AwaitResult() already ran
    ::close(observerPidFd);
}

BOOST_AUTO_TEST_CASE(testMonitorLaunchFailureKillsAndReapsCleanly) {
    CScopedTmpDirEnv tmpEnv;
    const std::string childRoot{makeChildIpcRoot(tmpEnv.dir(), "case2b")};

    TPid capturedPid{0};
    TSpawner::TPidFdOpenFn pidFdOpen = forcedPidFdOutcome({-1, ENOSYS}, &capturedPid);
    TSpawner::TMonitorLaunchFn alwaysFail = [](std::function<void()>) { return false; };

    // Registry insert left at the production default - it must succeed so
    // this test isolates monitor-launch failure specifically (LI8's other
    // half from case 2a).
    TSpawner spawner{pidFdOpen, TSpawner::TRegistryInsertFn{}, alwaysFail,
                      TSpawner::TAwaitResultFn{}};
    TPid childPid{0};
    const bool spawned =
        spawner.spawn(ML_SANDBOX2_LIFECYCLE_PAYLOAD, childIpcArgs(childRoot), childPid);

    BOOST_TEST_REQUIRE(spawned == false);
    BOOST_CHECK_EQUAL(childPid, 0);
    BOOST_TEST_REQUIRE(capturedPid > 0);
    BOOST_CHECK(spawner.hasChild(capturedPid) == false); // eraseRegistryEntry() ran

    const int observerPidFd{testPidfdOpen(capturedPid)};
    BOOST_TEST_REQUIRE(observerPidFd >= 0);
    BOOST_CHECK(pidfdReadableWithin(observerPidFd, 3000));
    ::close(observerPidFd);
}

// =====================================================================
// Gate 3 (LI6): stale generation must not erase/mutate a newer registration.
// =====================================================================

BOOST_AUTO_TEST_CASE(testStaleMonitorGenerationCannotEraseNewerRegistration) {
    CScopedTmpDirEnv tmpEnv;
    const std::string childRoot{makeChildIpcRoot(tmpEnv.dir(), "case3")};

    TSpawner::SPidRegistry* registry{nullptr};
    std::shared_ptr<sandbox2::Result> capturedResult;
    std::function<void()> monitorBody; // closes over the ORIGINAL (stale) generation.

    TSpawner::TRegistryInsertFn insertFn =
        capturingRegistryInsert(&registry, nullptr, nullptr, nullptr);
    TSpawner::TMonitorLaunchFn monitorLaunch = captureMonitorBodyWithoutRunning(&monitorBody);
    TSpawner::TAwaitResultFn awaitResultFn = capturingAwaitResult(&capturedResult);

    TSpawner spawner{TSpawner::TPidFdOpenFn{}, insertFn, monitorLaunch, awaitResultFn};
    TPid childPid{0};
    BOOST_TEST_REQUIRE(
        spawner.spawn(ML_SANDBOX2_LIFECYCLE_PAYLOAD, childIpcArgs(childRoot), childPid));
    BOOST_TEST_REQUIRE(childPid > 0);
    BOOST_TEST_REQUIRE(registry != nullptr);

    std::uint64_t originalGeneration{0};
    std::uint64_t newerGeneration{0};
    std::shared_ptr<sandbox2::Sandbox2> sandboxHandle;
    {
        std::lock_guard<std::mutex> lock(registry->s_Mutex);
        const auto it = registry->s_Children.find(childPid);
        BOOST_TEST_REQUIRE(it != registry->s_Children.end());
        originalGeneration = it->second.s_Generation;
        sandboxHandle = it->second.s_Sandbox;
        // Simulate a second, newer registration reusing the same numeric
        // PID racing this call's slow first monitor - exactly what
        // defaultRegistryInsert would do for a fresh insert under the same
        // key (bump generation, move to E_Monitoring).
        newerGeneration = ++registry->s_NextGeneration;
        it->second.s_Generation = newerGeneration;
        it->second.s_State = TSpawner::EChildLifecycleState::E_Monitoring;
    }
    BOOST_TEST_REQUIRE(sandboxHandle != nullptr);
    BOOST_TEST_REQUIRE(newerGeneration != originalGeneration);

    // End the real sandboxee so the stale monitor body's (real)
    // AwaitResult() call returns instead of hanging.
    sandboxHandle->Kill();

    BOOST_TEST_REQUIRE(static_cast<bool>(monitorBody));
    monitorBody(); // the STALE monitor, still closed over originalGeneration.

    BOOST_TEST_REQUIRE(capturedResult != nullptr); // reached marker: AwaitResult() did run

    // Negative + cleanup assertion (LI6): the stale monitor must not have
    // erased or mutated the newer entry.
    std::lock_guard<std::mutex> lock(registry->s_Mutex);
    const auto it = registry->s_Children.find(childPid);
    BOOST_TEST_REQUIRE(it != registry->s_Children.end());
    BOOST_CHECK_EQUAL(it->second.s_Generation, newerGeneration);
    BOOST_CHECK(it->second.s_State == TSpawner::EChildLifecycleState::E_Monitoring);
}

// =====================================================================
// Gate 4 (V9, LI7): a stale/expired identity must never let terminateChild()
// signal whatever unrelated process now owns a reused numeric PID.
// =====================================================================

//! There is no seam to force the OS's PID allocator to reuse a specific
//! number deterministically, so this fabricates the reused-PID scenario
//! directly in the registry (the only way to make it deterministic) and
//! proves terminateChild() acts on the CURRENTLY-registered identity's own
//! pidfd - never a numeric kill(pid) - by making that identity a real,
//! test-owned (never spawner-owned) forked process and observing it
//! actually receive the signal via a normal blocking waitpid(), not a
//! numeric ::kill() call anywhere in this file.
BOOST_AUTO_TEST_CASE(testTerminateChildSignalsOnlyTheCurrentlyRegisteredIdentity) {
    CScopedTmpDirEnv tmpEnv;
    const std::string childRoot{makeChildIpcRoot(tmpEnv.dir(), "case4")};

    TSpawner::SPidRegistry* registry{nullptr};
    std::shared_ptr<sandbox2::Result> capturedResultA;
    std::function<void()> monitorBodyA;

    TSpawner::TRegistryInsertFn insertFn =
        capturingRegistryInsert(&registry, nullptr, nullptr, nullptr);
    TSpawner::TMonitorLaunchFn monitorLaunch = captureMonitorBodyWithoutRunning(&monitorBodyA);
    TSpawner::TAwaitResultFn awaitResultFn = capturingAwaitResult(&capturedResultA);

    TSpawner spawner{TSpawner::TPidFdOpenFn{}, insertFn, monitorLaunch, awaitResultFn};
    TPid pidA{0};
    BOOST_TEST_REQUIRE(spawner.spawn(ML_SANDBOX2_LIFECYCLE_PAYLOAD, childIpcArgs(childRoot), pidA));
    BOOST_TEST_REQUIRE(pidA > 0);

    // Reap A for real - end its life and run its own monitor cleanup - so
    // the registry no longer has a live entry for pidA, simulating "the
    // original sandboxee already exited and was reaped".
    std::shared_ptr<sandbox2::Sandbox2> sandboxA;
    {
        std::lock_guard<std::mutex> lock(registry->s_Mutex);
        const auto it = registry->s_Children.find(pidA);
        BOOST_TEST_REQUIRE(it != registry->s_Children.end());
        sandboxA = it->second.s_Sandbox;
    }
    sandboxA->Kill();
    BOOST_TEST_REQUIRE(static_cast<bool>(monitorBodyA));
    monitorBodyA();
    BOOST_CHECK(registry->s_Children.count(pidA) == 0);

    // Fabricate "an unrelated process B now owns pidA's numeric PID": a
    // real, test-owned, throwaway forked process - never spawner-owned, so
    // this is test-fixture setup/teardown, not the thing LI7/the "no
    // numeric ::kill() on a spawner-owned child" constraint is about.
    const pid_t pidB{::fork()};
    BOOST_TEST_REQUIRE(pidB >= 0);
    if (pidB == 0) {
        // Plain test-fixture child: default SIGTERM disposition (terminate)
        // is exactly what this test wants to observe.
        for (;;) {
            ::pause();
        }
    }
    const int pidFdB{testPidfdOpen(pidB)};
    BOOST_TEST_REQUIRE(pidFdB >= 0);

    {
        std::lock_guard<std::mutex> lock(registry->s_Mutex);
        TSpawner::SSandboxedChild fabricated;
        fabricated.s_State = TSpawner::EChildLifecycleState::E_Monitoring;
        fabricated.s_Generation = ++registry->s_NextGeneration;
        fabricated.s_PidFd = pidFdB;
        fabricated.s_PidFdOutcome = TSpawner::EPidFdOutcome::E_Acquired;
        fabricated.s_Outcome = std::make_shared<TSpawner::CCasOutcomeLatch>();
        registry->s_Children[pidA] = std::move(fabricated); // same numeric key A used to own.
    }

    // The call under test, addressed at the numeric PID that used to
    // identify A.
    BOOST_TEST_REQUIRE(spawner.terminateChild(pidA));

    // Mechanism + negative assertion: this must have signalled B via B's
    // OWN pidfd (captured at B's own registration), never a numeric
    // ::kill(pidA, ...) - confirmed by actually observing B die of SIGTERM
    // via a normal blocking waitpid() on the test's own direct child, not
    // polling.
    int status{0};
    BOOST_TEST_REQUIRE(::waitpid(pidB, &status, 0) == pidB);
    BOOST_CHECK(WIFSIGNALED(status) != 0);
    BOOST_CHECK_EQUAL(WTERMSIG(status), SIGTERM);

    ::close(pidFdB);
}

// =====================================================================
// Gate 5 (V11, MG4): timeout-vs-completion race, both interleavings, plus a
// genuine concurrent stress run - all against CCasOutcomeLatch directly (the
// sole coordination primitive design.md assigns this race to). No timeout
// caller exists anywhere in the codebase yet (an accepted, documented gap -
// see task-4-report.md), so there is nothing on the spawn()/monitorBody
// integration side to additionally exercise for this gate.
// =====================================================================

BOOST_AUTO_TEST_CASE(testCasOutcomeLatchResolvesExactlyOnceBothOrderings) {
    using TLatch = TSpawner::CCasOutcomeLatch;
    using EState = TSpawner::EOutcomeState;
    {
        TLatch latch;
        EState completed{EState::E_Completed};
        EState timedOut{EState::E_TimedOut};
        const bool completionWon{latch.tryResolve(completed)};
        const bool timeoutWon{latch.tryResolve(timedOut)};
        BOOST_CHECK(completionWon);
        BOOST_CHECK(timeoutWon == false);
        BOOST_CHECK(timedOut == EState::E_Completed); // loser observes the winner's value
        BOOST_CHECK(latch.load() == EState::E_Completed);
    }
    {
        TLatch latch;
        EState timedOut{EState::E_TimedOut};
        EState completed{EState::E_Completed};
        const bool timeoutWon{latch.tryResolve(timedOut)};
        const bool completionWon{latch.tryResolve(completed)};
        BOOST_CHECK(timeoutWon);
        BOOST_CHECK(completionWon == false);
        BOOST_CHECK(completed == EState::E_TimedOut);
        BOOST_CHECK(latch.load() == EState::E_TimedOut);
    }
}

BOOST_AUTO_TEST_CASE(testCasOutcomeLatchUnderRealConcurrencyResolvesExactlyOnce) {
    using TLatch = TSpawner::CCasOutcomeLatch;
    using EState = TSpawner::EOutcomeState;
    for (int trial = 0; trial < 200; ++trial) {
        TLatch latch;
        std::promise<void> startPromise;
        std::shared_future<void> start{startPromise.get_future()};
        std::atomic<int> completedWins{0};
        std::atomic<int> timedOutWins{0};

        auto race = [&](EState desiredInitial, std::atomic<int>& winCounter) {
            start.wait(); // test-controlled synchronization point, never sleep().
            EState desired{desiredInitial};
            if (latch.tryResolve(desired)) {
                ++winCounter;
            }
        };
        std::thread t1(race, EState::E_Completed, std::ref(completedWins));
        std::thread t2(race, EState::E_TimedOut, std::ref(timedOutWins));
        startPromise.set_value();
        t1.join();
        t2.join();

        // Exactly one side ever wins, regardless of scheduling order - the
        // property MG4/V11 exist to guarantee.
        BOOST_CHECK_EQUAL(completedWins.load() + timedOutWins.load(), 1);
    }
}

// =====================================================================
// Gate 6 (LI4/V10 cleanup): descriptor baseline. SFdBaselineFixture (above)
// already asserts this after every case in this suite; this case names it
// explicitly against one concrete spawn/terminate/cleanup cycle.
// =====================================================================

BOOST_AUTO_TEST_CASE(testDescriptorCountReturnsToBaselineAfterSpawnTerminateCleanup) {
    const std::size_t before{openFdCount()};

    CScopedTmpDirEnv tmpEnv;
    const std::string childRoot{makeChildIpcRoot(tmpEnv.dir(), "case6")};

    std::shared_ptr<sandbox2::Result> capturedResult;
    std::function<void()> monitorBody;
    TSpawner::TPidFdOpenFn pidFdOpen = forcedPidFdOutcome({-1, ENOSYS}); // avoids the SIGTERM-survives hang.
    TSpawner::TMonitorLaunchFn monitorLaunch = captureMonitorBodyWithoutRunning(&monitorBody);
    TSpawner::TAwaitResultFn awaitResultFn = capturingAwaitResult(&capturedResult);

    TSpawner spawner{pidFdOpen, TSpawner::TRegistryInsertFn{}, monitorLaunch, awaitResultFn};
    TPid childPid{0};
    BOOST_TEST_REQUIRE(
        spawner.spawn(ML_SANDBOX2_LIFECYCLE_PAYLOAD, childIpcArgs(childRoot), childPid));
    BOOST_TEST_REQUIRE(childPid > 0);
    BOOST_TEST_REQUIRE(spawner.terminateChild(childPid));
    BOOST_TEST_REQUIRE(static_cast<bool>(monitorBody));
    monitorBody(); // real cleanup path: closes the pidfd, erases the entry.

    BOOST_CHECK_EQUAL(openFdCount(), before);
}

// =====================================================================
// Gate 7: controller-exit orphan behavior - spawner-side half ONLY.
//
// RULING (per the task brief): the orphan-CLEANUP half (does an abandoned
// sandboxee eventually get reaped by something else in the system) is out
// of scope for this unit test and is NOT claimed as covered here - see
// task-4-report.md, which records it as an MG6 accepted-risk candidate for
// the epic/PR-E to close.
// =====================================================================

BOOST_AUTO_TEST_CASE(testDestructorDoesNotJoinAndReturnsUnderOneSecond) {
    CScopedTmpDirEnv tmpEnv;
    const std::string childRoot{makeChildIpcRoot(tmpEnv.dir(), "case7")};

    std::shared_ptr<sandbox2::Sandbox2> capturedSandbox;
    TSpawner::TRegistryInsertFn insertFn =
        capturingRegistryInsert(nullptr, nullptr, nullptr, &capturedSandbox);

    // Real monitor launch (a genuine background thread, like the production
    // default) AND real AwaitResult (left as the default, empty seam): a
    // genuine background thread is blocked in the real AwaitResult() on a
    // genuinely live, never-self-exiting child when the spawner below is
    // destroyed - this is LI9/V8's actual scenario, not a simulation of it.
    // Unlike the plain default monitor-launch seam, this variant also
    // signals monitorDonePromise once that thread's cleanup has fully run,
    // which this test needs afterward to deterministically avoid racing
    // the suite-wide SFdBaselineFixture's end-of-case descriptor count
    // (the real cleanup closes the child's pidfd on that same thread,
    // asynchronously with respect to this test case's own control flow).
    std::promise<void> monitorDonePromise;
    std::future<void> monitorDone{monitorDonePromise.get_future()};
    TSpawner::TMonitorLaunchFn monitorLaunch = realMonitorLaunchWithCompletionSignal(&monitorDonePromise);

    auto spawner = std::make_unique<TSpawner>(TSpawner::TPidFdOpenFn{}, insertFn, monitorLaunch,
                                               TSpawner::TAwaitResultFn{});
    TPid childPid{0};
    BOOST_TEST_REQUIRE(
        spawner->spawn(ML_SANDBOX2_LIFECYCLE_PAYLOAD, childIpcArgs(childRoot), childPid));
    BOOST_TEST_REQUIRE(childPid > 0);
    BOOST_CHECK(spawner->hasChild(childPid)); // reached marker: genuinely running

    const auto start = std::chrono::steady_clock::now();
    spawner.reset(); // ~CSandboxedProcessSpawner() with a live child and a real
                      // monitor thread genuinely blocked in AwaitResult() on it.
    const auto elapsed = std::chrono::steady_clock::now() - start;

    BOOST_CHECK(elapsed < std::chrono::seconds(1)); // V8's explicit bound

    // Test hygiene, not part of the gate 7 assertion itself: reap the
    // still-running sandboxee via its identity-bound Sandbox2 handle
    // (never a numeric ::kill()) so this test process doesn't leave a
    // permanently-blocked monitor thread behind, then block (no polling)
    // until that thread's own cleanup has fully finished, so the next
    // test case's fd-baseline snapshot cannot race this one's cleanup.
    BOOST_TEST_REQUIRE(capturedSandbox != nullptr);
    capturedSandbox->Kill();
    monitorDone.wait();
}

// =====================================================================
// Gate 8 (V8): monitor outlives spawner - destroy the spawner while a
// monitor thread is genuinely still running (blocked on a test-controlled
// gate), release the gate, assert its cleanup runs safely against the
// registry it co-owns via shared_ptr.
// =====================================================================

BOOST_AUTO_TEST_CASE(testMonitorCleanupRunsSafelyAfterSpawnerDestruction) {
    CScopedTmpDirEnv tmpEnv;
    const std::string childRoot{makeChildIpcRoot(tmpEnv.dir(), "case8")};

    std::promise<void> gatePromise;
    std::shared_future<void> gate{gatePromise.get_future()};
    std::promise<void> monitorDonePromise;
    std::future<void> monitorDone{monitorDonePromise.get_future()};

    TSpawner::TAwaitResultFn awaitResultFn = [gate](sandbox2::Sandbox2& sandbox) -> sandbox2::Result {
        gate.wait(); // test-controlled synchronization point - never sleep().
        return sandbox.AwaitResult();
    };
    TSpawner::TMonitorLaunchFn monitorLaunch = realMonitorLaunchWithCompletionSignal(&monitorDonePromise);

    std::shared_ptr<sandbox2::Sandbox2> capturedSandbox;
    TSpawner::SPidRegistry* registryRaw{nullptr};
    TSpawner::TRegistryInsertFn insertFn =
        capturingRegistryInsert(&registryRaw, nullptr, nullptr, &capturedSandbox);

    TPid childPid{0};
    {
        TSpawner spawner{TSpawner::TPidFdOpenFn{}, insertFn, monitorLaunch, awaitResultFn};
        BOOST_TEST_REQUIRE(
            spawner.spawn(ML_SANDBOX2_LIFECYCLE_PAYLOAD, childIpcArgs(childRoot), childPid));
        BOOST_TEST_REQUIRE(childPid > 0);
        BOOST_CHECK(spawner.hasChild(childPid)); // reached marker
    } // spawner destroyed here; the monitor thread is still genuinely blocked on `gate`.

    // registryRaw is only safe to dereference now because the monitor
    // thread's own shared_ptr<SPidRegistry> copy (captured in monitorBody's
    // closure by spawn(), per design) keeps the same object alive - the
    // spawner's own shared_ptr, which is what made this pointer valid
    // originally, is gone. That co-ownership is exactly the property this
    // test exists to exercise.

    // Release the gate and make the sandboxee actually exit, so the
    // now-unblocked real AwaitResult() call inside the monitor thread can
    // return - identity-bound cleanup via the co-owned Sandbox2 handle,
    // never a numeric ::kill().
    BOOST_TEST_REQUIRE(capturedSandbox != nullptr);
    gatePromise.set_value();
    capturedSandbox->Kill();

    // Block (no polling) until the monitor thread's entire body - including
    // its registry cleanup - has fully returned.
    monitorDone.wait();

    // V8/no-crash assertion: reaching this line at all, after the spawner
    // is long gone, is the primary proof. The check below additionally
    // confirms the monitor's registry erase actually ran.
    BOOST_TEST_REQUIRE(registryRaw != nullptr);
    std::lock_guard<std::mutex> lock(registryRaw->s_Mutex);
    BOOST_CHECK(registryRaw->s_Children.count(childPid) == 0);
}

BOOST_AUTO_TEST_SUITE_END()
