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

// Linux-only lifecycle test for CSandboxedProcessSpawner (Task 4). Drives the
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
// configure time, not vendored in this checkout). Open concern: the
// gate-7 orphan-cleanup half (whether an abandoned sandboxee is eventually
// reaped by something else in the system, once the controller exits) is
// deliberately out of scope for this file - see the ruling above gate 7's
// test case.

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

//! Forward-declared here, defined further down once its own helpers
//! (makeChildIpcRoot, forcedPidFdOutcome, ...) are in scope. Starts the
//! lazily-created global Sandbox2 forkserver exactly once, no matter how
//! many test cases construct SFdBaselineFixture, so that its comms
//! descriptors are already open (and therefore already part of the
//! baseline) before the *first* test case's fd count is snapshotted.
void warmUpForkserverOnce();

//! Applied to the whole suite: every test case must leave the process with
//! exactly the descriptors it started with - no leaked pidfd, socketpair, or
//! Sandbox2 comms descriptor survives a spawn/terminate/cleanup cycle.
//!
//! Runs a one-time warm-up spawn (see warmUpForkserverOnce(), defined after
//! the helpers it needs) the first time this fixture is constructed, i.e.
//! for the very first test case that actually runs - never during Boost.Test's
//! own module/framework initialisation. An earlier version did this warm-up
//! via BOOST_GLOBAL_FIXTURE instead: that constructor runs before Boost.Test
//! has finished setting up its own test-tree/observer state, and forking a
//! real process that deep inside framework init corrupted that state (the
//! module reported "Incorrect setup: no test case executed" after every test
//! case had genuinely passed). Doing the warm-up lazily, inside the first
//! ordinary per-case fixture construction, avoids that entirely.
struct SFdBaselineFixture {
    SFdBaselineFixture()
        : s_Baseline((warmUpForkserverOnce(), openFdCount())) {}
    ~SFdBaselineFixture() { BOOST_CHECK_EQUAL(openFdCount(), s_Baseline); }
    std::size_t s_Baseline;
};

// ---------------------------------------------------------------------
// $TMPDIR / child-IPC-root scaffolding, matching the interlock
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
//! the native controller is responsible for creating, and returns its
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
//! (co-owned by design, since the monitor can outlive the spawner) - see
//! gate 8's test case comment for the one place this matters.
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
//! force ENOSYS/ESRCH/EMFILE/ENFILE/"other" classifications deterministically,
//! independent of what the real, presumably-modern, CI kernel would
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
//! polling or joining the (deliberately detached, since the monitor thread
//! must be able to outlive the spawner) thread itself.
//!
//! donePromise is heap-owned (shared_ptr), matching the precedent already
//! established for the ENOSYS case above (capturingAwaitResult's caller):
//! gates 7 and 8 both call spawn() before blocking on the returned future,
//! but if a BOOST_TEST_REQUIRE between spawn() and the wait throws, the
//! stack-local std::promise a raw pointer would have pointed at is
//! destroyed while this detached thread is still running and will later
//! call donePromise->set_value() on freed stack memory - self-review round
//! 2, finding 4.
//!
//! bodyKeepAliveOut is an optional extra output: when non-null, the seam
//! hands the caller its OWN shared_ptr<function<void()>> reference to the
//! monitor closure (self-review round 2, finding 2). Gate 8 needs this: it
//! destroys the spawner and then dereferences a raw SPidRegistry* obtained
//! from this same closure's registry-insert seam. That raw pointer is only
//! valid for as long as SOME shared_ptr<SPidRegistry> copy - normally the
//! monitor closure's own - is still alive. Previously the detached thread
//! destroyed its only copy of the closure immediately after running it and
//! before signalling completion, so by the time gate 8's test thread woke
//! up and dereferenced the raw pointer, the registry had already been
//! freed (a deterministic UAF, not merely racy). Handing the test its own
//! extra reference here means the object survives regardless of when the
//! detached thread releases its own copy. This does not change the
//! fd-baseline story: gates 7/8 already keep the Sandbox2 handle alive via
//! their own `capturedSandbox` copy for the same span, so this reference
//! extends nothing that wasn't already being kept alive.
TSpawner::TMonitorLaunchFn realMonitorLaunchWithCompletionSignal(
    std::shared_ptr<std::promise<void>> donePromise,
    std::shared_ptr<std::function<void()>>* bodyKeepAliveOut = nullptr) {
    return [donePromise, bodyKeepAliveOut](std::function<void()> body) -> bool {
        auto bodyPtr = std::make_shared<std::function<void()>>(std::move(body));
        if (bodyKeepAliveOut != nullptr) {
            *bodyKeepAliveOut = bodyPtr;
        }
        std::thread([bodyPtr, donePromise]() mutable {
            (*bodyPtr)();
            bodyPtr.reset(); // release this thread's reference BEFORE signalling, so a waiter
                // observing "done" is guaranteed this thread no longer holds the
                // closure (and therefore the Sandbox2 handle it captured) - restores
                // the ordering guarantee an earlier round's I5 fix established
                // (self-review round 3, finding S1).
            donePromise->set_value();
        })
            .detach();
        return true;
    };
}

//! AwaitResult seam that always delegates to the real
//! sandbox2::Sandbox2::AwaitResult() (never fabricates a sandbox2::Result -
//! its constructor is not part of any header available in this checkout)
//! and additionally stashes a copy for the test to
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

// ---------------------------------------------------------------------
// I6: forkserver / fork() warm-up, run once before ANY per-case fixture.
// ---------------------------------------------------------------------

//! Sandbox2's global forkserver is created lazily on the first RunAsync()
//! anywhere in this process, and holds its own comms descriptors for the
//! rest of the process's lifetime. SFdBaselineFixture (above) snapshots the
//! fd count before each case's first spawn(); if this test binary/suite
//! ever runs with this suite as the FIRST thing to spawn anything in the
//! whole process (e.g. via `--run_test=` filtering, or a future link-order
//! change), the first case's fd-baseline check would see the forkserver's
//! descriptors appear mid-case and spuriously fail. This is the same root
//! cause as the "gate 4 fork() implicit test-ordering dependency" concern
//! (gate 4 also forks - see testTerminateChildSignalsOnlyTheCurrentlyRegisteredIdentity
//! - and pays the same one-time lazy-init cost the first time anything in
//! this binary spawns or forks) - fixed once, here, for both.
//!
//! Runs once no matter how many test cases construct SFdBaselineFixture:
//! std::call_once guards the actual warm-up spawn behind a static flag, so
//! the real work happens only the first time a test case's fixture runs -
//! never during Boost.Test's own module/framework initialisation. An
//! earlier version did this warm-up in a BOOST_GLOBAL_FIXTURE constructor
//! instead: that constructor runs before Boost.Test has finished setting up
//! its own test-tree/observer state, and forking a real process that deep
//! inside framework init corrupted that state (the module reported
//! "Incorrect setup: no test case executed" after every test case had
//! genuinely passed, even though the run itself succeeded). Running the
//! warm-up lazily, from inside the first ordinary per-case fixture
//! construction, avoids that entirely while keeping the same guarantee:
//! the forkserver's comms descriptors are already open by the time any
//! case's own baseline is captured.
void warmUpForkserverOnce() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        CScopedTmpDirEnv tmpEnv;
        const std::string childRoot{makeChildIpcRoot(tmpEnv.dir(), "forkserver-warmup")};

        std::shared_ptr<sandbox2::Result> capturedResult;
        std::function<void()> monitorBody;
        // ENOSYS forces the Sandbox2::Kill() termination path below (rather
        // than requiring a real pidfd_send_signal/SIGTERM round-trip),
        // keeping this warm-up simple and unconditional regardless of what
        // the real kernel supports.
        TSpawner::TPidFdOpenFn pidFdOpen = forcedPidFdOutcome({-1, ENOSYS});
        TSpawner::TMonitorLaunchFn monitorLaunch =
            captureMonitorBodyWithoutRunning(&monitorBody);
        TSpawner::TAwaitResultFn awaitResultFn = capturingAwaitResult(&capturedResult);

        TSpawner spawner{pidFdOpen, TSpawner::TRegistryInsertFn{}, monitorLaunch, awaitResultFn};
        TPid childPid{0};
        // Best-effort: if this somehow fails, every real test case's own
        // spawn() will surface the underlying problem on its own merits -
        // this warm-up only exists to make the FIRST case's fd baseline
        // deterministic, not to assert anything itself.
        if (spawner.spawn(ML_SANDBOX2_LIFECYCLE_PAYLOAD, childIpcArgs(childRoot), childPid) &&
            childPid > 0) {
            // Only await completion if termination was actually requested
            // successfully: if Sandbox2::Kill() threw and terminateChild()
            // returned false, the sandboxee may still be running, and an
            // unconditional monitorBody() call would block this call -
            // and therefore the first test case that triggers it - inside
            // AwaitResult() with no bound and no diagnostic (production's
            // wall-time limit is unbounded).
            if (spawner.terminateChild(childPid) && monitorBody) {
                monitorBody(); // real cleanup path: closes the pidfd, erases the entry.
            }
        }
    });
}

} // namespace

BOOST_FIXTURE_TEST_SUITE(CSandboxedProcessSpawnerLifecycleTest_Linux, SFdBaselineFixture)

// =====================================================================
// Gate 1: every pidfd classification.
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
//! rather than register a child with an undefined termination
//! fallback. Runs each of ESRCH/EMFILE/ENFILE/EPERM through the real
//! spawn() path via the pidfd seam.
BOOST_AUTO_TEST_CASE(testSpawnFailsClosedOnEveryNonKernelUnsupportedPidfdFailure) {
    const int errnosToTry[] = {ESRCH, EMFILE, ENFILE, EPERM};
    for (int forcedErrno : errnosToTry) {
        CScopedTmpDirEnv tmpEnv;
        const std::string childRoot{makeChildIpcRoot(
            tmpEnv.dir(), std::string("case1-failed-") + std::to_string(forcedErrno))};

        TPid capturedPid{0};
        TSpawner::TPidFdOpenFn pidFdOpen = forcedPidFdOutcome({-1, forcedErrno}, &capturedPid);
        TSpawner spawner{pidFdOpen, TSpawner::TRegistryInsertFn{},
                         TSpawner::TMonitorLaunchFn{}, TSpawner::TAwaitResultFn{}};

        TPid childPid{0};
        const bool spawned = spawner.spawn(ML_SANDBOX2_LIFECYCLE_PAYLOAD,
                                           childIpcArgs(childRoot), childPid);

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
        // I4: a fresh pidfd_open() on a PID the kill-and-reap guard has
        // already Kill()ed and AwaitResult()ed can legitimately fail with
        // ESRCH (fully reaped already - the common case) rather than
        // succeed, so accept both outcomes as proof of cleanup instead of
        // requiring a live pidfd.
        const int observerPidFd{testPidfdOpen(capturedPid)};
        if (observerPidFd < 0) {
            BOOST_CHECK_EQUAL(errno, ESRCH); // already fully reaped - this IS proof of cleanup
        } else {
            BOOST_CHECK(pidfdReadableWithin(observerPidFd, 3000));
            ::close(observerPidFd);
        }
    }
}

//! ENOSYS classification: terminateChild() must fall back to
//! Sandbox2::Kill() (hard-coded SIGKILL, uncatchable). There is no seam
//! around Kill() itself (unlike AwaitResult()), so this cannot be verified
//! via a spy on the call. Instead this asserts the
//! only externally observable effect Kill()/SIGKILL and
//! pidfd_send_signal()/SIGTERM can be told apart by: the payload installs a
//! SIGTERM handler that does nothing and keeps running, so only an
//! uncatchable signal can end it - if it dies, SIGKILL (via Kill()) must
//! have been what ended it.
BOOST_AUTO_TEST_CASE(testTerminateChildFallsBackToKillWhenKernelUnsupportsPidfd) {
    CScopedTmpDirEnv tmpEnv;
    const std::string childRoot{makeChildIpcRoot(tmpEnv.dir(), "case1-enosys")};

    TSpawner::SPidRegistry* registry{nullptr};
    std::function<void()> monitorBody;

    // Heap-owned box for the captured sandbox2::Result, not a plain stack
    // local: monitorBody() below is run with a bounded wait (fixing the
    // review finding that a terminateChild() regression to a no-op would
    // otherwise hang this call forever, since production AwaitResult() has
    // no wall-clock bound of its own - see spawn()'s
    // set_walltime_limit(absl::ZeroDuration()) in
    // CSandboxedProcessSpawner_Linux.cc, and there is no seam to override it
    // for just this test). If the wait times out, the still-running
    // background thread is detached rather than joined (so this test case,
    // and the whole suite, fails fast instead of hanging) - anything that
    // thread can still touch after this function returns must therefore
    // live on the heap, not on this stack frame.
    auto capturedResult = std::make_shared<std::shared_ptr<sandbox2::Result>>();

    TSpawner::TPidFdOpenFn pidFdOpen = forcedPidFdOutcome({-1, ENOSYS});
    TSpawner::TRegistryInsertFn insertFn =
        capturingRegistryInsert(&registry, nullptr, nullptr, nullptr);
    TSpawner::TMonitorLaunchFn monitorLaunch = captureMonitorBodyWithoutRunning(&monitorBody);
    TSpawner::TAwaitResultFn awaitResultFn = capturingAwaitResult(capturedResult.get());

    TSpawner spawner{pidFdOpen, insertFn, monitorLaunch, awaitResultFn};
    TPid childPid{0};
    BOOST_TEST_REQUIRE(spawner.spawn(ML_SANDBOX2_LIFECYCLE_PAYLOAD,
                                     childIpcArgs(childRoot), childPid));
    BOOST_TEST_REQUIRE(childPid > 0);
    BOOST_CHECK(spawner.hasChild(childPid)); // positive control / reached marker

    BOOST_TEST_REQUIRE(spawner.terminateChild(childPid));

    BOOST_TEST_REQUIRE(static_cast<bool>(monitorBody));

    // Run the real cleanup path (calls the injected AwaitResult() exactly
    // once) on a separate thread, bounded by a std::promise/future wait -
    // same synchronization primitive gate 8 already uses in this file, just
    // with a timeout instead of an unconditional wait(), since here nothing
    // else in the test independently guarantees the payload will ever die.
    auto monitorDonePromise = std::make_shared<std::promise<void>>();
    std::future<void> monitorDoneFuture{monitorDonePromise->get_future()};
    std::thread monitorThread(
        [ body = monitorBody, monitorDonePromise, capturedResult ]() mutable {
            body();
            monitorDonePromise->set_value();
        });
    const std::future_status waitStatus{monitorDoneFuture.wait_for(std::chrono::seconds(5))};
    if (waitStatus == std::future_status::ready) {
        monitorThread.join();
    } else {
        // Regression path: terminateChild()'s E_KernelUnsupported branch
        // apparently didn't actually end the payload (e.g. sent the wrong
        // signal, or Kill() regressed to a no-op), so the injected
        // AwaitResult() is still blocked with no bound of its own. Detach
        // instead of join() so this test fails on the assertion below
        // within a few seconds rather than hanging indefinitely - every
        // object the thread can still reach (monitorDonePromise, and
        // monitorBody's own closure, copied above) is heap-owned via
        // shared_ptr/std::function-by-value. Critically, capturedResult
        // (the outer shared_ptr) is ALSO captured by value into this
        // lambda: capturingAwaitResult() only holds a raw pointer into the
        // heap-allocated inner shared_ptr<Result>, baked into
        // monitorBody/awaitResultFn's closure by value, so without a
        // shared_ptr copy of capturedResult riding along in this thread's
        // own capture list, BOOST_TEST_REQUIRE below failing/unwinding this
        // stack frame would drop the last reference and free the object
        // out from under the still-running detached thread - a
        // use-after-free once the real AwaitResult() unblocks and writes
        // through that raw pointer. Capturing capturedResult here keeps it
        // alive for as long as the detached thread might still run,
        // independent of this function's own lifetime.
        monitorThread.detach();
    }
    // Fails fast (instead of hanging) if terminateChild() regressed to
    // never actually killing the payload: a timeout here IS the failure,
    // not a hang. Plain BOOST_REQUIRE, not BOOST_TEST_REQUIRE: the latter
    // tries to stream both operands for its failure message, and
    // std::future_status has no operator<<.
    BOOST_REQUIRE(waitStatus == std::future_status::ready);

    BOOST_TEST_REQUIRE(*capturedResult != nullptr);
    // Self-review round 2, finding 1: Sandbox2::Kill() does not produce a
    // WIFSIGNALED-style SIGNALED/SIGKILL result. It sets the monitor's
    // external-kill flag, and the monitor's status classification (pinned
    // sandboxed-api v20241008, monitor_ptrace.cc) checks that flag AHEAD of
    // the WIFSIGNALED path, so a Kill()ed sandboxee is reported as
    // EXTERNAL_KILL with reason_code() == 0, never SIGNALED/SIGKILL.
    // EXTERNAL_KILL is actually the STRONGER discriminator here: it is only
    // reachable via Sandbox2::Kill(), whereas SIGNALED could also be
    // produced by an external SIGKILL unrelated to this mechanism.
    BOOST_CHECK((*capturedResult)->final_status() == sandbox2::Result::EXTERNAL_KILL); // mechanism: Kill()
    BOOST_CHECK((*capturedResult)->reason_code() == 0);
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
    BOOST_TEST_REQUIRE(spawner.spawn(ML_SANDBOX2_LIFECYCLE_PAYLOAD,
                                     childIpcArgs(childRoot), childPid));
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
// Gate 2: allocation/resource failure.
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
    const bool spawned = spawner.spawn(ML_SANDBOX2_LIFECYCLE_PAYLOAD,
                                       childIpcArgs(childRoot), childPid);

    BOOST_TEST_REQUIRE(spawned == false);
    BOOST_CHECK_EQUAL(childPid, 0); // no live unowned child, no registry entry, no leaked descriptor
    BOOST_TEST_REQUIRE(capturedPid > 0);
    BOOST_CHECK(spawner.hasChild(capturedPid) == false); // no registry entry

    // I4: accept either ESRCH (already fully reaped) or a live-but-exited
    // pidfd as proof the guard's Kill()+AwaitResult() already ran.
    const int observerPidFd{testPidfdOpen(capturedPid)};
    if (observerPidFd < 0) {
        BOOST_CHECK_EQUAL(errno, ESRCH); // already fully reaped - this IS proof of cleanup
    } else {
        BOOST_CHECK(pidfdReadableWithin(observerPidFd, 3000));
        ::close(observerPidFd);
    }
}

BOOST_AUTO_TEST_CASE(testMonitorLaunchFailureKillsAndReapsCleanly) {
    CScopedTmpDirEnv tmpEnv;
    const std::string childRoot{makeChildIpcRoot(tmpEnv.dir(), "case2b")};

    TPid capturedPid{0};
    TSpawner::TPidFdOpenFn pidFdOpen = forcedPidFdOutcome({-1, ENOSYS}, &capturedPid);
    TSpawner::TMonitorLaunchFn alwaysFail = [](std::function<void()>) {
        return false;
    };

    // Registry insert left at the production default - it must succeed so
    // this test isolates monitor-launch failure specifically (the other
    // half of gate 2's failure coverage, alongside case 2a's registry-insert
    // failure).
    TSpawner spawner{pidFdOpen, TSpawner::TRegistryInsertFn{}, alwaysFail,
                     TSpawner::TAwaitResultFn{}};
    TPid childPid{0};
    const bool spawned = spawner.spawn(ML_SANDBOX2_LIFECYCLE_PAYLOAD,
                                       childIpcArgs(childRoot), childPid);

    BOOST_TEST_REQUIRE(spawned == false);
    BOOST_CHECK_EQUAL(childPid, 0);
    BOOST_TEST_REQUIRE(capturedPid > 0);
    BOOST_CHECK(spawner.hasChild(capturedPid) == false); // eraseRegistryEntry() ran

    // I4: accept either ESRCH (already fully reaped) or a live-but-exited
    // pidfd as proof eraseRegistryEntry()/the guard's cleanup already ran.
    const int observerPidFd{testPidfdOpen(capturedPid)};
    if (observerPidFd < 0) {
        BOOST_CHECK_EQUAL(errno, ESRCH); // already fully reaped - this IS proof of cleanup
    } else {
        BOOST_CHECK(pidfdReadableWithin(observerPidFd, 3000));
        ::close(observerPidFd);
    }
}

// =====================================================================
// Gate 3: stale generation must not erase/mutate a newer registration.
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
    BOOST_TEST_REQUIRE(spawner.spawn(ML_SANDBOX2_LIFECYCLE_PAYLOAD,
                                     childIpcArgs(childRoot), childPid));
    BOOST_TEST_REQUIRE(childPid > 0);
    BOOST_TEST_REQUIRE(registry != nullptr);

    std::uint64_t originalGeneration{0};
    std::uint64_t newerGeneration{0};
    std::shared_ptr<sandbox2::Sandbox2> sandboxHandle;
    {
        std::lock_guard<std::mutex> lock(registry->s_Mutex);
        const auto it = registry->s_Children.find(childPid);
        BOOST_REQUIRE(it != registry->s_Children.end()); // BOOST_REQUIRE: map iterators aren't streamable
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

    // Negative + cleanup assertion: the stale monitor must not have
    // erased or mutated the newer entry.
    std::lock_guard<std::mutex> lock(registry->s_Mutex);
    const auto it = registry->s_Children.find(childPid);
    BOOST_REQUIRE(it != registry->s_Children.end()); // BOOST_REQUIRE: map iterators aren't streamable
    BOOST_CHECK_EQUAL(it->second.s_Generation, newerGeneration);
    BOOST_CHECK(it->second.s_State == TSpawner::EChildLifecycleState::E_Monitoring);

    // Self-review round 2, finding 3: this case uses the REAL pidfd path
    // (empty TPidFdOpenFn{}), so the entry above still holds a genuine open
    // pidfd. The stale monitor body correctly skipped closing it (generation
    // mismatch - that skip is exactly what gate 3 asserts above), but that also
    // means nothing else in this case ever closes it: production's
    // defaultRegistryInsert only closes a stale entry's pidfd when a NEWER
    // spawn() replaces it, which never happens in this fabricated scenario.
    // Close it explicitly so SFdBaselineFixture's end-of-case descriptor
    // count matches the suite-wide baseline instead of leaking one fd on
    // every run.
    ::close(it->second.s_PidFd);
}

// =====================================================================
// Gate 4: a stale/expired identity must never let terminateChild()
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
    BOOST_TEST_REQUIRE(spawner.spawn(ML_SANDBOX2_LIFECYCLE_PAYLOAD,
                                     childIpcArgs(childRoot), pidA));
    BOOST_TEST_REQUIRE(pidA > 0);

    // Reap A for real - end its life and run its own monitor cleanup - so
    // the registry no longer has a live entry for pidA, simulating "the
    // original sandboxee already exited and was reaped".
    std::shared_ptr<sandbox2::Sandbox2> sandboxA;
    {
        std::lock_guard<std::mutex> lock(registry->s_Mutex);
        const auto it = registry->s_Children.find(pidA);
        BOOST_REQUIRE(it != registry->s_Children.end()); // BOOST_REQUIRE: map iterators aren't streamable
        sandboxA = it->second.s_Sandbox;
    }
    sandboxA->Kill();
    BOOST_TEST_REQUIRE(static_cast<bool>(monitorBodyA));
    monitorBodyA();
    BOOST_CHECK(registry->s_Children.count(pidA) == 0);

    // Fabricate "an unrelated process B now owns pidA's numeric PID": a
    // real, test-owned, throwaway forked process - never spawner-owned, so
    // this is test-fixture setup/teardown, not the thing the "no
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
// Gate 5: timeout-vs-completion race, both interleavings, plus a
// genuine concurrent stress run - all against CCasOutcomeLatch directly (the
// sole coordination primitive this race is assigned to). No timeout
// caller exists anywhere in the codebase yet (an accepted, documented gap),
// so there is nothing on the spawn()/monitorBody
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
        // property this latch exists to guarantee.
        BOOST_CHECK_EQUAL(completedWins.load() + timedOutWins.load(), 1);
    }
}

// =====================================================================
// Gate 6 (cleanup): descriptor baseline. SFdBaselineFixture (above)
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
    BOOST_TEST_REQUIRE(spawner.spawn(ML_SANDBOX2_LIFECYCLE_PAYLOAD,
                                     childIpcArgs(childRoot), childPid));
    BOOST_TEST_REQUIRE(childPid > 0);
    BOOST_TEST_REQUIRE(spawner.terminateChild(childPid));
    BOOST_TEST_REQUIRE(static_cast<bool>(monitorBody));
    monitorBody(); // real cleanup path: closes the pidfd, erases the entry.

    // monitorBody's closure (built by
    // spawn()'s defaultMonitorLaunch path) captures `sandbox` -
    // shared_ptr<sandbox2::Sandbox2> - BY VALUE and never releases it during
    // execution; invoking the closure does not destroy the closure itself.
    // This `monitorBody` local therefore still keeps the Sandbox2 instance
    // (and its supervisor-side comms socketpair fd, only closed by
    // ~Comms()/~Sandbox2()) alive until it goes out of scope. Release it
    // explicitly here, BEFORE the fd-baseline check, so ~Sandbox2() (and the
    // comms fd close) has already run when openFdCount() is taken.
    monitorBody = nullptr;

    BOOST_CHECK_EQUAL(openFdCount(), before);
}

// =====================================================================
// Gate 7: controller-exit orphan behavior - spawner-side half ONLY.
//
// RULING (per the task brief): the orphan-CLEANUP half (does an abandoned
// sandboxee eventually get reaped by something else in the system) is out
// of scope for this unit test and is NOT claimed as covered here. It is an
// accepted, documented gap left for a future follow-up to close.
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
    // destroyed - this is the monitor-outlives-spawner scenario gate 8 below
    // exercises in full, not a simulation of it.
    // Unlike the plain default monitor-launch seam, this variant also
    // signals monitorDonePromise once that thread's cleanup has fully run,
    // which this test needs afterward to deterministically avoid racing
    // the suite-wide SFdBaselineFixture's end-of-case descriptor count
    // (the real cleanup closes the child's pidfd on that same thread,
    // asynchronously with respect to this test case's own control flow).
    // Heap-owned (shared_ptr), not a stack local, so a BOOST_TEST_REQUIRE
    // throwing before monitorDone.wait() below cannot free this out from
    // under the still-running detached thread (finding 4).
    auto monitorDonePromise = std::make_shared<std::promise<void>>();
    std::future<void> monitorDone{monitorDonePromise->get_future()};
    TSpawner::TMonitorLaunchFn monitorLaunch =
        realMonitorLaunchWithCompletionSignal(monitorDonePromise);

    auto spawner = std::make_unique<TSpawner>(TSpawner::TPidFdOpenFn{}, insertFn, monitorLaunch,
                                              TSpawner::TAwaitResultFn{});
    TPid childPid{0};
    BOOST_TEST_REQUIRE(spawner->spawn(ML_SANDBOX2_LIFECYCLE_PAYLOAD,
                                      childIpcArgs(childRoot), childPid));
    BOOST_TEST_REQUIRE(childPid > 0);
    BOOST_CHECK(spawner->hasChild(childPid)); // reached marker: genuinely running

    const auto start = std::chrono::steady_clock::now();
    spawner.reset(); // ~CSandboxedProcessSpawner() with a live child and a real
                     // monitor thread genuinely blocked in AwaitResult() on it.
    const auto elapsed = std::chrono::steady_clock::now() - start;

    BOOST_CHECK(elapsed < std::chrono::seconds(1)); // destructor must not block on the live child

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
// Gate 8: monitor outlives spawner - destroy the spawner while a
// monitor thread is genuinely still running (blocked on a test-controlled
// gate), release the gate, assert its cleanup runs safely against the
// registry it co-owns via shared_ptr.
// =====================================================================

BOOST_AUTO_TEST_CASE(testMonitorCleanupRunsSafelyAfterSpawnerDestruction) {
    CScopedTmpDirEnv tmpEnv;
    const std::string childRoot{makeChildIpcRoot(tmpEnv.dir(), "case8")};

    std::promise<void> gatePromise;
    std::shared_future<void> gate{gatePromise.get_future()};
    // Heap-owned (shared_ptr), not a stack local, matching gate 7 (finding
    // 4): a throw between spawn() and monitorDone.wait() below must not free
    // this out from under the still-running detached thread.
    auto monitorDonePromise = std::make_shared<std::promise<void>>();
    std::future<void> monitorDone{monitorDonePromise->get_future()};

    TSpawner::TAwaitResultFn awaitResultFn =
        [gate](sandbox2::Sandbox2& sandbox) -> sandbox2::Result {
        gate.wait(); // test-controlled synchronization point - never sleep().
        return sandbox.AwaitResult();
    };
    // Self-review round 2, finding 2: also request the seam's own extra
    // shared_ptr<function<void()>> reference to the monitor closure
    // (monitorBodyKeepAlive), held by this test until after the registryRaw
    // dereference below. Previously the ONLY surviving
    // shared_ptr<SPidRegistry> once the spawner block below exits was the
    // detached thread's own copy inside that closure, and the thread
    // destroyed its copy immediately after running the body and BEFORE
    // signalling monitorDone - so by the time this test woke up from
    // monitorDone.wait() and dereferenced registryRaw, the registry had
    // already been freed (a deterministic use-after-free, not merely
    // racy). Holding monitorBodyKeepAlive here keeps the same object alive
    // regardless of when the detached thread releases its own copy.
    std::shared_ptr<std::function<void()>> monitorBodyKeepAlive;
    TSpawner::TMonitorLaunchFn monitorLaunch =
        realMonitorLaunchWithCompletionSignal(monitorDonePromise, &monitorBodyKeepAlive);

    std::shared_ptr<sandbox2::Sandbox2> capturedSandbox;
    TSpawner::SPidRegistry* registryRaw{nullptr};
    TSpawner::TRegistryInsertFn insertFn =
        capturingRegistryInsert(&registryRaw, nullptr, nullptr, &capturedSandbox);

    TPid childPid{0};
    {
        TSpawner spawner{TSpawner::TPidFdOpenFn{}, insertFn, monitorLaunch, awaitResultFn};
        BOOST_TEST_REQUIRE(spawner.spawn(ML_SANDBOX2_LIFECYCLE_PAYLOAD,
                                         childIpcArgs(childRoot), childPid));
        BOOST_TEST_REQUIRE(childPid > 0);
        BOOST_CHECK(spawner.hasChild(childPid)); // reached marker
    } // spawner destroyed here; the monitor thread is still genuinely blocked on `gate`.

    // registryRaw is safe to dereference below because monitorBodyKeepAlive
    // (captured just above) holds its own shared_ptr<SPidRegistry> reference
    // via the monitor closure - independent of whatever the detached monitor
    // thread's own copy of that same closure does or does not still hold by
    // this point. The spawner's own shared_ptr, which is what made this
    // pointer valid originally, is gone; this test-owned reference is what
    // now keeps the object alive, exercising the same underlying
    // co-ownership property the monitor thread relies on in production.

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

    // No-crash assertion: reaching this line at all, after the spawner
    // is long gone, is the primary proof. The check below additionally
    // confirms the monitor's registry erase actually ran.
    BOOST_TEST_REQUIRE(registryRaw != nullptr);
    std::lock_guard<std::mutex> lock(registryRaw->s_Mutex);
    BOOST_CHECK(registryRaw->s_Children.count(childPid) == 0);
    // monitorBodyKeepAlive is not explicitly reset: it goes out of scope
    // here, after every dereference of registryRaw above, which is all that
    // matters for finding 2. Its (and capturedSandbox's) destruction here
    // still runs on this thread, strictly before SFdBaselineFixture's
    // end-of-case descriptor check, so this does not reintroduce the fd-
    // baseline race the original early-destroy trick (I5) guarded against.
}

BOOST_AUTO_TEST_SUITE_END()
