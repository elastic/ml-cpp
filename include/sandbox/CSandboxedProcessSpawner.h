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
#ifndef INCLUDED_ml_sandbox_CSandboxedProcessSpawner_h
#define INCLUDED_ml_sandbox_CSandboxedProcessSpawner_h

#include <core/CProcess.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Sandbox2 headers are unavailable on non-Linux configure runs (see
// include/sandbox/CPytorchInferenceSandboxPolicy.h). Only a forward
// declaration is needed here: this header stores sandbox2::Sandbox2 solely
// behind a shared_ptr, never by value, so non-Linux builds never need the
// real type.
namespace sandbox2 {
class Sandbox2;
}

#ifdef SANDBOX2_AVAILABLE
// The Sandbox2-completion injectable seam (TAwaitResultFn, below) names
// sandbox2::Result in a std::function signature, which needs the complete
// type - the forward declaration above is not enough for that one seam.
// Non-Linux/no-Sandbox2 configures never see this include, matching
// include/sandbox/CPytorchInferenceSandboxPolicy.h's pattern for the same
// reason.
#include <sandboxed_api/sandbox2/sandbox2.h>
#endif

namespace ml {
namespace sandbox {

//! \brief
//! Spawn and own the lifecycle of processes inside a Sandbox2 isolation
//! boundary.
//!
//! DESCRIPTION:\n
//! Replaces numeric-PID process control (core::CDetachedProcessSpawner's
//! model) with identity-bound handles, because a sandboxed child's PID can
//! be reused by an unrelated process while a stale monitor or a delayed
//! terminateChild() call is still in flight. The lifecycle below is the
//! explicit state machine every live registry entry moves through:
//! Prepared -> Launched -> IdentityCaptured -> Registered -> Monitoring ->
//! Reaped is the full happy-path transition set, with
//! TerminationRequested/CleanupRequired/Failed as the additional states a
//! termination request or a failure path can move through. This header
//! declares the state shape and public API only - spawn()'s kill-and-reap
//! guard, injectable seams, and pidfd outcome classification are
//! implemented in CSandboxedProcessSpawner_Linux.cc.
class CSandboxedProcessSpawner {
public:
    using TStrVec = std::vector<std::string>;

    //! Explicit lifecycle states a registry entry moves through: Prepared ->
    //! Launched -> IdentityCaptured -> Registered -> Monitoring -> Reaped.
    //! E_IdentityCaptured marks the point where pid() has been captured but
    //! the child is not yet registered - a distinct state from E_Launched
    //! because the kill-and-reap guard must be armed as soon as pid() is
    //! known, before registration, not folded into a coarser "Launched"
    //! state. No state is skipped and no state is inferred from a
    //! combination of booleans.
    enum class EChildLifecycleState {
        E_Prepared, //!< Launch spec validated; process not yet started.
        E_Launched, //!< Sandbox2::RunAsync() succeeded; pid() not yet captured.
        E_IdentityCaptured, //!< pid() captured; kill-and-reap guard armed so a failure from
                            //!< here on cannot leave a live, unowned child.
        E_Registered, //!< Registry insertion succeeded.
        E_Monitoring, //!< Monitor thread handoff succeeded; guard disarmed because the
                      //!< monitor now owns reaping the child.
        E_TerminationRequested, //!< terminateChild() issued a request; child not yet confirmed exited.
        E_CleanupRequired, //!< Sandbox2 completion observed; registry entry pending removal.
        E_Reaped, //!< AwaitResult() returned; every descriptor closed exactly once.
        E_Failed //!< spawn() failed at or after this state; no live unowned child remains.
    };

    //! One-shot outcome of the timeout-vs-completion race, replacing
    //! independent-boolean coordination with a single atomic latch.
    //! Exactly one of TimedOut/Completed wins via
    //! compare_exchange_strong from Pending; the loser observes the
    //! winner's value and must not perform cleanup.
    enum class EOutcomeState { E_Pending, E_TimedOut, E_Completed };

    //! \brief One-shot CAS latch: Pending -> TimedOut|Completed, never back.
    //!
    //! DESCRIPTION:\n
    //! The only coordination mechanism between a timeout path and a
    //! Sandbox2-completion path racing to decide who performs cleanup for
    //! the same child. A single compare_exchange_strong call decides the
    //! winner; the loser's compare_exchange_strong fails and returns the
    //! value the winner set, so it can branch without a second flag.
    class CCasOutcomeLatch {
    public:
        CCasOutcomeLatch() = default;

        CCasOutcomeLatch(const CCasOutcomeLatch&) = delete;
        CCasOutcomeLatch& operator=(const CCasOutcomeLatch&) = delete;

        //! Attempt to move the latch from Pending to \p desired. Returns
        //! true iff this call won the race (the latch was Pending and is
        //! now \p desired); false means some call - possibly this one on a
        //! retry, possibly a racing call - already set it to another value,
        //! which is written back into \p desired for the caller to inspect.
        bool tryResolve(EOutcomeState& desired) {
            EOutcomeState expected{EOutcomeState::E_Pending};
            return m_State.compare_exchange_strong(expected, desired)
                       ? true
                       : (desired = expected, false);
        }

        //! \return the latch's current value. For diagnostics only - never
        //! branch cleanup logic on a load() result instead of tryResolve()'s
        //! own return value, or the check-then-act gap reintroduces the
        //! two-boolean race this latch replaces.
        EOutcomeState load() const { return m_State.load(); }

    private:
        std::atomic<EOutcomeState> m_State{EOutcomeState::E_Pending};
    };

    //! Raw outcome of the injectable pidfd-acquisition seam: the fd returned
    //! by pidfd_open (or -1) and errno on failure. classifyPidFdOutcome()
    //! maps this to EPidFdOutcome.
    struct SPidFdAcquisitionResult {
        int s_Fd{-1};
        int s_Errno{0};
    };

    //! Three-way classification of a pidfd-acquisition attempt: whether
    //! spawn() registers the child at all, and which terminateChild()
    //! mechanism applies for a registered child.
    //!
    //! E_Acquired: s_Fd >= 0. terminateChild() sends a request via
    //! pidfd_send_signal(SIGTERM) on the held pidfd.
    //!
    //! E_KernelUnsupported: s_Fd < 0 and s_Errno == ENOSYS - the running
    //! kernel predates pidfd support entirely (pre-5.3). This is the *only*
    //! classification for which terminateChild() falls back to
    //! Sandbox2::Kill() (SIGKILL via the owned monitor, identity-safe, no
    //! numeric-PID lookup). Recorded on the registry entry at registration
    //! time - terminateChild() must use that recorded value, never
    //! re-derive it by re-calling pidfd_open.
    //!
    //! E_Failed: s_Fd < 0 and s_Errno is anything else (ESRCH, EMFILE,
    //! ENFILE, ...). This is a resource or identity error, not "no kernel
    //! support" - it must never be treated the same
    //! as E_KernelUnsupported. spawn() fails registration outright on this
    //! outcome rather than registering a child whose termination would need
    //! an undefined fallback.
    enum class EPidFdOutcome { E_Acquired, E_KernelUnsupported, E_Failed };

    //! Pure classification of SPidFdAcquisitionResult: no syscalls or side
    //! effects. Implemented outside the SANDBOX2_AVAILABLE block so it
    //! compiles and is unit-testable on every platform.
    static EPidFdOutcome classifyPidFdOutcome(const SPidFdAcquisitionResult& result);

public:
    //! \brief A live sandboxed child and the handles needed to manage it
    //! safely through every lifecycle state.
    //!
    //! Public so TRegistryInsertFn and test seams can name this type.
    //! s_Generation lets a stale monitor ignore a newer registration on
    //! the same numeric PID. s_Sandbox is co-owned with the monitor thread
    //! via shared_ptr (the monitor can outlive this spawner).
    struct SSandboxedChild {
        EChildLifecycleState s_State{EChildLifecycleState::E_Prepared};
        std::uint64_t s_Generation{0};
        std::shared_ptr<sandbox2::Sandbox2> s_Sandbox;
        int s_PidFd{-1};
        //! Classification recorded at registration time. Only E_Acquired
        //! and E_KernelUnsupported reach the registry; default is fail-closed.
        EPidFdOutcome s_PidFdOutcome{EPidFdOutcome::E_Failed};
        std::shared_ptr<CCasOutcomeLatch> s_Outcome;
    };

    //! \brief The live sandboxed children, and the lock that guards them.
    //!
    //! DESCRIPTION:\n
    //! Held behind a shared_ptr because a monitor thread that removes a
    //! child outlives the spawn() call that started it, and can outlive
    //! this object: the controller may tear the spawner down while a
    //! sandboxed pytorch_inference is still running, and the monitor
    //! only learns that the sandboxee exited some time later. A raw pointer
    //! back to the spawner would be dangling by then, so the monitor
    //! co-owns the registry instead, and the spawner's destructor needs no
    //! synchronisation with in-flight monitors.
    struct SPidRegistry {
        mutable std::mutex s_Mutex;
        std::uint64_t s_NextGeneration{0};
        std::map<core::CProcess::TPid, SSandboxedChild> s_Children;
    };
    using TPidRegistryPtr = std::shared_ptr<SPidRegistry>;

    //! Injectable seams for tests. Each has a production default; an empty
    //! std::function selects it.

    //! pidfd-acquisition seam: wraps the pidfd_open syscall.
    using TPidFdOpenFn = std::function<SPidFdAcquisitionResult(core::CProcess::TPid)>;

    //! Registry-allocation seam: performs the locked map insertion
    //! (replacing any stale entry for the same PID, mirroring the
    //! production default) and returns the new entry's generation. The
    //! production default never throws for ordinary insertion; a test
    //! overriding this seam can throw std::bad_alloc, or return a
    //! deliberately colliding generation, to exercise those failure and
    //! collision paths deterministically without waiting on real resource
    //! exhaustion.
    using TRegistryInsertFn =
        std::function<std::uint64_t(SPidRegistry&, core::CProcess::TPid, SSandboxedChild)>;

    //! Monitor-thread creation/detach seam. Returns false - never throws -
    //! if std::thread construction or detach() failed, so a test can force
    //! that failure deterministically without depending on the OS
    //! actually running out of threads. The production default constructs
    //! std::thread(monitorBody) and detaches it, converting any
    //! std::system_error from either step into a false return.
    using TMonitorLaunchFn = std::function<bool(std::function<void()> monitorBody)>;

#ifdef SANDBOX2_AVAILABLE
    //! Sandbox2-completion seam: wraps AwaitResult() so tests control when
    //! and what result is reported. Available only where sandbox2::Result is
    //! a complete type (SANDBOX2_AVAILABLE).
    using TAwaitResultFn = std::function<sandbox2::Result(sandbox2::Sandbox2&)>;
#endif

    CSandboxedProcessSpawner();

    //! Test-only constructor injecting the four seams above. Each parameter
    //! defaults to an empty std::function; spawn()
    //! (CSandboxedProcessSpawner_Linux.cc) treats an empty seam as "use the
    //! production behaviour", so production callers should keep using the
    //! plain default constructor and never need to name these types.
    CSandboxedProcessSpawner(TPidFdOpenFn pidFdOpenFn,
                             TRegistryInsertFn registryInsertFn,
                             TMonitorLaunchFn monitorLaunchFn
#ifdef SANDBOX2_AVAILABLE
                             ,
                             TAwaitResultFn awaitResultFn
#endif
    );

    ~CSandboxedProcessSpawner();

    //! Spawn a sandboxed process. Returns true only after registry
    //! insertion and monitor handoff both succeed; on any other
    //! outcome returns false with childPid left at 0 and no live unowned
    //! child, no registry entry, and no leaked descriptor.
    bool spawn(const std::string& processPath, const TStrVec& args, core::CProcess::TPid& childPid);

    //! Request termination of a sandboxed child previously started by this
    //! object, targeting its identity-bound handle rather than a recycled
    //! numeric PID.
    bool terminateChild(core::CProcess::TPid pid);

    //! \return true if this object owns a sandboxed child with the given
    //! PID that is still live (not yet Reaped or Failed).
    bool hasChild(core::CProcess::TPid pid) const;

private:
    const TPidRegistryPtr m_PidRegistry{std::make_shared<SPidRegistry>()};

    //! Seam storage for the test-only constructor. Left empty (default
    //! std::function) by the plain default constructor, which
    //! CSandboxedProcessSpawner_Linux.cc reads as "use the production
    //! behaviour" for every seam.
    TPidFdOpenFn m_PidFdOpenFn;
    TRegistryInsertFn m_RegistryInsertFn;
    TMonitorLaunchFn m_MonitorLaunchFn;
#ifdef SANDBOX2_AVAILABLE
    TAwaitResultFn m_AwaitResultFn;
#endif
};

} // namespace sandbox
} // namespace ml

#endif // INCLUDED_ml_sandbox_CSandboxedProcessSpawner_h
