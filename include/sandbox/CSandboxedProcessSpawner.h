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
//! terminateChild() call is still in flight (design.md §"Spawn lifecycle
//! and ownership", LI7). The lifecycle below is the explicit state machine
//! every live registry entry moves through; see design.md's mermaid
//! diagram for the full transition set. This header declares the state
//! shape and public API only - spawn()'s kill-and-reap guard, injectable
//! seams, and pidfd outcome classification land in later tasks of
//! docs/projects/mlcpp-sandbox2-pr2873/pr-d-lifecycle.plan.md.
class CSandboxedProcessSpawner {
public:
    using TStrVec = std::vector<std::string>;

    //! Explicit lifecycle states a registry entry moves through, mirroring
    //! design.md's mermaid diagram one-for-one. No state is skipped and no
    //! state is inferred from a combination of booleans.
    enum class EChildLifecycleState {
        E_Prepared,             //!< Launch spec validated; process not yet started.
        E_Launched,             //!< Sandbox2::RunAsync() succeeded; pid() not yet captured.
        E_IdentityCaptured,     //!< pid() captured; kill-and-reap guard armed (LI1).
        E_Registered,           //!< Registry insertion succeeded.
        E_Monitoring,           //!< Monitor thread handoff succeeded; guard disarmed (LI2).
        E_TerminationRequested, //!< terminateChild() issued a request; child not yet confirmed exited.
        E_CleanupRequired,      //!< Sandbox2 completion observed; registry entry pending removal.
        E_Reaped,               //!< AwaitResult() returned; every descriptor closed exactly once (LI4).
        E_Failed                //!< spawn() failed at or after this state; no live unowned child remains.
    };

    //! One-shot outcome of the timeout-vs-completion race (design.md
    //! MG4/V11), replacing independent-boolean coordination with a single
    //! atomic latch. Exactly one of TimedOut/Completed wins via
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
            return m_State.compare_exchange_strong(expected, desired) ? true
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

public:
    CSandboxedProcessSpawner();
    ~CSandboxedProcessSpawner();

    //! Spawn a sandboxed process. Returns true only after registry
    //! insertion and monitor handoff both succeed (LI2); on any other
    //! outcome returns false with childPid left at 0 and no live unowned
    //! child, no registry entry, and no leaked descriptor (LI3).
    bool spawn(const std::string& processPath, const TStrVec& args, core::CProcess::TPid& childPid);

    //! Request termination of a sandboxed child previously started by this
    //! object, targeting its identity-bound handle rather than a recycled
    //! numeric PID (LI7).
    bool terminateChild(core::CProcess::TPid pid);

    //! \return true if this object owns a sandboxed child with the given
    //! PID that is still live (not yet Reaped or Failed).
    bool hasChild(core::CProcess::TPid pid) const;

private:
    //! \brief A live sandboxed child and the handles needed to manage it
    //! safely through every lifecycle state.
    //!
    //! DESCRIPTION:\n
    //! Shape only in this task - no lifecycle logic lands here yet. Carries
    //! the explicit state, a monotonic generation (so a stale monitor
    //! cannot erase or mutate a newer registration racing the same PID,
    //! LI6), the Sandbox2 handle (co-owned with any monitor thread via
    //! shared_ptr, since a monitor can outlive this spawner and must never
    //! hold a raw pointer back into it, LI5), the pidfd used for
    //! identity-bound termination when the kernel provides one, and the
    //! one-shot outcome latch used to resolve a timeout-vs-completion race
    //! for this specific child (MG4/V11).
    struct SSandboxedChild {
        EChildLifecycleState s_State{EChildLifecycleState::E_Prepared};
        std::uint64_t s_Generation{0};
        std::shared_ptr<sandbox2::Sandbox2> s_Sandbox;
        int s_PidFd{-1};
        std::shared_ptr<CCasOutcomeLatch> s_Outcome;
    };

    //! \brief The live sandboxed children, and the lock that guards them.
    //!
    //! DESCRIPTION:\n
    //! Held behind a shared_ptr because a monitor thread that removes a
    //! child outlives the spawn() call that started it, and can outlive
    //! this object: the controller may tear the spawner down while a
    //! sandboxed pytorch_inference is still running (LI9), and the monitor
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

    const TPidRegistryPtr m_PidRegistry{std::make_shared<SPidRegistry>()};
};

} // namespace sandbox
} // namespace ml

#endif // INCLUDED_ml_sandbox_CSandboxedProcessSpawner_h
