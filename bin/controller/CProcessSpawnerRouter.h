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
#ifndef INCLUDED_ml_controller_CProcessSpawnerRouter_h
#define INCLUDED_ml_controller_CProcessSpawnerRouter_h

#include <core/CDetachedProcessSpawner.h>
#include <core/CProcess.h>

#include <sandbox/CSandboxedProcessSpawner.h>

#include <memory>
#include <string>
#include <vector>

namespace ml {
namespace controller {

//! \brief
//! Routes an already-decided process spawn request to the Sandbox2 or
//! legacy spawner.
//!
//! DESCRIPTION:\n
//! Unlike the frozen prior-art router this design supersedes, this class
//! never inspects \p args to decide how to route a spawn: the caller (the
//! CCommandProcessor built in a companion task) has already validated any
//! operator kill-switch token and decided the route before calling spawn().
//! This router's only job is to dispatch that already-decided route to the
//! right backend and enforce the fail-closed rules around Sandbox2
//! availability - it must never re-derive the route or retry a failed
//! Sandbox2 launch through the legacy spawner.
//!
//! Processes listed in sandboxedProcessPaths are routed to Sandbox2 when
//! the route is E_Sandbox2 and this build has Sandbox2 support; all other
//! permitted processes - and any explicit E_Legacy route - use the legacy
//! (posix_spawn-based) spawner.
//!
class CProcessSpawnerRouter {
public:
    using TStrVec = std::vector<std::string>;

    //! The route a spawn() call has already been assigned, decided upstream
    //! of this class (by CCommandProcessor). This router never derives a
    //! route itself from \p args or from \p processPath alone.
    enum class ERoute {
        //! Use Sandbox2 for processes listed in sandboxedProcessPaths (when
        //! this build has Sandbox2 support); every other permitted process
        //! is unaffected and always goes via the legacy spawner, exactly
        //! like today's CDetachedProcessSpawner-only paths.
        E_Sandbox2,
        //! Operator kill-switch route: the caller has already validated the
        //! disableSandbox token against this exact processPath and stripped
        //! it from args. Always dispatches to the legacy spawner.
        E_Legacy
    };

    //! Why the caller chose ERoute::E_Legacy. The router never derives this
    //! (it never re-parses args): CCommandProcessor passes the provenance it
    //! already knows from making the decision, purely so the
    //! `sandbox2_launch` signal's additive "legacy_reason" field can
    //! distinguish a deliberate operator
    //! kill switch from the dormant default that is in effect for the whole
    //! rollout window - mode == "degraded" alone cannot.
    enum class ELegacyReason {
        //! The route is E_Sandbox2; no legacy_reason is emitted at all.
        E_NotLegacy,
        //! A validated --disableSandbox token was present.
        E_KillSwitch,
        //! No token, and ML_SANDBOX2_DEFAULT_ENFORCED is not enabled.
        E_DormantDefault
    };

public:
    CProcessSpawnerRouter(const TStrVec& permittedProcessPaths,
                          const TStrVec& sandboxedProcessPaths);

    //! Dispatch a spawn request per the already-decided \p route. Returns
    //! false immediately on a Sandbox2 failure - never retries via the
    //! legacy spawner ("no automatic fallback").
    //! \param legacyReason provenance of an E_Legacy \p route, for the
    //!        `sandbox2_launch` signal only - never used to dispatch. Must be E_NotLegacy
    //!        (the default) when \p route is E_Sandbox2.
    bool spawn(ERoute route,
               const std::string& processPath,
               const TStrVec& args,
               core::CProcess::TPid& childPid,
               ELegacyReason legacyReason = ELegacyReason::E_NotLegacy);

    //! Terminate a child previously spawned by either backend.
    bool terminateChild(core::CProcess::TPid pid);

    //! \return true if either backend owns a still-live child with this PID.
    bool hasChild(core::CProcess::TPid pid) const;

    //! \return true if \p processPath is configured as a sandboxed process
    //! path. This is the single implementation of that predicate: the router
    //! uses it for dispatch and `sandbox2_launch`-signal gating, and CCommandProcessor
    //! calls it (through its own router member) to decide whether the
    //! operator kill-switch token is meaningful for a process path and
    //! whether the dormant-by-default Sandbox2 route applies. Keeping two
    //! independent std::find copies would let a future change to one (e.g.
    //! path normalisation) silently desync token validation from signal
    //! emission.
    bool isSandboxedProcessPath(const std::string& processPath) const;

private:
    //! Emit the `sandbox2_launch` structured once-per-launch signal for a
    //! Sandbox2-eligible spawn() call,
    //! after the dispatch outcome is known. Fires on every outcome,
    //! including \p spawnSucceeded == false (the fail_closed case) - never
    //! gated behind the caller's own success handling. Must only be called
    //! when the process path is a configured sandboxed process path; never
    //! for unrelated processes (e.g. autodetect).
    //! \param deploymentId SChildIpcLaunchSpec::s_ChildId, already derived
    //!        once by spawn() *before* dispatch - never re-derived here, so
    //!        the value in this signal cannot disagree with the value the
    //!        dispatch decision was made against.
    void emitLaunchSignal(ERoute route,
                          ELegacyReason legacyReason,
                          const std::string& deploymentId,
                          const TStrVec& args,
                          bool spawnSucceeded) const;

private:
    core::CDetachedProcessSpawner m_LegacySpawner;

    //! Null until - and unless - a spawn() call actually dispatches to the
    //! Sandbox2 route, at which point spawn() creates it in place (see the
    //! .cc's SANDBOX2_AVAILABLE branch). A router that only ever takes the
    //! legacy route - which is every router during the whole dormant-default
    //! rollout window, and every router in a non-Sandbox2 build - therefore
    //! never constructs *or* destructs any Sandbox2 machinery.
    //!
    //! Held behind a pointer rather than by value for two reasons:
    //!
    //! 1. Lifecycle: constructing Sandbox2 state (a PID registry with its
    //!    own mutex, and, in future tasks, forkserver/monitor resources) for
    //!    a router that will never launch a sandboxed process is pure
    //!    liability - it puts Sandbox2 objects into the construction and
    //!    teardown path of every controller and of every controller unit
    //!    test, including the ones that predate Sandbox2 entirely.
    //! 2. ODR safety: sizeof(sandbox::CSandboxedProcessSpawner) *differs*
    //!    between translation units compiled with and without
    //!    SANDBOX2_AVAILABLE, because its m_AwaitResultFn seam only exists
    //!    under that macro (include/sandbox/CSandboxedProcessSpawner.h). A
    //!    by-value member propagated that difference into
    //!    sizeof(CProcessSpawnerRouter) and sizeof(CCommandProcessor), so
    //!    any binary that mixed the two views of this header - as
    //!    ml_test_controller did on Linux - had inline constructors and
    //!    destructors disagreeing about member offsets and corrupted memory
    //!    at teardown. std::unique_ptr is the same size either way, so this
    //!    class's layout no longer depends on the macro at all. (The
    //!    underlying macro mismatch is fixed in bin/controller/CMakeLists.txt
    //!    as well; this member simply stops the layout being sensitive to
    //!    it.)
    //!
    //! Not synchronised: like m_LegacySpawner's own contract, every router
    //! entry point is called from the controller's single
    //! command-processing thread (bin/controller/CCommandProcessor.cc), so
    //! the lazy creation below needs no lock.
    std::unique_ptr<sandbox::CSandboxedProcessSpawner> m_SandboxSpawner;

    TStrVec m_SandboxedProcessPaths;
};

} // namespace controller
} // namespace ml

#endif // INCLUDED_ml_controller_CProcessSpawnerRouter_h
