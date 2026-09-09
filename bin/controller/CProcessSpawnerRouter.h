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

public:
    CProcessSpawnerRouter(const TStrVec& permittedProcessPaths,
                          const TStrVec& sandboxedProcessPaths);

    //! Dispatch a spawn request per the already-decided \p route. Returns
    //! false immediately on a Sandbox2 failure - never retries via the
    //! legacy spawner (V2, "no automatic fallback").
    bool spawn(ERoute route,
               const std::string& processPath,
               const TStrVec& args,
               core::CProcess::TPid& childPid);

    //! Terminate a child previously spawned by either backend.
    bool terminateChild(core::CProcess::TPid pid);

    //! \return true if either backend owns a still-live child with this PID.
    bool hasChild(core::CProcess::TPid pid) const;

    //! \return true if \p processPath is configured as a sandboxed process
    //! path. This is the single implementation of that predicate: the router
    //! uses it for dispatch and H4-signal gating, and CCommandProcessor
    //! calls it (through its own router member) to decide whether the
    //! operator kill-switch token is meaningful for a process path and
    //! whether the dormant-by-default Sandbox2 route applies. Keeping two
    //! independent std::find copies would let a future change to one (e.g.
    //! path normalisation) silently desync token validation from signal
    //! emission.
    bool isSandboxedProcessPath(const std::string& processPath) const;

private:
    //! Emit the H4 structured once-per-launch signal (design.md §Failure
    //! behavior and observability) for a Sandbox2-eligible spawn() call,
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
                          const std::string& deploymentId,
                          const TStrVec& args,
                          bool spawnSucceeded) const;

private:
    core::CDetachedProcessSpawner m_LegacySpawner;

    //! Always present: CSandboxedProcessSpawner compiles - and is safely
    //! constructible/queryable - on every platform (see
    //! lib/sandbox/CSandboxedProcessSpawner_Linux.cc), so no #ifdef is
    //! needed around this member's declaration. Its spawn()/terminateChild()
    //! are only ever *called* from this router behind an explicit
    //! SANDBOX2_AVAILABLE check - see the .cc.
    sandbox::CSandboxedProcessSpawner m_SandboxSpawner;

    TStrVec m_SandboxedProcessPaths;
};

} // namespace controller
} // namespace ml

#endif // INCLUDED_ml_controller_CProcessSpawnerRouter_h
