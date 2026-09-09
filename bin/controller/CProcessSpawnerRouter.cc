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
#include "CProcessSpawnerRouter.h"

#include <core/CLogger.h>

#include <algorithm>

namespace ml {
namespace controller {

CProcessSpawnerRouter::CProcessSpawnerRouter(const TStrVec& permittedProcessPaths,
                                              const TStrVec& sandboxedProcessPaths)
    : m_LegacySpawner{permittedProcessPaths}, m_SandboxedProcessPaths{sandboxedProcessPaths} {
}

bool CProcessSpawnerRouter::isSandboxedProcessPath(const std::string& processPath) const {
    return std::find(m_SandboxedProcessPaths.begin(), m_SandboxedProcessPaths.end(), processPath) !=
           m_SandboxedProcessPaths.end();
}

bool CProcessSpawnerRouter::spawn(ERoute route,
                                   const std::string& processPath,
                                   const TStrVec& args,
                                   core::CProcess::TPid& childPid) {
    if (route == ERoute::E_Legacy) {
        // Operator kill-switch route: the caller has already validated the
        // --disableSandbox token against this exact processPath and
        // stripped it from args before this call - this router never
        // re-parses args to decide anything (unlike the frozen prior art's
        // spawn(), which re-derived disableSandbox from args itself).
        LOG_INFO(<< "Launching '" << processPath
                 << "' without Sandbox2 (operator kill switch --disableSandbox); "
                 << "the in-process seccomp filter applies");
        return m_LegacySpawner.spawn(processPath, args, childPid);
    }

    // route == ERoute::E_Sandbox2: dispatch on whether processPath is
    // configured as sandboxed, not on anything derived from args.
    if (this->isSandboxedProcessPath(processPath)) {
#ifdef SANDBOX2_AVAILABLE
        // No automatic fallback to the legacy spawner on a Sandbox2
        // failure (V2, MG1): a process that must be sandboxed either
        // launches inside Sandbox2 or does not launch at all.
        return m_SandboxSpawner.spawn(processPath, args, childPid);
#else
        // Build/deployment contradiction: processPath is configured as
        // sandboxed, but this build has no Sandbox2 support (non-Linux).
        // pytorch_inference should never be listed as sandboxed on such a
        // platform - fail closed and say why, rather than silently falling
        // through to the legacy spawner as the frozen router's #ifdef
        // Linux masked this exact case by doing.
        LOG_ERROR(<< "Refusing to launch '" << processPath
                  << "': configured as a sandboxed process path, but this "
                  << "build was not compiled with Sandbox2 support");
        return false;
#endif
    }

    // Not a sandboxed process path: unrelated processes always go via the
    // legacy spawner, unchanged from today's behaviour.
    return m_LegacySpawner.spawn(processPath, args, childPid);
}

bool CProcessSpawnerRouter::terminateChild(core::CProcess::TPid pid) {
    if (m_LegacySpawner.terminateChild(pid)) {
        return true;
    }
#ifdef SANDBOX2_AVAILABLE
    if (m_SandboxSpawner.terminateChild(pid)) {
        return true;
    }
#endif
    return false;
}

bool CProcessSpawnerRouter::hasChild(core::CProcess::TPid pid) const {
    if (m_LegacySpawner.hasChild(pid)) {
        return true;
    }
#ifdef SANDBOX2_AVAILABLE
    if (m_SandboxSpawner.hasChild(pid)) {
        return true;
    }
#endif
    return false;
}

} // namespace controller
} // namespace ml
