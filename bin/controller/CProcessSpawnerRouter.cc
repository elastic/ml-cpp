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
    return std::find(m_SandboxedProcessPaths.begin(), m_SandboxedProcessPaths.end(),
                     processPath) != m_SandboxedProcessPaths.end();
}

bool CProcessSpawnerRouter::spawn(const std::string& processPath,
                                  const TStrVec& args,
                                  core::CProcess::TPid& childPid,
                                  std::string* failureReason) {
    bool disableSandbox{false};
    TStrVec effectiveArgs;
    effectiveArgs.reserve(args.size());
    for (const auto& arg : args) {
        if (arg == "--disableSandbox") {
            disableSandbox = true;
        } else {
            effectiveArgs.push_back(arg);
        }
    }

#ifdef Linux
    if (disableSandbox == false && this->isSandboxedProcessPath(processPath)) {
        return m_SandboxSpawner.spawn(processPath, effectiveArgs, childPid, failureReason);
    }
#endif

    if (disableSandbox && this->isSandboxedProcessPath(processPath)) {
        LOG_INFO(<< "Launching '" << processPath << "' without Sandbox2 (operator "
                 << "kill switch --disableSandbox); the in-process seccomp filter applies");
    }

    return m_LegacySpawner.spawn(processPath, effectiveArgs, childPid, failureReason);
}

bool CProcessSpawnerRouter::terminateChild(core::CProcess::TPid pid) {
    if (m_LegacySpawner.terminateChild(pid)) {
        return true;
    }
#ifdef Linux
    if (m_SandboxSpawner.terminateChild(pid)) {
        return true;
    }
#endif
    return false;
}

bool CProcessSpawnerRouter::hasChild(core::CProcess::TPid pid) const {
    return m_LegacySpawner.hasChild(pid)
#ifdef Linux
           || m_SandboxSpawner.hasChild(pid)
#endif
        ;
}

} // namespace controller
} // namespace ml
