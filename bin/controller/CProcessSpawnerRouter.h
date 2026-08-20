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

#ifdef Linux
#include <sandbox/CSandboxedProcessSpawner.h>
#endif

#include <string>
#include <vector>

namespace ml {
namespace controller {

//! \brief
//! Routes process spawn requests to the legacy or Sandbox2 spawner.
//!
//! DESCRIPTION:\n
//! Strips the operator kill switch flag (--disableSandbox) before dispatch.
//! Processes listed in sandboxedProcessPaths are launched via Sandbox2 unless
//! the kill switch is set; all other permitted processes use posix_spawn.
//!
class CProcessSpawnerRouter {
public:
    using TStrVec = std::vector<std::string>;

public:
    CProcessSpawnerRouter(const TStrVec& permittedProcessPaths,
                          const TStrVec& sandboxedProcessPaths);

    bool spawn(const std::string& processPath,
               const TStrVec& args,
               core::CProcess::TPid& childPid,
               std::string* failureReason = nullptr);

    bool terminateChild(core::CProcess::TPid pid);

    bool hasChild(core::CProcess::TPid pid) const;

private:
    bool isSandboxedProcessPath(const std::string& processPath) const;

private:
    core::CDetachedProcessSpawner m_LegacySpawner;
#ifdef Linux
    sandbox::CSandboxedProcessSpawner m_SandboxSpawner;
#endif
    TStrVec m_SandboxedProcessPaths;
};

} // namespace controller
} // namespace ml

#endif // INCLUDED_ml_controller_CProcessSpawnerRouter_h
