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

#include <sandbox/CPytorchInferenceSandboxPolicy.h>

#include <algorithm>
#include <cstdlib>
#include <sstream>

namespace {

//! Scan \p args for a "--modelid=<value>" token, using the same linear
//! string-prefix scan style CCommandProcessor uses for --disableSandbox
//! (bin/controller/CCommandProcessor.cc), rather than pulling in
//! boost::program_options for a single optional field. Returns "" if
//! absent. Independent of any --disableSandbox scan - this never mutates
//! or consumes \p args.
std::string scanModelId(const ml::controller::CProcessSpawnerRouter::TStrVec& args) {
    const std::string prefix{"--modelid="};
    for (const auto& arg : args) {
        if (arg.compare(0, prefix.size(), prefix) == 0) {
            return arg.substr(prefix.size());
        }
    }
    return std::string();
}

//! Minimal JSON string escaping for the two string fields
//! (deployment_id/model_id) that are derived from operator/caller-supplied
//! input (a launch argument and a validated path component) rather than
//! from a fixed internal vocabulary - PR F's ES-side observability code
//! parses this line by name and type, so it must stay valid JSON even if
//! either value contains a quote or backslash.
std::string jsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        default:
            out += c;
        }
    }
    return out;
}

} // namespace

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

void CProcessSpawnerRouter::emitLaunchSignal(ERoute route, const TStrVec& args, bool spawnSucceeded) const {
    // Derive deployment_id exactly as CSandboxedProcessSpawner_Linux.cc
    // does before constructing a Sandbox2 policy (see its trustedTmpDir
    // derivation just before its own validateChildIpcLaunchSpec() call):
    // this duplicates that validation call for observability purposes,
    // which is expected - the function is pure, cross-platform-safe (only
    // ::realpath and env/stat calls), and this signal must fire
    // independently of whether the Linux spawner's own gating call ever
    // ran (e.g. the legacy/degraded route never reaches it at all).
    const char* tmpDirEnv{::getenv("TMPDIR")};
    const std::string trustedTmpDir{tmpDirEnv != nullptr ? tmpDirEnv : "/tmp"};
    const sandbox::SChildIpcValidationResult validated{
        sandbox::validateChildIpcLaunchSpec(trustedTmpDir, args)};

    const bool isLegacyRoute{route == ERoute::E_Legacy};

    // Controller ruling (binding, PR E Task 4): degraded is decided purely
    // by route, regardless of the legacy spawn's own success/failure;
    // enforced/fail_closed are only decided for the no-token Sandbox2
    // route, keyed off the spawn outcome itself.
    std::string mode;
    if (isLegacyRoute) {
        mode = "degraded";
    } else {
        mode = spawnSucceeded ? "enforced" : "fail_closed";
    }
    const bool sandbox2Established{mode == "enforced"};

    std::ostringstream signal;
    signal << "{\"event\":\"sandbox2_launch\""
           << ",\"deployment_id\":\"" << jsonEscape(validated.s_Spec.s_ChildId) << "\""
           << ",\"model_id\":\"" << jsonEscape(scanModelId(args)) << "\""
           << ",\"route\":\"" << (isLegacyRoute ? "legacy" : "sandbox2") << "\""
           << ",\"sandbox2_established\":" << (sandbox2Established ? "true" : "false")
           << ",\"mode\":\"" << mode << "\""
           << "}";
    LOG_INFO(<< signal.str());
}

bool CProcessSpawnerRouter::spawn(ERoute route,
                                   const std::string& processPath,
                                   const TStrVec& args,
                                   core::CProcess::TPid& childPid) {
    // The H4 signal (design.md §Failure behavior and observability) fires
    // only for processes actually eligible for sandboxing - never for
    // unrelated permitted processes like autodetect - and exactly once per
    // spawn() call, on every outcome, computed once up front so neither
    // dispatch branch below can accidentally skip or duplicate it.
    const bool sandboxEligible{this->isSandboxedProcessPath(processPath)};

    bool spawned{false};
    if (route == ERoute::E_Legacy) {
        // Operator kill-switch route: the caller has already validated the
        // --disableSandbox token against this exact processPath and
        // stripped it from args before this call - this router never
        // re-parses args to decide anything (unlike the frozen prior art's
        // spawn(), which re-derived disableSandbox from args itself).
        LOG_INFO(<< "Launching '" << processPath
                 << "' without Sandbox2 (operator kill switch --disableSandbox); "
                 << "the in-process seccomp filter applies");
        spawned = m_LegacySpawner.spawn(processPath, args, childPid);
    } else if (sandboxEligible) {
        // route == ERoute::E_Sandbox2, and processPath is configured as
        // sandboxed.
#ifdef SANDBOX2_AVAILABLE
        // No automatic fallback to the legacy spawner on a Sandbox2
        // failure (V2, MG1): a process that must be sandboxed either
        // launches inside Sandbox2 or does not launch at all.
        spawned = m_SandboxSpawner.spawn(processPath, args, childPid);
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
        spawned = false;
#endif
    } else {
        // Not a sandboxed process path: unrelated processes always go via
        // the legacy spawner, unchanged from today's behaviour.
        spawned = m_LegacySpawner.spawn(processPath, args, childPid);
    }

    if (sandboxEligible) {
        this->emitLaunchSignal(route, args, spawned);
    }

    return spawned;
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
