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

#include <sandbox/CMlSandboxAvailability.h>
#include <sandbox/CPytorchInferenceSandboxPolicy.h>
#include <sandbox/CSandbox2Diagnostics.h>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <sstream>

namespace {

//! Scan \p args for a "--modelid=<value>" token, using the same linear
//! string-prefix scan style CCommandProcessor uses for --disableSandbox
//! (bin/controller/CCommandProcessor.cc), rather than pulling in
//! boost::program_options for a single optional field. Returns "" if
//! absent. Independent of any --disableSandbox scan - this never mutates
//! or consumes \p args.
//!
//! Only matches the "=" form ("--modelid=<value>"), not the space-separated
//! "--modelid <value>" form boost::program_options also accepts elsewhere
//! in this codebase: the "=" form is the wire contract a future change's
//! ES-side observability code relies on for model_id in the sandbox2_launch
//! signal (docs/sandbox2_production_failure_modes.md).
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
//! from a fixed internal vocabulary - a future change's ES-side
//! observability code parses this line by name and type, so it must stay
//! valid JSON even if
//! either value contains a quote, a backslash, or a control character.
//! deployment_id is a filesystem path component and model_id comes straight
//! off the command line, so a raw newline/tab/NUL in either would otherwise
//! split or corrupt what must stay a single-line JSON object.
std::string jsonEscape(const std::string& s) {
    static const char* const HEX_DIGITS{"0123456789abcdef"};
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        const auto byte = static_cast<unsigned char>(c);
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (byte < 0x20) {
                // Every remaining C0 control character, as the \u00XX escape
                // JSON requires (RFC 8259 section 7).
                out += "\\u00";
                out += HEX_DIGITS[(byte >> 4) & 0xF];
                out += HEX_DIGITS[byte & 0xF];
            } else {
                out += c;
            }
        }
    }
    return out;
}

//! Derive the per-launch deployment_id (SChildIpcLaunchSpec::s_ChildId) from
//! the path-bearing launch options in \p args, exactly as
//! CSandboxedProcessSpawner_Linux.cc does before constructing a Sandbox2
//! policy (same trustedTmpDir derivation - getenv("TMPDIR"), defaulting to
//! "/tmp"). Called once per spawn(), *before* either backend runs, so the
//! sandbox2_launch signal and the dispatch decision see one and the same
//! filesystem
//! state: validateChildIpcLaunchSpec() does live ::realpath() calls, and a
//! post-spawn second call could observe a different (or, on the
//! legacy/degraded and failed-Sandbox2 paths, an absent) per-child IPC
//! directory and report an empty deployment_id on exactly the degraded and
//! fail_closed modes the signal exists to make debuggable.
//! Returns "" when no path-bearing option was present at all.
std::string deriveDeploymentId(const ml::controller::CProcessSpawnerRouter::TStrVec& args) {
    const char* tmpDirEnv{::getenv("TMPDIR")};
    const std::string trustedTmpDir{tmpDirEnv != nullptr ? tmpDirEnv : "/tmp"};
    // validateChildIpcLaunchSpec() does live ::realpath() calls, which
    // require $TMPDIR/ml-child-ipc/<child-id> to already exist. This is the
    // *other* production call site that reaches that function (the one
    // inside CSandboxedProcessSpawner_Linux.cc::spawn() is the other), and
    // it runs strictly before spawn() dispatches to either backend - on the
    // legacy route just as much as the Sandbox2 route, since the signal
    // below always wants a real deployment_id. Ensure the directory exists
    // here too, rather than relying on the Sandbox2 spawner (which may not
    // even run on this route) to have already done it. A creation failure
    // is not logged again here: an empty deployment_id in the signal is
    // itself the observable symptom, and the Sandbox2 spawner (when that
    // route is actually taken) logs the failure with detail.
    ml::sandbox::ensureChildIpcDirectory(trustedTmpDir, args);
    return ml::sandbox::validateChildIpcLaunchSpec(trustedTmpDir, args).s_Spec.s_ChildId;
}

} // namespace

namespace ml {
namespace controller {

CProcessSpawnerRouter::CProcessSpawnerRouter(const TStrVec& permittedProcessPaths,
                                             const TStrVec& sandboxedProcessPaths,
                                             TConfinementFn confinementFn)
    : m_LegacySpawner{permittedProcessPaths}, m_SandboxedProcessPaths{sandboxedProcessPaths},
      m_ConfinementFn{confinementFn ? std::move(confinementFn) : TConfinementFn{[] {
          return sandbox::hostConfinement();
      }}} {
}

const std::string& CProcessSpawnerRouter::lastSpawnFailureReason() const {
    return m_LastSpawnFailureReason;
}

bool CProcessSpawnerRouter::isSandboxedProcessPath(const std::string& processPath) const {
    return std::find(m_SandboxedProcessPaths.begin(), m_SandboxedProcessPaths.end(),
                     processPath) != m_SandboxedProcessPaths.end();
}

void CProcessSpawnerRouter::emitLaunchSignal(ERoute route,
                                             ELegacyReason legacyReason,
                                             const std::string& deploymentId,
                                             const TStrVec& args,
                                             bool spawnSucceeded,
                                             bool landlockFallback) const {
    const bool isLegacyRoute{route == ERoute::E_Legacy};

    // degraded is decided purely by route, regardless of the legacy
    // spawn's own success/failure;
    // enforced/fail_closed apply when route == E_Sandbox2, keyed off the
    // spawn outcome (failed Sandbox2 launch, or no Sandbox2 support on a
    // --requireSandbox launch).
    std::string mode;
    if (isLegacyRoute) {
        mode = "degraded";
    } else if (landlockFallback) {
        // A Sandbox2-routed launch that this host could not honour, run
        // under Landlock instead. Reported distinctly rather than as
        // "enforced" (no Sandbox2 was established) or "fail_closed" (the
        // deployment did start): a consumer must be able to tell that the
        // operator's request was met by something weaker.
        mode = spawnSucceeded ? "landlock" : "fail_closed";
    } else {
        mode = spawnSucceeded ? "enforced" : "fail_closed";
    }
    const bool sandbox2Established{mode == "enforced"};

    // Additive field, emitted *only* on the legacy route (route ==
    // "legacy", i.e. mode == "degraded"): mode alone conflates a deliberate
    // operator kill switch with the permanent no-token default. Omitted
    // entirely - never "" and never null - on route == "sandbox2", i.e. on
    // both the "enforced" and "fail_closed" modes, since neither can have a
    // legacy reason.
    std::string legacyReasonField;
    if (isLegacyRoute) {
        const char* reason{legacyReason == ELegacyReason::E_KillSwitch ? "kill_switch" : "no_token_default"};
        if (legacyReason == ELegacyReason::E_NotLegacy) {
            // A caller that routed to legacy without naming why: report the
            // no-token default (the overwhelmingly common case for callers
            // that never send either routing token) rather than falsely
            // claiming an operator kill switch.
            LOG_WARN(<< "Legacy route with no recorded provenance; reporting the "
                        "no-token default in the sandbox2_launch signal");
        }
        legacyReasonField = std::string{",\"legacy_reason\":\""} + reason + "\"";
    }

    // Additive field, emitted on *every* signal line regardless of route:
    // a build-time-constant fact (backed by CMlSandboxAvailability, itself
    // backed by the SANDBOX2_AVAILABLE compile definition), not per-launch
    // state, so it is computed once here rather than threaded through as a
    // parameter. Lets a consumer (e.g. a future ES-side rollout logic)
    // distinguish a Linux build that has Sandbox2 support but a caller sent
    // no routing token (route == "legacy", legacy_reason ==
    // "no_token_default", sandbox2_compiled_in == true) from a build with
    // no Sandbox2 support at all (sandbox2_compiled_in == false) - the two
    // are otherwise indistinguishable from the sandbox2_launch signal alone.
    static const bool sandbox2CompiledIn{sandbox::CMlSandboxAvailability::isCompiledIn()};

    std::ostringstream signal;
    signal << "{\"event\":\"sandbox2_launch\""
           << ",\"deployment_id\":\"" << jsonEscape(deploymentId) << "\""
           << ",\"model_id\":\"" << jsonEscape(scanModelId(args)) << "\""
           << ",\"route\":\"" << (isLegacyRoute ? "legacy" : "sandbox2") << "\""
           << legacyReasonField
           << ",\"sandbox2_established\":" << (sandbox2Established ? "true" : "false")
           << ",\"mode\":\"" << mode << "\""
           << ",\"sandbox2_compiled_in\":" << (sandbox2CompiledIn ? "true" : "false")
           << "}";
    LOG_INFO(<< signal.str());
}

const std::string CProcessSpawnerRouter::RESTRICT_FILESYSTEM_TOKEN{"--restrictFilesystem"};

bool CProcessSpawnerRouter::spawn(ERoute route,
                                  const std::string& processPath,
                                  const TStrVec& args,
                                  core::CProcess::TPid& childPid,
                                  ELegacyReason legacyReason) {
    // The sandbox2_launch signal fires only for processes actually
    // eligible for sandboxing - never for
    // unrelated permitted processes like autodetect - and exactly once per
    // spawn() call, on every outcome, computed once up front so neither
    // dispatch branch below can accidentally skip or duplicate it.
    const bool sandboxEligible{this->isSandboxedProcessPath(processPath)};

    // Derived exactly once per spawn() call, before either backend runs, so
    // the sandbox2_launch signal below reports the same childId the
    // dispatch decision was taken against - see deriveDeploymentId()'s comment for why a
    // post-spawn second derivation is not equivalent. Skipped entirely for
    // processes that can never emit the signal, so unrelated permitted
    // processes (autodetect etc.) pay no ::realpath() cost.
    const std::string deploymentId{sandboxEligible ? deriveDeploymentId(args)
                                                   : std::string()};

    m_LastSpawnFailureReason.clear();
    bool spawned{false};
    // Set when the Sandbox2 route degraded to the Landlock fallback, so the
    // signal below reports what actually bounded the child.
    bool landlockFallback{false};
    if (route == ERoute::E_Legacy) {
        // Legacy route decided upstream: either the operator kill-switch
        // token (validated against this exact processPath and stripped from
        // args by CCommandProcessor) or the permanent no-token default. This
        // router never re-parses args to decide anything (unlike the frozen
        // prior art's spawn(), which re-derived disableSandbox from args
        // itself), so it cannot - and must not - derive which of the two it
        // was; CCommandProcessor logs that provenance at the point it is
        // actually known, and passes it in as legacyReason purely so the
        // sandbox2_launch signal below can report it.
        LOG_INFO(<< "Launching '" << processPath << "' without Sandbox2 (legacy route selected by the controller); "
                 << "the in-process seccomp filter applies");
        spawned = m_LegacySpawner.spawn(processPath, args, childPid);
    } else if (sandboxEligible) {
    // route == ERoute::E_Sandbox2, and processPath is configured as
    // sandboxed.
#ifdef SANDBOX2_AVAILABLE
        // Decide the rung before constructing or launching anything, from
        // the one cached verdict the startup self-check also logged - never
        // an independent probe here, because two probes can disagree (one
        // once did, when the controller's non-dumpable flag broke the later
        // one) and then the log says one thing while the route does
        // another. Deciding first matters: on a host without user
        // namespaces a Sandbox2 launch fails only after an opaque
        // SETUP_ERROR, and on one that permits namespaces but denies mounts
        // inside them the forkserver deadlocks instead of returning.
        const sandbox::SHostConfinement host{m_ConfinementFn()};
        switch (host.s_Level) {
        case sandbox::EConfinementLevel::E_Sandbox2:
            // First - and only - point at which any Sandbox2 machinery is
            // constructed. A router that never reaches this case (every
            // router that never dispatches a validated --requireSandbox
            // token, every router on a host that cannot run Sandbox2, and
            // every router in a build without Sandbox2 support) never creates
            // a CSandboxedProcessSpawner at all, so no Sandbox2 state enters
            // its construction or teardown path. Single-threaded by the same
            // contract as the legacy spawner - see the member's declaration.
            if (m_SandboxSpawner == nullptr) {
                m_SandboxSpawner = std::make_unique<sandbox::CSandboxedProcessSpawner>();
            }
            // No automatic fallback on a Sandbox2 *failure*: a host that can
            // run Sandbox2 but fails this launch has a problem worth
            // surfacing, not papering over with a weaker boundary.
            spawned = m_SandboxSpawner->spawn(processPath, args, childPid);
            break;
        case sandbox::EConfinementLevel::E_Landlock: {
            // A supported, deliberate degradation: INFO, with what an
            // administrator would change to get full isolation.
            LOG_INFO(<< sandbox::landlockFallbackMessage(host, processPath));
            TStrVec landlockArgs{args};
            landlockArgs.emplace_back(RESTRICT_FILESYSTEM_TOKEN);
            spawned = m_LegacySpawner.spawn(processPath, landlockArgs, childPid);
            landlockFallback = true;
            break;
        }
        case sandbox::EConfinementLevel::E_Unavailable:
            // Refuse here, in the controller, rather than launching a child
            // that would only discover it cannot confine itself: that way
            // Elasticsearch gets an immediate, explained failure instead of
            // a pipe-connection timeout, and no untrusted model is ever
            // started unconfined while the operator asked for a sandbox.
            m_LastSpawnFailureReason = sandbox::noConfinementMessage(host, processPath);
            LOG_ERROR(<< m_LastSpawnFailureReason);
            spawned = false;
            break;
        }
#else
        // Build/deployment contradiction: processPath is configured as
        // sandboxed, but this build has no Sandbox2 support (non-Linux).
        // pytorch_inference should never be listed as sandboxed on such a
        // platform - fail closed and say why, rather than silently falling
        // through to the legacy spawner as the frozen router's #ifdef
        // Linux masked this exact case by doing.
        LOG_ERROR(<< "Refusing to launch '" << processPath << "': configured as a sandboxed process path, but this "
                  << "build was not compiled with Sandbox2 support");
        spawned = false;
#endif
    } else {
        // Not a sandboxed process path: ERoute::E_Sandbox2 is the processor's
        // default enum value but is not a routing decision here - always use
        // the legacy spawner, unchanged from today's behaviour.
        spawned = m_LegacySpawner.spawn(processPath, args, childPid);
    }

    if (sandboxEligible) {
        this->emitLaunchSignal(route, legacyReason, deploymentId, args, spawned, landlockFallback);
    }

    return spawned;
}

bool CProcessSpawnerRouter::terminateChild(core::CProcess::TPid pid) {
    if (m_LegacySpawner.terminateChild(pid)) {
        return true;
    }
#ifdef SANDBOX2_AVAILABLE
    // A null m_SandboxSpawner means no spawn() call ever dispatched to the
    // Sandbox2 route, so there can be no sandboxed child to terminate. Ask
    // rather than construct: creating the spawner here would defeat the
    // lazy lifecycle and could only ever return false anyway.
    if (m_SandboxSpawner != nullptr && m_SandboxSpawner->terminateChild(pid)) {
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
    // Null means no sandboxed child was ever spawned - see terminateChild().
    if (m_SandboxSpawner != nullptr && m_SandboxSpawner->hasChild(pid)) {
        return true;
    }
#endif
    return false;
}

} // namespace controller
} // namespace ml
