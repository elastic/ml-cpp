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
#include "CCommandProcessor.h"

#include <core/CLogger.h>
#include <core/CProcess.h>
#include <core/CStringUtils.h>

#include <algorithm>
#include <cstdlib>
#include <istream>
#include <string>

namespace {
const std::string TAB(1, '\t');
const std::string EMPTY_STRING;
//! The only controller-control token the command wire format defines
//! today. Any other unrecognised "--" prefixed token is passed through to
//! the spawned
//! process unchanged - this task does not invent a general token schema.
const std::string DISABLE_SANDBOX_TOKEN{"--disableSandbox"};

//! Internal controller option gating the no-token default route. Not an
//! operator setting and not part of the command wire format: it exists so
//! the typed-routing machinery can ship dormant (legacy default) until
//! Elasticsearch owns the setting that turns mandatory Sandbox2 on.
const char* SANDBOX2_DEFAULT_ENFORCED_ENV{"ML_SANDBOX2_DEFAULT_ENFORCED"};

//! Exactly "1" and nothing else is truthy - one canonical spelling, the
//! same one the sandboxee's own ML_SANDBOXED=1 contract uses. Anything else
//! (unset, "", "0", "true", "TRUE", "yes") leaves the option off.
bool sandbox2DefaultEnforced() {
    const char* value{::getenv(SANDBOX2_DEFAULT_ENFORCED_ENV)};
    return value != nullptr && std::string{value} == "1";
}
}

namespace ml {
namespace controller {

// Initialise statics
const std::string CCommandProcessor::START{"start"};
const std::string CCommandProcessor::KILL{"kill"};

CCommandProcessor::CCommandProcessor(const TStrVec& permittedProcessPaths,
                                     const TStrVec& sandboxedProcessPaths,
                                     std::ostream& responseStream)
    : m_Spawner{permittedProcessPaths, sandboxedProcessPaths},
      m_Sandbox2DefaultEnabled{sandbox2DefaultEnforced()}, m_ResponseWriter{responseStream} {
    if (m_Sandbox2DefaultEnabled) {
        LOG_INFO(<< SANDBOX2_DEFAULT_ENFORCED_ENV << "=1: a start command with no " << DISABLE_SANDBOX_TOKEN
                 << " token requires Sandbox2 for configured sandboxed process paths");
    }
}

void CCommandProcessor::processCommands(std::istream& commandStream) {
    std::string command;
    while (std::getline(commandStream, command)) {
        if (command.empty() == false) {
            this->handleCommand(command);
        }
    }
}

bool CCommandProcessor::handleCommand(const std::string& command) {
    // Command lines must be tab-separated
    TStrVec tokens;
    {
        std::string remainder;
        core::CStringUtils::tokenise(TAB, command, tokens, remainder);
        if (remainder.empty() == false) {
            tokens.emplace_back(std::move(remainder));
        }
    }

    // Multiple consecutive tabs might have caused empty tokens
    tokens.erase(std::remove(tokens.begin(), tokens.end(), EMPTY_STRING), tokens.end());

    if (tokens.size() < 3) {
        if (tokens.empty() == false) {
            LOG_ERROR(<< "Ignoring command with only " << tokens.size()
                      << ((tokens.size() == 1) ? " token" : " tokens"));
        }
        return false;
    }

    // Split into ID, verb and other tokens
    std::uint32_t id{0};
    if (core::CStringUtils::stringToType(tokens[0], id) == false || id == 0) {
        LOG_ERROR(<< "Invalid command ID in " << tokens);
        return false;
    }

    std::string verb{std::move(tokens[1])};
    tokens.erase(tokens.begin(), tokens.begin() + 2);

    if (verb == START) {
        return this->handleStart(id, std::move(tokens));
    }
    if (verb == KILL) {
        return this->handleKill(id, std::move(tokens));
    }

    std::string error{"Did not understand verb '" + verb + '\''};
    LOG_ERROR(<< error << " in command with ID " << id);
    m_ResponseWriter.writeResponse(id, false, error);
    return false;
}

bool CCommandProcessor::handleStart(std::uint32_t id, TStrVec tokens) {
    std::string processPath{std::move(tokens[0])};
    tokens.erase(tokens.begin());

    // Scan for the operator kill-switch token before any spawn decision is
    // made. Never "last one wins"/"first one wins" on duplicates - count
    // them all and reject outright if there's more than one.
    std::size_t disableSandboxCount{0};
    TStrVec::iterator firstDisableSandbox{tokens.end()};
    for (auto iter = tokens.begin(); iter != tokens.end(); ++iter) {
        if (*iter == DISABLE_SANDBOX_TOKEN) {
            if (disableSandboxCount == 0) {
                firstDisableSandbox = iter;
            }
            ++disableSandboxCount;
        }
    }

    if (disableSandboxCount >= 2) {
        std::string error{"Rejecting command: '" + DISABLE_SANDBOX_TOKEN + "' specified " +
                          core::CStringUtils::typeToString(disableSandboxCount) +
                          " times for process '" + processPath + '\''};
        LOG_ERROR(<< error << " in command with ID " << id);
        m_ResponseWriter.writeResponse(id, false, error);
        return false;
    }

    // One shared predicate with the router (which uses the same call to gate
    // dispatch and sandbox2_launch-signal emission), never a second std::find over a
    // second copy of the list.
    const bool isConfiguredSandboxedPath{m_Spawner.isSandboxedProcessPath(processPath)};

    CProcessSpawnerRouter::ERoute route{CProcessSpawnerRouter::ERoute::E_Sandbox2};
    // Provenance of a legacy route, recorded at the one place it is known so
    // the router's sandbox2_launch signal can report it as "legacy_reason". Stays
    // E_NotLegacy for every E_Sandbox2 route, where the field is omitted.
    CProcessSpawnerRouter::ELegacyReason legacyReason{
        CProcessSpawnerRouter::ELegacyReason::E_NotLegacy};
    if (disableSandboxCount == 0) {
        // No token: the route is only a decision at all for a configured
        // sandboxed process path (every other permitted process dispatches
        // to the legacy spawner either way, and must not be described as an
        // explicitly-selected legacy route in the log).
        //
        // Ships dormant: with the internal option off (the default), the
        // no-token case stays on the legacy route - byte-for-byte the
        // pre-typed-routing behaviour on every platform, including builds
        // with no Sandbox2 support at all. With the option on it becomes
        // mandatory Sandbox2 (E_Sandbox2, no automatic fallback). The
        // follow-up that flips the option is the Elasticsearch-side
        // operator-setting change, not this one.
        if (isConfiguredSandboxedPath && m_Sandbox2DefaultEnabled == false) {
            route = CProcessSpawnerRouter::ERoute::E_Legacy;
            legacyReason = CProcessSpawnerRouter::ELegacyReason::E_DormantDefault;
            LOG_DEBUG(<< "Routing '" << processPath << "' to the legacy path: no "
                      << DISABLE_SANDBOX_TOKEN << " token and "
                      << SANDBOX2_DEFAULT_ENFORCED_ENV << " is not set to 1");
        }
    } else if (disableSandboxCount == 1) {
        if (isConfiguredSandboxedPath == false) {
            std::string error{"Rejecting command: '" + DISABLE_SANDBOX_TOKEN +
                              "' is only valid for the configured sandboxed process, "
                              "not '" +
                              processPath + '\''};
            LOG_ERROR(<< error << " in command with ID " << id);
            m_ResponseWriter.writeResponse(id, false, error);
            return false;
        }

        // Operator kill-switch validated against this exact processPath:
        // strip it before it reaches the spawner and route to legacy. This
        // is the one place the route's operator provenance is known, so it
        // is logged here rather than in the router, which only ever sees an
        // already-decided route.
        LOG_INFO(<< "Routing '" << processPath << "' to the legacy path: operator kill switch "
                 << DISABLE_SANDBOX_TOKEN << " in command with ID " << id);
        route = CProcessSpawnerRouter::ERoute::E_Legacy;
        legacyReason = CProcessSpawnerRouter::ELegacyReason::E_KillSwitch;
        tokens.erase(firstDisableSandbox);
    }

    core::CProcess::TPid childPid{0};
    if (m_Spawner.spawn(route, processPath, tokens, childPid, legacyReason) == false) {
        std::string error{"Failed to start process '" + processPath + '\''};
        LOG_ERROR(<< error << " in command with ID " << id);
        m_ResponseWriter.writeResponse(id, false, error);
        return false;
    }

    m_ResponseWriter.writeResponse(id, true, "Process '" + processPath + "' started");
    return true;
}

bool CCommandProcessor::handleKill(std::uint32_t id, TStrVec tokens) {
    core::CProcess::TPid pid{0};
    if (tokens.size() != 1 ||
        core::CStringUtils::stringToType(tokens[0], pid) == false || pid == 0) {
        std::string error{"Unexpected arguments for kill command: " +
                          core::CContainerPrinter::print(tokens)};
        LOG_ERROR(<< error << " in command with ID " << id);
        m_ResponseWriter.writeResponse(id, false, error);
        return false;
    }

    if (m_Spawner.terminateChild(pid) == false) {
        std::string error{"Failed to kill process with PID " + tokens[0]};
        LOG_WARN(<< error << " in command with ID " << id);
        m_ResponseWriter.writeResponse(id, false, error);
        return false;
    }

    m_ResponseWriter.writeResponse(id, true, "Process with PID " + tokens[0] + " killed");
    return true;
}
}
}
