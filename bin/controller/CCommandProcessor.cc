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
//! Operator kill-switch: forces the legacy route for the configured
//! sandboxed process path. Mutually exclusive with REQUIRE_SANDBOX_TOKEN -
//! a start command naming both is ambiguous about its own route and is
//! rejected outright, never resolved by precedence.
const std::string DISABLE_SANDBOX_TOKEN{"--disableSandbox"};

//! Operator opt-in: forces the Sandbox2 route (E_Sandbox2, no automatic
//! legacy fallback) for the configured sandboxed process path. Symmetric
//! counterpart to DISABLE_SANDBOX_TOKEN - together these are the only two
//! controller-control tokens the command wire format defines; any other
//! unrecognised "--" prefixed token is passed through to the spawned
//! process unchanged.
const std::string REQUIRE_SANDBOX_TOKEN{"--requireSandbox"};
}

namespace ml {
namespace controller {

// Initialise statics
const std::string CCommandProcessor::START{"start"};
const std::string CCommandProcessor::KILL{"kill"};

CCommandProcessor::CCommandProcessor(const TStrVec& permittedProcessPaths,
                                     const TStrVec& sandboxedProcessPaths,
                                     std::ostream& responseStream)
    : m_Spawner{permittedProcessPaths, sandboxedProcessPaths}, m_ResponseWriter{responseStream} {
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

    // Scan for both routing tokens before any spawn decision is made.
    // Never "last one wins"/"first one wins" on duplicates of either token -
    // count them all and reject outright if either appears more than once.
    std::size_t disableSandboxCount{0};
    TStrVec::iterator firstDisableSandbox{tokens.end()};
    std::size_t requireSandboxCount{0};
    TStrVec::iterator firstRequireSandbox{tokens.end()};
    for (auto iter = tokens.begin(); iter != tokens.end(); ++iter) {
        if (*iter == DISABLE_SANDBOX_TOKEN) {
            if (disableSandboxCount == 0) {
                firstDisableSandbox = iter;
            }
            ++disableSandboxCount;
        } else if (*iter == REQUIRE_SANDBOX_TOKEN) {
            if (requireSandboxCount == 0) {
                firstRequireSandbox = iter;
            }
            ++requireSandboxCount;
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

    if (requireSandboxCount >= 2) {
        std::string error{"Rejecting command: '" + REQUIRE_SANDBOX_TOKEN + "' specified " +
                          core::CStringUtils::typeToString(requireSandboxCount) +
                          " times for process '" + processPath + '\''};
        LOG_ERROR(<< error << " in command with ID " << id);
        m_ResponseWriter.writeResponse(id, false, error);
        return false;
    }

    if (disableSandboxCount == 1 && requireSandboxCount == 1) {
        std::string error{"Rejecting command: '" + DISABLE_SANDBOX_TOKEN + "' and '" +
                          REQUIRE_SANDBOX_TOKEN +
                          "' are mutually exclusive, both specified for process '" +
                          processPath + '\''};
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
    if (requireSandboxCount == 1) {
        if (isConfiguredSandboxedPath == false) {
            std::string error{"Rejecting command: '" + REQUIRE_SANDBOX_TOKEN +
                              "' is only valid for the configured sandboxed process, "
                              "not '" +
                              processPath + '\''};
            LOG_ERROR(<< error << " in command with ID " << id);
            m_ResponseWriter.writeResponse(id, false, error);
            return false;
        }

        // Operator opt-in validated against this exact processPath: strip
        // it before it reaches the spawner. Route is already E_Sandbox2
        // (the default above), so nothing else changes here beyond
        // stripping and logging the decision at the one place its
        // provenance is known.
        LOG_INFO(<< "Routing '" << processPath << "' to Sandbox2: operator opt-in "
                 << REQUIRE_SANDBOX_TOKEN << " in command with ID " << id);
        tokens.erase(firstRequireSandbox);
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
    } else {
        // No token at all: the route is only a decision at all for a
        // configured sandboxed process path (every other permitted process
        // dispatches to the legacy spawner either way, and must not be
        // described as an explicitly-selected legacy route in the log).
        //
        // Permanent behaviour, not a rollout seam: a caller that sends
        // neither token always takes the legacy route - byte-for-byte the
        // pre-typed-routing behaviour on every platform, including builds
        // with no Sandbox2 support at all. Elasticsearch is expected to
        // always send exactly one of the two tokens on every start command
        // for a sandboxed-eligible process, so this branch exists for
        // non-ES callers (support/debug scripts, direct controller
        // invocation) and the test harness.
        if (isConfiguredSandboxedPath) {
            route = CProcessSpawnerRouter::ERoute::E_Legacy;
            legacyReason = CProcessSpawnerRouter::ELegacyReason::E_NoTokenDefault;
            LOG_DEBUG(<< "Routing '" << processPath << "' to the legacy path: neither "
                      << DISABLE_SANDBOX_TOKEN << " nor " << REQUIRE_SANDBOX_TOKEN
                      << " token was present");
        }
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
