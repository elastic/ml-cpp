/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CCommandProcessor.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CProcess.h>
#include <core/CStringUtils.h>

#include <algorithm>
#include <istream>

namespace {
const std::string TAB(1, '\t');
const std::string EMPTY_STRING;
}

namespace ml {
namespace controller {

// Initialise statics
const std::string CCommandProcessor::START{"start"};
const std::string CCommandProcessor::KILL{"kill"};

CCommandProcessor::CCommandProcessor(const TStrVec& permittedProcessPaths,
                                     std::ostream& responseStream)
    : m_Spawner{permittedProcessPaths}, m_ResponseWriter{responseStream} {
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
        LOG_ERROR(<< "Invalid command ID in " << core::CContainerPrinter::print(tokens));
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

    if (m_Spawner.spawn(processPath, tokens) == false) {
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
        LOG_ERROR(<< error << " in command with ID " << id);
        m_ResponseWriter.writeResponse(id, false, error);
        return false;
    }

    m_ResponseWriter.writeResponse(id, true, "Process with PID " + tokens[0] + " killed");
    return true;
}
}
}
