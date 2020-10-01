/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_controller_CCommandProcessor_h
#define INCLUDED_ml_controller_CCommandProcessor_h

#include <core/CDetachedProcessSpawner.h>

#include "CResponseJsonWriter.h"

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace ml {
namespace controller {

//! \brief
//! Processes commands received on a C++ stream.
//!
//! DESCRIPTION:\n
//! Reads from the supplied stream until end-of-file is reached.
//!
//! Each (newline terminated) line is assumed to be a tab-separated
//! command to be executed.
//!
//! Each command has the following format:
//! ID verb arguments...
//!
//! The ID is expected to be a unique positive integer.  This is reported
//! in error messages and in the response objects that are sent when the
//! command is complete.
//!
//! Available verbs are:
//! 1) start - in this case the arguments consist of the process name
//!    followed by the arguments to pass to the new process
//! 2) kill - in this case the argument is the PID of the process to kill
//!
//! IMPLEMENTATION DECISIONS:\n
//! Commands are case sensitive.
//!
//! Only processes on a supplied list may be executed; requests to start
//! other processes are ignored.
//!
//! Only processes started by this controller may be killed; requests to
//! kill other processes are ignored.
//!
class CCommandProcessor {
public:
    using TStrVec = std::vector<std::string>;

public:
    //! Possible commands
    static const std::string START;
    static const std::string KILL;

public:
    CCommandProcessor(const TStrVec& permittedProcessPaths, std::ostream& responseStream);

    //! Action commands read from the supplied \p commandStream until
    //! end-of-file is reached.
    void processCommands(std::istream& commandStream);

    //! Parse and handle a single command.
    bool handleCommand(const std::string& command);

private:
    //! Handle a start command.
    //! \param id The command ID.
    //! \param tokens Tokens to the command excluding the command ID and verb.
    bool handleStart(std::uint32_t id, TStrVec tokens);

    //! Handle a kill command.
    //! \param id The command ID.
    //! \param tokens Expected to contain one element, namely the process
    //!               ID of the process to be killed.
    bool handleKill(std::uint32_t id, TStrVec tokens);

private:
    //! Used to spawn/kill the requested processes.
    core::CDetachedProcessSpawner m_Spawner;

    //! Used to write responses in JSON format to the response stream.
    CResponseJsonWriter m_ResponseWriter;
};
}
}

#endif // INCLUDED_ml_controller_CCommandProcessor_h
