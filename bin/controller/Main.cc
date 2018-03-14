/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */
//! \brief
//! Controller to start other Ml processes.
//!
//! DESCRIPTION:\n
//! Starts other Ml processes based on commands sent to it
//! through a named pipe.
//!
//! Each command has the following format:
//! verb arguments...
//!
//! These components must be separated using tabs, and the overall
//! command must be terminated with a newline.  (This implies that
//! keys and arguments cannot contain tabs or newlines.)
//!
//! IMPLEMENTATION DECISIONS:\n
//! Standalone program.
//!
//! Only accepts requests to start the following processes:
//! 1) ./autoconfig
//! 2) ./autodetect
//! 3) ./normalize
//! 4) ./categorize
//!
//! The assumption here is that the working directory of this
//! process will be the directory containing these other
//! processes.
//!
//! Always logs to a named pipe and accepts commands from
//! a named pipe.
//!
//! Additionally, reads from STDIN and will exit when it detects
//! EOF on STDIN.  This is so that it can exit if the JVM that
//! started it dies before the command named pipe is set up.
//!
#include <core/CLogger.h>
#include <core/CNamedPipeFactory.h>
#include <core/COsFileFuncs.h>
#include <core/CProcess.h>
#include <core/CProgName.h>
#include <core/CStringUtils.h>
#include <core/CThread.h>

#include <ver/CBuildInfo.h>

#include "CBlockingCallCancellerThread.h"
#include "CCmdLineParser.h"
#include "CCommandProcessor.h"

#include <iostream>
#include <string>

#include <errno.h>
#include <stdlib.h>
#include <string.h>


int main(int argc, char **argv) {
    const std::string &defaultNamedPipePath =
        ml::core::CNamedPipeFactory::defaultPath();
    const std::string &progName = ml::core::CProgName::progName();

    // Read command line options
    std::string jvmPidStr =
        ml::core::CStringUtils::typeToString(ml::core::CProcess::instance().parentId());
    std::string logPipe;
    std::string commandPipe;
    if (ml::controller::CCmdLineParser::parse(argc,
                                              argv,
                                              jvmPidStr,
                                              logPipe,
                                              commandPipe) == false) {
        return EXIT_FAILURE;
    }

    if (logPipe.empty()) {
        logPipe = defaultNamedPipePath + progName + "_log_" + jvmPidStr;
    }
    if (commandPipe.empty()) {
        commandPipe = defaultNamedPipePath + progName + "_command_" + jvmPidStr;
    }

    // This needs to be started before reconfiguring logging just in case
    // nothing connects to the other end of the logging pipe.  This could
    // happen if say:
    // 1) The pre-seccomp code in the Java process starts this process
    // 2) A bootstrap check, e.g. jar hell, fails
    // 3) The Java process exits with an error status
    // 4) No plugin code ever runs
    // This thread will detect the death of the parent process because this
    // process's STDIN will be closed.
    ml::controller::CBlockingCallCancellerThread cancellerThread(ml::core::CThread::currentThreadId(),
                                                                 std::cin);
    if (cancellerThread.start() == false) {
        // This log message will probably never been seen as it will go to the
        // real stderr of this process rather than the log pipe...
        LOG_FATAL("Could not start blocking call canceller thread");
        return EXIT_FAILURE;
    }

    if (ml::core::CLogger::instance().reconfigureLogToNamedPipe(logPipe) == false) {
        LOG_FATAL("Could not reconfigure logging");
        cancellerThread.stop();
        return EXIT_FAILURE;
    }

    // Log the program version immediately after reconfiguring the logger.  This
    // must be done from the program, and NOT a shared library, as each program
    // statically links its own version library.
    LOG_INFO(ml::ver::CBuildInfo::fullInfo());

    // Unlike other programs we DON'T reduce the process priority here, because
    // the controller is critical to the overall system.  Also its resource
    // requirements should always be very low.

    ml::core::CNamedPipeFactory::TIStreamP commandStream =
        ml::core::CNamedPipeFactory::openPipeStreamRead(commandPipe);
    if (commandStream == 0) {
        LOG_FATAL("Could not open command pipe");
        cancellerThread.stop();
        return EXIT_FAILURE;
    }

    // Change directory to the directory containing this program, because the
    // permitted paths all assume the current working directory contains the
    // permitted programs
    const std::string &progDir = ml::core::CProgName::progDir();
    if (ml::core::COsFileFuncs::chdir(progDir.c_str()) == -1) {
        LOG_FATAL("Could not change directory to '" << progDir << "': " <<
                  ::strerror(errno));
        cancellerThread.stop();
        return EXIT_FAILURE;
    }

    ml::controller::CCommandProcessor::TStrVec permittedProcessPaths;
    permittedProcessPaths.push_back("./autoconfig");
    permittedProcessPaths.push_back("./autodetect");
    permittedProcessPaths.push_back("./categorize");
    permittedProcessPaths.push_back("./normalize");

    ml::controller::CCommandProcessor processor(permittedProcessPaths);
    processor.processCommands(*commandStream);

    cancellerThread.stop();

    // This message makes it easier to spot process crashes in a log file - if
    // this isn't present in the log for a given PID and there's no other log
    // message indicating early exit then the process has probably core dumped
    LOG_INFO("Ml controller exiting");

    return EXIT_SUCCESS;
}

