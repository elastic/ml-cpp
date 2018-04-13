/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CProcess.h>

#include <core/CLogger.h>
#include <core/CScopedFastLock.h>

#include <stdlib.h>

namespace ml {
namespace core {

const char* CProcess::STARTING_MSG("Process Starting.");
const char* CProcess::STARTED_MSG("Process Started.");
const char* CProcess::STOPPING_MSG("Process Shutting Down.");
const char* CProcess::STOPPED_MSG("Process Exiting.");

CProcess::CProcess() : m_IsService(false), m_Initialised(false), m_Running(false), m_MlMainFunc(0) {
}

CProcess& CProcess::instance() {
    static CProcess instance;
    return instance;
}

bool CProcess::isService() const {
    return m_IsService;
}

CProcess::TPid CProcess::id() const {
    return ::getpid();
}

CProcess::TPid CProcess::parentId() const {
    return ::getppid();
}

bool CProcess::startDispatcher(TMlMainFunc mlMain, int argc, char* argv[]) {
    if (mlMain == 0) {
        LOG_ABORT("NULL mlMain() function passed");
    }

    m_MlMainFunc = mlMain;
    m_Args.reserve(argc);
    for (int count = 0; count < argc; ++count) {
        m_Args.push_back(argv[count]);
    }

    // For Unix, this method just passes on the call
    m_Running = true;

    bool success(m_MlMainFunc(argc, argv) == EXIT_SUCCESS);

    // Only log process status messages if the logger has been reconfigured to
    // log somewhere more sensible that STDERR.  (This prevents us spoiling the
    // output from --version and --help.)
    if (CLogger::instance().hasBeenReconfigured()) {
        LOG_INFO(STOPPED_MSG);
    }

    return success;
}

bool CProcess::isInitialised() const {
    return m_Initialised;
}

void CProcess::initialisationComplete(const TShutdownFunc& shutdownFunc) {
    CScopedFastLock lock(m_ShutdownFuncMutex);

    if (!m_Initialised) {
        if (CLogger::instance().hasBeenReconfigured()) {
            LOG_INFO(STARTED_MSG);
        }
        m_Initialised = true;
    }

    m_ShutdownFunc = shutdownFunc;
}

void CProcess::initialisationComplete() {
    CScopedFastLock lock(m_ShutdownFuncMutex);

    if (!m_Initialised) {
        if (CLogger::instance().hasBeenReconfigured()) {
            LOG_INFO(STARTED_MSG);
        }
        m_Initialised = true;
    }

    // Clear the shutdown function
    TShutdownFunc emptyFunc;
    m_ShutdownFunc.swap(emptyFunc);
}

bool CProcess::isRunning() const {
    return m_Running;
}

bool CProcess::shutdown() {
    if (CLogger::instance().hasBeenReconfigured()) {
        LOG_INFO(STOPPING_MSG);
    }

    CScopedFastLock lock(m_ShutdownFuncMutex);

    if (!m_ShutdownFunc) {
        return false;
    }

    m_ShutdownFunc();

    return true;
}
}
}
