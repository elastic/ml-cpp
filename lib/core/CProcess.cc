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

CProcess::CProcess(void)
    : m_IsService(false), m_Initialised(false), m_Running(false), m_MlMainFunc(0) {}

CProcess& CProcess::instance(void) {
    static CProcess instance;
    return instance;
}

bool CProcess::isService(void) const {
    return m_IsService;
}

CProcess::TPid CProcess::id(void) const {
    return ::getpid();
}

CProcess::TPid CProcess::parentId(void) const {
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

bool CProcess::isInitialised(void) const {
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

void CProcess::initialisationComplete(void) {
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

bool CProcess::isRunning(void) const {
    return m_Running;
}

bool CProcess::shutdown(void) {
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
