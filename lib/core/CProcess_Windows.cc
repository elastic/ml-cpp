/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CProcess.h>

#include <core/CLogger.h>
#include <core/CScopedFastLock.h>
#include <core/CWindowsError.h>

#include <boost/scoped_array.hpp>

#include <TlHelp32.h>

#include <stdlib.h>
#include <string.h>


namespace
{

// The wait hint of 10 seconds is about three times the observed time Ml
// services generally take to shut down.  However, it is better to give too high
// a hint than too low a hint, as we don't want the service controller reporting
// an error prematurely.
const DWORD STOP_WAIT_HINT_MSECS(10000);

//! This needs to be called quickly after startup, because it will only work
//! while the parent process is still running.
DWORD findParentProcessId(void)
{
    HANDLE snapshotHandle(CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0));
    if (snapshotHandle == INVALID_HANDLE_VALUE)
    {
        // Log at the point of retrieval, as this is too early in the program
        // lifecycle
        return 0;
    }

    PROCESSENTRY32 processEntry;
    ::memset(&processEntry, 0, sizeof(PROCESSENTRY32));
    processEntry.dwSize = sizeof(PROCESSENTRY32);

    DWORD pid(GetCurrentProcessId());
    DWORD ppid(0);

    if (Process32First(snapshotHandle, &processEntry) != FALSE)
    {
        do
        {
            if (processEntry.th32ProcessID == pid)
            {
                ppid = processEntry.th32ParentProcessID;
                break;
            }
        }
        while (Process32Next(snapshotHandle, &processEntry) != FALSE);
    }

    CloseHandle(snapshotHandle);

    return ppid;
}

// If this is zero it indicates that an error occurred when getting it
const DWORD PPID(findParentProcessId());

}


namespace ml
{
namespace core
{

const char *CProcess::STARTING_MSG("Process Starting.");
const char *CProcess::STARTED_MSG("Process Started.");
const char *CProcess::STOPPING_MSG("Process Shutting Down.");
const char *CProcess::STOPPED_MSG("Process Exiting.");


CProcess::CProcess(void)
    : m_IsService(false),
      m_Initialised(false),
      m_Running(false),
      m_MlMainFunc(0),
      m_ServiceHandle(0)
{
}

CProcess &CProcess::instance(void)
{
    static CProcess instance;
    return instance;
}

bool CProcess::isService(void) const
{
    return m_IsService;
}

CProcess::TPid CProcess::id(void) const
{
    return GetCurrentProcessId();
}

CProcess::TPid CProcess::parentId(void) const
{
    if (PPID == 0)
    {
        LOG_ERROR("Failed to find parent process ID");
    }
    return PPID;
}

bool CProcess::startDispatcher(TMlMainFunc mlMain,
                               int argc,
                               char *argv[])
{
    if (mlMain == 0)
    {
        LOG_ABORT("NULL mlMain() function passed");
    }

    if (argc <= 0)
    {
        // Arguments are invalid, but at this point it's debatable whether
        // logging will work if we're running as a service, as ServiceMain()
        // hasn't yet run.
        LOG_ERROR("Windows service dispatcher received argc " << argc);
        return false;
    }

    m_MlMainFunc = mlMain;
    m_Args.reserve(argc);
    for (int count = 0; count < argc; ++count)
    {
        m_Args.push_back(argv[count]);
    }

    SERVICE_TABLE_ENTRY serviceTable[] = {
        { argv[0], &CProcess::serviceMain },
        { 0, 0 }
    };

    // Start on the assumption we ARE running as a Windows service
    m_IsService = true;
    if (StartServiceCtrlDispatcher(serviceTable) != FALSE)
    {
        // We're running as a Windows service, so Windows will
        // call serviceMain.  The service dispatcher won't return
        // until Windows wants the process to exit.
        return true;
    }

    DWORD errCode(GetLastError());
    if (errCode != ERROR_FAILED_SERVICE_CONTROLLER_CONNECT)
    {
        // We're supposed to be running as a service, but something's gone
        // wrong.  At this point it's debatable whether logging will work, as
        // ServiceMain() hasn't yet run.
        LOG_ERROR("Windows service dispatcher failed " <<
                  CWindowsError(errCode));
        return false;
    }

    // We're not running as a Windows service, so just pass on the call
    m_IsService = false;
    m_Running = true;

    bool success(m_MlMainFunc(argc, argv) == EXIT_SUCCESS);

    // Only log process status messages if the logger has been reconfigured to
    // log somewhere more sensible that STDERR.  (This prevents us spoiling the
    // output from --version and --help.)
    if (CLogger::instance().hasBeenReconfigured())
    {
        LOG_INFO(STOPPED_MSG);
    }

    return success;
}

bool CProcess::isInitialised(void) const
{
    return m_Initialised;
}

void CProcess::initialisationComplete(const TShutdownFunc &shutdownFunc)
{
    CScopedFastLock lock(m_ShutdownFuncMutex);

    if (!m_Initialised)
    {
        if (CLogger::instance().hasBeenReconfigured())
        {
            LOG_INFO(STARTED_MSG);
        }
        m_Initialised = true;
    }

    m_ShutdownFunc = shutdownFunc;

    // Report status in case we're running as a service
    // (This will do nothing if we're running as a console process)
    this->serviceCtrlHandler(SERVICE_CONTROL_INTERROGATE);
}

void CProcess::initialisationComplete(void)
{
    CScopedFastLock lock(m_ShutdownFuncMutex);

    if (!m_Initialised)
    {
        if (CLogger::instance().hasBeenReconfigured())
        {
            LOG_INFO(STARTED_MSG);
        }
        m_Initialised = true;
    }

    // Clear the shutdown function
    TShutdownFunc emptyFunc;
    m_ShutdownFunc.swap(emptyFunc);

    // Report status in case we're running as a service
    // (This will do nothing if we're running as a console process)
    this->serviceCtrlHandler(SERVICE_CONTROL_INTERROGATE);
}

bool CProcess::isRunning(void) const
{
    return m_Running;
}

void WINAPI CProcess::serviceMain(DWORD argc, char *argv[])
{
    // This is a static method, so get the singleton instance
    CProcess &process = CProcess::instance();

    // Since we're an "own process" service the name is not required
    process.m_ServiceHandle = RegisterServiceCtrlHandler("",
                                                         &serviceCtrlHandler);
    if (process.m_ServiceHandle == 0)
    {
        return;
    }

    if (process.m_MlMainFunc != 0)
    {
        using TScopedCharPArray = boost::scoped_array<char *>;

        // Merge the arguments from the service itself with the arguments
        // passed to the original main() call
        int mergedArgC(static_cast<int>(process.m_Args.size()));
        if (argv != 0 && argc > 1)
        {
            mergedArgC += static_cast<int>(argc - 1);
        }

        size_t index(0);
        TScopedCharPArray mergedArgV(new char *[mergedArgC]);
        for (TStrVecCItr iter = process.m_Args.begin();
             iter != process.m_Args.end();
             ++iter)
        {
            mergedArgV[index++] = const_cast<char *>(iter->c_str());
        }

        if (argv != 0 && argc > 1)
        {
            for (size_t arg = 1; arg < argc; ++arg)
            {
                mergedArgV[index++] = argv[arg];
            }
        }

        process.m_Running = true;
        int ret(process.m_MlMainFunc(mergedArgC, mergedArgV.get()));

        // Only log process status messages if the logger has been reconfigured
        // to log somewhere more sensible that STDERR.  (This prevents us
        // spoiling the output from --version and --help.)
        if (CLogger::instance().hasBeenReconfigured())
        {
            LOG_INFO(STOPPED_MSG);
        }

        // Update the service status to say we've stopped
        SERVICE_STATUS serviceStatus;
        ::memset(&serviceStatus, 0, sizeof(serviceStatus));

        serviceStatus.dwServiceType = SERVICE_WIN32_OWN_PROCESS;
        serviceStatus.dwControlsAccepted = 0;
        serviceStatus.dwWin32ExitCode = (ret == EXIT_SUCCESS ? NO_ERROR : ERROR_SERVICE_SPECIFIC_ERROR);
        serviceStatus.dwServiceSpecificExitCode = static_cast<DWORD>(ret);
        serviceStatus.dwCheckPoint = 0;
        serviceStatus.dwWaitHint = 0;
        serviceStatus.dwCurrentState = SERVICE_STOPPED;
        SetServiceStatus(process.m_ServiceHandle, &serviceStatus);
    }
}

void WINAPI CProcess::serviceCtrlHandler(DWORD ctrlType)
{
    // This is a static method, so get the singleton instance
    CProcess &process = CProcess::instance();

    // If we're not running as a service, do nothing
    if (process.m_ServiceHandle == 0)
    {
        return;
    }

    SERVICE_STATUS serviceStatus;
    ::memset(&serviceStatus, 0, sizeof(serviceStatus));

    serviceStatus.dwServiceType = SERVICE_WIN32_OWN_PROCESS;

    // We only accept the basic start/stop commands, nothing fancy like
    // pause/continue.  Also, as per Microsoft's SetServiceStatus()
    // documentation, we don't accept stop until we've finished initialising, as
    // apparently doing this can cause a crash.
    serviceStatus.dwControlsAccepted = SERVICE_ACCEPT_SHUTDOWN;
    if (process.isInitialised())
    {
        serviceStatus.dwControlsAccepted |= SERVICE_ACCEPT_STOP;
    }
    serviceStatus.dwWin32ExitCode = NO_ERROR;
    serviceStatus.dwServiceSpecificExitCode = 0;
    serviceStatus.dwCheckPoint = 0;

    switch (ctrlType)
    {
        case SERVICE_CONTROL_INTERROGATE:
        {
            serviceStatus.dwWaitHint = 0;
            if (process.isRunning())
            {
                if (process.isInitialised())
                {
                    serviceStatus.dwCurrentState = SERVICE_RUNNING;
                }
                else
                {
                    serviceStatus.dwCurrentState = SERVICE_START_PENDING;
                }
            }
            else
            {
                serviceStatus.dwCurrentState = SERVICE_STOPPED;
            }
            SetServiceStatus(process.m_ServiceHandle, &serviceStatus);
            break;
        }
        case SERVICE_CONTROL_SHUTDOWN:
        case SERVICE_CONTROL_STOP:
        {
            serviceStatus.dwWaitHint = STOP_WAIT_HINT_MSECS;
            serviceStatus.dwCurrentState = SERVICE_STOP_PENDING;
            SetServiceStatus(process.m_ServiceHandle, &serviceStatus);
            if (process.shutdown() == false)
            {
                // This won't stop the process gracefully, and will trigger an
                // error message from Windows, but that's probably better than
                // having a rogue process hanging around after it's been told to
                // stop
                ::exit(EXIT_SUCCESS);
            }
            break;
        }
    }
}

bool CProcess::shutdown(void)
{
    if (CLogger::instance().hasBeenReconfigured())
    {
        LOG_INFO(STOPPING_MSG);
    }

    CScopedFastLock lock(m_ShutdownFuncMutex);

    if (!m_ShutdownFunc)
    {
        return false;
    }

    m_ShutdownFunc();

    return true;
}


}
}

