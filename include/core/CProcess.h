/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CProcess_h
#define INCLUDED_ml_core_CProcess_h

#include <core/CFastMutex.h>
#include <core/CNonCopyable.h>
#include <core/ImportExport.h>
#include <core/WindowsSafe.h>

#include <functional>
#include <string>
#include <vector>

#ifndef Windows
#include <unistd.h>
#endif

namespace ml {
namespace core {

//! \brief
//! Encapsulates actions related to process lifecycle
//!
//! DESCRIPTION:\n
//! Encapsulates actions related to process lifecycle, for example,
//! control of a Windows service.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Implemented as a singleton as it only makes sense to have one
//! object representing the state of the overall process.
//!
//! The Unix implementation does very little.  The Windows
//! implementation implements the calls that are required by
//! Windows for applications that want to run as Windows services.
//!
class CORE_EXPORT CProcess : private CNonCopyable {
public:
    //! These messages need to be 100% standard across all services
    static const char* const STARTING_MSG;
    static const char* const STARTED_MSG;
    static const char* const STOPPING_MSG;
    static const char* const STOPPED_MSG;

public:
    //! Prototype of the mlMain() function
    using TMlMainFunc = int (*)(int, char* []);

    //! Vector of process arguments
    using TStrVec = std::vector<std::string>;
    using TStrVecCItr = TStrVec::const_iterator;

    //! The shutdown function
    using TShutdownFunc = std::function<void()>;

    //! Process ID type
#ifdef Windows
    using TPid = DWORD;
#else
    using TPid = pid_t;
#endif

public:
    //! Access to singleton
    static CProcess& instance();

    //! Is this process running as a Windows service?
    bool isService() const;

    //! Get the process ID
    TPid id() const;

    //! Get the parent process ID
    TPid parentId() const;

    //! If this process is not running as a Windows service, this call will
    //! immediately pass control to the mlMain() function.  If this
    //! process is running as a Windows service, the thread that calls this
    //! method will become the service dispatcher thread.
    bool startDispatcher(TMlMainFunc mlMain, int argc, char* argv[]);

    //! Check if the application is initialised
    bool isInitialised() const;

    //! Record successful completion of the application's initialisation
    //! phase.  This must be passed a shutdown function that can be used
    //! to stop the application gracefully if requested.
    void initialisationComplete(const TShutdownFunc& shutdownFunc);

    //! Record successful completion of the application's initialisation
    //! phase.  No shutdown function is passed, so the application will
    //! not be able to stop gracefully.
    void initialisationComplete();

    //! Check if the application is running
    bool isRunning() const;

    //! Instruct the application to shutdown gracefully.  This will only
    //! succeed if initialisation has been reported to be complete.  (Even
    //! if this method returns success, the application will only shut
    //! down as gracefully if the shutdown function works as it should.)
    bool shutdown();

#ifdef Windows
    //! Windows service main function
    static void WINAPI serviceMain(DWORD argc, char* argv[]);

    //! Windows service control function
    static void WINAPI serviceCtrlHandler(DWORD ctrlType);
#endif

private:
    // Constructor for a singleton is private.
    CProcess();

private:
    //! Is this process running as a Windows service?
    bool m_IsService;

    //! Is this process initialised?
    bool m_Initialised;

    //! Is this process running?
    bool m_Running;

    //! Address of the mlMain() function to call
    TMlMainFunc m_MlMainFunc;

    //! Original arguments passed to the program's main() function
    TStrVec m_Args;

#ifdef Windows
    //! Service handle (will be 0 if we're not running as a service)
    SERVICE_STATUS_HANDLE m_ServiceHandle;
#endif

    //! Lock to protect the shutdown function object
    CFastMutex m_ShutdownFuncMutex;

    //! Function to call if the process is instructed to shut down.
    //! Will be empty until initialisation is complete.
    TShutdownFunc m_ShutdownFunc;
};
}
}

#endif // INCLUDED_ml_core_CProcess_h
