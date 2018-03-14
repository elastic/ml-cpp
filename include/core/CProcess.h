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
    static const char* STARTING_MSG;
    static const char* STARTED_MSG;
    static const char* STOPPING_MSG;
    static const char* STOPPED_MSG;

public:
    //! Prototype of the mlMain() function
    typedef int (*TMlMainFunc)(int, char* []);

    //! Vector of process arguments
    typedef std::vector<std::string> TStrVec;
    typedef TStrVec::const_iterator TStrVecCItr;

    //! The shutdown function
    typedef std::function<void()> TShutdownFunc;

    //! Process ID type
#ifdef Windows
    typedef DWORD TPid;
#else
    typedef pid_t TPid;
#endif

public:
    //! Access to singleton
    static CProcess& instance(void);

    //! Is this process running as a Windows service?
    bool isService(void) const;

    //! Get the process ID
    TPid id(void) const;

    //! Get the parent process ID
    TPid parentId(void) const;

    //! If this process is not running as a Windows service, this call will
    //! immediately pass control to the mlMain() function.  If this
    //! process is running as a Windows service, the thread that calls this
    //! method will become the service dispatcher thread.
    bool startDispatcher(TMlMainFunc mlMain, int argc, char* argv[]);

    //! Check if the application is initialised
    bool isInitialised(void) const;

    //! Record successful completion of the application's initialisation
    //! phase.  This must be passed a shutdown function that can be used
    //! to stop the application gracefully if requested.
    void initialisationComplete(const TShutdownFunc& shutdownFunc);

    //! Record successful completion of the application's initialisation
    //! phase.  No shutdown function is passed, so the application will
    //! not be able to stop gracefully.
    void initialisationComplete(void);

    //! Check if the application is running
    bool isRunning(void) const;

    //! Instruct the application to shutdown gracefully.  This will only
    //! succeed if initialisation has been reported to be complete.  (Even
    //! if this method returns success, the application will only shut
    //! down as gracefully if the shutdown function works as it should.)
    bool shutdown(void);

#ifdef Windows
    //! Windows service main function
    static void WINAPI serviceMain(DWORD argc, char* argv[]);

    //! Windows service control function
    static void WINAPI serviceCtrlHandler(DWORD ctrlType);
#endif

private:
    // Constructor for a singleton is private.
    CProcess(void);

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
