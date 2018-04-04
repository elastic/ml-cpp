/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CLogger_h
#define INCLUDED_ml_core_CLogger_h

#include <core/CNamedPipeFactory.h>
#include <core/CNonCopyable.h>
#include <core/ImportExport.h>
#include <core/LogMacros.h>

#include <log4cxx/helpers/properties.h>
#include <log4cxx/level.h>
#include <log4cxx/logger.h>

#include <stdio.h>

class CLoggerTest;

namespace ml
{
namespace core
{


//! \brief
//! Core logging class in Ml
//!
//! DESCRIPTION:\n
//! Core logging class in Ml.  Access to the actual logging
//! commands should be through macros.
//!
//! Errors that mean something has gone wrong, but the application
//! will continue to run, possibly losing a small amount of data or
//! with slightly inferior results, should be logged with the
//! LOG_ERROR macro.
//!
//! Errors that mean an application is not going to work at all
//! and will soon exit should be logged with the LOG_FATAL macro.
//! The LOG_FATAL macro itself does not change the program flow in
//! any way, so other code must still be written to effect the
//! shutdown of the application.
//!
//! The LOG_ABORT macro should only be used in situations that
//! would NEVER occur if our code were free of bugs.  Do NOT use
//! LOG_ABORT for situations such as invalid config files, unexpected
//! data received on a socket, or other things that might happen
//! at a customer site.  Effectively, a LOG_ABORT is a fancy version
//! of assert().
//!
//! IMPLEMENTATION DECISIONS:\n
//! Wrapper around Apache log4cxx.
//!
//! Singleton for simplicity.
//!
//! By default, logging is to stderr.  However, it's possible to
//! tell the logger to reinitialise itself using a properties file.
//! The vast majority of Ml processes that we ship will
//! log to a named pipe that's being read from by the Elasticsearch
//! JVM.
//!
//! The TRACE level of logging should never be enabled in shipped
//! product, but can be useful when a unit test needs to log more
//! detailed information.
//!
class CORE_EXPORT CLogger : private CNonCopyable
{
    public:
        //! Used to set the level we should log at
        enum ELevel
        {
            E_Fatal,
            E_Error,
            E_Warn,
            E_Info,
            E_Debug,
            E_Trace
        };

    public:
        //! Access to singleton - use MACROS to get to this when logging
        //! messages
        static CLogger     &instance();

        //! Reconfigure to either a named pipe or a properties file.
        //! If both are supplied the named pipe takes precedence.
        bool               reconfigure(const std::string &pipeName,
                                       const std::string &propertiesFile);

        //! Tell the logger to log to a named pipe rather than a file.
        bool               reconfigureLogToNamedPipe(const std::string &pipeName);

        //! Tell the logger to reconfigure itself by reading a specified
        //! properties file, if the file exists.
        bool               reconfigureFromFile(const std::string &propertiesFile);

        //! Tell the logger to reconfigure itself to log JSON.
        bool               reconfigureLogJson();

        //! Set the logging level on the fly - useful when unit tests need to
        //! log at a lower level than the shipped programs
        bool               setLoggingLevel(ELevel level);

        //! Has the logger been reconfigured?  Callers should note that there
        //! is nothing to stop the logger being reconfigured between a call to
        //! this method and them using the result.
        bool               hasBeenReconfigured() const;

        //! Log all environment variables.  Callers are responsible for ensuring
        //! that this method is not called at the same time as a putenv() or
        //! setenv() call in another thread.
        void               logEnvironment() const;

        //! Access to underlying logger (must only be called from macros)
        log4cxx::LoggerPtr logger();

#ifdef Windows
        //! Throw a fatal exception
        __declspec(noreturn) static void fatal();
#else
        //! Throw a fatal exception
        __attribute__ ((noreturn)) static void fatal();
#endif

    private:
        //! Constructor for a singleton is private.
        CLogger();
        ~CLogger();

        //! Replace Ml specific patterns in log4cxx properties.  In
        //! addition to the patterns usually supported by log4cxx, Ml will
        //! substitute:
        //! 1) %D with the path to the Ml base log directory
        //! 2) %N with the program's name
        //! 3) %P with the program's process ID
        void massageProperties(log4cxx::helpers::Properties &props) const;

        using TLogCharLogStrMap = std::map<log4cxx::logchar, log4cxx::LogString>;
        using TLogCharLogStrMapCItr = TLogCharLogStrMap::const_iterator;

        //! Replace Ml specific mappings in a single string
        void massageString(const TLogCharLogStrMap &mappings,
                           const log4cxx::LogString &oldStr,
                           log4cxx::LogString &newStr) const;

        //! Helper for other reconfiguration methods
        bool reconfigureFromProps(log4cxx::helpers::Properties &props);

        //! Reset the logger, this is a helper for unit testing as
        //! CLogger is a singleton, so we can not just create new instances
        void reset();
    private:
        log4cxx::LoggerPtr        m_Logger;

        //! Has the logger ever been reconfigured?  This is not protected by a
        //! lock despite the fact that it may be accessed from different
        //! threads.  It is declared volatile to prevent the compiler optimising
        //! away reads of it.
        volatile bool             m_Reconfigured;

        //! Cache the program name
        std::string               m_ProgramName;

        //! When logging to a named pipe this stores the C FILE pointer to
        //! access the pipe.  Should be NULL otherwise.
        CNamedPipeFactory::TFileP m_PipeFile;

        //! When logging to a pipe, the file descriptor that stderr was
        //! originally associated with.
        int                       m_OrigStderrFd;

        //! friend class for testing
        friend class ::CLoggerTest;
};


}
}

#endif // INCLUDED_ml_core_CLogger_h

