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

#include <boost/log/sources/severity_logger.hpp>

#include <functional>
#include <iosfwd>

#include <stdio.h> // fileno() is not C++ so need the C header

namespace ml {
namespace core {

//! \brief
//! Core logging class in ML.
//!
//! DESCRIPTION:\n
//! Core logging class in ML.  Access to the actual logging
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
//! Wrapper around Boost.Log.
//!
//! Singleton for simplicity.
//!
//! By default, logging is to stderr.  However, it's possible to
//! tell the logger to reinitialise itself using a properties file.
//! The vast majority of ML processes that we ship will
//! log to a named pipe that's being read from by the Elasticsearch
//! JVM.
//!
//! The TRACE level of logging should never be enabled in shipped
//! product, but can be useful when a unit test needs to log more
//! detailed information.
//!
class CORE_EXPORT CLogger : private CNonCopyable {
public:
    using TFatalErrorHandler = std::function<void(std::string)>;

    //! Used to set the level we should log at
    enum ELevel { E_Trace, E_Debug, E_Info, E_Warn, E_Error, E_Fatal };

    using TLevelSeverityLogger = boost::log::sources::severity_logger_mt<ELevel>;

    //! \brief Sets the fatal error handler to a specified value for
    //! the object lifetime.
    class CORE_EXPORT CScopeSetFatalErrorHandler : private CNonCopyable {
    public:
        CScopeSetFatalErrorHandler(const TFatalErrorHandler& handler);
        ~CScopeSetFatalErrorHandler();

    private:
        TFatalErrorHandler m_OriginalFatalErrorHandler;
    };

public:
    //! Access to singleton - use MACROS to get to this when logging
    //! messages
    static CLogger& instance();

    //! Reconfigure to either a named pipe or a properties file.
    //! If both are supplied the named pipe takes precedence.
    bool reconfigure(const std::string& pipeName, const std::string& propertiesFile);

    //! Tell the logger to log to a named pipe rather than a file.
    bool reconfigureLogToNamedPipe(const std::string& pipeName);

    //! Tell the logger to reconfigure itself by reading a specified
    //! properties file, if the file exists.
    bool reconfigureFromFile(const std::string& propertiesFile);

    //! Tell the logger to reconfigure itself to log JSON.
    bool reconfigureLogJson();

    //! Set the logging level on the fly - useful when unit tests need to
    //! log at a lower level than the shipped programs
    bool setLoggingLevel(ELevel level);

    //! Map the level enum to a string.
    static const std::string& levelToString(ELevel level);

    //! Has the logger been reconfigured?  Callers should note that there
    //! is nothing to stop the logger being reconfigured between a call to
    //! this method and them using the result.
    bool hasBeenReconfigured() const;

    //! Log all environment variables.  Callers are responsible for ensuring
    //! that this method is not called at the same time as a putenv() or
    //! setenv() call in another thread.
    void logEnvironment() const;

    //! Access to underlying logger (must only be called from macros)
    TLevelSeverityLogger& logger();

    //! Throw a fatal exception
    [[noreturn]] static void fatal();

    //! Register a new global fatal error handler.
    //!
    //! \note This is not thread safe as the intention is that it is invoked
    //! once, usually at the beginning of main or in single threaded test code.
    //! \note The default behaviour is to exit the process with an error status.
    void fatalErrorHandler(const TFatalErrorHandler& handler);

    //! Get the current fatal error handler.
    const TFatalErrorHandler& fatalErrorHandler() const;

    //! Handle a fatal problem using the registered fatal error handler.
    void handleFatal(std::string message);

    //! Attribute names for efficient access to our custom attributes
    boost::log::attribute_name fileAttributeName() const;
    boost::log::attribute_name lineAttributeName() const;
    boost::log::attribute_name functionAttributeName() const;

    //! Reset the logger, this is primarily a helper for unit testing as
    //! CLogger is a singleton, so we can not just create new instances
    void reset();

private:
    //! Constructor for a singleton is private.
    CLogger();
    ~CLogger();

    //! Helper for other reconfiguration methods
    bool reconfigureFromSettings(std::istream& settingsStrm);

    //! The default implementation of the fatal error handler causes the process
    //! to exit with non zero return code.
    [[noreturn]] static void defaultFatalErrorHandler(std::string message);

private:
    TLevelSeverityLogger m_Logger;

    //! Has the logger ever been reconfigured?  This is not protected by a
    //! lock despite the fact that it may be accessed from different
    //! threads.  It is declared volatile to prevent the compiler optimising
    //! away reads of it.
    volatile bool m_Reconfigured;

    //! Custom Boost.Log attribute names
    boost::log::attribute_name m_FileAttributeName;
    boost::log::attribute_name m_LineAttributeName;
    boost::log::attribute_name m_FunctionAttributeName;

    //! When logging to a named pipe this stores the C FILE pointer to
    //! access the pipe.  Should be NULL otherwise.
    CNamedPipeFactory::TFileP m_PipeFile;

    //! When logging to a pipe, the file descriptor that stderr was
    //! originally associated with.
    int m_OrigStderrFd;

    //! The default handler for fatal errors.
    TFatalErrorHandler m_FatalErrorHandler;
};

CORE_EXPORT std::ostream& operator<<(std::ostream& strm, CLogger::ELevel level);
CORE_EXPORT std::istream& operator>>(std::istream& strm, CLogger::ELevel& level);
}
}

#endif // INCLUDED_ml_core_CLogger_h
