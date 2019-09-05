/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CLogger.h>

#include <core/CCrashHandler.h>
#include <core/COsFileFuncs.h>
#include <core/CUname.h>
#include <core/CoreTypes.h>

#include <boost/log/attributes.hpp>
#include <boost/log/common.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks.hpp>
#include <boost/log/sources/logger.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/log/utility/manipulators/add_value.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

// environ is a global variable from the C runtime library
#ifdef Windows
__declspec(dllimport)
#endif
    extern char** environ;

namespace {
const std::string TRACE{"TRACE"};
const std::string DEBUG{"DEBUG"};
const std::string INFO{"INFO"};
const std::string WARN{"WARN"};
const std::string ERROR{"ERROR"};
const std::string FATAL{"FATAL"};

// To ensure the singleton is constructed before multiple threads may require it
// call instance() during the static initialisation phase of the program.  Of
// course, the instance may already be constructed before this if another static
// object has used it.
const ml::core::CLogger& DO_NOT_USE_THIS_VARIABLE = ml::core::CLogger::instance();
}

namespace ml {
namespace core {

CLogger::CLogger()
    : m_Reconfigured(false), m_OrigStderrFd(-1),
      m_FatalErrorHandler(defaultFatalErrorHandler) {
    CCrashHandler::installCrashHandler();
    // This adds LineID, TimeStamp, ProcessID and ThreadID
    boost::log::add_common_attributes();
    this->reset();
}

CLogger::~CLogger() {
    m_Logger.reset();

    if (m_PipeFile != nullptr) {
        // Revert the stderr file descriptor.
        if (m_OrigStderrFd != -1) {
            COsFileFuncs::dup2(m_OrigStderrFd, ::fileno(stderr));
        }
        m_PipeFile.reset();
    }
}

void CLogger::reset() {
    if (m_PipeFile != nullptr) {
        // Revert the stderr file descriptor.
        if (m_OrigStderrFd != -1) {
            COsFileFuncs::dup2(m_OrigStderrFd, ::fileno(stderr));
        }
        m_PipeFile.reset();
    }

    m_Reconfigured = false;

    TLevelSeverityLoggerPtr newLogger{std::make_shared<TLevelSeverityLogger>()};

    // TODO log4j.appender.A1.layout.ConversionPattern=%d %d{%Z} %-5p [PID] %F@%L %m%n

    boost::log::formatter formatter =
        boost::log::expressions::stream
        << boost::log::expressions::format_date_time<boost::posix_time::ptime>(
               "TimeStamp", "%Y-%m-%d, %H:%M:%S,%f %z")
        << " " << std::setw(5) << boost::log::expressions::attr<ELevel>("Severity") << " ["
        << boost::log::expressions::attr<boost::log::attributes::current_process_id::value_type>("ProcessID")
        << "] "
        << "TODO file@line"
        << " - " << boost::log::expressions::smessage;

    /* TODO
            // We can't use the log macros if the pointer to the logger is NULL
            std::cerr << "Could not initialise logger: " << e.what() << std::endl;
    */

    m_Logger = newLogger;
}

CLogger& CLogger::instance() {
    static CLogger instance;
    return instance;
}

bool CLogger::hasBeenReconfigured() const {
    return m_Reconfigured;
}

void CLogger::logEnvironment() const {
    std::string env("Environment variables:");
    // environ is a global variable from the C runtime library
    if (environ == nullptr) {
        env += " (None found)";
    } else {
        for (char** envPtr = environ; *envPtr != nullptr; ++envPtr) {
            env += core_t::LINE_ENDING;
            env += *envPtr;
        }
    }
    LOG_INFO(<< env);
}

CLogger::TLevelSeverityLoggerPtr CLogger::logger() {
    return m_Logger;
}

void CLogger::fatal() {
    throw std::runtime_error("Ml Fatal Exception");
}

void CLogger::fatalErrorHandler(const TFatalErrorHandler& handler) {
    // This is purposely not thread safe. This is only meant to be called once from
    // the main thread, typically from main of an executable or in single threaded
    // test code.
    m_FatalErrorHandler = handler;
}

const CLogger::TFatalErrorHandler& CLogger::fatalErrorHandler() const {
    return m_FatalErrorHandler;
}

void CLogger::handleFatal(std::string message) {
    m_FatalErrorHandler(std::move(message));
}

bool CLogger::setLoggingLevel(ELevel level) {

    // TODO m_Logger->setLevel(level);

    return true;
}

const std::string& CLogger::levelToString(ELevel level) {

    switch (level) {
    case E_Trace:
        return TRACE;
    case E_Debug:
        return DEBUG;
    case E_Info:
        return INFO;
    case E_Warn:
        return WARN;
    case E_Error:
        return ERROR;
    case E_Fatal:
        return FATAL;
    }

    // Default to warning on invalid input
    return WARN;
}

bool CLogger::reconfigure(const std::string& pipeName, const std::string& propertiesFile) {
    if (pipeName.empty()) {
        if (propertiesFile.empty()) {
            // Both empty is OK - it just means we keep logging to stderr
            return true;
        }
        return this->reconfigureFromFile(propertiesFile);
    }
    return this->reconfigureLogToNamedPipe(pipeName);
}

bool CLogger::reconfigureLogToNamedPipe(const std::string& pipeName) {
    if (m_Reconfigured) {
        LOG_ERROR(<< "Cannot log to a named pipe after logger reconfiguration");
        return false;
    }

    m_PipeFile = CNamedPipeFactory::openPipeFileWrite(pipeName);
    if (m_PipeFile == nullptr) {
        LOG_ERROR(<< "Cannot log to named pipe " << pipeName
                  << " as it could not be opened for writing");
        return false;
    }

    // Boost.Log logs to the std::clog stream, which in turn outputs to the OS
    // standard error file.  Rather than redirect at the C++ stream level it's
    // better to redirect at the level of the OS file because then errors
    // written directly to standard error by library/OS code will get redirected
    // to the pipe too, so switch this for a FILE opened on the named pipe.
    m_OrigStderrFd = COsFileFuncs::dup(::fileno(stderr));
    COsFileFuncs::dup2(::fileno(m_PipeFile.get()), ::fileno(stderr));

    if (this->reconfigureLogJson() == false) {
        return false;
    }

    LOG_DEBUG(<< "Logger is logging to named pipe " << pipeName);

    return true;
}

bool CLogger::reconfigureLogJson() {

    // TODO use CJsonLogLayout
    return true;
}

bool CLogger::reconfigureFromFile(const std::string& propertiesFile) {
    COsFileFuncs::TStat statBuf;
    if (COsFileFuncs::stat(propertiesFile.c_str(), &statBuf) != 0) {
        LOG_ERROR(<< "Unable to access properties file " << propertiesFile
                  << " for logger re-initialisation: " << std::strerror(errno));
        return false;
    }

    std::ifstream fileStrm(propertiesFile);
    if (this->reconfigureFromSettings(fileStrm) == false) {
        return false;
    }

    LOG_DEBUG(<< "Logger re-initialised using properties file " << propertiesFile);

    return true;
}

bool CLogger::reconfigureFromSettings(std::istream& settingsStrm) {

    //m_Logger = TODO

    if (m_Logger == nullptr) {
        // We can't use the log macros if the pointer to the logger is NULL
        std::cerr << "Failed to reinitialise logger" << std::endl;
        return false;
    }

    m_Reconfigured = true;

    // Start the new log file off with "uname -a" information so we know what
    // hardware problems occurred on
    LOG_DEBUG(<< "uname -a: " << CUname::all());

    return true;
}

void CLogger::defaultFatalErrorHandler(std::string message) {
    LOG_FATAL(<< message);
    std::exit(EXIT_FAILURE);
}

CLogger::CScopeSetFatalErrorHandler::CScopeSetFatalErrorHandler(const TFatalErrorHandler& handler)
    : m_OriginalFatalErrorHandler{CLogger::instance().fatalErrorHandler()} {
    CLogger::instance().fatalErrorHandler(handler);
}

CLogger::CScopeSetFatalErrorHandler::~CScopeSetFatalErrorHandler() {
    CLogger::instance().fatalErrorHandler(m_OriginalFatalErrorHandler);
}

std::ostream& operator<<(std::ostream& strm, CLogger::ELevel level) {
    strm << CLogger::levelToString(level);
    return strm;
}
}
}
