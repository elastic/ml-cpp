/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CLogger.h>

#include <core/CCrashHandler.h>
#include <core/COsFileFuncs.h>
#include <core/CProcess.h>
#include <core/CProgName.h>
#include <core/CResourceLocator.h>
#include <core/CTimezone.h>
#include <core/CUname.h>
#include <core/CoreTypes.h>

#include <log4cxx/appender.h>
#include <log4cxx/helpers/exception.h>
#include <log4cxx/helpers/fileinputstream.h>
#include <log4cxx/helpers/transcoder.h>
#include <log4cxx/logmanager.h>
#include <log4cxx/logstring.h>
#include <log4cxx/propertyconfigurator.h>
#include <log4cxx/writerappender.h>

#include <iostream>
#include <sstream>
#include <stdexcept>

#include <errno.h>
#include <stdlib.h>
#include <string.h>

// environ is a global variable from the C runtime library
#ifdef Windows
__declspec(dllimport)
#endif
    extern char** environ;

namespace {
// To ensure the singleton is constructed before multiple threads may require it
// call instance() during the static initialisation phase of the program.  Of
// course, the instance may already be constructed before this if another static
// object has used it.
const ml::core::CLogger& DO_NOT_USE_THIS_VARIABLE = ml::core::CLogger::instance();
}

namespace ml {
namespace core {

CLogger::CLogger()
    : m_Logger(0), m_Reconfigured(false), m_ProgramName(CProgName::progName()),
      m_OrigStderrFd(-1) {
    CCrashHandler::installCrashHandler();
    this->reset();
}

CLogger::~CLogger() {
    log4cxx::LogManager::shutdown();
    m_Logger = nullptr;

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

    // Configure the logger
    try {
        // This info can come from an XML file or other source.
        // When the logger first starts up, configure it by setting properties
        // equivalent to this properties file:
        //
        // # Set root logger level to DEBUG and its only appender to A1.
        // log4j.rootLogger=DEBUG, A1
        //
        // # A1 is set to be a ConsoleAppender.
        // log4j.appender.A1=org.apache.log4j.ConsoleAppender
        // log4j.appender.A1.Target=System.err
        //
        // # A1 uses PatternLayout.
        // log4j.appender.A1.layout=org.apache.log4j.PatternLayout
        // log4j.appender.A1.layout.ConversionPattern=%d %d{%Z} %-5p [PID] %F@%L %m%n
        //
        // Having this hardcoded configuration means that the unit tests and
        // other utility programs just work with minimal effort.  Longer
        // running daemon processes are expected to reconfigure the logging
        // using a real properties file.

        log4cxx::helpers::Properties props;
        props.put(LOG4CXX_STR("log4j.rootLogger"), LOG4CXX_STR("DEBUG, A1"));
        props.put(LOG4CXX_STR("log4j.appender.A1"),
                  LOG4CXX_STR("org.apache.log4j.ConsoleAppender"));
        props.put(LOG4CXX_STR("log4j.appender.A1.Target"), LOG4CXX_STR("System.err"));
        props.put(LOG4CXX_STR("log4j.appender.A1.layout"),
                  LOG4CXX_STR("org.apache.log4j.PatternLayout"));

        // The pattern includes the process ID to make it easier to see if a
        // process dies and restarts
        std::ostringstream strm;
        strm << "%d %d{%Z} [" << CProcess::instance().id() << "] %-5p %F@%L %m%n";
        log4cxx::LogString logPattern;
        log4cxx::helpers::Transcoder::decode(strm.str(), logPattern);
        props.put(LOG4CXX_STR("log4j.appender.A1.layout.ConversionPattern"), logPattern);

        // Make sure the timezone names have been frigged on Windows before
        // configuring the properties
        CTimezone::instance();

        log4cxx::PropertyConfigurator::configure(props);

        m_Logger = log4cxx::Logger::getRootLogger();
    } catch (log4cxx::helpers::Exception& e) {
        if (m_Logger != nullptr) {
            // (Can't use the Ml LOG_ERROR macro here, as the object
            // it references is only part constructed.)
            LOG4CXX_ERROR(m_Logger, "Could not initialise logger: " << e.what());
        } else {
            // We can't use the log macros if the pointer to the logger is NULL
            std::cerr << "Could not initialise logger: " << e.what() << std::endl;
        }
    }
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

log4cxx::LoggerPtr CLogger::logger() {
    return m_Logger;
}

void CLogger::fatal() {
    throw std::runtime_error("Ml Fatal Exception");
}

bool CLogger::setLoggingLevel(ELevel level) {
    log4cxx::LevelPtr levelToSet(0);

    switch (level) {
    case E_Fatal:
        levelToSet = log4cxx::Level::getFatal();
        break;
    case E_Error:
        levelToSet = log4cxx::Level::getError();
        break;
    case E_Warn:
        levelToSet = log4cxx::Level::getWarn();
        break;
    case E_Info:
        levelToSet = log4cxx::Level::getInfo();
        break;
    case E_Debug:
        levelToSet = log4cxx::Level::getDebug();
        break;
    case E_Trace:
        levelToSet = log4cxx::Level::getTrace();
        break;
    }

    // Defend against corrupt argument
    if (levelToSet == nullptr) {
        return false;
    }

    // Get a smart pointer to the current logger - this may be different to the
    // active logger when we call its setLevel() method, but because it's a
    // smart pointer, at least it will still exist
    log4cxx::LoggerPtr loggerToChange(m_Logger);
    if (loggerToChange == nullptr) {
        return false;
    }

    loggerToChange->setLevel(levelToSet);

    // The appenders of the logger can also have seperate thresholds, and where
    // these are more restrictive than the logger itself, the above level
    // change will have no effect.  Therefore, we adjust all appender thresholds
    // here as well for appenders that write to a file or the console.
    log4cxx::AppenderList appendersToChange(loggerToChange->getAllAppenders());
    for (log4cxx::AppenderList::iterator iter = appendersToChange.begin();
         iter != appendersToChange.end(); ++iter) {
        log4cxx::Appender* appenderToChange(*iter);

        // Unfortunately, thresholds are a concept lower down the inheritance
        // hierarchy than the Appender base class, so we have to downcast.
        log4cxx::WriterAppender* writerToChange(
            dynamic_cast<log4cxx::WriterAppender*>(appenderToChange));
        if (writerToChange != nullptr) {
            writerToChange->setThreshold(levelToSet);
        }
    }

    return true;
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

    // When logging to standard error, log4cxx logs to the C FILE called stderr,
    // so switch this for a FILE opened on the named pipe.
    m_OrigStderrFd = COsFileFuncs::dup(::fileno(stderr));
    COsFileFuncs::dup2(::fileno(m_PipeFile.get()), ::fileno(stderr));

    if (this->reconfigureLogJson() == false) {
        return false;
    }

    LOG_DEBUG(<< "Logger is logging to named pipe " << pipeName);

    return true;
}

bool CLogger::reconfigureLogJson() {
    log4cxx::helpers::Properties props;
    log4cxx::LogString logStr;
    log4cxx::helpers::Transcoder::decode(m_ProgramName, logStr);
    props.put(LOG4CXX_STR("log4j.logger.") + logStr, LOG4CXX_STR("DEBUG, A2"));
    props.put(LOG4CXX_STR("log4j.appender.A2"),
              LOG4CXX_STR("org.apache.log4j.ConsoleAppender"));
    props.put(LOG4CXX_STR("log4j.appender.A2.Target"), LOG4CXX_STR("System.err"));
    props.put(LOG4CXX_STR("log4j.appender.A2.layout"),
              LOG4CXX_STR("org.apache.log4j.CJsonLogLayout"));

    return this->reconfigureFromProps(props);
}

bool CLogger::reconfigureFromFile(const std::string& propertiesFile) {
    COsFileFuncs::TStat statBuf;
    if (COsFileFuncs::stat(propertiesFile.c_str(), &statBuf) != 0) {
        LOG_ERROR(<< "Unable to access properties file " << propertiesFile
                  << " for logger re-initialisation: " << ::strerror(errno));
        return false;
    }

    // Load the properties prior to starting the reconfiguration.  This means
    // we get the chance to massage the properties to include the name of the
    // current application, before log4cxx uses them.
    log4cxx::helpers::Properties props;
    try {
        // InputStreamPtr is a smart pointer
        log4cxx::helpers::InputStreamPtr inputStream(
            new log4cxx::helpers::FileInputStream(propertiesFile));
        props.load(inputStream);
    } catch (const log4cxx::helpers::Exception& e) {
        LOG_ERROR(<< "Unable to read from properties file " << propertiesFile
                  << " for logger re-initialisation: " << e.what());
        return false;
    }

    // Massage the properties with our extensions
    this->massageProperties(props);

    if (this->reconfigureFromProps(props) == false) {
        return false;
    }

    LOG_DEBUG(<< "Logger re-initialised using properties file " << propertiesFile);

    return true;
}

bool CLogger::reconfigureFromProps(log4cxx::helpers::Properties& props) {
    // Now attempt the reconfiguration using the new properties
    try {
        log4cxx::LogManager::resetConfiguration();
        log4cxx::PropertyConfigurator::configure(props);

        // Get a logger with the same name as our log identifier, and use this
        // for logging.  Anything written to this logger will also go to the
        // root logger (due to additivity) and using a logger named after our
        // log identifier has the advantage that any messages sent to a remote
        // TCP server can be identified as having come from this process.
        m_Logger = log4cxx::Logger::getLogger(m_ProgramName);

        if (m_Logger == nullptr) {
            // We can't use the log macros if the pointer to the logger is NULL
            std::cerr << "Failed to reinitialise logger for " << m_ProgramName << std::endl;
            return false;
        }
    } catch (log4cxx::helpers::Exception& e) {
        if (m_Logger != nullptr) {
            LOG_ERROR(<< "Failed to reinitialise logger: " << e.what());
        } else {
            // We can't use the log macros if the pointer to the logger is NULL
            std::cerr << "Failed to reinitialise logger: " << e.what() << std::endl;
        }

        return false;
    }

    m_Reconfigured = true;

    // Start the new log file off with "uname -a" information so we know what
    // hardware problems occurred on
    LOG_DEBUG(<< "uname -a: " << CUname::all());

    return true;
}

void CLogger::massageProperties(log4cxx::helpers::Properties& props) const {
    // Get the process ID as a string
    std::ostringstream pidStrm;
    pidStrm << CProcess::instance().id();

    // Set up ML specific mappings
    // 1) %N with the program name
    // 2) %P with the program's process ID
    TLogCharLogStrMap mappings;
    log4cxx::LogString logStr;
    log4cxx::helpers::Transcoder::decode(m_ProgramName, logStr);
    mappings.insert(TLogCharLogStrMap::value_type(static_cast<log4cxx::logchar>('N'), logStr));
    logStr.clear();
    log4cxx::helpers::Transcoder::decode(pidStrm.str(), logStr);
    mappings.insert(TLogCharLogStrMap::value_type(static_cast<log4cxx::logchar>('P'), logStr));

    // Map the properties
    using TLogStringVec = std::vector<log4cxx::LogString>;
    using TLogStringVecCItr = TLogStringVec::const_iterator;

    TLogStringVec propNames(props.propertyNames());
    for (TLogStringVecCItr iter = propNames.begin(); iter != propNames.end(); ++iter) {
        log4cxx::LogString oldKey(*iter);
        log4cxx::LogString newKey;
        newKey.reserve(oldKey.length());
        this->massageString(mappings, oldKey, newKey);

        log4cxx::LogString oldValue(props.get(oldKey));
        log4cxx::LogString newValue;
        newValue.reserve(oldValue.length());
        this->massageString(mappings, oldValue, newValue);

        if (newValue != oldValue || newKey != oldKey) {
            props.put(newKey, newValue);
        }
    }
}

void CLogger::massageString(const TLogCharLogStrMap& mappings,
                            const log4cxx::LogString& oldStr,
                            log4cxx::LogString& newStr) const {
    newStr.clear();

    for (log4cxx::LogString::const_iterator iter = oldStr.begin();
         iter != oldStr.end(); ++iter) {
        // We ONLY want to replace the patterns in our map - other patterns are
        // left for log4cxx itself
        if (*iter == static_cast<log4cxx::logchar>('%')) {
            ++iter;
            if (iter == oldStr.end()) {
                newStr += static_cast<log4cxx::logchar>('%');
                break;
            }

            TLogCharLogStrMapCItr mapping = mappings.find(*iter);
            if (mapping == mappings.end()) {
                newStr += static_cast<log4cxx::logchar>('%');
                newStr += *iter;
            } else {
                newStr += mapping->second;
            }
        } else {
            newStr += *iter;
        }
    }
}
}
}
