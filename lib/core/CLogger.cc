/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CLogger.h>

#include <core/CCrashHandler.h>
#include <core/CJsonLogLayout.h>
#include <core/COsFileFuncs.h>
#include <core/CProcess.h>
#include <core/CUname.h>
#include <core/CoreTypes.h>

#include <boost/core/null_deleter.hpp>
#include <boost/log/attributes.hpp>
#include <boost/log/common.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks.hpp>
#include <boost/log/sources/logger.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/log/utility/setup.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

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

// These must use boost::shared_ptr, not std, as that's what the Boost.Log interface
// uses
using TOStreamPtr = boost::shared_ptr<std::ostream>;
using TTextOStream = boost::log::sinks::text_ostream_backend;
using TTextOStreamSynchronousSink = boost::log::sinks::synchronous_sink<TTextOStream>;
using TTextOStreamSynchronousSinkPtr =
    boost::shared_ptr<boost::log::sinks::synchronous_sink<TTextOStream>>;

class CTimeStampFormatterFactory
    : public boost::log::basic_formatter_factory<char, boost::posix_time::ptime> {
private:
    static const std::string FORMAT;

public:
    formatter_type create_formatter(const boost::log::attribute_name& name,
                                    const args_map& args) {
        args_map::const_iterator iter{args.find(FORMAT)};
        if (iter != args.end()) {
            return boost::log::expressions::stream
                   << boost::log::expressions::format_date_time<boost::posix_time::ptime>(
                          boost::log::expressions::attr<boost::posix_time::ptime>(name),
                          iter->second);
        } else {
            return boost::log::expressions::stream
                   << boost::log::expressions::attr<boost::posix_time::ptime>(name);
        }
    }
};

const std::string CTimeStampFormatterFactory::FORMAT{"format"};

void resetSink(const TTextOStreamSynchronousSinkPtr& newSink) {
    auto loggingCorePtr = boost::log::core::get();
    loggingCorePtr->flush();
    loggingCorePtr->remove_all_sinks();
    if (newSink != nullptr) {
        loggingCorePtr->add_sink(newSink);
    }
}
}

namespace ml {
namespace core {

CLogger::CLogger()
    : m_Reconfigured(false), m_FileAttributeName("File"),
      m_LineAttributeName("Line"), m_FunctionAttributeName("Function"),
      m_OrigStderrFd(-1), m_FatalErrorHandler(defaultFatalErrorHandler) {
    CCrashHandler::installCrashHandler();
    // These formatter factories are not needed for programmatic configuration
    // of the format, but without them formats defined in settings files don't
    // work correctly.
    boost::log::register_simple_formatter_factory<ELevel, char>("Severity");
    boost::log::register_simple_filter_factory<ELevel, char>("Severity");
    boost::log::register_formatter_factory(
        "TimeStamp", boost::make_shared<CTimeStampFormatterFactory>());
    auto loggingCorePtr = boost::log::core::get();
    // By default Boost.Log logs in local time but doesn't remember the timezone.
    // For the case of logging to the ES JVM over a named pipe we want to send
    // the timestamp as milliseconds since the epoch, which requires either a
    // UTC timestamp or knowledge of the timezone.  For logging to a log file or
    // the console we want to print the timezone.  However, logging to a log
    // file or the console is only for internal development.  The pragmatic
    // approach here is to always log in UTC.  (This can be revisited in the
    // future if having console logs in UTC is ever a problem, but any fix would
    // be non-trivial.)
    loggingCorePtr->add_global_attribute(boost::log::aux::default_attribute_names::timestamp(),
                                         boost::log::attributes::utc_clock());
    loggingCorePtr->add_global_attribute(
        boost::log::aux::default_attribute_names::process_id(),
        boost::log::attributes::current_process_id());
    loggingCorePtr->add_global_attribute(
        boost::log::aux::default_attribute_names::thread_id(),
        boost::log::attributes::current_thread_id());
    this->reset();
}

CLogger::~CLogger() {
    resetSink(nullptr);

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

    // This must be boost::shared_ptr, not std, as that's what the Boost Log
    // interface uses
    auto backend = boost::make_shared<TTextOStream>();
    // Need a shared_ptr to std::cerr that will NOT delete it
    backend->add_stream(TOStreamPtr(&std::cerr, boost::null_deleter()));

    auto sinkPtr = boost::make_shared<TTextOStreamSynchronousSink>(backend);

    // Default format is as close as possible to log4cxx's %d %d{%Z} %-5p [PID] %F@%L %m%n
    sinkPtr->set_formatter(
        boost::log::expressions::stream
        << boost::log::expressions::format_date_time<boost::posix_time::ptime>(
               boost::log::aux::default_attribute_names::timestamp(), "%Y-%m-%d %H:%M:%S,%f")
        << " UTC [" << ml::core::CProcess::instance().id() << "] "
        << boost::log::expressions::attr<ELevel>(
               boost::log::aux::default_attribute_names::severity())
        << ' ' << boost::log::expressions::attr<std::string>(m_FileAttributeName)
        << '@' << boost::log::expressions::attr<int>(m_LineAttributeName) << ' '
        << boost::log::expressions::smessage);

    resetSink(sinkPtr);

    // Default filter is level debug and higher
    boost::log::core::get()->set_filter(
        boost::log::expressions::attr<ELevel>(
            boost::log::aux::default_attribute_names::severity()) >= E_Debug);

    m_Reconfigured = false;
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

CLogger::TLevelSeverityLogger& CLogger::logger() {
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

boost::log::attribute_name CLogger::fileAttributeName() const {
    return m_FileAttributeName;
}

boost::log::attribute_name CLogger::lineAttributeName() const {
    return m_LineAttributeName;
}

boost::log::attribute_name CLogger::functionAttributeName() const {
    return m_FunctionAttributeName;
}

bool CLogger::setLoggingLevel(ELevel level) {

    boost::log::core::get()->set_filter(
        boost::log::expressions::attr<ELevel>(
            boost::log::aux::default_attribute_names::severity()) >= level);
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

    // By default Boost.Log logs to the std::clog stream, which in turn outputs
    // to the OS standard error file.  Rather than redirect at the C++ stream
    // level it's better to redirect at the level of the OS file because then
    // errors written directly to standard error by library/OS code will get
    // redirected to the pipe too, so switch this for a FILE opened on the
    // named pipe.
    m_OrigStderrFd = COsFileFuncs::dup(::fileno(stderr));
    COsFileFuncs::dup2(::fileno(m_PipeFile.get()), ::fileno(stderr));

    if (this->reconfigureLogJson() == false) {
        return false;
    }

    LOG_DEBUG(<< "Logger is logging to named pipe " << pipeName);

    return true;
}

bool CLogger::reconfigureLogJson() {

    // This must be boost::shared_ptr, not std, as that's what the Boost Log
    // interface uses
    auto backend{boost::make_shared<TTextOStream>()};
    // Need a shared_ptr to std::cerr that will NOT delete it
    backend->add_stream(TOStreamPtr(&std::cerr, boost::null_deleter()));

    auto sinkPtr{boost::make_shared<TTextOStreamSynchronousSink>(backend)};

    CJsonLogLayout jsonLogLayout;
    sinkPtr->set_formatter(jsonLogLayout);

    resetSink(sinkPtr);

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

    // This is not ideal.  The settings file adds to the current config rather
    // than replacing it, so we have to remove the current config before parsing
    // the new settings.  But if there's an error in the new settings this means
    // we've lost the old ones.  Configuration files are currently only an
    // internal debugging feature, so it's not worth putting more effort into
    // this problem unless that changes.
    resetSink(nullptr);
    boost::log::core::get()->reset_filter();

    try {
        boost::log::init_from_stream(settingsStrm);
    } catch (std::exception& e) {
        this->reset();
        std::cerr << "Error initializing logger from settings: " << e.what() << std::endl;
        return false;
    }

    m_Reconfigured = true;

    // Start the new log off with "uname -a" information so we know what
    // hardware any subsequent problems occurred on
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
    strm << std::setw(5) << std::left << CLogger::levelToString(level);
    return strm;
}

std::istream& operator>>(std::istream& strm, CLogger::ELevel& level) {

    std::string token;
    strm >> token;

    if (token == FATAL) {
        level = CLogger::E_Fatal;
    } else if (token == ERROR) {
        level = CLogger::E_Error;
    } else if (token == WARN) {
        level = CLogger::E_Warn;
    } else if (token == INFO) {
        level = CLogger::E_Info;
    } else if (token == DEBUG) {
        level = CLogger::E_Debug;
    } else if (token == TRACE) {
        level = CLogger::E_Trace;
    } else {
        // This is fatal on the assumption that we only specify logging levels
        // in configuration files during internal development.  Obviously this
        // would need changing if specifying the logging level in a way that
        // required this parsing was available to end users.
        HANDLE_FATAL(<< "Input error: unknown logging level: '" << token << "'");
    }

    return strm;
}
}
}
