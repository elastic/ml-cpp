/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

// The lack of include guards is deliberate in this file, to allow per-file
// redefinition of logging macros

#include <boost/current_function.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/utility/manipulators/add_value.hpp>

#include <sstream>

// Location info
#ifdef LOG_LOCATION_INFO
#undef LOG_LOCATION_INFO
#endif
#define LOG_LOCATION_INFO                                                                 \
    << boost::log::add_value(ml::core::CLogger::instance().lineAttributeName(), __LINE__) \
    << boost::log::add_value(ml::core::CLogger::instance().fileAttributeName(), __FILE__) \
    << boost::log::add_value(ml::core::CLogger::instance().functionAttributeName(),       \
                             BOOST_CURRENT_FUNCTION)

// Log at a level known at compile time

#ifdef LOG_TRACE
#undef LOG_TRACE
#endif
#ifdef PROMOTE_LOGGING_TO_INFO
// When this option is set all LOG_TRACE macros are promoted to LOG_INFO
#define LOG_TRACE(message)                                                                  \
    BOOST_LOG_STREAM_SEV(ml::core::CLogger::instance().logger(), ml::core::CLogger::E_Info) \
    LOG_LOCATION_INFO                                                                       \
    message
#elif defined(EXCLUDE_TRACE_LOGGING)
// When this option is set TRACE logging is expanded to dummy code that can be
// eliminated from the compiled program (certainly when optimisation is
// enabled) - this avoids the overhead of checking the logging level at all for
// this low level logging
#define LOG_TRACE(message)                                                     \
    static_cast<void>([&]() { std::ostringstream() << "" message; })
#else
#define LOG_TRACE(message)                                                                   \
    BOOST_LOG_STREAM_SEV(ml::core::CLogger::instance().logger(), ml::core::CLogger::E_Trace) \
    LOG_LOCATION_INFO                                                                        \
    message
#endif
#ifdef LOG_DEBUG
#undef LOG_DEBUG
#endif
#ifdef PROMOTE_LOGGING_TO_INFO
// When this option is set all LOG_DEBUG macros are promoted to LOG_INFO
#define LOG_DEBUG(message)                                                                  \
    BOOST_LOG_STREAM_SEV(ml::core::CLogger::instance().logger(), ml::core::CLogger::E_Info) \
    LOG_LOCATION_INFO                                                                       \
    message
#else
#define LOG_DEBUG(message)                                                                   \
    BOOST_LOG_STREAM_SEV(ml::core::CLogger::instance().logger(), ml::core::CLogger::E_Debug) \
    LOG_LOCATION_INFO                                                                        \
    message
#endif
#ifdef LOG_INFO
#undef LOG_INFO
#endif
#define LOG_INFO(message)                                                                   \
    BOOST_LOG_STREAM_SEV(ml::core::CLogger::instance().logger(), ml::core::CLogger::E_Info) \
    LOG_LOCATION_INFO                                                                       \
    message
#ifdef LOG_WARN
#undef LOG_WARN
#endif
#define LOG_WARN(message)                                                                   \
    BOOST_LOG_STREAM_SEV(ml::core::CLogger::instance().logger(), ml::core::CLogger::E_Warn) \
    LOG_LOCATION_INFO                                                                       \
    message
#ifdef LOG_ERROR
#undef LOG_ERROR
#endif
#define LOG_ERROR(message)                                                                   \
    BOOST_LOG_STREAM_SEV(ml::core::CLogger::instance().logger(), ml::core::CLogger::E_Error) \
    LOG_LOCATION_INFO                                                                        \
    message
#ifdef LOG_FATAL
#undef LOG_FATAL
#endif
#define LOG_FATAL(message)                                                                   \
    BOOST_LOG_STREAM_SEV(ml::core::CLogger::instance().logger(), ml::core::CLogger::E_Fatal) \
    LOG_LOCATION_INFO                                                                        \
    message

#ifdef HANDLE_FATAL
#undef HANDLE_FATAL
#endif
#define HANDLE_FATAL(message)                                                  \
    {                                                                          \
        std::ostringstream ss;                                                 \
        ss message;                                                            \
        ml::core::CLogger::instance().handleFatal(ss.str());                   \
    }

#ifdef LOG_ABORT
#undef LOG_ABORT
#endif
#define LOG_ABORT(message)                                                                   \
    BOOST_LOG_STREAM_SEV(ml::core::CLogger::instance().logger(), ml::core::CLogger::E_Fatal) \
    LOG_LOCATION_INFO                                                                        \
    message;                                                                                 \
    ml::core::CLogger::fatal()

// Log at a level specified at runtime as a string, for example
// LOG_AT_LEVEL(ml::core::CLogger::E_Warn, << "Stay away from here " << username)

#ifdef LOG_AT_LEVEL
#undef LOG_AT_LEVEL
#endif
#define LOG_AT_LEVEL(level, message)                                           \
    BOOST_LOG_STREAM_SEV(ml::core::CLogger::instance().logger(), level)        \
    LOG_LOCATION_INFO                                                          \
    message
