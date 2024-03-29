/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

// The lack of include guards is deliberate in this file, to allow per-file
// redefinition of logging macros

#include <core/CContainerPrinter.h>

#include <boost/current_function.hpp>
#include <boost/log/utility/manipulators/add_value.hpp>

#include <cstddef>
#include <sstream>
#include <string>

// Location info
#ifdef LOG_LOCATION_INFO
#undef LOG_LOCATION_INFO
#endif
#define LOG_LOCATION_INFO                                                                 \
    << boost::log::add_value(ml::core::CLogger::instance().lineAttributeName(), __LINE__) \
    << boost::log::add_value(ml::core::CLogger::instance().fileAttributeName(), __FILE__) \
    << boost::log::add_value(ml::core::CLogger::instance().functionAttributeName(),       \
                             BOOST_CURRENT_FUNCTION)                                      \
    << ml::core::CScopePrintContainers {}

// Log at a level known at compile time

#ifdef LOG_TRACE
#undef LOG_TRACE
#endif
#ifdef SUPPRESS_USAGE_WARNING
#undef SUPPRESS_USAGE_WARNING
#endif
#ifdef EXCLUDE_TRACE_LOGGING
// Compiling trace logging is expensive so if we don't want it just discard
// the code in the preprocessor.
#define LOG_TRACE(message)
// Avoids compiler warning in the case a variable is only used in LOG_TRACE.
#define SUPPRESS_USAGE_WARNING(variable) static_cast<void>(&(variable))
#else
#ifdef PROMOTE_LOGGING_TO_INFO
// When this option is set all LOG_TRACE macros are promoted to LOG_INFO
#define LOG_TRACE(message)                                                                  \
    BOOST_LOG_STREAM_SEV(ml::core::CLogger::instance().logger(), ml::core::CLogger::E_Info) \
    LOG_LOCATION_INFO                                                                       \
    message
#else
#define LOG_TRACE(message)                                                                   \
    BOOST_LOG_STREAM_SEV(ml::core::CLogger::instance().logger(), ml::core::CLogger::E_Trace) \
    LOG_LOCATION_INFO                                                                        \
    message
#endif
#define SUPPRESS_USAGE_WARNING(variable)
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
#define LOG_WARN(message)                                                                 \
    do {                                                                                  \
        std::size_t countOfWarnMessages;                                                  \
        bool skipWarnMessage;                                                             \
        std::tie(countOfWarnMessages, skipWarnMessage) =                                  \
            ml::core::CLogger::instance().throttler().skip(__FILE__, __LINE__);           \
        if (skipWarnMessage == false) {                                                   \
            BOOST_LOG_STREAM_SEV(ml::core::CLogger::instance().logger(),                  \
                                 ml::core::CLogger::E_Warn)                               \
            LOG_LOCATION_INFO                                                             \
            message << (countOfWarnMessages > 1                                           \
                            ? " | repeated [" + std::to_string(countOfWarnMessages) + "]" \
                            : "");                                                        \
        }                                                                                 \
    } while (0)
#ifdef LOG_ERROR
#undef LOG_ERROR
#endif
#define LOG_ERROR(message)                                                                 \
    do {                                                                                   \
        std::size_t countOfErrorMessages;                                                  \
        bool skipErrorMessage;                                                             \
        std::tie(countOfErrorMessages, skipErrorMessage) =                                 \
            ml::core::CLogger::instance().throttler().skip(__FILE__, __LINE__);            \
        if (skipErrorMessage == false) {                                                   \
            BOOST_LOG_STREAM_SEV(ml::core::CLogger::instance().logger(),                   \
                                 ml::core::CLogger::E_Error)                               \
            LOG_LOCATION_INFO                                                              \
            message << (countOfErrorMessages > 1                                           \
                            ? " | repeated [" + std::to_string(countOfErrorMessages) + "]" \
                            : "");                                                         \
        }                                                                                  \
    } while (0)
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
#define HANDLE_FATAL(message)                                                              \
    ml::core::CLogger::instance().handleFatal(                                             \
        static_cast<std::ostringstream*>(                                                  \
            (std::ostringstream{} << ml::core::CScopePrintContainers {} message).stream()) \
            ->str())

#ifdef LOG_ABORT
#undef LOG_ABORT
#endif
#define LOG_ABORT(message)                                                                   \
    BOOST_LOG_STREAM_SEV(ml::core::CLogger::instance().logger(), ml::core::CLogger::E_Fatal) \
    LOG_LOCATION_INFO                                                                        \
    message;                                                                                 \
    ml::core::CLogger::fatal()

// Log at a level specified at runtime, for example
// LOG_AT_LEVEL(ml::core::CLogger::E_Warn, << "Stay away from here " << username)

#ifdef LOG_AT_LEVEL
#undef LOG_AT_LEVEL
#endif
#define LOG_AT_LEVEL(level, message)                                           \
    BOOST_LOG_STREAM_SEV(ml::core::CLogger::instance().logger(), level)        \
    LOG_LOCATION_INFO                                                          \
    message
