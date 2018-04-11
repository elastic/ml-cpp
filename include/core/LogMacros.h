/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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

// The lack of include guards is deliberate in this file, to allow per-file
// redefinition of logging macros

#include <log4cxx/logger.h>

// Log at a level known at compile time

#ifdef LOG_TRACE
#undef LOG_TRACE
#endif
#ifdef PROMOTE_LOGGING_TO_INFO
// When this option is set all LOG_TRACE macros are promoted to LOG_INFO
#define LOG_TRACE(message) LOG4CXX_INFO(ml::core::CLogger::instance().logger(), "" message)
#elif defined(EXCLUDE_TRACE_LOGGING)
// When this option is set TRACE logging is expanded to nothing - this avoids
// the overhead of checking the logging level at all for this low level logging
#define LOG_TRACE(message)
#else
#define LOG_TRACE(message) LOG4CXX_TRACE(ml::core::CLogger::instance().logger(), "" message)
#endif
#ifdef LOG_DEBUG
#undef LOG_DEBUG
#endif
#ifdef PROMOTE_LOGGING_TO_INFO
// When this option is set all LOG_DEBUG macros are promoted to LOG_INFO
#define LOG_DEBUG(message) LOG4CXX_INFO(ml::core::CLogger::instance().logger(), "" message)
#else
#define LOG_DEBUG(message) LOG4CXX_DEBUG(ml::core::CLogger::instance().logger(), "" message)
#endif
#ifdef LOG_INFO
#undef LOG_INFO
#endif
#define LOG_INFO(message) LOG4CXX_INFO(ml::core::CLogger::instance().logger(), "" message)
#ifdef LOG_WARN
#undef LOG_WARN
#endif
#define LOG_WARN(message) LOG4CXX_WARN(ml::core::CLogger::instance().logger(), "" message)
#ifdef LOG_ERROR
#undef LOG_ERROR
#endif
#define LOG_ERROR(message) LOG4CXX_ERROR(ml::core::CLogger::instance().logger(), "" message)
#ifdef LOG_FATAL
#undef LOG_FATAL
#endif
#define LOG_FATAL(message) LOG4CXX_FATAL(ml::core::CLogger::instance().logger(), "" message)
#ifdef LOG_ABORT
#undef LOG_ABORT
#endif
#define LOG_ABORT(message)                                                                                                                 \
    LOG4CXX_FATAL(ml::core::CLogger::instance().logger(), "" message);                                                                     \
    ml::core::CLogger::fatal()

// Log at a level specified at runtime as a string, for example
// LOG_AT_LEVEL("WARN", << "Stay away from here " << username)

#ifdef LOG_AT_LEVEL
#undef LOG_AT_LEVEL
#endif
#define LOG_AT_LEVEL(level, message) LOG4CXX_LOGLS(ml::core::CLogger::instance().logger(), log4cxx::Level::toLevel(level), "" message)
