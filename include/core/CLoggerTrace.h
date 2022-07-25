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

#ifndef INCLUDED_ml_core_CLoggerTrace_h
#define INCLUDED_ml_core_CLoggerTrace_h

#ifdef EXCLUDE_TRACE_LOGGING
// Compiling trace logging is expensive so if we don't want it just discard
// the code in the preprocessor.
#ifdef LOG_TRACE
#undef LOG_TRACE
#endif
#define LOG_TRACE(message)
// Avoids compiler warning in the case a variable is only used in LOG_TRACE.
#ifdef SUPPRESS_USAGE_WARNING
#undef SUPPRESS_USAGE_WARNING
#endif
#define SUPPRESS_USAGE_WARNING(variable) static_cast<void>(&(variable))
#else
#include <core/CLogger.h>
#endif

#endif // INCLUDED_ml_core_CLoggerTrace_h
