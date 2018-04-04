/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CStrPTime_h
#define INCLUDED_ml_core_CStrPTime_h

#include <core/CNonInstantiatable.h>
#include <core/ImportExport.h>

#include <time.h>

namespace ml {
namespace core {

//! \brief
//! Parse the date/time string in the buffer buf, according to the
//! string pointed to by format.
//!
//! DESCRIPTION:\n
//! Parse the date/time string in the buffer buf, according to the
//! string pointed to by format, and fill in the elements of the structure
//! pointed to by tm.  The resulting values will be relative to the local
//! time zone.
//!
//! See strptime man page for details.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This has been broken into a class of its own because the strptime()
//! functions on Linux don't support the %z format, which is required
//! to parse localhost_access_log files.
//!
//! Also, strptime() on Linux is supposed to skip over a timezone name
//! indicated by the %Z format, but (at least on Fedora 9) it doesn't.
//! So Linux requires special handling for %Z too.
//!
class CORE_EXPORT CStrPTime : private CNonInstantiatable {
public:
    static char* strPTime(const char* buf, const char* format, struct tm* tm);
};
}
}

#endif // INCLUDED_ml_core_CStrPTime_h
