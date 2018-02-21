/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CStrFTime_h
#define INCLUDED_ml_core_CStrFTime_h

#include <core/CNonInstantiatable.h>
#include <core/ImportExport.h>

#include <time.h>


namespace ml
{
namespace core
{


//! \brief
//! Format the date/time struct into a string in the buffer buf, according to
//! the string pointed to by format.
//!
//! DESCRIPTION:\n
//! Format the date/time struct into a string in the buffer buf, according to
//! the string pointed to by format.
//!
//! See strftime man page for details.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This has been broken into a class of its own because the strftime()
//! function on Windows handles %z differently to the strftime() function on
//! Unix.  Windows formats %z as the textual representation of the time zone,
//! whereas Unix formats it as a numeric offset.  We want the numeric offset on
//! all platforms.
//!
class CORE_EXPORT CStrFTime : private CNonInstantiatable
{
    public:
        static size_t strFTime(char *buf,
                               size_t maxSize,
                               const char *format,
                               struct tm *tm);
};


}
}

#endif // INCLUDED_ml_core_CStrFTime_h

