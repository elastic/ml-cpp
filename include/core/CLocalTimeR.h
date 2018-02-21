/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CLocalTimeR_h
#define INCLUDED_ml_core_CLocalTimeR_h

#include <core/CNonInstantiatable.h>
#include <core/ImportExport.h>

#include <time.h>


namespace ml
{
namespace core
{


//! \brief
//! Portable wrapper for the localtime_r() function.
//!
//! DESCRIPTION:\n
//! Portable wrapper for the localtime_r() function.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This has been broken into a class of its own because Windows has a
//! localtime_s() function with slightly different semantics to Unix's
//! localtime_r().
//!
class CORE_EXPORT CLocalTimeR : private CNonInstantiatable
{
    public:
        static struct tm *localTimeR(const time_t *clock,
                                     struct tm *result);
};


}
}

#endif // INCLUDED_ml_core_CLocalTimeR_h

