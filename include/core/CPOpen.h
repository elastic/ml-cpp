/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CPOpen_h
#define INCLUDED_ml_core_CPOpen_h

#include <core/CNonInstantiatable.h>
#include <core/ImportExport.h>

#include <stdio.h>


namespace ml
{
namespace core
{


//! \brief
//! Portable wrapper for the popen()/pclose() functions.
//!
//! DESCRIPTION:\n
//! Portable wrapper for the popen()/pclose() functions.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This has been broken into a class of its own because Windows has a
//! _popen() and _pclose() functions rather than Unix's popen() and pclose().
//!
class CORE_EXPORT CPOpen : private CNonInstantiatable
{
    public:
        static FILE *pOpen(const char *command,
                           const char *mode);

        static int pClose(FILE *stream);
};


}
}

#endif // INCLUDED_ml_core_CPOpen_h

