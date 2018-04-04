/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CUname_h
#define INCLUDED_ml_core_CUname_h

#include <core/CNonInstantiatable.h>
#include <core/ImportExport.h>

#include <string>

namespace ml {
namespace core {

//! \brief
//! Portable wrapper for the uname() function.
//!
//! DESCRIPTION:\n
//! Portable wrapper for the uname() function.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This has been broken into a class of its own because Windows does
//! not implement Unix's uname() function.
//!
//! Each member of the utsname struct should be implemented as a
//! separate method, as usually only one is required.
//!
class CORE_EXPORT CUname : private CNonInstantiatable {
public:
    //! uname -s
    static std::string sysName();
    //! uname -n
    static std::string nodeName();
    //! uname -r
    static std::string release();
    //! uname -v
    static std::string version();
    //! uname -m
    static std::string machine();
    //! uname -a (or possibly a cut down version on some platforms)
    static std::string all();

    //! Return the platform name in the format <platform>-<arch>
    //! e.g. linux-x86_64
    static std::string mlPlatform();

    //! On Unix this is equivalent to uname -r; on Windows it's the
    //! underlying Windows NT version.
    static std::string mlOsVer();
};
}
}

#endif // INCLUDED_ml_core_CUname_h
