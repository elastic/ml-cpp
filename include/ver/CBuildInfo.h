/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_ver_CBuildInfo_h
#define INCLUDED_ml_ver_CBuildInfo_h

#include <core/CNonInstantiatable.h>

#include <string>

namespace ml {
namespace ver {

//! \brief
//! Wrapper for version/build numbers
//!
//! DESCRIPTION:\n
//! Wrapper for version/build numbers
//!
//! Only use this class from within a program's own code, NEVER from within
//! a library.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The implementation file for this class is generated on the fly at build time
//! by the Makefile from a template.  There are two possible templates:
//! - One for developers' individual builds
//! - One for full builds that could be given to a customer
//! Placeholders within these templates are substituted with the exact version
//! number, build number, build year and developer's name.
//!
//! The version library is a static library that is linked to each Ml
//! program.  It must never be linked into a shared library.  This rule means
//! that it can show up situations where a programs from different builds
//! have been mixed up - each program will have its own distinct copy of the
//! version library embedded in it.
//!
class CBuildInfo : private core::CNonInstantiatable {
public:
    //! Get the version number to be printed out
    static const std::string& versionNumber();

    //! Get the build number to be printed out
    static const std::string& buildNumber();

    //! Get the copyright message to be printed out
    static const std::string& copyright();

    //! Get the full information to be printed out (this includes the name
    //! of the program, plus the version number, build number and copyright)
    static std::string fullInfo();

private:
    static const std::string VERSION_NUMBER;
    static const std::string BUILD_NUMBER;
    static const std::string COPYRIGHT;
};
}
}

#endif // INCLUDED_ml_core_CBuildInfo_h
