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
#ifndef INCLUDED_ml_ver_CBuildInfo_h
#define INCLUDED_ml_ver_CBuildInfo_h

#include <string>

namespace ml {
namespace ver {

//! \brief
//! Wrapper for version/build numbers.
//!
//! DESCRIPTION:\n
//! Wrapper for version/build numbers.
//!
//! Only use this class from within a program's own code, NEVER from within
//! a library.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The version library is a static library that is linked to each ML
//! program.  It must never be linked into a shared library.  This rule means
//! that it can show up situations where a programs from different builds
//! have been mixed up - each program will have its own distinct copy of the
//! version library embedded in it.
//!
class CBuildInfo {
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
    //! Disallow instantiation
    CBuildInfo() = delete;
    CBuildInfo(const CBuildInfo&) = delete;

private:
    static const std::string VERSION_NUMBER;
    static const std::string BUILD_NUMBER;
    static const std::string COPYRIGHT;
};
}
}

#endif // INCLUDED_ml_core_CBuildInfo_h
