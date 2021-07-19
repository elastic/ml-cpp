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
#ifndef INCLUDED_ml_core_CProgName_h
#define INCLUDED_ml_core_CProgName_h

#include <core/CNonInstantiatable.h>
#include <core/ImportExport.h>

#include <string>

namespace ml {
namespace core {

//! \brief
//! Get the name/location of the current program.
//!
//! DESCRIPTION:\n
//! Get the simple name of the current program and the absolute
//! path of the directory that it was loaded from.
//!
//! Being able to find out where the program executable is located
//! means other files can be picked up relative to this location.
//!
//! IMPLEMENTATION DECISIONS:\n
//! For the name, just the program name is returned, with no path
//! or extension.
//!
class CORE_EXPORT CProgName : private CNonInstantiatable {
public:
    //! Get the name of the current program.  On error, an empty string is
    //! returned.
    static std::string progName();

    //! Get the directory where the current program's executable image is
    //! located.  On error, an empty string is returned.
    static std::string progDir();
};
}
}

#endif // INCLUDED_ml_core_CProgName_h
