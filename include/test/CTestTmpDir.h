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
#ifndef INCLUDED_ml_test_CTestTmpDir_h
#define INCLUDED_ml_test_CTestTmpDir_h

#include <core/CNonInstantiatable.h>

#include <test/ImportExport.h>

#include <string>

namespace ml {
namespace test {

//! \brief
//! Return the name of the temporary directory for the system.
//!
//! DESCRIPTION:\n
//! Return the name of the temporary directory for the system.
//!
//! IMPLEMENTATION DECISIONS:\n
//! On Unix the temporary directory is /tmp.  On Windows it's a
//! sub-directory of the current user's home directory.
//!
class TEST_EXPORT CTestTmpDir : private core::CNonInstantiatable {
public:
    //! Returns /tmp on Unix or an expansion of %TEMP% on Windows
    static std::string tmpDir();
};
}
}

#endif // INCLUDED_ml_test_CTestTmpDir_h
