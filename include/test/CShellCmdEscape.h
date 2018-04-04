/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_test_CShellCmdEscape_h
#define INCLUDED_ml_test_CShellCmdEscape_h

#include <core/CNonInstantiatable.h>

#include <test/ImportExport.h>

#include <string>

namespace ml {
namespace test {

//! \brief
//! Escape special characters in a shell command
//!
//! DESCRIPTION:\n
//! Escape special characters in a shell command
//!
//! IMPLEMENTATION DECISIONS:\n
//! On Unix characters are escaped for sh/ksh/bash.  On Windows,
//! for cmd.exe.
//!
class TEST_EXPORT CShellCmdEscape : private core::CNonInstantiatable {
public:
    //! Modifies the command such that special characters are appropriately
    //! escaped
    static void escapeCmd(std::string& cmd);
};
}
}

#endif // INCLUDED_ml_test_CShellCmdEscape_h
