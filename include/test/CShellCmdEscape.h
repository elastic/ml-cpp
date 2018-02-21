/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */
#ifndef INCLUDED_ml_test_CShellCmdEscape_h
#define INCLUDED_ml_test_CShellCmdEscape_h

#include <core/CNonInstantiatable.h>

#include <test/ImportExport.h>

#include <string>


namespace ml
{
namespace test
{


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
class TEST_EXPORT CShellCmdEscape : private core::CNonInstantiatable
{
    public:
        //! Modifies the command such that special characters are appropriately
        //! escaped
        static void escapeCmd(std::string &cmd);
};


}
}

#endif // INCLUDED_ml_test_CShellCmdEscape_h

