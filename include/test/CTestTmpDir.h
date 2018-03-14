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
        static std::string tmpDir(void);
};


}
}

#endif // INCLUDED_ml_test_CTestTmpDir_h

