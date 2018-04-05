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
#ifndef INCLUDED_ml_core_CShellArgQuoter_h
#define INCLUDED_ml_core_CShellArgQuoter_h

#include <core/CNonInstantiatable.h>
#include <core/ImportExport.h>

#include <string>

namespace ml {
namespace core {

//! \brief
//! Quote a shell argument.
//!
//! DESCRIPTION:\n
//! Quote a shell argument, escaping any special characters within
//! the argument as necessary.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Arguments are escaped such that variable expansion does NOT
//! take place within them.  For example, if the argument is
//! $ES_HOME, then it will be escaped such that the program
//! the argument is being passed to receives literally $ES_HOME
//! and not the directory path that the environment variable
//! expands to.
//!
class CORE_EXPORT CShellArgQuoter : private CNonInstantiatable {
public:
    //! Returns /tmp on Unix or an expansion of %TEMP% on Windows
    static std::string quote(const std::string& arg);
};
}
}

#endif // INCLUDED_ml_core_CShellArgQuoter_h
