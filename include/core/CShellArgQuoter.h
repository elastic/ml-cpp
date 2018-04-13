/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
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
