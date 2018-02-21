/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CCrashHandler_h
#define INCLUDED_ml_core_CCrashHandler_h

#include <core/CNonInstantiatable.h>
#include <core/ImportExport.h>

namespace ml
{
namespace core
{

//! \brief
//! Helper class to register a crashhandler and getting better traces.
//!
//! DESCRIPTION:\n
//! Last line of defense when the autodetect process crashes. Tries to get out
//! useful information on a SIGSEGV and logs them.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The basic implementation does nothing.  Platform-specific implementations
//! exist for platforms where we have decided to do something.
//!
//! Currently the only platform-specific implementation is for Linux. This might
//! work for Mac as well.
//!
//! On Linux the handler returns signal number, code, errno (meaning can be found
//! in siginfo.h),  address, library, base and normalized_address.
//! To turn that into a line number use:
//!
//! addr2line -e library normalized_address
//! (library can be the symbol file)
class CORE_EXPORT CCrashHandler : private CNonInstantiatable
{
    public:
        static void installCrashHandler(void);
};

}
}

#endif // INCLUDED_ml_core_CCrashHandler_h
