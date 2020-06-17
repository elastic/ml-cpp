/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CNamedPipeFactory_h
#define INCLUDED_ml_core_CNamedPipeFactory_h

#include <core/CNonInstantiatable.h>
#include <core/ImportExport.h>
#include <core/WindowsSafe.h>

#include <atomic>
#include <iosfwd>
#include <memory>
#include <string>

#include <stdio.h> // fdopen() is not C++ so need the C header

namespace ml {
namespace core {

//! \brief
//! Class to create named pipes.
//!
//! DESCRIPTION:\n
//! Creates and opens a named pipe for reading or writing, and returns
//! either a C++ stream or C FILE connected to that named pipe.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Although anonymous pipes are portable between Windows
//! and Unix, named pipes are not.  This class encapsulates
//! the platform specific code.  Note that it's only the
//! server side of the named pipe that requires this class -
//! the client side can use standard file functions.
//!
//! Windows named pipes always have names of the form:
//!
//! \\.\pipe\mypipename
//!
//! Unix named pipes can go in any writable directory of the
//! file system.
//!
//! Returned streams/FILEs are for binary data; Windows CRLF
//! translation is not applied.
//!
//! This class only allows named pipes to be created for either
//! reading or writing, not both.  Some platforms support
//! bi-directional named pipes, but the level of complexity
//! involved in getting a portable interface would be too high.
//! For bi-directional communications, the best thing is simply
//! to open two named pipes, one for reading and another for
//! writing.  These should be handled in separate threads on
//! at least one side of the connection, to avoid a deadlock
//! due to buffers filling up.
//!
class CORE_EXPORT CNamedPipeFactory : private CNonInstantiatable {
public:
    using TIStreamP = std::shared_ptr<std::istream>;
    using TOStreamP = std::shared_ptr<std::ostream>;
    using TFileP = std::shared_ptr<FILE>;

public:
    //! Character that can safely be used to test whether named pipes are
    //! connected.  The Java side of the pipe will silently ignore it.
    //! (Obviously this is specific to Elastic.)
    static const char TEST_CHAR;

public:
    //! Initialise and open a named pipe for reading, returning a C++ stream
    //! that can be used to read from it.  Returns a NULL pointer on
    //! failure.
    static TIStreamP openPipeStreamRead(const std::string& fileName,
                                        const std::atomic_bool& isCancelled);

    //! Initialise and open a named pipe for writing, returning a C++ stream
    //! that can be used to write to it.  Returns a NULL pointer on failure.
    static TOStreamP openPipeStreamWrite(const std::string& fileName,
                                         const std::atomic_bool& isCancelled);

    //! Initialise and open a named pipe for writing, returning a C FILE
    //! that can be used to read from it.  Returns a NULL pointer on
    //! failure.
    static TFileP openPipeFileRead(const std::string& fileName,
                                   const std::atomic_bool& isCancelled);

    //! Initialise and open a named pipe for writing, returning a C FILE
    //! that can be used to write to it.  Returns a NULL pointer on failure.
    static TFileP openPipeFileWrite(const std::string& fileName,
                                    const std::atomic_bool& isCancelled);

    //! Does the supplied file name refer to a named pipe?
    static bool isNamedPipe(const std::string& fileName);

    //! Default path for named pipes.
    static std::string defaultPath();

    //! Log warnings that have been stored because they were detected very
    //! early in the program lifecycle.  Programs using named pipes should
    //! call this method once, after setting up logging.
    static void logDeferredWarnings();

private:
#ifdef Windows
    using TPipeHandle = HANDLE;
#else
    using TPipeHandle = int;
#endif

private:
    //! Initialise and open a named pipe for writing, returning a handle
    //! file descriptor that can be used to access it.  This is the core
    //! implementation of the higher level encapsulations that the public
    //! interface provides.
    static TPipeHandle initPipeHandle(const std::string& fileName,
                                      bool forWrite,
                                      const std::atomic_bool& isCancelled);
};
}
}

#endif // INCLUDED_ml_core_CNamedPipeFactory_h
