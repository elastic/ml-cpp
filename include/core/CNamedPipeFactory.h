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
#ifndef INCLUDED_ml_core_CNamedPipeFactory_h
#define INCLUDED_ml_core_CNamedPipeFactory_h

#include <core/CNonInstantiatable.h>
#include <core/ImportExport.h>
#include <core/WindowsSafe.h>

#include <boost/shared_ptr.hpp>

#include <iosfwd>
#include <string>

#include <stdio.h>


namespace ml
{
namespace core
{

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
class CORE_EXPORT CNamedPipeFactory : private CNonInstantiatable
{
    public:
        using TIStreamP = boost::shared_ptr<std::istream>;
        using TOStreamP = boost::shared_ptr<std::ostream>;
        using TFileP = boost::shared_ptr<FILE>;

    public:
        //! Character that can safely be used to test whether named pipes are
        //! connected.  The Java side of the pipe will silently ignore it.
        //! (Obviously this is specific to Elastic.)
        static const char TEST_CHAR;

    public:
        //! Initialise and open a named pipe for reading, returning a C++ stream
        //! that can be used to read from it.  Returns a NULL pointer on
        //! failure.
        static TIStreamP openPipeStreamRead(const std::string &fileName);

        //! Initialise and open a named pipe for writing, returning a C++ stream
        //! that can be used to write to it.  Returns a NULL pointer on failure.
        static TOStreamP openPipeStreamWrite(const std::string &fileName);

        //! Initialise and open a named pipe for writing, returning a C FILE
        //! that can be used to read from it.  Returns a NULL pointer on
        //! failure.
        static TFileP openPipeFileRead(const std::string &fileName);

        //! Initialise and open a named pipe for writing, returning a C FILE
        //! that can be used to write to it.  Returns a NULL pointer on failure.
        static TFileP openPipeFileWrite(const std::string &fileName);

        //! Does the supplied file name refer to a named pipe?
        static bool isNamedPipe(const std::string &fileName);

        //! Default path for named pipes.
        static std::string defaultPath();

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
        static TPipeHandle initPipeHandle(const std::string &fileName, bool forWrite);
};


}
}

#endif // INCLUDED_ml_core_CNamedPipeFactory_h

