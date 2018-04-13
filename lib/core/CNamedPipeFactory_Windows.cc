/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CNamedPipeFactory.h>

#include <core/CLogger.h>
#include <core/CWindowsError.h>

#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/stream.hpp>

#include <fcntl.h>
#include <io.h>


namespace
{

//! fclose() doesn't check for NULL pointers, so wrap it for use as a shared_ptr
//! deleter
void safeFClose(FILE *file)
{
    if (file != 0)
    {
        ::fclose(file);
    }
}

//! On Windows ALL named pipes are under this path
const std::string PIPE_PREFIX("\\\\.\\pipe\\");

}

namespace ml
{
namespace core
{


// Initialise static
const char CNamedPipeFactory::TEST_CHAR('\n');


CNamedPipeFactory::TIStreamP CNamedPipeFactory::openPipeStreamRead(const std::string &fileName)
{
    TPipeHandle handle = CNamedPipeFactory::initPipeHandle(fileName, false);
    if (handle == INVALID_HANDLE_VALUE)
    {
        return TIStreamP();
    }
    using TFileDescriptorSourceStream = boost::iostreams::stream<boost::iostreams::file_descriptor_source>;
    return TIStreamP(new TFileDescriptorSourceStream(
            boost::iostreams::file_descriptor_source(handle, boost::iostreams::close_handle)));
}

CNamedPipeFactory::TOStreamP CNamedPipeFactory::openPipeStreamWrite(const std::string &fileName)
{
    TPipeHandle handle = CNamedPipeFactory::initPipeHandle(fileName, true);
    if (handle == INVALID_HANDLE_VALUE)
    {
        return TOStreamP();
    }
    using TFileDescriptorSinkStream = boost::iostreams::stream<boost::iostreams::file_descriptor_sink>;
    return TOStreamP(new TFileDescriptorSinkStream(
            boost::iostreams::file_descriptor_sink(handle, boost::iostreams::close_handle)));
}

CNamedPipeFactory::TFileP CNamedPipeFactory::openPipeFileRead(const std::string &fileName)
{
    TPipeHandle handle = CNamedPipeFactory::initPipeHandle(fileName, false);
    if (handle == INVALID_HANDLE_VALUE)
    {
        return TFileP();
    }
    return TFileP(::fdopen(::_open_osfhandle(reinterpret_cast<intptr_t>(handle), _O_RDONLY),
                           "rb"),
                  safeFClose);
}

CNamedPipeFactory::TFileP CNamedPipeFactory::openPipeFileWrite(const std::string &fileName)
{
    TPipeHandle handle = CNamedPipeFactory::initPipeHandle(fileName, true);
    if (handle == INVALID_HANDLE_VALUE)
    {
        return TFileP();
    }
    return TFileP(::fdopen(::_open_osfhandle(reinterpret_cast<intptr_t>(handle), 0),
                           "wb"),
                  safeFClose);
}

bool CNamedPipeFactory::isNamedPipe(const std::string &fileName)
{
    return fileName.length() > PIPE_PREFIX.length() &&
           fileName.compare(0, PIPE_PREFIX.length(), PIPE_PREFIX) == 0;
}

std::string CNamedPipeFactory::defaultPath()
{
    return PIPE_PREFIX;
}

CNamedPipeFactory::TPipeHandle CNamedPipeFactory::initPipeHandle(const std::string &fileName, bool forWrite)
{
    // Size of named pipe buffer
    static const DWORD BUFFER_SIZE(4096);

    // If the name already exists, ensure it refers to a named pipe
    HANDLE handle(CreateNamedPipe(fileName.c_str(),
                                  // Input pipes are opened as duplex so we can
                                  // write a test byte to them to work around
                                  // the Java security manager problem
                                  forWrite ? PIPE_ACCESS_OUTBOUND : PIPE_ACCESS_DUPLEX,
                                  PIPE_TYPE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
                                  1,
                                  forWrite ? BUFFER_SIZE : 1,
                                  forWrite ? 1 : BUFFER_SIZE,
                                  NMPWAIT_USE_DEFAULT_WAIT,
                                  0));
    if (handle == INVALID_HANDLE_VALUE)
    {
        LOG_ERROR("Unable to create named pipe " << fileName <<
                  ": " << CWindowsError());
        return INVALID_HANDLE_VALUE;
    }

    // There is a problem with connecting named pipes on Windows to a JVM with a
    // security manager.  Each time a request is made to open a particular file
    // in Java, the security manager will check whether opening that file is
    // permitted.  On Windows, the first time this check is made for any given
    // file name, the file is opened.  For named pipes this unfortunately means
    // that the server side of the named pipe will see a very short-lived
    // connection followed by end-of-file.  When this happens we need to
    // disconnect the short-lived pipe to the security manager and reconnect to
    // the file that will back the Java FileInputStream or FileOutputStream.
    // This cannot be as simple as just always disconnecting and reconnecting
    // once, because this leads to a race condition: we cannot guarantee that
    // the security manager's spurious file open will occur while this code is
    // trying to connect.  Sometimes the security manager can try to open the
    // pipe before this code calls the ConnectNamedPipe() function, but then
    // the real file open on the Java side comes after the ConnectNamedPipe()
    // call has started.  In this case there is no short-lived connection.  To
    // cope with this we wait a short time after a connection and then try to
    // write to the pipe to check whether it's still open on the remote side.
    // This means that all Java code that reads from Ml named pipes must
    // tolerate a test character appearing at the beginning of the data it
    // receives.  We use a newline character, as the named pipes carry lineified
    // JSON and it's easy to make them tolerate blank lines.
    bool sufferedShortLivedConnection(false);
    DWORD attempt(0);
    do
    {
        ++attempt;
        // This call will block if there is no other connection to the named
        // pipe
        if (ConnectNamedPipe(handle, 0) == FALSE)
        {
            // ERROR_PIPE_CONNECTED means the pipe was already connected so
            // there was no need to connect it again - not a problem
            DWORD errCode(GetLastError());
            if (errCode != ERROR_PIPE_CONNECTED)
            {
                LOG_ERROR("Unable to connect named pipe " << fileName <<
                          ": " << CWindowsError(errCode));
                // Close the pipe (even though it was successfully opened) so
                // that the net effect of this failed call is nothing
                CloseHandle(handle);
                return INVALID_HANDLE_VALUE;
            }
        }

        // Allow time for the security manager problem to manifest itself
        Sleep(100 / attempt);

        // Check that the other end of the pipe has not disconnected (which
        // relies on the Java side of all connections tolerating an initial
        // blank line)
        DWORD bytesWritten(0);
        if (WriteFile(handle,
                      &TEST_CHAR,
                      sizeof(TEST_CHAR),
                      &bytesWritten,
                      0) == FALSE || bytesWritten == 0)
        {
            DisconnectNamedPipe(handle);
            sufferedShortLivedConnection = true;
        }
        else
        {
            sufferedShortLivedConnection = false;
        }
    }
    while (sufferedShortLivedConnection);

    return handle;
}


}
}

