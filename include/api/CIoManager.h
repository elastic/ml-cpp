/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CIoManager_h
#define INCLUDED_ml_api_CIoManager_h

#include <core/CNamedPipeFactory.h>
#include <core/CNonCopyable.h>

#include <api/ImportExport.h>

#include <iosfwd>
#include <string>

namespace ml {
namespace api {

//! \brief
//! Manages the various IO streams of an API command.
//!
//! DESCRIPTION:\n
//! Ml C++ commands that can be run from Java can take input
//! from and send output to:
//! 1) STDIN/STDOUT
//! 2) Standard files on disk
//! 3) Named pipes
//!
//! Option (3) will be used in production, but the others are useful
//! for backwards compatibility and testing.
//!
//! Additionally, state persistence and restoration can be via
//! named pipes.
//!
//! This class:
//! 1) Is the single point from which the appropriate C++ stream for
//!    a given function can be retrieved.
//! 2) Manages the lifecycle for streams other than STDIN/STDOUT.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Input/output streams are returned as references because they are
//! always required.  Persist/restore streams are returned as pointers
//! because some processes may not require both.
//!
class API_EXPORT CIoManager : private core::CNonCopyable {
public:
    //! Leave \p inputFileName/\p outputFileName empty to indicate
    //! STDIN/STDOUT.  Leave \p restoreFileName/\p persistFileName empty to
    //! indicate no state restore or persist.
    CIoManager(const std::string& inputFileName,
               bool isInputFileNamedPipe,
               const std::string& outputFileName,
               bool isOutputFileNamedPipe,
               const std::string& restoreFileName = std::string(),
               bool isRestoreFileNamedPipe = true,
               const std::string& persistFileName = std::string(),
               bool isPersistFileNamedPipe = true);

    //! This will close any streams and unlink named pipes.  All
    //! input/output/restore/persist operations must be complete at the time
    //! this object is destroyed.
    ~CIoManager();

    //! Set up the necessary streams given the constructor arguments.
    bool initIo();

    //! Get the stream to get input data from.
    std::istream& inputStream();

    //! Get the stream to write output to.
    std::ostream& outputStream();

    //! Get the stream to restore state from.  If NULL then don't restore state.
    core::CNamedPipeFactory::TIStreamP restoreStream();

    //! Get the stream to persist state to.  If NULL then don't persist state.
    core::CNamedPipeFactory::TOStreamP persistStream();

private:
    //! Have the streams been successfully initialised?
    bool m_IoInitialised;

    //! Name of file/pipe to get input from.  Empty implies STDIN.
    std::string m_InputFileName;

    //! Is the input file a named pipe?
    bool m_IsInputFileNamedPipe;

    //! If this object owns the input stream then a pointer to it.  If
    //! std::cin is being used then this will be NULL.
    core::CNamedPipeFactory::TIStreamP m_InputStream;

    //! Name of file/pipe to write output to.  Empty implies STDOUT.
    std::string m_OutputFileName;

    //! Is the input file a named pipe?
    bool m_IsOutputFileNamedPipe;

    //! If this object owns the output stream then a pointer to it.  If
    //! std::cout is being used then this will be NULL.
    core::CNamedPipeFactory::TOStreamP m_OutputStream;

    //! Name of file/pipe to restore state from.  Empty implies don't
    //! restore state.
    std::string m_RestoreFileName;

    //! Is the restore file a named pipe?
    bool m_IsRestoreFileNamedPipe;

    //! If this object owns the restore stream then a pointer to it.  A
    //! NULL pointer implies state is not being restored.
    core::CNamedPipeFactory::TIStreamP m_RestoreStream;

    //! Name of file/pipe to persist state to.  Empty implies don't persist
    //! state.
    std::string m_PersistFileName;

    //! Is the persist file a named pipe?
    bool m_IsPersistFileNamedPipe;

    //! If this object owns the persist stream then a pointer to it.  A
    //! NULL pointer implies state is not being persisted.
    core::CNamedPipeFactory::TOStreamP m_PersistStream;
};
}
}

#endif // INCLUDED_ml_api_CIoManager_h
