/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */
#include <api/CIoManager.h>

#include <core/CBlockingCallCancellerThread.h>
#include <core/CLogger.h>

#include <atomic>
#include <fstream>
#include <ios>
#include <iostream>

namespace ml {
namespace api {

namespace {

bool setUpIStream(const std::string& fileName,
                  bool isFileNamedPipe,
                  core::CBlockingCallCancellerThread& cancellerThread,
                  core::CNamedPipeFactory::TIStreamP& stream,
                  const std::string& pipeType) {
    if (fileName.empty()) {
        stream.reset();
        return true;
    }
    if (isFileNamedPipe) {
        cancellerThread.reset();
        cancellerThread.start();
        stream = core::CNamedPipeFactory::openPipeStreamRead(
            fileName, cancellerThread.hasCancelledBlockingCall());
        cancellerThread.stop();
        if (stream == nullptr) {
            LOG_ERROR(<< "Failed to open " << pipeType << " pipe for reading: " << fileName);
            return false;
        }
        if (stream->bad()) {
            LOG_ERROR(<< pipeType << " pipe stream is bad after opening: " << fileName);
            return false;
        }
        return true;
    }
    std::ifstream* fileStream{nullptr};
    stream.reset(fileStream = new std::ifstream(fileName, std::ios::binary | std::ios::in));
    if (!fileStream->is_open()) {
        LOG_ERROR(<< "Failed to open " << pipeType << " file for reading: " << fileName);
        return false;
    }
    return true;
}

bool setUpOStream(const std::string& fileName,
                  bool isFileNamedPipe,
                  core::CBlockingCallCancellerThread& cancellerThread,
                  core::CNamedPipeFactory::TOStreamP& stream,
                  const std::string& pipeType) {
    if (fileName.empty()) {
        stream.reset();
        return true;
    }
    if (isFileNamedPipe) {
        cancellerThread.reset();
        cancellerThread.start();
        stream = core::CNamedPipeFactory::openPipeStreamWrite(
            fileName, cancellerThread.hasCancelledBlockingCall());
        cancellerThread.stop();
        if (stream == nullptr) {
            LOG_ERROR(<< "Failed to open " << pipeType << " pipe for writing: " << fileName);
            return false;
        }
        if (stream->bad()) {
            LOG_ERROR(<< pipeType << " pipe stream is bad after opening: " << fileName);
            return false;
        }
        return true;
    }
    std::ofstream* fileStream{nullptr};
    stream.reset(fileStream = new std::ofstream(fileName, std::ios::binary | std::ios::out));
    if (!fileStream->is_open()) {
        LOG_ERROR(<< "Failed to open " << pipeType << " file for writing: " << fileName);
        return false;
    }
    return true;
}
}

CIoManager::CIoManager(core::CBlockingCallCancellerThread& cancellerThread,
                       const std::string& inputFileName,
                       bool isInputFileNamedPipe,
                       const std::string& outputFileName,
                       bool isOutputFileNamedPipe,
                       const std::string& restoreFileName,
                       bool isRestoreFileNamedPipe,
                       const std::string& persistFileName,
                       bool isPersistFileNamedPipe)
    : m_CancellerThread{cancellerThread}, m_IoInitialised{false}, m_InputFileName{inputFileName},
      m_IsInputFileNamedPipe{isInputFileNamedPipe && !inputFileName.empty()},
      m_OutputFileName{outputFileName}, m_IsOutputFileNamedPipe{isOutputFileNamedPipe &&
                                                                !outputFileName.empty()},
      m_RestoreFileName{restoreFileName},
      m_IsRestoreFileNamedPipe{isRestoreFileNamedPipe && !restoreFileName.empty()},
      m_PersistFileName{persistFileName}, m_IsPersistFileNamedPipe{
                                              isPersistFileNamedPipe &&
                                              !persistFileName.empty()} {
    // On some platforms input/output can be considerably faster if C and C++ IO
    // functionality is NOT synchronised.
    bool wasSynchronised{std::ios::sync_with_stdio(false)};
    if (wasSynchronised) {
        LOG_TRACE(<< "C++ streams no longer synchronised with C stdio");
    }

    // Untie the standard streams so that if std::cin and std::cout are used as
    // the primary IO mechanism they don't ruin each others buffering.
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    std::cerr.tie(nullptr);
}

CIoManager::~CIoManager() {
}

bool CIoManager::initIo() {
    LOG_DEBUG(<< "Initializing IO streams...");
    LOG_DEBUG(<< "  Input: " << (m_InputFileName.empty() ? "<none>" : m_InputFileName) 
              << (m_IsInputFileNamedPipe ? " (named pipe)" : " (file)"));
    LOG_DEBUG(<< "  Output: " << (m_OutputFileName.empty() ? "<none>" : m_OutputFileName)
              << (m_IsOutputFileNamedPipe ? " (named pipe)" : " (file)"));
    LOG_DEBUG(<< "  Restore: " << (m_RestoreFileName.empty() ? "<none>" : m_RestoreFileName)
              << (m_IsRestoreFileNamedPipe ? " (named pipe)" : " (file)"));
    LOG_DEBUG(<< "  Persist: " << (m_PersistFileName.empty() ? "<none>" : m_PersistFileName)
              << (m_IsPersistFileNamedPipe ? " (named pipe)" : " (file)"));
    
    if (!setUpIStream(m_InputFileName, m_IsInputFileNamedPipe,
                      m_CancellerThread, m_InputStream, "input")) {
        LOG_ERROR(<< "Failed to set up input stream");
        m_IoInitialised = false;
        return false;
    }
    LOG_DEBUG(<< "Input stream set up successfully");
    
    if (!setUpOStream(m_OutputFileName, m_IsOutputFileNamedPipe,
                      m_CancellerThread, m_OutputStream, "output")) {
        LOG_ERROR(<< "Failed to set up output stream");
        m_IoInitialised = false;
        return false;
    }
    LOG_DEBUG(<< "Output stream set up successfully");
    
    if (!setUpIStream(m_RestoreFileName, m_IsRestoreFileNamedPipe,
                      m_CancellerThread, m_RestoreStream, "restore")) {
        LOG_ERROR(<< "Failed to set up restore stream");
        m_IoInitialised = false;
        return false;
    }
    LOG_DEBUG(<< "Restore stream set up successfully");
    
    if (!setUpOStream(m_PersistFileName, m_IsPersistFileNamedPipe,
                      m_CancellerThread, m_PersistStream, "persist")) {
        LOG_ERROR(<< "Failed to set up persist stream");
        m_IoInitialised = false;
        return false;
    }
    LOG_DEBUG(<< "Persist stream set up successfully");
    
    m_IoInitialised = true;
    LOG_DEBUG(<< "All IO streams initialized successfully");
    return true;
}

std::istream& CIoManager::inputStream() {
    if (m_InputStream != nullptr) {
        return *m_InputStream;
    }

    if (!m_IoInitialised) {
        LOG_ERROR(<< "Accessing input stream before IO is initialised");
    }

    return std::cin;
}

std::ostream& CIoManager::outputStream() {
    if (m_OutputStream != nullptr) {
        return *m_OutputStream;
    }

    if (!m_IoInitialised) {
        LOG_ERROR(<< "Accessing output stream before IO is initialised");
    }

    return std::cout;
}

core::CNamedPipeFactory::TIStreamP CIoManager::restoreStream() {
    if (!m_IoInitialised) {
        LOG_ERROR(<< "Accessing restore stream before IO is initialised");
    }

    return m_RestoreStream;
}

core::CNamedPipeFactory::TOStreamP CIoManager::persistStream() {
    if (!m_IoInitialised) {
        LOG_ERROR(<< "Accessing persist stream before IO is initialised");
    }

    return m_PersistStream;
}
}
}
