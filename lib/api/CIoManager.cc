/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
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
                  core::CNamedPipeFactory::TIStreamP& stream) {
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
        return stream != nullptr && !stream->bad();
    }
    std::ifstream* fileStream{nullptr};
    stream.reset(fileStream = new std::ifstream(fileName, std::ios::binary | std::ios::in));
    return fileStream->is_open();
}

bool setUpOStream(const std::string& fileName,
                  bool isFileNamedPipe,
                  core::CBlockingCallCancellerThread& cancellerThread,
                  core::CNamedPipeFactory::TOStreamP& stream) {
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
        return stream != nullptr && !stream->bad();
    }
    std::ofstream* fileStream{nullptr};
    stream.reset(fileStream = new std::ofstream(fileName, std::ios::binary | std::ios::out));
    return fileStream->is_open();
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
    m_IoInitialised = setUpIStream(m_InputFileName, m_IsInputFileNamedPipe,
                                   m_CancellerThread, m_InputStream) &&
                      setUpOStream(m_OutputFileName, m_IsOutputFileNamedPipe,
                                   m_CancellerThread, m_OutputStream) &&
                      setUpIStream(m_RestoreFileName, m_IsRestoreFileNamedPipe,
                                   m_CancellerThread, m_RestoreStream) &&
                      setUpOStream(m_PersistFileName, m_IsPersistFileNamedPipe,
                                   m_CancellerThread, m_PersistStream);
    return m_IoInitialised;
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
