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
#include <api/CIoManager.h>

#include <core/CLogger.h>

#include <fstream>
#include <ios>
#include <iostream>

namespace ml {
namespace api {

namespace {

bool setUpIStream(const std::string& fileName, bool isFileNamedPipe, core::CNamedPipeFactory::TIStreamP& stream) {
    if (fileName.empty()) {
        stream.reset();
        return true;
    }
    if (isFileNamedPipe) {
        stream = core::CNamedPipeFactory::openPipeStreamRead(fileName);
        return stream != nullptr && !stream->bad();
    }
    std::ifstream* fileStream(nullptr);
    stream.reset(fileStream = new std::ifstream(fileName.c_str()));
    return fileStream->is_open();
}

bool setUpOStream(const std::string& fileName, bool isFileNamedPipe, core::CNamedPipeFactory::TOStreamP& stream) {
    if (fileName.empty()) {
        stream.reset();
        return true;
    }
    if (isFileNamedPipe) {
        stream = core::CNamedPipeFactory::openPipeStreamWrite(fileName);
        return stream != nullptr && !stream->bad();
    }
    std::ofstream* fileStream(nullptr);
    stream.reset(fileStream = new std::ofstream(fileName.c_str()));
    return fileStream->is_open();
}
}

CIoManager::CIoManager(const std::string& inputFileName,
                       bool isInputFileNamedPipe,
                       const std::string& outputFileName,
                       bool isOutputFileNamedPipe,
                       const std::string& restoreFileName,
                       bool isRestoreFileNamedPipe,
                       const std::string& persistFileName,
                       bool isPersistFileNamedPipe)
    : m_IoInitialised(false),
      m_InputFileName(inputFileName),
      m_IsInputFileNamedPipe(isInputFileNamedPipe && !inputFileName.empty()),
      m_OutputFileName(outputFileName),
      m_IsOutputFileNamedPipe(isOutputFileNamedPipe && !outputFileName.empty()),
      m_RestoreFileName(restoreFileName),
      m_IsRestoreFileNamedPipe(isRestoreFileNamedPipe && !restoreFileName.empty()),
      m_PersistFileName(persistFileName),
      m_IsPersistFileNamedPipe(isPersistFileNamedPipe && !persistFileName.empty()) {
    // On some platforms input/output can be considerably faster if C and C++ IO
    // functionality is NOT synchronised.
    bool wasSynchronised(std::ios::sync_with_stdio(false));
    if (wasSynchronised) {
        LOG_TRACE("C++ streams no longer synchronised with C stdio");
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
    m_IoInitialised = setUpIStream(m_InputFileName, m_IsInputFileNamedPipe, m_InputStream) &&
                      setUpOStream(m_OutputFileName, m_IsOutputFileNamedPipe, m_OutputStream) &&
                      setUpIStream(m_RestoreFileName, m_IsRestoreFileNamedPipe, m_RestoreStream) &&
                      setUpOStream(m_PersistFileName, m_IsPersistFileNamedPipe, m_PersistStream);
    return m_IoInitialised;
}

std::istream& CIoManager::inputStream() {
    if (m_InputStream != nullptr) {
        return *m_InputStream;
    }

    if (!m_IoInitialised) {
        LOG_ERROR("Accessing input stream before IO is initialised");
    }

    return std::cin;
}

std::ostream& CIoManager::outputStream() {
    if (m_OutputStream != nullptr) {
        return *m_OutputStream;
    }

    if (!m_IoInitialised) {
        LOG_ERROR("Accessing output stream before IO is initialised");
    }

    return std::cout;
}

core::CNamedPipeFactory::TIStreamP CIoManager::restoreStream() {
    if (!m_IoInitialised) {
        LOG_ERROR("Accessing restore stream before IO is initialised");
    }

    return m_RestoreStream;
}

core::CNamedPipeFactory::TOStreamP CIoManager::persistStream() {
    if (!m_IoInitialised) {
        LOG_ERROR("Accessing persist stream before IO is initialised");
    }

    return m_PersistStream;
}
}
}
