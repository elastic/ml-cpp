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
#include <core/CStateCompressor.h>

#include <core/CCompressOStream.h>
#include <core/CLogger.h>

namespace ml {
namespace core {

const std::string CStateCompressor::COMPRESSED_ATTRIBUTE("compressed");
const std::string CStateCompressor::END_OF_STREAM_ATTRIBUTE("eos");

CStateCompressor::CStateCompressor(CDataAdder& compressedAdder)
    : m_FilterSink(compressedAdder),
      m_OutStream(std::make_shared<CCompressOStream>(std::ref(m_FilterSink))) {
    LOG_TRACE(<< "New compressor");
}

CDataAdder::TOStreamP CStateCompressor::addStreamed(const std::string& baseId) {
    m_FilterSink.baseId(baseId);
    return m_OutStream;
}

bool CStateCompressor::streamComplete(CDataAdder::TOStreamP& /*strm*/, bool /*force*/) {
    LOG_TRACE(<< "Stream Complete");
    m_OutStream->close();
    return m_FilterSink.allWritesSuccessful();
}

std::size_t CStateCompressor::numCompressedDocs() const {
    return m_FilterSink.numCompressedDocs();
}

CStateCompressor::CChunkFilter::CChunkFilter(CDataAdder& adder)
    : m_Adder(adder), m_CurrentDocNum(1), m_BytesDone(0),
      m_MaxDocSize(adder.maxDocumentSize()), m_WritesSuccessful(true) {
}

std::streamsize CStateCompressor::CChunkFilter::write(const char* s, std::streamsize n) {
    // Write up to n characters from the buffer
    // s to the output sequence, returning the
    // number of characters written
    std::streamsize written = 0;
    while (n > 0) {
        if (!m_OStream) {
            const std::string& currentDocId = m_Adder.makeCurrentDocId(m_BaseId, m_CurrentDocNum);
            LOG_TRACE(<< "Add streamed: " << currentDocId);

            m_OStream = m_Adder.addStreamed(currentDocId);
            if (!m_OStream) {
                LOG_ERROR(<< "Failed to connect to store");
                return 0;
            }
            if (m_OStream->bad()) {
                LOG_ERROR(<< "Error connecting to store");
                return 0;
            }

            std::string header(1, '{');

            header += '\"';
            header += COMPRESSED_ATTRIBUTE;
            header += "\" : [ ";

            LOG_TRACE(<< "Write: " << header);
            m_OStream->write(header.c_str(), header.size());
            m_BytesDone += header.size();
        } else {
            LOG_TRACE(<< "Write: ,");
            m_OStream->write(",", 1);
            m_BytesDone += 1;
        }
        this->writeInternal(s, written, n);
        if (m_BytesDone >= (m_MaxDocSize - 1)) {
            LOG_TRACE(<< "Terminated stream " << m_CurrentDocNum);
            this->closeStream(false);
            m_OStream.reset();
            m_BytesDone = 0;
        }
    }
    LOG_TRACE(<< "Returning " << written);
    return written;
}

void CStateCompressor::CChunkFilter::close() {
    this->closeStream(true);
}

void CStateCompressor::CChunkFilter::closeStream(bool isFinal) {
    if (m_OStream) {
        std::string footer(1, ']');
        if (isFinal) {
            footer += ",\"";
            footer += END_OF_STREAM_ATTRIBUTE;
            footer += "\":true";
        }
        footer += '}';
        LOG_TRACE(<< "Write: " << footer);
        m_OStream->write(footer.c_str(), footer.size());

        // always evaluate streamComplete(...)
        m_WritesSuccessful = m_Adder.streamComplete(m_OStream, isFinal) && m_WritesSuccessful;
        ++m_CurrentDocNum;
    }
}

void CStateCompressor::CChunkFilter::baseId(const std::string& baseId) {
    m_BaseId = baseId;
}

void CStateCompressor::CChunkFilter::writeInternal(const char* s,
                                                   std::streamsize& written,
                                                   std::streamsize& n) {
    std::size_t bytesToWrite = std::min(std::size_t(n), m_MaxDocSize - m_BytesDone);
    LOG_TRACE(<< "Writing string: " << std::string(&s[written], bytesToWrite));
    m_OStream->write("\"", 1);
    m_OStream->write(&s[written], bytesToWrite);
    m_OStream->write("\"", 1);
    written += bytesToWrite;
    n -= bytesToWrite;
    m_BytesDone += bytesToWrite + 2;
}

bool CStateCompressor::CChunkFilter::allWritesSuccessful() {
    return m_WritesSuccessful;
}

std::size_t CStateCompressor::CChunkFilter::numCompressedDocs() const {
    return m_CurrentDocNum - 1;
}

} // core
} // ml
