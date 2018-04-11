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
#include <core/CStateCompressor.h>

#include <core/CCompressOStream.h>
#include <core/CLogger.h>

#include <boost/make_shared.hpp>
#include <boost/ref.hpp>

namespace ml {
namespace core {

const std::string CStateCompressor::COMPRESSED_ATTRIBUTE("compressed");
const std::string CStateCompressor::END_OF_STREAM_ATTRIBUTE("eos");

CStateCompressor::CStateCompressor(CDataAdder& compressedAdder)
    : m_FilterSink(compressedAdder),
      m_OutStream(boost::make_shared<CCompressOStream>(boost::ref(m_FilterSink))) {
    LOG_TRACE(<< "New compressor");
}

CDataAdder::TOStreamP
CStateCompressor::addStreamed(const std::string& index, const std::string& baseId) {
    LOG_TRACE(<< "StateCompressor asking for index " << index);

    m_FilterSink.index(index, baseId);
    return m_OutStream;
}

bool CStateCompressor::streamComplete(CDataAdder::TOStreamP& /*strm*/, bool /*force*/) {
    LOG_TRACE(<< "Stream Complete");
    m_OutStream->close();
    return m_FilterSink.allWritesSuccessful();
}

size_t CStateCompressor::numCompressedDocs() const {
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
            LOG_TRACE(<< "Add streamed: " << m_Index << ", " << currentDocId);

            m_OStream = m_Adder.addStreamed(m_Index, currentDocId);
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

void CStateCompressor::CChunkFilter::index(const std::string& index,
                                           const std::string& baseId) {
    m_Index = index;
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

size_t CStateCompressor::CChunkFilter::numCompressedDocs() const {
    return m_CurrentDocNum - 1;
}

} // core
} // ml
