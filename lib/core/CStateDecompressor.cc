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
#include <core/CStateDecompressor.h>

#include <core/CBase64Filter.h>
#include <core/CStateCompressor.h>

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>

#include <boost/iostreams/filter/gzip.hpp>

#include <cstring>
#include <string>

namespace ml {
namespace core {

const std::string CStateDecompressor::EMPTY_DATA("H4sIAAAAAAAA/4uOBQApu0wNAgAAAA==");

CStateDecompressor::CStateDecompressor(CDataSearcher& compressedSearcher)
    : m_Searcher(compressedSearcher), m_FilterSource(compressedSearcher) {
    m_InFilter.reset(new TFilteredInput);
    m_InFilter->push(boost::iostreams::gzip_decompressor());
    m_InFilter->push(CBase64Decoder());
    m_InFilter->push(boost::ref(m_FilterSource));
}

CDataSearcher::TIStreamP CStateDecompressor::search(size_t /*currentDocNum*/, size_t /*limit*/) {
    return m_InFilter;
}

void CStateDecompressor::setStateRestoreSearch(const std::string& index) {
    m_Searcher.setStateRestoreSearch(index);
}

void CStateDecompressor::setStateRestoreSearch(const std::string& index, const std::string& id) {
    m_Searcher.setStateRestoreSearch(index, id);
}

CStateDecompressor::CDechunkFilter::CDechunkFilter(CDataSearcher& searcher)
    : m_Initialised(false),
      m_SentData(false),
      m_Searcher(searcher),
      m_CurrentDocNum(1),
      m_EndOfStream(false),
      m_BufferOffset(0),
      m_NestedLevel(1) {
}

std::streamsize CStateDecompressor::CDechunkFilter::read(char* s, std::streamsize n) {
    if (m_EndOfStream) {
        LOG_TRACE("EOS -1");
        return -1;
    }
    // return number of bytes read, -1 for EOF
    std::streamsize bytesDone = 0;
    while (bytesDone < n) {
        if (!m_IStream) {
            // Get a new input stream
            LOG_TRACE("Getting new stream, for document number " << m_CurrentDocNum);

            m_IStream = m_Searcher.search(m_CurrentDocNum, 1);
            if (!m_IStream) {
                LOG_ERROR("Unable to connect to data store");
                return this->endOfStream(s, n, bytesDone);
            }
            if (m_IStream->bad()) {
                LOG_ERROR("Error connecting to data store");
                return this->endOfStream(s, n, bytesDone);
            }

            if (m_IStream->fail()) {
                m_EndOfStream = true;
                // This is not fatal - we just didn't find the given document number
                // Presume that we have finished
                LOG_TRACE("No more documents to find");
                return this->endOfStream(s, n, bytesDone);
            }

            m_InputStreamWrapper.reset(new rapidjson::IStreamWrapper(*m_IStream));
            m_Reader.reset(new rapidjson::Reader);

            if (!this->readHeader()) {
                return this->endOfStream(s, n, bytesDone);
            }
        }

        this->handleRead(s, n, bytesDone);
        if (m_EndOfStream) {
            return this->endOfStream(s, n, bytesDone);
        }

        if ((m_IStream) && (m_IStream->eof())) {
            LOG_TRACE("Stream EOF");
            m_IStream.reset();
            ++m_CurrentDocNum;
        }
    }
    LOG_TRACE("Returning " << bytesDone << ": " << std::string(s, bytesDone));
    return bytesDone;
}

bool CStateDecompressor::CDechunkFilter::parseNext() {
    if (m_Reader->HasParseError()) {
        const char* error(rapidjson::GetParseError_En(m_Reader->GetParseErrorCode()));
        LOG_ERROR("Error parsing JSON at offset " << m_Reader->GetErrorOffset() << ": " << ((error != nullptr) ? error : "No message"));
        return false;
    }

    const int parseFlags = rapidjson::kParseDefaultFlags;

    return m_Reader->IterativeParseNext<parseFlags>(*m_InputStreamWrapper, m_Handler);
}

bool CStateDecompressor::CDechunkFilter::readHeader() {
    m_Reader->IterativeParseInit();

    if (this->parseNext() == false) {
        LOG_ERROR("Failed to find valid JSON");
        m_Initialised = false;
        m_IStream.reset();
        ++m_CurrentDocNum;
        return false;
    }

    while (this->parseNext()) {
        if (m_Handler.s_Type == SRapidJsonHandler::E_TokenKey &&
            CStateCompressor::COMPRESSED_ATTRIBUTE.compare(
                0, CStateCompressor::COMPRESSED_ATTRIBUTE.length(), m_Handler.s_CompressedChunk, m_Handler.s_CompressedChunkLength) == 0) {
            if (this->parseNext() && m_Handler.s_Type == SRapidJsonHandler::E_TokenArrayStart) {
                m_Initialised = true;
                m_BufferOffset = 0;
                return true;
            }
        } else if (m_Handler.s_Type == SRapidJsonHandler::E_TokenObjectStart) {
            ++m_NestedLevel;
        }
    }
    // If we are here, we have got an empty document from downstream,
    // so the stream is finished
    LOG_TRACE("Failed to find 'compressed' data array!");
    m_Initialised = false;
    m_IStream.reset();
    ++m_CurrentDocNum;
    return false;
}

void CStateDecompressor::CDechunkFilter::handleRead(char* s, std::streamsize n, std::streamsize& bytesDone) {
    // Extract data from the JSON array "compressed"
    if (!m_Initialised) {
        return;
    }

    // Copy any outstanding data
    if (m_BufferOffset > 0) {
        std::streamsize toCopy = std::min((n - bytesDone), (m_Handler.s_CompressedChunkLength - m_BufferOffset));
        std::memcpy(s + bytesDone, m_Handler.s_CompressedChunk + m_BufferOffset, toCopy);
        bytesDone += toCopy;
        m_BufferOffset += toCopy;
    }

    // Expect to have data in an array
    while (bytesDone < n && this->parseNext()) {
        m_BufferOffset = 0;
        if (m_Handler.s_Type == SRapidJsonHandler::E_TokenArrayEnd) {
            LOG_TRACE("Come to end of array");
            if (this->parseNext() && m_Handler.s_Type == SRapidJsonHandler::E_TokenKey &&
                CStateCompressor::END_OF_STREAM_ATTRIBUTE.compare(0,
                                                                  CStateCompressor::END_OF_STREAM_ATTRIBUTE.length(),
                                                                  m_Handler.s_CompressedChunk,
                                                                  m_Handler.s_CompressedChunkLength) == 0) {
                LOG_DEBUG("Explicit end-of-stream marker found in document with index " << m_CurrentDocNum);

                // Read the value of the CStateCompressor::END_OF_STREAM_ATTRIBUTE field and the closing brace
                if (this->parseNext() && m_Handler.s_Type != SRapidJsonHandler::E_TokenBool) {
                    LOG_ERROR("Expecting bool value to follow  " << CStateCompressor::END_OF_STREAM_ATTRIBUTE << ", got "
                                                                 << m_Handler.s_Type);
                }

                while (m_NestedLevel > 0) {
                    if (this->parseNext() && m_Handler.s_Type != SRapidJsonHandler::E_TokenObjectEnd) {
                        LOG_ERROR("Expecting end object to follow " << CStateCompressor::END_OF_STREAM_ATTRIBUTE << ", got "
                                                                    << m_Handler.s_Type);
                    }

                    --m_NestedLevel;
                }

                // Don't search for any more documents after seeing this - any
                // that exist will be stale - see bug 1248 in Bugzilla
                m_EndOfStream = true;
            }
            m_IStream.reset();
            ++m_CurrentDocNum;
            break;
        }
        m_SentData = true;
        if (m_Handler.s_CompressedChunkLength <= (n - bytesDone)) {
            std::memcpy(s + bytesDone, m_Handler.s_CompressedChunk, m_Handler.s_CompressedChunkLength);
            bytesDone += m_Handler.s_CompressedChunkLength;
        } else {
            std::streamsize toCopy = n - bytesDone;
            std::memcpy(s + bytesDone, m_Handler.s_CompressedChunk, toCopy);
            bytesDone += toCopy;
            m_BufferOffset = toCopy;
            break;
        }
    }
}

std::streamsize CStateDecompressor::CDechunkFilter::endOfStream(char* s, std::streamsize n, std::streamsize bytesDone) {
    // return [ ] if not m_Initialised
    m_EndOfStream = true;
    if (!m_SentData && bytesDone == 0) {
        std::streamsize toCopy = std::min(std::streamsize(EMPTY_DATA.size()), n);
        ::memcpy(s, EMPTY_DATA.c_str(), toCopy);
        return toCopy;
    }

    LOG_TRACE("Returning " << bytesDone << " of " << n << " bytes");

    return (bytesDone == 0) ? -1 : bytesDone;
}

void CStateDecompressor::CDechunkFilter::close() {
}

bool CStateDecompressor::CDechunkFilter::SRapidJsonHandler::Bool(bool) {
    s_Type = E_TokenBool;
    return true;
}

bool CStateDecompressor::CDechunkFilter::SRapidJsonHandler::String(const char* str, rapidjson::SizeType length, bool) {
    s_Type = E_TokenString;
    s_CompressedChunk = str;
    s_CompressedChunkLength = length;
    return true;
}

bool CStateDecompressor::CDechunkFilter::SRapidJsonHandler::Key(const char* str, rapidjson::SizeType length, bool) {
    s_Type = E_TokenKey;
    s_CompressedChunk = str;
    s_CompressedChunkLength = length;
    return true;
}

bool CStateDecompressor::CDechunkFilter::SRapidJsonHandler::StartObject() {
    s_Type = E_TokenObjectStart;
    return true;
}

bool CStateDecompressor::CDechunkFilter::SRapidJsonHandler::EndObject(rapidjson::SizeType) {
    s_Type = E_TokenObjectEnd;
    return true;
}

bool CStateDecompressor::CDechunkFilter::SRapidJsonHandler::StartArray() {
    s_Type = E_TokenArrayStart;
    return true;
}

bool CStateDecompressor::CDechunkFilter::SRapidJsonHandler::EndArray(rapidjson::SizeType) {
    s_Type = E_TokenArrayEnd;
    return true;
}

} // core
} // ml
