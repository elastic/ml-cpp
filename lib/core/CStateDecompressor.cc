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
#include <core/CStateDecompressor.h>

#include <core/CBase64Filter.h>
#include <core/CStateCompressor.h>

#include <boost/iostreams/filter/gzip.hpp>
// This file must be manually included when
// using basic_parser to implement a parser.
#include <boost/json/basic_parser_impl.hpp>

#include <algorithm>
#include <cstring>
#include <string>

namespace json = boost::json;

namespace ml {
namespace core {

const std::string CStateDecompressor::EMPTY_DATA{"H4sIAAAAAAAA/4uOBQApu0wNAgAAAA=="};

CStateDecompressor::CStateDecompressor(CDataSearcher& compressedSearcher)
    : m_FilterSource{compressedSearcher} {
    m_InFilter.reset(new TFilteredInput{});
    m_InFilter->push(boost::iostreams::gzip_decompressor());
    m_InFilter->push(CBase64Decoder{});
    m_InFilter->push(boost::ref(m_FilterSource));
}

CDataSearcher::TIStreamP CStateDecompressor::search(std::size_t /*currentDocNum*/,
                                                    std::size_t /*limit*/) {
    return m_InFilter;
}

CStateDecompressor::CDechunkFilter::CDechunkFilter(CDataSearcher& searcher)
    : m_Initialised{false}, m_SentData{false}, m_Searcher{searcher},
      m_CurrentDocNum{1}, m_EndOfStream{false},
      m_Reader(nullptr), m_BufferOffset{0}, m_NestedLevel{1} {
}

std::streamsize CStateDecompressor::CDechunkFilter::read(char* s, std::streamsize n) {
    if (m_EndOfStream) {
        LOG_TRACE(<< "EOS -1");
        return -1;
    }
    // return number of bytes read, -1 for EOF
    std::streamsize bytesDone = 0;
    while (bytesDone < n) {
        if (!m_IStream) {
            // Get a new input stream
            LOG_TRACE(<< "Getting new stream, for document number " << m_CurrentDocNum);

            m_IStream = m_Searcher.search(m_CurrentDocNum, 1);
            if (!m_IStream) {
                LOG_ERROR(<< "Unable to connect to data store");
                return this->endOfStream(s, n, bytesDone);
            }
            if (m_IStream->bad()) {
                LOG_ERROR(<< "Error connecting to data store");
                return this->endOfStream(s, n, bytesDone);
            }

            if (m_IStream->fail()) {
                m_EndOfStream = true;
                // This is not fatal - we just didn't find the given document number
                // Presume that we have finished
                LOG_TRACE(<< "No more documents to find");
                return this->endOfStream(s, n, bytesDone);
            }

            m_InputStreamWrapper.reset(new CBoostJsonUnbufferedIStreamWrapper(*m_IStream));

            m_Reader.reset(new json::basic_parser<SBoostJsonHandler>(json::parse_options{}));

            if (!this->readHeader()) {
                return this->endOfStream(s, n, bytesDone);
            }
        }

        this->handleRead(s, n, bytesDone);
        if (m_EndOfStream) {
            return this->endOfStream(s, n, bytesDone);
        }

        if ((m_IStream) && (m_IStream->eof())) {
            LOG_TRACE(<< "Stream EOF");
            m_IStream.reset();
            ++m_CurrentDocNum;
        }
    }
    LOG_TRACE(<< "Returning " << bytesDone << ": " << std::string(s, bytesDone));
    return bytesDone;
}

bool CStateDecompressor::CDechunkFilter::parseNext() {
    bool ret{true};
    m_Reader->handler().s_HaveCompleteToken = false;
    do {
        char c = m_InputStreamWrapper->take();
        if (c == '\0') {
            std::string message;
            if (m_ParsingStarted == false) {
                message = "Encountered NULL character in stream before parsing has started.";
                ret = false;
            }
            if (m_Reader->handler().s_Type == SBoostJsonHandler::E_TokenObjectEnd) {
                message = "Encountered NULL character in stream after object end.";
                ret = false;
            }
            if (ret == false && message.empty() == false) {
                std::string jsonStr(m_Reader->handler().s_CompressedChunk,
                    m_Reader->handler().s_CompressedChunkLength);
                LOG_WARN(<< "Error parsing JSON: \"" << jsonStr << "\". " << message);
            }
            break;
        }

        m_ParsingStarted = true;

        // Note that we're passing one byte at a time to the basic_parser.
        // This is because, for every complete JSON token seen, we want parseNext
        // to immediately return. That is, we don't want to trigger multiple parser
        // callbacks for every call to parseNext as that will clobber the state
        // we've been building up.
        json::error_code ec;
        m_Reader->write_some(true, &c, 1, ec);
        if (ec) {
            LOG_ERROR(<< "Error parsing JSON: " << ec.message());
            ret = false;
            break;
        }
    } while (m_Reader->handler().s_HaveCompleteToken == false);

    return ret;
}

bool CStateDecompressor::CDechunkFilter::readHeader() {
    if (this->parseNext() == false) {
        LOG_ERROR(<< "Failed to find valid JSON");
        m_Initialised = false;
        m_IStream.reset();
        ++m_CurrentDocNum;
        return false;
    }

    while (this->parseNext()) {
        if (m_Reader->handler().s_Type == SBoostJsonHandler::E_TokenKey &&
            CStateCompressor::COMPRESSED_ATTRIBUTE.compare(
                0, CStateCompressor::COMPRESSED_ATTRIBUTE.length(),
                m_Reader->handler().s_CompressedChunk,
                m_Reader->handler().s_CompressedChunkLength) == 0) {
            if (this->parseNext() && m_Reader->handler().s_Type ==
                                         SBoostJsonHandler::E_TokenArrayStart) {
                m_Initialised = true;
                m_BufferOffset = 0;
                return true;
            }
        } else if (m_Reader->handler().s_Type == SBoostJsonHandler::E_TokenObjectStart) {
            ++m_NestedLevel;
        }
    }
    // If we are here, we have got an empty document from downstream,
    // so the stream is finished
    LOG_WARN(<< "Failed to find 'compressed' data array!");
    m_Initialised = false;
    m_IStream.reset();
    ++m_CurrentDocNum;
    return false;
}

void CStateDecompressor::CDechunkFilter::handleRead(char* s,
                                                    std::streamsize n,
                                                    std::streamsize& bytesDone) {
    // Extract data from the JSON array "compressed"
    if (!m_Initialised) {
        return;
    }

    // Copy any outstanding data
    if (m_BufferOffset > 0) {
        std::streamsize toCopy = std::min(
            (n - bytesDone), (m_Reader->handler().s_CompressedChunkLength - m_BufferOffset));
        std::memcpy(s + bytesDone,
                    m_Reader->handler().s_CompressedChunk + m_BufferOffset, toCopy);
        bytesDone += toCopy;
        m_BufferOffset += toCopy;
    }

    // Expect to have data in an array
    while (bytesDone < n && this->parseNext()) {
        m_BufferOffset = 0;
        if (m_Reader->handler().s_Type == SBoostJsonHandler::E_TokenArrayEnd) {
            LOG_TRACE(<< "Come to end of array");
            if (this->parseNext() && m_Reader->handler().s_Type == SBoostJsonHandler::E_TokenKey &&
                CStateCompressor::END_OF_STREAM_ATTRIBUTE.compare(
                    0, CStateCompressor::END_OF_STREAM_ATTRIBUTE.length(),
                    m_Reader->handler().s_CompressedChunk,
                    m_Reader->handler().s_CompressedChunkLength) == 0) {
                LOG_DEBUG(<< "Explicit end-of-stream marker found in document with index "
                          << m_CurrentDocNum);

                // Read the value of the CStateCompressor::END_OF_STREAM_ATTRIBUTE field and the closing brace
                if (this->parseNext() &&
                    m_Reader->handler().s_Type != SBoostJsonHandler::E_TokenBool) {
                    LOG_ERROR(<< "Expecting bool value to follow  "
                              << CStateCompressor::END_OF_STREAM_ATTRIBUTE
                              << ", got " << m_Reader->handler().s_Type);
                }

                while (m_NestedLevel > 0) {
                    if (this->parseNext() && m_Reader->handler().s_Type !=
                                                 SBoostJsonHandler::E_TokenObjectEnd) {
                        LOG_ERROR(<< "Expecting end object to follow "
                                  << CStateCompressor::END_OF_STREAM_ATTRIBUTE
                                  << ", got " << m_Reader->handler().s_Type);
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
        if (m_Reader->handler().s_CompressedChunkLength <= (n - bytesDone)) {
            std::memcpy(s + bytesDone, m_Reader->handler().s_CompressedChunk,
                        m_Reader->handler().s_CompressedChunkLength);
            bytesDone += m_Reader->handler().s_CompressedChunkLength;
        } else {
            std::streamsize toCopy = n - bytesDone;
            std::memcpy(s + bytesDone, m_Reader->handler().s_CompressedChunk, toCopy);
            bytesDone += toCopy;
            m_BufferOffset = toCopy;
            break;
        }
    }
}

std::streamsize CStateDecompressor::CDechunkFilter::endOfStream(char* s,
                                                                std::streamsize n,
                                                                std::streamsize bytesDone) {
    // return [ ] if not m_Initialised - i.e. if no valid json could be found
    m_EndOfStream = true;
    if (!m_SentData && bytesDone == 0) {
        std::streamsize toCopy = std::min(std::streamsize(EMPTY_DATA.size()), n);
        ::memcpy(s, EMPTY_DATA.c_str(), toCopy);
        return toCopy;
    }

    LOG_TRACE(<< "Returning " << bytesDone << " of " << n << " bytes");

    return (bytesDone == 0) ? -1 : bytesDone;
}

void CStateDecompressor::CDechunkFilter::close() {
}

bool CStateDecompressor::CDechunkFilter::SBoostJsonHandler::on_bool(bool, json::error_code& ec) {
    s_Type = E_TokenBool;
    if (ec) {
        LOG_ERROR(<< "Parse error: " << ec.message());
        return false;
    }
    s_HaveCompleteToken = true;
    LOG_TRACE(<< "m_Reader->handler().s_Type = SBoostJsonHandler::E_TokenBool");
    return true;
}

bool CStateDecompressor::CDechunkFilter::SBoostJsonHandler::on_string(std::string_view str,
                                                                      std::size_t length,
                                                                      json::error_code& ec) {
    s_Type = E_TokenString;
    if (ec) {
        LOG_ERROR(<< "Parse error: " << ec.message());
        return false;
    }
    s_HaveCompleteToken = true;

    LOG_TRACE(<< "m_Reader->handler().s_Type = SBoostJsonHandler::E_TokenString");
    LOG_TRACE(<< "s_CompressedChunk = " << s_CompressedChunk
              << ", s_CompressedChunkLength = " << s_CompressedChunkLength);
    if (str.front() == '"') {
        s_StringEnd = true;
        s_NewToken = true;
        return true;
    }
    s_CompressedChunk[length - 1] = str.front();
    s_CompressedChunkLength = length;
    s_NewToken = true;
    return true;
}

bool CStateDecompressor::CDechunkFilter::SBoostJsonHandler::on_string_part(std::string_view str,
                                                                           std::size_t length,
                                                                           json::error_code& ec) {
    s_Type = E_TokenStringPart;
    if (ec) {
        LOG_ERROR(<< "Parse error: " << ec.message());
        return false;
    }
    LOG_TRACE(<< "m_Reader->handler().s_Type = SBoostJsonHandler::E_TokenStringPart");
    LOG_TRACE(<< "s_CompressedChunk = " << s_CompressedChunk
              << ", s_CompressedChunkLength = " << s_CompressedChunkLength);
    s_StringEnd = false;
    if (s_NewToken == true) {
        s_NewToken = false;
        memset(static_cast<void*>(s_CompressedChunk), '\0', COMPRESSED_CHUNK_BUFFER_LENGTH);
    }
    s_CompressedChunk[length - 1] = str.front();
    s_CompressedChunkLength = length;
    return true;
}

bool CStateDecompressor::CDechunkFilter::SBoostJsonHandler::on_key(std::string_view str,
                                                                   std::size_t length,
                                                                   json::error_code& ec) {
    s_Type = E_TokenKey;
    if (ec) {
        LOG_ERROR(<< "Parse error: " << ec.message());
        return false;
    }
    s_HaveCompleteToken = true;

    LOG_TRACE(<< "m_Reader->handler().s_Type = SBoostJsonHandler::E_TokenKey");
    LOG_TRACE(<< "s_CompressedChunk = " << s_CompressedChunk
              << ", s_CompressedChunkLength = " << s_CompressedChunkLength);
    if (str.front() == '"') {
        s_NewToken = true;
        return true;
    }
    s_CompressedChunk[length - 1] = str.front();
    s_CompressedChunkLength = length;
    s_NewToken = true;
    return true;
}

bool CStateDecompressor::CDechunkFilter::SBoostJsonHandler::on_key_part(std::string_view str,
                                                                        std::size_t length,
                                                                        json::error_code& ec) {
    s_Type = E_TokenKeyPart;
    if (ec) {
        LOG_ERROR(<< "Parse error: " << ec.message());
        return false;
    }
    LOG_TRACE(<< "m_Reader->handler().s_Type = SBoostJsonHandler::E_TokenKeyPart");
    LOG_TRACE(<< "s_CompressedChunk = " << s_CompressedChunk
              << ", s_CompressedChunkLength = " << s_CompressedChunkLength);
    if (s_NewToken == true) {
        s_NewToken = false;
        memset(s_CompressedChunk, '\0', COMPRESSED_CHUNK_BUFFER_LENGTH);
    }
    s_CompressedChunk[length - 1] = str.front();
    s_CompressedChunkLength = length;
    return true;
}

bool CStateDecompressor::CDechunkFilter::SBoostJsonHandler::on_object_begin(json::error_code& ec) {
    s_Type = E_TokenObjectStart;
    if (ec) {
        LOG_ERROR(<< "Parse error: " << ec.message());
        return false;
    }
    s_HaveCompleteToken = true;

    LOG_TRACE(<< "m_Reader->handler().s_Type = SBoostJsonHandler::E_TokenObjectStart");
    return true;
}

bool CStateDecompressor::CDechunkFilter::SBoostJsonHandler::on_object_end(std::size_t,
                                                                          json::error_code& ec) {
    s_Type = E_TokenObjectEnd;
    if (ec) {
        LOG_ERROR(<< "Parse error: " << ec.message());
        return false;
    }
    s_HaveCompleteToken = true;

    LOG_TRACE(<< "m_Reader->handler().s_Type = SBoostJsonHandler::E_TokenObjectEnd");
    return true;
}

bool CStateDecompressor::CDechunkFilter::SBoostJsonHandler::on_array_begin(json::error_code& ec) {
    s_Type = E_TokenArrayStart;
    if (ec) {
        LOG_ERROR(<< "Parse error: " << ec.message());
        return false;
    }
    s_HaveCompleteToken = true;

    LOG_TRACE(<< "m_Reader->handler().s_Type = SBoostJsonHandler::E_TokenArrayStart");
    return true;
}

bool CStateDecompressor::CDechunkFilter::SBoostJsonHandler::on_array_end(std::size_t,
                                                                         json::error_code& ec) {
    s_Type = E_TokenArrayEnd;
    if (ec) {
        LOG_ERROR(<< "Parse error: " << ec.message());
        return false;
    }
    s_HaveCompleteToken = true;

    LOG_TRACE(<< "m_Reader->handler().s_Type = SBoostJsonHandler::E_TokenArrayEnd");
    return true;
}

bool CStateDecompressor::CDechunkFilter::SBaseBoostJsonHandler::on_document_begin(json::error_code& ec) {
    return (ec) ? false : true;
}

bool CStateDecompressor::CDechunkFilter::SBaseBoostJsonHandler::on_document_end(json::error_code& ec) {
    return (ec) ? false : true;
}

bool CStateDecompressor::CDechunkFilter::SBaseBoostJsonHandler::on_number_part(
    std::string_view /* s */,
    json::error_code& ec) {
    return (ec) ? false : true;
}

bool CStateDecompressor::CDechunkFilter::SBaseBoostJsonHandler::on_int64(int64_t /* i */,
                                                                         std::string_view /* s */,
                                                                         json::error_code& ec) {
    return (ec) ? false : true;
}

bool CStateDecompressor::CDechunkFilter::SBaseBoostJsonHandler::on_uint64(uint64_t /* u */,
                                                                          std::string_view /* s */,
                                                                          json::error_code& ec) {
    return (ec) ? false : true;
}

bool CStateDecompressor::CDechunkFilter::SBaseBoostJsonHandler::on_double(double /* d */,
                                                                          std::string_view /* s */,
                                                                          json::error_code& ec) {
    return (ec) ? false : true;
}

bool CStateDecompressor::CDechunkFilter::SBaseBoostJsonHandler::on_null(json::error_code& ec) {
    return (ec) ? false : true;
}

bool CStateDecompressor::CDechunkFilter::SBaseBoostJsonHandler::on_comment_part(
    std::string_view /* s */,
    json::error_code& ec) {
    return (ec) ? false : true;
}

bool CStateDecompressor::CDechunkFilter::SBaseBoostJsonHandler::on_comment(std::string_view /* s */,
                                                                           json::error_code& ec) {
    return (ec) ? false : true;
}

} // core
} // ml
