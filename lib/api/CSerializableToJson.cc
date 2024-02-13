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

#include <api/CSerializableToJson.h>

#include <core/CBase64Filter.h>
#include <core/CBoostJsonParser.h>
#include <core/CBoostJsonUnbufferedIStreamWrapper.h>
#include <core/CStreamWriter.h>

#include <boost/json.hpp>

#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/stream.hpp>

#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace json = boost::json;
namespace ml {
namespace api {
namespace {
namespace io = boost::iostreams;
using TFilteredInput = io::filtering_stream<io::input>;
using TFilteredOutput = io::filtering_stream<io::output>;
using TStreamWriter = core::CStreamWriter;
using TGenericLineWriter = TStreamWriter;

void compressAndEncode(std::function<void(TGenericLineWriter&)> addToJsonStream,
                       std::ostream& sink) {
    TFilteredOutput outFilter;
    outFilter.push(io::gzip_compressor());
    outFilter.push(core::CBase64Encoder());
    outFilter.push(sink);
    TGenericLineWriter writer{outFilter};
    addToJsonStream(writer);
    outFilter.flush();
}

auto decodeAndDecompress(std::istream& inputStream) {
    auto result = std::make_shared<TFilteredInput>();
    result->push(io::gzip_decompressor());
    result->push(core::CBase64Decoder());
    result->push(inputStream);
    return result;
}

void consumeSpace(std::istream& stream) {
    while (std::isspace(stream.peek()) != 0) {
        stream.get();
    }
}

const std::string JSON_DOC_NUM_TAG{"doc_num"};
const std::string JSON_EOS_TAG{"eos"};
}

std::string CSerializableToJsonStream::jsonString() const {
    std::ostringstream jsonStream;
    {
        TStreamWriter writer{jsonStream};
        this->addToJsonStream(writer);
    }
    return jsonStream.str();
}

CSerializableToCompressedChunkedJson::CSerializableToCompressedChunkedJson(std::size_t maxDocumentSize)
    : m_MaxDocumentSize{std::min(maxDocumentSize, MAX_DOCUMENT_SIZE)} {
}

std::stringstream CSerializableToCompressedChunkedJson::jsonCompressedStream() const {
    std::stringstream result;
    compressAndEncode(this->callableAddToJsonStream(), result);
    return result;
}

void CSerializableToCompressedChunkedJson::addCompressedToJsonStream(
    const std::string& compressedDocTag,
    const std::string& payloadTag,
    TBoostJsonWriter& writer) const {

    using TCharVec = std::vector<char>;

    TCharVec buffer;
    buffer.reserve(m_MaxDocumentSize);
    io::stream<io::back_insert_device<TCharVec>> bufferStream{io::back_inserter(buffer)};
    compressAndEncode(this->callableAddToJsonStream(), bufferStream);

    std::size_t docNum{0};
    for (std::size_t i = 0; i < buffer.size(); i += m_MaxDocumentSize) {
        std::size_t bytesToWrite = std::min(m_MaxDocumentSize, buffer.size() - i);

        writer.onObjectBegin();
        writer.onKey(compressedDocTag);
        writer.onObjectBegin();
        writer.onKey(JSON_DOC_NUM_TAG);
        writer.onUint64(docNum);
        writer.onKey(payloadTag);
        writer.onString(std::string(&buffer[i], bytesToWrite));
        if (i + bytesToWrite == buffer.size()) {
            writer.onKey(JSON_EOS_TAG);
            writer.onBool(true);
        }
        writer.onObjectEnd();
        writer.onObjectEnd();

        ++docNum;
    }
}

CSerializableFromCompressedChunkedJson::TIStreamPtr
CSerializableFromCompressedChunkedJson::rawJsonStream(const std::string& compressedDocTag,
                                                      const std::string& payloadTag,
                                                      TIStreamPtr inputStream,
                                                      std::iostream& buffer) {
    if (inputStream != nullptr) {
        try {
            json::value doc;
            json::error_code ec;
            json::stream_parser p;
            std::string line;
            bool done{false};
            while (inputStream->eof() == false && done == false) {
                if (inputStream->peek() == '\0') {
                    inputStream->get();
                    continue;
                }
                std::getline(*inputStream, line);
                ec = core::CBoostJsonParser::parse(line.data(), line.length(), doc);
                assertNoParseError(ec);
                assertIsJsonObject(doc);
                try {
                    auto chunk = ifExists(compressedDocTag, getAsObjectFrom,
                                          doc.as_object());
                    buffer.write(ifExists(payloadTag, getAsStringFrom, chunk),
                                 ifExists(payloadTag, getStringLengthFrom, chunk));
                    done = chunk.contains(JSON_EOS_TAG);
                } catch (const std::runtime_error& e) {
                    LOG_WARN(<< "Caught exception: " << e.what());
                    continue;
                }
            }

            consumeSpace(*inputStream);

            return decodeAndDecompress(buffer);

        } catch (const std::runtime_error& e) { LOG_ERROR(<< e.what()); }
    }
    return nullptr;
}
}
}
