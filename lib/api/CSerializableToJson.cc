/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CSerializableToJson.h>

#include <core/CBase64Filter.h>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/rapidjson.h>

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

namespace ml {
namespace api {
namespace {
namespace io = boost::iostreams;
using TFilteredInput = io::filtering_stream<io::input>;
using TFilteredOutput = io::filtering_stream<io::output>;
using TGenericLineWriter = core::CRapidJsonLineWriter<rapidjson::OStreamWrapper>;

void compressAndEncode(std::function<void(TGenericLineWriter&)> addToJsonStream,
                       std::ostream& sink) {
    TFilteredOutput outFilter;
    outFilter.push(io::gzip_compressor());
    outFilter.push(core::CBase64Encoder());
    outFilter.push(sink);
    rapidjson::OStreamWrapper osw{outFilter};
    TGenericLineWriter writer{osw};
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

const std::string JSON_DOC_NUM_TAG{"doc_num"};
const std::string JSON_EOS_TAG{"eos"};
}

std::string CSerializableToJsonStream::jsonString() const {
    std::ostringstream jsonStream;
    {
        rapidjson::OStreamWrapper osw{jsonStream};
        TGenericLineWriter writer{osw};
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
    TRapidJsonWriter& writer) const {

    using TCharVec = std::vector<char>;

    TCharVec buffer;
    buffer.reserve(m_MaxDocumentSize);
    io::stream<io::back_insert_device<TCharVec>> output{io::back_inserter(buffer)};
    compressAndEncode(this->callableAddToJsonStream(), output);

    std::size_t docNum{0};
    for (std::size_t i = 0; i < buffer.size(); i += m_MaxDocumentSize) {
        rapidjson::SizeType bytesToWrite{static_cast<rapidjson::SizeType>(
            std::min(m_MaxDocumentSize, buffer.size() - i))};

        writer.StartObject();
        writer.Key(compressedDocTag);
        writer.StartObject();
        writer.Key(JSON_DOC_NUM_TAG);
        writer.Uint64(docNum);
        writer.Key(payloadTag);
        writer.String(&buffer[i], bytesToWrite);
        if (i + bytesToWrite == buffer.size()) {
            writer.Key(JSON_EOS_TAG);
            writer.Bool(true);
        }
        writer.EndObject();
        writer.EndObject();

        ++docNum;
    }
}

CSerializableFromCompressedChunkedJson::TIStreamPtr
CSerializableFromCompressedChunkedJson::rawJsonStream(const std::string& compressedDocTag,
                                                      const std::string& payloadTag,
                                                      TIStreamPtr inputStream,
                                                      std::iostream& buffer) {
    if (inputStream != nullptr) {
        rapidjson::IStreamWrapper isw{*inputStream};
        try {
            rapidjson::Document doc;
            bool done{false};
            do {
                doc.ParseStream<rapidjson::kParseStopWhenDoneFlag>(isw);
                auto chunk = ifExists(compressedDocTag, getAsObjectFrom, doc);
                buffer.write(ifExists(payloadTag, getAsStringFrom, chunk),
                             ifExists(payloadTag, getStringLengthFrom, chunk));
                done = chunk.HasMember(JSON_EOS_TAG);
            } while (done == false && inputStream->eof() == false);
            return decodeAndDecompress(buffer);
        } catch (const std::runtime_error& e) { LOG_ERROR(<< e.what()); }
    }
    return nullptr;
}
}
}
