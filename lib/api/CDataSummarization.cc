/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataSummarization.h>

#include <core/CBase64Filter.h>
#include <core/Constants.h>

#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>

namespace ml {
namespace api {

namespace {
// clang-format off
const std::string JSON_DOC_NUM_TAG{"doc_num"};
const std::string JSON_COMPRESSED_DATA_SUMMARIZATION_TAG{"compressed_data_summarization"};
const std::string JSON_DATA_SUMMARIZATION_TAG{"data_summarization"};
const std::string JSON_EOS_TAG{"eos"};

// clang-format on
const std::size_t MAX_DOCUMENT_SIZE(16 * core::constants::BYTES_IN_MEGABYTES);
}

void CDataSummarization::addToDocumentCompressed(TRapidJsonWriter& writer) const {
    std::stringstream compressedStream{this->jsonCompressedStream()};
    std::streamsize processed{0};
    compressedStream.seekg(0, compressedStream.end);
    std::streamsize remained{compressedStream.tellg()};
    compressedStream.seekg(0, compressedStream.beg);
    std::size_t docNum{0};
    std::string buffer;
    while (remained > 0) {
        std::size_t bytesToProcess{std::min(MAX_DOCUMENT_SIZE, static_cast<size_t>(remained))};
        buffer.clear();
        std::copy_n(std::istreambuf_iterator<char>(compressedStream.seekg(processed)),
                    bytesToProcess, std::back_inserter(buffer));
        remained -= bytesToProcess;
        processed += bytesToProcess;
        writer.StartObject();
        writer.Key(JSON_COMPRESSED_DATA_SUMMARIZATION_TAG);
        writer.StartObject();
        writer.Key(JSON_DOC_NUM_TAG);
        writer.Uint64(docNum);
        writer.Key(JSON_DATA_SUMMARIZATION_TAG);
        writer.String(buffer);
        if (remained == 0) {
            writer.Key(JSON_EOS_TAG);
            writer.Bool(true);
        }
        writer.EndObject();
        writer.EndObject();
        ++docNum;
    }
}

std::stringstream CDataSummarization::jsonCompressedStream() const {
    std::stringstream compressedStream;
    using TFilteredOutput = boost::iostreams::filtering_stream<boost::iostreams::output>;
    {
        TFilteredOutput outFilter;
        outFilter.push(boost::iostreams::gzip_compressor());
        outFilter.push(core::CBase64Encoder());
        outFilter.push(compressedStream);
        this->jsonStream(outFilter);
    }
    return compressedStream;
}

std::string CDataSummarization::jsonString() const {
    std::ostringstream jsonStrm;
    this->jsonStream(jsonStrm);
    return jsonStrm.str();
}

void CDataSummarization::jsonStream(std::ostream& jsonStrm) const {
    rapidjson::OStreamWrapper wrapper{jsonStrm};
    TGenericLineWriter writer{wrapper};
    this->addToJsonStream(writer);
    jsonStrm.flush();
}

void CDataSummarization::addToJsonStream(TGenericLineWriter& writer) const {
    writer.StartObject();
    // TODO #1829 implement this method
    writer.EndObject();
}
}
}
