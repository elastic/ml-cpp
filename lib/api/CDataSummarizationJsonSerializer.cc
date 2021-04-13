/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataSummarizationJsonSerializer.h>

#include <core/CBase64Filter.h>
#include <core/CDataFrame.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/Constants.h>

#include <maths/CDataFrameCategoryEncoder.h>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>

#include <memory>
#include <sstream>
#include <string>
#include <utility>

namespace ml {
namespace api {

namespace {
using TRowItr = core::CDataFrame::TRowItr;
using TFilteredInput = boost::iostreams::filtering_stream<boost::iostreams::input>;

std::stringstream decompressStream(std::stringstream&& compressedStream) {
    std::stringstream decompressedStream;
    {
        TFilteredInput inFilter;
        inFilter.push(boost::iostreams::gzip_decompressor());
        inFilter.push(core::CBase64Decoder());
        inFilter.push(compressedStream);
        boost::iostreams::copy(inFilter, decompressedStream);
    }
    return decompressedStream;
}

// clang-format off
const std::string JSON_COMPRESSED_DATA_SUMMARIZATION_TAG{"compressed_data_summarization"};
const std::string JSON_DATA_SUMMARIZATION_TAG{"data_summarization"};
const std::string JSON_NUM_COLUMNS_TAG{"num_columns"};
const std::string JSON_COLUMN_NAMES_TAG{"column_names"};
const std::string JSON_COLUMN_IS_CATEGORICAL_TAG{"column_is_categorical"};
const std::string JSON_CATEGORICAL_COLUMN_VALUES_TAG{"categorical_column_values"};
const std::string JSON_ENCODINGS_TAG{"encodings"};
const std::string JSON_DATA_TAG{"data"};
// clang-format on
}

CDataSummarizationJsonSerializer::CDataSummarizationJsonSerializer(const core::CDataFrame& frame,
                                                                   core::CPackedBitVector rowMask,
                                                                   std::stringstream encodings)
    : m_RowMask(std::move(rowMask)), m_Frame(frame),
      m_Encodings(std::move(encodings)) {
}

void CDataSummarizationJsonSerializer::addToDocumentCompressed(TRapidJsonWriter& writer) const {
    CSerializableToJsonDocumentCompressed::addToDocumentCompressed(
        writer, JSON_COMPRESSED_DATA_SUMMARIZATION_TAG, JSON_DATA_SUMMARIZATION_TAG);
}

std::string CDataSummarizationJsonSerializer::jsonString() const {
    std::ostringstream jsonStrm;
    this->jsonStream(jsonStrm);
    return jsonStrm.str();
}

void CDataSummarizationJsonSerializer::addToJsonStream(TGenericLineWriter& writer) const {
    writer.StartObject();

    writer.Key(JSON_NUM_COLUMNS_TAG);
    writer.Uint64(m_Frame.numberColumns());

    writer.Key(JSON_COLUMN_NAMES_TAG);
    writer.StartArray();
    for (const auto& columnName : m_Frame.columnNames()) {
        writer.String(columnName);
    }
    writer.EndArray();

    writer.Key(JSON_COLUMN_IS_CATEGORICAL_TAG);
    writer.StartArray();
    for (auto columnIsCategorical : m_Frame.columnIsCategorical()) {
        writer.Bool(columnIsCategorical);
    }
    writer.EndArray();

    writer.Key(JSON_ENCODINGS_TAG);
    rapidjson::Document d;
    d.Parse(m_Encodings.str());
    // TODO check if parsing was successful
    writer.write(d);

    writer.Key(JSON_CATEGORICAL_COLUMN_VALUES_TAG);
    writer.StartArray();
    for (auto& categoricalColumnValuesItem : m_Frame.categoricalColumnValues()) {
        writer.StartArray();
        for (auto& categoricalValue : categoricalColumnValuesItem) {
            writer.String(categoricalValue);
        }
        writer.EndArray();
    }
    writer.EndArray();

    writer.Key(JSON_DATA_TAG);
    writer.StartArray();
    auto writeRowsToJson = [&](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            writer.StartArray();
            for (std::size_t i = 0; i < m_Frame.numberColumns(); ++i) {
                auto value = (*row)[i];
                if (m_Frame.categoricalColumnValues()[i].empty()) {
                    writer.String(std::to_string(value));
                } else {
                    writer.String(
                        m_Frame.categoricalColumnValues()[i][static_cast<std::size_t>(value)]);
                }
            }
            writer.EndArray();
        }
    };
    m_Frame.readRows(1, 0, m_Frame.numberRows(), writeRowsToJson, &m_RowMask);
    writer.EndArray();
    writer.EndObject();
}

CDataSummarizationJsonSerializer::TDataSummarization
CDataSummarizationJsonSerializer::fromDocumentCompressed(const TIStreamSPtr& istream) {
    rapidjson::IStreamWrapper isw(*istream);
    rapidjson::Document d;
    d.ParseStream(isw);
    if (d.HasMember(JSON_COMPRESSED_DATA_SUMMARIZATION_TAG) &&
        d[JSON_COMPRESSED_DATA_SUMMARIZATION_TAG].IsObject()) {
        auto& compressedDataSummarization = d[JSON_COMPRESSED_DATA_SUMMARIZATION_TAG];
        if (compressedDataSummarization.HasMember(JSON_DATA_SUMMARIZATION_TAG) &&
            compressedDataSummarization[JSON_DATA_SUMMARIZATION_TAG].IsString()) {
            std::stringstream compressedStream{
                compressedDataSummarization[JSON_DATA_SUMMARIZATION_TAG].GetString()};
            auto decompressedSPtr =
                std::make_shared<std::stringstream>(decompressStream(std::stringstream(
                    compressedDataSummarization[JSON_DATA_SUMMARIZATION_TAG].GetString())));
            return CDataSummarizationJsonSerializer::fromJsonStream(decompressedSPtr);
        } else {
            LOG_ERROR(<< "Field " << JSON_DATA_SUMMARIZATION_TAG
                      << " not found or is not a string.");
        }
    } else {
        LOG_ERROR(<< "Field " << JSON_COMPRESSED_DATA_SUMMARIZATION_TAG
                  << " not found or is not an object.");
    }
    return {nullptr, nullptr};
}

CDataSummarizationJsonSerializer::TDataSummarization
CDataSummarizationJsonSerializer::fromJsonStream(const TIStreamSPtr& istream) {
    using TStrVec = std::vector<std::string>;
    using TStrVecVec = std::vector<TStrVec>;
    using TBoolVec = std::vector<bool>;
    using TDataFrameUPtr = std::unique_ptr<core::CDataFrame>;
    using TEncoderUPtr = std::unique_ptr<maths::CDataFrameCategoryEncoder>;
    std::size_t numColumns;
    TStrVec columnNames;
    TStrVecVec categoricalColumnValues;
    TBoolVec columnIsCategorical;
    TDataFrameUPtr frame;
    TEncoderUPtr encoder;

    rapidjson::IStreamWrapper isw(*istream);
    rapidjson::Document d;
    d.ParseStream(isw);
    if (d.HasMember(JSON_NUM_COLUMNS_TAG) && d[JSON_NUM_COLUMNS_TAG].IsUint64()) {
        numColumns = d[JSON_NUM_COLUMNS_TAG].GetUint64();
        columnNames.reserve(numColumns);
        columnIsCategorical.reserve(numColumns);
        categoricalColumnValues.resize(numColumns, {});
    } else {
        LOG_ERROR(<< "Data summarization field '" << JSON_NUM_COLUMNS_TAG
                  << "'  is missing or has an unexpected format.");
        return {nullptr, nullptr};
    }

    if (d.HasMember(JSON_COLUMN_NAMES_TAG) && d[JSON_COLUMN_NAMES_TAG].IsArray()) {
        for (auto& item : d[JSON_COLUMN_NAMES_TAG].GetArray()) {
            columnNames.push_back(item.GetString());
        }
    } else {
        LOG_ERROR(<< "Data summarization field '" << JSON_COLUMN_NAMES_TAG
                  << "'  is missing or has an unexpected format.");
        return {nullptr, nullptr};
    }

    if (d.HasMember(JSON_COLUMN_IS_CATEGORICAL_TAG) &&
        d[JSON_COLUMN_IS_CATEGORICAL_TAG].IsArray()) {
        for (auto& item : d[JSON_COLUMN_IS_CATEGORICAL_TAG].GetArray()) {
            columnIsCategorical.push_back(item.GetBool());
        }
    } else {
        LOG_ERROR(<< "Data summarization field '" << JSON_COLUMN_IS_CATEGORICAL_TAG
                  << "'  is missing or has an unexpected format.");
        return {nullptr, nullptr};
    }
    if (d.HasMember(JSON_CATEGORICAL_COLUMN_VALUES_TAG) &&
        d[JSON_CATEGORICAL_COLUMN_VALUES_TAG].IsArray()) {
        std::size_t i{0};
        for (auto& item : d[JSON_CATEGORICAL_COLUMN_VALUES_TAG].GetArray()) {
            for (auto& categoricalValue : item.GetArray()) {
                categoricalColumnValues[i].push_back(categoricalValue.GetString());
            }
            ++i;
        }
    } else {
        LOG_ERROR(<< "Data summarization field '" << JSON_CATEGORICAL_COLUMN_VALUES_TAG
                  << "'  is missing or has an unexpected format.");
        return {nullptr, nullptr};
    }

    if (d.HasMember(JSON_ENCODINGS_TAG) && d[JSON_ENCODINGS_TAG].IsObject()) {
        std::stringstream jsonStrm;
        rapidjson::OStreamWrapper wrapper{jsonStrm};
        TGenericLineWriter writer{wrapper};
        writer.StartObject();
        writer.Key(JSON_ENCODINGS_TAG);
        writer.write(d[JSON_ENCODINGS_TAG].GetObject());
        writer.EndObject();
        core::CJsonStateRestoreTraverser traverser(jsonStrm);
        encoder = std::make_unique<maths::CDataFrameCategoryEncoder>(traverser);
    } else {
        LOG_ERROR(<< "Data summarization field '" << JSON_ENCODINGS_TAG
                  << "'  is missing or has an unexpected format.");
        return {nullptr, nullptr};
    }

    if (d.HasMember(JSON_DATA_TAG) && d[JSON_DATA_TAG].IsArray()) {
        frame = core::makeMainStorageDataFrame(numColumns).first;
        frame->columnNames(columnNames);
        frame->categoricalColumns(columnIsCategorical);
        frame->categoricalColumnValues(categoricalColumnValues);
        TStrVec rowVec;
        rowVec.reserve(numColumns);

        for (auto& row : d[JSON_DATA_TAG].GetArray()) {
            for (auto& item : row.GetArray()) {
                rowVec.push_back(item.GetString());
            }
            frame->parseAndWriteRow(
                core::CVectorRange<const TStrVec>(rowVec, 0, rowVec.size()));
            rowVec.clear();
        }
        frame->finishWritingRows();
    } else {
        LOG_ERROR(<< "Data summarization field '" << JSON_DATA_TAG
                  << "'  is missing or has an unexpected format.");
        return {nullptr, nullptr};
    }
    return {std::move(frame), std::move(encoder)};
}
}
}
