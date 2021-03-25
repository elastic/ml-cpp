/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataSummarization.h>

#include <core/CBase64Filter.h>
#include <core/CDataFrame.h>
#include <core/CPackedBitVector.h>
#include <core/Constants.h>

#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>

#include <string>

namespace ml {
namespace api {

namespace {
using TRowItr = core::CDataFrame::TRowItr;

// clang-format off
const std::string JSON_COMPRESSED_DATA_SUMMARIZATION_TAG{"compressed_data_summarization"};
const std::string JSON_DATA_SUMMARIZATION_TAG{"data_summarization"};
const std::string JSON_NUM_ROWS_TAG{"num_rows"};
const std::string JSON_NUM_COLUMNS_TAG{"num_columns"};
const std::string JSON_COLUMN_NAMES_TAG{"column_names"};
const std::string JSON_COLUMN_IS_CATEGORICAL_TAG{"column_is_categorical"};
const std::string JSON_DATA_TAG{"data"};
// clang-format on
const std::size_t MAX_DOCUMENT_SIZE(16 * core::constants::BYTES_IN_MEGABYTES);
}

CDataSummarization::CDataSummarization(const core::CDataFrame& frame, core::CPackedBitVector rowMask)
    : m_RowMask(rowMask), m_Frame(frame) {
}

void CDataSummarization::addToDocumentCompressed(TRapidJsonWriter& writer) const {
    CSerializableToJsonDocumentCompressed::addToDocumentCompressed(writer, JSON_COMPRESSED_DATA_SUMMARIZATION_TAG);
}

std::string CDataSummarization::jsonString() const {
    std::ostringstream jsonStrm;
    this->jsonStream(jsonStrm);
    return jsonStrm.str();
}

void CDataSummarization::addToJsonStream(TGenericLineWriter& writer) const {
    writer.StartObject();
    writer.Key(JSON_NUM_ROWS_TAG);
    writer.Uint64(static_cast<std::size_t>(m_RowMask.manhattan()));
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
}
}
