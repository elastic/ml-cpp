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

#include <api/CDataSummarizationJsonWriter.h>

#include <core/CDataFrame.h>
#include <core/CJsonStatePersistInserter.h>

#include <maths/analytics/CDataFrameCategoryEncoder.h>

#include <api/CInferenceModelDefinition.h>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/rapidjson.h>

#include <memory>
#include <sstream>
#include <utility>

namespace ml {
namespace api {

namespace {
using TDoubleVec = std::vector<double>;
using TStrVec = std::vector<std::string>;
using TStrVecVec = std::vector<TStrVec>;
using TRowItr = core::CDataFrame::TRowItr;

class CEncoderNameIndexMapBuilder final
    : public maths::analytics::CDataFrameCategoryEncoder::CVisitor {
public:
    using TStrSizePrVec = std::vector<std::pair<std::string, std::size_t>>;
    using TStrSizePrVecCItr = TStrSizePrVec::const_iterator;

public:
    CEncoderNameIndexMapBuilder(TStrVec fieldNames, TStrVecVec categoryNames)
        : m_FeatureNameProvider{std::move(fieldNames), std::move(categoryNames)} {}

    void addIdentityEncoding(std::size_t inputColumnIndex) override {
        m_Map.emplace_back(m_FeatureNameProvider.identityEncodingName(inputColumnIndex),
                           m_Map.size());
    }
    void addOneHotEncoding(std::size_t inputColumnIndex, std::size_t hotCategory) override {
        m_Map.emplace_back(m_FeatureNameProvider.oneHotEncodingName(inputColumnIndex, hotCategory),
                           m_Map.size());
    }
    void addTargetMeanEncoding(std::size_t inputColumnIndex, const TDoubleVec&, double) override {
        m_Map.emplace_back(m_FeatureNameProvider.targetMeanEncodingName(inputColumnIndex),
                           m_Map.size());
    }
    void addFrequencyEncoding(std::size_t inputColumnIndex, const TDoubleVec&) override {
        m_Map.emplace_back(m_FeatureNameProvider.frequencyEncodingName(inputColumnIndex),
                           m_Map.size());
    }

    TStrSizePrVecCItr begin() const { return m_Map.begin(); }
    TStrSizePrVecCItr end() const { return m_Map.end(); }

private:
    CTrainedModel::CFeatureNameProvider m_FeatureNameProvider;
    TStrSizePrVec m_Map;
};
}

CDataSummarizationJsonWriter::CDataSummarizationJsonWriter(
    const core::CDataFrame& frame,
    core::CPackedBitVector rowMask,
    std::size_t numberColumns,
    const maths::analytics::CDataFrameCategoryEncoder& encodings)
    : m_RowMask{std::move(rowMask)}, m_NumberColumns{numberColumns}, m_Frame{frame}, m_Encodings{encodings} {
}

void CDataSummarizationJsonWriter::addCompressedToJsonStream(TRapidJsonWriter& writer) const {
    this->CSerializableToCompressedChunkedJson::addCompressedToJsonStream(
        JSON_COMPRESSED_DATA_SUMMARIZATION_TAG, JSON_DATA_SUMMARIZATION_TAG, writer);
}

void CDataSummarizationJsonWriter::addToJsonStream(TGenericLineWriter& writer) const {

    // Note that the data frame has extra columns added to it when running training.
    // These are at the end and should not be serialized since it is wasteful and they
    // are reinitialised anyway. m_NumberColumns is the number of supplied columns,
    // i.e. the feature values and target variable.

    writer.StartObject();

    writer.Key(JSON_NUM_COLUMNS_TAG);
    writer.Uint64(m_NumberColumns);

    writer.Key(JSON_COLUMN_NAMES_TAG);
    writer.StartArray();
    for (std::size_t i = 0; i < m_NumberColumns; ++i) {
        writer.String(m_Frame.columnNames()[i]);
    }
    writer.EndArray();

    writer.Key(JSON_COLUMN_IS_CATEGORICAL_TAG);
    writer.StartArray();
    for (std::size_t i = 0; i < m_NumberColumns; ++i) {
        writer.Bool(m_Frame.columnIsCategorical()[i]);
    }
    writer.EndArray();

    CEncoderNameIndexMapBuilder encodingIndices{m_Frame.columnNames(),
                                                m_Frame.categoricalColumnValues()};
    m_Encodings.accept(encodingIndices);
    writer.Key(JSON_ENCODING_NAME_INDEX_MAP_TAG);
    writer.StartArray();
    for (const auto& index : encodingIndices) {
        writer.StartObject();
        writer.Key(JSON_ENCODING_NAME_INDEX_MAP_KEY_TAG);
        writer.String(index.first);
        writer.Key(JSON_ENCODING_NAME_INDEX_MAP_VALUE_TAG);
        writer.Uint64(index.second);
        writer.EndObject();
    }
    writer.EndArray();

    rapidjson::Document doc;
    std::stringstream encodings;
    {
        core::CJsonStatePersistInserter inserter{encodings};
        m_Encodings.acceptPersistInserter(inserter);
    }
    rapidjson::ParseResult ok(doc.Parse(encodings.str()));
    if (static_cast<bool>(ok) == false) {
        LOG_ERROR(<< "Failed parsing encoding json: "
                  << rapidjson::GetParseError_En(doc.GetParseError())
                  << ". Please report this error.");
    } else {
        writer.Key(JSON_ENCODINGS_TAG);
        writer.write(doc);
    }

    writer.Key(JSON_CATEGORICAL_COLUMN_VALUES_TAG);
    writer.StartArray();
    for (std::size_t i = 0; i < m_NumberColumns; ++i) {
        writer.StartArray();
        for (const auto& category : m_Frame.categoricalColumnValues()[i]) {
            writer.String(category);
        }
        writer.EndArray();
    }
    writer.EndArray();

    writer.Key(JSON_DATA_TAG);
    writer.StartArray();
    auto writeRowsToJson = [&](const TRowItr& beginRows, const TRowItr& endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            writer.StartArray();
            for (std::size_t i = 0; i < m_NumberColumns; ++i) {
                auto value = (*row)[i];
                if (core::CDataFrame::isMissing(value)) {
                    writer.String(m_Frame.missingString());
                } else if (m_Frame.categoricalColumnValues()[i].empty()) {
                    writer.String(value.toString());
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
