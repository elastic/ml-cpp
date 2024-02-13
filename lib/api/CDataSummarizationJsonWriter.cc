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

#include <boost/json.hpp>

#include <memory>
#include <sstream>
#include <utility>

namespace json = boost::json;

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

void CDataSummarizationJsonWriter::addCompressedToJsonStream(TBoostJsonWriter& writer) const {
    this->CSerializableToCompressedChunkedJson::addCompressedToJsonStream(
        JSON_COMPRESSED_DATA_SUMMARIZATION_TAG, JSON_DATA_SUMMARIZATION_TAG, writer);
}

void CDataSummarizationJsonWriter::addToJsonStream(TGenericLineWriter& writer) const {

    // Note that the data frame has extra columns added to it when running training.
    // These are at the end and should not be serialized since it is wasteful and they
    // are reinitialised anyway. m_NumberColumns is the number of supplied columns,
    // i.e. the feature values and target variable.

    writer.onObjectBegin();

    writer.onKey(JSON_NUM_COLUMNS_TAG);
    writer.onUint64(m_NumberColumns);

    writer.onKey(JSON_COLUMN_NAMES_TAG);
    writer.onArrayBegin();
    for (std::size_t i = 0; i < m_NumberColumns; ++i) {
        writer.onString(m_Frame.columnNames()[i]);
    }
    writer.onArrayEnd();

    writer.onKey(JSON_COLUMN_IS_CATEGORICAL_TAG);
    writer.onArrayBegin();
    for (std::size_t i = 0; i < m_NumberColumns; ++i) {
        writer.onBool(m_Frame.columnIsCategorical()[i]);
    }
    writer.onArrayEnd();

    CEncoderNameIndexMapBuilder encodingIndices{m_Frame.columnNames(),
                                                m_Frame.categoricalColumnValues()};
    m_Encodings.accept(encodingIndices);
    writer.onKey(JSON_ENCODING_NAME_INDEX_MAP_TAG);
    writer.onArrayBegin();
    for (const auto& index : encodingIndices) {
        writer.onObjectBegin();
        writer.onKey(JSON_ENCODING_NAME_INDEX_MAP_KEY_TAG);
        writer.onString(index.first);
        writer.onKey(JSON_ENCODING_NAME_INDEX_MAP_VALUE_TAG);
        writer.onUint64(index.second);
        writer.onObjectEnd();
    }
    writer.onArrayEnd();

    std::stringstream encodings;
    {
        core::CJsonStatePersistInserter inserter{encodings};
        m_Encodings.acceptPersistInserter(inserter);
    }
    std::string encodingsStr = encodings.str();
    core::CStringUtils::trim("\n", encodingsStr);
    writer.onKey(JSON_ENCODINGS_TAG);
    writer.onRawString(encodingsStr);

    writer.onKey(JSON_CATEGORICAL_COLUMN_VALUES_TAG);
    writer.onArrayBegin();
    for (std::size_t i = 0; i < m_NumberColumns; ++i) {
        writer.onArrayBegin();
        for (const auto& category : m_Frame.categoricalColumnValues()[i]) {
            writer.onString(category);
        }
        writer.onArrayEnd();
    }
    writer.onArrayEnd();

    writer.onKey(JSON_DATA_TAG);
    writer.onArrayBegin();
    auto writeRowsToJson = [&](const TRowItr& beginRows, const TRowItr& endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            writer.onArrayBegin();
            for (std::size_t i = 0; i < m_NumberColumns; ++i) {
                auto value = (*row)[i];
                if (core::CDataFrame::isMissing(value)) {
                    writer.onString(m_Frame.missingString());
                } else if (m_Frame.categoricalColumnValues()[i].empty()) {
                    writer.onString(value.toString());
                } else {
                    writer.onString(
                        m_Frame.categoricalColumnValues()[i][static_cast<std::size_t>(value)]);
                }
            }
            writer.onArrayEnd();
        }
    };
    m_Frame.readRows(1, 0, m_Frame.numberRows(), writeRowsToJson, &m_RowMask);
    writer.onArrayEnd();
    writer.onObjectEnd();
}
}
}
