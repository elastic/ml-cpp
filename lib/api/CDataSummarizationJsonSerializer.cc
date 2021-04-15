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

#include <api/CInferenceModelDefinition.h>

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

    rapidjson::Document doc;
    doc.Parse(m_Encodings.str());
    rapidjson::ParseResult ok(doc.Parse(m_Encodings.str()));
    if (static_cast<bool>(ok) == false) {
        LOG_ERROR(<< "Failed parsing encoding json. Please report this error.");
    } else {
        writer.Key(JSON_ENCODINGS_TAG);
        writer.write(doc);
    }

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

CRetrainableModelJsonDeserializer::TDataSummarization
CRetrainableModelJsonDeserializer::dataSummarizationFromDocumentCompressed(const TIStreamSPtr& istream) {
    rapidjson::IStreamWrapper isw(*istream);
    rapidjson::Document doc;
    doc.ParseStream(isw);
    if (doc.HasMember(JSON_COMPRESSED_DATA_SUMMARIZATION_TAG) &&
        doc[JSON_COMPRESSED_DATA_SUMMARIZATION_TAG].IsObject()) {
        auto& compressedDataSummarization = doc[JSON_COMPRESSED_DATA_SUMMARIZATION_TAG];
        if (compressedDataSummarization.HasMember(JSON_DATA_SUMMARIZATION_TAG) &&
            compressedDataSummarization[JSON_DATA_SUMMARIZATION_TAG].IsString()) {
            std::stringstream compressedStream{
                compressedDataSummarization[JSON_DATA_SUMMARIZATION_TAG].GetString()};
            auto decompressedSPtr =
                std::make_shared<std::stringstream>(decompressStream(std::stringstream(
                    compressedDataSummarization[JSON_DATA_SUMMARIZATION_TAG].GetString())));
            return CRetrainableModelJsonDeserializer::dataSummarizationFromJsonStream(decompressedSPtr);
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

CRetrainableModelJsonDeserializer::TDataSummarization
CRetrainableModelJsonDeserializer::dataSummarizationFromJsonStream(const TIStreamSPtr& istream) {
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
    rapidjson::Document doc;
    doc.ParseStream(isw);
    if (doc.HasMember(JSON_NUM_COLUMNS_TAG) && doc[JSON_NUM_COLUMNS_TAG].IsUint64()) {
        numColumns = doc[JSON_NUM_COLUMNS_TAG].GetUint64();
        columnNames.reserve(numColumns);
        columnIsCategorical.reserve(numColumns);
        categoricalColumnValues.resize(numColumns, {});
    } else {
        LOG_ERROR(<< "Data summarization field '" << JSON_NUM_COLUMNS_TAG
                  << "'  is missing or has an unexpected format.");
        return {nullptr, nullptr};
    }

    if (doc.HasMember(JSON_COLUMN_NAMES_TAG) && doc[JSON_COLUMN_NAMES_TAG].IsArray()) {
        for (auto& item : doc[JSON_COLUMN_NAMES_TAG].GetArray()) {
            columnNames.push_back(item.GetString());
        }
    } else {
        LOG_ERROR(<< "Data summarization field '" << JSON_COLUMN_NAMES_TAG
                  << "'  is missing or has an unexpected format.");
        return {nullptr, nullptr};
    }

    if (doc.HasMember(JSON_COLUMN_IS_CATEGORICAL_TAG) &&
        doc[JSON_COLUMN_IS_CATEGORICAL_TAG].IsArray()) {
        for (auto& item : doc[JSON_COLUMN_IS_CATEGORICAL_TAG].GetArray()) {
            columnIsCategorical.push_back(item.GetBool());
        }
    } else {
        LOG_ERROR(<< "Data summarization field '" << JSON_COLUMN_IS_CATEGORICAL_TAG
                  << "'  is missing or has an unexpected format.");
        return {nullptr, nullptr};
    }
    if (doc.HasMember(JSON_CATEGORICAL_COLUMN_VALUES_TAG) &&
        doc[JSON_CATEGORICAL_COLUMN_VALUES_TAG].IsArray()) {
        std::size_t i{0};
        for (auto& item : doc[JSON_CATEGORICAL_COLUMN_VALUES_TAG].GetArray()) {
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

    if (doc.HasMember(JSON_ENCODINGS_TAG) && doc[JSON_ENCODINGS_TAG].IsObject()) {
        std::stringstream jsonStrm;
        rapidjson::OStreamWrapper wrapper{jsonStrm};
        CDataSummarizationJsonSerializer::TGenericLineWriter writer{wrapper};
        writer.StartObject();
        writer.Key(JSON_ENCODINGS_TAG);
        writer.write(doc[JSON_ENCODINGS_TAG].GetObject());
        writer.EndObject();
        core::CJsonStateRestoreTraverser traverser(jsonStrm);
        encoder = std::make_unique<maths::CDataFrameCategoryEncoder>(traverser);
    } else {
        LOG_ERROR(<< "Data summarization field '" << JSON_ENCODINGS_TAG
                  << "'  is missing or has an unexpected format.");
        return {nullptr, nullptr};
    }

    if (doc.HasMember(JSON_DATA_TAG) && doc[JSON_DATA_TAG].IsArray()) {
        frame = core::makeMainStorageDataFrame(numColumns).first;
        frame->columnNames(columnNames);
        frame->categoricalColumns(columnIsCategorical);
        frame->categoricalColumnValues(categoricalColumnValues);
        TStrVec rowVec;
        rowVec.reserve(numColumns);

        for (auto& row : doc[JSON_DATA_TAG].GetArray()) {
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

CRetrainableModelJsonDeserializer::TBestForest
CRetrainableModelJsonDeserializer::bestForestFromJsonStream(const core::CDataSearcher::TIStreamP& istream) {
    using TNodeVec = maths::CBoostedTreeFactory::TNodeVec;
    using TNodeVecVec = maths::CBoostedTreeFactory::TNodeVecVec;
    rapidjson::IStreamWrapper isw(*istream);
    rapidjson::Document d;
    d.ParseStream(isw);
    if (d.HasMember(CInferenceModelDefinition::JSON_TRAINED_MODEL_TAG) &&
        d[CInferenceModelDefinition::JSON_TRAINED_MODEL_TAG].IsObject()) {
        auto trainedModel = d[CInferenceModelDefinition::JSON_TRAINED_MODEL_TAG].GetObject();
        if (trainedModel.HasMember(CEnsemble::JSON_ENSEMBLE_TAG) &&
            trainedModel[CEnsemble::JSON_ENSEMBLE_TAG].IsObject()) {
            auto ensemble = trainedModel[CEnsemble::JSON_ENSEMBLE_TAG].GetObject();
            if (ensemble.HasMember(CEnsemble::JSON_TRAINED_MODELS_TAG) &&
                ensemble[CEnsemble::JSON_TRAINED_MODELS_TAG].IsArray()) {
                auto trainedModels =
                    ensemble[CEnsemble::JSON_TRAINED_MODELS_TAG].GetArray();
                auto forest = std::make_unique<TNodeVecVec>();
                for (auto& tree : trainedModels) {
                    TNodeVec nodes;
                    nodes.emplace_back(); // add root
                    for (auto& node : tree[CTree::JSON_TREE_TAG][CTree::JSON_TREE_STRUCTURE_TAG]
                                          .GetArray()) {
                        std::size_t nodeIndex{
                            node[CTree::CTreeNode::JSON_NODE_INDEX_TAG].GetUint64()};
                        std::size_t numberSamples{
                            node[CTree::CTreeNode::JSON_NUMBER_SAMPLES_TAG].GetUint64()};
                        if (node.HasMember(CTree::CTreeNode::JSON_LEAF_VALUE_TAG)) {
                            // TODO #1851 this can be/is a vector
                            maths::CBoostedTreeNode::TVector nodeValue(1);
                            nodeValue[0] =
                                node[CTree::CTreeNode::JSON_LEAF_VALUE_TAG].GetDouble();
                            nodes[nodeIndex].numberSamples(numberSamples);
                            nodes[nodeIndex].nodeValue({nodeValue});

                        } else {
                            // TODO #1852 identify correct split feature;
                            std::size_t splitFeature{0};
                            double gain{
                                node[CTree::CTreeNode::JSON_SPLIT_GAIN_TAG].GetDouble()};
                            double splitValue{
                                node[CTree::CTreeNode::JSON_THRESHOLD_TAG].GetDouble()};
                            bool assignMissingToLeft{
                                node[CTree::CTreeNode::JSON_DEFAULT_LEFT_TAG].GetBool()};
                            nodes[nodeIndex].split(splitFeature, splitValue,
                                                   assignMissingToLeft, gain, 0.0, nodes);
                            nodes[nodeIndex].numberSamples(numberSamples);
                        }
                    }
                    forest->push_back(nodes);
                }
                return forest;
            }
        }
    }
    return nullptr;
}

CRetrainableModelJsonDeserializer::TBestForest
CRetrainableModelJsonDeserializer::bestForestFromDocumentCompressed(
    const core::CDataSearcher::TIStreamP& istream) {
    rapidjson::IStreamWrapper isw(*istream);
    rapidjson::Document d;
    d.ParseStream(isw);
    if (d.HasMember(CInferenceModelDefinition::JSON_COMPRESSED_INFERENCE_MODEL_TAG) &&
        d[CInferenceModelDefinition::JSON_COMPRESSED_INFERENCE_MODEL_TAG].IsObject()) {
        auto& compressedDataSummarization =
            d[CInferenceModelDefinition::JSON_COMPRESSED_INFERENCE_MODEL_TAG];
        if (compressedDataSummarization.HasMember(CInferenceModelDefinition::JSON_DEFINITION_TAG) &&
            compressedDataSummarization[CInferenceModelDefinition::JSON_DEFINITION_TAG]
                .IsString()) {
            std::stringstream compressedStream{
                compressedDataSummarization[CInferenceModelDefinition::JSON_DEFINITION_TAG]
                    .GetString()};
            auto decompressedSPtr = std::make_shared<std::stringstream>(decompressStream(
                std::stringstream(compressedDataSummarization[CInferenceModelDefinition::JSON_DEFINITION_TAG]
                                      .GetString())));
            return CRetrainableModelJsonDeserializer::bestForestFromJsonStream(decompressedSPtr);
        } else {
            LOG_ERROR(<< "Field " << CInferenceModelDefinition::JSON_DEFINITION_TAG
                      << " not found or is not a string.");
        }
    } else {
        LOG_ERROR(<< "Field " << CInferenceModelDefinition::JSON_COMPRESSED_INFERENCE_MODEL_TAG
                  << " not found or is not an object.");
    }
    return nullptr;
}
}
}
