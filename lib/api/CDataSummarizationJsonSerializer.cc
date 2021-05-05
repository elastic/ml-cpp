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

#include <maths/CBoostedTree.h>
#include <maths/CDataFrameCategoryEncoder.h>

#include <api/CInferenceModelDefinition.h>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/rapidjson.h>

#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/device/array.hpp>
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
using Device = boost::iostreams::basic_array_source<char>;
using TStreamBuffer = boost::iostreams::stream_buffer<Device>;
using TStrVec = std::vector<std::string>;

using TUint64Optional = boost::optional<std::uint64_t>;

std::uint64_t getUint64(const rapidjson::Value& element,
                        const std::string& tag,
                        std::uint64_t fallback = 0) {
    if (element.HasMember(tag) == false || element[tag].IsUint64() == false) {
        LOG_ERROR(<< "Field '" << tag << "' is missing or has incorrect type. Using default value "
                  << fallback << " instead.");
        return fallback;
    }
    return element[tag].GetUint64();
}

double getDouble(const rapidjson::Value& element, const std::string& tag, double fallback = 0.0) {
    if (element.HasMember(tag) == false || element[tag].IsDouble() == false) {
        LOG_ERROR(<< "Field '" << tag << "' is missing or has incorrect type. Using default value "
                  << fallback << " instead.");
        return fallback;
    }
    return element[tag].GetDouble();
}

bool getBool(const rapidjson::Value& element, const std::string& tag, bool fallback = true) {
    if (element.HasMember(tag) == false || element[tag].IsBool() == false) {
        LOG_ERROR(<< "Field '" << tag << "' is missing or has incorrect type. Using default value "
                  << fallback << " instead.");
        return fallback;
    }
    return element[tag].GetBool();
}

auto decompressStream(boost::iostreams::stream_buffer<Device>& buffer) {
    auto decompressedStream = std::make_shared<std::stringstream>();
    {
        TFilteredInput inFilter;
        inFilter.push(boost::iostreams::gzip_decompressor());
        inFilter.push(core::CBase64Decoder());
        inFilter.push(buffer);
        boost::iostreams::copy(inFilter, *decompressedStream);
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
                                                                   std::size_t numberColumns,
                                                                   std::stringstream encodings)
    : m_RowMask{std::move(rowMask)}, m_NumberColumns{numberColumns}, m_Frame{frame},
      m_Encodings{std::move(encodings)} {
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

CRetrainableModelJsonDeserializer::TEncoderUPtr
CRetrainableModelJsonDeserializer::dataSummarizationFromDocumentCompressed(TIStreamSPtr istream,
                                                                           core::CDataFrame& frame) {
    rapidjson::IStreamWrapper isw{*istream};
    rapidjson::Document doc;
    doc.ParseStream(isw);
    if (doc.HasMember(JSON_COMPRESSED_DATA_SUMMARIZATION_TAG) == false ||
        doc[JSON_COMPRESSED_DATA_SUMMARIZATION_TAG].IsObject() == false) {
        LOG_ERROR(<< "Field " << JSON_COMPRESSED_DATA_SUMMARIZATION_TAG
                  << " not found or is not an object.");
        return nullptr;
    }
    auto& compressedDataSummarization = doc[JSON_COMPRESSED_DATA_SUMMARIZATION_TAG];
    if (compressedDataSummarization.HasMember(JSON_DATA_SUMMARIZATION_TAG) == false ||
        compressedDataSummarization[JSON_DATA_SUMMARIZATION_TAG].IsString() == false) {
        LOG_ERROR(<< "Field " << JSON_DATA_SUMMARIZATION_TAG << " not found or is not a string.");
        return nullptr;
    }
    TStreamBuffer buffer{
        compressedDataSummarization[JSON_DATA_SUMMARIZATION_TAG].GetString(),
        compressedDataSummarization[JSON_DATA_SUMMARIZATION_TAG].GetStringLength()};
    return dataSummarizationFromJsonStream(decompressStream(buffer), frame);
}

CRetrainableModelJsonDeserializer::TEncoderUPtr
CRetrainableModelJsonDeserializer::dataSummarizationFromJsonStream(TIStreamSPtr istream,
                                                                   core::CDataFrame& frame) {
    using TStrVecVec = std::vector<TStrVec>;
    using TBoolVec = std::vector<bool>;

    std::size_t numColumns;
    TStrVec columnNames;
    TStrVecVec categoricalColumnValues;
    TBoolVec columnIsCategorical;
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
                  << "' is missing or has an unexpected format.");
        return nullptr;
    }

    if (doc.HasMember(JSON_COLUMN_NAMES_TAG) && doc[JSON_COLUMN_NAMES_TAG].IsArray()) {
        for (const auto& item : doc[JSON_COLUMN_NAMES_TAG].GetArray()) {
            columnNames.push_back(item.GetString());
        }
    } else {
        LOG_ERROR(<< "Data summarization field '" << JSON_COLUMN_NAMES_TAG
                  << "' is missing or has an unexpected format.");
        return nullptr;
    }

    if (doc.HasMember(JSON_COLUMN_IS_CATEGORICAL_TAG) &&
        doc[JSON_COLUMN_IS_CATEGORICAL_TAG].IsArray()) {
        for (const auto& item : doc[JSON_COLUMN_IS_CATEGORICAL_TAG].GetArray()) {
            columnIsCategorical.push_back(item.GetBool());
        }
    } else {
        LOG_ERROR(<< "Data summarization field '" << JSON_COLUMN_IS_CATEGORICAL_TAG
                  << "' is missing or has an unexpected format.");
        return nullptr;
    }
    if (doc.HasMember(JSON_CATEGORICAL_COLUMN_VALUES_TAG) &&
        doc[JSON_CATEGORICAL_COLUMN_VALUES_TAG].IsArray()) {
        std::size_t i{0};
        for (const auto& item : doc[JSON_CATEGORICAL_COLUMN_VALUES_TAG].GetArray()) {
            for (const auto& categoricalValue : item.GetArray()) {
                categoricalColumnValues[i].push_back(categoricalValue.GetString());
            }
            ++i;
        }
    } else {
        LOG_ERROR(<< "Data summarization field '" << JSON_CATEGORICAL_COLUMN_VALUES_TAG
                  << "' is missing or has an unexpected format.");
        return nullptr;
    }

    if (doc.HasMember(JSON_ENCODINGS_TAG) && doc[JSON_ENCODINGS_TAG].IsObject()) {
        std::stringstream jsonStrm;
        rapidjson::OStreamWrapper wrapper{jsonStrm};
        CDataSummarizationJsonSerializer::TGenericLineWriter writer{wrapper};
        writer.StartObject();
        writer.Key(JSON_ENCODINGS_TAG);
        writer.write(doc[JSON_ENCODINGS_TAG].GetObject());
        writer.EndObject();
        core::CJsonStateRestoreTraverser traverser{jsonStrm};
        encoder = std::make_unique<maths::CDataFrameCategoryEncoder>(traverser);
    } else {
        LOG_ERROR(<< "Data summarization field '" << JSON_ENCODINGS_TAG
                  << "' is missing or has an unexpected format.");
        return nullptr;
    }

    if (doc.HasMember(JSON_DATA_TAG) && doc[JSON_DATA_TAG].IsArray()) {
        frame.columnNames(columnNames);
        frame.categoricalColumns(columnIsCategorical);
        frame.categoricalColumnValues(categoricalColumnValues);
        TStrVec rowVec;
        rowVec.reserve(numColumns);

        for (const auto& row : doc[JSON_DATA_TAG].GetArray()) {
            for (const auto& item : row.GetArray()) {
                rowVec.push_back(item.GetString());
            }
            frame.parseAndWriteRow(
                core::CVectorRange<const TStrVec>(rowVec, 0, rowVec.size()));
            rowVec.clear();
        }
        frame.finishWritingRows();
    } else {
        LOG_ERROR(<< "Data summarization field '" << JSON_DATA_TAG
                  << "' is missing or has an unexpected format.");
        return nullptr;
    }
    return encoder;
}

CRetrainableModelJsonDeserializer::TNodeVecVecUPtr
CRetrainableModelJsonDeserializer::bestForestFromJsonStream(const core::CDataSearcher::TIStreamP& istream) {
    using TNodeVec = maths::CBoostedTreeFactory::TNodeVec;
    using TNodeVecVec = maths::CBoostedTreeFactory::TNodeVecVec;
    rapidjson::IStreamWrapper isw{*istream};
    rapidjson::Document doc;
    doc.ParseStream(isw);
    if (doc.HasMember(CInferenceModelDefinition::JSON_TRAINED_MODEL_TAG) == false ||
        doc[CInferenceModelDefinition::JSON_TRAINED_MODEL_TAG].IsObject() == false) {
        LOG_ERROR(<< "Object '" << CInferenceModelDefinition::JSON_TRAINED_MODEL_TAG
                  << "' is missing in the model definition.");
        return nullptr;
    }
    auto trainedModel = doc[CInferenceModelDefinition::JSON_TRAINED_MODEL_TAG].GetObject();
    if (trainedModel.HasMember(CEnsemble::JSON_ENSEMBLE_TAG) == false ||
        trainedModel[CEnsemble::JSON_ENSEMBLE_TAG].IsObject() == false) {
        LOG_ERROR(<< "Object '" << CEnsemble::JSON_ENSEMBLE_TAG
                  << "' is missing in the model definition.");
        return nullptr;
    }
    auto ensemble = trainedModel[CEnsemble::JSON_ENSEMBLE_TAG].GetObject();
    if (ensemble.HasMember(CEnsemble::JSON_TRAINED_MODELS_TAG) == false ||
        ensemble[CEnsemble::JSON_TRAINED_MODELS_TAG].IsArray() == false) {
        LOG_ERROR(<< "Array '" << CEnsemble::JSON_TRAINED_MODELS_TAG
                  << "' is missing in the model definition.");
        return nullptr;
    }
    auto trainedModels = ensemble[CEnsemble::JSON_TRAINED_MODELS_TAG].GetArray();
    auto forest = std::make_unique<TNodeVecVec>();
    forest->reserve(trainedModels.Size());
    TNodeVec nodes;
    for (const auto& tree : trainedModels) {
        if (tree.HasMember(CTree::JSON_TREE_TAG) == false ||
            tree[CTree::JSON_TREE_TAG].IsObject() == false ||
            tree[CTree::JSON_TREE_TAG].HasMember(CTree::JSON_TREE_STRUCTURE_TAG) == false ||
            tree[CTree::JSON_TREE_TAG][CTree::JSON_TREE_STRUCTURE_TAG].IsArray() == false) {
            LOG_ERROR(<< "Array '" << CTree::JSON_TREE_TAG << "/" << CTree::JSON_TREE_STRUCTURE_TAG
                      << "' is missing in the model definition.");
            return nullptr;
        }
        auto treeArray =
            tree[CTree::JSON_TREE_TAG][CTree::JSON_TREE_STRUCTURE_TAG].GetArray();
        nodes.clear();
        nodes.reserve(treeArray.Size());
        nodes.emplace_back(); // add root
        for (const auto& node : treeArray) {
            std::size_t nodeIndex{getUint64(node, CTree::CTreeNode::JSON_NODE_INDEX_TAG)};
            std::size_t numberSamples{getUint64(node, CTree::CTreeNode::JSON_NUMBER_SAMPLES_TAG)};
            nodes[nodeIndex].numberSamples(numberSamples);

            if (node.HasMember(CTree::CTreeNode::JSON_LEAF_VALUE_TAG)) {
                // leaf node
                if (node[CTree::CTreeNode::JSON_LEAF_VALUE_TAG].IsArray()) {
                    auto leafValueArray =
                        node[CTree::CTreeNode::JSON_LEAF_VALUE_TAG].GetArray();
                    maths::CBoostedTreeNode::TVector nodeValue(leafValueArray.Size());
                    for (rapidjson::SizeType i = 0; i < leafValueArray.Size(); ++i) {
                        nodeValue[static_cast<long>(i)] = leafValueArray[i].GetDouble();
                    }
                    nodes[nodeIndex].value({nodeValue});
                } else {
                    maths::CBoostedTreeNode::TVector nodeValue(1);
                    nodeValue[0] = getDouble(node, CTree::CTreeNode::JSON_LEAF_VALUE_TAG);
                    nodes[nodeIndex].value({nodeValue});
                }

            } else {
                // inner node
                std::size_t splitFeature{getUint64(node, CTree::CTreeNode::JSON_SPLIT_FEATURE_TAG)};
                double gain{getDouble(node, CTree::CTreeNode::JSON_SPLIT_GAIN_TAG)};
                double splitValue{getDouble(node, CTree::CTreeNode::JSON_THRESHOLD_TAG)};
                bool assignMissingToLeft{getBool(node, CTree::CTreeNode::JSON_DEFAULT_LEFT_TAG)};
                std::size_t leftChildIndex{getUint64(node, CTree::CTreeNode::JSON_LEFT_CHILD_TAG)};
                std::size_t rightChildIndex{
                    getUint64(node, CTree::CTreeNode::JSON_RIGHT_CHILD_TAG)};
                nodes[nodeIndex].split(splitFeature, splitValue,
                                       assignMissingToLeft, gain, 0.0, nodes);
                nodes[nodeIndex].numberSamples(numberSamples);
                nodes[nodeIndex].leftChildIndex(
                    static_cast<maths::CBoostedTreeNode::TNodeIndex>(leftChildIndex));
                nodes[nodeIndex].rightChildIndex(
                    static_cast<maths::CBoostedTreeNode::TNodeIndex>(rightChildIndex));
            }
        }
        forest->push_back(nodes);
    }
    return forest;
}

CRetrainableModelJsonDeserializer::TNodeVecVecUPtr
CRetrainableModelJsonDeserializer::bestForestFromDocumentCompressed(
    const core::CDataSearcher::TIStreamP& istream) {
    rapidjson::IStreamWrapper isw{*istream};
    rapidjson::Document doc;
    doc.ParseStream(isw);
    if (doc.HasMember(CInferenceModelDefinition::JSON_COMPRESSED_INFERENCE_MODEL_TAG) == false ||
        doc[CInferenceModelDefinition::JSON_COMPRESSED_INFERENCE_MODEL_TAG].IsObject() == false) {
        LOG_ERROR(<< "Field " << CInferenceModelDefinition::JSON_COMPRESSED_INFERENCE_MODEL_TAG
                  << " not found or is not an object.");
        return nullptr;
    }
    auto& compressedDataSummarization =
        doc[CInferenceModelDefinition::JSON_COMPRESSED_INFERENCE_MODEL_TAG];
    if (compressedDataSummarization.HasMember(CInferenceModelDefinition::JSON_DEFINITION_TAG) == false ||
        compressedDataSummarization[CInferenceModelDefinition::JSON_DEFINITION_TAG]
                .IsString() == false) {
        LOG_ERROR(<< "Field " << CInferenceModelDefinition::JSON_DEFINITION_TAG
                  << " not found or is not a string.");
        return nullptr;
    }
    TStreamBuffer buffer{compressedDataSummarization[CInferenceModelDefinition::JSON_DEFINITION_TAG]
                             .GetString(),
                         compressedDataSummarization[CInferenceModelDefinition::JSON_DEFINITION_TAG]
                             .GetStringLength()};
    return bestForestFromJsonStream(decompressStream(buffer));
}
}
}
