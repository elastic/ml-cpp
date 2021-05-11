/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataSummarizationJsonSerializer.h>

#include <core/CBase64Filter.h>
#include <core/CDataFrame.h>
#include <core/CJsonStatePersistInserter.h>
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
#include <stdexcept>
#include <string>
#include <utility>

namespace ml {
namespace api {

namespace {
using TBoolVec = std::vector<bool>;
using TDoubleVec = std::vector<double>;
using TStrVec = std::vector<std::string>;
using TStrVecVec = std::vector<TStrVec>;
using TRowItr = core::CDataFrame::TRowItr;
using TFilteredInput = boost::iostreams::filtering_stream<boost::iostreams::input>;
using Device = boost::iostreams::basic_array_source<char>;
using TStreamBuffer = boost::iostreams::stream_buffer<Device>;

template<typename GET, typename VALUE>
auto ifExists(const std::string& tag, const GET& get, const VALUE& value)
    -> decltype(get(value[tag])) {
    if (value.HasMember(tag)) {
        try {
            return get(value[tag]);
        } catch (const std::runtime_error& e) {
            throw std::runtime_error("Field '" + tag + "' " + e.what() + ".");
        }
    }
    throw std::runtime_error{"Field '" + tag + "' is missing."};
}

auto getObject(const rapidjson::Value& value) {
    if (value.IsObject() == false) {
        throw std::runtime_error{"is not an object"};
    }
    return value.GetObject();
}

auto getArray(const rapidjson::Value& value) {
    if (value.IsArray() == false) {
        throw std::runtime_error{"is not an array"};
    }
    return value.GetArray();
}

bool getBool(const rapidjson::Value& value) {
    if (value.IsBool() == false) {
        throw std::runtime_error{"is not a bool"};
    }
    return value.GetBool();
}

std::uint64_t getUint64(const rapidjson::Value& value) {
    if (value.IsUint64() == false) {
        throw std::runtime_error{"is not a uint64"};
    }
    return value.GetUint64();
}

double getDouble(const rapidjson::Value& value) {
    if (value.IsDouble() == false) {
        throw std::runtime_error{"is not a double"};
    }
    return value.GetDouble();
}

auto getString(const rapidjson::Value& value) {
    if (value.IsString() == false) {
        throw std::runtime_error{"is not a string"};
    }
    return value.GetString();
}

std::size_t getStringLength(const rapidjson::Value& value) {
    if (value.IsString() == false) {
        throw std::runtime_error{"is not a string"};
    }
    return value.GetStringLength();
}

auto decompressStream(boost::iostreams::stream_buffer<Device>& buffer) {
    auto result = std::make_shared<TFilteredInput>();
    result->push(boost::iostreams::gzip_decompressor());
    result->push(core::CBase64Decoder());
    result->push(buffer);
    return result;
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

CDataSummarizationJsonWriter::CDataSummarizationJsonWriter(const core::CDataFrame& frame,
                                                           core::CPackedBitVector rowMask,
                                                           std::size_t numberColumns,
                                                           std::stringstream encodings)
    : m_RowMask{std::move(rowMask)}, m_NumberColumns{numberColumns}, m_Frame{frame},
      m_Encodings{std::move(encodings)} {
}

void CDataSummarizationJsonWriter::addToDocumentCompressed(TRapidJsonWriter& writer) const {
    this->CSerializableToJsonDocumentCompressed::addToDocumentCompressed(
        writer, JSON_COMPRESSED_DATA_SUMMARIZATION_TAG, JSON_DATA_SUMMARIZATION_TAG);
}

std::string CDataSummarizationJsonWriter::jsonString() const {
    std::ostringstream jsonStrm;
    this->jsonStream(jsonStrm);
    return jsonStrm.str();
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

CRetrainableModelJsonReader::TEncoderUPtr
CRetrainableModelJsonReader::dataSummarizationFromJsonStream(TIStreamSPtr istream,
                                                             core::CDataFrame& frame) {
    if (istream == nullptr) {
        return nullptr;
    }
    try {
        return dataSummarizationFromJson(*istream, frame);
    } catch (const std::runtime_error& e) { LOG_ERROR(<< e.what()); }
    return nullptr;
}

CRetrainableModelJsonReader::TEncoderUPtr
CRetrainableModelJsonReader::dataSummarizationFromJson(std::istream& istream,
                                                       core::CDataFrame& frame) {
    rapidjson::IStreamWrapper isw(istream);
    rapidjson::Document doc;
    doc.ParseStream(isw);

    std::size_t numberColumns{ifExists(JSON_NUM_COLUMNS_TAG, getUint64, doc)};

    TStrVec columnNames;
    TBoolVec columnIsCategorical;
    TStrVecVec categoricalColumnValues;
    columnNames.reserve(numberColumns);
    columnIsCategorical.reserve(numberColumns);
    categoricalColumnValues.resize(numberColumns, {});

    for (const auto& name : ifExists(JSON_COLUMN_NAMES_TAG, getArray, doc)) {
        columnNames.push_back(getString(name));
    }

    for (const auto& isCategorical :
         ifExists(JSON_COLUMN_IS_CATEGORICAL_TAG, getArray, doc)) {
        columnIsCategorical.push_back(getBool(isCategorical));
    }

    std::size_t i{0};
    for (const auto& categories :
         ifExists(JSON_CATEGORICAL_COLUMN_VALUES_TAG, getArray, doc)) {
        for (const auto& category : getArray(categories)) {
            categoricalColumnValues[i].push_back(getString(category));
        }
        ++i;
    }

    ifExists(JSON_ENCODINGS_TAG, getObject, doc); // Validate the nested object exists.
    TEncoderUPtr encodings;
    std::stringstream jsonStrm;
    rapidjson::OStreamWrapper wrapper{jsonStrm};
    CDataSummarizationJsonWriter::TGenericLineWriter writer{wrapper};
    writer.StartObject();
    writer.Key(JSON_ENCODINGS_TAG);
    writer.write(doc[JSON_ENCODINGS_TAG]);
    writer.EndObject();
    core::CJsonStateRestoreTraverser traverser{jsonStrm};
    encodings = std::make_unique<maths::CDataFrameCategoryEncoder>(traverser);

    frame.columnNames(columnNames);
    frame.categoricalColumns(columnIsCategorical);
    frame.categoricalColumnValues(categoricalColumnValues);

    TStrVec rowVec;
    rowVec.reserve(numberColumns);
    for (const auto& row : ifExists(JSON_DATA_TAG, getArray, doc)) {
        for (const auto& column : getArray(row)) {
            rowVec.emplace_back(getString(column));
        }
        frame.parseAndWriteRow(
            core::CVectorRange<const TStrVec>(rowVec, 0, rowVec.size()));
        rowVec.clear();
    }
    frame.finishWritingRows();

    return encodings;
}

CRetrainableModelJsonReader::TEncoderUPtr
CRetrainableModelJsonReader::dataSummarizationFromDocumentCompressed(TIStreamSPtr istream,
                                                                     core::CDataFrame& frame) {
    if (istream == nullptr) {
        return nullptr;
    }
    try {
        rapidjson::IStreamWrapper isw{*istream};
        rapidjson::Document doc;
        doc.ParseStream(isw);
        auto compressedDataSummarization =
            ifExists(JSON_COMPRESSED_DATA_SUMMARIZATION_TAG, getObject, doc);
        TStreamBuffer buffer{ifExists(JSON_DATA_SUMMARIZATION_TAG, getString,
                                      compressedDataSummarization),
                             ifExists(JSON_DATA_SUMMARIZATION_TAG, getStringLength,
                                      compressedDataSummarization)};
        return dataSummarizationFromJsonStream(decompressStream(buffer), frame);
    } catch (const std::runtime_error& e) { LOG_ERROR(<< e.what()); }
    return nullptr;
}

CRetrainableModelJsonReader::TNodeVecVecUPtr
CRetrainableModelJsonReader::bestForestFromJsonStream(TIStreamSPtr istream) {
    if (istream == nullptr) {
        return nullptr;
    }
    try {
        return bestForestFromJson(*istream);
    } catch (const std::runtime_error& e) { LOG_ERROR(<< e.what()); }
    return nullptr;
}

CRetrainableModelJsonReader::TNodeVecVecUPtr
CRetrainableModelJsonReader::bestForestFromJson(std::istream& istream) {
    using TNodeVec = maths::CBoostedTreeFactory::TNodeVec;
    using TNodeVecVec = maths::CBoostedTreeFactory::TNodeVecVec;

    rapidjson::IStreamWrapper isw{istream};
    rapidjson::Document doc;
    doc.ParseStream(isw);
    auto inferenceModel =
        ifExists(CInferenceModelDefinition::JSON_TRAINED_MODEL_TAG, getObject, doc);
    auto ensemble = ifExists(CEnsemble::JSON_ENSEMBLE_TAG, getObject, inferenceModel);
    auto trainedModels = ifExists(CEnsemble::JSON_TRAINED_MODELS_TAG, getArray, ensemble);
    auto forest = std::make_unique<TNodeVecVec>();
    forest->reserve(trainedModels.Size());
    TStrVec featureNames;
    TNodeVec nodes;
    for (const auto& trainedModel : trainedModels) {
        auto tree = ifExists(CTree::JSON_TREE_TAG, getObject, trainedModel);
        featureNames.clear();
        for (const auto& name : ifExists(CTree::JSON_FEATURE_NAMES_TAG, getArray, tree)) {
            featureNames.emplace_back(getString(name));
        }
        auto treeNodes = ifExists(CTree::JSON_TREE_STRUCTURE_TAG, getArray, tree);
        nodes.clear();
        nodes.reserve(treeNodes.Size());
        nodes.emplace_back(); // Add the root.
        for (const auto& node : treeNodes) {
            std::size_t nodeIndex{
                ifExists(CTree::CTreeNode::JSON_NODE_INDEX_TAG, getUint64, node)};
            std::size_t numberSamples{ifExists(
                CTree::CTreeNode::JSON_NUMBER_SAMPLES_TAG, getUint64, node)};
            nodes[nodeIndex].numberSamples(numberSamples);
            if (node.HasMember(CTree::CTreeNode::JSON_LEAF_VALUE_TAG)) {
                // Add a leaf node.
                if (node[CTree::CTreeNode::JSON_LEAF_VALUE_TAG].IsArray()) {
                    auto leafValueArray =
                        getArray(node[CTree::CTreeNode::JSON_LEAF_VALUE_TAG]);
                    maths::CBoostedTreeNode::TVector nodeValue(leafValueArray.Size());
                    for (rapidjson::SizeType i = 0; i < leafValueArray.Size(); ++i) {
                        nodeValue[static_cast<long>(i)] = getDouble(leafValueArray[i]);
                    }
                    nodes[nodeIndex].value(nodeValue);
                } else {
                    maths::CBoostedTreeNode::TVector nodeValue(1);
                    nodeValue[0] = ifExists(CTree::CTreeNode::JSON_LEAF_VALUE_TAG,
                                            getDouble, node);
                    nodes[nodeIndex].value(nodeValue);
                }
            } else {
                // Add a split node.
                std::size_t splitFeature{ifExists(
                    CTree::CTreeNode::JSON_SPLIT_FEATURE_TAG, getUint64, node)};
                double gain{ifExists(CTree::CTreeNode::JSON_SPLIT_GAIN_TAG, getDouble, node)};
                double splitValue{ifExists(CTree::CTreeNode::JSON_THRESHOLD_TAG,
                                           getDouble, node)};
                bool assignMissingToLeft{ifExists(
                    CTree::CTreeNode::JSON_DEFAULT_LEFT_TAG, getBool, node)};
                std::size_t leftChildIndex{ifExists(
                    CTree::CTreeNode::JSON_LEFT_CHILD_TAG, getUint64, node)};
                std::size_t rightChildIndex{ifExists(
                    CTree::CTreeNode::JSON_RIGHT_CHILD_TAG, getUint64, node)};
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

CRetrainableModelJsonReader::TNodeVecVecUPtr
CRetrainableModelJsonReader::bestForestFromDocumentCompressed(TIStreamSPtr istream) {
    if (istream == nullptr) {
        return nullptr;
    }
    try {
        rapidjson::IStreamWrapper isw{*istream};
        rapidjson::Document doc;
        doc.ParseStream(isw);
        auto compressedDataSummarization = ifExists(
            CInferenceModelDefinition::JSON_COMPRESSED_INFERENCE_MODEL_TAG, getObject, doc);
        TStreamBuffer buffer{ifExists(CInferenceModelDefinition::JSON_DEFINITION_TAG,
                                      getString, compressedDataSummarization),
                             ifExists(CInferenceModelDefinition::JSON_DEFINITION_TAG,
                                      getStringLength, compressedDataSummarization)};
        return bestForestFromJsonStream(decompressStream(buffer));
    } catch (const std::exception& e) { LOG_ERROR(<< e.what()); }
    return nullptr;
}
}
}
