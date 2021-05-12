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
using TStrSizeUMap = CRetrainableModelJsonReader::TStrSizeUMap;

class CEncoderNameIndexMapBuilder final : public maths::CDataFrameCategoryEncoder::CVisitor {
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

std::size_t lookup(const TStrSizeUMap& encodingsIndices,
                   const TStrVec& featureNames,
                   std::size_t featureIndex) {
    if (featureIndex > featureNames.size()) {
        throw std::runtime_error{"Feature name index '" +
                                 std::to_string(featureIndex) + "' out of bounds '" +
                                 std::to_string(featureNames.size()) + "'."};
    }
    auto encodingIndex = encodingsIndices.find(featureNames[featureIndex]);
    if (encodingIndex == encodingsIndices.end()) {
        throw std::runtime_error{"No encoding index for '" + featureNames[featureIndex] + "'."};
    }
    return encodingIndex->second;
}

// clang-format off
const std::string JSON_COMPRESSED_DATA_SUMMARIZATION_TAG{"compressed_data_summarization"};
const std::string JSON_DATA_SUMMARIZATION_TAG{"data_summarization"};
const std::string JSON_NUM_COLUMNS_TAG{"num_columns"};
const std::string JSON_COLUMN_NAMES_TAG{"column_names"};
const std::string JSON_COLUMN_IS_CATEGORICAL_TAG{"column_is_categorical"};
const std::string JSON_CATEGORICAL_COLUMN_VALUES_TAG{"categorical_column_values"};
const std::string JSON_ENCODING_NAME_INDEX_MAP_TAG{"encodings_indices"};
const std::string JSON_ENCODING_NAME_INDEX_MAP_KEY_TAG{"encoding_name"};
const std::string JSON_ENCODING_NAME_INDEX_MAP_VALUE_TAG{"encoding_index"};
const std::string JSON_ENCODINGS_TAG{"encodings"};
const std::string JSON_DATA_TAG{"data"};
// clang-format on
}

CDataSummarizationJsonWriter::CDataSummarizationJsonWriter(const core::CDataFrame& frame,
                                                           core::CPackedBitVector rowMask,
                                                           std::size_t numberColumns,
                                                           const maths::CDataFrameCategoryEncoder& encodings)
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

CRetrainableModelJsonReader::TEncoderUPtrStrSizeUMapPr
CRetrainableModelJsonReader::dataSummarizationFromJsonStream(TIStreamSPtr istream,
                                                             core::CDataFrame& frame) {
    if (istream != nullptr) {
        try {
            return doDataSummarizationFromJsonStream(*istream, frame);
        } catch (const std::runtime_error& e) { LOG_ERROR(<< e.what()); }
    }
    return {nullptr, TStrSizeUMap{}};
}

CRetrainableModelJsonReader::TEncoderUPtrStrSizeUMapPr
CRetrainableModelJsonReader::doDataSummarizationFromJsonStream(std::istream& istream,
                                                               core::CDataFrame& frame) {
    rapidjson::IStreamWrapper isw(istream);
    rapidjson::Document doc;
    doc.ParseStream(isw);

    std::size_t numberColumns{ifExists(JSON_NUM_COLUMNS_TAG, getAsUint64From, doc)};

    TStrVec columnNames;
    TBoolVec columnIsCategorical;
    TStrVecVec categoricalColumnValues;
    columnNames.reserve(numberColumns);
    columnIsCategorical.reserve(numberColumns);
    categoricalColumnValues.resize(numberColumns, {});

    for (const auto& name : ifExists(JSON_COLUMN_NAMES_TAG, getAsArrayFrom, doc)) {
        columnNames.push_back(getAsStringFrom(name));
    }

    for (const auto& isCategorical :
         ifExists(JSON_COLUMN_IS_CATEGORICAL_TAG, getAsArrayFrom, doc)) {
        columnIsCategorical.push_back(getAsBoolFrom(isCategorical));
    }

    std::size_t i{0};
    for (const auto& categories :
         ifExists(JSON_CATEGORICAL_COLUMN_VALUES_TAG, getAsArrayFrom, doc)) {
        for (const auto& category : getAsArrayFrom(categories)) {
            categoricalColumnValues[i].push_back(getAsStringFrom(category));
        }
        ++i;
    }

    TStrSizeUMap encodingIndices;
    for (const auto& entry :
         ifExists(JSON_ENCODING_NAME_INDEX_MAP_TAG, getAsArrayFrom, doc)) {
        encodingIndices.emplace(
            ifExists(JSON_ENCODING_NAME_INDEX_MAP_KEY_TAG, getAsStringFrom, entry),
            ifExists(JSON_ENCODING_NAME_INDEX_MAP_VALUE_TAG, getAsUint64From, entry));
    }

    ifExists(JSON_ENCODINGS_TAG, getAsObjectFrom, doc); // Validate the nested object exists.
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
    for (const auto& row : ifExists(JSON_DATA_TAG, getAsArrayFrom, doc)) {
        for (const auto& column : getAsArrayFrom(row)) {
            rowVec.emplace_back(getAsStringFrom(column));
        }
        frame.parseAndWriteRow(
            core::CVectorRange<const TStrVec>(rowVec, 0, rowVec.size()));
        rowVec.clear();
    }
    frame.finishWritingRows();

    return {std::move(encodings), std::move(encodingIndices)};
}

CRetrainableModelJsonReader::TEncoderUPtrStrSizeUMapPr
CRetrainableModelJsonReader::dataSummarizationFromCompressedJsonStream(TIStreamSPtr istream,
                                                                       core::CDataFrame& frame) {
    std::stringstream buffer;
    return dataSummarizationFromJsonStream(
        rawJsonStream(JSON_COMPRESSED_DATA_SUMMARIZATION_TAG,
                      JSON_DATA_SUMMARIZATION_TAG, std::move(istream), buffer),
        frame);
}

CRetrainableModelJsonReader::TNodeVecVecUPtr
CRetrainableModelJsonReader::bestForestFromJsonStream(TIStreamSPtr istream,
                                                      const TStrSizeUMap& encodingIndices) {
    if (istream != nullptr) {
        try {
            return doBestForestFromJsonStream(*istream, encodingIndices);
        } catch (const std::runtime_error& e) { LOG_ERROR(<< e.what()); }
    }
    return nullptr;
}

CRetrainableModelJsonReader::TNodeVecVecUPtr
CRetrainableModelJsonReader::doBestForestFromJsonStream(std::istream& istream,
                                                        const TStrSizeUMap& encodingIndices) {
    using TNodeVec = maths::CBoostedTreeFactory::TNodeVec;
    using TNodeVecVec = maths::CBoostedTreeFactory::TNodeVecVec;

    rapidjson::IStreamWrapper isw{istream};
    rapidjson::Document doc;
    doc.ParseStream(isw);
    auto inferenceModel = ifExists(CInferenceModelDefinition::JSON_TRAINED_MODEL_TAG,
                                   getAsObjectFrom, doc);
    auto ensemble = ifExists(CEnsemble::JSON_ENSEMBLE_TAG, getAsObjectFrom, inferenceModel);
    auto trainedModels = ifExists(CEnsemble::JSON_TRAINED_MODELS_TAG, getAsArrayFrom, ensemble);
    auto forest = std::make_unique<TNodeVecVec>();
    forest->reserve(trainedModels.Size());
    TStrVec featureNames;
    TNodeVec nodes;
    for (const auto& trainedModel : trainedModels) {
        auto tree = ifExists(CTree::JSON_TREE_TAG, getAsObjectFrom, trainedModel);
        featureNames.clear();
        for (const auto& name :
             ifExists(CTree::JSON_FEATURE_NAMES_TAG, getAsArrayFrom, tree)) {
            featureNames.emplace_back(getAsStringFrom(name));
        }
        auto treeNodes = ifExists(CTree::JSON_TREE_STRUCTURE_TAG, getAsArrayFrom, tree);
        nodes.clear();
        nodes.reserve(treeNodes.Size());
        nodes.emplace_back(); // Add the root.
        for (const auto& node : treeNodes) {
            std::size_t nodeIndex{ifExists(CTree::CTreeNode::JSON_NODE_INDEX_TAG,
                                           getAsUint64From, node)};
            std::size_t numberSamples{ifExists(CTree::CTreeNode::JSON_NUMBER_SAMPLES_TAG,
                                               getAsUint64From, node)};
            nodes[nodeIndex].numberSamples(numberSamples);
            if (node.HasMember(CTree::CTreeNode::JSON_LEAF_VALUE_TAG)) {
                // Add a leaf node.
                if (node[CTree::CTreeNode::JSON_LEAF_VALUE_TAG].IsArray()) {
                    auto leafValueArray =
                        getAsArrayFrom(node[CTree::CTreeNode::JSON_LEAF_VALUE_TAG]);
                    maths::CBoostedTreeNode::TVector nodeValue(leafValueArray.Size());
                    for (rapidjson::SizeType i = 0; i < leafValueArray.Size(); ++i) {
                        nodeValue[static_cast<long>(i)] =
                            getAsDoubleFrom(leafValueArray[i]);
                    }
                    nodes[nodeIndex].value(nodeValue);
                } else {
                    maths::CBoostedTreeNode::TVector nodeValue(1);
                    nodeValue[0] = ifExists(CTree::CTreeNode::JSON_LEAF_VALUE_TAG,
                                            getAsDoubleFrom, node);
                    nodes[nodeIndex].value(nodeValue);
                }
            } else {
                // Add a split node.
                std::size_t splitFeature{lookup(encodingIndices, featureNames,
                                                ifExists(CTree::CTreeNode::JSON_SPLIT_FEATURE_TAG,
                                                         getAsUint64From, node))};
                double gain{ifExists(CTree::CTreeNode::JSON_SPLIT_GAIN_TAG,
                                     getAsDoubleFrom, node)};
                double splitValue{ifExists(CTree::CTreeNode::JSON_THRESHOLD_TAG,
                                           getAsDoubleFrom, node)};
                bool assignMissingToLeft{ifExists(CTree::CTreeNode::JSON_DEFAULT_LEFT_TAG,
                                                  getAsBoolFrom, node)};
                std::size_t leftChildIndex{ifExists(CTree::CTreeNode::JSON_LEFT_CHILD_TAG,
                                                    getAsUint64From, node)};
                std::size_t rightChildIndex{ifExists(CTree::CTreeNode::JSON_RIGHT_CHILD_TAG,
                                                     getAsUint64From, node)};
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
CRetrainableModelJsonReader::bestForestFromCompressedJsonStream(TIStreamSPtr istream,
                                                                const TStrSizeUMap& encodingIndices) {
    std::stringstream buffer;
    return bestForestFromJsonStream(
        rawJsonStream(CInferenceModelDefinition::JSON_COMPRESSED_INFERENCE_MODEL_TAG,
                      CInferenceModelDefinition::JSON_DEFINITION_TAG,
                      std::move(istream), buffer),
        encodingIndices);
}
}
}
