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

#include <api/CRetrainableModelJsonReader.h>

#include <core/CBoostJsonUnbufferedIStreamWrapper.h>
#include <core/CDataFrame.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CVectorRange.h>

#include <maths/analytics/CBoostedTree.h>
#include <maths/analytics/CDataFrameCategoryEncoder.h>

#include <api/CDataSummarizationJsonWriter.h>
#include <api/CInferenceModelDefinition.h>

#include <boost/json.hpp>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace json = boost::json;

namespace ml {
namespace api {

namespace {
using TBoolVec = std::vector<bool>;
using TStrVec = std::vector<std::string>;
using TStrVecVec = std::vector<TStrVec>;
using TStrSizeUMap = CRetrainableModelJsonReader::TStrSizeUMap;

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
//    core::CBoostJsonUnbufferedIStreamWrapper isw{istream};
    json::value doc;
    json::error_code ec;
    json::stream_parser p;
    std::string line;
    while (std::getline(istream, line) && !ec) {
        p.write(line, ec);
    }
    assertNoParseError(ec);
    doc = p.release();
    assertIsJsonObject(doc);

    LOG_DEBUG(<< "Parsed JSON doc: " << doc);

    std::size_t numberColumns{ifExists(JSON_NUM_COLUMNS_TAG, getAsUint64From, doc.as_object())};

    TStrVec columnNames;
    TBoolVec columnIsCategorical;
    TStrVecVec categoricalColumnValues;
    columnNames.reserve(numberColumns);
    columnIsCategorical.reserve(numberColumns);
    categoricalColumnValues.resize(numberColumns, {});

    for (const auto& name : ifExists(JSON_COLUMN_NAMES_TAG, getAsArrayFrom, doc.as_object())) {
        columnNames.push_back(getAsStringFrom(name));
    }

    for (const auto& isCategorical :
         ifExists(JSON_COLUMN_IS_CATEGORICAL_TAG, getAsArrayFrom, doc.as_object())) {
        columnIsCategorical.push_back(getAsBoolFrom(isCategorical));
    }

    std::size_t i{0};
    for (const auto& categories :
         ifExists(JSON_CATEGORICAL_COLUMN_VALUES_TAG, getAsArrayFrom, doc.as_object())) {
        for (const auto& category : getAsArrayFrom(categories)) {
            categoricalColumnValues[i].push_back(getAsStringFrom(category));
        }
        ++i;
    }

    TStrSizeUMap encodingIndices;
    for (const auto& entry :
         ifExists(JSON_ENCODING_NAME_INDEX_MAP_TAG, getAsArrayFrom, doc.as_object())) {
        encodingIndices.emplace(
            ifExists(JSON_ENCODING_NAME_INDEX_MAP_KEY_TAG, getAsStringFrom, entry.as_object()),
            ifExists(JSON_ENCODING_NAME_INDEX_MAP_VALUE_TAG, getAsUint64From, entry.as_object()));
    }

    ifExists(JSON_ENCODINGS_TAG, getAsObjectFrom, doc.as_object()); // Validate the nested object exists.
    TEncoderUPtr encodings;
    std::stringstream jsonStrm;
    CDataSummarizationJsonWriter::TGenericLineWriter writer{jsonStrm};
    writer.StartObject();
    writer.Key(JSON_ENCODINGS_TAG);
    writer.write(doc.as_object()[JSON_ENCODINGS_TAG]);
    writer.EndObject();
    core::CJsonStateRestoreTraverser traverser{jsonStrm};
    encodings = std::make_unique<maths::analytics::CDataFrameCategoryEncoder>(traverser);

    frame.columnNames(columnNames);
    frame.categoricalColumns(columnIsCategorical);
    frame.categoricalColumnValues(categoricalColumnValues);

    TStrVec rowVec;
    rowVec.reserve(numberColumns);
    for (const auto& row : ifExists(JSON_DATA_TAG, getAsArrayFrom, doc.as_object())) {
        for (const auto& column : getAsArrayFrom(row)) {
            rowVec.emplace_back(getAsStringFrom(column));
        }
        frame.parseAndWriteRow(core::make_const_range(rowVec, 0, rowVec.size()));
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
        } catch (const std::runtime_error& e) {
            LOG_ERROR(<< e.what());
        }
    }
    return nullptr;
}

CRetrainableModelJsonReader::TNodeVecVecUPtr
CRetrainableModelJsonReader::doBestForestFromJsonStream(std::istream& istream,
                                                        const TStrSizeUMap& encodingIndices) {
    using TNodeVec = maths::analytics::CBoostedTreeFactory::TNodeVec;
    using TNodeVecVec = maths::analytics::CBoostedTreeFactory::TNodeVecVec;

    json::stream_parser p;
    json::error_code ec;
    std::string line;
    while (std::getline(istream, line)) {
        LOG_DEBUG(<< "write_some: " << line);
        p.write_some(line);
    }
    p.finish( ec );
    assertNoParseError(ec);

    json::value doc = p.release();

    assertIsJsonObject(doc);

    LOG_DEBUG(<< "doc: " << doc);

    auto inferenceModel = ifExists(CInferenceModelDefinition::JSON_TRAINED_MODEL_TAG,
                                   getAsObjectFrom, doc.as_object());
    auto ensemble = ifExists(CEnsemble::JSON_ENSEMBLE_TAG, getAsObjectFrom, inferenceModel);
    auto trainedModels = ifExists(CEnsemble::JSON_TRAINED_MODELS_TAG, getAsArrayFrom, ensemble);
    auto forest = std::make_unique<TNodeVecVec>();
    forest->reserve(trainedModels.size());
    TStrVec featureNames;
    TNodeVec nodes;
    for (const auto& trainedModel : trainedModels) {
        auto tree = ifExists(CTree::JSON_TREE_TAG, getAsObjectFrom, trainedModel.as_object());
        featureNames.clear();
        for (const auto& name :
             ifExists(CTree::JSON_FEATURE_NAMES_TAG, getAsArrayFrom, tree)) {
            featureNames.emplace_back(getAsStringFrom(name));
        }
        auto treeNodes = ifExists(CTree::JSON_TREE_STRUCTURE_TAG, getAsArrayFrom, tree);
        nodes.clear();
        nodes.reserve(treeNodes.size());
        nodes.emplace_back(); // Add the root.
        for (const auto& node : treeNodes) {
            std::size_t nodeIndex{ifExists(CTree::CTreeNode::JSON_NODE_INDEX_TAG,
                                           getAsUint64From, node.as_object())};
            std::size_t numberSamples{ifExists(CTree::CTreeNode::JSON_NUMBER_SAMPLES_TAG,
                                               getAsUint64From, node.as_object())};
            nodes[nodeIndex].numberSamples(numberSamples);
            if (node.as_object().contains(CTree::CTreeNode::JSON_LEAF_VALUE_TAG)) {
                // Add a leaf node.
                if (node.as_object().at(CTree::CTreeNode::JSON_LEAF_VALUE_TAG).is_array()) {
                    auto leafValueArray =
                        getAsArrayFrom(node.as_object().at(CTree::CTreeNode::JSON_LEAF_VALUE_TAG));
                    maths::analytics::CBoostedTreeNode::TVector nodeValue(
                        leafValueArray.size());
                    for (std::size_t i = 0; i < leafValueArray.size(); ++i) {
                        nodeValue[static_cast<long>(i)] =
                            getAsDoubleFrom(leafValueArray[i]);
                    }
                    nodes[nodeIndex].value(nodeValue);
                } else {
                    maths::analytics::CBoostedTreeNode::TVector nodeValue(1);
                    nodeValue[0] = ifExists(CTree::CTreeNode::JSON_LEAF_VALUE_TAG,
                                            getAsDoubleFrom, node.as_object());
                    nodes[nodeIndex].value(nodeValue);
                }
            } else {
                // Add a split node.
                std::size_t splitFeature{lookup(encodingIndices, featureNames,
                                                ifExists(CTree::CTreeNode::JSON_SPLIT_FEATURE_TAG,
                                                         getAsUint64From, node.as_object()))};
                double gain{ifExists(CTree::CTreeNode::JSON_SPLIT_GAIN_TAG,
                                     getAsDoubleFrom, node.as_object())};
                double splitValue{ifExists(CTree::CTreeNode::JSON_THRESHOLD_TAG,
                                           getAsDoubleFrom, node.as_object())};
                bool assignMissingToLeft{ifExists(CTree::CTreeNode::JSON_DEFAULT_LEFT_TAG,
                                                  getAsBoolFrom, node.as_object())};
                std::size_t leftChildIndex{ifExists(CTree::CTreeNode::JSON_LEFT_CHILD_TAG,
                                                    getAsUint64From, node.as_object())};
                std::size_t rightChildIndex{ifExists(CTree::CTreeNode::JSON_RIGHT_CHILD_TAG,
                                                     getAsUint64From, node.as_object())};
                nodes[nodeIndex].split(splitFeature, splitValue,
                                       assignMissingToLeft, gain, 0.0, 0.0, nodes);
                nodes[nodeIndex].numberSamples(numberSamples);
                nodes[nodeIndex].leftChildIndex(
                    static_cast<maths::analytics::CBoostedTreeNode::TNodeIndex>(leftChildIndex));
                nodes[nodeIndex].rightChildIndex(
                    static_cast<maths::analytics::CBoostedTreeNode::TNodeIndex>(rightChildIndex));
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
