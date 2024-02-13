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

#include <core/CDataFrame.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CVectorRange.h>

#include <maths/analytics/CBoostedTree.h>
#include <maths/analytics/CDataFrameCategoryEncoder.h>

#include <api/CDataSummarizationJsonWriter.h>
#include <api/CInferenceModelDefinition.h>

#include <boost/json.hpp>
// This file must be manually included when
// using basic_parser to implement a parser.
#include <boost/json/basic_parser_impl.hpp>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace json = boost::json;

namespace {
// A bespoke SAX style parser for handling non-conformant input JSON that
// potentially contains duplicate object keys. To handle this case the parser
// instead wraps each offending key, value pair into another object and inserts
// these object wrappers in an array.
//
// Note that the json::value constructed by this parser is only ever meant to be
// used as an internal intermediary, never to be serialized for external used.
class custom_parser {
    struct handler {
        static inline std::string IDENTITY_ENCODING_TAG = "identity_encoding";
        static inline std::string ONE_HOT_ENCODING_TAG = "one_hot_encoding";
        static inline std::string FREQUENCY_ENCODING_TAG = "frequency_encoding";
        static inline std::string TARGET_MEAN_ENCODING_TAG = "target_mean_encoding";

        constexpr static std::size_t max_object_size =
            std::numeric_limits<std::size_t>::max();
        constexpr static std::size_t max_array_size =
            std::numeric_limits<std::size_t>::max();
        constexpr static std::size_t max_key_size = std::numeric_limits<std::size_t>::max();
        constexpr static std::size_t max_string_size =
            std::numeric_limits<std::size_t>::max();

        bool on_document_begin(json::error_code&) {
            s_Value.emplace_object();
            s_CurrentValue.push(&s_Value);
            return true;
        }
        bool on_document_end(json::error_code&) { return true; }
        bool on_object_begin(json::error_code&) {
            LOG_TRACE(<< "on_object_begin: s_Depth = " << s_CurrentValue.size());
            if (s_Keys.empty() == false) {
                if (s_Keys.top() == "encoding_vector") {
                    s_CurrentValue.top()->as_object()[s_Keys.top()] = json::array{};
                    s_CurrentValue.push(
                        &s_CurrentValue.top()->as_object()[s_Keys.top()]);
                } else {
                    if (s_CurrentValue.top()->is_array()) {
                        s_CurrentValue.top()->as_array().push_back(json::object{});
                        s_CurrentValue.push(&s_CurrentValue.top()->as_array().back());
                    } else {
                        s_CurrentValue.top()->as_object()[s_Keys.top()] = json::object{};
                        s_CurrentValue.push(
                            &s_CurrentValue.top()->as_object()[s_Keys.top()]);
                    }
                }
            }
            return true;
        }
        bool on_object_end(std::size_t, json::error_code&) {
            LOG_TRACE(<< "on_object_end: s_Depth = " << s_CurrentValue.size());
            s_CurrentValue.pop();
            if (s_Keys.empty() == false && s_EncodingTags.count(s_Keys.top()) > 0) {
                s_Keys.pop();
                s_CurrentValue.pop();
            }
            if (s_CurrentValue.empty() == false && s_CurrentValue.top()->is_array() == false) {
                s_Keys.pop();
            }
            return true;
        }
        bool on_array_begin(json::error_code&) {
            LOG_TRACE(<< "on_array_begin: s_Depth = " << s_CurrentValue.size());
            if (s_CurrentValue.empty() == false) {
                if (s_CurrentValue.top()->is_array()) {
                    s_CurrentValue.top()->as_array().push_back(json::array{});
                    s_CurrentValue.push(&s_CurrentValue.top()->as_array().back());
                } else {
                    s_CurrentValue.top()->as_object()[s_Keys.top()] = json::array{};
                    s_CurrentValue.push(
                        &s_CurrentValue.top()->as_object()[s_Keys.top()]);
                }
            }

            return true;
        }
        bool on_array_end(std::size_t, json::error_code&) {
            LOG_TRACE(<< "on_array_end: s_Depth = " << s_CurrentValue.size());

            s_CurrentValue.pop();
            if (s_CurrentValue.empty() == false && s_CurrentValue.top()->is_array() == false) {
                s_Keys.pop();
            }
            return true;
        }
        bool on_key_part(std::string_view, std::size_t, json::error_code&) {
            return true;
        }
        bool on_key(std::string_view s, std::size_t /*n*/, json::error_code& ec) {
            std::string str{s};
            s_Keys.push(str);
            if (s_CurrentValue.top()->is_array()) {
                s_CurrentValue.top()->as_array().push_back(json::object{});
                s_CurrentValue.push(&s_CurrentValue.top()->as_array().back());
            }
            return ec ? false : true;
        }
        bool on_string_part(std::string_view, std::size_t, json::error_code&) {
            return true;
        }
        bool on_string(std::string_view s, std::size_t /*n*/, json::error_code& ec) {
            if (s_CurrentValue.top()->is_array()) {
                s_CurrentValue.top()->as_array().push_back(json::string(s));
            } else {
                std::string k{s_Keys.top()};
                std::string v{s};
                LOG_TRACE(<< "on_string: key = " << k << ", value = " << v);
                s_CurrentValue.top()->as_object()[k] = json::string(v);
            }
            if (s_CurrentValue.empty() == false && s_CurrentValue.top()->is_array() == false) {
                s_Keys.pop();
            }
            return ec ? false : true;
        }
        bool on_number_part(std::string_view, json::error_code&) {
            return true;
        }
        bool on_int64(std::int64_t i, std::string_view, json::error_code& ec) {
            LOG_TRACE(<< "on_int64: " << i);
            if (s_CurrentValue.top()->is_array()) {
                s_CurrentValue.top()->as_array().push_back(json::value(i));
            } else {
                s_CurrentValue.top()->as_object()[s_Keys.top()] = json::value(i);
            }
            if (s_CurrentValue.empty() == false && s_CurrentValue.top()->is_array() == false) {
                s_Keys.pop();
            }
            return ec ? false : true;
        }
        bool on_uint64(std::uint64_t u, std::string_view, json::error_code& ec) {
            LOG_TRACE(<< "on_uint64: " << u);
            if (s_CurrentValue.top()->is_array()) {
                s_CurrentValue.top()->as_array().push_back(json::value(u));
            } else {
                s_CurrentValue.top()->as_object()[s_Keys.top()] = json::value(u);
            }
            if (s_CurrentValue.empty() == false && s_CurrentValue.top()->is_array() == false) {
                s_Keys.pop();
            }
            return ec ? false : true;
        }
        bool on_double(double d, std::string_view, json::error_code& ec) {
            LOG_TRACE(<< "on_double: " << d);
            if (s_CurrentValue.top()->is_array()) {
                s_CurrentValue.top()->as_array().push_back(json::value(d));
            } else {
                s_CurrentValue.top()->as_object()[s_Keys.top()] = json::value(d);
            }
            if (s_CurrentValue.empty() == false && s_CurrentValue.top()->is_array() == false) {
                s_Keys.pop();
            }
            return ec ? false : true;
        }
        bool on_bool(bool b, json::error_code& ec) {
            LOG_TRACE(<< "on_bool: " << b);
            if (s_CurrentValue.top()->is_array()) {
                s_CurrentValue.top()->as_array().push_back(json::value(b));
            } else {
                s_CurrentValue.top()->as_object()[s_Keys.top()] = json::value(b);
            }
            if (s_CurrentValue.empty() == false && s_CurrentValue.top()->is_array() == false) {
                s_Keys.pop();
            }
            return ec ? false : true;
        }
        bool on_null(json::error_code&) {
            LOG_TRACE(<< "on_null: ");
            return true;
        }
        bool on_comment_part(std::string_view, json::error_code&) {
            return true;
        }
        bool on_comment(std::string_view, json::error_code&) { return true; }

        std::stack<std::string> s_Keys;
        json::value s_Value;
        std::stack<json::value*> s_CurrentValue;
        std::set<std::string> s_EncodingTags{IDENTITY_ENCODING_TAG, ONE_HOT_ENCODING_TAG,
                                             TARGET_MEAN_ENCODING_TAG,
                                             FREQUENCY_ENCODING_TAG};
    };

    json::basic_parser<handler> p_;

public:
    custom_parser() : p_(json::parse_options()) {}

    ~custom_parser() {}

    std::size_t write(char const* data, std::size_t size, json::error_code& ec) {
        auto const n = p_.write_some(false, data, size, ec);
        if (!ec && n < size)
            ec = json::error::extra_data;
        return n;
    }

    json::value release() const { return std::move(p_.handler().s_Value); }
};

bool parse(std::string_view s, json::value& value, json::error_code& ec) {
    // Parse with the custom parser and return false on error
    custom_parser p;
    p.write(s.data(), s.size(), ec);
    value = p.release();
    return ec ? false : true;
}
}

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
    json::value doc;
    json::error_code ec;
    std::string line;
    while (std::getline(istream, line) && !ec) {
        LOG_TRACE(<< "Parsing line: " << line);
        parse(line, doc, ec);
    }
    assertNoParseError(ec);
    assertIsJsonObject(doc);

    LOG_TRACE(<< "Parsed JSON doc: " << doc);

    std::size_t numberColumns{
        ifExists(JSON_NUM_COLUMNS_TAG, getAsUint64From, doc.as_object())};

    TStrVec columnNames;
    TBoolVec columnIsCategorical;
    TStrVecVec categoricalColumnValues;
    columnNames.reserve(numberColumns);
    columnIsCategorical.reserve(numberColumns);
    categoricalColumnValues.resize(numberColumns, {});

    for (const auto& name :
         ifExists(JSON_COLUMN_NAMES_TAG, getAsArrayFrom, doc.as_object())) {
        columnNames.push_back(getAsStringFrom(name));
    }

    for (const auto& isCategorical :
         ifExists(JSON_COLUMN_IS_CATEGORICAL_TAG, getAsArrayFrom, doc.as_object())) {
        columnIsCategorical.push_back(getAsBoolFrom(isCategorical));
    }

    std::size_t i{0};
    for (const auto& categories : ifExists(JSON_CATEGORICAL_COLUMN_VALUES_TAG,
                                           getAsArrayFrom, doc.as_object())) {
        for (const auto& category : getAsArrayFrom(categories)) {
            categoricalColumnValues[i].push_back(getAsStringFrom(category));
        }
        ++i;
    }

    TStrSizeUMap encodingIndices;
    for (const auto& entry : ifExists(JSON_ENCODING_NAME_INDEX_MAP_TAG,
                                      getAsArrayFrom, doc.as_object())) {
        encodingIndices.emplace(ifExists(JSON_ENCODING_NAME_INDEX_MAP_KEY_TAG,
                                         getAsStringFrom, entry.as_object()),
                                ifExists(JSON_ENCODING_NAME_INDEX_MAP_VALUE_TAG,
                                         getAsUint64From, entry.as_object()));
    }

    TEncoderUPtr encodings =
        std::make_unique<maths::analytics::CDataFrameCategoryEncoder>(doc, true);

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

    if (encodings->numberEncodedColumns() != encodingIndices.size()) {
        LOG_FATAL(<< "Size mis-match: Encoded columns [" << encodings->numberEncodedColumns()
                  << "] != Encoding indices [" << encodingIndices.size() << "]");
    }

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
    using TNodeVec = maths::analytics::CBoostedTreeFactory::TNodeVec;
    using TNodeVecVec = maths::analytics::CBoostedTreeFactory::TNodeVecVec;

    json::stream_parser p;
    json::error_code ec;
    std::string line;
    while (std::getline(istream, line)) {
        LOG_TRACE(<< "write_some: " << line);
        p.write_some(line);
    }
    p.finish(ec);
    assertNoParseError(ec);

    json::value doc = p.release();

    assertIsJsonObject(doc);

    LOG_TRACE(<< "doc: " << doc);

    auto inferenceModel = ifExists(CInferenceModelDefinition::JSON_TRAINED_MODEL_TAG,
                                   getAsObjectFrom, doc.as_object());
    auto ensemble = ifExists(CEnsemble::JSON_ENSEMBLE_TAG, getAsObjectFrom, inferenceModel);
    auto trainedModels = ifExists(CEnsemble::JSON_TRAINED_MODELS_TAG, getAsArrayFrom, ensemble);
    auto forest = std::make_unique<TNodeVecVec>();
    forest->reserve(trainedModels.size());
    TStrVec featureNames;
    TNodeVec nodes;
    for (const auto& trainedModel : trainedModels) {
        auto tree = ifExists(CTree::JSON_TREE_TAG, getAsObjectFrom,
                             trainedModel.as_object());
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
                    auto leafValueArray = getAsArrayFrom(
                        node.as_object().at(CTree::CTreeNode::JSON_LEAF_VALUE_TAG));
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
                std::size_t splitFeature{
                    lookup(encodingIndices, featureNames,
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
