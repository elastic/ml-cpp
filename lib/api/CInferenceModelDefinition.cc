/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CInferenceModelDefinition.h>

#include <core/CPersistUtils.h>
#include <core/CRapidJsonLineWriter.h>
#include <core/RestoreMacros.h>

#include <unordered_map>

namespace ml {
namespace api {

namespace {
using TRapidJsonWriter = core::CRapidJsonLineWriter<rapidjson::StringBuffer>;

const std::string SPLIT_FEATURE_TAG{"split_feature"};
const std::string DEFAULT_LEFT_TAG{"assign_missing_to_left"};
const std::string NODE_VALUE_TAG{"node_value"};
const std::string SPLIT_INDEX_TAG{"split_index"};
const std::string SPLIT_VALUE_TAG{"split_value"};
const std::string LEFT_CHILD_TAG{"left_child"};
const std::string RIGHT_CHILD_TAG{"right_child"};
const std::string TREE_TAG{"a"};
const std::string TREE_NODE_TAG{"a"};

const std::string JSON_AGGREGATE_OUTPUT_TAG{"aggregate_output"};
const std::string JSON_CLASSIFICATION_LABELS_TAG{"classification_labels"};
const std::string JSON_COLUMNS_TAG{"columns"};
const std::string JSON_TARGET_MAP_TAG{"target_map"};
const std::string JSON_DEFAULT_VALUE_TAG{"default_value"};
const std::string JSON_DECISION_TYPE_TAG{"decision_type"};
const std::string JSON_DEFAULT_LEFT_TAG{"default_left"};
const std::string JSON_FEATURE_NAME_TAG{"feature_name"};
const std::string JSON_FEATURE_NAMES_TAG{"feature_names"};
const std::string JSON_FIELD_TAG{"field"};
const std::string JSON_FREQUENCY_ENCODING_TAG{"frequency_encoding"};
const std::string JSON_FREQUENCY_MAP_TAG{"frequency_map"};
const std::string JSON_HOT_MAP_TAG{"hot_map"};
const std::string JSON_LEAF_VALUE_TAG{"leaf_value"};
const std::string JSON_LEFT_CHILD_TAG{"left_child"};
const std::string JSON_LTE{"lte"};
const std::string JSON_NODE_INDEX_TAG{"node_index"};
const std::string JSON_ONE_HOT_ENCODING_TAG{"one_hot_encoding"};
const std::string JSON_PREPROCESSING_TAG{"preprocessing"};
const std::string JSON_INPUT_TAG{"input"};
const std::string JSON_RIGHT_CHILD_TAG{"right_child"};
const std::string JSON_SPLIT_FEATURE_TAG{"split_feature"};
const std::string JSON_SPLIT_GAIN_TAG{"split_gain"};
const std::string JSON_TARGET_MEAN_ENCODING_TAG{"target_mean_encoding"};
const std::string JSON_TARGET_TYPE_CLASSIFICATION{"classification"};
const std::string JSON_TARGET_TYPE_REGRESSION{"regression"};
const std::string JSON_TARGET_TYPE_TAG{"target_type"};
const std::string JSON_THRESHOLD_TAG{"threshold"};
const std::string JSON_TRAINED_MODEL_TAG{"trained_model"};
const std::string JSON_TRAINED_MODELS_TAG{"trained_models"};
const std::string JSON_TREE_STRUCTURE_TAG{"tree_structure"};
const std::string JSON_WEIGHTED_SUM_TAG{"weighted_sum"};
const std::string JSON_WEIGHTED_MODE_TAG{"weighted_mode"};
const std::string JSON_WEIGHTS_TAG{"weights"};

template<typename T>
void addJsonArray(const std::string& tag,
                  const std::vector<T>& vector,
                  rapidjson::Value& parentObject,
                  TRapidJsonWriter& writer) {
    rapidjson::Value array = writer.makeArray(vector.size());
    for (const auto& item : vector) {
        rapidjson::Value value;
        value.SetString(core::CStringUtils::typeToString(item), writer.getRawAllocator());
        array.PushBack(value, writer.getRawAllocator());
    }
    writer.addMember(tag, array, parentObject);
}
}

void CTree::CTreeNode::addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) {
    writer.addMember(JSON_NODE_INDEX_TAG, rapidjson::Value(m_NodeIndex).Move(), parentObject);
    writer.addMember(JSON_SPLIT_FEATURE_TAG,
                     rapidjson::Value(m_SplitFeature).Move(), parentObject);
    if (m_SplitGain.is_initialized()) {
        writer.addMember(JSON_SPLIT_GAIN_TAG,
                         rapidjson::Value(m_SplitGain.get()).Move(), parentObject);
    }
    writer.addMember(JSON_THRESHOLD_TAG, rapidjson::Value(m_Threshold).Move(), parentObject);
    writer.addMember(JSON_LEAF_VALUE_TAG, rapidjson::Value(m_LeafValue).Move(), parentObject);
    writer.addMember(JSON_DEFAULT_LEFT_TAG,
                     rapidjson::Value(m_DefaultLeft).Move(), parentObject);
    switch (m_DecisionType) {
    case E_LTE:
        writer.addMember(JSON_DECISION_TYPE_TAG, JSON_LTE, parentObject);
        break;
    default:
        writer.addMember(JSON_DECISION_TYPE_TAG, JSON_LTE, parentObject);
        break;
    }
    if (m_LeftChild.is_initialized()) {
        writer.addMember(JSON_LEFT_CHILD_TAG,
                         rapidjson::Value(m_LeftChild.get()).Move(), parentObject);
    }
    if (m_RightChild.is_initialized()) {
        writer.addMember(JSON_RIGHT_CHILD_TAG,
                         rapidjson::Value(m_RightChild.get()).Move(), parentObject);
    }
}

CTree::CTreeNode::CTreeNode(size_t nodeIndex,
                            double threshold,
                            bool defaultLeft,
                            double leafValue,
                            size_t splitFeature,
                            const CTree::CTreeNode::TOptionalSize& leftChild,
                            const CTree::CTreeNode::TOptionalSize& rightChild,
                            const CTree::CTreeNode::TOptionalDouble& splitGain)
    : m_NodeIndex(nodeIndex), m_Threshold(threshold), m_DefaultLeft(defaultLeft),
      m_LeafValue(leafValue), m_SplitFeature(splitFeature),
      m_LeftChild(leftChild), m_RightChild(rightChild), m_SplitGain(splitGain) {
}

void CEnsemble::addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) {
    CTrainedModel::addToDocument(parentObject, writer);
    rapidjson::Value trainedModelsArray = writer.makeArray(m_TrainedModels.size());
    for (auto trainedModel : m_TrainedModels) {
        rapidjson::Value trainedModelObject = writer.makeObject();
        trainedModel.addToDocument(trainedModelObject, writer);
        trainedModelsArray.PushBack(trainedModelObject, writer.getRawAllocator());
    }
    writer.addMember(JSON_TRAINED_MODELS_TAG, trainedModelsArray, parentObject);

    // aggregate output
    rapidjson::Value aggregateOutputObject = writer.makeObject();
    m_AggregateOutput->addToDocument(aggregateOutputObject, writer);
    writer.addMember(JSON_AGGREGATE_OUTPUT_TAG, aggregateOutputObject, parentObject);
}

void CEnsemble::featureNames(const CTrainedModel::TStringVec& featureNames) {
    for (auto trainedModel : m_TrainedModels) {
        trainedModel.featureNames(featureNames);
    }
}

void CEnsemble::aggregateOutput(TAggregateOutputUPtr&& aggregateOutput) {
    m_AggregateOutput.swap(aggregateOutput);
}

std::size_t CEnsemble::size() const {
    return m_TrainedModels.size();
}

CEnsemble::TTreeVec& CEnsemble::trainedModels() {
    return m_TrainedModels;
}

const CEnsemble::TAggregateOutputUPtr& CEnsemble::aggregateOutput() const {
    return m_AggregateOutput;
}

void CEnsemble::targetType(CTrainedModel::ETargetType targetType) {
    this->CTrainedModel::targetType(targetType);
    for (auto& trainedModel : m_TrainedModels) {
        trainedModel.targetType(targetType);
    }
}

CTrainedModel::ETargetType CEnsemble::targetType() const {
    return this->CTrainedModel::targetType();
}

void CTree::addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) {
    CTrainedModel::addToDocument(parentObject, writer);
    rapidjson::Value treeStructureArray = writer.makeArray(m_TreeStructure.size());
    for (auto treeNode : m_TreeStructure) {
        rapidjson::Value treeNodeObject = writer.makeObject();
        treeNode.addToDocument(treeNodeObject, writer);
        treeStructureArray.PushBack(treeNodeObject, writer.getRawAllocator());
    }
    writer.addMember(JSON_TREE_STRUCTURE_TAG, treeStructureArray, parentObject);
}

std::size_t CTree::size() const {
    return m_TreeStructure.size();
}

CTree::TTreeNodeVec& CTree::treeStructure() {
    return m_TreeStructure;
}

std::string CInferenceModelDefinition::jsonString() {
    rapidjson::StringBuffer stringBuffer;
    core::CRapidJsonLineWriter<rapidjson::StringBuffer> writer(stringBuffer);
    rapidjson::Value doc = writer.makeObject();

    //input
    rapidjson::Value inputObject = writer.makeObject();
    m_Input.addToDocument(inputObject, writer);
    writer.addMember(JSON_INPUT_TAG, inputObject, doc);

    // preprocessing
    rapidjson::Value preprocessingArray = writer.makeArray();
    for (const auto& encoding : m_Preprocessing) {
        rapidjson::Value encodingValue = writer.makeObject();
        encoding->addToDocument(encodingValue, writer);
        rapidjson::Value encodingEnclosingObject = writer.makeObject();
        writer.addMember(encoding->typeString(), encodingValue, encodingEnclosingObject);
        preprocessingArray.PushBack(encodingEnclosingObject, writer.getRawAllocator());
    }
    writer.addMember(JSON_PREPROCESSING_TAG, preprocessingArray, doc);

    //trained_model
    if (m_TrainedModel) {
        rapidjson::Value trainedModelValue = writer.makeObject();
        m_TrainedModel->addToDocument(trainedModelValue, writer);
        writer.addMember(JSON_TRAINED_MODEL_TAG, trainedModelValue, doc);
    } else {
        LOG_ERROR(<< "Trained model is not initialized");
    }
    writer.write(doc);
    return stringBuffer.GetString();
}

void CTrainedModel::addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) {
    //    rapidjson::Value featureNamesArray = writer.makeArray(m_FeatureNames.size());
    //    for (const auto& featureName : m_FeatureNames) {
    //        rapidjson::Value featureNameValue;
    //        featureNameValue.SetString(featureName, writer.getRawAllocator());
    //        featureNamesArray.PushBack(featureNameValue, writer.getRawAllocator());
    //    }
    //    writer.addMember(JSON_FEATURE_NAMES_TAG, featureNamesArray, parentObject);
    addJsonArray(JSON_FEATURE_NAME_TAG, m_FeatureNames, parentObject, writer);

    if (m_ClassificationLabels) {
        rapidjson::Value classificationLabelsArray =
            writer.makeArray(m_ClassificationLabels.get().size());
        for (const auto& classificationLabel : *m_ClassificationLabels) {
            rapidjson::Value classificationLabelValue;
            classificationLabelValue.SetString(classificationLabel,
                                               writer.getRawAllocator());
            classificationLabelsArray.PushBack(classificationLabelValue,
                                               writer.getRawAllocator());
        }
        writer.addMember(JSON_CLASSIFICATION_LABELS_TAG, classificationLabelsArray, parentObject);
    }

    switch (m_TargetType) {
    case E_Classification:
        writer.addMember(JSON_TARGET_TYPE_TAG, JSON_TARGET_TYPE_CLASSIFICATION, parentObject);
        break;
    case E_Regression:
        writer.addMember(JSON_TARGET_TYPE_TAG, JSON_TARGET_TYPE_REGRESSION, parentObject);
        break;
    default:
        writer.addMember(JSON_TARGET_TYPE_TAG, JSON_TARGET_TYPE_REGRESSION, parentObject);
        break;
    }
}

const CTrainedModel::TStringVec& CTrainedModel::featureNames() const {
    return m_FeatureNames;
}

void CTrainedModel::featureNames(const CTrainedModel::TStringVec& featureNames) {
    m_FeatureNames = featureNames;
}

void CTrainedModel::targetType(CTrainedModel::ETargetType targetType) {
    m_TargetType = targetType;
}

CTrainedModel::ETargetType CTrainedModel::targetType() const {
    return m_TargetType;
}

void CInferenceModelDefinition::fieldNames(const std::vector<std::string>& fieldNames) {
    m_FieldNames = fieldNames;
    m_Input.columns(fieldNames);
}

void CInferenceModelDefinition::encodings(const CInferenceModelDefinition::TEncodingUPtrVec& encodings) {
    //    if (encodings.empty()) {
    //        return;
    //    }
    //    using TOneHotEncodingUPtr = std::unique_ptr<COneHotEncoding>;
    //    using TOneHotEncodingUMap = std::unordered_map<std::string, TOneHotEncodingUPtr>;
    //    TOneHotEncodingUMap oneHotEncodingMaps;
    //    for (const auto& encoding : encodings) {
    //        std::size_t inputColumnIndex = encoding->inputColumnIndex();
    //        std::string fieldName = m_FieldNames[inputColumnIndex];
    //        if (encoding->type() == maths::EEncoding::E_OneHot) {
    //            std::size_t categoryUInt =
    //                static_cast<maths::CDataFrameCategoryEncoder::COneHotEncoding*>(
    //                    encoding.get())
    //                    ->hotCategory();
    //            std::string category = m_ReverseCategoryNameMap[inputColumnIndex][categoryUInt];
    //            std::string encodedFieldName = fieldName + "_" + category;
    //            if (oneHotEncodingMaps.find(fieldName) == oneHotEncodingMaps.end()) {
    //                auto apiEncoding = std::make_unique<api::COneHotEncoding>(
    //                    fieldName, api::COneHotEncoding::TStringStringUMap());
    //                oneHotEncodingMaps.emplace(fieldName, std::move(apiEncoding));
    //            }
    //            oneHotEncodingMaps[fieldName]->hotMap().emplace(category, encodedFieldName);
    //        } else if (encoding->type() == maths::EEncoding::E_TargetMean) {
    //            auto* enc = static_cast<maths::CDataFrameCategoryEncoder::CMappedEncoding*>(
    //                encoding.get());
    //            double defaultValue{enc->fallback()};
    //            std::string featureName{fieldName + "_targetmean"};
    //
    //            std::map<std::string, double> map;
    //            for (std::size_t categoryUInt = 0; categoryUInt < enc->map().size(); ++categoryUInt) {
    //                std::string category = m_ReverseCategoryNameMap[inputColumnIndex][categoryUInt];
    //                map.emplace(category, enc->map()[categoryUInt]);
    //            }
    //            m_Preprocessing.emplace_back(std::make_unique<CTargetMeanEncoding>(
    //                fieldName, defaultValue, featureName, map));
    //        } else if (encoding->type() == maths::EEncoding::E_Frequency) {
    //            auto* enc = static_cast<maths::CDataFrameCategoryEncoder::CMappedEncoding*>(
    //                encoding.get());
    //            std::string featureName{fieldName + "_frequency"};
    //            std::map<std::string, double> map;
    //            for (std::size_t categoryUInt = 0; categoryUInt < enc->map().size(); ++categoryUInt) {
    //                std::string category = m_ReverseCategoryNameMap[inputColumnIndex][categoryUInt];
    //                map.emplace(category, enc->map()[categoryUInt]);
    //            }
    //            m_Preprocessing.emplace_back(
    //                std::make_unique<CFrequencyEncoding>(fieldName, featureName, map));
    //        }
    //    }
    //
    //    for (auto& oneHotEncodingMapping : oneHotEncodingMaps) {
    //        m_Preprocessing.emplace_back(std::move(oneHotEncodingMapping.second));
    //    }
}

void CInferenceModelDefinition::trainedModel(std::unique_ptr<CTrainedModel>&& trainedModel) {
    m_TrainedModel.swap(trainedModel);
}

rapidjson::Value&& CInferenceModelDefinition::jsonObject() {
    rapidjson::Document doc;
    doc.Parse(this->jsonString());
    if (doc.GetParseError()) {
        HANDLE_FATAL(<< "Internal error: generated inference model JSON cannot be parsed. "
                     << "Please report this problem.")
    }
    return std::move(doc);
}

const CInferenceModelDefinition::TStrSizeUMapVec&
CInferenceModelDefinition::categoryNameMap() const {
    return m_CategoryNameMap;
}

void CInferenceModelDefinition::categoryNameMap(const CInferenceModelDefinition::TStrSizeUMapVec& categoryNameMap) {
    m_CategoryNameMap = categoryNameMap;
    m_ReverseCategoryNameMap.reserve(categoryNameMap.size());
    for (const auto& categoryNameMapping : categoryNameMap) {
        if (categoryNameMapping.empty() == false) {
            TSizeStrUMap map;
            for (const auto& categoryMappingPair : categoryNameMapping) {
                map.emplace(categoryMappingPair.second, categoryMappingPair.first);
            }
            m_ReverseCategoryNameMap.emplace_back(std::move(map));
        } else {
            m_ReverseCategoryNameMap.emplace_back();
        }
    }
}

std::unique_ptr<CTrainedModel>& CInferenceModelDefinition::trainedModel() {
    return m_TrainedModel;
}

CInferenceModelDefinition::CInferenceModelDefinition(
    const CInferenceModelDefinition::TStringVec& fieldNames,
    const CInferenceModelDefinition::TStrSizeUMapVec& categoryNameMap) {
    this->fieldNames(fieldNames);
    this->categoryNameMap(categoryNameMap);
}

const CInput& CInferenceModelDefinition::input() const {
    return m_Input;
}

CInferenceModelDefinition::TApiEncodingUPtrVec& CInferenceModelDefinition::preprocessing() {
    return m_Preprocessing;
}

const std::unique_ptr<CTrainedModel>& CInferenceModelDefinition::trainedModel() const {
    return m_TrainedModel;
}

const std::string& CInferenceModelDefinition::typeString() const {
    return m_TypeString;
}

void CInferenceModelDefinition::typeString(const std::string& typeString) {
    CInferenceModelDefinition::m_TypeString = typeString;
}

const CInput::TStringVec& CInput::columns() const {
    return m_Columns;
}

void CInput::columns(const TStringVec& columns) {
    m_Columns = columns;
}

void CInput::addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) {
    addJsonArray(JSON_COLUMNS_TAG, m_Columns, parentObject, writer);
}

const std::string& CTargetMeanEncoding::typeString() const {
    return JSON_TARGET_MEAN_ENCODING_TAG;
}

void CTargetMeanEncoding::addToDocument(rapidjson::Value& parentObject,
                                        CSerializableToJson::TRapidJsonWriter& writer) {
    this->CEncoding::addToDocument(parentObject, writer);
    writer.addMember(JSON_DEFAULT_VALUE_TAG,
                     rapidjson::Value(m_DefaultValue).Move(), parentObject);
    writer.addMember(JSON_FEATURE_NAME_TAG, m_FeatureName, parentObject);

    rapidjson::Value map = writer.makeObject();
    for (const auto& mapping : m_TargetMap) {
        writer.addMember(mapping.first, rapidjson::Value(mapping.second).Move(), map);
    }
    writer.addMember(JSON_TARGET_MAP_TAG, map, parentObject);
}

CTargetMeanEncoding::CTargetMeanEncoding(const std::string& field,
                                         double defaultValue,
                                         const std::string& featureName,
                                         std::map<std::string, double>&& targetMap)
    : CEncoding(field), m_DefaultValue(defaultValue),
      m_FeatureName(featureName), m_TargetMap(std::move(targetMap)) {
}

double CTargetMeanEncoding::defaultValue() const {
    return m_DefaultValue;
}

const std::string& CTargetMeanEncoding::featureName() const {
    return m_FeatureName;
}

const std::map<std::string, double>& CTargetMeanEncoding::targetMap() const {
    return m_TargetMap;
}

CFrequencyEncoding::CFrequencyEncoding(const std::string& field,
                                       const std::string& featureName,
                                       const std::map<std::string, double>& frequencyMap)
    : CEncoding(field), m_FeatureName(featureName), m_FrequencyMap(frequencyMap) {
}

void CEncoding::field(const std::string& field) {
    m_Field = field;
}

void CEncoding::addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) {
    writer.addMember(JSON_FIELD_TAG, m_Field, parentObject);
}

CEncoding::CEncoding(const std::string& field) : m_Field(field) {
}

void CFrequencyEncoding::addToDocument(rapidjson::Value& parentObject,
                                       CSerializableToJson::TRapidJsonWriter& writer) {
    this->CEncoding::addToDocument(parentObject, writer);
    writer.addMember(JSON_FEATURE_NAME_TAG, m_FeatureName, parentObject);
    rapidjson::Value frequencyMap = writer.makeObject();
    for (const auto& mapping : m_FrequencyMap) {
        writer.addMember(mapping.first, rapidjson::Value(mapping.second).Move(), frequencyMap);
    }
    writer.addMember(JSON_FREQUENCY_MAP_TAG, frequencyMap, parentObject);
}

const std::string& CFrequencyEncoding::typeString() const {
    return JSON_FREQUENCY_ENCODING_TAG;
}

const std::string& CFrequencyEncoding::featureName() const {
    return m_FeatureName;
}

const std::map<std::string, double>& CFrequencyEncoding::frequencyMap() const {
    return m_FrequencyMap;
}

COneHotEncoding::TStringStringUMap& COneHotEncoding::hotMap() {
    return m_HotMap;
}

const std::string& COneHotEncoding::typeString() const {
    return JSON_ONE_HOT_ENCODING_TAG;
}

void COneHotEncoding::addToDocument(rapidjson::Value& parentObject,
                                    CSerializableToJson::TRapidJsonWriter& writer) {
    this->CEncoding::addToDocument(parentObject, writer);
    rapidjson::Value hotMap = writer.makeObject();
    for (const auto& mapping : m_HotMap) {
        writer.addMember(mapping.first, mapping.second, hotMap);
    }
    writer.addMember(JSON_HOT_MAP_TAG, hotMap, parentObject);
}

COneHotEncoding::COneHotEncoding(const std::string& field,
                                 const COneHotEncoding::TStringStringUMap& hotMap)
    : CEncoding(field), m_HotMap(hotMap) {
}

CWeightedSum::CWeightedSum(const std::vector<double>& weights)
    : m_Weights{weights} {
}
CWeightedSum::CWeightedSum(std::size_t size, double weight)
    : m_Weights(size, weight) {
}

void CWeightedSum::addToDocument(rapidjson::Value& parentObject,
                                 CSerializableToJson::TRapidJsonWriter& writer) {
    addJsonArray(JSON_WEIGHTS_TAG, m_Weights, parentObject, writer);
}

const std::string& CWeightedSum::stringType() {
    return JSON_WEIGHTED_SUM_TAG;
}

CWeightedMode::CWeightedMode(const std::vector<double>& weights)
    : m_Weights(weights) {
}

const std::string& CWeightedMode::stringType() {
    return JSON_WEIGHTED_MODE_TAG;
}

void CWeightedMode::addToDocument(rapidjson::Value& parentObject,
                                  CSerializableToJson::TRapidJsonWriter& writer) {
    addJsonArray(JSON_WEIGHTS_TAG, m_Weights, parentObject, writer);
}

CWeightedMode::CWeightedMode(std::size_t size, double weight)
    : m_Weights(size, weight) {
}
}
}
