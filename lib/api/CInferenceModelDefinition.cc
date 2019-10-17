/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CInferenceModelDefinition.h>

#include <core/CPersistUtils.h>

#include <unordered_map>

namespace ml {
namespace api {

namespace {

const std::string JSON_AGGREGATE_OUTPUT_TAG{"aggregate_output"};
const std::string JSON_CLASSIFICATION_LABELS_TAG{"classification_labels"};
const std::string JSON_DECISION_TYPE_TAG{"decision_type"};
const std::string JSON_DEFAULT_LEFT_TAG{"default_left"};
const std::string JSON_DEFAULT_VALUE_TAG{"default_value"};
const std::string JSON_ENSEMBLE_TAG{"ensemble"};
const std::string JSON_FEATURE_NAME_TAG{"feature_name"};
const std::string JSON_FEATURE_NAMES_TAG{"feature_names"};
const std::string JSON_FIELD_NAMES_TAG{"field_names"};
const std::string JSON_FIELD_TAG{"field"};
const std::string JSON_FREQUENCY_ENCODING_TAG{"frequency_encoding"};
const std::string JSON_FREQUENCY_MAP_TAG{"frequency_map"};
const std::string JSON_HOT_MAP_TAG{"hot_map"};
const std::string JSON_INPUT_TAG{"input"};
const std::string JSON_LEAF_VALUE_TAG{"leaf_value"};
const std::string JSON_LEFT_CHILD_TAG{"left_child"};
const std::string JSON_LOGISTIC_REGRESSION_TAG{"logistic_regression"};
const std::string JSON_LT{"lt"};
const std::string JSON_NODE_INDEX_TAG{"node_index"};
const std::string JSON_ONE_HOT_ENCODING_TAG{"one_hot_encoding"};
const std::string JSON_PREPROCESSORS_TAG{"preprocessors"};
const std::string JSON_RIGHT_CHILD_TAG{"right_child"};
const std::string JSON_SPLIT_FEATURE_TAG{"split_feature"};
const std::string JSON_SPLIT_GAIN_TAG{"split_gain"};
const std::string JSON_TARGET_MAP_TAG{"target_map"};
const std::string JSON_TARGET_MEAN_ENCODING_TAG{"target_mean_encoding"};
const std::string JSON_TARGET_TYPE_CLASSIFICATION{"classification"};
const std::string JSON_TARGET_TYPE_REGRESSION{"regression"};
const std::string JSON_TARGET_TYPE_TAG{"target_type"};
const std::string JSON_THRESHOLD_TAG{"threshold"};
const std::string JSON_TRAINED_MODEL_TAG{"trained_model"};
const std::string JSON_TRAINED_MODELS_TAG{"trained_models"};
const std::string JSON_TREE_STRUCTURE_TAG{"tree_structure"};
const std::string JSON_TREE_TAG{"tree"};
const std::string JSON_WEIGHTED_MODE_TAG{"weighted_mode"};
const std::string JSON_WEIGHTED_SUM_TAG{"weighted_sum"};
const std::string JSON_WEIGHTS_TAG{"weights"};

template<typename T>
void addJsonArray(const std::string& tag,
                  const std::vector<T>& vector,
                  rapidjson::Value& parentObject,
                  CSerializableToJson::TRapidJsonWriter& writer) {
    rapidjson::Value array = writer.makeArray(vector.size());
    for (const auto& item : vector) {
        rapidjson::Value value;
        value.SetString(core::CStringUtils::typeToString(item), writer.getRawAllocator());
        array.PushBack(value, writer.getRawAllocator());
    }
    writer.addMember(tag, array, parentObject);
}
}

void CTree::CTreeNode::addToDocument(rapidjson::Value& parentObject,
                                     TRapidJsonWriter& writer) const {
    writer.addMember(JSON_NODE_INDEX_TAG, rapidjson::Value(m_NodeIndex).Move(), parentObject);

    if (m_LeftChild) {
        // internal node
        writer.addMember(
            JSON_SPLIT_FEATURE_TAG,
            rapidjson::Value(static_cast<std::uint64_t>(m_SplitFeature)).Move(),
            parentObject);
        if (m_SplitGain.is_initialized()) {
            writer.addMember(JSON_SPLIT_GAIN_TAG,
                             rapidjson::Value(m_SplitGain.get()).Move(), parentObject);
        }
        writer.addMember(JSON_THRESHOLD_TAG, rapidjson::Value(m_Threshold).Move(), parentObject);
        writer.addMember(JSON_DEFAULT_LEFT_TAG,
                         rapidjson::Value(m_DefaultLeft).Move(), parentObject);
        switch (m_DecisionType) {
        case E_LT:
            writer.addMember(JSON_DECISION_TYPE_TAG, JSON_LT, parentObject);
            break;
        }
        writer.addMember(
            JSON_LEFT_CHILD_TAG,
            rapidjson::Value(static_cast<std::uint64_t>(m_LeftChild.get())).Move(),
            parentObject);
        writer.addMember(
            JSON_RIGHT_CHILD_TAG,
            rapidjson::Value(static_cast<std::uint64_t>(m_RightChild.get())).Move(),
            parentObject);
    } else {
        // leaf node
        writer.addMember(JSON_LEAF_VALUE_TAG,
                         rapidjson::Value(m_LeafValue).Move(), parentObject);
    }
}

CTree::CTreeNode::CTreeNode(TNodeIndex nodeIndex,
                            double threshold,
                            bool defaultLeft,
                            double leafValue,
                            size_t splitFeature,
                            const TOptionalNodeIndex& leftChild,
                            const TOptionalNodeIndex& rightChild,
                            const TOptionalDouble& splitGain)
    : m_DefaultLeft(defaultLeft), m_NodeIndex(nodeIndex), m_LeftChild(leftChild),
      m_RightChild(rightChild), m_SplitFeature(splitFeature),
      m_Threshold(threshold), m_LeafValue(leafValue), m_SplitGain(splitGain) {
}

void CEnsemble::addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) const {
    rapidjson::Value ensembleObject = writer.makeObject();
    this->CTrainedModel::addToDocument(ensembleObject, writer);
    rapidjson::Value trainedModelsArray = writer.makeArray(m_TrainedModels.size());
    for (const auto& trainedModel : m_TrainedModels) {
        rapidjson::Value trainedModelObject = writer.makeObject();
        trainedModel->addToDocument(trainedModelObject, writer);
        trainedModelsArray.PushBack(trainedModelObject, writer.getRawAllocator());
    }
    writer.addMember(JSON_TRAINED_MODELS_TAG, trainedModelsArray, ensembleObject);

    // aggregate output
    rapidjson::Value aggregateOutputObject = writer.makeObject();
    m_AggregateOutput->addToDocument(aggregateOutputObject, writer);
    writer.addMember(JSON_AGGREGATE_OUTPUT_TAG, aggregateOutputObject, ensembleObject);
    writer.addMember(JSON_ENSEMBLE_TAG, ensembleObject, parentObject);
}

void CEnsemble::featureNames(const CTrainedModel::TStringVec& featureNames) {
    this->CTrainedModel::featureNames(featureNames);
    for (auto& trainedModel : m_TrainedModels) {
        trainedModel->featureNames(featureNames);
    }
}

void CEnsemble::aggregateOutput(TAggregateOutputUPtr&& aggregateOutput) {
    m_AggregateOutput.swap(aggregateOutput);
}

std::size_t CEnsemble::size() const {
    return m_TrainedModels.size();
}

CEnsemble::TTrainedModelUPtrVec& CEnsemble::trainedModels() {
    return m_TrainedModels;
}

const CEnsemble::TAggregateOutputUPtr& CEnsemble::aggregateOutput() const {
    return m_AggregateOutput;
}

void CEnsemble::targetType(CTrainedModel::ETargetType targetType) {
    this->CTrainedModel::targetType(targetType);
    for (auto& trainedModel : m_TrainedModels) {
        trainedModel->targetType(targetType);
    }
}

CTrainedModel::ETargetType CEnsemble::targetType() const {
    return this->CTrainedModel::targetType();
}

const CTrainedModel::TStringVec& CEnsemble::featureNames() const {
    return this->CTrainedModel::featureNames();
}

void CTree::addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) const {
    rapidjson::Value object = writer.makeObject();
    this->CTrainedModel::addToDocument(object, writer);
    rapidjson::Value treeStructureArray = writer.makeArray(m_TreeStructure.size());
    for (const auto& treeNode : m_TreeStructure) {
        rapidjson::Value treeNodeObject = writer.makeObject();
        treeNode.addToDocument(treeNodeObject, writer);
        treeStructureArray.PushBack(treeNodeObject, writer.getRawAllocator());
    }
    writer.addMember(JSON_TREE_STRUCTURE_TAG, treeStructureArray, object);
    writer.addMember(JSON_TREE_TAG, object, parentObject);
}

std::size_t CTree::size() const {
    return m_TreeStructure.size();
}

CTree::TTreeNodeVec& CTree::treeStructure() {
    return m_TreeStructure;
}

std::string CInferenceModelDefinition::jsonString() {

    std::ostringstream stream;
    {
        core::CJsonOutputStreamWrapper wrapper{stream};
        CSerializableToJson::TRapidJsonWriter writer{wrapper};
        rapidjson::Value doc = writer.makeObject();
        this->addToDocument(doc, writer);
        writer.write(doc);
        stream.flush();
    }
    // string writer puts the json object in an array, so we strip the external brackets
    std::string jsonStr{stream.str()};
    std::string resultString(jsonStr, 1, jsonStr.size() - 2);
    return resultString;
}

void CInferenceModelDefinition::addToDocument(rapidjson::Value& parentObject,
                                              TRapidJsonWriter& writer) const { //input
    rapidjson::Value inputObject = writer.makeObject();
    m_Input.addToDocument(inputObject, writer);
    writer.addMember(JSON_INPUT_TAG, inputObject, parentObject);

    // preprocessors
    rapidjson::Value preprocessingArray = writer.makeArray();
    for (const auto& encoding : m_Preprocessors) {
        rapidjson::Value encodingValue = writer.makeObject();
        encoding->addToDocument(encodingValue, writer);
        rapidjson::Value encodingEnclosingObject = writer.makeObject();
        writer.addMember(encoding->typeString(), encodingValue, encodingEnclosingObject);
        preprocessingArray.PushBack(encodingEnclosingObject, writer.getRawAllocator());
    }
    writer.addMember(JSON_PREPROCESSORS_TAG, preprocessingArray, parentObject);

    //trained_model
    if (m_TrainedModel) {
        rapidjson::Value trainedModelValue = writer.makeObject();
        m_TrainedModel->addToDocument(trainedModelValue, writer);
        writer.addMember(JSON_TRAINED_MODEL_TAG, trainedModelValue, parentObject);
    } else {
        LOG_ERROR(<< "Trained model is not initialized");
    }
}

void CTrainedModel::addToDocument(rapidjson::Value& parentObject,
                                  TRapidJsonWriter& writer) const {
    addJsonArray(JSON_FEATURE_NAMES_TAG, m_FeatureNames, parentObject, writer);

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
    }
}

const CTrainedModel::TStringVec& CTrainedModel::featureNames() const {
    return m_FeatureNames;
}

void CTrainedModel::featureNames(const TStringVec& featureNames) {
    m_FeatureNames = featureNames;
}

void CTrainedModel::targetType(ETargetType targetType) {
    m_TargetType = targetType;
}

CTrainedModel::ETargetType CTrainedModel::targetType() const {
    return m_TargetType;
}

void CInferenceModelDefinition::fieldNames(const TStringVec& fieldNames) {
    m_FieldNames = fieldNames;
    m_Input.fieldNames(fieldNames);
}

void CInferenceModelDefinition::trainedModel(std::unique_ptr<CTrainedModel>&& trainedModel) {
    m_TrainedModel.swap(trainedModel);
}

std::unique_ptr<CTrainedModel>& CInferenceModelDefinition::trainedModel() {
    return m_TrainedModel;
}

const CInput& CInferenceModelDefinition::input() const {
    return m_Input;
}

CInferenceModelDefinition::TApiEncodingUPtrVec& CInferenceModelDefinition::preprocessors() {
    return m_Preprocessors;
}

const std::string& CInferenceModelDefinition::typeString() const {
    return m_TypeString;
}

void CInferenceModelDefinition::typeString(const std::string& typeString) {
    CInferenceModelDefinition::m_TypeString = typeString;
}

const CInput::TStringVec& CInput::fieldNames() const {
    return m_FieldNames;
}

void CInput::fieldNames(const TStringVec& columns) {
    m_FieldNames = columns;
}

void CInput::addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) const {
    addJsonArray(JSON_FIELD_NAMES_TAG, m_FieldNames, parentObject, writer);
}

const std::string& CTargetMeanEncoding::typeString() const {
    return JSON_TARGET_MEAN_ENCODING_TAG;
}

void CTargetMeanEncoding::addToDocument(rapidjson::Value& parentObject,
                                        TRapidJsonWriter& writer) const {
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
                                         std::string featureName,
                                         TStringDoubleUMap&& targetMap)
    : CEncoding(field), m_DefaultValue(defaultValue),
      m_FeatureName(std::move(featureName)), m_TargetMap(std::move(targetMap)) {
}

double CTargetMeanEncoding::defaultValue() const {
    return m_DefaultValue;
}

const std::string& CTargetMeanEncoding::featureName() const {
    return m_FeatureName;
}

const CTargetMeanEncoding::TStringDoubleUMap& CTargetMeanEncoding::targetMap() const {
    return m_TargetMap;
}

CFrequencyEncoding::CFrequencyEncoding(const std::string& field,
                                       std::string featureName,
                                       TStringDoubleUMap frequencyMap)
    : CEncoding(field), m_FeatureName(std::move(featureName)),
      m_FrequencyMap(std::move(frequencyMap)) {
}

void CEncoding::field(const std::string& field) {
    m_Field = field;
}

void CEncoding::addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) const {
    writer.addMember(JSON_FIELD_TAG, m_Field, parentObject);
}

CEncoding::CEncoding(std::string field) : m_Field(std::move(field)) {
}

void CFrequencyEncoding::addToDocument(rapidjson::Value& parentObject,
                                       CSerializableToJson::TRapidJsonWriter& writer) const {
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

const CFrequencyEncoding::TStringDoubleUMap& CFrequencyEncoding::frequencyMap() const {
    return m_FrequencyMap;
}

COneHotEncoding::TStringStringUMap& COneHotEncoding::hotMap() {
    return m_HotMap;
}

const std::string& COneHotEncoding::typeString() const {
    return JSON_ONE_HOT_ENCODING_TAG;
}

void COneHotEncoding::addToDocument(rapidjson::Value& parentObject,
                                    CSerializableToJson::TRapidJsonWriter& writer) const {
    this->CEncoding::addToDocument(parentObject, writer);
    rapidjson::Value hotMap = writer.makeObject();
    for (const auto& mapping : m_HotMap) {
        writer.addMember(mapping.first, mapping.second, hotMap);
    }
    writer.addMember(JSON_HOT_MAP_TAG, hotMap, parentObject);
}

COneHotEncoding::COneHotEncoding(const std::string& field, TStringStringUMap hotMap)
    : CEncoding(field), m_HotMap(std::move(hotMap)) {
}

CWeightedSum::CWeightedSum(TDoubleVec&& weights)
    : m_Weights{std::move(weights)} {
}
CWeightedSum::CWeightedSum(std::size_t size, double weight)
    : m_Weights(size, weight) {
}

void CWeightedSum::addToDocument(rapidjson::Value& parentObject,
                                 CSerializableToJson::TRapidJsonWriter& writer) const {
    rapidjson::Value object = writer.makeObject();
    rapidjson::Value array = writer.makeArray(m_Weights.size());
    for (const auto item : m_Weights) {
        array.PushBack(rapidjson::Value(item).Move(), writer.getRawAllocator());
    }
    writer.addMember(JSON_WEIGHTS_TAG, array, object);
    writer.addMember(this->stringType(), object, parentObject);
}

const std::string& CWeightedSum::stringType() const {
    return JSON_WEIGHTED_SUM_TAG;
}

CWeightedMode::CWeightedMode(TDoubleVec&& weights)
    : m_Weights(std::move(weights)) {
}

const std::string& CWeightedMode::stringType() const {
    return JSON_WEIGHTED_MODE_TAG;
}

void CWeightedMode::addToDocument(rapidjson::Value& parentObject,
                                  CSerializableToJson::TRapidJsonWriter& writer) const {
    rapidjson::Value object = writer.makeObject();
    rapidjson::Value array = writer.makeArray(m_Weights.size());
    for (const auto item : m_Weights) {
        array.PushBack(rapidjson::Value(item).Move(), writer.getRawAllocator());
    }
    writer.addMember(JSON_WEIGHTS_TAG, array, object);
    writer.addMember(this->stringType(), object, parentObject);
}

CWeightedMode::CWeightedMode(std::size_t size, double weight)
    : m_Weights(size, weight) {
}

CLogisticRegression::CLogisticRegression(CLogisticRegression::TDoubleVec&& weights)
    : m_Weights(std::move(weights)) {
}

CLogisticRegression::CLogisticRegression(std::size_t size, double weight)
    : m_Weights(size, weight) {
}

void CLogisticRegression::addToDocument(rapidjson::Value& parentObject,
                                        TRapidJsonWriter& writer) const {
    rapidjson::Value object = writer.makeObject();
    rapidjson::Value array = writer.makeArray(m_Weights.size());
    for (const auto item : m_Weights) {
        array.PushBack(rapidjson::Value(item).Move(), writer.getRawAllocator());
    }
    writer.addMember(JSON_WEIGHTS_TAG, array, object);
    writer.addMember(this->stringType(), object, parentObject);
}

const std::string& CLogisticRegression::stringType() const {
    return JSON_LOGISTIC_REGRESSION_TAG;
}
}
}
