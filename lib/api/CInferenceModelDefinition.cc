/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CInferenceModelDefinition.h>

#include <core/CPersistUtils.h>

#include <cmath>
#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace ml {
namespace api {

namespace {
// clang-format off
const std::string JSON_AGGREGATE_OUTPUT_TAG{"aggregate_output"};
const std::string JSON_CLASSIFICATION_LABELS_TAG{"classification_labels"};
const std::string JSON_CLASSIFICATION_WEIGHTS_TAG{"classification_weights"};
const std::string JSON_DECISION_TYPE_TAG{"decision_type"};
const std::string JSON_DEFAULT_LEFT_TAG{"default_left"};
const std::string JSON_DEFAULT_VALUE_TAG{"default_value"};
const std::string JSON_ENSEMBLE_MODEL_SIZE_TAG{"ensemble_model_size"};
const std::string JSON_ENSEMBLE_TAG{"ensemble"};
const std::string JSON_FEATURE_NAME_LENGTH_TAG{"feature_name_length"};
const std::string JSON_FEATURE_NAME_LENGTHS_TAG{"feature_name_lengths"};
const std::string JSON_FEATURE_NAME_TAG{"feature_name"};
const std::string JSON_FEATURE_NAMES_TAG{"feature_names"};
const std::string JSON_FIELD_LENGTH_TAG{"field_length"};
const std::string JSON_FIELD_NAMES_TAG{"field_names"};
const std::string JSON_FIELD_TAG{"field"};
const std::string JSON_FIELD_VALUE_LENGTHS_TAG{"field_value_lengths"};
const std::string JSON_FREQUENCY_ENCODING_TAG{"frequency_encoding"};
const std::string JSON_FREQUENCY_MAP_TAG{"frequency_map"};
const std::string JSON_HOT_MAP_TAG{"hot_map"};
const std::string JSON_LEAF_VALUE_TAG{"leaf_value"};
const std::string JSON_LEFT_CHILD_TAG{"left_child"};
const std::string JSON_LOGISTIC_REGRESSION_TAG{"logistic_regression"};
const std::string JSON_LT{"lt"};
const std::string JSON_NODE_INDEX_TAG{"node_index"};
const std::string JSON_NUM_CLASSES_TAG{"num_classes"};
const std::string JSON_NUM_CLASSIFICATION_WEIGHTS_TAG{"num_classification_weights"};
const std::string JSON_NUM_LEAVES_TAG{"num_leaves"};
const std::string JSON_NUM_NODES_TAG{"num_nodes"};
const std::string JSON_NUM_OPERATIONS_TAG{"num_operations"};
const std::string JSON_NUM_OUTPUT_PROCESSOR_WEIGHTS_TAG{"num_output_processor_weights"};
const std::string JSON_NUMBER_SAMPLES_TAG{"number_samples"};
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
const std::string JSON_TRAINED_MODEL_SIZE_TAG{"trained_model_size"};
const std::string JSON_TRAINED_MODEL_TAG{"trained_model"};
const std::string JSON_TRAINED_MODELS_TAG{"trained_models"};
const std::string JSON_TREE_STRUCTURE_TAG{"tree_structure"};
const std::string JSON_TREE_TAG{"tree"};
const std::string JSON_TREE_SIZES_TAG{"tree_sizes"};
const std::string JSON_WEIGHTED_MODE_TAG{"weighted_mode"};
const std::string JSON_WEIGHTED_SUM_TAG{"weighted_sum"};
const std::string JSON_WEIGHTS_TAG{"weights"};
// clang-format on

auto toJson(const std::string& value, CSerializableToJson::TRapidJsonWriter& writer) {
    rapidjson::Value result;
    result.SetString(value, writer.getRawAllocator());
    return result;
}

auto toJson(double value, CSerializableToJson::TRapidJsonWriter&) {
    return rapidjson::Value{value};
}

auto toJson(std::size_t value) {
    return rapidjson::Value{static_cast<std::uint64_t>(value)};
}

auto toJson(std::size_t value, CSerializableToJson::TRapidJsonWriter&) {
    return toJson(value);
}

template<typename T>
void addJsonArray(const std::string& tag,
                  const std::vector<T>& vector,
                  rapidjson::Value& parentObject,
                  CSerializableToJson::TRapidJsonWriter& writer) {
    rapidjson::Value array{writer.makeArray(vector.size())};
    for (const auto& value : vector) {
        array.PushBack(toJson(value, writer), writer.getRawAllocator());
    }
    writer.addMember(tag, array, parentObject);
}
}

void CTree::CTreeNode::addToDocument(rapidjson::Value& parentObject,
                                     TRapidJsonWriter& writer) const {
    writer.addMember(JSON_NODE_INDEX_TAG, rapidjson::Value(m_NodeIndex).Move(), parentObject);
    writer.addMember(JSON_NUMBER_SAMPLES_TAG, toJson(m_NumberSamples).Move(), parentObject);

    if (m_LeftChild) {
        // internal node
        writer.addMember(JSON_SPLIT_FEATURE_TAG, toJson(m_SplitFeature).Move(), parentObject);
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
        writer.addMember(JSON_LEFT_CHILD_TAG, toJson(m_LeftChild.get()).Move(), parentObject);
        writer.addMember(JSON_RIGHT_CHILD_TAG, toJson(m_RightChild.get()).Move(), parentObject);
    } else if (m_LeafValue.size() > 1) {
        // leaf node
        addJsonArray(JSON_LEAF_VALUE_TAG, m_LeafValue, parentObject, writer);
    } else {
        // leaf node
        writer.addMember(JSON_LEAF_VALUE_TAG,
                         rapidjson::Value(m_LeafValue[0]).Move(), parentObject);
    }
}

CTree::CTreeNode::CTreeNode(TNodeIndex nodeIndex,
                            double threshold,
                            bool defaultLeft,
                            TDoubleVec leafValue,
                            std::size_t splitFeature,
                            std::size_t numberSamples,
                            const TOptionalNodeIndex& leftChild,
                            const TOptionalNodeIndex& rightChild,
                            const TOptionalDouble& splitGain)
    : m_DefaultLeft(defaultLeft), m_NodeIndex(nodeIndex), m_LeftChild(leftChild),
      m_RightChild(rightChild), m_SplitFeature(splitFeature),
      m_NumberSamples(numberSamples), m_Threshold(threshold),
      m_LeafValue(std::move(leafValue)), m_SplitGain(splitGain) {
}

size_t CTree::CTreeNode::splitFeature() const {
    return m_SplitFeature;
}

void CTree::CTreeNode::splitFeature(std::size_t splitFeature) {
    m_SplitFeature = splitFeature;
}

bool CTree::CTreeNode::leaf() const {
    return m_LeftChild.is_initialized() == false;
}

CTrainedModel::TSizeInfoUPtr CTree::sizeInfo() const {
    return std::make_unique<CSizeInfo>(*this);
}

CTree::CSizeInfo::CSizeInfo(const CTree& tree)
    : CTrainedModel::CSizeInfo(tree), m_Tree{tree} {
}

void CTree::CSizeInfo::addToDocument(rapidjson::Value& parentObject,
                                     TRapidJsonWriter& writer) const {
    std::size_t numLeaves{0};
    std::size_t numNodes{0};
    for (const auto& node : m_Tree.m_TreeStructure) {
        if (node.leaf()) {
            ++numLeaves;
        } else {
            ++numNodes;
        }
    }
    writer.addMember(JSON_NUM_NODES_TAG, toJson(numNodes).Move(), parentObject);
    writer.addMember(JSON_NUM_LEAVES_TAG, toJson(numLeaves).Move(), parentObject);
}

std::size_t CTree::CSizeInfo::numOperations() const {
    std::size_t numLeaves{0};
    std::size_t numNodes{0};
    for (const auto& node : m_Tree.m_TreeStructure) {
        if (node.leaf()) {
            ++numLeaves;
        } else {
            ++numNodes;
        }
    }
    // Strictly speaking, this formula is correct only for balanced trees, but it will
    // give a good average estimate for other binary trees as well.
    return static_cast<std::size_t>(std::ceil(std::log2(numNodes + numLeaves + 1)));
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

void CEnsemble::featureNames(const TStringVec& featureNames) {
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

void CEnsemble::targetType(ETargetType targetType) {
    this->CTrainedModel::targetType(targetType);
    for (auto& trainedModel : m_TrainedModels) {
        trainedModel->targetType(targetType);
    }
}

CTrainedModel::TStringVec CEnsemble::removeUnusedFeatures() {
    std::unordered_set<std::string> set;
    for (auto& trainedModel : this->trainedModels()) {
        TStringVec vec(trainedModel->removeUnusedFeatures());
        set.insert(vec.begin(), vec.end());
    }
    TStringVec selectedFeatureNames;
    selectedFeatureNames.reserve(set.size());
    std::copy(set.begin(), set.end(), std::back_inserter(selectedFeatureNames));
    std::sort(selectedFeatureNames.begin(), selectedFeatureNames.end());
    this->CTrainedModel::featureNames(selectedFeatureNames);
    return selectedFeatureNames;
}

const CTrainedModel::TStringVec& CEnsemble::featureNames() const {
    return this->CTrainedModel::featureNames();
}

void CEnsemble::classificationLabels(const TStringVec& classificationLabels) {
    this->CTrainedModel::classificationLabels(classificationLabels);
    for (auto& trainedModel : m_TrainedModels) {
        trainedModel->classificationLabels(classificationLabels);
    }
}

void CEnsemble::classificationWeights(TDoubleVec classificationWeights) {
    for (auto& trainedModel : m_TrainedModels) {
        trainedModel->classificationWeights(classificationWeights);
    }
    this->CTrainedModel::classificationWeights(std::move(classificationWeights));
}

CTrainedModel::TSizeInfoUPtr CEnsemble::sizeInfo() const {
    return std::make_unique<CSizeInfo>(*this);
}

CEnsemble::CSizeInfo::CSizeInfo(const CEnsemble& ensemble)
    : CTrainedModel::CSizeInfo(ensemble), m_Ensemble{&ensemble} {
}

std::size_t CEnsemble::CSizeInfo::numOperations() const {
    std::size_t numOperations{0};
    for (const auto& model : m_Ensemble->m_TrainedModels) {
        numOperations += model->sizeInfo()->numOperations();
    }
    return numOperations;
}

void CEnsemble::CSizeInfo::addToDocument(rapidjson::Value& parentObject,
                                         TRapidJsonWriter& writer) const {
    this->CTrainedModel::CSizeInfo::addToDocument(parentObject, writer);
    rapidjson::Value featureNameLengthsArray{
        writer.makeArray(m_Ensemble->featureNames().size())};
    for (const auto& featureName : m_Ensemble->featureNames()) {
        featureNameLengthsArray.PushBack(toJson(featureName.size()).Move(),
                                         writer.getRawAllocator());
    }
    writer.addMember(JSON_FEATURE_NAME_LENGTHS_TAG, featureNameLengthsArray, parentObject);

    rapidjson::Value treeSizesArray{writer.makeArray(m_Ensemble->m_TrainedModels.size())};
    for (const auto& trainedModel : m_Ensemble->m_TrainedModels) {
        rapidjson::Value item{writer.makeObject()};
        trainedModel->sizeInfo()->addToDocument(item, writer);
        treeSizesArray.PushBack(item, writer.getRawAllocator());
    }
    writer.addMember(JSON_TREE_SIZES_TAG, treeSizesArray, parentObject);

    std::size_t numOutputProcessorWeights{m_Ensemble->m_TrainedModels.size()};
    writer.addMember(JSON_NUM_OUTPUT_PROCESSOR_WEIGHTS_TAG,
                     toJson(numOutputProcessorWeights).Move(), parentObject);
    std::size_t numOperations{this->numOperations()};
    writer.addMember(JSON_NUM_OPERATIONS_TAG, toJson(numOperations).Move(), parentObject);
}

void CTree::addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) const {
    rapidjson::Value object{writer.makeObject()};
    this->CTrainedModel::addToDocument(object, writer);
    rapidjson::Value treeStructureArray{writer.makeArray(m_TreeStructure.size())};
    for (const auto& treeNode : m_TreeStructure) {
        rapidjson::Value treeNodeObject{writer.makeObject()};
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

CTrainedModel::TStringVec CTree::removeUnusedFeatures() {
    std::unordered_map<std::size_t, std::size_t> selectedFeatureIndices;
    for (auto& treeNode : m_TreeStructure) {
        if (treeNode.leaf() == false) {
            std::size_t adjustedIndex{selectedFeatureIndices
                                          .emplace(treeNode.splitFeature(),
                                                   selectedFeatureIndices.size())
                                          .first->second};
            treeNode.splitFeature(adjustedIndex);
        }
    }
    TStringVec selectedFeatureNames(selectedFeatureIndices.size());
    auto& featureNames = this->featureNames();
    for (auto i = selectedFeatureIndices.begin(); i != selectedFeatureIndices.end(); ++i) {
        selectedFeatureNames[i->second] = std::move(featureNames[i->first]);
    }
    this->featureNames(std::move(selectedFeatureNames));
    return this->featureNames();
}

std::string CInferenceModelDefinition::jsonString() {

    std::ostringstream stream;
    {
        core::CJsonOutputStreamWrapper wrapper{stream};
        CSerializableToJson::TRapidJsonWriter writer{wrapper};
        rapidjson::Value doc{writer.makeObject()};
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
                                              TRapidJsonWriter& writer) const {
    // preprocessors
    rapidjson::Value preprocessingArray{writer.makeArray()};
    for (const auto& encoding : m_Preprocessors) {
        rapidjson::Value encodingValue{writer.makeObject()};
        encoding->addToDocument(encodingValue, writer);
        rapidjson::Value encodingEnclosingObject{writer.makeObject()};
        writer.addMember(encoding->typeString(), encodingValue, encodingEnclosingObject);
        preprocessingArray.PushBack(encodingEnclosingObject, writer.getRawAllocator());
    }
    writer.addMember(JSON_PREPROCESSORS_TAG, preprocessingArray, parentObject);

    // trained_model
    if (m_TrainedModel) {
        rapidjson::Value trainedModelValue{writer.makeObject()};
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
        addJsonArray(JSON_CLASSIFICATION_LABELS_TAG, *m_ClassificationLabels,
                     parentObject, writer);
    }

    if (m_ClassificationWeights) {
        addJsonArray(JSON_CLASSIFICATION_WEIGHTS_TAG, *m_ClassificationWeights,
                     parentObject, writer);
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

CTrainedModel::TStringVec& CTrainedModel::featureNames() {
    return m_FeatureNames;
}

void CTrainedModel::featureNames(TStringVec&& featureNames) {
    m_FeatureNames = std::move(featureNames);
}

const CTrainedModel::TOptionalStringVec& CTrainedModel::classificationLabels() const {
    return m_ClassificationLabels;
}

void CTrainedModel::classificationLabels(const TStringVec& classificationLabels) {
    m_ClassificationLabels = classificationLabels;
}

const CTrainedModel::TOptionalDoubleVec& CTrainedModel::classificationWeights() const {
    return m_ClassificationWeights;
}

void CTrainedModel::classificationWeights(TDoubleVec classificationWeights) {
    m_ClassificationWeights = std::move(classificationWeights);
}

CTrainedModel::CSizeInfo::CSizeInfo(const CTrainedModel& trainedModel)
    : m_TrainedModel{trainedModel} {
}

void CTrainedModel::CSizeInfo::addToDocument(rapidjson::Value& parentObject,
                                             TRapidJsonWriter& writer) const {
    if (m_TrainedModel.targetType() == E_Classification) {
        writer.addMember(JSON_NUM_CLASSIFICATION_WEIGHTS_TAG,
                         toJson(m_TrainedModel.classificationWeights()->size()).Move(),
                         parentObject);
        writer.addMember(JSON_NUM_CLASSES_TAG,
                         toJson(m_TrainedModel.classificationLabels()->size()).Move(),
                         parentObject);
    }
}

void CInferenceModelDefinition::fieldNames(TStringVec&& fieldNames) {
    m_FieldNames = std::move(fieldNames);
}

void CInferenceModelDefinition::trainedModel(TTrainedModelUPtr&& trainedModel) {
    m_TrainedModel = std::move(trainedModel);
}

CInferenceModelDefinition::TTrainedModelUPtr& CInferenceModelDefinition::trainedModel() {
    return m_TrainedModel;
}

const CInferenceModelDefinition::TTrainedModelUPtr&
CInferenceModelDefinition::trainedModel() const {
    return m_TrainedModel;
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

const CInferenceModelDefinition::TStringVec& CInferenceModelDefinition::fieldNames() const {
    return m_FieldNames;
}

size_t CInferenceModelDefinition::dependentVariableColumnIndex() const {
    return m_DependentVariableColumnIndex;
}

void CInferenceModelDefinition::dependentVariableColumnIndex(std::size_t dependentVariableColumnIndex) {
    m_DependentVariableColumnIndex = dependentVariableColumnIndex;
}

CInferenceModelDefinition::TSizeInfoUPtr CInferenceModelDefinition::sizeInfo() const {
    return std::make_unique<CSizeInfo>(*this);
}

CInferenceModelDefinition::CSizeInfo::CSizeInfo(const CInferenceModelDefinition& definition)
    : m_Definition{definition} {
}

std::string CInferenceModelDefinition::CSizeInfo::jsonString() {
    std::ostringstream stream;
    {
        // we use this scope to finish writing the object in the CJsonOutputStreamWrapper destructor
        core::CJsonOutputStreamWrapper wrapper{stream};
        CSerializableToJson::TRapidJsonWriter writer{wrapper};
        rapidjson::Value doc{writer.makeObject()};
        this->addToDocument(doc, writer);
        writer.write(doc);
        stream.flush();
    }
    // string writer puts the json object in an array, so we strip the external brackets
    std::string jsonStr{stream.str()};
    std::string resultString(jsonStr, 1, jsonStr.size() - 2);
    return resultString;
}

void CInferenceModelDefinition::CSizeInfo::addToDocument(rapidjson::Value& parentObject,
                                                         TRapidJsonWriter& writer) const {
    // parse trained models
    TTrainedModelSizeUPtr trainedModelSize;
    if (m_Definition.trainedModel()) {
        m_Definition.trainedModel()->sizeInfo().swap(trainedModelSize);
    }

    // preprocessors
    rapidjson::Value preprocessingArray{writer.makeArray()};
    for (const auto& preprocessor : m_Definition.preprocessors()) {
        auto encodingSizeInfo = preprocessor->sizeInfo();
        rapidjson::Value encodingValue{writer.makeObject()};
        encodingSizeInfo->addToDocument(encodingValue, writer);
        rapidjson::Value encodingEnclosingObject{writer.makeObject()};
        writer.addMember(encodingSizeInfo->typeString(), encodingValue, encodingEnclosingObject);
        preprocessingArray.PushBack(encodingEnclosingObject, writer.getRawAllocator());
    }
    writer.addMember(JSON_PREPROCESSORS_TAG, preprocessingArray, parentObject);
    rapidjson::Value trainedModelSizeObject{writer.makeObject()};
    rapidjson::Value ensembleModelSizeObject{writer.makeObject()};
    trainedModelSize->addToDocument(ensembleModelSizeObject, writer);
    writer.addMember(JSON_ENSEMBLE_MODEL_SIZE_TAG, ensembleModelSizeObject,
                     trainedModelSizeObject);
    writer.addMember(JSON_TRAINED_MODEL_SIZE_TAG, trainedModelSizeObject, parentObject);
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

    rapidjson::Value map{writer.makeObject()};
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

CTargetMeanEncoding::CSizeInfo::CSizeInfo(const CTargetMeanEncoding& encoding)
    : CEncoding::CSizeInfo::CSizeInfo(&encoding), m_Encoding{encoding} {
}

void CTargetMeanEncoding::CSizeInfo::addToDocument(rapidjson::Value& parentObject,
                                                   TRapidJsonWriter& writer) const {
    this->CEncoding::CSizeInfo::addToDocument(parentObject, writer);
    std::size_t featureNameLength{m_Encoding.featureName().size()};
    TSizeVec fieldValueLengths;
    fieldValueLengths.reserve(m_Encoding.targetMap().size());
    for (const auto& item : m_Encoding.targetMap()) {
        fieldValueLengths.push_back(item.first.size());
    }
    writer.addMember(JSON_FEATURE_NAME_LENGTH_TAG,
                     toJson(featureNameLength).Move(), parentObject);
    addJsonArray(JSON_FIELD_VALUE_LENGTHS_TAG, fieldValueLengths, parentObject, writer);
}

CEncoding::TSizeInfoUPtr CTargetMeanEncoding::sizeInfo() const {
    return std::make_unique<CTargetMeanEncoding::CSizeInfo>(*this);
}

const std::string& CTargetMeanEncoding::CSizeInfo::typeString() const {
    return JSON_TARGET_MEAN_ENCODING_TAG;
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

const std::string& CEncoding::field() const {
    return m_Field;
}

CEncoding::CSizeInfo::CSizeInfo(const CEncoding* encoding)
    : m_Encoding(encoding) {
}

void CEncoding::CSizeInfo::addToDocument(rapidjson::Value& parentObject,
                                         TRapidJsonWriter& writer) const {
    writer.addMember(JSON_FIELD_LENGTH_TAG,
                     toJson(m_Encoding->field().size()).Move(), parentObject);
}

const CEncoding* CEncoding::CSizeInfo::encoding() const {
    return m_Encoding;
}

void CFrequencyEncoding::addToDocument(rapidjson::Value& parentObject,
                                       CSerializableToJson::TRapidJsonWriter& writer) const {
    this->CEncoding::addToDocument(parentObject, writer);
    writer.addMember(JSON_FEATURE_NAME_TAG, m_FeatureName, parentObject);
    rapidjson::Value frequencyMap{writer.makeObject()};
    for (const auto& mapping : m_FrequencyMap) {
        writer.addMember(mapping.first, rapidjson::Value(mapping.second).Move(), frequencyMap);
    }
    writer.addMember(JSON_FREQUENCY_MAP_TAG, frequencyMap, parentObject);
}

const std::string& CFrequencyEncoding::CSizeInfo::typeString() const {
    return JSON_FREQUENCY_ENCODING_TAG;
}

const std::string& CFrequencyEncoding::featureName() const {
    return m_FeatureName;
}

const CFrequencyEncoding::TStringDoubleUMap& CFrequencyEncoding::frequencyMap() const {
    return m_FrequencyMap;
}

CFrequencyEncoding::CSizeInfo::CSizeInfo(const CFrequencyEncoding& encoding)
    : CEncoding::CSizeInfo::CSizeInfo(&encoding), m_Encoding{encoding} {
}

void CFrequencyEncoding::CSizeInfo::addToDocument(rapidjson::Value& parentObject,
                                                  TRapidJsonWriter& writer) const {
    this->CEncoding::CSizeInfo::addToDocument(parentObject, writer);
    std::size_t featureNameLength{m_Encoding.featureName().size()};
    TSizeVec fieldValueLengths;
    fieldValueLengths.reserve(m_Encoding.frequencyMap().size());
    for (const auto& item : m_Encoding.frequencyMap()) {
        fieldValueLengths.push_back(item.first.size());
    }
    writer.addMember(JSON_FEATURE_NAME_LENGTH_TAG,
                     toJson(featureNameLength).Move(), parentObject);
    addJsonArray(JSON_FIELD_VALUE_LENGTHS_TAG, fieldValueLengths, parentObject, writer);
}

const std::string& CFrequencyEncoding::typeString() const {
    return JSON_FREQUENCY_ENCODING_TAG;
}

CEncoding::TSizeInfoUPtr CFrequencyEncoding::sizeInfo() const {
    return std::make_unique<CFrequencyEncoding::CSizeInfo>(*this);
}

COneHotEncoding::TStringStringUMap& COneHotEncoding::hotMap() {
    return m_HotMap;
}

const COneHotEncoding::TStringStringUMap& COneHotEncoding::hotMap() const {
    return m_HotMap;
}

const std::string& COneHotEncoding::typeString() const {
    return JSON_ONE_HOT_ENCODING_TAG;
}

void COneHotEncoding::addToDocument(rapidjson::Value& parentObject,
                                    CSerializableToJson::TRapidJsonWriter& writer) const {
    this->CEncoding::addToDocument(parentObject, writer);
    rapidjson::Value hotMap{writer.makeObject()};
    for (const auto& mapping : m_HotMap) {
        writer.addMember(mapping.first, mapping.second, hotMap);
    }
    writer.addMember(JSON_HOT_MAP_TAG, hotMap, parentObject);
}

COneHotEncoding::CSizeInfo::CSizeInfo(const COneHotEncoding& encoding)
    : CEncoding::CSizeInfo::CSizeInfo(&encoding), m_Encoding{encoding} {
}

COneHotEncoding::COneHotEncoding(const std::string& field, TStringStringUMap hotMap)
    : CEncoding(field), m_HotMap(std::move(hotMap)) {
}

void COneHotEncoding::CSizeInfo::addToDocument(rapidjson::Value& parentObject,
                                               TRapidJsonWriter& writer) const {
    this->CEncoding::CSizeInfo::addToDocument(parentObject, writer);
    TSizeVec fieldValueLengths;
    fieldValueLengths.reserve(m_Encoding.hotMap().size());
    TSizeVec featureNameLengths;
    featureNameLengths.reserve(m_Encoding.hotMap().size());
    for (const auto& item : m_Encoding.hotMap()) {
        fieldValueLengths.push_back(item.first.size());
        featureNameLengths.push_back(item.second.size());
    }
    addJsonArray(JSON_FIELD_VALUE_LENGTHS_TAG, fieldValueLengths, parentObject, writer);
    addJsonArray(JSON_FEATURE_NAME_LENGTHS_TAG, featureNameLengths, parentObject, writer);
}

const std::string& COneHotEncoding::CSizeInfo::typeString() const {
    return JSON_ONE_HOT_ENCODING_TAG;
}

CEncoding::TSizeInfoUPtr COneHotEncoding::sizeInfo() const {
    return std::make_unique<COneHotEncoding::CSizeInfo>(*this);
}

CWeightedSum::CWeightedSum(TDoubleVec&& weights)
    : m_Weights{std::move(weights)} {
}
CWeightedSum::CWeightedSum(std::size_t size, double weight)
    : m_Weights(size, weight) {
}

void CWeightedSum::addToDocument(rapidjson::Value& parentObject,
                                 CSerializableToJson::TRapidJsonWriter& writer) const {
    rapidjson::Value object{writer.makeObject()};
    addJsonArray(JSON_WEIGHTS_TAG, m_Weights, object, writer);
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
    rapidjson::Value object{writer.makeObject()};
    addJsonArray(JSON_WEIGHTS_TAG, m_Weights, object, writer);
    writer.addMember(this->stringType(), object, parentObject);
}

CWeightedMode::CWeightedMode(std::size_t size, double weight)
    : m_Weights(size, weight) {
}

CLogisticRegression::CLogisticRegression(TDoubleVec&& weights)
    : m_Weights(std::move(weights)) {
}

CLogisticRegression::CLogisticRegression(std::size_t size, double weight)
    : m_Weights(size, weight) {
}

void CLogisticRegression::addToDocument(rapidjson::Value& parentObject,
                                        TRapidJsonWriter& writer) const {
    rapidjson::Value object{writer.makeObject()};
    addJsonArray(JSON_WEIGHTS_TAG, m_Weights, object, writer);
    writer.addMember(this->stringType(), object, parentObject);
}

const std::string& CLogisticRegression::stringType() const {
    return JSON_LOGISTIC_REGRESSION_TAG;
}
}
}
