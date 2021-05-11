/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CInferenceModelDefinition.h>

#include <core/CPersistUtils.h>
#include <core/CStringUtils.h>

#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

#include <cmath>
#include <iterator>
#include <memory>

namespace ml {
namespace api {

namespace {

auto toRapidjsonValue(std::size_t value) {
    return rapidjson::Value{static_cast<std::uint64_t>(value)};
}

void addJsonArray(const std::string& tag,
                  const std::vector<std::size_t>& vector,
                  rapidjson::Value& parentObject,
                  CSerializableToJsonDocument::TRapidJsonWriter& writer) {
    rapidjson::Value array{writer.makeArray(vector.size())};
    for (const auto& value : vector) {
        array.PushBack(static_cast<std::uint64_t>(value), writer.getRawAllocator());
    }
    writer.addMember(tag, array, parentObject);
}

void addJsonArray(const std::string& tag,
                  const std::vector<double>& vector,
                  CSerializableToJsonStream::TGenericLineWriter& writer) {
    writer.Key(tag);
    writer.StartArray();
    for (const auto& value : vector) {
        writer.Double(value);
    }
    writer.EndArray();
}

void addJsonArray(const std::string& tag,
                  const std::vector<std::string>& vector,
                  CSerializableToJsonStream::TGenericLineWriter& writer) {
    writer.Key(tag);
    writer.StartArray();
    for (const auto& value : vector) {
        writer.String(value);
    }
    writer.EndArray();
}
}

void CTree::CTreeNode::addToJsonStream(TGenericLineWriter& writer) const {
    writer.Key(JSON_NODE_INDEX_TAG);
    writer.Uint64(m_NodeIndex);
    writer.Key(JSON_NUMBER_SAMPLES_TAG);
    writer.Uint64(m_NumberSamples);

    if (m_LeftChild) {
        // internal node
        writer.Key(JSON_SPLIT_FEATURE_TAG);
        writer.Uint64(m_SplitFeature);
        if (m_SplitGain.is_initialized()) {
            writer.Key(JSON_SPLIT_GAIN_TAG);
            writer.Double(m_SplitGain.get());
        }
        writer.Key(JSON_THRESHOLD_TAG);
        writer.Double(m_Threshold);
        writer.Key(JSON_DEFAULT_LEFT_TAG);
        writer.Bool(m_DefaultLeft);
        switch (m_DecisionType) {
        case E_LT:
            writer.Key(JSON_DECISION_TYPE_TAG);
            writer.String(JSON_LT);
            break;
        }
        writer.Key(JSON_LEFT_CHILD_TAG);
        writer.Uint(m_LeftChild.get());
        writer.Key(JSON_RIGHT_CHILD_TAG);
        writer.Uint(m_RightChild.get());
    } else if (m_LeafValue.size() > 1) {
        // leaf node
        addJsonArray(JSON_LEAF_VALUE_TAG, m_LeafValue, writer);
    } else {
        // leaf node
        writer.Key(JSON_LEAF_VALUE_TAG);
        writer.Double(m_LeafValue[0]);
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

void CTree::CSizeInfo::addToJsonDocument(rapidjson::Value& parentObject,
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
    writer.addMember(JSON_NUM_NODES_TAG, toRapidjsonValue(numNodes).Move(), parentObject);
    writer.addMember(JSON_NUM_LEAVES_TAG, toRapidjsonValue(numLeaves).Move(), parentObject);
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

void CEnsemble::addToJsonStream(TGenericLineWriter& writer) const {
    writer.Key(JSON_ENSEMBLE_TAG);
    writer.StartObject();
    this->CTrainedModel::addToJsonStream(writer);
    writer.Key(JSON_TRAINED_MODELS_TAG);
    writer.StartArray();
    for (const auto& trainedModel : m_TrainedModels) {
        writer.StartObject();
        trainedModel->addToJsonStream(writer);
        writer.EndObject();
    }
    writer.EndArray();

    // aggregate output
    writer.Key(JSON_AGGREGATE_OUTPUT_TAG);
    m_AggregateOutput->addToJsonStream(writer);
    writer.EndObject();
}

void CEnsemble::featureNames(TStringVec featureNames) {
    for (auto& trainedModel : m_TrainedModels) {
        trainedModel->featureNames(featureNames);
    }
    this->CTrainedModel::featureNames(std::move(featureNames));
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

const CEnsemble::TStringVec& CEnsemble::removeUnusedFeatures() {
    boost::unordered_set<std::string> uniqueUsedFeatures;
    for (auto& trainedModel : this->trainedModels()) {
        const auto& usedFeatures = trainedModel->removeUnusedFeatures();
        uniqueUsedFeatures.insert(usedFeatures.begin(), usedFeatures.end());
    }
    TStringVec usedFeatures(uniqueUsedFeatures.begin(), uniqueUsedFeatures.end());
    std::sort(usedFeatures.begin(), usedFeatures.end());
    // It is very important that we call the base class method here because we don't
    // want to reset feature names of the individual models which may be referenced
    // by position in their feature name vector.
    this->CTrainedModel::featureNames(std::move(usedFeatures));
    return this->featureNames();
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

void CEnsemble::CSizeInfo::addToJsonDocument(rapidjson::Value& parentObject,
                                             TRapidJsonWriter& writer) const {
    this->CTrainedModel::CSizeInfo::addToJsonDocument(parentObject, writer);
    rapidjson::Value featureNameLengthsArray{
        writer.makeArray(m_Ensemble->featureNames().size())};
    for (const auto& featureName : m_Ensemble->featureNames()) {
        featureNameLengthsArray.PushBack(
            toRapidjsonValue(core::CStringUtils::utf16LengthOfUtf8String(featureName))
                .Move(),
            writer.getRawAllocator());
    }
    writer.addMember(JSON_FEATURE_NAME_LENGTHS_TAG, featureNameLengthsArray, parentObject);

    rapidjson::Value treeSizesArray{writer.makeArray(m_Ensemble->m_TrainedModels.size())};
    for (const auto& trainedModel : m_Ensemble->m_TrainedModels) {
        rapidjson::Value item{writer.makeObject()};
        trainedModel->sizeInfo()->addToJsonDocument(item, writer);
        treeSizesArray.PushBack(item, writer.getRawAllocator());
    }
    writer.addMember(JSON_TREE_SIZES_TAG, treeSizesArray, parentObject);

    std::size_t numOutputProcessorWeights{m_Ensemble->m_TrainedModels.size()};
    writer.addMember(JSON_NUM_OUTPUT_PROCESSOR_WEIGHTS_TAG,
                     toRapidjsonValue(numOutputProcessorWeights).Move(), parentObject);
    std::size_t numOperations{this->numOperations()};
    writer.addMember(JSON_NUM_OPERATIONS_TAG,
                     toRapidjsonValue(numOperations).Move(), parentObject);
}

void CTree::addToJsonStream(TGenericLineWriter& writer) const {
    writer.Key(JSON_TREE_TAG);
    writer.StartObject();
    this->CTrainedModel::addToJsonStream(writer);
    writer.Key(JSON_TREE_STRUCTURE_TAG);
    writer.StartArray();
    for (const auto& treeNode : m_TreeStructure) {
        writer.StartObject();
        treeNode.addToJsonStream(writer);
        writer.EndObject();
    }
    writer.EndArray();
    writer.EndObject();
}

std::size_t CTree::size() const {
    return m_TreeStructure.size();
}

CTree::TTreeNodeVec& CTree::treeStructure() {
    return m_TreeStructure;
}

const CTree::TStringVec& CTree::removeUnusedFeatures() {
    boost::unordered_map<std::size_t, std::size_t> usedFeatureIndices;
    for (auto& treeNode : m_TreeStructure) {
        if (treeNode.leaf() == false) {
            std::size_t adjustedIndex{
                usedFeatureIndices
                    .emplace(treeNode.splitFeature(), usedFeatureIndices.size())
                    .first->second};
            treeNode.splitFeature(adjustedIndex);
        }
    }
    TStringVec usedFeatures(usedFeatureIndices.size());
    auto& featureNames = this->featureNames();
    for (const auto& index : usedFeatureIndices) {
        usedFeatures[index.second] = std::move(featureNames[index.first]);
    }
    this->featureNames(std::move(usedFeatures));
    return this->featureNames();
}

std::string CInferenceModelDefinition::jsonString() const {
    std::ostringstream jsonStrm;
    this->jsonStream(jsonStrm);
    return jsonStrm.str();
}

void CInferenceModelDefinition::addToDocumentCompressed(TRapidJsonWriter& writer) const {
    CSerializableToJsonDocumentCompressed::addToDocumentCompressed(
        writer, JSON_COMPRESSED_INFERENCE_MODEL_TAG, JSON_DEFINITION_TAG);
}

void CInferenceModelDefinition::addToJsonStream(TGenericLineWriter& writer) const {
    writer.StartObject();
    // preprocessors
    writer.Key(JSON_PREPROCESSORS_TAG);
    writer.StartArray();
    for (const auto& customEncoding : m_CustomPreprocessors) {
        customEncoding->addToJsonStream(writer);
    }
    for (const auto& encoding : m_Preprocessors) {
        writer.StartObject();
        writer.Key(encoding->typeString());
        writer.StartObject();
        encoding->addToJsonStream(writer);
        writer.EndObject();
        writer.EndObject();
    }
    writer.EndArray();

    // trained_model
    if (m_TrainedModel) {
        writer.Key(JSON_TRAINED_MODEL_TAG);
        writer.StartObject();
        m_TrainedModel->addToJsonStream(writer);
        writer.EndObject();
    } else {
        LOG_ERROR(<< "Trained model is not initialized");
    }
    writer.EndObject();
}

void CTrainedModel::addToJsonStream(TGenericLineWriter& writer) const {
    addJsonArray(JSON_FEATURE_NAMES_TAG, m_FeatureNames, writer);

    if (m_ClassificationLabels) {
        addJsonArray(JSON_CLASSIFICATION_LABELS_TAG, *m_ClassificationLabels, writer);
    }

    if (m_ClassificationWeights) {
        addJsonArray(JSON_CLASSIFICATION_WEIGHTS_TAG, *m_ClassificationWeights, writer);
    }

    switch (m_TargetType) {
    case E_Classification:
        writer.Key(JSON_TARGET_TYPE_TAG);
        writer.String(JSON_TARGET_TYPE_CLASSIFICATION);
        break;
    case E_Regression:
        writer.Key(JSON_TARGET_TYPE_TAG);
        writer.String(JSON_TARGET_TYPE_REGRESSION);
        break;
    }
}

const CTrainedModel::TStringVec& CTrainedModel::featureNames() const {
    return m_FeatureNames;
}

CTrainedModel::TStringVec& CTrainedModel::featureNames() {
    return m_FeatureNames;
}

void CTrainedModel::featureNames(TStringVec featureNames) {
    m_FeatureNames = std::move(featureNames);
}

void CTrainedModel::targetType(ETargetType targetType) {
    m_TargetType = targetType;
}

CTrainedModel::ETargetType CTrainedModel::targetType() const {
    return m_TargetType;
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

CTrainedModel::CFeatureNameProvider::CFeatureNameProvider(TStrVec fieldNames, TStrVecVec categoryNames)
    : m_FieldNames{std::move(fieldNames)}, m_CategoryNames{std::move(categoryNames)} {
}

const std::string&
CTrainedModel::CFeatureNameProvider::fieldName(std::size_t inputColumnIndex) const {
    return m_FieldNames[inputColumnIndex];
}

const std::string&
CTrainedModel::CFeatureNameProvider::category(std::size_t inputColumnIndex,
                                              std::size_t hotCategory) const {
    return m_CategoryNames[inputColumnIndex][hotCategory];
}

std::string CTrainedModel::CFeatureNameProvider::identityEncodingName(std::size_t inputColumnIndex) const {
    return m_FieldNames[inputColumnIndex];
}

std::string
CTrainedModel::CFeatureNameProvider::oneHotEncodingName(std::size_t inputColumnIndex,
                                                        std::size_t hotCategory) const {
    return m_FieldNames[inputColumnIndex] + "_" +
           m_CategoryNames[inputColumnIndex][hotCategory];
}

std::string CTrainedModel::CFeatureNameProvider::targetMeanEncodingName(std::size_t inputColumnIndex) const {
    return m_FieldNames[inputColumnIndex] + "_targetmean";
}

std::string CTrainedModel::CFeatureNameProvider::frequencyEncodingName(std::size_t inputColumnIndex) const {
    return m_FieldNames[inputColumnIndex] + "_frequency";
}

CTrainedModel::CSizeInfo::CSizeInfo(const CTrainedModel& trainedModel)
    : m_TrainedModel{trainedModel} {
}

void CTrainedModel::CSizeInfo::addToJsonDocument(rapidjson::Value& parentObject,
                                                 TRapidJsonWriter& writer) const {
    if (m_TrainedModel.targetType() == E_Classification) {
        writer.addMember(
            JSON_NUM_CLASSIFICATION_WEIGHTS_TAG,
            toRapidjsonValue(m_TrainedModel.classificationWeights()->size()).Move(),
            parentObject);
        writer.addMember(
            JSON_NUM_CLASSES_TAG,
            toRapidjsonValue(m_TrainedModel.classificationLabels()->size()).Move(),
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

CInferenceModelDefinition::TApiCustomEncodingUPtrVec&
CInferenceModelDefinition::customPreprocessors() {
    return m_CustomPreprocessors;
}

const std::string& CInferenceModelDefinition::typeString() const {
    return JSON_COMPRESSED_INFERENCE_MODEL_TAG;
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
        TRapidJsonWriter writer{wrapper};
        rapidjson::Value doc{writer.makeObject()};
        this->addToJsonDocument(doc, writer);
        writer.write(doc);
        stream.flush();
    }
    // string writer puts the json object in an array, so we strip the external brackets
    std::string jsonStr{stream.str()};
    std::string resultString(jsonStr, 1, jsonStr.size() - 2);
    return resultString;
}

const std::string& CInferenceModelDefinition::CSizeInfo::typeString() const {
    return JSON_MODEL_SIZE_INFO_TAG;
}

void CInferenceModelDefinition::CSizeInfo::addToJsonDocument(rapidjson::Value& parentObject,
                                                             TRapidJsonWriter& writer) const {
    using TTrainedModelSizeUPtr = std::unique_ptr<CTrainedModel::CSizeInfo>;

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
        encodingSizeInfo->addToJsonDocument(encodingValue, writer);
        rapidjson::Value encodingEnclosingObject{writer.makeObject()};
        writer.addMember(encodingSizeInfo->typeString(), encodingValue, encodingEnclosingObject);
        preprocessingArray.PushBack(encodingEnclosingObject, writer.getRawAllocator());
    }
    writer.addMember(JSON_PREPROCESSORS_TAG, preprocessingArray, parentObject);
    rapidjson::Value trainedModelSizeObject{writer.makeObject()};
    rapidjson::Value ensembleModelSizeObject{writer.makeObject()};
    trainedModelSize->addToJsonDocument(ensembleModelSizeObject, writer);
    writer.addMember(JSON_ENSEMBLE_MODEL_SIZE_TAG, ensembleModelSizeObject,
                     trainedModelSizeObject);
    writer.addMember(JSON_TRAINED_MODEL_SIZE_TAG, trainedModelSizeObject, parentObject);
}

const std::string& CTargetMeanEncoding::typeString() const {
    return JSON_TARGET_MEAN_ENCODING_TAG;
}

void CTargetMeanEncoding::addToJsonStream(TGenericLineWriter& writer) const {
    this->CEncoding::addToJsonStream(writer);
    writer.Key(JSON_DEFAULT_VALUE_TAG);
    writer.Double(m_DefaultValue);
    writer.Key(JSON_FEATURE_NAME_TAG);
    writer.String(m_FeatureName);

    writer.Key(JSON_TARGET_MAP_TAG);
    writer.StartObject();
    for (const auto& mapping : m_TargetMap) {
        writer.Key(mapping.first);
        writer.Double(mapping.second);
    }
    writer.EndObject();
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

void CTargetMeanEncoding::CSizeInfo::addToJsonDocument(rapidjson::Value& parentObject,
                                                       TRapidJsonWriter& writer) const {
    this->CEncoding::CSizeInfo::addToJsonDocument(parentObject, writer);
    std::size_t featureNameLength{
        core::CStringUtils::utf16LengthOfUtf8String(m_Encoding.featureName())};
    TSizeVec fieldValueLengths;
    fieldValueLengths.reserve(m_Encoding.targetMap().size());
    for (const auto& item : m_Encoding.targetMap()) {
        fieldValueLengths.push_back(core::CStringUtils::utf16LengthOfUtf8String(item.first));
    }
    writer.addMember(JSON_FEATURE_NAME_LENGTH_TAG,
                     toRapidjsonValue(featureNameLength).Move(), parentObject);
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

void CEncoding::addToJsonStream(TGenericLineWriter& writer) const {
    writer.Key(JSON_FIELD_TAG);
    writer.String(m_Field);
}

CEncoding::CEncoding(std::string field) : m_Field(std::move(field)) {
}

const std::string& CEncoding::field() const {
    return m_Field;
}

CEncoding::CSizeInfo::CSizeInfo(const CEncoding* encoding)
    : m_Encoding(encoding) {
}

void CEncoding::CSizeInfo::addToJsonDocument(rapidjson::Value& parentObject,
                                             TRapidJsonWriter& writer) const {
    writer.addMember(
        JSON_FIELD_LENGTH_TAG,
        toRapidjsonValue(core::CStringUtils::utf16LengthOfUtf8String(m_Encoding->field()))
            .Move(),
        parentObject);
}

const CEncoding* CEncoding::CSizeInfo::encoding() const {
    return m_Encoding;
}

void CFrequencyEncoding::addToJsonStream(TGenericLineWriter& writer) const {
    this->CEncoding::addToJsonStream(writer);
    writer.Key(JSON_FEATURE_NAME_TAG);
    writer.String(m_FeatureName);
    writer.Key(JSON_FREQUENCY_MAP_TAG);
    writer.StartObject();
    for (const auto& mapping : m_FrequencyMap) {
        writer.Key(mapping.first);
        writer.Double(mapping.second);
    }
    writer.EndObject();
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

void CFrequencyEncoding::CSizeInfo::addToJsonDocument(rapidjson::Value& parentObject,
                                                      TRapidJsonWriter& writer) const {
    this->CEncoding::CSizeInfo::addToJsonDocument(parentObject, writer);
    std::size_t featureNameLength{
        core::CStringUtils::utf16LengthOfUtf8String(m_Encoding.featureName())};
    TSizeVec fieldValueLengths;
    fieldValueLengths.reserve(m_Encoding.frequencyMap().size());
    for (const auto& item : m_Encoding.frequencyMap()) {
        fieldValueLengths.push_back(core::CStringUtils::utf16LengthOfUtf8String(item.first));
    }
    writer.addMember(JSON_FEATURE_NAME_LENGTH_TAG,
                     toRapidjsonValue(featureNameLength).Move(), parentObject);
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

void COneHotEncoding::addToJsonStream(TGenericLineWriter& writer) const {
    this->CEncoding::addToJsonStream(writer);
    writer.Key(JSON_HOT_MAP_TAG);
    writer.StartObject();
    for (const auto& mapping : m_HotMap) {
        writer.Key(mapping.first);
        writer.String(mapping.second);
    }
    writer.EndObject();
}

COneHotEncoding::CSizeInfo::CSizeInfo(const COneHotEncoding& encoding)
    : CEncoding::CSizeInfo::CSizeInfo(&encoding), m_Encoding{encoding} {
}

COneHotEncoding::COneHotEncoding(const std::string& field, TStringStringUMap hotMap)
    : CEncoding(field), m_HotMap(std::move(hotMap)) {
}

void COneHotEncoding::CSizeInfo::addToJsonDocument(rapidjson::Value& parentObject,
                                                   TRapidJsonWriter& writer) const {
    this->CEncoding::CSizeInfo::addToJsonDocument(parentObject, writer);
    TSizeVec fieldValueLengths;
    fieldValueLengths.reserve(m_Encoding.hotMap().size());
    TSizeVec featureNameLengths;
    featureNameLengths.reserve(m_Encoding.hotMap().size());
    for (const auto& item : m_Encoding.hotMap()) {
        fieldValueLengths.push_back(core::CStringUtils::utf16LengthOfUtf8String(item.first));
        featureNameLengths.push_back(core::CStringUtils::utf16LengthOfUtf8String(item.second));
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

COpaqueEncoding::COpaqueEncoding(const rapidjson::Document& object) {
    m_Object.CopyFrom(object, m_Object.GetAllocator());
}

void COpaqueEncoding::addToJsonStream(TGenericLineWriter& writer) const {

    if (m_Object.IsArray()) {
        // These are prepended to the array of other encoders so we don't wrap in
        // a StartArray and EndArray.
        for (const auto& val : m_Object.GetArray()) {
            writer.write(val);
        }
    } else if (m_Object.IsObject() && m_Object.ObjectEmpty() == false) {
        writer.write(m_Object);
    }
}

CWeightedSum::CWeightedSum(TDoubleVec&& weights)
    : m_Weights{std::move(weights)} {
}
CWeightedSum::CWeightedSum(std::size_t size, double weight)
    : m_Weights(size, weight) {
}

void CWeightedSum::addToJsonStream(TGenericLineWriter& writer) const {
    writer.StartObject();
    writer.Key(this->stringType());
    writer.StartObject();
    addJsonArray(JSON_WEIGHTS_TAG, m_Weights, writer);
    writer.EndObject();
    writer.EndObject();
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

void CWeightedMode::addToJsonStream(TGenericLineWriter& writer) const {
    writer.StartObject();
    writer.Key(this->stringType());
    addJsonArray(JSON_WEIGHTS_TAG, m_Weights, writer);
    writer.EndObject();
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

void CLogisticRegression::addToJsonStream(TGenericLineWriter& writer) const {
    writer.StartObject();
    writer.Key(this->stringType());
    writer.StartObject();
    addJsonArray(JSON_WEIGHTS_TAG, m_Weights, writer);
    writer.EndObject();
    writer.EndObject();
}

const std::string& CLogisticRegression::stringType() const {
    return JSON_LOGISTIC_REGRESSION_TAG;
}

CExponent::CExponent(TDoubleVec&& weights) : m_Weights(std::move(weights)) {
}

CExponent::CExponent(std::size_t size, double weight)
    : m_Weights(size, weight) {
}

void CExponent::addToJsonStream(TGenericLineWriter& writer) const {
    writer.StartObject();
    writer.Key(this->stringType());
    writer.StartObject();
    addJsonArray(JSON_WEIGHTS_TAG, m_Weights, writer);
    writer.EndObject();
    writer.EndObject();
}

const std::string& CExponent::stringType() const {
    return JSON_EXPONENT_TAG;
}

// clang-format off
const std::string CAggregateOutput::JSON_WEIGHTS_TAG{"weights"};
const std::string CEncoding::CSizeInfo::JSON_FEATURE_NAME_LENGTH_TAG{"feature_name_length"};
const std::string CEncoding::CSizeInfo::JSON_FIELD_LENGTH_TAG{"field_length"};
const std::string CEncoding::CSizeInfo::JSON_FIELD_VALUE_LENGTHS_TAG{"field_value_lengths"};
const std::string CEncoding::JSON_FEATURE_NAME_TAG{"feature_name"};
const std::string CEncoding::JSON_FIELD_TAG{"field"};
const std::string CEnsemble::CSizeInfo::JSON_FEATURE_NAME_LENGTHS_TAG{"feature_name_lengths"};
const std::string CEnsemble::CSizeInfo::JSON_NUM_OPERATIONS_TAG{"num_operations"};
const std::string CEnsemble::CSizeInfo::JSON_NUM_OUTPUT_PROCESSOR_WEIGHTS_TAG{"num_output_processor_weights"};
const std::string CEnsemble::CSizeInfo::JSON_TREE_SIZES_TAG{"tree_sizes"};
const std::string CEnsemble::JSON_AGGREGATE_OUTPUT_TAG{"aggregate_output"};
const std::string CEnsemble::JSON_ENSEMBLE_TAG{"ensemble"};
const std::string CEnsemble::JSON_TRAINED_MODELS_TAG{"trained_models"};
const std::string CExponent::JSON_EXPONENT_TAG{"exponent"};
const std::string CFrequencyEncoding::JSON_FREQUENCY_ENCODING_TAG{"frequency_encoding"};
const std::string CFrequencyEncoding::JSON_FREQUENCY_MAP_TAG{"frequency_map"};
const std::string CInferenceModelDefinition::CSizeInfo::JSON_ENSEMBLE_MODEL_SIZE_TAG{"ensemble_model_size"};
const std::string CInferenceModelDefinition::CSizeInfo::JSON_MODEL_SIZE_INFO_TAG{"model_size_info"};
const std::string CInferenceModelDefinition::CSizeInfo::JSON_TRAINED_MODEL_SIZE_TAG{"trained_model_size"};
const std::string CInferenceModelDefinition::JSON_COMPRESSED_INFERENCE_MODEL_TAG{"compressed_inference_model"};
const std::string CInferenceModelDefinition::JSON_DEFINITION_TAG{"definition"};
const std::string CInferenceModelDefinition::JSON_PREPROCESSORS_TAG{"preprocessors"};
const std::string CInferenceModelDefinition::JSON_TRAINED_MODEL_TAG{"trained_model"};
const std::string CLogisticRegression::JSON_LOGISTIC_REGRESSION_TAG{"logistic_regression"};
const std::string COneHotEncoding::CSizeInfo::JSON_FEATURE_NAME_LENGTHS_TAG{"feature_name_lengths"};
const std::string COneHotEncoding::JSON_HOT_MAP_TAG{"hot_map"};
const std::string COneHotEncoding::JSON_ONE_HOT_ENCODING_TAG{"one_hot_encoding"};
const std::string CTargetMeanEncoding::JSON_DEFAULT_VALUE_TAG{"default_value"};
const std::string CTargetMeanEncoding::JSON_TARGET_MAP_TAG{"target_map"};
const std::string CTargetMeanEncoding::JSON_TARGET_MEAN_ENCODING_TAG{"target_mean_encoding"};
const std::string CTrainedModel::CSizeInfo::JSON_NUM_CLASSES_TAG{"num_classes"};
const std::string CTrainedModel::CSizeInfo::JSON_NUM_CLASSIFICATION_WEIGHTS_TAG{"num_classification_weights"};
const std::string CTrainedModel::JSON_CLASSIFICATION_LABELS_TAG{"classification_labels"};
const std::string CTrainedModel::JSON_CLASSIFICATION_WEIGHTS_TAG{"classification_weights"};
const std::string CTrainedModel::JSON_FEATURE_NAMES_TAG{"feature_names"};
const std::string CTrainedModel::JSON_TARGET_TYPE_CLASSIFICATION{"classification"};
const std::string CTrainedModel::JSON_TARGET_TYPE_REGRESSION{"regression"};
const std::string CTrainedModel::JSON_TARGET_TYPE_TAG{"target_type"};
const std::string CTree::CSizeInfo::JSON_NUM_LEAVES_TAG{"num_leaves"};
const std::string CTree::CSizeInfo::JSON_NUM_NODES_TAG{"num_nodes"};
const std::string CTree::CTreeNode::JSON_DECISION_TYPE_TAG{"decision_type"};
const std::string CTree::CTreeNode::JSON_DEFAULT_LEFT_TAG{"default_left"};
const std::string CTree::CTreeNode::JSON_LEAF_VALUE_TAG{"leaf_value"};
const std::string CTree::CTreeNode::JSON_LEFT_CHILD_TAG{"left_child"};
const std::string CTree::CTreeNode::JSON_LT{"lt"};
const std::string CTree::CTreeNode::JSON_NODE_INDEX_TAG{"node_index"};
const std::string CTree::CTreeNode::JSON_NUMBER_SAMPLES_TAG{"number_samples"};
const std::string CTree::CTreeNode::JSON_RIGHT_CHILD_TAG{"right_child"};
const std::string CTree::CTreeNode::JSON_SPLIT_FEATURE_TAG{"split_feature"};
const std::string CTree::CTreeNode::JSON_SPLIT_GAIN_TAG{"split_gain"};
const std::string CTree::CTreeNode::JSON_THRESHOLD_TAG{"threshold"};
const std::string CTree::JSON_TREE_STRUCTURE_TAG{"tree_structure"};
const std::string CTree::JSON_TREE_TAG{"tree"};
const std::string CWeightedMode::JSON_WEIGHTED_MODE_TAG{"weighted_mode"};
const std::string CWeightedSum::JSON_WEIGHTED_SUM_TAG{"weighted_sum"};
// clang-format on
}
}
