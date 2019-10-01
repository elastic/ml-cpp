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
const std::string SPLIT_FEATURE_TAG{"split_feature"};
const std::string DEFAULT_LEFT_TAG{"assign_missing_to_left"};
const std::string NODE_VALUE_TAG{"node_value"};
const std::string SPLIT_INDEX_TAG{"split_index"};
const std::string SPLIT_VALUE_TAG{"split_value"};
const std::string LEFT_CHILD_TAG{"left_child"};
const std::string RIGHT_CHILD_TAG{"right_child"};
const std::string TREE_TAG{"a"};
const std::string TREE_NODE_TAG{"a"};

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
}

bool CTreeNode::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    try {
        do {
            const std::string& name = traverser.name();
            RESTORE(SPLIT_FEATURE_TAG,
                    core::CPersistUtils::restore(SPLIT_FEATURE_TAG, m_SplitFeature, traverser))
            RESTORE(DEFAULT_LEFT_TAG,
                    core::CPersistUtils::restore(DEFAULT_LEFT_TAG, m_DefaultLeft, traverser))
            RESTORE(NODE_VALUE_TAG,
                    core::CPersistUtils::restore(NODE_VALUE_TAG, m_LeafValue, traverser))
            RESTORE(SPLIT_INDEX_TAG,
                    core::CPersistUtils::restore(SPLIT_INDEX_TAG, m_NodeIndex, traverser))
            RESTORE(SPLIT_VALUE_TAG,
                    core::CPersistUtils::restore(SPLIT_VALUE_TAG, m_Threshold, traverser))
            RESTORE(LEFT_CHILD_TAG,
                    core::CPersistUtils::restore(LEFT_CHILD_TAG, m_LeftChild, traverser))
            RESTORE(RIGHT_CHILD_TAG,
                    core::CPersistUtils::restore(RIGHT_CHILD_TAG, m_RightChild, traverser))
            // TODO split_gain is missing

        } while (traverser.next());
    } catch (std::exception& e) {
        LOG_ERROR(<< "Failed to restore state! " << e.what());
        return false;
    }
    return true;
}

void CTreeNode::addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) {
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

bool CEnsemble::acceptRestoreTraverser(ml::core::CStateRestoreTraverser& traverser) {
    auto restoreTree = [this](ml::core::CStateRestoreTraverser& traverser) -> bool {
        CTree tree;
        if (traverser.traverseSubLevel(std::bind(&CTree::acceptRestoreTraverser, &tree,
                                                 std::placeholders::_1)) == true) {
            m_TrainedModels.push_back(tree);
            return true;
        }
        return false;
    };
    do {
        const std::string& name = traverser.name();
        RESTORE(TREE_TAG, restoreTree(traverser))
        // TODO restore aggregated output
    } while (traverser.next());
    return true;
}

void CEnsemble::addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) {
    CBasicEvaluator::addToDocument(parentObject, writer);
    rapidjson::Value trainedModelsArray = writer.makeArray(m_TrainedModels.size());
    for (auto trainedModel : m_TrainedModels) {
        rapidjson::Value trainedModelObject = writer.makeObject();
        trainedModel.addToDocument(trainedModelObject, writer);
        trainedModelsArray.PushBack(trainedModelObject, writer.getRawAllocator());
    }
    writer.addMember(JSON_TRAINED_MODELS_TAG, trainedModelsArray, parentObject);
}

void CEnsemble::featureNames(const CBasicEvaluator::TStringVec& featureNames) {
    for (auto trainedModel : m_TrainedModels) {
        trainedModel.featureNames(featureNames);
    }
}

bool CTree::acceptRestoreTraverser(ml::core::CStateRestoreTraverser& traverser) {
    auto restoreTreeNode = [this](ml::core::CStateRestoreTraverser& traverser) -> bool {
        CTreeNode treeNode;
        if (traverser.traverseSubLevel(std::bind(&CTreeNode::acceptRestoreTraverser, &treeNode,
                                                 std::placeholders::_1)) == true) {
            m_TreeStructure.push_back(treeNode);
            return true;
        }
        return false;
    };
    do {
        const std::string& name = traverser.name();
        RESTORE(TREE_NODE_TAG, restoreTreeNode(traverser))
    } while (traverser.next());
    return true;
}

void CTree::addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) {
    CBasicEvaluator::addToDocument(parentObject, writer);
    rapidjson::Value treeStructureArray = writer.makeArray(m_TreeStructure.size());
    for (auto treeNode : m_TreeStructure) {
        rapidjson::Value treeNodeObject = writer.makeObject();
        treeNode.addToDocument(treeNodeObject, writer);
        treeStructureArray.PushBack(treeNodeObject, writer.getRawAllocator());
    }
    writer.addMember(JSON_TREE_STRUCTURE_TAG, treeStructureArray, parentObject);
}

std::string CInferenceModelDefinition::jsonString() {
    rapidjson::StringBuffer stringBuffer;
    core::CRapidJsonLineWriter<rapidjson::StringBuffer> writer(stringBuffer);
    rapidjson::Value doc = writer.makeObject();

    //input

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
    }
    else {
        LOG_ERROR(<< "Trained model is not initialized");
    }
    writer.write(doc);
    return stringBuffer.GetString();
}

void CBasicEvaluator::addToDocument(TRapidJsonWriter::TValue& parentObject,
                                    TRapidJsonWriter& writer) {
    rapidjson::Value featureNamesArray = writer.makeArray(m_FeatureNames.size());
    for (const auto& featureName : m_FeatureNames) {
        rapidjson::Value featureNameValue;
        featureNameValue.SetString(featureName, writer.getRawAllocator());
        featureNamesArray.PushBack(featureNameValue, writer.getRawAllocator());
    }
    writer.addMember(JSON_FEATURE_NAMES_TAG, featureNamesArray, parentObject);

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

const CBasicEvaluator::TStringVec& CBasicEvaluator::featureNames() const {
    return m_FeatureNames;
}

void CBasicEvaluator::featureNames(const CBasicEvaluator::TStringVec& featureNames) {
    m_FeatureNames = featureNames;
}

void CBasicEvaluator::classificationLabels(const CBasicEvaluator::TStringVec& classificationLabels) {
    m_ClassificationLabels = classificationLabels;
}

void CBasicEvaluator::targetType(CBasicEvaluator::ETargetType targetType) {
    m_TargetType = targetType;
}

void CInferenceModelDefinition::fieldNames(const std::vector<std::string>& fieldNames) {
    m_FieldNames = fieldNames;
    m_Input.columns(fieldNames);

    // TODO this is not correct?
    m_TrainedModel->featureNames(fieldNames);
}

void CInferenceModelDefinition::encodings(const CInferenceModelDefinition::TEncodingUPtrVec& encodings) {
    if (encodings.empty()) {
        return;
    }
    using TOneHotEncodingUPtr = std::unique_ptr<COneHotEncoding>;
    using TOneHotEncodingUMap= std::unordered_map<std::string, TOneHotEncodingUPtr >;
    TOneHotEncodingUMap oneHotEncodingMaps;
    for (const auto& encoding : encodings) {
        std::size_t inputColumnIndex = encoding->inputColumnIndex();
        std::string fieldName = m_FieldNames[inputColumnIndex];
        if (encoding->type() == maths::EEncoding::E_OneHot) {
            std::size_t categoryUInt =
                static_cast<maths::CDataFrameCategoryEncoder::COneHotEncoding*>(
                    encoding.get())
                    ->hotCategory();
            std::string category = m_ReverseCategoryNameMap[inputColumnIndex][categoryUInt];
            std::string encodedFieldName = fieldName + "_" + category;
            if (oneHotEncodingMaps.find(fieldName) == oneHotEncodingMaps.end()) {
               auto  apiEncoding = std::make_unique< api::COneHotEncoding>(fieldName, api::COneHotEncoding::TStringStringUMap());
                oneHotEncodingMaps.emplace(fieldName, std::move(apiEncoding));
            }
            oneHotEncodingMaps[fieldName]->hotMap().emplace(category, encodedFieldName);
        } else if (encoding->type() == maths::EEncoding::E_TargetMean) {
            auto* enc = static_cast<maths::CDataFrameCategoryEncoder::CMappedEncoding*>(
                encoding.get());
            double defaultValue{enc->fallback()};
            std::string featureName{fieldName + "_targetmean"};

            std::map<std::string, double> map;
            for (std::size_t categoryUInt = 0; categoryUInt < enc->map().size(); ++categoryUInt) {
                std::string category = m_ReverseCategoryNameMap[inputColumnIndex][categoryUInt];
                map.emplace(category, enc->map()[categoryUInt]);
            }
            m_Preprocessing.emplace_back(std::make_unique<CTargetMeanEncoding>(fieldName, defaultValue, featureName, map));
        } else if (encoding->type() == maths::EEncoding::E_Frequency) {
            auto* enc = static_cast<maths::CDataFrameCategoryEncoder::CMappedEncoding*>(
                encoding.get());
            std::string featureName{fieldName + "_frequency"};
            std::map<std::string, double> map;
            for (std::size_t categoryUInt = 0; categoryUInt < enc->map().size(); ++categoryUInt) {
                std::string category = m_ReverseCategoryNameMap[inputColumnIndex][categoryUInt];
                map.emplace(category, enc->map()[categoryUInt]);
            }
            m_Preprocessing.emplace_back(std::make_unique<CFrequencyEncoding>(fieldName, featureName, map));
        }
    }

    for (auto& oneHotEncodingMapping : oneHotEncodingMaps) {
        m_Preprocessing.emplace_back(std::move(oneHotEncodingMapping.second));
    }
}

void CInferenceModelDefinition::trainedModel(std::unique_ptr<CBasicEvaluator> &&trainedModel) {
    m_TrainedModel.swap(trainedModel);
}

rapidjson::Value && CInferenceModelDefinition::jsonObject() {
    rapidjson::Document doc;
    doc.Parse(this->jsonString());
    if (doc.GetParseError()) {
        HANDLE_FATAL(<< "Internal error: generated inference model JSON cannot be parsed. "
        << "Please report this problem.")
    }
    return std::move(doc);
}

const CInferenceModelDefinition::TStrSizeUMapVec &CInferenceModelDefinition::categoryNameMap() const {
    return m_CategoryNameMap;
}

void CInferenceModelDefinition::categoryNameMap(const CInferenceModelDefinition::TStrSizeUMapVec &categoryNameMap) {
    m_CategoryNameMap = categoryNameMap;
    m_ReverseCategoryNameMap.reserve(categoryNameMap.size());
    for (const auto& categoryNameMapping: categoryNameMap) {
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

const CInput::TStringVec & CInput::columns() const {
    return m_Columns;
}

void CInput::columns(const TStringVec& columns) {
    m_Columns = columns;
}

void CInput::addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) {
    rapidjson::Value columnsArray = writer.makeArray(m_Columns.size());
    for (const auto &column : m_Columns) {
        rapidjson::Value columnValue;
        columnValue.SetString(column, writer.getRawAllocator());
        columnsArray.PushBack(columnValue, writer.getRawAllocator());
    }
    writer.addMember(JSON_COLUMNS_TAG, columnsArray, parentObject);
}

void CTargetMeanEncoding::defaultValue(double defaultValue) {
    m_DefaultValue = defaultValue;
}

void CTargetMeanEncoding::featureName(const std::string& featureName) {
    m_FeatureName = featureName;
}

void CTargetMeanEncoding::targetMap(const std::map<std::string, double>& targetMap) {
    m_TargetMap = targetMap;
}

const std::string &CTargetMeanEncoding::typeString() const {
    return JSON_TARGET_MEAN_ENCODING_TAG;
}

void CTargetMeanEncoding::addToDocument(rapidjson::Value &parentObject, CSerializableToJson::TRapidJsonWriter &writer) {
    this->CEncoding::addToDocument(parentObject, writer);
    writer.addMember(JSON_DEFAULT_VALUE_TAG, rapidjson::Value(m_DefaultValue).Move(), parentObject);
    writer.addMember(JSON_FEATURE_NAME_TAG, m_FeatureName, parentObject);

    rapidjson::Value map = writer.makeObject();
    for (const auto& mapping: m_TargetMap) {
        writer.addMember(mapping.first, rapidjson::Value(mapping.second).Move(), map);
    }
    writer.addMember(JSON_TARGET_MAP_TAG, map, parentObject);
}

CTargetMeanEncoding::CTargetMeanEncoding(const std::string &field, double defaultValue, const std::string &featureName,
                                         const std::map<std::string, double> &targetMap) : CEncoding(field),
                                                                                           m_DefaultValue(defaultValue),
                                                                                           m_FeatureName(featureName),
                                                                                           m_TargetMap(targetMap) {}

CFrequencyEncoding::CFrequencyEncoding(const std::string &field, const std::string &featureName,
                                       const std::map<std::string, double> &frequencyMap):CEncoding(field),
                                                                                          m_FeatureName(featureName),
                                                                                          m_FrequencyMap(
                                                                                                  frequencyMap) {}

void CEncoding::field(const std::string& field) {
    m_Field = field;
}

void CEncoding::addToDocument(rapidjson::Value& parentObject, TRapidJsonWriter& writer) {
    writer.addMember(JSON_FIELD_TAG, m_Field, parentObject);
}

CEncoding::CEncoding(const std::string &field) : m_Field(field) {}

void CFrequencyEncoding::featureName(const std::string& featureName) {
    m_FeatureName = featureName;
}

void CFrequencyEncoding::frequencyMap(const std::map<std::string, double>& frequencyMap) {
    m_FrequencyMap = frequencyMap;
}

void CFrequencyEncoding::addToDocument(rapidjson::Value &parentObject, CSerializableToJson::TRapidJsonWriter &writer) {
    this->CEncoding::addToDocument(parentObject, writer);
    writer.addMember(JSON_FEATURE_NAME_TAG, m_FeatureName, parentObject);
    rapidjson::Value frequencyMap = writer.makeObject();
    for (const auto& mapping: m_FrequencyMap) {
        writer.addMember(mapping.first, rapidjson::Value(mapping.second).Move(), frequencyMap);
    }
    writer.addMember(JSON_FREQUENCY_MAP_TAG, frequencyMap, parentObject);
}

const std::string &CFrequencyEncoding::typeString() const {
    return JSON_FREQUENCY_ENCODING_TAG;
}

COneHotEncoding::TStringStringUMap & COneHotEncoding::hotMap() {
    return m_HotMap;
}

const std::string &COneHotEncoding::typeString() const {
    return JSON_ONE_HOT_ENCODING_TAG;
}

void COneHotEncoding::addToDocument(rapidjson::Value &parentObject, CSerializableToJson::TRapidJsonWriter &writer) {
    this->CEncoding::addToDocument(parentObject, writer);
    rapidjson::Value hotMap = writer.makeObject();
    for (const auto& mapping: m_HotMap) {
        writer.addMember(mapping.first, mapping.second, hotMap);
    }
    writer.addMember(JSON_HOT_MAP_TAG, hotMap, parentObject);
}

COneHotEncoding::COneHotEncoding(const std::string &field, const COneHotEncoding::TStringStringUMap &hotMap)
        : CEncoding(field), m_HotMap(hotMap) {}
}
}
