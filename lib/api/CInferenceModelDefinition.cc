/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CInferenceModelDefinition.h>

#include <core/CPersistUtils.h>
#include <core/CRapidJsonLineWriter.h>
#include <core/RestoreMacros.h>

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

const std::string JSON_FEATURE_NAMES_TAG{"feature_names"};
const std::string JSON_TRAINED_MODELS_TAG{"trained_models"};
const std::string JSON_TRAINED_MODEL_TAG{"trained_model"};
const std::string JSON_TREE_STRUCTURE_TAG{"tree_structure"};
const std::string JSON_NODE_INDEX_TAG{"node_index"};
const std::string JSON_SPLIT_FEATURE_TAG{"split_feature"};
const std::string JSON_SPLIT_GAIN_TAG{"split_gain"};
const std::string JSON_THRESHOLD_TAG{"threshold"};
const std::string JSON_LEAF_VALUE_TAG{"leaf_value"};
const std::string JSON_DEFAULT_LEFT_TAG{"default_left"};
const std::string JSON_DECISION_TYPE_TAG{"decision_type"};
const std::string JSON_LEFT_CHILD_TAG{"left_child"};
const std::string JSON_RIGHT_CHILD_TAG{"right_child"};
const std::string JSON_LTE{"lte"};
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
    if (m_TrainedModel) {
        rapidjson::Value trainedModelValue = writer.makeObject();
        m_TrainedModel->addToDocument(trainedModelValue, writer);
        writer.addMember(JSON_TRAINED_MODEL_TAG, trainedModelValue, doc);
    }
    writer.write(doc);
    return stringBuffer.GetString();
}

void CBasicEvaluator::addToDocument(TRapidJsonWriter::TValue& parentObject,
                                    TRapidJsonWriter& writer) {
    rapidjson::Value featureNamesArray = writer.makeArray(m_FeatureNames.size());
    for (auto featureName : m_FeatureNames) {
        rapidjson::Value featureNameValue;
        featureNameValue.SetString(featureName, writer.getRawAllocator());
        featureNamesArray.PushBack(featureNameValue, writer.getRawAllocator());
    }

    writer.addMember(JSON_FEATURE_NAMES_TAG, featureNamesArray, parentObject);
}

const CBasicEvaluator::TStringVec& CBasicEvaluator::featureNames() const {
    return m_FeatureNames;
}

void CBasicEvaluator::featureNames(const CBasicEvaluator::TStringVec& featureNames) {
    m_FeatureNames = featureNames;
}

void CInferenceModelDefinition::fieldNames(const std::vector<std::string>& fieldNames) {
    m_Input.columns(fieldNames);
    m_TrainedModel.get(fieldNames);
}

const CInput::TStringVecOptional& CInput::columns() const {
    return m_Columns;
}

void CInput::columns(const TStringVec& columns) {
    m_Columns = columns;
}
}
}
