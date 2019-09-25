/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/SInferenceModelDefinition.h>
#include <core/RestoreMacros.h>

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
}

bool ml::api::STreeNode::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser) {
    try {
        do {
            const std::string& name = traverser.name();
            RESTORE(SPLIT_FEATURE_TAG,
                    core::CPersistUtils::restore(SPLIT_FEATURE_TAG, m_SplitIndex, traverser))
            RESTORE(DEFAULT_LEFT_TAG,
                    core::CPersistUtils::restore(DEFAULT_LEFT_TAG, m_DefaultLeft, traverser))
            RESTORE(NODE_VALUE_TAG,
                    core::CPersistUtils::restore(NODE_VALUE_TAG, m_NodeValue, traverser))
            RESTORE(SPLIT_INDEX_TAG,
                    core::CPersistUtils::restore(SPLIT_INDEX_TAG, m_SplitIndex, traverser))
            RESTORE(SPLIT_VALUE_TAG,
                    core::CPersistUtils::restore(SPLIT_VALUE_TAG, m_Threshold, traverser))
            RESTORE(LEFT_CHILD_TAG,
                    core::CPersistUtils::restore(LEFT_CHILD_TAG, m_LeftChild, traverser))
            RESTORE(RIGHT_CHILD_TAG,
                    core::CPersistUtils::restore(RIGHT_CHILD_TAG, m_RightChild, traverser))

        } while (traverser.next());
    } catch (std::exception& e) {
        LOG_ERROR(<< "Failed to restore state! " << e.what());
        return false;
    }

    return true;}

bool ml::api::SEnsemble::acceptRestoreTraverser(ml::core::CStateRestoreTraverser &traverser) {
    auto restoreTree = [this](ml::core::CStateRestoreTraverser &traverser){
        STree tree;
        RESTORE(TREE_TAG, std::bind(&STree::acceptRestoreTraverser, &tree, std::placeholders::_1))
        m_TrainedModels.push_back(tree);
        return true;
    };
    do {
        const std::string& name = traverser.name();
        RESTORE(TREE_TAG, restoreTree)
        // TODO restore aggregated output
    } while (traverser.next());
    return true;
}

bool ml::api::STree::acceptRestoreTraverser(ml::core::CStateRestoreTraverser &traverser) {
    auto restoreTreeNode = [this](ml::core::CStateRestoreTraverser &traverser){
        STreeNode treeNode;
        RESTORE(TREE_TAG, std::bind(&STreeNode::acceptRestoreTraverser, &treeNode, std::placeholders::_1))
        m_TreeStructure.push_back(treeNode);
        return true;
    };
    do {
        const std::string& name = traverser.name();
        RESTORE(TREE_NODE_TAG, restoreTreeNode)
        // TODO restore aggregated output
    } while (traverser.next());
    return true;
}
