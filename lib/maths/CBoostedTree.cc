/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBoostedTree.h>

#include <core/CDataFrame.h>
#include <core/CJsonStatePersistInserter.h>

#include <maths/CBoostedTreeImpl.h>
#include <maths/CBoostedTreeLoss.h>
#include <maths/CBoostedTreeUtils.h>
#include <maths/CLinearAlgebraPersist.h>
#include <maths/CLinearAlgebraShims.h>

#include <limits>
#include <sstream>
#include <utility>

namespace ml {
namespace maths {
using namespace boosted_tree;
using namespace boosted_tree_detail;

namespace {
const std::string LEFT_CHILD_TAG{"left_child"};
const std::string RIGHT_CHILD_TAG{"right_child"};
const std::string SPLIT_FEATURE_TAG{"split_feature"};
const std::string ASSIGN_MISSING_TO_LEFT_TAG{"assign_missing_to_left "};
const std::string NODE_VALUE_TAG{"node_value"};
const std::string SPLIT_VALUE_TAG{"split_value"};
const std::string NUMBER_SAMPLES_TAG{"number_samples"};
}

CBoostedTreeNode::CBoostedTreeNode(std::size_t numberLossParameters)
    : m_NodeValue{TVector::Zero(numberLossParameters)} {
}

CBoostedTreeNode::TNodeIndex CBoostedTreeNode::leafIndex(const CEncodedDataFrameRowRef& row,
                                                         const TNodeVec& tree,
                                                         TNodeIndex index) const {
    if (this->isLeaf()) {
        return index;
    }
    double value{row[m_SplitFeature]};
    bool missing{CDataFrameUtils::isMissing(value)};
    return (missing && m_AssignMissingToLeft) || (missing == false && value < m_SplitValue)
               ? tree[m_LeftChild.get()].leafIndex(row, tree, m_LeftChild.get())
               : tree[m_RightChild.get()].leafIndex(row, tree, m_RightChild.get());
}

CBoostedTreeNode::TNodeIndexNodeIndexPr CBoostedTreeNode::split(std::size_t splitFeature,
                                                                double splitValue,
                                                                bool assignMissingToLeft,
                                                                double gain,
                                                                double curvature,
                                                                TNodeVec& tree) {
    m_SplitFeature = splitFeature;
    m_SplitValue = splitValue;
    m_AssignMissingToLeft = assignMissingToLeft;
    m_LeftChild = static_cast<TNodeIndex>(tree.size());
    m_RightChild = static_cast<TNodeIndex>(tree.size() + 1);
    m_Gain = gain;
    m_Curvature = curvature;
    TNodeIndexNodeIndexPr result{m_LeftChild.get(), m_RightChild.get()};
    // Don't access members after calling resize because this object is likely an
    // element of the vector being resized.
    tree.resize(tree.size() + 2);
    return result;
}

std::size_t CBoostedTreeNode::memoryUsage() const {
    return core::CMemory::dynamicSize(m_NodeValue);
}

std::size_t CBoostedTreeNode::estimateMemoryUsage(std::size_t numberLossParameters) {
    return sizeof(CBoostedTreeNode) + las::estimateMemoryUsage<TVector>(numberLossParameters);
}

void CBoostedTreeNode::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    core::CPersistUtils::persist(LEFT_CHILD_TAG, m_LeftChild, inserter);
    core::CPersistUtils::persist(RIGHT_CHILD_TAG, m_RightChild, inserter);
    core::CPersistUtils::persist(SPLIT_FEATURE_TAG, m_SplitFeature, inserter);
    core::CPersistUtils::persist(ASSIGN_MISSING_TO_LEFT_TAG, m_AssignMissingToLeft, inserter);
    core::CPersistUtils::persist(NODE_VALUE_TAG, m_NodeValue, inserter);
    core::CPersistUtils::persist(SPLIT_VALUE_TAG, m_SplitValue, inserter);
    core::CPersistUtils::persist(NUMBER_SAMPLES_TAG, m_NumberSamples, inserter);
}

bool CBoostedTreeNode::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE(LEFT_CHILD_TAG,
                core::CPersistUtils::restore(LEFT_CHILD_TAG, m_LeftChild, traverser))
        RESTORE(RIGHT_CHILD_TAG,
                core::CPersistUtils::restore(RIGHT_CHILD_TAG, m_RightChild, traverser))
        RESTORE(SPLIT_FEATURE_TAG,
                core::CPersistUtils::restore(SPLIT_FEATURE_TAG, m_SplitFeature, traverser))
        RESTORE(ASSIGN_MISSING_TO_LEFT_TAG,
                core::CPersistUtils::restore(ASSIGN_MISSING_TO_LEFT_TAG,
                                             m_AssignMissingToLeft, traverser))
        RESTORE(NODE_VALUE_TAG,
                core::CPersistUtils::restore(NODE_VALUE_TAG, m_NodeValue, traverser))
        RESTORE(SPLIT_VALUE_TAG,
                core::CPersistUtils::restore(SPLIT_VALUE_TAG, m_SplitValue, traverser))
        RESTORE(NUMBER_SAMPLES_TAG,
                core::CPersistUtils::restore(NUMBER_SAMPLES_TAG, m_NumberSamples, traverser))
    } while (traverser.next());
    return true;
}

std::string CBoostedTreeNode::print(const TNodeVec& tree) const {
    std::ostringstream result;
    return this->doPrint("", tree, result).str();
}

std::ostringstream& CBoostedTreeNode::doPrint(std::string pad,
                                              const TNodeVec& tree,
                                              std::ostringstream& result) const {
    result << "\n" << pad;
    if (this->isLeaf()) {
        result << m_NodeValue;
    } else {
        result << "split feature '" << m_SplitFeature << "' @ " << m_SplitValue;
        tree[m_LeftChild.get()].doPrint(pad + "  ", tree, result);
        tree[m_RightChild.get()].doPrint(pad + "  ", tree, result);
    }
    return result;
}

void CBoostedTreeNode::accept(CVisitor& visitor) const {
    visitor.addNode(m_SplitFeature, m_SplitValue, m_AssignMissingToLeft, m_NodeValue,
                    m_Gain, m_NumberSamples, m_LeftChild, m_RightChild);
}

void CBoostedTreeNode::numberSamples(std::size_t numberSamples) {
    m_NumberSamples = numberSamples;
}

std::size_t CBoostedTreeNode::numberSamples() const {
    return m_NumberSamples;
}

CBoostedTree::CBoostedTree(core::CDataFrame& frame,
                           TTrainingStateCallback recordTrainingState,
                           TImplUPtr&& impl)
    : CDataFramePredictiveModel{frame, std::move(recordTrainingState)}, m_Impl{std::move(impl)} {
}

CBoostedTree::~CBoostedTree() = default;

void CBoostedTree::train() {
    m_Impl->train(this->frame(), this->trainingStateRecorder());
}

void CBoostedTree::predict() const {
    m_Impl->predict(this->frame());
}

CTreeShapFeatureImportance* CBoostedTree::shap() const {
    return m_Impl->shap();
}

std::size_t CBoostedTree::columnHoldingDependentVariable() const {
    return m_Impl->columnHoldingDependentVariable();
}

CBoostedTree::TDoubleVec CBoostedTree::readPrediction(const TRowRef& row) const {
    const auto& loss = m_Impl->loss();
    auto result = loss.transform(boosted_tree_detail::readPrediction(
        row, m_Impl->numberInputColumns(), loss.numberParameters()));
    return result.toStdVector();
}

CBoostedTree::TDoubleVec CBoostedTree::readAndAdjustPrediction(const TRowRef& row) const {

    const auto& loss = m_Impl->loss();

    auto prediction = loss.transform(boosted_tree_detail::readPrediction(
        row, m_Impl->numberInputColumns(), loss.numberParameters()));

    switch (loss.type()) {
    case CLoss::E_BinaryClassification:
    case CLoss::E_MulticlassClassification:
        prediction = m_Impl->classificationWeights().array() * prediction.array();
        break;
    case CLoss::E_Regression:
        break;
    }

    return prediction.toStdVector();
}

const CBoostedTree::TNodeVecVec& CBoostedTree::trainedModel() const {
    return m_Impl->trainedModel();
}

const CBoostedTree::TDoubleVec& CBoostedTree::featureWeightsForTraining() const {
    return m_Impl->featureSampleProbabilities();
}

const std::string& CBoostedTree::bestHyperparametersName() {
    return CBoostedTreeImpl::bestHyperparametersName();
}

const std::string& CBoostedTree::bestRegularizationHyperparametersName() {
    return CBoostedTreeImpl::bestRegularizationHyperparametersName();
}

CBoostedTree::TStrVec CBoostedTree::bestHyperparameterNames() {
    return CBoostedTreeImpl::bestHyperparameterNames();
}

bool CBoostedTree::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    return m_Impl->acceptRestoreTraverser(traverser);
}

void CBoostedTree::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    m_Impl->acceptPersistInserter(inserter);
}

void CBoostedTree::accept(CBoostedTree::CVisitor& visitor) const {
    m_Impl->accept(visitor);
}

const CBoostedTreeHyperparameters& CBoostedTree::bestHyperparameters() const {
    return m_Impl->bestHyperparameters();
}
}
}
