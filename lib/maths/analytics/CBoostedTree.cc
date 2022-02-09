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

#include <maths/analytics/CBoostedTree.h>

#include <core/CDataFrame.h>
#include <core/CJsonStatePersistInserter.h>
#include <core/RestoreMacros.h>

#include <maths/analytics/CBoostedTreeImpl.h>
#include <maths/analytics/CBoostedTreeLeafNodeStatistics.h>
#include <maths/analytics/CBoostedTreeLoss.h>
#include <maths/analytics/CBoostedTreeUtils.h>

#include <maths/common/CLinearAlgebraPersist.h>
#include <maths/common/CLinearAlgebraShims.h>

#include <algorithm>
#include <limits>
#include <sstream>
#include <utility>

namespace ml {
namespace maths {
namespace analytics {
using namespace boosted_tree;
using namespace boosted_tree_detail;

namespace {
const std::string ASSIGN_MISSING_TO_LEFT_TAG{"assign_missing_to_left "};
const std::string CURVATURE_TAG{"curvature"};
const std::string GAIN_TAG{"gain"};
const std::string LEFT_CHILD_TAG{"left_child"};
const std::string MISSING_SPLIT_TAG{"missing_split_tag"};
const std::string NODE_VALUE_TAG{"node_value"};
const std::string NUMBER_SAMPLES_TAG{"number_samples"};
const std::string RIGHT_CHILD_TAG{"right_child"};
const std::string SPLIT_TAG{"split_tag"};
const std::string SPLIT_FEATURE_TAG{"split_feature"};
const std::string SPLIT_VALUE_TAG{"split_value"};
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
    return this->assignToLeft(row)
               ? tree[m_LeftChild.get()].leafIndex(row, tree, m_LeftChild.get())
               : tree[m_RightChild.get()].leafIndex(row, tree, m_RightChild.get());
}

bool CBoostedTreeNode::assignToLeft(const CEncodedDataFrameRowRef& row) const {
    double value{row[m_SplitFeature]};
    bool missing{CDataFrameUtils::isMissing(value)};
    return (missing && m_AssignMissingToLeft) || (missing == false && value < m_SplitValue);
}

CBoostedTreeNode::TNodeIndex CBoostedTreeNode::leafIndex(const TRowRef& row,
                                                         const TSizeVec& extraColumns,
                                                         const TNodeVec& tree,
                                                         TNodeIndex index) const {
    if (this->isLeaf()) {
        return index;
    }
    return this->assignToLeft(row, extraColumns)
               ? tree[m_LeftChild.get()].leafIndex(row, extraColumns, tree,
                                                   m_LeftChild.get())
               : tree[m_RightChild.get()].leafIndex(row, extraColumns, tree,
                                                    m_RightChild.get());
}

bool CBoostedTreeNode::assignToLeft(const TRowRef& row, const TSizeVec& extraColumns) const {
    auto* splits = beginSplits(row, extraColumns);
    std::uint8_t split{
        CPackedUInt8Decorator{splits[m_SplitFeature >> 2]}.readBytes()[m_SplitFeature & 0x3]};
    return (split == m_MissingSplit && m_AssignMissingToLeft) ||
           (split != m_MissingSplit && split <= m_Split);
}

CBoostedTreeNode::TNodeIndexNodeIndexPr
CBoostedTreeNode::split(const TFloatVecVec& candidateSplits,
                        std::size_t splitFeature,
                        double splitValue,
                        bool assignMissingToLeft,
                        double gain,
                        double curvature,
                        TNodeVec& tree) {
    m_SplitFeature = splitFeature;
    m_SplitValue = splitValue;
    if (splitFeature < candidateSplits.size()) {
        const auto& featureCandidateSplits = candidateSplits[splitFeature];
        m_Split = static_cast<std::uint8_t>(
            std::lower_bound(featureCandidateSplits.begin(),
                             featureCandidateSplits.end(), splitValue) -
            featureCandidateSplits.begin());
        m_MissingSplit = static_cast<std::uint8_t>(missingSplit(featureCandidateSplits));
    }
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
    return sizeof(CBoostedTreeNode) +
           common::las::estimateMemoryUsage<TVector>(numberLossParameters);
}

std::size_t CBoostedTreeNode::deployedSize() const {
    // These are derived from the main terms from TreeSizeInfo.java:
    //   - Java uses separate objects for internal and leaf nodes.
    //   - Each object has a reference.
    //   - Each internal node additionally has 1 enum, 1 double, 3 ints, 1 boolean
    //     and 1 long. We assume enum is int sized and after alignment the boolean
    //     also consumes 4 bytes.
    //   - Each leaf has 1 double array (equal to the node value size) note Java
    //     arrays have 20 bytes overhead and 1 long.
    //
    // (We assume the JVM is are not using compressed refs in the so 8 bytes per
    // reference which is the worst case.)
    return this->isLeaf() ? 20 + (m_NodeValue.size() + 3) * 8 : 5 * 4 + 3 * 8;
}

void CBoostedTreeNode::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    core::CPersistUtils::persist(ASSIGN_MISSING_TO_LEFT_TAG, m_AssignMissingToLeft, inserter);
    core::CPersistUtils::persist(CURVATURE_TAG, m_Curvature, inserter);
    core::CPersistUtils::persist(GAIN_TAG, m_Gain, inserter);
    core::CPersistUtils::persist(LEFT_CHILD_TAG, m_LeftChild, inserter);
    core::CPersistUtils::persist(MISSING_SPLIT_TAG,
                                 static_cast<std::size_t>(m_MissingSplit), inserter);
    core::CPersistUtils::persist(NODE_VALUE_TAG, m_NodeValue, inserter);
    core::CPersistUtils::persist(NUMBER_SAMPLES_TAG, m_NumberSamples, inserter);
    core::CPersistUtils::persist(RIGHT_CHILD_TAG, m_RightChild, inserter);
    core::CPersistUtils::persist(SPLIT_TAG, static_cast<std::size_t>(m_Split), inserter);
    core::CPersistUtils::persist(SPLIT_FEATURE_TAG, m_SplitFeature, inserter);
    core::CPersistUtils::persist(SPLIT_VALUE_TAG, m_SplitValue, inserter);
}

bool CBoostedTreeNode::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE(ASSIGN_MISSING_TO_LEFT_TAG,
                core::CPersistUtils::restore(ASSIGN_MISSING_TO_LEFT_TAG,
                                             m_AssignMissingToLeft, traverser))
        RESTORE(CURVATURE_TAG,
                core::CPersistUtils::restore(CURVATURE_TAG, m_Curvature, traverser))
        RESTORE(GAIN_TAG, core::CPersistUtils::restore(GAIN_TAG, m_Gain, traverser))
        RESTORE(LEFT_CHILD_TAG,
                core::CPersistUtils::restore(LEFT_CHILD_TAG, m_LeftChild, traverser))
        RESTORE_SETUP_TEARDOWN(
            MISSING_SPLIT_TAG, std::size_t missingSplit,
            core::CPersistUtils::restore(MISSING_SPLIT_TAG, missingSplit, traverser),
            m_MissingSplit = static_cast<std::uint8_t>(missingSplit))
        RESTORE(NODE_VALUE_TAG,
                core::CPersistUtils::restore(NODE_VALUE_TAG, m_NodeValue, traverser))
        RESTORE(NUMBER_SAMPLES_TAG,
                core::CPersistUtils::restore(NUMBER_SAMPLES_TAG, m_NumberSamples, traverser))
        RESTORE(RIGHT_CHILD_TAG,
                core::CPersistUtils::restore(RIGHT_CHILD_TAG, m_RightChild, traverser))
        RESTORE_SETUP_TEARDOWN(SPLIT_TAG, std::size_t split,
                               core::CPersistUtils::restore(SPLIT_TAG, split, traverser),
                               m_Split = static_cast<std::uint8_t>(split))
        RESTORE(SPLIT_FEATURE_TAG,
                core::CPersistUtils::restore(SPLIT_FEATURE_TAG, m_SplitFeature, traverser))
        RESTORE(SPLIT_VALUE_TAG,
                core::CPersistUtils::restore(SPLIT_VALUE_TAG, m_SplitValue, traverser))
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

CBoostedTree::THyperparameterImportanceVec CBoostedTree::hyperparameterImportance() const {
    return m_Impl->hyperparameterImportance();
}

std::size_t CBoostedTree::numberTrainingRows() const {
    return static_cast<std::size_t>(m_Impl->allTrainingRowsMask().manhattan());
}

double CBoostedTree::trainFractionPerFold() const {
    return m_Impl->trainFractionPerFold();
}

std::size_t CBoostedTree::columnHoldingDependentVariable() const {
    return m_Impl->columnHoldingDependentVariable();
}

CBoostedTree::TDouble2Vec CBoostedTree::readPrediction(const TRowRef& row) const {
    const auto& loss = m_Impl->loss();
    return loss
        .transform(boosted_tree_detail::readPrediction(row, m_Impl->extraColumns(),
                                                       loss.numberParameters()))
        .to<TDouble2Vec>();
}

CBoostedTree::TDouble2Vec CBoostedTree::readAndAdjustPrediction(const TRowRef& row) const {

    const auto& loss = m_Impl->loss();

    auto prediction = loss.transform(boosted_tree_detail::readPrediction(
        row, m_Impl->extraColumns(), loss.numberParameters()));

    switch (loss.type()) {
    case E_BinaryClassification:
    case E_MulticlassClassification:
        prediction = m_Impl->classificationWeights().array() * prediction.array();
        break;
    case E_MseRegression:
    case E_MsleRegression:
    case E_HuberRegression:
        break;
    }

    return prediction.to<TDouble2Vec>();
}

const CBoostedTree::TNodeVecVec& CBoostedTree::trainedModel() const {
    return m_Impl->trainedModel();
}

CBoostedTreeImpl& CBoostedTree::impl() {
    return *m_Impl;
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

CBoostedTree::TDoubleVec CBoostedTree::classificationWeights() const {
    const auto& weights = m_Impl->classificationWeights();
    TDoubleVec result(weights.size());
    for (int i = 0; i < weights.size(); ++i) {
        result[i] = weights(i);
    }
    return result;
}
}
}
}
