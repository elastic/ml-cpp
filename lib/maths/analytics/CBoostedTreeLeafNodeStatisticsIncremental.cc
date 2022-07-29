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

#include <maths/analytics/CBoostedTreeLeafNodeStatisticsIncremental.h>

#include <core/CDataFrame.h>
#include <core/CLogger.h>
#include <core/CMemory.h>

#include <maths/analytics/CBoostedTree.h>
#include <maths/analytics/CBoostedTreeUtils.h>
#include <maths/analytics/CDataFrameCategoryEncoder.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CTools.h>

#include <limits>
#include <utility>

namespace ml {
namespace maths {
namespace analytics {
using namespace boosted_tree_detail;

namespace {
const std::size_t ASSIGN_MISSING_TO_LEFT{0};
const std::size_t ASSIGN_MISSING_TO_RIGHT{1};
}

CBoostedTreeLeafNodeStatisticsIncremental::CBoostedTreeLeafNodeStatisticsIncremental(
    std::size_t id,
    const TSizeVec& extraColumns,
    std::size_t numberLossParameters,
    const core::CDataFrame& frame,
    const TRegularization& regularization,
    const TFloatVecVec& candidateSplits,
    const TSizeVec& treeFeatureBag,
    const TSizeVec& nodeFeatureBag,
    std::size_t depth,
    const core::CPackedBitVector& rowMask,
    CWorkspace& workspace)
    : CBoostedTreeLeafNodeStatistics{id, depth, extraColumns,
                                     numberLossParameters, candidateSplits} {

    this->computeAggregateLossDerivatives(CNoLookAheadBound{},
                                          workspace.numberThreads(), frame,
                                          treeFeatureBag, rowMask, workspace);

    // Lazily copy the mask and derivatives to avoid unnecessary allocations.

    m_PreviousSplit = this->rootPreviousSplit(workspace);

    this->derivatives().swap(workspace.reducedDerivatives(treeFeatureBag));
    this->bestSplitStatistics() = this->computeBestSplitStatistics(
        workspace.numberThreads(), regularization, nodeFeatureBag);
    workspace.reducedDerivatives(treeFeatureBag).swap(this->derivatives());

    if (this->gain() >= workspace.minimumGain()) {
        this->rowMask() = rowMask;
        CSplitsDerivatives tmp{workspace.derivatives()[0]};
        this->derivatives() = std::move(tmp);
    }
}

CBoostedTreeLeafNodeStatisticsIncremental::CBoostedTreeLeafNodeStatisticsIncremental(
    std::size_t id,
    const CBoostedTreeLeafNodeStatisticsIncremental& parent,
    const core::CDataFrame& frame,
    const TRegularization& regularization,
    const TSizeVec& treeFeatureBag,
    const TSizeVec& nodeFeatureBag,
    bool isLeftChild,
    const CBoostedTreeNode& split,
    CWorkspace& workspace)
    : CBoostedTreeLeafNodeStatistics{id, parent.depth() + 1, parent.extraColumns(),
                                     parent.numberLossParameters(),
                                     parent.candidateSplits()} {

    this->computeRowMaskAndAggregateLossDerivatives(
        CNoLookAheadBound{},
        TThreading::numberThreadsForAggregateLossDerivatives(
            workspace.numberThreads(), treeFeatureBag.size(), parent.minimumChildRowCount()),
        frame, isLeftChild, split, treeFeatureBag, parent.rowMask(), workspace);
    this->derivatives().swap(workspace.reducedDerivatives(treeFeatureBag));

    // Set the split feature and value for this node in the tree being retrained.
    std::size_t parentSplitFeature{parent.bestSplit().first};
    m_PreviousSplit =
        (isLeftChild ? parent.leftChildPreviousSplit(parentSplitFeature, workspace)
                     : parent.rightChildPreviousSplit(parentSplitFeature, workspace));

    this->bestSplitStatistics() = this->computeBestSplitStatistics(
        workspace.numberThreads(), regularization, nodeFeatureBag);
    workspace.reducedDerivatives(treeFeatureBag).swap(this->derivatives());

    // Lazily copy the mask and derivatives to avoid unnecessary allocations.
    if (this->gain() >= workspace.minimumGain()) {
        CSplitsDerivatives tmp{workspace.reducedDerivatives(treeFeatureBag)};
        this->rowMask() = workspace.reducedMask(parent.rowMask().size());
        this->derivatives() = std::move(tmp);
    }
}

CBoostedTreeLeafNodeStatisticsIncremental::CBoostedTreeLeafNodeStatisticsIncremental(
    std::size_t id,
    CBoostedTreeLeafNodeStatisticsIncremental&& parent,
    const TRegularization& regularization,
    const TSizeVec& treeFeatureBag,
    const TSizeVec& nodeFeatureBag,
    bool isLeftChild,
    CWorkspace& workspace)
    : CBoostedTreeLeafNodeStatistics{id,
                                     parent.depth() + 1,
                                     parent.extraColumns(),
                                     parent.numberLossParameters(),
                                     parent.candidateSplits(),
                                     std::move(parent.derivatives())} {

    // TODO if sum(splits) > |feature| * |rows| aggregate.
    this->derivatives().subtract(workspace.numberThreads(),
                                 workspace.reducedDerivatives(treeFeatureBag),
                                 treeFeatureBag);

    // Set the split feature and value for this node in the tree being retrained.
    std::size_t parentSplitFeature{parent.bestSplit().first};
    m_PreviousSplit =
        (isLeftChild ? parent.leftChildPreviousSplit(parentSplitFeature, workspace)
                     : parent.rightChildPreviousSplit(parentSplitFeature, workspace));

    this->bestSplitStatistics() = this->computeBestSplitStatistics(
        workspace.numberThreads(), regularization, nodeFeatureBag);

    // Lazily compute the row mask to avoid unnecessary work.
    if (this->gain() >= workspace.minimumGain()) {
        this->rowMask() = std::move(parent.rowMask());
        this->rowMask() ^= workspace.reducedMask(this->rowMask().size());
    }
}

CBoostedTreeLeafNodeStatisticsIncremental::CBoostedTreeLeafNodeStatisticsIncremental(
    const TSizeVec& extraColumns,
    std::size_t numberLossParameters,
    const TFloatVecVec& candidateSplits,
    CSplitsDerivatives derivatives)
    : CBoostedTreeLeafNodeStatistics{0, // Id
                                     0, // Depth
                                     extraColumns,
                                     numberLossParameters,
                                     candidateSplits,
                                     std::move(derivatives)} {
}

CBoostedTreeLeafNodeStatisticsIncremental::TPtrPtrPr
CBoostedTreeLeafNodeStatisticsIncremental::split(std::size_t leftChildId,
                                                 std::size_t rightChildId,
                                                 double /*gainThreshold*/,
                                                 const core::CDataFrame& frame,
                                                 const TRegularization& regularization,
                                                 const TSizeVec& treeFeatureBag,
                                                 const TSizeVec& nodeFeatureBag,
                                                 const CBoostedTreeNode& split,
                                                 CWorkspace& workspace) {
    TPtr leftChild;
    TPtr rightChild;
    if (this->leftChildHasFewerRows()) {
        leftChild = std::make_shared<CBoostedTreeLeafNodeStatisticsIncremental>(
            leftChildId, *this, frame, regularization, treeFeatureBag,
            nodeFeatureBag, true /*is left child*/, split, workspace);
        rightChild = std::make_shared<CBoostedTreeLeafNodeStatisticsIncremental>(
            rightChildId, std::move(*this), regularization, treeFeatureBag,
            nodeFeatureBag, false /*is left child*/, workspace);
        return {std::move(leftChild), std::move(rightChild)};
    }

    rightChild = std::make_shared<CBoostedTreeLeafNodeStatisticsIncremental>(
        rightChildId, *this, frame, regularization, treeFeatureBag,
        nodeFeatureBag, false /*is left child*/, split, workspace);
    leftChild = std::make_shared<CBoostedTreeLeafNodeStatisticsIncremental>(
        leftChildId, std::move(*this), regularization, treeFeatureBag,
        nodeFeatureBag, true /*is left child*/, workspace);
    return {std::move(leftChild), std::move(rightChild)};
}

std::size_t CBoostedTreeLeafNodeStatisticsIncremental::staticSize() const {
    return sizeof(*this);
}

CBoostedTreeLeafNodeStatisticsIncremental::SSplitStatistics
CBoostedTreeLeafNodeStatisticsIncremental::computeBestSplitStatistics(
    std::size_t numberThreads,
    const TRegularization& regularization,
    const TSizeVec& featureBag) const {

    // We have four possible regularization terms we'll use:
    //   1. Tree size: gamma * "node count"
    //   2. Sum square weights: lambda * sum{"leaf weight" ^ 2)}
    //   3. Tree depth: alpha * sum{exp(("depth" / "target depth" - 1.0) / "tolerance")}
    //   4. Tree topology change: we get a fixed penalty for choosing a different split
    //      feature and a smaller penalty for choosing a different split value for the
    //      same feature which is proportional to the difference.

    // We seek to find the value at the minimum of the quadratic expansion of the
    // regularized loss function. For a given leaf this expansion is
    //
    //   L(w) = 1/2 w^t H(\lambda) w + g^t w
    //
    // where H(\lambda) = \sum_i H_i + \lambda I, g = \sum_i g_i and w is the leaf's
    // weight. Here, g_i and H_i denote an example's loss gradient and Hessian and i
    // ranges over the examples in the leaf. Writing this as the sum of a quadratic
    // form and constant, i.e. x(w)^t H(\lambda) x(w) + constant, and noting that H
    // is positive definite, we see that we'll minimise loss by choosing w such that
    // x is zero, i.e. w^* = arg\min_w(L(w)) satisfies x(w) = 0. This gives
    //
    //   L(w^*) = -1/2 g^t H(\lambda)^{-1} g

    using TFeatureBestSplitSearchVec = std::vector<TFeatureBestSplitSearch>;
    using TSplitStatisticsVec = std::vector<SSplitStatistics>;

    numberThreads = TThreading::numberThreadsForComputeBestSplitStatistics(
        numberThreads, featureBag.size(), this->numberLossParameters(),
        this->derivatives().numberDerivatives(featureBag));
    LOG_TRACE(<< "number threads = " << numberThreads);

    TFeatureBestSplitSearchVec featureBestSplitSearches;
    TSplitStatisticsVec splitStats(numberThreads);
    featureBestSplitSearches.reserve(numberThreads);

    for (std::size_t i = 0; i < numberThreads; ++i) {
        featureBestSplitSearches.push_back(
            this->featureBestSplitSearch(regularization, splitStats[i]));
    }

    core::parallel_for_each(featureBag.begin(), featureBag.end(), featureBestSplitSearches);

    SSplitStatistics result;
    for (std::size_t i = 0; i < numberThreads; ++i) {
        if (splitStats[i] > result) {
            result = splitStats[i];
        }
    }
    LOG_TRACE(<< "best split: " << result.print());

    return result;
}

CBoostedTreeLeafNodeStatisticsIncremental::TFeatureBestSplitSearch
CBoostedTreeLeafNodeStatisticsIncremental::featureBestSplitSearch(
    const TRegularization& regularization,
    SSplitStatistics& bestSplitStatistics) const {

    using TDoubleAry = std::array<double, 2>;
    using TDoubleVector = common::CDenseVector<double>;
    using TDoubleVectorAry = std::array<TDoubleVector, 2>;
    using TDoubleMatrix = common::CDenseMatrix<double>;
    using TDoubleMatrixAry = std::array<TDoubleMatrix, 2>;

    int d{static_cast<int>(this->numberLossParameters())};
    double lambda{regularization.leafWeightPenaltyMultiplier().value()};

    auto minimumLoss = TThreading::makeThreadLocalMinimumLossFunction(d, lambda);

    TDoubleVector g_{d};
    TDoubleMatrix h_{d, d};
    TDoubleVectorAry gl_{TDoubleVector{d}, TDoubleVector{d}};
    TDoubleVectorAry gr_{TDoubleVector{d}, TDoubleVector{d}};
    TDoubleMatrixAry hl_{TDoubleMatrix{d, d}, TDoubleMatrix{d, d}};
    TDoubleMatrixAry hr_{TDoubleMatrix{d, d}, TDoubleMatrix{d, d}};

    return [
        // Inputs
        minimumLoss, &regularization,
        // State
        g = std::move(g_), h = std::move(h_), gl = std::move(gl_),
        gr = std::move(gr_), hl = std::move(hl_), hr = std::move(hr_),
        // Results
        &bestSplitStatistics, this
    ](std::size_t feature) mutable {

        const auto& derivatives = this->derivatives();
        const auto& candidateSplits = this->candidateSplits();

        std::size_t c{derivatives.missingCount(feature)};
        g = derivatives.missingGradient(feature);
        h = derivatives.missingCurvature(feature);
        for (auto featureDerivatives = derivatives.beginDerivatives(feature);
             featureDerivatives != derivatives.endDerivatives(feature); ++featureDerivatives) {
            c += featureDerivatives->count();
            g += featureDerivatives->gradient();
            h += featureDerivatives->curvature();
        }
        std::size_t cl[]{derivatives.missingCount(feature), 0};
        gl[ASSIGN_MISSING_TO_LEFT] = derivatives.missingGradient(feature);
        gl[ASSIGN_MISSING_TO_RIGHT] = TDoubleVector::Zero(g.rows());
        gr[ASSIGN_MISSING_TO_LEFT] = g - derivatives.missingGradient(feature);
        gr[ASSIGN_MISSING_TO_RIGHT] = g;
        hl[ASSIGN_MISSING_TO_LEFT] = derivatives.missingCurvature(feature);
        hl[ASSIGN_MISSING_TO_RIGHT] = TDoubleMatrix::Zero(h.rows(), h.cols());
        hr[ASSIGN_MISSING_TO_LEFT] = h - derivatives.missingCurvature(feature);
        hr[ASSIGN_MISSING_TO_RIGHT] = h;

        double maximumGain{-INF};
        double splitAt{-INF};
        std::size_t leftChildRowCount{0};
        bool assignMissingToLeft{true};
        std::size_t size{derivatives.numberDerivatives(feature)};
        TDoubleAry gain;
        auto gainMoments = common::CBasicStatistics::momentsAccumulator(0.0, 0.0, 0.0);

        for (std::size_t split = 0; split + 1 < size; ++split) {

            std::size_t count{derivatives.count(feature, split)};
            if (count == 0) {
                continue;
            }

            const auto& gradient = derivatives.gradient(feature, split);
            const auto& curvature = derivatives.curvature(feature, split);

            cl[ASSIGN_MISSING_TO_LEFT] += count;
            cl[ASSIGN_MISSING_TO_RIGHT] += count;
            gl[ASSIGN_MISSING_TO_LEFT] += gradient;
            gl[ASSIGN_MISSING_TO_RIGHT] += gradient;
            gr[ASSIGN_MISSING_TO_LEFT] -= gradient;
            gr[ASSIGN_MISSING_TO_RIGHT] -= gradient;
            hl[ASSIGN_MISSING_TO_LEFT] += curvature;
            hl[ASSIGN_MISSING_TO_RIGHT] += curvature;
            hr[ASSIGN_MISSING_TO_LEFT] -= curvature;
            hr[ASSIGN_MISSING_TO_RIGHT] -= curvature;

            // Note in the following we scale the tree change penalty by 2 to undo
            // the scaling which is applied when computing maximum gain.

            if (cl[ASSIGN_MISSING_TO_LEFT] == 0 || cl[ASSIGN_MISSING_TO_LEFT] == c) {
                gain[ASSIGN_MISSING_TO_LEFT] = -INF;
            } else {
                double minLossLeft{minimumLoss(gl[ASSIGN_MISSING_TO_LEFT],
                                               hl[ASSIGN_MISSING_TO_LEFT])};
                double minLossRight{minimumLoss(gr[ASSIGN_MISSING_TO_LEFT],
                                                hr[ASSIGN_MISSING_TO_LEFT])};
                gain[ASSIGN_MISSING_TO_LEFT] =
                    minLossLeft + minLossRight -
                    2.0 * this->penaltyForTreeChange(regularization, feature, split);
                gainMoments.add(gain[ASSIGN_MISSING_TO_LEFT]);
            }

            if (cl[ASSIGN_MISSING_TO_RIGHT] == 0 || cl[ASSIGN_MISSING_TO_RIGHT] == c) {
                gain[ASSIGN_MISSING_TO_RIGHT] = -INF;
            } else {
                double minLossLeft{minimumLoss(gl[ASSIGN_MISSING_TO_RIGHT],
                                               hl[ASSIGN_MISSING_TO_RIGHT])};
                double minLossRight{minimumLoss(gr[ASSIGN_MISSING_TO_RIGHT],
                                                hr[ASSIGN_MISSING_TO_RIGHT])};
                gain[ASSIGN_MISSING_TO_RIGHT] =
                    minLossLeft + minLossRight -
                    2.0 * this->penaltyForTreeChange(regularization, feature, split);
                gainMoments.add(gain[ASSIGN_MISSING_TO_RIGHT]);
            }

            if (gain[ASSIGN_MISSING_TO_LEFT] > maximumGain) {
                maximumGain = gain[ASSIGN_MISSING_TO_LEFT];
                splitAt = candidateSplits[feature][split];
                leftChildRowCount = cl[ASSIGN_MISSING_TO_LEFT];
                assignMissingToLeft = true;
            }
            if (gain[ASSIGN_MISSING_TO_RIGHT] > maximumGain) {
                maximumGain = gain[ASSIGN_MISSING_TO_RIGHT];
                splitAt = candidateSplits[feature][split];
                leftChildRowCount = cl[ASSIGN_MISSING_TO_RIGHT];
                assignMissingToLeft = false;
            }
        }

        double penaltyForDepth{regularization.penaltyForDepth(this->depth())};
        double penaltyForDepthPlusOne{regularization.penaltyForDepth(this->depth() + 1)};

        // The gain is the difference between the quadratic minimum for loss with
        // no split and the loss with the minimum loss split we found.
        double totalGain{0.5 * (maximumGain - minimumLoss(g, h)) -
                         regularization.treeSizePenaltyMultiplier().value() -
                         regularization.depthPenaltyMultiplier().value() *
                             (2.0 * penaltyForDepthPlusOne - penaltyForDepth)};

        SSplitStatistics candidate{
            totalGain,
            common::CBasicStatistics::variance(gainMoments),
            h.trace() / static_cast<double>(this->numberLossParameters()),
            feature,
            splitAt,
            std::min(leftChildRowCount, c - leftChildRowCount),
            2 * leftChildRowCount < c,
            assignMissingToLeft};
        LOG_TRACE(<< "candidate split: " << candidate.print());

        if (candidate > bestSplitStatistics) {
            bestSplitStatistics = candidate;
        }
    };
}

double
CBoostedTreeLeafNodeStatisticsIncremental::penaltyForTreeChange(const TRegularization& regularization,
                                                                std::size_t feature,
                                                                std::size_t split) const {
    if (m_PreviousSplit == std::nullopt) {
        return 0.0;
    }

    if (feature != m_PreviousSplit->s_Feature) {
        return regularization.treeTopologyChangePenalty().value();
    }

    const auto& candidateSplits = this->candidateSplits()[feature];
    if (candidateSplits.empty()) {
        return 0.0;
    }

    double a{*candidateSplits.begin()};
    double b{*(candidateSplits.end() - 1)};
    if (a == b) {
        return 0.0;
    }

    double splitAt{candidateSplits[split]};
    double previousSplitAt{common::CTools::truncate(m_PreviousSplit->s_SplitAt, a, b)};
    return 0.25 * regularization.treeTopologyChangePenalty().value() *
           std::fabs(splitAt - previousSplitAt) / (b - a);
}

CBoostedTreeLeafNodeStatisticsIncremental::TOptionalPreviousSplit
CBoostedTreeLeafNodeStatisticsIncremental::rootPreviousSplit(const CWorkspace& workspace) const {

    if (workspace.retraining() == nullptr) {
        return {};
    }

    const auto& tree = *workspace.retraining();
    const auto& node = root(tree);
    if (node.isLeaf()) {
        return {};
    }

    return TOptionalPreviousSplit{std::in_place, rootIndex(),
                                  node.splitFeature(), node.splitValue()};
}

CBoostedTreeLeafNodeStatisticsIncremental::TOptionalPreviousSplit
CBoostedTreeLeafNodeStatisticsIncremental::leftChildPreviousSplit(std::size_t feature,
                                                                  const CWorkspace& workspace) const {
    if (workspace.retraining() == nullptr || m_PreviousSplit == std::nullopt ||
        m_PreviousSplit->s_Feature != feature) {
        return {};
    }

    const auto& tree = *workspace.retraining();
    if (tree[m_PreviousSplit->s_NodeIndex].isLeaf()) {
        return {};
    }

    std::size_t leftChildIndex{tree[m_PreviousSplit->s_NodeIndex].leftChildIndex()};
    const auto& node = tree[leftChildIndex];

    return TOptionalPreviousSplit{std::in_place, leftChildIndex,
                                  node.splitFeature(), node.splitValue()};
}

CBoostedTreeLeafNodeStatisticsIncremental::TOptionalPreviousSplit
CBoostedTreeLeafNodeStatisticsIncremental::rightChildPreviousSplit(std::size_t feature,
                                                                   const CWorkspace& workspace) const {
    if (workspace.retraining() == nullptr || m_PreviousSplit == std::nullopt ||
        m_PreviousSplit->s_Feature != feature) {
        return {};
    }

    const auto& tree = *workspace.retraining();
    if (tree[m_PreviousSplit->s_NodeIndex].isLeaf()) {
        return {};
    }

    std::size_t rightChildIndex{tree[m_PreviousSplit->s_NodeIndex].leftChildIndex()};
    const auto& node = tree[rightChildIndex];

    return TOptionalPreviousSplit{std::in_place, rightChildIndex,
                                  node.splitFeature(), node.splitValue()};
}
}
}
}
