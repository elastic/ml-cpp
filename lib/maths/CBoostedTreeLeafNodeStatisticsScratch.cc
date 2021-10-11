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

#include <maths/CBoostedTreeLeafNodeStatisticsScratch.h>

#include <core/CDataFrame.h>
#include <core/CLogger.h>
#include <core/CMemory.h>

#include <maths/CBoostedTree.h>
#include <maths/CDataFrameCategoryEncoder.h>
#include <maths/CTools.h>

#include <limits>

namespace ml {
namespace maths {
using namespace boosted_tree_detail;

namespace {
const std::size_t ASSIGN_MISSING_TO_LEFT{0};
const std::size_t ASSIGN_MISSING_TO_RIGHT{1};
}

CBoostedTreeLeafNodeStatisticsScratch::CBoostedTreeLeafNodeStatisticsScratch(
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

    this->computeAggregateLossDerivatives(CLookAheadBound{}, workspace.numberThreads(),
                                          frame, treeFeatureBag, rowMask, workspace);

    // Lazily copy the mask and derivatives to avoid unnecessary allocations.

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

CBoostedTreeLeafNodeStatisticsScratch::CBoostedTreeLeafNodeStatisticsScratch(
    std::size_t id,
    const CBoostedTreeLeafNodeStatisticsScratch& parent,
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
        CLookAheadBound{},
        TThreading::numberThreadsForAggregateLossDerivatives(
            workspace.numberThreads(), treeFeatureBag.size(), parent.minimumChildRowCount()),
        frame, isLeftChild, split, treeFeatureBag, parent.rowMask(), workspace);

    // Lazily copy the mask and derivatives to avoid unnecessary allocations.

    this->derivatives().swap(workspace.reducedDerivatives(treeFeatureBag));
    this->bestSplitStatistics() = this->computeBestSplitStatistics(
        workspace.numberThreads(), regularization, nodeFeatureBag);
    workspace.reducedDerivatives(treeFeatureBag).swap(this->derivatives());

    if (this->gain() >= workspace.minimumGain()) {
        CSplitsDerivatives tmp{workspace.reducedDerivatives(treeFeatureBag)};
        this->rowMask() = workspace.reducedMask(parent.rowMask().size());
        this->derivatives() = std::move(tmp);
    }
}

CBoostedTreeLeafNodeStatisticsScratch::CBoostedTreeLeafNodeStatisticsScratch(
    std::size_t id,
    CBoostedTreeLeafNodeStatisticsScratch&& parent,
    const TRegularization& regularization,
    const TSizeVec& treeFeatureBag,
    const TSizeVec& nodeFeatureBag,
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

    this->bestSplitStatistics() = this->computeBestSplitStatistics(
        workspace.numberThreads(), regularization, nodeFeatureBag);

    // Lazily compute the row mask to avoid unnecessary work.
    if (this->gain() >= workspace.minimumGain()) {
        this->rowMask() = std::move(parent.rowMask());
        this->rowMask() ^= workspace.reducedMask(this->rowMask().size());
    }
}

CBoostedTreeLeafNodeStatisticsScratch::CBoostedTreeLeafNodeStatisticsScratch(
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

CBoostedTreeLeafNodeStatisticsScratch::TPtrPtrPr
CBoostedTreeLeafNodeStatisticsScratch::split(std::size_t leftChildId,
                                             std::size_t rightChildId,
                                             double gainThreshold,
                                             const core::CDataFrame& frame,
                                             const TRegularization& regularization,
                                             const TSizeVec& treeFeatureBag,
                                             const TSizeVec& nodeFeatureBag,
                                             const CBoostedTreeNode& split,
                                             CWorkspace& workspace) {
    TPtr leftChild;
    TPtr rightChild;
    if (this->leftChildHasFewerRows()) {
        if (this->bestSplitStatistics().s_LeftChildMaxGain > gainThreshold) {
            leftChild = std::make_shared<CBoostedTreeLeafNodeStatisticsScratch>(
                leftChildId, *this, frame, regularization, treeFeatureBag,
                nodeFeatureBag, true /*is left child*/, split, workspace);
            if (this->bestSplitStatistics().s_RightChildMaxGain > gainThreshold) {
                rightChild = std::make_shared<CBoostedTreeLeafNodeStatisticsScratch>(
                    rightChildId, std::move(*this), regularization,
                    treeFeatureBag, nodeFeatureBag, workspace);
            }
        } else if (this->bestSplitStatistics().s_RightChildMaxGain > gainThreshold) {
            rightChild = std::make_shared<CBoostedTreeLeafNodeStatisticsScratch>(
                rightChildId, *this, frame, regularization, treeFeatureBag,
                nodeFeatureBag, false /*is left child*/, split, workspace);
        }
        return {std::move(leftChild), std::move(rightChild)};
    }

    if (this->bestSplitStatistics().s_RightChildMaxGain > gainThreshold) {
        rightChild = std::make_shared<CBoostedTreeLeafNodeStatisticsScratch>(
            rightChildId, *this, frame, regularization, treeFeatureBag,
            nodeFeatureBag, false /*is left child*/, split, workspace);
        if (this->bestSplitStatistics().s_LeftChildMaxGain > gainThreshold) {
            leftChild = std::make_shared<CBoostedTreeLeafNodeStatisticsScratch>(
                leftChildId, std::move(*this), regularization, treeFeatureBag,
                nodeFeatureBag, workspace);
        }
    } else if (this->bestSplitStatistics().s_LeftChildMaxGain > gainThreshold) {
        leftChild = std::make_shared<CBoostedTreeLeafNodeStatisticsScratch>(
            leftChildId, *this, frame, regularization, treeFeatureBag,
            nodeFeatureBag, true /*is left child*/, split, workspace);
    }
    return {std::move(leftChild), std::move(rightChild)};
}

std::size_t CBoostedTreeLeafNodeStatisticsScratch::staticSize() const {
    return sizeof(*this);
}

CBoostedTreeLeafNodeStatisticsScratch::SSplitStatistics
CBoostedTreeLeafNodeStatisticsScratch::computeBestSplitStatistics(std::size_t numberThreads,
                                                                  const TRegularization& regularization,
                                                                  const TSizeVec& featureBag) const {

    // We have three possible regularization terms we'll use:
    //   1. Tree size: gamma * "node count"
    //   2. Sum square weights: lambda * sum{"leaf weight" ^ 2)}
    //   3. Tree depth: alpha * sum{exp(("depth" / "target depth" - 1.0) / "tolerance")}

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
    using TChildrenGainStatisticsVec = std::vector<SChildrenGainStatistics>;

    const auto& derivatives = this->derivatives();
    numberThreads = TThreading::numberThreadsForComputeBestSplitStatistics(
        numberThreads, featureBag.size(), this->numberLossParameters(),
        derivatives.numberDerivatives(featureBag));
    LOG_TRACE(<< "number threads = " << numberThreads);

    TFeatureBestSplitSearchVec featureBestSplitSearches;
    TSplitStatisticsVec splitStats(numberThreads);
    TChildrenGainStatisticsVec childrenGainStatistics(numberThreads);
    featureBestSplitSearches.reserve(numberThreads);

    for (std::size_t i = 0; i < numberThreads; ++i) {
        featureBestSplitSearches.push_back(this->featureBestSplitSearch(
            regularization, splitStats[i], childrenGainStatistics[i]));
    }

    core::parallel_for_each(featureBag.begin(), featureBag.end(), featureBestSplitSearches);

    SSplitStatistics result;
    SChildrenGainStatistics bestSplitChildrenGainStatistics;
    for (std::size_t i = 0; i < numberThreads; ++i) {
        if (splitStats[i] > result) {
            result = splitStats[i];
            bestSplitChildrenGainStatistics = childrenGainStatistics[i];
        }
    }

    if (derivatives.numberLossParameters() <= 2 && result.s_Gain > 0) {
        double lambda{regularization.leafWeightPenaltyMultiplier().value()};
        double childPenaltyForDepth{regularization.penaltyForDepth(this->depth() + 1)};
        double childPenaltyForDepthPlusOne{
            regularization.penaltyForDepth(this->depth() + 2)};
        double childPenalty{regularization.treeSizePenaltyMultiplier().value() +
                            regularization.depthPenaltyMultiplier().value() *
                                (2.0 * childPenaltyForDepthPlusOne - childPenaltyForDepth)};
        result.s_LeftChildMaxGain =
            0.5 * this->childMaxGain(bestSplitChildrenGainStatistics.s_GainLeft,
                                     bestSplitChildrenGainStatistics.s_MinLossLeft, lambda) -
            childPenalty;

        result.s_RightChildMaxGain =
            0.5 * this->childMaxGain(bestSplitChildrenGainStatistics.s_GainRight,
                                     bestSplitChildrenGainStatistics.s_MinLossRight, lambda) -
            childPenalty;
    }

    return result;
}

CBoostedTreeLeafNodeStatisticsScratch::TFeatureBestSplitSearch
CBoostedTreeLeafNodeStatisticsScratch::featureBestSplitSearch(
    const TRegularization& regularization,
    SSplitStatistics& bestSplitStatistics,
    SChildrenGainStatistics& childrenGainStatisticsGlobal) const {

    using TDoubleAry = std::array<double, 2>;
    using TDoubleVector = CDenseVector<double>;
    using TDoubleVectorAry = std::array<TDoubleVector, 2>;
    using TDoubleMatrix = CDenseMatrix<double>;
    using TDoubleMatrixAry = std::array<TDoubleMatrix, 2>;

    int d{static_cast<int>(this->numberLossParameters())};
    double lambda{regularization.leafWeightPenaltyMultiplier().value()};

    auto minimumLoss_ = TThreading::makeThreadLocalMinimumLossFunction(d, lambda);

    TDoubleVector g_{d};
    TDoubleMatrix h_{d, d};
    TDoubleVectorAry gl_{TDoubleVector{d}, TDoubleVector{d}};
    TDoubleVectorAry gr_{TDoubleVector{d}, TDoubleVector{d}};
    TDoubleMatrixAry hl_{TDoubleMatrix{d, d}, TDoubleMatrix{d, d}};
    TDoubleMatrixAry hr_{TDoubleMatrix{d, d}, TDoubleMatrix{d, d}};

    return [
        // Inputs
        minimumLoss = std::move(minimumLoss_), &regularization,
        // State
        g = std::move(g_), h = std::move(h_), gl = std::move(gl_),
        gr = std::move(gr_), hl = std::move(hl_), hr = std::move(hr_),
        // Results
        &bestSplitStatistics, &childrenGainStatisticsGlobal, this
    ](std::size_t feature) mutable {

        const auto& candidateSplits = this->candidateSplits();
        const auto& derivatives = this->derivatives();

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
        TDoubleAry minLossLeft;
        TDoubleAry minLossRight;
        SChildrenGainStatistics childrenGainStatisticsPerFeature;

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

            if (cl[ASSIGN_MISSING_TO_LEFT] == 0 || cl[ASSIGN_MISSING_TO_LEFT] == c) {
                gain[ASSIGN_MISSING_TO_LEFT] = -INF;
            } else {
                minLossLeft[ASSIGN_MISSING_TO_LEFT] = minimumLoss(
                    gl[ASSIGN_MISSING_TO_LEFT], hl[ASSIGN_MISSING_TO_LEFT]);
                minLossRight[ASSIGN_MISSING_TO_LEFT] = minimumLoss(
                    gr[ASSIGN_MISSING_TO_LEFT], hr[ASSIGN_MISSING_TO_LEFT]);
                gain[ASSIGN_MISSING_TO_LEFT] = minLossLeft[ASSIGN_MISSING_TO_LEFT] +
                                               minLossRight[ASSIGN_MISSING_TO_LEFT];
            }

            if (cl[ASSIGN_MISSING_TO_RIGHT] == 0 || cl[ASSIGN_MISSING_TO_RIGHT] == c) {
                gain[ASSIGN_MISSING_TO_RIGHT] = -INF;
            } else {
                minLossLeft[ASSIGN_MISSING_TO_RIGHT] = minimumLoss(
                    gl[ASSIGN_MISSING_TO_RIGHT], hl[ASSIGN_MISSING_TO_RIGHT]);
                minLossRight[ASSIGN_MISSING_TO_RIGHT] = minimumLoss(
                    gr[ASSIGN_MISSING_TO_RIGHT], hr[ASSIGN_MISSING_TO_RIGHT]);
                gain[ASSIGN_MISSING_TO_RIGHT] = minLossLeft[ASSIGN_MISSING_TO_RIGHT] +
                                                minLossRight[ASSIGN_MISSING_TO_RIGHT];
            }

            if (gain[ASSIGN_MISSING_TO_LEFT] > maximumGain) {
                maximumGain = gain[ASSIGN_MISSING_TO_LEFT];
                splitAt = candidateSplits[feature][split];
                leftChildRowCount = cl[ASSIGN_MISSING_TO_LEFT];
                assignMissingToLeft = true;
                // If gain > -INF then minLossLeft and minLossRight were initialized.
                childrenGainStatisticsPerFeature = {
                    minLossLeft[ASSIGN_MISSING_TO_LEFT], minLossRight[ASSIGN_MISSING_TO_LEFT],
                    gl[ASSIGN_MISSING_TO_LEFT](0), gr[ASSIGN_MISSING_TO_LEFT](0)};
            }
            if (gain[ASSIGN_MISSING_TO_RIGHT] > maximumGain) {
                maximumGain = gain[ASSIGN_MISSING_TO_RIGHT];
                splitAt = candidateSplits[feature][split];
                leftChildRowCount = cl[ASSIGN_MISSING_TO_RIGHT];
                assignMissingToLeft = false;
                // If gain > -INF then minLossLeft and minLossRight were initialized.
                childrenGainStatisticsPerFeature = {
                    minLossLeft[ASSIGN_MISSING_TO_RIGHT],
                    minLossRight[ASSIGN_MISSING_TO_RIGHT],
                    gl[ASSIGN_MISSING_TO_RIGHT](0), gr[ASSIGN_MISSING_TO_RIGHT](0)};
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
        SSplitStatistics candidateSplitStatistics{
            totalGain,
            h.trace() / static_cast<double>(this->numberLossParameters()),
            feature,
            splitAt,
            std::min(leftChildRowCount, c - leftChildRowCount),
            2 * leftChildRowCount < c,
            assignMissingToLeft};
        LOG_TRACE(<< "candidate split: " << candidateSplitStatistics.print());

        if (candidateSplitStatistics > bestSplitStatistics) {
            bestSplitStatistics = candidateSplitStatistics;
            childrenGainStatisticsGlobal = childrenGainStatisticsPerFeature;
        }
    };
}

double CBoostedTreeLeafNodeStatisticsScratch::childMaxGain(double childGain,
                                                           double minLossChild,
                                                           double lambda) const {

    // This computes the maximum possible gain we can expect splitting a child node given
    // we know the sum of the positive (g^+) and negative gradients (g^-) at its parent,
    // the minimum curvature on the positive and negative gradient set (hmin^+ and hmin^-)
    // and largest and smallest gradient (gmax and gmin, respectively). The highest possible
    // gain consistent with these constraints can be shown to be:
    //
    //   (g^+)^2 / (hmin^+ * g^+ / gmax + lambda) + (g^-)^2 / (hmin^- * g^- / gmin + lambda)
    //
    // Since gchild = gchild^+ + gchild^-, we can improve estimates on g^+ and g^- for the
    // child as:
    //   g^+ = max(min(gchild - g^-, g^+), 0),
    //   g^- = max(min(gchild - g^+, g^-), 0).

    double positiveDerivativesGSum =
        std::max(std::min(childGain - this->derivatives().negativeDerivativesGSum(),
                          this->derivatives().positiveDerivativesGSum()),
                 0.0);
    double negativeDerivativesGSum =
        std::min(std::max(childGain - this->derivatives().positiveDerivativesGSum(),
                          this->derivatives().negativeDerivativesGSum()),
                 0.0);
    double lookAheadGain{
        ((positiveDerivativesGSum != 0.0)
             ? CTools::pow2(positiveDerivativesGSum) /
                   (this->derivatives().positiveDerivativesHMin() * positiveDerivativesGSum /
                        this->derivatives().positiveDerivativesGMax() +
                    lambda + 1e-10)
             : 0.0) +
        ((negativeDerivativesGSum != 0.0)
             ? CTools::pow2(negativeDerivativesGSum) /
                   (this->derivatives().negativeDerivativesHMin() * negativeDerivativesGSum /
                        this->derivatives().negativeDerivativesGMin() +
                    lambda + 1e-10)
             : 0.0)};
    return lookAheadGain - minLossChild;
}
}
}
