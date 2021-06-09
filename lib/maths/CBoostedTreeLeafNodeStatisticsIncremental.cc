/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBoostedTreeLeafNodeStatisticsIncremental.h>

#include <core/CDataFrame.h>
#include <core/CImmutableRadixSet.h>
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
struct SChildredGainStats {
    double s_MinLossLeft = -INF;
    double s_MinLossRight = -INF;
    double s_GLeft = -INF;
    double s_GRight = -INF;
};

const std::size_t ASSIGN_MISSING_TO_LEFT{0};
const std::size_t ASSIGN_MISSING_TO_RIGHT{1};
}

CBoostedTreeLeafNodeStatisticsIncremental::CBoostedTreeLeafNodeStatisticsIncremental(
    std::size_t id,
    const TSizeVec& extraColumns,
    std::size_t numberLossParameters,
    std::size_t numberThreads,
    const core::CDataFrame& frame,
    const CDataFrameCategoryEncoder& encoder,
    const TRegularization& regularization,
    const TImmutableRadixSetVec& candidateSplits,
    const TSizeVec& treeFeatureBag,
    const TSizeVec& nodeFeatureBag,
    std::size_t depth,
    const core::CPackedBitVector& rowMask,
    CWorkspace& workspace)
    : CBoostedTreeLeafNodeStatistics{id, depth, extraColumns,
                                     numberLossParameters, candidateSplits} {

    this->computeAggregateLossDerivatives(CNoLookAheadBound{}, numberThreads, frame,
                                          encoder, treeFeatureBag, rowMask, workspace);

    // Lazily copy the mask and derivatives to avoid unnecessary allocations.

    m_PreviousSplit = this->rootPreviousSplit(workspace);

    this->derivatives().swap(workspace.reducedDerivatives());
    this->bestSplitStatistics() =
        this->computeBestSplitStatistics(regularization, nodeFeatureBag);
    workspace.reducedDerivatives().swap(this->derivatives());

    if (this->gain() > workspace.minimumGain()) {
        this->rowMask() = rowMask;
        CSplitsDerivatives tmp{workspace.derivatives()[0]};
        this->derivatives() = std::move(tmp);
    }
}

CBoostedTreeLeafNodeStatisticsIncremental::CBoostedTreeLeafNodeStatisticsIncremental(
    std::size_t id,
    const CBoostedTreeLeafNodeStatisticsIncremental& parent,
    std::size_t numberThreads,
    const core::CDataFrame& frame,
    const CDataFrameCategoryEncoder& encoder,
    const TRegularization& regularization,
    const TSizeVec& treeFeatureBag,
    const TSizeVec& nodeFeatureBag,
    bool isLeftChild,
    const CBoostedTreeNode& split,
    CWorkspace& workspace)
    : CBoostedTreeLeafNodeStatistics{id, parent.depth() + 1, parent.extraColumns(),
                                     parent.numberLossParameters(),
                                     parent.candidateSplits()} {

    numberThreads = std::min(numberThreads, maximumNumberThreadsToAggregateDerivatives(
                                                parent, treeFeatureBag));

    this->computeRowMaskAndAggregateLossDerivatives(
        CNoLookAheadBound{}, numberThreads, frame, encoder, isLeftChild, split,
        treeFeatureBag, parent.rowMask(), workspace);
    this->derivatives().swap(workspace.reducedDerivatives());

    // Lazily copy the mask and derivatives to avoid unnecessary allocations.

    // Set the split feature and value for this node in the tree being retrained.
    std::size_t parentSplitFeature{parent.bestSplit().first};
    m_PreviousSplit =
        (isLeftChild ? parent.leftChildPreviousSplit(parentSplitFeature, workspace)
                     : parent.rightChildPreviousSplit(parentSplitFeature, workspace));

    this->bestSplitStatistics() =
        this->computeBestSplitStatistics(regularization, nodeFeatureBag);
    workspace.reducedDerivatives().swap(this->derivatives());

    if (this->gain() >= workspace.minimumGain()) {
        CSplitsDerivatives tmp{workspace.reducedDerivatives()};
        this->rowMask() = workspace.reducedMask(parent.rowMask().size());
        this->derivatives() = std::move(tmp);
    }
}

CBoostedTreeLeafNodeStatisticsIncremental::CBoostedTreeLeafNodeStatisticsIncremental(
    std::size_t id,
    CBoostedTreeLeafNodeStatisticsIncremental&& parent,
    const TRegularization& regularization,
    const TSizeVec& nodeFeatureBag,
    bool isLeftChild,
    CWorkspace& workspace)
    : CBoostedTreeLeafNodeStatistics{id,
                                     parent.depth() + 1,
                                     parent.extraColumns(),
                                     parent.numberLossParameters(),
                                     parent.candidateSplits(),
                                     std::move(parent.derivatives())} {

    // Lazily compute the row mask to avoid unnecessary work.

    this->derivatives().subtract(workspace.reducedDerivatives());

    // Set the split feature and value for this node in the tree being retrained.
    std::size_t parentSplitFeature{parent.bestSplit().first};
    m_PreviousSplit =
        (isLeftChild ? parent.leftChildPreviousSplit(parentSplitFeature, workspace)
                     : parent.rightChildPreviousSplit(parentSplitFeature, workspace));

    this->bestSplitStatistics() =
        this->computeBestSplitStatistics(regularization, nodeFeatureBag);

    if (this->gain() >= workspace.minimumGain()) {
        this->rowMask() = std::move(parent.rowMask());
        this->rowMask() ^= workspace.reducedMask(this->rowMask().size());
    }
}

CBoostedTreeLeafNodeStatisticsIncremental::TPtrPtrPr
CBoostedTreeLeafNodeStatisticsIncremental::split(std::size_t leftChildId,
                                                 std::size_t rightChildId,
                                                 std::size_t numberThreads,
                                                 double /*gainThreshold*/,
                                                 const core::CDataFrame& frame,
                                                 const CDataFrameCategoryEncoder& encoder,
                                                 const TRegularization& regularization,
                                                 const TSizeVec& treeFeatureBag,
                                                 const TSizeVec& nodeFeatureBag,
                                                 const CBoostedTreeNode& split,
                                                 CWorkspace& workspace) {
    TPtr leftChild;
    TPtr rightChild;
    if (this->leftChildHasFewerRows()) {
        leftChild = std::make_shared<CBoostedTreeLeafNodeStatisticsIncremental>(
            leftChildId, *this, numberThreads, frame, encoder, regularization,
            treeFeatureBag, nodeFeatureBag, true /*is left child*/, split, workspace);
        rightChild = std::make_shared<CBoostedTreeLeafNodeStatisticsIncremental>(
            rightChildId, std::move(*this), regularization, nodeFeatureBag,
            false /*is left child*/, workspace);
        return {std::move(leftChild), std::move(rightChild)};
    }

    rightChild = std::make_shared<CBoostedTreeLeafNodeStatisticsIncremental>(
        rightChildId, *this, numberThreads, frame, encoder, regularization,
        treeFeatureBag, nodeFeatureBag, false /*is left child*/, split, workspace);
    leftChild = std::make_shared<CBoostedTreeLeafNodeStatisticsIncremental>(
        leftChildId, std::move(*this), regularization, nodeFeatureBag,
        true /*is left child*/, workspace);
    return {std::move(leftChild), std::move(rightChild)};
}

std::size_t CBoostedTreeLeafNodeStatisticsIncremental::staticSize() const {
    return sizeof(*this);
}

CBoostedTreeLeafNodeStatisticsIncremental::SSplitStatistics
CBoostedTreeLeafNodeStatisticsIncremental::computeBestSplitStatistics(
    const TRegularization& regularization,
    const TSizeVec& featureBag) const {

    // We have four possible regularization terms we'll use:
    //   1. Tree size: gamma * "node count"
    //   2. Sum square weights: lambda * sum{"leaf weight" ^ 2)}
    //   3. Tree depth: alpha * sum{exp(("depth" / "target depth" - 1.0) / "tolerance")}
    //   4. Tree topology change: we get a fixed penalty for choosing a different split
    //      feature and a smaller penalty for choosing a different split value for the
    //      same feature which is proportional to the difference.

    SSplitStatistics result;

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

    using TDoubleVector = CDenseVector<double>;
    using TDoubleMatrix = CDenseMatrix<double>;
    using TMinimumLoss = std::function<double(const TDoubleVector&, const TDoubleMatrix&)>;

    int d{static_cast<int>(this->numberLossParameters())};

    TMinimumLoss minimumLoss;

    double lambda{regularization.leafWeightPenaltyMultiplier()};
    Eigen::MatrixXd hessian{d, d};
    Eigen::MatrixXd hessian_{d, d};
    Eigen::VectorXd hessianInvg{d};
    if (this->numberLossParameters() == 1) {
        // There is a significant overhead for using a matrix decomposition when g and h
        // are scalar so we have special case handling.
        minimumLoss = [&](const TDoubleVector& g, const TDoubleMatrix& h) -> double {
            return CTools::pow2(g(0)) / (h(0, 0) + lambda);
        };
    } else {
        minimumLoss = [&](const TDoubleVector& g, const TDoubleMatrix& h) -> double {
            hessian_ = hessian =
                (h + lambda * TDoubleMatrix::Identity(d, d)).selfadjointView<Eigen::Lower>();
            // Since the Hessian is positive semidefinite, the trace is larger than the
            // largest eigenvalue. Therefore, H_eps = H + eps * trace(H) * I will have
            // condition number at least eps. As long as eps >> double epsilon we should
            // be able to invert it accurately.
            double eps{std::max(1e-5 * hessian.trace(), 1e-10)};
            for (std::size_t i = 0; i < 2; ++i) {
                Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt{hessian};
                hessianInvg = llt.solve(g);
                if ((hessian_ * hessianInvg - g).norm() < 1e-2 * g.norm()) {
                    return g.transpose() * hessianInvg;
                } else {
                    hessian_.diagonal().array() += eps;
                    hessian = hessian_;
                }
            }
            return -INF / 2.0; // We couldn't invert the Hessian: discard this split.
        };
    }

    TDoubleVector g{d};
    TDoubleMatrix h{d, d};
    TDoubleVector gl[]{TDoubleVector{d}, TDoubleVector{d}};
    TDoubleVector gr[]{TDoubleVector{d}, TDoubleVector{d}};
    TDoubleMatrix hl[]{TDoubleMatrix{d, d}, TDoubleMatrix{d, d}};
    TDoubleMatrix hr[]{TDoubleMatrix{d, d}, TDoubleMatrix{d, d}};

    double gain[2];

    const auto& derivatives = this->derivatives();

    for (auto feature : featureBag) {
        std::size_t c{derivatives.missingCount(feature)};
        g = derivatives.missingGradient(feature);
        h = derivatives.missingCurvature(feature);
        for (const auto& featureDerivatives : derivatives.derivatives(feature)) {
            c += featureDerivatives.count();
            g += featureDerivatives.gradient();
            h += featureDerivatives.curvature();
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
        std::size_t size{derivatives.derivatives(feature).size()};

        for (std::size_t split = 0; split + 1 < size; ++split) {

            std::size_t count{derivatives.count(feature, split)};
            if (count == 0) {
                continue;
            }

            const TMemoryMappedDoubleVector& gradient{derivatives.gradient(feature, split)};
            const TMemoryMappedDoubleMatrix& curvature{derivatives.curvature(feature, split)};

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
            }

            if (gain[ASSIGN_MISSING_TO_LEFT] > maximumGain) {
                maximumGain = gain[ASSIGN_MISSING_TO_LEFT];
                splitAt = this->candidateSplits()[feature][split];
                leftChildRowCount = cl[ASSIGN_MISSING_TO_LEFT];
                assignMissingToLeft = true;
            }
            if (gain[ASSIGN_MISSING_TO_RIGHT] > maximumGain) {
                maximumGain = gain[ASSIGN_MISSING_TO_RIGHT];
                splitAt = this->candidateSplits()[feature][split];
                leftChildRowCount = cl[ASSIGN_MISSING_TO_RIGHT];
                assignMissingToLeft = false;
            }
        }

        double penaltyForDepth{regularization.penaltyForDepth(this->depth())};
        double penaltyForDepthPlusOne{regularization.penaltyForDepth(this->depth() + 1)};

        // The gain is the difference between the quadratic minimum for loss with
        // no split and the loss with the minimum loss split we found.
        double totalGain{0.5 * (maximumGain - minimumLoss(g, h)) -
                         regularization.treeSizePenaltyMultiplier() -
                         regularization.depthPenaltyMultiplier() *
                             (2.0 * penaltyForDepthPlusOne - penaltyForDepth)};
        SSplitStatistics candidate{
            totalGain,
            h.trace() / static_cast<double>(this->numberLossParameters()),
            feature,
            splitAt,
            std::min(leftChildRowCount, c - leftChildRowCount),
            2 * leftChildRowCount < c,
            assignMissingToLeft};
        LOG_TRACE(<< "candidate split: " << candidate.print());

        if (candidate > result) {
            result = candidate;
        }
    }

    LOG_TRACE(<< "best split: " << result.print());

    return result;
}

double
CBoostedTreeLeafNodeStatisticsIncremental::penaltyForTreeChange(const TRegularization& regularization,
                                                                std::size_t feature,
                                                                std::size_t split) const {
    if (m_PreviousSplit == boost::none) {
        return 0.0;
    }

    if (feature != m_PreviousSplit->s_Feature) {
        return regularization.treeTopologyChangePenalty();
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
    double previousSplitAt{CTools::truncate(m_PreviousSplit->s_SplitAt, a, b)};
    return 0.25 * regularization.treeTopologyChangePenalty() *
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

    return SPreviousSplit{rootIndex(), node.splitFeature(), node.splitValue()};
}

CBoostedTreeLeafNodeStatisticsIncremental::TOptionalPreviousSplit
CBoostedTreeLeafNodeStatisticsIncremental::leftChildPreviousSplit(std::size_t feature,
                                                                  const CWorkspace& workspace) const {
    if (workspace.retraining() == nullptr || m_PreviousSplit == boost::none ||
        m_PreviousSplit->s_Feature != feature) {
        return {};
    }

    const auto& tree = *workspace.retraining();
    if (tree[m_PreviousSplit->s_NodeIndex].isLeaf()) {
        return {};
    }

    std::size_t leftChildIndex{tree[m_PreviousSplit->s_NodeIndex].leftChildIndex()};
    const auto& node = tree[leftChildIndex];

    return SPreviousSplit{leftChildIndex, node.splitFeature(), node.splitValue()};
}

CBoostedTreeLeafNodeStatisticsIncremental::TOptionalPreviousSplit
CBoostedTreeLeafNodeStatisticsIncremental::rightChildPreviousSplit(std::size_t feature,
                                                                   const CWorkspace& workspace) const {
    if (workspace.retraining() == nullptr || m_PreviousSplit == boost::none ||
        m_PreviousSplit->s_Feature != feature) {
        return {};
    }

    const auto& tree = *workspace.retraining();
    if (tree[m_PreviousSplit->s_NodeIndex].isLeaf()) {
        return {};
    }

    std::size_t rightChildIndex{tree[m_PreviousSplit->s_NodeIndex].leftChildIndex()};
    auto& node = tree[rightChildIndex];

    return SPreviousSplit{rightChildIndex, node.splitFeature(), node.splitValue()};
}
}
}
