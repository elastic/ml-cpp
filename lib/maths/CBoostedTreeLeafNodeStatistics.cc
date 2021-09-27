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

#include <maths/CBoostedTreeLeafNodeStatistics.h>

#include <core/CDataFrame.h>
#include <core/CLogger.h>

#include <maths/CBoostedTree.h>
#include <maths/CDataFrameCategoryEncoder.h>
#include <maths/CTools.h>

#include <algorithm>
#include <limits>

namespace ml {
namespace maths {
using namespace boosted_tree_detail;
using TRowItr = core::CDataFrame::TRowItr;

namespace {
const std::size_t ASSIGN_MISSING_TO_LEFT{0};
const std::size_t ASSIGN_MISSING_TO_RIGHT{1};
}

CBoostedTreeLeafNodeStatistics::CBoostedTreeLeafNodeStatistics(
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
    : m_Id{id}, m_Depth{depth}, m_ExtraColumns{extraColumns},
      m_NumberLossParameters{numberLossParameters}, m_CandidateSplits{candidateSplits} {

    this->computeAggregateLossDerivatives(workspace.numberThreads(), frame,
                                          treeFeatureBag, rowMask, workspace);

    // Lazily copy the mask and derivatives to avoid unnecessary allocations.

    m_Derivatives.swap(workspace.reducedDerivatives(treeFeatureBag));
    m_BestSplit = this->computeBestSplitStatistics(workspace.numberThreads(),
                                                   regularization, nodeFeatureBag);
    workspace.reducedDerivatives(treeFeatureBag).swap(m_Derivatives);

    if (this->gain() >= workspace.minimumGain()) {
        m_RowMask = rowMask;
        CSplitsDerivatives tmp{workspace.derivatives()[0]};
        m_Derivatives = std::move(tmp);
    }
}

CBoostedTreeLeafNodeStatistics::CBoostedTreeLeafNodeStatistics(
    std::size_t id,
    const CBoostedTreeLeafNodeStatistics& parent,
    const core::CDataFrame& frame,
    const TRegularization& regularization,
    const TSizeVec& treeFeatureBag,
    const TSizeVec& nodeFeatureBag,
    bool isLeftChild,
    const CBoostedTreeNode& split,
    CWorkspace& workspace)
    : m_Id{id}, m_Depth{parent.m_Depth + 1}, m_ExtraColumns{parent.m_ExtraColumns},
      m_NumberLossParameters{parent.m_NumberLossParameters}, m_CandidateSplits{
                                                                 parent.m_CandidateSplits} {

    this->computeRowMaskAndAggregateLossDerivatives(
        TThreading::numberThreadsForAggregateLossDerivatives(
            workspace.numberThreads(), treeFeatureBag.size(), parent.minimumChildRowCount()),
        frame, isLeftChild, split, treeFeatureBag, parent.m_RowMask, workspace);

    m_Derivatives.swap(workspace.reducedDerivatives(treeFeatureBag));
    m_BestSplit = this->computeBestSplitStatistics(workspace.numberThreads(),
                                                   regularization, nodeFeatureBag);
    workspace.reducedDerivatives(treeFeatureBag).swap(m_Derivatives);

    // Lazily copy the mask and derivatives to avoid unnecessary allocations.
    if (this->gain() >= workspace.minimumGain()) {
        CSplitsDerivatives tmp{workspace.reducedDerivatives(treeFeatureBag)};
        m_RowMask = workspace.reducedMask(parent.m_RowMask.size());
        m_Derivatives = std::move(tmp);
    }
}

CBoostedTreeLeafNodeStatistics::CBoostedTreeLeafNodeStatistics(
    std::size_t id,
    CBoostedTreeLeafNodeStatistics&& parent,
    const TRegularization& regularization,
    const TSizeVec& treeFeatureBag,
    const TSizeVec& nodeFeatureBag,
    CWorkspace& workspace)
    : m_Id{id}, m_Depth{parent.m_Depth + 1}, m_ExtraColumns{parent.m_ExtraColumns},
      m_NumberLossParameters{parent.m_NumberLossParameters},
      m_CandidateSplits{parent.m_CandidateSplits}, m_Derivatives{std::move(
                                                       parent.m_Derivatives)} {

    // TODO if sum(splits) > |feature| * |rows| aggregate.
    m_Derivatives.subtract(workspace.numberThreads(),
                           workspace.reducedDerivatives(treeFeatureBag), treeFeatureBag);
    m_BestSplit = this->computeBestSplitStatistics(workspace.numberThreads(),
                                                   regularization, nodeFeatureBag);

    // Lazily compute the row mask to avoid unnecessary work.
    if (this->gain() >= workspace.minimumGain()) {
        m_RowMask = std::move(parent.m_RowMask);
        m_RowMask ^= workspace.reducedMask(m_RowMask.size());
    }
}

CBoostedTreeLeafNodeStatistics::TPtrPtrPr
CBoostedTreeLeafNodeStatistics::split(std::size_t leftChildId,
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
        if (this->m_BestSplit.s_LeftChildMaxGain > gainThreshold) {
            leftChild = std::make_shared<CBoostedTreeLeafNodeStatistics>(
                leftChildId, *this, frame, regularization, treeFeatureBag,
                nodeFeatureBag, true /*is left child*/, split, workspace);
            if (this->m_BestSplit.s_RightChildMaxGain > gainThreshold) {
                rightChild = std::make_shared<CBoostedTreeLeafNodeStatistics>(
                    rightChildId, std::move(*this), regularization,
                    treeFeatureBag, nodeFeatureBag, workspace);
            }
        } else if (this->m_BestSplit.s_RightChildMaxGain > gainThreshold) {
            rightChild = std::make_shared<CBoostedTreeLeafNodeStatistics>(
                rightChildId, *this, frame, regularization, treeFeatureBag,
                nodeFeatureBag, false /*is left child*/, split, workspace);
        }
        return {std::move(leftChild), std::move(rightChild)};
    }

    if (this->m_BestSplit.s_RightChildMaxGain > gainThreshold) {
        rightChild = std::make_shared<CBoostedTreeLeafNodeStatistics>(
            rightChildId, *this, frame, regularization, treeFeatureBag,
            nodeFeatureBag, false /*is left child*/, split, workspace);
        if (this->m_BestSplit.s_LeftChildMaxGain > gainThreshold) {
            leftChild = std::make_shared<CBoostedTreeLeafNodeStatistics>(
                leftChildId, std::move(*this), regularization, treeFeatureBag,
                nodeFeatureBag, workspace);
        }
    } else if (this->m_BestSplit.s_LeftChildMaxGain > gainThreshold) {
        leftChild = std::make_shared<CBoostedTreeLeafNodeStatistics>(
            leftChildId, *this, frame, regularization, treeFeatureBag,
            nodeFeatureBag, true /*is left child*/, split, workspace);
    }
    return {std::move(leftChild), std::move(rightChild)};
}

CBoostedTreeLeafNodeStatistics::CBoostedTreeLeafNodeStatistics(const TSizeVec& extraColumns,
                                                               std::size_t numberLossParameters,
                                                               const TFloatVecVec& candidateSplits,
                                                               CSplitsDerivatives derivatives)
    : m_Id{0}, m_Depth{0}, m_ExtraColumns{extraColumns}, m_NumberLossParameters{numberLossParameters},
      m_CandidateSplits{candidateSplits}, m_Derivatives{std::move(derivatives)} {
}

bool CBoostedTreeLeafNodeStatistics::operator<(const CBoostedTreeLeafNodeStatistics& rhs) const {
    return COrderings::lexicographical_compare(m_BestSplit, m_Id, rhs.m_BestSplit, rhs.m_Id);
}

double CBoostedTreeLeafNodeStatistics::gain() const {
    return m_BestSplit.s_Gain;
}

double CBoostedTreeLeafNodeStatistics::leftChildMaxGain() const {
    return m_BestSplit.s_LeftChildMaxGain;
}

double CBoostedTreeLeafNodeStatistics::rightChildMaxGain() const {
    return m_BestSplit.s_RightChildMaxGain;
}

double CBoostedTreeLeafNodeStatistics::curvature() const {
    return m_BestSplit.s_Curvature;
}

CBoostedTreeLeafNodeStatistics::TSizeDoublePr CBoostedTreeLeafNodeStatistics::bestSplit() const {
    return {m_BestSplit.s_Feature, m_BestSplit.s_SplitAt};
}

std::size_t CBoostedTreeLeafNodeStatistics::minimumChildRowCount() const {
    return m_BestSplit.s_MinimumChildRowCount;
}

bool CBoostedTreeLeafNodeStatistics::leftChildHasFewerRows() const {
    return m_BestSplit.s_LeftChildHasFewerRows;
}

bool CBoostedTreeLeafNodeStatistics::assignMissingToLeft() const {
    return m_BestSplit.s_AssignMissingToLeft;
}

std::size_t CBoostedTreeLeafNodeStatistics::id() const {
    return m_Id;
}

core::CPackedBitVector& CBoostedTreeLeafNodeStatistics::rowMask() {
    return m_RowMask;
}

std::size_t CBoostedTreeLeafNodeStatistics::memoryUsage() const {
    return core::CMemory::dynamicSize(m_RowMask) + core::CMemory::dynamicSize(m_Derivatives);
}

std::size_t
CBoostedTreeLeafNodeStatistics::estimateMemoryUsage(std::size_t numberFeatures,
                                                    std::size_t numberSplitsPerFeature,
                                                    std::size_t numberLossParameters) {
    // See CBoostedTreeImpl::estimateMemoryUsage for a discussion of the cost
    // of the row mask.
    std::size_t splitsDerivativesSize{CSplitsDerivatives::estimateMemoryUsage(
        numberFeatures, numberSplitsPerFeature, numberLossParameters)};
    return sizeof(CBoostedTreeLeafNodeStatistics) + splitsDerivativesSize;
}

void CBoostedTreeLeafNodeStatistics::computeAggregateLossDerivatives(
    std::size_t numberThreads,
    const core::CDataFrame& frame,
    const TSizeVec& featureBag,
    const core::CPackedBitVector& rowMask,
    CWorkspace& workspace) const {

    workspace.newLeaf(numberThreads);

    core::CDataFrame::TRowFuncVec aggregators;
    aggregators.reserve(numberThreads);

    for (std::size_t i = 0; i < numberThreads; ++i) {
        auto& splitsDerivatives = workspace.derivatives()[i];
        splitsDerivatives.zero();
        aggregators.push_back([&](const TRowItr& beginRows, const TRowItr& endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                this->addRowDerivatives(featureBag, *row, splitsDerivatives);
            }
        });
    }

    frame.readRows(0, frame.numberRows(), aggregators, &rowMask);
}

void CBoostedTreeLeafNodeStatistics::computeRowMaskAndAggregateLossDerivatives(
    std::size_t numberThreads,
    const core::CDataFrame& frame,
    bool isLeftChild,
    const CBoostedTreeNode& split,
    const TSizeVec& featureBag,
    const core::CPackedBitVector& parentRowMask,
    CWorkspace& workspace) const {

    workspace.newLeaf(numberThreads);

    core::CDataFrame::TRowFuncVec aggregators;
    aggregators.reserve(numberThreads);

    for (std::size_t i = 0; i < numberThreads; ++i) {
        auto& mask = workspace.masks()[i];
        auto& splitsDerivatives = workspace.derivatives()[i];
        mask.clear();
        splitsDerivatives.zero();
        aggregators.push_back([&](const TRowItr& beginRows, const TRowItr& endRows) {
            for (auto row_ = beginRows; row_ != endRows; ++row_) {
                auto row = *row_;
                if (split.assignToLeft(row, m_ExtraColumns) == isLeftChild) {
                    std::size_t index{row.index()};
                    mask.extend(false, index - mask.size());
                    mask.extend(true);
                    this->addRowDerivatives(featureBag, row, splitsDerivatives);
                }
            }
        });
    }

    frame.readRows(0, frame.numberRows(), aggregators, &parentRowMask);
}

void CBoostedTreeLeafNodeStatistics::addRowDerivatives(const TSizeVec& featureBag,
                                                       const TRowRef& row,
                                                       CSplitsDerivatives& splitsDerivatives) const {

    auto derivatives = readLossDerivatives(row, m_ExtraColumns, m_NumberLossParameters);

    if (derivatives.size() == 2) {
        if (derivatives(0) >= 0.0) {
            splitsDerivatives.addPositiveDerivatives(derivatives);
        } else {
            splitsDerivatives.addNegativeDerivatives(derivatives);
        }
    }

    const auto* splits = beginSplits(row, m_ExtraColumns);
    for (auto feature : featureBag) {
        std::size_t split{static_cast<std::size_t>(
            CPackedUInt8Decorator{splits[feature >> 2]}.readBytes()[feature & 0x3])};
        splitsDerivatives.addDerivatives(feature, split, derivatives);
    }
}

CBoostedTreeLeafNodeStatistics::SSplitStatistics
CBoostedTreeLeafNodeStatistics::computeBestSplitStatistics(std::size_t numberThreads,
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

    using TSplitSearchVec = std::vector<std::function<void(std::size_t)>>;
    using TSplitStatisticsVec = std::vector<SSplitStatistics>;
    using TChildrenGainStatisticsVec = std::vector<SChildrenGainStatistics>;

    numberThreads = TThreading::numberThreadsForComputeBestSplitStatistics(
        numberThreads, featureBag.size(), m_NumberLossParameters,
        m_Derivatives.numberDerivatives(featureBag));
    LOG_TRACE(<< "number threads = " << numberThreads);

    TSplitSearchVec bestSplitSearches;
    TSplitStatisticsVec splitStats(numberThreads);
    TChildrenGainStatisticsVec childrenGainStatistics(numberThreads);
    bestSplitSearches.reserve(numberThreads);

    for (std::size_t i = 0; i < numberThreads; ++i) {
        bestSplitSearches.push_back(this->featureBestSplitSearch(
            regularization, splitStats[i], childrenGainStatistics[i]));
    }

    core::parallel_for_each(featureBag.begin(), featureBag.end(), bestSplitSearches);

    SSplitStatistics result;
    SChildrenGainStatistics childrenGainStatisticsGlobal;
    for (std::size_t i = 0; i < numberThreads; ++i) {
        if (splitStats[i] > result) {
            result = splitStats[i];
            childrenGainStatisticsGlobal = childrenGainStatistics[i];
        }
    }

    if (m_Derivatives.numberLossParameters() <= 2 && result.s_Gain > 0) {
        double lambda{regularization.leafWeightPenaltyMultiplier()};
        double childPenaltyForDepth{regularization.penaltyForDepth(m_Depth + 1)};
        double childPenaltyForDepthPlusOne{regularization.penaltyForDepth(m_Depth + 2)};
        double childPenalty{regularization.treeSizePenaltyMultiplier() +
                            regularization.depthPenaltyMultiplier() *
                                (2.0 * childPenaltyForDepthPlusOne - childPenaltyForDepth)};
        result.s_LeftChildMaxGain =
            0.5 * this->childMaxGain(childrenGainStatisticsGlobal.s_GLeft,
                                     childrenGainStatisticsGlobal.s_MinLossLeft, lambda) -
            childPenalty;

        result.s_RightChildMaxGain =
            0.5 * this->childMaxGain(childrenGainStatisticsGlobal.s_GRight,
                                     childrenGainStatisticsGlobal.s_MinLossRight, lambda) -
            childPenalty;
    }

    LOG_TRACE(<< "best split: " << result.print());

    return result;
}

CBoostedTreeLeafNodeStatistics::TBestSplitSearch CBoostedTreeLeafNodeStatistics::featureBestSplitSearch(
    const TRegularization& regularization,
    SSplitStatistics& bestSplitStatistics,
    SChildrenGainStatistics& childrenGainStatisticsGlobal) const {
    using TDoubleAry = std::array<double, 2>;
    using TDoubleVector = CDenseVector<double>;
    using TDoubleVectorAry = std::array<TDoubleVector, 2>;
    using TDoubleMatrix = CDenseMatrix<double>;
    using TDoubleMatrixAry = std::array<TDoubleMatrix, 2>;

    int d{static_cast<int>(m_NumberLossParameters)};
    double lambda{regularization.leafWeightPenaltyMultiplier()};
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

        std::size_t c{m_Derivatives.missingCount(feature)};
        g = m_Derivatives.missingGradient(feature);
        h = m_Derivatives.missingCurvature(feature);
        for (auto derivatives = m_Derivatives.beginDerivatives(feature);
             derivatives != m_Derivatives.endDerivatives(feature); ++derivatives) {
            c += derivatives->count();
            g += derivatives->gradient();
            h += derivatives->curvature();
        }
        std::size_t cl[]{m_Derivatives.missingCount(feature), 0};
        gl[ASSIGN_MISSING_TO_LEFT] = m_Derivatives.missingGradient(feature);
        gl[ASSIGN_MISSING_TO_RIGHT] = TDoubleVector::Zero(g.rows());
        gr[ASSIGN_MISSING_TO_LEFT] = g - m_Derivatives.missingGradient(feature);
        gr[ASSIGN_MISSING_TO_RIGHT] = g;
        hl[ASSIGN_MISSING_TO_LEFT] = m_Derivatives.missingCurvature(feature);
        hl[ASSIGN_MISSING_TO_RIGHT] = TDoubleMatrix::Zero(h.rows(), h.cols());
        hr[ASSIGN_MISSING_TO_LEFT] = h - m_Derivatives.missingCurvature(feature);
        hr[ASSIGN_MISSING_TO_RIGHT] = h;

        double maximumGain{-INF};
        double splitAt{-INF};
        std::size_t leftChildRowCount{0};
        bool assignMissingToLeft{true};
        std::size_t size{m_Derivatives.numberDerivatives(feature)};
        TDoubleAry gain;
        TDoubleAry minLossLeft;
        TDoubleAry minLossRight;
        SChildrenGainStatistics childrenGainStatisticsPerFeature;

        for (std::size_t split = 0; split + 1 < size; ++split) {

            std::size_t count{m_Derivatives.count(feature, split)};
            if (count == 0) {
                continue;
            }

            const auto& gradient = m_Derivatives.gradient(feature, split);
            const auto& curvature = m_Derivatives.curvature(feature, split);

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
                splitAt = m_CandidateSplits[feature][split];
                leftChildRowCount = cl[ASSIGN_MISSING_TO_LEFT];
                assignMissingToLeft = true;
                // If gain > -INF then minLossLeft and minLossRight were initialized.
                childrenGainStatisticsPerFeature = {
                    minLossLeft[ASSIGN_MISSING_TO_LEFT], minLossRight[ASSIGN_MISSING_TO_LEFT],
                    gl[ASSIGN_MISSING_TO_LEFT](0), gr[ASSIGN_MISSING_TO_LEFT](0)};
            }
            if (gain[ASSIGN_MISSING_TO_RIGHT] > maximumGain) {
                maximumGain = gain[ASSIGN_MISSING_TO_RIGHT];
                splitAt = m_CandidateSplits[feature][split];
                leftChildRowCount = cl[ASSIGN_MISSING_TO_RIGHT];
                assignMissingToLeft = false;
                // If gain > -INF then minLossLeft and minLossRight were initialized.
                childrenGainStatisticsPerFeature = {
                    minLossLeft[ASSIGN_MISSING_TO_RIGHT],
                    minLossRight[ASSIGN_MISSING_TO_RIGHT],
                    gl[ASSIGN_MISSING_TO_RIGHT](0), gr[ASSIGN_MISSING_TO_RIGHT](0)};
            }
        }

        double penaltyForDepth{regularization.penaltyForDepth(m_Depth)};
        double penaltyForDepthPlusOne{regularization.penaltyForDepth(m_Depth + 1)};

        // The gain is the difference between the quadratic minimum for loss with
        // no split and the loss with the minimum loss split we found.
        double totalGain{0.5 * (maximumGain - minimumLoss(g, h)) -
                         regularization.treeSizePenaltyMultiplier() -
                         regularization.depthPenaltyMultiplier() *
                             (2.0 * penaltyForDepthPlusOne - penaltyForDepth)};
        SSplitStatistics candidateSplitStatistics{
            totalGain,
            h.trace() / static_cast<double>(m_NumberLossParameters),
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

double CBoostedTreeLeafNodeStatistics::childMaxGain(double gChild,
                                                    double minLossChild,
                                                    double lambda) const {

    // This computes the maximum possible gain we can expect splitting a child node given
    // we know the sum of the positive (g^+) and negative gradients (g^-) at its parent,
    // the minimum curvature on the positive and negative gradient set (hmin^+ and hmin^-)
    // and largest and smallest gradient (gmax and gmin, respectively). The highest possible
    // gain consistent with these constraints can be shown to be:
    // (g^+)^2 / (hmin^+ * g^+ / gmax + lambda) + (g^-)^2 / (hmin^- * g^- / gmin + lambda)
    // Since gchild = gchild^+ + gchild^-, we can improve estimates on g^+ and g^- for the child as:
    // g^+ = max(min(gchild - g^-, g^+), 0),
    // g^- = max(min(gchild - g^+, g^-), 0).
    double positiveDerivativesGSum =
        std::max(std::min(gChild - m_Derivatives.negativeDerivativesGSum(),
                          m_Derivatives.positiveDerivativesGSum()),
                 0.0);
    double negativeDerivativesGSum =
        std::min(std::max(gChild - m_Derivatives.positiveDerivativesGSum(),
                          m_Derivatives.negativeDerivativesGSum()),
                 0.0);
    double lookAheadGain{((positiveDerivativesGSum != 0.0)
                              ? CTools::pow2(positiveDerivativesGSum) /
                                    (m_Derivatives.positiveDerivativesHMin() * positiveDerivativesGSum /
                                         m_Derivatives.positiveDerivativesGMax() +
                                     lambda + 1e-10)
                              : 0.0) +
                         ((negativeDerivativesGSum != 0.0)
                              ? CTools::pow2(negativeDerivativesGSum) /
                                    (m_Derivatives.negativeDerivativesHMin() * negativeDerivativesGSum /
                                         m_Derivatives.negativeDerivativesGMin() +
                                     lambda + 1e-10)
                              : 0.0)};
    return lookAheadGain - minLossChild;
}
}
}
