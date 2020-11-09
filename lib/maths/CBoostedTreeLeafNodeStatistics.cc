/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <cmath>
#include <maths/CBoostedTreeLeafNodeStatistics.h>

#include <core/CDataFrame.h>
#include <core/CImmutableRadixSet.h>
#include <core/CLogger.h>

#include <maths/CBoostedTree.h>
#include <maths/CDataFrameCategoryEncoder.h>
#include <maths/CTools.h>

#include <limits>

namespace ml {
namespace maths {
using namespace boosted_tree_detail;
using TRowItr = core::CDataFrame::TRowItr;

namespace {
const std::size_t ASSIGN_MISSING_TO_LEFT{0};
const std::size_t ASSIGN_MISSING_TO_RIGHT{1};

void incrementStatsComputed(CBoostedTreeLeafNodeStatistics::TAnalysisInstrumentationPtr instrumentation) {
    if (instrumentation != nullptr) {
        instrumentation->statisticsComputed() += 1;
    }
}

void incrementStatsNotComputed(CBoostedTreeLeafNodeStatistics::TAnalysisInstrumentationPtr instrumentation) {
    if (instrumentation != nullptr) {
        instrumentation->statisticsNotComputed() += 1;
    }
}

void checkBounds(CBoostedTreeLeafNodeStatistics::TPtr child, double maxGain, double threshold, double leafGain) {
    if (child != nullptr && child->gain() > maxGain && child->gain() > threshold) {
        LOG_DEBUG(<< "The upper bound " << maxGain << " is wrong: threshold "
                  << threshold << " split " << child->print() 
                  << " leaf gain: " << leafGain);
    }
}
}

CBoostedTreeLeafNodeStatistics::CBoostedTreeLeafNodeStatistics(
    std::size_t id,
    const TSizeVec& extraColumns,
    std::size_t numberLossParameters,
    std::size_t numberThreads,
    const core::CDataFrame& frame,
    const CDataFrameCategoryEncoder& encoder,
    const TRegularization& regularization,
    const TImmutableRadixSetVec& candidateSplits,
    const TSizeVec& featureBag,
    std::size_t depth,
    const core::CPackedBitVector& rowMask,
    CWorkspace& workspace)
    : m_Id{id}, m_Depth{depth}, m_ExtraColumns{extraColumns},
      m_NumberLossParameters{numberLossParameters}, m_CandidateSplits{candidateSplits} {

    this->computeAggregateLossDerivatives(numberThreads, frame, encoder, rowMask, workspace);

    // Lazily copy the mask and derivatives to avoid unnecessary allocations.

    m_Derivatives.swap(workspace.reducedDerivatives());
    m_BestSplit = this->computeBestSplitStatistics(regularization, featureBag);
    workspace.reducedDerivatives().swap(m_Derivatives);

    if (this->gain() > workspace.minimumGain()) {
        m_RowMask = rowMask;
        CSplitsDerivatives tmp{workspace.derivatives()[0]};
        m_Derivatives = std::move(tmp);
    }
}

CBoostedTreeLeafNodeStatistics::CBoostedTreeLeafNodeStatistics(
    std::size_t id,
    const CBoostedTreeLeafNodeStatistics& parent,
    std::size_t numberThreads,
    const core::CDataFrame& frame,
    const CDataFrameCategoryEncoder& encoder,
    const TRegularization& regularization,
    const TSizeVec& featureBag,
    bool isLeftChild,
    const CBoostedTreeNode& split,
    CWorkspace& workspace)
    : m_Id{id}, m_Depth{parent.m_Depth + 1}, m_ExtraColumns{parent.m_ExtraColumns},
      m_NumberLossParameters{parent.m_NumberLossParameters}, m_CandidateSplits{
                                                                 parent.m_CandidateSplits} {

    // The number of threads we'll use breaks down as follows:
    //   - We need a minimum number of rows per thread to ensure reasonable
    //     load balancing.
    //   - We need a minimum amount of work per thread to make the overheads
    //     of distributing worthwhile.
    std::size_t features{featureBag.size()};
    std::size_t rows{parent.minimumChildRowCount()};
    std::size_t rowsPerThreadConstraint{rows / 64};
    std::size_t workPerThreadConstraint{(features * rows) / (8 * 128)};
    std::size_t maximumNumberThreads{std::max(
        std::min(rowsPerThreadConstraint, workPerThreadConstraint), std::size_t{1})};
    numberThreads = std::min(numberThreads, maximumNumberThreads);

    this->computeRowMaskAndAggregateLossDerivatives(
        numberThreads, frame, encoder, isLeftChild, split, parent.m_RowMask, workspace);

    // Lazily copy the mask and derivatives to avoid unnecessary allocations.

    m_Derivatives.swap(workspace.reducedDerivatives());
    m_BestSplit = this->computeBestSplitStatistics(regularization, featureBag);
    workspace.reducedDerivatives().swap(m_Derivatives);

    if (this->gain() >= workspace.minimumGain()) {
        CSplitsDerivatives tmp{workspace.reducedDerivatives()};
        m_RowMask = workspace.reducedMask(parent.m_RowMask.size());
        m_Derivatives = std::move(tmp);
    }
}

CBoostedTreeLeafNodeStatistics::CBoostedTreeLeafNodeStatistics(std::size_t id,
                                                               CBoostedTreeLeafNodeStatistics&& parent,
                                                               const TRegularization& regularization,
                                                               const TSizeVec& featureBag,
                                                               CWorkspace& workspace)
    : m_Id{id}, m_Depth{parent.m_Depth + 1}, m_ExtraColumns{parent.m_ExtraColumns},
      m_NumberLossParameters{parent.m_NumberLossParameters},
      m_CandidateSplits{parent.m_CandidateSplits}, m_Derivatives{std::move(
                                                       parent.m_Derivatives)} {

    // Lazily compute the row mask to avoid unnecessary work.

    m_Derivatives.subtract(workspace.reducedDerivatives());
    m_BestSplit = this->computeBestSplitStatistics(regularization, featureBag);
    if (this->gain() >= workspace.minimumGain()) {
        m_RowMask = std::move(parent.m_RowMask);
        m_RowMask ^= workspace.reducedMask(m_RowMask.size());
    }
}

CBoostedTreeLeafNodeStatistics::TPtrPtrPr
CBoostedTreeLeafNodeStatistics::split(std::size_t leftChildId,
                                      std::size_t rightChildId,
                                      std::size_t numberThreads,
                                      double gainThreshold,
                                      const core::CDataFrame& frame,
                                      const CDataFrameCategoryEncoder& encoder,
                                      const TRegularization& regularization,
                                      const TSizeVec& featureBag,
                                      const CBoostedTreeNode& split,
                                      CWorkspace& workspace,
                                      TAnalysisInstrumentationPtr instrumentation) {
    TPtr leftChild;
    TPtr rightChild;
    // if (gainThreshold > 0.0) {
    //     LOG_INFO(<< "threshold " << gainThreshold << " left bound "
    //              << this->m_BestSplit.s_LeftChildMaxGain << " right bound "
    //              << this->m_BestSplit.s_RightChildMaxGain);
    // }
    if (true || this->leftChildHasFewerRows()) {
        if (true || this->m_BestSplit.s_LeftChildMaxGain > gainThreshold) {
            if (this->m_BestSplit.s_LeftChildMaxGain == -73037177.503704607) {
                LOG_DEBUG(<<"Breakpoint");
            }
            leftChild = std::make_shared<CBoostedTreeLeafNodeStatistics>(
                leftChildId, *this, numberThreads, frame, encoder, regularization,
                featureBag, true /*is left child*/, split, workspace);
            checkBounds(leftChild, this->m_BestSplit.s_LeftChildMaxGain, gainThreshold, this->gain());
            incrementStatsComputed(instrumentation);
            if (true || this->m_BestSplit.s_RightChildMaxGain > gainThreshold) {
                rightChild = std::make_shared<CBoostedTreeLeafNodeStatistics>(
                    rightChildId, std::move(*this), regularization, featureBag, workspace);
                checkBounds(rightChild, this->m_BestSplit.s_RightChildMaxGain, gainThreshold, this->gain());
                incrementStatsComputed(instrumentation);
            } else {
                incrementStatsNotComputed(instrumentation);
            }
        } else {
            incrementStatsNotComputed(instrumentation);
            if (true || this->m_BestSplit.s_RightChildMaxGain > gainThreshold) {
                rightChild = std::make_shared<CBoostedTreeLeafNodeStatistics>(
                    rightChildId, *this, numberThreads, frame, encoder, regularization,
                    featureBag, false /*is left child*/, split, workspace);
                checkBounds(rightChild, this->m_BestSplit.s_RightChildMaxGain, gainThreshold, this->gain());
                incrementStatsComputed(instrumentation);
            } else {
                incrementStatsNotComputed(instrumentation);
            }
        }
        return {std::move(leftChild), std::move(rightChild)};
    }

    if (true || this->m_BestSplit.s_RightChildMaxGain > gainThreshold) {
        rightChild = std::make_shared<CBoostedTreeLeafNodeStatistics>(
            rightChildId, *this, numberThreads, frame, encoder, regularization,
            featureBag, false /*is left child*/, split, workspace);
        checkBounds(rightChild, this->m_BestSplit.s_RightChildMaxGain, gainThreshold, this->gain());
        incrementStatsComputed(instrumentation);
        if (true || this->m_BestSplit.s_LeftChildMaxGain > gainThreshold) {
            leftChild = std::make_shared<CBoostedTreeLeafNodeStatistics>(
                leftChildId, std::move(*this), regularization, featureBag, workspace);
            checkBounds(leftChild, this->m_BestSplit.s_LeftChildMaxGain, gainThreshold, this->gain());
            incrementStatsComputed(instrumentation);
        } else {
            incrementStatsNotComputed(instrumentation);
        }
    } else {
        incrementStatsNotComputed(instrumentation);
        if (true || this->m_BestSplit.s_LeftChildMaxGain > gainThreshold) {
            leftChild = std::make_shared<CBoostedTreeLeafNodeStatistics>(
                leftChildId, *this, numberThreads, frame, encoder, regularization,
                featureBag, true /*is left child*/, split, workspace);
            checkBounds(leftChild, this->m_BestSplit.s_LeftChildMaxGain, gainThreshold, this->gain());
            incrementStatsComputed(instrumentation);
        } else {
            incrementStatsNotComputed(instrumentation);
        }
    }
    return {std::move(leftChild), std::move(rightChild)};
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
    const CDataFrameCategoryEncoder& encoder,
    const core::CPackedBitVector& rowMask,
    CWorkspace& workspace) const {

    workspace.newLeaf(numberThreads);

    core::CDataFrame::TRowFuncVec aggregators;
    aggregators.reserve(numberThreads);

    for (std::size_t i = 0; i < numberThreads; ++i) {
        auto& splitsDerivatives = workspace.derivatives()[i];
        splitsDerivatives.zero();
        aggregators.push_back([&](TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                this->addRowDerivatives(encoder.encode(*row), splitsDerivatives);
            }
        });
    }

    frame.readRows(0, frame.numberRows(), aggregators, &rowMask);
}

void CBoostedTreeLeafNodeStatistics::computeRowMaskAndAggregateLossDerivatives(
    std::size_t numberThreads,
    const core::CDataFrame& frame,
    const CDataFrameCategoryEncoder& encoder,
    bool isLeftChild,
    const CBoostedTreeNode& split,
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
        aggregators.push_back([&](TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                auto encodedRow = encoder.encode(*row);
                if (split.assignToLeft(encodedRow) == isLeftChild) {
                    std::size_t index{row->index()};
                    mask.extend(false, index - mask.size());
                    mask.extend(true);
                    this->addRowDerivatives(encodedRow, splitsDerivatives);
                }
            }
        });
    }

    frame.readRows(0, frame.numberRows(), aggregators, &parentRowMask);
}

void CBoostedTreeLeafNodeStatistics::addRowDerivatives(const CEncodedDataFrameRowRef& row,
                                                       CSplitsDerivatives& splitsDerivatives) const {

    auto derivatives = readLossDerivatives(row.unencodedRow(), m_ExtraColumns,
                                           m_NumberLossParameters);

    if (derivatives.size() == 2) {
        if (derivatives(0) >= 0.0) {
            splitsDerivatives.addPositiveDerivatives(derivatives);
        } else {
            splitsDerivatives.addNegativeDerivatives(derivatives);
        }
    }

    std::size_t numberFeatures{m_CandidateSplits.size()};
    for (std::size_t feature = 0; feature < numberFeatures; ++feature) {
        double featureValue{row[feature]};
        if (CDataFrameUtils::isMissing(featureValue)) {
            splitsDerivatives.addMissingDerivatives(feature, derivatives);
        } else {
            std::ptrdiff_t split{m_CandidateSplits[feature].upperBound(featureValue)};
            splitsDerivatives.addDerivatives(feature, split, derivatives);
        }
    }
}

// void CBoostedTreeLeafNodeStatistics::addRowDerivativesUpdateBounds(
//     const CEncodedDataFrameRowRef& row,
//     CSplitsDerivatives& splitsDerivatives) const {

//     auto derivatives = readLossDerivatives(row.unencodedRow(), m_ExtraColumns,
//                                            m_NumberLossParameters);
//     if (derivatives(0) >= 0.0) {
//         splitsDerivatives.addPositiveDerivatives(derivatives);
//     } else {
//         splitsDerivatives.addNegativeDerivatives(derivatives);
//     }

//     std::size_t numberFeatures{m_CandidateSplits.size()};
//     for (std::size_t feature = 0; feature < numberFeatures; ++feature) {
//         double featureValue{row[feature]};
//         if (CDataFrameUtils::isMissing(featureValue)) {
//             splitsDerivatives.addMissingDerivatives(feature, derivatives);
//         } else {
//             std::ptrdiff_t split{m_CandidateSplits[feature].upperBound(featureValue)};
//             splitsDerivatives.addDerivatives(feature, split, derivatives);
//         }
//     }
// }

CBoostedTreeLeafNodeStatistics::SSplitStatistics
CBoostedTreeLeafNodeStatistics::computeBestSplitStatistics(const TRegularization& regularization,
                                                           const TSizeVec& featureBag) const {

    // We have three possible regularization terms we'll use:
    //   1. Tree size: gamma * "node count"
    //   2. Sum square weights: lambda * sum{"leaf weight" ^ 2)}
    //   3. Tree depth: alpha * sum{exp(("depth" / "target depth" - 1.0) / "tolerance")}

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

    int d{static_cast<int>(m_NumberLossParameters)};

    TMinimumLoss minimumLoss;

    double lambda{regularization.leafWeightPenaltyMultiplier()};
    Eigen::MatrixXd hessian{d, d};
    Eigen::MatrixXd hessian_{d, d};
    Eigen::VectorXd hessianInvg{d};
    if (m_NumberLossParameters == 1) {
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
    double minLossLeft[2];
    double minLossRight[2];
    double maxGainLeftChild{-INF};
    double maxGainRightChild{-INF};
    double maxGainLeftChildTop{-INF};
    double maxGainRightChildTop{-INF};
    double gChildLeft{0.0};
    double gChildRight{0.0};

    for (auto feature : featureBag) {
        std::size_t c{m_Derivatives.missingCount(feature)};
        g = m_Derivatives.missingGradient(feature);
        h = m_Derivatives.missingCurvature(feature);
        for (const auto& derivatives : m_Derivatives.derivatives(feature)) {
            c += derivatives.count();
            g += derivatives.gradient();
            h += derivatives.curvature();
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
        std::size_t size{m_Derivatives.derivatives(feature).size()};

        for (std::size_t split = 0; split + 1 < size; ++split) {

            std::size_t count{m_Derivatives.count(feature, split)};
            if (count == 0) {
                continue;
            }

            const TMemoryMappedDoubleVector& gradient{m_Derivatives.gradient(feature, split)};
            const TMemoryMappedDoubleMatrix& curvature{m_Derivatives.curvature(feature, split)};

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
                maxGainLeftChild = minLossLeft[ASSIGN_MISSING_TO_LEFT];
                maxGainRightChild = minLossRight[ASSIGN_MISSING_TO_LEFT];
                gChildLeft = gl[ASSIGN_MISSING_TO_LEFT](0);
                gChildRight = gr[ASSIGN_MISSING_TO_LEFT](0);
                if (gChildLeft == -30650.3759765625) {
                    LOG_DEBUG(<< " Breakpoint ");
                }
            }
            if (gain[ASSIGN_MISSING_TO_RIGHT] > maximumGain) {
                maximumGain = gain[ASSIGN_MISSING_TO_RIGHT];
                splitAt = m_CandidateSplits[feature][split];
                leftChildRowCount = cl[ASSIGN_MISSING_TO_RIGHT];
                assignMissingToLeft = false;
                maxGainLeftChild = minLossLeft[ASSIGN_MISSING_TO_RIGHT];
                maxGainRightChild = minLossRight[ASSIGN_MISSING_TO_RIGHT];
                gChildLeft = gl[ASSIGN_MISSING_TO_RIGHT](0);
                gChildRight = gr[ASSIGN_MISSING_TO_RIGHT](0);
                if (gChildLeft == -30650.3759765625) {
                    LOG_DEBUG(<< " Breakpoint ");
                }
            }
        }

        double penaltyForDepth{regularization.penaltyForDepth(m_Depth)};
        double penaltyForDepthPlusOne{regularization.penaltyForDepth(m_Depth + 1)};

        // The gain is the difference between the quadratic minimum for loss with
        // no split and the loss with the minimum loss split we found.   
        double gain{0.5 * (maximumGain - minimumLoss(g, h)) -
                    regularization.treeSizePenaltyMultiplier() -
                    regularization.depthPenaltyMultiplier() *
                        (2.0 * penaltyForDepthPlusOne - penaltyForDepth)};
        if (gain == 56730970.470521882) {
            LOG_DEBUG(<<"Breakpoint maximumGain = " << maximumGain << " minimumLoss " << minimumLoss(g, h)
            << " diifference " << ( maximumGain - minimumLoss(g, h))
            << " regularization " << -(-
                    regularization.treeSizePenaltyMultiplier() -
                    regularization.depthPenaltyMultiplier() *
                        (2.0 * penaltyForDepthPlusOne - penaltyForDepth))
            << " g " << (g(0)) << " feature " << feature << " again " << (m_Derivatives.missingGradient(feature)(0)));
        }
        SSplitStatistics candidate{gain,
                                   h.trace() / static_cast<double>(m_NumberLossParameters),
                                   feature,
                                   splitAt,
                                   std::min(leftChildRowCount, c - leftChildRowCount),
                                   2 * leftChildRowCount < c,
                                   assignMissingToLeft,
                                   gChildLeft,
                                   gChildRight};
        LOG_TRACE(<< "candidate split: " << candidate.print());

        if (candidate > result) {
            result = candidate;
            maxGainLeftChildTop = maxGainLeftChild;
            maxGainRightChildTop = maxGainRightChild;
        }
    }
    if (result.s_Gain > 0) {
    if (result.s_LeftChildMaxGain != 0.0) {
        maxGainLeftChild = computeChildGain(result.s_LeftChildMaxGain, lambda, maxGainLeftChildTop);
    }

    if (result.s_RightChildMaxGain != 0.0) {
        maxGainRightChild = computeChildGain(result.s_RightChildMaxGain, lambda, maxGainRightChildTop);
    }

    double childPenaltyForDepth{regularization.penaltyForDepth(m_Depth + 1)};
    double childPenaltyForDepthPlusOne{regularization.penaltyForDepth(m_Depth + 2)};
    double childPenalty{regularization.treeSizePenaltyMultiplier() +
                        regularization.depthPenaltyMultiplier() *
                            (2.0 * childPenaltyForDepthPlusOne - childPenaltyForDepth)};
    result.s_LeftChildMaxGain = 0.5 * maxGainLeftChild - childPenalty;
    result.s_RightChildMaxGain = 0.5 * maxGainRightChild - childPenalty;
    if (result.s_LeftChildMaxGain == -73037177.503704607) {
        LOG_DEBUG(<< "Breakpoint");
    }
    }
    else {
        result.s_LeftChildMaxGain = -INF;
        result.s_RightChildMaxGain = -INF;
    }

    LOG_TRACE(<< "best split: " << result.print());

    return result;
}

double CBoostedTreeLeafNodeStatistics::computeChildGain(double gChild,
                                                        double lambda,
                                                        double maxGainChild) const {
    double positiveDerivativesGSum =
        std::max(std::min(gChild - m_Derivatives.negativeDerivativesGSum(),
                          m_Derivatives.positiveDerivativesGSum()),
                 0.0);
    double negativeDerivativesGSum =
        std::min(std::max(gChild - m_Derivatives.positiveDerivativesGSum(),
                          m_Derivatives.negativeDerivativesGSum()),
                 0.0);
    // double positiveDerivativesGSum(m_Derivatives.positiveDerivativesGSum());
    // double negativeDerivativesGSum(m_Derivatives.negativeDerivativesGSum());

    if (positiveDerivativesGSum == 0.0 && negativeDerivativesGSum == 0.0) {
        LOG_INFO(<< "Something is wrong positive " << m_Derivatives.positiveDerivativesGSum() 
        << " negative " <<  m_Derivatives.negativeDerivativesGSum() 
        << " gChild " << gChild);
    }
    // double positiveDerivativesGSum = m_Derivatives.positiveDerivativesGSum();
    // double negativeDerivativesGSum = m_Derivatives.negativeDerivativesGSum();
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
    if (isnan(lookAheadGain)) {
        LOG_INFO(<< "Look ahead gain " << lookAheadGain << " negativeDerivativesGSum "
                 << negativeDerivativesGSum << " positiveDerivativesGSum " << positiveDerivativesGSum
                 << " gChild " << gChild << " m_Derivatives.negativeDerivativesGSum() "
                 << m_Derivatives.negativeDerivativesGSum() << " m_Derivatives.positiveDerivativesGSum() "
                 << m_Derivatives.positiveDerivativesGSum());
    }
    return lookAheadGain - maxGainChild;
}
}
}
