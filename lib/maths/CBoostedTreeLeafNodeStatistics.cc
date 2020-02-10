/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBoostedTreeLeafNodeStatistics.h>

#include <core/CDataFrame.h>
#include <core/CImmutableRadixSet.h>
#include <core/CLogger.h>
#include <core/CMemory.h>

#include <maths/CBoostedTree.h>
#include <maths/CDataFrameCategoryEncoder.h>
#include <maths/CTools.h>

#include <maths/CMathsFuncs.h>

namespace ml {
namespace maths {
using namespace boosted_tree_detail;
using TRowItr = core::CDataFrame::TRowItr;

namespace {
const std::size_t ASSIGN_MISSING_TO_LEFT{0};
const std::size_t ASSIGN_MISSING_TO_RIGHT{1};
const double SMALLEST_RELATIVE_CURVATURE{1e-20};
}

CBoostedTreeLeafNodeStatistics::CBoostedTreeLeafNodeStatistics(
    std::size_t id,
    std::size_t numberInputColumns,
    std::size_t numberLossParameters,
    std::size_t numberThreads,
    const core::CDataFrame& frame,
    const CDataFrameCategoryEncoder& encoder,
    const TRegularization& regularization,
    const TImmutableRadixSetVec& candidateSplits,
    const TSizeVec& featureBag,
    std::size_t depth,
    const core::CPackedBitVector& rowMask)
    : m_Id{id}, m_Depth{depth}, m_NumberInputColumns{numberInputColumns},
      m_NumberLossParameters{numberLossParameters}, m_CandidateSplits{candidateSplits}, m_RowMask{rowMask} {

    this->computeAggregateLossDerivatives(numberThreads, frame, encoder);
    m_BestSplit = this->computeBestSplitStatistics(regularization, featureBag);
}

CBoostedTreeLeafNodeStatistics::CBoostedTreeLeafNodeStatistics(
    std::size_t id,
    std::size_t numberInputColumns,
    std::size_t numberLossParameters,
    std::size_t numberThreads,
    const core::CDataFrame& frame,
    const CDataFrameCategoryEncoder& encoder,
    const TRegularization& regularization,
    const TImmutableRadixSetVec& candidateSplits,
    const TSizeVec& featureBag,
    bool isLeftChild,
    std::size_t depth,
    const CBoostedTreeNode& split,
    const core::CPackedBitVector& parentRowMask)
    : m_Id{id}, m_Depth{depth}, m_NumberInputColumns{numberInputColumns},
      m_NumberLossParameters{numberLossParameters}, m_CandidateSplits{candidateSplits} {

    this->computeRowMaskAndAggregateLossDerivatives(
        numberThreads, frame, encoder, isLeftChild, split, parentRowMask);
    m_BestSplit = this->computeBestSplitStatistics(regularization, featureBag);
}

CBoostedTreeLeafNodeStatistics::CBoostedTreeLeafNodeStatistics(
    std::size_t id,
    const CBoostedTreeLeafNodeStatistics& parent,
    const CBoostedTreeLeafNodeStatistics& sibling,
    const TRegularization& regularization,
    const TSizeVec& featureBag,
    core::CPackedBitVector rowMask)
    : m_Id{id}, m_Depth{sibling.m_Depth}, m_NumberInputColumns{sibling.m_NumberInputColumns},
      m_NumberLossParameters{sibling.m_NumberLossParameters},
      m_CandidateSplits{sibling.m_CandidateSplits}, m_RowMask{std::move(rowMask)} {

    m_Derivatives.resize(m_CandidateSplits.size());
    m_MissingDerivatives.reserve(m_CandidateSplits.size());

    for (std::size_t i = 0; i < m_CandidateSplits.size(); ++i) {
        std::size_t numberSplits{m_CandidateSplits[i].size() + 1};
        m_Derivatives[i].reserve(numberSplits);
        for (std::size_t j = 0; j < numberSplits; ++j) {
            // Numeric errors mean that it's possible the sum curvature for a candidate
            // split is identically zero while the gradient is epsilon. This can cause
            // the node gain to appear infinite (when there is no weight regularisation)
            // which in turns causes problems initialising the region we search for optimal
            // hyperparameter values. We can safely force the gradient and curvature to
            // be zero if we detect that the count is zero. Also, none of our loss functions
            // have negative curvature therefore we shouldn't allow the cumulative curvature
            // to be negative either. In this case we force it to be a v.small multiple
            // of the magnitude of the gradient since this is the closest feasible estimate.
            std::size_t count{parent.m_Derivatives[i][j].s_Count -
                              sibling.m_Derivatives[i][j].s_Count};
            if (count > 0) {
                TDoubleVector gradient{parent.m_Derivatives[i][j].s_Gradient -
                                       sibling.m_Derivatives[i][j].s_Gradient};
                TDoubleMatrix curvature{parent.m_Derivatives[i][j].s_Curvature -
                                        sibling.m_Derivatives[i][j].s_Curvature};
                for (int k = 0; k < gradient.size(); ++k) {
                    curvature(k, k) =
                        std::max(curvature(k, k), SMALLEST_RELATIVE_CURVATURE *
                                                      std::fabs(gradient(k)));
                }
                m_Derivatives[i].emplace_back(count, std::move(gradient),
                                              std::move(curvature));
            } else {
                m_Derivatives[i].emplace_back(
                    0, TDoubleVector{TDoubleVector::Zero(m_NumberLossParameters)},
                    TDoubleMatrix{TDoubleMatrix::Zero(m_NumberLossParameters,
                                                      m_NumberLossParameters)});
            }
        }
        std::size_t count{parent.m_MissingDerivatives[i].s_Count -
                          sibling.m_MissingDerivatives[i].s_Count};
        if (count > 0) {
            TDoubleVector gradient{parent.m_MissingDerivatives[i].s_Gradient -
                                   sibling.m_MissingDerivatives[i].s_Gradient};
            TDoubleVector curvature{parent.m_MissingDerivatives[i].s_Curvature -
                                    sibling.m_MissingDerivatives[i].s_Curvature};
            for (int k = 0; k < gradient.size(); ++k) {
                curvature(k, k) =
                    std::max(curvature(k, k),
                             SMALLEST_RELATIVE_CURVATURE * std::fabs(gradient(k)));
            }
            m_MissingDerivatives.emplace_back(count, std::move(gradient),
                                              std::move(curvature));
        } else {
            m_MissingDerivatives.emplace_back(
                0, TDoubleVector{TDoubleVector::Zero(m_NumberLossParameters)},
                TDoubleMatrix{TDoubleMatrix::Zero(m_NumberLossParameters, m_NumberLossParameters)});
        }
    }

    m_BestSplit = this->computeBestSplitStatistics(regularization, featureBag);
}

CBoostedTreeLeafNodeStatistics::TPtrPtrPr
CBoostedTreeLeafNodeStatistics::split(std::size_t leftChildId,
                                      std::size_t rightChildId,
                                      std::size_t numberThreads,
                                      const core::CDataFrame& frame,
                                      const CDataFrameCategoryEncoder& encoder,
                                      const TRegularization& regularization,
                                      const TImmutableRadixSetVec& candidateSplits,
                                      const TSizeVec& featureBag,
                                      const CBoostedTreeNode& split,
                                      bool leftChildHasFewerRows) {
    if (leftChildHasFewerRows) {
        auto leftChild = std::make_shared<CBoostedTreeLeafNodeStatistics>(
            leftChildId, m_NumberInputColumns, m_NumberLossParameters,
            numberThreads, frame, encoder, regularization, candidateSplits,
            featureBag, true /*is left child*/, m_Depth + 1, split, m_RowMask);
        core::CPackedBitVector rightChildRowMask{m_RowMask};
        rightChildRowMask ^= leftChild->rowMask();
        auto rightChild = std::make_shared<CBoostedTreeLeafNodeStatistics>(
            rightChildId, *this, *leftChild, regularization, featureBag,
            std::move(rightChildRowMask));

        return std::make_pair(leftChild, rightChild);
    }

    auto rightChild = std::make_shared<CBoostedTreeLeafNodeStatistics>(
        rightChildId, m_NumberInputColumns, m_NumberLossParameters,
        numberThreads, frame, encoder, regularization, candidateSplits,
        featureBag, false /*is left child*/, m_Depth + 1, split, m_RowMask);
    core::CPackedBitVector leftChildRowMask{m_RowMask};
    leftChildRowMask ^= rightChild->rowMask();
    auto leftChild = std::make_shared<CBoostedTreeLeafNodeStatistics>(
        leftChildId, *this, *rightChild, regularization, featureBag,
        std::move(leftChildRowMask));

    return std::make_pair(leftChild, rightChild);
}

bool CBoostedTreeLeafNodeStatistics::operator<(const CBoostedTreeLeafNodeStatistics& rhs) const {
    return m_BestSplit < rhs.m_BestSplit;
}

double CBoostedTreeLeafNodeStatistics::gain() const {
    return m_BestSplit.s_Gain;
}

double CBoostedTreeLeafNodeStatistics::curvature() const {
    return m_BestSplit.s_Curvature;
}

CBoostedTreeLeafNodeStatistics::TSizeDoublePr CBoostedTreeLeafNodeStatistics::bestSplit() const {
    return {m_BestSplit.s_Feature, m_BestSplit.s_SplitAt};
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
    std::size_t mem{core::CMemory::dynamicSize(m_RowMask)};
    mem += core::CMemory::dynamicSize(m_Derivatives);
    mem += core::CMemory::dynamicSize(m_MissingDerivatives);
    return mem;
}

std::size_t
CBoostedTreeLeafNodeStatistics::estimateMemoryUsage(std::size_t numberRows,
                                                    std::size_t numberCols,
                                                    std::size_t numberSplitsPerFeature,
                                                    std::size_t numberLossParameters) {
    // We will typically get the close to the best compression for most of the
    // leaves when the set of splits becomes large, corresponding to the worst
    // case for memory usage. This is because the rows will be spread over many
    // rows so the masks will mainly contain 0 bits in this case.
    std::size_t rowMaskSize{numberRows / PACKED_BIT_VECTOR_MAXIMUM_ROWS_PER_BYTE};
    std::size_t derivativesSize{(numberCols - 1) * numberSplitsPerFeature *
                                SDerivatives::estimateMemoryUsage(numberLossParameters)};
    std::size_t missingDerivativesSize{
        (numberCols - 1) * SDerivatives::estimateMemoryUsage(numberLossParameters)};
    return sizeof(CBoostedTreeLeafNodeStatistics) + rowMaskSize +
           derivativesSize + missingDerivativesSize;
}

void CBoostedTreeLeafNodeStatistics::computeAggregateLossDerivatives(
    std::size_t numberThreads,
    const core::CDataFrame& frame,
    const CDataFrameCategoryEncoder& encoder) {

    auto result = frame.readRows(
        numberThreads, 0, frame.numberRows(),
        core::bindRetrievableState(
            [&](CSplitDerivativesAccumulator& splitDerivativesAccumulator,
                TRowItr beginRows, TRowItr endRows) {
                for (auto row = beginRows; row != endRows; ++row) {
                    this->addRowDerivatives(encoder.encode(*row), splitDerivativesAccumulator);
                }
            },
            CSplitDerivativesAccumulator{m_CandidateSplits, m_NumberLossParameters}),
        &m_RowMask);
    auto& state = result.first;

    CSplitDerivativesAccumulator accumulatedDerivatives{std::move(state[0].s_FunctionState)};
    for (std::size_t i = 1; i < state.size(); ++i) {
        accumulatedDerivatives.merge(state[i].s_FunctionState);
    }

    std::tie(m_Derivatives, m_MissingDerivatives) = accumulatedDerivatives.read();
}

void CBoostedTreeLeafNodeStatistics::computeRowMaskAndAggregateLossDerivatives(
    std::size_t numberThreads,
    const core::CDataFrame& frame,
    const CDataFrameCategoryEncoder& encoder,
    bool isLeftChild,
    const CBoostedTreeNode& split,
    const core::CPackedBitVector& parentRowMask) {

    auto result = frame.readRows(
        numberThreads, 0, frame.numberRows(),
        core::bindRetrievableState(
            [&](std::pair<core::CPackedBitVector, CSplitDerivativesAccumulator>& state,
                TRowItr beginRows, TRowItr endRows) {
                auto& mask = state.first;
                auto& splitDerivativesAccumulator = state.second;
                for (auto row = beginRows; row != endRows; ++row) {
                    auto encodedRow = encoder.encode(*row);
                    if (split.assignToLeft(encodedRow) == isLeftChild) {
                        std::size_t index{row->index()};
                        mask.extend(false, index - mask.size());
                        mask.extend(true);
                        this->addRowDerivatives(encodedRow, splitDerivativesAccumulator);
                    }
                }
            },
            std::make_pair(core::CPackedBitVector{},
                           CSplitDerivativesAccumulator{m_CandidateSplits, m_NumberLossParameters})),
        &parentRowMask);
    auto& state = result.first;

    for (auto& mask_ : state) {
        auto& mask = mask_.s_FunctionState.first;
        mask.extend(false, parentRowMask.size() - mask.size());
    }

    m_RowMask = std::move(state[0].s_FunctionState.first);
    CSplitDerivativesAccumulator derivatives{std::move(state[0].s_FunctionState.second)};
    for (std::size_t i = 1; i < state.size(); ++i) {
        m_RowMask |= state[i].s_FunctionState.first;
        derivatives.merge(state[i].s_FunctionState.second);
    }

    std::tie(m_Derivatives, m_MissingDerivatives) = derivatives.read();
}

void CBoostedTreeLeafNodeStatistics::addRowDerivatives(
    const CEncodedDataFrameRowRef& row,
    CSplitDerivativesAccumulator& splitAggregateDerivatives) const {

    const TRowRef& unencodedRow{row.unencodedRow()};
    auto gradient = readLossGradient(unencodedRow, m_NumberInputColumns, m_NumberLossParameters);
    auto curvature = readLossCurvature(unencodedRow, m_NumberInputColumns, m_NumberLossParameters);

    for (std::size_t i = 0; i < m_CandidateSplits.size(); ++i) {
        double featureValue{row[i]};
        if (CDataFrameUtils::isMissing(featureValue)) {
            splitAggregateDerivatives.addMissingDerivatives(i, gradient, curvature);
        } else {
            std::ptrdiff_t j{m_CandidateSplits[i].upperBound(featureValue)};
            splitAggregateDerivatives.addDerivatives(i, j, gradient, curvature);
        }
    }
}

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
    // ranges over the examples in the leaf. Completing the square and noting that
    // the minimum is given when the quadratic form is zero we have
    //
    //   L(w^*) = -1/2 g^t H(\lambda) g
    //
    // where w^* = arg\min_w{ L(w) }.

    using TMinimumLoss =
        std::function<double(const TDoubleVector& g, const TDoubleMatrix& h)>;

    TMinimumLoss minimumLoss;
    
    double lambda{regularization.leafWeightPenaltyMultiplier()};
    if (m_NumberLossParameters == 1) {
        minimumLoss = [lambda](const TDoubleVector& g, const TDoubleMatrix& h) -> double {
            return CTools::pow2(g(0)) / (h(0, 0) + lambda);
        };
    } else {
        // TODO this turns out to be extremely expensive even when m_NumberLossParameters
        // is one. I'm not sure why yet. Maybe solving in-place would help.
        minimumLoss = [lambda](const TDoubleVector& g, const TDoubleMatrix& h) -> double {
            return g.transpose() *
                   (h + lambda * TDoubleMatrix::Identity(h.rows(), h.cols()))
                       .selfadjointView<Eigen::Upper>()
                       .ldlt()
                       .solve(g);
        };
    }

    for (auto i : featureBag) {
        std::size_t c{m_MissingDerivatives[i].s_Count};
        TDoubleVector g{m_MissingDerivatives[i].s_Gradient};
        TDoubleMatrix h{m_MissingDerivatives[i].s_Curvature};
        for (const auto& derivatives : m_Derivatives[i]) {
            c += derivatives.s_Count;
            g += derivatives.s_Gradient;
            h += derivatives.s_Curvature;
        }
        std::size_t cl[]{m_MissingDerivatives[i].s_Count, 0};
        TDoubleVector gl[]{m_MissingDerivatives[i].s_Gradient,
                           TDoubleVector::Zero(g.rows())};
        TDoubleMatrix hl[]{m_MissingDerivatives[i].s_Curvature,
                           TDoubleMatrix::Zero(h.rows(), h.cols())};
        TDoubleVector gr[]{g - m_MissingDerivatives[i].s_Gradient, g};
        TDoubleMatrix hr[]{h - m_MissingDerivatives[i].s_Curvature, h};

        double maximumGain{-INF};
        double splitAt{-INF};
        bool leftChildHasFewerRows{true};
        bool assignMissingToLeft{true};

        for (std::size_t j = 0; j + 1 < m_Derivatives[i].size(); ++j) {

            cl[ASSIGN_MISSING_TO_LEFT] += m_Derivatives[i][j].s_Count;
            gl[ASSIGN_MISSING_TO_LEFT] += m_Derivatives[i][j].s_Gradient;
            gr[ASSIGN_MISSING_TO_LEFT] -= m_Derivatives[i][j].s_Gradient;
            hl[ASSIGN_MISSING_TO_LEFT] += m_Derivatives[i][j].s_Curvature;
            hr[ASSIGN_MISSING_TO_LEFT] -= m_Derivatives[i][j].s_Curvature;

            cl[ASSIGN_MISSING_TO_RIGHT] += m_Derivatives[i][j].s_Count;
            gl[ASSIGN_MISSING_TO_RIGHT] += m_Derivatives[i][j].s_Gradient;
            gr[ASSIGN_MISSING_TO_RIGHT] -= m_Derivatives[i][j].s_Gradient;
            hl[ASSIGN_MISSING_TO_RIGHT] += m_Derivatives[i][j].s_Curvature;
            hr[ASSIGN_MISSING_TO_RIGHT] -= m_Derivatives[i][j].s_Curvature;

            double gain[2];
            gain[ASSIGN_MISSING_TO_LEFT] =
                minimumLoss(gl[ASSIGN_MISSING_TO_LEFT], hl[ASSIGN_MISSING_TO_LEFT]) +
                minimumLoss(gr[ASSIGN_MISSING_TO_LEFT], hr[ASSIGN_MISSING_TO_LEFT]);
            gain[ASSIGN_MISSING_TO_RIGHT] =
                minimumLoss(gl[ASSIGN_MISSING_TO_RIGHT], hl[ASSIGN_MISSING_TO_RIGHT]) +
                minimumLoss(gr[ASSIGN_MISSING_TO_RIGHT], hr[ASSIGN_MISSING_TO_RIGHT]);

            if (gain[ASSIGN_MISSING_TO_LEFT] > maximumGain) {
                maximumGain = gain[ASSIGN_MISSING_TO_LEFT];
                splitAt = m_CandidateSplits[i][j];
                leftChildHasFewerRows = (2 * cl[ASSIGN_MISSING_TO_LEFT] < c);
                assignMissingToLeft = true;
            }
            if (gain[ASSIGN_MISSING_TO_RIGHT] > maximumGain) {
                maximumGain = gain[ASSIGN_MISSING_TO_RIGHT];
                splitAt = m_CandidateSplits[i][j];
                leftChildHasFewerRows = (2 * cl[ASSIGN_MISSING_TO_RIGHT] < c);
                assignMissingToLeft = false;
            }
        }

        double penaltyForDepth{regularization.penaltyForDepth(m_Depth)};
        double penaltyForDepthPlusOne{regularization.penaltyForDepth(m_Depth + 1)};
        double gain{0.5 * (maximumGain - minimumLoss(g, h)) -
                    regularization.treeSizePenaltyMultiplier() -
                    regularization.depthPenaltyMultiplier() *
                        (2.0 * penaltyForDepthPlusOne - penaltyForDepth)};

        SSplitStatistics candidate{gain,
                                   h.trace() / static_cast<double>(m_NumberLossParameters),
                                   i,
                                   splitAt,
                                   leftChildHasFewerRows,
                                   assignMissingToLeft};
        LOG_TRACE(<< "candidate split: " << candidate.print());

        if (candidate > result) {
            result = candidate;
        }
    }

    LOG_TRACE(<< "best split: " << result.print());

    return result;
}
}
}
