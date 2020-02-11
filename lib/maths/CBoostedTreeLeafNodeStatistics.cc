/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBoostedTreeLeafNodeStatistics.h>

#include <core/CDataFrame.h>
#include <core/CImmutableRadixSet.h>
#include <core/CLogger.h>

#include <maths/CBoostedTree.h>
#include <maths/CDataFrameCategoryEncoder.h>
#include <maths/CTools.h>

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
    CBoostedTreeLeafNodeStatistics&& parent,
    const CBoostedTreeLeafNodeStatistics& sibling,
    const TRegularization& regularization,
    const TSizeVec& featureBag,
    core::CPackedBitVector rowMask)
    : m_Id{id}, m_Depth{sibling.m_Depth}, m_NumberInputColumns{sibling.m_NumberInputColumns},
      m_NumberLossParameters{sibling.m_NumberLossParameters},
      m_CandidateSplits{sibling.m_CandidateSplits}, m_RowMask{std::move(rowMask)},
      m_Derivatives{std::move(parent.m_Derivatives)} {

    m_Derivatives.subtract(sibling.m_Derivatives);
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
                                      const CBoostedTreeNode& split) {

    if (this->leftChildHasFewerRows()) {
        auto leftChild = std::make_shared<CBoostedTreeLeafNodeStatistics>(
            leftChildId, m_NumberInputColumns, m_NumberLossParameters,
            numberThreads, frame, encoder, regularization, candidateSplits,
            featureBag, true /*is left child*/, m_Depth + 1, split, m_RowMask);
        core::CPackedBitVector rightChildRowMask{m_RowMask};
        rightChildRowMask ^= leftChild->rowMask();
        auto rightChild = std::make_shared<CBoostedTreeLeafNodeStatistics>(
            rightChildId, std::move(*this), *leftChild, regularization,
            featureBag, std::move(rightChildRowMask));

        return std::make_pair(leftChild, rightChild);
    }

    auto rightChild = std::make_shared<CBoostedTreeLeafNodeStatistics>(
        rightChildId, m_NumberInputColumns, m_NumberLossParameters,
        numberThreads, frame, encoder, regularization, candidateSplits,
        featureBag, false /*is left child*/, m_Depth + 1, split, m_RowMask);
    core::CPackedBitVector leftChildRowMask{m_RowMask};
    leftChildRowMask ^= rightChild->rowMask();
    auto leftChild = std::make_shared<CBoostedTreeLeafNodeStatistics>(
        leftChildId, std::move(*this), *rightChild, regularization, featureBag,
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
    return core::CMemory::dynamicSize(m_RowMask) + core::CMemory::dynamicSize(m_Derivatives);
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
    std::size_t perSplitDerivativesSize{CPerSplitDerivatives::estimateMemoryUsage(
        numberCols, numberSplitsPerFeature, numberLossParameters)};
    return sizeof(CBoostedTreeLeafNodeStatistics) + rowMaskSize + perSplitDerivativesSize;
}

void CBoostedTreeLeafNodeStatistics::computeAggregateLossDerivatives(
    std::size_t numberThreads,
    const core::CDataFrame& frame,
    const CDataFrameCategoryEncoder& encoder) {

    auto result = frame.readRows(
        numberThreads, 0, frame.numberRows(),
        core::bindRetrievableState(
            [&](CPerSplitDerivatives& perSplitDerivatives, TRowItr beginRows, TRowItr endRows) {
                for (auto row = beginRows; row != endRows; ++row) {
                    this->addRowDerivatives(encoder.encode(*row), perSplitDerivatives);
                }
            },
            CPerSplitDerivatives{m_CandidateSplits, m_NumberLossParameters}),
        &m_RowMask);

    m_Derivatives = std::move(result.first[0].s_FunctionState);
    for (std::size_t i = 1; i < result.first.size(); ++i) {
        m_Derivatives.add(result.first[i].s_FunctionState);
    }
    m_Derivatives.remapCurvature();
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
            [&](std::pair<core::CPackedBitVector, CPerSplitDerivatives>& state,
                TRowItr beginRows, TRowItr endRows) {
                auto& mask = state.first;
                auto& perSplitDerivatives = state.second;
                for (auto row = beginRows; row != endRows; ++row) {
                    auto encodedRow = encoder.encode(*row);
                    if (split.assignToLeft(encodedRow) == isLeftChild) {
                        std::size_t index{row->index()};
                        mask.extend(false, index - mask.size());
                        mask.extend(true);
                        this->addRowDerivatives(encodedRow, perSplitDerivatives);
                    }
                }
            },
            std::make_pair(core::CPackedBitVector{},
                           CPerSplitDerivatives{m_CandidateSplits, m_NumberLossParameters})),
        &parentRowMask);

    for (auto& mask_ : result.first) {
        auto& mask = mask_.s_FunctionState.first;
        mask.extend(false, parentRowMask.size() - mask.size());
    }

    m_RowMask = std::move(result.first[0].s_FunctionState.first);
    m_Derivatives = std::move(result.first[0].s_FunctionState.second);
    for (std::size_t i = 1; i < result.first.size(); ++i) {
        m_RowMask |= result.first[i].s_FunctionState.first;
        m_Derivatives.add(result.first[i].s_FunctionState.second);
    }
    m_Derivatives.remapCurvature();
}

void CBoostedTreeLeafNodeStatistics::addRowDerivatives(const CEncodedDataFrameRowRef& row,
                                                       CPerSplitDerivatives& perSplitDerivatives) const {

    const TRowRef& unencodedRow{row.unencodedRow()};
    auto gradient = readLossGradient(unencodedRow, m_NumberInputColumns, m_NumberLossParameters);
    auto curvature = readLossCurvature(unencodedRow, m_NumberInputColumns, m_NumberLossParameters);
    for (std::size_t feature = 0; feature < m_CandidateSplits.size(); ++feature) {
        double featureValue{row[feature]};
        if (CDataFrameUtils::isMissing(featureValue)) {
            perSplitDerivatives.addMissingDerivatives(feature, gradient, curvature);
        } else {
            std::ptrdiff_t split{m_CandidateSplits[feature].upperBound(featureValue)};
            perSplitDerivatives.addDerivatives(feature, split, gradient, curvature);
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

    using TDoubleVector = CDenseVector<double>;
    using TDoubleMatrix = CDenseMatrix<double>;
    using TMinimumLoss = std::function<double(const TDoubleVector&, const TDoubleMatrix&)>;

    int d{static_cast<int>(m_NumberLossParameters)};

    TMinimumLoss minimumLoss;

    double lambda{regularization.leafWeightPenaltyMultiplier()};
    Eigen::MatrixXd placeholder{d, d};
    if (m_NumberLossParameters == 1) {
        // There is a large fixed overhead for using ldl^t even when g and h are
        // scalar so we have special case handling.
        minimumLoss = [&](const TDoubleVector& g, const TDoubleMatrix& h) -> double {
            return CTools::pow2(g(0)) / (h(0, 0) + lambda);
        };
    } else {
        // TODO use Cholesky (but need to handle positive semi-definite case).
        minimumLoss = [&](const TDoubleVector& g, const TDoubleMatrix& h) -> double {
            placeholder =
                (h + lambda * TDoubleMatrix::Identity(d, d)).selfadjointView<Eigen::Lower>();
            Eigen::LDLT<Eigen::Ref<Eigen::MatrixXd>> ldlt{placeholder};
            return g.transpose() * ldlt.solve(g);
        };
    }

    TDoubleVector g{d};
    TDoubleMatrix h{d, d};
    TDoubleVector gl[]{TDoubleVector{d}, TDoubleVector{d}};
    TDoubleVector gr[]{TDoubleVector{d}, TDoubleVector{d}};
    TDoubleMatrix hl[]{TDoubleMatrix{d, d}, TDoubleMatrix{d, d}};
    TDoubleMatrix hr[]{TDoubleMatrix{d, d}, TDoubleMatrix{d, d}};

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
        bool leftChildHasFewerRows{true};
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

            double gain[2];
            gain[ASSIGN_MISSING_TO_LEFT] =
                minimumLoss(gl[ASSIGN_MISSING_TO_LEFT], hl[ASSIGN_MISSING_TO_LEFT]) +
                minimumLoss(gr[ASSIGN_MISSING_TO_LEFT], hr[ASSIGN_MISSING_TO_LEFT]);
            gain[ASSIGN_MISSING_TO_RIGHT] =
                minimumLoss(gl[ASSIGN_MISSING_TO_RIGHT], hl[ASSIGN_MISSING_TO_RIGHT]) +
                minimumLoss(gr[ASSIGN_MISSING_TO_RIGHT], hr[ASSIGN_MISSING_TO_RIGHT]);

            if (gain[ASSIGN_MISSING_TO_LEFT] > maximumGain) {
                maximumGain = gain[ASSIGN_MISSING_TO_LEFT];
                splitAt = m_CandidateSplits[feature][split];
                leftChildHasFewerRows = (2 * cl[ASSIGN_MISSING_TO_LEFT] < c);
                assignMissingToLeft = true;
            }
            if (gain[ASSIGN_MISSING_TO_RIGHT] > maximumGain) {
                maximumGain = gain[ASSIGN_MISSING_TO_RIGHT];
                splitAt = m_CandidateSplits[feature][split];
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
                                   feature,
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
