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

#include <core/CMemory.h>

#include <maths/CBoostedTree.h>
#include <maths/CDataFrameCategoryEncoder.h>
#include <maths/CDataFrameUtils.h>
#include <maths/COrderings.h>
#include <memory>

namespace ml {
namespace maths {
using namespace boosted_tree_detail;
using TRowItr = core::CDataFrame::TRowItr;

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

const core::CPackedBitVector& CBoostedTreeLeafNodeStatistics::rowMask() const {
    return m_RowMask;
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
    // of the row mask which is accounted for there.
    std::size_t splitsDerivativesSize{CSplitsDerivatives::estimateMemoryUsage(
        numberFeatures, numberSplitsPerFeature, numberLossParameters)};
    return sizeof(CBoostedTreeLeafNodeStatistics) + splitsDerivativesSize;
}

std::string CBoostedTreeLeafNodeStatistics::print() const {
    return m_BestSplit.print();
}

CBoostedTreeLeafNodeStatistics::CBoostedTreeLeafNodeStatistics(std::size_t id,
                                                               std::size_t depth,
                                                               TSizeVecCRef extraColumns,
                                                               std::size_t numberLossParameters,
                                                               const TImmutableRadixSetVec& candidateSplits,
                                                               CSplitsDerivatives derivatives)
    : m_Id{id}, m_Depth{depth}, m_ExtraColumns{extraColumns}, m_NumberLossParameters{numberLossParameters},
      m_CandidateSplits{candidateSplits}, m_Derivatives{std::move(derivatives)} {
}

std::size_t CBoostedTreeLeafNodeStatistics::maximumNumberThreadsToAggregateDerivatives(
    const CBoostedTreeLeafNodeStatistics& parent,
    const TSizeVec& treeFeatureBag) {
    // The number of threads we'll use breaks down as follows:
    //   - We need a minimum number of rows per thread to ensure reasonable
    //     load balancing.
    //   - We need a minimum amount of work per thread to make the overheads
    //     of distributing worthwhile.
    std::size_t features{treeFeatureBag.size()};
    std::size_t rows{parent.minimumChildRowCount()};
    std::size_t rowsPerThreadConstraint{rows / 64};
    std::size_t workPerThreadConstraint{(features * rows) / (8 * 128)};
    return std::max(std::min(rowsPerThreadConstraint, workPerThreadConstraint),
                    std::size_t{1});
}

void CBoostedTreeLeafNodeStatistics::computeAggregateLossDerivatives(
    CLookAheadBound bound,
    std::size_t numberThreads,
    const core::CDataFrame& frame,
    const CDataFrameCategoryEncoder& encoder,
    const TSizeVec& featureBag,
    const core::CPackedBitVector& rowMask,
    CWorkspace& workspace) const {
    this->computeAggregateLossDerivativesWith(bound, numberThreads, frame, encoder,
                                              featureBag, rowMask, workspace);
}

void CBoostedTreeLeafNodeStatistics::computeAggregateLossDerivatives(
    CNoLookAheadBound bound,
    std::size_t numberThreads,
    const core::CDataFrame& frame,
    const CDataFrameCategoryEncoder& encoder,
    const TSizeVec& featureBag,
    const core::CPackedBitVector& rowMask,
    CWorkspace& workspace) const {
    this->computeAggregateLossDerivativesWith(bound, numberThreads, frame, encoder,
                                              featureBag, rowMask, workspace);
}

void CBoostedTreeLeafNodeStatistics::computeRowMaskAndAggregateLossDerivatives(
    CLookAheadBound bound,
    std::size_t numberThreads,
    const core::CDataFrame& frame,
    const CDataFrameCategoryEncoder& encoder,
    bool isLeftChild,
    const CBoostedTreeNode& split,
    const TSizeVec& featureBag,
    const core::CPackedBitVector& parentRowMask,
    CWorkspace& workspace) const {
    this->computeRowMaskAndAggregateLossDerivativesWith(
        bound, numberThreads, frame, encoder, isLeftChild, split, featureBag,
        parentRowMask, workspace);
}

void CBoostedTreeLeafNodeStatistics::computeRowMaskAndAggregateLossDerivatives(
    CNoLookAheadBound bound,
    std::size_t numberThreads,
    const core::CDataFrame& frame,
    const CDataFrameCategoryEncoder& encoder,
    bool isLeftChild,
    const CBoostedTreeNode& split,
    const TSizeVec& featureBag,
    const core::CPackedBitVector& parentRowMask,
    CWorkspace& workspace) const {
    this->computeRowMaskAndAggregateLossDerivativesWith(
        bound, numberThreads, frame, encoder, isLeftChild, split, featureBag,
        parentRowMask, workspace);
}

template<typename BOUND>
void CBoostedTreeLeafNodeStatistics::computeAggregateLossDerivativesWith(
    BOUND bound,
    std::size_t numberThreads,
    const core::CDataFrame& frame,
    const CDataFrameCategoryEncoder& encoder,
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
                this->addRowDerivatives(bound, featureBag, encoder.encode(*row),
                                        splitsDerivatives);
            }
        });
    }

    frame.readRows(0, frame.numberRows(), aggregators, &rowMask);
}

template<typename BOUND>
void CBoostedTreeLeafNodeStatistics::computeRowMaskAndAggregateLossDerivativesWith(
    BOUND bound,
    std::size_t numberThreads,
    const core::CDataFrame& frame,
    const CDataFrameCategoryEncoder& encoder,
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
            for (auto row = beginRows; row != endRows; ++row) {
                auto encodedRow = encoder.encode(*row);
                if (split.assignToLeft(encodedRow) == isLeftChild) {
                    std::size_t index{row->index()};
                    mask.extend(false, index - mask.size());
                    mask.extend(true);
                    this->addRowDerivatives(bound, featureBag, encodedRow, splitsDerivatives);
                }
            }
        });
    }

    frame.readRows(0, frame.numberRows(), aggregators, &parentRowMask);
}

void CBoostedTreeLeafNodeStatistics::addRowDerivatives(CLookAheadBound,
                                                       const TSizeVec& featureBag,
                                                       const CEncodedDataFrameRowRef& row,
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

    for (auto feature : featureBag) {
        double featureValue{row[feature]};
        if (CDataFrameUtils::isMissing(featureValue)) {
            splitsDerivatives.addMissingDerivatives(feature, derivatives);
        } else {
            std::ptrdiff_t split{m_CandidateSplits[feature].upperBound(featureValue)};
            splitsDerivatives.addDerivatives(feature, split, derivatives);
        }
    }
}

void CBoostedTreeLeafNodeStatistics::addRowDerivatives(CNoLookAheadBound,
                                                       const TSizeVec& featureBag,
                                                       const CEncodedDataFrameRowRef& row,
                                                       CSplitsDerivatives& splitsDerivatives) const {

    auto derivatives = readLossDerivatives(row.unencodedRow(), m_ExtraColumns,
                                           m_NumberLossParameters);

    for (auto feature : featureBag) {
        double featureValue{row[feature]};
        if (CDataFrameUtils::isMissing(featureValue)) {
            splitsDerivatives.addMissingDerivatives(feature, derivatives);
        } else {
            std::ptrdiff_t split{m_CandidateSplits[feature].upperBound(featureValue)};
            splitsDerivatives.addDerivatives(feature, split, derivatives);
        }
    }
}

CBoostedTreeLeafNodeStatistics::SSplitStatistics&
CBoostedTreeLeafNodeStatistics::bestSplitStatistics() {
    return m_BestSplit;
}

CBoostedTreeLeafNodeStatistics::CSplitsDerivatives&
CBoostedTreeLeafNodeStatistics::derivatives() {
    return m_Derivatives;
}

const CBoostedTreeLeafNodeStatistics::CSplitsDerivatives&
CBoostedTreeLeafNodeStatistics::derivatives() const {
    return m_Derivatives;
}

std::size_t CBoostedTreeLeafNodeStatistics::depth() const {
    return m_Depth;
}

CBoostedTreeLeafNodeStatistics::TSizeVecCRef
CBoostedTreeLeafNodeStatistics::extraColumns() const {
    return m_ExtraColumns;
}

std::size_t CBoostedTreeLeafNodeStatistics::numberLossParameters() const {
    return m_NumberLossParameters;
}

const CBoostedTreeLeafNodeStatistics::TImmutableRadixSetVec&
CBoostedTreeLeafNodeStatistics::candidateSplits() const {
    return m_CandidateSplits;
}

CBoostedTreeLeafNodeStatistics::TSizeVec
CBoostedTreeLeafNodeStatistics::CWorkspace::featuresToInclude() const {

    TSizeVec result;
    if (m_TreeToRetrain != nullptr) {
        for (const auto& node : *m_TreeToRetrain) {
            if (node.isLeaf() == false) {
                result.push_back(node.splitFeature());
            }
        }
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());

    return result;
}
}
}
