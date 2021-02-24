/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBoostedTreeLeafNodeStatistics.h>

#include <core/CMemory.h>

#include <maths/COrderings.h>

namespace ml {
namespace maths {
std::size_t CBoostedTreeLeafNodeStatistics::memoryUsage() const {
    return core::CMemory::dynamicSize(m_RowMask);
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

const core::CPackedBitVector& CBoostedTreeLeafNodeStatistics::rowMask() const {
    return m_RowMask;
}

core::CPackedBitVector& CBoostedTreeLeafNodeStatistics::rowMask() {
    return m_RowMask;
}

std::string CBoostedTreeLeafNodeStatistics::print() const {
    return m_BestSplit.print();
}

CBoostedTreeLeafNodeStatistics::CBoostedTreeLeafNodeStatistics(std::size_t id,
                                                               std::size_t depth,
                                                               TSizeVecCRef extraColumns,
                                                               std::size_t numberLossParameters,
                                                               const TImmutableRadixSetVec& candidateSplits)
    : m_Id{id}, m_Depth{depth}, m_ExtraColumns{extraColumns},
      m_NumberLossParameters{numberLossParameters}, m_CandidateSplits{candidateSplits} {
}

CBoostedTreeLeafNodeStatistics::SSplitStatistics&
CBoostedTreeLeafNodeStatistics::bestSplitStatistics() {
    return m_BestSplit;
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
}
}
