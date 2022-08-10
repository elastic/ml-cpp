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

#include <maths/analytics/CBoostedTreeUtils.h>

#include <core/CContainerPrinter.h>
#include <core/Concurrency.h>

#include <maths/analytics/CBoostedTree.h>
#include <maths/analytics/CBoostedTreeHyperparameters.h>
#include <maths/analytics/CBoostedTreeLoss.h>
#include <maths/analytics/CDataFrameCategoryEncoder.h>

#include <maths/common/CLinearAlgebraEigen.h>
#include <maths/common/MathsTypes.h>

namespace ml {
namespace maths {
namespace analytics {
namespace boosted_tree_detail {
using namespace boosted_tree;
namespace {
auto branchSizes(std::size_t n) {
    TSizeVec branchSizes;
    std::size_t nextPow5{5};
    for (/**/; nextPow5 < n; nextPow5 *= 5) {
        branchSizes.push_back(nextPow5);
    }
    std::reverse(branchSizes.begin(), branchSizes.end());
    return std::make_pair(std::move(branchSizes), nextPow5);
}
}

CSearchTree::CSearchTree(const TFloatVec& values) : m_Size{values.size()} {
    if (m_Size > 0) {
        m_Min = values[0].cstorage();
        std::size_t nextPow5;
        std::tie(m_BranchSizes, nextPow5) = branchSizes(values.size());
        // Allow space for padding with inf.
        m_Values.reserve(m_Size + 4 * m_BranchSizes.size());
        this->build(values, 0, nextPow5);
        LOG_TRACE(<< "flat tree = " << m_Values);
    }
    LOG_TRACE(<< "size = " << m_Size << ", padded size = " << m_InitialTreeSize);
}

void CSearchTree::build(const TFloatVec& values, std::size_t a, std::size_t b) {
    // Building depth first gives the lower expected jump size.
    std::size_t stride{(b - a) / 5};
    if (stride == 0) {
        return;
    }
    LOG_TRACE(<< "stride = " << stride);
    for (std::size_t i = a + stride; i < b; i += stride) {
        m_Values.push_back(i < values.size() ? values[i].cstorage() : INF);
    }
    for (std::size_t i = a; i < b; i += stride) {
        if (i < values.size()) {
            this->build(values, i, i + stride);
        }
    }
}

std::string CSearchTree::printNode(std::size_t node) const {
    return core::CContainerPrinter::print(m_Values.begin() + node,
                                          m_Values.begin() + node + 4);
}

const CBoostedTreeNode& root(const std::vector<CBoostedTreeNode>& tree) {
    return tree[rootIndex()];
}

CBoostedTreeNode& root(std::vector<CBoostedTreeNode>& tree) {
    return tree[rootIndex()];
}

void zeroPrediction(const TRowRef& row, const TSizeVec& extraColumns, std::size_t numberLossParameters) {
    for (std::size_t i = 0; i < numberLossParameters; ++i) {
        row.writeColumn(extraColumns[E_Prediction] + i, 0.0);
    }
}

void writePrediction(const TRowRef& row,
                     const TSizeVec& extraColumns,
                     std::size_t numberLossParameters,
                     const TMemoryMappedFloatVector& value) {
    for (std::size_t i = 0; i < numberLossParameters; ++i) {
        row.writeColumn(extraColumns[E_Prediction] + i, value(i));
    }
}

void writePreviousPrediction(const TRowRef& row,
                             const TSizeVec& extraColumns,
                             std::size_t numberLossParameters,
                             const TMemoryMappedFloatVector& value) {
    for (std::size_t i = 0; i < numberLossParameters; ++i) {
        row.writeColumn(extraColumns[E_PreviousPrediction] + i, value(i));
    }
}

void zeroLossGradient(const TRowRef& row, const TSizeVec& extraColumns, std::size_t numberLossParameters) {
    for (std::size_t i = 0; i < numberLossParameters; ++i) {
        row.writeColumn(extraColumns[E_Gradient] + i, 0.0);
    }
}

void writeLossGradient(const TRowRef& row,
                       const CEncodedDataFrameRowRef& encodedRow,
                       bool newExample,
                       const TSizeVec& extraColumns,
                       const CLoss& loss,
                       const TMemoryMappedFloatVector& prediction,
                       double actual,
                       double weight) {
    auto writer = [&row, &extraColumns](std::size_t i, double value) {
        row.writeColumn(extraColumns[E_Gradient] + i, value);
    };
    // We wrap the writer in another lambda which we know takes advantage
    // of std::function small size optimization to avoid heap allocations.
    loss.gradient(encodedRow, newExample, prediction, actual,
                  [&writer](std::size_t i, double value) { writer(i, value); }, weight);
}

void zeroLossCurvature(const TRowRef& row, const TSizeVec& extraColumns, std::size_t numberLossParameters) {
    for (std::size_t i = 0, size = lossHessianUpperTriangleSize(numberLossParameters);
         i < size; ++i) {
        row.writeColumn(extraColumns[E_Curvature] + i, 0.0);
    }
}

void writeLossCurvature(const TRowRef& row,
                        const CEncodedDataFrameRowRef& encodedRow,
                        bool newExample,
                        const TSizeVec& extraColumns,
                        const CLoss& loss,
                        const TMemoryMappedFloatVector& prediction,
                        double actual,
                        double weight) {
    auto writer = [&row, &extraColumns](std::size_t i, double value) {
        row.writeColumn(extraColumns[E_Curvature] + i, value);
    };
    // We wrap the writer in another lambda which we know takes advantage
    // of std::function small size optimization to avoid heap allocations.
    loss.curvature(encodedRow, newExample, prediction, actual,
                   [&writer](std::size_t i, double value) { writer(i, value); }, weight);
}
}
}
}
}
