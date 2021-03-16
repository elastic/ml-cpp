/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBoostedTreeUtils.h>

#include <maths/CBoostedTree.h>
#include <maths/CBoostedTreeLoss.h>
#include <maths/CDataFrameCategoryEncoder.h>

namespace ml {
namespace maths {
namespace boosted_tree_detail {
using namespace boosted_tree;

std::size_t rootIndex() {
    return 0;
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

void zeroLossGradient(const TRowRef& row, const TSizeVec& extraColumns, std::size_t numberLossParameters) {
    for (std::size_t i = 0; i < numberLossParameters; ++i) {
        row.writeColumn(extraColumns[E_Gradient] + i, 0.0);
    }
}

void writeLossGradient(const TRowRef& row,
                       const TSizeVec& extraColumns,
                       const CDataFrameCategoryEncoder& encoder,
                       const CLoss& loss,
                       const TMemoryMappedFloatVector& prediction,
                       double actual,
                       double weight) {
    auto writer = [&row, &extraColumns](std::size_t i, double value) {
        row.writeColumn(extraColumns[E_Gradient] + i, value);
    };
    // We wrap the writer in another lambda which we know takes advantage
    // of std::function small size optimization to avoid heap allocations.
    loss.gradient(encoder.encode(row), prediction, actual,
                  [&writer](std::size_t i, double value) { writer(i, value); }, weight);
}

void zeroLossCurvature(const TRowRef& row, const TSizeVec& extraColumns, std::size_t numberLossParameters) {
    for (std::size_t i = 0, size = lossHessianUpperTriangleSize(numberLossParameters);
         i < size; ++i) {
        row.writeColumn(extraColumns[E_Curvature] + i, 0.0);
    }
}

void writeLossCurvature(const TRowRef& row,
                        const TSizeVec& extraColumns,
                        const CDataFrameCategoryEncoder& encoder,
                        const CLoss& loss,
                        const TMemoryMappedFloatVector& prediction,
                        double actual,
                        double weight) {
    auto writer = [&row, &extraColumns](std::size_t i, double value) {
        row.writeColumn(extraColumns[E_Curvature] + i, value);
    };
    // We wrap the writer in another lambda which we know takes advantage
    // of std::function small size optimization to avoid heap allocations.
    loss.curvature(encoder.encode(row), prediction, actual,
                   [&writer](std::size_t i, double value) { writer(i, value); }, weight);
}
}
}
}
