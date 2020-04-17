/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBoostedTreeUtils.h>

#include <maths/CBoostedTree.h>
#include <maths/CBoostedTreeLoss.h>

namespace ml {
namespace maths {
namespace boosted_tree_detail {
using namespace boosted_tree;
namespace {
enum EExtraColumn { E_Prediction = 0, E_Gradient, E_Curvature, E_Weight };
}

TSizeAlignmentPrVec extraColumns(std::size_t numberLossParameters) {
    return {{numberLossParameters, core::CAlignment::E_Unaligned},
            {numberLossParameters, core::CAlignment::E_Aligned16},
            {numberLossParameters * numberLossParameters, core::CAlignment::E_Unaligned},
            {1, core::CAlignment::E_Unaligned}};
}

TMemoryMappedFloatVector readPrediction(const TRowRef& row,
                                        const TSizeVec& extraColumns,
                                        std::size_t numberLossParameters) {
    return {row.data() + extraColumns[E_Prediction], static_cast<int>(numberLossParameters)};
}

void zeroPrediction(const TRowRef& row, const TSizeVec& extraColumns, std::size_t numberLossParameters) {
    for (std::size_t i = 0; i < numberLossParameters; ++i) {
        row.writeColumn(extraColumns[E_Prediction] + i, 0.0);
    }
}

TAlignedMemoryMappedFloatVector readLossDerivatives(const TRowRef& row,
                                                    const TSizeVec& extraColumns,
                                                    std::size_t numberLossParameters) {
    return {row.data() + extraColumns[E_Gradient],
            static_cast<int>(numberLossParameters +
                             lossHessianUpperTriangleSize(numberLossParameters))};
}

TMemoryMappedFloatVector readLossGradient(const TRowRef& row,
                                          const TSizeVec& extraColumns,
                                          std::size_t numberLossParameters) {
    return {row.data() + extraColumns[E_Gradient], static_cast<int>(numberLossParameters)};
}

void zeroLossGradient(const TRowRef& row, const TSizeVec& extraColumns, std::size_t numberLossParameters) {
    for (std::size_t i = 0; i < numberLossParameters; ++i) {
        row.writeColumn(extraColumns[E_Gradient] + i, 0.0);
    }
}

void writeLossGradient(const TRowRef& row,
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
    loss.gradient(prediction, actual,
                  [&writer](std::size_t i, double value) { writer(i, value); }, weight);
}

TMemoryMappedFloatVector readLossCurvature(const TRowRef& row,
                                           const TSizeVec& extraColumns,
                                           std::size_t numberLossParameters) {
    return {row.data() + extraColumns[E_Curvature],
            static_cast<int>(lossHessianUpperTriangleSize(numberLossParameters))};
}

void zeroLossCurvature(const TRowRef& row, const TSizeVec& extraColumns, std::size_t numberLossParameters) {
    for (std::size_t i = 0, size = lossHessianUpperTriangleSize(numberLossParameters);
         i < size; ++i) {
        row.writeColumn(extraColumns[E_Curvature] + i, 0.0);
    }
}

void writeLossCurvature(const TRowRef& row,
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
    loss.curvature(prediction, actual,
                   [&writer](std::size_t i, double value) { writer(i, value); }, weight);
}

double readExampleWeight(const TRowRef& row, const TSizeVec& extraColumns) {
    return row[extraColumns[E_Weight]];
}

void writeExampleWeight(const TRowRef& row, const TSizeVec& extraColumns, double weight) {
    row.writeColumn(extraColumns[E_Weight], weight);
}

double readActual(const TRowRef& row, std::size_t dependentVariable) {
    return row[dependentVariable];
}
}
}
}
