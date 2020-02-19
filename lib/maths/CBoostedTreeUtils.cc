/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBoostedTreeUtils.h>

#include <maths/CBoostedTree.h>

namespace ml {
namespace maths {
namespace boosted_tree_detail {
using namespace boosted_tree;

TMemoryMappedFloatVector readPrediction(const TRowRef& row,
                                        std::size_t numberInputColumns,
                                        std::size_t numberLossParamaters) {
    return {row.data() + predictionColumn(numberInputColumns),
            static_cast<int>(numberLossParamaters)};
}

void zeroPrediction(const TRowRef& row, std::size_t numberInputColumns, std::size_t numberLossParamaters) {
    std::size_t offset{predictionColumn(numberInputColumns)};
    for (std::size_t i = 0; i < numberLossParamaters; ++i) {
        row.writeColumn(offset + i, 0.0);
    }
}

TMemoryMappedFloatVector readLossGradient(const TRowRef& row,
                                          std::size_t numberInputColumns,
                                          std::size_t numberLossParameters) {
    return {row.data() + lossGradientColumn(numberInputColumns, numberLossParameters),
            static_cast<int>(numberLossParameters)};
}

void zeroLossGradient(const TRowRef& row, std::size_t numberInputColumns, std::size_t numberLossParameters) {
    std::size_t offset{lossGradientColumn(numberInputColumns, numberLossParameters)};
    for (std::size_t i = 0; i < numberLossParameters; ++i) {
        row.writeColumn(offset + i, 0.0);
    }
}

void writeLossGradient(const TRowRef& row,
                       std::size_t numberInputColumns,
                       const CLoss& loss,
                       const TMemoryMappedFloatVector& prediction,
                       double actual,
                       double weight) {
    std::size_t offset{lossGradientColumn(numberInputColumns, prediction.size())};
    auto writer = [&row, offset](std::size_t i, double value) {
        row.writeColumn(offset + i, value);
    };
    // We wrap the writer in another lambda which we know takes advantage
    // of std::function small size optimization to avoid heap allocations.
    loss.gradient(prediction, actual,
                  [&writer](std::size_t i, double value) { writer(i, value); }, weight);
}

TMemoryMappedFloatVector readLossCurvature(const TRowRef& row,
                                           std::size_t numberInputColumns,
                                           std::size_t numberLossParameters) {
    return {row.data() + lossCurvatureColumn(numberInputColumns, numberLossParameters),
            static_cast<int>(lossHessianStoredSize(numberLossParameters))};
}

void zeroLossCurvature(const TRowRef& row, std::size_t numberInputColumns, std::size_t numberLossParameters) {
    std::size_t offset{lossCurvatureColumn(numberInputColumns, numberLossParameters)};
    for (std::size_t i = 0, size = lossHessianStoredSize(numberLossParameters);
         i < size; ++i) {
        row.writeColumn(offset + i, 0.0);
    }
}

void writeLossCurvature(const TRowRef& row,
                        std::size_t numberInputColumns,
                        const CLoss& loss,
                        const TMemoryMappedFloatVector& prediction,
                        double actual,
                        double weight) {
    std::size_t offset{lossCurvatureColumn(numberInputColumns, prediction.size())};
    auto writer = [&row, offset](std::size_t i, double value) {
        row.writeColumn(offset + i, value);
    };
    // We wrap the writer in another lambda which we know takes advantage
    // of std::function small size optimization to avoid heap allocations.
    loss.curvature(prediction, actual,
                   [&writer](std::size_t i, double value) { writer(i, value); }, weight);
}

double readExampleWeight(const TRowRef& row,
                         std::size_t numberInputColumns,
                         std::size_t numberLossParameters) {
    return row[exampleWeightColumn(numberInputColumns, numberLossParameters)];
}

double readActual(const TRowRef& row, std::size_t dependentVariable) {
    return row[dependentVariable];
}
}
}
}
