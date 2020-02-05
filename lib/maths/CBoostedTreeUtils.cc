/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBoostedTreeUtils.h>

namespace ml {
namespace maths {
namespace boosted_tree_detail {

TDouble1Vec readPrediction(const TRowRef& row,
                           std::size_t numberInputColumns,
                           std::size_t numberLossParamaters) {
    const auto* start = row.data() + predictionColumn(numberInputColumns);
    return TDouble1Vec(start, start + numberLossParamaters);
}

void writePrediction(const TRowRef& row, std::size_t numberInputColumns, const TDouble1Vec& prediction) {
    std::size_t offset{predictionColumn(numberInputColumns)};
    for (std::size_t i = 0; i < prediction.size(); ++i) {
        row.writeColumn(offset + i, prediction[i]);
    }
}

TDouble1Vec readLossGradient(const TRowRef& row,
                             std::size_t numberInputColumns,
                             std::size_t numberLossParameters) {
    const auto* start = row.data() + lossGradientColumn(numberInputColumns, numberLossParameters);
    return TDouble1Vec(start, start + numberLossParameters);
}

void writeLossGradient(const TRowRef& row, std::size_t numberInputColumns, const TDouble1Vec& gradient) {
    std::size_t offset{lossGradientColumn(numberInputColumns, gradient.size())};
    for (std::size_t i = 0; i < gradient.size(); ++i) {
        row.writeColumn(offset + i, gradient[i]);
    }
}

TDouble1Vec readLossCurvature(const TRowRef& row,
                              std::size_t numberInputColumns,
                              std::size_t numberLossParameters) {
    const auto* start = row.data() + lossCurvatureColumn(numberInputColumns, numberLossParameters);
    return TDouble1Vec(start, start + lossHessianStoredSize(numberLossParameters));
}

void writeLossCurvature(const TRowRef& row,
                        std::size_t numberInputColumns,
                        const TDouble1Vec& curvature) {
    // This comes from solving x(x+1)/2 = n.
    std::size_t numberLossParameters{
        numberLossParametersForHessianStoredSize(curvature.size())};
    std::size_t offset{lossCurvatureColumn(numberInputColumns, numberLossParameters)};
    for (std::size_t i = 0; i < curvature.size(); ++i) {
        row.writeColumn(offset + i, curvature[i]);
    }
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
