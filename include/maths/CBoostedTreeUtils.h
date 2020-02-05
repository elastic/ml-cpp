/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CBoostedTreeUtils_h
#define INCLUDED_ml_maths_CBoostedTreeUtils_h

#include <core/CDataFrame.h>
#include <core/CSmallVector.h>

#include <cmath>
#include <cstddef>

namespace ml {
namespace maths {
namespace boosted_tree_detail {
using TDouble1Vec = core::CSmallVector<double, 1>;
using TRowRef = core::CDataFrame::TRowRef;

inline std::size_t lossHessianStoredSize(std::size_t numberLossParameters) {
    return numberLossParameters * (numberLossParameters + 1) / 2;
}

inline std::size_t numberLossParametersForHessianStoredSize(std::size_t lossHessianStoredSize) {
    return static_cast<std::size_t>(
        (std::sqrt(8.0 * static_cast<double>(lossHessianStoredSize) + 1.0) - 1.0) / 2.0 + 0.5);
}

inline std::size_t predictionColumn(std::size_t numberInputColumns) {
    return numberInputColumns;
}

inline std::size_t lossGradientColumn(std::size_t numberInputColumns,
                                      std::size_t numberLossParameters) {
    return predictionColumn(numberInputColumns) + numberLossParameters;
}

inline std::size_t lossCurvatureColumn(std::size_t numberInputColumns,
                                       std::size_t numberLossParameters) {
    return lossGradientColumn(numberInputColumns, numberLossParameters) + numberLossParameters;
}

inline std::size_t exampleWeightColumn(std::size_t numberInputColumns,
                                       std::size_t numberLossParameters) {
    return lossCurvatureColumn(numberInputColumns, numberLossParameters) +
           lossHessianStoredSize(numberLossParameters);
}

MATHS_EXPORT
TDouble1Vec readPrediction(const TRowRef& row,
                           std::size_t numberInputColumns,
                           std::size_t numberLossParamaters);

MATHS_EXPORT
void writePrediction(const TRowRef& row, std::size_t numberInputColumns, const TDouble1Vec& prediction);

MATHS_EXPORT
TDouble1Vec readLossGradient(const TRowRef& row,
                             std::size_t numberInputColumns,
                             std::size_t numberLossParameters);

MATHS_EXPORT
void writeLossGradient(const TRowRef& row, std::size_t numberInputColumns, const TDouble1Vec& gradient);

MATHS_EXPORT
TDouble1Vec readLossCurvature(const TRowRef& row,
                              std::size_t numberInputColumns,
                              std::size_t numberLossParameters);

MATHS_EXPORT
void writeLossCurvature(const TRowRef& row,
                        std::size_t numberInputColumns,
                        const TDouble1Vec& curvature);

MATHS_EXPORT
double readExampleWeight(const TRowRef& row,
                         std::size_t numberInputColumns,
                         std::size_t numberLossParameters);

MATHS_EXPORT
double readActual(const TRowRef& row, std::size_t dependentVariable);

// The maximum number of rows encoded by a single byte in the packed bit vector
// assuming best compression.
constexpr std::size_t PACKED_BIT_VECTOR_MAXIMUM_ROWS_PER_BYTE{256};
constexpr double INF{std::numeric_limits<double>::max()};
}
}
}

#endif // INCLUDED_ml_maths_CBoostedTreeUtils_h
