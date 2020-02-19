/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CBoostedTreeUtils_h
#define INCLUDED_ml_maths_CBoostedTreeUtils_h

#include <core/CDataFrame.h>

#include <maths/CLinearAlgebraEigen.h>
#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <cmath>
#include <cstddef>

namespace ml {
namespace maths {
namespace boosted_tree {
class CLoss;
}
namespace boosted_tree_detail {
using TRowRef = core::CDataFrame::TRowRef;
using TMemoryMappedFloatVector = CMemoryMappedDenseVector<CFloatStorage>;

inline std::size_t lossHessianStoredSize(std::size_t numberLossParameters) {
    return numberLossParameters * (numberLossParameters + 1) / 2;
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

//! Read the prediction from \p row.
MATHS_EXPORT
TMemoryMappedFloatVector readPrediction(const TRowRef& row,
                                        std::size_t numberInputColumns,
                                        std::size_t numberLossParamaters);

//! Zero the prediction of \p row.
MATHS_EXPORT
void zeroPrediction(const TRowRef& row, std::size_t numberInputColumns, std::size_t numberLossParamaters);

//! Read the loss gradient from \p row.
MATHS_EXPORT
TMemoryMappedFloatVector readLossGradient(const TRowRef& row,
                                          std::size_t numberInputColumns,
                                          std::size_t numberLossParameters);

//! Zero the loss gradient of \p row.
MATHS_EXPORT
void zeroLossGradient(const TRowRef& row, std::size_t numberInputColumns, std::size_t numberLossParameters);

//! Write the loss gradient to \p row.
MATHS_EXPORT
void writeLossGradient(const TRowRef& row,
                       std::size_t numberInputColumns,
                       const boosted_tree::CLoss& loss,
                       const TMemoryMappedFloatVector& prediction,
                       double actual,
                       double weight = 1.0);

MATHS_EXPORT
TMemoryMappedFloatVector readLossCurvature(const TRowRef& row,
                                           std::size_t numberInputColumns,
                                           std::size_t numberLossParameters);

MATHS_EXPORT
void zeroLossCurvature(const TRowRef& row, std::size_t numberInputColumns, std::size_t numberLossParameters);

MATHS_EXPORT
void writeLossCurvature(const TRowRef& row,
                        std::size_t numberInputColumns,
                        const boosted_tree::CLoss& curvature,
                        const TMemoryMappedFloatVector& prediction,
                        double actual,
                        double weight = 1.0);

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
