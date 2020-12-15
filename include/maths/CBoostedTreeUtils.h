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
using TSizeVec = std::vector<std::size_t>;
using TRowRef = core::CDataFrame::TRowRef;
using TMemoryMappedFloatVector = CMemoryMappedDenseVector<CFloatStorage>;
using TSizeAlignmentPrVec = std::vector<std::pair<std::size_t, core::CAlignment::EType>>;
using TAlignedMemoryMappedFloatVector =
    CMemoryMappedDenseVector<CFloatStorage, Eigen::Aligned16>;

enum EExtraColumn { E_Prediction = 0, E_Gradient, E_Curvature, E_Weight };

enum EHyperparameters {
    E_Alpha = 0,
    E_DownsampleFactor,
    E_Eta,
    E_EtaGrowthRatePerTree,
    E_FeatureBagFraction,
    E_Gamma,
    E_Lambda,
    E_SoftTreeDepthLimit,
    E_SoftTreeDepthTolerance,
    E_LastHyperparameter //put all new hyperparameter before this one
};

struct SHyperparameterImportance {
    SHyperparameterImportance(EHyperparameters hyperparameter,
                              double value,
                              double absoluteImportance,
                              double relativeImportance)
        : s_Hyperparameter(hyperparameter), s_Value(value),
          s_AbsoluteImportance(absoluteImportance),
          s_RelativeImportance(relativeImportance){};
    EHyperparameters s_Hyperparameter;
    double s_Value;
    double s_AbsoluteImportance;
    double s_RelativeImportance;
};

//! Get the size of upper triangle of the loss Hessain.
inline std::size_t lossHessianUpperTriangleSize(std::size_t numberLossParameters) {
    return numberLossParameters * (numberLossParameters + 1) / 2;
}

//! Get the extra columns needed by training.
inline TSizeAlignmentPrVec extraColumns(std::size_t numberLossParameters) {
    return {{numberLossParameters, core::CAlignment::E_Unaligned},
            {numberLossParameters, core::CAlignment::E_Aligned16},
            {numberLossParameters * numberLossParameters, core::CAlignment::E_Unaligned},
            {1, core::CAlignment::E_Unaligned}};
}

//! Read the prediction from \p row.
inline TMemoryMappedFloatVector readPrediction(const TRowRef& row,
                                               const TSizeVec& extraColumns,
                                               std::size_t numberLossParameters) {
    return {row.data() + extraColumns[E_Prediction], static_cast<int>(numberLossParameters)};
}

//! Zero the prediction of \p row.
MATHS_EXPORT
void zeroPrediction(const TRowRef& row, const TSizeVec& extraColumns, std::size_t numberLossParameters);

//! Read all the loss derivatives from \p row into an aligned vector.
inline TAlignedMemoryMappedFloatVector
readLossDerivatives(const TRowRef& row, const TSizeVec& extraColumns, std::size_t numberLossParameters) {
    return {row.data() + extraColumns[E_Gradient],
            static_cast<int>(numberLossParameters +
                             lossHessianUpperTriangleSize(numberLossParameters))};
}

//! Zero the loss gradient of \p row.
MATHS_EXPORT
void zeroLossGradient(const TRowRef& row, const TSizeVec& extraColumns, std::size_t numberLossParameters);

//! Write the loss gradient to \p row.
MATHS_EXPORT
void writeLossGradient(const TRowRef& row,
                       const TSizeVec& extraColumns,
                       const boosted_tree::CLoss& loss,
                       const TMemoryMappedFloatVector& prediction,
                       double actual,
                       double weight = 1.0);

//! Read the loss flat column major Hessian from \p row.
inline TMemoryMappedFloatVector readLossCurvature(const TRowRef& row,
                                                  const TSizeVec& extraColumns,
                                                  std::size_t numberLossParameters) {
    return {row.data() + extraColumns[E_Curvature],
            static_cast<int>(lossHessianUpperTriangleSize(numberLossParameters))};
}

//! Zero the loss Hessian of \p row.
MATHS_EXPORT
void zeroLossCurvature(const TRowRef& row, const TSizeVec& extraColumns, std::size_t numberLossParameters);

//! Write the loss Hessian to \p row.
MATHS_EXPORT
void writeLossCurvature(const TRowRef& row,
                        const TSizeVec& extraColumns,
                        const boosted_tree::CLoss& curvature,
                        const TMemoryMappedFloatVector& prediction,
                        double actual,
                        double weight = 1.0);

//! Read the example weight from \p row.
inline double readExampleWeight(const TRowRef& row, const TSizeVec& extraColumns) {
    return row[extraColumns[E_Weight]];
}

//! Write the example weight to \p row .
inline void writeExampleWeight(const TRowRef& row, const TSizeVec& extraColumns, double weight) {
    row.writeColumn(extraColumns[E_Weight], weight);
}

//! Read the actual value for the target from \p row.
inline double readActual(const TRowRef& row, std::size_t dependentVariable) {
    return row[dependentVariable];
}

constexpr double INF{std::numeric_limits<double>::max()};
}
}
}

#endif // INCLUDED_ml_maths_CBoostedTreeUtils_h
