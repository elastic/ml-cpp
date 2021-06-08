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
class CBoostedTreeNode;
template<typename>
class CBoostedTreeRegularization;
class CDataFrameCategoryEncoder;
namespace boosted_tree_detail {
using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;
using TRowRef = core::CDataFrame::TRowRef;
using TMemoryMappedFloatVector = CMemoryMappedDenseVector<CFloatStorage>;
using TSizeAlignmentPrVec = std::vector<std::pair<std::size_t, core::CAlignment::EType>>;
using TAlignedMemoryMappedFloatVector =
    CMemoryMappedDenseVector<CFloatStorage, Eigen::Aligned16>;
using TRegularization = CBoostedTreeRegularization<double>;

enum EExtraColumn {
    E_Prediction = 0,
    E_Gradient,
    E_Curvature,
    E_Weight,
    E_PreviousPrediction
};

enum EHyperparameter {
    E_DownsampleFactor = 0,
    E_Alpha,
    E_Lambda,
    E_Gamma,
    E_SoftTreeDepthLimit,
    E_SoftTreeDepthTolerance,
    E_Eta,
    E_EtaGrowthRatePerTree,
    E_MaximumNumberTrees,
    E_FeatureBagFraction,
    E_PredictionChangeCost,
    E_TreeTopologyChangePenalty
};

constexpr std::size_t NUMBER_HYPERPARAMETERS = E_TreeTopologyChangePenalty + 1; // This must be last hyperparameter

struct SHyperparameterImportance {
    enum EType { E_Double = 0, E_Uint64 };
    EHyperparameter s_Hyperparameter;
    double s_Value;
    double s_AbsoluteImportance;
    double s_RelativeImportance;
    bool s_Supplied;
    EType s_Type;
};

//! Get the index of the root node in a canonical tree node vector.
inline std::size_t rootIndex() {
    return 0;
}

//! Get the root node of \p tree.
MATHS_EXPORT
const CBoostedTreeNode& root(const std::vector<CBoostedTreeNode>& tree);

//! Get the root node of \p tree.
MATHS_EXPORT
CBoostedTreeNode& root(std::vector<CBoostedTreeNode>& tree);

//! Get the size of upper triangle of the loss Hessain.
inline std::size_t lossHessianUpperTriangleSize(std::size_t numberLossParameters) {
    return numberLossParameters * (numberLossParameters + 1) / 2;
}

//! Get the extra columns needed by training.
inline TSizeAlignmentPrVec extraColumnsForTrain(std::size_t numberLossParameters) {
    return {{numberLossParameters, core::CAlignment::E_Unaligned},
            {numberLossParameters, core::CAlignment::E_Aligned16},
            {numberLossParameters * numberLossParameters, core::CAlignment::E_Unaligned},
            {1, core::CAlignment::E_Unaligned}};
}

//! Get the extra columns needed by incremental training.
inline TSizeAlignmentPrVec extraColumnsForIncrementalTrain(std::size_t numberLossParameters) {
    return {{numberLossParameters, core::CAlignment::E_Unaligned}};
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

//! Write \p value to \p row prediction column(s).
MATHS_EXPORT
void writePrediction(const TRowRef& row,
                     const TSizeVec& extraColumns,
                     std::size_t numberLossParameters,
                     const TMemoryMappedFloatVector& value);

//! Write \p value to \p row previous prediction column(s).
MATHS_EXPORT
void writePreviousPrediction(const TRowRef& row,
                             const TSizeVec& extraColumns,
                             std::size_t numberLossParameters,
                             const TMemoryMappedFloatVector& value);

//! Read the previous prediction for \p row if training incementally.
MATHS_EXPORT
inline TMemoryMappedFloatVector readPreviousPrediction(const TRowRef& row,
                                                       const TSizeVec& extraColumns,
                                                       std::size_t numberLossParameters) {
    return {row.data() + extraColumns[E_PreviousPrediction],
            static_cast<int>(numberLossParameters)};
}

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
                       bool newExample,
                       const TSizeVec& extraColumns,
                       const CDataFrameCategoryEncoder& encoder,
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
                        bool newExample,
                        const TSizeVec& extraColumns,
                        const CDataFrameCategoryEncoder& encoder,
                        const boosted_tree::CLoss& loss,
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

//! Compute the probabilities with which to select each tree for retraining.
//!
//! TODO should this be a member of CBoostedTreeImpl.
MATHS_EXPORT
TDoubleVec
retrainTreeSelectionProbabilities(std::size_t numberThreads,
                                  const core::CDataFrame& frame,
                                  const TSizeVec& extraColumns,
                                  std::size_t dependentVariable,
                                  const CDataFrameCategoryEncoder& encoder,
                                  const core::CPackedBitVector& trainingDataRowMask,
                                  const boosted_tree::CLoss& loss,
                                  const std::vector<std::vector<CBoostedTreeNode>>& forest);

constexpr double INF{std::numeric_limits<double>::max()};
}
}
}

#endif // INCLUDED_ml_maths_CBoostedTreeUtils_h
