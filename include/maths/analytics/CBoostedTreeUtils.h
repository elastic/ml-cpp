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

#ifndef INCLUDED_ml_maths_analytics_CBoostedTreeUtils_h
#define INCLUDED_ml_maths_analytics_CBoostedTreeUtils_h

#include <core/CDataFrame.h>

#include <maths/analytics/ImportExport.h>

#include <maths/common/CLinearAlgebraEigen.h>
#include <maths/common/MathsTypes.h>

#include <cmath>
#include <cstddef>

namespace ml {
namespace maths {
namespace analytics {
namespace boosted_tree {
class CLoss;
}
class CBoostedTreeNode;
class CDataFrameCategoryEncoder;
class CEncodedDataFrameRowRef;
namespace boosted_tree_detail {
using TDoubleVec = std::vector<double>;
using TFloatVec = std::vector<common::CFloatStorage>;
using TSizeVec = std::vector<std::size_t>;
using TRowRef = core::CDataFrame::TRowRef;
using TMemoryMappedFloatVector = common::CMemoryMappedDenseVector<common::CFloatStorage>;
using TSizeAlignmentPrVec = std::vector<std::pair<std::size_t, core::CAlignment::EType>>;
using TAlignedMemoryMappedFloatVector =
    common::CMemoryMappedDenseVector<common::CFloatStorage, Eigen::Aligned16>;

enum EExtraColumnTag {
    E_Prediction = 0,
    E_Gradient,
    E_Curvature,
    E_Weight,
    E_PreviousPrediction,
    E_BeginSplits
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
    E_MaximumNumberTrees, //!< Train only.
    E_FeatureBagFraction,
    E_PredictionChangeCost,     //!< Incremental train only.
    E_RetrainedTreeEta,         //!< Incremental train only.
    E_TreeTopologyChangePenalty //!< Incremental train only.
};

constexpr std::size_t NUMBER_EXTRA_COLUMNS{E_BeginSplits + 1}; // This must be last extra column
constexpr std::size_t NUMBER_HYPERPARAMETERS{E_TreeTopologyChangePenalty + 1}; // This must be last hyperparameter
constexpr std::size_t UNIT_ROW_WEIGHT_COLUMN{std::numeric_limits<std::size_t>::max()};

//! \brief Hyperparameter importance information.
struct MATHS_ANALYTICS_EXPORT SHyperparameterImportance {
    enum EType { E_Double = 0, E_Uint64 };
    EHyperparameter s_Hyperparameter;
    double s_Value;
    double s_AbsoluteImportance;
    double s_RelativeImportance;
    bool s_Supplied;
    EType s_Type;
};

//! \brief An extremely fast ordered search tree.
//!
//! DESCRIPTION:\n
//! This provides a single query upperBound(x), i.e. find the smallest value in
//! an ordered set greater than the specified query point x. Where possible it
//! uses SSE instructions to perform the 4 way comparison. This means it achieves
//! a branch factor of 5 and complexity ceil(log(n) / log(5)) in the set size n.
//!
//! IMPLEMENTATION DECISIONS:\n
//! We align the storage to 16 bytes so we can use aligned loads for the data to
//! compare at each node in the tree. We also pad the data slightly, with infinity,
//! so we can always safely load four values at once and to maintain the spacing.
//! In total though this only adds up to 16 bytes overhead.
//!
//! How does this compare to std::upper_bound performance-wise?
//!
//! The following are representative figures for bare metal for 10000000 lookups:
//!
//!  collection size | std::upper_bound | CSearchTree::upperBound | speedup
//!  --------------- | ---------------- | ----------------------- | -------
//!        100       |      215 ms      |          59 ms          |  3.6 X
//!       10000      |      580 ms      |         130 ms          |  4.5 X
//!
//! One might reasonably expect larger speedups for larger data set sizes because
//! of the higher branch factor. We posit that one pays fixed overheads due to
//! cache misses traversing larger data sets and this reduces the % improve from
//! executing fewer instructions.
class MATHS_ANALYTICS_EXPORT CSearchTree {
public:
    CSearchTree() = default;
    explicit CSearchTree(const TFloatVec& values);

    //! Check if it's empty.
    bool empty() const { return m_Size == 0; }

    //! Get the number of items in the set.
    std::size_t size() const { return m_Size; }

    //! A drop in replacement for std::upper_bound on a sorted collection.
    std::size_t upperBound(common::CFloatStorage x) const;

private:
    using TAlignedFloatVec = std::vector<float, core::CAlignedAllocator<float>>;

private:
    static constexpr float INF{std::numeric_limits<float>::infinity()};

private:
    void build(const TFloatVec& values, std::size_t a, std::size_t b);
    std::string printNode(std::size_t node) const;
    static std::size_t nextPow5(std::size_t n);

private:
    std::size_t m_Size{0};
    std::size_t m_InitialTreeSize{0};
    float m_Min{-INF};
    TAlignedFloatVec m_Values;
};

//! Get the index of the root node in a canonical tree node vector.
inline std::size_t rootIndex() {
    return 0;
}

//! Get the root node of \p tree.
MATHS_ANALYTICS_EXPORT
const CBoostedTreeNode& root(const std::vector<CBoostedTreeNode>& tree);

//! Get the root node of \p tree.
MATHS_ANALYTICS_EXPORT
CBoostedTreeNode& root(std::vector<CBoostedTreeNode>& tree);

//! Get the split used for storing missing values.
inline std::size_t missingSplit(const TFloatVec& candidateSplits) {
    return candidateSplits.size() + 1;
}

//! Get the split used for storing missing values.
inline std::size_t missingSplit(const CSearchTree& candidateSplits) {
    return candidateSplits.size() + 1;
}

//! Get the size of upper triangle of the loss Hessain.
inline std::size_t lossHessianUpperTriangleSize(std::size_t numberLossParameters) {
    return numberLossParameters * (numberLossParameters + 1) / 2;
}

//! Get the tags for extra columns needed by training.
inline TSizeVec extraColumnTagsForTrain() {
    return {E_Prediction, E_Gradient, E_Curvature, E_Weight};
}

//! Get the extra columns needed by training.
inline TSizeAlignmentPrVec extraColumnsForTrain(std::size_t numberLossParameters) {
    return {{numberLossParameters, core::CAlignment::E_Unaligned}, // prediction
            {numberLossParameters, core::CAlignment::E_Aligned16}, // gradient
            {numberLossParameters * numberLossParameters, core::CAlignment::E_Unaligned}}; // curvature
}

//! Get the tags for extra columns needed by training.
inline TSizeVec extraColumnTagsForIncrementalTrain() {
    return {E_PreviousPrediction};
}

//! Get the extra columns needed by incremental training.
inline TSizeAlignmentPrVec extraColumnsForIncrementalTrain(std::size_t numberLossParameters) {
    return {{numberLossParameters, core::CAlignment::E_Unaligned}}; // previous prediction
}

//! Get the extra columns needed for prediction.
inline TSizeAlignmentPrVec extraColumnsForPredict(std::size_t numberLossParameters) {
    return {{numberLossParameters, core::CAlignment::E_Unaligned}}; // prediction
}

//! Get the tags for extra columns needed for prediction.
inline TSizeVec extraColumnTagsForPredict() {
    return {E_Prediction};
}

//! Read the prediction from \p row.
inline TMemoryMappedFloatVector readPrediction(const TRowRef& row,
                                               const TSizeVec& extraColumns,
                                               std::size_t numberLossParameters) {
    return {row.data() + extraColumns[E_Prediction], static_cast<int>(numberLossParameters)};
}

//! Zero the prediction of \p row.
MATHS_ANALYTICS_EXPORT
void zeroPrediction(const TRowRef& row, const TSizeVec& extraColumns, std::size_t numberLossParameters);

//! Write \p value to \p row prediction column(s).
MATHS_ANALYTICS_EXPORT
void writePrediction(const TRowRef& row,
                     const TSizeVec& extraColumns,
                     std::size_t numberLossParameters,
                     const TMemoryMappedFloatVector& value);

//! Write \p value to \p row previous prediction column(s).
MATHS_ANALYTICS_EXPORT
void writePreviousPrediction(const TRowRef& row,
                             const TSizeVec& extraColumns,
                             std::size_t numberLossParameters,
                             const TMemoryMappedFloatVector& value);

//! Read the previous prediction for \p row if training incementally.
MATHS_ANALYTICS_EXPORT
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
MATHS_ANALYTICS_EXPORT
void zeroLossGradient(const TRowRef& row, const TSizeVec& extraColumns, std::size_t numberLossParameters);

//! Write the loss gradient to \p row.
MATHS_ANALYTICS_EXPORT
void writeLossGradient(const TRowRef& row,
                       const CEncodedDataFrameRowRef& encodedRow,
                       bool newExample,
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
MATHS_ANALYTICS_EXPORT
void zeroLossCurvature(const TRowRef& row, const TSizeVec& extraColumns, std::size_t numberLossParameters);

//! Write the loss Hessian to \p row.
MATHS_ANALYTICS_EXPORT
void writeLossCurvature(const TRowRef& row,
                        const CEncodedDataFrameRowRef& encodedRow,
                        bool newExample,
                        const TSizeVec& extraColumns,
                        const boosted_tree::CLoss& loss,
                        const TMemoryMappedFloatVector& prediction,
                        double actual,
                        double weight = 1.0);

//! Read the example weight from \p row.
inline double readExampleWeight(const TRowRef& row, const TSizeVec& extraColumns) {
    std::size_t weightColumn{extraColumns[E_Weight]};
    return weightColumn == UNIT_ROW_WEIGHT_COLUMN
               ? 1.0
               : static_cast<double>(row[weightColumn]);
}

//! Get a writable pointer to the start of the row split indices.
inline core::CFloatStorage* beginSplits(const TRowRef& row, const TSizeVec& extraColumns) {
    return row.data() + extraColumns[E_BeginSplits];
}

//! Read the actual value for the target from \p row.
inline double readActual(const TRowRef& row, std::size_t dependentVariable) {
    return row[dependentVariable];
}

constexpr double INF{std::numeric_limits<double>::max()};
}
}
}
}

#endif // INCLUDED_ml_maths_analytics_CBoostedTreeUtils_h
