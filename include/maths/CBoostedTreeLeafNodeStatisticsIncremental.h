/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CBoostedTreeLeafNodeStatisticsIncremental_h
#define INCLUDED_ml_maths_CBoostedTreeLeafNodeStatisticsIncremental_h

#include <core/CMemory.h>
#include <core/CPackedBitVector.h>

#include <maths/CBoostedTreeHyperparameters.h>
#include <maths/CBoostedTreeLeafNodeStatistics.h>
#include <maths/CBoostedTreeUtils.h>
#include <maths/CChecksum.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/CLinearAlgebraShims.h>
#include <maths/CMathsFuncs.h>
#include <maths/COrderings.h>
#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <boost/operators.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

namespace ml {
namespace core {
class CDataFrame;
}
namespace maths {
class CBoostedTreeNode;
class CDataFrameCategoryEncoder;
class CEncodedDataFrameRowRef;

//! \brief Maintains a collection of statistics about a leaf of the regression
//! tree as it is built.
//!
//! DESCRIPTION:\N
//! The regression tree is grown top down by greedily selecting the split with
//! the maximum gain (in the loss). This finds and scores the maximum gain split
//! of a single leaf of the tree.
//!
//! This version is used for training incrementally.
class MATHS_EXPORT CBoostedTreeLeafNodeStatisticsIncremental final
    : public CBoostedTreeLeafNodeStatistics {
public:
    CBoostedTreeLeafNodeStatisticsIncremental(std::size_t id,
                                              const TSizeVec& extraColumns,
                                              std::size_t numberLossParameters,
                                              std::size_t numberThreads,
                                              const core::CDataFrame& frame,
                                              const CDataFrameCategoryEncoder& encoder,
                                              const TRegularization& regularization,
                                              const TImmutableRadixSetVec& candidateSplits,
                                              const TSizeVec& treeFeatureBag,
                                              const TSizeVec& nodeFeatureBag,
                                              std::size_t depth,
                                              const core::CPackedBitVector& rowMask,
                                              CWorkspace& workspace);

    //! Only called by split but is public so it's accessible to std::make_shared.
    CBoostedTreeLeafNodeStatisticsIncremental(std::size_t id,
                                              const CBoostedTreeLeafNodeStatisticsIncremental& parent,
                                              std::size_t numberThreads,
                                              const core::CDataFrame& frame,
                                              const CDataFrameCategoryEncoder& encoder,
                                              const TRegularization& regularization,
                                              const TSizeVec& treeFeatureBag,
                                              const TSizeVec& nodeFeatureBag,
                                              bool isLeftChild,
                                              const CBoostedTreeNode& split,
                                              CWorkspace& workspace);

    //! Only called by split but is public so it's accessible to std::make_shared.
    CBoostedTreeLeafNodeStatisticsIncremental(std::size_t id,
                                              CBoostedTreeLeafNodeStatisticsIncremental&& parent,
                                              const TRegularization& regularization,
                                              const TSizeVec& nodeFeatureBag,
                                              CWorkspace& workspace);

    CBoostedTreeLeafNodeStatisticsIncremental(const CBoostedTreeLeafNodeStatisticsIncremental&) = delete;
    CBoostedTreeLeafNodeStatisticsIncremental&
    operator=(const CBoostedTreeLeafNodeStatisticsIncremental&) = delete;

    // Move construction/assignment not possible due to const reference member.

    //! Apply the split defined by \p split.
    //!
    //! \return Shared pointers to the left and right child node statistics.
    TPtrPtrPr split(std::size_t leftChildId,
                    std::size_t rightChildId,
                    std::size_t numberThreads,
                    double gainThreshold,
                    const core::CDataFrame& frame,
                    const CDataFrameCategoryEncoder& encoder,
                    const TRegularization& regularization,
                    const TSizeVec& treeFeatureBag,
                    const TSizeVec& nodeFeatureBag,
                    const CBoostedTreeNode& split,
                    CWorkspace& workspace) override;

    //! Get the size of this object.
    std::size_t staticSize() const override;

private:
    //! \brief Describes a split in the tree being incrementally retrained.
    struct SPreviousSplit {
        std::size_t s_Feature;
        double s_SplitAt;
    };
    using TOptionalPreviousSplit = boost::optional<SPreviousSplit>;

private:
    SSplitStatistics computeBestSplitStatistics(const TRegularization& regularization,
                                                const TSizeVec& featureBag) const;
    double penaltyForTreeChange(const TRegularization& regularization,
                                std::size_t feature,
                                std::size_t split) const;

private:
    TOptionalPreviousSplit m_PreviousSplit;
};
}
}

#endif // INCLUDED_ml_maths_CBoostedTreeLeafNodeStatisticsIncremental_h
