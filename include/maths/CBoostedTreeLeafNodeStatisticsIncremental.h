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

    //! Get the memory used by this object.
    std::size_t memoryUsage() const override;

    //! Estimate the maximum leaf statistics' memory usage training on a data frame
    //! with \p numberCols columns using \p numberSplitsPerFeature for a loss function
    //! with \p numberLossParameters parameters.
    static std::size_t estimateMemoryUsage(std::size_t numberCols,
                                           std::size_t numberSplitsPerFeature,
                                           std::size_t numberLossParameters);

private:
};
}
}

#endif // INCLUDED_ml_maths_CBoostedTreeLeafNodeStatisticsIncremental_h
