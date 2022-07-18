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

#ifndef INCLUDED_ml_maths_analytics_CBoostedTreeLeafNodeStatisticsIncremental_h
#define INCLUDED_ml_maths_analytics_CBoostedTreeLeafNodeStatisticsIncremental_h

#include <core/CMemory.h>
#include <core/CPackedBitVector.h>

#include <maths/analytics/CBoostedTreeHyperparameters.h>
#include <maths/analytics/CBoostedTreeLeafNodeStatistics.h>
#include <maths/analytics/CBoostedTreeUtils.h>
#include <maths/analytics/ImportExport.h>

#include <maths/common/CChecksum.h>
#include <maths/common/CLinearAlgebraEigen.h>
#include <maths/common/CLinearAlgebraShims.h>
#include <maths/common/CMathsFuncs.h>
#include <maths/common/COrderings.h>
#include <maths/common/MathsTypes.h>

#include <boost/operators.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

namespace CBoostedTreeLeafNodeStatisticsTest {
struct testComputeBestSplitStatisticsThreading;
}
namespace ml {
namespace core {
class CDataFrame;
}
namespace maths {
namespace analytics {
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
class MATHS_ANALYTICS_EXPORT CBoostedTreeLeafNodeStatisticsIncremental final
    : public CBoostedTreeLeafNodeStatistics {
public:
    CBoostedTreeLeafNodeStatisticsIncremental(std::size_t id,
                                              const TSizeVec& extraColumns,
                                              std::size_t numberLossParameters,
                                              const core::CDataFrame& frame,
                                              const TRegularization& regularization,
                                              const TFloatVecVec& candidateSplits,
                                              const TSizeVec& treeFeatureBag,
                                              const TSizeVec& nodeFeatureBag,
                                              std::size_t depth,
                                              const core::CPackedBitVector& rowMask,
                                              CWorkspace& workspace);

    //! Only called by split but is public so it's accessible to std::make_shared.
    CBoostedTreeLeafNodeStatisticsIncremental(std::size_t id,
                                              const CBoostedTreeLeafNodeStatisticsIncremental& parent,
                                              const core::CDataFrame& frame,
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
                                              const TSizeVec& treeFeatureBag,
                                              const TSizeVec& nodeFeatureBag,
                                              bool isLeftChild,
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
                    double gainThreshold,
                    const core::CDataFrame& frame,
                    const TRegularization& regularization,
                    const TSizeVec& treeFeatureBag,
                    const TSizeVec& nodeFeatureBag,
                    const CBoostedTreeNode& split,
                    CWorkspace& workspace) override;

    //! Get the size of this object.
    std::size_t staticSize() const override;

private:
    using TFeatureBestSplitSearch = std::function<void(std::size_t)>;

    //! \brief Describes a split of the tree being incrementally retrained.
    struct MATHS_ANALYTICS_EXPORT SPreviousSplit {
        SPreviousSplit(std::size_t nodeIndex, std::size_t feature, double splitAt)
            : s_NodeIndex{nodeIndex}, s_Feature{feature}, s_SplitAt{splitAt} {}

        std::size_t s_NodeIndex;
        std::size_t s_Feature;
        double s_SplitAt;
    };
    using TOptionalPreviousSplit = std::optional<SPreviousSplit>;

private:
    CBoostedTreeLeafNodeStatisticsIncremental(const TSizeVec& extraColumns,
                                              std::size_t numberLossParameters,
                                              const TFloatVecVec& candidateSplits,
                                              CSplitsDerivatives derivatives);
    SSplitStatistics computeBestSplitStatistics(std::size_t numberThreads,
                                                const TRegularization& regularization,
                                                const TSizeVec& featureBag) const;
    TFeatureBestSplitSearch featureBestSplitSearch(const TRegularization& regularization,
                                                   SSplitStatistics& bestSplitStatistics) const;
    double penaltyForTreeChange(const TRegularization& regularization,
                                std::size_t feature,
                                std::size_t split) const;
    TOptionalPreviousSplit rootPreviousSplit(const CWorkspace& workspace) const;
    TOptionalPreviousSplit leftChildPreviousSplit(std::size_t feature,
                                                  const CWorkspace& workspace) const;
    TOptionalPreviousSplit rightChildPreviousSplit(std::size_t feature,
                                                   const CWorkspace& workspace) const;

private:
    TOptionalPreviousSplit m_PreviousSplit;

    friend struct CBoostedTreeLeafNodeStatisticsTest::testComputeBestSplitStatisticsThreading;
};
}
}
}

#endif // INCLUDED_ml_maths_analytics_CBoostedTreeLeafNodeStatisticsIncremental_h
