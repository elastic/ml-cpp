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

#ifndef INCLUDED_ml_maths_analytics_CBoostedTreeLeafNodeStatisticsScratch_h
#define INCLUDED_ml_maths_analytics_CBoostedTreeLeafNodeStatisticsScratch_h

#include <maths/analytics/CBoostedTreeLeafNodeStatistics.h>
#include <maths/analytics/ImportExport.h>

#include <maths/common/MathsTypes.h>

#include <cstddef>
#include <functional>

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
//! This version is used for training from scratch.
class MATHS_ANALYTICS_EXPORT CBoostedTreeLeafNodeStatisticsScratch final
    : public CBoostedTreeLeafNodeStatistics {
public:
    CBoostedTreeLeafNodeStatisticsScratch(std::size_t id,
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
    CBoostedTreeLeafNodeStatisticsScratch(std::size_t id,
                                          const CBoostedTreeLeafNodeStatisticsScratch& parent,
                                          const core::CDataFrame& frame,
                                          const TRegularization& regularization,
                                          const TSizeVec& treeFeatureBag,
                                          const TSizeVec& nodeFeatureBag,
                                          bool isLeftChild,
                                          const CBoostedTreeNode& split,
                                          CWorkspace& workspace);

    //! Only called by split but is public so it's accessible to std::make_shared.
    CBoostedTreeLeafNodeStatisticsScratch(std::size_t id,
                                          CBoostedTreeLeafNodeStatisticsScratch&& parent,
                                          const TRegularization& regularization,
                                          const TSizeVec& treeFeatureBag,
                                          const TSizeVec& nodeFeatureBag,
                                          CWorkspace& workspace);

    CBoostedTreeLeafNodeStatisticsScratch(const CBoostedTreeLeafNodeStatisticsScratch&) = delete;
    CBoostedTreeLeafNodeStatisticsScratch&
    operator=(const CBoostedTreeLeafNodeStatisticsScratch&) = delete;

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

    //! \brief Statistics used to compute the gain bound.
    struct MATHS_ANALYTICS_EXPORT SChildrenGainStatistics {
        double s_MinLossLeft{-boosted_tree_detail::INF};
        double s_MinLossRight{-boosted_tree_detail::INF};
        double s_GainLeft{-boosted_tree_detail::INF};
        double s_GainRight{-boosted_tree_detail::INF};
    };

private:
    CBoostedTreeLeafNodeStatisticsScratch(const TSizeVec& extraColumns,
                                          std::size_t numberLossParameters,
                                          const TFloatVecVec& candidateSplits,
                                          CSplitsDerivatives derivatives);
    SSplitStatistics computeBestSplitStatistics(std::size_t numberThreads,
                                                const TRegularization& regularization,
                                                const TSizeVec& featureBag) const;
    TFeatureBestSplitSearch
    featureBestSplitSearch(const TRegularization& regularization,
                           SSplitStatistics& bestSplitStats,
                           SChildrenGainStatistics& childrenGainStatsGlobal) const;
    double childMaxGain(double childGain, double minLossChild, double lambda) const;

    friend struct CBoostedTreeLeafNodeStatisticsTest::testComputeBestSplitStatisticsThreading;
};
}
}
}

#endif // INCLUDED_ml_maths_analytics_CBoostedTreeLeafNodeStatisticsScratch_h
