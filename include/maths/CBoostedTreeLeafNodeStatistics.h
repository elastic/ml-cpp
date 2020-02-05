/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CBoostedTreeLeafNodeStatistics_h
#define INCLUDED_ml_maths_CBoostedTreeLeafNodeStatistics_h

#include <core/CImmutableRadixSet.h>
#include <core/CPackedBitVector.h>
#include <core/CSmallVector.h>

#include <maths/CBoostedTreeHyperparameters.h>
#include <maths/CBoostedTreeUtils.h>
#include <maths/COrderings.h>

#include <boost/operators.hpp>

#include <cstddef>
#include <limits>
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
class CBoostedTreeLeafNodeStatistics final {
public:
    using TSizeVec = std::vector<std::size_t>;
    using TSizeDoublePr = std::pair<std::size_t, double>;
    using TRegularization = CBoostedTreeRegularization<double>;
    using TImmutableRadixSetVec = std::vector<core::CImmutableRadixSet<double>>;
    using TPtr = std::shared_ptr<CBoostedTreeLeafNodeStatistics>;
    using TPtrPtrPr = std::pair<TPtr, TPtr>;

public:
    CBoostedTreeLeafNodeStatistics(std::size_t id,
                                   std::size_t numberInputColumns,
                                   std::size_t numberLossParameters,
                                   std::size_t numberThreads,
                                   const core::CDataFrame& frame,
                                   const CDataFrameCategoryEncoder& encoder,
                                   const TRegularization& regularization,
                                   const TImmutableRadixSetVec& candidateSplits,
                                   const TSizeVec& featureBag,
                                   std::size_t depth,
                                   const core::CPackedBitVector& rowMask);

    //! Only called by split but is public so it's accessible to std::make_shared.
    CBoostedTreeLeafNodeStatistics(std::size_t id,
                                   std::size_t numberInputColumns,
                                   std::size_t numberLossParameters,
                                   std::size_t numberThreads,
                                   const core::CDataFrame& frame,
                                   const CDataFrameCategoryEncoder& encoder,
                                   const TRegularization& regularization,
                                   const TImmutableRadixSetVec& candidateSplits,
                                   const TSizeVec& featureBag,
                                   bool isLeftChild,
                                   std::size_t depth,
                                   const CBoostedTreeNode& split,
                                   const core::CPackedBitVector& parentRowMask);

    //! Only called by split but is public so it's accessible to std::make_shared.
    CBoostedTreeLeafNodeStatistics(std::size_t id,
                                   const CBoostedTreeLeafNodeStatistics& parent,
                                   const CBoostedTreeLeafNodeStatistics& sibling,
                                   const TRegularization& regularization,
                                   const TSizeVec& featureBag,
                                   core::CPackedBitVector rowMask);

    CBoostedTreeLeafNodeStatistics(const CBoostedTreeLeafNodeStatistics&) = delete;
    CBoostedTreeLeafNodeStatistics& operator=(const CBoostedTreeLeafNodeStatistics&) = delete;

    // Move construction/assignment not possible due to const reference member.

    //! Apply the split defined by \p split.
    //!
    //! \return Shared pointers to the left and right child node statistics.
    TPtrPtrPr split(std::size_t leftChildId,
                    std::size_t rightChildId,
                    std::size_t numberThreads,
                    const core::CDataFrame& frame,
                    const CDataFrameCategoryEncoder& encoder,
                    const TRegularization& regularization,
                    const TImmutableRadixSetVec& candidateSplits,
                    const TSizeVec& featureBag,
                    const CBoostedTreeNode& split,
                    bool leftChildHasFewerRows);

    //! Order two leaves by decreasing gain in splitting them.
    bool operator<(const CBoostedTreeLeafNodeStatistics& rhs) const;

    //! Get the gain in loss of the best split of this leaf.
    double gain() const;

    //! Get the total curvature of node.
    double curvature() const;

    //! Get the best (feature, feature value) split.
    TSizeDoublePr bestSplit() const;

    //! Check if the left child has fewer rows than the right child.
    bool leftChildHasFewerRows() const;

    //! Check if we should assign the missing feature rows to the left child
    //! of the split.
    bool assignMissingToLeft() const;

    //! Get the node's identifier.
    std::size_t id() const;

    //! Get the row mask for this leaf node.
    core::CPackedBitVector& rowMask();

    //! Get the memory used by this object.
    std::size_t memoryUsage() const;

    //! Estimate the maximum leaf statistics' memory usage training on a data frame
    //! with \p numberRows rows and \p numberCols columns using \p featureBagFraction
    //! and \p numberSplitsPerFeature.
    static std::size_t estimateMemoryUsage(std::size_t numberRows,
                                           std::size_t numberCols,
                                           std::size_t numberSplitsPerFeature);

private:
    using TDouble1Vec = core::CSmallVector<double, 1>;

    //! \brief Statistics relating to a split of the node.
    struct SSplitStatistics : private boost::less_than_comparable<SSplitStatistics> {
        SSplitStatistics() = default;
        SSplitStatistics(double gain,
                         double curvature,
                         std::size_t feature,
                         double splitAt,
                         bool leftChildHasFewerRows,
                         bool assignMissingToLeft)
            : s_Gain{gain}, s_Curvature{curvature}, s_Feature{feature}, s_SplitAt{splitAt},
              s_LeftChildHasFewerRows{leftChildHasFewerRows}, s_AssignMissingToLeft{assignMissingToLeft} {
        }

        bool operator<(const SSplitStatistics& rhs) const {
            return COrderings::lexicographical_compare(
                s_Gain, s_Curvature, s_Feature, rhs.s_Gain, rhs.s_Curvature, rhs.s_Feature);
        }

        std::string print() const {
            std::ostringstream result;
            result << "split feature '" << s_Feature << "' @ " << s_SplitAt
                   << ", gain = " << s_Gain;
            return result.str();
        }

        double s_Gain = -boosted_tree_detail::INF;
        double s_Curvature = 0.0;
        std::size_t s_Feature = -1;
        double s_SplitAt = boosted_tree_detail::INF;
        bool s_LeftChildHasFewerRows = true;
        bool s_AssignMissingToLeft = true;
    };

    //! \brief Aggregate derivatives.
    struct SAggregateDerivatives {
        void add(std::size_t count, const TDouble1Vec& gradient, const TDouble1Vec& curvature) {
            s_Count += count;
            s_Gradient += gradient[0];
            s_Curvature += curvature[0];
        }

        void merge(const SAggregateDerivatives& other) {
            this->add(other.s_Count, {other.s_Gradient}, {other.s_Curvature});
        }

        std::string print() const {
            std::ostringstream result;
            result << "count = " << s_Count << ", gradient = " << s_Gradient
                   << ", curvature = " << s_Curvature;
            return result.str();
        }

        std::size_t s_Count = 0;
        double s_Gradient = 0.0;
        double s_Curvature = 0.0;
    };

    using TAggregateDerivativesVec = std::vector<SAggregateDerivatives>;
    using TAggregateDerivativesVecVec = std::vector<TAggregateDerivativesVec>;

    //! \brief A collection of aggregate derivatives for candidate feature splits.
    struct SSplitAggregateDerivatives {
        SSplitAggregateDerivatives(const TImmutableRadixSetVec& candidateSplits)
            : s_Derivatives(candidateSplits.size()),
              s_MissingDerivatives(candidateSplits.size()) {
            for (std::size_t i = 0; i < candidateSplits.size(); ++i) {
                s_Derivatives[i].resize(candidateSplits[i].size() + 1);
            }
        }

        void merge(const SSplitAggregateDerivatives& other) {
            for (std::size_t i = 0; i < s_Derivatives.size(); ++i) {
                for (std::size_t j = 0; j < s_Derivatives[i].size(); ++j) {
                    s_Derivatives[i][j].merge(other.s_Derivatives[i][j]);
                }
                s_MissingDerivatives[i].merge(other.s_MissingDerivatives[i]);
            }
        }

        auto move() {
            return std::make_pair(std::move(s_Derivatives), std::move(s_MissingDerivatives));
        }

        TAggregateDerivativesVecVec s_Derivatives;
        TAggregateDerivativesVec s_MissingDerivatives;
    };

private:
    void computeAggregateLossDerivatives(std::size_t numberThreads,
                                         const core::CDataFrame& frame,
                                         const CDataFrameCategoryEncoder& encoder);
    void computeRowMaskAndAggregateLossDerivatives(std::size_t numberThreads,
                                                   const core::CDataFrame& frame,
                                                   const CDataFrameCategoryEncoder& encoder,
                                                   bool isLeftChild,
                                                   const CBoostedTreeNode& split,
                                                   const core::CPackedBitVector& parentRowMask);
    void addRowDerivatives(const CEncodedDataFrameRowRef& row,
                           SSplitAggregateDerivatives& splitAggregateDerivatives) const;
    SSplitStatistics computeBestSplitStatistics(const TRegularization& regularization,
                                                const TSizeVec& featureBag) const;

private:
    std::size_t m_Id;
    std::size_t m_Depth;
    std::size_t m_NumberInputColumns;
    std::size_t m_NumberLossParameters;
    const TImmutableRadixSetVec& m_CandidateSplits;
    core::CPackedBitVector m_RowMask;
    TAggregateDerivativesVecVec m_Derivatives;
    TAggregateDerivativesVec m_MissingDerivatives;
    SSplitStatistics m_BestSplit;
};
}
}

#endif // INCLUDED_ml_maths_CBoostedTreeLeafNodeStatistics_h
