/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CBoostedTreeLeafNodeStatistics_h
#define INCLUDED_ml_maths_CBoostedTreeLeafNodeStatistics_h

#include <core/CImmutableRadixSet.h>
#include <core/CMemory.h>
#include <core/CPackedBitVector.h>
#include <core/CSmallVector.h>

#include <maths/CBoostedTreeHyperparameters.h>
#include <maths/CBoostedTreeUtils.h>
#include <maths/CChecksum.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/CLinearAlgebraShims.h>
#include <maths/COrderings.h>
#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <boost/operators.hpp>

#include <cstddef>
#include <cstdint>
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
class MATHS_EXPORT CBoostedTreeLeafNodeStatistics final {
public:
    using TDoubleVec = std::vector<double>;
    using TSizeVec = std::vector<std::size_t>;
    using TSizeDoublePr = std::pair<std::size_t, double>;
    using TRegularization = CBoostedTreeRegularization<double>;
    using TImmutableRadixSet = core::CImmutableRadixSet<double>;
    using TImmutableRadixSetVec = std::vector<TImmutableRadixSet>;
    using TPtr = std::shared_ptr<CBoostedTreeLeafNodeStatistics>;
    using TPtrPtrPr = std::pair<TPtr, TPtr>;
    using TMemoryMappedFloatVector = CMemoryMappedDenseVector<CFloatStorage>;
    using TMemoryMappedDoubleVector = CMemoryMappedDenseVector<double>;
    using TMemoryMappedDoubleMatrix = CMemoryMappedDenseMatrix<double>;

    //! \brief Accumulates aggregate derivatives.
    class MATHS_EXPORT CDerivatives {
    public:
        //! Bounds the minimum diagonal of the Hessian.
        static constexpr double SMALLEST_RELATIVE_CURVATURE{1e-20};

        //! See core::CMemory.
        static bool dynamicSizeAlwaysZero() { return true; }

    public:
        CDerivatives(std::size_t numberLossParameters, double* storage)
            : m_Count{0}, m_Gradient{storage, static_cast<int>(numberLossParameters)},
              m_Curvature{storage + numberLossParameters,
                          static_cast<int>(numberLossParameters),
                          static_cast<int>(numberLossParameters)} {}

        //! Get the accumulated count.
        std::size_t count() const { return m_Count; }

        //! Get the accumulated gradient.
        const TMemoryMappedDoubleVector& gradient() const { return m_Gradient; }

        //! Get the accumulated curvature.
        const TMemoryMappedDoubleMatrix& curvature() const {
            return m_Curvature;
        }

        //! Add \p count, \p gradient and \p curvature to the accumulator.
        void add(std::size_t count,
                 const TMemoryMappedFloatVector& gradient,
                 const TMemoryMappedFloatVector& curvature) {
            m_Count += count;
            m_Gradient += gradient;
            this->curvatureTriangleView() += curvature;
        }

        //! Compute the accumulation of both collections of derivatives.
        void merge(const CDerivatives& other) {
            m_Count += other.m_Count;
            m_Gradient += other.m_Gradient;
            m_Curvature += other.m_Curvature;
        }

        //! Set to the difference of \p lhs and \p rhs.
        void assignDifference(const CDerivatives& lhs, const CDerivatives& rhs) {
            // Numeric errors mean that it's possible the sum curvature for a candidate
            // split is identically zero while the gradient is epsilon. This can cause
            // the node gain to appear infinite (when there is no weight regularisation)
            // which in turns causes problems initialising the region we search for optimal
            // hyperparameter values. We can safely force the gradient and curvature to
            // be zero if we detect that the count is zero.
            std::size_t count{lhs.m_Count - rhs.m_Count};
            if (count > 0) {
                m_Count = count;
                m_Gradient.array() = lhs.m_Gradient - rhs.m_Gradient;
                m_Curvature.array() = lhs.m_Curvature - rhs.m_Curvature;
                // None of our loss functions have negative curvature therefore we
                // shouldn't allow the cumulative curvature to be negative either.
                // In this case we force it to be a v.small multiple of the magnitude
                // of the gradient since this is the closest feasible estimate.
                for (int i = 0; i < m_Gradient.size(); ++i) {
                    m_Curvature(i, i) = std::max(m_Curvature(i, i),
                                                 SMALLEST_RELATIVE_CURVATURE *
                                                     std::fabs(m_Gradient(i)));
                }
            }
        }

        //! Remap the accumulated curvature to lower triangle row major format.
        void remapCurvature() {
            // We accumulate curvatures in the first n (n + 1) / elements however
            // users of TMemoryMappedDoubleMatrix expect them stored column major
            // in the lower triangle of n x n matrix. This copies them backwards
            // to their correct positions.
            for (std::ptrdiff_t j = m_Curvature.cols() - 1,
                                k = m_Curvature.rows() * (m_Curvature.rows() + 1) / 2 - 1;
                 j >= 0; --j) {
                for (std::ptrdiff_t i = m_Curvature.rows() - 1; i >= j; --i, --k) {
                    m_Curvature(i, j) = m_Curvature.array()(k);
                }
            }
        }

        //! Get a checksum of this object.
        std::uint64_t checksum(std::uint64_t seed = 0) const {
            seed = CChecksum::calculate(seed, m_Count);
            seed = CChecksum::calculate(seed, m_Gradient);
            return CChecksum::calculate(seed, m_Curvature);
        }

    private:
        TMemoryMappedDoubleVector curvatureTriangleView() {
            return {m_Curvature.data(), m_Curvature.rows() * (m_Curvature.rows() + 1) / 2};
        }

    private:
        std::size_t m_Count = 0;
        TMemoryMappedDoubleVector m_Gradient;
        TMemoryMappedDoubleMatrix m_Curvature;
    };

    //! \brief A collection of aggregate derivatives for candidate feature splits.
    class MATHS_EXPORT CPerSplitDerivatives {
    public:
        using TDerivativesVec = std::vector<CDerivatives>;

    public:
        explicit CPerSplitDerivatives(std::size_t numberLossParameters = 0) 
            : m_NumberLossParameters{numberLossParameters} {}
        CPerSplitDerivatives(const TImmutableRadixSetVec& candidateSplits,
                             std::size_t numberLossParameters)
            : m_NumberLossParameters{numberLossParameters} {
            this->map(candidateSplits);
        }
        CPerSplitDerivatives(const CPerSplitDerivatives& other)
            : m_NumberLossParameters{other.m_NumberLossParameters} {
            this->map(other.m_Derivatives);
            this->merge(other);
        }
        CPerSplitDerivatives(CPerSplitDerivatives&&) = default;

        CPerSplitDerivatives& operator=(const CPerSplitDerivatives& other) = delete;
        CPerSplitDerivatives& operator=(CPerSplitDerivatives&&) = default;

        //! \return The aggregate count for \p feature and \p split.
        std::size_t count(std::size_t feature, std::size_t split) const {
            return m_Derivatives[feature][split].count();
        }

        //! \return The aggregate gradient for \p feature and \p split.
        const TMemoryMappedDoubleVector& gradient(std::size_t feature, std::size_t split) const {
            return m_Derivatives[feature][split].gradient();
        }

        //! \return The aggregate curvature for \p feature and \p split.
        const TMemoryMappedDoubleMatrix& curvature(std::size_t feature, std::size_t split) const {
            return m_Derivatives[feature][split].curvature();
        }

        //! \return All the split aggregate derivatives for \p feature.
        const TDerivativesVec& derivatives(std::size_t feature) const {
            return m_Derivatives[feature];
        }

        //! \return The count for missing \p feature.
        std::size_t missingCount(std::size_t feature) const {
            return m_MissingDerivatives[feature].count();
        }

        //! \return The aggregate gradients for missing \p feature.
        const TMemoryMappedDoubleVector& missingGradient(std::size_t feature) const {
            return m_MissingDerivatives[feature].gradient();
        }

        //! \return The aggregate curvatures for missing \p feature.
        const TMemoryMappedDoubleMatrix& missingCurvature(std::size_t feature) const {
            return m_MissingDerivatives[feature].curvature();
        }

        //! Add \p gradient and \p curvature to the accumulated derivatives for
        //! the split \p split of feature \p feature.
        void addDerivatives(std::size_t feature,
                            std::size_t split,
                            const TMemoryMappedFloatVector& gradient,
                            const TMemoryMappedFloatVector& curvature) {
            m_Derivatives[feature][split].add(1, gradient, curvature);
        }

        //! Add \p gradient and \p curvature to the accumulated derivatives for
        //! missing values of \p feature.
        void addMissingDerivatives(std::size_t feature,
                                   const TMemoryMappedFloatVector& gradient,
                                   const TMemoryMappedFloatVector& curvature) {
            m_MissingDerivatives[feature].add(1, gradient, curvature);
        }

        //! Compute the accumulation of both collections of per split derivatives.
        void merge(const CPerSplitDerivatives& other) {
            for (std::size_t i = 0; i < other.m_Derivatives.size(); ++i) {
                for (std::size_t j = 0; j < other.m_Derivatives[i].size(); ++j) {
                    m_Derivatives[i][j].merge(other.m_Derivatives[i][j]);
                }
                m_MissingDerivatives[i].merge(other.m_MissingDerivatives[i]);
            }
        }

        //! Set to the difference of \p lhs and \p rhs.
        static CPerSplitDerivatives difference(const CPerSplitDerivatives& lhs,
                                               const CPerSplitDerivatives& rhs) {
            CPerSplitDerivatives result{lhs.m_NumberLossParameters};
            result.map(lhs.m_Derivatives);
            for (std::size_t i = 0; i < lhs.m_Derivatives.size(); ++i) {
                for (std::size_t j = 0; j < lhs.m_Derivatives[i].size(); ++j) {
                    result.m_Derivatives[i][j].assignDifference(
                        lhs.m_Derivatives[i][j], rhs.m_Derivatives[i][j]);
                }
                result.m_MissingDerivatives[i].assignDifference(
                    lhs.m_MissingDerivatives[i], rhs.m_MissingDerivatives[i]);
            }
            return result;
        }

        //! Remap the accumulated curvature to lower triangle row major format.
        void remapCurvature() {
            for (std::size_t i = 0; i < m_Derivatives.size(); ++i) {
                for (std::size_t j = 0; j < m_Derivatives[i].size(); ++j) {
                    m_Derivatives[i][j].remapCurvature();
                }
                m_MissingDerivatives[i].remapCurvature();
            }
        }

        //! Get the memory used by this object.
        std::size_t memoryUsage() const {
            return core::CMemory::dynamicSize(m_Derivatives) +
                   core::CMemory::dynamicSize(m_MissingDerivatives) +
                   core::CMemory::dynamicSize(m_Storage);
        }

        //! Estimate the split derivatives' memory usage for a data frame with
        //! \p numberCols columns using \p numberSplitsPerFeature for a loss
        //! function with \p numberLossParameters parameters.
        static std::size_t estimateMemoryUsage(std::size_t numberCols,
                                               std::size_t numberSplitsPerFeature,
                                               std::size_t numberLossParameters) {
            std::size_t derivativesSize{(numberCols - 1) * (numberSplitsPerFeature + 1) *
                                        sizeof(CDerivatives)};
            std::size_t storageSize{(numberCols - 1) * (numberSplitsPerFeature + 1) *
                                    numberLossParameters *
                                    (numberLossParameters + 1) * sizeof(double)};
            return sizeof(CPerSplitDerivatives) + derivativesSize + storageSize;
        }

        //! Get a checksum of this object.
        std::uint64_t checksum(std::uint64_t seed = 0) const {
            seed = CChecksum::calculate(seed, m_NumberLossParameters);
            seed = CChecksum::calculate(seed, m_Derivatives);
            seed = CChecksum::calculate(seed, m_MissingDerivatives);
            return seed;
        }

    private:
        using TDerivativesVecVec = std::vector<TDerivativesVec>;

    private:
        static std::size_t number(const TDerivativesVec& derivatives) {
            return derivatives.size();
        }
        static std::size_t number(const TImmutableRadixSet& splits) {
            return splits.size() + 1;
        }
        template<typename SPLITS>
        void map(const SPLITS& splits) {
            std::size_t totalNumberSplits{
                std::accumulate(splits.begin(), splits.end(), std::size_t{0},
                                [](std::size_t size, const auto& featureSplits) {
                                    return size + number(featureSplits);
                                })};

            int numberGradients{static_cast<int>(m_NumberLossParameters)};
            int numberCurvatures{numberGradients * numberGradients};
            int numberDerivatives{numberGradients + numberCurvatures};

            m_Derivatives.resize(splits.size());
            m_MissingDerivatives.reserve(splits.size());
            m_Storage.resize((totalNumberSplits + splits.size()) * numberDerivatives, 0.0);

            double* storage{&m_Storage[0]};
            for (std::size_t i = 0; i < splits.size(); ++i, storage += numberDerivatives) {
                std::size_t size{number(splits[i])};
                m_Derivatives[i].reserve(size);
                for (std::size_t j = 0; j < size; ++j, storage += numberDerivatives) {
                    m_Derivatives[i].emplace_back(m_NumberLossParameters, storage);
                }
                m_MissingDerivatives.emplace_back(m_NumberLossParameters, storage);
            }
        }

    private:
        std::size_t m_NumberLossParameters = 0;
        TDerivativesVecVec m_Derivatives;
        TDerivativesVec m_MissingDerivatives;
        TDoubleVec m_Storage;
    };

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
                    const CBoostedTreeNode& split);

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
    //! with \p numberRows rows and \p numberCols columns using \p numberSplitsPerFeature
    //! for a loss function with \p numberLossParameters parameters.
    static std::size_t estimateMemoryUsage(std::size_t numberRows,
                                           std::size_t numberCols,
                                           std::size_t numberSplitsPerFeature,
                                           std::size_t numberLossParameters);

private:
    //! \brief Statistics relating to a split of the node.
    struct MATHS_EXPORT SSplitStatistics
        : private boost::less_than_comparable<SSplitStatistics> {
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
                           CPerSplitDerivatives& splitDerivatives) const;
    SSplitStatistics computeBestSplitStatistics(const TRegularization& regularization,
                                                const TSizeVec& featureBag) const;

private:
    std::size_t m_Id;
    std::size_t m_Depth;
    std::size_t m_NumberInputColumns;
    std::size_t m_NumberLossParameters;
    const TImmutableRadixSetVec& m_CandidateSplits;
    core::CPackedBitVector m_RowMask;
    CPerSplitDerivatives m_Derivatives;
    SSplitStatistics m_BestSplit;
};
}
}

#endif // INCLUDED_ml_maths_CBoostedTreeLeafNodeStatistics_h
