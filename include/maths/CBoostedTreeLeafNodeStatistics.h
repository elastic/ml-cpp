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
    using TDoubleVector = CDenseVector<double>;
    using TDoubleMatrix = CDenseMatrix<double>;

    //! \brief Aggregate derivatives (gradient and Hessian).
    struct MATHS_EXPORT SDerivatives {
        SDerivatives(std::size_t count, TDoubleVector gradient, TDoubleMatrix curvature)
            : s_Count{count}, s_Gradient{std::move(gradient)}, s_Curvature{std::move(curvature)} {
        }
        //! \warning This assumes that the curvature is stored flat row major.
        SDerivatives(std::size_t count,
                     const TMemoryMappedDoubleVector& gradient,
                     const TMemoryMappedDoubleVector& curvature)
            : s_Count{count}, s_Gradient{gradient.size()}, s_Curvature{gradient.size(),
                                                                       gradient.size()} {
            for (int i = 0; i < gradient.size(); ++i) {
                s_Gradient[i] = gradient[i];
            }
            // We only copy the upper triangle and always use selfadjointView
            // when working with the actual Hessian.
            for (int i = 0, k = 0; i < gradient.size(); ++i) {
                for (int j = i; j < gradient.size(); ++j, ++k) {
                    s_Curvature(i, j) = curvature(k);
                }
            }
        }

        static std::size_t estimateMemoryUsage(std::size_t numberLossParameters) {
            return sizeof(SDerivatives) +
                   las::estimateMemoryUsage<TDoubleVector>(numberLossParameters) +
                   las::estimateMemoryUsage<TDoubleMatrix>(numberLossParameters,
                                                           numberLossParameters);
        }

        std::size_t s_Count;
        TDoubleVector s_Gradient;
        //! Note only the upper triangle is initialized since this is symmetric.
        TDoubleMatrix s_Curvature;
    };

    using TDerivativesVec = std::vector<SDerivatives>;
    using TDerivativesVecVec = std::vector<TDerivativesVec>;

    //! \brief Accumulates aggregate derivatives.
    class MATHS_EXPORT CDerivativesAccumulator {
    public:
        CDerivativesAccumulator(const TMemoryMappedDoubleVector& gradient,
                                const TMemoryMappedDoubleVector& curvature)
            : CDerivativesAccumulator{0, gradient, curvature} {}
        CDerivativesAccumulator(std::size_t count,
                                const TMemoryMappedDoubleVector& gradient,
                                const TMemoryMappedDoubleVector& curvature)
            : m_Count{count}, m_Gradient{gradient}, m_Curvature{curvature} {}

        //! Get the accumulated count.
        std::size_t count() const { return m_Count; }

        //! Get the accumulated gradient.
        const TMemoryMappedDoubleVector& gradient() const { return m_Gradient; }

        //! Get the accumulated curvature.
        const TMemoryMappedDoubleVector& curvature() const {
            return m_Curvature;
        }

        //! Add \p count, \p gradient and \p curvature to the accumulator.
        void add(std::size_t count,
                 const TMemoryMappedFloatVector& gradient,
                 const TMemoryMappedFloatVector& curvature) {
            m_Count += count;
            m_Gradient += gradient;
            m_Curvature += curvature;
        }

        //! Compute the accumulation of both collections of derivatives.
        void merge(const CDerivativesAccumulator& other) {
            m_Count += other.m_Count;
            m_Gradient += other.m_Gradient;
            m_Curvature += other.m_Curvature;
        }

        //! Get a checksum of this object.
        std::uint64_t checksum(std::uint64_t seed = 0) const {
            seed = CChecksum::calculate(seed, m_Count);
            seed = CChecksum::calculate(seed, m_Gradient);
            return CChecksum::calculate(seed, m_Curvature);
        }

    private:
        std::size_t m_Count = 0;
        TMemoryMappedDoubleVector m_Gradient;
        TMemoryMappedDoubleVector m_Curvature;
    };

    using TDerivativesAccumulatorVec = std::vector<CDerivativesAccumulator>;
    using TDerivativesAccumulatorVecVec = std::vector<TDerivativesAccumulatorVec>;

    //! \brief A collection of aggregate derivatives for candidate feature splits.
    class MATHS_EXPORT CSplitDerivativesAccumulator {
    public:
        CSplitDerivativesAccumulator(const TImmutableRadixSetVec& candidateSplits,
                                     std::size_t numberLossParameters)
            : m_NumberLossParameters{numberLossParameters} {

            std::size_t totalNumberCandidateSplits{std::accumulate(
                candidateSplits.begin(), candidateSplits.end(), std::size_t{0},
                [](std::size_t size, const TImmutableRadixSet& splits) {
                    return size + splits.size() + 1;
                })};
            int numberGradients{static_cast<int>(numberLossParameters)};
            int numberCurvatures{static_cast<int>(
                boosted_tree_detail::lossHessianStoredSize(numberLossParameters))};
            int numberDerivatives{numberGradients + numberCurvatures};
            m_Derivatives.resize(candidateSplits.size());
            m_DerivativesStorage.resize(totalNumberCandidateSplits * numberDerivatives, 0.0);
            m_MissingDerivatives.reserve(candidateSplits.size());
            m_MissingDerivativesStorage.resize(candidateSplits.size() * numberDerivatives, 0.0);

            double* storage{nullptr};
            for (std::size_t i = 0, m = 0, n = 0; i < candidateSplits.size();
                 ++i, m += numberDerivatives) {

                m_Derivatives[i].reserve(candidateSplits[i].size() + 1);
                for (std::size_t j = 0; j <= candidateSplits[i].size();
                     ++j, n += numberDerivatives) {
                    storage = &m_DerivativesStorage[n];
                    m_Derivatives[i].emplace_back(
                        TMemoryMappedDoubleVector{storage, numberGradients},
                        TMemoryMappedDoubleVector{storage + numberGradients, numberCurvatures});
                }

                storage = &m_MissingDerivativesStorage[m];
                m_MissingDerivatives.emplace_back(
                    TMemoryMappedDoubleVector{storage, numberGradients},
                    TMemoryMappedDoubleVector{storage + numberGradients, numberCurvatures});
            }
        }

        CSplitDerivativesAccumulator(const CSplitDerivativesAccumulator& other)
            : m_NumberLossParameters{other.m_NumberLossParameters},
              m_DerivativesStorage{other.m_DerivativesStorage},
              m_MissingDerivativesStorage{other.m_MissingDerivativesStorage} {

            int numberGradients{static_cast<int>(m_NumberLossParameters)};
            int numberCurvatures{static_cast<int>(
                boosted_tree_detail::lossHessianStoredSize(m_NumberLossParameters))};
            int numberDerivatives{numberGradients + numberCurvatures};

            m_Derivatives.resize(other.m_Derivatives.size());
            m_MissingDerivatives.reserve(other.m_MissingDerivatives.size());

            double* storage{nullptr};
            for (std::size_t i = 0, m = 0, n = 0;
                 i < other.m_Derivatives.size(); ++i, m += numberDerivatives) {

                m_Derivatives[i].reserve(other.m_Derivatives[i].size());
                for (std::size_t j = 0; j < other.m_Derivatives[i].size();
                     ++j, n += numberDerivatives) {
                    storage = &m_DerivativesStorage[n];
                    m_Derivatives[i].emplace_back(
                        other.m_Derivatives[i][j].count(),
                        TMemoryMappedDoubleVector{storage, numberGradients},
                        TMemoryMappedDoubleVector{storage + numberGradients, numberCurvatures});
                }

                storage = &m_MissingDerivativesStorage[m];
                m_MissingDerivatives.emplace_back(
                    other.m_MissingDerivatives[i].count(),
                    TMemoryMappedDoubleVector{storage, numberGradients},
                    TMemoryMappedDoubleVector{storage + numberGradients, numberCurvatures});
            }
        }

        CSplitDerivativesAccumulator(CSplitDerivativesAccumulator&&) = default;

        CSplitDerivativesAccumulator&
        operator=(const CSplitDerivativesAccumulator& other) = delete;
        CSplitDerivativesAccumulator& operator=(CSplitDerivativesAccumulator&&) = default;

        //! Compute the accumulation of both collections of per split derivatives.
        void merge(const CSplitDerivativesAccumulator& other) {
            for (std::size_t i = 0; i < m_Derivatives.size(); ++i) {
                for (std::size_t j = 0; j < m_Derivatives[i].size(); ++j) {
                    m_Derivatives[i][j].merge(other.m_Derivatives[i][j]);
                }
                m_MissingDerivatives[i].merge(other.m_MissingDerivatives[i]);
            }
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

        //! Read out the accumulated derivatives.
        auto read() const {
            TDerivativesVecVec derivatives;
            derivatives.resize(m_Derivatives.size());
            for (std::size_t i = 0; i < m_Derivatives.size(); ++i) {
                derivatives[i].reserve(m_Derivatives[i].size());
                for (const auto& derivative : m_Derivatives[i]) {
                    derivatives[i].emplace_back(derivative.count(), derivative.gradient(),
                                                derivative.curvature());
                }
            }

            TDerivativesVec missingDerivatives;
            missingDerivatives.reserve(m_MissingDerivatives.size());
            for (const auto& derivative : m_MissingDerivatives) {
                missingDerivatives.emplace_back(derivative.count(), derivative.gradient(),
                                                derivative.curvature());
            }

            return std::make_pair(std::move(derivatives), std::move(missingDerivatives));
        }

        //! Get a checksum of this object.
        std::uint64_t checksum(std::uint64_t seed = 0) const {
            seed = CChecksum::calculate(seed, m_NumberLossParameters);
            seed = CChecksum::calculate(seed, m_Derivatives);
            seed = CChecksum::calculate(seed, m_MissingDerivatives);
            return seed;
        }

    private:
        std::size_t m_NumberLossParameters;
        TDerivativesAccumulatorVecVec m_Derivatives;
        TDoubleVec m_DerivativesStorage;
        TDerivativesAccumulatorVec m_MissingDerivatives;
        TDoubleVec m_MissingDerivativesStorage;
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
                           CSplitDerivativesAccumulator& splitDerivativesAccumulator) const;
    SSplitStatistics computeBestSplitStatistics(const TRegularization& regularization,
                                                const TSizeVec& featureBag) const;

private:
    std::size_t m_Id;
    std::size_t m_Depth;
    std::size_t m_NumberInputColumns;
    std::size_t m_NumberLossParameters;
    const TImmutableRadixSetVec& m_CandidateSplits;
    core::CPackedBitVector m_RowMask;
    TDerivativesVecVec m_Derivatives;
    TDerivativesVec m_MissingDerivatives;
    SSplitStatistics m_BestSplit;
};
}
}

#endif // INCLUDED_ml_maths_CBoostedTreeLeafNodeStatistics_h
