/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CBoostedTreeLeafNodeStatistics_h
#define INCLUDED_ml_maths_CBoostedTreeLeafNodeStatistics_h

#include <core/CAlignment.h>
#include <core/CImmutableRadixSet.h>
#include <core/CMemory.h>
#include <core/CPackedBitVector.h>
#include <core/CSmallVector.h>

#include <maths/CBoostedTreeHyperparameters.h>
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
    using TMemoryMappedFloatVector = CMemoryMappedDenseVector<CFloatStorage, Eigen::Aligned16>;
    using TMemoryMappedDoubleVector = CMemoryMappedDenseVector<double, Eigen::Aligned16>;
    using TMemoryMappedDoubleMatrix = CMemoryMappedDenseMatrix<double, Eigen::Aligned16>;

    //! \brief Accumulates aggregate derivatives.
    class MATHS_EXPORT CDerivatives {
    public:
        //! Bounds the minimum diagonal of the Hessian.
        static constexpr double SMALLEST_RELATIVE_CURVATURE{1e-20};

        //! See core::CMemory.
        static bool dynamicSizeAlwaysZero() { return true; }

    public:
        CDerivatives(std::size_t numberLossParameters, double* storageGradients, double* storageCurvatures)
            : m_Gradient{storageGradients, static_cast<int>(numberLossParameters)},
              m_Curvature{storageCurvatures, static_cast<int>(numberLossParameters),
                          static_cast<int>(numberLossParameters)} {}

        //! Get the accumulated count.
        std::size_t count() const { return m_Count; }

        //! Get the accumulated gradient.
        const TMemoryMappedDoubleVector& gradient() const { return m_Gradient; }

        //! Get the accumulated curvature.
        const TMemoryMappedDoubleMatrix& curvature() const {
            return m_Curvature;
        }

        //! Zero all values.
        void zero() {
            m_Count = 0;
            this->flatView().setZero();
        }

        //! Add \p count and \p derivatives to the accumulator.
        void add(std::size_t count, const TMemoryMappedFloatVector& derivatives) {
            m_Count += count;
            this->upperTriangularFlatView() += derivatives;
        }

        //! Compute the accumulation of both collections of derivatives.
        void add(const CDerivatives& other) {
            m_Count += other.m_Count;
            this->flatView() += const_cast<CDerivatives*>(&other)->flatView();
        }

        //! Set to the difference of \p lhs and \p rhs.
        void subtract(const CDerivatives& rhs) {
            m_Count -= rhs.m_Count;
            if (m_Count > 0) {
                m_Gradient -= rhs.m_Gradient;
                m_Curvature -= rhs.m_Curvature;
                // None of our loss functions have negative curvature therefore we
                // shouldn't allow the cumulative curvature to be negative either.
                // In this case we force it to be a very small multiple of the
                // magnitude of the gradient since this is the closest feasible
                // estimate.
                for (int i = 0; i < m_Gradient.size(); ++i) {
                    m_Curvature(i, i) = std::max(m_Curvature(i, i),
                                                 SMALLEST_RELATIVE_CURVATURE *
                                                     std::fabs(m_Gradient(i)));
                }
            } else {
                // Numeric errors mean that it's possible the sum curvature for a
                // candidate split is identically zero while the gradient is epsilon.
                // This can cause the node gain to appear infinite (when there is no
                // weight regularisation) which in turns causes problems initialising
                // the region we search for optimal hyperparameter values. We can
                // safely force the gradient and curvature to be zero if we detect
                // that the count is zero.
                m_Gradient.setZero();
                m_Curvature.setZero();
            }
        }

        //! Remap the accumulated curvature to lower triangle row major format.
        void remapCurvature() {
            // For performance, we accumulate curvatures into the first n + n (n + 1) / 2
            // elements of the array backing upperTriangularFlatView. However, the memory
            // mapped matrix class expects them to be stored column major in the lower
            // triangle of an n x n matrix. This copies them backwards to their correct
            // positions.
            TMemoryMappedDoubleVector derivatives{this->upperTriangularFlatView()};
            for (std::ptrdiff_t j = m_Curvature.cols() - 1, k = derivatives.rows() - 1;
                 j >= 0; --j) {
                for (std::ptrdiff_t i = m_Curvature.rows() - 1; i >= j; --i, --k) {
                    m_Curvature(i, j) = derivatives(k);
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
        TMemoryMappedDoubleVector upperTriangularFlatView() {
            // Gradient + upper triangle of the Hessian.
            auto n = m_Gradient.rows();
            return {m_Gradient.data(), n * (n + 3) / 2};
        }

        TMemoryMappedDoubleVector flatView() {
            // Gradient + pad + full Hessian.
            auto n = m_Curvature.data() - m_Gradient.data() +
                     m_Curvature.rows() * m_Curvature.cols();
            return {m_Gradient.data(), n};
        }

    private:
        std::size_t m_Count = 0;
        TMemoryMappedDoubleVector m_Gradient;
        TMemoryMappedDoubleMatrix m_Curvature;
    };

    //! \brief A collection of aggregate derivatives for candidate feature splits.
    class MATHS_EXPORT CSplitsDerivatives {
    public:
        using TDerivativesVec = std::vector<CDerivatives>;

    public:
        explicit CSplitsDerivatives(std::size_t numberLossParameters = 0)
            : m_NumberLossParameters{numberLossParameters},
              m_PositiveDerivativesSum{TDerivatives2x1::Zero()},
              m_NegativeDerivativesSum{TDerivatives2x1::Zero()},
              m_PositiveDerivativesMax{-boosted_tree_detail::INF},
              m_PositiveDerivativesMin{boosted_tree_detail::INF},
              m_NegativeDerivativesMin{boosted_tree_detail::INF, boosted_tree_detail::INF} {}
        CSplitsDerivatives(const TImmutableRadixSetVec& candidateSplits, std::size_t numberLossParameters)
            : m_NumberLossParameters{numberLossParameters},
              m_PositiveDerivativesSum{TDerivatives2x1::Zero()},
              m_NegativeDerivativesSum{TDerivatives2x1::Zero()},
              m_PositiveDerivativesMax{-boosted_tree_detail::INF},
              m_PositiveDerivativesMin{boosted_tree_detail::INF},
              m_NegativeDerivativesMin{boosted_tree_detail::INF, boosted_tree_detail::INF} {
            this->map(candidateSplits);
        }
        CSplitsDerivatives(const CSplitsDerivatives& other)
            : m_NumberLossParameters{other.m_NumberLossParameters},
              m_PositiveDerivativesSum{TDerivatives2x1::Zero()},
              m_NegativeDerivativesSum{TDerivatives2x1::Zero()},
              m_PositiveDerivativesMax{-boosted_tree_detail::INF},
              m_PositiveDerivativesMin{boosted_tree_detail::INF},
              m_NegativeDerivativesMin{boosted_tree_detail::INF, boosted_tree_detail::INF} {
            this->map(other.m_Derivatives);
            this->add(other);
        }
        CSplitsDerivatives(CSplitsDerivatives&&) = default;

        CSplitsDerivatives& operator=(const CSplitsDerivatives& other) = delete;
        CSplitsDerivatives& operator=(CSplitsDerivatives&&) = default;

        //! Re-initialize recycling the allocated memory.
        void reinitialize(const TImmutableRadixSetVec& candidateSplits,
                          std::size_t numberLossParameters) {
            m_NumberLossParameters = numberLossParameters;
            for (auto& derivatives : m_Derivatives) {
                derivatives.clear();
            }
            m_MissingDerivatives.clear();
            m_Storage.clear();
            this->map(candidateSplits);
        }

        //! Efficiently swap this and \p other.
        void swap(CSplitsDerivatives& other) {
            std::swap(m_NumberLossParameters, other.m_NumberLossParameters);
            m_Derivatives.swap(other.m_Derivatives);
            m_MissingDerivatives.swap(other.m_MissingDerivatives);
            m_Storage.swap(other.m_Storage);
            std::swap(m_PositiveDerivativesSum, other.m_PositiveDerivativesSum);
            std::swap(m_NegativeDerivativesSum, other.m_NegativeDerivativesSum);
            std::swap(m_PositiveDerivativesMax, other.m_PositiveDerivativesMax);
            std::swap(m_PositiveDerivativesMin, other.m_PositiveDerivativesMin);
            std::swap(m_NegativeDerivativesMin, other.m_NegativeDerivativesMin);
        }

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
        //! the \p split of \p feature.
        void addDerivatives(std::size_t feature,
                            std::size_t split,
                            const TMemoryMappedFloatVector& derivatives) {
            m_Derivatives[feature][split].add(1, derivatives);
        }

        //! Add \p gradient and \p curvature to the accumulated derivatives for
        //! missing values of \p feature.
        void addMissingDerivatives(std::size_t feature,
                                   const TMemoryMappedFloatVector& derivatives) {
            m_MissingDerivatives[feature].add(1, derivatives);
        }

        //! Zero all values.
        void zero() {
            m_PositiveDerivativesSum.fill(0.0);
            m_NegativeDerivativesSum.fill(0.0);
            m_PositiveDerivativesMax = -boosted_tree_detail::INF;
            m_PositiveDerivativesMin = boosted_tree_detail::INF;
            m_NegativeDerivativesMin.fill(boosted_tree_detail::INF);

            for (std::size_t i = 0; i < m_Derivatives.size(); ++i) {
                for (std::size_t j = 0; j < m_Derivatives[i].size(); ++j) {
                    m_Derivatives[i][j].zero();
                }
                m_MissingDerivatives[i].zero();
            }
        }

        //! Compute the accumulation of both collections of per split derivatives.
        void add(const CSplitsDerivatives& other) {
            m_PositiveDerivativesSum += other.m_PositiveDerivativesSum;
            m_NegativeDerivativesSum += other.m_NegativeDerivativesSum;
            m_PositiveDerivativesMax = std::max(m_PositiveDerivativesMax,
                                                other.m_PositiveDerivativesMax);
            m_PositiveDerivativesMin = std::min(m_PositiveDerivativesMin,
                                                other.m_PositiveDerivativesMin);
            m_NegativeDerivativesMin =
                m_NegativeDerivativesMin.cwiseMin(other.m_NegativeDerivativesMin);

            for (std::size_t i = 0; i < other.m_Derivatives.size(); ++i) {
                for (std::size_t j = 0; j < other.m_Derivatives[i].size(); ++j) {
                    m_Derivatives[i][j].add(other.m_Derivatives[i][j]);
                }
                m_MissingDerivatives[i].add(other.m_MissingDerivatives[i]);
            }
        }

        //! Subtract \p rhs.
        void subtract(const CSplitsDerivatives& rhs) {
            m_PositiveDerivativesSum -= rhs.m_PositiveDerivativesSum;
            m_NegativeDerivativesSum -= rhs.m_NegativeDerivativesSum;
            for (std::size_t i = 0; i < m_Derivatives.size(); ++i) {
                for (std::size_t j = 0; j < m_Derivatives[i].size(); ++j) {
                    m_Derivatives[i][j].subtract(rhs.m_Derivatives[i][j]);
                }
                m_MissingDerivatives[i].subtract(rhs.m_MissingDerivatives[i]);
            }
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
        static std::size_t estimateMemoryUsage(std::size_t numberFeatures,
                                               std::size_t numberSplitsPerFeature,
                                               std::size_t numberLossParameters) {
            std::size_t derivativesSize{numberFeatures * (numberSplitsPerFeature + 1) *
                                        sizeof(CDerivatives)};
            std::size_t storageSize{numberFeatures * (numberSplitsPerFeature + 1) * numberLossParameters *
                                    (numberLossParameters + 1) * sizeof(double)};
            return sizeof(CSplitsDerivatives) + derivativesSize + storageSize;
        }

        //! Get a checksum of this object.
        std::uint64_t checksum(std::uint64_t seed = 0) const {
            seed = CChecksum::calculate(seed, m_NumberLossParameters);
            seed = CChecksum::calculate(seed, m_Derivatives);
            seed = CChecksum::calculate(seed, m_MissingDerivatives);
            return seed;
        }

        std::size_t numberLossParameters() const {
            return m_NumberLossParameters;
        }

        void addPositiveDerivatives(const TMemoryMappedFloatVector& derivatives) {
            m_PositiveDerivativesSum += derivatives;
            m_PositiveDerivativesMin = std::min(
                m_PositiveDerivativesMin, static_cast<double>(derivatives(1)));
            m_PositiveDerivativesMax = std::max(
                m_PositiveDerivativesMax, static_cast<double>(derivatives(0)));
        }

        void addNegativeDerivatives(const TMemoryMappedFloatVector& derivatives) {
            m_NegativeDerivativesSum += derivatives;
            m_NegativeDerivativesMin = m_NegativeDerivativesMin.cwiseMin(derivatives);
        }

        double positiveDerivativesGSum() const {
            return m_PositiveDerivativesSum(0);
        }

        double negativeDerivativesGSum() const {
            return m_NegativeDerivativesSum(0);
        }

        double positiveDerivativesGMax() const {
            return m_PositiveDerivativesMax;
        }

        double positiveDerivativesHMin() const {
            return m_PositiveDerivativesMin;
        }

        double negativeDerivativesGMin() const {
            return m_NegativeDerivativesMin(0);
        }

        double negativeDerivativesHMin() const {
            return m_NegativeDerivativesMin(1);
        }

    private:
        using TDerivativesVecVec = std::vector<TDerivativesVec>;
        using TAlignedDoubleVec = std::vector<double, core::CAlignedAllocator<double>>;
        using TDerivatives2x1 = Eigen::Matrix<double, 2, 1>;

    private:
        static std::size_t number(const TDerivativesVec& derivatives) {
            return derivatives.size();
        }
        static std::size_t number(const TImmutableRadixSet& splits) {
            return splits.size() + 1;
        }
        template<typename SPLITS>
        void map(const SPLITS& splits) {
            // This function maps the memory in a single presized buffer containing
            // enough space to store all gradient vectors and curvatures. For each
            // feature the layout in this buffer is as follows:
            //
            // "split grad" "split hessian"       "missing grad" "missing hessian"
            //       |            |                     |              |
            //       V            V                     V              V
            // |     n     |      n^2      | ... |      n       |      n^2       |
            //
            // Note we ensure 16 byte alignment because we're using aligned memory
            // mapped vectors which have much better performance.

            std::size_t numberFeatures{splits.size()};
            std::size_t totalNumberSplits{
                std::accumulate(splits.begin(), splits.end(), std::size_t{0},
                                [](std::size_t size, const auto& featureSplits) {
                                    return size + number(featureSplits);
                                })};

            std::size_t numberGradients{this->numberGradients()};
            std::size_t numberDerivatives{this->numberDerivatives()};

            m_Derivatives.resize(numberFeatures);
            m_MissingDerivatives.reserve(numberFeatures);

            m_Storage.resize((totalNumberSplits + numberFeatures) * numberDerivatives, 0.0);

            double* storage{&m_Storage[0]};
            for (std::size_t i = 0; i < numberFeatures; ++i, storage += numberDerivatives) {
                std::size_t size{number(splits[i])};
                m_Derivatives[i].reserve(size);
                for (std::size_t j = 0; j < size; ++j, storage += numberDerivatives) {
                    m_Derivatives[i].emplace_back(m_NumberLossParameters, storage,
                                                  storage + numberGradients);
                }
                m_MissingDerivatives.emplace_back(m_NumberLossParameters, storage,
                                                  storage + numberGradients);
            }
        }

        std::size_t numberDerivatives() const {
            return this->numberGradients() + this->numberCurvatures();
        }

        std::size_t numberGradients() const {
            return core::CAlignment::roundup<double>(core::CAlignment::E_Aligned16,
                                                     m_NumberLossParameters);
        }

        std::size_t numberCurvatures() const {
            return core::CAlignment::roundup<double>(
                core::CAlignment::E_Aligned16, m_NumberLossParameters * m_NumberLossParameters);
        }

    private:
        std::size_t m_NumberLossParameters = 0;
        TDerivativesVecVec m_Derivatives;
        TDerivativesVec m_MissingDerivatives;
        TAlignedDoubleVec m_Storage;
        TDerivatives2x1 m_PositiveDerivativesSum;
        TDerivatives2x1 m_NegativeDerivativesSum;
        double m_PositiveDerivativesMax;
        double m_PositiveDerivativesMin;
        TDerivatives2x1 m_NegativeDerivativesMin;
    };

    //! \brief The derivatives and row masks objects to use for computations.
    //!
    //! DESCRIPTION:\n
    //! These are heavyweight objects and get passed in to minimise the number of
    //! times they need to be allocated. This has the added advantage of keeping
    //! the cache warm since the critical path is always working on the derivatives
    //! objects stored in this class.
    class MATHS_EXPORT CWorkspace {
    public:
        using TPackedBitVectorVec = std::vector<core::CPackedBitVector>;
        using TSplitsDerivativesVec = std::vector<CSplitsDerivatives>;

    public:
        CWorkspace() = default;
        CWorkspace(CWorkspace&&) = default;
        CWorkspace& operator=(const CWorkspace& other) = delete;
        CWorkspace& operator=(CWorkspace&&) = default;

        //! Re-initialize the masks and derivatives.
        void reinitialize(std::size_t numberThreads,
                          const TImmutableRadixSetVec& candidateSplits,
                          std::size_t numberLossParameters) {
            m_MinimumGain = 0.0;
            m_Masks.resize(numberThreads);
            m_Derivatives.reserve(numberThreads);
            for (auto& mask : m_Masks) {
                mask.clear();
            }
            for (auto& derivatives : m_Derivatives) {
                derivatives.reinitialize(candidateSplits, numberLossParameters);
            }
            for (std::size_t i = m_Derivatives.size(); i < numberThreads; ++i) {
                m_Derivatives.emplace_back(candidateSplits, numberLossParameters);
            }
        }

        //! Get the minimum leaf gain which will generate a split.
        double minimumGain() const { return m_MinimumGain; }

        //! Update the minimum gain to be at least \p gain.
        void minimumGain(double gain) {
            m_MinimumGain = std::max(m_MinimumGain, gain);
        }

        //! Reset the minimum gain to its initial value.
        void resetMinimumGain() { m_MinimumGain = 0.0; }

        //! Start working on a new leaf.
        void newLeaf(std::size_t numberThreads) {
            m_NumberThreads = numberThreads;
            m_ReducedMasks = false;
            m_ReducedDerivatives = false;
        }

        //! Get the reduction of the per thread aggregate derivatives.
        CSplitsDerivatives& reducedDerivatives() {
            if (m_ReducedDerivatives == false) {
                for (std::size_t i = 1; i < m_NumberThreads; ++i) {
                    m_Derivatives[0].add(m_Derivatives[i]);
                }
                m_Derivatives[0].remapCurvature();
                m_ReducedDerivatives = true;
            }
            return m_Derivatives[0];
        }

        //! Get the reduction of the per thread masks.
        const core::CPackedBitVector& reducedMask(std::size_t size) {
            if (m_ReducedMasks == false) {
                m_Masks[0].extend(false, size - m_Masks[0].size());
                for (std::size_t i = 1; i < m_NumberThreads; ++i) {
                    m_Masks[i].extend(false, size - m_Masks[i].size());
                    m_Masks[0] |= m_Masks[i];
                }
                m_ReducedMasks = true;
            }
            return m_Masks[0];
        }

        //! Get the workspace row masks.
        TPackedBitVectorVec& masks() { return m_Masks; }

        //! Get the workspace derivatives.
        TSplitsDerivativesVec& derivatives() { return m_Derivatives; }

        //! Get the memory used by this object.
        std::size_t memoryUsage() const {
            return core::CMemory::dynamicSize(m_Masks) +
                   core::CMemory::dynamicSize(m_Derivatives);
        }

    private:
        std::size_t m_NumberThreads = 0;
        double m_MinimumGain = 0.0;
        bool m_ReducedMasks = false;
        bool m_ReducedDerivatives = false;
        TPackedBitVectorVec m_Masks;
        TSplitsDerivativesVec m_Derivatives;
    };

public:
    CBoostedTreeLeafNodeStatistics(std::size_t id,
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
    CBoostedTreeLeafNodeStatistics(std::size_t id,
                                   const CBoostedTreeLeafNodeStatistics& parent,
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
    CBoostedTreeLeafNodeStatistics(std::size_t id,
                                   CBoostedTreeLeafNodeStatistics&& parent,
                                   const TRegularization& regularization,
                                   const TSizeVec& nodeFeatureBag,
                                   CWorkspace& workspace);

    CBoostedTreeLeafNodeStatistics(const CBoostedTreeLeafNodeStatistics&) = delete;
    CBoostedTreeLeafNodeStatistics& operator=(const CBoostedTreeLeafNodeStatistics&) = delete;

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
                    CWorkspace& workspace);

    //! Order two leaves by decreasing gain in splitting them.
    bool operator<(const CBoostedTreeLeafNodeStatistics& rhs) const;

    //! Get the gain in loss of the best split of this leaf.
    double gain() const;

    //! Get the gain upper bound for the left child.
    double leftChildMaxGain() const;

    //! Get the gain upper bound for the right child.
    double rightChildMaxGain() const;

    //! Get the total curvature of node.
    double curvature() const;

    //! Get the best (feature, feature value) split.
    TSizeDoublePr bestSplit() const;

    //! Get the row count of the child node with the fewest rows.
    std::size_t minimumChildRowCount() const;

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

    //! Get the best split info as a string.
    std::string print() const { return m_BestSplit.print(); }

    //! Estimate the maximum leaf statistics' memory usage training on a data frame
    //! with \p numberCols columns using \p numberSplitsPerFeature for a loss function
    //! with \p numberLossParameters parameters.
    static std::size_t estimateMemoryUsage(std::size_t numberCols,
                                           std::size_t numberSplitsPerFeature,
                                           std::size_t numberLossParameters);

private:
    using TSizeVecCRef = std::reference_wrapper<const TSizeVec>;

    //! \brief Statistics relating to a split of the node.
    struct MATHS_EXPORT SSplitStatistics
        : private boost::less_than_comparable<SSplitStatistics> {
        SSplitStatistics() = default;
        SSplitStatistics(double gain,
                         double curvature,
                         std::size_t feature,
                         double splitAt,
                         std::size_t minimumChildRowCount,
                         bool leftChildHasFewerRows,
                         bool assignMissingToLeft)
            : s_Gain{CMathsFuncs::isNan(gain) ? -boosted_tree_detail::INF : gain},
              s_Curvature{curvature}, s_Feature{feature}, s_SplitAt{splitAt},
              s_MinimumChildRowCount{static_cast<std::uint32_t>(minimumChildRowCount)},
              s_LeftChildHasFewerRows{leftChildHasFewerRows}, s_AssignMissingToLeft{assignMissingToLeft} {
        }

        bool operator<(const SSplitStatistics& rhs) const {
            return COrderings::lexicographical_compare(
                s_Gain, s_Curvature, s_Feature, s_SplitAt, // <- lhs
                rhs.s_Gain, rhs.s_Curvature, rhs.s_Feature, rhs.s_SplitAt);
        }

        std::string print() const {
            std::ostringstream result;
            result << "split feature '" << s_Feature << "' @ " << s_SplitAt
                   << ", gain = " << s_Gain << ", leftChildMaxGain = " << s_LeftChildMaxGain
                   << ", rightChildMaxGain = " << s_RightChildMaxGain;
            return result.str();
        }

        double s_Gain = -boosted_tree_detail::INF;
        double s_Curvature = 0.0;
        std::size_t s_Feature = std::numeric_limits<std::size_t>::max();
        double s_SplitAt = boosted_tree_detail::INF;
        std::uint32_t s_MinimumChildRowCount = 0;
        bool s_LeftChildHasFewerRows = true;
        bool s_AssignMissingToLeft = true;
        double s_LeftChildMaxGain = boosted_tree_detail::INF;
        double s_RightChildMaxGain = boosted_tree_detail::INF;
    };

private:
    void computeAggregateLossDerivatives(std::size_t numberThreads,
                                         const core::CDataFrame& frame,
                                         const CDataFrameCategoryEncoder& encoder,
                                         const TSizeVec& featureBag,
                                         const core::CPackedBitVector& rowMask,
                                         CWorkspace& workspace) const;
    void computeRowMaskAndAggregateLossDerivatives(std::size_t numberThreads,
                                                   const core::CDataFrame& frame,
                                                   const CDataFrameCategoryEncoder& encoder,
                                                   bool isLeftChild,
                                                   const CBoostedTreeNode& split,
                                                   const TSizeVec& featureBag,
                                                   const core::CPackedBitVector& parentRowMask,
                                                   CWorkspace& workspace) const;
    void addRowDerivatives(const TSizeVec& featureBag,
                           const CEncodedDataFrameRowRef& row,
                           CSplitsDerivatives& splitsDerivatives) const;

    SSplitStatistics computeBestSplitStatistics(const TRegularization& regularization,
                                                const TSizeVec& featureBag) const;

    double childMaxGain(double gChild, double minLossChild, double lambda) const;

private:
    std::size_t m_Id;
    std::size_t m_Depth;
    TSizeVecCRef m_ExtraColumns;
    std::size_t m_NumberLossParameters;
    const TImmutableRadixSetVec& m_CandidateSplits;
    core::CPackedBitVector m_RowMask;
    CSplitsDerivatives m_Derivatives;
    SSplitStatistics m_BestSplit;
};
}
}

#endif // INCLUDED_ml_maths_CBoostedTreeLeafNodeStatistics_h
