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

#ifndef INCLUDED_ml_maths_analytics_CBoostedTreeLeafNodeStatistics_h
#define INCLUDED_ml_maths_analytics_CBoostedTreeLeafNodeStatistics_h

#include <core/CAlignment.h>
#include <core/CDataFrame.h>
#include <core/CMemoryFwd.h>
#include <core/CPackedBitVector.h>
#include <core/Concurrency.h>

#include <maths/analytics/CBoostedTreeHyperparameters.h>
#include <maths/analytics/CBoostedTreeLeafNodeStatisticsThreading.h>
#include <maths/analytics/ImportExport.h>

#include <maths/common/CLinearAlgebraEigen.h>
#include <maths/common/CLinearAlgebraShims.h>
#include <maths/common/CMathsFuncs.h>
#include <maths/common/COrderings.h>
#include <maths/common/MathsTypes.h>

#include <boost/operators.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <list>
#include <numeric>
#include <vector>

namespace CBoostedTreeLeafNodeStatisticsTest {
struct testComputeBestSplitStatisticsThreading;
}
namespace ml {
namespace maths {
namespace analytics {
class CBoostedTreeNode;
class CDataFrameCategoryEncoder;
class CEncodedDataFrameRowRef;

//! \brief Manages accessing the bytes of common::CFloatStorage.
//!
//! DESCRIPTION:\n
//! We pack the row's split indices (four to a float) into the raw bytes of the
//! data frame. This means we only need to lookup aggregate derivative bucket to
//! update in CBoostedTreeLeafNodeStatistics::addRowDerivatives. This manages
//! reading and writing bits to an array of four std::uint8_t types.
class MATHS_ANALYTICS_EXPORT CPackedUInt8Decorator : public core::CFloatStorage {
public:
    using TUInt8Ary = std::array<std::uint8_t, sizeof(common::CFloatStorage)>;

public:
    explicit CPackedUInt8Decorator(core::CFloatStorage storage)
        : core::CFloatStorage{storage} {}
    explicit CPackedUInt8Decorator(TUInt8Ary bytes) {
        std::memcpy(&this->storage(), &bytes[0], sizeof(common::CFloatStorage));
    }
    TUInt8Ary readBytes() const {
        TUInt8Ary bytes;
        std::memcpy(&bytes[0], &this->cstorage(), sizeof(common::CFloatStorage));
        return bytes;
    }
};

//! \brief Maintains a collection of statistics about a leaf of the regression
//! tree as it is built.
//!
//! DESCRIPTION:\N
//! The regression tree is grown top down by greedily selecting the split with
//! the maximum gain (in the loss). This finds and scores the maximum gain split
//! of a single leaf of the tree.
class MATHS_ANALYTICS_EXPORT CBoostedTreeLeafNodeStatistics {
public:
    using TDoubleVec = std::vector<double>;
    using TSizeVec = std::vector<std::size_t>;
    using TSizeDoublePr = std::pair<std::size_t, double>;
    using TFloatVec = std::vector<common::CFloatStorage>;
    using TFloatVecVec = std::vector<TFloatVec>;
    using TRegularization = CBoostedTreeHyperparameters;
    using TPtr = std::unique_ptr<CBoostedTreeLeafNodeStatistics>;
    using TPtrPtrPr = std::pair<TPtr, TPtr>;
    using TMemoryMappedFloatVector =
        common::CMemoryMappedDenseVector<common::CFloatStorage, Eigen::Aligned16>;
    using TMemoryMappedDoubleVector =
        common::CMemoryMappedDenseVector<double, Eigen::Aligned16>;
    using TMemoryMappedDoubleMatrix =
        common::CMemoryMappedDenseMatrix<double, Eigen::Aligned16>;
    using TThreading = CBoostedTreeLeafNodeStatisticsThreading;

    //! \brief Accumulates aggregate derivatives.
    class MATHS_ANALYTICS_EXPORT CDerivatives {
    public:
        //! Bounds the minimum diagonal of the Hessian.
        static constexpr double SMALLEST_RELATIVE_CURVATURE{1e-20};

        //! See core::CMemory.
        static constexpr bool dynamicSizeAlwaysZero() { return true; }

    public:
        CDerivatives(int n, double* storageGradients, double* storageCurvatures, double* storageCount)
            : m_Count{storageCount}, m_Gradient(storageGradients, n),
              m_Curvature(storageCurvatures, n, n) {}

        //! Set the accumulated count.
        std::size_t count() const { return static_cast<std::size_t>(*m_Count); }

        //! Get the accumulated gradient.
        const TMemoryMappedDoubleVector& gradient() const { return m_Gradient; }

        //! Get the accumulated curvature.
        const TMemoryMappedDoubleMatrix& curvature() const {
            return m_Curvature;
        }

        //! Add \p count and \p derivatives.
        void add(std::size_t count, const TMemoryMappedFloatVector& derivatives) {
            this->flatView() += derivatives;
            *m_Count += static_cast<double>(count);
        }

        //! Compute the accumulation of both collections of derivatives.
        void add(const CDerivatives& rhs) {
            this->flatView() += const_cast<CDerivatives*>(&rhs)->flatView();
            *m_Count += *rhs.m_Count;
        }

        //! Set to the difference of \p lhs and \p rhs.
        void subtract(const CDerivatives& rhs) {
            *m_Count -= *rhs.m_Count;
            if (*m_Count > 0) {
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
            // elements of the array backing flatView. However, the memory mapped matrix
            // class expects them to be stored column major in the lower triangle of an
            // n x n matrix. This copies them backwards to their correct positions.
            TMemoryMappedDoubleVector derivatives{this->flatView()};
            for (std::ptrdiff_t j = m_Curvature.cols() - 1, k = derivatives.rows() - 1;
                 j >= 0; --j) {
                for (std::ptrdiff_t i = m_Curvature.rows() - 1; i >= j; --i, --k) {
                    m_Curvature(i, j) = derivatives(k);
                }
            }
        }

        //! Get a checksum of this object.
        std::uint64_t checksum(std::uint64_t seed = 0) const;

    private:
        TMemoryMappedDoubleVector flatView() {
            // Gradient + upper triangle of the Hessian.
            auto n = m_Gradient.rows();
            return {m_Gradient.data(), n * (n + 3) / 2};
        }

    private:
        double* m_Count;
        TMemoryMappedDoubleVector m_Gradient;
        TMemoryMappedDoubleMatrix m_Curvature;
    };

    //! \brief A collection of aggregate derivatives for candidate feature splits.
    class MATHS_ANALYTICS_EXPORT CSplitsDerivatives {
    public:
        using TDerivativesVec = std::vector<CDerivatives>;
        using TLoopBodyVec = std::vector<std::function<void(std::size_t)>>;

    public:
        explicit CSplitsDerivatives(std::size_t dimensionGradient = 0)
            : m_DimensionGradient{dimensionGradient} {}
        CSplitsDerivatives(const TFloatVecVec& candidateSplits, std::size_t dimensionGradient)
            : m_DimensionGradient{dimensionGradient} {
            this->initializeAndMapStorage(candidateSplits);
        }
        CSplitsDerivatives(const CSplitsDerivatives& other)
            : m_DimensionGradient{other.m_DimensionGradient},
              m_PositiveDerivativesSum{other.m_PositiveDerivativesSum},
              m_NegativeDerivativesSum{other.m_NegativeDerivativesSum},
              m_PositiveDerivativesMax{other.m_PositiveDerivativesMax},
              m_PositiveDerivativesMin{other.m_PositiveDerivativesMin},
              m_NegativeDerivativesMin{other.m_NegativeDerivativesMin} {
            this->copyAndMapStorage(other.m_Storage, other.m_Derivatives);
        }
        CSplitsDerivatives(CSplitsDerivatives&&) = default;

        CSplitsDerivatives& operator=(const CSplitsDerivatives& other) = delete;
        CSplitsDerivatives& operator=(CSplitsDerivatives&&) = default;

        //! Re-initialize recycling the allocated memory.
        void reinitialize(const TFloatVecVec& candidateSplits, std::size_t dimensionGradient) {
            m_DimensionGradient = dimensionGradient;
            for (auto& derivatives : m_Derivatives) {
                derivatives.clear();
            }
            m_Storage.clear();
            this->initializeAndMapStorage(candidateSplits);
        }

        //! Efficiently swap this and \p other.
        void swap(CSplitsDerivatives& other) {
            std::swap(m_DimensionGradient, other.m_DimensionGradient);
            std::swap(m_PositiveDerivativesSum, other.m_PositiveDerivativesSum);
            std::swap(m_NegativeDerivativesSum, other.m_NegativeDerivativesSum);
            std::swap(m_PositiveDerivativesMax, other.m_PositiveDerivativesMax);
            std::swap(m_PositiveDerivativesMin, other.m_PositiveDerivativesMin);
            std::swap(m_NegativeDerivativesMin, other.m_NegativeDerivativesMin);
            m_Storage.swap(other.m_Storage);
            m_Derivatives.swap(other.m_Derivatives);
        }

        //! Copy the derivatives from \p other.
        //!
        //! \note Copying is an expensive operation so we use an explicit function
        //! instead of operator= to avoid accidental copies.
        void copy(const CSplitsDerivatives& other) {
            if (this->conformable(other) == false) {
                CSplitsDerivatives tmp{other};
                std::swap(*this, tmp);
            } else {
                m_PositiveDerivativesSum = other.m_PositiveDerivativesSum;
                m_NegativeDerivativesSum = other.m_NegativeDerivativesSum;
                m_PositiveDerivativesMax = other.m_PositiveDerivativesMax;
                m_PositiveDerivativesMin = other.m_PositiveDerivativesMin;
                m_NegativeDerivativesMin = other.m_NegativeDerivativesMin;
                std::copy(other.m_Storage.begin(), other.m_Storage.end(),
                          m_Storage.begin());
            }
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

        //! \return The number of split aggregate derivatives for \p feature.
        std::size_t numberDerivatives(std::size_t feature) const {
            return m_Derivatives[feature].size() - 1;
        }

        //! \return The total number of split aggregate derivatives for \p featureBag.
        std::size_t numberDerivatives(const TSizeVec& featureBag) const {
            return std::accumulate(featureBag.begin(), featureBag.end(),
                                   std::size_t{0}, [this](auto result, auto feature) {
                                       return result + this->numberDerivatives(feature);
                                   });
        }

        //! \return An iterator to the begining of the split aggregate derivatives
        //! for \p feature.
        auto beginDerivatives(std::size_t feature) const {
            return m_Derivatives[feature].begin();
        }

        //! \return An iterator to the end of the split aggregate derivatives for
        //! \p feature.
        auto endDerivatives(std::size_t feature) const {
            return m_Derivatives[feature].end() - 1;
        }

        //! \return The count for missing \p feature.
        std::size_t missingCount(std::size_t feature) const {
            return m_Derivatives[feature].back().count();
        }

        //! \return The aggregate gradients for missing \p feature.
        const TMemoryMappedDoubleVector& missingGradient(std::size_t feature) const {
            return m_Derivatives[feature].back().gradient();
        }

        //! \return The aggregate curvatures for missing \p feature.
        const TMemoryMappedDoubleMatrix& missingCurvature(std::size_t feature) const {
            return m_Derivatives[feature].back().curvature();
        }

        //! \return The sum of positive loss gradients.
        double positiveDerivativesGSum() const {
            return m_PositiveDerivativesSum(0);
        }

        //! \return The sum of negative loss gradients.
        double negativeDerivativesGSum() const {
            return m_NegativeDerivativesSum(0);
        }

        //! \return The largest positive gradient.
        double positiveDerivativesGMax() const {
            return m_PositiveDerivativesMax;
        }

        //! \return The smallest loss curvature.
        double positiveDerivativesHMin() const {
            return m_PositiveDerivativesMin;
        }

        //! \return The smallest negative loss gradient.
        double negativeDerivativesGMin() const {
            return m_NegativeDerivativesMin(0);
        }

        //! \return The smallest loss curvature.
        double negativeDerivativesHMin() const {
            return m_NegativeDerivativesMin(1);
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
            m_Derivatives[feature].back().add(1, derivatives);
        }

        //! Update the positive derivative statistics.
        void addPositiveDerivatives(const TMemoryMappedFloatVector& derivatives) {
            m_PositiveDerivativesSum += derivatives;
            m_PositiveDerivativesMin = std::min(
                m_PositiveDerivativesMin, static_cast<double>(derivatives(1)));
            m_PositiveDerivativesMax = std::max(
                m_PositiveDerivativesMax, static_cast<double>(derivatives(0)));
        }

        //! Update the negative derivative statistics.
        void addNegativeDerivatives(const TMemoryMappedFloatVector& derivatives) {
            m_NegativeDerivativesSum += derivatives;
            m_NegativeDerivativesMin = m_NegativeDerivativesMin.cwiseMin(derivatives);
        }

        //! Zero all values.
        void zero() {
            m_PositiveDerivativesSum.fill(0.0);
            m_NegativeDerivativesSum.fill(0.0);
            m_PositiveDerivativesMax = -INF;
            m_PositiveDerivativesMin = INF;
            m_NegativeDerivativesMin.fill(INF);
            std::fill(m_Storage.begin(), m_Storage.end(), 0.0);
        }

        //! Compute the accumulation of both collections of per split derivatives.
        void add(std::size_t numberThreads, const CSplitsDerivatives& rhs, const TSizeVec& featureBag) {
            m_PositiveDerivativesSum += rhs.m_PositiveDerivativesSum;
            m_NegativeDerivativesSum += rhs.m_NegativeDerivativesSum;
            m_PositiveDerivativesMax =
                std::max(m_PositiveDerivativesMax, rhs.m_PositiveDerivativesMax);
            m_PositiveDerivativesMin =
                std::min(m_PositiveDerivativesMin, rhs.m_PositiveDerivativesMin);
            m_NegativeDerivativesMin =
                m_NegativeDerivativesMin.cwiseMin(rhs.m_NegativeDerivativesMin);
            numberThreads = TThreading::numberThreadsForAddSplitsDerivatives(
                numberThreads, featureBag.size(), m_DimensionGradient,
                this->numberDerivatives(featureBag));
            TLoopBodyVec add(numberThreads, [&](std::size_t i) {
                for (std::size_t j = 0; j < rhs.m_Derivatives[i].size(); ++j) {
                    m_Derivatives[i][j].add(rhs.m_Derivatives[i][j]);
                }
            });
            core::parallel_for_each(featureBag.begin(), featureBag.end(), add);
        }

        //! Subtract \p rhs.
        void subtract(std::size_t numberThreads,
                      const CSplitsDerivatives& rhs,
                      const TSizeVec& featureBag) {
            m_PositiveDerivativesSum -= rhs.m_PositiveDerivativesSum;
            m_NegativeDerivativesSum -= rhs.m_NegativeDerivativesSum;
            numberThreads = TThreading::numberThreadsForSubtractSplitsDerivatives(
                numberThreads, featureBag.size(), m_DimensionGradient,
                this->numberDerivatives(featureBag));
            TLoopBodyVec subtract(numberThreads, [&](std::size_t i) {
                for (std::size_t j = 0; j < m_Derivatives[i].size(); ++j) {
                    m_Derivatives[i][j].subtract(rhs.m_Derivatives[i][j]);
                }
            });
            core::parallel_for_each(featureBag.begin(), featureBag.end(), subtract);
        }

        //! Remap the accumulated curvature to lower triangle row major format.
        void remapCurvature(std::size_t numberThreads, const TSizeVec& featureBag) {
            numberThreads = TThreading::numberThreadsForRemapSplitsDerivatives(
                numberThreads, featureBag.size(), m_DimensionGradient,
                this->numberDerivatives(featureBag));
            TLoopBodyVec remap(numberThreads, [this](std::size_t i) {
                for (auto& derivatives : m_Derivatives[i]) {
                    derivatives.remapCurvature();
                }
            });
            core::parallel_for_each(featureBag.begin(), featureBag.end(), remap);
        }

        //! Get the memory used by this object.
        std::size_t memoryUsage() const;

        //! Estimate the split derivatives' memory usage for a data frame with
        //! \p numberCols columns using \p numberSplitsPerFeature for a loss
        //! function with \p dimensionGradient parameters.
        static std::size_t estimateMemoryUsage(std::size_t numberFeatures,
                                               std::size_t numberSplitsPerFeature,
                                               std::size_t dimensionGradient);

        //! Get a checksum of this object.
        std::uint64_t checksum(std::uint64_t seed = 0) const;

        //! Get the number of loss function parameters.
        std::size_t dimensionGradient() const { return m_DimensionGradient; }

    private:
        using TDerivativesVecVec = std::vector<TDerivativesVec>;
        using TAlignedDoubleVec = std::vector<double, core::CAlignedAllocator<double>>;
        using TDerivatives2x1 = Eigen::Matrix<double, 2, 1>;

    private:
        void initializeAndMapStorage(const TFloatVecVec& splits) {
            std::size_t numberFeatures{splits.size()};
            std::size_t totalNumberSplits{
                std::accumulate(splits.begin(), splits.end(), std::size_t{0},
                                [](std::size_t size, const auto& featureSplits) {
                                    return size + numberSplits(featureSplits);
                                })};
            std::size_t splitsSize{this->splitSize()};
            m_Derivatives.resize(numberFeatures);
            m_Storage.resize((totalNumberSplits + numberFeatures) * splitsSize, 0.0);
            this->mapStorage(splits);
        }

        void copyAndMapStorage(const TAlignedDoubleVec storage,
                               const TDerivativesVecVec& derivatives) {
            m_Derivatives.resize(derivatives.size());
            m_Storage.assign(storage.begin(), storage.end());
            this->mapStorage(derivatives);
        }

        template<typename SPLITS>
        void mapStorage(const SPLITS& splits) {
            // This function maps the memory in a single presized buffer containing
            // enough space to store all gradient vectors and curvatures. For each
            // feature the layout in this buffer is as follows:
            //
            // "split grad" "split hessian" "count"       "missing grad" "missing hessian" "count"
            //       |            |            |                |              |              |
            //       V            V            V                V              V              V
            // |     n     |      n^2      |   1   | ... |      n       |      n^2       |    1    |
            //
            // Note we ensure 16 byte alignment because we're using aligned memory
            // mapped vectors which have better performance.

            auto dimensionGradient = static_cast<int>(m_DimensionGradient);
            std::size_t numberFeatures{splits.size()};
            std::size_t numberGradients{this->numberGradients()};
            std::size_t countOffset{this->countOffset()};
            std::size_t splitSize{this->splitSize()};
            double* storage{m_Storage.data()};
            for (std::size_t i = 0; i < numberFeatures; ++i, storage += splitSize) {
                std::size_t size{numberSplits(splits[i])};
                m_Derivatives[i].reserve(size);
                for (std::size_t j = 0; j < size; ++j, storage += splitSize) {
                    m_Derivatives[i].emplace_back(dimensionGradient, storage,
                                                  storage + numberGradients,
                                                  storage + countOffset);
                }
            }
        }

        static std::size_t numberSplits(const TDerivativesVec& derivatives) {
            return derivatives.size();
        }

        static std::size_t numberSplits(const TFloatVec& splits) {
            return splits.size() + 2;
        }

        std::size_t numberDerivatives() const {
            return this->numberGradients() + this->numberCurvatures();
        }

        std::size_t numberGradients() const {
            return core::CAlignment::roundup<double>(core::CAlignment::E_Aligned16,
                                                     m_DimensionGradient);
        }

        std::size_t numberCurvatures() const {
            return core::CAlignment::roundup<double>(
                core::CAlignment::E_Aligned16, m_DimensionGradient * m_DimensionGradient);
        }

        std::size_t curvaturesPad() const {
            return this->numberCurvatures() - m_DimensionGradient * m_DimensionGradient;
        }

        std::size_t countOffset() const {
            return this->numberDerivatives() - this->curvaturesPad();
        }

        std::size_t splitSize() const {
            return this->numberDerivatives() +
                   (this->curvaturesPad() > 0
                        ? 0
                        : core::CAlignment::roundup<double>(core::CAlignment::E_Aligned16, 1));
        }

        bool conformable(const CSplitsDerivatives& other) const {
            return other.m_Derivatives.size() == m_Derivatives.size() &&
                   std::equal(other.m_Derivatives.begin(),
                              other.m_Derivatives.end(), m_Derivatives.begin(),
                              [](const auto& lhs, const auto& rhs) {
                                  return lhs.size() == rhs.size();
                              });
        }

    private:
        std::size_t m_DimensionGradient{0};
        TDerivativesVecVec m_Derivatives;
        TAlignedDoubleVec m_Storage;
        TDerivatives2x1 m_PositiveDerivativesSum{TDerivatives2x1::Zero()};
        TDerivatives2x1 m_NegativeDerivativesSum{TDerivatives2x1::Zero()};
        double m_PositiveDerivativesMax{-INF};
        double m_PositiveDerivativesMin{INF};
        TDerivatives2x1 m_NegativeDerivativesMin{INF, INF};
    };

    //! \brief The derivatives and row masks objects to use for computations.
    //!
    //! DESCRIPTION:\n
    //! These are heavyweight objects and get passed in to minimise the number of
    //! times they need to be allocated. This has the added advantage of keeping
    //! the cache warm since the critical path is always working on the derivatives
    //! objects stored in this class.
    class MATHS_ANALYTICS_EXPORT CWorkspace {
    public:
        using TPackedBitVectorVec = std::vector<core::CPackedBitVector>;
        using TSplitsDerivativesVec = std::vector<CSplitsDerivatives>;
        using TNodeVec = std::vector<CBoostedTreeNode>;

    public:
        explicit CWorkspace(std::size_t dimensionGradient)
            : m_DimensionGradient{dimensionGradient} {}
        CWorkspace(CWorkspace&&) = default;
        CWorkspace& operator=(const CWorkspace& other) = delete;
        CWorkspace& operator=(CWorkspace&&) = default;

        //! Get the number of threads available.
        std::size_t numberThreads() const { return m_Derivatives.size(); }

        //! Get a list of features which must be included in training.
        TSizeVec featuresToInclude() const;

        //! Define the tree to retrain.
        void retraining(const TNodeVec& tree) { m_TreeToRetrain = &tree; }

        //! Re-initialize the masks and derivatives.
        void reinitialize(std::size_t numberThreads, const TFloatVecVec& candidateSplits) {
            m_MinimumGain = 0.0;
            m_Masks.resize(numberThreads);
            m_Derivatives.reserve(numberThreads);
            for (auto& mask : m_Masks) {
                mask.clear();
            }
            for (auto& derivatives : m_Derivatives) {
                derivatives.reinitialize(candidateSplits, m_DimensionGradient);
            }
            for (std::size_t j = m_Derivatives.size(); j < numberThreads; ++j) {
                m_Derivatives.emplace_back(candidateSplits, m_DimensionGradient);
            }
        }

        //! Get the tree being retrained if there is one.
        const TNodeVec* retraining() const { return m_TreeToRetrain; }

        //! Get the minimum leaf gain which will generate a split.
        double minimumGain() const { return m_MinimumGain; }

        //! Update the minimum gain to be at least \p gain.
        void minimumGain(double gain) {
            m_MinimumGain = std::max(m_MinimumGain, gain);
        }

        //! Reset the minimum gain to its initial value.
        void resetMinimumGain() { m_MinimumGain = 0.0; }

        //! Start working on a new leaf.
        void newLeaf(std::size_t numberToReduce) {
            m_NumberToReduce = numberToReduce;
            m_ReducedMasks = false;
            m_ReducedDerivatives = false;
        }

        //! Get the reduction of the per thread aggregate derivatives.
        CSplitsDerivatives& reducedDerivatives(const TSizeVec& featureBag) {
            if (m_ReducedDerivatives == false) {
                for (std::size_t i = 1; i < m_NumberToReduce; ++i) {
                    m_Derivatives[0].add(this->numberThreads(), m_Derivatives[i], featureBag);
                }
                m_Derivatives[0].remapCurvature(this->numberThreads(), featureBag);
                m_ReducedDerivatives = true;
            }
            return m_Derivatives[0];
        }

        //! Get the reduction of the per thread masks.
        const core::CPackedBitVector& reducedMask(std::size_t size) {
            if (m_ReducedMasks == false) {
                m_Masks[0].extend(false, size - m_Masks[0].size());
                for (std::size_t i = 1; i < m_NumberToReduce; ++i) {
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

        //! Capture the derivatives object to use later.
        void recycle(CSplitsDerivatives derivatives) {
            m_FreeDerivativesPool.emplace_back(std::move(derivatives));
        }

        //! Create or recycle a derivatives object and initialize with \p other.
        CSplitsDerivatives copy(const CSplitsDerivatives& other) {
            if (m_FreeDerivativesPool.empty()) {
                return CSplitsDerivatives{other};
            }
            auto result = std::move(m_FreeDerivativesPool.back());
            m_FreeDerivativesPool.pop_back();
            result.copy(other);
            return result;
        }

        //! Get the memory used by this object.
        std::size_t memoryUsage() const;

    private:
        using TSplitsDerivativesList = std::list<CSplitsDerivatives>;

    private:
        const TNodeVec* m_TreeToRetrain{nullptr};
        std::size_t m_DimensionGradient{1};
        std::size_t m_NumberToReduce{0};
        double m_MinimumGain{0.0};
        bool m_ReducedMasks{false};
        bool m_ReducedDerivatives{false};
        TPackedBitVectorVec m_Masks;
        TSplitsDerivativesVec m_Derivatives;
        TSplitsDerivativesList m_FreeDerivativesPool;
    };

public:
    static constexpr double INF{CBoostedTreeHyperparameters::INF};

public:
    virtual ~CBoostedTreeLeafNodeStatistics() = default;

    CBoostedTreeLeafNodeStatistics(const CBoostedTreeLeafNodeStatistics&) = delete;
    CBoostedTreeLeafNodeStatistics& operator=(const CBoostedTreeLeafNodeStatistics&) = delete;

    // Move construction/assignment not possible due to const reference member.

    //! Apply the split defined by \p split.
    //!
    //! \return Shared pointers to the left and right child node statistics.
    virtual TPtrPtrPr split(std::size_t leftChildId,
                            std::size_t rightChildId,
                            double gainThreshold,
                            const core::CDataFrame& frame,
                            const TRegularization& regularization,
                            const TSizeVec& treeFeatureBag,
                            const TSizeVec& nodeFeatureBag,
                            const CBoostedTreeNode& split,
                            CWorkspace& workspace) = 0;

    //! Order two leaves by decreasing gain in splitting them.
    bool operator<(const CBoostedTreeLeafNodeStatistics& rhs) const;

    //! Get the gain in loss of the best split of this leaf.
    double gain() const;

    //! Get the variance in gain we see for alternative split candidates.
    double gainVariance() const;

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
    const core::CPackedBitVector& rowMask() const;

    //! Get the row mask for this leaf node.
    core::CPackedBitVector& rowMask();

    //! Get the size of this object.
    virtual std::size_t staticSize() const = 0;

    //! Get the memory used by this object.
    std::size_t memoryUsage() const;

    //! Estimate the maximum leaf statistics' memory usage training on a data frame
    //! with \p numberFeatures columns using \p numberSplitsPerFeature for a loss function
    //! with \p dimensionGradient parameters.
    static std::size_t estimateMemoryUsage(std::size_t numberFeatures,
                                           std::size_t numberSplitsPerFeature,
                                           std::size_t dimensionGradient);

    //! Get the best split info as a string.
    std::string print() const;

protected:
    using TFeatureBestSplitSearch = std::function<void(std::size_t)>;

    //! \brief Statistics relating to a split of the node.
    struct SSplitStatistics : private boost::less_than_comparable<SSplitStatistics> {
        SSplitStatistics() = default;
        SSplitStatistics(double gain,
                         double curvature,
                         std::size_t feature,
                         double splitAt,
                         std::size_t minimumChildRowCount,
                         bool leftChildHasFewerRows,
                         bool assignMissingToLeft)
            : SSplitStatistics{gain, 0.0, curvature, feature, splitAt, minimumChildRowCount, leftChildHasFewerRows, assignMissingToLeft} {
        }
        SSplitStatistics(double gain,
                         double gainVariance,
                         double curvature,
                         std::size_t feature,
                         double splitAt,
                         std::size_t minimumChildRowCount,
                         bool leftChildHasFewerRows,
                         bool assignMissingToLeft)
            : s_Gain{common::CMathsFuncs::isNan(gain) ? -INF : gain},
              s_GainVariance{common::CMathsFuncs::isNan(gain) ? 0.0 : gainVariance},
              s_Curvature{curvature}, s_Feature{feature}, s_SplitAt{splitAt},
              s_MinimumChildRowCount{static_cast<std::uint32_t>(minimumChildRowCount)},
              s_LeftChildHasFewerRows{leftChildHasFewerRows}, s_AssignMissingToLeft{assignMissingToLeft} {
        }

        bool operator<(const SSplitStatistics& rhs) const {
            return common::COrderings::lexicographicalCompare(
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

        double s_Gain{-INF};
        double s_GainVariance{0.0};
        double s_Curvature{0.0};
        std::size_t s_Feature{std::numeric_limits<std::size_t>::max()};
        double s_SplitAt{INF};
        std::uint32_t s_MinimumChildRowCount{0};
        bool s_LeftChildHasFewerRows{true};
        bool s_AssignMissingToLeft{true};
        double s_LeftChildMaxGain{INF};
        double s_RightChildMaxGain{INF};
    };

    class CLookAheadBound {};
    class CNoLookAheadBound {};

protected:
    CBoostedTreeLeafNodeStatistics(std::size_t id,
                                   std::size_t depth,
                                   const TSizeVec& extraColumns,
                                   std::size_t dimensionGradient,
                                   const TFloatVecVec& candidateSplits,
                                   CSplitsDerivatives derivatives = CSplitsDerivatives{});

    void computeAggregateLossDerivatives(CLookAheadBound,
                                         std::size_t numberThreads,
                                         const core::CDataFrame& frame,
                                         const TSizeVec& featureBag,
                                         const core::CPackedBitVector& rowMask,
                                         CWorkspace& workspace) const;
    void computeAggregateLossDerivatives(CNoLookAheadBound,
                                         std::size_t numberThreads,
                                         const core::CDataFrame& frame,
                                         const TSizeVec& featureBag,
                                         const core::CPackedBitVector& rowMask,
                                         CWorkspace& workspace) const;
    void computeRowMaskAndAggregateLossDerivatives(CLookAheadBound,
                                                   std::size_t numberThreads,
                                                   const core::CDataFrame& frame,
                                                   bool isLeftChild,
                                                   const CBoostedTreeNode& split,
                                                   const TSizeVec& featureBag,
                                                   const core::CPackedBitVector& parentRowMask,
                                                   CWorkspace& workspace) const;
    void computeRowMaskAndAggregateLossDerivatives(CNoLookAheadBound,
                                                   std::size_t numberThreads,
                                                   const core::CDataFrame& frame,
                                                   bool isLeftChild,
                                                   const CBoostedTreeNode& split,
                                                   const TSizeVec& featureBag,
                                                   const core::CPackedBitVector& parentRowMask,
                                                   CWorkspace& workspace) const;

    SSplitStatistics& bestSplitStatistics();
    CSplitsDerivatives& derivatives();
    const CSplitsDerivatives& derivatives() const;
    std::size_t depth() const;
    const TSizeVec& extraColumns() const;
    std::size_t dimensionGradient() const;
    const TFloatVecVec& candidateSplits() const;

private:
    using TRowRef = core::CDataFrame::TRowRef;

private:
    template<typename BOUND>
    void computeAggregateLossDerivativesWith(BOUND bound,
                                             std::size_t numberThreads,
                                             const core::CDataFrame& frame,
                                             const TSizeVec& featureBag,
                                             const core::CPackedBitVector& rowMask,
                                             CWorkspace& workspace) const;
    template<typename BOUND>
    void computeRowMaskAndAggregateLossDerivativesWith(BOUND bound,
                                                       std::size_t numberThreads,
                                                       const core::CDataFrame& frame,
                                                       bool isLeftChild,
                                                       const CBoostedTreeNode& split,
                                                       const TSizeVec& featureBag,
                                                       const core::CPackedBitVector& parentRowMask,
                                                       CWorkspace& workspace) const;
    void addRowDerivatives(CLookAheadBound,
                           const TSizeVec& featureBag,
                           const TRowRef& row,
                           CSplitsDerivatives& splitsDerivatives) const;
    void addRowDerivatives(CNoLookAheadBound,
                           const TSizeVec& featureBag,
                           const TRowRef& row,
                           CSplitsDerivatives& splitsDerivatives) const;

private:
    std::size_t m_Id;
    std::size_t m_Depth;
    std::size_t m_DimensionGradient;
    const TSizeVec& m_ExtraColumns;
    const TFloatVecVec& m_CandidateSplits;
    CSplitsDerivatives m_Derivatives;
    core::CPackedBitVector m_RowMask;
    SSplitStatistics m_BestSplit;

    friend struct CBoostedTreeLeafNodeStatisticsTest::testComputeBestSplitStatisticsThreading;
};
}
}
}

#endif // INCLUDED_ml_maths_analytics_CBoostedTreeLeafNodeStatistics_h
