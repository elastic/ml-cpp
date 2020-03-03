/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CKMeansOnline_h
#define INCLUDED_ml_maths_CKMeansOnline_h

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/CPersistUtils.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <iterator>
#include <maths/CBasicStatistics.h>
#include <maths/CChecksum.h>
#include <maths/CInformationCriteria.h>
#include <maths/CKMeans.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraShims.h>
#include <maths/CLinearAlgebraTools.h>
#include <maths/CPRNG.h>
#include <maths/CRestoreParams.h>
#include <maths/CTypeTraits.h>
#include <maths/Constants.h>

#include <cmath>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

namespace ml {
namespace maths {

//! \brief Computes k-means of a set of points online using \f$O(k)\f$
//! memory.
//!
//! DESCRIPTION:\n
//! This is a sketch data structure of points and their spherical variance
//! which can then be clustered using the k-means objective (of minimizing
//! the within class variance). See also CNaturalBreaksClassifier for a
//! discussion of the sketch strategy.
//!
//! IMPLEMENTATION:\n
//! This class is templated on the point type for greater flexibility.
//! It must support addition, subtraction, have free functions for
//! coordinate-wise min and max, satisfy the constraints imposed by
//! CBasicStatistics::SSampleCentralMoments, support coordinate access
//! by the brackets operator and have member functions called dimension
//! and euclidean - which gives the Euclidean norm of the vector.
template<typename POINT>
class CKMeansOnline {
public:
    using TSizeVec = std::vector<std::size_t>;
    using TSizeVecVec = std::vector<TSizeVec>;
    using TDoublePoint = typename SFloatingPoint<POINT, double>::Type;
    using TDoublePointVec = std::vector<TDoublePoint>;
    using TSphericalCluster = typename CSphericalCluster<POINT>::Type;
    using TSphericalClusterVec = std::vector<TSphericalCluster>;
    using TSphericalClusterVecVec = std::vector<TSphericalClusterVec>;
    using TKMeansOnlineVec = std::vector<CKMeansOnline>;

protected:
    //! \brief Checks if a cluster should be deleted based on its count.
    class CShouldDelete {
    public:
        CShouldDelete(double minimumCategoryCount)
            : m_MinimumCategoryCount(minimumCategoryCount) {}

        template<typename CLUSTER>
        bool operator()(const CLUSTER& cluster) const {
            return CBasicStatistics::count(cluster.first) < m_MinimumCategoryCount;
        }

    private:
        double m_MinimumCategoryCount;
    };

    using TFloatPoint = typename SFloatingPoint<POINT, CFloatStorage>::Type;
    using TFloatCoordinate = typename SCoordinate<TFloatPoint>::Type;
    using TFloatPointDoublePr = std::pair<TFloatPoint, double>;
    using TFloatPointDoublePrVec = std::vector<TFloatPointDoublePr>;
    using TFloatPointMeanAccumulator =
        typename CBasicStatistics::SSampleMean<TFloatPoint>::TAccumulator;
    using TFloatPointMeanAccumulatorDoublePr = std::pair<TFloatPointMeanAccumulator, double>;
    using TFloatPointMeanAccumulatorDoublePrVec = std::vector<TFloatPointMeanAccumulatorDoublePr>;
    using TDoublePointMeanAccumulator =
        typename CBasicStatistics::SSampleMean<TDoublePoint>::TAccumulator;
    using TDoublePointMeanVarAccumulator =
        typename CBasicStatistics::SSampleMeanVar<TDoublePoint>::TAccumulator;

protected:
    //! The minimum permitted size for the clusterer.
    static const std::size_t MINIMUM_SPACE;

    //! The maximum allowed size of the points buffer.
    static const std::size_t MAXIMUM_BUFFER_SIZE;

    //! The number of times to seed the clustering in reduce.
    static const std::size_t NUMBER_SEEDS;

    //! The maximum number of iterations to use for k-means in reduce.
    static const std::size_t MAX_ITERATIONS;

    static const core::TPersistenceTag K_TAG;
    static const core::TPersistenceTag CLUSTERS_TAG;
    static const core::TPersistenceTag POINTS_TAG;
    static const core::TPersistenceTag RNG_TAG;

public:
    //! \param[in] k The maximum space in numbers of clusters.
    //! A cluster comprises one float point vector, one count and
    //! a double holding the spherical variance.
    //! \param[in] decayRate The rate at which we data ages out
    //! of the clusterer.
    //! \param[in] minimumCategoryCount The minimum permitted count
    //! for a cluster.
    //! \note This will store as much information about the points
    //! subject to this constraint so will generally hold \p k
    //! clusters.
    CKMeansOnline(std::size_t k, double decayRate = 0.0, double minimumCategoryCount = MINIMUM_CATEGORY_COUNT)
        : m_K(std::max(k, MINIMUM_SPACE)), m_DecayRate(decayRate),
          m_MinimumCategoryCount(minimumCategoryCount) {
        m_Clusters.reserve(m_K + MAXIMUM_BUFFER_SIZE + 1u);
        m_PointsBuffer.reserve(MAXIMUM_BUFFER_SIZE);
    }

    //! Construct a new classifier with the specified space limit
    //! \p space and categories \p categories.
    CKMeansOnline(std::size_t k,
                  double decayRate,
                  double minimumCategoryCount,
                  TFloatPointMeanAccumulatorDoublePrVec& clusters)
        : m_K(std::max(k, MINIMUM_SPACE)), m_DecayRate(decayRate),
          m_MinimumCategoryCount(minimumCategoryCount) {
        m_Clusters.swap(clusters);
        m_Clusters.reserve(m_K + MAXIMUM_BUFFER_SIZE + 1u);
        m_PointsBuffer.reserve(MAXIMUM_BUFFER_SIZE);
    }

    //! Create from part of a state document.
    bool acceptRestoreTraverser(const SDistributionRestoreParams& params,
                                core::CStateRestoreTraverser& traverser) {
        m_DecayRate = params.s_DecayRate;
        m_MinimumCategoryCount = params.s_MinimumCategoryCount;

        do {
            const std::string& name{traverser.name()};
            RESTORE(RNG_TAG, m_Rng.fromString(traverser.value()))
            RESTORE(K_TAG, core::CPersistUtils::restore(K_TAG, m_K, traverser))
            RESTORE(CLUSTERS_TAG,
                    core::CPersistUtils::restore(CLUSTERS_TAG, m_Clusters, traverser))
            RESTORE(POINTS_TAG, core::CPersistUtils::restore(POINTS_TAG, m_PointsBuffer, traverser))
        } while (traverser.next());
        return true;
    }

    //! Persist state by passing to the supplied inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const {
        inserter.insertValue(RNG_TAG, m_Rng.toString());
        core::CPersistUtils::persist(K_TAG, m_K, inserter);
        core::CPersistUtils::persist(CLUSTERS_TAG, m_Clusters, inserter);
        core::CPersistUtils::persist(POINTS_TAG, m_PointsBuffer, inserter);
    }

    //! Efficiently swap the contents of this and \p other.
    void swap(CKMeansOnline& other) {
        std::swap(m_Rng, other.m_Rng);
        std::swap(m_K, other.m_K);
        std::swap(m_DecayRate, other.m_DecayRate);
        std::swap(m_MinimumCategoryCount, other.m_MinimumCategoryCount);
        m_Clusters.swap(other.m_Clusters);
        m_PointsBuffer.swap(other.m_PointsBuffer);
    }

    //! Get the total number of clusters.
    std::size_t size() const {
        return std::min(m_Clusters.size() + m_PointsBuffer.size(), m_K);
    }

    //! Get the clusters being maintained.
    void clusters(TSphericalClusterVec& result) const {
        result.clear();
        result.reserve(m_Clusters.size());
        for (std::size_t i = 0u; i < m_Clusters.size(); ++i) {
            const TFloatPoint& m{CBasicStatistics::mean(m_Clusters[i].first)};
            double n{CBasicStatistics::count(m_Clusters[i].first)};
            double v{m_Clusters[i].second};
            result.emplace_back(m, SCountAndVariance(n, v));
        }
    }

    //! Get our best estimate of the \p k means clustering of the
    //! k-means maintained by this object.
    //!
    //! \param[in] k The desired size for the clustering.
    //! \param[out] result Filled in with the \p k means clustering.
    bool kmeans(std::size_t k, TSphericalClusterVecVec& result) {
        LOG_TRACE(<< "split");

        result.clear();

        if (k == 0) {
            LOG_ERROR(<< "Bad request for zero categories");
            return false;
        }

        this->reduce();
        LOG_TRACE(<< "raw clusters = " << this->print());

        TSphericalClusterVec clusters;
        this->clusters(clusters);

        return kmeans(m_Rng, clusters, k, result);
    }

    //! Get our best estimate of the \p k means clustering of
    //! \p clusters.
    //!
    //! \param[in] rng The random number generator.
    //! \param[in] clusters The spherical clusters to cluster.
    //! \param[in] k The desired size for the clustering.
    //! \param[out] result Filled in with the \p k means clustering
    //! of \p clusters.
    template<typename RNG>
    static bool
    kmeans(RNG& rng, TSphericalClusterVec& clusters, std::size_t k, TSphericalClusterVecVec& result) {
        result.clear();

        if (k == 0) {
            LOG_ERROR(<< "Bad request for zero categories");
            return false;
        }
        if (clusters.empty()) {
            return true;
        }

        result.reserve(std::min(k, clusters.size()));

        if (k >= clusters.size()) {
            TSphericalClusterVec cluster(1);
            for (std::size_t i = 0u; i < clusters.size(); ++i) {
                cluster[0] = clusters[i];
                result.push_back(cluster);
            }
            return true;
        } else if (k == 1) {
            result.push_back(clusters);
            return true;
        }

        CKMeans<TSphericalCluster> kmeans;
        kmeans.setPoints(clusters);
        CBasicStatistics::SMin<double>::TAccumulator minCost;
        TSphericalClusterVec centres;
        TSphericalClusterVecVec candidates;
        for (std::size_t i = 0u; i < NUMBER_SEEDS; ++i) {
            CKMeansPlusPlusInitialization<TSphericalCluster, RNG> seedCentres(rng);
            seedCentres.run(clusters, k, centres);
            kmeans.setCentres(centres);
            kmeans.run(MAX_ITERATIONS);
            kmeans.clusters(candidates);
            CSphericalGaussianInfoCriterion<TSphericalCluster, E_BIC> criterion;
            criterion.add(candidates);
            double cost = criterion.calculate();
            if (minCost.add(cost)) {
                result.swap(candidates);
            }
        }

        LOG_TRACE(<< "result = " << core::CContainerPrinter::print(result));

        return true;
    }

    //! Split this into n online k-means clusterers corresponding to
    //! \p split.
    //!
    //! \param[in] split The desired partition of the k clusters.
    //! \param[out] result Filled in with the clusterers representing
    //! \p split if it is a valid partition and cleared otherwise.
    bool split(const TSizeVecVec& split, TKMeansOnlineVec& result) {
        result.clear();
        this->reduce();
        if (!this->checkSplit(split)) {
            return false;
        }

        result.reserve(split.size());
        TFloatPointMeanAccumulatorDoublePrVec clusters;
        for (std::size_t i = 0u; i < split.size(); ++i) {
            clusters.clear();
            clusters.reserve(split[i].size());
            for (std::size_t j = 0u; j < split[i].size(); ++j) {
                clusters.push_back(m_Clusters[split[i][j]]);
            }
            result.emplace_back(m_K, m_DecayRate, m_MinimumCategoryCount, clusters);
        }

        return true;
    }

    //! Add \p x to the clusterer.
    //!
    //! \param[in] x A point to add to the clusterer.
    //! \param[in] count The count weight of this point.
    void add(const TDoublePoint& x, double count = 1.0) {
        if (m_PointsBuffer.size() < MAXIMUM_BUFFER_SIZE) {
            m_PointsBuffer.emplace_back(x, count);
        } else {
            m_Clusters.emplace_back(CBasicStatistics::momentsAccumulator(count, x), 0.0);
            this->reduce();
        }
    }

    //! Merge \p other with this clusterer.
    //!
    //! \param[in] other Another clusterer to merge with this one.
    void merge(const CKMeansOnline& other) {
        LOG_TRACE(<< "Merge");

        for (const auto& point : other.m_PointsBuffer) {
            m_Clusters.emplace_back(
                CBasicStatistics::momentsAccumulator(point.second, point.first), 0.0);
        }
        m_Clusters.insert(m_Clusters.end(), other.m_Clusters.begin(),
                          other.m_Clusters.end());

        this->reduce();

        // Reclaim memory from the vector buffer.
        TFloatPointMeanAccumulatorDoublePrVec categories(m_Clusters);
        m_Clusters.swap(categories);
    }

    //! Set the rate at which information is aged out.
    void decayRate(double decayRate) { m_DecayRate = decayRate; }

    //! Propagate the clusters forwards by \p time.
    void propagateForwardsByTime(double time) {
        if (time < 0.0) {
            LOG_ERROR(<< "Can't propagate backwards in time");
            return;
        }

        double alpha{std::exp(-m_DecayRate * time)};
        LOG_TRACE(<< "alpha = " << alpha);

        this->age(alpha);
    }

    //! Age by a factor \p alpha, which should be in the range (0, 1).
    void age(double alpha) {
        LOG_TRACE(<< "clusters = " << core::CContainerPrinter::print(m_Clusters));

        for (std::size_t i = 0u; i < m_Clusters.size(); ++i) {
            m_Clusters[i].first.age(alpha);
        }

        // Prune any dead categories: we're not interested in
        // maintaining categories with low counts.
        m_Clusters.erase(std::remove_if(m_Clusters.begin(), m_Clusters.end(),
                                        CShouldDelete(m_MinimumCategoryCount)),
                         m_Clusters.end());

        LOG_TRACE(<< "clusters = " << core::CContainerPrinter::print(m_Clusters));
    }

    //! Get the current points buffer.
    bool buffering() const { return m_PointsBuffer.size() > 0; }

    //! Get \p n samples of the distribution corresponding to the
    //! categories we are maintaining.
    //!
    //! \param[in] numberSamples The desired number of samples.
    //! \param[out] result Filled in with the samples of the distribution.
    void sample(std::size_t numberSamples, TDoublePointVec& result) const {
        result.clear();
        if (numberSamples == 0) {
            return;
        }

        using TDoubleVec = std::vector<double>;
        using TDoubleSizePr = std::pair<double, std::size_t>;

        // See, for example, Effective C++ item 3.
        const_cast<CKMeansOnline*>(this)->reduce();
        LOG_TRACE(<< "categories = " << core::CContainerPrinter::print(m_Clusters));

        TDoubleVec counts;
        counts.reserve(m_Clusters.size());
        double Z{0.0};
        for (std::size_t i = 0u; i < m_Clusters.size(); ++i) {
            double ni{CBasicStatistics::count(m_Clusters[i].first)};
            counts.push_back(ni);
            Z += ni;
        }
        Z /= static_cast<double>(numberSamples);
        for (std::size_t i = 0u; i < counts.size(); ++i) {
            counts[i] /= Z;
        }
        LOG_TRACE(<< "weights = " << core::CContainerPrinter::print(counts)
                  << ", Z = " << Z << ", n = " << numberSamples);

        result.reserve(2 * numberSamples);

        TDoubleVec weights;
        TDoublePointVec clusterSamples;
        for (std::size_t i = 0u; i < m_Clusters.size(); ++i) {
            double ni{counts[i]};

            clusterSamples.clear();
            TDoublePoint m{CBasicStatistics::mean(m_Clusters[i].first)};
            if (m_Clusters[i].second == 0.0) {
                clusterSamples.push_back(m);
            } else {
                std::size_t ni_{static_cast<std::size_t>(std::ceil(ni))};
                TDoublePoint v(m_Clusters[i].second);
                sampleGaussian(ni_, m, v.asDiagonal(), clusterSamples);
            }

            ni /= static_cast<double>(clusterSamples.size());

            result.insert(result.end(), clusterSamples.begin(), clusterSamples.end());
            weights.insert(weights.end(), clusterSamples.size(), ni);
        }
        LOG_TRACE(<< "samples = " << core::CContainerPrinter::print(result));
        LOG_TRACE(<< "weights = " << core::CContainerPrinter::print(weights));

        TDoublePointVec finalSamples;
        finalSamples.reserve(static_cast<std::size_t>(
            std::ceil(std::accumulate(weights.begin(), weights.end(), 0.0))));
        TDoublePointMeanAccumulator sample{las::zero(result[0])};
        for (;;) {
            CBasicStatistics::SMin<TDoubleSizePr>::TAccumulator nearest;
            const TDoublePoint& sample_{CBasicStatistics::mean(sample)};
            for (std::size_t j = 0u; j < result.size(); ++j) {
                if (weights[j] > 0.0) {
                    nearest.add({las::distance(result[j], sample_), j});
                }
            }
            if (nearest.count() == 0) {
                break;
            }

            std::size_t j{nearest[0].second};
            const TDoublePoint& xj{result[j]};
            do {
                double nj{std::min(1.0 - CBasicStatistics::count(sample), weights[j])};
                sample.add(xj, nj);
                weights[j] -= nj;
                if (CBasicStatistics::count(sample) > ALMOST_ONE) {
                    finalSamples.push_back(CBasicStatistics::mean(sample));
                    sample = TDoublePointMeanAccumulator{las::zero(result[0])};
                }
            } while (weights[j] > 0.0);
        }

        result = std::move(finalSamples);
        LOG_TRACE(<< "# samples = " << result.size());
        LOG_TRACE(<< "samples = " << core::CContainerPrinter::print(result));
    }

    //! Print this classifier for debug.
    std::string print() const {
        return core::CContainerPrinter::print(m_Clusters);
    }

    //! Get a checksum for this object.
    std::uint64_t checksum(std::uint64_t seed = 0) const {
        seed = CChecksum::calculate(seed, m_K);
        seed = CChecksum::calculate(seed, m_DecayRate);
        seed = CChecksum::calculate(seed, m_Clusters);
        return CChecksum::calculate(seed, m_PointsBuffer);
    }

    //! Get the memory used by this component
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
        mem->setName("CKMeansOnline");
        core::CMemoryDebug::dynamicSize("m_Clusters", m_Clusters, mem);
        core::CMemoryDebug::dynamicSize("m_PointsBuffer", m_PointsBuffer, mem);
    }

    //! Get the memory used by this component
    std::size_t memoryUsage() const {
        std::size_t mem = core::CMemory::dynamicSize(m_Clusters);
        mem += core::CMemory::dynamicSize(m_PointsBuffer);
        return mem;
    }

protected:
    //! Sanity check \p split.
    bool checkSplit(const TSizeVecVec& split) const {
        if (split.empty()) {
            LOG_ERROR(<< "Bad split = " << core::CContainerPrinter::print(split));
            return false;
        }
        for (std::size_t i = 0u; i < split.size(); ++i) {
            if (split[i].empty()) {
                LOG_ERROR(<< "Bad split = " << core::CContainerPrinter::print(split));
                return false;
            }
            for (std::size_t j = 0u; j < split[i].size(); ++j) {
                if (split[i][j] >= m_Clusters.size()) {
                    LOG_ERROR(<< "Bad split = " << core::CContainerPrinter::print(split));
                    return false;
                }
            }
        }
        return true;
    }

    //! Reduce the number of clusters to m_K by k-means clustering.
    void reduce() {
        // Add all the points as new spherical clusters and reduce.
        deduplicate(m_PointsBuffer);
        for (const auto& point : m_PointsBuffer) {
            m_Clusters.emplace_back(
                CBasicStatistics::momentsAccumulator(point.second, point.first), 0.0);
        }
        m_PointsBuffer.clear();

        if (m_Clusters.size() < m_K) {
            return;
        }

        LOG_TRACE(<< "clusters = " << core::CContainerPrinter::print(m_Clusters));
        LOG_TRACE(<< "# clusters = " << m_Clusters.size());

        TSphericalClusterVecVec kclusters;
        {
            TSphericalClusterVec clusters;
            this->clusters(clusters);
            kmeans(m_Rng, clusters, m_K, kclusters);
        }

        m_Clusters.resize(kclusters.size());
        for (std::size_t i = 0u; i < kclusters.size(); ++i) {
            TDoublePointMeanVarAccumulator cluster{las::zero(kclusters[i][0])};
            for (const auto& point : kclusters[i]) {
                cluster.add(point);
            }
            double n{CBasicStatistics::count(cluster)};
            const TDoublePoint& m{CBasicStatistics::mean(cluster)};
            m_Clusters[i].first = CBasicStatistics::momentsAccumulator(
                TFloatCoordinate(n), TFloatPoint(m));
            m_Clusters[i].second = variance(cluster);
        }

        LOG_TRACE(<< "reduced clusters = " << core::CContainerPrinter::print(m_Clusters));
        LOG_TRACE(<< "# reduced clusters = " << m_Clusters.size());
    }

    //! Remove any duplicates in \p points.
    //!
    //! \note We assume \p points is small so the bruteforce approach is fast.
    static void deduplicate(TFloatPointDoublePrVec& points) {
        TSizeVec remove;
        remove.reserve(points.size());
        for (std::size_t i = 0; i < points.size(); ++i) {
            for (std::size_t j = i + 1; j < points.size(); ++j) {
                if (points[i].first == points[j].first) {
                    points[i].second += points[j].second;
                    remove.push_back(j);
                }
            }
            if (remove.size() > 0) {
                remove.push_back(points.size());
                auto back = points.begin() + remove[0];
                auto end = back;
                for (std::size_t j = 1; j < remove.size(); ++j) {
                    auto start = end + 1;
                    end = points.begin() + remove[j];
                    for (/**/; start != end; ++start, ++back) {
                        *back = std::move(*start);
                    }
                }
                points.erase(back, points.end());
                remove.clear();
            }
        }
    }

    //! Get the spherically symmetric variance from \p moments.
    static double variance(const TDoublePointMeanVarAccumulator& moments) {
        const TDoublePoint& v{CBasicStatistics::maximumLikelihoodVariance(moments)};
        return las::L1(v) / static_cast<double>(las::dimension(v));
    }

private:
    static constexpr double ALMOST_ONE = 0.99999;

private:
    //! The random number generator.
    CPRNG::CXorOShiro128Plus m_Rng;

    //! The number of clusters to maintain.
    std::size_t m_K;

    //! The rate at which the categories lose information.
    double m_DecayRate;

    //! The minimum permitted count for a cluster.
    double m_MinimumCategoryCount;

    //! The clusters we are maintaining.
    TFloatPointMeanAccumulatorDoublePrVec m_Clusters;

    //! A buffer of the points added while the space constraint
    //! is satisfied.
    TFloatPointDoublePrVec m_PointsBuffer;
};

template<typename POINT>
const std::size_t CKMeansOnline<POINT>::MINIMUM_SPACE = 4u;
template<typename POINT>
const std::size_t CKMeansOnline<POINT>::MAXIMUM_BUFFER_SIZE = 6u;
template<typename POINT>
const std::size_t CKMeansOnline<POINT>::NUMBER_SEEDS = 5u;
template<typename POINT>
const std::size_t CKMeansOnline<POINT>::MAX_ITERATIONS = 10u;

template<typename POINT>
const core::TPersistenceTag CKMeansOnline<POINT>::K_TAG("a", "k");
template<typename POINT>
const core::TPersistenceTag CKMeansOnline<POINT>::CLUSTERS_TAG("b", "clusters");
template<typename POINT>
const core::TPersistenceTag CKMeansOnline<POINT>::POINTS_TAG("c", "points");
template<typename POINT>
const core::TPersistenceTag CKMeansOnline<POINT>::RNG_TAG("d", "rng");
}
}

#endif // INCLUDED_ml_maths_CKMeansOnline_h
