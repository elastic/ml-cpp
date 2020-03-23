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
#include <iterator>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace ml {
namespace maths {

//! \brief Computes k-means of a set of points online using \f$O(k)\f$ memory.
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
template<typename POINT, typename STORAGE_POINT = typename SFloatingPoint<POINT, CFloatStorage>::Type>
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
    using TStoragePoint = STORAGE_POINT;
    using TStorageCoordinate = typename SCoordinate<TStoragePoint>::Type;
    using TStoragePointDoublePr = std::pair<TStoragePoint, double>;
    using TStoragePointDoublePrVec = std::vector<TStoragePointDoublePr>;
    using TStoragePointMeanAccumulator =
        typename CBasicStatistics::SSampleMean<TStoragePoint>::TAccumulator;
    using TStoragePointMeanAccumulatorDoublePr =
        std::pair<TStoragePointMeanAccumulator, double>;
    using TStoragePointMeanAccumulatorDoublePrVec =
        std::vector<TStoragePointMeanAccumulatorDoublePr>;
    using TDoublePointMeanAccumulator =
        typename CBasicStatistics::SSampleMean<TDoublePoint>::TAccumulator;
    using TDoublePointMeanVarAccumulator =
        typename CBasicStatistics::SSampleMeanVar<TDoublePoint>::TAccumulator;

public:
    //! The minimum permitted size for the clusterer.
    static constexpr std::size_t MINIMUM_SPACE = 4;

    //! The maximum allowed size of the points buffer.
    static constexpr std::size_t BUFFER_SIZE = 6;

    //! The number of times to seed the clustering in reduce.
    static constexpr std::size_t NUMBER_SEEDS = 5;

    //! The maximum number of iterations to use for k-means in reduce.
    static constexpr std::size_t MAX_ITERATIONS = 10;

    static const core::TPersistenceTag K_TAG;
    static const core::TPersistenceTag BUFFER_SIZE_TAG;
    static const core::TPersistenceTag NUMBER_SEEDS_TAG;
    static const core::TPersistenceTag MAX_ITERATIONS_TAG;
    static const core::TPersistenceTag CLUSTERS_TAG;
    static const core::TPersistenceTag POINTS_TAG;
    static const core::TPersistenceTag RNG_TAG;

public:
    //! \param[in] k The numbers of clusters to maintain. A cluster comprises
    //! one float point vector, one count and a double holding the spherical
    //! variance.
    //! \param[in] decayRate The rate to age old data out of the clusters.
    //! \param[in] minClusterSize The minimum permitted number of points in
    //! a cluster.
    //! \param[in] bufferSize The number of points to buffer before reclustering.
    //! \param[in] numberSeeds The number of seeds to use when reclustering.
    //! \param[in] maxIterations The maximum number of iterations to use when
    //! reclustering.
    CKMeansOnline(std::size_t k,
                  double decayRate = 0.0,
                  double minClusterSize = MINIMUM_CATEGORY_COUNT,
                  std::size_t bufferSize = BUFFER_SIZE,
                  std::size_t numberSeeds = NUMBER_SEEDS,
                  std::size_t maxIterations = MAX_ITERATIONS)
        : m_K{std::max(k, MINIMUM_SPACE)}, m_BufferSize{bufferSize}, m_NumberSeeds{numberSeeds},
          m_MaxIterations{maxIterations}, m_DecayRate{decayRate}, m_MinClusterSize{minClusterSize} {
        m_Clusters.reserve(m_K + m_BufferSize + 1);
    }

    //! Construct with \p clusters.
    CKMeansOnline(std::size_t k,
                  double decayRate,
                  double minClusterSize,
                  TStoragePointMeanAccumulatorDoublePrVec& clusters)
        : CKMeansOnline{k, decayRate, minClusterSize} {
        m_Clusters.swap(clusters);
        m_Clusters.reserve(m_K + m_BufferSize + 1);
    }

    //! Create from part of a state document.
    bool acceptRestoreTraverser(const SDistributionRestoreParams& params,
                                core::CStateRestoreTraverser& traverser) {
        m_DecayRate = params.s_DecayRate;
        m_MinClusterSize = params.s_MinimumCategoryCount;
        TStoragePointDoublePrVec points;

        do {
            const std::string& name{traverser.name()};
            RESTORE(RNG_TAG, m_Rng.fromString(traverser.value()))
            RESTORE(K_TAG, core::CPersistUtils::restore(K_TAG, m_K, traverser))
            RESTORE(BUFFER_SIZE_TAG,
                    core::CPersistUtils::restore(BUFFER_SIZE_TAG, m_BufferSize, traverser))
            RESTORE(NUMBER_SEEDS_TAG,
                    core::CPersistUtils::restore(NUMBER_SEEDS_TAG, m_NumberSeeds, traverser))
            RESTORE(MAX_ITERATIONS_TAG,
                    core::CPersistUtils::restore(MAX_ITERATIONS_TAG, m_MaxIterations, traverser))
            RESTORE(K_TAG, core::CPersistUtils::restore(K_TAG, m_K, traverser))
            RESTORE(CLUSTERS_TAG,
                    core::CPersistUtils::restore(CLUSTERS_TAG, m_Clusters, traverser))
            RESTORE(POINTS_TAG, core::CPersistUtils::restore(POINTS_TAG, points, traverser))
        } while (traverser.next());

        for (const auto& point : points) {
            m_Clusters.emplace_back(
                CBasicStatistics::momentsAccumulator(point.second, point.first), 0.0);
        }

        return true;
    }

    //! Persist state by passing to the supplied inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const {
        inserter.insertValue(RNG_TAG, m_Rng.toString());
        core::CPersistUtils::persist(K_TAG, m_K, inserter);
        core::CPersistUtils::persist(BUFFER_SIZE_TAG, m_BufferSize, inserter);
        core::CPersistUtils::persist(NUMBER_SEEDS_TAG, m_NumberSeeds, inserter);
        core::CPersistUtils::persist(MAX_ITERATIONS_TAG, m_MaxIterations, inserter);
        core::CPersistUtils::persist(CLUSTERS_TAG, m_Clusters, inserter);
    }

    //! Efficiently swap the contents of this and \p other.
    void swap(CKMeansOnline& other) {
        std::swap(m_Rng, other.m_Rng);
        std::swap(m_K, other.m_K);
        std::swap(m_BufferSize, other.m_BufferSize);
        std::swap(m_NumberSeeds, other.m_NumberSeeds);
        std::swap(m_MaxIterations, other.m_MaxIterations);
        std::swap(m_DecayRate, other.m_DecayRate);
        std::swap(m_MinClusterSize, other.m_MinClusterSize);
        m_Clusters.swap(other.m_Clusters);
    }

    //! Get the total number of clusters.
    std::size_t size() const { return std::min(m_Clusters.size(), m_K); }

    //! Get the clusters being maintained.
    void clusters(TSphericalClusterVec& result) const {
        const_cast<CKMeansOnline*>(this)->clusters(result, std::false_type{});
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

        return kmeans(m_Rng, std::move(clusters), k, result, m_NumberSeeds, m_MaxIterations);
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
    static bool kmeans(RNG& rng,
                       TSphericalClusterVec clusters,
                       std::size_t k,
                       TSphericalClusterVecVec& result,
                       std::size_t numberSeeds = NUMBER_SEEDS,
                       std::size_t maxIterations = MAX_ITERATIONS) {
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
        kmeans.setPoints(std::move(clusters));

        CBasicStatistics::SMin<double>::TAccumulator minCost;
        TSphericalClusterVec centres;
        TSphericalClusterVecVec candidates;
        for (std::size_t i = 0; i < numberSeeds; ++i) {
            CKMeansPlusPlusInitialization<TSphericalCluster, RNG> seedCentres(rng);
            seedCentres.run(kmeans.beginPoints(), kmeans.endPoints(), k, centres);
            kmeans.setCentres(centres);
            kmeans.run(maxIterations);
            kmeans.clusters(candidates);
            candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                            [](TSphericalClusterVec& cluster) {
                                                return cluster.empty();
                                            }),
                             candidates.end());
            CSphericalGaussianInfoCriterion<TSphericalCluster, E_BIC> criterion;
            criterion.add(candidates);
            double cost{criterion.calculate()};
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
        if (!this->checkSplit(split)) {
            return false;
        }

        result.reserve(split.size());
        TStoragePointMeanAccumulatorDoublePrVec clusters;
        for (std::size_t i = 0u; i < split.size(); ++i) {
            clusters.clear();
            clusters.reserve(split[i].size());
            for (std::size_t j = 0u; j < split[i].size(); ++j) {
                clusters.push_back(m_Clusters[split[i][j]]);
            }
            result.emplace_back(m_K, m_DecayRate, m_MinClusterSize, clusters);
        }

        return true;
    }

    //! Add \p x to the clusterer.
    //!
    //! \param[in] x A point to add to the clusterer.
    //! \param[in] count The count weight of this point.
    void add(const TDoublePoint& x, double count = 1.0) {
        m_Clusters.emplace_back(CBasicStatistics::momentsAccumulator(count, x), 0.0);
        if (m_Clusters.size() > m_K + m_BufferSize) {
            this->reduce();
        }
    }

    //! Merge \p other with this clusterer.
    //!
    //! \param[in] other Another clusterer to merge with this one.
    void merge(const CKMeansOnline& other) {
        LOG_TRACE(<< "Merge");

        m_Clusters.insert(m_Clusters.end(), other.m_Clusters.begin(),
                          other.m_Clusters.end());

        this->reduce();

        // Reclaim memory from the vector buffer.
        TStoragePointMeanAccumulatorDoublePrVec categories(m_Clusters);
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
                                        [this](const auto& cluster) {
                                            return CBasicStatistics::count(
                                                       cluster.first) < m_MinClusterSize;
                                        }),
                         m_Clusters.end());

        LOG_TRACE(<< "clusters = " << core::CContainerPrinter::print(m_Clusters));
    }

    //! Check if there are points in the buffer.
    bool buffering() const { return m_Clusters.size() > m_K; }

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
        return CChecksum::calculate(seed, m_Clusters);
    }

    //! Get the memory used by this component
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
        mem->setName("CKMeansOnline");
        core::CMemoryDebug::dynamicSize("m_Clusters", m_Clusters, mem);
    }

    //! Get the memory used by this component
    std::size_t memoryUsage() const {
        return core::CMemory::dynamicSize(m_Clusters);
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
        deduplicate(m_Clusters);

        if (m_Clusters.size() < m_K) {
            return;
        }

        LOG_TRACE(<< "clusters = " << core::CContainerPrinter::print(m_Clusters));
        LOG_TRACE(<< "# clusters = " << m_Clusters.size());

        TSphericalClusterVec oldClusters;
        this->clusters(oldClusters, std::true_type{});

        TDoublePointMeanVarAccumulator empty{las::zero(oldClusters[0])};

        TSphericalClusterVecVec newClusters;
        kmeans(m_Rng, std::move(oldClusters), m_K, newClusters, m_NumberSeeds, m_MaxIterations);

        m_Clusters.resize(newClusters.size());

        TDoublePointMeanVarAccumulator centroid;
        for (std::size_t i = 0; i < newClusters.size(); ++i) {
            centroid = empty;
            for (const auto& point : newClusters[i]) {
                centroid.add(point);
            }
            double n{CBasicStatistics::count(centroid)};
            TDoublePoint& m{CBasicStatistics::moment<0>(centroid)};
            CBasicStatistics::count(m_Clusters[i].first) = n;
            CBasicStatistics::moment<0>(m_Clusters[i].first) = std::move(m);
            m_Clusters[i].second = variance(centroid);
        }

        LOG_TRACE(<< "reduced clusters = " << core::CContainerPrinter::print(m_Clusters));
        LOG_TRACE(<< "# reduced clusters = " << m_Clusters.size());
    }

    //! Remove any duplicates in \p points.
    //!
    //! \note We assume \p points is small so the bruteforce approach is fast.
    static void deduplicate(TStoragePointMeanAccumulatorDoublePrVec& clusters) {
        if (clusters.size() > 1) {
            std::stable_sort(clusters.begin(), clusters.end(),
                             [](const auto& lhs, const auto& rhs) {
                                 return CBasicStatistics::mean(lhs.first) <
                                        CBasicStatistics::mean(rhs.first);
                             });
            auto back = clusters.begin();
            for (auto i = back + 1; i != clusters.end(); ++i) {
                if (CBasicStatistics::mean(back->first) == CBasicStatistics::mean(i->first)) {
                    back->first += i->first;
                    double n[]{CBasicStatistics::count(back->first),
                               CBasicStatistics::count(i->first)};
                    back->second = (n[0] * back->second + n[1] * i->second) /
                                   (n[0] + n[1]);
                } else if (++back != i) {
                    *back = std::move(*i);
                }
            }
            clusters.erase(back + 1, clusters.end());
        }
    }

    //! Get the clusters being maintained optionally moving into \p result.
    template<typename MOVE>
    void clusters(TSphericalClusterVec& result, MOVE move) {
        result.clear();
        result.reserve(m_Clusters.size());
        bool moved{false};
        for (std::size_t i = 0; i < m_Clusters.size(); ++i) {
            TStoragePoint& m{CBasicStatistics::moment<0>(m_Clusters[i].first)};
            double n{CBasicStatistics::count(m_Clusters[i].first)};
            double v{m_Clusters[i].second};
            moved |= append(m, n, v, result, move);
        }
        if (moved) {
            m_Clusters.clear();
        }
    }

    //! Move append \p m into \p result.
    static bool append(POINT& m, double n, double v, TSphericalClusterVec& result, std::true_type) {
        result.emplace_back(std::move(m), SCountAndVariance(n, v));
        return true;
    }
    //! Copy append \p m into \p result.
    template<typename OTHER_POINT>
    static bool
    append(OTHER_POINT& m, double n, double v, TSphericalClusterVec& result, std::true_type) {
        result.emplace_back(m, SCountAndVariance(n, v));
        return false;
    }
    //! Copy append \p m into \p result.
    template<typename OTHER_POINT>
    static bool
    append(OTHER_POINT& m, double n, double v, TSphericalClusterVec& result, std::false_type) {
        result.emplace_back(m, SCountAndVariance(n, v));
        return false;
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

    //! The number of points to buffer before clustering.
    std::size_t m_BufferSize;

    //! The number of seeds to use when reclustering.
    std::size_t m_NumberSeeds;

    //! The number of iterations of k-means to use when reclustering.
    std::size_t m_MaxIterations;

    //! The rate at which the categories lose information.
    double m_DecayRate;

    //! The minimum permitted number of points in a cluster.
    double m_MinClusterSize;

    //! The clusters we are maintaining.
    TStoragePointMeanAccumulatorDoublePrVec m_Clusters;
};

template<typename POINT, typename STORAGE_POINT>
const core::TPersistenceTag CKMeansOnline<POINT, STORAGE_POINT>::K_TAG("a", "k");
template<typename POINT, typename STORAGE_POINT>
const core::TPersistenceTag CKMeansOnline<POINT, STORAGE_POINT>::CLUSTERS_TAG("b", "clusters");
template<typename POINT, typename STORAGE_POINT>
const core::TPersistenceTag CKMeansOnline<POINT, STORAGE_POINT>::POINTS_TAG("c", "points");
template<typename POINT, typename STORAGE_POINT>
const core::TPersistenceTag CKMeansOnline<POINT, STORAGE_POINT>::RNG_TAG("d", "rng");
template<typename POINT, typename STORAGE_POINT>
const core::TPersistenceTag
    CKMeansOnline<POINT, STORAGE_POINT>::BUFFER_SIZE_TAG("e", "buffer_size");
template<typename POINT, typename STORAGE_POINT>
const core::TPersistenceTag
    CKMeansOnline<POINT, STORAGE_POINT>::NUMBER_SEEDS_TAG("f", "number_seeds");
template<typename POINT, typename STORAGE_POINT>
const core::TPersistenceTag
    CKMeansOnline<POINT, STORAGE_POINT>::MAX_ITERATIONS_TAG("g", "max_iterations");
}
}

#endif // INCLUDED_ml_maths_CKMeansOnline_h
