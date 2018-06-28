/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CClusterEvaluation_h
#define INCLUDED_ml_maths_CClusterEvaluation_h

#include <core/CLogger.h>

#include <maths/CBasicStatistics.h>
#include <maths/CLinearAlgebraShims.h>
#include <maths/CPRNG.h>
#include <maths/CSampling.h>

#include <boost/ref.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

namespace ml {
namespace maths {

//! \brief Defines a collection of statistics for evaluating clustering.
class CClusterEvaluation {
private:
    using TDoubleVec = std::vector<double>;
    using TDoubleVecVec = std::vector<TDoubleVec>;
    using TMinAccumulator = CBasicStatistics::SMin<double>::TAccumulator;
    template<typename POINT>
    using TPointCRef = boost::reference_wrapper<const POINT>;
    template<typename POINT>
    using TPointCRefVec = std::vector<TPointCRef<POINT>>;
    template<typename POINT>
    using TPointCRefVecVec = std::vector<TPointCRefVec<POINT>>;
    template<typename POINT>
    using TPointCRefVecVecVec = std::vector<TPointCRefVecVec<POINT>>;

public:
    //! \brief Computes the dissimilarity of a point to a cluster.
    //!
    //! This is defined as \f$\frac{1}{|C|}\sum_{i\in C}{\|y_i - x\|}\f$.
    template<typename POINT>
    class CClusterDissimilarity {
    public:
        using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

    public:
        CClusterDissimilarity(const POINT& x) : m_X{&x} {}

        //! Update and return \p min with the dissimilarity of \p cluster
        //! to the point.
        template<typename OTHER_POINT>
        TMinAccumulator operator()(TMinAccumulator min,
                                   const std::vector<OTHER_POINT>& cluster) const {
            if (!cluster.empty()) {
                min.add(this->operator()(cluster));
            }
            return min;
        }

        //! Compute the dissimilarity of \p cluster and the point.
        template<typename OTHER_POINT>
        double operator()(const std::vector<OTHER_POINT>& cluster) const {
            TMeanAccumulator dissimilarity;
            for (const auto& y_ : cluster) {
                const auto& y = boost::unwrap_ref(y_);
                if (m_X != &y) {
                    dissimilarity.add(las::distance(*m_X, y));
                }
            }
            return CBasicStatistics::mean(dissimilarity);
        }

    private:
        const POINT* m_X;
    };

public:
    //! Compute Silhouette statistics.
    //!
    //! For details see https://en.wikipedia.org/wiki/Silhouette_(clustering).
    //!
    //! \param[in] clusters The clusters for which to compute the Silhouette
    //! statistics.
    //! \param[out] result Filled in with the statistics.
    template<typename POINT>
    static void silhouetteExact(const std::vector<std::vector<POINT>>& clusters,
                                TDoubleVecVec& result) {
        result.clear();
        result.reserve(clusters.size());
        for (auto cluster = clusters.begin(); cluster != clusters.end(); ++cluster) {
            result.emplace_back(cluster->size());
            auto s = result.back().begin();
            for (const auto& point : *cluster) {
                CClusterDissimilarity<POINT> dissimilarity(point);
                double a{dissimilarity(*cluster)};
                TMinAccumulator b_;
                b_ = std::accumulate(clusters.begin(), cluster, b_, dissimilarity);
                b_ = std::accumulate(cluster + 1, clusters.end(), b_, dissimilarity);
                double b{b_.count() > 0 ? b_[0] : a};
                *s++ = (b - a) / std::max(a, b);
            }
        }
    }

    //! Compute approximate Silhouette statistics if the number of points is large.
    //!
    //! See silhouetteExact for a discussion on the Silhouette statistics.
    //! Rather the the \f$O(n^2)\f$ complexity in the number of points
    //! to compute the exact Silhouette statistics, this computes approximate
    //! statistics (together with an estimate of the uncertainty) with
    //! complexity \f$O(n^(3/2))\f$.
    //!
    //! \param[in] clusters The clusters for which to compute the Silhouette
    //! statistics.
    //! \param[out] means Filled in with our best estimates of the statistics.
    //! \param[out] variances Filled in with an estimate of the error variance
    //! in \p means.
    template<typename POINT>
    static void silhouetteApprox(const std::vector<std::vector<POINT>>& clusters,
                                 TDoubleVecVec& means,
                                 TDoubleVecVec& variances) {
        if (numberPoints(clusters) < 1000) {
            silhouetteExact(clusters, means);
            variances.reserve(clusters.size());
            std::for_each(clusters.begin(), clusters.end(),
                          [&variances](const std::vector<POINT>& cluster) {
                              variances.emplace_back(cluster.size(), 0.0);
                          });
        } else {
            using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
            using TMeanVarAccumulatorVec = std::vector<TMeanVarAccumulator>;

            means.clear();
            variances.clear();
            means.reserve(clusters.size());
            variances.reserve(clusters.size());

            TMeanVarAccumulatorVec moments;
            moments.reserve(largest(clusters));

            TPointCRefVecVecVec<POINT> bags(10);
            generateBags(clusters, bags);

            for (std::size_t i = 0u; i != clusters.size(); ++i) {
                std::size_t ni{clusters[i].size()};

                moments.assign(ni, TMeanVarAccumulator());
                for (const auto& bag : bags) {
                    auto m = moments.begin();
                    for (const auto& point : clusters[i]) {
                        CClusterDissimilarity<POINT> dissimilarity(point);
                        double a{dissimilarity(*(bag.begin() + i))};
                        TMinAccumulator b_;
                        b_ = std::accumulate(bag.begin(), bag.begin() + i, b_, dissimilarity);
                        b_ = std::accumulate(bag.begin() + i + 1, bag.end(), b_, dissimilarity);
                        double b{b_.count() > 0 ? b_[0] : a};
                        (m++)->add((b - a) / std::max(a, b));
                    }
                }

                means.emplace_back(ni);
                variances.emplace_back(ni);
                for (std::size_t j = 0; j < moments.size(); ++j) {
                    means.back()[j] = CBasicStatistics::mean(moments[j]);
                    variances.back()[j] = 0.1 * CBasicStatistics::variance(moments[j]);
                }
            }
        }
    }

private:
    //! Generate the bags of samples of \p clusters to use for the
    //! approximate Silhouette statistics.
    //!
    //! \param[in] clusters The clusters to sample.
    //! \param[in,out] result Should be pre-sized with the number
    //! of samplings required.
    template<typename POINT>
    static void generateBags(const std::vector<std::vector<POINT>>& clusters,
                             TPointCRefVecVecVec<POINT>& result) {
        using TSizeVec = std::vector<std::size_t>;

        if (result.empty()) {
            LOG_ERROR("Asked for no bags");
            return;
        }

        std::size_t n{numberPoints(clusters)};
        LOG_TRACE("# points = " << n);
        TDoubleVec weights;
        weights.reserve(clusters.size());
        std::for_each(clusters.begin(), clusters.end(),
                      [&weights, n](const std::vector<POINT>& cluster) {
                          weights.push_back(static_cast<double>(cluster.size()) /
                                            static_cast<double>(n));
                      });
        TSizeVec samples;
        CSampling::weightedSample(numberSamples(clusters), weights, samples);
        LOG_TRACE("samples = " << core::CContainerPrinter::print(samples));
        TDoubleVec uniform;
        uniform.reserve(largest(clusters));

        CPRNG::CXorOShiro128Plus rng;
        for (auto&& bag : result) {
            bag.resize(clusters.size());
            for (std::size_t i = 0u; i < clusters.size(); ++i) {
                std::size_t ni{clusters[i].size()};
                LOG_TRACE("# cluster points = " << ni);
                if (ni > 0) {
                    bag[i].reserve(ni);
                    uniform.assign(ni, 1.0 / static_cast<double>(ni));
                    TSizeVec indices;
                    CSampling::categoricalSampleWithoutReplacement(
                        rng, uniform, std::max(samples[i], std::size_t(1)), indices);
                    for (auto j : indices) {
                        bag[i].emplace_back(clusters[i][j]);
                    }
                }
            }
        }
    }

    //! Get the number of points in the largest cluster.
    template<typename POINT>
    static std::size_t largest(const std::vector<std::vector<POINT>>& clusters) {
        std::size_t result{0u};
        std::for_each(clusters.begin(), clusters.end(),
                      [&result](const std::vector<POINT>& cluster) {
                          result = std::max(result, cluster.size());
                      });
        return result;
    }

    //! Get the number of points in all clusters.
    template<typename POINT>
    static std::size_t numberPoints(const std::vector<std::vector<POINT>>& clusters) {
        return std::accumulate(clusters.begin(), clusters.end(), 0,
                               [](std::size_t n, const std::vector<POINT>& cluster) {
                                   return n + cluster.size();
                               });
    }

    //! Compute the number of samples to use for the approximate
    //! Silhouette statistics.
    template<typename POINT>
    static std::size_t numberSamples(const std::vector<std::vector<POINT>>& clusters) {
        double n_{std::max(static_cast<double>(numberPoints(clusters)), 1000.0)};
        std::size_t n{static_cast<std::size_t>(n_ / (10.0 + std::sqrt(n_ - 1000.0)) + 0.5)};
        return std::max(n, 10 * clusters.size());
    }
};
}
}

#endif // INCLUDED_ml_maths_CClusterEvaluation_h
