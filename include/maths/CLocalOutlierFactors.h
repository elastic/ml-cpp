/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CLocalOutlierFactors_h
#define INCLUDED_ml_maths_CLocalOutlierFactors_h

#include <core/CHashing.h>

#include <maths/CBasicStatistics.h>
#include <maths/CGramSchmidt.h>
#include <maths/CKdTree.h>
#include <maths/CLinearAlgebraShims.h>
#include <maths/CPRNG.h>
#include <maths/CSampling.h>
#include <maths/CTools.h>
#include <maths/ImportExport.h>

#include <boost/math/distributions/normal.hpp>
#include <boost/make_shared.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace ml
{
namespace maths
{

//! \brief Utilities for computing outlier scores for collections
//! of points.
class MATHS_EXPORT CLocalOutlierFactors
{
    private:
        template<typename POINT>
        using TKdTree = CKdTree<CAnnotatedVector<POINT, std::size_t>>;

    public:
        using TDoubleVec = std::vector<double>;
        using TDoubleVecVec = std::vector<TDoubleVec>;

        //! The available algorithm.
        enum EAlgorithm
        {
            E_Lof,
            E_Ldof,
            E_DistancekNN,
            E_TotalDistancekNN
        };

    public:
        //! Compute the normalized LOF scores for \p points.
        //!
        //! \note This automatically chooses to project the data and
        //! the number of nearest neighbours to use based on the data
        //! characteristics.
        template<typename POINT>
        static void normalizedLof(const std::vector<POINT> &points, TDoubleVec &scores)
        {
            normalizedLof(numberNeighbours(E_Lof, points), project(points), points, scores);
        }

        //! Compute the normalized LOF scores for \p points.
        //!
        //! See https://en.wikipedia.org/wiki/Local_outlier_factor
        //! for details.
        //!
        //! \param[in] k The number of nearest neighbours to use.
        //! \param[in] project If true, compute geometric average scores
        //! over multiple random projections of the data.
        //! \param[in] points The points for which to compute scores.
        //! \param[out] scores The scores of \p points.
        template<typename POINT>
        static void normalizedLof(std::size_t k, bool project,
                                  const std::vector<POINT> &points,
                                  TDoubleVec &scores)
        {
            CLof<POINT, TKdTree<POINT>> lof(k);
            compute(lof, project, points, scores);
        }

        //! Compute the normalized local distance based outlier scores
        //! for \p points.
        //!
        //! \note This automatically chooses to project the data and
        //! the number of nearest neighbours to use based on the data
        //! characteristics.
        template<typename POINT>
        static void normalizedLdof(const std::vector<POINT> &points, TDoubleVec &scores)
        {
            normalizedLdof(numberNeighbours(E_Ldof, points), project(points), points, scores);
        }

        //! Compute normalized local distance based outlier scores
        //! for \p points.
        //!
        //! See https://arxiv.org/pdf/0903.3257.pdf for details.
        //!
        //! \param[in] k The number of nearest neighbours to use.
        //! \param[in] project If true compute geometric average scores
        //! over multiple random projections of the data.
        //! \param[in] points The points for which to compute scores.
        //! \param[out] scores The scores of \p points.
        template<typename POINT>
        static void normalizedLdof(std::size_t k, bool project,
                                   const std::vector<POINT> &points,
                                   TDoubleVec &scores)
        {
            CLdof<POINT, TKdTree<POINT>> ldof(k);
            compute(ldof, project, points, scores);
        }

        //! Compute the normalized distance to the k-th nearest neighbour
        //! outlier scores for \p points.
        //!
        //! \note This automatically chooses to project the data and
        //! the number of nearest neighbours to use based on the data
        //! characteristics.
        template<typename POINT>
        static void normalizedDistancekNN(const std::vector<POINT> &points,
                                          TDoubleVec &scores)
        {
            normalizedDistancekNN(numberNeighbours(E_DistancekNN, points),
                                  project(points), points, scores);
        }

        //! Compute the normalized distance to the k-th nearest neighbour
        //! outlier scores for \p points.
        //!
        //! \param[in] k The number of nearest neighbours to use.
        //! \param[in] project If true compute geometric average scores
        //! over multiple random projections of the data.
        //! \param[in] points The points for which to compute scores.
        //! \param[out] scores The scores of \p points.
        template<typename POINT>
        static void normalizedDistancekNN(std::size_t k, bool project,
                                          const std::vector<POINT> &points,
                                          TDoubleVec &scores)
        {
            CDistancekNN<POINT, TKdTree<POINT>> knn(k);
            compute(knn, project, points, scores);
        }

        //! Compute the normalized mean distance to the k nearest
        //! neighbours outlier scores for \p points.
        //!
        //! \note This automatically chooses to project the data and
        //! the number of nearest neighbours to use based on the data
        //! characteristics.
        template<typename POINT>
        static void normalizedTotalDistancekNN(const std::vector<POINT> &points,
                                               TDoubleVec &scores)
        {
            normalizedTotalDistancekNN(numberNeighbours(E_TotalDistancekNN, points),
                                       project(points), points, scores);
        }

        //! Compute the normalized mean distance to the k nearest
        //! neighbours outlier scores for \p points.
        //!
        //! \param[in] k The number of nearest neighbours to use.
        //! \param[in] project If true compute geometric average scores
        //! over multiple random projections of the data.
        //! \param[in] points The points for which to compute scores.
        //! \param[out] scores The scores of \p points.
        template<typename POINT>
        static void normalizedTotalDistancekNN(std::size_t k, bool project,
                                               const std::vector<POINT> &points,
                                               TDoubleVec &scores)
        {
            CTotalDistancekNN< POINT, TKdTree<POINT>> knn(k);
            compute(knn, project, points, scores);
        }

        //! Compute the outlier scores using a mean ensemble of approaches.
        //!
        //! \note This automatically chooses to project the data and
        //! the number of nearest neighbours to use based on the data
        //! characteristics.
        template<typename POINT>
        static void ensemble(const std::vector<POINT> &points, TDoubleVec &scores)
        {
            CEnsemble<POINT, TKdTree<POINT>> method;
            method.lof(points)
                  .ldof(points)
                  .knn(points)
                  .tnn(points);
            compute(method, project(points), points, scores);
        }

    protected:
        using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
        using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
        using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;

    protected:
        //! Check whether to project the data.
        template<typename POINT>
        static bool project(const std::vector<POINT> &points)
        {
            return (!points.empty() ? las::dimension(points[0]) : 1) > 4;
        }

        //! Get the number of nearest neighbours to use.
        template<typename POINT>
        static std::size_t numberNeighbours(EAlgorithm algorithm,
                                            const std::vector<POINT> &points)
        {
            double d{std::max(std::sqrt(static_cast<double>(
                                 !points.empty() ? las::dimension(points[0]) : 1)), 4.0)};
            double n{static_cast<double>(points.size())};
            double result;
            switch (algorithm)
            {
            case E_Lof:
                result = std::max(std::min(5.0 * d, 20.0), std::pow(n, 1.0 / 3.0)); break;
            case E_Ldof:
                result = std::max(std::min(5.0 * d, 30.0), std::pow(n, 1.0 / 3.0)); break;
            case E_DistancekNN:
            case E_TotalDistancekNN:
                result = std::max(std::min(d, 10.0), std::pow(n, 1.0 / 3.0)); break;
            }
            return static_cast<std::size_t>(result + 0.5);
        }

        //! Compute the outlier score using \p compute.
        template<typename COMPUTE, typename POINT>
        static void compute(COMPUTE &compute,
                            bool project,
                            const std::vector<POINT> &points,
                            TDoubleVec &scores)
        {
            using TPoint = CAnnotatedVector<POINT, std::size_t>;
            using TPointVec = std::vector<TPoint>;

            scores.assign(points.size(), 0.0);

            if (!points.empty())
            {
                if (!project)
                {
                    TPointVec points_;
                    points_.reserve(points.size());
                    for (const auto &point : points)
                    {
                        points_.emplace_back(point, points_.size());
                    }
                    compute(std::move(points_), scores);
                }
                else
                {
                    std::size_t dimension{las::dimension(points[0])};
                    std::size_t bags, projectedDimension;
                    computeBagsAndProjectedDimension(points, bags, projectedDimension);

                    // Use standard 2-stable Gaussian projections since we are
                    // interested in Euclidean distances.
                    TDoubleVec coordinates;
                    CPRNG::CXorOShiro128Plus rng;
                    CSampling::normalSample(rng, 0.0, 1.0,
                                            bags * projectedDimension * dimension,
                                            coordinates);

                    TPointVec projected_;
                    projected_.reserve(points.size());
                    std::vector<POINT> projection(projectedDimension, las::zero(points[0]));
                    TMeanAccumulatorVec scores_(points.size());
                    for (std::size_t i = 0u; i < coordinates.size(); /**/)
                    {
                        // Create an orthonormal basis for the bag.
                        for (std::size_t p = 0u; p < projectedDimension; ++p)
                        {
                            for (std::size_t d = 0u; d < dimension; ++i, ++d)
                            {
                                projection[p][d] = coordinates[i];
                            }
                        }
                        CGramSchmidt::basis(projection);

                        // Project onto the basis.
                        projected_.clear();
                        for (const auto &point : points)
                        {
                            TPoint projected(SConstant<POINT>::get(projectedDimension, 0),
                                             projected_.size());
                            for (std::size_t p = 0u; p < projection.size(); ++p)
                            {
                                projected(p) = las::inner(projection[p], point);
                            }
                            projected_.push_back(std::move(projected));
                        }

                        // Compute the scores and update the overall score.
                        compute(std::move(projected_), scores);
                        for (std::size_t j = 0u; j < scores.size(); ++j)
                        {
                            scores_[j].add(std::log(scores[j]));
                        }
                    }

                    for (std::size_t i = 0u; i < scores_.size(); ++i)
                    {
                        scores[i] = std::exp(CBasicStatistics::mean(scores_[i]));
                    }
                }
            }
        }

        //! Compute the number of bags and the projection dimension.
        template<typename POINT>
        static void computeBagsAndProjectedDimension(const std::vector<POINT> &points,
                                                     std::size_t &bags,
                                                     std::size_t &projectedDimension)
        {
            // Note because we get big wins in the lookup speed of
            // the k-d tree we still get a big win if the product
            // "number of projections" x "projection dimension" is
            // O(dimension) of the full space.

            std::size_t dimension{las::dimension(points[0])};
            double rootd{std::sqrt(std::min(static_cast<double>(dimension), 100.0))};
            double logn{std::log(static_cast<double>(points.size()))};
            projectedDimension = static_cast<std::size_t>(
                    std::max(std::min(rootd, logn), 4.0) + 0.5);
            bags = std::max(dimension / 2 / projectedDimension, std::size_t(2));
        }

        //! \brief The interface for an outlier calculation method.
        template<typename POINT, typename NEAREST_NEIGHBOURS>
        class CMethod
        {
            public:
                using TPoint = CAnnotatedVector<POINT, std::size_t>;
                using TPointVec = std::vector<TPoint>;

            public:
                CMethod(NEAREST_NEIGHBOURS lookup) : m_Lookup(lookup) {}
                virtual ~CMethod(void) = default;

                void run(std::size_t k, TPointVec &&points, TDoubleVec &scores)
                {
                    this->setup(points);
                    m_Lookup.build(points);
                    for (const auto &point : m_Lookup)
                    {
                        m_Lookup.nearestNeighbours(k+1, point, m_Neighbours);
                        this->add(point, m_Neighbours, scores);
                    }
                    this->compute(scores);
                }

                const NEAREST_NEIGHBOURS &lookup(void) const
                {
                    return m_Lookup;
                }

                virtual void setup(const TPointVec &/*points*/) {}

                virtual void add(const TPoint &point,
                                 const TPointVec &neighbours,
                                 TDoubleVec &scores) = 0;

                virtual void compute(TDoubleVec &scores)
                {
                    normalize(scores);
                }

            private:
                NEAREST_NEIGHBOURS m_Lookup;
                TPointVec m_Neighbours;
        };

        //! \brief Computes the normalized version of the local
        //! outlier factor score.
        template<typename POINT, typename NEAREST_NEIGHBOURS>
        class CLof final : public CMethod<POINT, NEAREST_NEIGHBOURS>
        {
            public:
                using TPoint = CAnnotatedVector<POINT, std::size_t>;
                using TPointVec = std::vector<TPoint>;
                using TSizeDoublePr = std::pair<std::size_t, double>;
                using TSizeDoublePrVec = std::vector<TSizeDoublePr>;
                using TSizeDoublePrVecVec = std::vector<TSizeDoublePrVec>;

            public:
                CLof(std::size_t k, NEAREST_NEIGHBOURS lookup = NEAREST_NEIGHBOURS()) :
                        CMethod<POINT, NEAREST_NEIGHBOURS>(lookup), m_K(k)
                {}

                void operator()(TPointVec &&points, TDoubleVec &scores)
                {
                    this->run(m_K, std::move(points), scores);
                }

                virtual void setup(const TPointVec &points)
                {
                    m_KDistances.assign(points.size(), TSizeDoublePrVec());
                    m_Lrd.assign(points.size(), 0.0);
                }

                virtual void add(const TPoint &point,
                                 const TPointVec &neighbours,
                                 TDoubleVec &/*scores*/)
                {
                    std::size_t i{point.annotation()};
                    m_KDistances[i].reserve(m_K);
                    for (std::size_t j = 1u; j <= m_K; ++j)
                    {
                        m_KDistances[i].emplace_back(neighbours[j].annotation(),
                                                     las::distance(point, neighbours[j]));
                    }
                }

                virtual void compute(TDoubleVec &scores)
                {
                    using TMinAccumulator = CBasicStatistics::SMin<double>::TAccumulator;

                    TMinAccumulator min;
                    for (const auto &point : this->lookup())
                    {
                        std::size_t i{point.annotation()};
                        TMeanAccumulator reachability_;
                        for (const auto &neighbour : m_KDistances[i])
                        {
                            reachability_.add(std::max(kdistance(m_KDistances[index(neighbour)]),
                                                       distance(neighbour)));
                        }
                        double reachability{CBasicStatistics::mean(reachability_)};
                        if (reachability > 0.0)
                        {
                            m_Lrd[i] = 1.0 / reachability;
                            min.add(reachability);
                        }
                        else
                        {
                            m_Lrd[i] = -1.0;
                        }
                    }
                    if (min.count() > 0)
                    {
                        for (auto &&lrd : m_Lrd)
                        {
                            if (lrd < 0.0)
                            {
                                lrd = min[0] / 2.0;
                            }
                        }
                        for (const auto &point : this->lookup())
                        {
                            std::size_t i{point.annotation()};
                            TMeanAccumulator score;
                            for (const auto &neighbour : m_KDistances[i])
                            {
                                score.add(m_Lrd[index(neighbour)]);
                            }
                            scores[i] = CBasicStatistics::mean(score) / m_Lrd[i];
                        }
                    }
                    normalize(scores);
                }

            private:
                static std::size_t index(const TSizeDoublePr &neighbour)
                {
                    return neighbour.first;
                }
                static double distance(const TSizeDoublePr &neighbour)
                {
                    return neighbour.second;
                }
                static double kdistance(const TSizeDoublePrVec &neighbours)
                {
                    return distance(neighbours.back());
                }

            private:
                std::size_t m_K;
                TSizeDoublePrVecVec m_KDistances;
                TDoubleVec m_Lrd;
        };

        //! \brief Computes the normalized version of the local
        //! distance based outlier score.
        template<typename POINT, typename NEAREST_NEIGHBOURS>
        class CLdof final : public CMethod<POINT, NEAREST_NEIGHBOURS>
        {
            public:
                using TPoint = CAnnotatedVector<POINT, std::size_t>;
                using TPointVec = std::vector<TPoint>;

            public:
                CLdof(std::size_t k, NEAREST_NEIGHBOURS lookup = NEAREST_NEIGHBOURS()) :
                        CMethod<POINT, NEAREST_NEIGHBOURS>(lookup), m_K(k)
                {}

                void operator()(TPointVec &&points, TDoubleVec &scores)
                {
                    this->run(m_K, std::move(points), scores);
                }

                virtual void add(const TPoint &point,
                                 const TPointVec &neighbours,
                                 TDoubleVec &scores)
                {
                    TMeanAccumulator d, D;
                    for (std::size_t i = 1u; i < neighbours.size(); ++i)
                    {
                        d.add(las::distance(point, neighbours[i]));
                        for (std::size_t j = 1u; j < i; ++j)
                        {
                            D.add(las::distance(neighbours[i], neighbours[j]));
                        }
                    }
                    scores[point.annotation()] =
                            CBasicStatistics::mean(D) > 0.0 ?
                            CBasicStatistics::mean(d) / CBasicStatistics::mean(D) : 0.0;
                }

            private:
                std::size_t m_K;
        };

        //! \brief Computes the normalized version of the distance
        //! to the k'th nearest neighbour score.
        template<typename POINT, typename NEAREST_NEIGHBOURS>
        class CDistancekNN final : public CMethod<POINT, NEAREST_NEIGHBOURS>
        {
            public:
                using TPoint = CAnnotatedVector<POINT, std::size_t>;
                using TPointVec = std::vector<TPoint>;

            public:
                CDistancekNN(std::size_t k, NEAREST_NEIGHBOURS lookup = NEAREST_NEIGHBOURS()) :
                        CMethod<POINT, NEAREST_NEIGHBOURS>(lookup), m_K(k)
                {}

                void operator()(TPointVec &&points, TDoubleVec &scores)
                {
                    this->run(m_K, std::move(points), scores);
                }

                virtual void add(const TPoint &point,
                                 const TPointVec &neighbours,
                                 TDoubleVec &scores)
                {
                    scores[point.annotation()] = las::distance(point, neighbours.back());
                }

            private:
                std::size_t m_K;
        };

        //! \brief Computes the normalized version of the total
        //! distance to the k nearest neighbours.
        template<typename POINT, typename NEAREST_NEIGHBOURS>
        class CTotalDistancekNN final : public CMethod<POINT, NEAREST_NEIGHBOURS>
        {
            public:
                using TPoint = CAnnotatedVector<POINT, std::size_t>;
                using TPointVec = std::vector<TPoint>;

            public:
                CTotalDistancekNN(std::size_t k, NEAREST_NEIGHBOURS lookup = NEAREST_NEIGHBOURS()) :
                        CMethod<POINT, NEAREST_NEIGHBOURS>(lookup), m_K(k)
                {}

                void operator()(TPointVec &&points, TDoubleVec &scores)
                {
                    this->run(m_K, std::move(points), scores);
                }

                virtual void add(const TPoint &point,
                                 const TPointVec &neighbours,
                                 TDoubleVec &scores)
                {
                    std::size_t i{point.annotation()};
                    for (std::size_t j = 1u; j < neighbours.size(); ++j)
                    {
                        scores[i] += las::distance(point, neighbours[j]);
                    }
                    scores[i] /= static_cast<double>(neighbours.size());
                }

            private:
                std::size_t m_K;
        };

        //! \brief Computes a mean ensemble of the other methods.
        template<typename POINT, typename NEAREST_NEIGHBOURS>
        class CEnsemble final : public CMethod<POINT, NEAREST_NEIGHBOURS>
        {
            public:
                using TPoint = CAnnotatedVector<POINT, std::size_t>;
                using TPointVec = std::vector<TPoint>;
                using TMethodPtr = boost::shared_ptr<CMethod<POINT, const NEAREST_NEIGHBOURS&>>;
                using TMethodPtrVec = std::vector<TMethodPtr>;

           public:
                CEnsemble(NEAREST_NEIGHBOURS lookup = NEAREST_NEIGHBOURS()) :
                        CMethod<POINT, NEAREST_NEIGHBOURS>(lookup), m_K(0)
                {}

                CEnsemble &lof(const std::vector<POINT> &points)
                {
                    std::size_t k{numberNeighbours(E_Lof, points)};
                    m_Methods.push_back(boost::make_shared<
                            CLof<POINT, const NEAREST_NEIGHBOURS&>>(k, this->lookup()));
                    m_K = std::max(m_K, k);
                    return *this;
                }
                CEnsemble &ldof(const std::vector<POINT> &points)
                {
                    std::size_t k{numberNeighbours(E_Ldof, points)};
                    m_Methods.push_back(boost::make_shared<
                            CLdof<POINT, const NEAREST_NEIGHBOURS&>>(k, this->lookup()));
                    m_K = std::max(m_K, k);
                    return *this;
                }
                CEnsemble &knn(const std::vector<POINT> &points)
                {
                    std::size_t k{numberNeighbours(E_DistancekNN, points)};
                    m_Methods.push_back(boost::make_shared<
                            CDistancekNN<POINT, const NEAREST_NEIGHBOURS&>>(k, this->lookup()));
                    m_K = std::max(m_K, k);
                    return *this;
                }
                CEnsemble &tnn(const std::vector<POINT> &points)
                {
                    std::size_t k{numberNeighbours(E_TotalDistancekNN, points)};
                    m_Methods.push_back(boost::make_shared<
                            CTotalDistancekNN<POINT, const NEAREST_NEIGHBOURS&>>(k, this->lookup()));
                    m_K = std::max(m_K, k);
                    return *this;
                }

                void operator()(TPointVec &&points, TDoubleVec &scores)
                {
                    this->run(m_K, std::move(points), scores);
                }

                virtual void setup(const TPointVec &points)
                {
                    for (auto &&method : m_Methods)
                    {
                        method->setup(points);
                    }
                    m_Scores.assign(m_Methods.size(), TDoubleVec(points.size()));
                }

                virtual void add(const TPoint &point,
                                 const TPointVec &neighbours,
                                 TDoubleVec &/*scores*/)
                {
                    for (std::size_t i = 0u; i < m_Methods.size(); ++i)
                    {
                        m_Methods[i]->add(point, neighbours, m_Scores[i]);
                    }
                }

                virtual void compute(TDoubleVec &scores)
                {
                    for (std::size_t i = 0u; i < m_Methods.size(); ++i)
                    {
                        m_Methods[i]->compute(m_Scores[i]);
                    }
                    for (std::size_t i = 0u; i < scores.size(); ++i)
                    {
                        for (std::size_t j = 0u; j < m_Scores.size(); ++j)
                        {
                            scores[i] = m_Scores[j][i];
                        }
                        scores[i] /= static_cast<double>(m_Scores.size());
                    }
                }

           private:
                std::size_t m_K;
                TMethodPtrVec m_Methods;
                TDoubleVecVec m_Scores;
        };

        //! The Gaussian normalization scheme proposed in "Interpreting
        //! and Unifying Outlier Scores" by Kreigel et al.
        //!
        //! This simply fits a Gaussian to a collection of scores for an
        //! outlier method and computes the right tail probability.
        static void normalize(TDoubleVec &scores)
        {
            using TMaxAccumulator = CBasicStatistics::COrderStatisticsHeap<double, std::greater<double>>;

            TMaxAccumulator max(std::max(scores.size() / 10, std::size_t(1)));
            max.add(scores);
            double cutoff{max.biggest()};

            TMeanVarAccumulator moments;
            for (auto score : scores)
            {
                moments.add(std::min(score, cutoff));
            }

            try
            {
                double mean{CBasicStatistics::mean(moments)};
                double variance{CBasicStatistics::variance(moments)};
                if (variance > 0.0)
                {
                    boost::math::normal normal(mean, std::sqrt(variance));
                    for (auto &&score : scores)
                    {
                        score = CTools::safeCdfComplement(normal, score);
                    }
                }
                else
                {
                    scores.assign(scores.size(), 1.0);
                }
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Failed to normalise scores: " << e.what());
            }
        }
};

}
}

#endif // INCLUDED_ml_maths_CLocalOutlierFactors_h
