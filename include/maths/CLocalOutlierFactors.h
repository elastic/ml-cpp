/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CLocalOutlierFactors_h
#define INCLUDED_ml_maths_CLocalOutlierFactors_h

#include <core/CHashing.h>
#include <core/Concurrency.h>

#include <maths/CBasicStatistics.h>
#include <maths/CKdTree.h>
#include <maths/CLinearAlgebraShims.h>
#include <maths/COrthogonaliser.h>
#include <maths/CPRNG.h>
#include <maths/CSampling.h>
#include <maths/CTools.h>
#include <maths/ImportExport.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

namespace ml {
namespace core {
class CDataFrame;
}
namespace maths {

//! \brief Utilities for computing outlier scores for collections
//! of points.
class MATHS_EXPORT CLocalOutlierFactors {
private:
    template<typename POINT>
    using TKdTree = CKdTree<CAnnotatedVector<POINT, std::size_t>>;

public:
    using TDoubleVec = std::vector<double>;
    using TDoubleVecVec = std::vector<TDoubleVec>;

    //! The available algorithms.
    enum EAlgorithm {
        E_Lof,
        E_Ldof,
        E_DistancekNN,
        E_TotalDistancekNN,
        E_Ensemble
    };

public:
    //! Compute the normalized LOF scores for \p points.
    //!
    //! \note This automatically chooses to project the data and
    //! the number of nearest neighbours to use based on the data
    //! characteristics.
    template<typename POINT>
    static void normalizedLof(std::vector<POINT> points, TDoubleVec& scores) {
        std::size_t k{defaultNumberOfNeighbours(points)};
        bool project{shouldProject(points)};
        normalizedLof(k, project, std::move(points), scores);
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
    static void
    normalizedLof(std::size_t k, bool project, std::vector<POINT> points, TDoubleVec& scores) {
        normalized<CLof>(k, project, std::move(points), scores);
    }

    //! Compute the normalized local distance based outlier scores
    //! for \p points.
    //!
    //! \note This automatically chooses to project the data and
    //! the number of nearest neighbours to use based on the data
    //! characteristics.
    template<typename POINT>
    static void normalizedLdof(std::vector<POINT> points, TDoubleVec& scores) {
        std::size_t k{defaultNumberOfNeighbours(points)};
        bool project{shouldProject(points)};
        normalizedLdof(k, project, std::move(points), scores);
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
    static void
    normalizedLdof(std::size_t k, bool project, std::vector<POINT> points, TDoubleVec& scores) {
        normalized<CLdof>(k, project, std::move(points), scores);
    }

    //! Compute the normalized distance to the k-th nearest neighbour
    //! outlier scores for \p points.
    //!
    //! \note This automatically chooses to project the data and
    //! the number of nearest neighbours to use based on the data
    //! characteristics.
    template<typename POINT>
    static void normalizedDistancekNN(std::vector<POINT> points, TDoubleVec& scores) {
        std::size_t k{defaultNumberOfNeighbours(points)};
        bool project{shouldProject(points)};
        normalizedDistancekNN(k, project, std::move(points), scores);
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
    static void normalizedDistancekNN(std::size_t k,
                                      bool project,
                                      std::vector<POINT> points,
                                      TDoubleVec& scores) {
        normalized<CDistancekNN>(k, project, std::move(points), scores);
    }

    //! Compute the normalized mean distance to the k nearest
    //! neighbours outlier scores for \p points.
    //!
    //! \note This automatically chooses to project the data and
    //! the number of nearest neighbours to use based on the data
    //! characteristics.
    template<typename POINT>
    static void normalizedTotalDistancekNN(std::vector<POINT> points, TDoubleVec& scores) {
        std::size_t k{defaultNumberOfNeighbours(points)};
        bool project{shouldProject(points)};
        normalizedTotalDistancekNN(k, project, std::move(points), scores);
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
    static void normalizedTotalDistancekNN(std::size_t k,
                                           bool project,
                                           std::vector<POINT> points,
                                           TDoubleVec& scores) {
        normalized<CTotalDistancekNN>(k, project, std::move(points), scores);
    }

    //! Compute the outlier scores using a mean ensemble of approaches.
    //!
    //! \note This automatically chooses to project the data and
    //! the number of nearest neighbours to use based on the data
    //! characteristics.
    template<typename POINT>
    static void ensemble(std::vector<POINT> points, TDoubleVec& scores) {
        if (points.empty()) {
            // Nothing to do
        } else if (shouldProject(points)) {
            using TPoint = decltype(SConstant<POINT>::get(0, 0));
            CEnsemble<TPoint, TKdTree<TPoint>> ensemble;
            ensemble.lof(points).ldof(points).knn(points).tnn(points);
            computeForBagOfProjections(ensemble, points, scores);
        } else {
            CEnsemble<POINT, TKdTree<POINT>> ensemble;
            ensemble.lof(points).ldof(points).knn(points).tnn(points);
            compute(ensemble, annotate(std::move(points)), scores);
        }
    }

    //! Estimate the amount of memory that will be used computing outliers.
    //!
    //! \param[in] algorithm The algorithm that will be used.
    //! \param[in] k The number of nearest neighbours which will be used.
    //! \param[in] numberPoints The number of points for outliers will be
    //! computed.
    //! \param[in] dimension The dimension of the points for which outliers
    //! will be computed.
    template<typename POINT>
    static std::size_t estimateMemoryUsage(EAlgorithm algorithm,
                                           std::size_t k,
                                           std::size_t numberPoints,
                                           std::size_t dimension) {
        // TODO handle the case we'll project the data properly.
        std::size_t result{TKdTree<POINT>::estimateMemoryUsage(numberPoints, dimension)};
        switch (algorithm) {
        case E_Ensemble:
            result += CLof<POINT, TKdTree<POINT>>::estimateOwnMemoryOverhead(k, numberPoints) +
                      CEnsemble<POINT, TKdTree<POINT>>::estimateOwnMemoryOverhead(
                          numberPoints, 4);
            break;
        case E_Lof:
            result += CLof<POINT, TKdTree<POINT>>::estimateOwnMemoryOverhead(k, numberPoints);
            break;
        case E_Ldof:
        case E_DistancekNN:
        case E_TotalDistancekNN:
            break;
        }
        return result;
    }

    //! Overload of estimateMemoryUsage which computes estimated usage
    //! for the default method and number of nearest neighbours.
    template<typename POINT>
    static std::size_t
    estimateMemoryUsage(EAlgorithm algorithm, std::size_t numberPoints, std::size_t dimension) {
        return estimateMemoryUsage<POINT>(
            algorithm, defaultNumberOfNeighbours(numberPoints, dimension),
            numberPoints, dimension);
    }

protected:
    using TSizeSizePr = std::pair<std::size_t, std::size_t>;
    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;

protected:
    //! Compute normalised outlier scores for a specified method.
    template<template<typename, typename> class METHOD, typename POINT>
    static void
    normalized(std::size_t k, bool project, std::vector<POINT> points, TDoubleVec& scores) {
        if (points.empty()) {
            // Nothing to do
        } else if (project) {
            using TPoint = decltype(SConstant<POINT>::get(0, 0));
            METHOD<TPoint, TKdTree<TPoint>> method(k);
            computeForBagOfProjections(method, points, scores);
        } else {
            METHOD<POINT, TKdTree<POINT>> method(k);
            compute(method, annotate(std::move(points)), scores);
        }
    }

    //! Check whether to project the data.
    template<typename POINT>
    static bool shouldProject(const std::vector<POINT>& points) {
        return shouldProject(points.size() > 0 ? las::dimension(points[0]) : 1);
    }

    //! Check whether to project the data.
    static bool shouldProject(std::size_t dimension) { return dimension > 4; }

    //! Get the number of nearest neighbours to use.
    template<typename POINT>
    static std::size_t defaultNumberOfNeighbours(const std::vector<POINT>& points) {
        if (points.empty()) {
            return 1;
        }

        std::size_t numberPoints{points.size()};
        std::size_t dimension{las::dimension(points[0])};
        if (shouldProject(points)) {
            std::tie(std::ignore, dimension) = computeBagsAndProjectedDimension(points);
        }

        return defaultNumberOfNeighbours(numberPoints, dimension);
    }

    //! Get the number of nearest neighbours to use.
    static std::size_t defaultNumberOfNeighbours(std::size_t numberPoints,
                                                 std::size_t dimension) {
        return static_cast<std::size_t>(
            CTools::truncate(std::min(5.0 * static_cast<double>(dimension),
                                      std::pow(static_cast<double>(numberPoints), 1.0 / 3.0)),
                             10.0, 50.0) +
            0.5);
    }

    //! Create points annotated with their index in \p points.
    template<typename POINT>
    static std::vector<CAnnotatedVector<POINT, std::size_t>>
    annotate(std::vector<POINT> points) {
        std::vector<CAnnotatedVector<POINT, std::size_t>> annotatedPoints;
        annotatedPoints.reserve(points.size());
        for (std::size_t i = 0; i < points.size(); ++i) {
            annotatedPoints.emplace_back(std::move(points[i]), i);
        }
        return annotatedPoints;
    }

    //! Compute the outlier scores of \p points using \p compute.
    template<typename COMPUTE, typename POINT>
    static void compute(COMPUTE& compute,
                        std::vector<CAnnotatedVector<POINT, std::size_t>> points,
                        TDoubleVec& scores) {
        scores.assign(points.size(), 0.0);
        compute(std::move(points), scores);
    }

    //! Compute the outlier scores of \p points using \p compute and
    //! a bag of random projections.
    template<typename COMPUTE, typename POINT>
    static void computeForBagOfProjections(COMPUTE& compute,
                                           const std::vector<POINT>& points,
                                           TDoubleVec& scores) {
        using TPoint = decltype(SConstant<POINT>::get(0, 0));
        using TPointVec = std::vector<TPoint>;
        using TAnnotatedPoint = CAnnotatedVector<TPoint, std::size_t>;
        using TAnnotatedPointVec = std::vector<TAnnotatedPoint>;
        using TMaxAccumulator = CBasicStatistics::SMax<double>::TAccumulator;
        using TMaxAccumulatorVec = std::vector<TMaxAccumulator>;

        std::size_t dimension{las::dimension(points[0])};
        std::size_t bags, projectedDimension;
        std::tie(bags, projectedDimension) = computeBagsAndProjectedDimension(points);

        // Use standard 2-stable Gaussian projections since we are
        // interested in Euclidean distances.
        TDoubleVec coordinates;
        CPRNG::CXorOShiro128Plus rng;
        CSampling::normalSample(rng, 0.0, 1.0, 2 * bags * projectedDimension * dimension,
                                coordinates);

        // Placeholder for projected points.
        TAnnotatedPointVec projectedPoints;
        projectedPoints.reserve(points.size());
        for (std::size_t i = 0; i < points.size(); ++i) {
            projectedPoints.emplace_back(SConstant<POINT>::get(projectedDimension, 0), i);
        }

        // We're interested if points are:
        //   1) Outlying in any projection of the data,
        //   2) Outlying in many projections of the data.
        //
        // We want to identify the former as outliers and the later as
        // the most significant outliers. To this end we average the
        // maximum score from any projection with the average score for
        // all projections.
        TMaxAccumulatorVec maxScores(points.size());
        TMeanAccumulatorVec meanScores(points.size());

        TPointVec projection(projectedDimension, las::zero(points[0]));
        for (std::size_t bag = 0, i = 0; bag < bags && i < coordinates.size(); /**/) {
            // Create an orthonormal basis for the bag.
            for (std::size_t p = 0; p < projectedDimension; ++p) {
                for (std::size_t d = 0; d < dimension; ++i, ++d) {
                    projection[p](d) = coordinates[i];
                }
            }
            if (COrthogonaliser::orthonormalBasis(projection) &&
                projection.size() == projectedDimension) {

                // Project onto the basis.
                core::parallel_for_each(0, points.size(), [&](std::size_t j) {
                    for (std::size_t d = 0; d < projectedDimension; ++d) {
                        projectedPoints[j](d) = las::inner(projection[d], points[j]);
                    }
                });

                // Compute the scores and update the overall score.
                scores.assign(points.size(), 0.0);
                compute(projectedPoints, scores);
                core::parallel_for_each(0, scores.size(), [&](std::size_t j) {
                    maxScores[j].add(scores[j]);
                    meanScores[j].add(scores[j]);
                });

                ++bag;
            }
        }

        core::parallel_for_each(0, meanScores.size(), [&](std::size_t i) {
            scores[i] = (maxScores[i][0] + CBasicStatistics::mean(meanScores[i])) / 2.0;
        });
    }

    //! Compute the number of bags and the projection dimension.
    template<typename POINT>
    static TSizeSizePr computeBagsAndProjectedDimension(const std::vector<POINT>& points) {
        // Note because we get big wins in the lookup speed of the k-d
        // by reducing the point dimensions we still get a speed up if
        // "number of projections" x "projection dimension" is O(dimension)
        // of the full space.

        std::size_t dimension{las::dimension(points[0])};
        double rootDimension{std::sqrt(std::min(static_cast<double>(dimension), 100.0))};
        double logNumberPoints{std::log(static_cast<double>(points.size()))};
        std::size_t projectedDimension{static_cast<std::size_t>(
            std::max(std::min(rootDimension, logNumberPoints), 4.0) + 0.5)};
        return {std::max(dimension / 2 / projectedDimension, std::size_t(2)), projectedDimension};
    }

    //! \brief The interface for an outlier calculation method.
    template<typename POINT, typename NEAREST_NEIGHBOURS>
    class CMethod {
    public:
        using TPoint = CAnnotatedVector<POINT, std::size_t>;
        using TPointVec = std::vector<TPoint>;

    public:
        CMethod(NEAREST_NEIGHBOURS lookup) : m_Lookup(std::move(lookup)) {}
        virtual ~CMethod() = default;

        void run(std::size_t k, TPointVec points, TDoubleVec& scores) {
            this->setup(points);
            m_Lookup.build(std::move(points));
            // We call add exactly once for each point. Scores is presized
            // so writes are safe because these are happening for different
            // elements in different threads.
            core::parallel_for_each(
                m_Lookup.begin(), m_Lookup.end(),
                [&, neighbours = TPointVec{} ](const TPoint& point) mutable {
                    m_Lookup.nearestNeighbours(k + 1, point, neighbours);
                    this->add(point, neighbours, scores);
                });
            this->compute(scores);
        }

        const NEAREST_NEIGHBOURS& lookup() const { return m_Lookup; }

        virtual void setup(const TPointVec& /*points*/) {}

        virtual void
        add(const TPoint& point, const TPointVec& neighbours, TDoubleVec& scores) = 0;

        virtual void compute(TDoubleVec& scores) { normalize(scores); }

    private:
        NEAREST_NEIGHBOURS m_Lookup;
    };

    //! \brief Computes the normalized version of the local outlier
    //! factor score.
    template<typename POINT, typename NEAREST_NEIGHBOURS>
    class CLof final : public CMethod<POINT, NEAREST_NEIGHBOURS> {
    public:
        using TPoint = CAnnotatedVector<POINT, std::size_t>;
        using TPointVec = std::vector<TPoint>;
        using TSizeDoublePr = std::pair<std::size_t, double>;
        using TSizeDoublePrVec = std::vector<TSizeDoublePr>;
        using TSizeDoublePrVecVec = std::vector<TSizeDoublePrVec>;

    public:
        CLof(std::size_t k, NEAREST_NEIGHBOURS lookup = NEAREST_NEIGHBOURS())
            : CMethod<POINT, NEAREST_NEIGHBOURS>(std::move(lookup)), m_K(k) {}

        void operator()(TPointVec points, TDoubleVec& scores) {
            this->run(m_K, std::move(points), scores);
        }

        virtual void setup(const TPointVec& points) {
            m_KDistances.assign(points.size(), TSizeDoublePrVec());
            m_Lrd.assign(points.size(), 0.0);
        }

        virtual void add(const TPoint& point, const TPointVec& neighbours, TDoubleVec&) {
            // This is called exactly once for each point therefore an
            // element of m_KDistances is only ever written by one thread.
            std::size_t i{point.annotation()};
            std::size_t k{std::min(m_K, neighbours.size() - 1)};
            m_KDistances[i].reserve(k);
            for (std::size_t j = 1; j <= k; ++j) {
                m_KDistances[i].emplace_back(neighbours[j].annotation(),
                                             las::distance(point, neighbours[j]));
            }
        }

        virtual void compute(TDoubleVec& scores) {
            using TMinAccumulator = CBasicStatistics::SMin<double>::TAccumulator;

            // We bind a minimum accumulator (by value) to each lambda (since one
            // copy is then accessed by each thread) and take the minimum of these
            // at the end.

            auto results = core::parallel_for_each(
                this->lookup().begin(), this->lookup().end(),
                core::bindRetrievableState(
                    [&](TMinAccumulator& min, const TPoint& point) {
                        std::size_t i{point.annotation()};
                        TMeanAccumulator reachability_;
                        for (const auto& neighbour : m_KDistances[i]) {
                            reachability_.add(
                                std::max(kdistance(m_KDistances[index(neighbour)]),
                                         distance(neighbour)));
                        }
                        double reachability{CBasicStatistics::mean(reachability_)};
                        if (reachability > 0.0) {
                            m_Lrd[i] = 1.0 / reachability;
                            min.add(reachability);
                        } else {
                            m_Lrd[i] = -1.0;
                        }
                    },
                    TMinAccumulator{}));

            TMinAccumulator min;
            for (const auto& result : results) {
                min += result.s_FunctionState;
            }

            if (min.count() > 0) {
                // Use twice the maximum "density" at any other point if there are
                // k-fold duplicates.
                for (auto& lrd : m_Lrd) {
                    if (lrd < 0.0) {
                        lrd = 2.0 / min[0];
                    }
                }
                core::parallel_for_each(
                    this->lookup().begin(), this->lookup().end(), [&](const TPoint& point) {
                        std::size_t i{point.annotation()};
                        TMeanAccumulator score;
                        for (const auto& neighbour : m_KDistances[i]) {
                            score.add(m_Lrd[index(neighbour)]);
                        }
                        scores[i] = CBasicStatistics::mean(score) / m_Lrd[i];
                    });
            }
            normalize(scores);
        }

        static std::size_t estimateOwnMemoryOverhead(std::size_t k, std::size_t numberPoints) {
            return numberPoints * (sizeof(TSizeDoublePrVec) +
                                   k * sizeof(TSizeDoublePr) + sizeof(double));
        }

    private:
        static std::size_t index(const TSizeDoublePr& neighbour) {
            return neighbour.first;
        }
        static double distance(const TSizeDoublePr& neighbour) {
            return neighbour.second;
        }
        static double kdistance(const TSizeDoublePrVec& neighbours) {
            return distance(neighbours.back());
        }

    private:
        std::size_t m_K;
        TSizeDoublePrVecVec m_KDistances;
        TDoubleVec m_Lrd;
    };

    //! \brief Computes the normalized version of the local distance
    //! based outlier score.
    template<typename POINT, typename NEAREST_NEIGHBOURS>
    class CLdof final : public CMethod<POINT, NEAREST_NEIGHBOURS> {
    public:
        using TPoint = CAnnotatedVector<POINT, std::size_t>;
        using TPointVec = std::vector<TPoint>;

    public:
        CLdof(std::size_t k, NEAREST_NEIGHBOURS lookup = NEAREST_NEIGHBOURS())
            : CMethod<POINT, NEAREST_NEIGHBOURS>(std::move(lookup)), m_K(k) {}

        void operator()(TPointVec points, TDoubleVec& scores) {
            this->run(m_K, std::move(points), scores);
        }

        virtual void add(const TPoint& point, const TPointVec& neighbours, TDoubleVec& scores) {
            TMeanAccumulator d, D;
            std::size_t k{std::min(m_K, neighbours.size() - 1)};
            for (std::size_t i = 1; i <= k; ++i) {
                d.add(las::distance(point, neighbours[i]));
                for (std::size_t j = 1; j < i; ++j) {
                    D.add(las::distance(neighbours[i], neighbours[j]));
                }
            }
            scores[point.annotation()] = CBasicStatistics::mean(D) > 0.0
                                             ? CBasicStatistics::mean(d) /
                                                   CBasicStatistics::mean(D)
                                             : 0.0;
        }

    private:
        std::size_t m_K;
    };

    //! \brief Computes the normalized version of the distance
    //! to the k'th nearest neighbour score.
    template<typename POINT, typename NEAREST_NEIGHBOURS>
    class CDistancekNN final : public CMethod<POINT, NEAREST_NEIGHBOURS> {
    public:
        using TPoint = CAnnotatedVector<POINT, std::size_t>;
        using TPointVec = std::vector<TPoint>;

    public:
        CDistancekNN(std::size_t k, NEAREST_NEIGHBOURS lookup = NEAREST_NEIGHBOURS())
            : CMethod<POINT, NEAREST_NEIGHBOURS>(std::move(lookup)), m_K(k) {}

        void operator()(TPointVec points, TDoubleVec& scores) {
            this->run(m_K, std::move(points), scores);
        }

        virtual void add(const TPoint& point, const TPointVec& neighbours, TDoubleVec& scores) {
            std::size_t k{std::min(m_K, neighbours.size() - 1)};
            scores[point.annotation()] = las::distance(point, neighbours[k]);
        }

    private:
        std::size_t m_K;
    };

    //! \brief Computes the normalized version of the total
    //! distance to the k nearest neighbours.
    template<typename POINT, typename NEAREST_NEIGHBOURS>
    class CTotalDistancekNN final : public CMethod<POINT, NEAREST_NEIGHBOURS> {
    public:
        using TPoint = CAnnotatedVector<POINT, std::size_t>;
        using TPointVec = std::vector<TPoint>;

    public:
        CTotalDistancekNN(std::size_t k, NEAREST_NEIGHBOURS lookup = NEAREST_NEIGHBOURS())
            : CMethod<POINT, NEAREST_NEIGHBOURS>(std::move(lookup)), m_K(k) {}

        void operator()(TPointVec points, TDoubleVec& scores) {
            this->run(m_K, std::move(points), scores);
        }

        virtual void add(const TPoint& point, const TPointVec& neighbours, TDoubleVec& scores) {
            std::size_t i{point.annotation()};
            std::size_t k{std::min(m_K, neighbours.size() - 1)};
            for (std::size_t j = 1; j <= k; ++j) {
                scores[i] += las::distance(point, neighbours[j]);
            }
            scores[i] /= static_cast<double>(k);
        }

    private:
        std::size_t m_K;
    };

    //! \brief Computes a mean ensemble of the other methods.
    template<typename POINT, typename NEAREST_NEIGHBOURS>
    class CEnsemble final : public CMethod<POINT, NEAREST_NEIGHBOURS> {
    public:
        using TPoint = CAnnotatedVector<POINT, std::size_t>;
        using TPointVec = std::vector<TPoint>;
        using TMethodUPtr = std::unique_ptr<CMethod<POINT, const NEAREST_NEIGHBOURS&>>;
        using TMethodUPtrVec = std::vector<TMethodUPtr>;

    public:
        CEnsemble(NEAREST_NEIGHBOURS lookup = NEAREST_NEIGHBOURS())
            : CMethod<POINT, NEAREST_NEIGHBOURS>(lookup), m_K(0) {}

        template<typename T>
        CEnsemble& lof(const std::vector<T>& points) {
            m_K = defaultNumberOfNeighbours(points);
            m_Methods.push_back(std::make_unique<CLof<POINT, const NEAREST_NEIGHBOURS&>>(
                m_K, this->lookup()));
            return *this;
        }
        template<typename T>
        CEnsemble& ldof(const std::vector<T>& points) {
            m_K = defaultNumberOfNeighbours(points);
            m_Methods.push_back(std::make_unique<CLdof<POINT, const NEAREST_NEIGHBOURS&>>(
                m_K, this->lookup()));
            return *this;
        }
        template<typename T>
        CEnsemble& knn(const std::vector<T>& points) {
            m_K = defaultNumberOfNeighbours(points);
            m_Methods.push_back(std::make_unique<CDistancekNN<POINT, const NEAREST_NEIGHBOURS&>>(
                m_K, this->lookup()));
            return *this;
        }
        template<typename T>
        CEnsemble& tnn(const std::vector<T>& points) {
            m_K = defaultNumberOfNeighbours(points);
            m_Methods.push_back(
                std::make_unique<CTotalDistancekNN<POINT, const NEAREST_NEIGHBOURS&>>(
                    m_K, this->lookup()));
            return *this;
        }

        void operator()(TPointVec points, TDoubleVec& scores) {
            this->run(m_K, std::move(points), scores);
        }

        virtual void setup(const TPointVec& points) {
            for (auto& method : m_Methods) {
                method->setup(points);
            }
            m_Scores.assign(m_Methods.size(), TDoubleVec(points.size()));
        }

        virtual void add(const TPoint& point, const TPointVec& neighbours, TDoubleVec&) {
            for (std::size_t i = 0; i < m_Methods.size(); ++i) {
                m_Methods[i]->add(point, neighbours, m_Scores[i]);
            }
        }

        virtual void compute(TDoubleVec& scores) {
            for (std::size_t i = 0; i < m_Methods.size(); ++i) {
                m_Methods[i]->compute(m_Scores[i]);
            }
            for (std::size_t i = 0; i < scores.size(); ++i) {
                for (std::size_t j = 0; j < m_Scores.size(); ++j) {
                    scores[i] += m_Scores[j][i];
                }
                scores[i] /= static_cast<double>(m_Scores.size());
            }
        }

        static std::size_t estimateOwnMemoryOverhead(std::size_t numberPoints,
                                                     std::size_t numberMethods) {
            return numberMethods * (sizeof(TDoubleVec) + numberPoints * sizeof(double));
        }

    private:
        std::size_t m_K;
        TMethodUPtrVec m_Methods;
        TDoubleVecVec m_Scores;
    };

    //! The Gaussian normalization scheme proposed in "Interpreting
    //! and Unifying Outlier Scores" by Kreigel et al.
    //!
    //! This simply fits a Gaussian to a collection of scores for an
    //! outlier method and computes the right tail probability.
    static void normalize(TDoubleVec& scores);

    //! Convert the c.d.f. complement of an outlier factor into a
    //! score in the range [0,100] with the higher the score the
    //! greater the outlier.
    static double cdfComplementToScore(double cdfComplement);
};

//! Compute outliers for \p frame and write to a new column.
MATHS_EXPORT
bool computeOutliers(std::size_t numberThreads, core::CDataFrame& frame);
}
}

#endif // INCLUDED_ml_maths_CLocalOutlierFactors_h
