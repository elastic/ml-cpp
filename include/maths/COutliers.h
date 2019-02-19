/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_COutliers_h
#define INCLUDED_ml_maths_COutliers_h

#include <core/CDataFrame.h>
#include <core/CHashing.h>
#include <core/CNonInstantiatable.h>
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
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace ml {
namespace maths {
namespace outliers_detail {
using TDoubleVec = std::vector<double>;
using TDouble1Vec = core::CSmallVector<double, 1>;
using TDouble2Vec = core::CSmallVector<double, 2>;
using TDouble1VecVec = std::vector<TDouble1Vec>;
using TDouble1VecVec2Vec = core::CSmallVector<TDouble1VecVec, 2>;
using TDouble1Vec2Vec = core::CSmallVector<TDouble1Vec, 2>;
using TProgressCallback = std::function<void(double)>;
using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

//! Get the distance in the complement space of the projection.
//!
//! We use Euclidean distances so this is just \f$\sqrt{d^2 - x^2}\f$ for
//! \f$d\f$ the total distance and \f$x\f$ is the distance in the projection
//! space.
inline double complementDistance(double distance, double projectionDistance) {
    return std::sqrt(std::max(CTools::pow2(distance) - CTools::pow2(projectionDistance), 0.0));
}

//! \brief The interface for a nearest neighbour outlier calculation method.
template<typename POINT, typename NEAREST_NEIGHBOURS>
class CNearestNeighbourMethod {
public:
    using TPointVec = std::vector<POINT>;

public:
    CNearestNeighbourMethod(bool featureSignificances,
                            std::size_t k,
                            NEAREST_NEIGHBOURS lookup,
                            TProgressCallback recordProgress)
        : m_FeatureSignificances{featureSignificances}, m_K{k}, m_Lookup{std::move(lookup)},
          m_RecordProgress{std::move(recordProgress)} {}
    virtual ~CNearestNeighbourMethod() = default;

    //! Check whether to compute coordinate scores.
    bool featureSignificances() const { return m_FeatureSignificances; }

    //! The number of points.
    std::size_t n() const { return m_Lookup.size(); }

    //! The number of nearest neighbours.
    std::size_t k() const { return m_K; }

    //! Get the nearest neighbours lookup.
    const NEAREST_NEIGHBOURS& lookup() const { return m_Lookup; }

    //! Compute the outlier scores for \p points.
    TDouble1VecVec2Vec run(const TPointVec& points, std::size_t numberScores) {

        this->setup(points);

        // We call add exactly once for each point. Scores is presized
        // so any writes to it are safe.
        TDouble1VecVec2Vec scores(this->numberMethods(), TDouble1VecVec(numberScores));
        core::parallel_for_each(points.begin(), points.end(),
                                [&, neighbours = TPointVec{} ](const POINT& point) mutable {
                                    m_Lookup.nearestNeighbours(m_K + 1, point, neighbours);
                                    this->add(point, neighbours, scores);
                                },
                                [this](double fractionalProgress) {
                                    this->recordProgress(fractionalProgress);
                                });

        this->compute(points, scores);

        return scores;
    }

    //! Recover any temporary memory used by run.
    virtual void recoverMemory() {}

    //! \name Progress Monitoring
    //@{
    //! Get the progress recorder.
    TProgressCallback& progressRecorder() { return m_RecordProgress; }
    //! Record \p fractionalProgress.
    void recordProgress(double fractionalProgress) {
        m_RecordProgress(fractionalProgress);
    }
    //@}

    //! Get a human readable description of the outlier detection method.
    virtual std::string print() const {
        return this->name() + "(n = " + std::to_string(this->n()) +
               ", k = " + std::to_string(m_K) + ")";
    }

    virtual void setup(const TPointVec&) {}
    virtual void add(const POINT&, const TPointVec&, TDouble1VecVec&) {}
    virtual void compute(const TPointVec&, TDouble1VecVec&) {}

private:
    virtual void add(const POINT& point, const TPointVec& neighbours, TDouble1VecVec2Vec& scores) {
        this->add(point, neighbours, scores[0]);
    }
    virtual void compute(const TPointVec& points, TDouble1VecVec2Vec& scores) {
        this->compute(points, scores[0]);
    }
    virtual std::size_t numberMethods() const { return 1; }
    virtual std::string name() const = 0;

private:
    bool m_FeatureSignificances;
    std::size_t m_K;
    NEAREST_NEIGHBOURS m_Lookup;
    TProgressCallback m_RecordProgress;
};

//! \brief Computes the local outlier factor score.
template<typename POINT, typename NEAREST_NEIGHBOURS>
class CLof final : public CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS> {
public:
    using TPointVec = std::vector<POINT>;
    using TCoordinate = typename SCoordinate<POINT>::Type;
    using TCoordinateVec = std::vector<TCoordinate>;
    using TUInt32CoordinatePr = std::pair<uint32_t, TCoordinate>;
    using TUInt32CoordinatePrVec = std::vector<TUInt32CoordinatePr>;
    static const TCoordinate UNSET_DISTANCE;

public:
    CLof(bool featureSignificances,
         std::size_t k,
         TProgressCallback recordProgress,
         NEAREST_NEIGHBOURS lookup = NEAREST_NEIGHBOURS())
        : CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS>{
              featureSignificances, k, std::move(lookup), std::move(recordProgress)} {}

    void recoverMemory() override {
        m_KDistances.resize(this->k() * m_StartAddresses);
        m_Lrd.resize(m_StartAddresses);
        m_CoordinateLrd.resize(m_Dimension * m_StartAddresses);
        m_KDistances.shrink_to_fit();
        m_Lrd.shrink_to_fit();
        m_CoordinateLrd.shrink_to_fit();
    }

    static std::size_t estimateOwnMemoryOverhead(bool featureSignificances,
                                                 std::size_t k,
                                                 std::size_t numberPoints,
                                                 std::size_t dimension) {
        return numberPoints *
               (k * sizeof(TUInt32CoordinatePr) +
                (featureSignificances ? dimension + 1 : 1) * sizeof(TCoordinate));
    }

private:
    void setup(const TPointVec& points) override {

        m_Dimension = las::dimension(points[0]);

        auto minmax = std::minmax_element(
            points.begin(), points.end(), [](const POINT& lhs, const POINT& rhs) {
                return lhs.annotation() < rhs.annotation();
            });
        m_StartAddresses = minmax.first->annotation();
        m_EndAddresses = minmax.second->annotation() + 1;

        std::size_t k{this->k()};
        m_KDistances.resize(k * m_StartAddresses, {0, UNSET_DISTANCE});
        m_KDistances.resize(k * m_EndAddresses, {0, UNSET_DISTANCE});
        m_Lrd.resize(m_StartAddresses, UNSET_DISTANCE);
        m_Lrd.resize(m_EndAddresses, UNSET_DISTANCE);

        if (this->featureSignificances()) {
            m_CoordinateLrd.resize(m_Dimension * m_StartAddresses, UNSET_DISTANCE);
            m_CoordinateLrd.resize(m_Dimension * m_EndAddresses, UNSET_DISTANCE);
        }
    }

    void add(const POINT& point, const TPointVec& neighbours, TDouble1VecVec&) override {
        // This is called exactly once for each point therefore an element
        // of m_KDistances is only ever written by one thread.
        std::size_t i{point.annotation()};
        std::size_t a(point == neighbours[0] ? 1 : 0);
        std::size_t b{std::min(this->k() + a - 1, neighbours.size() + a - 2)};
        for (std::size_t j = a; j <= b; ++j) {
            std::size_t index{this->kDistanceIndex(i, j - a)};
            m_KDistances[index].first =
                static_cast<std::uint32_t>(neighbours[j].annotation());
            m_KDistances[index].second = las::distance(point, neighbours[j]);
        }
    }

    void compute(const TPointVec& points, TDouble1VecVec& scores) override {
        this->computeLocalReachabilityDistances(points);
        this->computeLocalOutlierFactors(points, scores);
    }

    std::string name() const override { return "lof"; }

    void computeLocalReachabilityDistances(const TPointVec& points) {

        // We bind minimum accumulators (by value) to each lambda (since
        // one copy is then accessed by each thread) and take the minimum
        // of these at the end.

        using TMinAccumulator = CBasicStatistics::SMin<double>::TAccumulator;

        auto results = core::parallel_for_each(
            points.begin(), points.end(),
            core::bindRetrievableState(
                [this](TMinAccumulator& min, const POINT& point) mutable {

                    std::size_t i{point.annotation()};
                    TMeanAccumulator reachability_;
                    for (std::size_t j = 0; j < this->k(); ++j) {
                        const auto& neighbour = m_KDistances[this->kDistanceIndex(i, j)];
                        if (distance(neighbour) != UNSET_DISTANCE) {
                            reachability_.add(this->reachabilityDistance(neighbour));
                        }
                    }
                    double reachability{CBasicStatistics::mean(reachability_)};
                    if (reachability > 0.0) {
                        m_Lrd[i] = 1.0 / reachability;
                        min.add(reachability);
                    }
                },
                TMinAccumulator{}));

        TMinAccumulator min;
        for (const auto& result : results) {
            min += result.s_FunctionState;
        }
        if (min.count() > 0) {
            // Use twice the maximum "density" at any other point if there
            // are k-fold duplicates.
            for (std::size_t i = m_StartAddresses; i < m_EndAddresses; ++i) {
                if (m_Lrd[i] == UNSET_DISTANCE) {
                    m_Lrd[i] = 2.0 / min[0];
                }
            }
        }

        if (this->featureSignificances()) {
            // The idea is to compute the score placing each point p at the
            // locations which minimise the lof on each axis aligned line
            // passing through p. If we let p'(x) denote the point obtained
            // by replacing the i'th coordinate of p with x then this is p'(y)
            // where
            //   y = argmin_{x}{sum_{i in neighbours of p}{rd(p'(x))}}
            // and rd(.) denotes the reachability distance. Unfortunately,
            // we need to look up nearest neighbours again.

            core::parallel_for_each(
                points.begin(), points.end(),
                [&, neighbours = TPointVec{} ](const POINT& point) mutable {

                    std::size_t i{point.annotation()};

                    this->lookup().nearestNeighbours(this->k() + 1, point, neighbours);
                    std::size_t a(point == neighbours[0] ? 1 : 0);
                    std::size_t b{std::min(this->k() + a - 1, neighbours.size() + a - 2)};

                    POINT point_;
                    for (std::size_t j = 0; j < m_Dimension; ++j) {
                        // p'(y) can be found by an iterative scheme which moves
                        // it as close as possible to the current point on the
                        // hypersphere with radius equal to the k-distance of the
                        // neighbour. Initializing with the p after two iterations
                        // we end up very close.
                        point_ = point;
                        for (std::size_t round = 0; round < 2; ++round) {
                            TMeanAccumulator centroid;
                            for (std::size_t k = a; k <= b; ++k) {
                                const POINT& neighbour{neighbours[k]};
                                double kdistance{this->kdistance(neighbour.annotation())};
                                double distance{las::distance(point_, neighbour)};
                                double step{std::min(kdistance / distance, 1.0)};
                                centroid.add((neighbour + (point_ - neighbour) * step)(j));
                            }
                            point_(j) = CBasicStatistics::mean(centroid);
                        }

                        TMeanAccumulator reachability_;
                        for (std::size_t k = a; k <= b; ++k) {
                            POINT& neighbour{neighbours[k]};
                            double kdistance{this->kdistance(neighbour.annotation())};
                            double distance{las::distance(point_, neighbour)};
                            reachability_.add(std::max(kdistance, distance));
                        }
                        double reachability{
                            std::max(CBasicStatistics::mean(reachability_), min[0])};
                        m_CoordinateLrd[this->coordinateLrdIndex(i, j)] = 1.0 / reachability;
                    }
                });
        }
    }

    void computeLocalOutlierFactors(const TPointVec& points, TDouble1VecVec& scores) {

        core::parallel_for_each(points.begin(), points.end(), [&](const POINT& point) mutable {

            std::size_t i{point.annotation()};

            TMeanAccumulator neighbourhoodLrd;
            for (std::size_t j = 0; j < this->k(); ++j) {
                const auto& neighbour = m_KDistances[this->kDistanceIndex(i, j)];
                if (distance(neighbour) != UNSET_DISTANCE) {
                    neighbourhoodLrd.add(m_Lrd[index(neighbour)]);
                }
            }

            scores[i].resize(this->featureSignificances() ? m_Dimension + 1 : 1);
            scores[i][0] = CBasicStatistics::mean(neighbourhoodLrd) / m_Lrd[i];

            // We choose to ignore the impact of moving the point on its
            // neighbours' local reachability distances when computing
            // scores for each coordinate.
            for (std::size_t j = 1; j < scores[i].size(); ++j) {
                scores[i][j] = CBasicStatistics::mean(neighbourhoodLrd) /
                               m_CoordinateLrd[this->coordinateLrdIndex(i, j - 1)];
            }
        });
    }

    std::size_t kDistanceIndex(std::size_t index, std::size_t neighbourIndex) const {
        return index * this->k() + neighbourIndex;
    }
    std::size_t coordinateLrdIndex(std::size_t index, std::size_t coordinate) const {
        return index * m_Dimension + coordinate;
    }

    static std::size_t index(const TUInt32CoordinatePr& neighbour) {
        return neighbour.first;
    }
    static double distance(const TUInt32CoordinatePr& neighbour) {
        return neighbour.second;
    }

    double reachabilityDistance(const TUInt32CoordinatePr& neighbour) const {
        return std::max(this->kdistance(index(neighbour)), distance(neighbour));
    }
    double kdistance(std::size_t i) const {
        for (std::size_t j = this->k(); j > 0; --j) {
            double dist{distance(m_KDistances[this->kDistanceIndex(i, j - 1)])};
            if (dist != UNSET_DISTANCE) {
                return dist;
            }
        }
        return UNSET_DISTANCE;
    }

private:
    std::size_t m_Dimension;
    std::size_t m_StartAddresses;
    std::size_t m_EndAddresses;
    // The k distances to the neighbouring points of each point are stored
    // flattened: [neighbours of 0, neighbours of 1,...].
    TUInt32CoordinatePrVec m_KDistances;
    TCoordinateVec m_Lrd;
    // The coordinate local reachability distances are stored flattened:
    // [coordinates of 0, coordinates of 2, ...].
    TCoordinateVec m_CoordinateLrd;
};

template<typename POINT, typename NEAREST_NEIGHBOURS>
const typename CLof<POINT, NEAREST_NEIGHBOURS>::TCoordinate
    CLof<POINT, NEAREST_NEIGHBOURS>::UNSET_DISTANCE(-1.0);

//! \brief Computes the local distance based outlier score.
template<typename POINT, typename NEAREST_NEIGHBOURS>
class CLdof final : public CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS> {
public:
    CLdof(bool featureSignificances,
          std::size_t k,
          TProgressCallback recordProgress,
          NEAREST_NEIGHBOURS lookup = NEAREST_NEIGHBOURS())
        : CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS>{
              featureSignificances, k, std::move(lookup), std::move(recordProgress)} {}

private:
    void add(const POINT& point, const std::vector<POINT>& neighbours, TDouble1VecVec& scores) override {

        using TPointMeanAccumulator = typename CBasicStatistics::SSampleMean<POINT>::TAccumulator;

        auto ldof = [](const TMeanAccumulator& d, const TMeanAccumulator& D) {
            return CBasicStatistics::mean(D) > 0.0
                       ? CBasicStatistics::mean(d) / CBasicStatistics::mean(D)
                       : 0.0;
        };

        std::size_t a(point == neighbours[0] ? 1 : 0);
        std::size_t b{std::min(this->k() + a - 1, neighbours.size() + a - 2)};

        auto& score = scores[point.annotation()];
        score.resize(this->featureSignificances() ? las::dimension(point) + 1 : 1);

        TMeanAccumulator d, D;
        for (std::size_t i = a; i <= b; ++i) {
            d.add(las::distance(point, neighbours[i]));
            for (std::size_t j = 1; j < i; ++j) {
                D.add(las::distance(neighbours[i], neighbours[j]));
            }
        }
        score[0] = ldof(d, D);

        if (score.size() > 1) {
            // The idea is to compute the score placing the point at the
            // locations which minimise the ldof on each axis aligned line
            // passing through the point. These are just the neighbours
            // coordinate centroids.
            TPointMeanAccumulator centroid{las::zero(point)};
            centroid.add(neighbours);
            POINT point_{point};
            for (std::size_t i = 1; i < score.size(); ++i) {
                d = TMeanAccumulator{};
                point_(i - 1) = CBasicStatistics::mean(centroid)(i - 1);
                for (std::size_t j = a; j <= b; ++j) {
                    d.add(las::distance(point_, neighbours[j]));
                }
                score[i] = ldof(d, D);
            }
        }
    }

    std::string name() const override { return "ldof"; }
};

//! \brief Computes the distance to the k'th nearest neighbour score.
template<typename POINT, typename NEAREST_NEIGHBOURS>
class CDistancekNN final : public CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS> {
public:
    CDistancekNN(bool featureSignificances,
                 std::size_t k,
                 TProgressCallback recordProgress,
                 NEAREST_NEIGHBOURS lookup = NEAREST_NEIGHBOURS())
        : CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS>{
              featureSignificances, k, std::move(lookup), std::move(recordProgress)} {}

private:
    void add(const POINT& point, const std::vector<POINT>& neighbours, TDouble1VecVec& scores) override {

        std::size_t k{std::min(this->k() + 1, neighbours.size() - 1) -
                      (point == neighbours[0] ? 0 : 1)};
        const auto& kthNeighbour = neighbours[k];

        auto& score = scores[point.annotation()];
        score.resize(this->featureSignificances() ? las::dimension(point) + 1 : 1);

        double d{las::distance(point, kthNeighbour)};
        score[0] = d;
        for (std::size_t i = 1; i < score.size(); ++i) {
            // The idea here is to compute the score for the complement
            // space of each coordinate projection.
            score[i] = complementDistance(d, kthNeighbour(i - 1) - point(i - 1));
        }
    }

    std::string name() const override { return "knn"; }
};

//! \brief Computes the total distance to the k nearest neighbours score.
template<typename POINT, typename NEAREST_NEIGHBOURS>
class CTotalDistancekNN final
    : public CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS> {
public:
    CTotalDistancekNN(bool featureSignificances,
                      std::size_t k,
                      TProgressCallback recordProgress,
                      NEAREST_NEIGHBOURS lookup = NEAREST_NEIGHBOURS())
        : CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS>{
              featureSignificances, k, std::move(lookup), std::move(recordProgress)} {}

private:
    void add(const POINT& point, const std::vector<POINT>& neighbours, TDouble1VecVec& scores) override {

        std::size_t a(point == neighbours[0] ? 1 : 0);
        std::size_t b{std::min(this->k() + a - 1, neighbours.size() + a - 2)};

        auto& score = scores[point.annotation()];
        score.assign(this->featureSignificances() ? las::dimension(point) + 1 : 1, 0.0);

        for (std::size_t i = a; i <= b; ++i) {
            double d{las::distance(point, neighbours[i])};
            score[0] += d;
            // The idea here is to compute the score for the complement
            // space of each coordinate projection.
            for (std::size_t j = 1; j < score.size(); ++j) {
                score[j] += complementDistance(d, neighbours[i](j - 1) - point(j - 1));
            }
        }
        for (std::size_t i = 0; i < score.size(); ++i) {
            score[i] /= static_cast<double>(this->k());
        }
    }

    std::string name() const override { return "tnn"; }
};

//! \brief A composite method.
//!
//! IMPLEMENTATION:\n
//! This is used in conjunction with CEnsemble so we can share nearest
//! neighbour lookups for all methods in an ensemble model.
template<typename POINT, typename NEAREST_NEIGHBOURS>
class CMultipleMethods final
    : public CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS> {
public:
    using TPointVec = std::vector<POINT>;
    using TMethodUPtr = std::unique_ptr<CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS>>;
    using TMethodUPtrVec = std::vector<TMethodUPtr>;

public:
    CMultipleMethods(std::size_t k,
                     TMethodUPtrVec methods,
                     NEAREST_NEIGHBOURS lookup = NEAREST_NEIGHBOURS())
        : CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS>{methods[0]->featureSignificances(),
                                                             k, std::move(lookup),
                                                             methods[0]->progressRecorder()},
          m_Methods{std::move(methods)} {}

    void recoverMemory() override {
        for (auto& model : m_Methods) {
            model->recoverMemory();
        }
    }

    std::string print() const override {
        std::string result;
        result += "{";
        for (const auto& method : m_Methods) {
            result += " " + method->print();
        }
        result += " }";
        return result;
    }

private:
    std::size_t numberMethods() const override { return m_Methods.size(); }

    void setup(const TPointVec& points) override {
        for (auto& method : m_Methods) {
            method->setup(points);
        }
    }

    void add(const POINT& point, const TPointVec& neighbours, TDouble1VecVec2Vec& scores) override {
        for (std::size_t i = 0; i < m_Methods.size(); ++i) {
            m_Methods[i]->add(point, neighbours, scores[i]);
        }
    }

    void compute(const TPointVec& points, TDouble1VecVec2Vec& scores) override {
        for (std::size_t i = 0; i < m_Methods.size(); ++i) {
            m_Methods[i]->compute(points, scores[i]);
        }
    }

    std::string name() const override { return "multiple"; }

private:
    TMethodUPtrVec m_Methods;
};

//! \brief This encapsulates creating a collection of models used for outlier
//! detection.
//!
//! DESCRIPTION:\n
//! A model is defined as one or more algorithm for computing outlier scores,
//! the number of nearest neighbours used to compute the score and a sample of
//! the original data points (possibly) projected onto a random subspace which
//! is searched for neighbours.
//!
//! The models can be built from a data stream by repeatedly calling addPoint.
//! The evaluation of the outlier score for a point is delagated.
template<typename POINT>
class CEnsemble {
private:
    class CModel;

public:
    using TSizeVec = std::vector<std::size_t>;
    using TSizeVecVec = std::vector<TSizeVec>;
    using TSizeSizePr = std::pair<std::size_t, std::size_t>;
    using TSizeSizePrVec = std::vector<TSizeSizePr>;
    using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using TMeanVarAccumulator2Vec = core::CSmallVector<TMeanVarAccumulator, 2>;
    using TPoint = CAnnotatedVector<decltype(SConstant<POINT>::get(0, 0)), std::size_t>;
    using TPointVec = std::vector<TPoint>;
    using TPointVecVec = std::vector<TPointVec>;
    using TKdTree = CKdTree<TPoint>;
    using TMatrix = typename SConformableMatrix<TPoint>::Type;
    using TMatrixVec = std::vector<TMatrix>;
    using TMethodUPtr = std::unique_ptr<CNearestNeighbourMethod<TPoint, const TKdTree&>>;
    using TMethodUPtrVec = std::vector<TMethodUPtr>;
    using TMethodFactory = std::function<TMethodUPtr(std::size_t, const TKdTree&)>;
    using TMethodFactoryVec = std::vector<TMethodFactory>;
    using TMethodSize = std::function<std::size_t(std::size_t, std::size_t, std::size_t)>;

    //! \brief Builds (online) one model of the points for the ensemble.
    class CModelBuilder {
    public:
        using TRowRef = core::CDataFrame::TRowRef;

    public:
        CModelBuilder(CPRNG::CXorOShiro128Plus& rng,
                      TSizeSizePrVec&& methodAndNumberNeighbours,
                      std::size_t sampleSize,
                      TMatrix&& projection);

        //! Maybe sample the point.
        void addPoint(const TRowRef& point) { m_Sampler.sample(point); }

        //! \note Only call once: this moves state into place.
        CModel make(const TMethodFactoryVec& methodFactories);

    private:
        using TSampler = CSampling::CRandomStreamSampler<TRowRef>;

    private:
        TSampler makeSampler(CPRNG::CXorOShiro128Plus& rng, std::size_t sampleSize);

    private:
        TSizeSizePrVec m_MethodsAndNumberNeighbours;
        std::size_t m_SampleSize;
        TSampler m_Sampler;
        TMatrix m_Projection;
        TPointVec m_SampledProjectedPoints;
    };
    using TModelBuilderVec = std::vector<CModelBuilder>;

    //! \brief Manages computing the probability that a point is an outlier
    //! given its scores from a collection of ensemble models.
    class CScorer {
    public:
        void add(const TMeanVarAccumulator2Vec& logScoreMoments,
                 const TMatrix& columnNormalizedProjection,
                 const TDouble1Vec2Vec& scores);

        //! Compute the posterior probability that the point is an outlier
        //! and optionally the feature significances.
        TDouble1Vec compute(double pOutlier) const;

        static std::size_t estimateMemoryUsage(std::size_t dimension) {
            return (dimension + 2) * sizeof(CFloatStorage);
        }

    private:
        using TFloat2Vec = core::CSmallVector<CFloatStorage, 2>;

    private:
        double ensembleSize() const { return m_State[0]; }
        CFloatStorage& ensembleSize() { return m_State[0]; }

        double logLikelihoodOutlier() const { return m_State[1]; }
        CFloatStorage& logLikelihoodOutlier() { return m_State[1]; }

        double significance(std::size_t index) const {
            return m_State[index + 2];
        }
        CFloatStorage& significance(std::size_t index) {
            return m_State[index + 2];
        }

        std::size_t numberSignificances() const { return m_State.size() - 2; }

    private:
        TFloat2Vec m_State;
    };
    using TScorerVec = std::vector<CScorer>;

public:
    static const double SAMPLE_SIZE_SCALE;
    static const double NEIGHBOURHOOD_FRACTION;

public:
    CEnsemble(const TMethodFactoryVec& methodFactories, TModelBuilderVec modelBuilders);
    CEnsemble(const CEnsemble&) = delete;
    CEnsemble& operator=(const CEnsemble&) = delete;
    CEnsemble(CEnsemble&&) = default;
    CEnsemble& operator=(CEnsemble&&) = default;

    //! Make the builders for the ensemble models.
    static TModelBuilderVec
    makeBuilders(const TSizeVecVec& algorithms,
                 std::size_t numberPoints,
                 std::size_t dimension,
                 CPRNG::CXorOShiro128Plus rng = CPRNG::CXorOShiro128Plus{});

    //! Compute the outlier scores for \p points.
    TScorerVec computeOutlierScores(const std::vector<POINT>& points) const;

    //! Estimate the amount of memory that will be used by the ensemble.
    static std::size_t
    estimateMemoryUsedToComputeOutlierScores(TMethodSize methodSize,
                                             std::size_t numberMethodsPerModel,
                                             bool featureSignificances,
                                             std::size_t totalNumberPoints,
                                             std::size_t partitionNumberPoints,
                                             std::size_t dimension) {
        std::size_t ensembleSize{computeEnsembleSize(numberMethodsPerModel,
                                                     totalNumberPoints, dimension)};
        std::size_t sampleSize{computeSampleSize(totalNumberPoints)};
        std::size_t numberModels{(ensembleSize + numberMethodsPerModel - 1) / numberMethodsPerModel};
        std::size_t maxNumberNeighbours{computeNumberNeighbours(sampleSize)};
        std::size_t projectionDimension{computeProjectionDimension(sampleSize, dimension)};
        std::size_t averageNumberNeighbours{(3 + maxNumberNeighbours) / 2};

        auto projectedPointsSize = [&] {
            return partitionNumberPoints * las::estimateMemoryUsage<TPoint>(projectionDimension);
        };
        auto scorersSize = [&] {
            return partitionNumberPoints *
                   (sizeof(CScorer) + CScorer::estimateMemoryUsage(
                                          featureSignificances ? dimension : 0));
        };
        auto modelSize = [&] {
            return CModel::estimateMemoryUsage(methodSize, sampleSize, averageNumberNeighbours,
                                               projectionDimension, dimension);
        };
        auto perPartitionOverhead = [&] {
            // The scores for a single method plus bookkeeping overhead
            // for a single partition.
            return numberMethodsPerModel * partitionNumberPoints *
                       (sizeof(TDouble1Vec) +
                        (featureSignificances ? projectionDimension * sizeof(double) : 0)) +
                   methodSize(averageNumberNeighbours, partitionNumberPoints, projectionDimension);
        };

        return projectedPointsSize() + scorersSize() +
               numberModels * modelSize() + perPartitionOverhead();
    }

    //! Get a human readable description of the ensemble.
    std::string print() const;

private:
    using TKdTreeUPtr = std::unique_ptr<TKdTree>;

    //! \brief A model of the points used as part of the ensemble.
    class CModel {
    public:
        CModel(const TMethodFactoryVec& methodFactories,
               TSizeSizePrVec methodAndNumberNeighbours,
               TPointVec samples,
               TMatrix projection);

        std::size_t numberPoints() const { return m_Lookup->size(); }

        void proportionOfRuntimePerMethod(double proportion);

        void addOutlierScores(const std::vector<POINT>& points, TScorerVec& scores) const;

        static std::size_t estimateMemoryUsage(TMethodSize methodSize,
                                               std::size_t sampleSize,
                                               std::size_t averageNumberNeighbours,
                                               std::size_t projectionDimension,
                                               std::size_t dimension) {
            auto lookupSize = [&] {
                return TKdTree::estimateMemoryUsage(sampleSize, projectionDimension);
            };
            auto projectionSize = [&] {
                return projectionDimension * dimension *
                       sizeof(typename SCoordinate<TPoint>::Type);
            };

            return lookupSize() + 2 * projectionSize() +
                   methodSize(averageNumberNeighbours, sampleSize, projectionDimension);
        }

        std::string print() const;

    private:
        TKdTreeUPtr m_Lookup;
        TMatrix m_Projection;
        TMatrix m_RowNormalizedProjection;
        TMethodUPtr m_Method;
        TMeanVarAccumulator2Vec m_LogScoreMoments;
    };
    using TModelVec = std::vector<CModel>;

private:
    static std::size_t computeEnsembleSize(std::size_t numberMethods,
                                           std::size_t numberPoints,
                                           std::size_t dimension) {
        // We want enough members such that we get:
        //   1. Reasonable coverage of original space,
        //   2. Reasonable coverage of the original point set.
        //
        // Using too few members turned up some pathologies in testing and
        // using many gives diminishing returns for the extra runtime and
        // memory usage so restrict to at least 6 and no more than 20.
        std::size_t projectionDimension{
            computeProjectionDimension(computeSampleSize(numberPoints), dimension)};
        std::size_t requiredNumberModels{(dimension + projectionDimension - 1) / projectionDimension};
        double target{std::max(static_cast<double>(numberMethods * requiredNumberModels),
                               std::sqrt(static_cast<double>(numberPoints)) / SAMPLE_SIZE_SCALE)};
        return static_cast<std::size_t>(std::min(std::max(target, 6.0), 20.0) + 0.5);
    }

    static std::size_t computeSampleSize(std::size_t numberPoints) {
        // We want an aggressive downsample of the original set. Except
        // for the case that there are many small clusters, this typically
        // improves QoR since it avoids outliers swamping one another. It
        // also greatly improves scalability.
        double target{2.0 * SAMPLE_SIZE_SCALE * std::sqrt(static_cast<double>(numberPoints))};
        return static_cast<std::size_t>(target + 0.5);
    }

    static std::size_t computeNumberNeighbours(std::size_t sampleSize) {
        // Use a fraction of the sample size but don't allow to get
        //   1. too small because the outlier metrics tend to be unstable in
        //      this regime or
        //   2. too big because they tend to be insentive to changes in this
        //      parameter when it's large, but the nearest neighbour search
        //      becomes much more expensive.
        double target{NEIGHBOURHOOD_FRACTION * static_cast<double>(sampleSize)};
        return static_cast<std::size_t>(std::min(std::max(target, 5.0), 100.0) + 0.5);
    }

    static std::size_t computeProjectionDimension(std::size_t numberPoints,
                                                  std::size_t dimension) {
        // We need a minimum number of points per dimension to get any sort
        // of stable density estimate. The dependency is exponential (curse
        // of dimensionality).
        double logNumberPoints{std::log(static_cast<double>(numberPoints)) / std::log(3.0)};
        double target{std::min(static_cast<double>(dimension), logNumberPoints)};
        return static_cast<std::size_t>(std::min(std::max(target, 2.0), 10.0) + 0.5);
    }

    static TMatrixVec createProjections(CPRNG::CXorOShiro128Plus& rng,
                                        std::size_t numberProjections,
                                        std::size_t projectionDimension,
                                        std::size_t dimension);

private:
    TModelVec m_Models;
};

template<typename POINT>
const double CEnsemble<POINT>::SAMPLE_SIZE_SCALE{5.0};
template<typename POINT>
const double CEnsemble<POINT>::NEIGHBOURHOOD_FRACTION{0.01};
}

//! \brief Utilities for computing outlier scores for collections of points.
class MATHS_EXPORT COutliers : private core::CNonInstantiatable {
public:
    using TDoubleVec = outliers_detail::TDoubleVec;
    using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using TProgressCallback = outliers_detail::TProgressCallback;
    template<typename POINT>
    using TAnnotatedPoint = CAnnotatedVector<POINT, std::size_t>;
    template<typename POINT>
    using TEnsemble = outliers_detail::CEnsemble<POINT>;
    template<typename POINT>
    using TLof =
        outliers_detail::CLof<TAnnotatedPoint<POINT>, CKdTree<TAnnotatedPoint<POINT>>>;

    //! The outlier detection methods which are available.
    enum EMethod {
        E_Lof,
        E_Ldof,
        E_DistancekNN,
        E_TotalDistancekNN,
        E_Ensemble
    };

    //! \brief The parameters for compute.
    struct SComputeParameters {
        //! The number of threads available.
        std::size_t s_NumberThreads;
        //! The number of partitions to use.
        std::size_t s_NumberPartitions;
        //! If true also compute the feature significances.
        bool s_FeatureSignificances;
        //! The prior probability that a point is an outlier.
        double s_ProbabilityOutlier;
    };

public:
    //! Compute outliers for \p frame and write to a new column.
    //!
    //! \param[in] params The calculation parameters.
    //! \param[in] frame The data frame whose rows hold the coordinated of
    //! the points for which to compute outliers.
    //! \param[in] recordProgress A function to which fractional progress
    //! is written.
    static void compute(const SComputeParameters& params,
                        core::CDataFrame& frame,
                        TProgressCallback recordProgress = noop);

    //! Estimate the amount of memory that will be used computing outliers
    //! for a data frame.
    //!
    //! \param[in] method The method that will be used.
    //! \param[in] k The number of nearest neighbours which will be used.
    //! \param[in] featureSignificances If true the feature significances
    //! will also be computed.
    //! \param[in] totalNumberPoints The total number of points for which
    //! outlier scores will be computed.
    //! \param[in] partitionNumberPoints The number of points per partition
    //! for which outlier scores will be computed.
    //! \param[in] dimension The dimension of the points for which outliers
    //! will be computed.
    template<typename POINT>
    static std::size_t estimateMemoryUsedByCompute(EMethod method,
                                                   std::size_t k,
                                                   bool featureSignificances,
                                                   std::size_t totalNumberPoints,
                                                   std::size_t partitionNumberPoints,
                                                   std::size_t dimension) {
        auto methodSize = [=](std::size_t, std::size_t numberPoints,
                              std::size_t projectionDimension) {
            return method == E_Lof
                       ? TLof<POINT>::estimateOwnMemoryOverhead(
                             featureSignificances, k, numberPoints, projectionDimension)
                       : 0;
        };
        return TEnsemble<POINT>::estimateMemoryUsedToComputeOutlierScores(
            methodSize, 1 /*number methods*/, featureSignificances,
            totalNumberPoints, partitionNumberPoints, dimension);
    }

    //! Overload of estimateMemoryUsedByCompute for default behaviour.
    template<typename POINT>
    static std::size_t estimateMemoryUsedByCompute(bool featureSignificances,
                                                   std::size_t totalNumberPoints,
                                                   std::size_t partitionNumberPoints,
                                                   std::size_t dimension) {
        auto methodSize = [=](std::size_t k, std::size_t numberPoints,
                              std::size_t projectionDimension) {
            // On average half of models use CLof.
            return TLof<POINT>::estimateOwnMemoryOverhead(
                       featureSignificances, k, numberPoints, projectionDimension) /
                   2;
        };
        return TEnsemble<POINT>::estimateMemoryUsedToComputeOutlierScores(
            methodSize, 2 /*number methods*/, featureSignificances,
            totalNumberPoints, partitionNumberPoints, dimension);
    }

    //! \name Test Interface
    //@{
    //! Compute the normalized LOF scores for \p points.
    //!
    //! See https://en.wikipedia.org/wiki/Local_outlier_factor for details.
    //!
    //! \param[in] k The number of nearest neighbours to use.
    //! \param[in] points The points for which to compute scores.
    //! \param[out] scores The scores of \p points.
    template<typename POINT>
    static void lof(std::size_t k, std::vector<POINT> points, TDoubleVec& scores) {
        compute<outliers_detail::CLof>(k, std::move(points), scores);
    }

    //! Compute normalized local distance based outlier scores for \p points.
    //!
    //! See https://arxiv.org/pdf/0903.3257.pdf for details.
    //!
    //! \param[in] k The number of nearest neighbours to use.
    //! \param[in] points The points for which to compute scores.
    //! \param[out] scores The scores of \p points.
    template<typename POINT>
    static void ldof(std::size_t k, std::vector<POINT> points, TDoubleVec& scores) {
        compute<outliers_detail::CLdof>(k, std::move(points), scores);
    }

    //! Compute the normalized distance to the k-th nearest neighbour
    //! outlier scores for \p points.
    //!
    //! \param[in] k The number of nearest neighbours to use.
    //! \param[in] points The points for which to compute scores.
    //! \param[out] scores The scores of \p points.
    template<typename POINT>
    static void distancekNN(std::size_t k, std::vector<POINT> points, TDoubleVec& scores) {
        compute<outliers_detail::CDistancekNN>(k, std::move(points), scores);
    }

    //! Compute the normalized mean distance to the k nearest neighbours
    //! outlier scores for \p points.
    //!
    //! \param[in] k The number of nearest neighbours to use.
    //! \param[in] points The points for which to compute scores.
    //! \param[out] scores The scores of \p points.
    template<typename POINT>
    static void totalDistancekNN(std::size_t k, std::vector<POINT> points, TDoubleVec& scores) {
        compute<outliers_detail::CTotalDistancekNN>(k, std::move(points), scores);
    }
    //@}

private:
    //! Compute normalised outlier scores for a specified method.
    template<template<typename, typename> class METHOD, typename POINT>
    static void compute(std::size_t k, std::vector<POINT> points, TDoubleVec& scores) {
        if (points.size() > 0) {
            auto annotatedPoints = annotate(std::move(points));
            CKdTree<TAnnotatedPoint<POINT>> lookup;
            lookup.reserve(points.size());
            lookup.build(annotatedPoints);

            METHOD<TAnnotatedPoint<POINT>, CKdTree<TAnnotatedPoint<POINT>>> scorer{
                false, k, noop, std::move(lookup)};
            auto scores_ = scorer.run(annotatedPoints, annotatedPoints.size());

            scores.resize(scores_[0].size());
            for (std::size_t i = 0; i < scores.size(); ++i) {
                scores[i] = scores_[0][i][0];
            }
        }
    }

    //! Create points annotated with their index in \p points.
    template<typename POINT>
    static std::vector<TAnnotatedPoint<POINT>> annotate(std::vector<POINT> points) {
        std::vector<TAnnotatedPoint<POINT>> annotatedPoints;
        annotatedPoints.reserve(points.size());
        for (std::size_t i = 0; i < points.size(); ++i) {
            annotatedPoints.emplace_back(std::move(points[i]), i);
        }
        return annotatedPoints;
    }

private:
    static void noop(double);
};
}
}

#endif // INCLUDED_ml_maths_COutliers_h
