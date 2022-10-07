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

#ifndef INCLUDED_ml_maths_analytics_COutliers_h
#define INCLUDED_ml_maths_analytics_COutliers_h

#include <core/CDataFrame.h>
#include <core/CHashing.h>
#include <core/CNonInstantiatable.h>
#include <core/CSmallVector.h>
#include <core/Concurrency.h>

#include <maths/analytics/ImportExport.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CKdTree.h>
#include <maths/common/CLinearAlgebraShims.h>
#include <maths/common/COrthogonaliser.h>
#include <maths/common/CPRNG.h>
#include <maths/common/CSampling.h>
#include <maths/common/CTools.h>

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
namespace analytics {
class CDataFrameOutliersInstrumentationInterface;

namespace outliers_detail {
using TDoubleVec = std::vector<double>;
using TDouble1Vec = core::CSmallVector<double, 1>;
using TDouble2Vec = core::CSmallVector<double, 2>;
using TDouble1VecVec = std::vector<TDouble1Vec>;
using TDouble1VecVec2Vec = core::CSmallVector<TDouble1VecVec, 2>;
using TDouble1Vec2Vec = core::CSmallVector<TDouble1Vec, 2>;
using TProgressCallback = std::function<void(double)>;
using TMemoryUsageCallback = std::function<void(std::int64_t)>;
using TMeanAccumulator = common::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;

//! \brief The interface for a nearest neighbour outlier calculation method.
template<typename POINT, typename NEAREST_NEIGHBOURS>
class CNearestNeighbourMethod {
public:
    using TPointVec = std::vector<POINT>;
    using TMatrix = typename common::SConformableMatrix<POINT>::Type;

public:
    CNearestNeighbourMethod(bool computeFeatureInfluence,
                            std::size_t k,
                            NEAREST_NEIGHBOURS lookup,
                            TProgressCallback recordProgress)
        : m_ComputeFeatureInfluence{computeFeatureInfluence}, m_K{k},
          m_Lookup{std::move(lookup)}, m_RecordProgress{std::move(recordProgress)} {}
    virtual ~CNearestNeighbourMethod() = default;

    //! Check whether to compute influences of features on the outlier scores.
    bool computeFeatureInfluence() const { return m_ComputeFeatureInfluence; }

    //! The number of points.
    std::size_t n() const { return m_Lookup.size(); }

    //! The number of nearest neighbours.
    std::size_t k() const { return m_K; }

    //! Get the nearest neighbours lookup.
    const NEAREST_NEIGHBOURS& lookup() const { return m_Lookup; }

    //! Compute the outlier scores for \p points.
    TDouble1VecVec2Vec
    run(const TPointVec& points, const TMatrix& projection, double eps, std::size_t numberScores) {

        this->setup(points, projection);

        // We call add exactly once for each point. Scores is presized
        // so any writes to it are safe.
        TDouble1VecVec2Vec scores(this->numberMethods(), TDouble1VecVec(numberScores));
        core::parallel_for_each(
            points.begin(), points.end(),
            [&, neighbours = TPointVec{} ](const POINT& point) mutable {
                m_Lookup.nearestNeighbours(m_K + 1, point, neighbours);
                this->add(point, projection, eps, neighbours, scores);
            },
            [this](double fractionalProgress) {
                this->recordProgress(fractionalProgress);
            });

        this->compute(points, projection, eps, scores);

        return scores;
    }

    //! Recover any temporary memory used by run.
    virtual void recoverMemory() {}

    //! Get the size of this object.
    virtual std::size_t staticSize() const { return sizeof(*this); }

    //! Get the memory that the method uses.
    virtual std::size_t memoryUsage() const { return 0; }

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

    virtual void setup(const TPointVec&, const TMatrix&) {}
    virtual void add(const POINT&, const TMatrix&, double, const TPointVec&, TDouble1VecVec&) {
    }
    virtual void compute(const TPointVec&, const TMatrix&, double, TDouble1VecVec&) {}

protected:
    static double distanceAtEps(const TMatrix& projection,
                                std::size_t j,
                                double eps,
                                const POINT& x,
                                const POINT& y) {
        double result{0.0};
        for (std::size_t i = 0; i < common::las::dimension(x); ++i) {
            result += common::CTools::pow2(x(i) + eps * projection(i, j) - y(i));
        }
        return std::sqrt(result);
    }

private:
    virtual void add(const POINT& point,
                     const TMatrix& projection,
                     double eps,
                     const TPointVec& neighbours,
                     TDouble1VecVec2Vec& scores) {
        this->add(point, projection, eps, neighbours, scores[0]);
    }
    virtual void compute(const TPointVec& points,
                         const TMatrix& projection,
                         double eps,
                         TDouble1VecVec2Vec& scores) {
        this->compute(points, projection, eps, scores[0]);
    }
    virtual std::size_t numberMethods() const { return 1; }
    virtual std::string name() const = 0;

private:
    bool m_ComputeFeatureInfluence;
    std::size_t m_K;
    NEAREST_NEIGHBOURS m_Lookup;
    TProgressCallback m_RecordProgress;
};

//! \brief Computes the local outlier factor score.
template<typename POINT, typename NEAREST_NEIGHBOURS>
class CLof final : public CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS> {
public:
    using TPointVec = std::vector<POINT>;
    using TMatrix = typename common::SConformableMatrix<POINT>::Type;
    using TCoordinate = typename common::SCoordinate<POINT>::Type;
    using TCoordinateVec = std::vector<TCoordinate>;
    using TUInt32CoordinatePr = std::pair<std::uint32_t, TCoordinate>;
    using TUInt32CoordinatePrVec = std::vector<TUInt32CoordinatePr>;
    using CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS>::distanceAtEps;
    static const TCoordinate UNSET_DISTANCE;

public:
    CLof(bool computeFeatureInfluence,
         std::size_t k,
         TProgressCallback recordProgress,
         NEAREST_NEIGHBOURS lookup = NEAREST_NEIGHBOURS())
        : CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS>{
              computeFeatureInfluence, k, std::move(lookup), std::move(recordProgress)} {}

    void recoverMemory() override {
        m_KDistances.resize(this->k() * m_StartAddresses);
        m_KDistances.shrink_to_fit();
        m_Lrd.resize(m_StartAddresses);
        m_Lrd.shrink_to_fit();
        if (this->computeFeatureInfluence()) {
            m_LrdAtEps.resize(m_NumberInfluences * m_StartAddresses);
            m_LrdAtEps.shrink_to_fit();
        }
    }

    std::size_t staticSize() const override { return sizeof(*this); }

    std::size_t memoryUsage() const override {
        return core::memory::dynamicSize(m_KDistances) +
               core::memory::dynamicSize(m_Lrd) + core::memory::dynamicSize(m_LrdAtEps);
    }

    static std::size_t estimateOwnMemoryOverhead(bool computeFeatureInfluence,
                                                 std::size_t k,
                                                 std::size_t numberPoints,
                                                 std::size_t dimension) {
        return numberPoints *
               (k * sizeof(TUInt32CoordinatePr) +
                (computeFeatureInfluence ? dimension + 1 : 1) * sizeof(TCoordinate));
    }

private:
    void setup(const TPointVec& points, const TMatrix& projection) override {

        m_NumberInfluences = common::las::columns(projection);

        auto minmax = std::minmax_element(
            points.begin(), points.end(), [](const POINT& lhs, const POINT& rhs) {
                return lhs.annotation() < rhs.annotation();
            });
        m_StartAddresses = minmax.first->annotation();
        m_EndAddresses = minmax.second->annotation() + 1;

        // In the following, we first shrink then grow to ensure we overwrite
        // values in the range [start addresses, end addresses).

        std::size_t k{this->k()};
        m_KDistances.resize(k * m_StartAddresses, {std::uint32_t{0}, UNSET_DISTANCE});
        m_KDistances.resize(k * m_EndAddresses, {std::uint32_t{0}, UNSET_DISTANCE});
        m_Lrd.resize(m_StartAddresses, UNSET_DISTANCE);
        m_Lrd.resize(m_EndAddresses, UNSET_DISTANCE);
        if (this->computeFeatureInfluence()) {
            m_LrdAtEps.resize(m_NumberInfluences * m_StartAddresses, UNSET_DISTANCE);
            m_LrdAtEps.resize(m_NumberInfluences * m_EndAddresses, UNSET_DISTANCE);
        }
    }

    void add(const POINT& point, const TMatrix&, double, const TPointVec& neighbours, TDouble1VecVec&) override {
        // This is called exactly once for each point therefore an element
        // of m_KDistances is only ever written by one thread.
        if (neighbours.size() < 2) {
            return;
        }
        std::size_t i{point.annotation()};
        std::size_t a(point == neighbours[0] ? 1 : 0);
        std::size_t b{std::min(this->k() + a - 1, neighbours.size() + a - 2)};
        for (std::size_t j = a; j <= b; ++j) {
            std::size_t index{this->kDistanceIndex(i, j - a)};
            m_KDistances[index].first =
                static_cast<std::uint32_t>(neighbours[j].annotation());
            m_KDistances[index].second = common::las::distance(point, neighbours[j]);
        }
    }

    void compute(const TPointVec& points,
                 const TMatrix& projection,
                 double eps,
                 TDouble1VecVec& scores) override {
        this->computeLocalReachabilityDistances(points, projection, eps);
        this->computeLocalOutlierFactors(points, eps, scores);
    }

    std::string name() const override { return "lof"; }

    void computeLocalReachabilityDistances(const TPointVec& points,
                                           const TMatrix& projection,
                                           double eps) {

        // We bind minimum accumulators (by value) to each lambda (since
        // one copy is then accessed by each thread) and take the minimum
        // of these at the end.

        using TMinAccumulator = common::CBasicStatistics::SMin<double>::TAccumulator;

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
                    double reachability{common::CBasicStatistics::mean(reachability_)};
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
            // Use twice the maximum "density" at any other point if there are
            // k-fold duplicates. Note it is possible that all lookup points are
            // duplicates, in which case we need to set their local reachability
            // density in this loop. The overwritten densities are reset in setup.
            for (std::size_t i = 0; i < m_EndAddresses; ++i) {
                if (m_Lrd[i] == UNSET_DISTANCE) {
                    m_Lrd[i] = 2.0 / min[0];
                }
            }
        }

        if (eps <= 0.0 || this->computeFeatureInfluence() == false) {
            return;
        }

        // Unfortunately, we need to look up nearest neighbours again or cache
        // the distance for each coordinate perturbation for each neighbour
        // which is prohibitive.

        core::parallel_for_each(
            points.begin(), points.end(),
            [&, neighbours = TPointVec{} ](const POINT& point) mutable {
                this->lookup().nearestNeighbours(this->k() + 1, point, neighbours);
                if (neighbours.size() < 2) {
                    return;
                }
                std::size_t i{point.annotation()};
                std::size_t a(point == neighbours[0] ? 1 : 0);
                std::size_t b{std::min(this->k() + a - 1, neighbours.size() + a - 2)};
                for (std::size_t k = 0; k < m_NumberInfluences; ++k) {
                    TMeanAccumulator reachability_;
                    for (std::size_t j = a; j <= b; ++j) {
                        reachability_.add(this->reachabilityDistance(
                            {neighbours[j].annotation(),
                             distanceAtEps(projection, k, eps, point, neighbours[j])}));
                    }
                    double reachability{common::CBasicStatistics::mean(reachability_)};
                    m_LrdAtEps[this->epsLrdIndex(i, k)] =
                        1.0 / std::max(reachability, min[0] / 2.0);
                }
            });
    }

    void computeLocalOutlierFactors(const TPointVec& points, double eps, TDouble1VecVec& scores) {

        core::parallel_for_each(points.begin(), points.end(), [&](const POINT& point) mutable {

            std::size_t i{point.annotation()};

            TMeanAccumulator neighbourhoodLrd;
            for (std::size_t j = 0; j < this->k(); ++j) {
                const auto& neighbour = m_KDistances[this->kDistanceIndex(i, j)];
                if (distance(neighbour) != UNSET_DISTANCE) {
                    neighbourhoodLrd.add(m_Lrd[index(neighbour)]);
                }
            }

            scores[i].resize(eps > 0.0 && this->computeFeatureInfluence()
                                 ? m_NumberInfluences + 1
                                 : 1);
            scores[i][0] = common::CBasicStatistics::mean(neighbourhoodLrd) / m_Lrd[i];

            // We choose to ignore the impact of moving the point on its
            // neighbours' local reachability distances when computing
            // scores for each coordinate.
            for (std::size_t j = 1; j < scores[i].size(); ++j) {
                scores[i][j] = common::CBasicStatistics::mean(neighbourhoodLrd) /
                               m_LrdAtEps[this->epsLrdIndex(i, j - 1)];
            }
        });
    }

    std::size_t kDistanceIndex(std::size_t index, std::size_t neighbourIndex) const {
        return index * this->k() + neighbourIndex;
    }
    std::size_t epsLrdIndex(std::size_t index, std::size_t coordinate) const {
        return index * m_NumberInfluences + coordinate;
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
    std::size_t m_NumberInfluences;
    std::size_t m_StartAddresses;
    std::size_t m_EndAddresses;
    // The k distances to the neighbouring points of each point are stored
    // flattened: [neighbours of 0, neighbours of 1,...].
    TUInt32CoordinatePrVec m_KDistances;
    TCoordinateVec m_Lrd;
    // The epsilon local reachability distances are stored flattened:
    // [coordinates of 0, coordinates of 2, ...].
    TCoordinateVec m_LrdAtEps;
};

template<typename POINT, typename NEAREST_NEIGHBOURS>
const typename CLof<POINT, NEAREST_NEIGHBOURS>::TCoordinate
    CLof<POINT, NEAREST_NEIGHBOURS>::UNSET_DISTANCE(-1.0);

//! \brief Computes the local distance based outlier score.
template<typename POINT, typename NEAREST_NEIGHBOURS>
class CLdof final : public CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS> {
public:
    CLdof(bool computeFeatureInfluence,
          std::size_t k,
          TProgressCallback recordProgress,
          NEAREST_NEIGHBOURS lookup = NEAREST_NEIGHBOURS())
        : CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS>{
              computeFeatureInfluence, k, std::move(lookup), std::move(recordProgress)} {}

private:
    using TMatrix = typename common::SConformableMatrix<POINT>::Type;
    using CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS>::distanceAtEps;

private:
    void add(const POINT& point,
             const TMatrix& projection,
             double eps,
             const std::vector<POINT>& neighbours,
             TDouble1VecVec& scores) override {

        std::size_t numberInfluences{common::las::columns(projection)};
        auto& score = scores[point.annotation()];
        score.assign(
            eps > 0.0 && this->computeFeatureInfluence() ? numberInfluences + 1 : 1, 0.0);

        if (neighbours.size() < 2) {
            return;
        }

        auto ldof = [](const TMeanAccumulator& d, const TMeanAccumulator& D) {
            return common::CBasicStatistics::mean(D) > 0.0
                       ? common::CBasicStatistics::mean(d) /
                             common::CBasicStatistics::mean(D)
                       : 0.0;
        };

        std::size_t a(point == neighbours[0] ? 1 : 0);
        std::size_t b{std::min(this->k() + a - 1, neighbours.size() + a - 2)};

        TMeanAccumulator d;
        TMeanAccumulator D;
        for (std::size_t i = a; i <= b; ++i) {
            d.add(common::las::distance(point, neighbours[i]));
            for (std::size_t j = 1; j < i; ++j) {
                D.add(common::las::distance(neighbours[i], neighbours[j]));
            }
        }
        score[0] = ldof(d, D);

        for (std::size_t j = 1; j < score.size(); ++j) {
            d = TMeanAccumulator{};
            for (std::size_t i = a; i <= b; ++i) {
                d.add(distanceAtEps(projection, j - 1, eps, point, neighbours[i]));
            }
            score[j] = ldof(d, D);
        }
    }

    std::string name() const override { return "ldof"; }
};

//! \brief Computes the distance to the k'th nearest neighbour score.
template<typename POINT, typename NEAREST_NEIGHBOURS>
class CDistancekNN final : public CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS> {
public:
    CDistancekNN(bool computeFeatureInfluence,
                 std::size_t k,
                 TProgressCallback recordProgress,
                 NEAREST_NEIGHBOURS lookup = NEAREST_NEIGHBOURS())
        : CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS>{
              computeFeatureInfluence, k, std::move(lookup), std::move(recordProgress)} {}

private:
    using TMatrix = typename common::SConformableMatrix<POINT>::Type;
    using CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS>::distanceAtEps;

private:
    void add(const POINT& point,
             const TMatrix& projection,
             double eps,
             const std::vector<POINT>& neighbours,
             TDouble1VecVec& scores) override {

        std::size_t numberInfluences{common::las::columns(projection)};
        auto& score = scores[point.annotation()];
        score.assign(
            eps > 0.0 && this->computeFeatureInfluence() ? numberInfluences + 1 : 1, 0.0);

        if (neighbours.size() < 2) {
            return;
        }

        std::size_t k{std::min(this->k() + 1, neighbours.size() - 1) -
                      (point == neighbours[0] ? 0 : 1)};
        const auto& kthNeighbour = neighbours[k];

        score[0] = common::las::distance(point, kthNeighbour);
        for (std::size_t i = 1; i < score.size(); ++i) {
            score[i] = distanceAtEps(projection, i - 1, eps, point, kthNeighbour);
        }
    }

    std::string name() const override { return "knn"; }
};

//! \brief Computes the total distance to the k nearest neighbours score.
template<typename POINT, typename NEAREST_NEIGHBOURS>
class CTotalDistancekNN final
    : public CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS> {
public:
    CTotalDistancekNN(bool computeFeatureInfluence,
                      std::size_t k,
                      TProgressCallback recordProgress,
                      NEAREST_NEIGHBOURS lookup = NEAREST_NEIGHBOURS())
        : CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS>{
              computeFeatureInfluence, k, std::move(lookup), std::move(recordProgress)} {}

private:
    using TMatrix = typename common::SConformableMatrix<POINT>::Type;
    using CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS>::distanceAtEps;

private:
    void add(const POINT& point,
             const TMatrix& projection,
             double eps,
             const std::vector<POINT>& neighbours,
             TDouble1VecVec& scores) override {

        std::size_t numberInfluences{common::las::columns(projection)};
        auto& score = scores[point.annotation()];
        score.assign(
            eps > 0.0 && this->computeFeatureInfluence() ? numberInfluences + 1 : 1, 0.0);

        if (neighbours.size() < 2) {
            return;
        }

        std::size_t a(point == neighbours[0] ? 1 : 0);
        std::size_t b{std::min(this->k() + a - 1, neighbours.size() + a - 2)};

        for (std::size_t i = a; i <= b; ++i) {
            score[0] += common::las::distance(point, neighbours[i]);
            for (std::size_t j = 1; j < score.size(); ++j) {
                score[j] += distanceAtEps(projection, j - 1, eps, point, neighbours[i]);
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
//! IMPLEMENTATION DECISIONS:\n
//! This is used in conjunction with CEnsemble so we can share nearest
//! neighbour lookups for all methods in an ensemble model.
template<typename POINT, typename NEAREST_NEIGHBOURS>
class CMultipleMethods final
    : public CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS> {
public:
    using TPointVec = std::vector<POINT>;
    using TMatrix = typename common::SConformableMatrix<POINT>::Type;
    using TMethodUPtr = std::unique_ptr<CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS>>;
    using TMethodUPtrVec = std::vector<TMethodUPtr>;

public:
    CMultipleMethods(std::size_t k,
                     TMethodUPtrVec methods,
                     NEAREST_NEIGHBOURS lookup = NEAREST_NEIGHBOURS())
        : CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS>{methods[0]->computeFeatureInfluence(),
                                                             k, std::move(lookup),
                                                             methods[0]->progressRecorder()},
          m_Methods{std::move(methods)} {}

    void recoverMemory() override {
        for (auto& model : m_Methods) {
            model->recoverMemory();
        }
    }

    std::size_t staticSize() const override { return sizeof(*this); }

    std::size_t memoryUsage() const override {
        return core::memory::dynamicSize(m_Methods);
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

    void setup(const TPointVec& points, const TMatrix& projection) override {
        for (auto& method : m_Methods) {
            method->setup(points, projection);
        }
    }

    void add(const POINT& point,
             const TMatrix& projection,
             double eps,
             const TPointVec& neighbours,
             TDouble1VecVec2Vec& scores) override {
        for (std::size_t i = 0; i < m_Methods.size(); ++i) {
            m_Methods[i]->add(point, projection, eps, neighbours, scores[i]);
        }
    }

    void compute(const TPointVec& points,
                 const TMatrix& projection,
                 double eps,
                 TDouble1VecVec2Vec& scores) override {
        for (std::size_t i = 0; i < m_Methods.size(); ++i) {
            m_Methods[i]->compute(points, projection, eps, scores[i]);
        }
    }

    std::string name() const override { return "multiple"; }

private:
    TMethodUPtrVec m_Methods;
};
}

//! \brief Utilities for computing outlier scores for collections of points.
class MATHS_ANALYTICS_EXPORT COutliers : private core::CNonInstantiatable {
public:
    using TDoubleVec = outliers_detail::TDoubleVec;
    using TMeanVarAccumulator = common::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using TProgressCallback = outliers_detail::TProgressCallback;
    using TMemoryUsageCallback = outliers_detail::TMemoryUsageCallback;
    template<typename POINT>
    using TAnnotatedPoint = common::CAnnotatedVector<POINT, std::size_t>;

    //! \name Method Names
    //@{
    static const std::string LOF;
    static const std::string LDOF;
    static const std::string DISTANCE_KNN;
    static const std::string TOTAL_DISTANCE_KNN;
    static const std::string ENSEMBLE;
    //@}

    //! Instrumentation phase.
    static const std::string COMPUTING_OUTLIERS;
    //! Used to compute numeric derivative for influence.
    static constexpr double EPS{0.01};

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
        //! Standardize the column values before computing outlier scores.
        bool s_StandardizeColumns;
        //! The methods to use.
        EMethod s_Method;
        //! The number of neighbours to use if non-zero.
        std::size_t s_NumberNeighbours;
        //! If true also compute the feature influence.
        bool s_ComputeFeatureInfluence;
        //! The fraction of true outliers among the points.
        double s_OutlierFraction;
    };

public:
    //! Compute outliers for \p frame and write to a new column.
    //!
    //! \param[in] params The calculation parameters.
    //! \param[in] frame The data frame whose rows hold the coordinated of
    //! the points for which to compute outliers.
    //! \param[in] instrumentation Manages writing out telemetry.
    static void compute(const SComputeParameters& params,
                        core::CDataFrame& frame,
                        CDataFrameOutliersInstrumentationInterface& instrumentation);

    //! Estimate the amount of memory that will be used computing outliers
    //! for a data frame.
    //!
    //! \param[in] params The calculation parameters.
    //! \param[in] totalNumberPoints The total number of points for which
    //! outlier scores will be computed.
    //! \param[in] partitionNumberPoints The number of points per partition
    //! for which outlier scores will be computed.
    //! \param[in] dimension The dimension of the points for which outliers
    //! will be computed.
    static std::size_t estimateMemoryUsedByCompute(const SComputeParameters& params,
                                                   std::size_t totalNumberPoints,
                                                   std::size_t partitionNumberPoints,
                                                   std::size_t dimension);

    //! Return string representation of the \p method.
    static const std::string& print(EMethod method);

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
    //! Estimate the amount of memory that will be used computing outliers
    //! for a data frame using POINT point type.
    template<typename POINT>
    static std::size_t estimateMemoryUsedByCompute(const SComputeParameters& params,
                                                   std::size_t totalNumberPoints,
                                                   std::size_t partitionNumberPoints,
                                                   std::size_t dimension);

    //! Compute normalised outlier scores for a specified method.
    template<template<typename, typename> class METHOD, typename POINT>
    static void compute(std::size_t k, std::vector<POINT> points, TDoubleVec& scores) {
        using TPoint = TAnnotatedPoint<POINT>;
        using TMethod = METHOD<TPoint, common::CKdTree<TPoint>>;
        using TMatrix = typename common::SConformableMatrix<TPoint>::Type;

        if (points.size() > 0) {
            auto annotatedPoints = annotate(std::move(points));
            common::CKdTree<TAnnotatedPoint<POINT>> lookup;
            lookup.reserve(points.size());
            lookup.build(annotatedPoints);

            TMethod scorer{false, k, noopRecordProgress, std::move(lookup)};
            TMatrix projection{common::SIdentity<TMatrix>::get(
                common::las::dimension(annotatedPoints[0]))};
            auto scores_ = scorer.run(annotatedPoints, projection, EPS,
                                      annotatedPoints.size());

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
    static void noopRecordProgress(double);
    static void noopRecordMemoryUsage(std::int64_t);
};
}
}
}

#endif // INCLUDED_ml_maths_analytics_COutliers_h
