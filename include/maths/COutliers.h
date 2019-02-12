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
using TDoubleVecVec = std::vector<TDoubleVec>;
using TProgressCallback = std::function<void(double)>;
using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

//! \brief The interface for a nearest neighbour outlier calculation method.
template<typename POINT, typename NEAREST_NEIGHBOURS>
class CNearestNeighbourMethod {
public:
    using TPointVec = std::vector<POINT>;

public:
    CNearestNeighbourMethod(std::size_t k, NEAREST_NEIGHBOURS lookup, TProgressCallback recordProgress)
        : m_K{k}, m_Lookup{std::move(lookup)}, m_RecordProgress{std::move(recordProgress)} {}
    virtual ~CNearestNeighbourMethod() = default;

    //! The number of points.
    std::size_t n() const { return m_Lookup.size(); }

    //! The number of nearest neighbours.
    std::size_t k() const { return m_K; }

    //! Compute the outlier scores for \p points.
    void run(const TPointVec& points, std::size_t numberScores, TDoubleVecVec& scores) {

        this->setup(points);

        // We call add exactly once for each point. Scores is presized
        // so any writes to it are safe.
        scores = TDoubleVecVec(this->numberMethods(), TDoubleVec(numberScores, 0.0));
        core::parallel_for_each(points.begin(), points.end(),
                                [&, neighbours = TPointVec{} ](const POINT& point) mutable {
                                    m_Lookup.nearestNeighbours(m_K + 1, point, neighbours);
                                    this->add(point, neighbours, scores);
                                },
                                [this](double fractionalProgress) {
                                    this->recordProgress(fractionalProgress);
                                });

        this->compute(points, scores);
    }

    //! \name Progress Monitoring
    //@{
    //! Get the progress recorder.
    TProgressCallback& progressRecorder() { return m_RecordProgress; }
    //! Record \p fractionalProgress.
    void recordProgress(double fractionalProgress) {
        m_RecordProgress(fractionalProgress);
    }
    //@}

    virtual std::string print() const {
        return this->name() + "(n = " + std::to_string(this->n()) +
               ", k = " + std::to_string(m_K) + ")";
    }

    virtual void setup(const TPointVec&) {}
    virtual void add(const POINT&, const TPointVec&, TDoubleVec&) {}
    virtual void compute(const TPointVec&, TDoubleVec&) {}

private:
    virtual void add(const POINT& point, const TPointVec& neighbours, TDoubleVecVec& scores) {
        this->add(point, neighbours, scores[0]);
    }
    virtual void compute(const TPointVec& points, TDoubleVecVec& scores) {
        this->compute(points, scores[0]);
    }
    virtual std::size_t numberMethods() const { return 1; }
    virtual std::string name() const = 0;

private:
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
    CLof(std::size_t k,
         TProgressCallback recordProgress,
         NEAREST_NEIGHBOURS lookup = NEAREST_NEIGHBOURS())
        : CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS>{
              k, std::move(lookup), std::move(recordProgress)} {}

    //! Estimate the additional bookkeeping memory used explicitly by the
    //! local outlier factors calculation.
    static std::size_t estimateOwnMemoryOverhead(std::size_t k, std::size_t numberPoints) {
        return numberPoints * (k * sizeof(TUInt32CoordinatePr) + sizeof(TCoordinate));
    }

private:
    void setup(const TPointVec& points) override {
        std::size_t addressSpace{
            std::max_element(points.begin(), points.end(),
                             [](const POINT& lhs, const POINT& rhs) {
                                 return lhs.annotation() < rhs.annotation();
                             })
                ->annotation() +
            1};
        m_KDistances.resize(this->k() * addressSpace, {0, UNSET_DISTANCE});
        m_Lrd.resize(addressSpace, UNSET_DISTANCE);
    }

    void add(const POINT& point, const TPointVec& neighbours, TDoubleVec&) override {
        // This is called exactly once for each point therefore an element
        // of m_KDistances is only ever written by one thread.
        std::size_t i{point.annotation() * this->k()};
        std::size_t a(point == neighbours[0] ? 1 : 0);
        std::size_t b{std::min(this->k() + a - 1, neighbours.size() + a - 2)};
        for (std::size_t j = a; j <= b; ++j) {
            m_KDistances[i + j - a].first =
                static_cast<uint32_t>(neighbours[j].annotation());
            m_KDistances[i + j - a].second = las::distance(point, neighbours[j]);
        }
    }

    void compute(const TPointVec& points, TDoubleVec& scores) override {
        using TMinAccumulator = CBasicStatistics::SMin<double>::TAccumulator;

        std::size_t k{this->k()};

        // We bind a minimum accumulator (by value) to each lambda (since
        // one copy is then accessed by each thread) and take the minimum
        // of these at the end.
        auto results = core::parallel_for_each(
            points.begin(), points.end(),
            core::bindRetrievableState(
                [&](TMinAccumulator& min, const POINT& point) {
                    std::size_t i{point.annotation()};
                    TMeanAccumulator reachability_;
                    for (std::size_t j = i * k; j < (i + 1) * k; ++j) {
                        const auto& neighbour = m_KDistances[j];
                        if (distance(neighbour) != UNSET_DISTANCE) {
                            reachability_.add(std::max(kdistance(index(neighbour)),
                                                       distance(neighbour)));
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
            for (auto& lrd : m_Lrd) {
                if (lrd < 0.0) {
                    lrd = 2.0 / min[0];
                }
            }
            core::parallel_for_each(points.begin(), points.end(), [&](const POINT& point) {
                std::size_t i{point.annotation()};
                TMeanAccumulator score;
                for (std::size_t j = i * k; j < (i + 1) * k; ++j) {
                    const auto& neighbour = m_KDistances[j];
                    if (distance(neighbour) != UNSET_DISTANCE) {
                        score.add(m_Lrd[index(neighbour)]);
                    }
                }
                scores[i] = CBasicStatistics::mean(score) / m_Lrd[i];
            });
        }
    }

    std::string name() const override { return "lof"; }

    static std::size_t index(const TUInt32CoordinatePr& neighbour) {
        return neighbour.first;
    }
    static double distance(const TUInt32CoordinatePr& neighbour) {
        return neighbour.second;
    }
    double kdistance(std::size_t index) const {
        for (std::size_t k{this->k()}, j = (index + 1) * k; j > index * k; --j) {
            if (distance(m_KDistances[j - 1]) != UNSET_DISTANCE) {
                return distance(m_KDistances[j - 1]);
            }
        }
        return UNSET_DISTANCE;
    }

private:
    // The k distances to the neighbouring points of each point are
    // stored flattened: [neighbours of 0, neighbours of 1,...].
    TUInt32CoordinatePrVec m_KDistances;
    TCoordinateVec m_Lrd;
};

template<typename POINT, typename NEAREST_NEIGHBOURS>
const typename CLof<POINT, NEAREST_NEIGHBOURS>::TCoordinate
    CLof<POINT, NEAREST_NEIGHBOURS>::UNSET_DISTANCE(-1.0);

//! \brief Computes the local distance based outlier score.
template<typename POINT, typename NEAREST_NEIGHBOURS>
class CLdof final : public CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS> {
public:
    CLdof(std::size_t k,
          TProgressCallback recordProgress,
          NEAREST_NEIGHBOURS lookup = NEAREST_NEIGHBOURS())
        : CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS>{
              k, std::move(lookup), std::move(recordProgress)} {}

private:
    void add(const POINT& point, const std::vector<POINT>& neighbours, TDoubleVec& scores) override {
        TMeanAccumulator d, D;
        std::size_t a(point == neighbours[0] ? 1 : 0);
        std::size_t b{std::min(this->k() + a - 1, neighbours.size() + a - 2)};
        for (std::size_t i = a; i <= b; ++i) {
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

    std::string name() const override { return "ldof"; }
};

//! \brief Computes the distance to the k'th nearest neighbour score.
template<typename POINT, typename NEAREST_NEIGHBOURS>
class CDistancekNN final : public CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS> {
public:
    CDistancekNN(std::size_t k,
                 TProgressCallback recordProgress,
                 NEAREST_NEIGHBOURS lookup = NEAREST_NEIGHBOURS())
        : CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS>{
              k, std::move(lookup), std::move(recordProgress)} {}

private:
    void add(const POINT& point, const std::vector<POINT>& neighbours, TDoubleVec& scores) override {
        std::size_t k{std::min(this->k() + 1, neighbours.size() - 1) -
                      (point == neighbours[0] ? 0 : 1)};
        scores[point.annotation()] = las::distance(point, neighbours[k]);
    }

    std::string name() const override { return "knn"; }
};

//! \brief Computes the total distance to the k nearest neighbours score.
template<typename POINT, typename NEAREST_NEIGHBOURS>
class CTotalDistancekNN final
    : public CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS> {
public:
    CTotalDistancekNN(std::size_t k,
                      TProgressCallback recordProgress,
                      NEAREST_NEIGHBOURS lookup = NEAREST_NEIGHBOURS())
        : CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS>{
              k, std::move(lookup), std::move(recordProgress)} {}

private:
    void add(const POINT& point, const std::vector<POINT>& neighbours, TDoubleVec& scores) override {
        std::size_t i{point.annotation()};
        std::size_t a(point == neighbours[0] ? 1 : 0);
        std::size_t b{std::min(this->k() + a - 1, neighbours.size() + a - 2)};
        for (std::size_t j = a; j <= b; ++j) {
            scores[i] += las::distance(point, neighbours[j]);
        }
        scores[i] /= static_cast<double>(this->k());
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
        : CNearestNeighbourMethod<POINT, NEAREST_NEIGHBOURS>{k, std::move(lookup),
                                                             methods[0]->progressRecorder()},
          m_Methods{std::move(methods)} {}

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

    void add(const POINT& point, const TPointVec& neighbours, TDoubleVecVec& scores) override {
        for (std::size_t i = 0; i < m_Methods.size(); ++i) {
            m_Methods[i]->add(point, neighbours, scores[i]);
        }
    }

    void compute(const TPointVec& points, TDoubleVecVec& scores) override {
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
    using TPoint = CAnnotatedVector<decltype(SConstant<POINT>::get(0, 0)), std::size_t>;
    using TPointVec = std::vector<TPoint>;
    using TPointVecVec = std::vector<TPointVec>;
    using TKdTree = CKdTree<TPoint>;
    using TMethodUPtr = std::unique_ptr<CNearestNeighbourMethod<TPoint, const TKdTree&>>;
    using TMethodUPtrVec = std::vector<TMethodUPtr>;
    using TMethodFactory = std::function<TMethodUPtr(std::size_t, const TKdTree&)>;
    using TMethodFactoryVec = std::vector<TMethodFactory>;
    using TMethodSize = std::function<std::size_t(std::size_t)>;

    //! \brief Builds (online) one model of the points for the ensemble.
    class CModelBuilder {
    public:
        using TRowRef = core::CDataFrame::TRowRef;

    public:
        CModelBuilder(CPRNG::CXorOShiro128Plus& rng,
                      TSizeSizePrVec&& methodAndNumberNeighbours,
                      std::size_t sampleSize,
                      TPointVec&& projection);

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
        TPointVec m_Projection;
        TPointVec m_SampledProjectedPoints;
    };
    using TModelBuilderVec = std::vector<CModelBuilder>;

public:
    static const double SAMPLE_SIZE_SCALE;
    static const double NEIGHBOURHOOD_FRACTION;

public:
    CEnsemble(const TMethodFactoryVec& methodFactories,
              TModelBuilderVec modelBuilders,
              double priorProbabilityOutlier = 0.05);
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
    void computeOutlierScores(const std::vector<POINT>& points, TDoubleVec& scores) const;

    //! Estimate the amount of memory that will be used by the ensemble.
    static std::size_t
    estimateMemoryUsedToComputeOutlierScores(TMethodSize methodSize,
                                             std::size_t numberMethods,
                                             std::size_t totalNumberPoints,
                                             std::size_t partitionNumberPoints,
                                             std::size_t dimension) {
        std::size_t ensembleSize{
            computeEnsembleSize(numberMethods, totalNumberPoints, dimension)};
        std::size_t sampleSize{computeSampleSize(totalNumberPoints)};
        std::size_t numberModels{(ensembleSize + numberMethods - 1) / numberMethods};
        std::size_t maxNumberNeighbours{computeNumberNeighbours(sampleSize)};
        std::size_t projectionDimension{computeProjectionDimension(sampleSize, dimension)};
        std::size_t averageNumberNeighbours{(3 + maxNumberNeighbours) / 2};
        // This is "own size" + "method scores size" + "scorers size" +
        // "projected points size" + "models size".
        return sizeof(CEnsemble) +
               partitionNumberPoints *
                   (sizeof(double) + sizeof(CScorer) +
                    las::estimateMemoryUsage<TPoint>(projectionDimension)) +
               numberModels * CModel::estimateMemoryUsage(methodSize, sampleSize,
                                                          averageNumberNeighbours,
                                                          projectionDimension, dimension);
    }

    //! Get a human readable description of the ensemble.
    std::string print() const;

private:
    using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using TMeanVarAccumulatorVec = std::vector<TMeanVarAccumulator>;
    using TKdTreeUPtr = std::unique_ptr<TKdTree>;

    //! \brief Manages computing the probability that a point is an outlier
    //! given its scores from a collection of ensemble models.
    class CScorer {
    public:
        void add(const TMeanVarAccumulatorVec& logScoreMoments, const TDoubleVec& scores);
        double compute(double pOutlier) const;

    private:
        double m_EnsembleSize = 0.0;
        double m_LogLikelihoodOutlierGivenScores = 0.0;
        double m_LogLikelihoodInlierGivenScores = 0.0;
    };
    using TScorerVec = std::vector<CScorer>;

    //! \brief A model of the points used as part of the ensemble.
    class CModel {
    public:
        CModel(const TMethodFactoryVec& methodFactories,
               TSizeSizePrVec methodAndNumberNeighbours,
               TPointVec samples,
               TPointVec projection);

        std::size_t numberPoints() const { return m_Lookup->size(); }

        void proportionOfRuntimePerMethod(double proportion);

        void addOutlierScores(const std::vector<POINT>& points, TScorerVec& scores) const;

        static std::size_t estimateMemoryUsage(TMethodSize methodSize,
                                               std::size_t sampleSize,
                                               std::size_t averageNumberNeighbours,
                                               std::size_t projectionDimension,
                                               std::size_t dimension) {
            return sizeof(CModel) +
                   TKdTree::estimateMemoryUsage(sampleSize, projectionDimension) +
                   projectionDimension * las::estimateMemoryUsage<TPoint>(dimension) +
                   methodSize(averageNumberNeighbours);
        }

        std::string print() const;

    private:
        TKdTreeUPtr m_Lookup;
        TPointVec m_Projection;
        TMethodUPtr m_Method;
        TMeanVarAccumulatorVec m_LogScoreMoments;
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

    static TPointVecVec createProjections(CPRNG::CXorOShiro128Plus& rng,
                                          std::size_t numberProjections,
                                          std::size_t projectionDimension,
                                          std::size_t dimension);

    static TPoint project(const TPointVec& projection, const POINT& point);

private:
    double m_PriorProbabilityOutlier;
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
    using TDoubleVecVec = outliers_detail::TDoubleVecVec;
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

public:
    //! Compute outliers for \p frame and write to a new column.
    static void compute(std::size_t numberThreads,
                        std::size_t numberPartitions,
                        core::CDataFrame& frame,
                        TProgressCallback recordProgress = noop);

    //! Estimate the amount of memory that will be used computing outliers
    //! for a data frame.
    //!
    //! \param[in] method The method that will be used.
    //! \param[in] k The number of nearest neighbours which will be used.
    //! \param[in] totalNumberPoints The total number of points for which
    //! outlier scores will be computed.
    //! \param[in] partitionNumberPoints The number of points per partition
    //! for which outlier scores will be computed.
    //! \param[in] dimension The dimension of the points for which outliers
    //! will be computed.
    template<typename POINT>
    static std::size_t estimateComputeMemoryUsage(EMethod method,
                                                  std::size_t k,
                                                  std::size_t totalNumberPoints,
                                                  std::size_t partitionNumberPoints,
                                                  std::size_t dimension) {
        auto methodSize = [method, k, partitionNumberPoints](std::size_t) {
            return method == E_Lof
                       ? TLof<POINT>::estimateOwnMemoryOverhead(k, partitionNumberPoints)
                       : 0;
        };
        return TEnsemble<POINT>::estimateMemoryUsedToComputeOutlierScores(
            methodSize, 1, totalNumberPoints, partitionNumberPoints, dimension);
    }

    //! Overload of estimateComputeMemoryUsage for default behaviour.
    template<typename POINT>
    static std::size_t estimateComputeMemoryUsage(std::size_t totalNumberPoints,
                                                  std::size_t partitionNumberPoints,
                                                  std::size_t dimension) {
        auto methodSize = [partitionNumberPoints](std::size_t k_) {
            // On average half of models use CLof.
            return TLof<POINT>::estimateOwnMemoryOverhead(k_, partitionNumberPoints) / 2;
        };
        return TEnsemble<POINT>::estimateMemoryUsedToComputeOutlierScores(
            methodSize, 2, totalNumberPoints, partitionNumberPoints, dimension);
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
                k, noop, std::move(lookup)};

            TDoubleVecVec scores_;
            scorer.run(annotatedPoints, annotatedPoints.size(), scores_);

            scores = std::move(scores_[0]);
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
