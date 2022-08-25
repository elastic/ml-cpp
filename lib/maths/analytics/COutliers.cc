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

#include "core/CContainerPrinter.h"
#include "maths/common/CBasicStatistics.h"
#include "maths/common/CSpline.h"
#include <limits>
#include <maths/analytics/COutliers.h>

#include <core/CDataFrame.h>
#include <core/CMemoryDef.h>
#include <core/CProgramCounters.h>
#include <core/CStopWatch.h>

#include <maths/analytics/CDataFrameAnalysisInstrumentationInterface.h>
#include <maths/analytics/CDataFrameUtils.h>

#include <maths/common/CBasicStatisticsPersist.h>
#include <maths/common/CIntegration.h>
#include <maths/common/CLinearAlgebraEigen.h>
#include <maths/common/CTools.h>

#include <maths/common/CMathsFuncs.h>

#include <boost/math/distributions/lognormal.hpp>

#include <cmath>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>

namespace ml {
namespace maths {
namespace analytics {
using namespace outliers_detail;

namespace {

const std::string EMPTY_STRING;

using TRowItr = core::CDataFrame::TRowItr;
using TStepCallback = std::function<void(const std::string&)>;
using TMemoryMappedFloatVector =
    common::CMemoryMappedDenseVector<common::CFloatStorage, Eigen::Aligned16>;
using TDenseFloatVector = common::CDenseVector<common::CFloatStorage>;

double shift(double score) {
    return std::exp(-2.0) + score;
}

template<typename T>
std::int64_t signedMemoryUsage(const T& obj) {
    return static_cast<std::int64_t>(core::memory::dynamicSize(obj));
}

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
    using TMeanVarAccumulator = common::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using TMeanVarAccumulator2Vec = core::CSmallVector<TMeanVarAccumulator, 2>;
    using TPoint =
        common::CAnnotatedVector<decltype(common::SConstant<POINT>::get(0, 0)), std::size_t>;
    using TPointVec = std::vector<TPoint>;
    using TPointVecVec = std::vector<TPointVec>;
    using TKdTree = common::CKdTree<TPoint>;
    using TMatrix = typename common::SConformableMatrix<TPoint>::Type;
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
        CModelBuilder(common::CPRNG::CXorOShiro128Plus& rng,
                      TSizeSizePrVec&& methodsAndNumberNeighbours,
                      std::size_t sampleSize,
                      TMatrix&& projection);

        //! Maybe sample the point.
        void addPoint(const TRowRef& point) { m_Sampler.sample(point); }

        //! \note Only call once: this moves state into place.
        CModel make(const TMethodFactoryVec& methodFactories);

    private:
        using TSampler = common::CSampling::CReservoirSampler<TRowRef>;

    private:
        TSampler makeSampler(common::CPRNG::CXorOShiro128Plus& rng, std::size_t sampleSize);

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
                 const TDouble1Vec2Vec& scores);

        //! Compute the posterior probability that the point is an outlier
        //! and optionally the feature influence.
        TDouble1Vec compute(double pOutlier) const;

        std::size_t memoryUsage() const {
            return core::memory::dynamicSize(m_State);
        }

        static std::size_t estimateMemoryUsage(std::size_t numberInfluences) {
            return sizeof(CScorer) + (numberInfluences + 2) * sizeof(common::CFloatStorage);
        }

    private:
        using TFloat2Vec = core::CSmallVector<common::CFloatStorage, 2>;

    private:
        double ensembleSize() const { return m_State[0]; }
        common::CFloatStorage& ensembleSize() { return m_State[0]; }

        double logLikelihoodOutlier(std::size_t index) const {
            return m_State[index + 1];
        }
        common::CFloatStorage& logLikelihoodOutlier(std::size_t index) {
            return m_State[index + 1];
        }

        std::size_t numberInfluences() const { return m_State.size() - 2; }

    private:
        TFloat2Vec m_State;
    };
    using TScorerVec = std::vector<CScorer>;

public:
    static const double SAMPLE_SIZE_SCALE;
    static const double NEIGHBOURHOOD_FRACTION;

public:
    CEnsemble(const TMethodFactoryVec& methodFactories,
              TModelBuilderVec builders,
              TMemoryUsageCallback recordMemoryUsage,
              TStepCallback recordStep);
    ~CEnsemble() { m_RecordMemoryUsage(-signedMemoryUsage(m_Models)); }

    CEnsemble(const CEnsemble&) = delete;
    CEnsemble& operator=(const CEnsemble&) = delete;
    CEnsemble(CEnsemble&&) noexcept = default;
    CEnsemble& operator=(CEnsemble&&) noexcept = default;

    //! Make the builders for the ensemble models.
    static TModelBuilderVec
    makeBuilders(const TSizeVecVec& methods,
                 std::size_t numberPoints,
                 std::size_t dimension,
                 std::size_t numberNeighbours,
                 common::CPRNG::CXorOShiro128Plus rng = common::CPRNG::CXorOShiro128Plus{});

    //! Compute the outlier scores for \p points.
    TScorerVec computeOutlierScores(const std::vector<POINT>& points) const;

    //! Estimate the amount of memory that will be used by the ensemble.
    static std::size_t estimateMemoryUsage(TMethodSize methodSize,
                                           std::size_t numberMethodsPerModel,
                                           bool computeFeatureInfluence,
                                           std::size_t totalNumberPoints,
                                           std::size_t partitionNumberPoints,
                                           std::size_t dimension);

    //! Get a human readable description of the ensemble.
    std::string print() const;

private:
    using TKdTreeUPtr = std::unique_ptr<TKdTree>;

    //! \brief A model of the points used as part of the ensemble.
    class CModel {
    public:
        CModel(const TMethodFactoryVec& methodFactories,
               TSizeSizePrVec methodsAndNumberNeighbours,
               TPointVec samples,
               TMatrix projection);

        std::size_t numberPoints() const { return m_Lookup->size(); }

        void proportionOfRuntimePerMethod(double proportion);

        void addOutlierScores(const std::vector<POINT>& points,
                              TScorerVec& scores,
                              const TMemoryUsageCallback& recordMemoryUsage) const;

        std::size_t memoryUsage() const {
            return core::CMemory::dynamicSize(m_Lookup) +
                   core::CMemory::dynamicSize(m_Projection) +
                   core::CMemory::dynamicSize(m_Method) +
                   core::CMemory::dynamicSize(m_LogScoreMoments);
        }

        static std::size_t estimateMemoryUsage(TMethodSize methodSize,
                                               std::size_t sampleSize,
                                               std::size_t numberNeighbours,
                                               std::size_t projectionDimension,
                                               std::size_t dimension);

        std::string print() const;

    private:
        TKdTreeUPtr m_Lookup;
        TMatrix m_Projection;
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

    static TMatrixVec createProjections(common::CPRNG::CXorOShiro128Plus& rng,
                                        std::size_t numberProjections,
                                        std::size_t projectionDimension,
                                        std::size_t dimension);

private:
    TModelVec m_Models;
    TMemoryUsageCallback m_RecordMemoryUsage;
    TStepCallback m_RecordStep;
};

template<typename POINT>
const double CEnsemble<POINT>::SAMPLE_SIZE_SCALE{5.0};
template<typename POINT>
const double CEnsemble<POINT>::NEIGHBOURHOOD_FRACTION{0.01};

template<typename POINT>
CEnsemble<POINT>::CEnsemble(const TMethodFactoryVec& methodFactories,
                            TModelBuilderVec builders,
                            TMemoryUsageCallback recordMemoryUsage,
                            TStepCallback recordStep)
    : m_RecordMemoryUsage{std::move(recordMemoryUsage)}, m_RecordStep{std::move(recordStep)} {

    m_Models.reserve(builders.size());
    for (auto& builder : builders) {
        m_Models.push_back(builder.make(methodFactories));
    }

    for (auto& model : m_Models) {
        model.proportionOfRuntimePerMethod(1.0 / static_cast<double>(m_Models.size()));
    }

    m_RecordMemoryUsage(core::memory::dynamicSize(m_Models));
}

template<typename POINT>
typename CEnsemble<POINT>::TModelBuilderVec
CEnsemble<POINT>::makeBuilders(const TSizeVecVec& methods,
                               std::size_t numberPoints,
                               std::size_t dimension,
                               std::size_t numberNeighbours,
                               common::CPRNG::CXorOShiro128Plus rng) {
    // Compute some constants of the ensemble:
    //   - The number of ensemble models, which is proportional n^(1/2),
    //   - The number of points which are sampled for each model, which is also
    //     proportional to the n^(1/2),
    //   - The maximum number of nearest neighbours, which is a fraction of the
    //     model sample size and
    //   - The dimension of the projected points, which is proportional to log(n)
    //     (to ensure there is sufficient data) and d^(1/2)
    std::size_t ensembleSize{computeEnsembleSize(methods.size(), numberPoints, dimension)};
    std::size_t sampleSize{computeSampleSize(numberPoints)};
    std::size_t numberModels{(ensembleSize + methods.size() - 1) / methods.size()};
    std::size_t minNumberNeighbours{numberNeighbours > 0 ? numberNeighbours : 3};
    std::size_t maxNumberNeighbours{
        numberNeighbours > 0 ? numberNeighbours : computeNumberNeighbours(sampleSize)};
    std::size_t projectionDimension{computeProjectionDimension(sampleSize, dimension)};

    TModelBuilderVec result;
    result.reserve(numberModels);

    TMatrixVec projections(createProjections(rng, numberModels, projectionDimension, dimension));

    for (std::size_t i = 0; i < projections.size(); ++i) {
        // Multiple algorithms are used for the same model which therefore share
        // the sampled points. This allows us to save on both memory and runtime
        // (because we can share the nearest neighbour lookup for the different
        // algorithms).
        TSizeSizePrVec methodsAndNumberNeighbours;
        methodsAndNumberNeighbours.reserve(methods.size());
        for (const auto& method : methods) {
            methodsAndNumberNeighbours.emplace_back(
                method[common::CSampling::uniformSample(rng, 0, methods.size())],
                common::CSampling::uniformSample(rng, minNumberNeighbours,
                                                 maxNumberNeighbours + 1));
        }

        result.emplace_back(rng, std::move(methodsAndNumberNeighbours),
                            sampleSize, std::move(projections[i]));

        rng.jump();
    }

    return result;
}

template<typename POINT>
typename CEnsemble<POINT>::TScorerVec
CEnsemble<POINT>::computeOutlierScores(const std::vector<POINT>& points) const {
    if (points.empty()) {
        return {};
    }

    LOG_TRACE(<< "Computing outlier scores for\n" << this->print());

    TScorerVec scores(points.size());
    m_RecordMemoryUsage(core::memory::dynamicSize(scores));

    for (const auto& model : m_Models) {
        model.addOutlierScores(points, scores, m_RecordMemoryUsage);
    }
    return scores;
}

template<typename POINT>
std::size_t CEnsemble<POINT>::estimateMemoryUsage(TMethodSize methodSize,
                                                  std::size_t numberMethodsPerModel,
                                                  bool computeFeatureInfluence,
                                                  std::size_t totalNumberPoints,
                                                  std::size_t partitionNumberPoints,
                                                  std::size_t dimension) {
    std::size_t ensembleSize{computeEnsembleSize(numberMethodsPerModel,
                                                 totalNumberPoints, dimension)};
    std::size_t sampleSize{computeSampleSize(totalNumberPoints)};
    std::size_t numberModels{(ensembleSize + numberMethodsPerModel - 1) / numberMethodsPerModel};
    std::size_t maxNumberNeighbours{computeNumberNeighbours(sampleSize)};
    std::size_t projectionDimension{computeProjectionDimension(sampleSize, dimension)};

    std::size_t pointsMemory{
        partitionNumberPoints *
        (sizeof(TPoint) + common::las::estimateMemoryUsage<TPoint>(dimension))};
    std::size_t scorersMemory{
        partitionNumberPoints *
        CScorer::estimateMemoryUsage(computeFeatureInfluence ? dimension : 0)};
    std::size_t modelMemory{CModel::estimateMemoryUsage(
        methodSize, sampleSize, maxNumberNeighbours, projectionDimension, dimension)};
    // The scores for a single method plus bookkeeping overhead for a single partition.
    std::size_t partitionScoringMemory{
        numberMethodsPerModel * partitionNumberPoints *
            (sizeof(TDouble1Vec) + (computeFeatureInfluence ? dimension * sizeof(double) : 0)) +
        methodSize(maxNumberNeighbours, partitionNumberPoints, dimension)};

    return pointsMemory + scorersMemory + numberModels * modelMemory + partitionScoringMemory;
}

template<typename POINT>
std::string CEnsemble<POINT>::print() const {
    std::ostringstream result;
    result << "ensemble: {";
    for (const auto& model : m_Models) {
        result << "\n  " << model.print();
    }
    result << "\n}";
    return result.str();
}

template<typename POINT>
typename CEnsemble<POINT>::TMatrixVec
CEnsemble<POINT>::createProjections(common::CPRNG::CXorOShiro128Plus& rng,
                                    std::size_t numberProjections,
                                    std::size_t projectionDimension,
                                    std::size_t dimension) {

    LOG_TRACE(<< "# projections = " << numberProjections << ", dimension = " << dimension
              << ", projection dimension = " << projectionDimension);

    TMatrixVec result;
    result.reserve(numberProjections);

    if (projectionDimension < dimension) {

        // We create bags of random projections that are orthogonalised together.
        // This gives us much better diversity across the models in the ensemble.
        std::size_t bag{dimension / projectionDimension};
        bag = projectionDimension * std::min(bag, numberProjections);
        LOG_TRACE(<< "# projections in bag = " << bag);

        // Use standard 2-stable Gaussian projections since we are interested
        // in Euclidean distances.
        TDoubleVec coordinates;
        std::size_t target{2 * numberProjections * projectionDimension * dimension};
        common::CSampling::normalSample(
            rng, 0.0, 1.0, bag * ((target + bag - 1) / bag), coordinates);
        LOG_TRACE(<< "# random samples = " << coordinates.size());

        for (auto coordinate = coordinates.begin();
             bag > 0 && dimension > 0 && result.size() < numberProjections &&
             coordinate != coordinates.end();
             /**/) {

            TPointVec projection{bag, common::SConstant<TPoint>::get(dimension, 0)};

            for (std::size_t i = 0; i < bag; ++i) {
                for (std::size_t j = 0; j < dimension; ++j, ++coordinate) {
                    projection[i](j) = *coordinate;
                }
            }

            // Orthogonalise the projection. This resizes the it if there are
            // linear dependencies.
            if (common::COrthogonaliser::orthonormalBasis(projection)) {

                std::size_t n{std::min(numberProjections - result.size(),
                                       projection.size() / projectionDimension)};

                for (std::size_t i = 0; i < n; ++i) {
                    result.emplace_back(projectionDimension, dimension);
                    for (std::size_t j = 0; j < projectionDimension; ++j) {
                        std::size_t index{projectionDimension * i + j};
                        result.back().row(j) = std::move(projection[index]);
                    }
                }
            }
        }
    } else {
        // Identity matices.
        result.resize(numberProjections, TMatrix::Identity(dimension, dimension));
    }

    return result;
}

template<typename POINT>
CEnsemble<POINT>::CModelBuilder::CModelBuilder(common::CPRNG::CXorOShiro128Plus& rng,
                                               TSizeSizePrVec&& methodsAndNumberNeighbours,
                                               std::size_t sampleSize,
                                               TMatrix&& projection)
    : m_MethodsAndNumberNeighbours{std::forward<TSizeSizePrVec>(methodsAndNumberNeighbours)},
      m_SampleSize{sampleSize}, m_Sampler{makeSampler(rng, sampleSize)},
      m_Projection{std::forward<TMatrix>(projection)} {

    m_SampledProjectedPoints.reserve(m_Sampler.targetSampleSize());
}

template<typename POINT>
typename CEnsemble<POINT>::CModel
CEnsemble<POINT>::CModelBuilder::make(const TMethodFactoryVec& methodFactories) {

    if (m_SampledProjectedPoints.size() > m_SampleSize) {
        // We want a random subset of size m_SampleSize. Randomly permute and
        // grab the first m_SampleSize elements is equivalent.
        common::CSampling::random_shuffle(m_Sampler.rng(),
                                          m_SampledProjectedPoints.begin(),
                                          m_SampledProjectedPoints.end());
        m_SampledProjectedPoints.resize(m_SampleSize);
    }

    for (std::size_t i = 0; i < m_SampledProjectedPoints.size(); ++i) {
        m_SampledProjectedPoints[i].annotation() = i;
    }

    return {methodFactories, std::move(m_MethodsAndNumberNeighbours),
            std::move(m_SampledProjectedPoints), std::move(m_Projection)};
}

template<typename POINT>
typename CEnsemble<POINT>::CModelBuilder::TSampler
CEnsemble<POINT>::CModelBuilder::makeSampler(common::CPRNG::CXorOShiro128Plus& rng,
                                             std::size_t sampleSize) {
    auto onSample = [this](std::size_t index, const TRowRef& row) {
        if (index >= m_SampledProjectedPoints.size()) {
            m_SampledProjectedPoints.emplace_back(
                m_Projection * CDataFrameUtils::rowTo<POINT>(row));
        } else {
            m_SampledProjectedPoints[index] = m_Projection *
                                              CDataFrameUtils::rowTo<POINT>(row);
        }
    };
    return {sampleSize, onSample, rng};
}

template<typename POINT>
void CEnsemble<POINT>::CScorer::add(const TMeanVarAccumulator2Vec& logScoreMoments,
                                    const TDouble1Vec2Vec& scores) {
    if (scores.empty()) {
        return;
    }

    // The basic idea is to map score percentile to a probability that the data
    // point is an outlier. We use a monotonic function for this which is flat
    // near the median, doesn't become too small for low percentiles and is
    // relatively sensitive for very high percentiles. Even then we don't become
    // too confident based on a single score. This could be explicitly calibrated
    // using the slope of the ROC curve, but one would need a very good collection
    // of benchmark sets to make this effective a priori. We also account for the
    // fact that the scores will be somewhat correlated.

    // We lower bound the coefficient of variation (to 1e-4) of the log-normal we
    // use to compute the score c.d.f. This is log(1 + CV^2) which for small CV is
    // very nearly CV^2.
    constexpr double MINIMUM_VARIANCE{1e-8};

    auto scoreCdfComplement = [&](std::size_t i, std::size_t j) {
        double location{common::CBasicStatistics::mean(logScoreMoments[i])};
        double scale{std::sqrt(std::max(
            common::CBasicStatistics::variance(logScoreMoments[i]), MINIMUM_VARIANCE))};
        try {
            boost::math::lognormal lognormal{location, scale};
            return common::CTools::safeCdfComplement(lognormal, shift(scores[i][j]));
        } catch (const std::exception& e) {
            // In this case, we use the initial value of 0.5 for the cdfComplement
            // which means P(outlier | score) = P(inlier | score) = 0.5. The outcome
            // is the score conveys no information about whether or not a point is
            // an outlier, it is effectively ignored. The rationale for keeping going
            // therefore is that this handling is good enough that the results may
            // still be useful.
            LOG_WARN(<< "Failed to normalise scores: '" << e.what()
                     << "'. Results maybe compromised.");
        }
        return 0.5;
    };

    auto pOutlierGiven = [] {
        static const TDoubleVec LOG_KNOTS{
            common::CTools::fastLog(1e-5), common::CTools::fastLog(1e-3),
            common::CTools::fastLog(0.01), common::CTools::fastLog(0.1),
            common::CTools::fastLog(0.2),  common::CTools::fastLog(0.4),
            common::CTools::fastLog(0.5),  common::CTools::fastLog(0.6),
            common::CTools::fastLog(0.8),  common::CTools::fastLog(1.0)};
        static const TDoubleVec KNOTS_P_OUTLIER{0.98, 0.87, 0.76, 0.65, 0.6,
                                                0.5,  0.5,  0.5,  0.3,  0.3};

        return [&](double cdfComplement) {
            double logCdfComplement{common::CTools::fastLog(
                std::max(cdfComplement, std::numeric_limits<double>::min()))};
            return P_OUTLIER.value(logCdfComplement);
        };
    }();

    static const double EXPECTED_P_OUTLIER{[=] {
        double result{0.0};
        for (double x = 0.0, dx = 0.05; x < 0.99; x += dx) {
            double interval;
            common::CIntegration::gaussLegendre<common::CIntegration::OrderTwo>(
                [=](double x_, double& r) {
                    r = pOutlierGiven(x_);
                    return true;
                },
                x, x + dx, interval);
            result += interval;
        }
        return result;
    }()};

    m_State.resize(scores[0].size() + 1, 0.0);

    this->ensembleSize() += 1.0;

    for (std::size_t j = 0; j < scores[0].size(); ++j) {
        double logLikelihoodOutlier{0.0};
        double logLikelihoodInlier{0.0};
        for (std::size_t i = 0; i < scores.size(); ++i) {
            double pOutlier{0.5 / EXPECTED_P_OUTLIER *
                            pOutlierGiven(scoreCdfComplement(i, j))};
            logLikelihoodOutlier += std::log(pOutlier);
            logLikelihoodInlier += std::log(1.0 - pOutlier);
        }
        double likelihoodOutlier{std::exp(logLikelihoodOutlier)};
        double likelihoodInlier{std::exp(logLikelihoodInlier)};
        double pOutlier{likelihoodOutlier / (likelihoodInlier + likelihoodOutlier)};
        double pInlier{1.0 - pOutlier};
        this->logLikelihoodOutlier(j) += std::log(pOutlier) - std::log(pInlier);
    }
}

template<typename POINT>
TDouble1Vec CEnsemble<POINT>::CScorer::compute(double pOutlier) const {

    // For large ensemble sizes, the classification becomes unrealistically
    // confident. For a high degree of confidence we instead require that the
    // point has a significant chance of being an outlier for a significant
    // fraction of the ensemble models, i.e. 4 out of 5.

    TDouble1Vec result(this->numberInfluences() + 1, 0.0);
    double Z{0.0};
    for (std::size_t i = 0; i < this->numberInfluences() + 1; ++i) {
        double logLikelihoodOutlier{this->logLikelihoodOutlier(i) /
                                    (0.8 * this->ensembleSize())};
        // The conditional probability follows from Bayes rule.
        double likelihoodOutlier{
            std::exp(logLikelihoodOutlier + common::CTools::fastLog(pOutlier))};
        double likelihoodInlier{std::exp(common::CTools::fastLog(1.0 - pOutlier))};
        result[i] = likelihoodOutlier / (likelihoodOutlier + likelihoodInlier);
        Z += std::fabs(result[i] - result[0]);
    }

    // The normalised feature influence.
    for (std::size_t i = 1; i < result.size(); ++i) {
        result[i] = Z == 0.0 ? 1.0 / static_cast<double>(result.size() - 1)
                             : std::fabs(result[i] - result[0]) / Z;
    }

    return result;
}

template<typename POINT>
CEnsemble<POINT>::CModel::CModel(const TMethodFactoryVec& methodFactories,
                                 TSizeSizePrVec methodsAndNumberNeighbours,
                                 TPointVec sample,
                                 TMatrix projection)
    : m_Lookup{std::make_unique<TKdTree>()}, m_Projection{std::move(projection)} {

    m_Lookup->reserve(sample.size());
    m_Lookup->build(sample);

    TMethodUPtrVec methods;
    methods.reserve(methodsAndNumberNeighbours.size());
    std::size_t maxk{0};

    for (const auto& methodAndNumberNeighbours : methodsAndNumberNeighbours) {
        std::size_t method;
        std::size_t k;
        std::tie(method, k) = methodAndNumberNeighbours;
        methods.push_back(std::move(methodFactories[method](k, *m_Lookup)));
        maxk = std::max(maxk, k);
    }

    m_LogScoreMoments.reserve(methods.size());

    TProgressCallback noop{[](double) {}};

    for (auto& method : methods) {
        method->progressRecorder().swap(noop);
        TDouble1VecVec2Vec scores(method->run(sample, m_Projection, 0.0, sample.size()));
        method->progressRecorder().swap(noop);

        m_LogScoreMoments.emplace_back();
        for (auto& score : scores[0]) {
            m_LogScoreMoments.back().add(common::CTools::fastLog(shift(score[0])));
        }
    }
    m_Method = std::make_unique<CMultipleMethods<TPoint, const TKdTree&>>(
        maxk, std::move(methods), *m_Lookup);
}

template<typename POINT>
void CEnsemble<POINT>::CModel::proportionOfRuntimePerMethod(double proportion) {
    TProgressCallback recordProgress{m_Method->progressRecorder()};
    m_Method->progressRecorder() = [proportion, recordProgress](double progress) {
        recordProgress(proportion * progress);
    };
}

template<typename POINT>
void CEnsemble<POINT>::CModel::addOutlierScores(const std::vector<POINT>& points,
                                                TScorerVec& scores,
                                                const TMemoryUsageCallback& recordMemoryUsage) const {
    // This index is used for addressing an array in the cache of nearest neighbour
    // distances for the local outlier factor method. We simply need to ensure that
    // it doesn't overlap the indices of any of the sampled model points.
    std::size_t index{this->numberPoints()};

    TPointVec points_;
    points_.reserve(points.size());
    TMeanAccumulator meanNorm;
    for (std::size_t i = 0; i < points.size(); ++i, ++index) {
        points_.emplace_back(m_Projection * points[i], index);
        meanNorm.add(common::las::norm(points[i]));
    }
    double eps{COutliers::EPS * common::CBasicStatistics::mean(meanNorm)};
    LOG_TRACE(<< "eps = " << eps);

    std::int64_t pointsMemory{signedMemoryUsage(points_)};
    recordMemoryUsage(pointsMemory);
    std::int64_t methodMemoryBeforeRun{signedMemoryUsage(m_Method)};

    // Run the method.
    TDouble1VecVec2Vec methodScores(m_Method->run(points_, m_Projection, eps, index));

    std::int64_t methodMemoryAfterRun{signedMemoryUsage(m_Method)};
    recordMemoryUsage(methodMemoryAfterRun - methodMemoryBeforeRun);

    // Recover temporary memory.
    m_Method->recoverMemory();

    recordMemoryUsage(signedMemoryUsage(m_Method) - methodMemoryAfterRun);
    std::int64_t scoresMemoryBeforeAdd{signedMemoryUsage(scores)};

    // Update the scores.
    TDouble1Vec2Vec pointScores(methodScores.size());
    for (std::size_t i = 0; i < points_.size(); ++i) {
        index = points_[i].annotation();
        for (std::size_t j = 0; j < methodScores.size(); ++j) {
            pointScores[j] = std::move(methodScores[j][index]);
        }
        scores[i].add(m_LogScoreMoments, pointScores);
    }

    recordMemoryUsage(signedMemoryUsage(scores) - scoresMemoryBeforeAdd - pointsMemory);
}

template<typename POINT>
std::size_t CEnsemble<POINT>::CModel::estimateMemoryUsage(TMethodSize methodSize,
                                                          std::size_t sampleSize,
                                                          std::size_t numberNeighbours,
                                                          std::size_t projectionDimension,
                                                          std::size_t dimension) {
    std::size_t lookupMemory{TKdTree::estimateMemoryUsage(sampleSize, projectionDimension)};
    std::size_t projectionMemory{projectionDimension * dimension *
                                 sizeof(typename common::SCoordinate<TPoint>::Type)};
    return sizeof(CModel) + lookupMemory + projectionMemory +
           methodSize(numberNeighbours, sampleSize, dimension);
}

template<typename POINT>
std::string CEnsemble<POINT>::CModel::print() const {
    return "projection = " + std::to_string(m_Projection.rows()) + " x " +
           std::to_string(m_Projection.cols()) + " method = " + m_Method->print() +
           " score moments = " + core::CContainerPrinter::print(m_LogScoreMoments);
}

template<typename POINT>
typename CEnsemble<POINT>::TMethodFactoryVec
methodFactories(bool computeFeatureInfluence, TProgressCallback recordProgress) {

    using TPoint = typename CEnsemble<POINT>::TPoint;
    using TKdTree = typename CEnsemble<POINT>::TKdTree;

    typename CEnsemble<POINT>::TMethodFactoryVec result;
    result.reserve(4);

    result.emplace_back([=](std::size_t k, const TKdTree& lookup) {
        return std::make_unique<CLof<TPoint, const TKdTree&>>(
            computeFeatureInfluence, k, recordProgress, lookup);
    });
    result.emplace_back([=](std::size_t k, const TKdTree& lookup) {
        return std::make_unique<CLdof<TPoint, const TKdTree&>>(
            computeFeatureInfluence, k, recordProgress, lookup);
    });
    result.emplace_back([=](std::size_t k, const TKdTree& lookup) {
        return std::make_unique<CDistancekNN<TPoint, const TKdTree&>>(
            computeFeatureInfluence, k, recordProgress, lookup);
    });
    result.emplace_back([=](std::size_t k, const TKdTree& lookup) {
        return std::make_unique<CTotalDistancekNN<TPoint, const TKdTree&>>(
            computeFeatureInfluence, k, recordProgress, lookup);
    });

    return result;
}

template<typename POINT>
CEnsemble<POINT> buildEnsemble(const COutliers::SComputeParameters& params,
                               core::CDataFrame& frame,
                               TProgressCallback recordProgress,
                               TMemoryUsageCallback recordMemoryUsage,
                               TStepCallback recordStep) {

    using TSizeVec = typename CEnsemble<POINT>::TSizeVec;
    using TSizeVecVec = typename CEnsemble<POINT>::TSizeVecVec;

    TSizeVecVec methods;
    methods.reserve(2);
    if (params.s_Method == COutliers::E_Ensemble) {
        methods.push_back(TSizeVec{static_cast<std::size_t>(COutliers::E_Lof),
                                   static_cast<std::size_t>(COutliers::E_Ldof)});
        methods.push_back(TSizeVec{static_cast<std::size_t>(COutliers::E_DistancekNN),
                                   static_cast<std::size_t>(COutliers::E_TotalDistancekNN)});
    } else {
        methods.push_back(TSizeVec{static_cast<std::size_t>(params.s_Method)});
    }

    auto builders = CEnsemble<POINT>::makeBuilders(
        methods, frame.numberRows(), frame.numberColumns(), params.s_NumberNeighbours);

    frame.readRows(1, [&builders](const TRowItr& beginRows, const TRowItr& endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            for (auto& builder : builders) {
                builder.addPoint(*row);
            }
        }
    });

    return CEnsemble<POINT>{
        methodFactories<POINT>(params.s_ComputeFeatureInfluence, std::move(recordProgress)),
        std::move(builders), std::move(recordMemoryUsage), std::move(recordStep)};
}

bool computeOutliersNoPartitions(const COutliers::SComputeParameters& params,
                                 core::CDataFrame& frame,
                                 CDataFrameAnalysisInstrumentationInterface& instrumentation) {

    using TPoint = TMemoryMappedFloatVector;
    using TPointVec = std::vector<TPoint>;

    CEnsemble<TPoint>::TScorerVec scores;

    // Use scoping to recover memory before resizing the data frame.
    {
        core::CStopWatch watch{true};
        CEnsemble<TPoint> ensemble{buildEnsemble<TPoint>(
            params, frame, instrumentation.progressCallback(),
            instrumentation.memoryUsageCallback(), instrumentation.flushCallback())};
        LOG_TRACE(<< "Ensemble = " << ensemble.print());
        core::CProgramCounters::counter(counter_t::E_DFOTimeToCreateEnsemble) =
            watch.stop();

        // The points will be entirely overwritten by readRows so the initial value
        // is not important. This is presized so that rowsToPoints only needs to
        // access and write to each element. Since it does this once per element it
        // is thread safe.
        TPointVec points(frame.numberRows(), TPoint{nullptr, 1});
        std::int64_t pointsMemory{signedMemoryUsage(points)};
        instrumentation.updateMemoryUsage(pointsMemory);

        auto rowsToPoints = [&points](const TRowItr& beginRows, const TRowItr& endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                points[row->index()] = CDataFrameUtils::rowTo<TPoint>(*row);
            }
        };

        std::uint64_t checksum{frame.checksum()};

        bool successful;
        std::tie(std::ignore, successful) = frame.readRows(params.s_NumberThreads, rowsToPoints);

        if (successful == false) {
            LOG_ERROR(<< "Failed to read the data frame");
            return false;
        }

        watch.reset(true);
        scores = ensemble.computeOutlierScores(points);
        core::CProgramCounters::counter(counter_t::E_DFOTimeToComputeScores) =
            watch.stop();

        instrumentation.updateMemoryUsage(-pointsMemory);

        // This is a sanity check against CEnsemble accidentally writing to the data
        // frame via one of the memory mapped vectors. All bets are off as to whether
        // we generate anything meaningful if this happens.
        if (checksum != frame.checksum()) {
            LOG_ERROR(<< "Accidentally modified the data frame");
            return false;
        }
    }

    std::size_t numberColumns{frame.numberColumns()};

    auto writeScores = [&](const TRowItr& beginRows, const TRowItr& endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            std::size_t index{numberColumns};
            for (auto value : scores[row->index()].compute(params.s_OutlierFraction)) {
                row->writeColumn(index++, value);
            }
        }
    };

    std::int64_t frameMemory{signedMemoryUsage(frame)};
    frame.resizeColumns(params.s_NumberThreads,
                        (params.s_ComputeFeatureInfluence ? 2 : 1) * numberColumns + 1);
    instrumentation.updateMemoryUsage(signedMemoryUsage(frame) - frameMemory);

    bool successful;
    std::tie(std::ignore, successful) = frame.writeColumns(params.s_NumberThreads, writeScores);
    if (successful == false) {
        LOG_ERROR(<< "Failed to write scores to the data frame");
        return false;
    }
    return true;
}

bool computeOutliersPartitioned(const COutliers::SComputeParameters& params,
                                core::CDataFrame& frame,
                                CDataFrameAnalysisInstrumentationInterface& instrumentation) {

    using TPoint = TDenseFloatVector;
    using TPointVec = std::vector<TPoint>;

    core::CStopWatch watch{true};
    CEnsemble<TPoint> ensemble{buildEnsemble<TPoint>(
        params, frame,
        [&](double progress) {
            instrumentation.updateProgress(
                progress / static_cast<double>(params.s_NumberPartitions));
        },
        instrumentation.memoryUsageCallback(), instrumentation.flushCallback())};
    core::CProgramCounters::counter(counter_t::E_DFOTimeToCreateEnsemble) =
        watch.stop();
    LOG_TRACE(<< "Ensemble = " << ensemble.print());

    std::size_t dimension{frame.numberColumns()};
    std::int64_t frameMemory{signedMemoryUsage(frame)};
    frame.resizeColumns(params.s_NumberThreads,
                        (params.s_ComputeFeatureInfluence ? 2 : 1) * dimension + 1);
    instrumentation.updateMemoryUsage(signedMemoryUsage(frame) - frameMemory);

    std::size_t rowsPerPartition{(frame.numberRows() + params.s_NumberPartitions - 1) /
                                 params.s_NumberPartitions};
    LOG_TRACE(<< "# rows = " << frame.numberRows()
              << ", # partitions = " << params.s_NumberPartitions
              << ", # rows per partition = " << rowsPerPartition);

    // This is presized so that rowsToPoints only needs to access and write to
    // each element. Since it does this once per element it is thread safe.
    TPointVec points(rowsPerPartition, common::SConstant<TPoint>::get(dimension, 0));
    instrumentation.updateMemoryUsage(signedMemoryUsage(points));

    for (std::size_t i = 0, beginPartitionRows = 0; i < params.s_NumberPartitions;
         ++i, beginPartitionRows += rowsPerPartition) {

        std::size_t endPartitionRows{beginPartitionRows + rowsPerPartition};
        LOG_TRACE(<< "rows [" << beginPartitionRows << "," << endPartitionRows << ")");
        points.resize(std::min(rowsPerPartition, frame.numberRows() - beginPartitionRows));

        auto rowsToPoints = [beginPartitionRows, dimension, &points](
                                const TRowItr& beginRows, const TRowItr& endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                for (std::size_t j = 0; j < dimension; ++j) {
                    points[row->index() - beginPartitionRows](j) = (*row)[j];
                }
            }
        };

        bool successful;
        std::tie(std::ignore, successful) = frame.readRows(
            params.s_NumberThreads, beginPartitionRows, endPartitionRows, rowsToPoints);

        if (successful == false) {
            LOG_ERROR(<< "Failed to read the data frame");
            return false;
        }

        watch.reset(true);
        auto scores = ensemble.computeOutlierScores(points);
        core::CProgramCounters::counter(counter_t::E_DFOTimeToComputeScores) +=
            watch.stop();

        auto writeScores = [&](const TRowItr& beginRows, const TRowItr& endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                std::size_t offset{row->index() - beginPartitionRows};
                std::size_t index{dimension};
                for (auto value : scores[offset].compute(params.s_OutlierFraction)) {
                    row->writeColumn(index++, value);
                }
            }
        };

        std::tie(std::ignore, successful) = frame.writeColumns(
            params.s_NumberThreads, beginPartitionRows, endPartitionRows, writeScores);
        if (successful == false) {
            LOG_ERROR(<< "Failed to write scores to the data frame");
            return false;
        }

        instrumentation.updateMemoryUsage(-signedMemoryUsage(scores));
    }

    return true;
}
}

void COutliers::compute(const SComputeParameters& params,
                        core::CDataFrame& frame,
                        CDataFrameOutliersInstrumentationInterface& instrumentation) {

    instrumentation.startNewProgressMonitoredTask(COMPUTING_OUTLIERS);
    instrumentation.parameters(params);

    core::CStopWatch stopWatch;
    stopWatch.start();
    std::uint64_t startTime{stopWatch.lap()};

    if (params.s_StandardizeColumns) {
        CDataFrameUtils::standardizeColumns(params.s_NumberThreads, frame);
    }

    bool successful{frame.inMainMemory() && params.s_NumberPartitions == 1
                        ? computeOutliersNoPartitions(params, frame, instrumentation)
                        : computeOutliersPartitioned(params, frame, instrumentation)};

    std::uint64_t elapsedTime{stopWatch.lap() - startTime};
    instrumentation.elapsedTime(elapsedTime);
    instrumentation.flush(COMPUTING_OUTLIERS);

    if (successful == false) {
        HANDLE_FATAL(<< "Internal error: computing outliers for data frame. There "
                     << "may be more details in the logs. Please report this problem.");
    }
}

std::size_t COutliers::estimateMemoryUsedByCompute(const SComputeParameters& params,
                                                   std::size_t totalNumberPoints,
                                                   std::size_t partitionNumberPoints,
                                                   std::size_t dimension) {
    return params.s_NumberPartitions == 1
               ? COutliers::estimateMemoryUsedByCompute<TMemoryMappedFloatVector>(
                     params, totalNumberPoints, partitionNumberPoints, dimension)
               : COutliers::estimateMemoryUsedByCompute<TDenseFloatVector>(
                     params, totalNumberPoints, partitionNumberPoints, dimension);
}

template<typename POINT>
std::size_t COutliers::estimateMemoryUsedByCompute(const SComputeParameters& params,
                                                   std::size_t totalNumberPoints,
                                                   std::size_t partitionNumberPoints,
                                                   std::size_t dimension) {
    using TLof = CLof<TAnnotatedPoint<POINT>, common::CKdTree<TAnnotatedPoint<POINT>>>;

    auto methodSize = [=](std::size_t k, std::size_t numberPoints, std::size_t dimension) {

        k = params.s_NumberNeighbours > 0 ? params.s_NumberNeighbours : k;

        if (params.s_Method == E_Ensemble) {
            // On average half of models use CLof.
            return TLof::estimateOwnMemoryOverhead(params.s_ComputeFeatureInfluence,
                                                   k, numberPoints, dimension) /
                   2;
        }
        if (params.s_Method == E_Lof) {
            return TLof::estimateOwnMemoryOverhead(params.s_ComputeFeatureInfluence,
                                                   k, numberPoints, dimension);
        }
        return std::size_t{0};
    };
    return CEnsemble<POINT>::estimateMemoryUsage(
        methodSize, params.s_Method == E_Ensemble ? 2 : 1 /*number methods*/,
        params.s_ComputeFeatureInfluence, totalNumberPoints,
        partitionNumberPoints, dimension);
}

void COutliers::noopRecordProgress(double) {
}

void COutliers::noopRecordMemoryUsage(std::int64_t) {
}

const std::string& COutliers::print(EMethod method) {
    switch (method) {
    case E_Lof:
        return LOF;
    case E_Ldof:
        return LDOF;
    case E_DistancekNN:
        return DISTANCE_KNN;
    case E_TotalDistancekNN:
        return TOTAL_DISTANCE_KNN;
    case E_Ensemble:
        return ENSEMBLE;
    }
    return EMPTY_STRING;
}

const std::string COutliers::LOF{"lof"};
const std::string COutliers::LDOF{"ldof"};
const std::string COutliers::DISTANCE_KNN{"distance_kth_nn"};
const std::string COutliers::TOTAL_DISTANCE_KNN{"distance_knn"};
const std::string COutliers::ENSEMBLE{"ensemble"};
const std::string COutliers::COMPUTING_OUTLIERS{"computing_outliers"};
}
}
}
