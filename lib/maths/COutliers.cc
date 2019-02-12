/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/COutliers.h>

#include <core/CDataFrame.h>

#include <maths/CDataFrameUtils.h>
#include <maths/CIntegration.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/CTools.h>

#include <boost/math/distributions/lognormal.hpp>

#include <numeric>
#include <sstream>
#include <tuple>

namespace ml {
namespace maths {
namespace outliers_detail {

double shift(double score) {
    return std::exp(-2.0) + score;
}

template<typename POINT>
CEnsemble<POINT>::CEnsemble(const TMethodFactoryVec& methodFactories,
                            TModelBuilderVec builders,
                            double priorProbabilityOutlier)
    : m_PriorProbabilityOutlier{priorProbabilityOutlier} {

    m_Models.reserve(builders.size());
    for (auto& builder : builders) {
        m_Models.push_back(builder.make(methodFactories));
    }

    for (auto& model : m_Models) {
        model.proportionOfRuntimePerMethod(1.0 / static_cast<double>(m_Models.size()));
    }
}

template<typename POINT>
typename CEnsemble<POINT>::TModelBuilderVec
CEnsemble<POINT>::makeBuilders(const TSizeVecVec& methods,
                               std::size_t numberPoints,
                               std::size_t dimension,
                               CPRNG::CXorOShiro128Plus rng) {
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
    std::size_t maxNumberNeighbours{computeNumberNeighbours(sampleSize)};
    std::size_t projectionDimension{computeProjectionDimension(sampleSize, dimension)};

    TModelBuilderVec result;
    result.reserve(numberModels);

    TPointVecVec projections{
        createProjections(rng, numberModels, projectionDimension, dimension)};

    // If we didn't find suitable projections this can be less than the target.
    numberModels = projections.size();

    for (std::size_t model = 0; model < numberModels; ++model) {
        // Multiple algorithms are used for the same model which therefore share
        // the sampled points. This allows us to save on both memory and runtime
        // (because we can share the nearest neighbour lookup for the different
        // algorithms).
        TSizeSizePrVec methodsAndNumberNeighbours;
        methodsAndNumberNeighbours.reserve(methods.size());
        for (const auto& method : methods) {
            methodsAndNumberNeighbours.emplace_back(
                method[CSampling::uniformSample(rng, 0, methods.size())],
                CSampling::uniformSample(rng, 3, maxNumberNeighbours + 1));
        }

        result.emplace_back(rng, std::move(methodsAndNumberNeighbours),
                            sampleSize, std::move(projections[model]));

        rng.discard(1ull < 63);
    }

    return result;
}

template<typename POINT>
void CEnsemble<POINT>::computeOutlierScores(const std::vector<POINT>& points,
                                            TDoubleVec& scores) const {
    if (points.empty()) {
        return;
    }

    LOG_TRACE(<< "Computing outlier scores");

    TScorerVec scores_(points.size());
    for (const auto& model : m_Models) {
        model.addOutlierScores(points, scores_);
    }

    scores.resize(points.size());
    for (std::size_t i = 0; i < scores_.size(); ++i) {
        scores[i] = scores_[i].compute(m_PriorProbabilityOutlier);
    }
}

template<typename POINT>
std::string CEnsemble<POINT>::print() const {
    std::ostringstream result;
    result << "prior probability outlier = " << m_PriorProbabilityOutlier;
    for (const auto& model : m_Models) {
        result << "\n" << model.print();
    }
    return result.str();
}

template<typename POINT>
typename CEnsemble<POINT>::TPointVecVec
CEnsemble<POINT>::createProjections(CPRNG::CXorOShiro128Plus& rng,
                                    std::size_t numberProjections,
                                    std::size_t projectionDimension,
                                    std::size_t dimension) {

    LOG_TRACE(<< "# projections = " << numberProjections << ", dimension = " << dimension
              << ", projection dimension = " << projectionDimension);

    TPointVecVec result;
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
        CSampling::normalSample(rng, 0.0, 1.0, bag * ((target + bag - 1) / bag), coordinates);
        LOG_TRACE(<< "# random samples = " << coordinates.size());

        for (auto coordinate = coordinates.begin();
             bag > 0 && dimension > 0 && result.size() < numberProjections &&
             coordinate != coordinates.end();
             /**/) {

            TPointVec projection{bag, SConstant<TPoint>::get(dimension, 0)};

            for (std::size_t i = 0; i < bag; ++i) {
                for (std::size_t j = 0; j < dimension; ++j, ++coordinate) {
                    projection[i](j) = *coordinate;
                }
            }

            // Orthogonalise the projection. This resizes the it if there are
            // linear dependencies.
            if (COrthogonaliser::orthonormalBasis(projection)) {

                std::size_t n{std::min(numberProjections - result.size(),
                                       projection.size() / projectionDimension)};

                for (std::size_t i = 0; i < n; ++i) {
                    result.push_back({});
                    for (std::size_t j = 0; j < projectionDimension; ++j) {
                        std::size_t index{projectionDimension * i + j};
                        result.back().push_back(std::move(projection[index]));
                    }
                }
            }
        }
    } else {
        // Identity.
        result.resize(numberProjections);
        for (std::size_t p = 0; p < numberProjections; ++p) {
            result[p].resize(projectionDimension, SConstant<TPoint>::get(dimension, 0));
            for (std::size_t d = 0; d < dimension; ++d) {
                result[p][d](d) = 1.0;
            }
        }
    }

    return result;
}

template<typename POINT>
typename CEnsemble<POINT>::TPoint
CEnsemble<POINT>::project(const TPointVec& projection, const POINT& point) {
    auto result = SConstant<TPoint>::get(projection.size(), 0);
    for (std::size_t pd = 0; pd < projection.size(); ++pd) {
        result(pd) = las::inner(projection[pd], point);
    }
    return result;
}

template<typename POINT>
CEnsemble<POINT>::CModelBuilder::CModelBuilder(CPRNG::CXorOShiro128Plus& rng,
                                               TSizeSizePrVec&& methodsAndNumberNeighbours,
                                               std::size_t sampleSize,
                                               TPointVec&& projection)
    : m_MethodsAndNumberNeighbours{std::forward<TSizeSizePrVec>(methodsAndNumberNeighbours)},
      m_SampleSize{sampleSize}, m_Sampler{makeSampler(rng, sampleSize)},
      m_Projection(std::forward<TPointVec>(projection)) {

    m_SampledProjectedPoints.reserve(m_Sampler.targetSampleSize());
}

template<typename POINT>
typename CEnsemble<POINT>::CModel
CEnsemble<POINT>::CModelBuilder::make(const TMethodFactoryVec& methodFactories) {

    if (m_SampledProjectedPoints.size() > m_SampleSize) {
        // We want a random subset of size m_SampleSize. Randomly permute and
        // grab the first m_SampleSize elements is equivalent.
        CSampling::random_shuffle(m_Sampler.rng(), m_SampledProjectedPoints.begin(),
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
CEnsemble<POINT>::CModelBuilder::makeSampler(CPRNG::CXorOShiro128Plus& rng,
                                             std::size_t sampleSize) {
    auto onSample = [this](std::size_t index, const TRowRef& row) {
        if (index >= m_SampledProjectedPoints.size()) {
            m_SampledProjectedPoints.push_back(
                project(m_Projection, CDataFrameUtils::rowTo<POINT>(row)));
        } else {
            m_SampledProjectedPoints[index] =
                project(m_Projection, CDataFrameUtils::rowTo<POINT>(row));
        }
    };
    return {sampleSize, onSample, rng};
}

template<typename POINT>
void CEnsemble<POINT>::CScorer::add(const TMeanVarAccumulatorVec& logScoreMoments,
                                    const TDoubleVec& scores) {

    // The basic idea is to map score percentile to a probability that the data
    // point is an outlier. We use a monotonic function for this which is flat
    // near the median, doesn't become too small for low percentiles and is
    // relatively sensitive for very high percentiles. Even then we don't become
    // too confident based on a single score. This could be explicitly calibrated
    // using the slope of the ROC curve, but one would need a very good collection
    // of benchmark sets to make this effective a priori. We also account for the
    // fact that the scores will be somewhat correlated.

    auto pOutlier = [](double cdfComplement) {
        static const double LOG_KNOTS[]{
            CTools::fastLog(1e-5), CTools::fastLog(1e-3), CTools::fastLog(0.01),
            CTools::fastLog(0.1),  CTools::fastLog(0.2),  CTools::fastLog(0.4),
            CTools::fastLog(0.5),  CTools::fastLog(0.6),  CTools::fastLog(0.8),
            CTools::fastLog(1.0)};
        static const double KNOT_PROBABILITIES[]{0.98, 0.87, 0.76, 0.65, 0.6,
                                                 0.5,  0.5,  0.5,  0.3,  0.3};

        double logCdfComplement{CTools::fastLog(std::max(cdfComplement, 1e-5))};
        auto k = std::upper_bound(std::begin(LOG_KNOTS), std::end(LOG_KNOTS), logCdfComplement);
        return CTools::linearlyInterpolate(
            *(k - 1), *k, KNOT_PROBABILITIES[k - std::begin(LOG_KNOTS) - 1],
            KNOT_PROBABILITIES[k - std::begin(LOG_KNOTS)], logCdfComplement);
    };

    static const double EXPECTED_PROBABILITY{[pOutlier] {
        double result{0.0};
        for (double x = 0.0; x < 0.99; x += 0.1) {
            double interval;
            CIntegration::gaussLegendre<CIntegration::OrderTwo>(
                [pOutlier](double x_, double& r) {
                    r = pOutlier(x_);
                    return true;
                },
                x, x + 0.1, interval);
            result += interval;
        }
        return result;
    }()};

    double logLikelihoodOutlierGivenScores{0.0};
    double logLikelihoodInlierGivenScores{0.0};

    for (std::size_t i = 0; i < scores.size(); ++i) {
        double location{CBasicStatistics::mean(logScoreMoments[i])};
        double scale{std::sqrt(CBasicStatistics::variance(logScoreMoments[i]))};
        double cdfComplement{0.5};
        if (scale > 0.0) {
            try {
                boost::math::lognormal lognormal{location, scale};
                cdfComplement = CTools::safeCdfComplement(lognormal, shift(scores[i]));
            } catch (const std::exception& e) {
                // In this case, we use the initial value of 0.5 for the cdfComplement
                // which means P(outlier | score) = P(inlier | score) = 0.5. The outcome
                // is the score conveys no information about whether or not a point is
                // an outlier, it is effectively ignored. The rationale for keeping going
                // therefore is that this handling is good enough that the results may
                // still be useful.
                LOG_ERROR(<< "Failed to normalise scores: " << e.what());
            }
        }

        double pOutlierGivenScore{pOutlier(cdfComplement) * 0.5 / EXPECTED_PROBABILITY};

        logLikelihoodOutlierGivenScores += CTools::fastLog(pOutlierGivenScore);
        logLikelihoodInlierGivenScores += CTools::fastLog(1.0 - pOutlierGivenScore);
    }

    double pOutlierGivenScores{std::exp(logLikelihoodOutlierGivenScores)};
    double pInlierGivenScores{std::exp(logLikelihoodInlierGivenScores)};
    double Z{pInlierGivenScores + pOutlierGivenScores};

    m_EnsembleSize += 1.0;
    m_LogLikelihoodOutlierGivenScores += CTools::fastLog(pOutlierGivenScores / Z);
    m_LogLikelihoodInlierGivenScores += CTools::fastLog(pInlierGivenScores / Z);
}

template<typename POINT>
double CEnsemble<POINT>::CScorer::compute(double pOutlier) const {

    double pInlier{1.0 - pOutlier};

    double logLikelihoodOutlier{m_LogLikelihoodOutlierGivenScores};
    double logLikelihoodInlier{m_LogLikelihoodInlierGivenScores};

    double renorm{std::max(logLikelihoodOutlier, logLikelihoodInlier)};
    logLikelihoodOutlier -= renorm;
    logLikelihoodInlier -= renorm;

    // For large ensemble sizes, the classification becomes unrealistically
    // confident. For a high degree of confidence we instead require that the
    // point has a significant chance of being an outlier for a significant
    // fraction of the ensemble models, i.e. 4 out of 5. In the following, we
    // use the fact that max(P(outlier), P(inlier)) is zero after renormalisation.

    (logLikelihoodInlier < logLikelihoodOutlier ? logLikelihoodInlier
                                                : logLikelihoodOutlier) /= 0.8 * m_EnsembleSize;

    // The conditional probability follows from Bayes rule.
    double likelihoodOutlier{std::exp(logLikelihoodOutlier + CTools::fastLog(pOutlier))};
    double likelihoodInlier{std::exp(logLikelihoodInlier + CTools::fastLog(pInlier))};
    return likelihoodOutlier / (likelihoodOutlier + likelihoodInlier);
}

template<typename POINT>
CEnsemble<POINT>::CModel::CModel(const TMethodFactoryVec& methodFactories,
                                 TSizeSizePrVec methodsAndNumberNeighbours,
                                 TPointVec sample,
                                 TPointVec projection)
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

    TDoubleVec scores;
    TProgressCallback noop{[](double) {}};
    for (auto& method : methods) {
        method->progressRecorder().swap(noop);
        method->run(sample, sample.size(), [&scores](TDoubleVecVec scores_) {
            scores = std::move(scores_[0]);
        });
        method->progressRecorder().swap(noop);
        for (auto& score : scores) {
            score = CTools::fastLog(shift(score));
        }
        TMeanVarAccumulator moments;
        moments.add(scores);
        m_LogScoreMoments.push_back(moments);
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
                                                TScorerVec& scores) const {
    // This index is used for addressing an array in the cache of nearest neighbour
    // distances for the local outlier factor method. We simply need to ensure that
    // it doesn't overlap the indices of any of the sampled model points.
    std::size_t index{this->numberPoints()};

    TPointVec points_;
    points_.reserve(points.size());
    for (std::size_t i = 0; i < points.size(); ++i, ++index) {
        points_.emplace_back(project(m_Projection, points[i]), index);
    }

    TDoubleVecVec methodScores;
    m_Method->run(points_, index, [&methodScores](TDoubleVecVec scores_) {
        methodScores = std::move(scores_);
    });

    TDoubleVec pointScores(methodScores.size());
    for (std::size_t i = 0; i < points_.size(); ++i) {
        index = points_[i].annotation();
        for (std::size_t j = 0; j < methodScores.size(); ++j) {
            pointScores[j] = methodScores[j][index];
        }
        scores[i].add(m_LogScoreMoments, pointScores);
    }
}

template<typename POINT>
std::string CEnsemble<POINT>::CModel::print() const {
    return "projection = " + std::to_string(m_Projection.size()) + " x " +
           std::to_string(las::dimension(m_Projection[0])) +
           " method = " + m_Method->print() +
           " score moments = " + core::CContainerPrinter::print(m_LogScoreMoments);
}
}

using namespace outliers_detail;

namespace {
using TRowItr = core::CDataFrame::TRowItr;

template<typename POINT>
typename CEnsemble<POINT>::TMethodFactoryVec methodFactories(TProgressCallback recordProgress) {
    using TPoint = typename CEnsemble<POINT>::TPoint;
    using TKdTree = typename CEnsemble<POINT>::TKdTree;

    typename CEnsemble<POINT>::TMethodFactoryVec result;
    result.reserve(4);

    result.emplace_back([recordProgress](std::size_t k, const TKdTree& lookup) {
        return std::make_unique<CLof<TPoint, const TKdTree&>>(k, recordProgress, lookup);
    });
    result.emplace_back([recordProgress](std::size_t k, const TKdTree& lookup) {
        return std::make_unique<CLdof<TPoint, const TKdTree&>>(k, recordProgress, lookup);
    });
    result.emplace_back([recordProgress](std::size_t k, const TKdTree& lookup) {
        return std::make_unique<CDistancekNN<TPoint, const TKdTree&>>(k, recordProgress, lookup);
    });
    result.emplace_back([recordProgress](std::size_t k, const TKdTree& lookup) {
        return std::make_unique<CTotalDistancekNN<TPoint, const TKdTree&>>(
            k, recordProgress, lookup);
    });

    return result;
}

template<typename POINT>
CEnsemble<POINT> buildEnsemble(core::CDataFrame& frame, TProgressCallback recordProgress) {

    using TSizeVec = typename CEnsemble<POINT>::TSizeVec;
    using TSizeVecVec = typename CEnsemble<POINT>::TSizeVecVec;

    TSizeVecVec methods(2);
    methods[0] = TSizeVec{COutliers::E_Lof, COutliers::E_Ldof};
    methods[1] = TSizeVec{COutliers::E_DistancekNN, COutliers::E_TotalDistancekNN};

    auto builders = CEnsemble<POINT>::makeBuilders(methods, frame.numberRows(),
                                                   frame.numberColumns());

    frame.readRows(1, [&builders](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            for (auto& builder : builders) {
                builder.addPoint(*row);
            }
        }
    });

    return {methodFactories<POINT>(recordProgress), std::move(builders)};
}

bool computeOutliersNoPartitions(std::size_t numberThreads,
                                 core::CDataFrame& frame,
                                 TProgressCallback recordProgress) {

    using TPoint = CMemoryMappedDenseVector<CFloatStorage>;
    using TPointVec = std::vector<TPoint>;

    CEnsemble<TPoint> ensemble{buildEnsemble<TPoint>(frame, recordProgress)};
    LOG_TRACE(<< "Ensemble = " << ensemble.print());

    // The points will be entirely overwritten by readRows so the initial value
    // is not important. This is presized so that rowsToPoints only needs to
    // access and write to each element. Since it does this once per element it
    // is thread safe.
    TPointVec points(frame.numberRows(), TPoint{nullptr, 1});

    auto rowsToPoints = [&points](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            points[row->index()] = CDataFrameUtils::rowTo<TPoint>(*row);
        }
    };

    std::uint64_t checksum{frame.checksum()};

    bool successful;
    std::tie(std::ignore, successful) = frame.readRows(numberThreads, rowsToPoints);

    if (successful == false) {
        LOG_ERROR(<< "Failed to read the data frame");
        return false;
    }

    TDoubleVec scores;
    ensemble.computeOutlierScores(points, scores);

    // This is a sanity check against CEnsemble accidentally writing to the data
    // frame via one of the memory mapped vectors. All bets are off as to whether
    // we generate anything meaningful if this happens.
    if (checksum != frame.checksum()) {
        LOG_ERROR(<< "Accidentally modified the data frame");
        return false;
    }

    auto writeScores = [&scores](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            row->writeColumn(row->numberColumns() - 1, scores[row->index()]);
        }
    };

    frame.resizeColumns(numberThreads, frame.numberColumns() + 1);

    if (frame.writeColumns(numberThreads, writeScores) == false) {
        LOG_ERROR(<< "Failed to write scores to the data frame");
        return false;
    }
    return true;
}
}

void COutliers::compute(std::size_t numberThreads,
                        core::CDataFrame& frame,
                        TProgressCallback recordProgress) {
    // TODO memory monitoring.
    // TODO different analysis types.

    CDataFrameUtils::standardizeColumns(numberThreads, frame);

    if (frame.inMainMemory()) {
        if (computeOutliersNoPartitions(numberThreads, frame, recordProgress) == false) {
            HANDLE_FATAL(<< "Internal error: computing outliers for data frame. There "
                         << "may be more details in the logs. Please report this problem.");
        }
        return;
    }

    // TODO this needs to work out the partitioning and run outlier detection
    // one partition at a time.
}

void COutliers::noop(double) {
}
}
}
