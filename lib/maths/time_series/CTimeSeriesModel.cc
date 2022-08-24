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

#include <maths/time_series/CTimeSeriesModel.h>

#include <core/CAllocationStrategy.h>
#include <core/CFunctional.h>
#include <core/CMemoryDef.h>
#include <core/CPersistUtils.h>
#include <core/RestoreMacros.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CBasicStatisticsPersist.h>
#include <maths/common/CModelDetail.h>
#include <maths/common/CMultivariateNormalConjugate.h>
#include <maths/common/CMultivariatePrior.h>
#include <maths/common/COrderings.h>
#include <maths/common/COrderingsSimultaneousSort.h>
#include <maths/common/CPrior.h>
#include <maths/common/CPriorStateSerialiser.h>
#include <maths/common/CTools.h>
#include <maths/common/Constants.h>
#include <maths/common/MathsTypes.h>

#include <maths/time_series/CDecayRateController.h>
#include <maths/time_series/CTimeSeriesDecomposition.h>
#include <maths/time_series/CTimeSeriesDecompositionStateSerialiser.h>
#include <maths/time_series/CTimeSeriesMultibucketFeatureSerialiser.h>
#include <maths/time_series/CTimeSeriesMultibucketFeatures.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <tuple>

namespace ml {
namespace maths {
namespace time_series {
namespace {

using TDoubleVec = std::vector<double>;
using TSizeDoublePr = std::pair<std::size_t, double>;
using TSizeVec = std::vector<std::size_t>;
using TDouble4Vec = core::CSmallVector<double, 4>;
using TDouble10Vec = core::CSmallVector<double, 10>;
using TDouble10VecVec = std::vector<TDouble10Vec>;
using TDouble10Vec1Vec = core::CSmallVector<TDouble10Vec, 1>;
using TDouble10Vec2Vec = core::CSmallVector<TDouble10Vec, 2>;
using TSize10Vec = core::CSmallVector<std::size_t, 10>;
using TDoubleDoublePr = std::pair<double, double>;
using TSizeDoublePr10Vec = core::CSmallVector<TSizeDoublePr, 10>;
using TCalculation2Vec = core::CSmallVector<maths_t::EProbabilityCalculation, 2>;
using TTail10Vec = core::CSmallVector<maths_t::ETail, 10>;
using TMeanAccumulator = common::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TUnivariatePriorPtr = std::shared_ptr<common::CPrior>;
using TMultivariatePriorCPtrSizePr1Vec = CTimeSeriesCorrelations::TMultivariatePriorCPtrSizePr1Vec;

//! The decay rate controllers we maintain.
enum EDecayRateController {
    E_TrendControl = 0,
    E_ResidualControl,
    E_NumberControls
};

const std::size_t MAXIMUM_CORRELATIONS{5000};
const double MINIMUM_CORRELATE_PRIOR_SAMPLE_COUNT{24.0};
const TSize10Vec NOTHING_TO_MARGINALIZE;
const TSizeDoublePr10Vec NOTHING_TO_CONDITION;

//! Expand \p calculation for computing multibucket anomalies.
TCalculation2Vec expand(maths_t::EProbabilityCalculation calculation) {
    switch (calculation) {
    case maths_t::E_TwoSided:
        return {maths_t::E_OneSidedBelow, maths_t::E_OneSidedAbove};
    case maths_t::E_OneSidedBelow:
    case maths_t::E_OneSidedAbove:
        return {calculation};
    }
    return {};
}

//! Aggregate one or more feature probabilities.
double aggregateFeatureProbabilities(const TDouble4Vec& probabilities, double correlation) {
    if (probabilities.size() > 1) {
        common::CJointProbabilityOfLessLikelySamples pJoint{correlation};
        for (auto p : probabilities) {
            pJoint.add(p);
        }
        double result;
        pJoint.calculate(result);
        return result;
    }
    return probabilities[0];
}

std::shared_ptr<common::CPrior> conditional(const common::CMultivariatePrior& prior,
                                            std::size_t dimension,
                                            const core::CSmallVector<double, 10>& value) {
    std::size_t dimensions{prior.dimension()};
    TSizeDoublePr10Vec condition(dimensions - 1);
    for (std::size_t i = 0, j = 0; i < dimensions; ++i) {
        if (i != dimension) {
            condition[j++] = std::make_pair(i, value[i]);
        }
    }
    return prior.univariate(NOTHING_TO_MARGINALIZE, condition).first;
}

const std::string VERSION_6_3_TAG("6.3");
const std::string VERSION_6_5_TAG("6.5");
const std::string VERSION_7_3_TAG("7.3");
const std::string VERSION_7_11_TAG("7.11");

// Models
// Version >= 6.3
const std::string ID_6_3_TAG{"a"};
const std::string IS_NON_NEGATIVE_6_3_TAG{"b"};
const std::string IS_FORECASTABLE_6_3_TAG{"c"};
//const std::string RNG_6_3_TAG{"d"}; Removed in 6.5
const std::string CONTROLLER_6_3_TAG{"e"};
const core::TPersistenceTag TREND_MODEL_6_3_TAG{"f", "trend_model"};
const core::TPersistenceTag RESIDUAL_MODEL_6_3_TAG{"g", "residual_model"};
const std::string ANOMALY_MODEL_6_3_TAG{"h"};
//const std::string RECENT_SAMPLES_6_3_TAG{"i"}; Removed in 6.5
//const std::string CANDIDATE_CHANGE_POINT_6_3_TAG{"j"}; Removed in 7.11
//const std::string CURRENT_CHANGE_INTERVAL_6_3_TAG{"k"}; Removed in 7.11
//const std::string CHANGE_DETECTOR_6_3_TAG{"l"}; Removed in 7.11
const std::string MULTIBUCKET_FEATURE_6_3_TAG{"m"};
const std::string MULTIBUCKET_FEATURE_MODEL_6_3_TAG{"n"};

// Anomaly model
// Version >= 7.3
const std::string LAST_ANOMALOUS_BUCKET_TIME_7_3_TAG{"d"};
// Version >= 6.5
const std::string ANOMALY_6_5_TAG{"e"};
const std::string ANOMALY_FEATURE_MODEL_6_5_TAG{"f"};
// Version < 6.5
// Discarded on state upgrade because features have changed.
// Anomaly only restored for 6.5 state.
const std::string FIRST_ANOMALOUS_BUCKET_TIME_6_5_TAG{"a"};
const std::string SUM_PREDICTION_ERROR_6_5_TAG{"b"};
const std::string MEAN_ABS_PREDICTION_ERROR_6_5_TAG{"c"};

// Correlations
const std::string K_MOST_CORRELATED_TAG{"a"};
const std::string CORRELATED_LOOKUP_TAG{"b"};
const std::string CORRELATION_MODELS_TAG{"c"};
// Correlations nested
const std::string FIRST_CORRELATE_ID_TAG{"a"};
const std::string SECOND_CORRELATE_ID_TAG{"b"};
const std::string CORRELATION_MODEL_TAG{"c"};
const std::string CORRELATION_TAG{"d"};

namespace forecast {
const std::string INFO_INSUFFICIENT_HISTORY{"Insufficient history to forecast"};
const std::string INFO_COULD_NOT_FORECAST_FOR_FULL_DURATION{
    "Unable to accurately forecast for the full requested time interval"};
const std::string ERROR_MULTIVARIATE{"Forecast not supported for multivariate features"};
}

namespace outliers {
constexpr double MINIMUM_WEIGHT{0.01};
const double MAXIMUM_P_VALUE{1e-3};
const double MINIMUM_P_VALUE{1e-5};
const double LOG_MAXIMUM_P_VALUE{std::log(MAXIMUM_P_VALUE)};
const double LOG_MINIMUM_P_VALUE{std::log(MINIMUM_P_VALUE)};
const double MINUS_LOG_TOLERANCE{
    -std::log(1.0 - 100.0 * std::numeric_limits<double>::epsilon())};

//! Derate the minimum outlier weight.
double deratedMinimumWeight(double derate) {
    derate = common::CTools::truncate(derate, 0.0, 1.0);
    return MINIMUM_WEIGHT + (0.5 - MINIMUM_WEIGHT) * derate;
}

//! Compute the one tail p-value of \p value.
double computePValue(const common::CPrior& prior,
                     const maths_t::TDoubleWeightsAry& weights,
                     double value) {
    double lowerBound;
    double upperBound;
    if (prior.minusLogJointCdf({value}, {weights}, lowerBound, upperBound) == false) {
        return 1.0;
    }
    if (upperBound >= MINUS_LOG_TOLERANCE) {
        double f{std::exp(-(lowerBound + upperBound) / 2.0)};
        return std::min(f, 1.0 - f);
    }
    if (prior.minusLogJointCdfComplement({value}, {weights}, lowerBound, upperBound) == false) {
        return 1.0;
    }
    return std::exp(-(lowerBound + upperBound) / 2.0);
}

double weight(const common::CPrior& prior,
              const maths_t::TDoubleWeightsAry& weights,
              double derate,
              double value) {

    double pValue{computePValue(prior, weights, value)};
    if (pValue >= MAXIMUM_P_VALUE) {
        return 1.0;
    }

    double minimumWeight{deratedMinimumWeight(derate)};
    if (pValue <= MINIMUM_P_VALUE) {
        return minimumWeight;
    }

    // We logarithmically interpolate between 1.0 and the minimum weight
    // on the interval [MAXIMUM_P_VALUE, MINIMUM_P_VALUE].

    double maximumExponent{-std::log(minimumWeight) / LOG_MINIMUM_P_VALUE /
                           (LOG_MINIMUM_P_VALUE - LOG_MAXIMUM_P_VALUE)};
    double logPValue{std::log(pValue)};
    double weight{std::exp(-maximumExponent * logPValue * (logPValue - LOG_MAXIMUM_P_VALUE))};
    LOG_TRACE(<< "sample = " << value << " p-value = " << pValue << ", weight = " << weight);

    return common::CMathsFuncs::isNan(weight) ? 1.0 : weight;
}
}
}

//! \brief A model of anomalous sections of a time series.
class CTimeSeriesAnomalyModel {
public:
    CTimeSeriesAnomalyModel();
    CTimeSeriesAnomalyModel(core_t::TTime bucketLength, double decayRate);

    //! Update the anomaly with prediction error and probability.
    //!
    //! Extends the current anomaly if \p probability is small; otherwise,
    //! it closes it. If the time series is currently anomalous, update the
    //! model with the anomaly feature vector.
    void sample(const common::CModelProbabilityParams& params,
                core_t::TTime time,
                double error,
                double bucketProbability,
                double overallProbability);

    //! Reset the mean error norm.
    void reset();

    //! If the time series is currently anomalous, compute the anomalousness
    //! of the anomaly feature vector.
    TDoubleDoublePr probability(double bucketProbability, double overallProbability) const;

    //! Age the model to account for \p time elapsed time.
    void propagateForwardsByTime(double time);

    //! Compute a checksum for this object.
    std::uint64_t checksum(std::uint64_t seed) const;

    //! Debug the memory used by this object.
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const;

    //! Get the memory used by this object.
    std::size_t memoryUsage() const;

    //! Initialize reading state from \p traverser.
    bool acceptRestoreTraverser(const common::SModelRestoreParams& params,
                                core::CStateRestoreTraverser& traverser);

    //! Persist by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

private:
    using TMultivariateNormalConjugate = common::CMultivariateNormalConjugate<2>;
    using TMultivariateNormalConjugateVec = std::vector<TMultivariateNormalConjugate>;

    //! \brief Extracts features related to anomalous time periods.
    class CAnomaly {
    public:
        //! See core::CMemory.
        static constexpr bool dynamicSizeAlwaysZero() { return true; }

    public:
        CAnomaly() = default;
        explicit CAnomaly(core_t::TTime time)
            : m_FirstAnomalousBucketTime(time), m_LastAnomalousBucketTime(time) {}

        //! Add a result to the anomaly.
        void update(core_t::TTime time, double predictionError) {
            m_LastAnomalousBucketTime = time;
            m_SumPredictionError += predictionError;
            m_MeanAbsPredictionError.add(std::fabs(predictionError));
        }

        //! Get the weight to apply to this anomaly on update.
        double weight() const {
            core_t::TTime length{m_LastAnomalousBucketTime - m_FirstAnomalousBucketTime};
            return 1.0 / (1.0 + std::max(static_cast<double>(length), 0.0));
        }

        //! Check if this anomaly is positive or negative.
        bool positive() const { return m_SumPredictionError > 0.0; }

        //! Get the feature vector for this anomaly.
        TDouble10Vec features() const {
            return {static_cast<double>(m_LastAnomalousBucketTime - m_FirstAnomalousBucketTime),
                    common::CBasicStatistics::mean(m_MeanAbsPredictionError)};
        }

        //! Compute a checksum for this object.
        std::uint64_t checksum(std::uint64_t seed) const {
            seed = common::CChecksum::calculate(seed, m_FirstAnomalousBucketTime);
            seed = common::CChecksum::calculate(seed, m_LastAnomalousBucketTime);
            seed = common::CChecksum::calculate(seed, m_SumPredictionError);
            return common::CChecksum::calculate(seed, m_MeanAbsPredictionError);
        }

        //! Initialize reading state from \p traverser.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
            do {
                const std::string& name{traverser.name()};
                RESTORE_BUILT_IN(FIRST_ANOMALOUS_BUCKET_TIME_6_5_TAG, m_FirstAnomalousBucketTime)
                RESTORE_BUILT_IN(LAST_ANOMALOUS_BUCKET_TIME_7_3_TAG, m_LastAnomalousBucketTime)
                RESTORE_BUILT_IN(SUM_PREDICTION_ERROR_6_5_TAG, m_SumPredictionError)
                RESTORE(MEAN_ABS_PREDICTION_ERROR_6_5_TAG,
                        m_MeanAbsPredictionError.fromDelimited(traverser.value()))
            } while (traverser.next());
            return true;
        }

        //! Persist by passing information to \p inserter.
        void acceptPersistInserter(core::CStatePersistInserter& inserter) const {
            inserter.insertValue(FIRST_ANOMALOUS_BUCKET_TIME_6_5_TAG, m_FirstAnomalousBucketTime);
            inserter.insertValue(LAST_ANOMALOUS_BUCKET_TIME_7_3_TAG, m_LastAnomalousBucketTime);
            inserter.insertValue(SUM_PREDICTION_ERROR_6_5_TAG, m_SumPredictionError,
                                 core::CIEEE754::E_SinglePrecision);
            inserter.insertValue(MEAN_ABS_PREDICTION_ERROR_6_5_TAG,
                                 m_MeanAbsPredictionError.toDelimited());
        }

    private:
        //! The time at which the first anomalous bucket was detected.
        core_t::TTime m_FirstAnomalousBucketTime{0};

        //! The time at which the last anomalous bucket was detected.
        core_t::TTime m_LastAnomalousBucketTime{0};

        //! The sum of the errors in our base model predictions for the
        //! anomaly.
        double m_SumPredictionError{0.0};

        //! The mean of minus the log probabilities from our base model
        //! in the anomaly.
        TMeanAccumulator m_MeanAbsPredictionError;
    };

    using TOptionalAnomaly = std::optional<CAnomaly>;

private:
    //! A unit weight.
    static const maths_t::TDouble10VecWeightsAry1Vec UNIT;

private:
    //! Update the anomaly model with a sample of the current feature vector.
    void sample(const common::CModelProbabilityParams& params, double weight);

    //! Compute the probability of the anomaly feature vector.
    bool anomalyProbability(double& result) const;

    //! Get the largest probability the model counts as anomalous.
    double largestAnomalyProbability() const {
        return 2.0 * common::LARGEST_SIGNIFICANT_PROBABILITY;
    }

    //! Get the scaled time.
    core_t::TTime scale(core_t::TTime time) const {
        return time / m_BucketLength;
    }

private:
    //! The data bucketing interval.
    core_t::TTime m_BucketLength{0};

    //! The current anomaly (if there is one).
    TOptionalAnomaly m_Anomaly;

    //! The model describing features of anomalous time periods.
    TMultivariateNormalConjugateVec m_AnomalyFeatureModels;
};

CTimeSeriesAnomalyModel::CTimeSeriesAnomalyModel() {
    m_AnomalyFeatureModels.reserve(2);
    m_AnomalyFeatureModels.push_back(
        TMultivariateNormalConjugate::nonInformativePrior(maths_t::E_ContinuousData));
    m_AnomalyFeatureModels.push_back(
        TMultivariateNormalConjugate::nonInformativePrior(maths_t::E_ContinuousData));
}

CTimeSeriesAnomalyModel::CTimeSeriesAnomalyModel(core_t::TTime bucketLength, double decayRate)
    : m_BucketLength(bucketLength) {
    m_AnomalyFeatureModels.reserve(2);
    m_AnomalyFeatureModels.push_back(TMultivariateNormalConjugate::nonInformativePrior(
        maths_t::E_ContinuousData, this->largestAnomalyProbability() * decayRate / 2.0));
    m_AnomalyFeatureModels.push_back(TMultivariateNormalConjugate::nonInformativePrior(
        maths_t::E_ContinuousData, this->largestAnomalyProbability() * decayRate / 2.0));
}

void CTimeSeriesAnomalyModel::sample(const common::CModelProbabilityParams& params,
                                     core_t::TTime time,
                                     double predictionError,
                                     double bucketProbability,
                                     double overallProbability) {

    if (overallProbability < this->largestAnomalyProbability()) {
        if (m_Anomaly == std::nullopt) {
            m_Anomaly.emplace(CAnomaly{this->scale(time)});
        }
        if (bucketProbability < this->largestAnomalyProbability()) {
            m_Anomaly->update(this->scale(time), predictionError);
            this->sample(params, m_Anomaly->weight());
        }
    } else if (m_Anomaly != std::nullopt) {
        this->sample(params, 1.0 - m_Anomaly->weight());
        m_Anomaly.reset();
    }
}

void CTimeSeriesAnomalyModel::sample(const common::CModelProbabilityParams& params,
                                     double weight) {
    // In case a rule triggered to skip model update,
    // this is the bit that we want to skip.
    // The rest of sample is necessary as it creates
    // the feature vector related to the current anomaly.
    double initialCountWeight{params.initialCountWeight()};
    if (initialCountWeight > 0.0) {
        auto& model = m_AnomalyFeatureModels[m_Anomaly->positive() ? 0 : 1];
        model.addSamples({m_Anomaly->features()},
                         {maths_t::countWeight(weight * initialCountWeight, 2)});
    }
}

void CTimeSeriesAnomalyModel::reset() {
    for (auto& model : m_AnomalyFeatureModels) {
        model = TMultivariateNormalConjugate::nonInformativePrior(
            maths_t::E_ContinuousData, model.decayRate());
    }
}

TDoubleDoublePr CTimeSeriesAnomalyModel::probability(double bucketProbability,
                                                     double overallProbability) const {

    double anomalyProbability{1.0};

    if (overallProbability < this->largestAnomalyProbability() &&
        this->anomalyProbability(anomalyProbability)) {

        // We logarithmically interpolate the anomaly probability and the
        // probability we've determined for the bucket. This determines
        // the weight assigned to the anomaly probability. We arrange for
        // the following properties for the weight (alpha) as a function
        // of the bucket and anomaly probabilities:
        //   1) The weight function is continuous,
        //   2) For small bucket probabilities we take the geometric mean
        //      (which corresponds to a weight equal to 0.5),
        //   3) For fixed anomaly probability the derivative of the weight
        //      w.r.t. minus log the bucket probability is negative and
        //      approaches 0.0 at the largest anomaly probability, and
        //   4) For fixed bucket probability the derivative of the weight
        //      w.r.t. minus log the anomaly probability is positive.
        // Note that condition 1) means we won't fall into the case that
        // a small perturbation in input data can lead to a large change in
        // results, condition 2) means that we will always correct anomalous
        // bucket probabilities based on how unusual they are in the context
        // of anomalous buckets we've seen before, condition 3) means that
        // the correction is continuous at the decision boundary for whether
        // to correct for the anomaly probability and is also important to
        // avoid the case that small perturbations lead to significant result
        // changes, finally condition 4) means that if the anomaly features
        // are highly unusual we can still assign the bucket a low probability
        // even if we don't think the bucket value is particularly unusual.
        // We relax the anomaly probability back to 1.0 by a factor lambda
        // based on how normal the individual bucket is.

        double a{-common::CTools::fastLog(this->largestAnomalyProbability())};
        double b{-common::CTools::fastLog(common::SMALL_PROBABILITY)};
        double lambda{bucketProbability == 0.0
                          ? 1.0
                          : std::min(this->largestAnomalyProbability() / bucketProbability, 1.0)};
        double logOverallProbability{common::CTools::fastLog(overallProbability)};
        double logAnomalyProbability{common::CTools::fastLog(anomalyProbability)};

        double x{std::max((b + logOverallProbability) / (b - a), 0.0)};
        double y{(1.0 - b / (b - logAnomalyProbability))};
        double alpha{0.5 * (1.0 - x + x * y)};

        overallProbability = std::exp((1.0 - alpha) * logOverallProbability +
                                      alpha * lambda * logAnomalyProbability);
        LOG_TRACE(<< "alpha = " << alpha << ", p(combined) = " << overallProbability);
    }

    return {overallProbability, anomalyProbability};
}

bool CTimeSeriesAnomalyModel::anomalyProbability(double& result) const {
    const auto& model = m_AnomalyFeatureModels[m_Anomaly->positive() ? 0 : 1];
    if (m_Anomaly == std::nullopt || model.isNonInformative()) {
        return false;
    }
    double pl;
    double pu;
    TTail10Vec tail;
    if (model.probabilityOfLessLikelySamples(
            maths_t::E_OneSidedAbove, {m_Anomaly->features()}, UNIT, pl, pu, tail) == false) {
        return false;
    }
    result = (pl + pu) / 2.0;
    LOG_TRACE(<< "features = " << m_Anomaly->features() << " p(anomaly) = " << result);
    return true;
}

void CTimeSeriesAnomalyModel::propagateForwardsByTime(double time) {
    m_AnomalyFeatureModels[0].propagateForwardsByTime(time);
    m_AnomalyFeatureModels[1].propagateForwardsByTime(time);
}

std::uint64_t CTimeSeriesAnomalyModel::checksum(std::uint64_t seed) const {
    seed = common::CChecksum::calculate(seed, m_BucketLength);
    seed = common::CChecksum::calculate(seed, m_Anomaly);
    seed = common::CChecksum::calculate(seed, m_AnomalyFeatureModels[0]);
    return common::CChecksum::calculate(seed, m_AnomalyFeatureModels[1]);
}

void CTimeSeriesAnomalyModel::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CTimeSeriesAnomalyModel");
    core::memory_debug::dynamicSize("m_Anomalies", m_Anomaly, mem);
    core::memory_debug::dynamicSize("m_AnomalyFeatureModels", m_AnomalyFeatureModels, mem);
}

std::size_t CTimeSeriesAnomalyModel::memoryUsage() const {
    return core::memory::dynamicSize(m_Anomaly) +
           core::memory::dynamicSize(m_AnomalyFeatureModels);
}

bool CTimeSeriesAnomalyModel::acceptRestoreTraverser(const common::SModelRestoreParams& params,
                                                     core::CStateRestoreTraverser& traverser) {
    m_BucketLength = core::unwrap_ref(params.s_Params).bucketLength();
    if (traverser.name() == VERSION_7_3_TAG) {
        std::size_t index{0};
        while (traverser.next()) {
            const std::string& name{traverser.name()};
            RESTORE_SETUP_TEARDOWN(ANOMALY_6_5_TAG, CAnomaly restored,
                                   traverser.traverseSubLevel([&](auto& traverser_) {
                                       return restored.acceptRestoreTraverser(traverser_);
                                   }),
                                   m_Anomaly.emplace(std::move(restored)))
            RESTORE(ANOMALY_FEATURE_MODEL_6_5_TAG, traverser.traverseSubLevel([&](auto& traverser_) {
                return m_AnomalyFeatureModels[index++].acceptRestoreTraverser(traverser_);
            }))
        }
    } else if (traverser.name() == VERSION_6_5_TAG) {
        std::size_t index{0};
        while (traverser.next()) {
            const std::string& name{traverser.name()};
            RESTORE(ANOMALY_FEATURE_MODEL_6_5_TAG, traverser.traverseSubLevel([&](auto& traverser_) {
                return m_AnomalyFeatureModels[index++].acceptRestoreTraverser(traverser_);
            }))
        }
    }
    // else we can't upgrade the state of the anomaly model pre 6.5.

    return true;
}

void CTimeSeriesAnomalyModel::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(VERSION_7_3_TAG, "");
    if (m_Anomaly) {
        inserter.insertLevel(ANOMALY_6_5_TAG, [this](auto& inserter_) {
            m_Anomaly->acceptPersistInserter(inserter_);
        });
    }
    inserter.insertLevel(ANOMALY_FEATURE_MODEL_6_5_TAG, [this](auto& inserter_) {
        m_AnomalyFeatureModels[0].acceptPersistInserter(inserter_);
    });
    inserter.insertLevel(ANOMALY_FEATURE_MODEL_6_5_TAG, [this](auto& inserter_) {
        m_AnomalyFeatureModels[1].acceptPersistInserter(inserter_);
    });
}

const maths_t::TDouble10VecWeightsAry1Vec CTimeSeriesAnomalyModel::UNIT{
    maths_t::CUnitWeights::unit<TDouble10Vec>(2)};

CUnivariateTimeSeriesModel::CUnivariateTimeSeriesModel(
    const common::CModelParams& params,
    std::size_t id,
    const CTimeSeriesDecompositionInterface& trendModel,
    const common::CPrior& residualModel,
    const TDecayRateController2Ary* controllers,
    const TMultibucketFeature* multibucketFeature,
    bool modelAnomalies)
    : common::CModel(params), m_Id(id), m_TrendModel(trendModel.clone()),
      m_ResidualModel(residualModel.clone()),
      m_MultibucketFeature(multibucketFeature != nullptr ? multibucketFeature->clone()
                                                         : nullptr),
      m_MultibucketFeatureModel(multibucketFeature != nullptr ? residualModel.clone() : nullptr),
      m_AnomalyModel(modelAnomalies ? std::make_unique<CTimeSeriesAnomalyModel>(
                                          params.bucketLength(),
                                          params.decayRate())
                                    : nullptr) {
    if (controllers != nullptr) {
        m_Controllers = std::make_unique<TDecayRateController2Ary>(*controllers);
    }
}

CUnivariateTimeSeriesModel::CUnivariateTimeSeriesModel(const common::SModelRestoreParams& params,
                                                       core::CStateRestoreTraverser& traverser)
    : common::CModel(params.s_Params) {
    if (traverser.traverseSubLevel([&](auto& traverser_) {
            return this->acceptRestoreTraverser(params, traverser_);
        }) == false) {
        traverser.setBadState();
    }
}

CUnivariateTimeSeriesModel::~CUnivariateTimeSeriesModel() {
    if (m_Correlations != nullptr) {
        m_Correlations->removeTimeSeries(m_Id);
    }
}

std::size_t CUnivariateTimeSeriesModel::identifier() const {
    return m_Id;
}

CUnivariateTimeSeriesModel* CUnivariateTimeSeriesModel::clone(std::size_t id) const {
    CUnivariateTimeSeriesModel* result{new CUnivariateTimeSeriesModel{*this, id}};
    if (m_Correlations != nullptr) {
        result->modelCorrelations(*m_Correlations);
    }
    return result;
}

CUnivariateTimeSeriesModel* CUnivariateTimeSeriesModel::cloneForPersistence() const {
    return new CUnivariateTimeSeriesModel{*this, m_Id};
}

CUnivariateTimeSeriesModel* CUnivariateTimeSeriesModel::cloneForForecast() const {
    return new CUnivariateTimeSeriesModel{*this, m_Id, true};
}

bool CUnivariateTimeSeriesModel::isForecastPossible() const {
    return m_IsForecastable && !m_ResidualModel->isNonInformative();
}

void CUnivariateTimeSeriesModel::modelCorrelations(CTimeSeriesCorrelations& model) {
    m_Correlations = &model;
    m_Correlations->addTimeSeries(m_Id, *this);
}

CUnivariateTimeSeriesModel::TSize2Vec1Vec CUnivariateTimeSeriesModel::correlates() const {
    TSize2Vec1Vec result;
    TSize1Vec correlated;
    TSize2Vec1Vec variables;
    TMultivariatePriorCPtrSizePr1Vec correlationModels;
    TModelCPtr1Vec correlatedTimeSeriesModels;
    this->correlationModels(correlated, variables, correlationModels, correlatedTimeSeriesModels);
    result.resize(correlated.size(), TSize2Vec(2));
    for (std::size_t i = 0; i < correlated.size(); ++i) {
        result[i][variables[i][0]] = m_Id;
        result[i][variables[i][1]] = correlated[i];
    }
    return result;
}

void CUnivariateTimeSeriesModel::addBucketValue(const TTimeDouble2VecSizeTrVec& values) {
    for (const auto& value : values) {
        m_ResidualModel->adjustOffset(
            {m_TrendModel->detrend(value.first, value.second[0], 0.0, m_IsNonNegative)},
            maths_t::CUnitWeights::SINGLE_UNIT);
    }
}

CUnivariateTimeSeriesModel::EUpdateResult
CUnivariateTimeSeriesModel::addSamples(const common::CModelAddSamplesParams& params,
                                       TTimeDouble2VecSizeTrVec samples) {
    if (samples.empty()) {
        return E_Success;
    }

    // Update the data characteristics.
    m_IsNonNegative = params.isNonNegative();
    maths_t::EDataType type{params.type()};
    m_ResidualModel->dataType(type);
    if (m_MultibucketFeatureModel != nullptr) {
        m_MultibucketFeatureModel->dataType(type);
    }
    m_TrendModel->dataType(type);

    EUpdateResult result{this->updateTrend(params, samples)};

    auto[residuals, decayRateMultiplier] =
        this->updateResidualModels(params, std::move(samples));

    // Age the anomaly model. Note that update requires the probability
    // to be calculated. This is expensive to compute and so unlike our
    // other model components is done in that function.
    if (m_AnomalyModel != nullptr) {
        m_AnomalyModel->propagateForwardsByTime(params.propagationInterval());
    }

    // Add the samples to the correlation models if there are any.
    if (m_Correlations != nullptr) {
        m_Correlations->addSamples(m_Id, params, residuals, decayRateMultiplier);
    }

    return result;
}

void CUnivariateTimeSeriesModel::skipTime(core_t::TTime gap) {
    m_TrendModel->skipTime(gap);
}

CUnivariateTimeSeriesModel::TDouble2Vec
CUnivariateTimeSeriesModel::mode(core_t::TTime time, const TDouble2VecWeightsAry& weights) const {
    return {m_ResidualModel->marginalLikelihoodMode(unpack(weights)) +
            m_TrendModel->value(time, 0.0, m_IsNonNegative).mean()};
}

CUnivariateTimeSeriesModel::TDouble2Vec1Vec
CUnivariateTimeSeriesModel::correlateModes(core_t::TTime time,
                                           const TDouble2VecWeightsAry1Vec& weights) const {
    TDouble2Vec1Vec result;

    TSize1Vec correlated;
    TSize2Vec1Vec variables;
    TMultivariatePriorCPtrSizePr1Vec correlationModels;
    TModelCPtr1Vec correlatedTimeSeriesModels;
    if (this->correlationModels(correlated, variables, correlationModels,
                                correlatedTimeSeriesModels)) {
        result.resize(correlated.size(), TDouble10Vec(2));
        double baseline[2];
        baseline[0] = m_TrendModel->value(time, 0.0, m_IsNonNegative).mean();
        for (std::size_t i = 0; i < correlated.size(); ++i) {
            baseline[1] = correlatedTimeSeriesModels[i]
                              ->m_TrendModel
                              ->value(time, 0.0, correlatedTimeSeriesModels[i]->m_IsNonNegative)
                              .mean();
            TDouble10Vec mode(correlationModels[i].first->marginalLikelihoodMode(
                CMultivariateTimeSeriesModel::unpack(weights[i])));
            std::size_t v0{variables[i][0]};
            std::size_t v1{variables[i][1]};
            result[i][v0] = baseline[0] + mode[v0];
            result[i][v1] = baseline[1] + mode[v1];
        }
    }

    return result;
}

CUnivariateTimeSeriesModel::TDouble2Vec1Vec
CUnivariateTimeSeriesModel::residualModes(const TDouble2VecWeightsAry& weights) const {
    TDouble2Vec1Vec result;
    TDouble1Vec modes(m_ResidualModel->marginalLikelihoodModes(unpack(weights)));
    result.reserve(modes.size());
    for (auto mode : modes) {
        result.push_back({mode});
    }
    return result;
}

void CUnivariateTimeSeriesModel::detrend(const TTime2Vec1Vec& time,
                                         double confidenceInterval,
                                         TDouble2Vec1Vec& value) const {
    if (value.empty()) {
        return;
    }

    if (value[0].size() == 1) {
        value[0][0] = m_TrendModel->detrend(time[0][0], value[0][0],
                                            confidenceInterval, m_IsNonNegative);
        return;
    }

    TSize1Vec correlated;
    TSize2Vec1Vec variables;
    TMultivariatePriorCPtrSizePr1Vec correlationModels;
    TModelCPtr1Vec correlatedTimeSeriesModels;
    if (this->correlationModels(correlated, variables, correlationModels,
                                correlatedTimeSeriesModels)) {
        for (std::size_t i = 0; i < variables.size(); ++i) {
            if (value[i].empty() == false) {
                std::size_t v0{variables[i][0]};
                std::size_t v1{variables[i][1]};
                value[i][v0] = m_TrendModel->detrend(
                    time[i][v0], value[i][v0], confidenceInterval, m_IsNonNegative);
                value[i][v1] = correlatedTimeSeriesModels[i]->m_TrendModel->detrend(
                    time[i][v1], value[i][v1], confidenceInterval,
                    correlatedTimeSeriesModels[i]->m_IsNonNegative);
            }
        }
    }
}

CUnivariateTimeSeriesModel::TDouble2Vec
CUnivariateTimeSeriesModel::predict(core_t::TTime time,
                                    const TSizeDoublePr1Vec& correlatedValue,
                                    TDouble2Vec hint) const {
    double correlateCorrection{0.0};
    if (correlatedValue.empty() == false) {
        TSize1Vec correlated{correlatedValue[0].first};
        TSize2Vec1Vec variables;
        TMultivariatePriorCPtrSizePr1Vec correlationModel;
        TModelCPtr1Vec correlatedTimeSeriesModels;
        if (m_Correlations->correlationModels(m_Id, correlated, variables, correlationModel,
                                              correlatedTimeSeriesModels)) {
            double sample{correlatedTimeSeriesModels[0]->m_TrendModel->detrend(
                time, correlatedValue[0].second, 0.0,
                correlatedTimeSeriesModels[0]->m_IsNonNegative)};
            TSize10Vec marginalize{variables[0][1]};
            TSizeDoublePr10Vec condition{{variables[0][1], sample}};
            const common::CMultivariatePrior* joint{correlationModel[0].first};
            TPriorPtr margin{
                joint->univariate(marginalize, NOTHING_TO_CONDITION).first};
            TPriorPtr conditional{
                joint->univariate(NOTHING_TO_MARGINALIZE, condition).first};
            correlateCorrection = conditional->marginalLikelihoodMean() -
                                  margin->marginalLikelihoodMean();
        }
    }

    double trend{0.0};
    if (m_TrendModel->initialized()) {
        trend = m_TrendModel->value(time, 0.0, m_IsNonNegative).mean();
    }

    if (hint.size() == 1) {
        hint[0] = m_TrendModel->detrend(time, hint[0], 0.0, m_IsNonNegative);
    }

    double median{
        m_ResidualModel->isNonInformative()
            ? m_ResidualModel->marginalLikelihoodMean()
            : (hint.empty()
                   ? common::CBasicStatistics::mean(
                         m_ResidualModel->marginalLikelihoodConfidenceInterval(0.0))
                   : m_ResidualModel->nearestMarginalLikelihoodMean(hint[0]))};
    double result{trend + median + correlateCorrection};

    return {m_IsNonNegative ? std::max(result, 0.0) : result};
}

CUnivariateTimeSeriesModel::TDouble2Vec3Vec
CUnivariateTimeSeriesModel::confidenceInterval(core_t::TTime time,
                                               double confidenceInterval,
                                               const TDouble2VecWeightsAry& weights_) const {
    if (m_ResidualModel->isNonInformative()) {
        return {};
    }

    double trend{m_TrendModel->initialized()
                     ? m_TrendModel->value(time, 0.0, m_IsNonNegative).mean()
                     : 0.0};

    TDoubleWeightsAry weights(unpack(weights_));
    double median{common::CBasicStatistics::mean(
        m_ResidualModel->marginalLikelihoodConfidenceInterval(0.0, weights))};
    TDoubleDoublePr interval{m_ResidualModel->marginalLikelihoodConfidenceInterval(
        confidenceInterval, weights)};

    double result[]{trend + interval.first, trend + median, trend + interval.second};

    return {{m_IsNonNegative ? std::max(result[0], 0.0) : result[0]},
            {m_IsNonNegative ? std::max(result[1], 0.0) : result[1]},
            {m_IsNonNegative ? std::max(result[2], 0.0) : result[2]}};
}

bool CUnivariateTimeSeriesModel::forecast(core_t::TTime firstDataTime,
                                          core_t::TTime lastDataTime,
                                          core_t::TTime startTime,
                                          core_t::TTime endTime,
                                          double confidenceInterval,
                                          const TDouble2Vec& minimum_,
                                          const TDouble2Vec& maximum_,
                                          const common::TForecastPushDatapointFunc& forecastPushDataPointFunc,
                                          std::string& messageOut) {

    core_t::TTime horizon{std::min(lastDataTime + (lastDataTime - firstDataTime),
                                   lastDataTime + m_TrendModel->maximumForecastInterval())};

    if (m_ResidualModel->isNonInformative() || startTime >= horizon) {
        messageOut = forecast::INFO_INSUFFICIENT_HISTORY;
        return true;
    }
    if (endTime > horizon) {
        // Truncate to the forecast horizon
        endTime = horizon;
        messageOut = forecast::INFO_COULD_NOT_FORECAST_FOR_FULL_DURATION;
    }

    using TDouble3Vec = core::CSmallVector<double, 3>;

    core_t::TTime bucketLength{this->params().bucketLength()};
    double minimum{m_IsNonNegative ? std::max(minimum_[0], 0.0) : minimum_[0]};
    double maximum{m_IsNonNegative ? std::max(maximum_[0], 0.0) : maximum_[0]};

    auto writer = [&](core_t::TTime time, const TDouble3Vec& prediction) {
        common::SErrorBar errorBar{
            time, bucketLength,
            common::CTools::truncate(prediction[0], minimum,
                                     maximum + prediction[0] - prediction[1]),
            common::CTools::truncate(prediction[1], minimum, maximum),
            common::CTools::truncate(
                prediction[2], minimum + prediction[2] - prediction[1], maximum)};
        forecastPushDataPointFunc(errorBar);
    };

    m_TrendModel->forecast(startTime, endTime, bucketLength, confidenceInterval,
                           this->params().minimumSeasonalVarianceScale(),
                           m_IsNonNegative, writer);

    return true;
}

bool CUnivariateTimeSeriesModel::probability(const common::CModelProbabilityParams& params,
                                             const TTime2Vec1Vec& time,
                                             const TDouble2Vec1Vec& value,
                                             common::SModelProbabilityResult& result) const {
    result = common::SModelProbabilityResult{};
    if (value.empty()) {
        return true;
    }
    return value[0].size() == 1
               ? this->uncorrelatedProbability(params, time, value, result)
               : this->correlatedProbability(params, time, value, result);
}

bool CUnivariateTimeSeriesModel::uncorrelatedProbability(
    const common::CModelProbabilityParams& params,
    const TTime2Vec1Vec& time_,
    const TDouble2Vec1Vec& value,
    common::SModelProbabilityResult& result) const {

    maths_t::EProbabilityCalculation calculation{params.calculation(0)};
    maths_t::TDoubleWeightsAry1Vec weights{unpack(params.weights()[0])};

    TDouble4Vec probabilities;
    common::SModelProbabilityResult::TFeatureProbability4Vec featureProbabilities;

    double pl;
    double pu;
    maths_t::ETail tail;
    core_t::TTime time{time_[0][0]};
    TDouble1Vec sample{m_TrendModel->detrend(
        time, value[0][0], params.seasonalConfidenceInterval(), m_IsNonNegative)};
    if (m_ResidualModel->probabilityOfLessLikelySamples(calculation, sample,
                                                        weights, pl, pu, tail)) {
        LOG_TRACE(<< "P(" << sample << " | weight = " << weights
                  << ", time = " << time << ") = " << (pl + pu) / 2.0);
        double pSingleBucket{(pl + pu) / 2.0};
        probabilities.push_back(pSingleBucket);
        featureProbabilities.emplace_back(
            common::SModelProbabilityResult::E_SingleBucketProbability, pSingleBucket);

        if (maths_t::seasonalVarianceScale(weights[0]) > 1) {
            LOG_DEBUG(<< "Variation at similar times is larger than is typical => the score is reduced.");
            result.s_ProbabilityExplanation.emplace_back("Variation at similar times is larger than is typical => the score is reduced. ");
        } 
        if (maths_t::countVarianceScale(weights[0]) > 1) {
            LOG_DEBUG(<< "The bucket contains fewer values than is typical => the score is reduced.");
            result.s_ProbabilityExplanation.emplace_back("The bucket contains fewer values than is typical => the score is reduced.");
        }

    } else {
        LOG_ERROR(<< "Failed to compute P(" << sample
                  << " | weight = " << weights << ", time = " << time << ")");
        return false;
    }

    double correlation{0.0};
    if (m_MultibucketFeatureModel != nullptr && params.useMultibucketFeatures()) {
        double pMultiBucket{1.0};
        TDouble1Vec feature;
        std::tie(feature, std::ignore) = m_MultibucketFeature->value();
        if (feature.empty() == false) {
            for (auto calculation_ : expand(calculation)) {
                maths_t::ETail dummy;
                if (m_MultibucketFeatureModel->probabilityOfLessLikelySamples(
                        calculation_, feature,
                        maths_t::CUnitWeights::SINGLE_UNIT, pl, pu, dummy)) {
                    LOG_TRACE(<< "P(" << feature << ") = " << (pl + pu) / 2.0);
                } else {
                    LOG_ERROR(<< "Failed to compute P(" << feature << ")");
                    return false;
                }
                pMultiBucket = std::min(pMultiBucket, (pl + pu) / 2.0);
            }
            correlation = m_MultibucketFeature->correlationWithBucketValue();
        }

        if (pMultiBucket < probabilities.back()) {
            LOG_DEBUG(<<"The function value is unusual for the multi-bucket, but the current bucket is normal => the score is reduced.");
            result.s_ProbabilityExplanation.emplace_back("The function value is unusual for the multi-bucket, but the current bucket is normal => the score is reduced.");
        }
        probabilities.push_back(pMultiBucket);
        featureProbabilities.emplace_back(
            common::SModelProbabilityResult::E_MultiBucketProbability, pMultiBucket);
    }

    double pOverall{aggregateFeatureProbabilities(probabilities, correlation)};

    if (m_AnomalyModel != nullptr && params.useAnomalyModel()) {
        TDouble2Vec seasonalWeight;
        this->seasonalWeight(0.0, time, seasonalWeight);
        double residual{
            (sample[0] - m_ResidualModel->nearestMarginalLikelihoodMean(sample[0])) /
            std::max(std::sqrt(seasonalWeight[0]), 1.0)};
        double pSingleBucket{probabilities[0]};

        m_AnomalyModel->sample(params, time, residual, pSingleBucket, pOverall);

        double pAnomaly;
        double pOverallOld{pOverall};
        std::tie(pOverall, pAnomaly) = m_AnomalyModel->probability(pSingleBucket, pOverall);
        if (pOverall < pOverallOld) {
            LOG_DEBUG(<< "The anomaly is shorter than other anomalies seen before (short spike) => the score is reduced.");
            result.s_ProbabilityExplanation.emplace_back("The anomaly is shorter than other anomalies seen before (short spike) => the score is reduced. ");
        } else if (pOverall > pOverallOld) {
            LOG_DEBUG(<< "The anomaly is longer than others we’ve seen before  => the score is increased.");
            result.s_ProbabilityExplanation.emplace_back("The anomaly is longer than others we’ve seen before  => the score is increased.");
        }
        probabilities.push_back(pAnomaly);
        featureProbabilities.emplace_back(
            common::SModelProbabilityResult::E_AnomalyModelProbability, pAnomaly);
    }

    result.s_Probability = pOverall;
    result.s_FeatureProbabilities = std::move(featureProbabilities);
    result.s_Tail = {tail};

    return true;
}

bool CUnivariateTimeSeriesModel::correlatedProbability(
    const common::CModelProbabilityParams& params,
    const TTime2Vec1Vec& time,
    const TDouble2Vec1Vec& value,
    common::SModelProbabilityResult& result) const {
    TSize1Vec correlated;
    TSize2Vec1Vec variables;
    TMultivariatePriorCPtrSizePr1Vec correlationModels;
    TModelCPtr1Vec correlatedTimeSeriesModels;
    if (this->correlationModels(correlated, variables, correlationModels,
                                correlatedTimeSeriesModels) == false) {
        return false;
    }

    double neff{effectiveCount(variables.size())};
    common::CProbabilityOfExtremeSample aggregator;
    common::CBasicStatistics::COrderStatisticsStack<double, 1> minProbability;

    // Declared outside the loop to minimize the number of times they are created.
    maths_t::EProbabilityCalculation calculation{params.calculation(0)};
    TDouble10Vec1Vec sample{TDouble10Vec(2)};
    maths_t::TDouble10VecWeightsAry1Vec weights{
        maths_t::CUnitWeights::unit<TDouble10Vec>(2)};
    TDouble10Vec2Vec pli;
    TDouble10Vec2Vec pui;
    TTail10Vec ti;
    core_t::TTime mostAnomalousTime{0};
    double mostAnomalousSample{0.0};
    TPriorPtr mostAnomalousCorrelationModel;

    TTail2Vec tail;
    bool conditional{false};
    TSize1Vec mostAnomalousCorrelate;

    TSize1Vec correlateIndices;
    if (params.mostAnomalousCorrelate() != std::nullopt) {
        if (*params.mostAnomalousCorrelate() >= variables.size()) {
            LOG_ERROR(<< "Unexpected correlate " << *params.mostAnomalousCorrelate());
            return false;
        }
        correlateIndices.push_back(*params.mostAnomalousCorrelate());
    } else {
        correlateIndices.resize(variables.size());
        std::iota(correlateIndices.begin(), correlateIndices.end(), 0);
    }

    for (std::size_t i = 0; i < correlateIndices.size(); ++i) {
        if (value[i].empty()) {
            aggregator.add(1.0, neff);
        } else {
            std::size_t correlateIndex{correlateIndices[i]};
            std::size_t v0{variables[correlateIndex][0]};
            std::size_t v1{variables[correlateIndex][1]};
            TDecompositionPtr trendModels[2];
            trendModels[v0] = m_TrendModel;
            trendModels[v1] = correlatedTimeSeriesModels[correlateIndex]->m_TrendModel;
            const auto& correlationModel = correlationModels[correlateIndex].first;
            bool isNonNegative[2];
            isNonNegative[v0] = m_IsNonNegative;
            isNonNegative[v1] = correlatedTimeSeriesModels[correlateIndex]->m_IsNonNegative;

            sample[0][0] = trendModels[0]->detrend(time[i][0], value[i][0],
                                                   params.seasonalConfidenceInterval(),
                                                   isNonNegative[0]);
            sample[0][1] = trendModels[1]->detrend(time[i][1], value[i][1],
                                                   params.seasonalConfidenceInterval(),
                                                   isNonNegative[1]);
            weights[0] = CMultivariateTimeSeriesModel::unpack(params.weights()[i]);

            if (correlationModel->probabilityOfLessLikelySamples(
                    calculation, sample, weights, {v0}, pli, pui, ti)) {
                LOG_TRACE(<< "Marginal P(" << sample
                          << " | weight = " << weights << ", coordinate = " << v0
                          << ") = " << (pli[0][0] + pui[0][0]) / 2.0);
                LOG_TRACE(<< "Conditional P(" << sample
                          << " | weight = " << weights << ", coordinate = " << v0
                          << ") = " << (pli[1][0] + pui[1][0]) / 2.0);
            } else {
                LOG_ERROR(<< "Failed to compute P(" << sample << " | weight = " << weights
                          << ", coordinate = " << v0 << ")");
                continue;
            }

            double pl{std::sqrt(pli[0][0] * pli[1][0])};
            double pu{std::sqrt(pui[0][0] * pui[1][0])};
            double pi{(pl + pu) / 2.0};

            aggregator.add(pi, neff);
            if (minProbability.add(pi)) {
                tail.assign(1, ti[0]);
                conditional = ((pli[1][0] + pui[1][0]) < (pli[0][0] + pui[0][0]));
                mostAnomalousCorrelate.assign(1, correlateIndex);
                mostAnomalousTime = time[i][v0];
                mostAnomalousSample = sample[0][v0];
                mostAnomalousCorrelationModel =
                    conditional ? correlationModel
                                      ->univariate(NOTHING_TO_MARGINALIZE,
                                                   {{v1, sample[0][v1]}})
                                      .first
                                : correlationModel
                                      ->univariate({v1}, NOTHING_TO_CONDITION)
                                      .first;
            }
        }
    }

    double pSingleBucket;
    aggregator.calculate(pSingleBucket);
    TDouble4Vec probabilities{pSingleBucket};
    common::SModelProbabilityResult::TFeatureProbability4Vec featureProbabilities;
    featureProbabilities.emplace_back(
        common::SModelProbabilityResult::E_SingleBucketProbability, pSingleBucket);

    double pOverall{pSingleBucket};

    if (m_AnomalyModel != nullptr && params.useAnomalyModel()) {
        TDouble2Vec seasonalWeight;
        this->seasonalWeight(0.0, mostAnomalousTime, seasonalWeight);
        double residual{(mostAnomalousSample - mostAnomalousCorrelationModel->nearestMarginalLikelihoodMean(
                                                   mostAnomalousSample)) /
                        std::max(std::sqrt(seasonalWeight[0]), 1.0)};

        m_AnomalyModel->sample(params, mostAnomalousTime, residual, pSingleBucket, pOverall);

        double pAnomaly;
        std::tie(pOverall, pAnomaly) = m_AnomalyModel->probability(pSingleBucket, pOverall);
        probabilities.push_back(pAnomaly);
        featureProbabilities.emplace_back(
            common::SModelProbabilityResult::E_AnomalyModelProbability, pAnomaly);
    }

    result.s_Probability = pOverall;
    result.s_Conditional = conditional;
    result.s_FeatureProbabilities = std::move(featureProbabilities);
    result.s_Tail = std::move(tail);
    result.s_MostAnomalousCorrelate = std::move(mostAnomalousCorrelate);

    return true;
}

void CUnivariateTimeSeriesModel::countWeights(core_t::TTime time,
                                              const TDouble2Vec& value,
                                              double trendCountWeight,
                                              double residualCountWeight,
                                              double outlierWeightDerate,
                                              double countVarianceScale,
                                              TDouble2VecWeightsAry& trendWeights,
                                              TDouble2VecWeightsAry& residuaWeights) const {
    if (m_TrendModel->seasonalComponents().empty() == false) {
        countVarianceScale = 1.0;
    }

    TDouble2Vec seasonalWeight;
    this->seasonalWeight(0.0, time, seasonalWeight);
    double sample{m_TrendModel->detrend(time, value[0], 0.0, m_IsNonNegative)};
    auto weights = maths_t::CUnitWeights::UNIT;
    maths_t::setCount(std::min(residualCountWeight / trendCountWeight, 1.0), weights);
    maths_t::setSeasonalVarianceScale(seasonalWeight[0], weights);
    double outlierWeight{outliers::weight(
        *m_ResidualModel, weights,
        std::max(outlierWeightDerate, m_TrendModel->outlierWeightDerate(time, sample)),
        sample)};

    double changeWeight{m_TrendModel->countWeight(time)};
    trendCountWeight /= countVarianceScale;
    residualCountWeight *= changeWeight;

    TDouble2Vec trendOutlierWeight{outlierWeight * changeWeight};
    TDouble2Vec residualOutlierWeight{outlierWeight};

    maths_t::setCount(TDouble2Vec{trendCountWeight}, trendWeights);
    maths_t::setCount(TDouble2Vec{residualCountWeight}, residuaWeights);
    maths_t::setOutlierWeight(trendOutlierWeight, trendWeights);
    maths_t::setOutlierWeight(residualOutlierWeight, residuaWeights);
    maths_t::setCountVarianceScale(TDouble2Vec{countVarianceScale}, trendWeights);
    maths_t::setCountVarianceScale(TDouble2Vec{countVarianceScale}, residuaWeights);
}

void CUnivariateTimeSeriesModel::addCountWeights(core_t::TTime time,
                                                 double trendCountWeight,
                                                 double residualCountWeight,
                                                 double countVarianceScale,
                                                 TDouble2VecWeightsAry& trendWeights,
                                                 TDouble2VecWeightsAry& residuaWeights) const {
    if (m_TrendModel->seasonalComponents().empty()) {
        countVarianceScale = 1.0;
    }

    residualCountWeight *= m_TrendModel->countWeight(time);
    trendCountWeight /= countVarianceScale;

    maths_t::addCount(TDouble2Vec{trendCountWeight}, trendWeights);
    maths_t::addCount(TDouble2Vec{residualCountWeight}, residuaWeights);
}

void CUnivariateTimeSeriesModel::seasonalWeight(double confidence,
                                                core_t::TTime time,
                                                TDouble2Vec& weight) const {
    double scale{m_TrendModel->varianceScaleWeight(
        time, m_ResidualModel->marginalLikelihoodVariance(), confidence)(1)};
    weight.assign(1, std::max(scale, this->params().minimumSeasonalVarianceScale()));
}

std::uint64_t CUnivariateTimeSeriesModel::checksum(std::uint64_t seed) const {
    seed = common::CChecksum::calculate(seed, m_IsNonNegative);
    seed = common::CChecksum::calculate(seed, m_Controllers);
    seed = common::CChecksum::calculate(seed, m_TrendModel);
    seed = common::CChecksum::calculate(seed, m_ResidualModel);
    seed = common::CChecksum::calculate(seed, m_MultibucketFeature);
    seed = common::CChecksum::calculate(seed, m_MultibucketFeatureModel);
    seed = common::CChecksum::calculate(seed, m_AnomalyModel);
    return common::CChecksum::calculate(seed, m_Correlations != nullptr);
}

void CUnivariateTimeSeriesModel::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CUnivariateTimeSeriesModel");
    core::memory_debug::dynamicSize("m_Controllers", m_Controllers, mem);
    core::memory_debug::dynamicSize("m_TrendModel", m_TrendModel, mem);
    core::memory_debug::dynamicSize("m_ResidualModel", m_ResidualModel, mem);
    core::memory_debug::dynamicSize("m_MultibucketFeature", m_MultibucketFeature, mem);
    core::memory_debug::dynamicSize("m_MultibucketFeatureModel",
                                    m_MultibucketFeatureModel, mem);
    core::memory_debug::dynamicSize("m_AnomalyModel", m_AnomalyModel, mem);
}

std::size_t CUnivariateTimeSeriesModel::memoryUsage() const {
    return core::memory::dynamicSize(m_Controllers) +
           core::memory::dynamicSize(m_TrendModel) +
           core::memory::dynamicSize(m_ResidualModel) +
           core::memory::dynamicSize(m_MultibucketFeature) +
           core::memory::dynamicSize(m_MultibucketFeatureModel) +
           core::memory::dynamicSize(m_AnomalyModel);
}

bool CUnivariateTimeSeriesModel::acceptRestoreTraverser(const common::SModelRestoreParams& params,
                                                        core::CStateRestoreTraverser& traverser) {
    bool stateMissingControllerChecks{false};
    if (traverser.name() == VERSION_6_3_TAG || traverser.name() == VERSION_7_11_TAG) {
        stateMissingControllerChecks = (traverser.name() == VERSION_6_3_TAG);
        while (traverser.next()) {
            const std::string& name{traverser.name()};
            RESTORE_BUILT_IN(ID_6_3_TAG, m_Id)
            RESTORE_BOOL(IS_NON_NEGATIVE_6_3_TAG, m_IsNonNegative)
            RESTORE_BOOL(IS_FORECASTABLE_6_3_TAG, m_IsForecastable)
            RESTORE_SETUP_TEARDOWN(
                CONTROLLER_6_3_TAG,
                m_Controllers = std::make_unique<TDecayRateController2Ary>(),
                core::CPersistUtils::restore(CONTROLLER_6_3_TAG, *m_Controllers, traverser),
                /**/)
            RESTORE(TREND_MODEL_6_3_TAG,
                    traverser.traverseSubLevel(
                        [&, serialiser = CTimeSeriesDecompositionStateSerialiser{}](auto& traverser_) {
                            return serialiser(params.s_DecompositionParams, m_TrendModel, traverser_);
                        }))
            RESTORE(RESIDUAL_MODEL_6_3_TAG,
                    traverser.traverseSubLevel(
                        [&, serialiser = common::CPriorStateSerialiser{} ](auto& traverser_) {
                            return serialiser(params.s_DistributionParams,
                                              m_ResidualModel, traverser_);
                        }))
            RESTORE(MULTIBUCKET_FEATURE_6_3_TAG, traverser.traverseSubLevel([
                this, serialiser = CTimeSeriesMultibucketFeatureSerialiser{}
            ](auto& traverser_) {
                return serialiser(m_MultibucketFeature, traverser_);
            }))
            RESTORE(MULTIBUCKET_FEATURE_MODEL_6_3_TAG,
                    traverser.traverseSubLevel(
                        [&, serialiser = common::CPriorStateSerialiser{} ](auto& traverser_) {
                            return serialiser(params.s_DistributionParams,
                                              m_MultibucketFeatureModel, traverser_);
                        }))
            RESTORE_SETUP_TEARDOWN(
                ANOMALY_MODEL_6_3_TAG,
                m_AnomalyModel = std::make_unique<CTimeSeriesAnomalyModel>(),
                traverser.traverseSubLevel([&](auto& traverser_) {
                    return m_AnomalyModel->acceptRestoreTraverser(params, traverser_);
                }),
                /**/)
        }
    } else {
        LOG_ERROR(<< "Unsupported version '" << traverser.name() << "'");
        return false;
    }

    if (m_Controllers != nullptr && stateMissingControllerChecks) {
        (*m_Controllers)[E_TrendControl].checks(CDecayRateController::E_PredictionBias |
                                                CDecayRateController::E_PredictionErrorIncrease);
    }
    if (m_Controllers != nullptr && stateMissingControllerChecks) {
        (*m_Controllers)[E_ResidualControl].checks(
            CDecayRateController::E_PredictionBias | CDecayRateController::E_PredictionErrorIncrease |
            CDecayRateController::E_PredictionErrorDecrease);
    }

    this->checkRestoredInvariants();

    return true;
}

void CUnivariateTimeSeriesModel::checkRestoredInvariants() const {
    VIOLATES_INVARIANT_NO_EVALUATION(m_TrendModel, ==, nullptr);
    VIOLATES_INVARIANT_NO_EVALUATION(m_ResidualModel, ==, nullptr);
}

void CUnivariateTimeSeriesModel::persistModelsState(core::CStatePersistInserter& inserter) const {
    if (m_TrendModel != nullptr) {
        inserter.insertLevel(TREND_MODEL_6_3_TAG, [
            this, serialiser = CTimeSeriesDecompositionStateSerialiser{}
        ](auto& inserter_) { serialiser(*m_TrendModel, inserter_); });
    }

    if (m_ResidualModel != nullptr) {
        inserter.insertLevel(RESIDUAL_MODEL_6_3_TAG, [
            this, serialiser = common::CPriorStateSerialiser{}
        ](auto& inserter_) { serialiser(*m_ResidualModel, inserter_); });
    }
}

void CUnivariateTimeSeriesModel::acceptPersistInserter(core::CStatePersistInserter& inserter) const {

    // Note that we don't persist this->params() or the correlations
    // because that state is reinitialized.
    inserter.insertValue(VERSION_7_11_TAG, "");
    inserter.insertValue(ID_6_3_TAG, m_Id);
    inserter.insertValue(IS_NON_NEGATIVE_6_3_TAG, static_cast<int>(m_IsNonNegative));
    inserter.insertValue(IS_FORECASTABLE_6_3_TAG, static_cast<int>(m_IsForecastable));
    if (m_Controllers != nullptr) {
        core::CPersistUtils::persist(CONTROLLER_6_3_TAG, *m_Controllers, inserter);
    }
    if (m_TrendModel != nullptr) {
        inserter.insertLevel(TREND_MODEL_6_3_TAG, [
            this, serialiser = CTimeSeriesDecompositionStateSerialiser{}
        ](auto& inserter_) { serialiser(*m_TrendModel, inserter_); });
    }
    if (m_ResidualModel != nullptr) {
        inserter.insertLevel(RESIDUAL_MODEL_6_3_TAG, [
            this, serialiser = common::CPriorStateSerialiser{}
        ](auto& inserter_) { serialiser(*m_ResidualModel, inserter_); });
    }
    if (m_MultibucketFeature != nullptr) {
        inserter.insertLevel(MULTIBUCKET_FEATURE_6_3_TAG, [
            this, serialiser = CTimeSeriesMultibucketFeatureSerialiser{}
        ](auto& inserter_) { serialiser(m_MultibucketFeature, inserter_); });
    }
    if (m_MultibucketFeatureModel != nullptr) {
        inserter.insertLevel(MULTIBUCKET_FEATURE_MODEL_6_3_TAG, [
            this, serialiser = common::CPriorStateSerialiser{}
        ](auto& inserter_) { serialiser(*m_MultibucketFeatureModel, inserter_); });
    }
    if (m_AnomalyModel != nullptr) {
        inserter.insertLevel(ANOMALY_MODEL_6_3_TAG, [this](auto& inserter_) {
            m_AnomalyModel->acceptPersistInserter(inserter_);
        });
    }
}

maths_t::EDataType CUnivariateTimeSeriesModel::dataType() const {
    return m_ResidualModel->dataType();
}

CUnivariateTimeSeriesModel::TDoubleWeightsAry
CUnivariateTimeSeriesModel::unpack(const TDouble2VecWeightsAry& weights) {
    TDoubleWeightsAry result{maths_t::CUnitWeights::UNIT};
    for (std::size_t i = 0; i < weights.size(); ++i) {
        result[i] = weights[i][0];
    }
    return result;
}

const CTimeSeriesDecompositionInterface& CUnivariateTimeSeriesModel::trendModel() const {
    return *m_TrendModel;
}

const common::CPrior& CUnivariateTimeSeriesModel::residualModel() const {
    return *m_ResidualModel;
}

const CUnivariateTimeSeriesModel::TDecayRateController2Ary*
CUnivariateTimeSeriesModel::decayRateControllers() const {
    return m_Controllers.get();
}

CUnivariateTimeSeriesModel::CUnivariateTimeSeriesModel(const CUnivariateTimeSeriesModel& other,
                                                       std::size_t id,
                                                       bool isForForecast)
    : common::CModel(other.params()), m_Id(id),
      m_IsNonNegative(other.m_IsNonNegative), m_IsForecastable(other.m_IsForecastable),
      m_TrendModel(other.m_TrendModel->clone()),
      m_ResidualModel(other.m_ResidualModel->clone()),
      m_MultibucketFeature(!isForForecast && other.m_MultibucketFeature
                               ? other.m_MultibucketFeature->clone()
                               : nullptr),
      m_MultibucketFeatureModel(!isForForecast && other.m_MultibucketFeatureModel != nullptr
                                    ? other.m_MultibucketFeatureModel->clone()
                                    : nullptr),
      m_AnomalyModel(!isForForecast && other.m_AnomalyModel != nullptr
                         ? std::make_unique<CTimeSeriesAnomalyModel>(*other.m_AnomalyModel)
                         : nullptr) {
    if (!isForForecast && other.m_Controllers != nullptr) {
        m_Controllers = std::make_unique<TDecayRateController2Ary>(*other.m_Controllers);
    }
}

CUnivariateTimeSeriesModel::EUpdateResult
CUnivariateTimeSeriesModel::updateTrend(const common::CModelAddSamplesParams& params,
                                        const TTimeDouble2VecSizeTrVec& samples) {

    for (const auto& sample : samples) {
        if (sample.second.size() != 1) {
            LOG_ERROR(<< "Dimension mismatch: '" << sample.second.size() << " != 1'");
            return E_Failure;
        }
    }

    // Time order is not a total order, for example if the data are polled
    // the times of all samples will be the same. So break ties using the
    // sample value.
    TSize1Vec timeorder(samples.size());
    std::iota(timeorder.begin(), timeorder.end(), 0);
    std::stable_sort(timeorder.begin(), timeorder.end(),
                     [&samples](std::size_t lhs, std::size_t rhs) {
                         return common::COrderings::lexicographical_compare(
                             samples[lhs].first, samples[lhs].second,
                             samples[rhs].first, samples[rhs].second);
                     });

    // Do the update.

    TFloatMeanAccumulatorVec window;
    EUpdateResult result{E_Success};
    auto componentChangeCallback = [&window, &result](TFloatMeanAccumulatorVec window_) {
        if (window_.empty() == false) {
            window = std::move(window_);
        }
        result = E_Reset;
    };
    const auto& weights = params.trendWeights();
    const auto& modelAnnotationCallback = params.annotationCallback();
    double occupancy{params.bucketOccupancy()};
    core_t::TTime firstValueTime{params.firstValueTime()};

    for (auto i : timeorder) {
        core_t::TTime time{samples[i].first};
        double value{samples[i].second[0]};
        TDoubleWeightsAry weight{unpack(weights[i])};
        m_TrendModel->addPoint(time, value, weight, componentChangeCallback,
                               modelAnnotationCallback, occupancy, firstValueTime);
    }

    if (result == E_Reset) {
        this->reinitializeStateGivenNewComponent(params, std::move(window));
    }

    return result;
}

CUnivariateTimeSeriesModel::TTimeDouble2VecSizeTrVecDoublePr
CUnivariateTimeSeriesModel::updateResidualModels(const common::CModelAddSamplesParams& params,
                                                 TTimeDouble2VecSizeTrVec samples) {

    for (auto& residual : samples) {
        residual.second[0] = m_TrendModel->detrend(
            residual.first, residual.second[0], 0.0, m_IsNonNegative);
    }

    // We add the samples in value order since it makes clustering more stable.
    TSize1Vec valueorder(samples.size());
    std::iota(valueorder.begin(), valueorder.end(), 0);
    std::stable_sort(valueorder.begin(), valueorder.end(),
                     [&](std::size_t lhs, std::size_t rhs) {
                         return samples[lhs].second < samples[rhs].second;
                     });

    TDouble1Vec residuals;
    maths_t::TDoubleWeightsAry1Vec weights;
    TMeanAccumulator averageTimeAccumulator;
    weights.reserve(samples.size());
    residuals.reserve(samples.size());

    for (auto i : valueorder) {
        core_t::TTime time{samples[i].first};
        auto weight = unpack(params.priorWeights()[i]);
        residuals.push_back(samples[i].second[0]);
        weights.push_back(weight);
        averageTimeAccumulator.add(static_cast<double>(time),
                                   maths_t::countForUpdate(weight));
    }
    core_t::TTime averageTime{static_cast<core_t::TTime>(
        common::CBasicStatistics::mean(averageTimeAccumulator))};

    m_ResidualModel->addSamples(residuals, weights);
    m_ResidualModel->propagateForwardsByTime(params.propagationInterval());

    if (m_MultibucketFeatureModel != nullptr) {
        TDouble2Vec seasonalWeight;
        for (std::size_t i = 0; i < valueorder.size(); ++i) {
            core_t::TTime time{samples[i].first};
            this->seasonalWeight(0.0, time, seasonalWeight);
            maths_t::setSeasonalVarianceScale(seasonalWeight[0], weights[i]);
        }

        m_MultibucketFeature->add(averageTime, this->params().bucketLength(),
                                  residuals, weights);

        const auto & [ feature, featureWeight ] = m_MultibucketFeature->value();

        if (feature.empty() == false) {
            m_MultibucketFeatureModel->addSamples(feature, featureWeight);
            m_MultibucketFeatureModel->propagateForwardsByTime(params.propagationInterval());
        }
    }

    double decayRateMultiplier{this->updateDecayRates(params, averageTime, residuals)};

    return {std::move(samples), decayRateMultiplier};
}

double CUnivariateTimeSeriesModel::updateDecayRates(const common::CModelAddSamplesParams& params,
                                                    core_t::TTime time,
                                                    const TDouble1Vec& samples) {
    double multiplier{1.0};

    if (m_Controllers == nullptr) {
        return multiplier;
    }

    TDouble1VecVec errors[2];
    errors[0].reserve(samples.size());
    errors[1].reserve(samples.size());
    for (auto sample : samples) {
        this->appendPredictionErrors(params.propagationInterval(), sample, errors);
    }
    {
        CDecayRateController& controller{(*m_Controllers)[E_TrendControl]};
        TDouble1Vec trendMean{m_TrendModel->meanValue(time)};
        multiplier = controller.multiplier(
            trendMean, errors[E_TrendControl], this->params().bucketLength(),
            this->params().learnRate(), this->params().decayRate());
        if (multiplier != 1.0) {
            m_TrendModel->decayRate(multiplier * m_TrendModel->decayRate());
            LOG_TRACE(<< "trend decay rate = " << m_TrendModel->decayRate());
        }
    }
    {
        CDecayRateController& controller{(*m_Controllers)[E_ResidualControl]};
        TDouble1Vec residualMean{m_ResidualModel->marginalLikelihoodMean()};
        multiplier = controller.multiplier(
            residualMean, errors[E_ResidualControl], this->params().bucketLength(),
            this->params().learnRate(), this->params().decayRate());
        if (multiplier != 1.0) {
            double decayRate{multiplier * m_ResidualModel->decayRate()};
            m_ResidualModel->decayRate(decayRate);
            if (m_MultibucketFeatureModel != nullptr) {
                m_MultibucketFeatureModel->decayRate(decayRate);
            }
            LOG_TRACE(<< "prior decay rate = " << decayRate);
        }
    }
    return multiplier;
}

void CUnivariateTimeSeriesModel::appendPredictionErrors(double interval,
                                                        double sample_,
                                                        TDouble1VecVec (&result)[2]) {
    using TDecompositionPtr1Vec = core::CSmallVector<TDecompositionPtr, 1>;
    TDouble1Vec sample{sample_};
    TDecompositionPtr1Vec trend{m_TrendModel};
    if (auto error = predictionError(interval, m_ResidualModel, sample)) {
        result[E_ResidualControl].push_back(*error);
    }
    if (auto error = predictionError(trend, sample)) {
        result[E_TrendControl].push_back(*error);
    }
}

void CUnivariateTimeSeriesModel::reinitializeStateGivenNewComponent(
    const common::CModelAddSamplesParams& params,
    TFloatMeanAccumulatorVec residuals) {

    if (m_Controllers != nullptr) {
        m_ResidualModel->decayRate(m_ResidualModel->decayRate() /
                                   (*m_Controllers)[E_ResidualControl].multiplier());
        m_TrendModel->decayRate(m_TrendModel->decayRate() /
                                (*m_Controllers)[E_TrendControl].multiplier());
        for (auto& controller : *m_Controllers) {
            controller.reset();
        }
    }

    // We can't properly handle periodicity in the variance of the rate if
    // using a Poisson process so remove it from model selection if we detect
    // seasonality.
    m_ResidualModel->removeModels(maths::common::CPrior::CModelFilter().remove(
        maths::common::CPrior::E_Poisson));
    m_ResidualModel->setToNonInformative(0.0, m_ResidualModel->decayRate());

    // Reinitialize the residual model with any values we've been given. Note
    // that if we have sparse data we reduce the sample weights so we smoothly
    // transition to modelling only non-empty values. This must be reflected
    // in the sample weights when reinitialising the residual model.
    if (residuals.empty() == false) {
        double emptyBucketWeight{CModel::emptyBucketWeight(params.bucketOccupancy())};
        maths_t::TDoubleWeightsAry1Vec weights(1);
        double buckets{std::accumulate(residuals.begin(), residuals.end(), 0.0,
                                       [](auto partialBuckets, const auto& residual) {
                                           return partialBuckets +
                                                  common::CBasicStatistics::count(residual);
                                       }) /
                       this->params().learnRate()};
        double time{emptyBucketWeight * buckets / static_cast<double>(residuals.size())};
        for (const auto& residual : residuals) {
            double weight{emptyBucketWeight * common::CBasicStatistics::count(residual)};
            if (weight > 0.0) {
                weights[0] = maths_t::countWeight(weight);
                m_ResidualModel->addSamples({common::CBasicStatistics::mean(residual)}, weights);
                m_ResidualModel->propagateForwardsByTime(time);
            }
        }
    }

    // Reset the multi-bucket residual model. We can't reinitialize this from
    // the initial values because they are not typically at the granularity of
    // the job's bucket length.
    if (m_MultibucketFeature != nullptr) {
        m_MultibucketFeature->clear();
    }
    if (m_MultibucketFeatureModel != nullptr) {
        m_MultibucketFeatureModel->removeModels(
            maths::common::CPrior::CModelFilter().remove(maths::common::CPrior::E_Poisson));
        m_MultibucketFeatureModel->setToNonInformative(0.0, m_ResidualModel->decayRate());
    }

    if (m_Correlations != nullptr) {
        m_Correlations->clearCorrelationModels(m_Id);
    }

    if (m_AnomalyModel != nullptr) {
        m_AnomalyModel->reset();
    }
}

bool CUnivariateTimeSeriesModel::correlationModels(TSize1Vec& correlated,
                                                   TSize2Vec1Vec& variables,
                                                   TMultivariatePriorCPtrSizePr1Vec& correlationModels,
                                                   TModelCPtr1Vec& correlatedTimeSeriesModels) const {
    if (m_Correlations != nullptr) {
        correlated = m_Correlations->correlated(m_Id);
        m_Correlations->correlationModels(m_Id, correlated, variables, correlationModels,
                                          correlatedTimeSeriesModels);
    }
    return correlated.empty() == false;
}

CTimeSeriesCorrelations::CTimeSeriesCorrelations(double minimumSignificantCorrelation,
                                                 double decayRate)
    : m_MinimumSignificantCorrelation(minimumSignificantCorrelation),
      m_Correlations(MAXIMUM_CORRELATIONS, decayRate) {
}

CTimeSeriesCorrelations::CTimeSeriesCorrelations(const CTimeSeriesCorrelations& other,
                                                 bool isForPersistence)
    : m_MinimumSignificantCorrelation(other.m_MinimumSignificantCorrelation),
      m_SampleData(other.m_SampleData), m_Correlations(other.m_Correlations),
      m_CorrelatedLookup(other.m_CorrelatedLookup),
      m_TimeSeriesModels(isForPersistence ? TModelCPtrVec() : other.m_TimeSeriesModels) {
    for (const auto& model : other.m_CorrelationDistributionModels) {
        m_CorrelationDistributionModels.emplace(
            model.first,
            std::make_pair(TMultivariatePriorPtr(model.second.first->clone()),
                           model.second.second));
    }
}

CTimeSeriesCorrelations::~CTimeSeriesCorrelations() = default;

CTimeSeriesCorrelations* CTimeSeriesCorrelations::clone() const {
    return new CTimeSeriesCorrelations(*this);
}

CTimeSeriesCorrelations* CTimeSeriesCorrelations::cloneForPersistence() const {
    return new CTimeSeriesCorrelations(*this, true);
}

void CTimeSeriesCorrelations::processSamples() {
    using TSizeSizePrMultivariatePriorPtrDoublePrUMapCItrVec =
        std::vector<TSizeSizePrMultivariatePriorPtrDoublePrUMap::const_iterator>;

    // The priors use a shared pseudo random number generator which
    // generates a fixed sequence of random numbers. Since the order
    // of the correlated priors map can change across persist and
    // restore we can effectively get a different sequence of random
    // numbers depending on whether we persist and restore or not.
    // We'd like results to be as independent as possible from the
    // action of persisting and restoring so sort the collection to
    // preserve the random number sequence.
    TSizeSizePrMultivariatePriorPtrDoublePrUMapCItrVec iterators;
    iterators.reserve(m_CorrelationDistributionModels.size());
    for (auto i = m_CorrelationDistributionModels.begin();
         i != m_CorrelationDistributionModels.end(); ++i) {
        iterators.push_back(i);
    }
    std::sort(iterators.begin(), iterators.end(),
              core::CFunctional::SDereference<common::COrderings::SFirstLess>());

    TDouble10Vec1Vec multivariateSamples;
    maths_t::TDouble10VecWeightsAry1Vec multivariateWeights;
    for (auto i : iterators) {
        std::size_t pid1{i->first.first};
        std::size_t pid2{i->first.second};
        auto i1 = m_SampleData.find(pid1);
        auto i2 = m_SampleData.find(pid2);
        if (i1 == m_SampleData.end() || i2 == m_SampleData.end()) {
            continue;
        }

        const TMultivariatePriorPtr& prior{i->second.first};
        SSampleData* samples1{&i1->second};
        SSampleData* samples2{&i2->second};
        std::size_t n1{samples1->s_Times.size()};
        std::size_t n2{samples2->s_Times.size()};
        std::size_t indices[]{0, 1};
        if (n1 < n2) {
            std::swap(samples1, samples2);
            std::swap(n1, n2);
            std::swap(indices[0], indices[1]);
        }
        multivariateSamples.assign(n1, TDouble10Vec(2));
        multivariateWeights.assign(n1, maths_t::CUnitWeights::unit<TDouble10Vec>(2));

        TSize1Vec& tags2{samples2->s_Tags};
        TTime1Vec& times2{samples2->s_Times};

        common::COrderings::simultaneousSort(tags2, times2, samples2->s_Samples,
                                             samples2->s_Weights);
        for (auto j = tags2.begin(); j != tags2.end(); /**/) {
            auto k = std::upper_bound(j, tags2.end(), *j);
            std::size_t a = j - tags2.begin();
            std::size_t b = k - tags2.begin();
            common::COrderings::simultaneousSort(
                core::make_range(times2, a, b),
                core::make_range(samples2->s_Samples, a, b),
                core::make_range(samples2->s_Weights, a, b));
            j = k;
        }

        for (std::size_t j1 = 0; j1 < n1; ++j1) {
            std::size_t j2{0};
            if (n2 > 1) {
                std::size_t tag{samples1->s_Tags[j1]};
                core_t::TTime time{samples1->s_Times[j1]};
                std::size_t a_ = std::lower_bound(tags2.begin(), tags2.end(), tag) -
                                 tags2.begin();
                std::size_t b_ = std::upper_bound(tags2.begin(), tags2.end(), tag) -
                                 tags2.begin();
                std::size_t b{common::CTools::truncate(
                    static_cast<std::size_t>(
                        std::lower_bound(times2.begin() + a_, times2.begin() + b_, time) -
                        times2.begin()),
                    std::size_t(1), n2 - 1)};
                std::size_t a{b - 1};
                j2 = std::abs(times2[a] - time) < std::abs(times2[b] - time) ? a : b;
            }
            multivariateSamples[j1][indices[0]] = samples1->s_Samples[j1];
            multivariateSamples[j1][indices[1]] = samples2->s_Samples[j2];
            for (std::size_t w = 0; w < maths_t::NUMBER_WEIGHT_STYLES; ++w) {
                multivariateWeights[j1][w][indices[0]] = samples1->s_Weights[j1][w];
                multivariateWeights[j1][w][indices[1]] = samples2->s_Weights[j2][w];
            }
        }
        LOG_TRACE(<< "correlate samples = " << multivariateSamples
                  << ", correlate weights = " << multivariateWeights);

        prior->dataType(samples1->s_Type == maths_t::E_IntegerData ||
                                samples2->s_Type == maths_t::E_IntegerData
                            ? maths_t::E_IntegerData
                            : maths_t::E_ContinuousData);
        prior->addSamples(multivariateSamples, multivariateWeights);
        prior->propagateForwardsByTime(std::min(samples1->s_Interval, samples2->s_Interval));
        prior->decayRate(std::sqrt(samples1->s_Multiplier * samples2->s_Multiplier) *
                         prior->decayRate());
        LOG_TRACE(<< "correlation prior:" << core_t::LINE_ENDING << prior->print());
        LOG_TRACE(<< "decayRate = " << prior->decayRate());
    }

    m_Correlations.capture();
    m_SampleData.clear();
}

void CTimeSeriesCorrelations::refresh(const CTimeSeriesCorrelateModelAllocator& allocator) {
    using TSizeSizePrVec = std::vector<TSizeSizePr>;

    if (m_Correlations.changed()) {
        TSizeSizePrVec correlated;
        TDoubleVec correlationCoeffs;
        m_Correlations.mostCorrelated(
            static_cast<std::size_t>(
                1.2 * static_cast<double>(allocator.maxNumberCorrelations())),
            correlated, &correlationCoeffs);
        LOG_TRACE(<< "correlated = " << correlated);
        LOG_TRACE(<< "correlationCoeffs = " << correlationCoeffs);

        std::ptrdiff_t cutoff{
            std::upper_bound(correlationCoeffs.begin(), correlationCoeffs.end(),
                             0.5 * m_MinimumSignificantCorrelation,
                             [](double lhs, double rhs) {
                                 return std::fabs(lhs) > std::fabs(rhs);
                             }) -
            correlationCoeffs.begin()};
        LOG_TRACE(<< "cutoff = " << cutoff);

        correlated.erase(correlated.begin() + cutoff, correlated.end());
        if (correlated.empty()) {
            m_CorrelationDistributionModels.clear();
            this->refreshLookup();
            return;
        }

        // Extract the correlated pairs which are and aren't already
        // being modeled.
        TSizeSizePrVec present;
        TSizeVec presentRank;
        TSizeSizePrVec missing;
        TSizeVec missingRank;
        std::size_t np{static_cast<std::size_t>(
            std::max(0.9 * static_cast<double>(correlated.size()), 1.0))};
        std::size_t nm{static_cast<std::size_t>(
            std::max(0.1 * static_cast<double>(correlated.size()), 1.0))};
        present.reserve(np);
        presentRank.reserve(np);
        missing.reserve(nm);
        missingRank.reserve(nm);
        for (std::size_t j = 0; j < correlated.size(); ++j) {
            bool isPresent{m_CorrelationDistributionModels.count(correlated[j]) > 0};
            (isPresent ? present : missing).push_back(correlated[j]);
            (isPresent ? presentRank : missingRank).push_back(j);
        }

        // Remove any weakly correlated models.
        std::size_t initial{m_CorrelationDistributionModels.size()};
        common::COrderings::simultaneousSort(present, presentRank);
        for (auto i = m_CorrelationDistributionModels.begin();
             i != m_CorrelationDistributionModels.end();
             /**/) {
            std::size_t j = std::lower_bound(present.begin(), present.end(), i->first) -
                            present.begin();
            if (j == present.size() || i->first != present[j]) {
                i = m_CorrelationDistributionModels.erase(i);
            } else {
                i->second.second = correlationCoeffs[presentRank[j]];
                ++i;
            }
        }

        // Remove the remaining most weakly correlated models subject
        // to the capacity constraint.
        common::COrderings::simultaneousSort(presentRank, present, std::greater<>());
        for (std::size_t i = 0; m_CorrelationDistributionModels.size() >
                                allocator.maxNumberCorrelations();
             ++i) {
            m_CorrelationDistributionModels.erase(present[i]);
        }

        if (allocator.areAllocationsAllowed()) {
            for (std::size_t i = 0, nextChunk =
                                        std::min(allocator.maxNumberCorrelations(),
                                                 initial + allocator.chunkSize());
                 m_CorrelationDistributionModels.size() < allocator.maxNumberCorrelations() &&
                 i < missing.size() &&
                 (m_CorrelationDistributionModels.size() <= initial ||
                  !allocator.exceedsLimit(m_CorrelationDistributionModels.size()));
                 nextChunk = std::min(allocator.maxNumberCorrelations(),
                                      nextChunk + allocator.chunkSize())) {
                for (/**/; i < missing.size() &&
                           m_CorrelationDistributionModels.size() < nextChunk;
                     ++i) {
                    m_CorrelationDistributionModels.emplace(
                        missing[i], TMultivariatePriorPtrDoublePr{
                                        allocator.newPrior(),
                                        correlationCoeffs[missingRank[i]]});
                }
            }
        }

        this->refreshLookup();
    }
}

const CTimeSeriesCorrelations::TSizeSizePrMultivariatePriorPtrDoublePrUMap&
CTimeSeriesCorrelations::correlationModels() const {
    return m_CorrelationDistributionModels;
}

void CTimeSeriesCorrelations::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CTimeSeriesCorrelations");
    core::memory_debug::dynamicSize("m_SampleData", m_SampleData, mem);
    core::memory_debug::dynamicSize("m_Correlations", m_Correlations, mem);
    core::memory_debug::dynamicSize("m_CorrelatedLookup", m_CorrelatedLookup, mem);
    core::memory_debug::dynamicSize("m_CorrelationDistributionModels",
                                    m_CorrelationDistributionModels, mem);
}

std::size_t CTimeSeriesCorrelations::memoryUsage() const {
    return core::memory::dynamicSize(m_SampleData) +
           core::memory::dynamicSize(m_Correlations) +
           core::memory::dynamicSize(m_CorrelatedLookup) +
           core::memory::dynamicSize(m_CorrelationDistributionModels);
}

bool CTimeSeriesCorrelations::acceptRestoreTraverser(const common::SDistributionRestoreParams& params,
                                                     core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name{traverser.name()};
        RESTORE(K_MOST_CORRELATED_TAG, traverser.traverseSubLevel([this](auto& traverser_) {
            return m_Correlations.acceptRestoreTraverser(traverser_);
        }))
        RESTORE_WITH_UTILS(CORRELATED_LOOKUP_TAG, m_CorrelatedLookup)
        RESTORE(CORRELATION_MODELS_TAG, traverser.traverseSubLevel([&](auto& traverser_) {
            return this->restoreCorrelationModels(params, traverser_);
        }))
    } while (traverser.next());
    return true;
}

void CTimeSeriesCorrelations::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    // Note we don't persist the minimum significant correlation or the
    // models because that state is reinitialized. The sample is only
    // maintained transitively during an update at the end of a bucket
    // and so always empty at the point persistence occurs.

    inserter.insertLevel(K_MOST_CORRELATED_TAG, [this](auto& inserter_) {
        m_Correlations.acceptPersistInserter(inserter_);
    });
    core::CPersistUtils::persist(CORRELATED_LOOKUP_TAG, m_CorrelatedLookup, inserter);
    inserter.insertLevel(CORRELATION_MODELS_TAG, [this](auto& inserter_) {
        this->persistCorrelationModels(inserter_);
    });
}

bool CTimeSeriesCorrelations::restoreCorrelationModels(const common::SDistributionRestoreParams& params,
                                                       core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name{traverser.name()};
        RESTORE_SETUP_TEARDOWN(CORRELATION_MODEL_TAG,
                               TSizeSizePrMultivariatePriorPtrDoublePrPr prior,
                               traverser.traverseSubLevel([&](auto& traverser_) {
                                   return restore(params, prior, traverser_);
                               }),
                               m_CorrelationDistributionModels.insert(std::move(prior)))
    } while (traverser.next());
    return true;
}

void CTimeSeriesCorrelations::persistCorrelationModels(core::CStatePersistInserter& inserter) const {
    using TSizeSizePrMultivariatePriorPtrDoublePrUMapCItrVec =
        std::vector<TSizeSizePrMultivariatePriorPtrDoublePrUMap::const_iterator>;
    TSizeSizePrMultivariatePriorPtrDoublePrUMapCItrVec ordered;
    ordered.reserve(m_CorrelationDistributionModels.size());
    for (auto prior = m_CorrelationDistributionModels.begin();
         prior != m_CorrelationDistributionModels.end(); ++prior) {
        ordered.push_back(prior);
    }
    std::sort(ordered.begin(), ordered.end(),
              core::CFunctional::SDereference<common::COrderings::SFirstLess>());
    for (auto prior : ordered) {
        inserter.insertLevel(CORRELATION_MODEL_TAG, [&](auto& inserter_) {
            return persist(*prior, inserter_);
        });
    }
}

bool CTimeSeriesCorrelations::restore(const common::SDistributionRestoreParams& params,
                                      TSizeSizePrMultivariatePriorPtrDoublePrPr& model,
                                      core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name{traverser.name()};
        RESTORE_BUILT_IN(FIRST_CORRELATE_ID_TAG, model.first.first)
        RESTORE_BUILT_IN(SECOND_CORRELATE_ID_TAG, model.first.second)
        RESTORE(CORRELATION_MODEL_TAG,
                traverser.traverseSubLevel(
                    [&, serialiser = common::CPriorStateSerialiser{} ](auto& traverser_) {
                        return serialiser(params, model.second.first, traverser_);
                    }))
        RESTORE_BUILT_IN(CORRELATION_TAG, model.second.second)

    } while (traverser.next());
    return true;
}

void CTimeSeriesCorrelations::persist(const TConstSizeSizePrMultivariatePriorPtrDoublePrPr& model,
                                      core::CStatePersistInserter& inserter) {
    inserter.insertValue(FIRST_CORRELATE_ID_TAG, model.first.first);
    inserter.insertValue(SECOND_CORRELATE_ID_TAG, model.first.second);
    inserter.insertLevel(CORRELATION_MODEL_TAG, [
        &model, serialiser = common::CPriorStateSerialiser{}
    ](auto& inserter_) { serialiser(*model.second.first, inserter_); });
    inserter.insertValue(CORRELATION_TAG, model.second.second, core::CIEEE754::E_SinglePrecision);
}

void CTimeSeriesCorrelations::addTimeSeries(std::size_t id,
                                            const CUnivariateTimeSeriesModel& model) {
    m_Correlations.addVariables(id + 1);
    core::CAllocationStrategy::resize(m_TimeSeriesModels,
                                      std::max(id + 1, m_TimeSeriesModels.size()));
    m_TimeSeriesModels[id] = &model;
}

void CTimeSeriesCorrelations::removeTimeSeries(std::size_t id) {
    this->clearCorrelationModels(id);
    m_TimeSeriesModels[id] = nullptr;
}

void CTimeSeriesCorrelations::clearCorrelationModels(std::size_t id) {
    auto correlated_ = m_CorrelatedLookup.find(id);
    if (correlated_ != m_CorrelatedLookup.end()) {
        TSize1Vec& correlated{correlated_->second};
        for (const auto& correlate : correlated) {
            m_CorrelationDistributionModels.erase({id, correlate});
            m_CorrelationDistributionModels.erase({correlate, id});
        }
        this->refreshLookup();
    }
    m_Correlations.removeVariables({id});
}

void CTimeSeriesCorrelations::addSamples(std::size_t id,
                                         const common::CModelAddSamplesParams& params,
                                         const TTimeDouble2VecSizeTrVec& samples,
                                         double multiplier) {
    SSampleData& data{m_SampleData[id]};
    data.s_Type = params.type();
    data.s_Times.reserve(samples.size());
    data.s_Samples.reserve(samples.size());
    data.s_Tags.reserve(samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i) {
        data.s_Times.push_back(samples[i].first);
        data.s_Samples.push_back(samples[i].second[0]);
        data.s_Tags.push_back(samples[i].third);
        data.s_Weights.push_back(
            CUnivariateTimeSeriesModel::unpack(params.priorWeights()[i]));
    }
    data.s_Interval = params.propagationInterval();
    data.s_Multiplier = multiplier;
    m_Correlations.add(id, common::CBasicStatistics::median(data.s_Samples));
}

CTimeSeriesCorrelations::TSize1Vec CTimeSeriesCorrelations::correlated(std::size_t id) const {
    auto correlated = m_CorrelatedLookup.find(id);
    return correlated != m_CorrelatedLookup.end() ? correlated->second : TSize1Vec();
}

bool CTimeSeriesCorrelations::correlationModels(std::size_t id,
                                                TSize1Vec& correlated,
                                                TSize2Vec1Vec& variables,
                                                TMultivariatePriorCPtrSizePr1Vec& correlationModels,
                                                TModelCPtr1Vec& correlatedTimeSeriesModels) const {

    variables.clear();
    correlationModels.clear();
    correlatedTimeSeriesModels.clear();

    if (correlated.empty()) {
        return false;
    }

    variables.reserve(correlated.size());
    correlationModels.reserve(correlated.size());
    correlatedTimeSeriesModels.reserve(correlated.size());
    std::size_t end{0};
    for (auto correlate : correlated) {
        auto i = m_CorrelationDistributionModels.find({id, correlate});
        TSize2Vec variable{0, 1};
        if (i == m_CorrelationDistributionModels.end()) {
            i = m_CorrelationDistributionModels.find({correlate, id});
            std::swap(variable[0], variable[1]);
        }
        if (i == m_CorrelationDistributionModels.end()) {
            LOG_ERROR(<< "Unexpectedly missing prior for correlation (" << id
                      << "," << correlate << ")");
            continue;
        }
        if (std::fabs(i->second.second) < m_MinimumSignificantCorrelation) {
            LOG_TRACE(<< "Correlation " << i->second.second << " is too small to model");
            continue;
        }
        if (i->second.first->numberSamples() < MINIMUM_CORRELATE_PRIOR_SAMPLE_COUNT) {
            LOG_TRACE(<< "Too few samples in correlate model");
            continue;
        }
        correlated[end] = correlate;
        correlationModels.emplace_back(i->second.first.get(), variable[0]);
        variables.push_back(std::move(variable));
        ++end;
    }

    correlated.resize(variables.size());
    for (auto correlate : correlated) {
        correlatedTimeSeriesModels.push_back(m_TimeSeriesModels[correlate]);
    }

    return correlationModels.empty() == false;
}

void CTimeSeriesCorrelations::refreshLookup() {
    m_CorrelatedLookup.clear();
    for (const auto& prior : m_CorrelationDistributionModels) {
        std::size_t x0{prior.first.first};
        std::size_t x1{prior.first.second};
        m_CorrelatedLookup[x0].push_back(x1);
        m_CorrelatedLookup[x1].push_back(x0);
    }
    for (auto& prior : m_CorrelatedLookup) {
        std::sort(prior.second.begin(), prior.second.end());
    }
}

CMultivariateTimeSeriesModel::CMultivariateTimeSeriesModel(
    const common::CModelParams& params,
    const CTimeSeriesDecompositionInterface& trend,
    const common::CMultivariatePrior& residualModel,
    const TDecayRateController2Ary* controllers,
    const TMultibucketFeature* multibucketFeature,
    bool modelAnomalies)
    : common::CModel(params), m_ResidualModel(residualModel.clone()),
      m_MultibucketFeature(multibucketFeature != nullptr ? multibucketFeature->clone()
                                                         : nullptr),
      m_MultibucketFeatureModel(multibucketFeature != nullptr ? residualModel.clone() : nullptr),
      m_AnomalyModel(modelAnomalies ? std::make_unique<CTimeSeriesAnomalyModel>(
                                          params.bucketLength(),
                                          params.decayRate())
                                    : nullptr) {
    if (controllers != nullptr) {
        m_Controllers = std::make_unique<TDecayRateController2Ary>(*controllers);
    }
    for (std::size_t d = 0; d < this->dimension(); ++d) {
        m_TrendModel.emplace_back(trend.clone());
    }
}

CMultivariateTimeSeriesModel::CMultivariateTimeSeriesModel(const CMultivariateTimeSeriesModel& other)
    : common::CModel(other.params()), m_IsNonNegative(other.m_IsNonNegative),
      m_ResidualModel(other.m_ResidualModel->clone()),
      m_MultibucketFeature(other.m_MultibucketFeature != nullptr
                               ? other.m_MultibucketFeature->clone()
                               : nullptr),
      m_MultibucketFeatureModel(other.m_MultibucketFeatureModel != nullptr
                                    ? other.m_MultibucketFeatureModel->clone()
                                    : nullptr),
      m_AnomalyModel(other.m_AnomalyModel != nullptr
                         ? std::make_unique<CTimeSeriesAnomalyModel>(*other.m_AnomalyModel)
                         : nullptr) {
    if (other.m_Controllers != nullptr) {
        m_Controllers = std::make_unique<TDecayRateController2Ary>(*other.m_Controllers);
    }
    m_TrendModel.reserve(other.m_TrendModel.size());
    for (const auto& trend : other.m_TrendModel) {
        m_TrendModel.emplace_back(trend->clone());
    }
}

CMultivariateTimeSeriesModel::CMultivariateTimeSeriesModel(const common::SModelRestoreParams& params,
                                                           core::CStateRestoreTraverser& traverser)
    : common::CModel(params.s_Params) {
    if (traverser.traverseSubLevel([&](auto& traverser_) {
            return this->acceptRestoreTraverser(params, traverser_);
        }) == false) {
        traverser.setBadState();
    }
}

CMultivariateTimeSeriesModel::~CMultivariateTimeSeriesModel() = default;

std::size_t CMultivariateTimeSeriesModel::identifier() const {
    return 0;
}

CMultivariateTimeSeriesModel* CMultivariateTimeSeriesModel::clone(std::size_t /*id*/) const {
    return new CMultivariateTimeSeriesModel{*this};
}

CMultivariateTimeSeriesModel* CMultivariateTimeSeriesModel::cloneForPersistence() const {
    return new CMultivariateTimeSeriesModel{*this};
}

CMultivariateTimeSeriesModel* CMultivariateTimeSeriesModel::cloneForForecast() const {
    // Note: placeholder as there is no forecast support for multivariate time series for now
    return new CMultivariateTimeSeriesModel{*this};
}

bool CMultivariateTimeSeriesModel::isForecastPossible() const {
    return false;
}

void CMultivariateTimeSeriesModel::modelCorrelations(CTimeSeriesCorrelations& /*model*/) {
    // no-op
}

CMultivariateTimeSeriesModel::TSize2Vec1Vec CMultivariateTimeSeriesModel::correlates() const {
    return {};
}

void CMultivariateTimeSeriesModel::addBucketValue(const TTimeDouble2VecSizeTrVec& /*value*/) {
    // no-op
}

CMultivariateTimeSeriesModel::EUpdateResult
CMultivariateTimeSeriesModel::addSamples(const common::CModelAddSamplesParams& params,
                                         TTimeDouble2VecSizeTrVec samples) {
    if (samples.empty()) {
        return E_Success;
    }

    std::size_t dimension{this->dimension()};
    samples.erase(std::remove_if(samples.begin(), samples.end(),
                                 [&](const auto& sample) {
                                     return sample.second.size() != dimension;
                                 }),
                  samples.end());

    // Update the data characteristics.
    m_IsNonNegative = params.isNonNegative();
    maths_t::EDataType type{params.type()};
    m_ResidualModel->dataType(type);
    if (m_MultibucketFeatureModel != nullptr) {
        m_MultibucketFeatureModel->dataType(type);
    }
    for (auto& trendModel : m_TrendModel) {
        trendModel->dataType(type);
    }

    EUpdateResult result{this->updateTrend(params, samples)};

    this->updateResidualModels(params, std::move(samples));

    // Age the anomaly model. Note that update requires the probability
    // to be calculated. This is expensive to compute and so unlike our
    // other model components is done in that function.
    if (m_AnomalyModel != nullptr) {
        m_AnomalyModel->propagateForwardsByTime(params.propagationInterval());
    }

    return result;
}

void CMultivariateTimeSeriesModel::skipTime(core_t::TTime gap) {
    for (const auto& trend : m_TrendModel) {
        trend->skipTime(gap);
    }
}

CMultivariateTimeSeriesModel::TDouble2Vec
CMultivariateTimeSeriesModel::mode(core_t::TTime time,
                                   const TDouble2VecWeightsAry& weights) const {
    std::size_t dimension{this->dimension()};
    TDouble2Vec result(dimension);
    TDouble10Vec mode(m_ResidualModel->marginalLikelihoodMode(unpack(weights)));
    for (std::size_t d = 0; d < dimension; ++d) {
        result[d] = mode[d] + m_TrendModel[d]->value(time, 0.0, m_IsNonNegative).mean();
    }
    return result;
}

CMultivariateTimeSeriesModel::TDouble2Vec1Vec
CMultivariateTimeSeriesModel::correlateModes(core_t::TTime /*time*/,
                                             const TDouble2VecWeightsAry1Vec& /*weights*/) const {
    return {};
}

CMultivariateTimeSeriesModel::TDouble2Vec1Vec
CMultivariateTimeSeriesModel::residualModes(const TDouble2VecWeightsAry& weights) const {
    TDouble10Vec1Vec modes(m_ResidualModel->marginalLikelihoodModes(unpack(weights)));
    TDouble2Vec1Vec result;
    result.reserve(modes.size());
    for (const auto& mode : modes) {
        result.push_back(mode);
    }
    return result;
}

void CMultivariateTimeSeriesModel::detrend(const TTime2Vec1Vec& time_,
                                           double confidenceInterval,
                                           TDouble2Vec1Vec& value) const {
    std::size_t dimension{this->dimension()};
    core_t::TTime time{time_[0][0]};
    for (std::size_t d = 0; d < dimension; ++d) {
        value[0][d] = m_TrendModel[d]->detrend(time, value[0][d],
                                               confidenceInterval, m_IsNonNegative);
    }
}

CMultivariateTimeSeriesModel::TDouble2Vec
CMultivariateTimeSeriesModel::predict(core_t::TTime time,
                                      const TSizeDoublePr1Vec& /*correlated*/,
                                      TDouble2Vec hint) const {
    std::size_t dimension{this->dimension()};

    if (hint.size() == dimension) {
        for (std::size_t d = 0; d < dimension; ++d) {
            hint[d] = m_TrendModel[d]->detrend(time, hint[d], 0.0, m_IsNonNegative);
        }
    }

    TSize10Vec marginalize(dimension - 1);
    std::iota(marginalize.begin(), marginalize.end(), 1);

    TDouble2Vec result(dimension);
    TDouble10Vec mean(m_ResidualModel->marginalLikelihoodMean());
    for (std::size_t d = 0; d < dimension; --marginalize[std::min(d, dimension - 2)], ++d) {
        double trend{0.0};
        if (m_TrendModel[d]->initialized()) {
            trend = m_TrendModel[d]->value(time, 0.0, m_IsNonNegative).mean();
        }
        double median{mean[d]};
        if (m_ResidualModel->isNonInformative() == false) {
            TUnivariatePriorPtr marginal{
                m_ResidualModel->univariate(marginalize, NOTHING_TO_CONDITION).first};
            median = hint.empty()
                         ? common::CBasicStatistics::mean(
                               marginal->marginalLikelihoodConfidenceInterval(0.0))
                         : marginal->nearestMarginalLikelihoodMean(hint[d]);
        }
        result[d] = trend + median;
        if (m_IsNonNegative) {
            result[d] = std::max(result[d], 0.0);
        }
    }

    return result;
}

CMultivariateTimeSeriesModel::TDouble2Vec3Vec
CMultivariateTimeSeriesModel::confidenceInterval(core_t::TTime time,
                                                 double confidenceInterval,
                                                 const TDouble2VecWeightsAry& weights_) const {

    if (m_ResidualModel->isNonInformative()) {
        return {};
    }

    std::size_t dimension{this->dimension()};

    TSize10Vec marginalize(dimension - 1);
    std::iota(marginalize.begin(), marginalize.end(), 1);

    TDouble2Vec3Vec result(3, TDouble2Vec(dimension));

    maths_t::TDoubleWeightsAry weights{maths_t::CUnitWeights::UNIT};
    for (std::size_t d = 0; d < dimension; --marginalize[std::min(d, dimension - 2)], ++d) {
        double trend{m_TrendModel[d]->initialized()
                         ? m_TrendModel[d]->value(time, 0.0, m_IsNonNegative).mean()
                         : 0.0};

        for (std::size_t i = 0; i < maths_t::NUMBER_WEIGHT_STYLES; ++i) {
            weights[i] = weights_[i][d];
        }

        TUnivariatePriorPtr marginal{
            m_ResidualModel->univariate(marginalize, NOTHING_TO_CONDITION).first};
        double median{common::CBasicStatistics::mean(
            marginal->marginalLikelihoodConfidenceInterval(0.0, weights))};
        TDoubleDoublePr interval{marginal->marginalLikelihoodConfidenceInterval(
            confidenceInterval, weights)};

        result[0][d] = trend + interval.first;
        result[1][d] = trend + median;
        result[2][d] = trend + interval.second;
        if (m_IsNonNegative) {
            result[0][d] = std::max(result[0][d], 0.0);
            result[1][d] = std::max(result[1][d], 0.0);
            result[2][d] = std::max(result[2][d], 0.0);
        }
    }

    return result;
}

bool CMultivariateTimeSeriesModel::forecast(core_t::TTime /*firstDataTime*/,
                                            core_t::TTime /*lastDataTime*/,
                                            core_t::TTime /*startTime*/,
                                            core_t::TTime /*endTime*/,
                                            double /*confidenceInterval*/,
                                            const TDouble2Vec& /*minimum*/,
                                            const TDouble2Vec& /*maximum*/,
                                            const common::TForecastPushDatapointFunc& /*forecastPushDataPointFunc*/,
                                            std::string& messageOut) {
    LOG_DEBUG(<< forecast::ERROR_MULTIVARIATE);
    messageOut = forecast::ERROR_MULTIVARIATE;
    return false;
}

bool CMultivariateTimeSeriesModel::probability(const common::CModelProbabilityParams& params,
                                               const TTime2Vec1Vec& time_,
                                               const TDouble2Vec1Vec& value,
                                               common::SModelProbabilityResult& result) const {
    TSize2Vec coordinates(params.coordinates());
    if (coordinates.empty()) {
        coordinates.resize(this->dimension());
        std::iota(coordinates.begin(), coordinates.end(), 0);
    }
    TTail2Vec tail(coordinates.size(), maths_t::E_UndeterminedTail);

    result = common::SModelProbabilityResult{
        1.0, false, {{common::SModelProbabilityResult::E_SingleBucketProbability, 1.0}}, tail, {}};

    std::size_t dimension{this->dimension()};
    core_t::TTime time{time_[0][0]};
    TDouble10Vec1Vec sample{TDouble10Vec(dimension)};
    for (std::size_t d = 0; d < dimension; ++d) {
        sample[0][d] = m_TrendModel[d]->detrend(
            time, value[0][d], params.seasonalConfidenceInterval(), m_IsNonNegative);
    }
    maths_t::TDouble10VecWeightsAry1Vec weights{unpack(params.weights()[0])};

    struct SJointProbability {
        common::CJointProbabilityOfLessLikelySamples s_MarginalLower;
        common::CJointProbabilityOfLessLikelySamples s_MarginalUpper;
        common::CJointProbabilityOfLessLikelySamples s_ConditionalLower;
        common::CJointProbabilityOfLessLikelySamples s_ConditionalUpper;
    };
    using TJointProbability2Vec = core::CSmallVector<SJointProbability, 2>;

    auto updateJointProbabilities = [&](const TDouble10Vec2Vec& pl,
                                        const TDouble10Vec2Vec& pu,
                                        SJointProbability& joint) {
        joint.s_MarginalLower.add(pl[0][0]);
        joint.s_MarginalUpper.add(pu[0][0]);
        joint.s_ConditionalLower.add(pl[1][0]);
        joint.s_ConditionalUpper.add(pu[1][0]);
    };

    TJointProbability2Vec jointProbabilities(
        m_MultibucketFeatureModel != nullptr && params.useMultibucketFeatures() ? 2 : 1);

    double correlation{0.0};
    for (std::size_t i = 0; i < coordinates.size(); ++i) {
        maths_t::EProbabilityCalculation calculation{params.calculation(i)};
        TSize10Vec coordinate{coordinates[i]};
        TDouble10Vec2Vec pSingleBucket[2];
        TTail10Vec tail_;
        if (m_ResidualModel->probabilityOfLessLikelySamples(
                calculation, sample, weights, coordinate, pSingleBucket[0],
                pSingleBucket[1], tail_) == false) {
            LOG_ERROR(<< "Failed to compute P(" << sample << " | weight = " << weights << ")");
            return false;
        }
        updateJointProbabilities(pSingleBucket[0], pSingleBucket[1],
                                 jointProbabilities[0]);
        tail[i] = tail_[0];

        if (m_MultibucketFeatureModel != nullptr && params.useMultibucketFeatures()) {
            TDouble10Vec1Vec feature;
            std::tie(feature, std::ignore) = m_MultibucketFeature->value();
            if (feature.empty() == false) {
                TDouble10Vec2Vec pMultiBucket[2]{{{1.0}, {1.0}}, {{1.0}, {1.0}}};
                for (auto calculation_ : expand(calculation)) {
                    TDouble10Vec2Vec pl;
                    TDouble10Vec2Vec pu;
                    TTail10Vec dummy;
                    if (m_MultibucketFeatureModel->probabilityOfLessLikelySamples(
                            calculation_, feature,
                            maths_t::CUnitWeights::singleUnit<TDouble10Vec>(dimension),
                            coordinate, pl, pu, dummy) == false) {
                        LOG_ERROR(<< "Failed to compute P(" << feature << ")");
                        return false;
                    }
                    pMultiBucket[0][0][0] = std::min(pMultiBucket[0][0][0], pl[0][0]);
                    pMultiBucket[1][0][0] = std::min(pMultiBucket[1][0][0], pu[0][0]);
                    pMultiBucket[0][1][0] = std::min(pMultiBucket[0][1][0], pl[1][0]);
                    pMultiBucket[1][1][0] = std::min(pMultiBucket[1][1][0], pu[1][0]);
                }
                updateJointProbabilities(pMultiBucket[0], pMultiBucket[1],
                                         jointProbabilities[1]);
                correlation = m_MultibucketFeature->correlationWithBucketValue();
            }
        }
    }

    TDouble4Vec probabilities;
    for (const auto& probability : jointProbabilities) {
        double marginalLower;
        double marginalUpper;
        double conditionalLower;
        double conditionalUpper;
        if (probability.s_MarginalLower.calculate(marginalLower) == false ||
            probability.s_MarginalUpper.calculate(marginalUpper) == false ||
            probability.s_ConditionalLower.calculate(conditionalLower) == false ||
            probability.s_ConditionalUpper.calculate(conditionalUpper) == false) {
            return false;
        }
        probabilities.push_back((std::sqrt(marginalLower * conditionalLower) +
                                 std::sqrt(marginalUpper * conditionalUpper)) /
                                2.0);
    }

    common::SModelProbabilityResult::EFeatureProbabilityLabel labels[]{
        common::SModelProbabilityResult::E_SingleBucketProbability,
        common::SModelProbabilityResult::E_MultiBucketProbability};
    common::SModelProbabilityResult::TFeatureProbability4Vec featureProbabilities;
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        featureProbabilities.emplace_back(labels[i], probabilities[i]);
    }

    double pOverall{aggregateFeatureProbabilities(probabilities, correlation)};

    if (m_AnomalyModel != nullptr && params.useAnomalyModel()) {
        double residual{0.0};
        TDouble10Vec nearest(m_ResidualModel->nearestMarginalLikelihoodMean(sample[0]));
        TDouble2Vec seasonalWeight;
        this->seasonalWeight(0.0, time, seasonalWeight);
        for (std::size_t i = 0; i < dimension; ++i) {
            residual += (sample[0][i] - nearest[i]) /
                        std::max(std::sqrt(seasonalWeight[i]), 1.0);
        }
        double pSingleBucket{probabilities[0]};

        m_AnomalyModel->sample(params, time, residual, pSingleBucket, pOverall);

        double pAnomaly;
        std::tie(pOverall, pAnomaly) = m_AnomalyModel->probability(pSingleBucket, pOverall);
        probabilities.push_back(pAnomaly);
        featureProbabilities.emplace_back(
            common::SModelProbabilityResult::E_AnomalyModelProbability, pAnomaly);
    }

    result.s_Probability = pOverall;
    result.s_FeatureProbabilities = std::move(featureProbabilities);
    result.s_Tail = tail;

    return true;
}

void CMultivariateTimeSeriesModel::countWeights(core_t::TTime time,
                                                const TDouble2Vec& value,
                                                double trendCountWeight,
                                                double residualCountWeight,
                                                double outlierWeightDerate,
                                                double countVarianceScale,
                                                TDouble2VecWeightsAry& trendWeights,
                                                TDouble2VecWeightsAry& residuaWeights) const {
    std::size_t dimension{this->dimension()};

    TDouble2Vec seasonalWeight;
    this->seasonalWeight(0.0, time, seasonalWeight);

    TDouble2Vec trendCountWeights(dimension, trendCountWeight);
    TDouble2Vec residualCountWeights(dimension, residualCountWeight);
    TDouble2Vec trendOutlierWeight(dimension);
    TDouble2Vec residualOutlierWeight(dimension);
    TDouble2Vec countVarianceScales(dimension, 1.0);
    TDouble10Vec sample(dimension);
    for (std::size_t d = 0; d < dimension; ++d) {
        sample[d] = m_TrendModel[d]->detrend(time, value[d], 0.0, m_IsNonNegative);
        if (m_TrendModel[d]->seasonalComponents().empty()) {
            trendCountWeights[d] /= countVarianceScale;
            countVarianceScales[d] = countVarianceScale;
        }
    }
    for (std::size_t d = 0; d < dimension; ++d) {
        double changeWeight{m_TrendModel[d]->countWeight(time)};
        auto weights = maths_t::CUnitWeights::UNIT;
        maths_t::setCount(std::min(residualCountWeight / trendCountWeight, 1.0), weights);
        maths_t::setSeasonalVarianceScale(seasonalWeight[d], weights);
        double outlierWeight{outliers::weight(
            *conditional(*m_ResidualModel, d, sample), weights,
            std::max(outlierWeightDerate,
                     m_TrendModel[d]->outlierWeightDerate(time, sample[d])),
            sample[d])};
        residualCountWeights[d] *= changeWeight;
        trendOutlierWeight[d] = outlierWeight * changeWeight;
        residualOutlierWeight[d] = outlierWeight;
    }

    maths_t::setCount(trendCountWeights, trendWeights);
    maths_t::setCount(residualCountWeights, residuaWeights);
    maths_t::setOutlierWeight(trendOutlierWeight, trendWeights);
    maths_t::setOutlierWeight(trendOutlierWeight, residuaWeights);
    maths_t::setCountVarianceScale(countVarianceScales, trendWeights);
    maths_t::setCountVarianceScale(countVarianceScales, residuaWeights);
}

void CMultivariateTimeSeriesModel::addCountWeights(core_t::TTime time,
                                                   double trendCountWeight,
                                                   double residualCountWeight,
                                                   double countVarianceScale,
                                                   TDouble2VecWeightsAry& trendWeights,
                                                   TDouble2VecWeightsAry& residuaWeights) const {
    std::size_t dimension{this->dimension()};

    TDouble2Vec trendCountWeights(dimension, trendCountWeight);
    TDouble2Vec residualCountWeights(dimension, residualCountWeight);
    for (std::size_t d = 0; d < dimension; ++d) {
        if (m_TrendModel[d]->seasonalComponents().empty()) {
            trendCountWeights[d] /= countVarianceScale;
        }
        residualCountWeights[d] *= m_TrendModel[d]->countWeight(time);
    }

    maths_t::addCount(trendCountWeights, trendWeights);
    maths_t::addCount(residualCountWeights, residuaWeights);
}

void CMultivariateTimeSeriesModel::seasonalWeight(double confidence,
                                                  core_t::TTime time,
                                                  TDouble2Vec& weight) const {
    std::size_t dimension{this->dimension()};
    weight.resize(dimension);
    TDouble10Vec variances(m_ResidualModel->marginalLikelihoodVariances());
    for (std::size_t d = 0; d < dimension; ++d) {
        double scale{m_TrendModel[d]->varianceScaleWeight(time, variances[d],
                                                          confidence)(1)};
        weight[d] = std::max(scale, this->params().minimumSeasonalVarianceScale());
    }
}

std::uint64_t CMultivariateTimeSeriesModel::checksum(std::uint64_t seed) const {
    seed = common::CChecksum::calculate(seed, m_IsNonNegative);
    seed = common::CChecksum::calculate(seed, m_Controllers);
    seed = common::CChecksum::calculate(seed, m_TrendModel);
    seed = common::CChecksum::calculate(seed, m_ResidualModel);
    seed = common::CChecksum::calculate(seed, m_MultibucketFeature);
    seed = common::CChecksum::calculate(seed, m_MultibucketFeatureModel);
    return common::CChecksum::calculate(seed, m_AnomalyModel);
}

void CMultivariateTimeSeriesModel::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CMultivariateTimeSeriesModel");
    core::memory_debug::dynamicSize("m_Controllers", m_Controllers, mem);
    core::memory_debug::dynamicSize("m_TrendModel", m_TrendModel, mem);
    core::memory_debug::dynamicSize("m_ResidualModel", m_ResidualModel, mem);
    core::memory_debug::dynamicSize("m_MultibucketFeature", m_MultibucketFeature, mem);
    core::memory_debug::dynamicSize("m_MultibucketFeatureModel",
                                    m_MultibucketFeatureModel, mem);
    core::memory_debug::dynamicSize("m_AnomalyModel", m_AnomalyModel, mem);
}

std::size_t CMultivariateTimeSeriesModel::memoryUsage() const {
    return core::memory::dynamicSize(m_Controllers) +
           core::memory::dynamicSize(m_TrendModel) +
           core::memory::dynamicSize(m_ResidualModel) +
           core::memory::dynamicSize(m_MultibucketFeature) +
           core::memory::dynamicSize(m_MultibucketFeatureModel) +
           core::memory::dynamicSize(m_AnomalyModel);
}

bool CMultivariateTimeSeriesModel::acceptRestoreTraverser(const common::SModelRestoreParams& params,
                                                          core::CStateRestoreTraverser& traverser) {
    bool stateMissingControllerChecks{false};
    if (traverser.name() == VERSION_6_3_TAG || traverser.name() == VERSION_7_11_TAG) {
        stateMissingControllerChecks = (traverser.name() == VERSION_6_3_TAG);
        while (traverser.next()) {
            const std::string& name{traverser.name()};
            RESTORE_BOOL(IS_NON_NEGATIVE_6_3_TAG, m_IsNonNegative)
            RESTORE_SETUP_TEARDOWN(
                CONTROLLER_6_3_TAG,
                m_Controllers = std::make_unique<TDecayRateController2Ary>(),
                core::CPersistUtils::restore(CONTROLLER_6_3_TAG, *m_Controllers, traverser),
                /**/)
            RESTORE_SETUP_TEARDOWN(
                TREND_MODEL_6_3_TAG, m_TrendModel.emplace_back(),
                traverser.traverseSubLevel([
                    &, serialiser = CTimeSeriesDecompositionStateSerialiser{}
                ](auto& traverser_) {
                    return serialiser(params.s_DecompositionParams, m_TrendModel.back(), traverser_);
                }),
                /**/)
            RESTORE(RESIDUAL_MODEL_6_3_TAG,
                    traverser.traverseSubLevel(
                        [&, serialiser = common::CPriorStateSerialiser{} ](auto& traverser_) {
                            return serialiser(params.s_DistributionParams,
                                              m_ResidualModel, traverser_);
                        }))
            RESTORE(MULTIBUCKET_FEATURE_6_3_TAG, traverser.traverseSubLevel([
                this, serialiser = CTimeSeriesMultibucketFeatureSerialiser{}
            ](auto& traverser_) {
                return serialiser(m_MultibucketFeature, traverser_);
            }))
            RESTORE(MULTIBUCKET_FEATURE_MODEL_6_3_TAG,
                    traverser.traverseSubLevel(
                        [&, serialiser = common::CPriorStateSerialiser{} ](auto& traverser_) {
                            return serialiser(params.s_DistributionParams,
                                              m_MultibucketFeatureModel, traverser_);
                        }))
            RESTORE_SETUP_TEARDOWN(
                ANOMALY_MODEL_6_3_TAG,
                m_AnomalyModel = std::make_unique<CTimeSeriesAnomalyModel>(),
                traverser.traverseSubLevel([&](auto& traverser_) {
                    return m_AnomalyModel->acceptRestoreTraverser(params, traverser_);
                }),
                /**/)
        }
    } else {
        LOG_ERROR(<< "Unsupported version '" << traverser.name() << "'");
        return false;
    }

    this->checkRestoredInvariants();

    if (m_Controllers != nullptr && stateMissingControllerChecks) {
        (*m_Controllers)[E_TrendControl].checks(CDecayRateController::E_PredictionBias |
                                                CDecayRateController::E_PredictionErrorIncrease);
    }
    if (m_Controllers != nullptr && stateMissingControllerChecks) {
        (*m_Controllers)[E_ResidualControl].checks(
            CDecayRateController::E_PredictionBias | CDecayRateController::E_PredictionErrorIncrease |
            CDecayRateController::E_PredictionErrorDecrease);
    }

    return true;
}

void CMultivariateTimeSeriesModel::checkRestoredInvariants() const {
    for (const auto& trendModel : m_TrendModel) {
        VIOLATES_INVARIANT_NO_EVALUATION(trendModel, ==, nullptr);
    }
    VIOLATES_INVARIANT_NO_EVALUATION(m_ResidualModel, ==, nullptr);
    VIOLATES_INVARIANT(m_TrendModel.size(), !=, this->dimension());
    VIOLATES_INVARIANT_NO_EVALUATION(m_Controllers, ==, nullptr);
    VIOLATES_INVARIANT((*m_Controllers)[E_TrendControl].dimension(), !=, this->dimension());
    VIOLATES_INVARIANT((*m_Controllers)[E_ResidualControl].dimension(), !=,
                       this->dimension());
}

void CMultivariateTimeSeriesModel::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    // Note that we don't persist this->params() because that state
    // is reinitialized.
    inserter.insertValue(VERSION_7_11_TAG, "");
    inserter.insertValue(IS_NON_NEGATIVE_6_3_TAG, static_cast<int>(m_IsNonNegative));
    if (m_Controllers) {
        core::CPersistUtils::persist(CONTROLLER_6_3_TAG, *m_Controllers, inserter);
    }
    for (const auto& trend : m_TrendModel) {
        inserter.insertLevel(TREND_MODEL_6_3_TAG, [
            &trend, serialiser = CTimeSeriesDecompositionStateSerialiser{}
        ](auto& inserter_) { serialiser(*trend, inserter_); });
    }
    if (m_ResidualModel != nullptr) {
        inserter.insertLevel(RESIDUAL_MODEL_6_3_TAG, [
            this, serialiser = common::CPriorStateSerialiser{}
        ](auto& inserter_) { serialiser(*m_ResidualModel, inserter_); });
    }
    if (m_MultibucketFeature != nullptr) {
        inserter.insertLevel(MULTIBUCKET_FEATURE_6_3_TAG, [
            this, serialiser = CTimeSeriesMultibucketFeatureSerialiser{}
        ](auto& inserter_) { serialiser(m_MultibucketFeature, inserter_); });
    }
    if (m_MultibucketFeatureModel != nullptr) {
        inserter.insertLevel(MULTIBUCKET_FEATURE_MODEL_6_3_TAG, [
            this, serialiser = common::CPriorStateSerialiser{}
        ](auto& inserter_) { serialiser(*m_MultibucketFeatureModel, inserter_); });
    }
    if (m_AnomalyModel != nullptr) {
        inserter.insertLevel(ANOMALY_MODEL_6_3_TAG, [this](auto& inserter_) {
            m_AnomalyModel->acceptPersistInserter(inserter_);
        });
    }
}

void CMultivariateTimeSeriesModel::persistModelsState(core::CStatePersistInserter& /* inserter*/) const {
    // NO-OP
}

maths_t::EDataType CMultivariateTimeSeriesModel::dataType() const {
    return m_ResidualModel->dataType();
}

CMultivariateTimeSeriesModel::TDouble10VecWeightsAry
CMultivariateTimeSeriesModel::unpack(const TDouble2VecWeightsAry& weights) {
    TDouble10VecWeightsAry result{maths_t::CUnitWeights::unit<TDouble10Vec>(weights[0])};
    for (std::size_t i = 0; i < weights.size(); ++i) {
        result[i] = weights[i];
    }
    return result;
}

const CMultivariateTimeSeriesModel::TDecompositionPtr10Vec&
CMultivariateTimeSeriesModel::trendModel() const {
    return m_TrendModel;
}

const common::CMultivariatePrior& CMultivariateTimeSeriesModel::residualModel() const {
    return *m_ResidualModel;
}

const CMultivariateTimeSeriesModel::TDecayRateController2Ary*
CMultivariateTimeSeriesModel::decayRateControllers() const {
    return m_Controllers.get();
}

CMultivariateTimeSeriesModel::EUpdateResult
CMultivariateTimeSeriesModel::updateTrend(const common::CModelAddSamplesParams& params,
                                          const TTimeDouble2VecSizeTrVec& samples) {

    std::size_t dimension{this->dimension()};

    for (const auto& sample : samples) {
        if (sample.second.size() != dimension) {
            LOG_ERROR(<< "Dimension mismatch: '" << sample.second.size()
                      << " != " << m_TrendModel.size() << "'");
            return E_Failure;
        }
    }

    // Time order is not a total order, for example if the data are polled
    // the times of all samples will be the same. So break ties using the
    // sample value.
    TSize1Vec timeorder(samples.size());
    std::iota(timeorder.begin(), timeorder.end(), 0);
    std::stable_sort(timeorder.begin(), timeorder.end(),
                     [&samples](std::size_t lhs, std::size_t rhs) {
                         return common::COrderings::lexicographical_compare(
                             samples[lhs].first, samples[lhs].second,
                             samples[rhs].first, samples[rhs].second);
                     });

    // Do the update.

    EUpdateResult result{E_Success};
    auto componentChangeCallback = [&result](TFloatMeanAccumulatorVec) {
        result = E_Reset;
    };
    const auto& weights = params.trendWeights();
    const auto& modelAnnotationCallback = params.annotationCallback();
    double occupancy{params.bucketOccupancy()};
    core_t::TTime firstValueTime{params.firstValueTime()};

    maths_t::TDoubleWeightsAry weight;
    for (auto i : timeorder) {
        core_t::TTime time{samples[i].first};
        TDouble10Vec value(samples[i].second);
        for (std::size_t d = 0; d < dimension; ++d) {
            for (std::size_t j = 0; j < maths_t::NUMBER_WEIGHT_STYLES; ++j) {
                weight[j] = weights[i][j][d];
            }
            m_TrendModel[d]->addPoint(time, value[d], weight, componentChangeCallback,
                                      modelAnnotationCallback, occupancy, firstValueTime);
        }
    }

    if (result == E_Reset) {
        TFloatMeanAccumulatorVec10Vec window(dimension);
        for (std::size_t d = 0; d < dimension; ++d) {
            window[d] = m_TrendModel[d]->residuals(m_IsNonNegative);
        }
        this->reinitializeStateGivenNewComponent(params, std::move(window));
    }

    return result;
}

void CMultivariateTimeSeriesModel::updateResidualModels(const common::CModelAddSamplesParams& params,
                                                        TTimeDouble2VecSizeTrVec samples) {

    std::size_t dimension{this->dimension()};

    for (auto& residual : samples) {
        core_t::TTime time{residual.first};
        for (std::size_t d = 0; d < dimension; ++d) {
            residual.second[d] = m_TrendModel[d]->detrend(time, residual.second[d],
                                                          0.0, m_IsNonNegative);
        }
    }

    // We add the samples in value order since it makes clustering more stable.
    TSize1Vec valueorder(samples.size());
    std::iota(valueorder.begin(), valueorder.end(), 0);
    std::stable_sort(valueorder.begin(), valueorder.end(),
                     [&](std::size_t lhs, std::size_t rhs) {
                         return samples[lhs].second < samples[rhs].second;
                     });

    TDouble10Vec1Vec residuals;
    maths_t::TDouble10VecWeightsAry1Vec weights;
    samples.reserve(samples.size());
    weights.reserve(samples.size());
    TMeanAccumulator averageTimeAccumulator;
    for (auto i : valueorder) {
        core_t::TTime time{samples[i].first};
        auto weight = unpack(params.priorWeights()[i]);
        residuals.push_back(samples[i].second);
        weights.push_back(weight);
        averageTimeAccumulator.add(static_cast<double>(time));
    }
    core_t::TTime averageTime{static_cast<core_t::TTime>(
        common::CBasicStatistics::mean(averageTimeAccumulator))};

    m_ResidualModel->addSamples(residuals, weights);
    m_ResidualModel->propagateForwardsByTime(params.propagationInterval());

    if (m_MultibucketFeatureModel != nullptr) {
        TDouble2Vec seasonalWeight;
        for (std::size_t i = 0; i < valueorder.size(); ++i) {
            core_t::TTime time{samples[valueorder[i]].first};
            this->seasonalWeight(0.0, time, seasonalWeight);
            maths_t::setSeasonalVarianceScale(TDouble10Vec(seasonalWeight), weights[i]);
        }
        m_MultibucketFeature->add(averageTime, this->params().bucketLength(),
                                  residuals, weights);

        const auto & [ feature, featureWeight ] = m_MultibucketFeature->value();

        if (feature.empty() == false) {
            m_MultibucketFeatureModel->addSamples(feature, featureWeight);
            m_MultibucketFeatureModel->propagateForwardsByTime(params.propagationInterval());
        }
    }

    this->updateDecayRates(params, averageTime, residuals);
}

void CMultivariateTimeSeriesModel::updateDecayRates(const common::CModelAddSamplesParams& params,
                                                    core_t::TTime time,
                                                    const TDouble10Vec1Vec& samples) {
    if (m_Controllers != nullptr) {
        TDouble1VecVec errors[2];
        errors[0].reserve(samples.size());
        errors[1].reserve(samples.size());
        for (const auto& sample : samples) {
            this->appendPredictionErrors(params.propagationInterval(), sample, errors);
        }
        {
            CDecayRateController& controller{(*m_Controllers)[E_TrendControl]};
            TDouble1Vec trendMean(this->dimension());
            for (std::size_t d = 0; d < trendMean.size(); ++d) {
                trendMean[d] = m_TrendModel[d]->meanValue(time);
            }
            double multiplier{controller.multiplier(
                trendMean, errors[E_TrendControl], this->params().bucketLength(),
                this->params().learnRate(), this->params().decayRate())};
            if (multiplier != 1.0) {
                for (const auto& trend : m_TrendModel) {
                    trend->decayRate(multiplier * trend->decayRate());
                }
                LOG_TRACE(<< "trend decay rate = " << m_TrendModel[0]->decayRate());
            }
        }
        {
            CDecayRateController& controller{(*m_Controllers)[E_ResidualControl]};
            TDouble1Vec residualMean(m_ResidualModel->marginalLikelihoodMean());
            double multiplier{controller.multiplier(
                residualMean, errors[E_ResidualControl], this->params().bucketLength(),
                this->params().learnRate(), this->params().decayRate())};
            if (multiplier != 1.0) {
                m_ResidualModel->decayRate(multiplier * m_ResidualModel->decayRate());
                LOG_TRACE(<< "prior decay rate = " << m_ResidualModel->decayRate());
            }
        }
    }
}

void CMultivariateTimeSeriesModel::appendPredictionErrors(double interval,
                                                          const TDouble10Vec& sample,
                                                          TDouble1VecVec (&result)[2]) {
    if (auto error = predictionError(interval, m_ResidualModel, sample)) {
        result[E_ResidualControl].push_back(*error);
    }
    if (auto error = predictionError(m_TrendModel, sample)) {
        result[E_TrendControl].push_back(*error);
    }
}

void CMultivariateTimeSeriesModel::reinitializeStateGivenNewComponent(
    const common::CModelAddSamplesParams& params,
    TFloatMeanAccumulatorVec10Vec residuals) {

    if (m_Controllers != nullptr) {
        m_ResidualModel->decayRate(m_ResidualModel->decayRate() /
                                   (*m_Controllers)[E_ResidualControl].multiplier());
        for (auto& trend : m_TrendModel) {
            trend->decayRate(trend->decayRate() /
                             (*m_Controllers)[E_TrendControl].multiplier());
        }
        for (auto& controller : *m_Controllers) {
            controller.reset();
        }
    }

    // Reinitialize the residual model with any values we've been given. Note
    // that if we have sparse data we reduce the sample weights so we smoothly
    // transition to modelling only non-empty values. This must be reflected
    // in the sample weights when reinitialising the residual model.
    m_ResidualModel->setToNonInformative(0.0, m_ResidualModel->decayRate());
    if (residuals.empty() == false) {
        std::size_t dimension{this->dimension()};
        double emptyBucketWeight{CModel::emptyBucketWeight(params.bucketOccupancy())};

        TDouble10VecVec samples;
        TDoubleVec weights;
        double time{0.0};
        for (std::size_t d = 0; d < dimension; ++d) {
            samples.resize(residuals[d].size(), TDouble10Vec(dimension));
            weights.resize(residuals[d].size(), std::numeric_limits<double>::max());
            double buckets{
                std::accumulate(residuals[d].begin(), residuals[d].end(), 0.0,
                                [](auto partialBuckets, const auto& residual) {
                                    return partialBuckets +
                                           common::CBasicStatistics::count(residual);
                                }) /
                this->params().learnRate()};
            time += buckets / static_cast<double>(residuals.size());
            for (std::size_t i = 0; i < residuals[d].size(); ++i) {
                samples[i][d] = common::CBasicStatistics::mean(residuals[d][i]);
                weights[i] = std::min(
                    weights[i], emptyBucketWeight * static_cast<double>(common::CBasicStatistics::count(
                                                        residuals[d][i])));
            }
        }
        time *= emptyBucketWeight / static_cast<double>(dimension);

        maths_t::TDouble10VecWeightsAry1Vec weight(1);
        for (std::size_t i = 0; i < samples.size(); ++i) {
            if (weights[i] > 0.0) {
                weight[0] = maths_t::countWeight(weights[i], dimension);
                m_ResidualModel->addSamples({samples[i]}, weight);
                m_ResidualModel->propagateForwardsByTime(time);
            }
        }
    }

    // Reset the multi-bucket residual model. We can't reinitialize this from
    // the initial values because they are not typically at the granularity of
    // the job's bucket length.
    if (m_MultibucketFeature != nullptr) {
        m_MultibucketFeature->clear();
    }
    if (m_MultibucketFeatureModel != nullptr) {
        m_MultibucketFeatureModel->setToNonInformative(0.0, m_ResidualModel->decayRate());
    }

    if (m_AnomalyModel != nullptr) {
        m_AnomalyModel->reset();
    }
}

std::size_t CMultivariateTimeSeriesModel::dimension() const {
    return m_ResidualModel->dimension();
}
}
}
}
