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

#include <maths/time_series/CTrendComponent.h>

#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/Constants.h>
#include <core/RestoreMacros.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CChecksum.h>
#include <maths/common/CIntegerTools.h>
#include <maths/common/CLeastSquaresOnlineRegression.h>
#include <maths/common/CLeastSquaresOnlineRegressionDetail.h>
#include <maths/common/CLinearAlgebra.h>
#include <maths/common/CNaiveBayes.h>
#include <maths/common/COrderings.h>
#include <maths/common/COrderingsSimultaneousSort.h>
#include <maths/common/CSampling.h>
#include <maths/common/CStatisticalTests.h>
#include <maths/common/CTools.h>

#include <maths/time_series/CSignal.h>

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/students_t.hpp>

#include <algorithm>
#include <cmath>
#include <exception>

namespace ml {
namespace maths {
namespace time_series {
namespace {
using TVector2x1 = CTrendComponent::TVector2x1;

const double TIME_SCALES[]{144.0, 72.0, 36.0, 12.0, 4.0, 1.0, 0.25, 0.05};
const std::size_t NUMBER_MODELS{std::size(TIME_SCALES)};
const double MINIMUM_WEIGHT_TO_USE_MODEL_FOR_PREDICTION{0.01};
const double MODEL_MSE_DECREASE_SIGNFICANT{0.05};
const double MAX_CONDITION{1e12};
const core_t::TTime UNSET_TIME{0};
const std::size_t NO_CHANGE_LABEL{0};
const std::size_t LEVEL_CHANGE_LABEL{1};
// BIC penalty multipliers for different polynomial orders
// Higher penalties make the model more conservative in selecting higher orders
const double BIC_PENALTY_ORDER_1{1.0};  // Linear model (baseline)
const double BIC_PENALTY_ORDER_2{4.0};  // Quadratic model (very strong penalty to prevent overfitting)
const double BIC_PENALTY_ORDER_3{8.0};  // Cubic model (extremely strong penalty to prevent overfitting)

class CChangeForecastFeatureWeight : public common::CNaiveBayesFeatureWeight {
public:
    void add(std::size_t class_, double logLikelihood) override {
        if (class_ == NO_CHANGE_LABEL) {
            m_LogLikelihood = logLikelihood;
        }
    }

    double calculate() const override {
        // Downweight features for which we don't have sufficient examples
        // of the time series not changing.
        // Note that m_LogLikelihood = 0.5 * (x - m)^2 / sigma^2 so 4.5
        // corresponds to the case the feature value is at the 3 sigma
        // point of the conditional distribution.
        return common::CTools::logisticFunction((4.5 + m_LogLikelihood) / 4.5, 0.1);
    }

private:
    double m_LogLikelihood{0.0};
};

//! Get the desired weight for the regression model.
double modelWeight(double targetDecayRate, double modelDecayRate) {
    return targetDecayRate == modelDecayRate
               ? 1.0
               : std::min(targetDecayRate, modelDecayRate) /
                     std::max(targetDecayRate, modelDecayRate);
}

//! We scale the time used for the regression model to improve the condition
//! of the design matrix.
double scaleTime(core_t::TTime time, core_t::TTime origin) {
    return static_cast<double>(time - origin) / static_cast<double>(core::constants::WEEK);
}

//! Get the \p confidence interval for \p prediction and \p variance.
TVector2x1 confidenceInterval(double prediction, double variance, double confidence) {
    if (variance > 0.0) {
        try {
            boost::math::normal normal{prediction, std::sqrt(variance)};
            double ql{boost::math::quantile(normal, (100.0 - confidence) / 200.0)};
            double qu{boost::math::quantile(normal, (100.0 + confidence) / 200.0)};
            return TVector2x1{{ql, qu}};
        } catch (const std::exception& e) {
            LOG_ERROR(<< "Failed calculating confidence interval: " << e.what()
                      << ", prediction = " << prediction << ", variance = " << variance
                      << ", confidence = " << confidence);
        }
    }
    return TVector2x1{prediction};
}

TVector2x1 confidenceIntervalT(double prediction, double variance, double confidence, double n_eff) {
    if (variance > 0.0 && n_eff > 0.0) {
        try {
            // Use t-distribution for small effective sample sizes
            double df = std::max(n_eff - 1.0, 1.0); // Degrees of freedom
            boost::math::students_t t{df};
            double t_critical = boost::math::quantile(t, (100.0 + confidence) / 200.0);
            double margin = t_critical * std::sqrt(variance);
            
            LOG_TRACE(<< "T-Distribution Debug: df=" << df << ", t_critical=" << t_critical 
                      << ", margin=" << margin << ", bounds=[" << (prediction - margin) 
                      << ", " << (prediction + margin) << "]");
            
            return TVector2x1{{prediction - margin, prediction + margin}};
        } catch (const std::exception& e) {
            LOG_ERROR(<< "Failed calculating t-confidence interval: " << e.what()
                      << ", prediction = " << prediction << ", variance = " << variance
                      << ", confidence = " << confidence << ", n_eff = " << n_eff);
        }
    }
    return TVector2x1{prediction};
}

common::CNaiveBayesFeatureDensityFromPrior naiveBayesExemplar(double decayRate) {
    return common::CNaiveBayesFeatureDensityFromPrior{common::CNormalMeanPrecConjugate::nonInformativePrior(
        maths_t::E_ContinuousData, TIME_SCALES[NUMBER_MODELS - 1] * decayRate)};
}

common::CNaiveBayes initialProbabilityOfChangeModel(double decayRate) {
    return common::CNaiveBayes{naiveBayesExemplar(decayRate),
                               TIME_SCALES[NUMBER_MODELS - 1] * decayRate};
}

common::CNormalMeanPrecConjugate initialMagnitudeOfChangeModel(double decayRate) {
    return common::CNormalMeanPrecConjugate::nonInformativePrior(
        maths_t::E_ContinuousData, decayRate);
}

const std::string VERSION_7_1_TAG("7.1");

const core::TPersistenceTag TARGET_DECAY_RATE_TAG{"a", "target_decay_rate"};
const core::TPersistenceTag FIRST_UPDATE_TAG{"b", "first_update"};
const core::TPersistenceTag LAST_UPDATE_TAG{"c", "last_update"};
const core::TPersistenceTag REGRESSION_ORIGIN_TAG{"d", "regression_origin"};
const core::TPersistenceTag MODEL_TAG{"e", "model"};
const core::TPersistenceTag PREDICTION_ERROR_VARIANCE_TAG{"f", "prediction_error_variance"};
const core::TPersistenceTag VALUE_MOMENTS_TAG{"g", "value_moments"};
const core::TPersistenceTag TIME_OF_LAST_LEVEL_CHANGE_TAG{"h", "time_of_last_level_change"};
const core::TPersistenceTag PROBABILITY_OF_LEVEL_CHANGE_MODEL_TAG{
    "i", "probability_of_level_change_model"};
const core::TPersistenceTag MAGNITUDE_OF_LEVEL_CHANGE_MODEL_TAG{"j", "magnitude_of_level_change_model"};
// Version 7.1
const core::TPersistenceTag WEIGHT_7_1_TAG{"a", "weight"};
const core::TPersistenceTag REGRESSION_7_1_TAG{"b", "regression"};
const core::TPersistenceTag MSE_7_1_TAG{"c", "mse"};
}

CTrendComponent::CTrendComponent(double decayRate)
    : m_DefaultDecayRate(decayRate), m_TargetDecayRate(decayRate),
      m_FirstUpdate(UNSET_TIME), m_LastUpdate(UNSET_TIME),
      m_RegressionOrigin(UNSET_TIME), m_PredictionErrorVariance(0.0),
      m_TimeOfLastLevelChange(UNSET_TIME),
      m_ProbabilityOfLevelChangeModel(initialProbabilityOfChangeModel(decayRate)),
      m_MagnitudeOfLevelChangeModel(initialMagnitudeOfChangeModel(decayRate)) {
    for (std::size_t i = 0; i < NUMBER_MODELS; ++i) {
        m_TrendModels.emplace_back(modelWeight(1.0, TIME_SCALES[i]));
    }
}

void CTrendComponent::swap(CTrendComponent& other) {
    std::swap(m_DefaultDecayRate, other.m_DefaultDecayRate);
    std::swap(m_TargetDecayRate, other.m_TargetDecayRate);
    std::swap(m_FirstUpdate, other.m_FirstUpdate);
    std::swap(m_LastUpdate, other.m_LastUpdate);
    std::swap(m_RegressionOrigin, other.m_RegressionOrigin);
    m_TrendModels.swap(other.m_TrendModels);
    std::swap(m_PredictionErrorVariance, other.m_PredictionErrorVariance);
    std::swap(m_ValueMoments, other.m_ValueMoments);
    std::swap(m_TimeOfLastLevelChange, other.m_TimeOfLastLevelChange);
    m_ProbabilityOfLevelChangeModel.swap(other.m_ProbabilityOfLevelChangeModel);
    m_MagnitudeOfLevelChangeModel.swap(other.m_MagnitudeOfLevelChangeModel);
}

void CTrendComponent::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(TARGET_DECAY_RATE_TAG, m_TargetDecayRate);
    inserter.insertValue(FIRST_UPDATE_TAG, m_FirstUpdate);
    inserter.insertValue(LAST_UPDATE_TAG, m_LastUpdate);
    inserter.insertValue(REGRESSION_ORIGIN_TAG, m_RegressionOrigin);
    for (const auto& model : m_TrendModels) {
        inserter.insertLevel(MODEL_TAG, [&model](auto& inserter_) {
            model.acceptPersistInserter(inserter_);
        });
    }
    inserter.insertValue(PREDICTION_ERROR_VARIANCE_TAG, m_PredictionErrorVariance,
                         core::CIEEE754::E_DoublePrecision);
    inserter.insertValue(VALUE_MOMENTS_TAG, m_ValueMoments.toDelimited());
    inserter.insertValue(TIME_OF_LAST_LEVEL_CHANGE_TAG, m_TimeOfLastLevelChange);
    inserter.insertLevel(PROBABILITY_OF_LEVEL_CHANGE_MODEL_TAG, [this](auto& inserter_) {
        m_ProbabilityOfLevelChangeModel.acceptPersistInserter(inserter_);
    });
    inserter.insertLevel(MAGNITUDE_OF_LEVEL_CHANGE_MODEL_TAG, [this](auto& inserter_) {
        m_MagnitudeOfLevelChangeModel.acceptPersistInserter(inserter_);
    });
}

bool CTrendComponent::acceptRestoreTraverser(const common::SDistributionRestoreParams& params,
                                             core::CStateRestoreTraverser& traverser) {
    std::size_t i{0};
    do {
        const std::string& name{traverser.name()};
        RESTORE_BUILT_IN(TARGET_DECAY_RATE_TAG, m_TargetDecayRate)
        RESTORE_BUILT_IN(FIRST_UPDATE_TAG, m_FirstUpdate)
        RESTORE_BUILT_IN(LAST_UPDATE_TAG, m_LastUpdate)
        RESTORE_BUILT_IN(REGRESSION_ORIGIN_TAG, m_RegressionOrigin)
        RESTORE(MODEL_TAG, traverser.traverseSubLevel([&](auto& traverser_) {
            return m_TrendModels[i++].acceptRestoreTraverser(traverser_);
        }))
        RESTORE_BUILT_IN(PREDICTION_ERROR_VARIANCE_TAG, m_PredictionErrorVariance)
        RESTORE(VALUE_MOMENTS_TAG, m_ValueMoments.fromDelimited(traverser.value()))
        RESTORE_BUILT_IN(TIME_OF_LAST_LEVEL_CHANGE_TAG, m_TimeOfLastLevelChange)
        RESTORE_NO_ERROR(PROBABILITY_OF_LEVEL_CHANGE_MODEL_TAG,
                         m_ProbabilityOfLevelChangeModel = common::CNaiveBayes(
                             naiveBayesExemplar(m_DefaultDecayRate), params, traverser))
        RESTORE_NO_ERROR(MAGNITUDE_OF_LEVEL_CHANGE_MODEL_TAG,
                         m_MagnitudeOfLevelChangeModel =
                             common::CNormalMeanPrecConjugate(params, traverser))
    } while (traverser.next());

    this->checkRestoredInvariants();

    return true;
}

void CTrendComponent::checkRestoredInvariants() const {
    VIOLATES_INVARIANT(m_FirstUpdate, >, m_LastUpdate);
}

bool CTrendComponent::initialized() const {
    return m_LastUpdate != UNSET_TIME;
}

void CTrendComponent::clear() {
    m_FirstUpdate = UNSET_TIME;
    m_LastUpdate = UNSET_TIME;
    m_RegressionOrigin = UNSET_TIME;
    for (std::size_t i = 0; i < NUMBER_MODELS; ++i) {
        m_TrendModels[i] = SModel(modelWeight(1.0, TIME_SCALES[i]));
    }
    m_PredictionErrorVariance = 0.0;
    m_ValueMoments = TMeanVarAccumulator();
    m_TimeOfLastLevelChange = UNSET_TIME;
    m_ProbabilityOfLevelChangeModel = initialProbabilityOfChangeModel(m_DefaultDecayRate);
    m_MagnitudeOfLevelChangeModel = initialMagnitudeOfChangeModel(m_DefaultDecayRate);
}

void CTrendComponent::shiftOrigin(core_t::TTime time) {
    time = common::CIntegerTools::floor(time, core::constants::WEEK);
    double scaledShift{scaleTime(time, m_RegressionOrigin)};
    if (scaledShift > 0.0) {
        for (auto& model : m_TrendModels) {
            model.s_Regression.shiftAbscissa(-scaledShift);
        }
        m_RegressionOrigin = time;
    }
}

void CTrendComponent::shiftSlope(core_t::TTime time, double shift) {
    // This shifts all models gradients by a fixed amount whilst keeping
    // the prediction at time fixed. This avoids a jump in the current
    // predicted value as a result of adjusting the gradient. Otherwise,
    // this changes by "shift" x "scaled time since the regression origin".
    if (std::fabs(shift) > 0.0) {
        for (std::size_t i = 0; i < NUMBER_MODELS; ++i) {
            double modelDecayRate{m_DefaultDecayRate * TIME_SCALES[i]};
            double shift_{std::min(modelDecayRate / m_TargetDecayRate, 1.0) * shift};
            m_TrendModels[i].s_Regression.shiftGradient(shift_);
            m_TrendModels[i].s_Regression.shiftOrdinate(
                -shift_ * scaleTime(time, m_RegressionOrigin));
        }
    }
}

void CTrendComponent::shiftLevel(double shift) {
    for (auto& model : m_TrendModels) {
        model.s_Regression.shiftOrdinate(shift);
    }
}

void CTrendComponent::shiftLevel(double shift,
                                 core_t::TTime valuesStartTime,
                                 core_t::TTime bucketLength,
                                 const TFloatMeanAccumulatorVec& values,
                                 const TSizeVec& segments,
                                 const TDoubleVec& shifts) {
    this->shiftLevel(shift);

    if (segments.size() <= 2) {
        m_MagnitudeOfLevelChangeModel.addSamples({shift}, maths_t::CUnitWeights::SINGLE_UNIT);
        return;
    }

    auto indexTime = [&](std::size_t i) {
        return valuesStartTime + bucketLength * static_cast<core_t::TTime>(i);
    };

    // If the last change is in the values' time window the next change
    // is the first one after the closest. Otherwise, it is the first
    // change.
    auto next = std::min(
        static_cast<std::size_t>(
            m_TimeOfLastLevelChange < valuesStartTime
                ? 1
                : std::min_element(segments.begin() + 1, segments.end() - 1,
                                   [&](auto lhs, auto rhs) {
                                       return std::abs(indexTime(lhs) - m_TimeOfLastLevelChange) <
                                              std::abs(indexTime(rhs) - m_TimeOfLastLevelChange);
                                   }) -
                      segments.begin() + 1),
        segments.size() - 2);

    std::size_t last{shifts.size() - 1};
    core_t::TTime time{valuesStartTime + static_cast<core_t::TTime>(segments[last]) * bucketLength};
    double magnitude{shifts[last] - shifts[next - 1]};
    if (m_TimeOfLastLevelChange != UNSET_TIME) {
        double dt{static_cast<double>(time - m_TimeOfLastLevelChange)};
        if (values.size() > segments[next] - 1) {
            double value{static_cast<double>(
                common::CBasicStatistics::mean(values[segments[next] - 1]))};
            m_ProbabilityOfLevelChangeModel.addTrainingDataPoint(LEVEL_CHANGE_LABEL,
                                                                 {{dt}, {value}});
        } else {
            LOG_DEBUG(<< "Size mis-match reading from values. Length = "
                      << values.size() << ", requested index = " << segments[next] - 1);
        }
    }
    m_TimeOfLastLevelChange = time;
    for (std::size_t i = segments[last]; i < values.size(); ++i, time += bucketLength) {
        this->dontShiftLevel(time, common::CBasicStatistics::mean(values[i]));
    }
    m_MagnitudeOfLevelChangeModel.addSamples({magnitude}, maths_t::CUnitWeights::SINGLE_UNIT);
}

void CTrendComponent::dontShiftLevel(core_t::TTime time, double value) {
    if (m_TimeOfLastLevelChange != UNSET_TIME) {
        double dt{static_cast<double>(time - m_TimeOfLastLevelChange)};
        m_ProbabilityOfLevelChangeModel.addTrainingDataPoint(NO_CHANGE_LABEL,
                                                             {{dt}, {value}});
    }
}

void CTrendComponent::linearScale(core_t::TTime time, double scale) {
    double shift{(scale - 1.0) * this->value(time, 0.0).mean()};
    this->shiftLevel(shift);
}

void CTrendComponent::add(core_t::TTime time, double value, double weight) {
    // Update the model weights: we weight the components based on the
    // relative difference in the component scale and the target scale.

    for (std::size_t i = 0; i < NUMBER_MODELS; ++i) {
        m_TrendModels[i].s_Weight.add(
            modelWeight(m_TargetDecayRate, m_DefaultDecayRate * TIME_SCALES[i]));
    }

    if (m_FirstUpdate == UNSET_TIME) {
        m_RegressionOrigin = common::CIntegerTools::floor(time, core::constants::WEEK);
    }

    // Update the models.

    double prediction{this->value(time, 0.0).mean()};

    // Track residuals for autocorrelation calculation
    if (this->initialized()) {
        double residual = value - prediction;
        m_RecentResiduals.push_back(residual);
        
        // Keep only the most recent residuals (e.g., last 30 points)
        const std::size_t MAX_RESIDUALS = 30;
        if (m_RecentResiduals.size() > MAX_RESIDUALS) {
            m_RecentResiduals.erase(m_RecentResiduals.begin());
        }
        
        // LOG_DEBUG(<< "Residual Debug: value=" << value << ", prediction=" << prediction 
        //           << ", residual=" << residual << ", residuals_count=" << m_RecentResiduals.size());
    }

    double count{this->count()};
    if (count > 0.0) {
        TMeanVarAccumulator moments{common::CBasicStatistics::momentsAccumulator(
            count, prediction, m_PredictionErrorVariance)};
        moments.add(value, weight);
        m_PredictionErrorVariance =
            common::CBasicStatistics::maximumLikelihoodVariance(moments);
    }

    double scaledTime{scaleTime(time, m_RegressionOrigin)};
    for (auto& model : m_TrendModels) {
        TVector3x1 mse;
        for (std::size_t order = 1; order <= TRegression::N; ++order) {
            mse(order - 1) = value - model.s_Regression.predict(order, scaledTime, MAX_CONDITION);
        }
        model.s_Mse.add(mse * mse, weight);
        model.s_Regression.add(scaledTime, value, weight);
    }
    m_ValueMoments.add(value, weight);

    m_FirstUpdate = m_FirstUpdate == UNSET_TIME ? time : std::min(m_FirstUpdate, time);
    m_LastUpdate = std::max(m_LastUpdate, time);
}

void CTrendComponent::dataType(maths_t::EDataType dataType) {
    m_ProbabilityOfLevelChangeModel.dataType(dataType);
    m_MagnitudeOfLevelChangeModel.dataType(dataType);
}

double CTrendComponent::defaultDecayRate() const {
    return m_DefaultDecayRate;
}

void CTrendComponent::decayRate(double decayRate) {
    m_TargetDecayRate = decayRate;
}

void CTrendComponent::propagateForwardsByTime(core_t::TTime interval) {
    TDoubleVec factors;
    this->smoothingFactors(interval, factors);
    double median{common::CBasicStatistics::median(factors)};
    for (std::size_t i = 0; i < NUMBER_MODELS; ++i) {
        m_TrendModels[i].s_Weight.age(median);
        m_TrendModels[i].s_Regression.age(factors[i]);
        m_TrendModels[i].s_Mse.age(std::sqrt(factors[i]));
    }
    double interval_{static_cast<double>(interval) /
                     static_cast<double>(core::constants::DAY)};
    m_ProbabilityOfLevelChangeModel.propagateForwardsByTime(interval_);
    m_MagnitudeOfLevelChangeModel.propagateForwardsByTime(interval_);
}

CTrendComponent::TVector2x1 CTrendComponent::value(core_t::TTime time, double confidence) const {
    if (this->initialized() == false) {
        return TVector2x1{0.0};
    }

    double a{this->weightOfPrediction(time)};
    double b{1.0 - a};
    double scaledTime{scaleTime(time, m_RegressionOrigin)};

    TMeanAccumulator prediction_;

    TDoubleVec weights;
    this->smoothingFactors(std::abs(time - m_LastUpdate), weights);
    double Z{0.0};
    for (std::size_t i = 0; i < NUMBER_MODELS; ++i) {
        weights[i] *= common::CBasicStatistics::mean(m_TrendModels[i].s_Weight);
        Z += weights[i];
    }
    for (std::size_t i = 0; i < NUMBER_MODELS; ++i) {
        if (weights[i] > MINIMUM_WEIGHT_TO_USE_MODEL_FOR_PREDICTION * Z) {
            prediction_.add(m_TrendModels[i].s_Regression.predict(scaledTime, MAX_CONDITION),
                            weights[i]);
        }
    }

    double prediction{a * common::CBasicStatistics::mean(prediction_) +
                      b * common::CBasicStatistics::mean(m_ValueMoments)};

    if (confidence > 0.0 && m_PredictionErrorVariance > 0.0) {
        // Use effective sample size to account for autocorrelation
        double n_eff = calculateEffectiveSampleSize();
        // Don't scale variance by n_eff - use n_eff only for degrees of freedom
        double variance{a * m_PredictionErrorVariance +
                        b * common::CBasicStatistics::variance(m_ValueMoments) /
                            std::max(common::CBasicStatistics::count(m_ValueMoments), 1.0)};
        // Add minimum variance floor to ensure confidence intervals are wide enough for pure noise data
        // double dataVariance = common::CBasicStatistics::variance(m_ValueMoments);
        // double minVariance = std::max(variance, dataVariance * 0.5); // Ensure at least 50% of data variance
        // variance = minVariance;
        
        // Use t-distribution when we have autocorrelation (n_eff < n) to get wider confidence intervals
        double n_raw = this->count();
        
        if (n_eff < n_raw) {
            return confidenceIntervalT(prediction, variance, confidence, n_eff);
        } else {
            return confidenceInterval(prediction, variance, confidence);
        }
    }

    return TVector2x1{prediction};
}

CTrendComponent::TPredictor CTrendComponent::predictor() const {

    if (this->initialized() == false) {
        return [](core_t::TTime) { return 0.0; };
    }

    TDoubleVec weights;
    TRegressionArrayVec parameters(NUMBER_MODELS);
    for (std::size_t i = 0; i < NUMBER_MODELS; ++i) {
        m_TrendModels[i].s_Regression.parameters(parameters[i], MAX_CONDITION);
    }

    return [weights, parameters, this](core_t::TTime time) mutable {
        TMeanAccumulator prediction;

        double a{this->weightOfPrediction(time)};
        double b{1.0 - a};
        double scaledTime{scaleTime(time, m_RegressionOrigin)};

        this->smoothingFactors(std::abs(time - m_LastUpdate), weights);

        double Z{0.0};
        for (std::size_t i = 0; i < NUMBER_MODELS; ++i) {
            weights[i] *= common::CBasicStatistics::mean(m_TrendModels[i].s_Weight);
            Z += weights[i];
        }
        for (std::size_t i = 0; i < NUMBER_MODELS; ++i) {
            if (weights[i] > MINIMUM_WEIGHT_TO_USE_MODEL_FOR_PREDICTION * Z) {
                prediction.add(m_TrendModels[i].s_Regression.predict(parameters[i], scaledTime),
                               weights[i]);
            }
        }

        return a * common::CBasicStatistics::mean(prediction) +
               b * common::CBasicStatistics::mean(m_ValueMoments);
    };
}

core_t::TTime CTrendComponent::maximumForecastInterval() const {
    double timescale{static_cast<double>(core::constants::DAY)};
    double interval{std::min(
        1.0 / (1.0 - std::exp(-TIME_SCALES[NUMBER_MODELS - 1] * m_DefaultDecayRate)),
        std::floor(static_cast<double>(std::numeric_limits<core_t::TTime>::max()) / timescale))};
    return static_cast<core_t::TTime>(interval * timescale);
}

CTrendComponent::TVector2x1 CTrendComponent::variance(double confidence) const {

    if (this->initialized() == false) {
        return TVector2x1{0.0};
    }

    double variance{m_PredictionErrorVariance};

    if (confidence > 0.0 && variance > 0.0) {
        // Use effective sample size to account for autocorrelation
        double n_eff = calculateEffectiveSampleSize();
        double df{std::max(n_eff, 2.0) - 1.0}; // Degrees of freedom based on effective sample size
        
        try {
            boost::math::chi_squared chi{df};
            double ql{boost::math::quantile(chi, (100.0 - confidence) / 200.0)};
            double qu{boost::math::quantile(chi, (100.0 + confidence) / 200.0)};
            
            double lower_bound = ql * variance / df;
            double upper_bound = qu * variance / df;
            return TVector2x1{{lower_bound, upper_bound}};
        } catch (const std::exception& e) {
            LOG_ERROR(<< "Failed calculating confidence interval: " << e.what()
                      << ", df = " << df << ", confidence = " << confidence);
        }
    }

    return TVector2x1{variance};
}

void CTrendComponent::forecast(core_t::TTime startTime,
                               core_t::TTime endTime,
                               core_t::TTime step,
                               double confidence,
                               bool isNonNegative,
                               const TSeasonalForecast& seasonal,
                               const TWriteForecastResult& writer) const {
    if (endTime < startTime) {
        LOG_ERROR(<< "Bad forecast range: [" << startTime << "," << endTime << "]");
        return;
    }
    if (confidence < 0.0 || confidence >= 100.0) {
        LOG_ERROR(<< "Bad confidence interval: " << confidence << "%");
        return;
    }

    endTime = startTime + common::CIntegerTools::ceil(endTime - startTime, step);

    LOG_TRACE(<< "forecasting = " << this->print());

    TSizeVec selectedModelOrders(this->selectModelOrdersForForecasting());
    LOG_TRACE(<< "Selected model orders = " << selectedModelOrders);

    TDoubleVec factors;
    this->smoothingFactors(step, factors);
    TDoubleVec modelWeights(this->initialForecastModelWeights(NUMBER_MODELS));
    TDoubleVec errorWeights(this->initialForecastModelWeights(NUMBER_MODELS + 1));
    TRegressionArrayVec models(NUMBER_MODELS);
    TMatrix3x3Vec modelCovariances(NUMBER_MODELS);
    TDoubleVec mse(NUMBER_MODELS);
    for (std::size_t i = 0; i < NUMBER_MODELS; ++i) {
        // Note in the following we multiply the bias by the sample count
        // when estimating the regression coefficients' covariance matrix.
        // This is because, unlike the noise like component of the residual,
        // it is overly optimistic to assume these will cancel in the
        // estimates of the regression coefficients when adding multiple
        // samples. Instead, we make the most pessimistic assumption that
        // this term is from a fixed random variable and so multiply its
        // variance by the number of samples to undo the scaling applied
        // in the regression model covariance method.
        const SModel& model{m_TrendModels[i]};
        std::size_t order{selectedModelOrders[i]};
        double n{model.s_Regression.count()};
        mse[i] = common::CBasicStatistics::mean(model.s_Mse)(order - 1);
        model.s_Regression.parameters(order, models[i], MAX_CONDITION);
        model.s_Regression.covariances(order, n * mse[i], modelCovariances[i], MAX_CONDITION);
        LOG_TRACE(<< "params      = " << models[i]);
        LOG_TRACE(<< "covariances = " << modelCovariances[i].toDelimited());
        LOG_TRACE(<< "mse         = " << mse[i]);
    }
    LOG_TRACE(<< "long time variance = "
              << common::CBasicStatistics::variance(m_ValueMoments));

    CForecastLevel level{m_ProbabilityOfLevelChangeModel,
                         m_MagnitudeOfLevelChangeModel, m_TimeOfLastLevelChange};

    TDoubleVec variances(NUMBER_MODELS + 1);
    for (core_t::TTime time = startTime; time < endTime; time += step) {
        double scaledDt{scaleTime(time, startTime)};
        TVector3x1 times({0.0, scaledDt, scaledDt * scaledDt});

        double a{this->weightOfPrediction(time)};
        double b{1.0 - a};

        for (std::size_t j = 0; j < NUMBER_MODELS; ++j) {
            modelWeights[j] *= factors[j];
            errorWeights[j] *= common::CTools::pow2(factors[j]);
        }

        for (std::size_t j = 0; j < NUMBER_MODELS; ++j) {
            variances[j] = times.inner(modelCovariances[j] * times) + mse[j];
        }
        variances[NUMBER_MODELS] = common::CBasicStatistics::variance(m_ValueMoments);
        for (auto v = variances.rbegin(); v != variances.rend(); ++v) {
            *v = *std::min_element(variances.rbegin(), v + 1);
        }

        TMeanAccumulator variance_;
        for (std::size_t j = 0; j < NUMBER_MODELS; ++j) {
            variance_.add(variances[j], errorWeights[j]);
        }
        double variance{a * common::CBasicStatistics::mean(variance_) +
                        b * common::CBasicStatistics::variance(m_ValueMoments)};

        TVector2x1 trend{confidenceInterval(
            this->value(modelWeights, models, scaleTime(time, m_RegressionOrigin)),
            variance, confidence)};
        TDouble3Vec seasonal_(seasonal(time));
        TDouble3Vec level_(level.forecast(time, seasonal_[1] + trend.mean(), confidence));

        TDouble3Vec forecast{level_[0] + trend(0) + seasonal_[0],
                             level_[1] + trend.mean() + seasonal_[1],
                             level_[2] + trend(1) + seasonal_[2]};
        forecast[0] = isNonNegative ? std::max(forecast[0], 0.0) : forecast[0];
        forecast[1] = isNonNegative ? std::max(forecast[1], 0.0) : forecast[1];
        forecast[2] = isNonNegative ? std::max(forecast[2], 0.0) : forecast[2];

        writer(time, forecast);
    }
}

core_t::TTime CTrendComponent::observedInterval() const {
    return m_LastUpdate - m_FirstUpdate;
}

double CTrendComponent::parameters() const {
    return static_cast<double>(TRegression::N);
}

std::uint64_t CTrendComponent::checksum(std::uint64_t seed) const {
    seed = common::CChecksum::calculate(seed, m_TargetDecayRate);
    seed = common::CChecksum::calculate(seed, m_FirstUpdate);
    seed = common::CChecksum::calculate(seed, m_LastUpdate);
    seed = common::CChecksum::calculate(seed, m_TrendModels);
    seed = common::CChecksum::calculate(seed, m_PredictionErrorVariance);
    seed = common::CChecksum::calculate(seed, m_ValueMoments);
    seed = common::CChecksum::calculate(seed, m_TimeOfLastLevelChange);
    seed = common::CChecksum::calculate(seed, m_ProbabilityOfLevelChangeModel);
    return common::CChecksum::calculate(seed, m_MagnitudeOfLevelChangeModel);
}

std::string CTrendComponent::print() const {
    std::ostringstream result;
    result << "\n===\n";
    result << "Trend Models:";
    for (const auto& model : m_TrendModels) {
        result << "\n" << model.s_Regression.print();
    }
    result << "\n===\n";
    result << "Probability of Change Model:";
    result << m_ProbabilityOfLevelChangeModel.print();
    result << "===\n";
    result << "Magnitude of Change Model:";
    result << m_MagnitudeOfLevelChangeModel.print();
    return result.str();
}

void CTrendComponent::smoothingFactors(core_t::TTime interval, TDoubleVec& result) const {
    result.assign(NUMBER_MODELS, 0.0);
    double factor{m_DefaultDecayRate * static_cast<double>(interval) /
                  static_cast<double>(core::constants::DAY)};
    for (std::size_t i = 0; i < NUMBER_MODELS; ++i) {
        result[i] = std::exp(-TIME_SCALES[i] * factor);
    }
}

CTrendComponent::TSizeVec CTrendComponent::selectModelOrdersForForecasting() const {
    // Use BIC (Bayesian Information Criterion) to select model order with F-test fallback
    // BIC = n * log(MSE) + penalty * k * log(n)
    // Lower BIC is better
    
    TSizeVec result(NUMBER_MODELS, 1);
    
    // Penalty multipliers for different orders
    const double penalties[]{BIC_PENALTY_ORDER_1, BIC_PENALTY_ORDER_2, BIC_PENALTY_ORDER_3};
    
    for (std::size_t i = 0; i < NUMBER_MODELS; ++i) {
        const SModel& model{m_TrendModels[i]};
        double n{common::CBasicStatistics::count(model.s_Mse)};
        
        if (n < 2.0) {
            continue;
        }
        
        // Use effective sample size to account for autocorrelation in time series
        double n_eff{calculateEffectiveSampleSize()};
        double logN_eff{std::log(n_eff)};
        double minBIC{std::numeric_limits<double>::max()};
        std::size_t bestOrder{1};
        
        // Let BIC criteria handle model selection without explicit noise detection
        
        // First pass: Use BIC to select model order with effective sample size
        LOG_TRACE(<< "BIC Model Selection Debug: n=" << n << ", n_eff=" << n_eff << ", logN_eff=" << logN_eff);
        for (std::size_t order = 1; order <= TRegression::N; ++order) {
            double mse{common::CBasicStatistics::mean(model.s_Mse)(order - 1)};
            
            // Skip if MSE is invalid or we don't have enough data
            if (mse <= 0.0 || n_eff < static_cast<double>(order + 1)) {
                LOG_TRACE(<< "BIC Debug: Skipping order " << order << " (mse=" << mse << ", n_eff=" << n_eff << ")");
                break;
            }
            
            
            // BIC = n_eff * log(MSE) + penalty * k * log(n_eff)
            // k = order + 1 (number of parameters for polynomial of given order)
            // Using n_eff accounts for autocorrelation in time series data
            double k{static_cast<double>(order + 1)};
            double penalty{penalties[order - 1]};
            double bic{(n_eff * std::log(mse)) + (penalty * k * logN_eff)};
            double logMSE = std::log(mse);
            double penaltyTerm = penalty * k * logN_eff;
            
            LOG_TRACE(<< "BIC Debug: order=" << order << ", mse=" << mse << ", logMSE=" << logMSE 
                      << ", k=" << k << ", penalty=" << penalty << ", penaltyTerm=" << penaltyTerm 
                      << ", bic=" << bic << ", minBIC=" << minBIC);
            
            if (bic < minBIC) {
                minBIC = bic;
                bestOrder = order;
                LOG_TRACE(<< "BIC Debug: New best order=" << order << " with BIC=" << bic);
            }
        }
        
        // Second pass: Use F-test as fallback to verify the BIC selection
        // Only upgrade to higher order if F-test also supports it
        LOG_TRACE(<< "BIC Final Selection: bestOrder=" << bestOrder << " before F-test verification");
        if (bestOrder > 1) {
            double mse0{common::CBasicStatistics::mean(model.s_Mse)(0)};
            double df0{n - 1.0};
            LOG_TRACE(<< "F-test Debug: Starting F-test verification, mse0=" << mse0 << ", df0=" << df0);
            
            for (std::size_t order = 2; order <= bestOrder; ++order) {
                double mse1{common::CBasicStatistics::mean(model.s_Mse)(order - 1)};
                double df1{n - static_cast<double>(order)};
                
                if (df1 < 0.0) {
                    LOG_TRACE(<< "F-test Debug: df1 < 0, reducing bestOrder from " << bestOrder << " to " << (order - 1));
                    bestOrder = order - 1;
                    break;
                }
                
                double p{common::CStatisticalTests::leftTailFTest(mse1, mse0, df1, df0)};
                LOG_TRACE(<< "F-test Debug: order=" << order << ", mse1=" << mse1 << ", df1=" << df1 
                          << ", p=" << p << ", threshold=" << MODEL_MSE_DECREASE_SIGNFICANT);
                
                if (p >= MODEL_MSE_DECREASE_SIGNFICANT) {
                    // F-test doesn't support this order, use previous order
                    LOG_TRACE(<< "F-test Debug: F-test rejects order " << order << ", reducing bestOrder to " << (order - 1));
                    bestOrder = order - 1;
                    break;
                }
                
                mse0 = mse1;
                df0 = df1;
            }
        }
        LOG_DEBUG(<< "Final Model Selection: bestOrder=" << bestOrder);
        
        result[i] = bestOrder;
    }

    return result;
}

CTrendComponent::TDoubleVec CTrendComponent::initialForecastModelWeights(std::size_t n) const {
    TDoubleVec result(n);
    for (std::size_t i = 0; i < n; ++i) {
        result[i] = std::exp(static_cast<double>(NUMBER_MODELS / 2) -
                             static_cast<double>(i));
    }
    return result;
}

double CTrendComponent::count() const {
    TMeanAccumulator result;
    for (const auto& model : m_TrendModels) {
        result.add(common::CTools::fastLog(model.s_Regression.count()),
                   common::CBasicStatistics::mean(model.s_Weight));
    }
    return std::exp(common::CBasicStatistics::mean(result));
}

double CTrendComponent::calculateEffectiveSampleSize() const {
    if (!this->initialized()) {
        return 0.0;
    }
    
    try {
        double n = this->count();
        
        // If we don't have enough data for autocorrelation analysis, return the raw count
        if (n < 10 || m_RecentResiduals.size() < 10) {
            return n;
        }
        
        // Calculate rho_1 using the standard estimator
        // rho_1 = sum((x_t - mean)(x_{t-1} - mean)) / sum((x_t - mean)^2)
        double mean = 0.0;
        for (double residual : m_RecentResiduals) {
            mean += residual;
        }
        mean /= m_RecentResiduals.size();
        
        double numerator = 0.0;
        double denominator = 0.0;
        
        for (std::size_t i = 1; i < m_RecentResiduals.size(); ++i) {
            double diff_i = m_RecentResiduals[i] - mean;
            double diff_prev = m_RecentResiduals[i-1] - mean;
            numerator += diff_i * diff_prev;
            denominator += diff_i * diff_i;
        }
        
        // Add the last term for denominator
        if (m_RecentResiduals.size() > 0) {
            double diff_0 = m_RecentResiduals[0] - mean;
            denominator += diff_0 * diff_0;
        }
        
        double rho_1 = (denominator > 0.0) ? std::abs(numerator / denominator) : 0.0;
        
        double n_eff = n * (1.0 - rho_1) / (1.0 + rho_1);
        // Ensure n_eff is reasonable
        return std::max(1.0, std::min(n, n_eff));
        
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to calculate effective sample size using autocorrelations: " << e.what());
        // Fall back to a conservative estimate
        double n = this->count();
        return std::max(2.0, n * 0.5);
    }
}

double CTrendComponent::value(const TDoubleVec& weights,
                              const TRegressionArrayVec& models,
                              double time) const {
    TMeanAccumulator prediction;
    for (std::size_t i = 0; i < models.size(); ++i) {
        prediction.add(TRegression::predict(models[i], time), weights[i]);
    }
    return common::CBasicStatistics::mean(prediction);
}

double CTrendComponent::weightOfPrediction(core_t::TTime time) const {
    double interval{static_cast<double>(m_LastUpdate - m_FirstUpdate)};
    if (interval == 0.0) {
        return 0.0;
    }

    double extrapolateInterval{static_cast<double>(common::CBasicStatistics::max(
        time - m_LastUpdate, m_FirstUpdate - time, core_t::TTime(0)))};
    if (extrapolateInterval == 0.0) {
        return 1.0;
    }

    return common::CTools::logisticFunction(extrapolateInterval / interval, 0.1, 1.0, -1.0);
}

CTrendComponent::SModel::SModel(double weight) {
    s_Weight.add(weight, 0.01);
}

void CTrendComponent::SModel::acceptPersistInserter(core::CStatePersistInserter& inserter) const {

    inserter.insertValue(VERSION_7_1_TAG, "");
    inserter.insertValue(WEIGHT_7_1_TAG, s_Weight.toDelimited());
    inserter.insertLevel(REGRESSION_7_1_TAG, [this](auto& inserter_) {
        s_Regression.acceptPersistInserter(inserter_);
    });
    inserter.insertValue(MSE_7_1_TAG, s_Mse.toDelimited());
}

bool CTrendComponent::SModel::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    if (traverser.name() == VERSION_7_1_TAG) {
        while (traverser.next()) {
            const std::string& name{traverser.name()};
            RESTORE(WEIGHT_7_1_TAG, s_Weight.fromDelimited(traverser.value()))
            RESTORE(REGRESSION_7_1_TAG, traverser.traverseSubLevel([this](auto& traverser_) {
                return s_Regression.acceptRestoreTraverser(traverser_);
            }))
            RESTORE(MSE_7_1_TAG, s_Mse.fromDelimited(traverser.value()))
        }
    } else {
        LOG_ERROR(<< "Input error: unsupported state serialization version '"
                  << traverser.name()
                  << "'. Currently supported minimum version: " << VERSION_7_1_TAG);
        return false;
    }
    return true;
}

std::uint64_t CTrendComponent::SModel::checksum(std::uint64_t seed) const {
    seed = common::CChecksum::calculate(seed, s_Weight);
    seed = common::CChecksum::calculate(seed, s_Regression);
    return common::CChecksum::calculate(seed, s_Mse);
}

CTrendComponent::CForecastLevel::CForecastLevel(const common::CNaiveBayes& probability,
                                                const common::CNormalMeanPrecConjugate& magnitude,
                                                core_t::TTime timeOfLastChange,
                                                std::size_t numberPaths)
    : m_Probability{probability}, m_Magnitude{magnitude},
      m_Levels(numberPaths, 0.0), m_TimesOfLastChange(numberPaths, timeOfLastChange),
      m_ProbabilitiesOfChange(numberPaths, 0.0) {
    m_Uniform01.reserve(numberPaths);
}

CTrendComponent::TDouble3Vec
CTrendComponent::CForecastLevel::forecast(core_t::TTime time, double prediction, double confidence) {
    TDouble3Vec result{0.0, 0.0, 0.0};

    if (m_Probability.initialized() && m_Probability.numberClasses() > 1) {
        common::CSampling::uniformSample(0.0, 1.0, m_Levels.size(), m_Uniform01);
        bool reorder{false};
        auto weightProvider = [weight =
                                   CChangeForecastFeatureWeight{}]() mutable->common::CNaiveBayesFeatureWeight& {
            weight = CChangeForecastFeatureWeight{};
            return weight;
        };
        for (std::size_t i = 0; i < m_Levels.size(); ++i) {
            double dt{static_cast<double>(time - m_TimesOfLastChange[i])};
            double x{m_Levels[i] + prediction};
            auto[p, pConfidence] = m_Probability.classProbability(
                LEVEL_CHANGE_LABEL, {{dt}, {x}}, weightProvider);
            // Here we decide whether to increase the probability we should have
            // seen a step change for this rollout. If we are no longer confident
            // in our predicted probability we do not predict changes based on
            // the principle of least surprise.
            if (pConfidence > 0.5) {
                m_ProbabilitiesOfChange[i] = std::max(m_ProbabilitiesOfChange[i], p);
            }
            if (m_Uniform01[i] < m_ProbabilitiesOfChange[i]) {
                double stepMean{m_Magnitude.marginalLikelihoodMean()};
                double stepVariance{m_Magnitude.marginalLikelihoodVariance()};
                m_Levels[i] += common::CSampling::normalSample(m_Rng, stepMean, stepVariance);
                m_TimesOfLastChange[i] = time;
                m_ProbabilitiesOfChange[i] = 0.0;
                reorder = true;
            }
        }
        if (reorder) {
            common::COrderings::simultaneousSort(m_Levels, m_TimesOfLastChange,
                                                 m_ProbabilitiesOfChange);
        }

        double rollouts{static_cast<double>(m_Levels.size())};
        std::size_t lower{std::min(
            static_cast<std::size_t>((100.0 - confidence) / 200.0 * rollouts + 0.5),
            m_Levels.size())};
        std::size_t upper{std::min(
            static_cast<std::size_t>((100.0 + confidence) / 200.0 * rollouts + 0.5),
            m_Levels.size() - 1)};

        result[0] = m_Levels[lower];
        result[1] = common::CBasicStatistics::median(m_Levels);
        result[2] = m_Levels[upper];
    }

    return result;
}
}
}
}
