/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CTrendComponent.h>

#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/Constants.h>
#include <core/RestoreMacros.h>

#include <maths/CChecksum.h>
#include <maths/CIntegerTools.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CRegressionDetail.h>
#include <maths/CSampling.h>
#include <maths/CTools.h>

#include <boost/bind.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/range.hpp>

#include <cmath>
#include <exception>
#include <numeric>

namespace ml {
namespace maths {
namespace {
using TOptionalDoubleDoublePr = boost::optional<std::pair<double, double>>;

const double TIME_SCALES[]{144.0, 72.0, 36.0, 12.0, 4.0, 1.0, 0.25, 0.05};
const std::size_t NUMBER_MODELS{boost::size(TIME_SCALES)};
const double MINIMUM_WEIGHT_TO_USE_MODEL_FOR_PREDICTION{0.01};
const double MAX_CONDITION{1e12};
const core_t::TTime UNSET_TIME{0};
const std::size_t NO_CHANGE_LABEL{0};
const std::size_t LEVEL_CHANGE_LABEL{1};

//! Get the desired weight for the regression model.
double modelWeight(double targetDecayRate, double modelDecayRate) {
    return targetDecayRate == modelDecayRate
               ? 1.0
               : std::min(targetDecayRate, modelDecayRate) /
                     std::max(targetDecayRate, modelDecayRate);
}

//! We scale the time used for the regression model to improve
//! the condition of the design matrix.
double scaleTime(core_t::TTime time, core_t::TTime origin) {
    return static_cast<double>(time - origin) / static_cast<double>(core::constants::WEEK);
}

//! Get the \p confidence interval for \p prediction and \p variance.
TOptionalDoubleDoublePr confidenceInterval(double prediction, double variance, double confidence) {
    if (variance == 0.0) {
        return std::make_pair(prediction, prediction);
    }
    try {
        boost::math::normal normal{prediction, std::sqrt(variance)};
        double ql{boost::math::quantile(normal, (100.0 - confidence) / 200.0)};
        double qu{boost::math::quantile(normal, (100.0 + confidence) / 200.0)};
        return std::make_pair(ql, qu);
    } catch (const std::exception& e) {
        LOG_ERROR("Failed calculating confidence interval: "
                  << e.what() << ", prediction = " << prediction
                  << ", variance = " << variance << ", confidence = " << confidence);
    }
    return TOptionalDoubleDoublePr{};
}

CNaiveBayesFeatureDensityFromPrior naiveBayesExemplar(double decayRate) {
    return CNaiveBayesFeatureDensityFromPrior{CNormalMeanPrecConjugate::nonInformativePrior(
        maths_t::E_ContinuousData, TIME_SCALES[NUMBER_MODELS - 1] * decayRate)};
}

CNaiveBayes initialProbabilityOfChangeModel(double decayRate) {
    return CNaiveBayes{naiveBayesExemplar(decayRate),
                       TIME_SCALES[NUMBER_MODELS - 1] * decayRate, -20.0};
}

CNormalMeanPrecConjugate initialMagnitudeOfChangeModel(double decayRate) {
    return CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData, decayRate);
}

const std::string TARGET_DECAY_RATE_TAG{"a"};
const std::string FIRST_UPDATE_TAG{"b"};
const std::string LAST_UPDATE_TAG{"c"};
const std::string REGRESSION_ORIGIN_TAG{"d"};
const std::string MODEL_TAG{"e"};
const std::string PREDICTION_ERROR_VARIANCE_TAG{"f"};
const std::string VALUE_MOMENTS_TAG{"g"};
const std::string TIME_OF_LAST_LEVEL_CHANGE_TAG{"h"};
const std::string PROBABILITY_OF_LEVEL_CHANGE_MODEL_TAG{"i"};
const std::string MAGNITUDE_OF_LEVEL_CHANGE_MODEL_TAG{"j"};
const std::string WEIGHT_TAG{"a"};
const std::string REGRESSION_TAG{"b"};
const std::string RESIDUAL_MOMENTS_TAG{"c"};
}

CTrendComponent::CTrendComponent(double decayRate)
    : m_DefaultDecayRate(decayRate), m_TargetDecayRate(decayRate),
      m_FirstUpdate(UNSET_TIME), m_LastUpdate(UNSET_TIME),
      m_RegressionOrigin(UNSET_TIME), m_PredictionErrorVariance(0.0),
      m_TimeOfLastLevelChange(UNSET_TIME),
      m_ProbabilityOfLevelChangeModel(initialProbabilityOfChangeModel(decayRate)),
      m_MagnitudeOfLevelChangeModel(initialMagnitudeOfChangeModel(decayRate)) {
    for (std::size_t i = 0u; i < NUMBER_MODELS; ++i) {
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
        inserter.insertLevel(
            MODEL_TAG, boost::bind(&SModel::acceptPersistInserter, &model, _1));
    }
    inserter.insertValue(PREDICTION_ERROR_VARIANCE_TAG, m_PredictionErrorVariance,
                         core::CIEEE754::E_DoublePrecision);
    inserter.insertValue(VALUE_MOMENTS_TAG, m_ValueMoments.toDelimited());
    inserter.insertValue(TIME_OF_LAST_LEVEL_CHANGE_TAG, m_TimeOfLastLevelChange);
    inserter.insertLevel(PROBABILITY_OF_LEVEL_CHANGE_MODEL_TAG,
                         boost::bind(&CNaiveBayes::acceptPersistInserter,
                                     &m_ProbabilityOfLevelChangeModel, _1));
    inserter.insertLevel(MAGNITUDE_OF_LEVEL_CHANGE_MODEL_TAG,
                         boost::bind(&CNormalMeanPrecConjugate::acceptPersistInserter,
                                     &m_MagnitudeOfLevelChangeModel, _1));
}

bool CTrendComponent::acceptRestoreTraverser(const SDistributionRestoreParams& params,
                                             core::CStateRestoreTraverser& traverser) {
    std::size_t i{0};
    do {
        const std::string& name{traverser.name()};
        RESTORE_BUILT_IN(TARGET_DECAY_RATE_TAG, m_TargetDecayRate)
        RESTORE_BUILT_IN(FIRST_UPDATE_TAG, m_FirstUpdate)
        RESTORE_BUILT_IN(LAST_UPDATE_TAG, m_LastUpdate)
        RESTORE_BUILT_IN(REGRESSION_ORIGIN_TAG, m_RegressionOrigin)
        RESTORE(MODEL_TAG, traverser.traverseSubLevel(boost::bind(
                               &SModel::acceptRestoreTraverser, &m_TrendModels[i++], _1)))
        RESTORE_BUILT_IN(PREDICTION_ERROR_VARIANCE_TAG, m_PredictionErrorVariance)
        RESTORE(VALUE_MOMENTS_TAG, m_ValueMoments.fromDelimited(traverser.value()))
        RESTORE_BUILT_IN(TIME_OF_LAST_LEVEL_CHANGE_TAG, m_TimeOfLastLevelChange)
        RESTORE_NO_ERROR(PROBABILITY_OF_LEVEL_CHANGE_MODEL_TAG,
                         m_ProbabilityOfLevelChangeModel = std::move(CNaiveBayes(
                             naiveBayesExemplar(m_DefaultDecayRate), params, traverser)))
        RESTORE_NO_ERROR(MAGNITUDE_OF_LEVEL_CHANGE_MODEL_TAG,
                         m_MagnitudeOfLevelChangeModel =
                             CNormalMeanPrecConjugate(params, traverser))
    } while (traverser.next());
    return true;
}

bool CTrendComponent::initialized() const {
    return m_LastUpdate != UNSET_TIME;
}

void CTrendComponent::clear() {
    m_FirstUpdate = UNSET_TIME;
    m_LastUpdate = UNSET_TIME;
    m_RegressionOrigin = UNSET_TIME;
    for (std::size_t i = 0u; i < NUMBER_MODELS; ++i) {
        m_TrendModels[i] = SModel(modelWeight(1.0, TIME_SCALES[i]));
    }
    m_PredictionErrorVariance = 0.0;
    m_ValueMoments = TMeanVarAccumulator();
    m_TimeOfLastLevelChange = UNSET_TIME;
    m_ProbabilityOfLevelChangeModel = initialProbabilityOfChangeModel(m_DefaultDecayRate);
    m_MagnitudeOfLevelChangeModel = initialMagnitudeOfChangeModel(m_DefaultDecayRate);
}

void CTrendComponent::shiftOrigin(core_t::TTime time) {
    time = CIntegerTools::floor(time, core::constants::WEEK);
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
    // the prediction at the time fixed. This avoids a jump in the current
    // predicted value as a result of adjusting the gradient. Otherwise,
    // this changes by "shift" x "scaled time since the regression origin".
    if (std::fabs(shift) > 0.0) {
        for (std::size_t i = 0u; i < NUMBER_MODELS; ++i) {
            double modelDecayRate{m_DefaultDecayRate * TIME_SCALES[i]};
            double shift_{std::min(modelDecayRate / m_TargetDecayRate, 1.0) * shift};
            m_TrendModels[i].s_Regression.shiftGradient(shift_);
            m_TrendModels[i].s_Regression.shiftOrdinate(
                -shift_ * scaleTime(time, m_RegressionOrigin));
        }
    }
}

void CTrendComponent::shiftLevel(core_t::TTime time, double value, double shift) {
    for (auto& model : m_TrendModels) {
        model.s_Regression.shiftOrdinate(shift);
    }
    if (m_TimeOfLastLevelChange != UNSET_TIME) {
        double dt{static_cast<double>(time - m_TimeOfLastLevelChange)};
        m_ProbabilityOfLevelChangeModel.addTrainingDataPoint(LEVEL_CHANGE_LABEL,
                                                             {{dt}, {value}});
    }
    m_MagnitudeOfLevelChangeModel.addSamples({shift}, maths_t::CUnitWeights::SINGLE_UNIT);
    m_TimeOfLastLevelChange = time;
}

void CTrendComponent::dontShiftLevel(core_t::TTime time, double value) {
    if (m_TimeOfLastLevelChange != UNSET_TIME) {
        double dt{static_cast<double>(time - m_TimeOfLastLevelChange)};
        m_ProbabilityOfLevelChangeModel.addTrainingDataPoint(NO_CHANGE_LABEL,
                                                             {{dt}, {value}});
    }
}

void CTrendComponent::linearScale(double scale) {
    for (auto& model : m_TrendModels) {
        model.s_Regression.linearScale(scale);
    }
}

void CTrendComponent::add(core_t::TTime time, double value, double weight) {
    // Update the model weights: we weight the components based on the
    // relative difference in the component scale and the target scale.

    for (std::size_t i = 0u; i < NUMBER_MODELS; ++i) {
        m_TrendModels[i].s_Weight.add(
            modelWeight(m_TargetDecayRate, m_DefaultDecayRate * TIME_SCALES[i]));
    }

    if (m_FirstUpdate == UNSET_TIME) {
        m_RegressionOrigin = CIntegerTools::floor(time, core::constants::WEEK);
    }

    // Update the models.

    double prediction{CBasicStatistics::mean(this->value(time, 0.0))};

    double count{this->count()};
    if (count > 0.0) {
        TMeanVarAccumulator moments{CBasicStatistics::accumulator(
            count, prediction, m_PredictionErrorVariance)};
        moments.add(value, weight);
        m_PredictionErrorVariance = CBasicStatistics::maximumLikelihoodVariance(moments);
    }

    double scaledTime{scaleTime(time, m_RegressionOrigin)};
    for (auto& model : m_TrendModels) {
        model.s_ResidualMoments.add(value - model.s_Regression.predict(scaledTime, MAX_CONDITION));
        model.s_Regression.add(scaledTime, value, weight);
    }
    m_ValueMoments.add(value);

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
    TDoubleVec factors(this->factors(interval));
    double median{CBasicStatistics::median(factors)};
    for (std::size_t i = 0u; i < NUMBER_MODELS; ++i) {
        m_TrendModels[i].s_Weight.age(median);
        m_TrendModels[i].s_Regression.age(factors[i]);
        m_TrendModels[i].s_ResidualMoments.age(std::sqrt(factors[i]));
    }
    double interval_{static_cast<double>(interval) /
                     static_cast<double>(core::constants::DAY)};
    m_ProbabilityOfLevelChangeModel.propagateForwardsByTime(interval_);
    m_MagnitudeOfLevelChangeModel.propagateForwardsByTime(interval_);
}

CTrendComponent::TDoubleDoublePr CTrendComponent::value(core_t::TTime time,
                                                        double confidence) const {
    if (!this->initialized()) {
        return {0.0, 0.0};
    }

    double a{this->weightOfPrediction(time)};
    double b{1.0 - a};
    double scaledTime{scaleTime(time, m_RegressionOrigin)};

    TMeanAccumulator prediction_;

    TDoubleVec weights(this->factors(std::abs(time - m_LastUpdate)));
    double Z{0.0};
    for (std::size_t i = 0u; i < NUMBER_MODELS; ++i) {
        weights[i] *= CBasicStatistics::mean(m_TrendModels[i].s_Weight);
        Z += weights[i];
    }
    for (std::size_t i = 0u; i < NUMBER_MODELS; ++i) {
        if (weights[i] > MINIMUM_WEIGHT_TO_USE_MODEL_FOR_PREDICTION * Z) {
            prediction_.add(m_TrendModels[i].s_Regression.predict(scaledTime, MAX_CONDITION),
                            weights[i]);
        }
    }

    double prediction{a * CBasicStatistics::mean(prediction_) +
                      b * CBasicStatistics::mean(m_ValueMoments)};

    if (confidence > 0.0 && m_PredictionErrorVariance > 0.0) {
        double variance{a * m_PredictionErrorVariance / std::max(this->count(), 1.0) +
                        b * CBasicStatistics::variance(m_ValueMoments) /
                            std::max(CBasicStatistics::count(m_ValueMoments), 1.0)};
        if (auto interval = confidenceInterval(prediction, variance, confidence)) {
            return *interval;
        }
    }

    return {prediction, prediction};
}

CTrendComponent::TDoubleDoublePr CTrendComponent::variance(double confidence) const {
    if (!this->initialized()) {
        return {0.0, 0.0};
    }

    double variance{m_PredictionErrorVariance};

    if (confidence > 0.0 && m_PredictionErrorVariance > 0.0) {
        double df{std::max(this->count(), 2.0) - 1.0};
        try {
            boost::math::chi_squared chi{df};
            double ql{boost::math::quantile(chi, (100.0 - confidence) / 200.0)};
            double qu{boost::math::quantile(chi, (100.0 + confidence) / 200.0)};
            return {ql * variance / df, qu * variance / df};
        } catch (const std::exception& e) {
            LOG_ERROR(<< "Failed calculating confidence interval: " << e.what()
                      << ", df = " << df << ", confidence = " << confidence);
        }
    }

    return {variance, variance};
}

void CTrendComponent::forecast(core_t::TTime startTime,
                               core_t::TTime endTime,
                               core_t::TTime step,
                               double confidence,
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

    endTime = startTime + CIntegerTools::ceil(endTime - startTime, step);

    LOG_TRACE(<< "forecasting = " << this->print());

    TDoubleVec factors(this->factors(step));
    TDoubleVec modelWeights(this->initialForecastModelWeights());
    TDoubleVec errorWeights(this->initialForecastErrorWeights());
    TRegressionArrayVec models(NUMBER_MODELS);
    TMatrixVec modelCovariances(NUMBER_MODELS);
    TDoubleVec residualVariances(NUMBER_MODELS);
    for (std::size_t i = 0u; i < NUMBER_MODELS; ++i) {
        // Note in the following we multiply the bias by the sample count
        // when estimating the regression coefficients' covariance matrix.
        // This is because, unlike noise like component of the residual,
        // it is overly optimistic to assume these will cancel in the
        // estimates of the regression coefficient errors when adding
        // multiple samples. We make the most pessimistic assumption that
        // this is a fixed random variable and so multiply its covariance
        // by the number of samples to undo the scaling applied in the
        // covariance method.
        const SModel& model{m_TrendModels[i]};
        double n{CBasicStatistics::count(model.s_ResidualMoments)};
        double bias{CBasicStatistics::mean(model.s_ResidualMoments)};
        double variance{CBasicStatistics::variance(model.s_ResidualMoments)};
        residualVariances[i] = CTools::pow2(bias) + variance;
        model.s_Regression.parameters(models[i], MAX_CONDITION);
        model.s_Regression.covariances(n * CTools::pow2(bias) + variance, modelCovariances[i], MAX_CONDITION);
        LOG_TRACE(<< "params      = " << core::CContainerPrinter::print(models[i]));
        LOG_TRACE(<< "covariances = " << modelCovariances[i].toDelimited())
        LOG_TRACE(<< "variances   = " << residualVariances[i]);
    }
    LOG_TRACE(<< "long time variance = " << CBasicStatistics::variance(m_ValueMoments));

    CForecastLevel level{m_ProbabilityOfLevelChangeModel,
                         m_MagnitudeOfLevelChangeModel, m_TimeOfLastLevelChange};

    TDoubleVec variances(NUMBER_MODELS + 1);
    for (core_t::TTime time = startTime; time < endTime; time += step) {
        double scaledDt{scaleTime(time, startTime)};
        TVector times({0.0, scaledDt, scaledDt * scaledDt});

        double a{this->weightOfPrediction(time)};
        double b{1.0 - a};

        for (std::size_t j = 0u; j < NUMBER_MODELS; ++j) {
            modelWeights[j] *= factors[j];
            errorWeights[j] *= CTools::pow2(factors[j]);
        }

        for (std::size_t j = 0u; j < NUMBER_MODELS; ++j) {
            variances[j] = times.inner(modelCovariances[j] * times) + residualVariances[j];
        }
        variances[NUMBER_MODELS] = CBasicStatistics::variance(m_ValueMoments);
        for (auto v = variances.rbegin(); v != variances.rend(); ++v) {
            *v = *std::min_element(variances.rbegin(), v + 1);
        }
        TMeanAccumulator variance_;
        for (std::size_t j = 0u; j < NUMBER_MODELS; ++j) {
            variance_.add(variances[j], errorWeights[j]);
        }

        double prediction{this->value(modelWeights, models,
                                      scaleTime(time, m_RegressionOrigin))};
        TDouble3Vec seasonal_(seasonal(time));
        TDouble3Vec level_(level.forecast(time, seasonal_[1] + prediction, confidence));

        double ql{0.0};
        double qu{0.0};
        double variance{a * CBasicStatistics::mean(variance_) +
                        b * CBasicStatistics::variance(m_ValueMoments)};
        if (auto interval = confidenceInterval(0.0, variance, confidence)) {
            boost::tie(ql, qu) = *interval;
        }

        writer(time, {level_[0] + seasonal_[0] + prediction + ql,
                      level_[1] + seasonal_[1] + prediction,
                      level_[2] + seasonal_[2] + prediction + qu});
    }
}

core_t::TTime CTrendComponent::observedInterval() const {
    return m_LastUpdate - m_FirstUpdate;
}

double CTrendComponent::parameters() const {
    return static_cast<double>(TRegression::N);
}

uint64_t CTrendComponent::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_TargetDecayRate);
    seed = CChecksum::calculate(seed, m_FirstUpdate);
    seed = CChecksum::calculate(seed, m_LastUpdate);
    seed = CChecksum::calculate(seed, m_TrendModels);
    seed = CChecksum::calculate(seed, m_PredictionErrorVariance);
    seed = CChecksum::calculate(seed, m_ValueMoments);
    seed = CChecksum::calculate(seed, m_TimeOfLastLevelChange);
    seed = CChecksum::calculate(seed, m_ProbabilityOfLevelChangeModel);
    return CChecksum::calculate(seed, m_MagnitudeOfLevelChangeModel);
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

CTrendComponent::TDoubleVec CTrendComponent::factors(core_t::TTime interval) const {
    TDoubleVec result(NUMBER_MODELS);
    double factor{m_DefaultDecayRate * static_cast<double>(interval) /
                  static_cast<double>(core::constants::DAY)};
    for (std::size_t i = 0u; i < NUMBER_MODELS; ++i) {
        result[i] = std::exp(-TIME_SCALES[i] * factor);
    }
    return result;
}

CTrendComponent::TDoubleVec CTrendComponent::initialForecastModelWeights() const {
    TDoubleVec result(NUMBER_MODELS);
    for (std::size_t i = 0u; i < NUMBER_MODELS; ++i) {
        result[i] = std::exp(static_cast<double>(NUMBER_MODELS / 2) -
                             static_cast<double>(i));
    }
    return result;
}

CTrendComponent::TDoubleVec CTrendComponent::initialForecastErrorWeights() const {
    TDoubleVec result(NUMBER_MODELS + 1);
    for (std::size_t i = 0u; i < NUMBER_MODELS; ++i) {
        result[i] = std::exp(static_cast<double>(NUMBER_MODELS / 2) -
                             static_cast<double>(i));
    }
    result[NUMBER_MODELS] = result[NUMBER_MODELS - 1] / std::exp(1.0);
    return result;
}

double CTrendComponent::count() const {
    TMeanAccumulator result;
    for (const auto& model : m_TrendModels) {
        result.add(CTools::fastLog(model.s_Regression.count()),
                   CBasicStatistics::mean(model.s_Weight));
    }
    return std::exp(CBasicStatistics::mean(result));
}

double CTrendComponent::value(const TDoubleVec& weights,
                              const TRegressionArrayVec& models,
                              double time) const {
    TMeanAccumulator prediction;
    for (std::size_t i = 0u; i < models.size(); ++i) {
        prediction.add(CRegression::predict(models[i], time), weights[i]);
    }
    return CBasicStatistics::mean(prediction);
}

double CTrendComponent::weightOfPrediction(core_t::TTime time) const {
    double interval{static_cast<double>(m_LastUpdate - m_FirstUpdate)};
    if (interval == 0.0) {
        return 0.0;
    }

    double extrapolateInterval{static_cast<double>(CBasicStatistics::max(
        time - m_LastUpdate, m_FirstUpdate - time, core_t::TTime(0)))};
    if (extrapolateInterval == 0.0) {
        return 1.0;
    }

    return CTools::logisticFunction(extrapolateInterval / interval, 0.1, 1.0, -1.0);
}

CTrendComponent::SModel::SModel(double weight) {
    s_Weight.add(weight, 0.01);
}

void CTrendComponent::SModel::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(WEIGHT_TAG, s_Weight.toDelimited());
    inserter.insertLevel(REGRESSION_TAG, boost::bind(&TRegression::acceptPersistInserter,
                                                     &s_Regression, _1));
    inserter.insertValue(RESIDUAL_MOMENTS_TAG, s_ResidualMoments.toDelimited());
}

bool CTrendComponent::SModel::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name{traverser.name()};
        RESTORE(WEIGHT_TAG, s_Weight.fromDelimited(traverser.value()))
        RESTORE(REGRESSION_TAG,
                traverser.traverseSubLevel(boost::bind(
                    &TRegression::acceptRestoreTraverser, &s_Regression, _1)))
        RESTORE(RESIDUAL_MOMENTS_TAG, s_ResidualMoments.fromDelimited(traverser.value()))
    } while (traverser.next());
    return true;
}

uint64_t CTrendComponent::SModel::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, s_Weight);
    seed = CChecksum::calculate(seed, s_Regression);
    return CChecksum::calculate(seed, s_ResidualMoments);
}

CTrendComponent::CForecastLevel::CForecastLevel(const CNaiveBayes& probability,
                                                const CNormalMeanPrecConjugate& magnitude,
                                                core_t::TTime timeOfLastChange,
                                                std::size_t numberPaths)
    : m_Probability(probability), m_Magnitude(magnitude), m_Levels(numberPaths),
      m_TimesOfLastChange(numberPaths, timeOfLastChange),
      m_ProbabilitiesOfChange(numberPaths, 0.0) {
    m_Uniform01.reserve(numberPaths);
}

CTrendComponent::TDouble3Vec
CTrendComponent::CForecastLevel::forecast(core_t::TTime time, double prediction, double confidence) {
    TDouble3Vec result{0.0, 0.0, 0.0};

    if (m_Probability.initialized()) {
        CSampling::uniformSample(0.0, 1.0, m_Levels.size(), m_Uniform01);
        bool reorder{false};
        for (std::size_t i = 0u; i < m_Levels.size(); ++i) {
            double dt{static_cast<double>(time - m_TimesOfLastChange[i])};
            double x = m_Levels[i] + prediction;
            double p{m_Probability.classProbability(LEVEL_CHANGE_LABEL, {{dt}, {x}})};
            m_ProbabilitiesOfChange[i] = std::max(m_ProbabilitiesOfChange[i], p);
            if (m_Uniform01[i] < m_ProbabilitiesOfChange[i]) {
                double stepMean{m_Magnitude.marginalLikelihoodMean()};
                double stepVariance{m_Magnitude.marginalLikelihoodVariance()};
                m_Levels[i] += CSampling::normalSample(m_Rng, stepMean, stepVariance);
                m_TimesOfLastChange[i] = time;
                m_ProbabilitiesOfChange[i] = 0.0;
                reorder = true;
            }
        }
        if (reorder) {
            COrderings::simultaneousSort(m_Levels, m_TimesOfLastChange, m_ProbabilitiesOfChange);
        }

        double rollouts{static_cast<double>(m_Levels.size())};
        std::size_t lower{std::min(
            static_cast<std::size_t>((100.0 - confidence) / 200.0 * rollouts + 0.5),
            m_Levels.size())};
        std::size_t upper{std::min(
            static_cast<std::size_t>((100.0 + confidence) / 200.0 * rollouts + 0.5),
            m_Levels.size() - 1)};

        result[0] = m_Levels[lower];
        result[1] = CBasicStatistics::median(m_Levels);
        result[2] = m_Levels[upper];
    }

    return result;
}
}
}
