/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
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

//! Get the desired weight for the regression model.
double modelWeight(double targetDecayRate, double modelDecayRate) {
    return targetDecayRate == modelDecayRate
               ? 1.0
               : std::min(targetDecayRate, modelDecayRate) / std::max(targetDecayRate, modelDecayRate);
}

//! We scale the time used for the regression model to improve
//! the condition of the design matrix.
double scaleTime(core_t::TTime time, core_t::TTime origin) {
    return static_cast<double>(time - origin) / static_cast<double>(core::constants::WEEK);
}

const std::string TARGET_DECAY_RATE_TAG{"a"};
const std::string FIRST_UPDATE_TAG{"b"};
const std::string LAST_UPDATE_TAG{"c"};
const std::string REGRESSION_ORIGIN_TAG{"d"};
const std::string MODEL_TAG{"e"};
const std::string PREDICTION_ERROR_VARIANCE_TAG{"f"};
const std::string VALUE_MOMENTS_TAG{"g"};
const std::string WEIGHT_TAG{"a"};
const std::string REGRESSION_TAG{"b"};
const std::string RESIDUAL_MOMENTS_TAG{"c"};

const double TIME_SCALES[]{144.0, 72.0, 36.0, 12.0, 4.0, 1.0, 0.25, 0.05};
const std::size_t NUMBER_MODELS{boost::size(TIME_SCALES)};
const double MAX_CONDITION{1e12};
const core_t::TTime UNSET_TIME{0};
}

CTrendComponent::CTrendComponent(double decayRate)
    : m_DefaultDecayRate(decayRate),
      m_TargetDecayRate(decayRate),
      m_FirstUpdate(UNSET_TIME),
      m_LastUpdate(UNSET_TIME),
      m_RegressionOrigin(UNSET_TIME),
      m_PredictionErrorVariance(0.0) {
    for (std::size_t i = 0u; i < NUMBER_MODELS; ++i) {
        m_Models.emplace_back(modelWeight(1.0, TIME_SCALES[i]));
    }
}

void CTrendComponent::swap(CTrendComponent& other) {
    std::swap(m_DefaultDecayRate, other.m_DefaultDecayRate);
    std::swap(m_TargetDecayRate, other.m_TargetDecayRate);
    std::swap(m_FirstUpdate, other.m_FirstUpdate);
    std::swap(m_LastUpdate, other.m_LastUpdate);
    std::swap(m_RegressionOrigin, other.m_RegressionOrigin);
    m_Models.swap(other.m_Models);
    std::swap(m_PredictionErrorVariance, other.m_PredictionErrorVariance);
    std::swap(m_ValueMoments, other.m_ValueMoments);
}

void CTrendComponent::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(TARGET_DECAY_RATE_TAG, m_TargetDecayRate);
    inserter.insertValue(FIRST_UPDATE_TAG, m_FirstUpdate);
    inserter.insertValue(LAST_UPDATE_TAG, m_LastUpdate);
    inserter.insertValue(REGRESSION_ORIGIN_TAG, m_RegressionOrigin);
    for (const auto& model : m_Models) {
        inserter.insertLevel(MODEL_TAG, boost::bind(&SModel::acceptPersistInserter, &model, _1));
    }
    inserter.insertValue(PREDICTION_ERROR_VARIANCE_TAG, m_PredictionErrorVariance, core::CIEEE754::E_DoublePrecision);
    inserter.insertValue(VALUE_MOMENTS_TAG, m_ValueMoments.toDelimited());
}

bool CTrendComponent::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    std::size_t i{0};
    do {
        const std::string& name{traverser.name()};
        RESTORE_BUILT_IN(TARGET_DECAY_RATE_TAG, m_TargetDecayRate)
        RESTORE_BUILT_IN(FIRST_UPDATE_TAG, m_FirstUpdate)
        RESTORE_BUILT_IN(LAST_UPDATE_TAG, m_LastUpdate)
        RESTORE_BUILT_IN(REGRESSION_ORIGIN_TAG, m_RegressionOrigin)
        RESTORE(MODEL_TAG, traverser.traverseSubLevel(boost::bind(&SModel::acceptRestoreTraverser, &m_Models[i++], _1)))
        RESTORE_BUILT_IN(PREDICTION_ERROR_VARIANCE_TAG, m_PredictionErrorVariance)
        RESTORE(VALUE_MOMENTS_TAG, m_ValueMoments.fromDelimited(traverser.value()))
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
        m_Models[i] = SModel(modelWeight(1.0, TIME_SCALES[i]));
    }
    m_PredictionErrorVariance = 0.0;
    m_ValueMoments = TMeanVarAccumulator();
}

void CTrendComponent::shiftOrigin(core_t::TTime time) {
    time = CIntegerTools::floor(time, core::constants::WEEK);
    double scaledShift{scaleTime(time, m_RegressionOrigin)};
    if (scaledShift > 0.0) {
        for (auto& model : m_Models) {
            model.s_Regression.shiftAbscissa(-scaledShift);
        }
        m_RegressionOrigin = time;
    }
}

void CTrendComponent::shiftSlope(double decayRate, double shift) {
    for (std::size_t i = 0u; i < NUMBER_MODELS; ++i) {
        double shift_{std::min(m_DefaultDecayRate * TIME_SCALES[i] / decayRate, 1.0) * shift};
        m_Models[i].s_Regression.shiftGradient(shift_);
    }
}

void CTrendComponent::add(core_t::TTime time, double value, double weight) {
    // Update the model weights: we weight the components based on the
    // relative difference in the component scale and the target scale.

    for (std::size_t i = 0u; i < NUMBER_MODELS; ++i) {
        m_Models[i].s_Weight.add(modelWeight(m_TargetDecayRate, m_DefaultDecayRate * TIME_SCALES[i]));
    }

    // Update the models.

    if (m_FirstUpdate == UNSET_TIME) {
        m_RegressionOrigin = CIntegerTools::floor(time, core::constants::WEEK);
    }

    double prediction{CBasicStatistics::mean(this->value(time, 0.0))};

    double count{this->count()};
    if (count > 0.0) {
        TMeanVarAccumulator moments{CBasicStatistics::accumulator(count, prediction, m_PredictionErrorVariance)};
        moments.add(value, weight);
        m_PredictionErrorVariance = CBasicStatistics::maximumLikelihoodVariance(moments);
    }

    double scaledTime{scaleTime(time, m_RegressionOrigin)};
    for (auto& model : m_Models) {
        model.s_Regression.add(scaledTime, value, weight);
        model.s_ResidualMoments.add(value - model.s_Regression.predict(scaledTime, MAX_CONDITION));
    }
    m_ValueMoments.add(value);

    m_FirstUpdate = m_FirstUpdate == UNSET_TIME ? time : std::min(m_FirstUpdate, time);
    m_LastUpdate = std::max(m_LastUpdate, time);
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
        m_Models[i].s_Weight.age(median);
        m_Models[i].s_Regression.age(factors[i]);
        m_Models[i].s_ResidualMoments.age(std::sqrt(factors[i]));
    }
}

CTrendComponent::TDoubleDoublePr CTrendComponent::value(core_t::TTime time, double confidence) const {
    if (!this->initialized()) {
        return {0.0, 0.0};
    }

    double a{this->weightOfPrediction(time)};
    double b{1.0 - a};
    double scaledTime{scaleTime(time, m_RegressionOrigin)};

    TMeanAccumulator prediction_;
    {
        TDoubleVec factors(this->factors(std::abs(time - m_LastUpdate)));
        for (std::size_t i = 0u; i < NUMBER_MODELS; ++i) {
            prediction_.add(m_Models[i].s_Regression.predict(scaledTime, MAX_CONDITION),
                            factors[i] * CBasicStatistics::mean(m_Models[i].s_Weight));
        }
    }

    double prediction{a * CBasicStatistics::mean(prediction_) + b * CBasicStatistics::mean(m_ValueMoments)};

    if (confidence > 0.0 && m_PredictionErrorVariance > 0.0) {
        double variance{a * m_PredictionErrorVariance / std::max(this->count(), 1.0) +
                        b * CBasicStatistics::variance(m_ValueMoments) /
                            std::max(CBasicStatistics::count(m_ValueMoments), 1.0)};
        try {
            boost::math::normal normal{prediction, std::sqrt(variance)};
            double ql{boost::math::quantile(normal, (100.0 - confidence) / 200.0)};
            double qu{boost::math::quantile(normal, (100.0 + confidence) / 200.0)};
            return {ql, qu};
        } catch (const std::exception& e) {
            LOG_ERROR("Failed calculating confidence interval: " << e.what() << ", prediction = " << prediction
                                                                 << ", variance = " << variance
                                                                 << ", confidence = " << confidence);
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
            LOG_ERROR("Failed calculating confidence interval: " << e.what() << ", df = " << df
                                                                 << ", confidence = " << confidence);
        }
    }

    return {variance, variance};
}

void CTrendComponent::forecast(core_t::TTime startTime,
                               core_t::TTime endTime,
                               core_t::TTime step,
                               double confidence,
                               TDouble3VecVec& result) const {
    result.clear();

    if (endTime < startTime) {
        LOG_ERROR("Bad forecast range: [" << startTime << "," << endTime << "]");
        return;
    }
    if (confidence < 0.0 || confidence >= 100.0) {
        LOG_ERROR("Bad confidence interval: " << confidence << "%");
        return;
    }

    endTime = startTime + CIntegerTools::ceil(endTime - startTime, step);

    core_t::TTime steps{(endTime - startTime) / step};
    result.resize(steps, TDouble3Vec(3));

    LOG_TRACE("forecasting = " << this->print());

    TDoubleVec factors(this->factors(step));

    TDoubleVec modelWeights(this->initialForecastModelWeights());
    TDoubleVec errorWeights(this->initialForecastErrorWeights());
    TRegressionArrayVec models(NUMBER_MODELS);
    TMatrixVec modelCovariances(NUMBER_MODELS);
    TDoubleVec residualVariances(NUMBER_MODELS);
    for (std::size_t i = 0u; i < NUMBER_MODELS; ++i) {
        m_Models[i].s_Regression.parameters(models[i], MAX_CONDITION);
        m_Models[i].s_Regression.covariances(m_PredictionErrorVariance, modelCovariances[i], MAX_CONDITION);
        modelCovariances[i] /= std::max(m_Models[i].s_Regression.count(), 1.0);
        residualVariances[i] = std::pow(CBasicStatistics::mean(m_Models[i].s_ResidualMoments), 2.0) +
                               CBasicStatistics::variance(m_Models[i].s_ResidualMoments);
        LOG_TRACE("params      = " << core::CContainerPrinter::print(models[i]));
        LOG_TRACE("covariances = " << modelCovariances[i].toDelimited())
    }
    LOG_TRACE("long time variance = " << CBasicStatistics::variance(m_ValueMoments));

    TDoubleVec variances(NUMBER_MODELS + 1);
    for (core_t::TTime time = startTime; time < endTime; time += step) {
        core_t::TTime pillar{(time - startTime) / step};
        double scaledDt{scaleTime(time, startTime)};
        TVector times({0.0, scaledDt, scaledDt * scaledDt});

        double a{this->weightOfPrediction(time)};
        double b{1.0 - a};

        for (std::size_t j = 0u; j < NUMBER_MODELS; ++j) {
            modelWeights[j] *= factors[j];
            errorWeights[j] *= std::pow(factors[j], 2.0);
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

        double prediction{this->value(modelWeights, models, scaleTime(time, m_RegressionOrigin))};
        double ql{0.0};
        double qu{0.0};
        double variance{a * CBasicStatistics::mean(variance_) + b * CBasicStatistics::variance(m_ValueMoments)};
        try {
            boost::math::normal normal{0.0, std::sqrt(variance)};
            ql = boost::math::quantile(normal, (100.0 - confidence) / 200.0);
            qu = boost::math::quantile(normal, (100.0 + confidence) / 200.0);
        } catch (const std::exception& e) {
            LOG_ERROR("Failed calculating confidence interval: " << e.what() << ", variance = " << variance
                                                                 << ", confidence = " << confidence);
        }

        result[pillar][0] = prediction + ql;
        result[pillar][1] = prediction;
        result[pillar][2] = prediction + qu;
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
    seed = CChecksum::calculate(seed, m_Models);
    seed = CChecksum::calculate(seed, m_PredictionErrorVariance);
    return CChecksum::calculate(seed, m_ValueMoments);
}

std::string CTrendComponent::print() const {
    std::ostringstream result;
    for (const auto& model : m_Models) {
        result << model.s_Regression.print() << "\n";
    }
    return result.str();
}

CTrendComponent::TDoubleVec CTrendComponent::factors(core_t::TTime interval) const {
    TDoubleVec result(NUMBER_MODELS);
    double factor{m_DefaultDecayRate * static_cast<double>(interval) / static_cast<double>(core::constants::DAY)};
    for (std::size_t i = 0u; i < NUMBER_MODELS; ++i) {
        result[i] = std::exp(-TIME_SCALES[i] * factor);
    }
    return result;
}

CTrendComponent::TDoubleVec CTrendComponent::initialForecastModelWeights() const {
    TDoubleVec result(NUMBER_MODELS);
    for (std::size_t i = 0u; i < NUMBER_MODELS; ++i) {
        result[i] = std::exp(static_cast<double>(NUMBER_MODELS / 2) - static_cast<double>(i));
    }
    return result;
}

CTrendComponent::TDoubleVec CTrendComponent::initialForecastErrorWeights() const {
    TDoubleVec result(NUMBER_MODELS + 1);
    for (std::size_t i = 0u; i < NUMBER_MODELS; ++i) {
        result[i] = std::exp(static_cast<double>(NUMBER_MODELS / 2) - static_cast<double>(i));
    }
    result[NUMBER_MODELS] = result[NUMBER_MODELS - 1] / std::exp(1.0);
    return result;
}

double CTrendComponent::count() const {
    TMeanAccumulator result;
    for (const auto& model : m_Models) {
        result.add(CTools::fastLog(model.s_Regression.count()), CBasicStatistics::mean(model.s_Weight));
    }
    return std::exp(CBasicStatistics::mean(result));
}

double CTrendComponent::value(const TDoubleVec& weights, const TRegressionArrayVec& models, double time) const {
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

    double extrapolateInterval{
        static_cast<double>(CBasicStatistics::max(time - m_LastUpdate, m_FirstUpdate - time, core_t::TTime(0)))};
    if (extrapolateInterval == 0.0) {
        return 1.0;
    }

    return CTools::smoothHeaviside(extrapolateInterval / interval, 1.0 / 12.0, -1.0);
}

CTrendComponent::SModel::SModel(double weight) {
    s_Weight.add(weight, 0.01);
}

void CTrendComponent::SModel::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(WEIGHT_TAG, s_Weight.toDelimited());
    inserter.insertLevel(REGRESSION_TAG, boost::bind(&TRegression::acceptPersistInserter, &s_Regression, _1));
    inserter.insertValue(RESIDUAL_MOMENTS_TAG, s_ResidualMoments.toDelimited());
}

bool CTrendComponent::SModel::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name{traverser.name()};
        RESTORE(WEIGHT_TAG, s_Weight.fromDelimited(traverser.value()))
        RESTORE(REGRESSION_TAG,
                traverser.traverseSubLevel(boost::bind(&TRegression::acceptRestoreTraverser, &s_Regression, _1)))
        RESTORE(RESIDUAL_MOMENTS_TAG, s_ResidualMoments.fromDelimited(traverser.value()))
    } while (traverser.next());
    return true;
}

uint64_t CTrendComponent::SModel::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, s_Weight);
    seed = CChecksum::calculate(seed, s_Regression);
    return CChecksum::calculate(seed, s_ResidualMoments);
}
}
}
