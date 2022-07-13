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

#include <maths/time_series/CDecayRateController.h>

#include <core/CContainerPrinter.h>
#include <core/CFunctional.h>
#include <core/CLogger.h>
#include <core/CPersistUtils.h>
#include <core/RestoreMacros.h>

#include <maths/common/CBasicStatisticsPersist.h>
#include <maths/common/CChecksum.h>
#include <maths/common/CSampling.h>
#include <maths/common/CTools.h>

#include <algorithm>
#include <cmath>

namespace ml {
namespace maths {
namespace time_series {
namespace {

const std::string TARGET_TAG{"a"};
const std::string PREDICTION_MEAN_TAG{"b"};
const std::string BIAS_TAG{"c"};
const std::string RECENT_ABS_ERROR_TAG{"d"};
const std::string HISTORICAL_ABS_ERROR_TAG{"e"};
const std::string RNG_TAG{"f"};
const std::string MULTIPLIER_TAG{"g"};
const std::string CHECKS_TAG{"h"};

//! The factor by which we'll increase the decay rate per bucket.
const double INCREASE_RATE{1.2};
//! The factor by which we'll decrease the decay rate per bucket.
const double DECREASE_RATE{1.0 / INCREASE_RATE};
//! The long term statistic decay rate multiplier.
const double SLOW_DECAY_RATE{5.0};
//! The short term statistic decay rate multiplier.
const double FAST_DECAY_RATE{25.0};
//! The minimum ratio between the prediction short and long term
//! errors which causes us to increase decay rate.
const double ERROR_INCREASING{2.0};
//! The minimum ratio between the prediction long and short term
//! errors which causes us to increase the decay rate.
const double ERROR_DECREASING{1.5};
//! The maximum ratio between the prediction short and long term
//! errors which causes us to decrease decay rate if permitted by
//! the bias test.
const double ERROR_NOT_INCREASING{1.2};
//! The maximum ratio between the prediction long and short term
//! errors which causes us to decrease decay rate if permitted by
//! the bias test.
const double ERROR_NOT_DECREASING{1.2};
//! The minimum ratio between the prediction bias and error which
//! causes us to increase decay rate.
const double BIASED{0.5};
//! The maximum ratio between the prediction bias and error which
//! causes us to decrease decay rate if permitted by the error
//! increasing or decreasing tests.
const double NOT_BIASED{0.3};
//! The minimum number of prediction residuals we need to see before
//! we'll attempt to control the decay rate.
const double MINIMUM_COUNT_TO_CONTROL{336.0};
//! The minimum coefficient of variation for the prediction error
//! at which we'll bother to control decay rate.
const double MINIMUM_COV_TO_CONTROL{1e-4};
//! The minimum decay rate multiplier permitted.
const double MINIMUM_MULTIPLIER{0.2};
//! The maximum decay rate multiplier permitted.
const double MAXIMUM_MULTIPLIER{40.0};
//! The bias stat index.
const std::size_t BIAS{0};
//! The recent prediction error index.
const std::size_t RECENT_ERROR{1};
//! The long time average prediction error index.
const std::size_t HISTORIC_ERROR{2};

//! Compute the \p learnRate and \p decayRate adjusted minimum
//! count to control.
double minimumCountToControl(double learnRate, double decayRate) {
    return 0.0005 * MINIMUM_COUNT_TO_CONTROL * learnRate / decayRate;
}

//! Adjust the decay rate multiplier for long bucket lengths.
double adjustMultiplier(double multiplier, core_t::TTime bucketLength_) {
    double bucketLength{static_cast<double>(bucketLength_)};
    return std::pow(multiplier, std::min(bucketLength / 1800.0, 1.0));
}

//! Adjust the maximum decay rate multiplier for long bucket lengths.
double adjustedMaximumMultiplier(core_t::TTime bucketLength_) {
    double bucketLength{static_cast<double>(bucketLength_)};
    return MAXIMUM_MULTIPLIER /
           (1.0 + common::CTools::truncate((bucketLength - 1800.0) / 86400.0, 0.0, 1.0));
}
}

CDecayRateController::CDecayRateController() {
    m_Multiplier.add(m_Target);
}

CDecayRateController::CDecayRateController(int checks, std::size_t dimension)
    : m_Checks{checks}, m_PredictionMean(dimension), m_Bias(dimension),
      m_RecentAbsError(dimension), m_HistoricalAbsError(dimension) {
    m_Multiplier.add(m_Target);
}

int CDecayRateController::checks() const {
    return m_Checks;
}

void CDecayRateController::checks(int checks) {
    m_Checks = checks;
}

void CDecayRateController::reset() {
    m_Target = 1.0;
    m_Multiplier = TMeanAccumulator();
    m_PredictionMean = TMeanAccumulator1Vec(m_PredictionMean.size());
    m_Bias = TMeanAccumulator1Vec(m_Bias.size());
    m_RecentAbsError = TMeanAccumulator1Vec(m_RecentAbsError.size());
    m_HistoricalAbsError = TMeanAccumulator1Vec(m_HistoricalAbsError.size());
    m_Multiplier.add(m_Target);
}

bool CDecayRateController::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    m_Multiplier = TMeanAccumulator();
    do {
        const std::string& name = traverser.name();
        RESTORE_BUILT_IN(CHECKS_TAG, m_Checks)
        RESTORE_BUILT_IN(TARGET_TAG, m_Target)
        RESTORE(MULTIPLIER_TAG, m_Multiplier.fromDelimited(traverser.value()))
        RESTORE(RNG_TAG, m_Rng.fromString(traverser.value()))
        RESTORE(PREDICTION_MEAN_TAG,
                core::CPersistUtils::restore(PREDICTION_MEAN_TAG, m_PredictionMean, traverser));
        RESTORE(BIAS_TAG, core::CPersistUtils::restore(BIAS_TAG, m_Bias, traverser))
        RESTORE(RECENT_ABS_ERROR_TAG,
                core::CPersistUtils::restore(RECENT_ABS_ERROR_TAG, m_RecentAbsError, traverser))
        RESTORE(HISTORICAL_ABS_ERROR_TAG,
                core::CPersistUtils::restore(HISTORICAL_ABS_ERROR_TAG,
                                             m_HistoricalAbsError, traverser))
    } while (traverser.next());
    if (common::CBasicStatistics::count(m_Multiplier) == 0.0) {
        m_Multiplier.add(m_Target);
    }
    this->checkRestoredInvariants();
    return true;
}

void CDecayRateController::checkRestoredInvariants() const {
    VIOLATES_INVARIANT(m_PredictionMean.size(), !=, m_Bias.size());
    VIOLATES_INVARIANT(m_Bias.size(), !=, m_RecentAbsError.size());
    VIOLATES_INVARIANT(m_RecentAbsError.size(), !=, m_HistoricalAbsError.size());
    VIOLATES_INVARIANT(m_PredictionMean.size(), <=, 0);
}

void CDecayRateController::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(CHECKS_TAG, m_Checks);
    inserter.insertValue(TARGET_TAG, m_Target, core::CIEEE754::E_DoublePrecision);
    inserter.insertValue(MULTIPLIER_TAG, m_Multiplier.toDelimited());
    inserter.insertValue(RNG_TAG, m_Rng.toString());
    core::CPersistUtils::persist(PREDICTION_MEAN_TAG, m_PredictionMean, inserter);
    core::CPersistUtils::persist(BIAS_TAG, m_Bias, inserter);
    core::CPersistUtils::persist(RECENT_ABS_ERROR_TAG, m_RecentAbsError, inserter);
    core::CPersistUtils::persist(HISTORICAL_ABS_ERROR_TAG, m_HistoricalAbsError, inserter);
}

double CDecayRateController::multiplier(const TDouble1Vec& prediction,
                                        const TDouble1VecVec& predictionErrors,
                                        core_t::TTime bucketLength,
                                        double learnRate,
                                        double decayRate) {
    // We could estimate the, presumably non-linear, function describing
    // the dynamics of the various error quantities and minimize the bias
    // and short term absolute prediction error using the decay rate as a
    // control variable. In practice, we want to bound the decay rate in
    // a range around the target decay rate and increase it when we detect
    // a bias or that the short term prediction error is significantly
    // greater than long term prediction error and vice versa. Using bang-
    // bang control, with some hysteresis, on the rate of change of the
    // decay rate does this.

    std::size_t dimension{m_PredictionMean.size()};
    double count{this->count()};
    TMeanAccumulator1Vec* stats_[3];
    stats_[BIAS] = &m_Bias;
    stats_[RECENT_ERROR] = &m_RecentAbsError;
    stats_[HISTORIC_ERROR] = &m_HistoricalAbsError;
    double numberPredictionErrors{static_cast<double>(predictionErrors.size())};

    for (auto predictionError : predictionErrors) {
        if (predictionError.empty()) {
            continue;
        }

        for (std::size_t d = 0; d < dimension; ++d) {
            // Truncate the prediction error to deal with large outliers.
            if (count > 0.0) {
                double bias{common::CBasicStatistics::mean(m_Bias[d])};
                double width{10.0 * common::CBasicStatistics::mean(m_HistoricalAbsError[d])};
                predictionError[d] = common::CTools::truncate(
                    predictionError[d], bias - width, bias + width);
            }

            // The idea of the following is to allow the model memory
            // length to increase whilst the prediction errors are less
            // than some tolerance expressed in terms of the data's
            // coefficient of variation. We achieve this by adding on
            // noise with this magnitude with the understanding that if
            // the prediction errors are less than this the sum will be
            // unbiased and the error magnitudes will be uniform in time
            // so the controller will actively decrease the decay rate.

            double weight{learnRate / numberPredictionErrors};
            double sd{MINIMUM_COV_TO_CONTROL *
                      std::fabs(common::CBasicStatistics::mean(m_PredictionMean[d]))};
            double tolerance{
                sd > 0.0 ? common::CSampling::normalSample(m_Rng, 0.0, sd * sd) : 0.0};
            m_PredictionMean[d].add(prediction[d], weight);
            (*stats_[BIAS])[d].add(predictionError[d] + tolerance, weight);
            (*stats_[RECENT_ERROR])[d].add(std::fabs(predictionError[d] + tolerance), weight);
            (*stats_[HISTORIC_ERROR])[d].add(std::fabs(predictionError[d] + tolerance), weight);
            LOG_TRACE(<< "stats = " << stats_);
            LOG_TRACE(<< "predictions = " << common::CBasicStatistics::mean(m_PredictionMean));
        }
    }

    if (count > 0.0) {
        TDouble3Ary factors;
        factors[BIAS] = std::exp(-FAST_DECAY_RATE * decayRate);
        factors[RECENT_ERROR] = std::exp(-FAST_DECAY_RATE * decayRate);
        factors[HISTORIC_ERROR] = std::exp(-SLOW_DECAY_RATE * decayRate);
        for (auto& component : m_PredictionMean) {
            component.age(factors[HISTORIC_ERROR]);
        }
        for (std::size_t i = 0; i < 3; ++i) {
            for (auto& component : *stats_[i]) {
                component.age(factors[i]);
            }
        }
    }

    double result{1.0};

    if (count > minimumCountToControl(learnRate, decayRate)) {
        using TMaxAccumulator = common::CBasicStatistics::SMax<double>::TAccumulator;

        // Compute the change to apply to the target decay rate.
        TMaxAccumulator change;
        for (std::size_t d = 0; d < dimension; ++d) {
            TDouble3Ary stats;
            for (std::size_t i = 0; i < 3; ++i) {
                stats[i] = std::fabs(common::CBasicStatistics::mean((*stats_[i])[d]));
            }
            change.add(this->change(stats, bucketLength));
        }

        m_Target *= common::CTools::truncate(m_Target * change[0], MINIMUM_MULTIPLIER,
                                             adjustedMaximumMultiplier(bucketLength)) /
                    m_Target;

        // We smooth the target decay rate. Over time this should
        // converge to the single decay rate which would minimize
        // the error measures. We want to find the multiplier to
        // apply to the current decay rate such that it equals the
        // smoothed decay rate. This is just the ratio of the new
        // to old.
        double mean{common::CBasicStatistics::mean(m_Multiplier)};
        m_Multiplier.add(m_Target);
        m_Multiplier.age(std::exp(-mean * decayRate));
        result = common::CBasicStatistics::mean(m_Multiplier) / mean;
    }

    return result;
}

double CDecayRateController::multiplier() const {
    return common::CBasicStatistics::mean(m_Multiplier);
}

std::size_t CDecayRateController::dimension() const {
    return m_PredictionMean.size();
}

void CDecayRateController::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CDecayRateController");
    core::CMemoryDebug::dynamicSize("m_PredictionMean", m_PredictionMean, mem);
    core::CMemoryDebug::dynamicSize("m_Bias", m_Bias, mem);
    core::CMemoryDebug::dynamicSize("m_RecentAbsError", m_RecentAbsError, mem);
    core::CMemoryDebug::dynamicSize("m_HistoricalAbsError", m_HistoricalAbsError, mem);
}

std::size_t CDecayRateController::memoryUsage() const {
    std::size_t mem{core::CMemory::dynamicSize(m_PredictionMean)};
    mem += core::CMemory::dynamicSize(m_Bias);
    mem += core::CMemory::dynamicSize(m_RecentAbsError);
    mem += core::CMemory::dynamicSize(m_HistoricalAbsError);
    return mem;
}

std::uint64_t CDecayRateController::checksum(std::uint64_t seed) const {
    seed = common::CChecksum::calculate(seed, m_Checks);
    seed = common::CChecksum::calculate(seed, m_Target);
    seed = common::CChecksum::calculate(seed, m_Multiplier);
    seed = common::CChecksum::calculate(seed, m_Rng);
    seed = common::CChecksum::calculate(seed, m_PredictionMean);
    seed = common::CChecksum::calculate(seed, m_Bias);
    seed = common::CChecksum::calculate(seed, m_RecentAbsError);
    return common::CChecksum::calculate(seed, m_HistoricalAbsError);
}

double CDecayRateController::count() const {
    return common::CBasicStatistics::count(m_HistoricalAbsError[0]);
}

double CDecayRateController::change(const TDouble3Ary& stats, core_t::TTime bucketLength) const {
    if (this->notControlling()) {
        return 1.0;
    }
    if (this->increaseDecayRateErrorIncreasing(stats) ||
        this->increaseDecayRateErrorDecreasing(stats) ||
        this->increaseDecayRateBiased(stats)) {
        return adjustMultiplier(INCREASE_RATE, bucketLength);
    }
    if (this->decreaseDecayRateErrorNotIncreasing(stats) &&
        this->decreaseDecayRateErrorNotDecreasing(stats) &&
        this->decreaseDecayRateNotBiased(stats)) {
        return adjustMultiplier(DECREASE_RATE, bucketLength);
    }
    return 1.0;
}

bool CDecayRateController::notControlling() const {
    return (m_Checks & E_PredictionErrorIncrease) == 0 &&
           (m_Checks & E_PredictionErrorDecrease) == 0 &&
           (m_Checks & E_PredictionBias) == 0;
}

bool CDecayRateController::increaseDecayRateErrorIncreasing(const TDouble3Ary& stats) const {
    return (m_Checks & E_PredictionErrorIncrease) &&
           stats[RECENT_ERROR] > ERROR_INCREASING * stats[HISTORIC_ERROR];
}

bool CDecayRateController::increaseDecayRateErrorDecreasing(const TDouble3Ary& stats) const {
    return (m_Checks & E_PredictionErrorDecrease) &&
           stats[HISTORIC_ERROR] > ERROR_DECREASING * stats[RECENT_ERROR];
}

bool CDecayRateController::increaseDecayRateBiased(const TDouble3Ary& stats) const {
    return (m_Checks & E_PredictionBias) && stats[BIAS] > BIASED * stats[RECENT_ERROR];
}

bool CDecayRateController::decreaseDecayRateErrorNotIncreasing(const TDouble3Ary& stats) const {
    return (m_Checks & E_PredictionErrorIncrease) == 0 ||
           stats[RECENT_ERROR] < ERROR_NOT_INCREASING * stats[HISTORIC_ERROR];
}

bool CDecayRateController::decreaseDecayRateErrorNotDecreasing(const TDouble3Ary& stats) const {
    return (m_Checks & E_PredictionErrorDecrease) == 0 ||
           stats[HISTORIC_ERROR] < ERROR_NOT_DECREASING * stats[RECENT_ERROR];
}

bool CDecayRateController::decreaseDecayRateNotBiased(const TDouble3Ary& stats) const {
    return (m_Checks & E_PredictionBias) == 0 ||
           stats[BIAS] < NOT_BIASED * stats[RECENT_ERROR];
}
}
}
}
