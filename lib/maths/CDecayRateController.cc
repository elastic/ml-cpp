/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
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

#include <maths/CDecayRateController.h>

#include <core/CContainerPrinter.h>
#include <core/CFunctional.h>
#include <core/CLogger.h>
#include <core/CPersistUtils.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>
#include <maths/CSampling.h>
#include <maths/CTools.h>

#include <boost/range.hpp>

#include <algorithm>
#include <math.h>

namespace ml {
namespace maths {
namespace {

const std::string TARGET_TAG{"a"};
const std::string PREDICTION_MEAN_TAG{"b"};
const std::string BIAS_TAG{"c"};
const std::string RECENT_ABS_ERROR_TAG{"d"};
const std::string HISTORICAL_ABS_ERROR_TAG{"e"};
const std::string RNG_TAG{"f"};
const std::string MULTIPLIER_TAG{"g"};

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
    return MAXIMUM_MULTIPLIER / (1.0 + CTools::truncate((bucketLength - 1800.0) / 86400.0, 0.0, 1.0));
}
}

CDecayRateController::CDecayRateController(void) : m_Checks(0), m_Target(1.0) {
    m_Multiplier.add(m_Target);
}

CDecayRateController::CDecayRateController(int checks, std::size_t dimension)
    : m_Checks(checks),
      m_Target(1.0),
      m_PredictionMean(dimension),
      m_Bias(dimension),
      m_RecentAbsError(dimension),
      m_HistoricalAbsError(dimension) {
    m_Multiplier.add(m_Target);
}

void CDecayRateController::reset(void) {
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
        RESTORE_BUILT_IN(TARGET_TAG, m_Target)
        RESTORE(MULTIPLIER_TAG, m_Multiplier.fromDelimited(traverser.value()))
        RESTORE(RNG_TAG, m_Rng.fromString(traverser.value()))
        RESTORE(PREDICTION_MEAN_TAG, core::CPersistUtils::restore(PREDICTION_MEAN_TAG, m_PredictionMean, traverser));
        RESTORE(BIAS_TAG, core::CPersistUtils::restore(BIAS_TAG, m_Bias, traverser))
        RESTORE(RECENT_ABS_ERROR_TAG, core::CPersistUtils::restore(RECENT_ABS_ERROR_TAG, m_RecentAbsError, traverser))
        RESTORE(HISTORICAL_ABS_ERROR_TAG,
                core::CPersistUtils::restore(HISTORICAL_ABS_ERROR_TAG, m_HistoricalAbsError, traverser))
    } while (traverser.next());
    if (CBasicStatistics::count(m_Multiplier) == 0.0) {
        m_Multiplier.add(m_Target);
    }
    return true;
}

void CDecayRateController::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(TARGET_TAG, m_Target);
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
    TMeanAccumulator1Vec* stats_[]{&m_Bias, &m_RecentAbsError, &m_HistoricalAbsError};
    double numberPredictionErrors{static_cast<double>(predictionErrors.size())};

    for (auto predictionError : predictionErrors) {
        if (predictionError.empty()) {
            continue;
        }

        for (std::size_t d = 0u; d < dimension; ++d) {
            // Truncate the prediction error to deal with large outliers.
            if (count > 0.0) {
                double bias{CBasicStatistics::mean(m_Bias[d])};
                double width{10.0 * CBasicStatistics::mean(m_HistoricalAbsError[d])};
                predictionError[d] = CTools::truncate(predictionError[d], bias - width, bias + width);
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
            double sd{MINIMUM_COV_TO_CONTROL * std::fabs(CBasicStatistics::mean(m_PredictionMean[d]))};
            double tolerance{sd > 0.0 ? CSampling::normalSample(m_Rng, 0.0, sd * sd) : 0.0};
            m_PredictionMean[d].add(prediction[d], weight);
            (*stats_[0])[d].add(predictionError[d] + tolerance, weight);
            (*stats_[1])[d].add(std::fabs(predictionError[d] + tolerance), weight);
            (*stats_[2])[d].add(std::fabs(predictionError[d] + tolerance), weight);
            LOG_TRACE("stats = " << core::CContainerPrinter::print(stats_));
            LOG_TRACE("predictions = " << CBasicStatistics::mean(m_PredictionMean));
        }
    }

    if (count > 0.0) {
        double factors[]{std::exp(-FAST_DECAY_RATE * decayRate),
                         std::exp(-FAST_DECAY_RATE * decayRate),
                         std::exp(-SLOW_DECAY_RATE * decayRate)};
        for (auto& component : m_PredictionMean) {
            component.age(factors[2]);
        }
        for (std::size_t i = 0u; i < 3; ++i) {
            for (auto& component : *stats_[i]) {
                component.age(factors[i]);
            }
        }
    }

    double result{1.0};

    if (count > minimumCountToControl(learnRate, decayRate)) {
        using TMaxAccumulator = CBasicStatistics::SMax<double>::TAccumulator;

        // Compute the change to apply to the target decay rate.
        TMaxAccumulator change;
        for (std::size_t d = 0u; d < dimension; ++d) {
            double stats[3];
            for (std::size_t i = 0u; i < 3; ++i) {
                stats[i] = std::fabs(CBasicStatistics::mean((*stats_[i])[d]));
            }
            change.add(this->change(stats, bucketLength));
        }

        m_Target *=
            CTools::truncate(m_Target * change[0], MINIMUM_MULTIPLIER, adjustedMaximumMultiplier(bucketLength)) /
            m_Target;

        // We smooth the target decay rate. Over time this should
        // converge to the single decay rate which would minimize
        // the error measures. We want to find the multiplier to
        // apply to the current decay rate such that it equals the
        // smoothed decay rate. This is just the ratio of the new
        // to old.
        double mean{CBasicStatistics::mean(m_Multiplier)};
        m_Multiplier.add(m_Target);
        m_Multiplier.age(std::exp(-mean * decayRate));
        result = CBasicStatistics::mean(m_Multiplier) / mean;
    }

    return result;
}

double CDecayRateController::multiplier(void) const {
    return CBasicStatistics::mean(m_Multiplier);
}

std::size_t CDecayRateController::dimension(void) const {
    return m_PredictionMean.size();
}

void CDecayRateController::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CDecayRateController");
    core::CMemoryDebug::dynamicSize("m_PredictionMean", m_PredictionMean, mem);
    core::CMemoryDebug::dynamicSize("m_Bias", m_Bias, mem);
    core::CMemoryDebug::dynamicSize("m_RecentAbsError", m_RecentAbsError, mem);
    core::CMemoryDebug::dynamicSize("m_HistoricalAbsError", m_HistoricalAbsError, mem);
}

std::size_t CDecayRateController::memoryUsage(void) const {
    std::size_t mem = core::CMemory::dynamicSize(m_PredictionMean);
    mem += core::CMemory::dynamicSize(m_Bias);
    mem += core::CMemory::dynamicSize(m_RecentAbsError);
    mem += core::CMemory::dynamicSize(m_HistoricalAbsError);
    return mem;
}

uint64_t CDecayRateController::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_PredictionMean);
    seed = CChecksum::calculate(seed, m_Bias);
    seed = CChecksum::calculate(seed, m_RecentAbsError);
    return CChecksum::calculate(seed, m_HistoricalAbsError);
}

double CDecayRateController::count(void) const {
    return CBasicStatistics::count(m_HistoricalAbsError[0]);
}

double CDecayRateController::change(const double (&stats)[3], core_t::TTime bucketLength) const {
    if (((m_Checks & E_PredictionErrorIncrease) && stats[1] > ERROR_INCREASING * stats[2]) ||
        ((m_Checks & E_PredictionErrorDecrease) && stats[2] > ERROR_DECREASING * stats[1]) ||
        ((m_Checks & E_PredictionBias) && stats[0] > BIASED * stats[1])) {
        return adjustMultiplier(INCREASE_RATE, bucketLength);
    }
    if ((!(m_Checks & E_PredictionErrorIncrease) || stats[1] < ERROR_NOT_INCREASING * stats[2]) &&
        (!(m_Checks & E_PredictionErrorDecrease) || stats[2] < ERROR_NOT_DECREASING * stats[1]) &&
        (!(m_Checks & E_PredictionBias) || stats[0] < NOT_BIASED * stats[1])) {
        return adjustMultiplier(DECREASE_RATE, bucketLength);
    }
    return 1.0;
}
}
}
