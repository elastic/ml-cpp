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

#include <maths/time_series/CTrendDecomposition.h>

#include <core/CLogger.h>
#include <core/CMemoryCircuitBreaker.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/common/CChecksum.h>
#include <maths/common/CMathsFuncs.h>
#include <maths/common/CRestoreParams.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

namespace ml {
namespace maths {
namespace time_series {
namespace {

// Version 7.11
const std::string VERSION_7_11_TAG("7.11");
const core::TPersistenceTag TIME_SHIFT_7_11_TAG{"a", "time_shift"};
const core::TPersistenceTag DECAY_RATE_7_11_TAG{"b", "decay_rate"};
const core::TPersistenceTag LAST_VALUE_TIME_7_11_TAG{"c", "last_value_time"};
const core::TPersistenceTag LAST_PROPAGATION_TIME_7_11_TAG{"d", "last_propagation_time"};
const core::TPersistenceTag TREND_7_11_TAG{"e", "trend"};

const std::string EMPTY_STRING;
}

CTrendDecomposition::CTrendDecomposition(double decayRate, core_t::TTime bucketLength)
    : CTimeSeriesDecompositionBase(decayRate, bucketLength),
      m_TimeShift{0}, m_DecayRate{decayRate}, m_LastValueTime{0}, m_LastPropagationTime{0},
      m_Trend{decayRate, bucketLength} {
}

CTrendDecomposition::CTrendDecomposition(const common::STimeSeriesDecompositionRestoreParams& params,
                                         core::CStateRestoreTraverser& traverser)
    : CTimeSeriesDecompositionBase(params.s_DecayRate, params.s_MinimumBucketLength),
      m_TimeShift{0}, m_DecayRate{params.s_DecayRate}, m_LastValueTime{0}, m_LastPropagationTime{0},
      m_Trend{params.s_DecayRate, params.s_MinimumBucketLength} {
    
    if (traverser.traverseSubLevel([&](auto& traverser_) {
            if (traverser_.name() == VERSION_7_11_TAG) {
                while (traverser_.next()) {
                    const std::string& name{traverser_.name()};
                    RESTORE_BUILT_IN(TIME_SHIFT_7_11_TAG, m_TimeShift)
                    RESTORE_BUILT_IN(DECAY_RATE_7_11_TAG, m_DecayRate)
                    RESTORE_BUILT_IN(LAST_VALUE_TIME_7_11_TAG, m_LastValueTime)
                    RESTORE_BUILT_IN(LAST_PROPAGATION_TIME_7_11_TAG, m_LastPropagationTime)
                    RESTORE(TREND_7_11_TAG, traverser_.traverseSubLevel([&](auto& traverser__) {
                        return m_Trend.acceptRestoreTraverser(params, traverser__);
                    }))
                }
                return true;
            }
            LOG_ERROR(<< "Input error: unsupported state serialization version '"
                      << traverser_.name()
                      << "'. Currently supported minimum version: " << VERSION_7_11_TAG);
            return false;
        }) == false) {
        traverser.setBadState();
    }
}

CTrendDecomposition::CTrendDecomposition(const CTrendDecomposition& other, bool isForForecast)
    : CTimeSeriesDecompositionBase(other.decayRate(), other.bucketLength()),
      m_TimeShift{other.m_TimeShift}, m_DecayRate{other.m_DecayRate},
      m_LastValueTime{other.m_LastValueTime}, m_LastPropagationTime{other.m_LastPropagationTime},
      m_Trend{other.m_Trend, isForForecast} {
}

void CTrendDecomposition::swap(CTrendDecomposition& other) {
    std::swap(m_TimeShift, other.m_TimeShift);
    std::swap(m_DecayRate, other.m_DecayRate);
    std::swap(m_LastValueTime, other.m_LastValueTime);
    std::swap(m_LastPropagationTime, other.m_LastPropagationTime);
    m_Trend.swap(other.m_Trend);
}

CTrendDecomposition& CTrendDecomposition::operator=(const CTrendDecomposition& other) {
    if (this != &other) {
        CTrendDecomposition copy{other};
        this->swap(copy);
    }
    return *this;
}

void CTrendDecomposition::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(VERSION_7_11_TAG, "");
    inserter.insertValue(TIME_SHIFT_7_11_TAG, m_TimeShift);
    inserter.insertValue(DECAY_RATE_7_11_TAG, m_DecayRate);
    inserter.insertValue(LAST_VALUE_TIME_7_11_TAG, m_LastValueTime);
    inserter.insertValue(LAST_PROPAGATION_TIME_7_11_TAG, m_LastPropagationTime);
    inserter.insertLevel(TREND_7_11_TAG, [this](auto& inserter_) {
        m_Trend.acceptPersistInserter(inserter_);
    });
}

CTrendDecomposition* CTrendDecomposition::clone(bool isForForecast) const {
    return new CTrendDecomposition{*this, isForForecast};
}

void CTrendDecomposition::dataType(maths_t::EDataType dataType) {
    m_Trend.dataType(dataType);
}

void CTrendDecomposition::decayRate(double decayRate) {
    m_DecayRate = decayRate;
    m_Trend.decayRate(decayRate);
}

double CTrendDecomposition::decayRate() const {
    return m_DecayRate;
}

bool CTrendDecomposition::initialized() const {
    return m_Trend.initialized();
}

void CTrendDecomposition::addPoint(core_t::TTime time,
                                   double value,
                                   const core::CMemoryCircuitBreaker& circuitBreaker,
                                   const maths_t::TDoubleWeightsAry& weights,
                                   const TComponentChangeCallback& componentChangeCallback,
                                   const maths_t::TModelAnnotationCallback& modelAnnotationCallback,
                                   double occupancy,
                                   core_t::TTime firstValueTime) {
                                       
    if (common::CMathsFuncs::isFinite(value) == false) {
        LOG_ERROR(<< "Discarding invalid value.");
        return;
    }

    time += m_TimeShift;

    core_t::TTime lastTime{std::max(m_LastValueTime, m_LastPropagationTime)};

    m_LastValueTime = std::max(m_LastValueTime, time);
    this->propagateForwardsTo(time);

    // Add the point to the trend component
    m_Trend.add(time, value, weights, componentChangeCallback, modelAnnotationCallback, 
                circuitBreaker);
}

void CTrendDecomposition::propagateForwardsTo(core_t::TTime time) {
    if (time > m_LastPropagationTime) {
        m_Trend.propagateForwards(m_LastPropagationTime, time);
    }
    m_LastPropagationTime = std::max(m_LastPropagationTime, time);
}

double CTrendDecomposition::meanValue(core_t::TTime time) const {
    time += m_TimeShift;
    return m_Trend.meanValue(time);
}

CTrendDecomposition::TVector2x1 
CTrendDecomposition::value(core_t::TTime time, double confidence, bool isNonNegative) const {
    time += m_TimeShift;
    
    TVector2x1 result = m_Trend.value(time, confidence);
    
    return isNonNegative ? max(result, 0.0) : result;
}

core_t::TTime CTrendDecomposition::maximumForecastInterval() const {
    return m_Trend.maximumForecastInterval();
}

void CTrendDecomposition::forecast(core_t::TTime startTime,
                                  core_t::TTime endTime,
                                  core_t::TTime step,
                                  double confidence,
                                  double minimumScale,
                                  bool isNonNegative,
                                  const TWriteForecastResult& writer) {
    
    if (endTime < startTime) {
        LOG_ERROR(<< "Bad forecast range: [" << startTime << "," << endTime << "]");
        return;
    }
    if (confidence < 0.0 || confidence >= 100.0) {
        LOG_ERROR(<< "Bad confidence interval: " << confidence << "%");
        return;
    }

    startTime += m_TimeShift;
    endTime += m_TimeShift;
    
    // Forecast only the trend component
    m_Trend.forecast(startTime, endTime, step, confidence, writer);
}

double CTrendDecomposition::detrend(core_t::TTime time,
                                   double value,
                                   double confidence,
                                   bool isNonNegative,
                                   core_t::TTime maximumTimeShift) const {
    time += m_TimeShift;
    
    if (maximumTimeShift > 0) {
        core_t::TTime bestShift{0};
        double bestError{std::numeric_limits<double>::max()};
        
        // Find the best shift within the allowed range
        for (core_t::TTime dt = -maximumTimeShift; dt <= maximumTimeShift;
             dt = std::min(maximumTimeShift, dt + bucketLength())) {
            TVector2x1 prediction{m_Trend.value(time + dt, confidence)};
            double current{std::fabs(value - prediction.mean())};
            if (current < bestError) {
                bestShift = dt;
                bestError = current;
            }
        }
        
        time += bestShift;
    }

    // Apply detrending
    TVector2x1 prediction{m_Trend.value(time, confidence)};
    double result{value - prediction.mean()};
    
    return result;
}

double CTrendDecomposition::meanVariance() const {
    return m_Trend.meanVariance();
}

CTrendDecomposition::TVector2x1
CTrendDecomposition::varianceScaleWeight(core_t::TTime time, double variance, double confidence) const {
    time += m_TimeShift;
    
    // For trend-only decomposition, we just use the trend variance scale
    return m_Trend.varianceScaleWeight(time, variance, confidence);
}

double CTrendDecomposition::countWeight(core_t::TTime time) const {
    time += m_TimeShift;
    return m_Trend.countWeight(time);
}

double CTrendDecomposition::outlierWeightDerate(core_t::TTime time, double error) const {
    time += m_TimeShift;
    return m_Trend.outlierWeightDerate(time, error);
}

CTrendDecomposition::TFloatMeanAccumulatorVec CTrendDecomposition::residuals(bool /*isNonNegative*/) const {
    return m_Trend.residuals();
}

void CTrendDecomposition::skipTime(core_t::TTime skipInterval) {
    m_Trend.skipTime(skipInterval);
    m_LastValueTime += skipInterval;
    m_LastPropagationTime += skipInterval;
}

void CTrendDecomposition::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CTrendDecomposition");
    core::CMemoryDebug::dynamicSize("m_Trend", m_Trend, mem);
}

std::size_t CTrendDecomposition::memoryUsage() const {
    return m_Trend.memoryUsage();
}

std::size_t CTrendDecomposition::staticSize() const {
    return sizeof(*this);
}

core_t::TTime CTrendDecomposition::timeShift() const {
    return m_TimeShift;
}

const maths_t::TSeasonalComponentVec& CTrendDecomposition::seasonalComponents() const {
    static const maths_t::TSeasonalComponentVec EMPTY;
    return EMPTY;
}

const maths_t::TCalendarComponentVec& CTrendDecomposition::calendarComponents() const {
    static const maths_t::TCalendarComponentVec EMPTY;
    return EMPTY;
}

bool CTrendDecomposition::usingTrendForPrediction() const {
    return m_Trend.initialized();
}

CTrendDecomposition::TFilteredPredictor CTrendDecomposition::predictor() const {
    auto trend = m_Trend.predictor();
    
    return [trend](core_t::TTime time, const TBoolVec& /*ignored*/) {
        return trend(time);
    };
}

}
}
}
