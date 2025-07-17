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

#include <maths/time_series/CCompleteDecomposition.h>

#include <core/CLogger.h>
#include <core/CMemoryDef.h>
#include <core/CMemoryUsage.h>
#include <core/CMemoryCircuitBreaker.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/common/CChecksum.h>
#include <maths/common/Constants.h>
#include <maths/common/CIntegerTools.h>
#include <maths/common/CMathsFuncs.h>
#include <maths/common/CMathsFuncsForMatrixAndVectorTypes.h>
#include <maths/common/CRestoreParams.h>

#include <maths/time_series/CTrendDecomposition.h>
#include <maths/time_series/CSeasonalDecomposition.h>
#include <maths/time_series/CCalendarDecomposition.h>
#include <maths/time_series/CTimeSeriesForecaster.h>
#include <maths/time_series/CTimeSeriesPredictor.h>

#include <algorithm>
#include <cmath>
#include <limits>
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
const core::TPersistenceTag CHANGE_POINT_TEST_7_11_TAG{"e", "change_point_test"};
const core::TPersistenceTag TREND_DECOMPOSITION_7_11_TAG{"f", "trend_decomposition"};
const core::TPersistenceTag SEASONAL_DECOMPOSITION_7_11_TAG{"g", "seasonal_decomposition"};
const core::TPersistenceTag CALENDAR_DECOMPOSITION_7_11_TAG{"h", "calendar_decomposition"};

const std::string EMPTY_STRING;
}

const core_t::TTime CCompleteDecomposition::DEFAULT_SMOOTHING_INTERVAL{14400};

CCompleteDecomposition::CCompleteDecomposition(double decayRate,
                                            core_t::TTime bucketLength,
                                            std::size_t seasonalComponentSize)
    : CTimeSeriesDecompositionBase(decayRate, bucketLength),
      m_TimeShift{0}, m_DecayRate{decayRate}, m_LastValueTime{0}, m_LastPropagationTime{0},
      m_ChangePointTest{decayRate, bucketLength},
      m_TrendDecomposition{std::make_unique<CTrendDecomposition>(decayRate, bucketLength)},
      m_SeasonalDecomposition{std::make_unique<CSeasonalDecomposition>(decayRate, bucketLength, seasonalComponentSize)},
      m_CalendarDecomposition{std::make_unique<CCalendarDecomposition>(decayRate, bucketLength, seasonalComponentSize)} {
    
    this->initializeMediator();
}

CCompleteDecomposition::CCompleteDecomposition(const common::STimeSeriesDecompositionRestoreParams& params,
                                             core::CStateRestoreTraverser& traverser)
    : CTimeSeriesDecompositionBase(params.s_DecayRate, params.s_MinimumBucketLength),
      m_TimeShift{0}, m_DecayRate{params.s_DecayRate}, m_LastValueTime{0}, m_LastPropagationTime{0},
      m_ChangePointTest{params.s_DecayRate, params.s_MinimumBucketLength},
      m_TrendDecomposition{std::make_unique<CTrendDecomposition>(params, traverser)},
      m_SeasonalDecomposition{std::make_unique<CSeasonalDecomposition>(params, traverser)},
      m_CalendarDecomposition{std::make_unique<CCalendarDecomposition>(params, traverser)} {
    
    if (traverser.traverseSubLevel([&](auto& traverser_) {
            if (traverser_.name() == VERSION_7_11_TAG) {
                while (traverser_.next()) {
                    const std::string& name{traverser_.name()};
                    RESTORE_BUILT_IN(TIME_SHIFT_7_11_TAG, m_TimeShift)
                    RESTORE_BUILT_IN(DECAY_RATE_7_11_TAG, m_DecayRate)
                    RESTORE_BUILT_IN(LAST_VALUE_TIME_7_11_TAG, m_LastValueTime)
                    RESTORE_BUILT_IN(LAST_PROPAGATION_TIME_7_11_TAG, m_LastPropagationTime)
                    RESTORE(CHANGE_POINT_TEST_7_11_TAG,
                            traverser_.traverseSubLevel([this](auto& traverser__) {
                                return m_ChangePointTest.acceptRestoreTraverser(traverser__);
                            }))
                    RESTORE(TREND_DECOMPOSITION_7_11_TAG,
                            traverser_.traverseSubLevel([&](auto& traverser__) {
                                bool success = true;
                                m_TrendDecomposition = std::make_unique<CTrendDecomposition>(params, traverser__);
                                return success;
                            }))
                    RESTORE(SEASONAL_DECOMPOSITION_7_11_TAG,
                            traverser_.traverseSubLevel([&](auto& traverser__) {
                                bool success = true;
                                m_SeasonalDecomposition = std::make_unique<CSeasonalDecomposition>(params, traverser__);
                                return success;
                            }))
                    RESTORE(CALENDAR_DECOMPOSITION_7_11_TAG,
                            traverser_.traverseSubLevel([&](auto& traverser__) {
                                bool success = true;
                                m_CalendarDecomposition = std::make_unique<CCalendarDecomposition>(params, traverser__);
                                return success;
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
    
    this->initializeMediator();
}

CCompleteDecomposition::CCompleteDecomposition(const CCompleteDecomposition& other, bool isForForecast)
    : CTimeSeriesDecompositionBase(other.decayRate(), other.bucketLength()),
      m_TimeShift{other.m_TimeShift}, m_DecayRate{other.m_DecayRate},
      m_LastValueTime{other.m_LastValueTime}, m_LastPropagationTime{other.m_LastPropagationTime},
      m_ChangePointTest{other.m_ChangePointTest, isForForecast},
      m_TrendDecomposition{std::make_unique<CTrendDecomposition>(*other.m_TrendDecomposition, isForForecast)},
      m_SeasonalDecomposition{std::make_unique<CSeasonalDecomposition>(*other.m_SeasonalDecomposition, isForForecast)},
      m_CalendarDecomposition{std::make_unique<CCalendarDecomposition>(*other.m_CalendarDecomposition, isForForecast)} {
    
    this->initializeMediator();
}

void CCompleteDecomposition::swap(CCompleteDecomposition& other) {
    std::swap(m_TimeShift, other.m_TimeShift);
    std::swap(m_DecayRate, other.m_DecayRate);
    std::swap(m_LastValueTime, other.m_LastValueTime);
    std::swap(m_LastPropagationTime, other.m_LastPropagationTime);
    m_ChangePointTest.swap(other.m_ChangePointTest);
    m_TrendDecomposition.swap(other.m_TrendDecomposition);
    m_SeasonalDecomposition.swap(other.m_SeasonalDecomposition);
    m_CalendarDecomposition.swap(other.m_CalendarDecomposition);
    m_Mediator.swap(other.m_Mediator);
}

CCompleteDecomposition& CCompleteDecomposition::operator=(const CCompleteDecomposition& other) {
    if (this != &other) {
        CCompleteDecomposition copy{other};
        this->swap(copy);
    }
    return *this;
}

void CCompleteDecomposition::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(VERSION_7_11_TAG, "");
    inserter.insertValue(TIME_SHIFT_7_11_TAG, m_TimeShift);
    inserter.insertValue(DECAY_RATE_7_11_TAG, m_DecayRate);
    inserter.insertValue(LAST_VALUE_TIME_7_11_TAG, m_LastValueTime);
    inserter.insertValue(LAST_PROPAGATION_TIME_7_11_TAG, m_LastPropagationTime);
    inserter.insertLevel(CHANGE_POINT_TEST_7_11_TAG, [this](auto& inserter_) {
        m_ChangePointTest.acceptPersistInserter(inserter_);
    });
    inserter.insertLevel(TREND_DECOMPOSITION_7_11_TAG, [this](auto& inserter_) {
        m_TrendDecomposition->acceptPersistInserter(inserter_);
    });
    inserter.insertLevel(SEASONAL_DECOMPOSITION_7_11_TAG, [this](auto& inserter_) {
        m_SeasonalDecomposition->acceptPersistInserter(inserter_);
    });
    inserter.insertLevel(CALENDAR_DECOMPOSITION_7_11_TAG, [this](auto& inserter_) {
        m_CalendarDecomposition->acceptPersistInserter(inserter_);
    });
}

CCompleteDecomposition* CCompleteDecomposition::clone(bool isForForecast) const {
    return new CCompleteDecomposition{*this, isForForecast};
}

void CCompleteDecomposition::dataType(maths_t::EDataType dataType) {
    m_TrendDecomposition->dataType(dataType);
    m_SeasonalDecomposition->dataType(dataType);
    m_CalendarDecomposition->dataType(dataType);
}

void CCompleteDecomposition::decayRate(double decayRate) {
    m_DecayRate = decayRate;
    m_TrendDecomposition->decayRate(decayRate);
    m_SeasonalDecomposition->decayRate(decayRate);
    m_CalendarDecomposition->decayRate(decayRate);
}

double CCompleteDecomposition::decayRate() const {
    return m_DecayRate;
}

bool CCompleteDecomposition::initialized() const {
    return m_TrendDecomposition->initialized() || 
           m_SeasonalDecomposition->initialized() ||
           m_CalendarDecomposition->initialized();
}

void CCompleteDecomposition::initializeMediator() {
    // For now, just create the mediator without handlers
    // A more comprehensive mediator implementation will be added later
    m_Mediator = std::make_unique<CTimeSeriesDecompositionDetail::CMediator>();
}

void CCompleteDecomposition::addPoint(core_t::TTime time,
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

    // Add point to each component
    m_TrendDecomposition->addPoint(time - m_TimeShift, value, circuitBreaker, weights, 
                                   componentChangeCallback, modelAnnotationCallback, 
                                   occupancy, firstValueTime);
    
    m_SeasonalDecomposition->addPoint(time - m_TimeShift, value, circuitBreaker, weights, 
                                      componentChangeCallback, modelAnnotationCallback, 
                                      occupancy, firstValueTime);
    
    m_CalendarDecomposition->addPoint(time - m_TimeShift, value, circuitBreaker, weights, 
                                      componentChangeCallback, modelAnnotationCallback, 
                                      occupancy, firstValueTime);
}

void CCompleteDecomposition::shiftTime(core_t::TTime time, core_t::TTime shift) {
    m_SeasonalDecomposition->shiftTime(time, shift);
    m_TimeShift += shift;
    m_LastValueTime += shift;
    m_LastPropagationTime += shift;
}

void CCompleteDecomposition::propagateForwardsTo(core_t::TTime time) {
    if (time > m_LastPropagationTime) {
        m_ChangePointTest.propagateForwards(m_LastPropagationTime, time);
        m_TrendDecomposition->propagateForwardsTo(time);
        m_SeasonalDecomposition->propagateForwardsTo(time);
        m_CalendarDecomposition->propagateForwardsTo(time);
    }
    m_LastPropagationTime = std::max(m_LastPropagationTime, time);
}

double CCompleteDecomposition::meanValue(core_t::TTime time) const {
    return m_TrendDecomposition->meanValue(time) + 
           m_SeasonalDecomposition->meanValue(time) + 
           m_CalendarDecomposition->meanValue(time);
}

CCompleteDecomposition::TVector2x1
CCompleteDecomposition::value(core_t::TTime time, double confidence, int components, bool smooth) const {
    // Create predictor if it doesn't exist
    if (!m_Predictor) {
        m_Predictor = std::make_unique<CTimeSeriesPredictor>(
            m_TrendDecomposition.get(), 
            m_SeasonalDecomposition.get(), 
            m_CalendarDecomposition.get());
    }
    
    // Create the prediction function based on the components
    auto predictionFunction = [this, time, confidence, components](core_t::TTime t) {
        TVector2x1 result{0.0};

        // Handle component-specific prediction requests
        if ((components & E_TrendForced) != 0 || 
            ((components & E_Trend) != 0 && m_TrendDecomposition->initialized())) {
            result += m_Predictor->trendValue(t, confidence, false);
        }

        if ((components & E_Seasonal) != 0) {
            result += m_Predictor->seasonalValue(t, confidence, false);
        }

        if ((components & E_Calendar) != 0) {
            result += m_Predictor->calendarValue(t, confidence, false);
        }

        return result;
    };

    // Apply smoothing if requested
    if (smooth && m_Smoother) {
        return this->smooth(predictionFunction, time);
    }
    
    // Otherwise return the raw prediction
    return predictionFunction(time);
}

CCompleteDecomposition::TVector2x1
CCompleteDecomposition::value(core_t::TTime time, double confidence, bool isNonNegative) const {
    // Create predictor if it doesn't exist
    if (!m_Predictor) {
        m_Predictor = std::make_unique<CTimeSeriesPredictor>(
            m_TrendDecomposition.get(), 
            m_SeasonalDecomposition.get(), 
            m_CalendarDecomposition.get());
    }
    
    // Use the predictor to get the combined value
    return m_Predictor->value(time, confidence, isNonNegative, m_TimeShift);
}

core_t::TTime CCompleteDecomposition::maximumForecastInterval() const {
    return m_TrendDecomposition->maximumForecastInterval();
}

void CCompleteDecomposition::forecast(core_t::TTime startTime,
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
    
    // Create forecaster if it doesn't exist
    if (!m_Forecaster) {
        m_Forecaster = std::make_unique<CTimeSeriesForecaster>(
            m_TrendDecomposition.get(), 
            m_SeasonalDecomposition.get(), 
            m_CalendarDecomposition.get());
    }
    
    // Delegate forecasting to the specialized forecaster class
    m_Forecaster->forecast(startTime, endTime, step, confidence, 
                          minimumScale, isNonNegative, m_TimeShift, writer);
}

double CCompleteDecomposition::detrend(core_t::TTime time,
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
                 
            TVector2x1 prediction = this->value(time + dt - m_TimeShift, confidence, false);
            double current{std::fabs(value - prediction.mean())};
            if (current < bestError) {
                bestShift = dt;
                bestError = current;
            }
        }
        
        time += bestShift;
    }

    // Apply detrending
    TVector2x1 prediction{this->value(time - m_TimeShift, confidence, isNonNegative)};
    double result{value - prediction.mean()};
    
    return result;
}

double CCompleteDecomposition::meanVariance() const {
    double trendVariance = m_TrendDecomposition->meanVariance();
    double seasonalVariance = m_SeasonalDecomposition->meanVariance();
    double calendarVariance = m_CalendarDecomposition->meanVariance();
    
    // Combine variances from all components
    return trendVariance + seasonalVariance + calendarVariance;
}

CCompleteDecomposition::TVector2x1
CCompleteDecomposition::varianceScaleWeight(core_t::TTime time, double variance, double confidence) const {
    time += m_TimeShift;
    
    // Combine variance scale weights from different components
    TVector2x1 trendScale = m_TrendDecomposition->varianceScaleWeight(time - m_TimeShift, variance, confidence);
    TVector2x1 seasonalScale = m_SeasonalDecomposition->varianceScaleWeight(time - m_TimeShift, variance, confidence);
    TVector2x1 calendarScale = m_CalendarDecomposition->varianceScaleWeight(time - m_TimeShift, variance, confidence);
    
    // Take the minimum of the lower bounds and the maximum of the upper bounds
    return {std::min({trendScale(0), seasonalScale(0), calendarScale(0)}),
            std::max({trendScale(1), seasonalScale(1), calendarScale(1)})};
}

double CCompleteDecomposition::countWeight(core_t::TTime time) const {
    time += m_TimeShift;
    
    // Take the minimum of all component count weights
    return std::min({m_TrendDecomposition->countWeight(time - m_TimeShift),
                     m_SeasonalDecomposition->countWeight(time - m_TimeShift),
                     m_CalendarDecomposition->countWeight(time - m_TimeShift)});
}

double CCompleteDecomposition::outlierWeightDerate(core_t::TTime time, double error) const {
    time += m_TimeShift;
    
    // Take the minimum of all component derate values
    return std::min({m_TrendDecomposition->outlierWeightDerate(time - m_TimeShift, error),
                     m_SeasonalDecomposition->outlierWeightDerate(time - m_TimeShift, error),
                     m_CalendarDecomposition->outlierWeightDerate(time - m_TimeShift, error)});
}

CCompleteDecomposition::TFloatMeanAccumulatorVec 
CCompleteDecomposition::residuals(bool isNonNegative) const {
    // Combine residuals from all components
    TFloatMeanAccumulatorVec result = m_TrendDecomposition->residuals(isNonNegative);
    
    // Add seasonal residuals
    TFloatMeanAccumulatorVec seasonalResiduals = m_SeasonalDecomposition->residuals(isNonNegative);
    result.insert(result.end(), seasonalResiduals.begin(), seasonalResiduals.end());
    
    // Add calendar residuals
    TFloatMeanAccumulatorVec calendarResiduals = m_CalendarDecomposition->residuals(isNonNegative);
    result.insert(result.end(), calendarResiduals.begin(), calendarResiduals.end());
    
    return result;
}

void CCompleteDecomposition::skipTime(core_t::TTime skipInterval) {
    m_TrendDecomposition->skipTime(skipInterval);
    m_SeasonalDecomposition->skipTime(skipInterval);
    m_CalendarDecomposition->skipTime(skipInterval);
    m_LastValueTime += skipInterval;
    m_LastPropagationTime += skipInterval;
}

std::uint64_t CCompleteDecomposition::checksum(std::uint64_t seed) const {
    seed = common::CChecksum::calculate(seed, m_TimeShift);
    seed = common::CChecksum::calculate(seed, m_DecayRate);
    seed = common::CChecksum::calculate(seed, m_LastValueTime);
    seed = common::CChecksum::calculate(seed, m_LastPropagationTime);
    seed = common::CChecksum::calculate(seed, m_ChangePointTest);
    seed = m_TrendDecomposition->checksum(seed);
    seed = m_SeasonalDecomposition->checksum(seed);
    seed = m_CalendarDecomposition->checksum(seed);
    return seed;
}

void CCompleteDecomposition::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CCompleteDecomposition");
    core::memory_debug::dynamicSize("m_TrendDecomposition", m_TrendDecomposition, mem);
    core::memory_debug::dynamicSize("m_SeasonalDecomposition", m_SeasonalDecomposition, mem);
    core::memory_debug::dynamicSize("m_CalendarDecomposition", m_CalendarDecomposition, mem);
    core::memory_debug::dynamicSize("m_ChangePointTest", m_ChangePointTest, mem);
    core::memory_debug::dynamicSize("m_Mediator", m_Mediator, mem);
    if (m_Forecaster) {
        core::memory_debug::dynamicSize("m_Forecaster", m_Forecaster, mem);
    }
    if (m_Predictor) {
        core::memory_debug::dynamicSize("m_Predictor", m_Predictor, mem);
    }
    if (m_Smoother) {
        core::memory_debug::dynamicSize("m_Smoother", m_Smoother, mem);
    }
}

std::size_t CCompleteDecomposition::memoryUsage() const {
    std::size_t mem = core::memory::dynamicSize(m_TrendDecomposition);
    mem += core::memory::dynamicSize(m_SeasonalDecomposition);
    mem += core::memory::dynamicSize(m_CalendarDecomposition);
    mem += core::memory::dynamicSize(m_ChangePointTest);
    mem += core::memory::dynamicSize(m_Mediator);
    
    // Include dynamically created components
    if (m_Forecaster) {
        mem += core::memory::dynamicSize(m_Forecaster);
    }
    if (m_Predictor) {
        mem += core::memory::dynamicSize(m_Predictor);
    }
    if (m_Smoother) {
        mem += core::memory::dynamicSize(m_Smoother);
    }
    
    return mem;
}

std::size_t CCompleteDecomposition::staticSize() const {
    return sizeof(*this);
}

core_t::TTime CCompleteDecomposition::timeShift() const {
    return m_TimeShift;
}

const maths_t::TSeasonalComponentVec& CCompleteDecomposition::seasonalComponents() const {
    return m_SeasonalDecomposition->seasonalComponents();
}

const maths_t::TCalendarComponentVec& CCompleteDecomposition::calendarComponents() const {
    return m_CalendarDecomposition->calendarComponents();
}

core_t::TTime CCompleteDecomposition::lastValueTime() const {
    return m_LastValueTime;
}

void CCompleteDecomposition::resetChangePointTest(core_t::TTime time) {
    m_ChangePointTest.reset(time);
}

CCompleteDecomposition::TFilteredPredictor CCompleteDecomposition::predictor() const {
    // Create predictor if it doesn't exist
    if (!m_Predictor) {
        m_Predictor = std::make_unique<CTimeSeriesPredictor>(
            m_TrendDecomposition.get(), 
            m_SeasonalDecomposition.get(), 
            m_CalendarDecomposition.get());
    }
    
    // Delegate to the specialized predictor class
    return m_Predictor->predictor();
}

const std::unique_ptr<CTrendDecomposition>& CCompleteDecomposition::trendDecomposition() const {
    return m_TrendDecomposition;
}
const std::unique_ptr<CSeasonalDecomposition>& CCompleteDecomposition::seasonalDecomposition() const {
    return m_SeasonalDecomposition;
}
const std::unique_ptr<CCalendarDecomposition>& CCompleteDecomposition::calendarDecomposition() const {
    return m_CalendarDecomposition;
}

} // namespace time_series
} // namespace maths
} // namespace ml
