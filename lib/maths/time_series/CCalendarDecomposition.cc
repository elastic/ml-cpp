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

#include <maths/time_series/CCalendarDecomposition.h>

#include <core/CLogger.h>
#include <core/CMemoryCircuitBreaker.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/common/CChecksum.h>
#include <maths/common/CIntegerTools.h>
#include <maths/common/CMathsFuncs.h>
#include <maths/common/CMathsFuncsForMatrixAndVectorTypes.h>
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
const core::TPersistenceTag CALENDAR_CYCLIC_TEST_7_11_TAG{"e", "calendar_cyclic_test"};
const core::TPersistenceTag CALENDAR_COMPONENTS_7_11_TAG{"f", "calendar_components"};

const std::string EMPTY_STRING;
}

CCalendarDecomposition::CCalendarDecomposition(double decayRate,
                                            core_t::TTime bucketLength,
                                            std::size_t seasonalComponentSize)
    : CTimeSeriesDecompositionBase(decayRate, bucketLength),
      m_TimeShift{0}, m_DecayRate{decayRate}, m_LastValueTime{0}, m_LastPropagationTime{0},
      m_CalendarCyclicTest{decayRate, bucketLength}, 
      m_CalendarComponents{decayRate, bucketLength, seasonalComponentSize} {
}

CCalendarDecomposition::CCalendarDecomposition(const common::STimeSeriesDecompositionRestoreParams& params,
                                              core::CStateRestoreTraverser& traverser)
    : CTimeSeriesDecompositionBase(params.s_DecayRate, params.s_MinimumBucketLength),
      m_TimeShift{0}, m_DecayRate{params.s_DecayRate}, m_LastValueTime{0}, m_LastPropagationTime{0},
      m_CalendarCyclicTest{params.s_DecayRate, params.s_MinimumBucketLength},
      m_CalendarComponents{params.s_DecayRate, params.s_MinimumBucketLength, params.s_ComponentSize} {
    
    if (traverser.traverseSubLevel([&](auto& traverser_) {
            if (traverser_.name() == VERSION_7_11_TAG) {
                while (traverser_.next()) {
                    const std::string& name{traverser_.name()};
                    RESTORE_BUILT_IN(TIME_SHIFT_7_11_TAG, m_TimeShift)
                    RESTORE_BUILT_IN(DECAY_RATE_7_11_TAG, m_DecayRate)
                    RESTORE_BUILT_IN(LAST_VALUE_TIME_7_11_TAG, m_LastValueTime)
                    RESTORE_BUILT_IN(LAST_PROPAGATION_TIME_7_11_TAG, m_LastPropagationTime)
                    RESTORE(CALENDAR_CYCLIC_TEST_7_11_TAG,
                            traverser_.traverseSubLevel([this](auto& traverser__) {
                                return m_CalendarCyclicTest.acceptRestoreTraverser(traverser__);
                            }))
                    RESTORE(CALENDAR_COMPONENTS_7_11_TAG,
                            traverser_.traverseSubLevel([&](auto& traverser__) {
                                return m_CalendarComponents.acceptRestoreTraverser(params, traverser__);
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

CCalendarDecomposition::CCalendarDecomposition(const CCalendarDecomposition& other, bool isForForecast)
    : CTimeSeriesDecompositionBase(other.decayRate(), other.bucketLength()),
      m_TimeShift{other.m_TimeShift}, m_DecayRate{other.m_DecayRate},
      m_LastValueTime{other.m_LastValueTime}, m_LastPropagationTime{other.m_LastPropagationTime},
      m_CalendarCyclicTest{other.m_CalendarCyclicTest, isForForecast},
      m_CalendarComponents{other.m_CalendarComponents} {
}

void CCalendarDecomposition::swap(CCalendarDecomposition& other) {
    std::swap(m_TimeShift, other.m_TimeShift);
    std::swap(m_DecayRate, other.m_DecayRate);
    std::swap(m_LastValueTime, other.m_LastValueTime);
    std::swap(m_LastPropagationTime, other.m_LastPropagationTime);
    m_CalendarCyclicTest.swap(other.m_CalendarCyclicTest);
    m_CalendarComponents.swap(other.m_CalendarComponents);
}

CCalendarDecomposition& CCalendarDecomposition::operator=(const CCalendarDecomposition& other) {
    if (this != &other) {
        CCalendarDecomposition copy{other};
        this->swap(copy);
    }
    return *this;
}

void CCalendarDecomposition::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(VERSION_7_11_TAG, "");
    inserter.insertValue(TIME_SHIFT_7_11_TAG, m_TimeShift);
    inserter.insertValue(DECAY_RATE_7_11_TAG, m_DecayRate);
    inserter.insertValue(LAST_VALUE_TIME_7_11_TAG, m_LastValueTime);
    inserter.insertValue(LAST_PROPAGATION_TIME_7_11_TAG, m_LastPropagationTime);
    inserter.insertLevel(CALENDAR_CYCLIC_TEST_7_11_TAG, [this](auto& inserter_) {
        m_CalendarCyclicTest.acceptPersistInserter(inserter_);
    });
    inserter.insertLevel(CALENDAR_COMPONENTS_7_11_TAG, [this](auto& inserter_) {
        m_CalendarComponents.acceptPersistInserter(inserter_);
    });
}

CCalendarDecomposition* CCalendarDecomposition::clone(bool isForForecast) const {
    return new CCalendarDecomposition{*this, isForForecast};
}

void CCalendarDecomposition::dataType(maths_t::EDataType dataType) {
    m_CalendarComponents.dataType(dataType);
}

void CCalendarDecomposition::decayRate(double decayRate) {
    m_DecayRate = decayRate;
    m_CalendarComponents.decayRate(decayRate);
}

double CCalendarDecomposition::decayRate() const {
    return m_DecayRate;
}

bool CCalendarDecomposition::initialized() const {
    // Check if any calendar components are initialized
    for (const auto& component : m_CalendarComponents.calendar()) {
        if (component.initialized()) {
            return true;
        }
    }
    return false;
}

void CCalendarDecomposition::addPoint(core_t::TTime time,
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

    // Make sure that we always attach this as the first thing we do.
    CTimeSeriesDecompositionDetail::CComponents::CScopeAttachComponentChangeCallback attach{
        m_CalendarComponents, componentChangeCallback, modelAnnotationCallback};

    time += m_TimeShift;

    core_t::TTime lastTime{std::max(m_LastValueTime, m_LastPropagationTime)};

    m_LastValueTime = std::max(m_LastValueTime, time);
    this->propagateForwardsTo(time);

    // Create message for the calendar cyclic test
    CTimeSeriesDecompositionDetail::SAddValue message{
        time,
        lastTime,
        m_TimeShift,
        value,
        weights,
        occupancy,
        firstValueTime,
        0.0, // No trend component
        0.0, // No seasonal component
        this->calculateCalendarPrediction(time, 0.0).mean(),
        CTimeSeriesDecompositionDetail::CNullTimeSeriesDecomposition{},
        []() { return [](core_t::TTime) { return 0.0; }; },
        []() { return [](core_t::TTime, const TBoolVec&) { return 0.0; }; },
        [](core_t::TTime, const TBoolVec&) { return 0.0; },
        circuitBreaker
    };
    
    // Process the message
    m_CalendarCyclicTest.handle(message);
    m_CalendarComponents.handleCalendarComponents(message);
}

void CCalendarDecomposition::propagateForwardsTo(core_t::TTime time) {
    if (time > m_LastPropagationTime) {
        m_CalendarCyclicTest.propagateForwards(m_LastPropagationTime, time);
        m_CalendarComponents.propagateForwards(m_LastPropagationTime, time);
    }
    m_LastPropagationTime = std::max(m_LastPropagationTime, time);
}

double CCalendarDecomposition::meanValue(core_t::TTime time) const {
    time += m_TimeShift;
    
    // Sum of the mean values from all calendar components
    double result = 0.0;
    for (const auto& component : m_CalendarComponents.calendar()) {
        if (component.initialized() && component.feature().inWindow(time)) {
            result += component.value(time, 0.0).mean();
        }
    }
    
    return result;
}

CCalendarDecomposition::TVector2x1 
CCalendarDecomposition::calculateCalendarPrediction(core_t::TTime time, double confidence) const {
    TVector2x1 prediction{0.0};
    
    for (const auto& component : m_CalendarComponents.calendar()) {
        if (component.initialized() && component.feature().inWindow(time)) {
            prediction += component.value(time, confidence);
        }
    }
    
    return prediction;
}

CCalendarDecomposition::TVector2x1 
CCalendarDecomposition::value(core_t::TTime time, double confidence, bool isNonNegative) const {
    time += m_TimeShift;
    
    TVector2x1 result = this->calculateCalendarPrediction(time, confidence);
    
    return isNonNegative ? max(result, 0.0) : result;
}

core_t::TTime CCalendarDecomposition::maximumForecastInterval() const {
    // Default forecast interval for calendar features
    return 3 * core::constants::MONTH;
}

CCalendarDecomposition::TDouble3Vec
CCalendarDecomposition::calculateCalendarForecastWithConfidenceInterval(core_t::TTime time, 
                                                                      double confidence,
                                                                      double minimumScale) const {
    m_CalendarComponents.interpolateForForecast(time);

    TVector2x1 bounds{this->calculateCalendarPrediction(time, confidence)};

    double variance{this->meanVariance()};
    double boundsScale{std::sqrt(std::max(
        minimumScale, this->varianceScaleWeight(time, variance, 0.0).mean()))};
    double prediction{bounds.mean()};
    double interval{boundsScale * (bounds(1) - bounds(0))};

    return {prediction - interval / 2.0, prediction, prediction + interval / 2.0};
}

void CCalendarDecomposition::forecast(core_t::TTime startTime,
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
    endTime = startTime + common::CIntegerTools::ceil(endTime - startTime, step);
    
    // Forecast only the calendar components
    for (core_t::TTime time = startTime; time < endTime; time += step) {
        TDouble3Vec result{this->calculateCalendarForecastWithConfidenceInterval(
            time, confidence, minimumScale)};
            
        if (isNonNegative) {
            result[0] = std::max(0.0, result[0]);
            result[1] = std::max(0.0, result[1]);
            result[2] = std::max(0.0, result[2]);
        }
        
        writer(time - m_TimeShift, result);
    }
}

double CCalendarDecomposition::detrend(core_t::TTime time,
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
                 
            TVector2x1 calendarPrediction = this->value(time + dt - m_TimeShift, confidence, false);
            double current{std::fabs(value - calendarPrediction.mean())};
            if (current < bestError) {
                bestShift = dt;
                bestError = current;
            }
        }
        
        time += bestShift;
    }

    // Apply detrending
    TVector2x1 prediction{this->value(time - m_TimeShift, confidence, false)};
    double result{value - prediction.mean()};
    
    return result;
}

double CCalendarDecomposition::meanVariance() const {
    double result = 0.0;
    std::size_t count = 0;
    
    for (const auto& component : m_CalendarComponents.calendar()) {
        if (component.initialized()) {
            result += component.meanVariance();
            ++count;
        }
    }
    
    return count > 0 ? result / static_cast<double>(count) : 0.0;
}

CCalendarDecomposition::TVector2x1
CCalendarDecomposition::varianceScaleWeight(core_t::TTime time, double variance, double confidence) const {
    time += m_TimeShift;
    
    // Calculate variance scale weight based on calendar components
    TVector2x1 result{1.0};
    bool initialized = false;
    
    for (const auto& component : m_CalendarComponents.calendar()) {
        if (component.initialized() && component.feature().inWindow(time)) {
            if (!initialized) {
                result = component.varianceScaleWeight(time, variance, confidence);
                initialized = true;
            } else {
                result = minmax(result, component.varianceScaleWeight(time, variance, confidence));
            }
        }
    }
    
    return result;
}

double CCalendarDecomposition::countWeight(core_t::TTime time) const {
    time += m_TimeShift;
    
    // Calculate count weight based on calendar components
    double result = 1.0;
    bool initialized = false;
    
    for (const auto& component : m_CalendarComponents.calendar()) {
        if (component.initialized() && component.feature().inWindow(time)) {
            if (!initialized) {
                result = component.countWeight(time);
                initialized = true;
            } else {
                result = std::min(result, component.countWeight(time));
            }
        }
    }
    
    return result;
}

double CCalendarDecomposition::outlierWeightDerate(core_t::TTime time, double error) const {
    time += m_TimeShift;
    
    // Calculate outlier weight derate based on calendar components
    double result = 1.0;
    bool initialized = false;
    
    for (const auto& component : m_CalendarComponents.calendar()) {
        if (component.initialized() && component.feature().inWindow(time)) {
            if (!initialized) {
                result = component.outlierWeightDerate(time, error);
                initialized = true;
            } else {
                result = std::min(result, component.outlierWeightDerate(time, error));
            }
        }
    }
    
    return result;
}

CCalendarDecomposition::TFloatMeanAccumulatorVec 
CCalendarDecomposition::residuals(bool /*isNonNegative*/) const {
    // Combine residuals from all calendar components
    TFloatMeanAccumulatorVec result;
    
    for (const auto& component : m_CalendarComponents.calendar()) {
        if (component.initialized()) {
            const auto& componentResiduals = component.residuals();
            result.insert(result.end(), componentResiduals.begin(), componentResiduals.end());
        }
    }
    
    return result;
}

void CCalendarDecomposition::skipTime(core_t::TTime skipInterval) {
    m_CalendarComponents.skipTime(skipInterval);
    m_LastValueTime += skipInterval;
    m_LastPropagationTime += skipInterval;
}

void CCalendarDecomposition::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CCalendarDecomposition");
    core::CMemoryDebug::dynamicSize("m_CalendarComponents", m_CalendarComponents, mem);
    core::CMemoryDebug::dynamicSize("m_CalendarCyclicTest", m_CalendarCyclicTest, mem);
}

std::size_t CCalendarDecomposition::memoryUsage() const {
    return core::CMemory::dynamicSize(m_CalendarComponents) + 
           core::CMemory::dynamicSize(m_CalendarCyclicTest);
}

std::size_t CCalendarDecomposition::staticSize() const {
    return sizeof(*this);
}

core_t::TTime CCalendarDecomposition::timeShift() const {
    return m_TimeShift;
}

const maths_t::TSeasonalComponentVec& CCalendarDecomposition::seasonalComponents() const {
    static const maths_t::TSeasonalComponentVec EMPTY;
    return EMPTY;
}

const maths_t::TCalendarComponentVec& CCalendarDecomposition::calendarComponents() const {
    return m_CalendarComponents.calendar();
}

CCalendarDecomposition::TFilteredPredictor CCalendarDecomposition::predictor() const {
    return [this](core_t::TTime time, const TBoolVec& /*ignored*/) {
        double result{0.0};
        time += m_TimeShift;

        for (const auto& component : m_CalendarComponents.calendar()) {
            if (component.initialized() && component.feature().inWindow(time)) {
                result += component.value(time, 0.0).mean();
            }
        }

        return result;
    };
}

void CCalendarDecomposition::interpolateForForecast(core_t::TTime time) {
    m_CalendarComponents.interpolateForForecast(time);
}

}
}
}
