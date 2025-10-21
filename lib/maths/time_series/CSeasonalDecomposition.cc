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

#include <maths/time_series/CSeasonalDecomposition.h>

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
const core::TPersistenceTag SEASONALITY_TEST_7_11_TAG{"e", "seasonality_test"};
const core::TPersistenceTag SEASONAL_COMPONENTS_7_11_TAG{"f", "seasonal_components"};

const std::string EMPTY_STRING;
}

const core_t::TTime CSeasonalDecomposition::SMOOTHING_INTERVAL{14400};

CSeasonalDecomposition::CSeasonalDecomposition(double decayRate,
                                             core_t::TTime bucketLength,
                                             std::size_t seasonalComponentSize)
    : CTimeSeriesDecompositionBase(decayRate, bucketLength),
      m_TimeShift{0}, m_DecayRate{decayRate}, m_LastValueTime{0}, m_LastPropagationTime{0},
      m_SeasonalityTest{decayRate, bucketLength}, 
      m_SeasonalComponents{decayRate, bucketLength, seasonalComponentSize} {
}

CSeasonalDecomposition::CSeasonalDecomposition(const common::STimeSeriesDecompositionRestoreParams& params,
                                              core::CStateRestoreTraverser& traverser)
    : CTimeSeriesDecompositionBase(params.s_DecayRate, params.s_MinimumBucketLength),
      m_TimeShift{0}, m_DecayRate{params.s_DecayRate}, m_LastValueTime{0}, m_LastPropagationTime{0},
      m_SeasonalityTest{params.s_DecayRate, params.s_MinimumBucketLength},
      m_SeasonalComponents{params.s_DecayRate, params.s_MinimumBucketLength, params.s_ComponentSize} {
    
    if (traverser.traverseSubLevel([&](auto& traverser_) {
            if (traverser_.name() == VERSION_7_11_TAG) {
                while (traverser_.next()) {
                    const std::string& name{traverser_.name()};
                    RESTORE_BUILT_IN(TIME_SHIFT_7_11_TAG, m_TimeShift)
                    RESTORE_BUILT_IN(DECAY_RATE_7_11_TAG, m_DecayRate)
                    RESTORE_BUILT_IN(LAST_VALUE_TIME_7_11_TAG, m_LastValueTime)
                    RESTORE_BUILT_IN(LAST_PROPAGATION_TIME_7_11_TAG, m_LastPropagationTime)
                    RESTORE(SEASONALITY_TEST_7_11_TAG,
                            traverser_.traverseSubLevel([this](auto& traverser__) {
                                return m_SeasonalityTest.acceptRestoreTraverser(traverser__);
                            }))
                    RESTORE(SEASONAL_COMPONENTS_7_11_TAG,
                            traverser_.traverseSubLevel([&](auto& traverser__) {
                                return m_SeasonalComponents.acceptRestoreTraverser(params, traverser__);
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

CSeasonalDecomposition::CSeasonalDecomposition(const CSeasonalDecomposition& other, bool isForForecast)
    : CTimeSeriesDecompositionBase(other.decayRate(), other.bucketLength()),
      m_TimeShift{other.m_TimeShift}, m_DecayRate{other.m_DecayRate},
      m_LastValueTime{other.m_LastValueTime}, m_LastPropagationTime{other.m_LastPropagationTime},
      m_SeasonalityTest{other.m_SeasonalityTest, isForForecast},
      m_SeasonalComponents{other.m_SeasonalComponents} {
}

void CSeasonalDecomposition::swap(CSeasonalDecomposition& other) {
    std::swap(m_TimeShift, other.m_TimeShift);
    std::swap(m_DecayRate, other.m_DecayRate);
    std::swap(m_LastValueTime, other.m_LastValueTime);
    std::swap(m_LastPropagationTime, other.m_LastPropagationTime);
    m_SeasonalityTest.swap(other.m_SeasonalityTest);
    m_SeasonalComponents.swap(other.m_SeasonalComponents);
}

CSeasonalDecomposition& CSeasonalDecomposition::operator=(const CSeasonalDecomposition& other) {
    if (this != &other) {
        CSeasonalDecomposition copy{other};
        this->swap(copy);
    }
    return *this;
}

void CSeasonalDecomposition::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(VERSION_7_11_TAG, "");
    inserter.insertValue(TIME_SHIFT_7_11_TAG, m_TimeShift);
    inserter.insertValue(DECAY_RATE_7_11_TAG, m_DecayRate);
    inserter.insertValue(LAST_VALUE_TIME_7_11_TAG, m_LastValueTime);
    inserter.insertValue(LAST_PROPAGATION_TIME_7_11_TAG, m_LastPropagationTime);
    inserter.insertLevel(SEASONALITY_TEST_7_11_TAG, [this](auto& inserter_) {
        m_SeasonalityTest.acceptPersistInserter(inserter_);
    });
    inserter.insertLevel(SEASONAL_COMPONENTS_7_11_TAG, [this](auto& inserter_) {
        m_SeasonalComponents.acceptPersistInserter(inserter_);
    });
}

CSeasonalDecomposition* CSeasonalDecomposition::clone(bool isForForecast) const {
    return new CSeasonalDecomposition{*this, isForForecast};
}

void CSeasonalDecomposition::dataType(maths_t::EDataType dataType) {
    m_SeasonalComponents.dataType(dataType);
}

void CSeasonalDecomposition::decayRate(double decayRate) {
    m_DecayRate = decayRate;
    m_SeasonalComponents.decayRate(decayRate);
}

double CSeasonalDecomposition::decayRate() const {
    return m_DecayRate;
}

bool CSeasonalDecomposition::initialized() const {
    // Check if any seasonal components are initialized
    for (const auto& component : m_SeasonalComponents.seasonal()) {
        if (component.initialized()) {
            return true;
        }
    }
    return false;
}

void CSeasonalDecomposition::addPoint(core_t::TTime time,
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
        m_SeasonalComponents, componentChangeCallback, modelAnnotationCallback};

    time += m_TimeShift;

    core_t::TTime lastTime{std::max(m_LastValueTime, m_LastPropagationTime)};

    m_LastValueTime = std::max(m_LastValueTime, time);
    this->propagateForwardsTo(time);
    
    // Create a test for seasonality
    auto testForSeasonality = [this](core_t::TTime time_, const TBoolVec& removedSeasonalMask) {
        auto predictor_ = this->predictor();
        return predictor_(time_, removedSeasonalMask) +
               this->smooth([&](core_t::TTime shiftedTime) {
                   return predictor_(shiftedTime - m_TimeShift, removedSeasonalMask);
               }, time_);
    };

    // Create message for the seasonality test
    CTimeSeriesDecompositionDetail::SAddValue message{
        time,
        lastTime,
        m_TimeShift,
        value,
        weights,
        occupancy,
        firstValueTime,
        0.0, // No trend component
        this->calculateSeasonalPrediction(time, 0.0).mean(),
        0.0, // No calendar component
        CTimeSeriesDecompositionDetail::CNullTimeSeriesDecomposition{},
        []() { return [](core_t::TTime) { return 0.0; }; },
        [this] { return this->predictor(); },
        testForSeasonality,
        circuitBreaker
    };
    
    // Process the message
    m_SeasonalityTest.handle(message);
    m_SeasonalComponents.handleSeasonalComponents(message);
}

void CSeasonalDecomposition::shiftTime(core_t::TTime time, core_t::TTime shift) {
    m_SeasonalityTest.shiftTime(time, shift);
    m_TimeShift += shift;
    m_LastValueTime += shift;
    m_LastPropagationTime += shift;
}

void CSeasonalDecomposition::propagateForwardsTo(core_t::TTime time) {
    if (time > m_LastPropagationTime) {
        m_SeasonalityTest.propagateForwards(m_LastPropagationTime, time);
        m_SeasonalComponents.propagateForwards(m_LastPropagationTime, time);
    }
    m_LastPropagationTime = std::max(m_LastPropagationTime, time);
}

double CSeasonalDecomposition::meanValue(core_t::TTime time) const {
    time += m_TimeShift;
    
    // Sum of the mean values from all seasonal components
    double result = 0.0;
    for (const auto& component : m_SeasonalComponents.seasonal()) {
        if (component.initialized() && component.time().inWindow(time)) {
            result += component.value(time, 0.0).mean();
        }
    }
    
    return result;
}

CSeasonalDecomposition::TVector2x1 
CSeasonalDecomposition::calculateSeasonalPrediction(core_t::TTime time, double confidence) const {
    TVector2x1 prediction{0.0};
    
    for (const auto& component : m_SeasonalComponents.seasonal()) {
        if (component.initialized() && component.time().inWindow(time)) {
            prediction += component.value(time, confidence);
        }
    }
    
    return prediction;
}

CSeasonalDecomposition::TVector2x1 
CSeasonalDecomposition::value(core_t::TTime time, double confidence, bool isNonNegative) const {
    time += m_TimeShift;
    
    TVector2x1 result = this->calculateSeasonalPrediction(time, confidence);
    
    // Apply smoothing
    result += this->smooth(
        [&](core_t::TTime time_) {
            return this->calculateSeasonalPrediction(time_ - m_TimeShift, confidence);
        },
        time);
    
    return isNonNegative ? max(result, 0.0) : result;
}

core_t::TTime CSeasonalDecomposition::maximumForecastInterval() const {
    // Default forecast interval
    return 3 * core::constants::WEEK;
}

CSeasonalDecomposition::TDouble3Vec
CSeasonalDecomposition::calculateSeasonalForecastWithConfidenceInterval(core_t::TTime time, 
                                                                       double confidence,
                                                                       double minimumScale) const {
    m_SeasonalComponents.interpolateForForecast(time);

    TVector2x1 bounds{this->calculateSeasonalPrediction(time, confidence)};

    // Decompose the smoothing into shift plus stretch and ensure that the
    // smoothed interval between the prediction bounds remains positive length.
    TVector2x1 smoothing{this->smooth([&](core_t::TTime time_) {
        return this->calculateSeasonalPrediction(time_, confidence);
    }, time)};
    
    double shift{smoothing.mean()};
    double stretch{std::max(smoothing(1) - smoothing(0), bounds(0) - bounds(1))};
    bounds += TVector2x1{{shift - stretch / 2.0, shift + stretch / 2.0}};

    double variance{this->meanVariance()};
    double boundsScale{std::sqrt(std::max(
        minimumScale, this->varianceScaleWeight(time, variance, 0.0).mean()))};
    double prediction{(bounds(0) + bounds(1)) / 2.0};
    double interval{boundsScale * (bounds(1) - bounds(0))};

    return {prediction - interval / 2.0, prediction, prediction + interval / 2.0};
}

void CSeasonalDecomposition::forecast(core_t::TTime startTime,
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
    
    // Forecast only the seasonal components
    for (core_t::TTime time = startTime; time < endTime; time += step) {
        TDouble3Vec result{this->calculateSeasonalForecastWithConfidenceInterval(
            time, confidence, minimumScale)};
            
        if (isNonNegative) {
            result[0] = std::max(0.0, result[0]);
            result[1] = std::max(0.0, result[1]);
            result[2] = std::max(0.0, result[2]);
        }
        
        writer(time - m_TimeShift, result);
    }
}

double CSeasonalDecomposition::detrend(core_t::TTime time,
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
                 
            TVector2x1 seasonalPrediction = this->value(time + dt - m_TimeShift, confidence, false);
            double current{std::fabs(value - seasonalPrediction.mean())};
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

double CSeasonalDecomposition::meanVariance() const {
    double result = 0.0;
    std::size_t count = 0;
    
    for (const auto& component : m_SeasonalComponents.seasonal()) {
        if (component.initialized()) {
            result += component.meanVariance();
            ++count;
        }
    }
    
    return count > 0 ? result / static_cast<double>(count) : 0.0;
}

CSeasonalDecomposition::TVector2x1
CSeasonalDecomposition::varianceScaleWeight(core_t::TTime time, double variance, double confidence) const {
    time += m_TimeShift;
    
    // Calculate variance scale weight based on seasonal components
    TVector2x1 result{1.0};
    bool initialized = false;
    
    for (const auto& component : m_SeasonalComponents.seasonal()) {
        if (component.initialized() && component.time().inWindow(time)) {
            if (!initialized) {
                result = component.varianceScaleWeight(time, variance, confidence);
                initialized = true;
            } else {
                result = minmax(result, component.varianceScaleWeight(time, variance, confidence));
            }
        }
    }
    
    // Apply smoothing
    result += this->smooth(
        [&](core_t::TTime time_) {
            TVector2x1 scale{1.0};
            bool initialized_ = false;
            
            for (const auto& component : m_SeasonalComponents.seasonal()) {
                if (component.initialized() && component.time().inWindow(time_)) {
                    if (!initialized_) {
                        scale = component.varianceScaleWeight(time_, variance, confidence);
                        initialized_ = true;
                    } else {
                        scale = minmax(
                            scale, component.varianceScaleWeight(time_, variance, confidence));
                    }
                }
            }
            
            return scale;
        },
        time);
    
    return result;
}

double CSeasonalDecomposition::countWeight(core_t::TTime time) const {
    time += m_TimeShift;
    
    // Calculate count weight based on seasonal components
    double result = 1.0;
    bool initialized = false;
    
    for (const auto& component : m_SeasonalComponents.seasonal()) {
        if (component.initialized() && component.time().inWindow(time)) {
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

double CSeasonalDecomposition::outlierWeightDerate(core_t::TTime time, double error) const {
    time += m_TimeShift;
    
    // Calculate outlier weight derate based on seasonal components
    double result = 1.0;
    bool initialized = false;
    
    for (const auto& component : m_SeasonalComponents.seasonal()) {
        if (component.initialized() && component.time().inWindow(time)) {
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

CSeasonalDecomposition::TFloatMeanAccumulatorVec 
CSeasonalDecomposition::residuals(bool /*isNonNegative*/) const {
    // Combine residuals from all seasonal components
    TFloatMeanAccumulatorVec result;
    
    for (const auto& component : m_SeasonalComponents.seasonal()) {
        if (component.initialized()) {
            const auto& componentResiduals = component.residuals();
            result.insert(result.end(), componentResiduals.begin(), componentResiduals.end());
        }
    }
    
    return result;
}

void CSeasonalDecomposition::skipTime(core_t::TTime skipInterval) {
    m_SeasonalComponents.skipTime(skipInterval);
    m_LastValueTime += skipInterval;
    m_LastPropagationTime += skipInterval;
}

void CSeasonalDecomposition::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CSeasonalDecomposition");
    core::CMemoryDebug::dynamicSize("m_SeasonalComponents", m_SeasonalComponents, mem);
    core::CMemoryDebug::dynamicSize("m_SeasonalityTest", m_SeasonalityTest, mem);
}

std::size_t CSeasonalDecomposition::memoryUsage() const {
    return core::CMemory::dynamicSize(m_SeasonalComponents) + 
           core::CMemory::dynamicSize(m_SeasonalityTest);
}

std::size_t CSeasonalDecomposition::staticSize() const {
    return sizeof(*this);
}

core_t::TTime CSeasonalDecomposition::timeShift() const {
    return m_TimeShift;
}

const maths_t::TSeasonalComponentVec& CSeasonalDecomposition::seasonalComponents() const {
    return m_SeasonalComponents.seasonal();
}

const maths_t::TCalendarComponentVec& CSeasonalDecomposition::calendarComponents() const {
    // This class focuses on seasonal components only - no calendar components
    static const maths_t::TCalendarComponentVec EMPTY;
    return EMPTY;
}

const maths_t::TSeasonalComponentVec& CSeasonalDecomposition::seasonalComponents() const {
    return m_SeasonalComponents.seasonal();
}

CSeasonalDecomposition::TFilteredPredictor CSeasonalDecomposition::predictor() const {
    return [this](core_t::TTime time, const TBoolVec& removedSeasonalMask) {
        double result{0.0};
        time += m_TimeShift;

        const auto& seasonal = m_SeasonalComponents.seasonal();
        for (std::size_t i = 0; i < seasonal.size(); ++i) {
            if (seasonal[i].initialized() &&
                (removedSeasonalMask.empty() || removedSeasonalMask[i] == false) &&
                seasonal[i].time().inWindow(time)) {
                result += seasonal[i].value(time, 0.0).mean();
            }
        }

        return result;
    };
}

void CSeasonalDecomposition::interpolateForForecast(core_t::TTime time) {
    m_SeasonalComponents.interpolateForForecast(time);
}

template<typename F>
auto CSeasonalDecomposition::smooth(const F& f, core_t::TTime time) const -> decltype(f(time)) {
    using TResult = decltype(f(time));
    
    TResult result{0};

    // Check if we're within the smoothing interval of a weekend/weekday boundary
    if (CTimeSeriesDecompositionDetail::CSeasonalTime::isWithinBoundary(time, SMOOTHING_INTERVAL)) {
        core_t::TTime boundary{CTimeSeriesDecompositionDetail::CSeasonalTime::boundaryTime(time)};
        core_t::TTime dt{std::abs(time - boundary)};
        double weight{static_cast<double>(dt) / static_cast<double>(SMOOTHING_INTERVAL)};
        
        TResult forTime{f(time)};
        
        core_t::TTime reflect{2 * boundary - time};
        TResult forReflect{f(reflect)};

        result = weight * forTime + (1.0 - weight) * forReflect;
    }
    
    return result;
}

}
}
}
