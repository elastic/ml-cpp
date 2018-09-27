/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CTimeSeriesDecomposition.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/Constants.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CCalendarComponent.h>
#include <maths/CChecksum.h>
#include <maths/CExpandingWindow.h>
#include <maths/CIntegerTools.h>
#include <maths/CPrior.h>
#include <maths/CRestoreParams.h>
#include <maths/CSeasonalTime.h>
#include <maths/CTimeSeriesChangeDetector.h>
#include <maths/CTools.h>

#include <boost/bind.hpp>
#include <boost/container/flat_map.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/random/normal_distribution.hpp>

#include <cmath>
#include <string>
#include <utility>

namespace ml {
namespace maths {
namespace {

using TDoubleDoublePr = maths_t::TDoubleDoublePr;
using TVector2x1 = CVectorNx1<double, 2>;

//! Convert a double pair to a 2x1 vector.
TVector2x1 vector2x1(const TDoubleDoublePr& p) {
    TVector2x1 result;
    result(0) = p.first;
    result(1) = p.second;
    return result;
}

//! Convert a 2x1 vector to a double pair.
TDoubleDoublePr pair(const TVector2x1& v) {
    return {v(0), v(1)};
}

//! Get the normal confidence interval.
TDoubleDoublePr confidenceInterval(double confidence, double variance) {
    if (variance > 0.0) {
        try {
            boost::math::normal normal(0.0, std::sqrt(variance));
            double ql{boost::math::quantile(normal, (100.0 - confidence) / 200.0)};
            double qu{boost::math::quantile(normal, (100.0 + confidence) / 200.0)};
            return {ql, qu};
        } catch (const std::exception& e) {
            LOG_ERROR(<< "Failed calculating confidence interval: " << e.what()
                      << ", variance = " << variance << ", confidence = " << confidence);
        }
    }
    return {0.0, 0.0};
}

// Version 6.3
const std::string VERSION_6_3_TAG("6.3");
const std::string LAST_VALUE_TIME_6_3_TAG{"a"};
const std::string LAST_PROPAGATION_TIME_6_3_TAG{"b"};
const std::string PERIODICITY_TEST_6_3_TAG{"c"};
const std::string CALENDAR_CYCLIC_TEST_6_3_TAG{"d"};
const std::string COMPONENTS_6_3_TAG{"e"};
const std::string TIME_SHIFT_6_3_TAG{"f"};
// Version < 6.3
const std::string DECAY_RATE_OLD_TAG{"a"};
const std::string LAST_VALUE_TIME_OLD_TAG{"b"};
const std::string CALENDAR_CYCLIC_TEST_OLD_TAG{"f"};
const std::string COMPONENTS_OLD_TAG{"g"};
const std::string LAST_PROPAGATION_TIME_OLD_TAG{"h"};

const std::string EMPTY_STRING;
}

CTimeSeriesDecomposition::CTimeSeriesDecomposition(double decayRate,
                                                   core_t::TTime bucketLength,
                                                   std::size_t seasonalComponentSize)
    : m_TimeShift{0}, m_LastValueTime{0}, m_LastPropagationTime{0},
      m_PeriodicityTest{decayRate, bucketLength}, m_CalendarCyclicTest{decayRate, bucketLength},
      m_Components{decayRate, bucketLength, seasonalComponentSize} {
    this->initializeMediator();
}

CTimeSeriesDecomposition::CTimeSeriesDecomposition(const STimeSeriesDecompositionRestoreParams& params,
                                                   core::CStateRestoreTraverser& traverser)
    : m_TimeShift{0}, m_LastValueTime{0}, m_LastPropagationTime{0},
      m_PeriodicityTest{params.s_DecayRate, params.s_MinimumBucketLength},
      m_CalendarCyclicTest{params.s_DecayRate, params.s_MinimumBucketLength},
      m_Components{params.s_DecayRate, params.s_MinimumBucketLength, params.s_ComponentSize} {
    traverser.traverseSubLevel(
        boost::bind(&CTimeSeriesDecomposition::acceptRestoreTraverser, this,
                    boost::cref(params.s_ChangeModelParams), _1));
    this->initializeMediator();
}

CTimeSeriesDecomposition::CTimeSeriesDecomposition(const CTimeSeriesDecomposition& other,
                                                   bool isForForecast)
    : m_TimeShift{other.m_TimeShift}, m_LastValueTime{other.m_LastValueTime},
      m_LastPropagationTime{other.m_LastPropagationTime},
      m_PeriodicityTest{other.m_PeriodicityTest, isForForecast},
      m_CalendarCyclicTest{other.m_CalendarCyclicTest, isForForecast}, m_Components{
                                                                           other.m_Components} {
    this->initializeMediator();
}

bool CTimeSeriesDecomposition::acceptRestoreTraverser(const SDistributionRestoreParams& params,
                                                      core::CStateRestoreTraverser& traverser) {
    if (traverser.name() == VERSION_6_3_TAG) {
        while (traverser.next()) {
            const std::string& name{traverser.name()};
            RESTORE_BUILT_IN(TIME_SHIFT_6_3_TAG, m_TimeShift)
            RESTORE_BUILT_IN(LAST_VALUE_TIME_6_3_TAG, m_LastValueTime)
            RESTORE_BUILT_IN(LAST_PROPAGATION_TIME_6_3_TAG, m_LastPropagationTime)
            RESTORE(PERIODICITY_TEST_6_3_TAG,
                    traverser.traverseSubLevel(boost::bind(&CPeriodicityTest::acceptRestoreTraverser,
                                                           &m_PeriodicityTest, _1)))
            RESTORE(CALENDAR_CYCLIC_TEST_6_3_TAG,
                    traverser.traverseSubLevel(boost::bind(&CCalendarTest::acceptRestoreTraverser,
                                                           &m_CalendarCyclicTest, _1)))
            RESTORE(COMPONENTS_6_3_TAG,
                    traverser.traverseSubLevel(
                        boost::bind(&CComponents::acceptRestoreTraverser, &m_Components,
                                    boost::cref(params), m_LastValueTime, _1)))
        }
    } else {
        // There is no version string this is historic state.
        double decayRate{0.012};
        do {
            const std::string& name{traverser.name()};
            RESTORE_BUILT_IN(DECAY_RATE_OLD_TAG, decayRate)
            RESTORE_BUILT_IN(LAST_VALUE_TIME_OLD_TAG, m_LastValueTime)
            RESTORE_BUILT_IN(LAST_PROPAGATION_TIME_OLD_TAG, m_LastPropagationTime)
            RESTORE(CALENDAR_CYCLIC_TEST_OLD_TAG,
                    traverser.traverseSubLevel(boost::bind(&CCalendarTest::acceptRestoreTraverser,
                                                           &m_CalendarCyclicTest, _1)))
            RESTORE(COMPONENTS_OLD_TAG,
                    traverser.traverseSubLevel(
                        boost::bind(&CComponents::acceptRestoreTraverser, &m_Components,
                                    boost::cref(params), m_LastValueTime, _1)))
        } while (traverser.next());
        this->decayRate(decayRate);
    }
    return true;
}

void CTimeSeriesDecomposition::swap(CTimeSeriesDecomposition& other) {
    std::swap(m_TimeShift, other.m_TimeShift);
    std::swap(m_LastValueTime, other.m_LastValueTime);
    std::swap(m_LastPropagationTime, other.m_LastPropagationTime);
    m_PeriodicityTest.swap(other.m_PeriodicityTest);
    m_CalendarCyclicTest.swap(other.m_CalendarCyclicTest);
    m_Components.swap(other.m_Components);
}

CTimeSeriesDecomposition& CTimeSeriesDecomposition::
operator=(const CTimeSeriesDecomposition& other) {
    if (this != &other) {
        CTimeSeriesDecomposition copy{other};
        this->swap(copy);
    }
    return *this;
}

void CTimeSeriesDecomposition::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(VERSION_6_3_TAG, "");
    inserter.insertValue(TIME_SHIFT_6_3_TAG, m_TimeShift);
    inserter.insertValue(LAST_VALUE_TIME_6_3_TAG, m_LastValueTime);
    inserter.insertValue(LAST_PROPAGATION_TIME_6_3_TAG, m_LastPropagationTime);
    inserter.insertLevel(PERIODICITY_TEST_6_3_TAG,
                         boost::bind(&CPeriodicityTest::acceptPersistInserter,
                                     &m_PeriodicityTest, _1));
    inserter.insertLevel(CALENDAR_CYCLIC_TEST_6_3_TAG,
                         boost::bind(&CCalendarTest::acceptPersistInserter,
                                     &m_CalendarCyclicTest, _1));
    inserter.insertLevel(COMPONENTS_6_3_TAG, boost::bind(&CComponents::acceptPersistInserter,
                                                         &m_Components, _1));
}

CTimeSeriesDecomposition* CTimeSeriesDecomposition::clone(bool isForForecast) const {
    return new CTimeSeriesDecomposition{*this, isForForecast};
}

void CTimeSeriesDecomposition::dataType(maths_t::EDataType dataType) {
    m_Components.dataType(dataType);
}

void CTimeSeriesDecomposition::decayRate(double decayRate) {
    // Periodic component tests use a fixed decay rate.
    m_Components.decayRate(decayRate);
}

double CTimeSeriesDecomposition::decayRate() const {
    return m_Components.decayRate();
}

bool CTimeSeriesDecomposition::initialized() const {
    return m_Components.initialized();
}

bool CTimeSeriesDecomposition::addPoint(core_t::TTime time,
                                        double value,
                                        const maths_t::TDoubleWeightsAry& weights) {
    CComponents::CScopeNotifyOnStateChange result{m_Components};

    time += m_TimeShift;

    core_t::TTime lastTime{std::max(m_LastValueTime, m_LastPropagationTime)};

    m_LastValueTime = std::max(m_LastValueTime, time);
    this->propagateForwardsTo(time);

    SAddValue message{time,
                      lastTime,
                      value,
                      weights,
                      CBasicStatistics::mean(this->value(time, 0.0, E_TrendForced)),
                      CBasicStatistics::mean(this->value(time, 0.0, E_Seasonal)),
                      CBasicStatistics::mean(this->value(time, 0.0, E_Calendar)),
                      [this](core_t::TTime time_) {
                          return CBasicStatistics::mean(
                              this->value(time_, 0.0, E_Seasonal | E_Calendar));
                      },
                      m_Components.periodicityTestConfig()};

    m_Components.handle(message);
    m_PeriodicityTest.handle(message);
    m_CalendarCyclicTest.handle(message);

    return result.changed();
}

bool CTimeSeriesDecomposition::applyChange(core_t::TTime time,
                                           double value,
                                           const SChangeDescription& change) {
    bool result{m_Components.usingTrendForPrediction() == false};
    m_Components.useTrendForPrediction();

    switch (change.s_Description) {
    case SChangeDescription::E_LevelShift:
        m_Components.shiftLevel(time, value, change.s_Value[0]);
        break;
    case SChangeDescription::E_LinearScale:
        m_Components.linearScale(time, change.s_Value[0]);
        break;
    case SChangeDescription::E_TimeShift: {
        core_t::TTime dt{static_cast<core_t::TTime>(change.s_Value[0])};
        m_PeriodicityTest.shiftTime(dt);
        m_TimeShift += dt;
        break;
    }
    }

    return result;
}

void CTimeSeriesDecomposition::propagateForwardsTo(core_t::TTime time) {
    if (time > m_LastPropagationTime) {
        m_PeriodicityTest.propagateForwards(m_LastPropagationTime, time);
        m_CalendarCyclicTest.propagateForwards(m_LastPropagationTime, time);
        m_Components.propagateForwards(m_LastPropagationTime, time);
    }
    m_LastPropagationTime = std::max(m_LastPropagationTime, time);
}

double CTimeSeriesDecomposition::meanValue(core_t::TTime time) const {
    return m_Components.meanValue(time);
}

TDoubleDoublePr CTimeSeriesDecomposition::value(core_t::TTime time,
                                                double confidence,
                                                int components,
                                                bool smooth) const {
    TVector2x1 baseline{0.0};

    time += m_TimeShift;

    if (components & E_TrendForced) {
        baseline += vector2x1(m_Components.trend().value(time, confidence));
    } else if (components & E_Trend) {
        if (m_Components.usingTrendForPrediction()) {
            baseline += vector2x1(m_Components.trend().value(time, confidence));
        }
    }

    if (components & E_Seasonal) {
        for (const auto& component : m_Components.seasonal()) {
            if (this->selected(time, components, component)) {
                baseline += vector2x1(component.value(time, confidence));
            }
        }
    }

    if (components & E_Calendar) {
        for (const auto& component : m_Components.calendar()) {
            if (component.initialized() && component.feature().inWindow(time)) {
                baseline += vector2x1(component.value(time, confidence));
            }
        }
    }

    if (smooth) {
        baseline += vector2x1(
            this->smooth(boost::bind(&CTimeSeriesDecomposition::value, this, _1,
                                     confidence, components & E_Seasonal, false),
                         time - m_TimeShift, components));
    }

    return pair(baseline);
}

void CTimeSeriesDecomposition::forecast(core_t::TTime startTime,
                                        core_t::TTime endTime,
                                        core_t::TTime step,
                                        double confidence,
                                        double minimumScale,
                                        const TWriteForecastResult& writer) {
    if (endTime < startTime) {
        LOG_ERROR(<< "Bad forecast range: [" << startTime << "," << endTime << "]");
        return;
    }
    if (confidence < 0.0 || confidence >= 100.0) {
        LOG_ERROR(<< "Bad confidence interval: " << confidence << "%");
        return;
    }

    auto seasonal = [this, confidence](core_t::TTime time) {
        TVector2x1 prediction(0.0);
        for (const auto& component : m_Components.seasonal()) {
            if (component.initialized() && component.time().inWindow(time)) {
                prediction += vector2x1(component.value(time, confidence));
            }
        }
        for (const auto& component : m_Components.calendar()) {
            if (component.initialized() && component.feature().inWindow(time)) {
                prediction += vector2x1(component.value(time, confidence));
            }
        }
        return pair(prediction);
    };

    startTime += m_TimeShift;
    endTime += m_TimeShift;
    endTime = startTime + CIntegerTools::ceil(endTime - startTime, step);

    double trendVariance{CBasicStatistics::mean(m_Components.trend().variance(0.0))};
    double seasonalVariance{m_Components.meanVariance() - trendVariance};
    double variance{this->meanVariance()};
    double scale0{std::sqrt(std::max(
        CBasicStatistics::mean(this->scale(startTime, variance, 0.0)), minimumScale))};
    TVector2x1 i0{vector2x1(confidenceInterval(confidence, seasonalVariance))};

    auto forecastSeasonal = [&](core_t::TTime time) {
        m_Components.interpolateForForecast(time);
        double scale{std::sqrt(std::max(
            CBasicStatistics::mean(this->scale(time, variance, 0.0)), minimumScale))};
        TVector2x1 prediction{vector2x1(seasonal(time)) +
                              vector2x1(this->smooth(seasonal, time, E_Seasonal)) +
                              (scale - scale0) * i0};
        return TDouble3Vec{prediction(0), (prediction(0) + prediction(1)) / 2.0,
                           prediction(1)};
    };

    m_Components.trend().forecast(startTime, endTime, step, confidence,
                                  forecastSeasonal, writer);
}

double CTimeSeriesDecomposition::detrend(core_t::TTime time,
                                         double value,
                                         double confidence,
                                         int components) const {
    if (!this->initialized()) {
        return value;
    }
    TDoubleDoublePr interval{this->value(time, confidence, components)};
    return std::min(value - interval.first, 0.0) + std::max(value - interval.second, 0.0);
}

double CTimeSeriesDecomposition::meanVariance() const {
    return m_Components.meanVarianceScale() * m_Components.meanVariance();
}

TDoubleDoublePr CTimeSeriesDecomposition::scale(core_t::TTime time,
                                                double variance,
                                                double confidence,
                                                bool smooth) const {
    if (!this->initialized()) {
        return {1.0, 1.0};
    }

    double mean{this->meanVariance()};
    if (mean == 0.0) {
        return {1.0, 1.0};
    }

    time += m_TimeShift;

    double components{0.0};
    TVector2x1 scale(0.0);
    if (m_Components.usingTrendForPrediction()) {
        scale += vector2x1(m_Components.trend().variance(confidence));
    }
    for (const auto& component : m_Components.seasonal()) {
        if (component.initialized() && component.time().inWindow(time)) {
            scale += vector2x1(component.variance(time, confidence));
            components += 1.0;
        }
    }
    for (const auto& component : m_Components.calendar()) {
        if (component.initialized() && component.feature().inWindow(time)) {
            scale += vector2x1(component.variance(time, confidence));
            components += 1.0;
        }
    }

    double bias{std::min(2.0 * mean / variance, 1.0)};
    if (m_Components.usingTrendForPrediction()) {
        bias *= (components + 1.0) / std::max(components, 1.0);
    }
    LOG_TRACE(<< "mean = " << mean << " variance = " << variance << " bias = " << bias
              << " scale = " << core::CContainerPrinter::print(scale));

    scale *= m_Components.meanVarianceScale() / mean;
    scale = TVector2x1{1.0} + bias * (scale - TVector2x1{1.0});

    if (smooth) {
        scale += vector2x1(this->smooth(boost::bind(&CTimeSeriesDecomposition::scale, this,
                                                    _1, variance, confidence, false),
                                        time, E_All));
    }

    return pair(scale);
}

bool CTimeSeriesDecomposition::mightAddComponents(core_t::TTime time) const {
    for (auto test : {CPeriodicityTest::E_Short, CPeriodicityTest::E_Long}) {
        if (m_PeriodicityTest.shouldTest(test, time)) {
            return true;
        }
    }
    return false;
}

CTimeSeriesDecomposition::TTimeDoublePrVec CTimeSeriesDecomposition::windowValues() const {
    return m_PeriodicityTest.windowValues();
}

void CTimeSeriesDecomposition::skipTime(core_t::TTime skipInterval) {
    m_LastValueTime += skipInterval;
    m_LastPropagationTime += skipInterval;
}

uint64_t CTimeSeriesDecomposition::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_LastValueTime);
    seed = CChecksum::calculate(seed, m_LastPropagationTime);
    seed = CChecksum::calculate(seed, m_PeriodicityTest);
    seed = CChecksum::calculate(seed, m_CalendarCyclicTest);
    return CChecksum::calculate(seed, m_Components);
}

void CTimeSeriesDecomposition::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CTimeSeriesDecomposition");
    core::CMemoryDebug::dynamicSize("m_Mediator", m_Mediator, mem);
    core::CMemoryDebug::dynamicSize("m_PeriodicityTest", m_PeriodicityTest, mem);
    core::CMemoryDebug::dynamicSize("m_CalendarCyclicTest", m_CalendarCyclicTest, mem);
    core::CMemoryDebug::dynamicSize("m_Components", m_Components, mem);
}

std::size_t CTimeSeriesDecomposition::memoryUsage() const {
    return core::CMemory::dynamicSize(m_Mediator) +
           core::CMemory::dynamicSize(m_PeriodicityTest) +
           core::CMemory::dynamicSize(m_CalendarCyclicTest) +
           core::CMemory::dynamicSize(m_Components);
}

std::size_t CTimeSeriesDecomposition::staticSize() const {
    return sizeof(*this);
}

core_t::TTime CTimeSeriesDecomposition::timeShift() const {
    return m_TimeShift;
}

const maths_t::TSeasonalComponentVec& CTimeSeriesDecomposition::seasonalComponents() const {
    return m_Components.seasonal();
}

void CTimeSeriesDecomposition::initializeMediator() {
    m_Mediator = std::make_shared<CMediator>();
    m_Mediator->registerHandler(m_PeriodicityTest);
    m_Mediator->registerHandler(m_CalendarCyclicTest);
    m_Mediator->registerHandler(m_Components);
}

template<typename F>
TDoubleDoublePr
CTimeSeriesDecomposition::smooth(const F& f, core_t::TTime time, int components) const {
    auto offset = [&f, time](core_t::TTime discontinuity) {
        TVector2x1 baselineMinusEps{vector2x1(f(discontinuity - 1))};
        TVector2x1 baselinePlusEps{vector2x1(f(discontinuity + 1))};
        return 0.5 *
               (1.0 - static_cast<double>(std::abs(time - discontinuity)) /
                          static_cast<double>(SMOOTHING_INTERVAL)) *
               (baselinePlusEps - baselineMinusEps);
    };

    for (const auto& component : m_Components.seasonal()) {
        if (!component.initialized() || !this->matches(components, component) ||
            component.time().windowRepeat() <= SMOOTHING_INTERVAL) {
            continue;
        }

        const CSeasonalTime& times{component.time()};

        bool timeInWindow{times.inWindow(time)};
        bool inWindowBefore{times.inWindow(time - SMOOTHING_INTERVAL)};
        bool inWindowAfter{times.inWindow(time + SMOOTHING_INTERVAL)};
        if ((!timeInWindow && inWindowBefore) ||
            (timeInWindow && inWindowBefore &&
             times.startOfWindow(time) != times.startOfWindow(time + SMOOTHING_INTERVAL))) {
            core_t::TTime discontinuity{times.startOfWindow(time - SMOOTHING_INTERVAL) +
                                        times.windowLength()};
            return pair(-offset(discontinuity));
        }
        if ((!timeInWindow && inWindowAfter) ||
            (timeInWindow && inWindowAfter &&
             times.startOfWindow(time) != times.startOfWindow(time + SMOOTHING_INTERVAL))) {
            core_t::TTime discontinuity{component.time().startOfWindow(time + SMOOTHING_INTERVAL)};
            return pair(offset(discontinuity));
        }
    }

    return {0.0, 0.0};
}

bool CTimeSeriesDecomposition::selected(core_t::TTime time,
                                        int components,
                                        const CSeasonalComponent& component) const {
    return component.initialized() && this->matches(components, component) &&
           component.time().inWindow(time);
}

bool CTimeSeriesDecomposition::matches(int components, const CSeasonalComponent& component) const {
    int seasonal{components & E_Seasonal};
    if (seasonal == E_Seasonal) {
        return true;
    }
    core_t::TTime period{component.time().period()};
    bool diurnal{(period % core::constants::DAY == 0) ||
                 (period % core::constants::WEEK == 0)};
    return (seasonal == E_Diurnal && diurnal) || (seasonal == E_NonDiurnal && !diurnal);
}

core_t::TTime CTimeSeriesDecomposition::lastValueTime() const {
    return m_LastValueTime;
}

const core_t::TTime CTimeSeriesDecomposition::SMOOTHING_INTERVAL{7200};
}
}
