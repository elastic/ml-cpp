/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CTimeSeriesDecomposition.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/Constants.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CCalendarComponent.h>
#include <maths/CChecksum.h>
#include <maths/CIntegerTools.h>
#include <maths/CSeasonalTime.h>
#include <maths/CTools.h>

#include <boost/bind.hpp>
#include <boost/container/flat_map.hpp>
#include <boost/make_shared.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/random/normal_distribution.hpp>

#include <cmath>
#include <string>
#include <utility>

namespace ml
{
namespace maths
{
namespace
{

using TDoubleDoublePr = maths_t::TDoubleDoublePr;
using TVector2x1 = CVectorNx1<double, 2>;

//! Convert a double pair to a 2x1 vector.
TVector2x1 vector2x1(const TDoubleDoublePr &p)
{
    TVector2x1 result;
    result(0) = p.first;
    result(1) = p.second;
    return result;
}

//! Convert a 2x1 vector to a double pair.
TDoubleDoublePr pair(const TVector2x1 &v)
{
    return std::make_pair(v(0), v(1));
}

//! Update the confidence interval to reflect the initial errors
//! in the regression coefficients.
template<std::size_t N>
void forecastErrors(double varianceDueToParameterDrift,
                    const CSymmetricMatrixNxN<double, N> &m,
                    double dt, double confidence, TVector2x1 &result)
{
    double variance{varianceDueToParameterDrift};
    double ti{dt};
    for (std::size_t i = 1; dt > 0.0 && i < m.rows(); ++i, ti *= dt)
    {
        if (m(i,i) == 0.0)
        {
            LOG_TRACE("Ignoring t^" << i << " as variance of coefficient is zero");
            continue;
        }
        variance += m(i,i) * ti * ti;
    }
    try
    {
        if (variance > 0.0)
        {
            boost::math::normal normal{0.0, std::sqrt(variance)};
            result(0) += boost::math::quantile(normal, (100.0 - confidence) / 200.0);
            result(1) += boost::math::quantile(normal, (100.0 + confidence) / 200.0);
        }
        else
        {
            result(0) = result(1) = 0.0;
        }
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Unable to compute error: " << e.what());
    }
}

const std::string DECAY_RATE_TAG{"a"};
const std::string LAST_VALUE_TIME_TAG{"b"};
const std::string LONG_TERM_TREND_TEST_TAG{"c"};
const std::string DIURNAL_TEST_TAG{"d"};
const std::string GENERAL_SEASONALITY_TEST_TAG{"e"};
const std::string CALENDAR_CYCLIC_TEST_TAG{"f"};
const std::string SEASONAL_COMPONENTS_TAG{"g"};
const std::string LAST_PROPAGATION_TIME_TAG{"h"};
const std::string EMPTY_STRING;

}

CTimeSeriesDecomposition::CTimeSeriesDecomposition(double decayRate,
                                                   core_t::TTime bucketLength,
                                                   std::size_t seasonalComponentSize) :
        m_DecayRate{decayRate},
        m_LastValueTime{0},
        m_LastPropagationTime{0},
        m_LongTermTrendTest{decayRate},
        m_DiurnalTest{decayRate, bucketLength},
        m_GeneralSeasonalityTest{decayRate, bucketLength},
        m_CalendarCyclicTest{decayRate, bucketLength},
        m_Components{decayRate, bucketLength, seasonalComponentSize}
{
    this->initializeMediator();
}

CTimeSeriesDecomposition::CTimeSeriesDecomposition(double decayRate,
                                                   core_t::TTime bucketLength,
                                                   std::size_t seasonalComponentSize,
                                                   core::CStateRestoreTraverser &traverser) :
        m_DecayRate{decayRate},
        m_LastValueTime{0},
        m_LastPropagationTime{0},
        m_LongTermTrendTest{decayRate},
        m_DiurnalTest{decayRate, bucketLength},
        m_GeneralSeasonalityTest{decayRate, bucketLength},
        m_CalendarCyclicTest{decayRate, bucketLength},
        m_Components{decayRate, bucketLength, seasonalComponentSize}
{
    traverser.traverseSubLevel(boost::bind(&CTimeSeriesDecomposition::acceptRestoreTraverser, this, _1));
    this->initializeMediator();
}

CTimeSeriesDecomposition::CTimeSeriesDecomposition(const CTimeSeriesDecomposition &other) :
        m_DecayRate{other.m_DecayRate},
        m_LastValueTime{other.m_LastValueTime},
        m_LastPropagationTime{other.m_LastPropagationTime},
        m_LongTermTrendTest{other.m_LongTermTrendTest},
        m_DiurnalTest{other.m_DiurnalTest},
        m_GeneralSeasonalityTest{other.m_GeneralSeasonalityTest},
        m_CalendarCyclicTest{other.m_CalendarCyclicTest},
        m_Components{other.m_Components}
{
    this->initializeMediator();
}

bool CTimeSeriesDecomposition::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser)
{
    do
    {
        const std::string &name{traverser.name()};
        RESTORE_BUILT_IN(DECAY_RATE_TAG, m_DecayRate);
        RESTORE_BUILT_IN(LAST_VALUE_TIME_TAG, m_LastValueTime)
        RESTORE_BUILT_IN(LAST_PROPAGATION_TIME_TAG, m_LastPropagationTime)
        RESTORE(LONG_TERM_TREND_TEST_TAG, traverser.traverseSubLevel(
                                              boost::bind(&CLongTermTrendTest::acceptRestoreTraverser,
                                                          &m_LongTermTrendTest, _1)));
        RESTORE(DIURNAL_TEST_TAG, traverser.traverseSubLevel(
                                      boost::bind(&CDiurnalTest::acceptRestoreTraverser,
                                                  &m_DiurnalTest, _1)));
        RESTORE(GENERAL_SEASONALITY_TEST_TAG, traverser.traverseSubLevel(
                                                  boost::bind(&CNonDiurnalTest::acceptRestoreTraverser,
                                                              &m_GeneralSeasonalityTest, _1)))
        RESTORE(CALENDAR_CYCLIC_TEST_TAG, traverser.traverseSubLevel(
                                              boost::bind(&CCalendarTest::acceptRestoreTraverser,
                                                          &m_CalendarCyclicTest, _1)))
        RESTORE(SEASONAL_COMPONENTS_TAG, traverser.traverseSubLevel(
                                             boost::bind(&CComponents::acceptRestoreTraverser,
                                                         &m_Components, _1)))
    }
    while (traverser.next());

    m_LongTermTrendTest.decayRate(m_DecayRate);
    m_Components.decayRate(m_DecayRate);
    if (m_LastPropagationTime == 0)
    {
        m_LastPropagationTime = m_LastValueTime;
    }

    return true;
}

void CTimeSeriesDecomposition::swap(CTimeSeriesDecomposition &other)
{
    std::swap(m_DecayRate, other.m_DecayRate);
    std::swap(m_LastValueTime, other.m_LastValueTime);
    std::swap(m_LastPropagationTime, other.m_LastPropagationTime);
    m_LongTermTrendTest.swap(other.m_LongTermTrendTest);
    m_DiurnalTest.swap(other.m_DiurnalTest);
    m_GeneralSeasonalityTest.swap(other.m_GeneralSeasonalityTest);
    m_CalendarCyclicTest.swap(other.m_CalendarCyclicTest);
    m_Components.swap(other.m_Components);
}

CTimeSeriesDecomposition &CTimeSeriesDecomposition::operator=(const CTimeSeriesDecomposition &other)
{
    if (this != &other)
    {
        CTimeSeriesDecomposition copy{other};
        this->swap(copy);
    }
    return *this;
}

void CTimeSeriesDecomposition::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertValue(DECAY_RATE_TAG, this->decayRate(), core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(LAST_VALUE_TIME_TAG, m_LastValueTime);
    inserter.insertValue(LAST_PROPAGATION_TIME_TAG, m_LastPropagationTime);
    inserter.insertLevel(LONG_TERM_TREND_TEST_TAG, boost::bind(&CLongTermTrendTest::acceptPersistInserter,
                                                               &m_LongTermTrendTest, _1));
    inserter.insertLevel(DIURNAL_TEST_TAG, boost::bind(&CDiurnalTest::acceptPersistInserter,
                                                       &m_DiurnalTest, _1));
    inserter.insertLevel(GENERAL_SEASONALITY_TEST_TAG, boost::bind(&CNonDiurnalTest::acceptPersistInserter,
                                                                   &m_GeneralSeasonalityTest, _1));
    inserter.insertLevel(CALENDAR_CYCLIC_TEST_TAG, boost::bind(&CCalendarTest::acceptPersistInserter,
                                                               &m_CalendarCyclicTest, _1));
    inserter.insertLevel(SEASONAL_COMPONENTS_TAG, boost::bind(&CComponents::acceptPersistInserter,
                                                              &m_Components, _1));
}

CTimeSeriesDecomposition *CTimeSeriesDecomposition::clone(void) const
{
    return new CTimeSeriesDecomposition{*this};
}

void CTimeSeriesDecomposition::decayRate(double decayRate)
{
    // Periodic component tests use a fixed decay rate.
    m_DecayRate = decayRate;
    m_LongTermTrendTest.decayRate(decayRate);
    m_Components.decayRate(decayRate);
}

double CTimeSeriesDecomposition::decayRate(void) const
{
    return m_DecayRate;
}

void CTimeSeriesDecomposition::forForecasting(void)
{
    m_Components.forecast();
}

bool CTimeSeriesDecomposition::initialized(void) const
{
    return m_Components.initialized();
}

bool CTimeSeriesDecomposition::addPoint(core_t::TTime time,
                                        double value,
                                        const maths_t::TWeightStyleVec &weightStyles,
                                        const maths_t::TDouble4Vec &weights)
{
    CComponents::CScopeNotifyOnStateChange result{m_Components};

    core_t::TTime lastTime{std::max(m_LastValueTime, m_LastPropagationTime)};

    m_LastValueTime = std::max(m_LastValueTime, time);
    this->propagateForwardsTo(time);

    SAddValue message{time, lastTime, value, weightStyles, weights,
                      CBasicStatistics::mean(this->baseline(time, 0.0, 0.0, E_Trend)),
                      CBasicStatistics::mean(this->baseline(time, 0.0, 0.0, E_NonDiurnal)),
                      CBasicStatistics::mean(this->baseline(time, 0.0, 0.0, E_Seasonal)),
                      CBasicStatistics::mean(this->baseline(time, 0.0, 0.0, E_Calendar))};

    m_Components.handle(message);

    m_LongTermTrendTest.handle(message);
    if (result.changed())
    {
        message.s_Trend = CBasicStatistics::mean(this->baseline(time, 0.0, 0.0, E_Trend));
    }

    m_DiurnalTest.handle(message);
    if (result.changed())
    {
        message.s_Seasonal = CBasicStatistics::mean(this->baseline(time, 0.0, 0.0, E_Seasonal));
    }

    m_GeneralSeasonalityTest.handle(message);
    if (result.changed())
    {
        message.s_NonDiurnal = CBasicStatistics::mean(this->baseline(time, 0.0, 0.0, E_NonDiurnal));
    }

    m_CalendarCyclicTest.handle(message);

    return result.changed();
}

void CTimeSeriesDecomposition::propagateForwardsTo(core_t::TTime time)
{
    if (time > m_LastPropagationTime)
    {
        if (!this->forecasting())
        {
            m_LongTermTrendTest.propagateForwards(m_LastPropagationTime, time);
            m_DiurnalTest.propagateForwards(m_LastPropagationTime, time);
            m_CalendarCyclicTest.propagateForwards(m_LastPropagationTime, time);
        }
        m_Components.propagateForwards(m_LastPropagationTime, time);
    }
    m_LastPropagationTime = std::max(m_LastPropagationTime, time);
}

bool CTimeSeriesDecomposition::testAndInterpolate(core_t::TTime time)
{
    CComponents::CScopeNotifyOnStateChange result(m_Components);

    SMessage message{time, std::max(m_LastValueTime, m_LastPropagationTime)};
    if (!this->forecasting())
    {
        m_LongTermTrendTest.test(message);
        m_DiurnalTest.test(message);
        m_GeneralSeasonalityTest.test(message);
        m_CalendarCyclicTest.test(message);
    }
    m_Components.interpolate(message);

    return result.changed();
}

double CTimeSeriesDecomposition::mean(core_t::TTime time) const
{
    return m_Components.meanValue(time);
}

TDoubleDoublePr CTimeSeriesDecomposition::baseline(core_t::TTime time,
                                                   double predictionConfidence,
                                                   double forecastConfidence,
                                                   EComponents components,
                                                   bool smooth) const
{
    if (!this->initialized())
    {
        return {0.0, 0.0};
    }

    TVector2x1 baseline{0.0};

    if (components & E_Trend)
    {
        CTrendCRef trend = m_Components.trend();

        baseline += vector2x1(trend.prediction(time, predictionConfidence));

        if (this->forecasting())
        {
            CTrendCRef::TMatrix m;
            if (trend.covariances(m))
            {
                double dt{std::max(trend.time(time) - trend.time(m_LastValueTime), 0.0)};
                forecastErrors(trend.varianceDueToParameterDrift(time),
                               m, dt, forecastConfidence, baseline);
            }
        }
    }

    if (components & E_Seasonal)
    {
        for (const auto &component : m_Components.seasonal())
        {
            if (this->selected(time, components, component))
            {
                baseline += vector2x1(component.value(time, predictionConfidence));
                if (this->forecasting())
                {
                    CSeasonalComponent::TMatrix m;
                    if (component.covariances(time, m))
                    {
                        const CSeasonalTime &seasonalTime{component.time()};
                        double dt{std::max(  seasonalTime.regression(time)
                                           - seasonalTime.regression(m_LastValueTime), 0.0)};
                        forecastErrors(component.varianceDueToParameterDrift(time),
                                       m, dt, forecastConfidence, baseline);
                    }
                }
            }
        }
    }

    if (components & E_Calendar)
    {
        for (const auto &component : m_Components.calendar())
        {
            if (component.feature().inWindow(time))
            {
                baseline += vector2x1(component.value(time, predictionConfidence));
            }
        }
    }

    if (smooth)
    {
        baseline += vector2x1(this->smooth(
                boost::bind(&CTimeSeriesDecomposition::baseline,
                            this, _1, predictionConfidence,
                            forecastConfidence, E_All, false), time, components));
    }

    return pair(baseline);
}

double CTimeSeriesDecomposition::detrend(core_t::TTime time, double value, double confidence) const
{
    if (!this->initialized())
    {
        return value;
    }
    TDoubleDoublePr baseline{this->baseline(time, confidence)};
    return std::min(value - baseline.first, 0.0) + std::max(value - baseline.second, 0.0);
}

double CTimeSeriesDecomposition::meanVariance(void) const
{
    return m_Components.meanVariance();
}

TDoubleDoublePr CTimeSeriesDecomposition::scale(core_t::TTime time,
                                                double variance,
                                                double confidence,
                                                bool smooth) const
{
    if (!this->initialized())
    {
        return {1.0, 1.0};
    }

    double mean{this->meanVariance()};
    if (mean == 0.0)
    {
        return {1.0, 1.0};
    }

    TVector2x1 scale{m_Components.trend().variance()};
    for (const auto &component : m_Components.seasonal())
    {
        if (component.initialized() && component.time().inWindow(time))
        {
            scale += vector2x1(component.variance(time, confidence));
        }
    }
    for (const auto &component : m_Components.calendar())
    {
        if (component.initialized() && component.feature().inWindow(time))
        {
            scale += vector2x1(component.variance(time, confidence));
        }
    }
    LOG_TRACE("mean = " << mean << " variance = " << core::CContainerPrinter::print(scale));

    double bias{std::min(2.0 * mean / variance, 1.0)};
    scale /= mean;
    scale  = TVector2x1{1.0} + bias * (scale - TVector2x1{1.0});

    if (smooth)
    {
        scale += vector2x1(this->smooth(
                boost::bind(&CTimeSeriesDecomposition::scale,
                            this, _1, variance, confidence, false), time, E_All));
    }

    return pair(scale);
}

void CTimeSeriesDecomposition::skipTime(core_t::TTime skipInterval)
{
    m_LastValueTime += skipInterval;
    m_LastPropagationTime += skipInterval;
    m_LongTermTrendTest.skipTime(skipInterval);
    m_DiurnalTest.skipTime(skipInterval);
    m_GeneralSeasonalityTest.skipTime(skipInterval);
    m_CalendarCyclicTest.advanceTimeTo(m_LastPropagationTime);
}

uint64_t CTimeSeriesDecomposition::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, m_DecayRate);
    seed = CChecksum::calculate(seed, m_LastValueTime);
    seed = CChecksum::calculate(seed, m_LastPropagationTime);
    seed = CChecksum::calculate(seed, m_LongTermTrendTest);
    seed = CChecksum::calculate(seed, m_DiurnalTest);
    seed = CChecksum::calculate(seed, m_GeneralSeasonalityTest);
    seed = CChecksum::calculate(seed, m_CalendarCyclicTest);
    return CChecksum::calculate(seed, m_Components);
}

void CTimeSeriesDecomposition::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    mem->setName("CTimeSeriesDecomposition");
    core::CMemoryDebug::dynamicSize("m_Mediator", m_Mediator, mem);
    core::CMemoryDebug::dynamicSize("m_LongTermTrendTest", m_LongTermTrendTest, mem);
    core::CMemoryDebug::dynamicSize("m_DiurnalTest", m_DiurnalTest, mem);
    core::CMemoryDebug::dynamicSize("m_GeneralSeasonalityTest", m_GeneralSeasonalityTest, mem);
    core::CMemoryDebug::dynamicSize("m_CalendarCyclicTest", m_CalendarCyclicTest, mem);
    core::CMemoryDebug::dynamicSize("m_Components", m_Components, mem);
}

std::size_t CTimeSeriesDecomposition::memoryUsage(void) const
{
    return  core::CMemory::dynamicSize(m_Mediator)
          + core::CMemory::dynamicSize(m_LongTermTrendTest)
          + core::CMemory::dynamicSize(m_DiurnalTest)
          + core::CMemory::dynamicSize(m_GeneralSeasonalityTest)
          + core::CMemory::dynamicSize(m_CalendarCyclicTest)
          + core::CMemory::dynamicSize(m_Components);
}

std::size_t CTimeSeriesDecomposition::staticSize(void) const
{
    return sizeof(*this);
}

const maths_t::TSeasonalComponentVec &CTimeSeriesDecomposition::seasonalComponents(void) const
{
    return m_Components.seasonal();
}

void CTimeSeriesDecomposition::initializeMediator(void)
{
    m_Mediator = boost::make_shared<CMediator>();
    m_Mediator->registerHandler(m_LongTermTrendTest);
    m_Mediator->registerHandler(m_DiurnalTest);
    m_Mediator->registerHandler(m_GeneralSeasonalityTest);
    m_Mediator->registerHandler(m_CalendarCyclicTest);
    m_Mediator->registerHandler(m_Components);
}

bool CTimeSeriesDecomposition::forecasting(void) const
{
    return m_Components.forecasting();
}

template<typename F>
TDoubleDoublePr CTimeSeriesDecomposition::smooth(const F &f,
                                                 core_t::TTime time,
                                                 EComponents components) const
{
    auto offset = [&f, time](core_t::TTime discontinuity)
                  {
                      TVector2x1 baselineMinusEps{vector2x1(f(discontinuity - 1))};
                      TVector2x1 baselinePlusEps{ vector2x1(f(discontinuity + 1))};
                      return 0.5 * (1.0 - static_cast<double>(std::abs(time - discontinuity))
                                        / static_cast<double>(SMOOTHING_INTERVAL))
                                 * (baselinePlusEps - baselineMinusEps);
                  };

    for (const auto &component : m_Components.seasonal())
    {
        if (   !component.initialized()
            || !this->matches(components, component)
            || component.time().windowRepeat() <= SMOOTHING_INTERVAL)
        {
            continue;
        }

        const CSeasonalTime &times{component.time()};

        bool timeInWindow{times.inWindow(time)};
        bool inWindowBefore{times.inWindow(time - SMOOTHING_INTERVAL)};
        bool inWindowAfter{times.inWindow(time + SMOOTHING_INTERVAL)};
        if (  (!timeInWindow && inWindowBefore)
            || (timeInWindow && inWindowBefore && times.startOfWindow(time) !=
                                                  times.startOfWindow(time + SMOOTHING_INTERVAL)))
        {
            core_t::TTime discontinuity{  times.startOfWindow(time - SMOOTHING_INTERVAL)
                                        + times.windowLength()};
            return pair(-offset(discontinuity));
        }
        if (  (!timeInWindow && inWindowAfter)
            || (timeInWindow && inWindowAfter && times.startOfWindow(time) !=
                                                 times.startOfWindow(time + SMOOTHING_INTERVAL)))
        {
            core_t::TTime discontinuity{component.time().startOfWindow(time + SMOOTHING_INTERVAL)};
            return pair(offset(discontinuity));
        }
    }

    return {0.0, 0.0};
}

bool CTimeSeriesDecomposition::selected(core_t::TTime time,
                                        EComponents components,
                                        const CSeasonalComponent &component) const
{
    return   component.initialized()
          && this->matches(components, component)
          && component.time().inWindow(time);
}

bool CTimeSeriesDecomposition::matches(EComponents components,
                                       const CSeasonalComponent &component) const
{
    int seasonal{components & E_Seasonal};
    if (seasonal == E_Seasonal)
    {
        return true;
    }
    core_t::TTime period{component.time().period()};
    bool diurnal{(period % core::constants::DAY == 0) || (period % core::constants::WEEK == 0)};
    return (seasonal == E_Diurnal && diurnal) || (seasonal == E_NonDiurnal && !diurnal);
}

core_t::TTime CTimeSeriesDecomposition::lastValueTime(void) const
{
    return m_LastValueTime;
}

const core_t::TTime CTimeSeriesDecomposition::SMOOTHING_INTERVAL{7200};
const std::size_t CTimeSeriesDecomposition::DEFAULT_COMPONENT_SIZE{36u};

}
}
