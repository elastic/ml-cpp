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

#include <maths/CTimeSeriesDecompositionDetail.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/Constants.h>
#include <core/CPersistUtils.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CTimezone.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CCalendarComponent.h>
#include <maths/CChecksum.h>
#include <maths/CIntegerTools.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraPersist.h>
#include <maths/CRegressionDetail.h>
#include <maths/CSeasonalComponentAdaptiveBucketing.h>
#include <maths/CSeasonalTime.h>
#include <maths/CSetTools.h>

#include <boost/config.hpp>
#include <boost/bind.hpp>
#include <boost/config.hpp>
#include <boost/container/flat_map.hpp>
#include <boost/container/flat_set.hpp>
#include <boost/make_shared.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/range.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace ml
{
namespace maths
{
namespace
{

using TDoubleDoublePr = maths_t::TDoubleDoublePr;
using TSeasonalComponentVec = maths_t::TSeasonalComponentVec;
using TCalendarComponentVec = maths_t::TCalendarComponentVec;
using TBoolVec = std::vector<bool>;
using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;
using TSizeVecVec = std::vector<TSizeVec>;
using TStrVec = std::vector<std::string>;
using TTimeVec = std::vector<core_t::TTime>;
using TTimeTimePr = std::pair<core_t::TTime, core_t::TTime>;
using TTimeTimePrVec = std::vector<TTimeTimePr>;
using TTimeTimePrDoubleFMap = boost::container::flat_map<TTimeTimePr, double>;
using TTimeTimePrSizeFMap = boost::container::flat_map<TTimeTimePr, std::size_t>;
using TSeasonalComponentPtrVec = std::vector<CSeasonalComponent*>;
using TCalendarComponentPtrVec = std::vector<CCalendarComponent*>;
using TRegression = CTimeSeriesDecompositionDetail::TRegression;
using TTrendCRef = CTimeSeriesDecompositionDetail::CTrendCRef;

const core_t::TTime HOUR  = core::constants::HOUR;
const core_t::TTime DAY   = core::constants::DAY;
const core_t::TTime WEEK  = core::constants::WEEK;
const core_t::TTime MONTH = 4 * WEEK;

//! Get the square of \p x.
double pow2(double x)
{
    return x * x;
}

//! Check if a period is daily oe weekly.
bool isDiurnal(core_t::TTime period)
{
    return period % DAY == 0 || period % WEEK == 0;
}

//! Scale \p interval to account for \p bucketLength.
core_t::TTime scale(core_t::TTime interval, core_t::TTime bucketLength)
{
    return interval * std::max(bucketLength / HOUR, core_t::TTime(1));
}

//! Compute the mean of \p mean of \p components.
template<typename MEAN_FUNCTION>
double meanOf(MEAN_FUNCTION mean, const TSeasonalComponentVec &components)
{
    // We can choose to partition the trend model into windows.
    // In particular, we check for the presence of weekday/end
    // patterns. In this function we want to compute the sum of
    // the mean average of the different components: we use an
    // additive decomposition of the trend. However, if we have
    // detected a partition we want to average the models for
    // the different windows.

    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

    double unwindowed{0.0};
    TTimeTimePrDoubleFMap windows;
    windows.reserve(components.size());
    for (const auto &component : components)
    {
        if (component.initialized())
        {
            TTimeTimePr window{component.time().window()};
            if (window.second - window.first == component.time().windowRepeat())
            {
                unwindowed += (component.*mean)();
            }
            else
            {
                windows[window] += (component.*mean)();
            }
        }
    }

    TMeanAccumulator windowed;
    for (const auto &window : windows)
    {
        double weight{static_cast<double>(
                          window.first.second - window.first.first)};
        windowed.add(window.second, weight);
    }

    return unwindowed + CBasicStatistics::mean(windowed);
}

//! Compute the values to add to the trend and each component.
//!
//! \param[in] trend The long term trend.
//! \param[in] seasonal The seasonal components.
//! \param[in] calendar The calendar components.
//! \param[in] deltas The delta offset to apply to the difference
//! between each component value and its mean, used to minimize
//! slope in the longer periods.
//! \param[in] time The time of value to decompose.
//! \param[in,out] decomposition Updated to contain the value to
//! add to each by component.
void decompose(const TTrendCRef &trend,
               const TSeasonalComponentPtrVec &seasonal,
               const TCalendarComponentPtrVec &calendar,
               const TDoubleVec &deltas,
               core_t::TTime time,
               double &error,
               TDoubleVec &decomposition,
               TDoubleVec &predictions)
{
    std::size_t m{seasonal.size()};
    std::size_t n{calendar.size()};

    double x0{CBasicStatistics::mean(trend.prediction(time, 0.0))};
    TDoubleVec x(m + n);
    double xhat{x0};
    for (std::size_t i = 0u; i < m; ++i)
    {
        x[i]  = CBasicStatistics::mean(seasonal[i]->value(time, 0.0));
        xhat += x[i];
    }
    for (std::size_t i = m; i < m + n; ++i)
    {
        x[i]  = CBasicStatistics::mean(calendar[i - m]->value(time, 0.0));
        xhat += x[i];
    }

    double w0{trend.initialized() ? 1.0 : 0.0};
    double Z{static_cast<double>(m + n) + w0};

    error = decomposition[0] - xhat;
    decomposition[0] = x0 + (decomposition[0] - xhat) * w0 / Z;
    double lastDelta{0.0};
    for (std::size_t i = 0u; i < m; ++i)
    {
        double d{deltas[i] - lastDelta};
        lastDelta = deltas[i];
        predictions[i] = x[i] - seasonal[i]->meanValue();
        decomposition[i + 1] = x[i] + (decomposition[i + 1] - xhat) / Z + d;
    }
    for (std::size_t i = m; i < m + n; ++i)
    {
        predictions[i] = x[i] - calendar[i - m]->meanValue();
        decomposition[i + 1] = x[i] + (decomposition[i + 1] - xhat) / Z;
    }
}

//! Convert the propagation decay rate into the corresponding regular
//! periodicity test decay rate.
double regularTestDecayRate(double decayRate)
{
    return CSeasonalTime::scaleDecayRate(decayRate, DAY, WEEK);
}

//! Propagate a test forwards to account for \p end - \p start elapsed
//! time.
template<typename TEST>
void propagateTestForwards(core_t::TTime start,
                           core_t::TTime end,
                           core_t::TTime interval,
                           const TEST &test)
{
    if (test)
    {
        start = CIntegerTools::floor(start, interval);
        end   = CIntegerTools::floor(end, interval);
        if (end > start)
        {
            double time{static_cast<double>(end - start) / static_cast<double>(interval)};
            test->propagateForwardsByTime(time);
        }
    }
}

//! Get the time of \p time suitable for use in a trend regression model.
double regressionTime(core_t::TTime time, core_t::TTime origin)
{
    return static_cast<double>(time - origin) / static_cast<double>(WEEK);
}

//! Apply the common shift to the slope of \p trend.
void shiftSlope(const TTimeTimePrDoubleFMap &slopes, TRegression &trend)
{
    CBasicStatistics::CMinMax<double> minmax;
    for (const auto &slope : slopes)
    {
        minmax.add(slope.second);
    }
    double shift{minmax.signMargin()};
    if (shift != 0.0)
    {
        trend.shiftGradient(shift);
    }
}


// Long Term Trend Test State Machine

// States
const std::size_t LT_INITIAL     = 0;
const std::size_t LT_TEST        = 1;
const std::size_t LT_NOT_TESTING = 2;
const std::size_t LT_ERROR       = 3;
const TStrVec LT_STATES{"INITIAL", "TEST", "NOT_TESTING", "ERROR"};
// Alphabet
const std::size_t LT_NEW_VALUE   = 0;
const std::size_t LT_FINISH_TEST = 1;
const std::size_t LT_RESET       = 2;
const TStrVec LT_ALPHABET{"NEW_VALUE", "FINISH_TEST", "RESET"};
// Transition Function
const TSizeVecVec LT_TRANSITION_FUNCTION
    {
        TSizeVec{LT_TEST,        LT_TEST,        LT_NOT_TESTING, LT_ERROR  },
        TSizeVec{LT_NOT_TESTING, LT_NOT_TESTING, LT_NOT_TESTING, LT_ERROR  },
        TSizeVec{LT_INITIAL,     LT_INITIAL,     LT_INITIAL,     LT_INITIAL}
    };

// Diurnal Test State Machine

// States
const std::size_t DW_INITIAL      = 0;
const std::size_t DW_SMALL_TEST   = 1;
const std::size_t DW_REGULAR_TEST = 2;
const std::size_t DW_NOT_TESTING  = 3;
const std::size_t DW_ERROR        = 4;
const TStrVec DW_STATES{"INITIAL", "SMALL_TEST", "REGULAR_TEST", "NOT_TESTING", "ERROR"};
// Alphabet
const std::size_t DW_NEW_VALUE              = 0;
const std::size_t DW_SMALL_TEST_TRUE        = 1;
const std::size_t DW_REGULAR_TEST_TIMED_OUT = 2;
const std::size_t DW_FINISH_TEST            = 3;
const std::size_t DW_RESET                  = 4;
const TStrVec DW_ALPHABET{"NEW_VALUE", "SMALL_TEST_TRUE", "REGULAR_TEST_TIMED_OUT", "FINISH_TEST", "RESET"};
// Transition Function
const TSizeVecVec DW_TRANSITION_FUNCTION
    {
        TSizeVec{DW_REGULAR_TEST, DW_SMALL_TEST,   DW_REGULAR_TEST, DW_NOT_TESTING, DW_ERROR  },
        TSizeVec{DW_ERROR,        DW_REGULAR_TEST, DW_ERROR,        DW_NOT_TESTING, DW_ERROR  },
        TSizeVec{DW_ERROR,        DW_ERROR,        DW_SMALL_TEST,   DW_NOT_TESTING, DW_ERROR  },
        TSizeVec{DW_NOT_TESTING,  DW_NOT_TESTING,  DW_NOT_TESTING,  DW_NOT_TESTING, DW_ERROR  },
        TSizeVec{DW_INITIAL,      DW_INITIAL,      DW_INITIAL,      DW_NOT_TESTING, DW_INITIAL}
    };

// General Seasonality Test State Machine

// States
const std::size_t GS_INITIAL     = 0;
const std::size_t GS_TEST        = 1;
const std::size_t GS_NOT_TESTING = 2;
const std::size_t GS_ERROR       = 3;
const TStrVec GS_STATES{"INITIAL", "TEST", "NOT_TESTING", "ERROR" };
// Alphabet
const std::size_t GS_NEW_VALUE = 0;
const std::size_t GS_RESET     = 1;
const TStrVec GS_ALPHABET{"NEW_VALUE", "RESET"};
// Transition Function
const TSizeVecVec GS_TRANSITION_FUNCTION
    {
        TSizeVec{GS_TEST,    GS_TEST,    GS_NOT_TESTING, GS_ERROR  },
        TSizeVec{GS_INITIAL, GS_INITIAL, GS_NOT_TESTING, GS_INITIAL}
    };

// Calendar Cyclic Test State Machine

// States
const std::size_t CC_INITIAL     = 0;
const std::size_t CC_TEST        = 1;
const std::size_t CC_NOT_TESTING = 2;
const std::size_t CC_ERROR       = 3;
const TStrVec CC_STATES{"INITIAL", "TEST", "NOT_TESTING", "ERROR"};
// Alphabet
const std::size_t CC_NEW_VALUE = 0;
const std::size_t CC_RESET     = 1;
const TStrVec CC_ALPHABET{"NEW_VALUE", "RESET"};
// Transition Function
const TSizeVecVec CC_TRANSITION_FUNCTION
    {
        TSizeVec{CC_TEST,    CC_TEST,    CC_NOT_TESTING, CC_ERROR  },
        TSizeVec{CC_INITIAL, CC_INITIAL, CC_NOT_TESTING, CC_INITIAL}
    };

// Components State Machine

// States
const std::size_t SC_NEW_COMPONENTS = 0;
const std::size_t SC_NORMAL         = 1;
const std::size_t SC_FORECASTING    = 2;
const std::size_t SC_DISABLED       = 3;
const std::size_t SC_ERROR          = 4;
const TStrVec SC_STATES{"NEW_COMPONENTS", "NORMAL", "FORECASTING", "DISABLED", "ERROR"};
// Alphabet
const std::size_t SC_ADDED_COMPONENTS = 0;
const std::size_t SC_INTERPOLATED     = 1;
const std::size_t SC_FORECAST         = 2;
const std::size_t SC_RESET            = 3;
const TStrVec SC_ALPHABET{"ADDED_COMPONENTS", "INTERPOLATED", "FORECAST", "RESET"};
// Transition Function
const TSizeVecVec SC_TRANSITION_FUNCTION
    {
        TSizeVec{SC_NEW_COMPONENTS, SC_NEW_COMPONENTS, SC_ERROR,       SC_DISABLED, SC_ERROR },
        TSizeVec{SC_NORMAL,         SC_NORMAL,         SC_FORECASTING, SC_DISABLED, SC_ERROR },
        TSizeVec{SC_ERROR,          SC_FORECASTING,    SC_FORECASTING, SC_DISABLED, SC_ERROR },
        TSizeVec{SC_NORMAL,         SC_NORMAL,         SC_NORMAL,      SC_NORMAL,   SC_NORMAL}
    };

// Long Term Trend Test Tags
const std::string MACHINE_TAG{"a"};
const std::string NEXT_TEST_TIME_TAG{"b"};
const std::string TEST_TAG{"c"};

// Diurnal Test Tags
//const std::string MACHINE_TAG{"a"};
//const std::string NEXT_TEST_TIME_TAG{"b"};
const std::string STARTED_REGULAR_TEST_TAG{"c"};
const std::string TIME_OUT_REGULAR_TEST_TAG{"d"};
const std::string REGULAR_TEST_TAG{"e"};
const std::string SMALL_TEST_TAG{"f"};
const std::string PERIODS_TAG{"g"};

// General Seasonality Tags
//const std::string MACHINE_TAG{"a"};
const std::string SHORT_TEST_TAG{"d"};
const std::string LONG_TEST_TAG{"e"};

// Calendar Cyclic Test Tags
//const std::string MACHINE_TAG{"a"};
const std::string LAST_MONTH_TAG{"b"};
//const std::string TEST_TAG{"c"};

// Seasonal Components Tags
//const std::string MACHINE_TAG{"a"};
const std::string TREND_TAG{"b"};
const std::string SEASONAL_TAG{"c"};
const std::string CALENDAR_TAG{"d"};
const std::string COMPONENT_TAG{"e"};
const std::string ERRORS_TAG{"f"};
const std::string REGRESSION_TAG{"g"};
const std::string VARIANCE_TAG{"h"};
const std::string TIME_ORIGIN_TAG{"i"};
const std::string LAST_UPDATE_TAG{"j"};
const std::string PARAMETER_PROCESS_TAG{"k"};

const core_t::TTime FOREVER{boost::numeric::bounds<core_t::TTime>::highest()};
const core_t::TTime UNSET_LAST_UPDATE{0};
const std::size_t MAXIMUM_COMPONENTS = 8;
const TTrendCRef NO_TREND;
const TSeasonalComponentVec NO_SEASONAL_COMPONENTS;
const TCalendarComponentVec NO_CALENDAR_COMPONENTS;

}

//////// SMessage ////////

CTimeSeriesDecompositionDetail::SMessage::SMessage(void) : s_Time{}, s_LastTime{} {}
CTimeSeriesDecompositionDetail::SMessage::SMessage(core_t::TTime time, core_t::TTime lastTime) :
        s_Time{time}, s_LastTime{lastTime}
{}

//////// SAddValue ////////

CTimeSeriesDecompositionDetail::SAddValue::SAddValue(core_t::TTime time,
                                                     core_t::TTime lastTime,
                                                     double value,
                                                     const maths_t::TWeightStyleVec &weightStyles,
                                                     const maths_t::TDouble4Vec &weights,
                                                     double trend,
                                                     double nonDiurnal,
                                                     double seasonal,
                                                     double calendar) :
        SMessage{time, lastTime},
        s_Value{value},
        s_WeightStyles{weightStyles},
        s_Weights{weights},
        s_Trend{trend},
        s_NonDiurnal{nonDiurnal},
        s_Seasonal{seasonal},
        s_Calendar{calendar}
{}

//////// SDetectedTrend ////////

CTimeSeriesDecompositionDetail::SDetectedTrend::SDetectedTrend(core_t::TTime time,
                                                               core_t::TTime lastTime,
                                                               const CTrendTest &test) :
        SMessage{time, lastTime}, s_Test{test}
{}

//////// SDetectedDiurnal ////////

CTimeSeriesDecompositionDetail::SDetectedDiurnal::SDetectedDiurnal(core_t::TTime time,
                                                                   core_t::TTime lastTime,
                                                                   const CPeriodicityTestResult &result,
                                                                   const CDiurnalPeriodicityTest &test) :
        SMessage{time, lastTime}, s_Result{result}, s_Test{test}
{}

//////// SDetectedNonDiurnal ////////

CTimeSeriesDecompositionDetail::SDetectedNonDiurnal::SDetectedNonDiurnal(core_t::TTime time,
                                                                         core_t::TTime lastTime,
                                                                         bool discardLongTermTrend,
                                                                         const CPeriodicityTestResult &result,
                                                                         const CGeneralPeriodicityTest &test) :
        SMessage{time, lastTime},
        s_DiscardLongTermTrend{discardLongTermTrend},
        s_Result{result}, s_Test{test}
{}

//////// SDetectedCalendar ////////

CTimeSeriesDecompositionDetail::SDetectedCalendar::SDetectedCalendar(core_t::TTime time,
                                                                     core_t::TTime lastTime,
                                                                     CCalendarFeature feature) :
        SMessage{time, lastTime}, s_Feature{feature}
{}

//////// SNewComponent ////////

CTimeSeriesDecompositionDetail::SNewComponents::SNewComponents(core_t::TTime time,
                                                               core_t::TTime lastTime,
                                                               EComponent component) :
        SMessage{time, lastTime}, s_Component{component}
{}

//////// CHandler ////////

CTimeSeriesDecompositionDetail::CHandler::CHandler(void) : m_Mediator{0} {}
CTimeSeriesDecompositionDetail::CHandler::~CHandler(void) {}

void CTimeSeriesDecompositionDetail::CHandler::handle(const SAddValue &/*message*/) {}

void CTimeSeriesDecompositionDetail::CHandler::handle(const SDetectedTrend &/*message*/) {}

void CTimeSeriesDecompositionDetail::CHandler::handle(const SDetectedDiurnal &/*message*/) {}

void CTimeSeriesDecompositionDetail::CHandler::handle(const SDetectedNonDiurnal &/*message*/) {}

void CTimeSeriesDecompositionDetail::CHandler::handle(const SDetectedCalendar &/*message*/) {}

void CTimeSeriesDecompositionDetail::CHandler::handle(const SNewComponents &/*message*/) {}

void CTimeSeriesDecompositionDetail::CHandler::mediator(CMediator *mediator)
{
    m_Mediator = mediator;
}

CTimeSeriesDecompositionDetail::CMediator *CTimeSeriesDecompositionDetail::CHandler::mediator(void) const
{
    return m_Mediator;
}

//////// CMediator ////////

template<typename M>
void CTimeSeriesDecompositionDetail::CMediator::forward(const M &message) const
{
    for (CHandler &handler : m_Handlers)
    {
        handler.handle(message);
    }
}

void CTimeSeriesDecompositionDetail::CMediator::registerHandler(CHandler &handler)
{
    m_Handlers.push_back(boost::ref(handler));
    handler.mediator(this);
}

void CTimeSeriesDecompositionDetail::CMediator::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    mem->setName("CMediator");
    core::CMemoryDebug::dynamicSize("m_Handlers", m_Handlers, mem);
}

std::size_t CTimeSeriesDecompositionDetail::CMediator::memoryUsage(void) const
{
    return core::CMemory::dynamicSize(m_Handlers);
}

//////// CLongTermTrendTest ////////

CTimeSeriesDecompositionDetail::CLongTermTrendTest::CLongTermTrendTest(double decayRate) :
        m_Machine{core::CStateMachine::create(LT_ALPHABET, LT_STATES, LT_TRANSITION_FUNCTION, LT_INITIAL)},
        m_MaximumDecayRate{decayRate},
        m_NextTestTime{},
        m_Test{new CTrendTest{decayRate}}
{}

CTimeSeriesDecompositionDetail::CLongTermTrendTest::CLongTermTrendTest(const CLongTermTrendTest &other) :
        m_Machine{other.m_Machine},
        m_MaximumDecayRate{other.m_MaximumDecayRate},
        m_NextTestTime{other.m_NextTestTime},
        m_Test{other.m_Test ? new CTrendTest{*other.m_Test} : 0}
{}

bool CTimeSeriesDecompositionDetail::CLongTermTrendTest::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser)
{
    do
    {
        const std::string &name{traverser.name()};
        RESTORE(MACHINE_TAG, traverser.traverseSubLevel(
                                 boost::bind(&core::CStateMachine::acceptRestoreTraverser, &m_Machine, _1)));
        RESTORE_BUILT_IN(NEXT_TEST_TIME_TAG, m_NextTestTime)
        RESTORE_SETUP_TEARDOWN(TEST_TAG,
                               m_Test.reset(new CTrendTest(m_MaximumDecayRate)),
                               traverser.traverseSubLevel(
                                   boost::bind(&CTrendTest::acceptRestoreTraverser, m_Test.get(), _1)),
                               /**/)
    }
    while (traverser.next());
    return true;
}

void CTimeSeriesDecompositionDetail::CLongTermTrendTest::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertLevel(MACHINE_TAG, boost::bind(&core::CStateMachine::acceptPersistInserter, &m_Machine, _1));
    inserter.insertValue(NEXT_TEST_TIME_TAG, m_NextTestTime);
    if (m_Test)
    {
        inserter.insertLevel(TEST_TAG, boost::bind(&CTrendTest::acceptPersistInserter, m_Test.get(), _1));
    }
}

void CTimeSeriesDecompositionDetail::CLongTermTrendTest::swap(CLongTermTrendTest &other)
{
    std::swap(m_Machine, other.m_Machine);
    std::swap(m_MaximumDecayRate, other.m_MaximumDecayRate);
    std::swap(m_NextTestTime, other.m_NextTestTime);
    m_Test.swap(other.m_Test);
}

void CTimeSeriesDecompositionDetail::CLongTermTrendTest::handle(const SAddValue &message)
{
    core_t::TTime time{message.s_Time};
    double value{message.s_Value - message.s_Seasonal - message.s_Calendar};
    double count{maths_t::countForUpdate(message.s_WeightStyles, message.s_Weights)};

    this->test(message);

    if (time >= m_NextTestTime - this->testInterval())
    {
        switch (m_Machine.state())
        {
        case LT_NOT_TESTING:
            break;
        case LT_TEST:
            m_Test->add(time, value, count);
            m_Test->captureVariance(time, value, count);
            break;
        case LT_INITIAL:
            this->apply(LT_NEW_VALUE, message);
            this->handle(message);
            break;
        default:
            LOG_ERROR("Test in a bad state: " << m_Machine.state());
            this->apply(LT_RESET, message);
            break;
        }
    }
}

void CTimeSeriesDecompositionDetail::CLongTermTrendTest::handle(const SNewComponents &message)
{
    switch (message.s_Component)
    {
    case SNewComponents::E_DiurnalSeasonal:
    case SNewComponents::E_GeneralSeasonal:
        if (m_Machine.state() != LT_NOT_TESTING)
        {
            this->apply(LT_RESET, message);
        }
        break;
    case SNewComponents::E_Trend:
    case SNewComponents::E_CalendarCyclic:
        break;
    }
}

void CTimeSeriesDecompositionDetail::CLongTermTrendTest::test(const SMessage &message)
{
    core_t::TTime time{message.s_Time};
    core_t::TTime lastTime{message.s_LastTime};

    if (this->shouldTest(time))
    {
        switch (m_Machine.state())
        {
        case LT_NOT_TESTING:
        case LT_INITIAL:
            break;
        case LT_TEST:
            if (m_Test->test())
            {
                this->mediator()->forward(SDetectedTrend(time, lastTime, *m_Test));
                this->apply(LT_FINISH_TEST, message);
            }
            break;
        default:
            LOG_ERROR("Test in a bad state: " << m_Machine.state());
            this->apply(LT_RESET, message);
            break;
        }
    }
}

void CTimeSeriesDecompositionDetail::CLongTermTrendTest::decayRate(double decayRate)
{
    if (m_Test)
    {
        m_Test->decayRate(std::min(decayRate, m_MaximumDecayRate));
    }
}

void CTimeSeriesDecompositionDetail::CLongTermTrendTest::propagateForwards(core_t::TTime start,
                                                                           core_t::TTime end)
{
    if (m_Test)
    {
        double time{static_cast<double>(end - start) / static_cast<double>(DAY)};
        m_Test->propagateForwardsByTime(time);
    }
}

void CTimeSeriesDecompositionDetail::CLongTermTrendTest::skipTime(core_t::TTime skipInterval)
{
    core_t::TTime testInterval{this->testInterval()};
    m_NextTestTime = CIntegerTools::floor(m_NextTestTime + skipInterval + testInterval, testInterval);
}

uint64_t CTimeSeriesDecompositionDetail::CLongTermTrendTest::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, m_Machine);
    seed = CChecksum::calculate(seed, m_MaximumDecayRate);
    seed = CChecksum::calculate(seed, m_NextTestTime);
    return CChecksum::calculate(seed, m_Test);
}

void CTimeSeriesDecompositionDetail::CLongTermTrendTest::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    mem->setName("CLongTermTrendTest");
    core::CMemoryDebug::dynamicSize("m_Test", m_Test, mem);
}

std::size_t CTimeSeriesDecompositionDetail::CLongTermTrendTest::memoryUsage(void) const
{
    return core::CMemory::dynamicSize(m_Test);
}

void CTimeSeriesDecompositionDetail::CLongTermTrendTest::apply(std::size_t symbol, const SMessage &message)
{
    core_t::TTime time{message.s_Time};

    std::size_t old{m_Machine.state()};
    m_Machine.apply(symbol);
    std::size_t state{m_Machine.state()};

    if (state != old)
    {
        LOG_TRACE(LT_STATES[old] << "," << LT_ALPHABET[symbol] << " -> " << LT_STATES[state]);

        if (old == LT_INITIAL)
        {
            m_NextTestTime = time + 3 * WEEK;
        }

        switch (state)
        {
        case LT_TEST:
            break;
        case LT_NOT_TESTING:
            m_NextTestTime = core_t::TTime{};
            m_Test.reset();
            break;
        case LT_INITIAL:
            m_NextTestTime = core_t::TTime{};
            m_Test.reset(new CTrendTest(m_MaximumDecayRate));
            break;
        default:
            LOG_ERROR("Test in a bad state: " << state);
            this->apply(LT_RESET, message);
            break;
        }
    }
}

bool CTimeSeriesDecompositionDetail::CLongTermTrendTest::shouldTest(core_t::TTime time)
{
    if (time >= m_NextTestTime)
    {
        m_NextTestTime = CIntegerTools::ceil(time + 1, this->testInterval());
        return true;
    }
    return false;
}

core_t::TTime CTimeSeriesDecompositionDetail::CLongTermTrendTest::testInterval(void) const
{
    return DAY;
}

//////// CDiurnalTest ////////

CTimeSeriesDecompositionDetail::CDiurnalTest::CDiurnalTest(double decayRate,
                                                           core_t::TTime bucketLength) :
        m_Machine{core::CStateMachine::create(DW_ALPHABET, DW_STATES, DW_TRANSITION_FUNCTION, DW_INITIAL)},
        m_DecayRate{decayRate},
        m_BucketLength{bucketLength},
        m_NextTestTime{},
        m_StartedRegularTest{},
        m_TimeOutRegularTest{}
{}

CTimeSeriesDecompositionDetail::CDiurnalTest::CDiurnalTest(const CDiurnalTest &other) :
        m_Machine{other.m_Machine},
        m_DecayRate{other.m_DecayRate},
        m_BucketLength{other.m_BucketLength},
        m_NextTestTime{other.m_NextTestTime},
        m_StartedRegularTest{other.m_StartedRegularTest},
        m_TimeOutRegularTest{other.m_TimeOutRegularTest},
        m_RegularTest{other.m_RegularTest ? new CDiurnalPeriodicityTest{*other.m_RegularTest} : 0},
        m_SmallTest{other.m_SmallTest ? new CRandomizedPeriodicityTest{*other.m_SmallTest} : 0},
        m_Periods{other.m_Periods}
{}

bool CTimeSeriesDecompositionDetail::CDiurnalTest::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser)
{
    do
    {
        const std::string &name{traverser.name()};
        RESTORE(MACHINE_TAG, traverser.traverseSubLevel(
                                 boost::bind(&core::CStateMachine::acceptRestoreTraverser, &m_Machine, _1)));
        RESTORE_BUILT_IN(NEXT_TEST_TIME_TAG, m_NextTestTime)
        RESTORE_BUILT_IN(STARTED_REGULAR_TEST_TAG, m_StartedRegularTest)
        RESTORE_BUILT_IN(TIME_OUT_REGULAR_TEST_TAG, m_TimeOutRegularTest)
        RESTORE_SETUP_TEARDOWN(REGULAR_TEST_TAG,
                               m_RegularTest.reset(new CDiurnalPeriodicityTest(regularTestDecayRate(m_DecayRate))),
                               traverser.traverseSubLevel(
                                   boost::bind(&CDiurnalPeriodicityTest::acceptRestoreTraverser, m_RegularTest.get(), _1)),
                               /**/)
        RESTORE_SETUP_TEARDOWN(SMALL_TEST_TAG,
                               m_SmallTest.reset(new CRandomizedPeriodicityTest),
                               traverser.traverseSubLevel(
                                   boost::bind(&CRandomizedPeriodicityTest::acceptRestoreTraverser, m_SmallTest.get(), _1)),
                               /**/)
        RESTORE(PERIODS_TAG, traverser.traverseSubLevel(
                                 boost::bind(&CPeriodicityTestResult::acceptRestoreTraverser, &m_Periods, _1)))
    }
    while (traverser.next());
    return true;
}

void CTimeSeriesDecompositionDetail::CDiurnalTest::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertLevel(MACHINE_TAG, boost::bind(&core::CStateMachine::acceptPersistInserter, &m_Machine, _1));
    inserter.insertValue(NEXT_TEST_TIME_TAG, m_NextTestTime);
    inserter.insertValue(STARTED_REGULAR_TEST_TAG, m_StartedRegularTest);
    inserter.insertValue(TIME_OUT_REGULAR_TEST_TAG, m_TimeOutRegularTest);
    if (m_RegularTest)
    {
        inserter.insertLevel(REGULAR_TEST_TAG, boost::bind(
                                 &CDiurnalPeriodicityTest::acceptPersistInserter, m_RegularTest.get(), _1));
    }
    if (m_SmallTest)
    {
        inserter.insertLevel(SMALL_TEST_TAG, boost::bind(
                                 &CRandomizedPeriodicityTest::acceptPersistInserter, m_SmallTest.get(), _1));
    }
    inserter.insertLevel(PERIODS_TAG, boost::bind(
                             &CPeriodicityTestResult::acceptPersistInserter, &m_Periods, _1));
}

void CTimeSeriesDecompositionDetail::CDiurnalTest::swap(CDiurnalTest &other)
{
    std::swap(m_Machine, other.m_Machine);
    std::swap(m_DecayRate, other.m_DecayRate);
    std::swap(m_BucketLength, other.m_BucketLength);
    std::swap(m_NextTestTime, other.m_NextTestTime);
    std::swap(m_StartedRegularTest, other.m_StartedRegularTest);
    std::swap(m_TimeOutRegularTest, other.m_TimeOutRegularTest);
    m_RegularTest.swap(other.m_RegularTest);
    m_SmallTest.swap(other.m_SmallTest);
    std::swap(m_Periods, other.m_Periods);
}

void CTimeSeriesDecompositionDetail::CDiurnalTest::handle(const SAddValue &message)
{
    core_t::TTime time{message.s_Time};
    double value{message.s_Value - message.s_Trend - message.s_NonDiurnal - message.s_Calendar};
    const maths_t::TWeightStyleVec &weightStyles{message.s_WeightStyles};
    const maths_t::TDouble4Vec &weights{message.s_Weights};

    this->test(message);

    switch (m_Machine.state())
    {
    case DW_NOT_TESTING:
        break;
    case DW_SMALL_TEST:
        m_SmallTest->add(time, value);
        break;
    case DW_REGULAR_TEST:
        if (time < this->timeOutRegularTest())
        {
            m_RegularTest->add(time, value, maths_t::countForUpdate(weightStyles, weights));
        }
        else
        {
            LOG_TRACE("Switching to small test at " << time);
            this->apply(DW_REGULAR_TEST_TIMED_OUT, message);
            this->handle(message);
        }
        break;
    case DW_INITIAL:
        this->apply(DW_NEW_VALUE, message);
        this->handle(message);
        break;
    default:
        LOG_ERROR("Test in a bad state: " << m_Machine.state());
        this->apply(DW_RESET, message);
        break;
    }
}

void CTimeSeriesDecompositionDetail::CDiurnalTest::handle(const SNewComponents &message)
{
    switch (message.s_Component)
    {
    case SNewComponents::E_GeneralSeasonal:
    case SNewComponents::E_Trend:
        if (m_Machine.state() != DW_NOT_TESTING)
        {
            this->apply(DW_RESET, message);
        }
        break;
    case SNewComponents::E_DiurnalSeasonal:
    case SNewComponents::E_CalendarCyclic:
        break;
    }
}

void CTimeSeriesDecompositionDetail::CDiurnalTest::test(const SMessage &message)
{
    core_t::TTime time{message.s_Time};
    core_t::TTime lastTime{message.s_Time};

    switch (m_Machine.state())
    {
    case DW_NOT_TESTING:
    case DW_INITIAL:
        break;
    case DW_SMALL_TEST:
        if (this->shouldTest(time))
        {
            LOG_TRACE("Small testing at " << time);
            if (m_SmallTest->test())
            {
                LOG_TRACE("Switching to full test at " << time);
                this->apply(DW_SMALL_TEST_TRUE, message);
            }
        }
        break;
    case DW_REGULAR_TEST:
        if (this->shouldTest(time))
        {
            LOG_TRACE("Regular testing at " << time);
            CPeriodicityTestResult result{m_RegularTest->test()};
            if (result.periodic() && result != m_Periods)
            {
                this->mediator()->forward(SDetectedDiurnal(time, lastTime, result, *m_RegularTest));
                m_Periods = result;
            }
            if (result.periodic())
            {
                if (m_RegularTest->seenSufficientData())
                {
                    LOG_TRACE("Finished testing");
                    this->apply(DW_FINISH_TEST, message);
                }
                else
                {
                    m_NextTestTime = std::max(CIntegerTools::ceil(
                            m_StartedRegularTest + 2 * WEEK, this->testInterval()),
                            m_NextTestTime);
                }
            }
        }
        break;
    default:
        LOG_ERROR("Test in a bad state: " << m_Machine.state());
        this->apply(DW_RESET, message);
        break;
    }
}

void CTimeSeriesDecompositionDetail::CDiurnalTest::propagateForwards(core_t::TTime start,
                                                                     core_t::TTime end)
{
    propagateTestForwards(start, end, DAY, m_RegularTest);
}

void CTimeSeriesDecompositionDetail::CDiurnalTest::skipTime(core_t::TTime skipInterval)
{
    core_t::TTime testInterval{this->testInterval()};
    m_NextTestTime = CIntegerTools::floor(m_NextTestTime + skipInterval + testInterval, testInterval);
    m_TimeOutRegularTest += skipInterval;
}

uint64_t CTimeSeriesDecompositionDetail::CDiurnalTest::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, m_Machine);
    seed = CChecksum::calculate(seed, m_DecayRate);
    seed = CChecksum::calculate(seed, m_BucketLength);
    seed = CChecksum::calculate(seed, m_NextTestTime);
    seed = CChecksum::calculate(seed, m_StartedRegularTest);
    seed = CChecksum::calculate(seed, m_TimeOutRegularTest);
    seed = CChecksum::calculate(seed, m_RegularTest);
    seed = CChecksum::calculate(seed, m_SmallTest);
    return CChecksum::calculate(seed, m_Periods);
}

void CTimeSeriesDecompositionDetail::CDiurnalTest::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    mem->setName("CDiurnalTest");
    core::CMemoryDebug::dynamicSize("m_RegularTest", m_RegularTest, mem);
    core::CMemoryDebug::dynamicSize("m_SmallTest", m_SmallTest, mem);
}

std::size_t CTimeSeriesDecompositionDetail::CDiurnalTest::memoryUsage(void) const
{
    std::size_t usage{core::CMemory::dynamicSize(m_RegularTest) + core::CMemory::dynamicSize(m_SmallTest)};
    if (m_Machine.state() == DW_INITIAL)
    {
        usage += this->extraMemoryOnInitialization();
    }
    return usage;
}

std::size_t CTimeSeriesDecompositionDetail::CDiurnalTest::extraMemoryOnInitialization(void) const
{
    static std::size_t result{0};
    if (result == 0)
    {
        TPeriodicityTestPtr regularTest(CDiurnalPeriodicityTest::create(
                                            m_BucketLength, regularTestDecayRate(m_DecayRate)));
        result = core::CMemory::dynamicSize(regularTest);
    }
    return result;
}

void CTimeSeriesDecompositionDetail::CDiurnalTest::apply(std::size_t symbol, const SMessage &message)
{
    core_t::TTime time{message.s_Time};

    std::size_t old{m_Machine.state()};
    m_Machine.apply(symbol);
    std::size_t state{m_Machine.state()};

    if (state != old)
    {
        LOG_TRACE(DW_STATES[old] << "," << DW_ALPHABET[symbol] << " -> " << DW_STATES[state]);

        if (old == DW_INITIAL)
        {
            m_NextTestTime = time;
            m_TimeOutRegularTest = time + scale(5 * WEEK, m_BucketLength);
        }

        switch (state)
        {
        case DW_SMALL_TEST:
            if (m_RegularTest)
            {
                m_RegularTest.reset();
            }
            if (!m_SmallTest)
            {
                m_SmallTest.reset(new CRandomizedPeriodicityTest);
            }
            break;
        case DW_REGULAR_TEST:
            if (m_SmallTest)
            {
                m_TimeOutRegularTest = time + scale(9 * WEEK, m_BucketLength);
                m_SmallTest.reset();
            }
            if (!m_RegularTest)
            {
                m_StartedRegularTest = time;
                m_RegularTest.reset(CDiurnalPeriodicityTest::create(
                                        m_BucketLength, regularTestDecayRate(m_DecayRate)));
                if (!m_RegularTest)
                {
                    this->apply(DW_NOT_TESTING, message);
                }
            }
            break;
        case DW_NOT_TESTING:
        case DW_INITIAL:
            m_NextTestTime = core_t::TTime{};
            m_TimeOutRegularTest = core_t::TTime{};
            m_SmallTest.reset();
            m_RegularTest.reset();
            break;
        default:
            LOG_ERROR("Test in a bad state: " << state);
            this->apply(DW_RESET, message);
            break;
        }
    }
}

bool CTimeSeriesDecompositionDetail::CDiurnalTest::shouldTest(core_t::TTime time)
{
    if (time >= m_NextTestTime)
    {
        m_NextTestTime = CIntegerTools::ceil(time + 1, this->testInterval());
        return true;
    }
    return false;
}

core_t::TTime CTimeSeriesDecompositionDetail::CDiurnalTest::testInterval(void) const
{
    switch (m_Machine.state())
    {
    case DW_SMALL_TEST:
        return DAY;
    case DW_REGULAR_TEST:
        return m_NextTestTime > m_StartedRegularTest + 2 * WEEK ? 2 * WEEK : DAY;
    default:
        break;
    }
    return FOREVER;
}

core_t::TTime CTimeSeriesDecompositionDetail::CDiurnalTest::timeOutRegularTest(void) const
{
    return m_TimeOutRegularTest + static_cast<core_t::TTime>(
                                      6.0 * static_cast<double>(WEEK)
                                          * (1.0 - m_RegularTest->populatedRatio()));
}

//////// CNonDiurnalTest ////////

CTimeSeriesDecompositionDetail::CNonDiurnalTest::CNonDiurnalTest(double decayRate,
                                                                 core_t::TTime bucketLength) :
        m_Machine{core::CStateMachine::create(
                      GS_ALPHABET, GS_STATES, GS_TRANSITION_FUNCTION,
                      bucketLength > LONG_BUCKET_LENGTHS.back() ? GS_NOT_TESTING : GS_INITIAL)},
        m_DecayRate{decayRate},
        m_BucketLength{bucketLength}
{}

CTimeSeriesDecompositionDetail::CNonDiurnalTest::CNonDiurnalTest(const CNonDiurnalTest &other) :
        m_Machine{other.m_Machine},
        m_DecayRate{other.m_DecayRate},
        m_BucketLength{other.m_BucketLength}
{
    for (std::size_t i = 0u; i < other.m_Tests.size(); ++i)
    {
        if (other.m_Tests[i])
        {
            m_Tests[i] = boost::make_shared<CScanningPeriodicityTest>(*other.m_Tests[i]);
        }
    }
}

bool CTimeSeriesDecompositionDetail::CNonDiurnalTest::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser)
{
    do
    {
        const std::string &name{traverser.name()};
        RESTORE(MACHINE_TAG, traverser.traverseSubLevel(
                                 boost::bind(&core::CStateMachine::acceptRestoreTraverser, &m_Machine, _1)))
        RESTORE_SETUP_TEARDOWN(SHORT_TEST_TAG,
                               m_Tests[E_Short].reset(this->newTest(E_Short)),
                               m_Tests[E_Short] && traverser.traverseSubLevel(
                                                       boost::bind(&CScanningPeriodicityTest::acceptRestoreTraverser,
                                                                   m_Tests[E_Short].get(), _1)),
                               /**/)
        RESTORE_SETUP_TEARDOWN(LONG_TEST_TAG,
                               m_Tests[E_Long].reset(this->newTest(E_Long)),
                               m_Tests[E_Long] && traverser.traverseSubLevel(
                                                      boost::bind(&CScanningPeriodicityTest::acceptRestoreTraverser,
                                                                  m_Tests[E_Long].get(), _1)),
                               /**/)
    }
    while (traverser.next());
    return true;
}

void CTimeSeriesDecompositionDetail::CNonDiurnalTest::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertLevel(MACHINE_TAG, boost::bind(&core::CStateMachine::acceptPersistInserter, &m_Machine, _1));
    if (m_Tests[E_Short])
    {
        inserter.insertLevel(SHORT_TEST_TAG, boost::bind(
                &CScanningPeriodicityTest::acceptPersistInserter, m_Tests[E_Short].get(), _1));
    }
    if (m_Tests[E_Long])
    {
        inserter.insertLevel(LONG_TEST_TAG, boost::bind(
                &CScanningPeriodicityTest::acceptPersistInserter, m_Tests[E_Long].get(), _1));
    }
}

void CTimeSeriesDecompositionDetail::CNonDiurnalTest::swap(CNonDiurnalTest &other)
{
    std::swap(m_Machine, other.m_Machine);
    std::swap(m_DecayRate, other.m_DecayRate);
    std::swap(m_BucketLength, other.m_BucketLength);
    m_Tests[0].swap(other.m_Tests[0]);
    m_Tests[1].swap(other.m_Tests[1]);
}

void CTimeSeriesDecompositionDetail::CNonDiurnalTest::handle(const SAddValue &message)
{
    core_t::TTime time{message.s_Time};
    double values[2];
    double aperiodic{message.s_Value - message.s_Seasonal - message.s_Calendar};
    values[E_Short] = aperiodic - message.s_Trend;
    values[E_Long]  = aperiodic;
    const maths_t::TWeightStyleVec &weightStyles{message.s_WeightStyles};
    const maths_t::TDouble4Vec &weights{message.s_Weights};

    this->test(message);

    switch (m_Machine.state())
    {
    case GS_TEST:
        for (auto test : {E_Short, E_Long})
        {
            if (m_Tests[test])
            {
                m_Tests[test]->add(time, values[test],
                                   maths_t::countForUpdate(weightStyles, weights));
            }
        }
        break;
    case GS_NOT_TESTING:
        break;
    case GS_INITIAL:
        this->apply(GS_NEW_VALUE, message);
        this->handle(message);
        break;
    default:
        LOG_ERROR("Test in a bad state: " << m_Machine.state());
        this->apply(GS_RESET, message);
        break;
    }
}

void CTimeSeriesDecompositionDetail::CNonDiurnalTest::handle(const SNewComponents &message)
{
    if (m_Machine.state() != GS_NOT_TESTING)
    {
        switch (message.s_Component)
        {
        case SNewComponents::E_GeneralSeasonal:
        case SNewComponents::E_Trend:
            this->apply(GS_RESET, message, {{0 * WEEK, 12 * WEEK}});
            break;
        case SNewComponents::E_DiurnalSeasonal:
            this->apply(GS_RESET, message, {{3 * WEEK, 12 * WEEK}});
            break;
        case SNewComponents::E_CalendarCyclic:
            break;
        }
    }
}

void CTimeSeriesDecompositionDetail::CNonDiurnalTest::test(const SMessage &message)
{
    core_t::TTime time{message.s_Time};
    core_t::TTime lastTime{message.s_LastTime};

    switch (m_Machine.state())
    {
    case GS_TEST:
        for (auto test_ : {E_Short, E_Long})
        {
            if (m_Tests[test_] && m_Tests[test_]->needToCompress(time))
            {
                CGeneralPeriodicityTest test;
                CPeriodicityTestResult result;
                boost::tie(test, result) = m_Tests[test_]->test();
                if (result.periodic())
                {
                    this->mediator()->forward(SDetectedNonDiurnal(
                            time, lastTime, test_ == E_Long, result, test));
                }
            }
        }
        break;
    case GS_NOT_TESTING:
    case GS_INITIAL:
        break;
    default:
        LOG_ERROR("Test in a bad state: " << m_Machine.state());
        this->apply(GS_RESET, message);
        break;
    }
}

void CTimeSeriesDecompositionDetail::CNonDiurnalTest::propagateForwards(core_t::TTime start,
                                                                        core_t::TTime end)
{
    propagateTestForwards(start, end, DAY, m_Tests[E_Short]);
    propagateTestForwards(start, end, WEEK, m_Tests[E_Long]);
}

void CTimeSeriesDecompositionDetail::CNonDiurnalTest::skipTime(core_t::TTime skipInterval)
{
    for (const auto &test : m_Tests)
    {
        if (test)
        {
            test->skipTime(skipInterval);
        }
    }
}

uint64_t CTimeSeriesDecompositionDetail::CNonDiurnalTest::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, m_Machine);
    seed = CChecksum::calculate(seed, m_DecayRate);
    seed = CChecksum::calculate(seed, m_BucketLength);
    return CChecksum::calculate(seed, m_Tests);
}

void CTimeSeriesDecompositionDetail::CNonDiurnalTest::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    mem->setName("CNonDiurnalTest");
    core::CMemoryDebug::dynamicSize("m_Tests", m_Tests, mem);
}

std::size_t CTimeSeriesDecompositionDetail::CNonDiurnalTest::memoryUsage(void) const
{
    std::size_t usage{core::CMemory::dynamicSize(m_Tests)};
    if (m_Machine.state() == GS_INITIAL)
    {
        usage += this->extraMemoryOnInitialization();
    }
    return usage;
}

std::size_t CTimeSeriesDecompositionDetail::CNonDiurnalTest::extraMemoryOnInitialization(void) const
{
    static std::size_t result{0};
    if (result == 0)
    {
        TScanningPeriodicityTestPtr shortTest(this->newTest(E_Short));
        TScanningPeriodicityTestPtr longTest(this->newTest(E_Long));
        result = core::CMemory::dynamicSize(shortTest) + core::CMemory::dynamicSize(longTest);
    }
    return result;
}

void CTimeSeriesDecompositionDetail::CNonDiurnalTest::apply(std::size_t symbol,
                                                            const SMessage &message,
                                                            const TTimeAry &offsets)
{
    core_t::TTime time{message.s_Time};

    std::size_t old{m_Machine.state()};
    m_Machine.apply(symbol);
    std::size_t state{m_Machine.state()};

    if (state != old)
    {
        LOG_TRACE(GS_STATES[old] << "," << GS_ALPHABET[symbol] << " -> " << GS_STATES[state]);

        switch (state)
        {
        case GS_TEST:
            if (std::all_of(m_Tests.begin(), m_Tests.end(),
                            [](const TScanningPeriodicityTestPtr &test) { return !test; }))
            {
                for (auto test : {E_Short, E_Long})
                {
                    m_Tests[test].reset(this->newTest(test));
                    if (m_Tests[test])
                    {
                        m_Tests[test]->initialize(time + offsets[test]);
                    }
                }
            }
            break;
        case GS_INITIAL:
            for (auto test : {E_Short, E_Long})
            {
                m_Tests[test].reset(this->newTest(test));
                if (m_Tests[test])
                {
                    m_Tests[test]->initialize(time + offsets[test]);
                }
            }
            break;
        case GS_NOT_TESTING:
            m_Tests[0].reset();
            m_Tests[1].reset();
            break;
        default:
            LOG_ERROR("Test in a bad state: " << state);
            this->apply(GS_RESET, message, offsets);
            break;
        }
    }
}

CScanningPeriodicityTest *CTimeSeriesDecompositionDetail::CNonDiurnalTest::newTest(ETest test) const
{
    using TTimeCRng = CScanningPeriodicityTest::TTimeCRng;
    switch (test)
    {
    case E_Short:
        if (m_BucketLength < SHORT_BUCKET_LENGTHS.back())
        {
            std::ptrdiff_t a{std::lower_bound(SHORT_BUCKET_LENGTHS.begin(),
                                              SHORT_BUCKET_LENGTHS.end(),
                                              m_BucketLength) - SHORT_BUCKET_LENGTHS.begin()};
            std::size_t b{SHORT_BUCKET_LENGTHS.size()};
            TTimeCRng bucketLengths(SHORT_BUCKET_LENGTHS, a, b);
            return new CScanningPeriodicityTest(bucketLengths, TEST_SIZE, m_DecayRate);
        }
        break;
    case E_Long:
        if (m_BucketLength < LONG_BUCKET_LENGTHS.back())
        {
            std::ptrdiff_t a{std::lower_bound(LONG_BUCKET_LENGTHS.begin(),
                                              LONG_BUCKET_LENGTHS.end(),
                                              m_BucketLength) - LONG_BUCKET_LENGTHS.begin()};
            std::size_t b{LONG_BUCKET_LENGTHS.size()};
            TTimeCRng bucketLengths(LONG_BUCKET_LENGTHS, a, b);
            return new CScanningPeriodicityTest(bucketLengths, TEST_SIZE, m_DecayRate);
        }
        break;
    }
    return 0;
}

const std::size_t CTimeSeriesDecompositionDetail::CNonDiurnalTest::TEST_SIZE{168};
const TTimeVec CTimeSeriesDecompositionDetail::CNonDiurnalTest::SHORT_BUCKET_LENGTHS
    {
        1, 5, 10, 30, 60, 300, 600, 1800, 3600, 7200
    };
const TTimeVec CTimeSeriesDecompositionDetail::CNonDiurnalTest::LONG_BUCKET_LENGTHS
    {
        21600, 43200, 86400, 172800, 345600, 691200
    };

//////// CCalendarCyclic ////////

CTimeSeriesDecompositionDetail::CCalendarTest::CCalendarTest(double decayRate,
                                                             core_t::TTime bucketLength) :
        m_Machine{core::CStateMachine::create(CC_ALPHABET, CC_STATES, CC_TRANSITION_FUNCTION,
                                              bucketLength > DAY ? CC_NOT_TESTING : CC_INITIAL)},
        m_DecayRate{decayRate},
        m_LastMonth{}
{}

CTimeSeriesDecompositionDetail::CCalendarTest::CCalendarTest(const CCalendarTest &other) :
        m_Machine{other.m_Machine},
        m_DecayRate{other.m_DecayRate},
        m_LastMonth{other.m_LastMonth},
        m_Test{other.m_Test ? new CCalendarCyclicTest(*other.m_Test) : 0}
{}

bool CTimeSeriesDecompositionDetail::CCalendarTest::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser)
{
    do
    {
        const std::string &name{traverser.name()};
        RESTORE(MACHINE_TAG, traverser.traverseSubLevel(
                                 boost::bind(&core::CStateMachine::acceptRestoreTraverser, &m_Machine, _1)))
        RESTORE_BUILT_IN(LAST_MONTH_TAG, m_LastMonth);
        RESTORE_SETUP_TEARDOWN(TEST_TAG,
                               m_Test.reset(new CCalendarCyclicTest(m_DecayRate)),
                               traverser.traverseSubLevel(
                                   boost::bind(&CCalendarCyclicTest::acceptRestoreTraverser, m_Test.get(), _1)),
                               /**/)
    }
    while (traverser.next());
    return true;
}

void CTimeSeriesDecompositionDetail::CCalendarTest::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertLevel(MACHINE_TAG, boost::bind(&core::CStateMachine::acceptPersistInserter, &m_Machine, _1));
    inserter.insertValue(LAST_MONTH_TAG, m_LastMonth);
    if (m_Test)
    {
        inserter.insertLevel(TEST_TAG, boost::bind(
                &CCalendarCyclicTest::acceptPersistInserter, m_Test.get(), _1));
    }
}

void CTimeSeriesDecompositionDetail::CCalendarTest::swap(CCalendarTest &other)
{
    std::swap(m_Machine, other.m_Machine);
    std::swap(m_DecayRate, other.m_DecayRate);
    std::swap(m_LastMonth, other.m_LastMonth);
    m_Test.swap(other.m_Test);
}

void CTimeSeriesDecompositionDetail::CCalendarTest::handle(const SAddValue &message)
{
    core_t::TTime time{message.s_Time};
    double error{message.s_Value - message.s_Trend - message.s_Seasonal - message.s_Calendar};
    const maths_t::TWeightStyleVec &weightStyles{message.s_WeightStyles};
    const maths_t::TDouble4Vec &weights{message.s_Weights};

    this->test(message);

    switch (m_Machine.state())
    {
    case CC_TEST:
        m_Test->add(time, error, maths_t::countForUpdate(weightStyles, weights));
        break;
    case CC_NOT_TESTING:
        break;
    case CC_INITIAL:
        this->apply(CC_NEW_VALUE, message);
        this->handle(message);
        break;
    default:
        LOG_ERROR("Test in a bad state: " << m_Machine.state());
        this->apply(CC_RESET, message);
        break;
    }
}

void CTimeSeriesDecompositionDetail::CCalendarTest::handle(const SNewComponents &message)
{
    if (m_Machine.state() != CC_NOT_TESTING)
    {
        switch (message.s_Component)
        {
        case SNewComponents::E_GeneralSeasonal:
        case SNewComponents::E_DiurnalSeasonal:
        case SNewComponents::E_Trend:
            this->apply(GS_RESET, message);
            break;
        case SNewComponents::E_CalendarCyclic:
            break;
        }
    }
}

void CTimeSeriesDecompositionDetail::CCalendarTest::test(const SMessage &message)
{
    core_t::TTime time{message.s_Time};
    core_t::TTime lastTime{message.s_LastTime};

    if (this->shouldTest(time))
    {
        switch (m_Machine.state())
        {
        case CC_TEST:
        {
            if (CCalendarCyclicTest::TOptionalFeature feature = m_Test->test())
            {
                this->mediator()->forward(SDetectedCalendar(time, lastTime, *feature));
            }
            break;
        }
        case CC_NOT_TESTING:
        case CC_INITIAL:
            break;
        default:
            LOG_ERROR("Test in a bad state: " << m_Machine.state());
            this->apply(CC_RESET, message);
            break;
        }
    }
}

void CTimeSeriesDecompositionDetail::CCalendarTest::propagateForwards(core_t::TTime start,
                                                                      core_t::TTime end)
{
    propagateTestForwards(start, end, DAY, m_Test);
}

void CTimeSeriesDecompositionDetail::CCalendarTest::advanceTimeTo(core_t::TTime time)
{
    m_LastMonth = this->month(time);
}

uint64_t CTimeSeriesDecompositionDetail::CCalendarTest::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, m_Machine);
    seed = CChecksum::calculate(seed, m_DecayRate);
    return CChecksum::calculate(seed, m_Test);
}

void CTimeSeriesDecompositionDetail::CCalendarTest::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    mem->setName("CCalendarTest");
    core::CMemoryDebug::dynamicSize("m_Test", m_Test, mem);
}

std::size_t CTimeSeriesDecompositionDetail::CCalendarTest::memoryUsage(void) const
{
    std::size_t usage{core::CMemory::dynamicSize(m_Test)};
    if (m_Machine.state() == CC_INITIAL)
    {
        usage += this->extraMemoryOnInitialization();
    }
    return usage;
}

std::size_t CTimeSeriesDecompositionDetail::CCalendarTest::extraMemoryOnInitialization(void) const
{
    static std::size_t result{0};
    if (result == 0)
    {
        TCalendarCyclicTestPtr test(new CCalendarCyclicTest(m_DecayRate));
        result = core::CMemory::dynamicSize(test);
    }
    return result;
}

void CTimeSeriesDecompositionDetail::CCalendarTest::apply(std::size_t symbol, const SMessage &message)
{
    core_t::TTime time{message.s_Time};

    std::size_t old{m_Machine.state()};
    m_Machine.apply(symbol);
    std::size_t state{m_Machine.state()};

    if (state != old)
    {
        LOG_TRACE(CC_STATES[old] << "," << CC_ALPHABET[symbol] << " -> " << CC_STATES[state]);

        switch (state)
        {
        case CC_TEST:
            if (!m_Test)
            {
                m_Test.reset(new CCalendarCyclicTest(m_DecayRate));
                m_LastMonth = this->month(time) + 2;
            }
            break;
        case CC_NOT_TESTING:
        case CC_INITIAL:
            m_Test.reset();
            m_LastMonth = int{};
            break;
        default:
            LOG_ERROR("Test in a bad state: " << state);
            this->apply(CC_RESET, message);
            break;
        }
    }
}

bool CTimeSeriesDecompositionDetail::CCalendarTest::shouldTest(core_t::TTime time)
{
    int month{this->month(time)};
    if (month == (m_LastMonth + 1) % 12)
    {
        m_LastMonth = month;
        return true;
    }
    return false;
}

int CTimeSeriesDecompositionDetail::CCalendarTest::month(core_t::TTime time) const
{
    int dummy;
    int month;
    core::CTimezone::instance().dateFields(time, dummy, dummy, dummy, month, dummy, dummy);
    return month;
}

//////// CTrendCRef ////////

CTimeSeriesDecompositionDetail::CTrendCRef::CTrendCRef(void) :
        m_Trend{0}, m_Variance{0.0}, m_TimeOrigin{0},
        m_LastUpdate{0}, m_ParameterProcess{0}
{}

CTimeSeriesDecompositionDetail::CTrendCRef::CTrendCRef(const TRegression &trend,
                                                       double variance,
                                                       core_t::TTime timeOrigin,
                                                       core_t::TTime lastUpdate,
                                                       const TRegressionParameterProcess &process) :
        m_Trend{&trend},
        m_Variance{variance},
        m_TimeOrigin{timeOrigin},
        m_LastUpdate{lastUpdate},
        m_ParameterProcess{&process}
{}

bool CTimeSeriesDecompositionDetail::CTrendCRef::initialized(void) const
{
    return m_Trend != 0;
}

double CTimeSeriesDecompositionDetail::CTrendCRef::count(void) const
{
    return m_Trend ? m_Trend->count() : 0.0;
}

TDoubleDoublePr CTimeSeriesDecompositionDetail::CTrendCRef::prediction(core_t::TTime time, double confidence) const
{
    if (!m_Trend)
    {
        return {0.0, 0.0};
    }

    double m{CRegression::predict(*m_Trend, this->time(time))};

    if (confidence > 0.0 && m_Variance > 0.0)
    {
        double sd{std::sqrt(m_Variance) / std::max(m_Trend->count(), 1.0)};

        try
        {
            boost::math::normal normal{m, sd};
            double ql{boost::math::quantile(normal, (100.0 - confidence) / 200.0)};
            double qu{boost::math::quantile(normal, (100.0 + confidence) / 200.0)};
            return {ql, qu};
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("Failed calculating confidence interval: " << e.what()
                      << ", m = " << m
                      << ", sd = " << sd
                      << ", confidence = " << confidence);
        }
    }
    return {m, m};
}

double CTimeSeriesDecompositionDetail::CTrendCRef::variance(void) const
{
    return m_Variance;
}

bool CTimeSeriesDecompositionDetail::CTrendCRef::covariances(TMatrix &result) const
{
    return m_Trend ? m_Trend->covariances(m_Variance, result) : false;
}

double CTimeSeriesDecompositionDetail::CTrendCRef::varianceDueToParameterDrift(core_t::TTime time) const
{
    return m_ParameterProcess ? m_ParameterProcess->predictionVariance(regressionTime(time, m_LastUpdate)) : 0.0;
}

double CTimeSeriesDecompositionDetail::CTrendCRef::time(core_t::TTime time) const
{
    return m_Trend ? regressionTime(time, m_TimeOrigin) : 0.0;
}

//////// CComponents ////////

CTimeSeriesDecompositionDetail::CComponents::CComponents(double decayRate,
                                                         core_t::TTime bucketLength,
                                                         std::size_t seasonalComponentSize) :
        m_Machine{core::CStateMachine::create(SC_ALPHABET, SC_STATES, SC_TRANSITION_FUNCTION, SC_NORMAL)},
        m_DecayRate{decayRate},
        m_BucketLength{bucketLength},
        m_SeasonalComponentSize{seasonalComponentSize},
        m_CalendarComponentSize{seasonalComponentSize / 3},
        m_Watcher{0}
{}

CTimeSeriesDecompositionDetail::CComponents::CComponents(const CComponents &other) :
        m_Machine{other.m_Machine},
        m_DecayRate{other.m_DecayRate},
        m_BucketLength{other.m_BucketLength},
        m_SeasonalComponentSize{other.m_SeasonalComponentSize},
        m_CalendarComponentSize{other.m_CalendarComponentSize},
        m_Trend{other.m_Trend ? new STrend(*other.m_Trend) : 0},
        m_Seasonal{other.m_Seasonal ? new SSeasonal{*other.m_Seasonal} : 0},
        m_Calendar{other.m_Calendar ? new SCalendar{*other.m_Calendar} : 0},
        m_Watcher{0}
 {}

bool CTimeSeriesDecompositionDetail::CComponents::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser)
{
    do
    {
        const std::string &name{traverser.name()};
        RESTORE(MACHINE_TAG, traverser.traverseSubLevel(
                                 boost::bind(&core::CStateMachine::acceptRestoreTraverser, &m_Machine, _1)));
        RESTORE_SETUP_TEARDOWN(TREND_TAG,
                               m_Trend.reset(new STrend),
                               traverser.traverseSubLevel(boost::bind(&STrend::acceptRestoreTraverser,
                                                                      m_Trend.get(), _1)),
                               /**/)
        RESTORE_SETUP_TEARDOWN(SEASONAL_TAG,
                               m_Seasonal.reset(new SSeasonal),
                               traverser.traverseSubLevel(boost::bind(&SSeasonal::acceptRestoreTraverser,
                                                                      m_Seasonal.get(),
                                                                      m_DecayRate, m_BucketLength, _1)),
                               /**/)
        RESTORE_SETUP_TEARDOWN(CALENDAR_TAG,
                               m_Calendar.reset(new SCalendar),
                               traverser.traverseSubLevel(boost::bind(&SCalendar::acceptRestoreTraverser,
                                                                      m_Calendar.get(),
                                                                      m_DecayRate, m_BucketLength, _1)),
                               /**/)
    }
    while (traverser.next());

    return true;
}

void CTimeSeriesDecompositionDetail::CComponents::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertLevel(MACHINE_TAG, boost::bind(&core::CStateMachine::acceptPersistInserter, &m_Machine, _1));
    if (m_Trend)
    {
        inserter.insertLevel(TREND_TAG, boost::bind(&STrend::acceptPersistInserter, m_Trend.get(), _1));
    }
    if (m_Seasonal)
    {
        inserter.insertLevel(SEASONAL_TAG, boost::bind(&SSeasonal::acceptPersistInserter, m_Seasonal.get(), _1));
    }
    if (m_Calendar)
    {
        inserter.insertLevel(CALENDAR_TAG, boost::bind(&SCalendar::acceptPersistInserter, m_Calendar.get(), _1));
    }
}

void CTimeSeriesDecompositionDetail::CComponents::swap(CComponents &other)
{
    std::swap(m_Machine, other.m_Machine);
    std::swap(m_DecayRate, other.m_DecayRate);
    std::swap(m_BucketLength, other.m_BucketLength);
    std::swap(m_SeasonalComponentSize, other.m_SeasonalComponentSize);
    std::swap(m_CalendarComponentSize, other.m_CalendarComponentSize);
    m_Trend.swap(other.m_Trend);
    m_Seasonal.swap(other.m_Seasonal);
    m_Calendar.swap(other.m_Calendar);
}

void CTimeSeriesDecompositionDetail::CComponents::handle(const SAddValue &message)
{
    switch (m_Machine.state())
    {
    case SC_NORMAL:
    case SC_NEW_COMPONENTS:
        if (m_Trend || m_Seasonal)
        {
            this->interpolate(message);

            core_t::TTime time{message.s_Time};
            double value{message.s_Value};
            const maths_t::TWeightStyleVec &weightStyles{message.s_WeightStyles};
            const maths_t::TDouble4Vec &weights{message.s_Weights};

            TSeasonalComponentPtrVec seasonal;
            TCalendarComponentPtrVec calendar;
            TComponentErrorsPtrVec seasonalErrors;
            TComponentErrorsPtrVec calendarErrors;
            TDoubleVec deltas;

            if (m_Seasonal)
            {
                m_Seasonal->componentsErrorsAndDeltas(time, seasonal, seasonalErrors, deltas);
            }
            if (m_Calendar)
            {
                m_Calendar->componentsAndErrors(time, calendar, calendarErrors);
            }

            double weight{maths_t::countForUpdate(weightStyles, weights)};
            std::size_t m{seasonal.size()};
            std::size_t n{calendar.size()};

            CTrendCRef trend{this->trend()};
            double error;
            TDoubleVec values(m + n + 1, value);
            TDoubleVec predictions(m + n);
            decompose(trend, seasonal, calendar, deltas, time, error, values, predictions);

            if (m_Trend)
            {
                TMeanVarAccumulator moments{
                        CBasicStatistics::accumulator(trend.count(),
                                                      CBasicStatistics::mean(trend.prediction(time, 0.0)),
                                                      m_Trend->s_Variance)};
                moments.add(values[0], weight);
                double t{trend.time(time)};

                // Note this condition can change as a result adding the new
                // value we need to check before as well.
                bool sufficientHistoryBeforeUpdate{m_Trend->s_Regression.sufficientHistoryToPredict()};
                TVector paramsDrift(m_Trend->s_Regression.parameters(t));

                m_Trend->s_Regression.add(t, values[0], weight);
                m_Trend->s_Variance = CBasicStatistics::maximumLikelihoodVariance(moments);

                paramsDrift -= TVector(m_Trend->s_Regression.parameters(t));

                if (   sufficientHistoryBeforeUpdate
                    && m_Trend->s_Regression.sufficientHistoryToPredict()
                    && m_Trend->s_LastUpdate != UNSET_LAST_UPDATE)
                {
                    double interval{regressionTime(time, m_Trend->s_LastUpdate)};
                    m_Trend->s_ParameterProcess.add(interval, paramsDrift, TVector(weight * interval));
                }
                m_Trend->s_LastUpdate = time;
            }
            for (std::size_t i = 1u; i <= m; ++i)
            {
                CSeasonalComponent *component{seasonal[i - 1]};
                CComponentErrors *error_{seasonalErrors[i - 1]};
                double wi{weight / component->time().fractionInWindow()};
                component->add(time, values[i], wi);
                error_->add(error, predictions[i - 1], wi);
            }
            for (std::size_t i = m + 1; i <= m + n; ++i)
            {
                CCalendarComponent *component{calendar[i - m - 1]};
                CComponentErrors *error_{calendarErrors[i - m - 1]};
                component->add(time, values[i], weight);
                error_->add(error, predictions[i - 1], weight);
            }
        }
        break;
    case SC_FORECASTING:
    case SC_DISABLED:
        break;
    default:
        LOG_ERROR("Components in a bad state: " << m_Machine.state());
        this->apply(SC_RESET, message);
        break;
    }
}

void CTimeSeriesDecompositionDetail::CComponents::handle(const SDetectedTrend &message)
{
    switch (m_Machine.state())
    {
    case SC_NORMAL:
    case SC_NEW_COMPONENTS:
    {
        if (m_Watcher)
        {
            *m_Watcher = true;
        }

        LOG_DEBUG("Detected long term trend at '" << message.s_Time << "'");
        core_t::TTime time{message.s_Time};
        core_t::TTime lastTime{message.s_LastTime};
        const TRegression &trend{message.s_Test.trend()};
        double variance{message.s_Test.variance()};
        core_t::TTime origin{message.s_Test.origin()};

        m_Trend.reset(new STrend);
        m_Trend->s_Regression = trend;
        m_Trend->s_Variance = variance;
        m_Trend->s_TimeOrigin = origin;
        m_Trend->s_LastUpdate = time;

        this->clearComponentErrors();
        this->apply(SC_ADDED_COMPONENTS, message);
        this->mediator()->forward(SNewComponents(time, lastTime, SNewComponents::E_Trend));
        break;
    }
    case SC_FORECASTING:
    case SC_DISABLED:
        break;
    default:
        LOG_ERROR("Components in a bad state: " << m_Machine.state());
        this->apply(SC_RESET, message);
        break;
    }
}

void CTimeSeriesDecompositionDetail::CComponents::handle(const SDetectedDiurnal &message)
{
    if (this->size() + m_SeasonalComponentSize > this->maxSize())
    {
        return;
    }

    switch (m_Machine.state())
    {
    case SC_NORMAL:
    case SC_NEW_COMPONENTS:
    {
        if (!m_Seasonal)
        {
            m_Seasonal.reset(new SSeasonal);
        }
        if (m_Watcher)
        {
            *m_Watcher = true;
        }

        core_t::TTime time{message.s_Time};
        core_t::TTime lastTime{message.s_LastTime};
        const CDiurnalPeriodicityTest &test{message.s_Test};
        const CPeriodicityTestResult &result{message.s_Result};

        TSeasonalComponentVec &components{m_Seasonal->s_Components};
        TComponentErrorsVec &errors{m_Seasonal->s_PredictionErrors};
        components.erase(std::remove_if(components.begin(), components.end(),
                                        [](const CSeasonalComponent &component)
                                        {
                                            return isDiurnal(component.time().period());
                                        }), components.end());

        CPeriodicityTest::TTimeTimePrMeanVarAccumulatorPrVecVec trends;
        test.trends(result, trends);

        this->clearComponentErrors();
        this->addSeasonalComponents(test, result, time, components, errors);
        this->apply(SC_ADDED_COMPONENTS, message);
        this->mediator()->forward(SNewComponents(time, lastTime, SNewComponents::E_DiurnalSeasonal));
        break;
    }
    case SC_FORECASTING:
    case SC_DISABLED:
        break;
    default:
        LOG_ERROR("Components in a bad state: " << m_Machine.state());
        this->apply(SC_RESET, message);
        break;
    }
}

void CTimeSeriesDecompositionDetail::CComponents::handle(const SDetectedNonDiurnal &message)
{
    if (this->size() + m_SeasonalComponentSize > this->maxSize())
    {
        return;
    }

    switch (m_Machine.state())
    {
    case SC_NORMAL:
    case SC_NEW_COMPONENTS:
    {
        if (!m_Seasonal)
        {
            m_Seasonal.reset(new SSeasonal);
        }

        core_t::TTime time{message.s_Time};
        core_t::TTime lastTime{message.s_LastTime};
        core_t::TTime period{message.s_Test.periods()[0]};
        const CGeneralPeriodicityTest &test{message.s_Test};
        CPeriodicityTestResult result{message.s_Result};

        if (m_Seasonal->haveComponent(period))
        {
            break;
        }
        if (m_Watcher)
        {
            *m_Watcher = true;
        }

        TSeasonalComponentVec &components{m_Seasonal->s_Components};
        TComponentErrorsVec &errors{m_Seasonal->s_PredictionErrors};

        this->clearComponentErrors();
        this->addSeasonalComponents(test, result, time, components, errors);
        this->apply(SC_ADDED_COMPONENTS, message);
        if (message.s_DiscardLongTermTrend)
        {
            m_Trend.reset();
        }
        this->mediator()->forward(SNewComponents(time, lastTime, SNewComponents::E_GeneralSeasonal));
        break;
    }
    case SC_FORECASTING:
    case SC_DISABLED:
        break;
    default:
        LOG_ERROR("Components in a bad state: " << m_Machine.state());
        this->apply(SC_RESET, message);
        break;
    }
}

void CTimeSeriesDecompositionDetail::CComponents::handle(const SDetectedCalendar &message)
{
    if (this->size() + m_CalendarComponentSize > this->maxSize())
    {
        return;
    }

    switch (m_Machine.state())
    {
    case SC_NORMAL:
    case SC_NEW_COMPONENTS:
    {
        if (!m_Calendar)
        {
            m_Calendar.reset(new SCalendar);
        }

        core_t::TTime time{message.s_Time};
        core_t::TTime lastTime{message.s_LastTime};
        CCalendarFeature feature{message.s_Feature};

        if (m_Calendar->haveComponent(feature))
        {
            break;
        }

        TCalendarComponentVec &components{m_Calendar->s_Components};
        TComponentErrorsVec &errors{m_Calendar->s_PredictionErrors};

        this->addCalendarComponent(feature, time, components, errors);
        this->apply(SC_ADDED_COMPONENTS, message);
        this->mediator()->forward(SNewComponents(time, lastTime, SNewComponents::E_CalendarCyclic));
        break;
    }
    case SC_FORECASTING:
    case SC_DISABLED:
        break;
    default:
        LOG_ERROR("Components in a bad state: " << m_Machine.state());
        this->apply(SC_RESET, message);
        break;
    }
}

void CTimeSeriesDecompositionDetail::CComponents::interpolate(const SMessage &message)
{
    core_t::TTime time{message.s_Time};
    core_t::TTime lastTime{message.s_LastTime};

    std::size_t state{m_Machine.state()};

    switch (state)
    {
    case SC_NORMAL:
    case SC_NEW_COMPONENTS:
    case SC_FORECASTING:
        this->canonicalize(time);
        if (this->shouldInterpolate(time, lastTime))
        {
            LOG_TRACE("Interpolating values at " << time);

            if (m_Seasonal)
            {
                m_Seasonal->interpolate(time, lastTime, !this->forecasting());
            }
            if (m_Calendar)
            {
                m_Calendar->interpolate(time, lastTime, !this->forecasting());
            }

            this->apply(SC_INTERPOLATED, message);
        }
        break;
    case SC_DISABLED:
        break;
    default:
        LOG_ERROR("Components in a bad state: " << state);
        this->apply(SC_RESET, message);
        break;
    }
}

void CTimeSeriesDecompositionDetail::CComponents::decayRate(double decayRate)
{
    m_DecayRate = decayRate;
    if (m_Seasonal)
    {
        m_Seasonal->decayRate(decayRate);
    }
    if (m_Calendar)
    {
        m_Calendar->decayRate(decayRate);
    }
}

void CTimeSeriesDecompositionDetail::CComponents::propagateForwards(core_t::TTime start,
                                                                    core_t::TTime end)
{
    if (m_Trend)
    {
        double time{static_cast<double>(end - start) / static_cast<double>(DAY)};
        double factor{std::exp(-m_DecayRate * time)};
        m_Trend->s_Regression.age(factor);
        m_Trend->s_ParameterProcess.age(factor);
    }
    if (m_Seasonal)
    {
        m_Seasonal->propagateForwards(start, end);
    }
    if (m_Calendar)
    {
        m_Calendar->propagateForwards(start, end);
    }
}

bool CTimeSeriesDecompositionDetail::CComponents::forecasting(void) const
{
    return m_Machine.state() == SC_FORECASTING;
}

void CTimeSeriesDecompositionDetail::CComponents::forecast(void)
{
    this->apply(SC_FORECAST, SMessage());
}

bool CTimeSeriesDecompositionDetail::CComponents::initialized(void) const
{
    return m_Trend ? true :
          (m_Seasonal ? m_Seasonal->initialized() :
          (m_Calendar ? m_Calendar->initialized() : false));
}

TTrendCRef CTimeSeriesDecompositionDetail::CComponents::trend(void) const
{
    return m_Trend ? m_Trend->reference() : NO_TREND;
}

const TSeasonalComponentVec &CTimeSeriesDecompositionDetail::CComponents::seasonal(void) const
{
    return m_Seasonal ? m_Seasonal->s_Components : NO_SEASONAL_COMPONENTS;
}

const maths_t::TCalendarComponentVec &CTimeSeriesDecompositionDetail::CComponents::calendar(void) const
{
    return m_Calendar ? m_Calendar->s_Components : NO_CALENDAR_COMPONENTS;
}

double CTimeSeriesDecompositionDetail::CComponents::meanValue(core_t::TTime time) const
{
    return this->initialized() ? (  CBasicStatistics::mean(this->trend().prediction(time, 0.0))
                                  + meanOf(&CSeasonalComponent::meanValue, this->seasonal())) : 0.0;
}

double CTimeSeriesDecompositionDetail::CComponents::meanVariance(void) const
{
    return this->initialized() ?  this->trend().variance()
                                + meanOf(&CSeasonalComponent::meanVariance, this->seasonal()) : 0.0;
}

uint64_t CTimeSeriesDecompositionDetail::CComponents::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, m_Machine);
    seed = CChecksum::calculate(seed, m_DecayRate);
    seed = CChecksum::calculate(seed, m_BucketLength);
    seed = CChecksum::calculate(seed, m_SeasonalComponentSize);
    seed = CChecksum::calculate(seed, m_CalendarComponentSize);
    seed = CChecksum::calculate(seed, m_Trend);
    seed = CChecksum::calculate(seed, m_Seasonal);
    return CChecksum::calculate(seed, m_Calendar);
}

void CTimeSeriesDecompositionDetail::CComponents::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    mem->setName("CComponents");
    core::CMemoryDebug::dynamicSize("m_Trend", m_Trend, mem);
    core::CMemoryDebug::dynamicSize("m_Seasonal", m_Seasonal, mem);
    core::CMemoryDebug::dynamicSize("m_Calendar", m_Calendar, mem);
}

std::size_t CTimeSeriesDecompositionDetail::CComponents::memoryUsage(void) const
{
    return  core::CMemory::dynamicSize(m_Trend)
          + core::CMemory::dynamicSize(m_Seasonal)
          + core::CMemory::dynamicSize(m_Calendar);
}

std::size_t CTimeSeriesDecompositionDetail::CComponents::size(void) const
{
    return (m_Seasonal ? m_Seasonal->size() : 0) + (m_Calendar ? m_Calendar->size() : 0);
}

std::size_t CTimeSeriesDecompositionDetail::CComponents::maxSize(void) const
{
    return MAXIMUM_COMPONENTS * m_SeasonalComponentSize;
}

void CTimeSeriesDecompositionDetail::CComponents::addSeasonalComponents(const CPeriodicityTest &test,
                                                                        const CPeriodicityTestResult &result,
                                                                        core_t::TTime time,
                                                                        TSeasonalComponentVec &components,
                                                                        TComponentErrorsVec &errors) const
{
    using TSeasonalTimePtr = boost::scoped_ptr<CSeasonalTime>;

    double bucketLength{static_cast<double>(m_BucketLength)};

    CPeriodicityTest::TTimeTimePrMeanVarAccumulatorPrVecVec trends;
    test.trends(result, trends);
    components.reserve(components.size() + result.components().size());
    for (std::size_t i = 0u; i < result.components().size(); ++i)
    {
        TSeasonalTimePtr seasonalTime(test.seasonalTime(result.components()[i]));
        components.emplace_back(*seasonalTime, m_SeasonalComponentSize,
                                m_DecayRate, bucketLength, CSplineTypes::E_Natural);
        components.back().initialize(seasonalTime->startOfWindowRepeat(time), time, trends[i]);
    }
    errors.resize(components.size());

    LOG_DEBUG("Detected " << test.print(result));
    LOG_DEBUG("Estimated new periods at '" << time << "'");
    LOG_TRACE("# components = " << components.size());

    COrderings::simultaneousSort(components, errors,
                                 [](const CSeasonalComponent &lhs, const CSeasonalComponent &rhs)
                                 {
                                     return lhs.time().period() < rhs.time().period();
                                 });
}

void CTimeSeriesDecompositionDetail::CComponents::addCalendarComponent(const CCalendarFeature &feature,
                                                                       core_t::TTime time,
                                                                       maths_t::TCalendarComponentVec &components,
                                                                       TComponentErrorsVec &errors) const
{
    double bucketLength{static_cast<double>(m_BucketLength)};
    components.emplace_back(feature, m_CalendarComponentSize,
                            m_DecayRate, bucketLength, CSplineTypes::E_Natural);
    components.back().initialize();
    errors.resize(components.size());

    LOG_DEBUG("Detected feature '" << feature.print() << "'");
    LOG_DEBUG("Estimated new calendar component at '" << time << "'");
}

void CTimeSeriesDecompositionDetail::CComponents::clearComponentErrors(void)
{
    if (m_Seasonal)
    {
        for (auto &&errors : m_Seasonal->s_PredictionErrors)
        {
            errors.clear();
        }
    }
    if (m_Calendar)
    {
        for (auto &&errors : m_Calendar->s_PredictionErrors)
        {
            errors.clear();
        }
    }
}

void CTimeSeriesDecompositionDetail::CComponents::apply(std::size_t symbol, const SMessage &message)
{
    if (symbol == SC_RESET)
    {
        m_Trend.reset();
        m_Seasonal.reset();
        m_Calendar.reset();
    }

    std::size_t old{m_Machine.state()};
    m_Machine.apply(symbol);
    std::size_t state{m_Machine.state()};

    if (state != old)
    {
        LOG_TRACE(SC_STATES[old] << "," << SC_ALPHABET[symbol] << " -> " << SC_STATES[state]);

        switch (state)
        {
        case SC_NORMAL:
        case SC_FORECASTING:
        case SC_NEW_COMPONENTS:
            this->interpolate(message);
            break;
        case SC_DISABLED:
            m_Trend.reset();
            m_Seasonal.reset();
            m_Calendar.reset();
            break;
        default:
            LOG_ERROR("Components in a bad state: " << m_Machine.state());
            this->apply(SC_RESET, message);
            break;
        }
    }
}

bool CTimeSeriesDecompositionDetail::CComponents::shouldInterpolate(core_t::TTime time,
                                                                    core_t::TTime last)
{
    std::size_t state{m_Machine.state()};
    if (state == SC_NEW_COMPONENTS)
    {
        return true;
    }
    return   this->forecasting()
          || (m_Seasonal && m_Seasonal->shouldInterpolate(time, last))
          || (m_Calendar && m_Calendar->shouldInterpolate(time, last));
}

void CTimeSeriesDecompositionDetail::CComponents::shiftOrigin(core_t::TTime time)
{
    if (!this->forecasting())
    {
        time -= static_cast<core_t::TTime>(static_cast<double>(DAY) / m_DecayRate / 2.0);
        if (m_Trend)
        {
            m_Trend->shiftOrigin(time);
        }
        if (m_Seasonal)
        {
            m_Seasonal->shiftOrigin(time);
        }
    }
}

void CTimeSeriesDecompositionDetail::CComponents::canonicalize(core_t::TTime time)
{
    this->shiftOrigin(time);

    if (m_Seasonal && m_Seasonal->prune(time, m_BucketLength))
    {
        m_Seasonal.reset();
    }
    if (m_Calendar && m_Calendar->prune(time, m_BucketLength))
    {
        m_Calendar.reset();
    }

    if (m_Seasonal && m_Trend)
    {
        TSeasonalComponentVec &seasonal{m_Seasonal->s_Components};

        TTimeTimePrDoubleFMap slope;
        slope.reserve(seasonal.size());

        for (auto &&component : seasonal)
        {
            if (component.sufficientHistoryToPredict(time))
            {
                const CSeasonalTime &time_{component.time()};
                double si{component.slope()};
                component.shiftSlope(-si);
                slope[time_.window()] += si;
            }
        }

        LOG_TRACE("slope = " << core::CContainerPrinter::print(slope));
        shiftSlope(slope, m_Trend->s_Regression);
    }
}

void CTimeSeriesDecompositionDetail::CComponents::notifyOnNewComponents(bool *watcher)
{
    m_Watcher = watcher;
}

CTimeSeriesDecompositionDetail::CComponents::CScopeNotifyOnStateChange::CScopeNotifyOnStateChange(CComponents &components) :
        m_Components{components}, m_Watcher{false}
{
    m_Components.notifyOnNewComponents(&m_Watcher);
}

CTimeSeriesDecompositionDetail::CComponents::CScopeNotifyOnStateChange::~CScopeNotifyOnStateChange(void)
{
    m_Components.notifyOnNewComponents(0);
}

bool CTimeSeriesDecompositionDetail::CComponents::CScopeNotifyOnStateChange::changed(void) const
{
    return m_Watcher;
}

bool CTimeSeriesDecompositionDetail::CComponents::CComponentErrors::fromDelimited(const std::string &str)
{
    TFloatMeanAccumulator *state[] =
        {
            &m_MeanErrorWithComponent,
            &m_MeanErrorWithoutComponent
        };

    std::string suffix = str;
    for (std::size_t i = 0u, n = 0; i < 2; ++i, suffix = suffix.substr(n + 1))
    {
        n = suffix.find(CBasicStatistics::EXTERNAL_DELIMITER);
        if (!state[i]->fromDelimited(suffix.substr(0, n)))
        {
            LOG_ERROR("Failed to parse '" << str << "'");
            return false;
        }
    }

    return true;
}

std::string CTimeSeriesDecompositionDetail::CComponents::CComponentErrors::toDelimited(void) const
{
    return  m_MeanErrorWithComponent.toDelimited() + CBasicStatistics::EXTERNAL_DELIMITER
          + m_MeanErrorWithoutComponent.toDelimited() + CBasicStatistics::EXTERNAL_DELIMITER;
}

void CTimeSeriesDecompositionDetail::CComponents::CComponentErrors::add(double error,
                                                                        double prediction,
                                                                        double weight)
{
    double errorWithComponent{winsorise(pow2(error), m_MeanErrorWithComponent)};
    double errorWithoutComponent{winsorise(pow2(error - prediction), m_MeanErrorWithoutComponent)};
    m_MeanErrorWithComponent.add(errorWithComponent, weight);
    m_MeanErrorWithoutComponent.add(errorWithoutComponent, weight);
}

void CTimeSeriesDecompositionDetail::CComponents::CComponentErrors::clear(void)
{
    m_MeanErrorWithComponent = TFloatMeanAccumulator();
    m_MeanErrorWithoutComponent = TFloatMeanAccumulator();
}

bool CTimeSeriesDecompositionDetail::CComponents::CComponentErrors::remove(core_t::TTime bucketLength,
                                                                           CSeasonalComponent &seasonal) const
{
    double count{CBasicStatistics::count(m_MeanErrorWithComponent)};
    double errorWithComponent{CBasicStatistics::mean(m_MeanErrorWithComponent)};
    double errorWithoutComponent{CBasicStatistics::mean(m_MeanErrorWithoutComponent)};
    return   count > static_cast<double>(10 * seasonal.time().period() / bucketLength)
          && std::max(  errorWithoutComponent
                      / errorWithComponent, seasonal.heteroscedasticity()) < 1.5;
}

bool CTimeSeriesDecompositionDetail::CComponents::CComponentErrors::remove(core_t::TTime bucketLength,
                                                                           CCalendarComponent &calendar) const
{
    double count{CBasicStatistics::count(m_MeanErrorWithComponent)};
    double errorWithComponent{CBasicStatistics::mean(m_MeanErrorWithComponent)};
    double errorWithoutComponent{CBasicStatistics::mean(m_MeanErrorWithoutComponent)};
    return   count > static_cast<double>(5 * calendar.feature().window() / bucketLength)
          && std::max(  errorWithoutComponent
                      / errorWithComponent, calendar.heteroscedasticity()) < 1.5;
}

void CTimeSeriesDecompositionDetail::CComponents::CComponentErrors::age(double factor)
{
    m_MeanErrorWithComponent.age(factor);
    m_MeanErrorWithoutComponent.age(factor);
}

uint64_t CTimeSeriesDecompositionDetail::CComponents::CComponentErrors::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, m_MeanErrorWithComponent);
    return CChecksum::calculate(seed, m_MeanErrorWithoutComponent);
}

double CTimeSeriesDecompositionDetail::CComponents::CComponentErrors::winsorise(double squareError,
                                                                                const TFloatMeanAccumulator &variance)
{
    return CBasicStatistics::count(variance) > 10.0 ?
           std::min(squareError, 36.0 * CBasicStatistics::mean(variance)) : squareError;
}

CTimeSeriesDecompositionDetail::CComponents::STrend::STrend(void) :
        s_Variance{0.0}, s_TimeOrigin{0}, s_LastUpdate{UNSET_LAST_UPDATE}
{}

bool CTimeSeriesDecompositionDetail::CComponents::STrend::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser)
{
    s_LastUpdate = UNSET_LAST_UPDATE;
    do
    {
        const std::string &name{traverser.name()};
        RESTORE(REGRESSION_TAG, traverser.traverseSubLevel(
                                    boost::bind(&TRegression::acceptRestoreTraverser, &s_Regression, _1)))
        RESTORE_BUILT_IN(VARIANCE_TAG, s_Variance)
        RESTORE_BUILT_IN(TIME_ORIGIN_TAG, s_TimeOrigin)
        RESTORE_BUILT_IN(LAST_UPDATE_TAG, s_LastUpdate)
        RESTORE(PARAMETER_PROCESS_TAG,
                traverser.traverseSubLevel(
                    boost::bind(&TRegressionParameterProcess::acceptRestoreTraverser, &s_ParameterProcess, _1)))
    }
    while (traverser.next());
    return true;
}

void CTimeSeriesDecompositionDetail::CComponents::STrend::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertLevel(REGRESSION_TAG, boost::bind(&TRegression::acceptPersistInserter, &s_Regression, _1));
    inserter.insertValue(VARIANCE_TAG, s_Variance, core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(TIME_ORIGIN_TAG, s_TimeOrigin);
    inserter.insertValue(LAST_UPDATE_TAG, s_LastUpdate);
    inserter.insertLevel(PARAMETER_PROCESS_TAG,
                         boost::bind(&TRegressionParameterProcess::acceptPersistInserter, &s_ParameterProcess, _1));
}

CTimeSeriesDecompositionDetail::CTrendCRef
CTimeSeriesDecompositionDetail::CComponents::STrend::reference(void) const
{
    return CTrendCRef(s_Regression, s_Variance, s_TimeOrigin, s_LastUpdate, s_ParameterProcess);
}

void CTimeSeriesDecompositionDetail::CComponents::STrend::shiftOrigin(core_t::TTime time)
{
    time = CIntegerTools::floor(time, WEEK);
    double shift{regressionTime(time, s_TimeOrigin)};
    if (shift > 0.0)
    {
        s_Regression.shiftAbscissa(-shift);
        s_TimeOrigin = time;
    }
}

uint64_t CTimeSeriesDecompositionDetail::CComponents::STrend::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, s_Regression);
    seed = CChecksum::calculate(seed, s_Variance);
    return CChecksum::calculate(seed, s_TimeOrigin);
}

bool CTimeSeriesDecompositionDetail::CComponents::SSeasonal::acceptRestoreTraverser(double decayRate,
                                                                                    core_t::TTime bucketLength,
                                                                                    core::CStateRestoreTraverser &traverser)
{
    do
    {
        const std::string &name{traverser.name()};
        RESTORE_NO_ERROR(COMPONENT_TAG, s_Components.emplace_back(
                                            decayRate, static_cast<double>(bucketLength), traverser))
        RESTORE(ERRORS_TAG, core::CPersistUtils::restore(ERRORS_TAG, s_PredictionErrors, traverser))
    }
    while (traverser.next());
    return true;
}

void CTimeSeriesDecompositionDetail::CComponents::SSeasonal::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    for (const auto &component : s_Components)
    {
        inserter.insertLevel(COMPONENT_TAG, boost::bind(
                &CSeasonalComponent::acceptPersistInserter, &component, _1));
    }
    core::CPersistUtils::persist(ERRORS_TAG, s_PredictionErrors, inserter);
}

void CTimeSeriesDecompositionDetail::CComponents::SSeasonal::decayRate(double decayRate)
{
    for (auto &&component : s_Components)
    {
        component.decayRate(decayRate);
    }
}

void CTimeSeriesDecompositionDetail::CComponents::SSeasonal::propagateForwards(core_t::TTime start,
                                                                               core_t::TTime end)
{
    for (std::size_t i = 0u; i < s_Components.size(); ++i)
    {
        core_t::TTime period{s_Components[i].time().period()};
        core_t::TTime a{CIntegerTools::floor(start, period)};
        core_t::TTime b{CIntegerTools::floor(end, period)};
        if (b > a)
        {
            double time{  static_cast<double>(b - a)
                        / static_cast<double>(CTools::truncate(period, DAY, WEEK))};
            s_Components[i].propagateForwardsByTime(time);
            s_PredictionErrors[i].age(std::exp(-s_Components[i].decayRate() * time));
        }
    }
}

std::size_t CTimeSeriesDecompositionDetail::CComponents::SSeasonal::size(void) const
{
    std::size_t result{0};
    for (const auto &component : s_Components)
    {
        result += component.size();
    }
    return result;
}

bool CTimeSeriesDecompositionDetail::CComponents::SSeasonal::haveComponent(core_t::TTime period) const
{
    for (const auto &component : s_Components)
    {
        core_t::TTime reference{component.time().period()};
        if (std::abs(period - reference) < reference / 10)
        {
            return true;
        }
    }
    return false;
}

void CTimeSeriesDecompositionDetail::CComponents::SSeasonal::componentsErrorsAndDeltas(core_t::TTime time,
                                                                                       TSeasonalComponentPtrVec &components,
                                                                                       TComponentErrorsPtrVec &errors,
                                                                                       TDoubleVec &deltas)
{
    std::size_t n = s_Components.size();

    components.reserve(n);
    errors.reserve(n);

    for (std::size_t i = 0u; i < n; ++i)
    {
        if (s_Components[i].time().inWindow(time))
        {
            components.push_back(&s_Components[i]);
            errors.push_back(&s_PredictionErrors[i]);
        }
    }

    deltas.resize(components.size(), 0.0);
    for (std::size_t i = 1u; i < components.size(); ++i)
    {
        core_t::TTime period{components[i-1]->time().period()};
        deltas[i-1] = 0.2 * components[i]->differenceFromMean(time, period);
    }
}

bool CTimeSeriesDecompositionDetail::CComponents::SSeasonal::shouldInterpolate(core_t::TTime time,
                                                                               core_t::TTime last) const
{
    for (const auto &component : s_Components)
    {
        core_t::TTime period{component.time().period()};
        core_t::TTime a{CIntegerTools::floor(last, period)};
        core_t::TTime b{CIntegerTools::floor(time, period)};
        if (b > a)
        {
            return true;
        }
    }
    return false;
}

void CTimeSeriesDecompositionDetail::CComponents::SSeasonal::interpolate(core_t::TTime time,
                                                                         core_t::TTime last,
                                                                         bool refine)
{
    for (auto &&component : s_Components)
    {
        core_t::TTime period{component.time().period()};
        core_t::TTime a{CIntegerTools::floor(last, period)};
        core_t::TTime b{CIntegerTools::floor(time, period)};
        if (b > a || !component.initialized())
        {
            component.interpolate(b, refine);
        }
    }
}

bool CTimeSeriesDecompositionDetail::CComponents::SSeasonal::initialized(void) const
{
    for (const auto &component : s_Components)
    {
        if (component.initialized())
        {
            return true;
        }
    }
    return false;
}

bool CTimeSeriesDecompositionDetail::CComponents::SSeasonal::prune(core_t::TTime time,
                                                                   core_t::TTime bucketLength)
{
    std::size_t n = s_Components.size();

    if (n > 1)
    {
        TTimeTimePrSizeFMap windowed;
        windowed.reserve(n);
        for (const auto &component : s_Components)
        {
            const CSeasonalTime &time_ = component.time();
            if (time_.windowed())
            {
                ++windowed[time_.window()];
            }
        }

        TBoolVec remove(n, false);
        TTimeTimePrDoubleFMap shifts;
        shifts.reserve(n);
        for (std::size_t i = 0u; i < n; ++i)
        {
            const CSeasonalTime &time_ = s_Components[i].time();
            auto j = windowed.find(time_.window());
            if (j == windowed.end() || j->second > 1)
            {
                if (s_PredictionErrors[i].remove(bucketLength, s_Components[i]))
                {
                    LOG_DEBUG("Removing seasonal component"
                              << " with period '" << time_.period() << "' at " << time);
                    remove[i] = true;
                    shifts[time_.window()] += s_Components[i].meanValue();
                    --j->second;
                }
            }
        }

        CSetTools::simultaneousRemoveIf(
                remove, s_Components, s_PredictionErrors, [](bool remove_) { return remove_; });

        for (auto &&shift : shifts)
        {
            if (windowed.count(shift.first) > 0)
            {
                for (auto &&component : s_Components)
                {
                    if (shift.first == component.time().window())
                    {
                        component.shiftLevel(shift.second);
                        break;
                    }
                }
            }
            else
            {
                bool fallback = true;
                for (auto &&component : s_Components)
                {
                    if (!component.time().windowed())
                    {
                        component.shiftLevel(shift.second);
                        fallback = false;
                        break;
                    }
                }
                if (fallback)
                {
                    TTimeTimePrVec shifted;
                    shifted.reserve(s_Components.size());
                    for (auto &&component : s_Components)
                    {
                        const CSeasonalTime &time_ = component.time();
                        if (std::find_if(shifted.begin(), shifted.end(),
                                         [&time_](const TTimeTimePr &window)
                                         {
                                             return !(   time_.windowEnd() <= window.first
                                                      || time_.windowStart() >= window.second);
                                         }) == shifted.end())
                        {
                            component.shiftLevel(shift.second);
                        }
                    }
                }
            }
        }
    }

    return s_Components.empty();
}

void CTimeSeriesDecompositionDetail::CComponents::SSeasonal::shiftOrigin(core_t::TTime time)
{
    for (auto &&component : s_Components)
    {
        component.shiftOrigin(time);
    }
}

uint64_t CTimeSeriesDecompositionDetail::CComponents::SSeasonal::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, s_Components);
    return CChecksum::calculate(seed, s_PredictionErrors);
}

void CTimeSeriesDecompositionDetail::CComponents::SSeasonal::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    mem->setName("SSeasonal");
    core::CMemoryDebug::dynamicSize("s_Components", s_Components, mem);
    core::CMemoryDebug::dynamicSize("s_PredictionErrors", s_PredictionErrors, mem);
}

std::size_t CTimeSeriesDecompositionDetail::CComponents::SSeasonal::memoryUsage(void) const
{
    return core::CMemory::dynamicSize(s_Components) + core::CMemory::dynamicSize(s_PredictionErrors);
}

bool CTimeSeriesDecompositionDetail::CComponents::SCalendar::acceptRestoreTraverser(double decayRate,
                                                                                    core_t::TTime bucketLength,
                                                                                    core::CStateRestoreTraverser &traverser)
{
    do
    {
        const std::string &name{traverser.name()};
        RESTORE_NO_ERROR(COMPONENT_TAG, s_Components.emplace_back(
                                            decayRate, static_cast<double>(bucketLength), traverser))
        RESTORE(ERRORS_TAG, core::CPersistUtils::restore(ERRORS_TAG, s_PredictionErrors, traverser))
    }
    while (traverser.next());
    return true;
}

void CTimeSeriesDecompositionDetail::CComponents::SCalendar::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    for (const auto &component : s_Components)
    {
        inserter.insertLevel(COMPONENT_TAG, boost::bind(
                &CCalendarComponent::acceptPersistInserter, &component, _1));
    }
    core::CPersistUtils::persist(ERRORS_TAG, s_PredictionErrors, inserter);
}

void CTimeSeriesDecompositionDetail::CComponents::SCalendar::decayRate(double decayRate)
{
    for (auto &&component : s_Components)
    {
        component.decayRate(decayRate);
    }
}

void CTimeSeriesDecompositionDetail::CComponents::SCalendar::propagateForwards(core_t::TTime start,
                                                                               core_t::TTime end)
{
    for (std::size_t i = 0u; i < s_Components.size(); ++i)
    {
        core_t::TTime a{CIntegerTools::floor(start, MONTH)};
        core_t::TTime b{CIntegerTools::floor(end, MONTH)};
        if (b > a)
        {
            double time{static_cast<double>(b - a) / static_cast<double>(MONTH)};
            s_Components[i].propagateForwardsByTime(time);
            s_PredictionErrors[i].age(std::exp(-s_Components[i].decayRate() * time));
        }
    }
}

std::size_t CTimeSeriesDecompositionDetail::CComponents::SCalendar::size(void) const
{
    std::size_t result{0};
    for (const auto &component : s_Components)
    {
        result += component.size();
    }
    return result;
}

bool CTimeSeriesDecompositionDetail::CComponents::SCalendar::haveComponent(CCalendarFeature feature) const
{
    for (const auto &component : s_Components)
    {
        if (component.feature() == feature)
        {
            return true;
        }
    }
    return false;
}

void CTimeSeriesDecompositionDetail::CComponents::SCalendar::componentsAndErrors(core_t::TTime time,
                                                                                 TCalendarComponentPtrVec &components,
                                                                                 TComponentErrorsPtrVec &errors)
{
    std::size_t n = s_Components.size();
    components.reserve(n);
    errors.reserve(n);
    for (std::size_t i = 0u; i < n; ++i)
    {
        if (s_Components[i].feature().inWindow(time))
        {
            components.push_back(&s_Components[i]);
            errors.push_back(&s_PredictionErrors[i]);
        }
    }
}

bool CTimeSeriesDecompositionDetail::CComponents::SCalendar::shouldInterpolate(core_t::TTime time,
                                                                               core_t::TTime last) const
{
    for (const auto &component : s_Components)
    {
        CCalendarFeature feature = component.feature();
        if (!feature.inWindow(time) && feature.inWindow(last))
        {
            return true;
        }
    }
    return false;
}

void CTimeSeriesDecompositionDetail::CComponents::SCalendar::interpolate(core_t::TTime time,
                                                                         core_t::TTime last,
                                                                         bool refine)
{
    for (auto &&component : s_Components)
    {
        CCalendarFeature feature = component.feature();
        if (!feature.inWindow(time) && feature.inWindow(last))
        {
            component.interpolate(time - feature.offset(time), refine);
        }
    }
}

bool CTimeSeriesDecompositionDetail::CComponents::SCalendar::initialized(void) const
{
    for (const auto &component : s_Components)
    {
        if (component.initialized())
        {
            return true;
        }
    }
    return false;
}

bool CTimeSeriesDecompositionDetail::CComponents::SCalendar::prune(core_t::TTime time,
                                                                   core_t::TTime bucketLength)
{
    TBoolVec remove(s_Components.size(), false);
    for (std::size_t i = 0u; i < s_Components.size(); ++i)
    {
        if (s_PredictionErrors[i].remove(bucketLength, s_Components[i]))
        {
            LOG_DEBUG("Removing calendar component"
                      << " '" << s_Components[i].feature().print() << "' at " << time);
            remove[i] = true;
        }
    }

    CSetTools::simultaneousRemoveIf(
            remove, s_Components, s_PredictionErrors, [](bool remove_) { return remove_; });

    return s_Components.empty();
}

uint64_t CTimeSeriesDecompositionDetail::CComponents::SCalendar::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, s_Components);
    return CChecksum::calculate(seed, s_PredictionErrors);
}

void CTimeSeriesDecompositionDetail::CComponents::SCalendar::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    mem->setName("SCalendar");
    core::CMemoryDebug::dynamicSize("s_Components", s_Components, mem);
    core::CMemoryDebug::dynamicSize("s_PredictionErrors", s_PredictionErrors, mem);
}

std::size_t CTimeSeriesDecompositionDetail::CComponents::SCalendar::memoryUsage(void) const
{
    return core::CMemory::dynamicSize(s_Components) + core::CMemory::dynamicSize(s_PredictionErrors);
}

}
}
