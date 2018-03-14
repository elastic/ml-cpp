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
#include <maths/CExpandingWindow.h>
#include <maths/CIntegerTools.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraPersist.h>
#include <maths/CPeriodicityHypothesisTests.h>
#include <maths/CRegressionDetail.h>
#include <maths/CSampling.h>
#include <maths/CSeasonalComponentAdaptiveBucketing.h>
#include <maths/CSeasonalTime.h>
#include <maths/CSetTools.h>
#include <maths/CStatisticalTests.h>
#include <maths/CTimeSeriesDecomposition.h>

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

namespace ml {
namespace maths {
namespace {

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
using TComponent5Vec = CPeriodicityHypothesisTestsResult::TComponent5Vec;
using TSeasonalComponentPtrVec = std::vector<CSeasonalComponent*>;
using TCalendarComponentPtrVec = std::vector<CCalendarComponent*>;

const core_t::TTime DAY   = core::constants::DAY;
const core_t::TTime WEEK  = core::constants::WEEK;
const core_t::TTime MONTH = 4 * WEEK;

//! Get the square of \p x.
double pow2(double x) {
    return x * x;
}

//! Compute the mean of \p mean of \p components.
template<typename MEAN_FUNCTION>
double meanOf(MEAN_FUNCTION mean, const TSeasonalComponentVec &components) {
    // We can choose to partition the trend model into windows.
    // In particular, we check for the presence of weekday/end
    // patterns. In this function we want to compute the sum of
    // the mean average of the different components: we use an
    // additive decomposition of the trend. However, if we have
    // detected a partition we want to average the models for
    // the different windows.

    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

    double                unwindowed{0.0};
    TTimeTimePrDoubleFMap windows;
    windows.reserve(components.size());
    for (const auto &component : components) {
        if (component.initialized()) {
            TTimeTimePr window{component.time().window()};
            if (window.second - window.first == component.time().windowRepeat()) {
                unwindowed += (component.*mean)();
            } else {
                windows[window] += (component.*mean)();
            }
        }
    }

    TMeanAccumulator windowed;
    for (const auto &window : windows) {
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
//! \param[in] time The time of value to decompose.
//! \param[in] deltas The delta offset to apply to the difference
//! between each component value and its mean, used to minimize
//! slope in the longer periods.
//! \param[in,out] decomposition Updated to contain the value to
//! add to each by component.
//! \param[out] predictions Filled in with the component predictions.
//! \param[out] error Filled in with the prediction error.
//! \param[out] scale Filled in with the normalization scaling.
void decompose(const CTrendComponent &trend,
               const TSeasonalComponentPtrVec &seasonal,
               const TCalendarComponentPtrVec &calendar,
               core_t::TTime time,
               const TDoubleVec &deltas,
               TDoubleVec &decomposition,
               TDoubleVec &predictions,
               double &error,
               double &scale) {
    std::size_t m{seasonal.size()};
    std::size_t n{calendar.size()};

    double     x0{CBasicStatistics::mean(trend.value(time, 0.0))};
    TDoubleVec x(m + n);
    double     xhat{x0};
    for (std::size_t i = 0u; i < m; ++i) {
        x[i]  = CBasicStatistics::mean(seasonal[i]->value(time, 0.0));
        xhat += x[i];
    }
    for (std::size_t i = m; i < m + n; ++i) {
        x[i]  = CBasicStatistics::mean(calendar[i - m]->value(time, 0.0));
        xhat += x[i];
    }

    // Note we are adding on the a proportion of the error to the
    // target value for each component. This constant controls the
    // proportion of the overall error we add. There is no need
    // to arrange for the sum error added to all components to be
    // equal to the actual error to avoid bias: noise will still
    // average down to zero (since the errors will be both positive
    // and negative). It will however affect the variance in the
    // limit the trend has been fit. This can be thought of as a
    // trade off between the rate at which each component reacts
    // to errors verses the error variance in the steady state with
    // smaller values of Z corresponding to greater responsiveness.
    double Z{std::max(0.5 * static_cast<double>(m + n + 1), 1.0)};

    error = decomposition[0] - xhat;
    decomposition[0] = x0 + (decomposition[0] - xhat) / Z;
    for (std::size_t i = 0u; i < m; ++i) {
        predictions[i] = x[i] - seasonal[i]->meanValue();
        decomposition[i + 1] = x[i] + (decomposition[i + 1] - xhat) / Z + deltas[i];
    }
    for (std::size_t i = m; i < m + n; ++i) {
        predictions[i] = x[i] - calendar[i - m]->meanValue();
        decomposition[i + 1] = x[i] + (decomposition[i + 1] - xhat) / Z;
    }

    // Because we add in more than the prediction error across the
    // different components, i.e. because Z < m + n + 1, we end up
    // with a bias in our variance estimates. We can mostly correct
    // the bias by scaling the variance estimate, but need to calculate
    // the scale.
    scale = Z / static_cast<double>(m + n + 1);
}

//! Propagate a test forwards to account for \p end - \p start
//! elapsed time in steps or size \p step.
template<typename T>
void stepwisePropagateForwards(core_t::TTime step,
                               core_t::TTime start,
                               core_t::TTime end,
                               const T &target) {
    if (target) {
        start = CIntegerTools::floor(start, step);
        end   = CIntegerTools::floor(end, step);
        if (end > start) {
            double time{static_cast<double>(end - start) / static_cast<double>(step)};
            target->propagateForwardsByTime(time);
        }
    }
}

//! Apply the common shift to the slope of \p trend.
void shiftSlope(const TTimeTimePrDoubleFMap &slopes,
                double decayRate,
                CTrendComponent &trend) {
    CBasicStatistics::CMinMax<double> minmax;
    for (const auto &slope : slopes) {
        minmax.add(slope.second);
    }
    double shift{minmax.signMargin()};
    if (shift != 0.0) {
        trend.shiftSlope(decayRate, shift);
    }
}

// Periodicity Test State Machine

// States
const std::size_t PT_INITIAL     = 0;
const std::size_t PT_TEST        = 1;
const std::size_t PT_NOT_TESTING = 2;
const std::size_t PT_ERROR       = 3;
const TStrVec     PT_STATES{"INITIAL", "TEST", "NOT_TESTING", "ERROR" };
// Alphabet
const std::size_t PT_NEW_VALUE = 0;
const std::size_t PT_RESET     = 1;
const TStrVec     PT_ALPHABET{"NEW_VALUE", "RESET"};
// Transition Function
const TSizeVecVec PT_TRANSITION_FUNCTION
{
    TSizeVec{PT_TEST,    PT_TEST,    PT_NOT_TESTING, PT_ERROR  },
    TSizeVec{PT_INITIAL, PT_INITIAL, PT_NOT_TESTING, PT_INITIAL}
};

// Calendar Cyclic Test State Machine

// States
const std::size_t CC_INITIAL     = 0;
const std::size_t CC_TEST        = 1;
const std::size_t CC_NOT_TESTING = 2;
const std::size_t CC_ERROR       = 3;
const TStrVec     CC_STATES{"INITIAL", "TEST", "NOT_TESTING", "ERROR"};
// Alphabet
const std::size_t CC_NEW_VALUE = 0;
const std::size_t CC_RESET     = 1;
const TStrVec     CC_ALPHABET{"NEW_VALUE", "RESET"};
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
const std::size_t SC_DISABLED       = 2;
const std::size_t SC_ERROR          = 3;
const TStrVec     SC_STATES{"NEW_COMPONENTS", "NORMAL", "DISABLED", "ERROR"};
// Alphabet
const std::size_t SC_ADDED_COMPONENTS = 0;
const std::size_t SC_INTERPOLATED     = 1;
const std::size_t SC_RESET            = 2;
const TStrVec     SC_ALPHABET{"ADDED_COMPONENTS", "INTERPOLATED", "RESET"};
// Transition Function
const TSizeVecVec SC_TRANSITION_FUNCTION
{
    TSizeVec{SC_NEW_COMPONENTS, SC_NEW_COMPONENTS, SC_DISABLED, SC_ERROR },
    TSizeVec{SC_NORMAL,         SC_NORMAL,         SC_DISABLED, SC_ERROR },
    TSizeVec{SC_NORMAL,         SC_NORMAL,         SC_NORMAL,   SC_NORMAL}
};

const std::string VERSION_6_3_TAG("6.3");

// Periodicity Test Tags
// Version 6.3
const std::string PERIODICITY_TEST_MACHINE_6_3_TAG{"a"};
const std::string SHORT_WINDOW_6_3_TAG{"b"};
const std::string LONG_WINDOW_6_3_TAG{"c"};
// Old versions can't be restored.

// Calendar Cyclic Test Tags
// Version 6.3
const std::string CALENDAR_TEST_MACHINE_6_3_TAG{"a"};
const std::string LAST_MONTH_6_3_TAG{"b"};
const std::string CALENDAR_TEST_6_3_TAG{"c"};
// These work for all versions.

// Components Tags
// Version 6.3
const std::string COMPONENTS_MACHINE_6_3_TAG{"a"};
const std::string DECAY_RATE_6_3_TAG{"b"};
const std::string TREND_6_3_TAG{"c"};
const std::string SEASONAL_6_3_TAG{"d"};
const std::string CALENDAR_6_3_TAG{"e"};
const std::string COMPONENT_6_3_TAG{"f"};
const std::string ERRORS_6_3_TAG{"g"};
const std::string MEAN_VARIANCE_SCALE_6_3_TAG{"h"};
const std::string MOMENTS_6_3_TAG{"i"};
const std::string MOMENTS_MINUS_TREND_6_3_TAG{"j"};
const std::string USING_TREND_FOR_PREDICTION_6_3_TAG{"k"};
// Version < 6.3
const std::string COMPONENTS_MACHINE_OLD_TAG{"a"};
const std::string TREND_OLD_TAG{"b"};
const std::string SEASONAL_OLD_TAG{"c"};
const std::string CALENDAR_OLD_TAG{"d"};
const std::string COMPONENT_OLD_TAG{"e"};
const std::string ERRORS_OLD_TAG{"f"};
const std::string REGRESSION_OLD_TAG{"g"};
const std::string VARIANCE_OLD_TAG{"h"};
const std::string TIME_ORIGIN_OLD_TAG{"i"};
const std::string LAST_UPDATE_OLD_TAG{"j"};

//////////////////////// Upgrade to Version 6.3 ////////////////////////

const double MODEL_WEIGHT_UPGRADING_TO_VERSION_6p3{48.0};

bool upgradeTrendModelToVersion6p3(const core_t::TTime bucketLength,
                                   CTrendComponent &trend,
                                   core::CStateRestoreTraverser &traverser) {
    using TRegression = CRegression::CLeastSquaresOnline<3, double>;

    TRegression   regression;
    double        variance{0.0};
    core_t::TTime origin{0};
    core_t::TTime lastUpdate{0};
    do {
        const std::string &name{traverser.name()};
        RESTORE(REGRESSION_OLD_TAG, traverser.traverseSubLevel(boost::bind(
                                                                   &TRegression::acceptRestoreTraverser, &regression, _1)))
        RESTORE_BUILT_IN(VARIANCE_OLD_TAG, variance)
        RESTORE_BUILT_IN(TIME_ORIGIN_OLD_TAG, origin)
        RESTORE_BUILT_IN(LAST_UPDATE_OLD_TAG, lastUpdate)
    } while (traverser.next());

    // Generate some samples from the old trend model.

    double weight{MODEL_WEIGHT_UPGRADING_TO_VERSION_6p3 * static_cast<double>(bucketLength)
                  / static_cast<double>(4 * WEEK)};

    CPRNG::CXorOShiro128Plus rng;
    for (core_t::TTime time = lastUpdate - 4 * WEEK;
         time < lastUpdate;
         time += bucketLength) {
        double time_{static_cast<double>(time - origin) / static_cast<double>(WEEK)};
        double sample{  regression.predict(time_)
                        + CSampling::normalSample(rng, 0.0, variance)};
        trend.add(time, sample, weight);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////

// Constants
const core_t::TTime         FOREVER{boost::numeric::bounds<core_t::TTime>::highest()};
const std::size_t           MAXIMUM_COMPONENTS{8};
const TSeasonalComponentVec NO_SEASONAL_COMPONENTS;
const TCalendarComponentVec NO_CALENDAR_COMPONENTS;

}

//////// SMessage ////////

CTimeSeriesDecompositionDetail::SMessage::SMessage(core_t::TTime time, core_t::TTime lastTime) :
    s_Time{time}, s_LastTime{lastTime} {
}

//////// SAddValue ////////

CTimeSeriesDecompositionDetail::SAddValue::SAddValue(core_t::TTime time,
                                                     core_t::TTime lastTime,
                                                     double value,
                                                     const maths_t::TWeightStyleVec &weightStyles,
                                                     const maths_t::TDouble4Vec &weights,
                                                     double trend,
                                                     double seasonal,
                                                     double calendar,
                                                     const TPredictor &predictor,
                                                     const CPeriodicityHypothesisTestsConfig &periodicityTestConfig) :
    SMessage{time, lastTime},
    s_Value{value},
    s_WeightStyles{weightStyles},
    s_Weights{weights},
    s_Trend{trend},
    s_Seasonal{seasonal},
    s_Calendar{calendar},
    s_Predictor{predictor},
    s_PeriodicityTestConfig{periodicityTestConfig} {
}

//////// SDetectedSeasonal ////////

CTimeSeriesDecompositionDetail::SDetectedSeasonal::SDetectedSeasonal(core_t::TTime time,
                                                                     core_t::TTime lastTime,
                                                                     const CPeriodicityHypothesisTestsResult &result,
                                                                     const CExpandingWindow &window,
                                                                     const TPredictor &predictor) :
    SMessage{time, lastTime},
    s_Result{result},
    s_Window{window},
    s_Predictor{predictor} {
}

//////// SDetectedCalendar ////////

CTimeSeriesDecompositionDetail::SDetectedCalendar::SDetectedCalendar(core_t::TTime time,
                                                                     core_t::TTime lastTime,
                                                                     CCalendarFeature feature) :
    SMessage{time, lastTime}, s_Feature{feature} {
}

//////// SNewComponent ////////

CTimeSeriesDecompositionDetail::SNewComponents::SNewComponents(core_t::TTime time,
                                                               core_t::TTime lastTime,
                                                               EComponent component) :
    SMessage{time, lastTime}, s_Component{component} {
}

//////// CHandler ////////

CTimeSeriesDecompositionDetail::CHandler::CHandler(void) : m_Mediator{0} {
}
CTimeSeriesDecompositionDetail::CHandler::~CHandler(void) {
}

void CTimeSeriesDecompositionDetail::CHandler::handle(const SAddValue & /*message*/) {
}

void CTimeSeriesDecompositionDetail::CHandler::handle(const SDetectedSeasonal & /*message*/) {
}

void CTimeSeriesDecompositionDetail::CHandler::handle(const SDetectedCalendar & /*message*/) {
}

void CTimeSeriesDecompositionDetail::CHandler::handle(const SNewComponents & /*message*/) {
}

void CTimeSeriesDecompositionDetail::CHandler::mediator(CMediator *mediator) {
    m_Mediator = mediator;
}

CTimeSeriesDecompositionDetail::CMediator *CTimeSeriesDecompositionDetail::CHandler::mediator(void) const {
    return m_Mediator;
}

//////// CMediator ////////

template<typename M>
void CTimeSeriesDecompositionDetail::CMediator::forward(const M &message) const
{
    for (CHandler &handler : m_Handlers) {
        handler.handle(message);
    }
}

void CTimeSeriesDecompositionDetail::CMediator::registerHandler(CHandler &handler) {
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

//////// CPeriodicityTest ////////

CTimeSeriesDecompositionDetail::CPeriodicityTest::CPeriodicityTest(double decayRate,
                                                                   core_t::TTime bucketLength) :
    m_Machine{core::CStateMachine::create(
                  PT_ALPHABET, PT_STATES, PT_TRANSITION_FUNCTION,
                  bucketLength > LONG_BUCKET_LENGTHS.back() ? PT_NOT_TESTING : PT_INITIAL)},
    m_DecayRate{decayRate},
    m_BucketLength{bucketLength} {
}

CTimeSeriesDecompositionDetail::CPeriodicityTest::CPeriodicityTest(const CPeriodicityTest &other) :
    m_Machine{other.m_Machine},
    m_DecayRate{other.m_DecayRate},
    m_BucketLength{other.m_BucketLength} {
    // Note that m_Windows is an array.
    for (std::size_t i = 0u; i < other.m_Windows.size(); ++i) {
        if (other.m_Windows[i]) {
            m_Windows[i] = boost::make_shared<CExpandingWindow>(*other.m_Windows[i]);
        }
    }
}

bool CTimeSeriesDecompositionDetail::CPeriodicityTest::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser) {
    do {
        const std::string &name{traverser.name()};
        RESTORE(PERIODICITY_TEST_MACHINE_6_3_TAG, traverser.traverseSubLevel(
                    boost::bind(&core::CStateMachine::acceptRestoreTraverser, &m_Machine, _1)))
        RESTORE_SETUP_TEARDOWN(SHORT_WINDOW_6_3_TAG,
                               m_Windows[E_Short].reset(this->newWindow(E_Short)),
                               m_Windows[E_Short] && traverser.traverseSubLevel(
                                   boost::bind(&CExpandingWindow::acceptRestoreTraverser,
                                               m_Windows[E_Short].get(), _1)),
                               /**/)
        RESTORE_SETUP_TEARDOWN(LONG_WINDOW_6_3_TAG,
                               m_Windows[E_Long].reset(this->newWindow(E_Long)),
                               m_Windows[E_Long] && traverser.traverseSubLevel(
                                   boost::bind(&CExpandingWindow::acceptRestoreTraverser,
                                               m_Windows[E_Long].get(), _1)),
                               /**/)
    } while (traverser.next());
    return true;
}

void CTimeSeriesDecompositionDetail::CPeriodicityTest::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertLevel(PERIODICITY_TEST_MACHINE_6_3_TAG,
                         boost::bind(&core::CStateMachine::acceptPersistInserter, &m_Machine, _1));
    if (m_Windows[E_Short]) {
        inserter.insertLevel(SHORT_WINDOW_6_3_TAG, boost::bind(
                                 &CExpandingWindow::acceptPersistInserter, m_Windows[E_Short].get(), _1));
    }
    if (m_Windows[E_Long]) {
        inserter.insertLevel(LONG_WINDOW_6_3_TAG, boost::bind(
                                 &CExpandingWindow::acceptPersistInserter, m_Windows[E_Long].get(), _1));
    }
}

void CTimeSeriesDecompositionDetail::CPeriodicityTest::swap(CPeriodicityTest &other) {
    std::swap(m_Machine, other.m_Machine);
    std::swap(m_DecayRate, other.m_DecayRate);
    std::swap(m_BucketLength, other.m_BucketLength);
    m_Windows[E_Short].swap(other.m_Windows[E_Short]);
    m_Windows[E_Long].swap(other.m_Windows[E_Long]);
}

void CTimeSeriesDecompositionDetail::CPeriodicityTest::handle(const SAddValue &message) {
    core_t::TTime                   time{message.s_Time};
    double                          value{message.s_Value};
    const maths_t::TWeightStyleVec &weightStyles{message.s_WeightStyles};
    const maths_t::TDouble4Vec &    weights{message.s_Weights};
    double                          weight{maths_t::countForUpdate(weightStyles, weights)};

    this->test(message);

    switch (m_Machine.state()) {
        case PT_TEST:
            for (auto &&window : m_Windows) {
                if (window) {
                    window->add(time, value, weight);
                }
            }
            break;
        case PT_NOT_TESTING:
            break;
        case PT_INITIAL:
            this->apply(PT_NEW_VALUE, message);
            this->handle(message);
            break;
        default:
            LOG_ERROR("Test in a bad state: " << m_Machine.state());
            this->apply(PT_RESET, message);
            break;
    }
}

void CTimeSeriesDecompositionDetail::CPeriodicityTest::handle(const SNewComponents & /*message*/) {
    // This can be a no-op because we always maintain the raw time
    // series values in the windows and apply corrections for other
    // components only when we test.
}

void CTimeSeriesDecompositionDetail::CPeriodicityTest::test(const SAddValue &message) {
    core_t::TTime                            time{message.s_Time};
    core_t::TTime                            lastTime{message.s_LastTime};
    const TPredictor &                       predictor{message.s_Predictor};
    const CPeriodicityHypothesisTestsConfig &config{message.s_PeriodicityTestConfig};

    switch (m_Machine.state()) {
        case PT_TEST:
            for (const auto &window : m_Windows) {
                if (this->shouldTest(window, time)) {
                    TFloatMeanAccumulatorVec          values(window->valuesMinusPrediction(predictor));
                    core_t::TTime                     start{CIntegerTools::floor(window->startTime(), m_BucketLength)};
                    core_t::TTime                     bucketLength{window->bucketLength()};
                    CPeriodicityHypothesisTestsResult result{testForPeriods(config, start, bucketLength, values)};
                    if (result.periodic()) {
                        this->mediator()->forward(SDetectedSeasonal{time, lastTime, result, *window, predictor});
                    }
                }
            }
            break;
        case PT_NOT_TESTING:
        case PT_INITIAL:
            break;
        default:
            LOG_ERROR("Test in a bad state: " << m_Machine.state());
            this->apply(PT_RESET, message);
            break;
    }
}

void CTimeSeriesDecompositionDetail::CPeriodicityTest::propagateForwards(core_t::TTime start,
                                                                         core_t::TTime end) {
    stepwisePropagateForwards(DAY,  start, end, m_Windows[E_Short]);
    stepwisePropagateForwards(WEEK, start, end, m_Windows[E_Long]);
}

uint64_t CTimeSeriesDecompositionDetail::CPeriodicityTest::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, m_Machine);
    seed = CChecksum::calculate(seed, m_DecayRate);
    seed = CChecksum::calculate(seed, m_BucketLength);
    return CChecksum::calculate(seed, m_Windows);
}

void CTimeSeriesDecompositionDetail::CPeriodicityTest::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    mem->setName("CPeriodicityTest");
    core::CMemoryDebug::dynamicSize("m_Windows", m_Windows, mem);
}

std::size_t CTimeSeriesDecompositionDetail::CPeriodicityTest::memoryUsage(void) const
{
    std::size_t usage{core::CMemory::dynamicSize(m_Windows)};
    if (m_Machine.state() == PT_INITIAL) {
        usage += this->extraMemoryOnInitialization();
    }
    return usage;
}

std::size_t CTimeSeriesDecompositionDetail::CPeriodicityTest::extraMemoryOnInitialization(void) const
{
    static std::size_t result{0};
    if (result == 0) {
        for (auto i : {E_Short, E_Long}) {
            TExpandingWindowPtr window(this->newWindow(i));
            result += core::CMemory::dynamicSize(window);
        }
    }
    return result;
}

void CTimeSeriesDecompositionDetail::CPeriodicityTest::apply(std::size_t symbol,
                                                             const SMessage &message) {
    core_t::TTime time{message.s_Time};

    std::size_t old{m_Machine.state()};
    m_Machine.apply(symbol);
    std::size_t state{m_Machine.state()};

    if (state != old) {
        LOG_TRACE(PT_STATES[old] << "," << PT_ALPHABET[symbol] << " -> " << PT_STATES[state]);

        auto initialize = [this](core_t::TTime time_) {
                              for (auto i : {E_Short, E_Long}) {
                                  m_Windows[i].reset(this->newWindow(i));
                                  if (m_Windows[i]) {
                                      m_Windows[i]->initialize(time_);
                                  }
                              }
                          };

        switch (state) {
            case PT_TEST:
                if (std::all_of(m_Windows.begin(), m_Windows.end(),
                                [](const TExpandingWindowPtr &window) {
                        return !window;
                    })) {
                    initialize(time);
                }
                break;
            case PT_INITIAL:
                initialize(time);
                break;
            case PT_NOT_TESTING:
                m_Windows[0].reset();
                m_Windows[1].reset();
                break;
            default:
                LOG_ERROR("Test in a bad state: " << state);
                this->apply(PT_RESET, message);
                break;
        }
    }
}

bool CTimeSeriesDecompositionDetail::CPeriodicityTest::shouldTest(const TExpandingWindowPtr &window,
                                                                  core_t::TTime time) const
{
    // We need to test more frequently than when we compress, because
    // this only happens after we've seen 336 buckets, this would thus
    // significantly delay when we first detect a daily periodic for
    // longer bucket lengths otherwise.

    auto shouldTest = [this, time](const TExpandingWindowPtr &window_) {
                          core_t::TTime length{time - window_->startTime()};
                          for (auto lengthToTest : {3 * DAY, 1 * WEEK, 2 * WEEK}) {
                              if (length >= lengthToTest && length < lengthToTest + m_BucketLength) {
                                  return true;
                              }
                          }
                          return false;
                      };
    return window && (window->needToCompress(time) || shouldTest(window));
}

CExpandingWindow *CTimeSeriesDecompositionDetail::CPeriodicityTest::newWindow(ETest test) const {
    using TTimeCRng = CExpandingWindow::TTimeCRng;

    auto newWindow = [this](const TTimeVec &bucketLengths) {
                         if (m_BucketLength <= bucketLengths.back()) {
                             std::ptrdiff_t a{std::lower_bound(bucketLengths.begin(),
                                                               bucketLengths.end(),
                                                               m_BucketLength) - bucketLengths.begin()};
                             std::size_t b{bucketLengths.size()};
                             TTimeCRng   bucketLengths_(bucketLengths, a, b);
                             return new CExpandingWindow(m_BucketLength, bucketLengths_, 336, m_DecayRate);
                         }
                         return static_cast<CExpandingWindow*>(0);
                     };

    switch (test) {
        case E_Short: return newWindow(SHORT_BUCKET_LENGTHS);
        case E_Long:  return newWindow(LONG_BUCKET_LENGTHS);
    }
    return 0;
}

const TTimeVec CTimeSeriesDecompositionDetail::CPeriodicityTest::SHORT_BUCKET_LENGTHS
{
    1, 5, 10, 30, 60, 300, 600, 1800, 3600
};
const TTimeVec CTimeSeriesDecompositionDetail::CPeriodicityTest::LONG_BUCKET_LENGTHS
{
    7200, 21600, 43200, 86400, 172800, 345600
};

//////// CCalendarCyclic ////////

CTimeSeriesDecompositionDetail::CCalendarTest::CCalendarTest(double decayRate,
                                                             core_t::TTime bucketLength) :
    m_Machine{core::CStateMachine::create(CC_ALPHABET, CC_STATES, CC_TRANSITION_FUNCTION,
                                          bucketLength > DAY ? CC_NOT_TESTING : CC_INITIAL)},
    m_DecayRate{decayRate},
    m_LastMonth{} {
}

CTimeSeriesDecompositionDetail::CCalendarTest::CCalendarTest(const CCalendarTest &other) :
    m_Machine{other.m_Machine},
    m_DecayRate{other.m_DecayRate},
    m_LastMonth{other.m_LastMonth},
    m_Test{other.m_Test ? new CCalendarCyclicTest(*other.m_Test) : 0} {
}

bool CTimeSeriesDecompositionDetail::CCalendarTest::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser) {
    do {
        const std::string &name{traverser.name()};
        RESTORE(CALENDAR_TEST_MACHINE_6_3_TAG, traverser.traverseSubLevel(
                    boost::bind(&core::CStateMachine::acceptRestoreTraverser, &m_Machine, _1)))
        RESTORE_BUILT_IN(LAST_MONTH_6_3_TAG, m_LastMonth);
        RESTORE_SETUP_TEARDOWN(CALENDAR_TEST_6_3_TAG,
                               m_Test.reset(new CCalendarCyclicTest(m_DecayRate)),
                               traverser.traverseSubLevel(
                                   boost::bind(&CCalendarCyclicTest::acceptRestoreTraverser, m_Test.get(), _1)),
                               /**/)
    } while (traverser.next());
    return true;
}

void CTimeSeriesDecompositionDetail::CCalendarTest::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertLevel(CALENDAR_TEST_MACHINE_6_3_TAG,
                         boost::bind(&core::CStateMachine::acceptPersistInserter, &m_Machine, _1));
    inserter.insertValue(LAST_MONTH_6_3_TAG, m_LastMonth);
    if (m_Test) {
        inserter.insertLevel(CALENDAR_TEST_6_3_TAG, boost::bind(
                                 &CCalendarCyclicTest::acceptPersistInserter, m_Test.get(), _1));
    }
}

void CTimeSeriesDecompositionDetail::CCalendarTest::swap(CCalendarTest &other) {
    std::swap(m_Machine, other.m_Machine);
    std::swap(m_DecayRate, other.m_DecayRate);
    std::swap(m_LastMonth, other.m_LastMonth);
    m_Test.swap(other.m_Test);
}

void CTimeSeriesDecompositionDetail::CCalendarTest::handle(const SAddValue &message) {
    core_t::TTime                   time{message.s_Time};
    double                          error{message.s_Value - message.s_Trend - message.s_Seasonal - message.s_Calendar};
    const maths_t::TWeightStyleVec &weightStyles{message.s_WeightStyles};
    const maths_t::TDouble4Vec &    weights{message.s_Weights};

    this->test(message);

    switch (m_Machine.state()) {
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

void CTimeSeriesDecompositionDetail::CCalendarTest::handle(const SNewComponents &message) {
    if (m_Machine.state() != CC_NOT_TESTING) {
        switch (message.s_Component) {
            case SNewComponents::E_GeneralSeasonal:
            case SNewComponents::E_DiurnalSeasonal:
                this->apply(CC_RESET, message);
                break;
            case SNewComponents::E_CalendarCyclic:
                break;
        }
    }
}

void CTimeSeriesDecompositionDetail::CCalendarTest::test(const SMessage &message) {
    core_t::TTime time{message.s_Time};
    core_t::TTime lastTime{message.s_LastTime};

    if (this->shouldTest(time)) {
        switch (m_Machine.state()) {
            case CC_TEST: {
                if (CCalendarCyclicTest::TOptionalFeature feature = m_Test->test()) {
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
                                                                      core_t::TTime end) {
    stepwisePropagateForwards(DAY, start, end, m_Test);
}

uint64_t CTimeSeriesDecompositionDetail::CCalendarTest::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, m_Machine);
    seed = CChecksum::calculate(seed, m_DecayRate);
    seed = CChecksum::calculate(seed, m_LastMonth);
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
    if (m_Machine.state() == CC_INITIAL) {
        usage += this->extraMemoryOnInitialization();
    }
    return usage;
}

std::size_t CTimeSeriesDecompositionDetail::CCalendarTest::extraMemoryOnInitialization(void) const
{
    static std::size_t result{0};
    if (result == 0) {
        TCalendarCyclicTestPtr test(new CCalendarCyclicTest(m_DecayRate));
        result = core::CMemory::dynamicSize(test);
    }
    return result;
}

void CTimeSeriesDecompositionDetail::CCalendarTest::apply(std::size_t symbol, const SMessage &message) {
    core_t::TTime time{message.s_Time};

    std::size_t old{m_Machine.state()};
    m_Machine.apply(symbol);
    std::size_t state{m_Machine.state()};

    if (state != old) {
        LOG_TRACE(CC_STATES[old] << "," << CC_ALPHABET[symbol] << " -> " << CC_STATES[state]);

        switch (state) {
            case CC_TEST:
                if (!m_Test) {
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

bool CTimeSeriesDecompositionDetail::CCalendarTest::shouldTest(core_t::TTime time) {
    int month{this->month(time)};
    if (month == (m_LastMonth + 1) % 12) {
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

//////// CComponents ////////

CTimeSeriesDecompositionDetail::CComponents::CComponents(double decayRate,
                                                         core_t::TTime bucketLength,
                                                         std::size_t seasonalComponentSize) :
    m_Machine{core::CStateMachine::create(SC_ALPHABET, SC_STATES, SC_TRANSITION_FUNCTION, SC_NORMAL)},
    m_DecayRate{decayRate},
    m_BucketLength{bucketLength},
    m_SeasonalComponentSize{seasonalComponentSize},
    m_CalendarComponentSize{seasonalComponentSize / 3},
    m_Trend{decayRate},
    m_UsingTrendForPrediction{false},
    m_Watcher{0} {
}

CTimeSeriesDecompositionDetail::CComponents::CComponents(const CComponents &other) :
    m_Machine{other.m_Machine},
    m_DecayRate{other.m_DecayRate},
    m_BucketLength{other.m_BucketLength},
    m_SeasonalComponentSize{other.m_SeasonalComponentSize},
    m_CalendarComponentSize{other.m_CalendarComponentSize},
    m_Trend{other.m_Trend},
    m_Seasonal{other.m_Seasonal ? new SSeasonal{*other.m_Seasonal} : 0},
    m_Calendar{other.m_Calendar ? new SCalendar{*other.m_Calendar} : 0},
    m_MeanVarianceScale{other.m_MeanVarianceScale},
    m_Moments{other.m_Moments},
    m_MomentsMinusTrend{other.m_MomentsMinusTrend},
    m_UsingTrendForPrediction{other.m_UsingTrendForPrediction},
    m_Watcher{0} {
}

bool CTimeSeriesDecompositionDetail::CComponents::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser) {
    if (traverser.name() == VERSION_6_3_TAG) {
        while (traverser.next()) {
            const std::string &name{traverser.name()};
            RESTORE(COMPONENTS_MACHINE_6_3_TAG, traverser.traverseSubLevel(
                        boost::bind(&core::CStateMachine::acceptRestoreTraverser, &m_Machine, _1)));
            RESTORE_BUILT_IN(DECAY_RATE_6_3_TAG, m_DecayRate);
            RESTORE(TREND_6_3_TAG, traverser.traverseSubLevel(boost::bind(
                                                                  &CTrendComponent::acceptRestoreTraverser, &m_Trend, _1)))
            RESTORE_SETUP_TEARDOWN(SEASONAL_6_3_TAG,
                                   m_Seasonal.reset(new SSeasonal),
                                   traverser.traverseSubLevel(boost::bind(
                                                                  &SSeasonal::acceptRestoreTraverser,
                                                                  m_Seasonal.get(), m_DecayRate, m_BucketLength, _1)),
                                   /**/)
            RESTORE_SETUP_TEARDOWN(CALENDAR_6_3_TAG,
                                   m_Calendar.reset(new SCalendar),
                                   traverser.traverseSubLevel(boost::bind(
                                                                  &SCalendar::acceptRestoreTraverser,
                                                                  m_Calendar.get(), m_DecayRate, m_BucketLength, _1)),
                                   /**/)
            RESTORE(MEAN_VARIANCE_SCALE_6_3_TAG, m_MeanVarianceScale.fromDelimited(traverser.value()))
            RESTORE(MOMENTS_6_3_TAG, m_Moments.fromDelimited(traverser.value()));
            RESTORE(MOMENTS_MINUS_TREND_6_3_TAG, m_MomentsMinusTrend.fromDelimited(traverser.value()));
            RESTORE_BUILT_IN(USING_TREND_FOR_PREDICTION_6_3_TAG, m_UsingTrendForPrediction)
        }

        this->decayRate(m_DecayRate);
    } else {
        // There is no version string this is historic state.
        do {
            const std::string &name{traverser.name()};
            RESTORE(COMPONENTS_MACHINE_OLD_TAG, traverser.traverseSubLevel(
                        boost::bind(&core::CStateMachine::acceptRestoreTraverser, &m_Machine, _1)));
            RESTORE_SETUP_TEARDOWN(TREND_OLD_TAG,
                                   /**/,
                                   traverser.traverseSubLevel(boost::bind(
                                                                  upgradeTrendModelToVersion6p3,
                                                                  m_BucketLength, boost::ref(m_Trend), _1)),
                                   m_UsingTrendForPrediction = true)
            RESTORE_SETUP_TEARDOWN(SEASONAL_OLD_TAG,
                                   m_Seasonal.reset(new SSeasonal),
                                   traverser.traverseSubLevel(boost::bind(
                                                                  &SSeasonal::acceptRestoreTraverser,
                                                                  m_Seasonal.get(), m_DecayRate, m_BucketLength, _1)),
                                   /**/)
            RESTORE_SETUP_TEARDOWN(CALENDAR_OLD_TAG,
                                   m_Calendar.reset(new SCalendar),
                                   traverser.traverseSubLevel(boost::bind(
                                                                  &SCalendar::acceptRestoreTraverser,
                                                                  m_Calendar.get(), m_DecayRate, m_BucketLength, _1)),
                                   /**/)
        } while (traverser.next());

        m_MeanVarianceScale.add(1.0, MODEL_WEIGHT_UPGRADING_TO_VERSION_6p3);
    }
    return true;
}

void CTimeSeriesDecompositionDetail::CComponents::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertValue(VERSION_6_3_TAG, "");
    inserter.insertLevel(COMPONENTS_MACHINE_6_3_TAG,
                         boost::bind(&core::CStateMachine::acceptPersistInserter, &m_Machine, _1));
    inserter.insertValue(DECAY_RATE_6_3_TAG, m_DecayRate, core::CIEEE754::E_SinglePrecision);
    inserter.insertLevel(TREND_6_3_TAG, boost::bind(&CTrendComponent::acceptPersistInserter, m_Trend, _1));
    if (m_Seasonal) {
        inserter.insertLevel(SEASONAL_6_3_TAG, boost::bind(&SSeasonal::acceptPersistInserter, m_Seasonal.get(), _1));
    }
    if (m_Calendar) {
        inserter.insertLevel(CALENDAR_6_3_TAG, boost::bind(&SCalendar::acceptPersistInserter, m_Calendar.get(), _1));
    }
    inserter.insertValue(MEAN_VARIANCE_SCALE_6_3_TAG, m_MeanVarianceScale.toDelimited());
    inserter.insertValue(MOMENTS_6_3_TAG, m_Moments.toDelimited());
    inserter.insertValue(MOMENTS_MINUS_TREND_6_3_TAG, m_MomentsMinusTrend.toDelimited());
    inserter.insertValue(USING_TREND_FOR_PREDICTION_6_3_TAG, m_UsingTrendForPrediction);
}

void CTimeSeriesDecompositionDetail::CComponents::swap(CComponents &other) {
    std::swap(m_Machine, other.m_Machine);
    std::swap(m_DecayRate, other.m_DecayRate);
    std::swap(m_BucketLength, other.m_BucketLength);
    std::swap(m_SeasonalComponentSize, other.m_SeasonalComponentSize);
    std::swap(m_CalendarComponentSize, other.m_CalendarComponentSize);
    m_Trend.swap(other.m_Trend);
    m_Seasonal.swap(other.m_Seasonal);
    m_Calendar.swap(other.m_Calendar);
    std::swap(m_MeanVarianceScale, other.m_MeanVarianceScale);
    std::swap(m_Moments, other.m_Moments);
    std::swap(m_MomentsMinusTrend, other.m_MomentsMinusTrend);
    std::swap(m_UsingTrendForPrediction, other.m_UsingTrendForPrediction);
}

void CTimeSeriesDecompositionDetail::CComponents::handle(const SAddValue &message) {
    switch (m_Machine.state()) {
        case SC_NORMAL:
        case SC_NEW_COMPONENTS: {
            this->interpolate(message);

            core_t::TTime                   time{message.s_Time};
            double                          value{message.s_Value};
            double                          trend{message.s_Trend};
            double                          seasonal{message.s_Seasonal};
            double                          calendar{message.s_Calendar};
            const maths_t::TWeightStyleVec &weightStyles{message.s_WeightStyles};
            const maths_t::TDouble4Vec &    weights{message.s_Weights};

            TSeasonalComponentPtrVec seasonalComponents;
            TCalendarComponentPtrVec calendarComponents;
            TComponentErrorsPtrVec   seasonalErrors;
            TComponentErrorsPtrVec   calendarErrors;
            TDoubleVec               deltas;

            if (m_Seasonal) {
                m_Seasonal->componentsErrorsAndDeltas(time, seasonalComponents, seasonalErrors, deltas);
            }
            if (m_Calendar) {
                m_Calendar->componentsAndErrors(time, calendarComponents, calendarErrors);
            }

            double      weight{maths_t::countForUpdate(weightStyles, weights)};
            std::size_t m{seasonalComponents.size()};
            std::size_t n{calendarComponents.size()};

            TDoubleVec values(m + n + 1, value);
            TDoubleVec predictions(m + n);
            double     error;
            double     scale;
            decompose(m_Trend, seasonalComponents, calendarComponents,
                      time, deltas, values, predictions, error, scale);

            core_t::TTime observedInterval{m_Trend.observedInterval()};

            m_Trend.add(time, values[0], weight);
            for (std::size_t i = 1u; i <= m; ++i) {
                CSeasonalComponent *component{seasonalComponents[i - 1]};
                CComponentErrors *  error_{seasonalErrors[i - 1]};
                double              wi{weight / component->time().fractionInWindow()};
                component->add(time, values[i], wi);
                error_->add(error, predictions[i - 1], wi);
            }
            for (std::size_t i = m + 1; i <= m + n; ++i) {
                CCalendarComponent *component{calendarComponents[i - m - 1]};
                CComponentErrors *  error_{calendarErrors[i - m - 1]};
                component->add(time, values[i], weight);
                error_->add(error, predictions[i - 1], weight);
            }

            m_MeanVarianceScale.add(scale, weight);
            m_Moments.add(value - seasonal - calendar, weight);
            m_MomentsMinusTrend.add(value - trend - seasonal - calendar, weight);

            if (!m_UsingTrendForPrediction && observedInterval > 6 * m_BucketLength) {
                double v0{CBasicStatistics::variance(m_Moments)};
                double v1{CBasicStatistics::variance(m_MomentsMinusTrend)};
                double df0{CBasicStatistics::count(m_Moments) - 1.0};
                double df1{CBasicStatistics::count(m_MomentsMinusTrend) - m_Trend.parameters()};
                m_UsingTrendForPrediction =
                    v1 < SIGNIFICANT_VARIANCE_REDUCTION[0] * v0 &&
                         df0 > 0.0 && df1 > 0.0 &&
                    CStatisticalTests::leftTailFTest(v1 / v0, df1, df0) <= MAXIMUM_SIGNIFICANCE;
                *m_Watcher = m_UsingTrendForPrediction;
            }
        }
        break;
        case SC_DISABLED:
            break;
        default:
            LOG_ERROR("Components in a bad state: " << m_Machine.state());
            this->apply(SC_RESET, message);
            break;
    }
}

void CTimeSeriesDecompositionDetail::CComponents::handle(const SDetectedSeasonal &message) {
    if (this->size() + m_SeasonalComponentSize > this->maxSize()) {
        return;
    }

    switch (m_Machine.state()) {
        case SC_NORMAL:
        case SC_NEW_COMPONENTS: {
            if (!m_Seasonal) {
                m_Seasonal.reset(new SSeasonal);
            }

            core_t::TTime                            time{message.s_Time};
            core_t::TTime                            lastTime{message.s_LastTime};
            const CPeriodicityHypothesisTestsResult &result{message.s_Result};
            const CExpandingWindow &                 window{message.s_Window};
            const TPredictor &                       predictor{message.s_Predictor};

            TSeasonalComponentVec &components{m_Seasonal->s_Components};
            TComponentErrorsVec &  errors{m_Seasonal->s_PredictionErrors};

            if (!this->addSeasonalComponents(result, window, predictor, m_Trend, components, errors)) {
                break;
            }
            if (m_Watcher) {
                *m_Watcher = true;
            }
            LOG_DEBUG("Detected seasonal components at " << time);

            m_UsingTrendForPrediction = true;
            this->clearComponentErrors();
            this->apply(SC_ADDED_COMPONENTS, message);
            this->mediator()->forward(SNewComponents(time, lastTime, SNewComponents::E_GeneralSeasonal));
            break;
        }
        case SC_DISABLED:
            break;
        default:
            LOG_ERROR("Components in a bad state: " << m_Machine.state());
            this->apply(SC_RESET, message);
            break;
    }
}

void CTimeSeriesDecompositionDetail::CComponents::handle(const SDetectedCalendar &message) {
    if (this->size() + m_CalendarComponentSize > this->maxSize()) {
        return;
    }

    switch (m_Machine.state()) {
        case SC_NORMAL:
        case SC_NEW_COMPONENTS: {
            if (!m_Calendar) {
                m_Calendar.reset(new SCalendar);
            }

            core_t::TTime    time{message.s_Time};
            core_t::TTime    lastTime{message.s_LastTime};
            CCalendarFeature feature{message.s_Feature};

            if (m_Calendar->haveComponent(feature)) {
                break;
            }

            TCalendarComponentVec &components{m_Calendar->s_Components};
            TComponentErrorsVec &  errors{m_Calendar->s_PredictionErrors};

            this->addCalendarComponent(feature, time, components, errors);
            this->apply(SC_ADDED_COMPONENTS, message);
            this->mediator()->forward(SNewComponents(time, lastTime, SNewComponents::E_CalendarCyclic));
            break;
        }
        case SC_DISABLED:
            break;
        default:
            LOG_ERROR("Components in a bad state: " << m_Machine.state());
            this->apply(SC_RESET, message);
            break;
    }
}

void CTimeSeriesDecompositionDetail::CComponents::interpolate(const SMessage &message, bool refine) {
    core_t::TTime time{message.s_Time};
    core_t::TTime lastTime{message.s_LastTime};

    std::size_t state{m_Machine.state()};

    switch (state) {
        case SC_NORMAL:
        case SC_NEW_COMPONENTS:
            this->canonicalize(time);
            if (this->shouldInterpolate(time, lastTime)) {
                LOG_TRACE("Interpolating values at " << time);

                if (m_Seasonal) {
                    m_Seasonal->interpolate(time, lastTime, refine);
                }
                if (m_Calendar) {
                    m_Calendar->interpolate(time, lastTime, refine);
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

void CTimeSeriesDecompositionDetail::CComponents::decayRate(double decayRate) {
    m_DecayRate = decayRate;
    m_Trend.decayRate(decayRate);
    if (m_Seasonal) {
        m_Seasonal->decayRate(decayRate);
    }
    if (m_Calendar) {
        m_Calendar->decayRate(decayRate);
    }
}

double CTimeSeriesDecompositionDetail::CComponents::decayRate(void) const
{
    return m_DecayRate;
}

void CTimeSeriesDecompositionDetail::CComponents::propagateForwards(core_t::TTime start,
                                                                    core_t::TTime end) {
    m_Trend.propagateForwardsByTime(end - start);
    if (m_Seasonal) {
        m_Seasonal->propagateForwards(start, end);
    }
    if (m_Calendar) {
        m_Calendar->propagateForwards(start, end);
    }
    double factor{std::exp(-m_DecayRate * static_cast<double>(end - start)
                           / static_cast<double>(DAY))};
    m_MeanVarianceScale.age(factor);
    m_Moments.age(factor);
    m_MomentsMinusTrend.age(factor);
}

bool CTimeSeriesDecompositionDetail::CComponents::initialized(void) const
{
    return m_UsingTrendForPrediction && m_Trend.initialized() ? true :
           (m_Seasonal && m_Calendar ? m_Seasonal->initialized() || m_Calendar->initialized() :
            (m_Seasonal ? m_Seasonal->initialized() :
             (m_Calendar ? m_Calendar->initialized() : false)));
}

const CTrendComponent &CTimeSeriesDecompositionDetail::CComponents::trend(void) const {
    return m_Trend;
}

const TSeasonalComponentVec &CTimeSeriesDecompositionDetail::CComponents::seasonal(void) const {
    return m_Seasonal ? m_Seasonal->s_Components : NO_SEASONAL_COMPONENTS;
}

const maths_t::TCalendarComponentVec &CTimeSeriesDecompositionDetail::CComponents::calendar(void) const {
    return m_Calendar ? m_Calendar->s_Components : NO_CALENDAR_COMPONENTS;
}

bool CTimeSeriesDecompositionDetail::CComponents::usingTrendForPrediction(void) const
{
    return m_UsingTrendForPrediction;
}

CPeriodicityHypothesisTestsConfig CTimeSeriesDecompositionDetail::CComponents::periodicityTestConfig(void) const
{
    CPeriodicityHypothesisTestsConfig result;
    for (const auto &component : this->seasonal()) {
        const CSeasonalTime &time{component.time()};
        result.hasDaily(  result.hasDaily()   || time.period() == DAY);
        result.hasWeekend(result.hasWeekend() || time.hasWeekend());
        result.hasWeekly( result.hasWeekly()  || time.period() == WEEK);
        if (time.hasWeekend()) {
            result.startOfWeek(time.windowRepeatStart());
        }
    }
    return result;
}

double CTimeSeriesDecompositionDetail::CComponents::meanValue(core_t::TTime time) const
{
    return this->initialized() ? ( (m_UsingTrendForPrediction ?
                                    CBasicStatistics::mean(m_Trend.value(time, 0.0)) : 0.0)
                                   + meanOf(&CSeasonalComponent::meanValue, this->seasonal())) : 0.0;
}

double CTimeSeriesDecompositionDetail::CComponents::meanVariance(void) const
{
    return this->initialized() ? ( (m_UsingTrendForPrediction ?
                                    CBasicStatistics::mean(this->trend().variance(0.0)) : 0.0)
                                   + meanOf(&CSeasonalComponent::meanVariance, this->seasonal())) : 0.0;
}

double CTimeSeriesDecompositionDetail::CComponents::meanVarianceScale(void) const
{
    return CBasicStatistics::mean(m_MeanVarianceScale);
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
    seed = CChecksum::calculate(seed, m_Calendar);
    seed = CChecksum::calculate(seed, m_MeanVarianceScale);
    seed = CChecksum::calculate(seed, m_Moments);
    seed = CChecksum::calculate(seed, m_MomentsMinusTrend);
    return CChecksum::calculate(seed, m_UsingTrendForPrediction);
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
    return core::CMemory::dynamicSize(m_Trend)
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

bool CTimeSeriesDecompositionDetail::CComponents::addSeasonalComponents(const CPeriodicityHypothesisTestsResult &result,
                                                                        const CExpandingWindow &window,
                                                                        const TPredictor &predictor,
                                                                        CTrendComponent &trend,
                                                                        TSeasonalComponentVec &components,
                                                                        TComponentErrorsVec &errors) const
{
    using TSeasonalTimePtr = boost::shared_ptr<CSeasonalTime>;
    using TSeasonalTimePtrVec = std::vector<TSeasonalTimePtr>;

    TSeasonalTimePtrVec newSeasonalTimes;

    for (const auto &candidate_ : result.components()) {
        TSeasonalTimePtr seasonalTime(candidate_.seasonalTime());
        if (std::find_if(components.begin(), components.end(),
                         [&seasonalTime](const CSeasonalComponent &component) {
                    return component.time().excludes(*seasonalTime);
                }) == components.end()) {
            LOG_DEBUG("Detected '" << candidate_.s_Description << "'");
            newSeasonalTimes.push_back(seasonalTime);
        }
    }

    if (newSeasonalTimes.size() > 0) {
        for (const auto &seasonalTime : newSeasonalTimes) {
            components.erase(std::remove_if(components.begin(), components.end(),
                                            [&seasonalTime](const CSeasonalComponent &component) {
                        return seasonalTime->excludes(component.time());
                    }), components.end());
        }

        std::sort(newSeasonalTimes.begin(), newSeasonalTimes.end(), maths::COrderings::SLess());

        TFloatMeanAccumulatorVec values;
        for (const auto &seasonalTime : newSeasonalTimes) {
            values = window.valuesMinusPrediction(predictor);
            components.emplace_back(*seasonalTime, m_SeasonalComponentSize,
                                    m_DecayRate, static_cast<double>(m_BucketLength),
                                    CSplineTypes::E_Natural);
            components.back().initialize(window.startTime(), window.endTime(), values);
            components.back().interpolate(CIntegerTools::floor(window.endTime(),
                                                               seasonalTime->period()));
        }

        CTrendComponent windowTrend{trend.defaultDecayRate()};
        values = window.valuesMinusPrediction(predictor);
        core_t::TTime time{window.startTime() + window.bucketLength() / 2};
        for (const auto &value : values) {
            // Because we now test before the window is fully compressed
            // we can get a run of unset values at the end of the window,
            // we should just ignore these.
            if (CBasicStatistics::count(value) > 0.0) {
                windowTrend.add(time, CBasicStatistics::mean(value), CBasicStatistics::count(value));
                windowTrend.propagateForwardsByTime(window.bucketLength());
            }
            time += window.bucketLength();
        }
        trend.swap(windowTrend);

        errors.resize(components.size());
        COrderings::simultaneousSort(components, errors,
                                     [](const CSeasonalComponent &lhs, const CSeasonalComponent &rhs) {
                    return lhs.time() < rhs.time();
                });
    }

    return newSeasonalTimes.size() > 0;
}

bool CTimeSeriesDecompositionDetail::CComponents::addCalendarComponent(const CCalendarFeature &feature,
                                                                       core_t::TTime time,
                                                                       maths_t::TCalendarComponentVec &components,
                                                                       TComponentErrorsVec &errors) const
{
    double bucketLength{static_cast<double>(m_BucketLength)};
    components.emplace_back(feature, m_CalendarComponentSize,
                            m_DecayRate, bucketLength, CSplineTypes::E_Natural);
    components.back().initialize();
    errors.resize(components.size());
    LOG_DEBUG("Detected feature '" << feature.print() << "' at " << time);
    return true;
}

void CTimeSeriesDecompositionDetail::CComponents::clearComponentErrors(void) {
    if (m_Seasonal) {
        for (auto &errors : m_Seasonal->s_PredictionErrors) {
            errors.clear();
        }
    }
    if (m_Calendar) {
        for (auto &errors : m_Calendar->s_PredictionErrors) {
            errors.clear();
        }
    }
}

void CTimeSeriesDecompositionDetail::CComponents::apply(std::size_t symbol, const SMessage &message) {
    if (symbol == SC_RESET) {
        m_Trend.clear();
        m_Seasonal.reset();
        m_Calendar.reset();
    }

    std::size_t old{m_Machine.state()};
    m_Machine.apply(symbol);
    std::size_t state{m_Machine.state()};

    if (state != old) {
        LOG_TRACE(SC_STATES[old] << "," << SC_ALPHABET[symbol] << " -> " << SC_STATES[state]);

        switch (state) {
            case SC_NORMAL:
            case SC_NEW_COMPONENTS:
                this->interpolate(message);
                break;
            case SC_DISABLED:
                m_Trend.clear();
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
                                                                    core_t::TTime last) {
    return m_Machine.state() == SC_NEW_COMPONENTS ||
           (m_Seasonal && m_Seasonal->shouldInterpolate(time, last)) ||
           (m_Calendar && m_Calendar->shouldInterpolate(time, last));
}

void CTimeSeriesDecompositionDetail::CComponents::shiftOrigin(core_t::TTime time) {
    time -= static_cast<core_t::TTime>(static_cast<double>(DAY) / m_DecayRate / 2.0);
    m_Trend.shiftOrigin(time);
    if (m_Seasonal) {
        m_Seasonal->shiftOrigin(time);
    }
}

void CTimeSeriesDecompositionDetail::CComponents::canonicalize(core_t::TTime time) {
    this->shiftOrigin(time);

    if (m_Seasonal && m_Seasonal->prune(time, m_BucketLength)) {
        m_Seasonal.reset();
    }
    if (m_Calendar && m_Calendar->prune(time, m_BucketLength)) {
        m_Calendar.reset();
    }

    if (m_Seasonal) {
        TSeasonalComponentVec &seasonal{m_Seasonal->s_Components};

        TTimeTimePrDoubleFMap slope;
        slope.reserve(seasonal.size());

        for (auto &component : seasonal) {
            if (component.slopeAccurate(time)) {
                const CSeasonalTime &time_{component.time()};
                double               si{component.slope()};
                component.shiftSlope(-si);
                slope[time_.window()] += si;
            }
        }

        LOG_TRACE("slope = " << core::CContainerPrinter::print(slope));
        shiftSlope(slope, m_DecayRate, m_Trend);
    }
}

void CTimeSeriesDecompositionDetail::CComponents::notifyOnNewComponents(bool *watcher) {
    m_Watcher = watcher;
}

CTimeSeriesDecompositionDetail::CComponents::CScopeNotifyOnStateChange::CScopeNotifyOnStateChange(CComponents &components) :
    m_Components{components}, m_Watcher{false} {
    m_Components.notifyOnNewComponents(&m_Watcher);
}

CTimeSeriesDecompositionDetail::CComponents::CScopeNotifyOnStateChange::~CScopeNotifyOnStateChange(void) {
    m_Components.notifyOnNewComponents(0);
}

bool CTimeSeriesDecompositionDetail::CComponents::CScopeNotifyOnStateChange::changed(void) const
{
    return m_Watcher;
}

bool CTimeSeriesDecompositionDetail::CComponents::CComponentErrors::fromDelimited(const std::string &str) {
    TFloatMeanAccumulator *state[] =
    {
        &m_MeanErrorWithComponent,
        &m_MeanErrorWithoutComponent
    };

    std::string suffix = str;
    for (std::size_t i = 0u, n = 0; i < 2; ++i, suffix = suffix.substr(n + 1)) {
        n = suffix.find(CBasicStatistics::EXTERNAL_DELIMITER);
        if (!state[i]->fromDelimited(suffix.substr(0, n))) {
            LOG_ERROR("Failed to parse '" << str << "'");
            return false;
        }
    }

    return true;
}

std::string CTimeSeriesDecompositionDetail::CComponents::CComponentErrors::toDelimited(void) const
{
    return m_MeanErrorWithComponent.toDelimited() + CBasicStatistics::EXTERNAL_DELIMITER
           + m_MeanErrorWithoutComponent.toDelimited() + CBasicStatistics::EXTERNAL_DELIMITER;
}

void CTimeSeriesDecompositionDetail::CComponents::CComponentErrors::add(double error,
                                                                        double prediction,
                                                                        double weight) {
    double errorWithComponent{winsorise(pow2(error), m_MeanErrorWithComponent)};
    double errorWithoutComponent{winsorise(pow2(error - prediction), m_MeanErrorWithoutComponent)};
    m_MeanErrorWithComponent.add(errorWithComponent, weight);
    m_MeanErrorWithoutComponent.add(errorWithoutComponent, weight);
}

void CTimeSeriesDecompositionDetail::CComponents::CComponentErrors::clear(void) {
    m_MeanErrorWithComponent = TFloatMeanAccumulator();
    m_MeanErrorWithoutComponent = TFloatMeanAccumulator();
}

bool CTimeSeriesDecompositionDetail::CComponents::CComponentErrors::remove(core_t::TTime bucketLength,
                                                                           CSeasonalComponent &seasonal) const
{
    double count{CBasicStatistics::count(m_MeanErrorWithComponent)};
    double errorWithComponent{CBasicStatistics::mean(m_MeanErrorWithComponent)};
    double errorWithoutComponent{CBasicStatistics::mean(m_MeanErrorWithoutComponent)};
    return count > static_cast<double>(10 * seasonal.time().period() / bucketLength) &&
           std::max(  errorWithoutComponent
                      / errorWithComponent, seasonal.heteroscedasticity()) < 1.5;
}

bool CTimeSeriesDecompositionDetail::CComponents::CComponentErrors::remove(core_t::TTime bucketLength,
                                                                           CCalendarComponent &calendar) const
{
    double count{CBasicStatistics::count(m_MeanErrorWithComponent)};
    double errorWithComponent{CBasicStatistics::mean(m_MeanErrorWithComponent)};
    double errorWithoutComponent{CBasicStatistics::mean(m_MeanErrorWithoutComponent)};
    return count > static_cast<double>(5 * calendar.feature().window() / bucketLength) &&
           std::max(  errorWithoutComponent
                      / errorWithComponent, calendar.heteroscedasticity()) < 1.5;
}

void CTimeSeriesDecompositionDetail::CComponents::CComponentErrors::age(double factor) {
    m_MeanErrorWithComponent.age(factor);
    m_MeanErrorWithoutComponent.age(factor);
}

uint64_t CTimeSeriesDecompositionDetail::CComponents::CComponentErrors::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, m_MeanErrorWithComponent);
    return CChecksum::calculate(seed, m_MeanErrorWithoutComponent);
}

double CTimeSeriesDecompositionDetail::CComponents::CComponentErrors::winsorise(double squareError,
                                                                                const TFloatMeanAccumulator &variance) {
    return CBasicStatistics::count(variance) > 10.0 ?
           std::min(squareError, 36.0 * CBasicStatistics::mean(variance)) : squareError;
}

bool CTimeSeriesDecompositionDetail::CComponents::SSeasonal::acceptRestoreTraverser(double decayRate,
                                                                                    core_t::TTime bucketLength_,
                                                                                    core::CStateRestoreTraverser &traverser) {
    double bucketLength{static_cast<double>(bucketLength_)};
    if (traverser.name() == VERSION_6_3_TAG) {
        while (traverser.next()) {
            const std::string &name{traverser.name()};
            RESTORE_NO_ERROR(COMPONENT_6_3_TAG, s_Components.emplace_back(
                                 decayRate, bucketLength, traverser))
            RESTORE(ERRORS_6_3_TAG, core::CPersistUtils::restore(
                        ERRORS_6_3_TAG, s_PredictionErrors, traverser))
        }
    } else {
        // There is no version string this is historic state.
        do {
            const std::string &name{traverser.name()};
            RESTORE_NO_ERROR(COMPONENT_OLD_TAG, s_Components.emplace_back(
                                 decayRate, bucketLength, traverser))
            RESTORE(ERRORS_OLD_TAG, core::CPersistUtils::restore(
                        ERRORS_OLD_TAG, s_PredictionErrors, traverser))
        } while (traverser.next());
    }
    return true;
}

void CTimeSeriesDecompositionDetail::CComponents::SSeasonal::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertValue(VERSION_6_3_TAG, "");
    for (const auto &component : s_Components) {
        inserter.insertLevel(COMPONENT_6_3_TAG, boost::bind(
                                 &CSeasonalComponent::acceptPersistInserter, &component, _1));
    }
    core::CPersistUtils::persist(ERRORS_6_3_TAG, s_PredictionErrors, inserter);
}

void CTimeSeriesDecompositionDetail::CComponents::SSeasonal::decayRate(double decayRate) {
    for (auto &component : s_Components) {
        component.decayRate(decayRate);
    }
}

void CTimeSeriesDecompositionDetail::CComponents::SSeasonal::propagateForwards(core_t::TTime start,
                                                                               core_t::TTime end) {
    for (std::size_t i = 0u; i < s_Components.size(); ++i) {
        core_t::TTime period{s_Components[i].time().period()};
        core_t::TTime a{CIntegerTools::floor(start, period)};
        core_t::TTime b{CIntegerTools::floor(end, period)};
        if (b > a) {
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
    for (const auto &component : s_Components) {
        result += component.size();
    }
    return result;
}

void CTimeSeriesDecompositionDetail::CComponents::SSeasonal::componentsErrorsAndDeltas(core_t::TTime time,
                                                                                       TSeasonalComponentPtrVec &components,
                                                                                       TComponentErrorsPtrVec &errors,
                                                                                       TDoubleVec &deltas) {
    std::size_t n{s_Components.size()};

    components.reserve(n);
    errors.reserve(n);

    for (std::size_t i = 0u; i < n; ++i) {
        if (s_Components[i].time().inWindow(time)) {
            components.push_back(&s_Components[i]);
            errors.push_back(&s_PredictionErrors[i]);
        }
    }

    deltas.resize(components.size(), 0.0);
    for (std::size_t i = 1u; i < components.size(); ++i) {
        int j{static_cast<int>(i - 1)};
        for (core_t::TTime period{components[i]->time().period()}; j > -1; --j) {
            core_t::TTime period_{components[j]->time().period()};
            if (period % period_ == 0) {
                double value{CBasicStatistics::mean(components[j]->value(time, 0.0))};
                double delta{0.2 * components[i]->delta(time, period_, value)};
                deltas[j] += delta;
                deltas[i] -= delta;
                break;
            }
        }
    }
}

bool CTimeSeriesDecompositionDetail::CComponents::SSeasonal::shouldInterpolate(core_t::TTime time,
                                                                               core_t::TTime last) const
{
    for (const auto &component : s_Components) {
        core_t::TTime period{component.time().period()};
        core_t::TTime a{CIntegerTools::floor(last, period)};
        core_t::TTime b{CIntegerTools::floor(time, period)};
        if (b > a) {
            return true;
        }
    }
    return false;
}

void CTimeSeriesDecompositionDetail::CComponents::SSeasonal::interpolate(core_t::TTime time,
                                                                         core_t::TTime last,
                                                                         bool refine) {
    for (auto &component : s_Components) {
        core_t::TTime period{component.time().period()};
        core_t::TTime a{CIntegerTools::floor(last, period)};
        core_t::TTime b{CIntegerTools::floor(time, period)};
        if (b > a || !component.initialized()) {
            component.interpolate(b, refine);
        }
    }
}

bool CTimeSeriesDecompositionDetail::CComponents::SSeasonal::initialized(void) const
{
    for (const auto &component : s_Components) {
        if (component.initialized()) {
            return true;
        }
    }
    return false;
}

bool CTimeSeriesDecompositionDetail::CComponents::SSeasonal::prune(core_t::TTime time,
                                                                   core_t::TTime bucketLength) {
    std::size_t n = s_Components.size();

    if (n > 1) {
        TTimeTimePrSizeFMap windowed;
        windowed.reserve(n);
        for (const auto &component : s_Components) {
            const CSeasonalTime &time_ = component.time();
            if (time_.windowed()) {
                ++windowed[time_.window()];
            }
        }

        TBoolVec              remove(n, false);
        TTimeTimePrDoubleFMap shifts;
        shifts.reserve(n);
        for (std::size_t i = 0u; i < n; ++i) {
            const CSeasonalTime &time_ = s_Components[i].time();
            auto                 j = windowed.find(time_.window());
            if (j == windowed.end() || j->second > 1) {
                if (s_PredictionErrors[i].remove(bucketLength, s_Components[i])) {
                    LOG_DEBUG("Removing seasonal component"
                              << " with period '" << time_.period() << "' at " << time);
                    remove[i] = true;
                    shifts[time_.window()] += s_Components[i].meanValue();
                    --j->second;
                }
            }
        }

        CSetTools::simultaneousRemoveIf(
            remove, s_Components, s_PredictionErrors, [](bool remove_) {
                    return remove_;
                });

        for (auto &shift : shifts) {
            if (windowed.count(shift.first) > 0) {
                for (auto &&component : s_Components) {
                    if (shift.first == component.time().window()) {
                        component.shiftLevel(shift.second);
                        break;
                    }
                }
            } else {
                bool fallback = true;
                for (auto &&component : s_Components) {
                    if (!component.time().windowed()) {
                        component.shiftLevel(shift.second);
                        fallback = false;
                        break;
                    }
                }
                if (fallback) {
                    TTimeTimePrVec shifted;
                    shifted.reserve(s_Components.size());
                    for (auto &&component : s_Components) {
                        const CSeasonalTime &time_ = component.time();
                        if (std::find_if(shifted.begin(), shifted.end(),
                                         [&time_](const TTimeTimePr &window) {
                                    return !(   time_.windowEnd() <= window.first ||
                                                time_.windowStart() >= window.second);
                                }) == shifted.end()) {
                            component.shiftLevel(shift.second);
                        }
                    }
                }
            }
        }
    }

    return s_Components.empty();
}

void CTimeSeriesDecompositionDetail::CComponents::SSeasonal::shiftOrigin(core_t::TTime time) {
    for (auto &component : s_Components) {
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
                                                                                    core_t::TTime bucketLength_,
                                                                                    core::CStateRestoreTraverser &traverser) {
    double bucketLength{static_cast<double>(bucketLength_)};
    if (traverser.name() == VERSION_6_3_TAG) {
        while (traverser.next()) {
            const std::string &name{traverser.name()};
            RESTORE_NO_ERROR(COMPONENT_6_3_TAG, s_Components.emplace_back(
                                 decayRate, bucketLength, traverser))
            RESTORE(ERRORS_6_3_TAG, core::CPersistUtils::restore(
                        ERRORS_6_3_TAG, s_PredictionErrors, traverser))
        }
    } else {
        // There is no version string this is historic state.
        do {
            const std::string &name{traverser.name()};
            RESTORE_NO_ERROR(COMPONENT_OLD_TAG, s_Components.emplace_back(
                                 decayRate, bucketLength, traverser))
            RESTORE(ERRORS_OLD_TAG, core::CPersistUtils::restore(
                        ERRORS_OLD_TAG, s_PredictionErrors, traverser))
        } while (traverser.next());
    }
    return true;
}

void CTimeSeriesDecompositionDetail::CComponents::SCalendar::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertValue(VERSION_6_3_TAG, "");
    for (const auto &component : s_Components) {
        inserter.insertLevel(COMPONENT_6_3_TAG, boost::bind(
                                 &CCalendarComponent::acceptPersistInserter, &component, _1));
    }
    core::CPersistUtils::persist(ERRORS_6_3_TAG, s_PredictionErrors, inserter);
}

void CTimeSeriesDecompositionDetail::CComponents::SCalendar::decayRate(double decayRate) {
    for (auto &component : s_Components) {
        component.decayRate(decayRate);
    }
}

void CTimeSeriesDecompositionDetail::CComponents::SCalendar::propagateForwards(core_t::TTime start,
                                                                               core_t::TTime end) {
    for (std::size_t i = 0u; i < s_Components.size(); ++i) {
        core_t::TTime a{CIntegerTools::floor(start, MONTH)};
        core_t::TTime b{CIntegerTools::floor(end, MONTH)};
        if (b > a) {
            double time{static_cast<double>(b - a) / static_cast<double>(MONTH)};
            s_Components[i].propagateForwardsByTime(time);
            s_PredictionErrors[i].age(std::exp(-s_Components[i].decayRate() * time));
        }
    }
}

std::size_t CTimeSeriesDecompositionDetail::CComponents::SCalendar::size(void) const
{
    std::size_t result{0};
    for (const auto &component : s_Components) {
        result += component.size();
    }
    return result;
}

bool CTimeSeriesDecompositionDetail::CComponents::SCalendar::haveComponent(CCalendarFeature feature) const
{
    for (const auto &component : s_Components) {
        if (component.feature() == feature) {
            return true;
        }
    }
    return false;
}

void CTimeSeriesDecompositionDetail::CComponents::SCalendar::componentsAndErrors(core_t::TTime time,
                                                                                 TCalendarComponentPtrVec &components,
                                                                                 TComponentErrorsPtrVec &errors) {
    std::size_t n = s_Components.size();
    components.reserve(n);
    errors.reserve(n);
    for (std::size_t i = 0u; i < n; ++i) {
        if (s_Components[i].feature().inWindow(time)) {
            components.push_back(&s_Components[i]);
            errors.push_back(&s_PredictionErrors[i]);
        }
    }
}

bool CTimeSeriesDecompositionDetail::CComponents::SCalendar::shouldInterpolate(core_t::TTime time,
                                                                               core_t::TTime last) const
{
    for (const auto &component : s_Components) {
        CCalendarFeature feature = component.feature();
        if (!feature.inWindow(time) && feature.inWindow(last)) {
            return true;
        }
    }
    return false;
}

void CTimeSeriesDecompositionDetail::CComponents::SCalendar::interpolate(core_t::TTime time,
                                                                         core_t::TTime last,
                                                                         bool refine) {
    for (auto &component : s_Components) {
        CCalendarFeature feature = component.feature();
        if (!feature.inWindow(time) && feature.inWindow(last)) {
            component.interpolate(time - feature.offset(time), refine);
        }
    }
}

bool CTimeSeriesDecompositionDetail::CComponents::SCalendar::initialized(void) const
{
    for (const auto &component : s_Components) {
        if (component.initialized()) {
            return true;
        }
    }
    return false;
}

bool CTimeSeriesDecompositionDetail::CComponents::SCalendar::prune(core_t::TTime time,
                                                                   core_t::TTime bucketLength) {
    TBoolVec remove(s_Components.size(), false);
    for (std::size_t i = 0u; i < s_Components.size(); ++i) {
        if (s_PredictionErrors[i].remove(bucketLength, s_Components[i])) {
            LOG_DEBUG("Removing calendar component"
                      << " '" << s_Components[i].feature().print() << "' at " << time);
            remove[i] = true;
        }
    }

    CSetTools::simultaneousRemoveIf(
        remove, s_Components, s_PredictionErrors, [](bool remove_) {
                return remove_;
            });

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
