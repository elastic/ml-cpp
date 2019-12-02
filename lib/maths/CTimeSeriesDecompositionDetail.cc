/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CTimeSeriesDecompositionDetail.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/CPersistUtils.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CTimezone.h>
#include <core/Constants.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CCalendarComponent.h>
#include <maths/CChecksum.h>
#include <maths/CExpandingWindow.h>
#include <maths/CIntegerTools.h>
#include <maths/CLeastSquaresOnlineRegressionDetail.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraPersist.h>
#include <maths/CPeriodicityHypothesisTests.h>
#include <maths/CSampling.h>
#include <maths/CSeasonalComponentAdaptiveBucketing.h>
#include <maths/CSeasonalTime.h>
#include <maths/CSetTools.h>
#include <maths/CStatisticalTests.h>
#include <maths/CTimeSeriesDecomposition.h>
#include <maths/CTimeSeriesSegmentation.h>
#include <maths/CTools.h>
#include <maths/Constants.h>

#include <boost/config.hpp>
#include <boost/container/flat_map.hpp>
#include <boost/container/flat_set.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/numeric/conversion/bounds.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
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
using TSizeSizeMap = std::map<std::size_t, std::size_t>;
using TStrVec = std::vector<std::string>;
using TTimeVec = std::vector<core_t::TTime>;
using TTimeTimePr = std::pair<core_t::TTime, core_t::TTime>;
using TTimeTimePrVec = std::vector<TTimeTimePr>;
using TTimeTimePrDoubleFMap = boost::container::flat_map<TTimeTimePr, double>;
using TTimeTimePrSizeFMap = boost::container::flat_map<TTimeTimePr, std::size_t>;
using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
using TComponent5Vec = CPeriodicityHypothesisTestsResult::TComponent5Vec;
using TSeasonalComponentPtrVec = std::vector<CSeasonalComponent*>;
using TCalendarComponentPtrVec = std::vector<CCalendarComponent*>;

const core_t::TTime DAY{core::constants::DAY};
const core_t::TTime WEEK{core::constants::WEEK};
const core_t::TTime MONTH{4 * WEEK};

//! We scale the time used for the regression model to improve
//! the condition of the design matrix.
double scaleTime(core_t::TTime time, core_t::TTime origin) {
    return static_cast<double>(time - origin) / static_cast<double>(WEEK);
}

//! Get the aging factor to apply for \p dt elapsed time.
double ageFactor(double decayRate, core_t::TTime dt, core_t::TTime scale = DAY) {
    return std::exp(-decayRate * static_cast<double>(dt) / static_cast<double>(scale));
}

//! Compute the mean of \p mean of \p components.
template<typename MEAN_FUNCTION>
double meanOf(MEAN_FUNCTION mean, const TSeasonalComponentVec& components) {
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
    for (const auto& component : components) {
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
    for (const auto& window : windows) {
        double weight{static_cast<double>(window.first.second - window.first.first)};
        windowed.add(window.second, weight);
    }

    return unwindowed + CBasicStatistics::mean(windowed);
}

//! Compute the values to add to the trend and each component.
//!
//! \param[in] trend The long term trend prediction.
//! \param[in] seasonal The seasonal components.
//! \param[in] calendar The calendar components.
//! \param[in] time The time of value to decompose.
//! \param[in] deltas The delta offset to apply to the difference
//! between each component value and its mean, used to minimize
//! slope in the longer periods.
//! \param[in,out] decomposition Updated to contain the value to
//! add to each by component.
//! \param[out] predictions Filled in with the component predictions.
//! \param[out] referenceError Filled in with the error w.r.t. the trend.
//! \param[out] error Filled in with the prediction error.
//! \param[out] scale Filled in with the normalization scaling.
void decompose(double trend,
               const TSeasonalComponentPtrVec& seasonal,
               const TCalendarComponentPtrVec& calendar,
               const core_t::TTime time,
               const TDoubleVec& deltas,
               double gain,
               TDoubleVec& decomposition,
               TDoubleVec& predictions,
               double& referenceError,
               double& error,
               double& scale) {
    std::size_t m{seasonal.size()};
    std::size_t n{calendar.size()};

    double x0{trend};
    TDoubleVec x(m + n);
    double xhat{x0};
    for (std::size_t i = 0u; i < m; ++i) {
        x[i] = CBasicStatistics::mean(seasonal[i]->value(time, 0.0));
        xhat += x[i];
    }
    for (std::size_t i = m; i < m + n; ++i) {
        x[i] = CBasicStatistics::mean(calendar[i - m]->value(time, 0.0));
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
    double Z{std::max(static_cast<double>(m + n + 1) / gain, 1.0)};

    error = decomposition[0] - xhat;
    referenceError = decomposition[0] - x0;
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

//! Propagate \p target forwards to account for \p end - \p start elapsed
//! time in steps or size \p step.
template<typename PTR>
double stepwisePropagateForwards(const PTR& target,
                                 core_t::TTime start,
                                 core_t::TTime end,
                                 core_t::TTime step,
                                 double scale = 1.0) {
    double time{0.0};
    if (target) {
        start = CIntegerTools::floor(start, step);
        end = CIntegerTools::floor(end, step);
        if (end > start) {
            time = scale * static_cast<double>(end - start) / static_cast<double>(step);
            target->propagateForwardsByTime(time);
        }
    }
    return time;
}

// Periodicity Test State Machine

// States
const std::size_t PT_INITIAL = 0;
const std::size_t PT_TEST = 1;
const std::size_t PT_NOT_TESTING = 2;
const std::size_t PT_ERROR = 3;
const TStrVec PT_STATES{"INITIAL", "TEST", "NOT_TESTING", "ERROR"};
// Alphabet
const std::size_t PT_NEW_VALUE = 0;
const std::size_t PT_RESET = 1;
const TStrVec PT_ALPHABET{"NEW_VALUE", "RESET"};
// Transition Function
const TSizeVecVec PT_TRANSITION_FUNCTION{
    TSizeVec{PT_TEST, PT_TEST, PT_NOT_TESTING, PT_ERROR},
    TSizeVec{PT_INITIAL, PT_INITIAL, PT_NOT_TESTING, PT_INITIAL}};

// Calendar Cyclic Test State Machine

// States
const std::size_t CC_INITIAL = 0;
const std::size_t CC_TEST = 1;
const std::size_t CC_NOT_TESTING = 2;
const std::size_t CC_ERROR = 3;
const TStrVec CC_STATES{"INITIAL", "TEST", "NOT_TESTING", "ERROR"};
// Alphabet
const std::size_t CC_NEW_VALUE = 0;
const std::size_t CC_RESET = 1;
const TStrVec CC_ALPHABET{"NEW_VALUE", "RESET"};
// Transition Function
const TSizeVecVec CC_TRANSITION_FUNCTION{
    TSizeVec{CC_TEST, CC_TEST, CC_NOT_TESTING, CC_ERROR},
    TSizeVec{CC_INITIAL, CC_INITIAL, CC_NOT_TESTING, CC_INITIAL}};

// Components State Machine

// States
const std::size_t SC_NEW_COMPONENTS = 0;
const std::size_t SC_NORMAL = 1;
const std::size_t SC_DISABLED = 2;
const std::size_t SC_ERROR = 3;
const TStrVec SC_STATES{"NEW_COMPONENTS", "NORMAL", "DISABLED", "ERROR"};
// Alphabet
const std::size_t SC_ADDED_COMPONENTS = 0;
const std::size_t SC_INTERPOLATED = 1;
const std::size_t SC_RESET = 2;
const TStrVec SC_ALPHABET{"ADDED_COMPONENTS", "INTERPOLATED", "RESET"};
// Transition Function
const TSizeVecVec SC_TRANSITION_FUNCTION{
    TSizeVec{SC_NEW_COMPONENTS, SC_NEW_COMPONENTS, SC_DISABLED, SC_ERROR},
    TSizeVec{SC_NORMAL, SC_NORMAL, SC_DISABLED, SC_ERROR},
    TSizeVec{SC_NORMAL, SC_NORMAL, SC_NORMAL, SC_NORMAL}};

const std::string VERSION_6_3_TAG("6.3");
const std::string VERSION_6_4_TAG("6.4");

// Periodicity Test Tags
const core::TPersistenceTag LINEAR_SCALES_7_2_TAG{"d", "linear_scales"};
// Version 6.3
const core::TPersistenceTag PERIODICITY_TEST_MACHINE_6_3_TAG{"a", "periodicity_test_machine"};
const core::TPersistenceTag SHORT_WINDOW_6_3_TAG{"b", "short_window"};
const core::TPersistenceTag LONG_WINDOW_6_3_TAG{"c", "long_window"};
// Old versions can't be restored.

// Calendar Cyclic Test Tags
// Version 6.3
const core::TPersistenceTag CALENDAR_TEST_MACHINE_6_3_TAG{"a", "calendar_test_machine"};
const core::TPersistenceTag LAST_MONTH_6_3_TAG{"b", "last_month"};
const core::TPersistenceTag CALENDAR_TEST_6_3_TAG{"c", "calendar_test"};
// These work for all versions.

// Components Tags
// Version 6.5
const core::TPersistenceTag TESTING_FOR_CHANGE_6_5_TAG{"m", "testing_for_change"};
// Version 6.4
const core::TPersistenceTag COMPONENT_6_4_TAG{"f", "component"};
const core::TPersistenceTag ERRORS_6_4_TAG{"g", "errors"};
const core::TPersistenceTag REGRESSION_ORIGIN_6_4_TAG{"a", "regression_origin"};
const core::TPersistenceTag MEAN_SUM_AMPLITUDES_6_4_TAG{"b", "mean_sum_amplitudes"};
const core::TPersistenceTag MEAN_SUM_AMPLITUDES_TREND_6_4_TAG{"c", "mean_sum_amplitudes_trend"};
// Version 6.3
const core::TPersistenceTag COMPONENTS_MACHINE_6_3_TAG{"a", "components_machine"};
const core::TPersistenceTag DECAY_RATE_6_3_TAG{"b", "decay_rate"};
const core::TPersistenceTag TREND_6_3_TAG{"c", "trend"};
const core::TPersistenceTag SEASONAL_6_3_TAG{"d", "seasonal"};
const core::TPersistenceTag CALENDAR_6_3_TAG{"e", "calendar"};
const core::TPersistenceTag COMPONENT_6_3_TAG{"f", "component"};
const core::TPersistenceTag MEAN_VARIANCE_SCALE_6_3_TAG{"h", "mean_variance_scale"};
const core::TPersistenceTag MOMENTS_6_3_TAG{"i", "moments"};
const core::TPersistenceTag MOMENTS_MINUS_TREND_6_3_TAG{"j", "moments_minus_trend"};
const core::TPersistenceTag USING_TREND_FOR_PREDICTION_6_3_TAG{"k", "using_trend_for_prediction"};
const core::TPersistenceTag GAIN_CONTROLLER_6_3_TAG{"l", "gain_controller"};
// Version < 6.3
const std::string COMPONENTS_MACHINE_OLD_TAG{"a"};
const std::string TREND_OLD_TAG{"b"};
const std::string SEASONAL_OLD_TAG{"c"};
const std::string CALENDAR_OLD_TAG{"d"};
const std::string COMPONENT_OLD_TAG{"e"};
const std::string REGRESSION_OLD_TAG{"g"};
const std::string VARIANCE_OLD_TAG{"h"};
const std::string TIME_ORIGIN_OLD_TAG{"i"};
const std::string LAST_UPDATE_OLD_TAG{"j"};

//////////////////////// Upgrade to Version 6.3 ////////////////////////

const double MODEL_WEIGHT_UPGRADING_TO_VERSION_6_3{48.0};

bool upgradeTrendModelToVersion_6_3(const core_t::TTime bucketLength,
                                    const core_t::TTime lastValueTime,
                                    CTrendComponent& trend,
                                    core::CStateRestoreTraverser& traverser) {
    using TRegression = CLeastSquaresOnlineRegression<3, double>;

    TRegression regression;
    double variance{0.0};
    core_t::TTime origin{0};
    do {
        const std::string& name{traverser.name()};
        RESTORE(REGRESSION_OLD_TAG, traverser.traverseSubLevel(std::bind(
                                        &TRegression::acceptRestoreTraverser,
                                        &regression, std::placeholders::_1)))
        RESTORE_BUILT_IN(VARIANCE_OLD_TAG, variance)
        RESTORE_BUILT_IN(TIME_ORIGIN_OLD_TAG, origin)
    } while (traverser.next());

    // Generate some samples from the old trend model.

    double weight{MODEL_WEIGHT_UPGRADING_TO_VERSION_6_3 *
                  static_cast<double>(bucketLength) / static_cast<double>(4 * WEEK)};

    CPRNG::CXorOShiro128Plus rng;
    for (core_t::TTime time = lastValueTime - 4 * WEEK; time < lastValueTime;
         time += bucketLength) {
        double time_{static_cast<double>(time - origin) / static_cast<double>(WEEK)};
        double sample{regression.predict(time_) + CSampling::normalSample(rng, 0.0, variance)};
        trend.add(time, sample, weight);
    }

    return true;
}

// This implements the mapping from restored states to their best
// equivalents; specifically:
// SC_NEW_COMPONENTS |-> SC_NEW_COMPONENTS
// SC_NORMAL |-> SC_NORMAL
// SC_FORECASTING |-> SC_NORMAL
// SC_DISABLED |-> SC_DISABLED
// SC_ERROR |-> SC_ERROR
// Note that we don't try and restore the periodicity test state
// (see CTimeSeriesDecomposition::acceptRestoreTraverser) and the
// calendar test state is unchanged.
const TSizeSizeMap SC_STATES_UPGRADING_TO_VERSION_6_3{{0, 0}, {1, 1}, {2, 1}, {3, 2}, {4, 3}};

////////////////////////////////////////////////////////////////////////

// Constants
const std::size_t MAXIMUM_COMPONENTS{8};
const TSeasonalComponentVec NO_SEASONAL_COMPONENTS;
const TCalendarComponentVec NO_CALENDAR_COMPONENTS;
}

//////// SMessage ////////

CTimeSeriesDecompositionDetail::SMessage::SMessage(core_t::TTime time, core_t::TTime lastTime)
    : s_Time{time}, s_LastTime{lastTime} {
}

//////// SAddValue ////////

CTimeSeriesDecompositionDetail::SAddValue::SAddValue(
    core_t::TTime time,
    core_t::TTime lastTime,
    double value,
    const maths_t::TDoubleWeightsAry& weights,
    double trend,
    double seasonal,
    double calendar,
    const TPredictor& predictor,
    const CPeriodicityHypothesisTestsConfig& periodicityTestConfig,
    TComponentChangeCallback componentChangeCallback)
    : SMessage{time, lastTime}, s_Value{value}, s_Weights{weights}, s_Trend{trend},
      s_Seasonal{seasonal}, s_Calendar{calendar}, s_Predictor{predictor},
      s_PeriodicityTestConfig{periodicityTestConfig}, s_ComponentChangeCallback{componentChangeCallback} {
}

//////// SDetectedSeasonal ////////

CTimeSeriesDecompositionDetail::SDetectedSeasonal::SDetectedSeasonal(
    core_t::TTime time,
    core_t::TTime lastTime,
    const CPeriodicityHypothesisTestsResult& result,
    const CExpandingWindow& window,
    const TPredictor& predictor)
    : SMessage{time, lastTime}, s_Result{result}, s_Window{window}, s_Predictor{predictor} {
}

//////// SDetectedCalendar ////////

CTimeSeriesDecompositionDetail::SDetectedCalendar::SDetectedCalendar(core_t::TTime time,
                                                                     core_t::TTime lastTime,
                                                                     CCalendarFeature feature)
    : SMessage{time, lastTime}, s_Feature{feature} {
}

//////// SNewComponent ////////

CTimeSeriesDecompositionDetail::SNewComponents::SNewComponents(core_t::TTime time,
                                                               core_t::TTime lastTime,
                                                               EComponent component)
    : SMessage{time, lastTime}, s_Component{component} {
}

//////// CHandler ////////

void CTimeSeriesDecompositionDetail::CHandler::handle(const SAddValue& /*message*/) {
}

void CTimeSeriesDecompositionDetail::CHandler::handle(const SDetectedSeasonal& /*message*/) {
}

void CTimeSeriesDecompositionDetail::CHandler::handle(const SDetectedCalendar& /*message*/) {
}

void CTimeSeriesDecompositionDetail::CHandler::handle(const SNewComponents& /*message*/) {
}

void CTimeSeriesDecompositionDetail::CHandler::mediator(CMediator* mediator) {
    m_Mediator = mediator;
}

CTimeSeriesDecompositionDetail::CMediator*
CTimeSeriesDecompositionDetail::CHandler::mediator() const {
    return m_Mediator;
}

//////// CMediator ////////

template<typename M>
void CTimeSeriesDecompositionDetail::CMediator::forward(const M& message) const {
    for (CHandler& handler : m_Handlers) {
        handler.handle(message);
    }
}

void CTimeSeriesDecompositionDetail::CMediator::registerHandler(CHandler& handler) {
    m_Handlers.push_back(std::ref(handler));
    handler.mediator(this);
}

void CTimeSeriesDecompositionDetail::CMediator::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CMediator");
    core::CMemoryDebug::dynamicSize("m_Handlers", m_Handlers, mem);
}

std::size_t CTimeSeriesDecompositionDetail::CMediator::memoryUsage() const {
    return core::CMemory::dynamicSize(m_Handlers);
}

//////// CPeriodicityTest ////////

CTimeSeriesDecompositionDetail::CPeriodicityTest::CPeriodicityTest(double decayRate,
                                                                   core_t::TTime bucketLength)
    : m_Machine{core::CStateMachine::create(
          PT_ALPHABET,
          PT_STATES,
          PT_TRANSITION_FUNCTION,
          bucketLength > LONGEST_BUCKET_LENGTH ? PT_NOT_TESTING : PT_INITIAL)},
      m_DecayRate{decayRate}, m_BucketLength{bucketLength} {
}

CTimeSeriesDecompositionDetail::CPeriodicityTest::CPeriodicityTest(const CPeriodicityTest& other,
                                                                   bool isForForecast)
    : CHandler(), m_Machine{other.m_Machine}, m_DecayRate{other.m_DecayRate},
      m_BucketLength{other.m_BucketLength} {
    // Note that m_Windows is an array.
    for (std::size_t i = 0u; !isForForecast && i < other.m_Windows.size(); ++i) {
        if (other.m_Windows[i] != nullptr) {
            m_Windows[i] = std::make_unique<CExpandingWindow>(*other.m_Windows[i]);
        }
    }
}

bool CTimeSeriesDecompositionDetail::CPeriodicityTest::acceptRestoreTraverser(
    core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name{traverser.name()};
        RESTORE(PERIODICITY_TEST_MACHINE_6_3_TAG,
                traverser.traverseSubLevel([this](core::CStateRestoreTraverser& traverser_) {
                    return m_Machine.acceptRestoreTraverser(traverser_);
                }))
        RESTORE_SETUP_TEARDOWN(
            SHORT_WINDOW_6_3_TAG, m_Windows[E_Short].reset(this->newWindow(E_Short)),
            m_Windows[E_Short] && traverser.traverseSubLevel(std::bind(
                                      &CExpandingWindow::acceptRestoreTraverser,
                                      m_Windows[E_Short].get(), std::placeholders::_1)),
            /**/)
        RESTORE_SETUP_TEARDOWN(
            LONG_WINDOW_6_3_TAG, m_Windows[E_Long].reset(this->newWindow(E_Long)),
            m_Windows[E_Long] && traverser.traverseSubLevel(std::bind(
                                     &CExpandingWindow::acceptRestoreTraverser,
                                     m_Windows[E_Long].get(), std::placeholders::_1)),
            /**/)
        RESTORE(LINEAR_SCALES_7_2_TAG,
                core::CPersistUtils::restore(LINEAR_SCALES_7_2_TAG, m_LinearScales, traverser))
    } while (traverser.next());
    return true;
}

void CTimeSeriesDecompositionDetail::CPeriodicityTest::acceptPersistInserter(
    core::CStatePersistInserter& inserter) const {
    inserter.insertLevel(PERIODICITY_TEST_MACHINE_6_3_TAG,
                         std::bind(&core::CStateMachine::acceptPersistInserter,
                                   &m_Machine, std::placeholders::_1));
    if (m_Windows[E_Short] != nullptr) {
        inserter.insertLevel(SHORT_WINDOW_6_3_TAG,
                             std::bind(&CExpandingWindow::acceptPersistInserter,
                                       m_Windows[E_Short].get(), std::placeholders::_1));
    }
    if (m_Windows[E_Long] != nullptr) {
        inserter.insertLevel(LONG_WINDOW_6_3_TAG,
                             std::bind(&CExpandingWindow::acceptPersistInserter,
                                       m_Windows[E_Long].get(), std::placeholders::_1));
    }
    core::CPersistUtils::persist(LINEAR_SCALES_7_2_TAG, m_LinearScales, inserter);
}

void CTimeSeriesDecompositionDetail::CPeriodicityTest::swap(CPeriodicityTest& other) {
    std::swap(m_Machine, other.m_Machine);
    std::swap(m_DecayRate, other.m_DecayRate);
    std::swap(m_BucketLength, other.m_BucketLength);
    m_Windows[E_Short].swap(other.m_Windows[E_Short]);
    m_Windows[E_Long].swap(other.m_Windows[E_Long]);
}

void CTimeSeriesDecompositionDetail::CPeriodicityTest::handle(const SAddValue& message) {

    core_t::TTime time{message.s_Time};
    double value{message.s_Value};
    const maths_t::TDoubleWeightsAry& weights{message.s_Weights};
    double weight{maths_t::countForUpdate(weights)};

    this->test(message);

    switch (m_Machine.state()) {
    case PT_TEST:
        for (auto& window : m_Windows) {
            if (window != nullptr) {
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
        LOG_ERROR(<< "Test in a bad state: " << m_Machine.state());
        this->apply(PT_RESET, message);
        break;
    }
}

void CTimeSeriesDecompositionDetail::CPeriodicityTest::handle(const SNewComponents& /*message*/) {
    // This can be a no-op because we always maintain the raw time
    // series values in the windows and apply corrections for other
    // components only when we test.
}

void CTimeSeriesDecompositionDetail::CPeriodicityTest::test(const SAddValue& message) {
    core_t::TTime time{message.s_Time};
    core_t::TTime lastTime{message.s_LastTime};
    const TPredictor& predictor{message.s_Predictor};
    const CPeriodicityHypothesisTestsConfig& config{message.s_PeriodicityTestConfig};

    switch (m_Machine.state()) {
    case PT_TEST:
        for (auto i : {E_Short, E_Long}) {
            if (this->shouldTest(i, time)) {

                this->pruneLinearScales();

                const auto& window = m_Windows[i];

                TFloatMeanAccumulatorVec values(
                    window->valuesMinusPrediction(this->scaledPredictor(predictor)));

                core_t::TTime start{CIntegerTools::floor(window->startTime(), m_BucketLength)};
                core_t::TTime bucketLength{window->bucketLength()};
                CPeriodicityHypothesisTestsResult result{
                    testForPeriods(config, start, bucketLength, values)};
                result.remove([i](const CPeriodicityHypothesisTestsResult::SComponent& component) {
                    return i == E_Long && component.s_Period <= WEEK;
                });

                if (result.periodic()) {
                    this->mediator()->forward(SDetectedSeasonal{
                        time, lastTime, result, *window, predictor});
                }
            }
        }
        break;
    case PT_NOT_TESTING:
    case PT_INITIAL:
        break;
    default:
        LOG_ERROR(<< "Test in a bad state: " << m_Machine.state());
        this->apply(PT_RESET, message);
        break;
    }
}

void CTimeSeriesDecompositionDetail::CPeriodicityTest::linearScale(core_t::TTime time,
                                                                   double scale) {
    m_LinearScales.emplace_back(time, scale);
}

void CTimeSeriesDecompositionDetail::CPeriodicityTest::shiftTime(core_t::TTime dt) {
    for (auto& window : m_Windows) {
        if (window != nullptr) {
            window->shiftTime(dt);
        }
    }
}

void CTimeSeriesDecompositionDetail::CPeriodicityTest::propagateForwards(core_t::TTime start,
                                                                         core_t::TTime end) {
    stepwisePropagateForwards(m_Windows[E_Short], start, end, DAY);
    stepwisePropagateForwards(m_Windows[E_Long], start, end, WEEK);
}

CTimeSeriesDecompositionDetail::TFloatMeanAccumulatorVec
CTimeSeriesDecompositionDetail::CPeriodicityTest::windowValues(const TPredictor& predictor) const {
    TFloatMeanAccumulatorVec result;
    for (auto i : {E_Short, E_Long}) {
        if (m_Windows[i] != nullptr) {
            result = m_Windows[i]->valuesMinusPrediction(this->scaledPredictor(predictor));
            break;
        }
    }
    return result;
}

uint64_t CTimeSeriesDecompositionDetail::CPeriodicityTest::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_Machine);
    seed = CChecksum::calculate(seed, m_DecayRate);
    seed = CChecksum::calculate(seed, m_BucketLength);
    seed = CChecksum::calculate(seed, m_Windows);
    return CChecksum::calculate(seed, m_LinearScales);
}

void CTimeSeriesDecompositionDetail::CPeriodicityTest::debugMemoryUsage(
    core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CPeriodicityTest");
    core::CMemoryDebug::dynamicSize("m_Windows", m_Windows, mem);
    core::CMemoryDebug::dynamicSize("m_LinearScales", m_LinearScales, mem);
}

std::size_t CTimeSeriesDecompositionDetail::CPeriodicityTest::memoryUsage() const {
    std::size_t usage{core::CMemory::dynamicSize(m_Windows)};
    if (m_Machine.state() == PT_INITIAL) {
        usage += this->extraMemoryOnInitialization();
    }
    return usage + core::CMemory::dynamicSize(m_LinearScales);
}

std::size_t CTimeSeriesDecompositionDetail::CPeriodicityTest::extraMemoryOnInitialization() const {
    static std::size_t result{0};
    if (result == 0) {
        for (auto i : {E_Short, E_Long}) {
            TExpandingWindowPtr window(this->newWindow(i, false));
            // The 0.3 is a rule-of-thumb estimate of the worst case
            // compression ratio we achieve on the test state.
            result += static_cast<std::size_t>(
                0.3 * static_cast<double>(core::CMemory::dynamicSize(window)));
        }
    }
    return result;
}

void CTimeSeriesDecompositionDetail::CPeriodicityTest::apply(std::size_t symbol,
                                                             const SMessage& message) {
    core_t::TTime time{message.s_Time};

    std::size_t old{m_Machine.state()};
    m_Machine.apply(symbol);
    std::size_t state{m_Machine.state()};

    if (state != old) {
        LOG_TRACE(<< PT_STATES[old] << "," << PT_ALPHABET[symbol] << " -> "
                  << PT_STATES[state]);

        auto initialize = [this](core_t::TTime time_) {
            for (auto i : {E_Short, E_Long}) {
                m_Windows[i].reset(this->newWindow(i));
                if (m_Windows[i] != nullptr) {
                    // Since all permitted bucket lengths are divisors
                    // of longer ones, this finds the unique rightmost
                    // time which is an integer multiple of all windows'
                    // bucket lengths. It is important to align start
                    // times so that we try the short test first when
                    // testing for shorter periods.
                    time_ = CIntegerTools::floor(time_, m_Windows[i]->bucketLength());
                }
            }
            for (auto& window : m_Windows) {
                if (window != nullptr) {
                    window->initialize(time_);
                }
            }
        };

        switch (state) {
        case PT_TEST:
            if (std::all_of(m_Windows.begin(), m_Windows.end(),
                            [](const TExpandingWindowPtr& window) {
                                return window == nullptr;
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
            LOG_ERROR(<< "Test in a bad state: " << state);
            this->apply(PT_RESET, message);
            break;
        }
    }
}

CExpandingWindow*
CTimeSeriesDecompositionDetail::CPeriodicityTest::newWindow(ETest test, bool deflate) const {

    using TTimeCRng = CExpandingWindow::TTimeCRng;

    // The choice of 336 is somewhat arbitrary since we scan over a
    // range of bucket lengths so consider multiple window durations.
    // However, this is divisible by 6 and is the number of hours in
    // two weeks. Together with the bucket lengths we use this means
    // we will typically get window lengths which are a whole number
    // of hours and are usually a multiple of one weeks. The testing
    // makes the best use of data when the true period is a divisor
    // of the window length and this improves the odds for the most
    // common seasonalities we observe.
    static const core_t::TTime TARGET_SIZE{336};
    static const TTimeVec SHORT{1,    5,    10,   30,    60,    300,  600,
                                1800, 3600, 7200, 21600, 43200, 86400};
    static const TTimeVec LONG{7200,  21600,  43200,
                               86400, 172800, LONGEST_BUCKET_LENGTH};

    auto newWindow = [&](const TTimeVec& buckets, core_t::TTime maxLength) {
        if (m_BucketLength <= buckets.back()) {
            // Find the next longer permitted bucket length.
            auto a = std::lower_bound(buckets.begin(), buckets.end(), m_BucketLength);
            // Compute the window length in buckets which doesn't exceed
            // the maximum length and is great than the minimum number of
            // buckets to test.
            core_t::TTime size{
                std::min(TARGET_SIZE, maxLength / buckets[a - buckets.begin()])};
            // Find the longest bucket length such that the window length
            // doesn't exceed the maximum length.
            auto b = std::find_if(a + 1, buckets.end(), [&](core_t::TTime l) {
                return size * l > maxLength;
            });
            return new CExpandingWindow(
                m_BucketLength,
                TTimeCRng(buckets, a - buckets.begin(), b - buckets.begin()),
                size, m_DecayRate, deflate);
        }
        return static_cast<CExpandingWindow*>(nullptr);
    };

    switch (test) {
    case E_Short:
        return newWindow(SHORT, 7200 * TARGET_SIZE);
    case E_Long:
        return newWindow(LONG, LONGEST_BUCKET_LENGTH * TARGET_SIZE);
    }
    return nullptr;
}

bool CTimeSeriesDecompositionDetail::CPeriodicityTest::shouldTest(ETest test,
                                                                  core_t::TTime time) const {
    // We need to test more frequently than we compress because it
    // would significantly delay when we first detect short periodic
    // components for longer bucket lengths otherwise.
    auto scheduledTest = [&]() {
        if (test == E_Short) {
            core_t::TTime length{time - m_Windows[test]->startTime()};
            for (auto schedule : {3 * DAY, 1 * WEEK, 2 * WEEK}) {
                if (length >= schedule && length < schedule + m_BucketLength) {
                    return true;
                }
            }
        }
        return false;
    };
    return m_Windows[test] != nullptr &&
           (m_Windows[test]->needToCompress(time) || scheduledTest());
}

CTimeSeriesDecompositionDetail::TPredictor
CTimeSeriesDecompositionDetail::CPeriodicityTest::scaledPredictor(const TPredictor& predictor) const {
    return [&predictor, this](core_t::TTime time) {
        auto i = std::lower_bound(m_LinearScales.begin(), m_LinearScales.end(), time,
                                  [](const TTimeDoublePr& lhs, core_t::TTime rhs) {
                                      return lhs.first < rhs;
                                  });
        return std::accumulate(i, m_LinearScales.end(), 1.0,
                               [](double quotient, const TTimeDoublePr& scale) {
                                   return quotient / scale.second;
                               }) *
               predictor(time);
    };
}

void CTimeSeriesDecompositionDetail::CPeriodicityTest::pruneLinearScales() {
    core_t::TTime cutoff{std::numeric_limits<core_t::TTime>::max()};
    for (const auto& window : m_Windows) {
        if (window != nullptr) {
            cutoff = std::min(cutoff, window->startTime());
        }
    }
    m_LinearScales.erase(std::remove_if(m_LinearScales.begin(), m_LinearScales.end(),
                                        [cutoff](const TTimeDoublePr& scale) {
                                            return scale.first < cutoff;
                                        }),
                         m_LinearScales.end());
}

const core_t::TTime CTimeSeriesDecompositionDetail::CPeriodicityTest::LONGEST_BUCKET_LENGTH{345600};

//////// CCalendarCyclic ////////

CTimeSeriesDecompositionDetail::CCalendarTest::CCalendarTest(double decayRate,
                                                             core_t::TTime bucketLength)
    : m_Machine{core::CStateMachine::create(CC_ALPHABET,
                                            CC_STATES,
                                            CC_TRANSITION_FUNCTION,
                                            bucketLength > DAY ? CC_NOT_TESTING : CC_INITIAL)},
      m_DecayRate{decayRate}, m_LastMonth{} {
}

CTimeSeriesDecompositionDetail::CCalendarTest::CCalendarTest(const CCalendarTest& other,
                                                             bool isForForecast)
    : CHandler(), m_Machine{other.m_Machine}, m_DecayRate{other.m_DecayRate},
      m_LastMonth{other.m_LastMonth}, m_Test{!isForForecast && other.m_Test
                                                 ? std::make_unique<CCalendarCyclicTest>(
                                                       *other.m_Test)
                                                 : nullptr} {
}

bool CTimeSeriesDecompositionDetail::CCalendarTest::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name{traverser.name()};
        RESTORE(CALENDAR_TEST_MACHINE_6_3_TAG,
                traverser.traverseSubLevel([this](core::CStateRestoreTraverser& traverser_) {
                    return m_Machine.acceptRestoreTraverser(traverser_);
                }))
        RESTORE_BUILT_IN(LAST_MONTH_6_3_TAG, m_LastMonth);
        RESTORE_SETUP_TEARDOWN(
            CALENDAR_TEST_6_3_TAG,
            m_Test = std::make_unique<CCalendarCyclicTest>(m_DecayRate),
            traverser.traverseSubLevel(std::bind(&CCalendarCyclicTest::acceptRestoreTraverser,
                                                 m_Test.get(), std::placeholders::_1)),
            /**/)
    } while (traverser.next());
    return true;
}

void CTimeSeriesDecompositionDetail::CCalendarTest::acceptPersistInserter(
    core::CStatePersistInserter& inserter) const {
    inserter.insertLevel(CALENDAR_TEST_MACHINE_6_3_TAG,
                         std::bind(&core::CStateMachine::acceptPersistInserter,
                                   &m_Machine, std::placeholders::_1));
    inserter.insertValue(LAST_MONTH_6_3_TAG, m_LastMonth);
    if (m_Test != nullptr) {
        inserter.insertLevel(CALENDAR_TEST_6_3_TAG,
                             std::bind(&CCalendarCyclicTest::acceptPersistInserter,
                                       m_Test.get(), std::placeholders::_1));
    }
}

void CTimeSeriesDecompositionDetail::CCalendarTest::swap(CCalendarTest& other) {
    std::swap(m_Machine, other.m_Machine);
    std::swap(m_DecayRate, other.m_DecayRate);
    std::swap(m_LastMonth, other.m_LastMonth);
    m_Test.swap(other.m_Test);
}

void CTimeSeriesDecompositionDetail::CCalendarTest::handle(const SAddValue& message) {
    core_t::TTime time{message.s_Time};
    double error{message.s_Value - message.s_Trend - message.s_Seasonal -
                 message.s_Calendar};
    const maths_t::TDoubleWeightsAry& weights{message.s_Weights};

    this->test(message);

    switch (m_Machine.state()) {
    case CC_TEST:
        m_Test->add(time, error, maths_t::countForUpdate(weights));
        break;
    case CC_NOT_TESTING:
        break;
    case CC_INITIAL:
        this->apply(CC_NEW_VALUE, message);
        this->handle(message);
        break;
    default:
        LOG_ERROR(<< "Test in a bad state: " << m_Machine.state());
        this->apply(CC_RESET, message);
        break;
    }
}

void CTimeSeriesDecompositionDetail::CCalendarTest::handle(const SNewComponents& message) {
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

void CTimeSeriesDecompositionDetail::CCalendarTest::test(const SMessage& message) {
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
            LOG_ERROR(<< "Test in a bad state: " << m_Machine.state());
            this->apply(CC_RESET, message);
            break;
        }
    }
}

void CTimeSeriesDecompositionDetail::CCalendarTest::propagateForwards(core_t::TTime start,
                                                                      core_t::TTime end) {
    stepwisePropagateForwards(m_Test, start, end, DAY);
}

uint64_t CTimeSeriesDecompositionDetail::CCalendarTest::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_Machine);
    seed = CChecksum::calculate(seed, m_DecayRate);
    seed = CChecksum::calculate(seed, m_LastMonth);
    return CChecksum::calculate(seed, m_Test);
}

void CTimeSeriesDecompositionDetail::CCalendarTest::debugMemoryUsage(
    core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CCalendarTest");
    core::CMemoryDebug::dynamicSize("m_Test", m_Test, mem);
}

std::size_t CTimeSeriesDecompositionDetail::CCalendarTest::memoryUsage() const {
    std::size_t usage{core::CMemory::dynamicSize(m_Test)};
    if (m_Machine.state() == CC_INITIAL) {
        usage += this->extraMemoryOnInitialization();
    }
    return usage;
}

std::size_t CTimeSeriesDecompositionDetail::CCalendarTest::extraMemoryOnInitialization() const {
    static std::size_t result{0};
    if (result == 0) {
        TCalendarCyclicTestPtr test = std::make_unique<CCalendarCyclicTest>(m_DecayRate);
        result = core::CMemory::dynamicSize(test);
    }
    return result;
}

void CTimeSeriesDecompositionDetail::CCalendarTest::apply(std::size_t symbol,
                                                          const SMessage& message) {
    core_t::TTime time{message.s_Time};

    std::size_t old{m_Machine.state()};
    m_Machine.apply(symbol);
    std::size_t state{m_Machine.state()};

    if (state != old) {
        LOG_TRACE(<< CC_STATES[old] << "," << CC_ALPHABET[symbol] << " -> "
                  << CC_STATES[state]);

        switch (state) {
        case CC_TEST:
            if (m_Test == nullptr) {
                m_Test = std::make_unique<CCalendarCyclicTest>(m_DecayRate);
                m_LastMonth = this->month(time) + 2;
            }
            break;
        case CC_NOT_TESTING:
        case CC_INITIAL:
            m_Test.reset();
            m_LastMonth = int{};
            break;
        default:
            LOG_ERROR(<< "Test in a bad state: " << state);
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

int CTimeSeriesDecompositionDetail::CCalendarTest::month(core_t::TTime time) const {
    int dummy;
    int month;
    core::CTimezone::instance().dateFields(time, dummy, dummy, dummy, month, dummy, dummy);
    return month;
}

//////// CComponents ////////

CTimeSeriesDecompositionDetail::CComponents::CComponents(double decayRate,
                                                         core_t::TTime bucketLength,
                                                         std::size_t seasonalComponentSize)
    : m_Machine{core::CStateMachine::create(SC_ALPHABET, SC_STATES, SC_TRANSITION_FUNCTION, SC_NORMAL)},
      m_DecayRate{decayRate}, m_BucketLength{bucketLength}, m_SeasonalComponentSize{seasonalComponentSize},
      m_CalendarComponentSize{seasonalComponentSize / 3}, m_Trend{decayRate} {
}

CTimeSeriesDecompositionDetail::CComponents::CComponents(const CComponents& other)
    : CHandler(), m_Machine{other.m_Machine}, m_DecayRate{other.m_DecayRate},
      m_BucketLength{other.m_BucketLength}, m_GainController{other.m_GainController},
      m_SeasonalComponentSize{other.m_SeasonalComponentSize},
      m_CalendarComponentSize{other.m_CalendarComponentSize}, m_Trend{other.m_Trend},
      m_Seasonal{other.m_Seasonal ? std::make_unique<CSeasonal>(*other.m_Seasonal) : nullptr},
      m_Calendar{other.m_Calendar ? std::make_unique<CCalendar>(*other.m_Calendar) : nullptr},
      m_MeanVarianceScale{other.m_MeanVarianceScale},
      m_PredictionErrorWithoutTrend{other.m_PredictionErrorWithoutTrend},
      m_PredictionErrorWithTrend{other.m_PredictionErrorWithTrend},
      m_UsingTrendForPrediction{other.m_UsingTrendForPrediction} {
}

bool CTimeSeriesDecompositionDetail::CComponents::acceptRestoreTraverser(
    const SDistributionRestoreParams& params,
    core_t::TTime lastValueTime,
    core::CStateRestoreTraverser& traverser) {
    if (traverser.name() == VERSION_6_3_TAG) {
        while (traverser.next()) {
            const std::string& name{traverser.name()};
            RESTORE(COMPONENTS_MACHINE_6_3_TAG,
                    traverser.traverseSubLevel([this](core::CStateRestoreTraverser& traverser_) {
                        return m_Machine.acceptRestoreTraverser(traverser_);
                    }))
            RESTORE_BUILT_IN(DECAY_RATE_6_3_TAG, m_DecayRate);
            RESTORE(GAIN_CONTROLLER_6_3_TAG,
                    traverser.traverseSubLevel(
                        std::bind(&CGainController::acceptRestoreTraverser,
                                  &m_GainController, std::placeholders::_1)))
            RESTORE(TREND_6_3_TAG, traverser.traverseSubLevel(std::bind(
                                       &CTrendComponent::acceptRestoreTraverser, &m_Trend,
                                       std::cref(params), std::placeholders::_1)))
            RESTORE_SETUP_TEARDOWN(
                SEASONAL_6_3_TAG, m_Seasonal = std::make_unique<CSeasonal>(),
                traverser.traverseSubLevel(
                    std::bind(&CSeasonal::acceptRestoreTraverser, m_Seasonal.get(),
                              m_DecayRate, m_BucketLength, std::placeholders::_1)),
                /**/)
            RESTORE_SETUP_TEARDOWN(
                CALENDAR_6_3_TAG, m_Calendar = std::make_unique<CCalendar>(),
                traverser.traverseSubLevel(
                    std::bind(&CCalendar::acceptRestoreTraverser, m_Calendar.get(),
                              m_DecayRate, m_BucketLength, std::placeholders::_1)),
                /**/)
            RESTORE(MEAN_VARIANCE_SCALE_6_3_TAG,
                    m_MeanVarianceScale.fromDelimited(traverser.value()))
            RESTORE(MOMENTS_6_3_TAG,
                    m_PredictionErrorWithoutTrend.fromDelimited(traverser.value()));
            RESTORE(MOMENTS_MINUS_TREND_6_3_TAG,
                    m_PredictionErrorWithTrend.fromDelimited(traverser.value()));
            RESTORE_BUILT_IN(TESTING_FOR_CHANGE_6_5_TAG, m_TestingForChange)
            RESTORE_BUILT_IN(USING_TREND_FOR_PREDICTION_6_3_TAG, m_UsingTrendForPrediction)
        }

        this->decayRate(m_DecayRate);
    } else {
        // There is no version string this is historic state.
        do {
            const std::string& name{traverser.name()};
            RESTORE(COMPONENTS_MACHINE_OLD_TAG,
                    traverser.traverseSubLevel([this](core::CStateRestoreTraverser& traverser_) {
                        return m_Machine.acceptRestoreTraverser(
                            traverser_, SC_STATES_UPGRADING_TO_VERSION_6_3);
                    }))
            RESTORE_SETUP_TEARDOWN(TREND_OLD_TAG,
                                   /**/,
                                   traverser.traverseSubLevel(std::bind(
                                       upgradeTrendModelToVersion_6_3, m_BucketLength, lastValueTime,
                                       std::ref(m_Trend), std::placeholders::_1)),
                                   m_UsingTrendForPrediction = true)
            RESTORE_SETUP_TEARDOWN(
                SEASONAL_OLD_TAG, m_Seasonal = std::make_unique<CSeasonal>(),
                traverser.traverseSubLevel(
                    std::bind(&CSeasonal::acceptRestoreTraverser, m_Seasonal.get(),
                              m_DecayRate, m_BucketLength, std::placeholders::_1)),
                /**/)
            RESTORE_SETUP_TEARDOWN(
                CALENDAR_OLD_TAG, m_Calendar = std::make_unique<CCalendar>(),
                traverser.traverseSubLevel(
                    std::bind(&CCalendar::acceptRestoreTraverser, m_Calendar.get(),
                              m_DecayRate, m_BucketLength, std::placeholders::_1)),
                /**/)
        } while (traverser.next());

        m_MeanVarianceScale.add(1.0, MODEL_WEIGHT_UPGRADING_TO_VERSION_6_3);
    }
    return true;
}

void CTimeSeriesDecompositionDetail::CComponents::acceptPersistInserter(
    core::CStatePersistInserter& inserter) const {

    inserter.insertValue(VERSION_6_3_TAG, "");
    inserter.insertLevel(COMPONENTS_MACHINE_6_3_TAG,
                         std::bind(&core::CStateMachine::acceptPersistInserter,
                                   &m_Machine, std::placeholders::_1));
    inserter.insertValue(DECAY_RATE_6_3_TAG, m_DecayRate, core::CIEEE754::E_SinglePrecision);
    inserter.insertLevel(GAIN_CONTROLLER_6_3_TAG,
                         std::bind(&CGainController::acceptPersistInserter,
                                   &m_GainController, std::placeholders::_1));
    inserter.insertLevel(TREND_6_3_TAG, std::bind(&CTrendComponent::acceptPersistInserter,
                                                  m_Trend, std::placeholders::_1));
    if (m_Seasonal) {
        inserter.insertLevel(SEASONAL_6_3_TAG,
                             std::bind(&CSeasonal::acceptPersistInserter,
                                       m_Seasonal.get(), std::placeholders::_1));
    }
    if (m_Calendar) {
        inserter.insertLevel(CALENDAR_6_3_TAG,
                             std::bind(&CCalendar::acceptPersistInserter,
                                       m_Calendar.get(), std::placeholders::_1));
    }
    inserter.insertValue(MEAN_VARIANCE_SCALE_6_3_TAG, m_MeanVarianceScale.toDelimited());
    inserter.insertValue(MOMENTS_6_3_TAG, m_PredictionErrorWithoutTrend.toDelimited());
    inserter.insertValue(MOMENTS_MINUS_TREND_6_3_TAG,
                         m_PredictionErrorWithTrend.toDelimited());
    inserter.insertValue(TESTING_FOR_CHANGE_6_5_TAG, m_TestingForChange);
    inserter.insertValue(USING_TREND_FOR_PREDICTION_6_3_TAG, m_UsingTrendForPrediction);
}

void CTimeSeriesDecompositionDetail::CComponents::swap(CComponents& other) {
    std::swap(m_Machine, other.m_Machine);
    std::swap(m_DecayRate, other.m_DecayRate);
    std::swap(m_BucketLength, other.m_BucketLength);
    std::swap(m_SeasonalComponentSize, other.m_SeasonalComponentSize);
    std::swap(m_CalendarComponentSize, other.m_CalendarComponentSize);
    m_Trend.swap(other.m_Trend);
    m_Seasonal.swap(other.m_Seasonal);
    m_Calendar.swap(other.m_Calendar);
    std::swap(m_GainController, other.m_GainController);
    std::swap(m_MeanVarianceScale, other.m_MeanVarianceScale);
    std::swap(m_PredictionErrorWithoutTrend, other.m_PredictionErrorWithoutTrend);
    std::swap(m_TestingForChange, other.m_TestingForChange);
    std::swap(m_PredictionErrorWithTrend, other.m_PredictionErrorWithTrend);
    std::swap(m_UsingTrendForPrediction, other.m_UsingTrendForPrediction);
}

void CTimeSeriesDecompositionDetail::CComponents::handle(const SAddValue& message) {
    switch (m_Machine.state()) {
    case SC_NORMAL:
    case SC_NEW_COMPONENTS: {
        this->interpolate(message);

        core_t::TTime time{message.s_Time};
        double value{message.s_Value};
        double trend{message.s_Trend};
        const maths_t::TDoubleWeightsAry& weights{message.s_Weights};
        m_ComponentChangeCallback = message.s_ComponentChangeCallback;

        TSeasonalComponentPtrVec seasonalComponents;
        TCalendarComponentPtrVec calendarComponents;
        TComponentErrorsPtrVec seasonalErrors;
        TComponentErrorsPtrVec calendarErrors;
        TDoubleVec deltas;

        if (m_Seasonal) {
            m_Seasonal->componentsErrorsAndDeltas(time, seasonalComponents,
                                                  seasonalErrors, deltas);
        }
        if (m_Calendar) {
            m_Calendar->componentsAndErrors(time, calendarComponents, calendarErrors);
        }

        double weight{maths_t::countForUpdate(weights)};
        std::size_t m{seasonalComponents.size()};
        std::size_t n{calendarComponents.size()};

        TDoubleVec values(m + n + 1, value);
        TDoubleVec predictions(m + n, 0.0);
        double referenceError;
        double error;
        double scale;
        decompose(trend, seasonalComponents, calendarComponents, time, deltas,
                  m_GainController.gain(), values, predictions, referenceError,
                  error, scale);

        TDoubleVec variances(m + n + 1, 0.0);
        if (m_UsingTrendForPrediction) {
            variances[0] = CBasicStatistics::mean(m_Trend.variance(0.0));
        }
        for (std::size_t i = 1u; i <= m; ++i) {
            variances[i] = CBasicStatistics::mean(
                seasonalComponents[i - 1]->variance(time, 0.0));
        }
        for (std::size_t i = m + 1; i <= m + n; ++i) {
            variances[i] = CBasicStatistics::mean(
                calendarComponents[i - m - 1]->variance(time, 0.0));
        }
        double variance{std::accumulate(variances.begin(), variances.end(), 0.0)};
        double expectedVarianceIncrease{1.0 / static_cast<double>(m + n + 1)};

        bool testForTrend{(m_TestingForChange == false) &&
                          (m_UsingTrendForPrediction == false) &&
                          (m_Trend.observedInterval() > 6 * m_BucketLength)};

        m_Trend.add(time, values[0], weight);
        m_Trend.dontShiftLevel(time, value);
        for (std::size_t i = 1; i <= m; ++i) {
            CSeasonalComponent* component{seasonalComponents[i - 1]};
            CComponentErrors* error_{seasonalErrors[i - 1]};
            double varianceIncrease{variance == 0.0 ? 1.0 : variances[i] / variance / expectedVarianceIncrease};
            component->add(time, values[i], weight);
            error_->add(referenceError, error, predictions[i - 1], varianceIncrease, weight);
        }
        for (std::size_t i = m + 1; i <= m + n; ++i) {
            CCalendarComponent* component{calendarComponents[i - m - 1]};
            CComponentErrors* error_{calendarErrors[i - m - 1]};
            double varianceIncrease{variance == 0.0 ? 1.0 : variances[i] / variance / expectedVarianceIncrease};
            component->add(time, values[i], weight);
            error_->add(referenceError, error, predictions[i - 1], varianceIncrease, weight);
        }

        m_MeanVarianceScale.add(scale, weight);
        m_PredictionErrorWithoutTrend.add(error + trend, weight);
        m_PredictionErrorWithTrend.add(error, weight);
        m_GainController.add(time, predictions);

        if (testForTrend && this->shouldUseTrendForPrediction()) {
            LOG_DEBUG(<< "Detected trend at " << time);
            m_ComponentChangeCallback({});
        }
    } break;
    case SC_DISABLED:
        break;
    default:
        LOG_ERROR(<< "Components in a bad state: " << m_Machine.state());
        this->apply(SC_RESET, message);
        break;
    }
}

void CTimeSeriesDecompositionDetail::CComponents::handle(const SDetectedSeasonal& message) {
    if (this->size() + m_SeasonalComponentSize > this->maxSize()) {
        return;
    }

    switch (m_Machine.state()) {
    case SC_NORMAL:
    case SC_NEW_COMPONENTS: {
        if (m_Seasonal == nullptr) {
            m_Seasonal = std::make_unique<CSeasonal>();
        }

        core_t::TTime time{message.s_Time};
        core_t::TTime lastTime{message.s_LastTime};
        const CPeriodicityHypothesisTestsResult& result{message.s_Result};
        const CExpandingWindow& window{message.s_Window};
        const TPredictor& predictor{message.s_Predictor};

        if (this->addSeasonalComponents(result, window, predictor)) {
            LOG_DEBUG(<< "Detected seasonal components at " << time);
            m_UsingTrendForPrediction = true;
            this->clearComponentErrors();
            this->apply(SC_ADDED_COMPONENTS, message);
            this->mediator()->forward(
                SNewComponents(time, lastTime, SNewComponents::E_GeneralSeasonal));
        }
        break;
    }
    case SC_DISABLED:
        break;
    default:
        LOG_ERROR(<< "Components in a bad state: " << m_Machine.state());
        this->apply(SC_RESET, message);
        break;
    }
}

void CTimeSeriesDecompositionDetail::CComponents::handle(const SDetectedCalendar& message) {
    if (this->size() + m_CalendarComponentSize > this->maxSize()) {
        return;
    }

    switch (m_Machine.state()) {
    case SC_NORMAL:
    case SC_NEW_COMPONENTS: {
        if (!m_Calendar) {
            m_Calendar = std::make_unique<CCalendar>();
        }

        core_t::TTime time{message.s_Time};
        core_t::TTime lastTime{message.s_LastTime};
        CCalendarFeature feature{message.s_Feature};

        if (m_Calendar->haveComponent(feature)) {
            break;
        }

        this->addCalendarComponent(feature, time);
        this->apply(SC_ADDED_COMPONENTS, message);
        this->mediator()->forward(
            SNewComponents(time, lastTime, SNewComponents::E_CalendarCyclic));
        break;
    }
    case SC_DISABLED:
        break;
    default:
        LOG_ERROR(<< "Components in a bad state: " << m_Machine.state());
        this->apply(SC_RESET, message);
        break;
    }
}

void CTimeSeriesDecompositionDetail::CComponents::testingForChange(bool value) {
    m_TestingForChange = value;
}

void CTimeSeriesDecompositionDetail::CComponents::shiftLevel(core_t::TTime time,
                                                             double value,
                                                             double shift) {
    m_Trend.shiftLevel(time, value, shift);
}

void CTimeSeriesDecompositionDetail::CComponents::linearScale(core_t::TTime time, double scale) {
    m_Trend.linearScale(scale);
    if (m_Seasonal) {
        m_Seasonal->linearScale(time, scale);
    }
    if (m_Calendar) {
        m_Calendar->linearScale(time, scale);
    }
}

void CTimeSeriesDecompositionDetail::CComponents::interpolateForForecast(core_t::TTime time) {
    if (this->shouldInterpolate(time, time - m_BucketLength)) {
        if (m_Seasonal) {
            m_Seasonal->interpolate(time, time - m_BucketLength, false);
        }
        if (m_Calendar) {
            m_Calendar->interpolate(time, time - m_BucketLength, true);
        }
    }
}

void CTimeSeriesDecompositionDetail::CComponents::dataType(maths_t::EDataType dataType) {
    m_Trend.dataType(dataType);
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

double CTimeSeriesDecompositionDetail::CComponents::decayRate() const {
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
    double factor{ageFactor(m_DecayRate, end - start)};
    m_MeanVarianceScale.age(factor);
    m_PredictionErrorWithTrend.age(factor);
    m_PredictionErrorWithoutTrend.age(factor);
    m_GainController.age(factor);
}

bool CTimeSeriesDecompositionDetail::CComponents::initialized() const {
    return m_UsingTrendForPrediction && m_Trend.initialized()
               ? true
               : (m_Seasonal && m_Calendar
                      ? m_Seasonal->initialized() || m_Calendar->initialized()
                      : (m_Seasonal ? m_Seasonal->initialized()
                                    : (m_Calendar ? m_Calendar->initialized() : false)));
}

const CTrendComponent& CTimeSeriesDecompositionDetail::CComponents::trend() const {
    return m_Trend;
}

const TSeasonalComponentVec& CTimeSeriesDecompositionDetail::CComponents::seasonal() const {
    return m_Seasonal ? m_Seasonal->components() : NO_SEASONAL_COMPONENTS;
}

const maths_t::TCalendarComponentVec&
CTimeSeriesDecompositionDetail::CComponents::calendar() const {
    return m_Calendar ? m_Calendar->components() : NO_CALENDAR_COMPONENTS;
}

bool CTimeSeriesDecompositionDetail::CComponents::usingTrendForPrediction() const {
    return m_UsingTrendForPrediction;
}

void CTimeSeriesDecompositionDetail::CComponents::useTrendForPrediction() {
    m_UsingTrendForPrediction = true;
}

CPeriodicityHypothesisTestsConfig
CTimeSeriesDecompositionDetail::CComponents::periodicityTestConfig() const {
    CPeriodicityHypothesisTestsConfig result;
    for (const auto& component : this->seasonal()) {
        const CSeasonalTime& time{component.time()};
        result.hasDaily(result.hasDaily() || time.period() == DAY);
        result.hasWeekend(result.hasWeekend() || time.hasWeekend());
        result.hasWeekly(result.hasWeekly() || time.period() == WEEK);
        if (time.hasWeekend()) {
            result.startOfWeek(time.windowRepeatStart());
        }
    }
    return result;
}

double CTimeSeriesDecompositionDetail::CComponents::meanValue(core_t::TTime time) const {
    return this->initialized()
               ? ((m_UsingTrendForPrediction
                       ? CBasicStatistics::mean(m_Trend.value(time, 0.0))
                       : 0.0) +
                  meanOf(&CSeasonalComponent::meanValue, this->seasonal()))
               : 0.0;
}

double CTimeSeriesDecompositionDetail::CComponents::meanVariance() const {
    return this->initialized()
               ? ((m_UsingTrendForPrediction
                       ? CBasicStatistics::mean(this->trend().variance(0.0))
                       : 0.0) +
                  meanOf(&CSeasonalComponent::meanVariance, this->seasonal()))
               : 0.0;
}

double CTimeSeriesDecompositionDetail::CComponents::meanVarianceScale() const {
    return CBasicStatistics::mean(m_MeanVarianceScale);
}

uint64_t CTimeSeriesDecompositionDetail::CComponents::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_Machine);
    seed = CChecksum::calculate(seed, m_DecayRate);
    seed = CChecksum::calculate(seed, m_BucketLength);
    seed = CChecksum::calculate(seed, m_SeasonalComponentSize);
    seed = CChecksum::calculate(seed, m_CalendarComponentSize);
    seed = CChecksum::calculate(seed, m_Trend);
    seed = CChecksum::calculate(seed, m_Seasonal);
    seed = CChecksum::calculate(seed, m_Calendar);
    seed = CChecksum::calculate(seed, m_MeanVarianceScale);
    seed = CChecksum::calculate(seed, m_PredictionErrorWithoutTrend);
    seed = CChecksum::calculate(seed, m_PredictionErrorWithTrend);
    seed = CChecksum::calculate(seed, m_GainController);
    seed = CChecksum::calculate(seed, m_TestingForChange);
    return CChecksum::calculate(seed, m_UsingTrendForPrediction);
}

void CTimeSeriesDecompositionDetail::CComponents::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CComponents");
    core::CMemoryDebug::dynamicSize("m_Trend", m_Trend, mem);
    core::CMemoryDebug::dynamicSize("m_Seasonal", m_Seasonal, mem);
    core::CMemoryDebug::dynamicSize("m_Calendar", m_Calendar, mem);
}

std::size_t CTimeSeriesDecompositionDetail::CComponents::memoryUsage() const {
    return core::CMemory::dynamicSize(m_Trend) + core::CMemory::dynamicSize(m_Seasonal) +
           core::CMemory::dynamicSize(m_Calendar);
}

std::size_t CTimeSeriesDecompositionDetail::CComponents::size() const {
    return (m_Seasonal ? m_Seasonal->size() : 0) + (m_Calendar ? m_Calendar->size() : 0);
}

std::size_t CTimeSeriesDecompositionDetail::CComponents::maxSize() const {
    return MAXIMUM_COMPONENTS * m_SeasonalComponentSize;
}

bool CTimeSeriesDecompositionDetail::CComponents::addSeasonalComponents(
    const CPeriodicityHypothesisTestsResult& result,
    const CExpandingWindow& window,
    const TPredictor& predictor) {
    using TSeasonalTimePtr = std::unique_ptr<CSeasonalTime>;
    using TSeasonalTimePtrBoolPr = std::pair<TSeasonalTimePtr, bool>;
    using TSeasonalTimePtrBoolPrVec = std::vector<TSeasonalTimePtrBoolPr>;

    TSeasonalTimePtrBoolPrVec newComponents;
    const TSeasonalComponentVec& components{m_Seasonal->components()};

    // Find the components to add to the decomposition.
    for (const auto& candidate : result.components()) {
        TSeasonalTimePtr seasonalTime(candidate.seasonalTime());
        if (std::find_if(components.begin(), components.end(),
                         [&seasonalTime](const CSeasonalComponent& component) {
                             return component.time().excludes(*seasonalTime);
                         }) == components.end()) {
            LOG_DEBUG(<< "Detected '" << candidate.s_Description << "'");
            newComponents.emplace_back(std::move(seasonalTime), candidate.s_PiecewiseScaled);
        }
    }

    if (newComponents.size() > 0) {
        // Remove any components which have been superseded.
        for (const auto& component : newComponents) {
            m_Seasonal->removeExcludedComponents(*component.first);
        }

        std::sort(newComponents.begin(), newComponents.end(), maths::COrderings::SLess());

        core_t::TTime startTime{window.startTime()};
        core_t::TTime endTime{window.endTime()};
        core_t::TTime dt{window.bucketLength()};

        TFloatMeanAccumulatorVec values(window.valuesMinusPrediction(predictor));
        TDoubleVec shifts(values.size());
        std::transform(values.begin(), values.end(), shifts.begin(),
                       [](const TFloatMeanAccumulator& value) {
                           return CBasicStatistics::mean(value);
                       });
        result.removeTrend(values);
        TFloatMeanAccumulator level{std::accumulate(values.begin(), values.end(),
                                                    TFloatMeanAccumulator{})};
        for (std::size_t i = 0; i < shifts.size(); ++i) {
            shifts[i] = CBasicStatistics::mean(values[i]) -
                        CBasicStatistics::mean(level) - shifts[i];
        }

        for (auto& component : newComponents) {

            TSeasonalTimePtr seasonalTime;
            bool piecewiseConstantLinearScaling;
            std::tie(seasonalTime, piecewiseConstantLinearScaling) = std::move(component);
            core_t::TTime period{seasonalTime->period()};

            // Extract the values, minus the prediction of existing components,
            // ignoring any values which fall outside the window of the component
            // to add.
            values = window.valuesMinusPrediction(predictor);
            if (seasonalTime->windowed()) {
                core_t::TTime time{startTime + dt / 2};
                for (auto& value : values) {
                    if (seasonalTime->inWindow(time) == false) {
                        value = TFloatMeanAccumulator{};
                    }
                    time += dt;
                }
            }
            if (piecewiseConstantLinearScaling) {
                this->adjustValuesForPiecewiseConstantScaling(period / dt, values);
            }
            for (std::size_t i = 0; i < values.size(); ++i) {
                if (CBasicStatistics::count(values[i]) > 0.0) {
                    CBasicStatistics::moment<0>(values[i]) += shifts[i];
                }
            }

            // If we see multiple repeats of the component in the window we use
            // a periodic boundary condition, which ensures that the prediction
            // at the repeat is continuous.
            auto boundaryCondition = period > seasonalTime->windowLength()
                                         ? CSplineTypes::E_Natural
                                         : CSplineTypes::E_Periodic;

            // Re-weight outlier values with respect to the predictions of the
            // new seasonal component.
            double bucketLength{static_cast<double>(m_BucketLength)};
            CSeasonalComponent newComponent{*seasonalTime, m_SeasonalComponentSize,
                                            m_DecayRate, bucketLength, boundaryCondition};
            newComponent.initialize(startTime, endTime, values);
            newComponent.interpolate(CIntegerTools::floor(endTime, period));
            this->reweightOutliers(
                startTime, dt,
                [&newComponent](core_t::TTime time) {
                    return CBasicStatistics::mean(newComponent.value(time, 0.0));
                },
                values);

            // Add the new seasonal component.
            m_Seasonal->add(*seasonalTime, m_SeasonalComponentSize, m_DecayRate,
                            bucketLength, boundaryCondition, startTime, endTime, values);
        }

        m_Seasonal->refreshForNewComponents();

        TDoubleVec predictions;
        m_GainController.clear();
        for (core_t::TTime time = startTime; time < endTime; time += m_BucketLength) {
            predictions.clear();
            if (m_Seasonal) {
                m_Seasonal->appendPredictions(time, predictions);
            }
            if (m_Calendar) {
                m_Calendar->appendPredictions(time, predictions);
            }
            m_GainController.seed(predictions);
            m_GainController.age(ageFactor(m_DecayRate, m_BucketLength));
        }

        values = window.valuesMinusPrediction(predictor);
        result.removeDiscontinuities(values);
        CTrendComponent newTrend{m_Trend.defaultDecayRate()};
        this->fitTrend(startTime, dt, values, newTrend);
        this->reweightOutliers(startTime, dt,
                               [&newTrend](core_t::TTime time) {
                                   return CBasicStatistics::mean(newTrend.value(time, 0.0));
                               },
                               values);

        // TODO Resetting the trend is bad from a forecast perspective so we
        // should avoid it if the difference is small.
        newTrend = CTrendComponent{m_Trend.defaultDecayRate()};
        this->fitTrend(startTime, dt, values, newTrend);
        m_Trend.swap(newTrend);

        // Callback passing the residuals.
        core_t::TTime time{startTime + dt / 2};
        for (auto& value : values) {
            if (CBasicStatistics::count(value) > 0.0) {
                CBasicStatistics::moment<0>(value) -=
                    CBasicStatistics::mean(m_Trend.value(time, 0.0));
            }
            time += dt;
        }
        m_ComponentChangeCallback(std::move(values));
    }

    return newComponents.size() > 0;
}

bool CTimeSeriesDecompositionDetail::CComponents::addCalendarComponent(const CCalendarFeature& feature,
                                                                       core_t::TTime time) {
    double bucketLength{static_cast<double>(m_BucketLength)};
    m_Calendar->add(feature, m_CalendarComponentSize, m_DecayRate, bucketLength);
    LOG_DEBUG(<< "Detected feature '" << feature.print() << "' at " << time);
    return true;
}

void CTimeSeriesDecompositionDetail::CComponents::adjustValuesForPiecewiseConstantScaling(
    std::size_t period,
    TFloatMeanAccumulatorVec& values) const {

    // Periodicity testing detected piecewise constant linear scaling
    // of the underlying seasonal component. Here, we adjust all values
    // so the scaling is constant and equal to the average scaling for
    // the last populated period.

    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

    TDoubleVec trend;
    TDoubleVec scales;
    TSizeVec segmentation(
        CTimeSeriesSegmentation::piecewiseLinearScaledPeriodic(values, period));
    std::tie(trend, scales) = CTimeSeriesSegmentation::piecewiseLinearScaledPeriodic(
        values, period, segmentation);
    LOG_TRACE(<< "trend = " << core::CContainerPrinter::print(trend));
    LOG_TRACE(<< "scales = " << core::CContainerPrinter::print(scales));
    LOG_TRACE(<< "segmentation = " << core::CContainerPrinter::print(segmentation));
    values = CTimeSeriesSegmentation::removePiecewiseLinearScaledPeriodic(
        values, segmentation, trend, scales);
    TMeanAccumulator scale;
    for (std::size_t i = scales.size(), j = period + 1; i > 0 && j > 0; --i) {
        std::size_t a{segmentation[i - 1]};
        std::size_t b{segmentation[i]};
        for (std::size_t k = a; k < b && j > 0; --j, ++k) {
            if (CBasicStatistics::count(values[k]) > 0.0) {
                scale.add(scales[i - 1]);
            }
        }
    }
    LOG_TRACE(<< "scale = " << CBasicStatistics::mean(scale));
    for (std::size_t i = 0; i < values.size(); ++i) {
        CBasicStatistics::moment<0>(values[i]) += CBasicStatistics::mean(scale) *
                                                  trend[i % trend.size()];
    }
}

void CTimeSeriesDecompositionDetail::CComponents::reweightOutliers(
    core_t::TTime startTime,
    core_t::TTime dt,
    TPredictor predictor,
    TFloatMeanAccumulatorVec& values) const {
    using TMinAccumulator =
        CBasicStatistics::COrderStatisticsHeap<std::pair<double, std::size_t>>;
    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

    double numberValues{std::accumulate(
        values.begin(), values.end(), 0.0, [](double n, const TFloatMeanAccumulator& value) {
            return n + (CBasicStatistics::count(value) > 0.0 ? 1.0 : 0.0);
        })};
    double numberOutliers{SEASONAL_OUTLIER_FRACTION * numberValues};

    if (numberOutliers > 1.0) {

        TMinAccumulator outliers{static_cast<std::size_t>(2.0 * numberOutliers)};
        TMeanAccumulator meanDifference;
        core_t::TTime time{startTime + dt / 2};
        for (std::size_t i = 0; i < values.size(); ++i, time += dt) {
            if (CBasicStatistics::count(values[i]) > 0.0) {
                double difference{
                    std::fabs(CBasicStatistics::mean(values[i]) - predictor(time))};
                outliers.add({-difference, i});
                meanDifference.add(difference);
            }
        }
        outliers.sort();
        TMeanAccumulator meanDifferenceOfOutliers;
        for (std::size_t i = 0u; i < static_cast<std::size_t>(numberOutliers); ++i) {
            meanDifferenceOfOutliers.add(-outliers[i].first);
        }
        meanDifference -= meanDifferenceOfOutliers;
        for (std::size_t i = 0; i < outliers.count(); ++i) {
            if (-outliers[i].first > SEASONAL_OUTLIER_DIFFERENCE_THRESHOLD *
                                         CBasicStatistics::mean(meanDifference)) {
                double weight{SEASONAL_OUTLIER_WEIGHT +
                              (1.0 - SEASONAL_OUTLIER_WEIGHT) *
                                  CTools::logisticFunction(static_cast<double>(i) / numberOutliers,
                                                           0.1, 1.0)};
                CBasicStatistics::count(values[outliers[i].second]) *= weight;
            }
        }
    }
}

void CTimeSeriesDecompositionDetail::CComponents::fitTrend(core_t::TTime startTime,
                                                           core_t::TTime dt,
                                                           const TFloatMeanAccumulatorVec& values,
                                                           CTrendComponent& trend) const {
    core_t::TTime time{startTime + dt / 2};
    for (const auto& value : values) {
        // Because we test before the window is fully compressed we can
        // get a run of unset values at the end of the window, we should
        // just ignore these.
        if (CBasicStatistics::count(value) > 0.0) {
            trend.add(time, CBasicStatistics::mean(value), CBasicStatistics::count(value));
            trend.propagateForwardsByTime(dt);
        }
        time += dt;
    }
}

void CTimeSeriesDecompositionDetail::CComponents::clearComponentErrors() {
    if (m_Seasonal) {
        m_Seasonal->clearPredictionErrors();
    }
    if (m_Calendar) {
        m_Calendar->clearPredictionErrors();
    }
}

void CTimeSeriesDecompositionDetail::CComponents::apply(std::size_t symbol,
                                                        const SMessage& message) {
    if (symbol == SC_RESET) {
        m_Trend.clear();
        m_Seasonal.reset();
        m_Calendar.reset();
    }

    std::size_t old{m_Machine.state()};
    m_Machine.apply(symbol);
    std::size_t state{m_Machine.state()};

    if (state != old) {
        LOG_TRACE(<< SC_STATES[old] << "," << SC_ALPHABET[symbol] << " -> "
                  << SC_STATES[state]);

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
            LOG_ERROR(<< "Components in a bad state: " << m_Machine.state());
            this->apply(SC_RESET, message);
            break;
        }
    }
}

bool CTimeSeriesDecompositionDetail::CComponents::shouldUseTrendForPrediction() {
    double v0{CBasicStatistics::variance(m_PredictionErrorWithoutTrend)};
    double v1{CBasicStatistics::variance(m_PredictionErrorWithTrend)};
    double df0{CBasicStatistics::count(m_PredictionErrorWithoutTrend) - 1.0};
    double df1{CBasicStatistics::count(m_PredictionErrorWithTrend) - m_Trend.parameters()};
    if (df0 > 0.0 && df1 > 0.0 && v0 > 0.0) {
        double relativeLogSignificance{
            CTools::fastLog(CStatisticalTests::leftTailFTest(v1 / v0, df1, df0)) /
            LOG_COMPONENT_STATISTICALLY_SIGNIFICANCE};
        double vt{*std::max_element(std::begin(COMPONENT_SIGNIFICANT_VARIANCE_REDUCTION),
                                    std::end(COMPONENT_SIGNIFICANT_VARIANCE_REDUCTION)) *
                  v0};
        double p{CTools::logisticFunction(relativeLogSignificance, 0.1, 1.0) *
                 (vt > v1 ? CTools::logisticFunction(vt / v1, 1.0, 1.0, +1.0)
                          : CTools::logisticFunction(v1 / vt, 0.1, 1.0, -1.0))};
        m_UsingTrendForPrediction = (p >= 0.25);
    }
    return m_UsingTrendForPrediction;
}

bool CTimeSeriesDecompositionDetail::CComponents::shouldInterpolate(core_t::TTime time,
                                                                    core_t::TTime last) {
    return m_Machine.state() == SC_NEW_COMPONENTS ||
           (m_Seasonal && m_Seasonal->shouldInterpolate(time, last)) ||
           (m_Calendar && m_Calendar->shouldInterpolate(time, last));
}

void CTimeSeriesDecompositionDetail::CComponents::interpolate(const SMessage& message) {
    core_t::TTime time{message.s_Time};
    core_t::TTime lastTime{message.s_LastTime};

    std::size_t state{m_Machine.state()};

    switch (state) {
    case SC_NORMAL:
    case SC_NEW_COMPONENTS:
        this->canonicalize(time);
        if (this->shouldInterpolate(time, lastTime)) {
            LOG_TRACE(<< "Interpolating values at " << time);

            // Remove components that contain invalid (not finite) values,
            // along with the associated prediction errors and signal that the
            // set of components has been modified.
            if (m_Seasonal) {
                if (m_Seasonal->removeComponentsWithBadValues(time)) {
                    m_ComponentChangeCallback({});
                }
                m_Seasonal->interpolate(time, lastTime, true);
            }
            if (m_Calendar) {
                if (m_Calendar->removeComponentsWithBadValues(time)) {
                    m_ComponentChangeCallback({});
                }
                m_Calendar->interpolate(time, lastTime, true);
            }

            this->apply(SC_INTERPOLATED, message);
        }
        break;
    case SC_DISABLED:
        break;
    default:
        LOG_ERROR(<< "Components in a bad state: " << state);
        this->apply(SC_RESET, message);
        break;
    }
}

void CTimeSeriesDecompositionDetail::CComponents::shiftOrigin(core_t::TTime time) {
    time -= static_cast<core_t::TTime>(static_cast<double>(DAY) / m_DecayRate / 2.0);
    m_Trend.shiftOrigin(time);
    if (m_Seasonal) {
        m_Seasonal->shiftOrigin(time);
    }
    m_GainController.shiftOrigin(time);
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
        TSeasonalComponentVec& seasonal{m_Seasonal->components()};
        double slope{0.0};
        for (auto& component : seasonal) {
            if (component.slopeAccurate(time)) {
                double slope_{component.slope()};
                slope += slope_;
                component.shiftSlope(time, -slope_);
            }
        }
        if (slope != 0.0) {
            m_Trend.shiftSlope(time, slope);
        }
    }
}

bool CTimeSeriesDecompositionDetail::CComponents::CGainController::acceptRestoreTraverser(
    core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name{traverser.name()};
        RESTORE_BUILT_IN(REGRESSION_ORIGIN_6_4_TAG, m_RegressionOrigin)
        RESTORE(MEAN_SUM_AMPLITUDES_6_4_TAG,
                m_MeanSumAmplitudes.fromDelimited(traverser.value()))
        RESTORE(MEAN_SUM_AMPLITUDES_TREND_6_4_TAG,
                traverser.traverseSubLevel(
                    std::bind(&TRegression::acceptRestoreTraverser,
                              &m_MeanSumAmplitudesTrend, std::placeholders::_1)))
    } while (traverser.next());
    return true;
}

void CTimeSeriesDecompositionDetail::CComponents::CGainController::acceptPersistInserter(
    core::CStatePersistInserter& inserter) const {
    inserter.insertValue(REGRESSION_ORIGIN_6_4_TAG, m_RegressionOrigin);
    inserter.insertValue(MEAN_SUM_AMPLITUDES_6_4_TAG, m_MeanSumAmplitudes.toDelimited());
    inserter.insertLevel(MEAN_SUM_AMPLITUDES_TREND_6_4_TAG,
                         std::bind(&TRegression::acceptPersistInserter,
                                   &m_MeanSumAmplitudesTrend, std::placeholders::_1));
}

void CTimeSeriesDecompositionDetail::CComponents::CGainController::clear() {
    m_RegressionOrigin = 0;
    m_MeanSumAmplitudes = TFloatMeanAccumulator{};
    m_MeanSumAmplitudesTrend = TRegression{};
}

double CTimeSeriesDecompositionDetail::CComponents::CGainController::gain() const {
    if (m_MeanSumAmplitudesTrend.count() > 0.0) {
        TRegression::TArray params;
        m_MeanSumAmplitudesTrend.parameters(params);
        if (params[1] > 0.01 * CBasicStatistics::mean(m_MeanSumAmplitudes)) {
            return 1.0;
        }
    }
    return 3.0;
}

void CTimeSeriesDecompositionDetail::CComponents::CGainController::seed(const TDoubleVec& predictions) {
    m_MeanSumAmplitudes.add(std::accumulate(
        predictions.begin(), predictions.end(), 0.0,
        [](double sum, double prediction) { return sum + std::fabs(prediction); }));
}

void CTimeSeriesDecompositionDetail::CComponents::CGainController::add(core_t::TTime time,
                                                                       const TDoubleVec& predictions) {
    if (predictions.size() > 0) {
        m_MeanSumAmplitudes.add(std::accumulate(
            predictions.begin(), predictions.end(), 0.0, [](double sum, double prediction) {
                return sum + std::fabs(prediction);
            }));
        m_MeanSumAmplitudesTrend.add(scaleTime(time, m_RegressionOrigin),
                                     CBasicStatistics::mean(m_MeanSumAmplitudes),
                                     CBasicStatistics::count(m_MeanSumAmplitudes));
    }
}

void CTimeSeriesDecompositionDetail::CComponents::CGainController::age(double factor) {
    m_MeanSumAmplitudes.age(factor);
    m_MeanSumAmplitudesTrend.age(factor);
}

void CTimeSeriesDecompositionDetail::CComponents::CGainController::shiftOrigin(core_t::TTime time) {
    time = CIntegerTools::floor(time, WEEK);
    if (time > m_RegressionOrigin) {
        m_MeanSumAmplitudesTrend.shiftAbscissa(-scaleTime(time, m_RegressionOrigin));
        m_RegressionOrigin = time;
    }
}

uint64_t CTimeSeriesDecompositionDetail::CComponents::CGainController::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_RegressionOrigin);
    seed = CChecksum::calculate(seed, m_MeanSumAmplitudes);
    return CChecksum::calculate(seed, m_MeanSumAmplitudesTrend);
}

bool CTimeSeriesDecompositionDetail::CComponents::CComponentErrors::fromDelimited(const std::string& str_) {
    std::string str{str_};
    std::size_t n{str.find(CBasicStatistics::EXTERNAL_DELIMITER)};
    if (m_MeanErrors.fromDelimited(str.substr(0, n)) == false) {
        LOG_ERROR(<< "Failed to parse '" << str << "'");
        return false;
    }
    str = str.substr(n + 1);
    if (m_MaxVarianceIncrease.fromDelimited(str) == false) {
        LOG_ERROR(<< "Failed to parse '" << str << "'");
        return false;
    }
    return true;
}

std::string CTimeSeriesDecompositionDetail::CComponents::CComponentErrors::toDelimited() const {
    return m_MeanErrors.toDelimited() + CBasicStatistics::EXTERNAL_DELIMITER +
           m_MaxVarianceIncrease.toDelimited();
}

void CTimeSeriesDecompositionDetail::CComponents::CComponentErrors::add(double referenceError,
                                                                        double error,
                                                                        double prediction,
                                                                        double varianceIncrease,
                                                                        double weight) {
    TVector errors;
    errors(0) = CTools::pow2(referenceError);
    errors(1) = CTools::pow2(error);
    errors(2) = CTools::pow2(error + prediction);
    m_MeanErrors.add(this->winsorise(errors), weight);
    m_MaxVarianceIncrease.add(varianceIncrease);
}

void CTimeSeriesDecompositionDetail::CComponents::CComponentErrors::clear() {
    m_MeanErrors = TVectorMeanAccumulator{};
    m_MaxVarianceIncrease = TMaxAccumulator{};
}

bool CTimeSeriesDecompositionDetail::CComponents::CComponentErrors::remove(core_t::TTime bucketLength,
                                                                           core_t::TTime period) const {
    double history{CBasicStatistics::count(m_MeanErrors) * static_cast<double>(bucketLength)};
    double referenceError{CBasicStatistics::mean(m_MeanErrors)(0)};
    double errorWithComponent{CBasicStatistics::mean(m_MeanErrors)(1)};
    double errorWithoutComponent{CBasicStatistics::mean(m_MeanErrors)(2)};
    return (history > static_cast<double>(WEEK) && errorWithComponent > referenceError) ||
           (history > 5.0 * static_cast<double>(period) && m_MaxVarianceIncrease[0] < 1.2 &&
            errorWithoutComponent < 1.2 * errorWithComponent);
}

void CTimeSeriesDecompositionDetail::CComponents::CComponentErrors::age(double factor) {
    m_MeanErrors.age(factor);
    m_MaxVarianceIncrease.age(factor);
}

uint64_t CTimeSeriesDecompositionDetail::CComponents::CComponentErrors::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_MeanErrors);
    return CChecksum::calculate(seed, m_MaxVarianceIncrease);
}

CTimeSeriesDecompositionDetail::CComponents::CComponentErrors::TVector
CTimeSeriesDecompositionDetail::CComponents::CComponentErrors::winsorise(const TVector& squareError) const {
    return CBasicStatistics::count(m_MeanErrors) > 10.0
               ? min(squareError, CFloatStorage{36} * CBasicStatistics::mean(m_MeanErrors))
               : squareError;
}

bool CTimeSeriesDecompositionDetail::CComponents::CSeasonal::acceptRestoreTraverser(
    double decayRate,
    core_t::TTime bucketLength_,
    core::CStateRestoreTraverser& traverser) {
    double bucketLength{static_cast<double>(bucketLength_)};
    if (traverser.name() == VERSION_6_4_TAG) {
        while (traverser.next()) {
            const std::string& name{traverser.name()};
            RESTORE_NO_ERROR(COMPONENT_6_4_TAG,
                             m_Components.emplace_back(decayRate, bucketLength, traverser))
            RESTORE(ERRORS_6_4_TAG,
                    core::CPersistUtils::restore(ERRORS_6_4_TAG, m_PredictionErrors, traverser))
        }
    } else if (traverser.name() == VERSION_6_3_TAG) {
        while (traverser.next()) {
            const std::string& name{traverser.name()};
            RESTORE_NO_ERROR(COMPONENT_6_3_TAG,
                             m_Components.emplace_back(decayRate, bucketLength, traverser))
        }
        m_PredictionErrors.resize(m_Components.size());
    } else {
        // There is no version string this is historic state.
        do {
            const std::string& name{traverser.name()};
            RESTORE_NO_ERROR(COMPONENT_OLD_TAG,
                             m_Components.emplace_back(decayRate, bucketLength, traverser))
        } while (traverser.next());
        m_PredictionErrors.resize(m_Components.size());
    }
    return true;
}

void CTimeSeriesDecompositionDetail::CComponents::CSeasonal::acceptPersistInserter(
    core::CStatePersistInserter& inserter) const {
    inserter.insertValue(VERSION_6_4_TAG, "");
    for (const auto& component : m_Components) {
        inserter.insertLevel(COMPONENT_6_4_TAG,
                             std::bind(&CSeasonalComponent::acceptPersistInserter,
                                       &component, std::placeholders::_1));
    }
    core::CPersistUtils::persist(ERRORS_6_4_TAG, m_PredictionErrors, inserter);
}

void CTimeSeriesDecompositionDetail::CComponents::CSeasonal::decayRate(double decayRate) {
    for (auto& component : m_Components) {
        component.decayRate(decayRate);
    }
}

bool CTimeSeriesDecompositionDetail::CComponents::CSeasonal::removeComponentsWithBadValues(core_t::TTime time) {

    TBoolVec remove(m_Components.size(), false);
    bool anyBadComponentsFound{false};
    for (std::size_t i = 0u; i < m_Components.size(); ++i) {
        const CSeasonalTime& time_{m_Components[i].time()};
        if (m_Components[i].isBad()) {
            LOG_DEBUG(<< "Removing seasonal component"
                      << " with period '" << time_.period() << "' at " << time
                      << ". Invalid values detected.");
            remove[i] = true;
            anyBadComponentsFound |= true;
        }
    }

    if (anyBadComponentsFound) {
        CSetTools::simultaneousRemoveIf(remove, m_Components, m_PredictionErrors,
                                        [](bool remove_) { return remove_; });

        return true;
    }
    return false;
}

void CTimeSeriesDecompositionDetail::CComponents::CSeasonal::propagateForwards(core_t::TTime start,
                                                                               core_t::TTime end) {
    for (std::size_t i = 0u; i < m_Components.size(); ++i) {
        core_t::TTime period{m_Components[i].time().period()};
        double time{stepwisePropagateForwards(
            &m_Components[i], start, end, CTools::truncate(period, DAY, WEEK), 1.0 / 3.0)};
        if (time > 0.0) {
            m_PredictionErrors[i].age(std::exp(-m_Components[i].decayRate() * time));
        }
    }
}

void CTimeSeriesDecompositionDetail::CComponents::CSeasonal::clearPredictionErrors() {
    for (auto& errors : m_PredictionErrors) {
        errors.clear();
    }
}

std::size_t CTimeSeriesDecompositionDetail::CComponents::CSeasonal::size() const {
    std::size_t result{0};
    for (const auto& component : m_Components) {
        result += component.size();
    }
    return result;
}

const maths_t::TSeasonalComponentVec&
CTimeSeriesDecompositionDetail::CComponents::CSeasonal::components() const {
    return m_Components;
}

maths_t::TSeasonalComponentVec&
CTimeSeriesDecompositionDetail::CComponents::CSeasonal::components() {
    return m_Components;
}

void CTimeSeriesDecompositionDetail::CComponents::CSeasonal::componentsErrorsAndDeltas(
    core_t::TTime time,
    TSeasonalComponentPtrVec& components,
    TComponentErrorsPtrVec& errors,
    TDoubleVec& deltas) {
    std::size_t n{m_Components.size()};

    components.reserve(n);
    errors.reserve(n);

    for (std::size_t i = 0u; i < n; ++i) {
        if (m_Components[i].time().inWindow(time)) {
            components.push_back(&m_Components[i]);
            errors.push_back(&m_PredictionErrors[i]);
        }
    }

    deltas.resize(components.size(), 0.0);
    for (std::size_t i = 1u; i < components.size(); ++i) {
        core_t::TTime period{components[i]->time().period()};
        for (int j{static_cast<int>(i - 1)}; j > -1; --j) {
            core_t::TTime period_{components[j]->time().period()};
            if (period % period_ == 0) {
                double value{CBasicStatistics::mean(components[j]->value(time, 0.0)) -
                             components[j]->meanValue()};
                double delta{0.1 * components[i]->delta(time, period_, value)};
                deltas[j] += delta;
                deltas[i] -= delta;
                break;
            }
        }
    }
}

void CTimeSeriesDecompositionDetail::CComponents::CSeasonal::appendPredictions(
    core_t::TTime time,
    TDoubleVec& predictions) const {
    predictions.reserve(predictions.size() + m_Components.size());
    for (const auto& component : m_Components) {
        if (component.time().inWindow(time)) {
            predictions.push_back(CBasicStatistics::mean(component.value(time, 0.0)) -
                                  component.meanValue());
        }
    }
}

bool CTimeSeriesDecompositionDetail::CComponents::CSeasonal::shouldInterpolate(
    core_t::TTime time,
    core_t::TTime last) const {
    for (const auto& component : m_Components) {
        core_t::TTime period{component.time().period()};
        core_t::TTime a{CIntegerTools::floor(last, period)};
        core_t::TTime b{CIntegerTools::floor(time, period)};
        if (b > a) {
            return true;
        }
    }
    return false;
}

void CTimeSeriesDecompositionDetail::CComponents::CSeasonal::interpolate(core_t::TTime time,
                                                                         core_t::TTime lastTime,
                                                                         bool refine) {
    for (auto& component : m_Components) {
        core_t::TTime period{component.time().period()};
        core_t::TTime a{CIntegerTools::floor(lastTime, period)};
        core_t::TTime b{CIntegerTools::floor(time, period)};
        if (b > a || !component.initialized()) {
            component.interpolate(b, refine);
        }
    }
}

bool CTimeSeriesDecompositionDetail::CComponents::CSeasonal::initialized() const {
    for (const auto& component : m_Components) {
        if (component.initialized()) {
            return true;
        }
    }
    return false;
}

void CTimeSeriesDecompositionDetail::CComponents::CSeasonal::add(
    const CSeasonalTime& seasonalTime,
    std::size_t size,
    double decayRate,
    double bucketLength,
    CSplineTypes::EBoundaryCondition boundaryCondition,
    core_t::TTime startTime,
    core_t::TTime endTime,
    const TFloatMeanAccumulatorVec& values) {
    m_Components.emplace_back(seasonalTime, size, decayRate, bucketLength, boundaryCondition);
    m_Components.back().initialize(startTime, endTime, values);
    m_Components.back().interpolate(CIntegerTools::floor(endTime, seasonalTime.period()));
}

void CTimeSeriesDecompositionDetail::CComponents::CSeasonal::refreshForNewComponents() {
    m_PredictionErrors.resize(m_Components.size());
    COrderings::simultaneousSort(
        m_Components, m_PredictionErrors,
        [](const CSeasonalComponent& lhs, const CSeasonalComponent& rhs) {
            return lhs.time() < rhs.time();
        });
}

void CTimeSeriesDecompositionDetail::CComponents::CSeasonal::removeExcludedComponents(
    const CSeasonalTime& time) {
    m_Components.erase(std::remove_if(m_Components.begin(), m_Components.end(),
                                      [&time](const CSeasonalComponent& component) {
                                          return time.excludes(component.time());
                                      }),
                       m_Components.end());
}

bool CTimeSeriesDecompositionDetail::CComponents::CSeasonal::prune(core_t::TTime time,
                                                                   core_t::TTime bucketLength) {
    std::size_t n{m_Components.size()};

    if (n > 1) {
        TTimeTimePrSizeFMap windowed;
        windowed.reserve(n);
        for (const auto& component : m_Components) {
            const CSeasonalTime& time_{component.time()};
            if (time_.windowed()) {
                ++windowed[time_.window()];
            }
        }

        TBoolVec remove(n, false);
        TTimeTimePrDoubleFMap shifts;
        shifts.reserve(n);
        for (std::size_t i = 0u; i < n; ++i) {
            const CSeasonalTime& time_{m_Components[i].time()};
            auto j = windowed.find(time_.window());
            if (j == windowed.end() || j->second > 1) {
                if (m_PredictionErrors[i].remove(bucketLength, time_.period())) {
                    LOG_DEBUG(<< "Removing seasonal component"
                              << " with period '" << time_.period() << "' at " << time);
                    remove[i] = true;
                    shifts[time_.window()] += m_Components[i].meanValue();
                    --j->second;
                }
            }
        }

        CSetTools::simultaneousRemoveIf(remove, m_Components, m_PredictionErrors,
                                        [](bool remove_) { return remove_; });

        for (auto& shift : shifts) {
            if (windowed.count(shift.first) > 0) {
                for (auto& component : m_Components) {
                    if (shift.first == component.time().window()) {
                        component.shiftLevel(shift.second);
                        break;
                    }
                }
            } else {
                bool fallback = true;
                for (auto& component : m_Components) {
                    if (!component.time().windowed()) {
                        component.shiftLevel(shift.second);
                        fallback = false;
                        break;
                    }
                }
                if (fallback) {
                    TTimeTimePrVec shifted;
                    shifted.reserve(m_Components.size());
                    for (auto& component : m_Components) {
                        const CSeasonalTime& time_ = component.time();
                        auto containsWindow = [&time_](const TTimeTimePr& window) {
                            return !(time_.windowEnd() <= window.first ||
                                     time_.windowStart() >= window.second);
                        };
                        if (std::find_if(shifted.begin(), shifted.end(),
                                         containsWindow) == shifted.end()) {
                            component.shiftLevel(shift.second);
                        }
                    }
                }
            }
        }
    }

    return m_Components.empty();
}

void CTimeSeriesDecompositionDetail::CComponents::CSeasonal::shiftOrigin(core_t::TTime time) {
    for (auto& component : m_Components) {
        component.shiftOrigin(time);
    }
}

void CTimeSeriesDecompositionDetail::CComponents::CSeasonal::linearScale(core_t::TTime time,
                                                                         double scale) {
    for (std::size_t i = 0; i < m_Components.size(); ++i) {
        m_Components[i].linearScale(time, scale);
        m_PredictionErrors[i].clear();
    }
}

uint64_t CTimeSeriesDecompositionDetail::CComponents::CSeasonal::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_Components);
    return CChecksum::calculate(seed, m_PredictionErrors);
}

void CTimeSeriesDecompositionDetail::CComponents::CSeasonal::debugMemoryUsage(
    core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CSeasonal");
    core::CMemoryDebug::dynamicSize("m_Components", m_Components, mem);
    core::CMemoryDebug::dynamicSize("m_PredictionErrors", m_PredictionErrors, mem);
}

std::size_t CTimeSeriesDecompositionDetail::CComponents::CSeasonal::memoryUsage() const {
    return core::CMemory::dynamicSize(m_Components) +
           core::CMemory::dynamicSize(m_PredictionErrors);
}

bool CTimeSeriesDecompositionDetail::CComponents::CCalendar::acceptRestoreTraverser(
    double decayRate,
    core_t::TTime bucketLength_,
    core::CStateRestoreTraverser& traverser) {
    double bucketLength{static_cast<double>(bucketLength_)};
    if (traverser.name() == VERSION_6_4_TAG) {
        while (traverser.next()) {
            const std::string& name{traverser.name()};
            RESTORE_NO_ERROR(COMPONENT_6_4_TAG,
                             m_Components.emplace_back(decayRate, bucketLength, traverser))
            RESTORE(ERRORS_6_4_TAG,
                    core::CPersistUtils::restore(ERRORS_6_4_TAG, m_PredictionErrors, traverser))
        }
    } else if (traverser.name() == VERSION_6_3_TAG) {
        while (traverser.next()) {
            const std::string& name{traverser.name()};
            RESTORE_NO_ERROR(COMPONENT_6_3_TAG,
                             m_Components.emplace_back(decayRate, bucketLength, traverser))
        }
        m_PredictionErrors.resize(m_Components.size());
    } else {
        // There is no version string this is historic state.
        do {
            const std::string& name{traverser.name()};
            RESTORE_NO_ERROR(COMPONENT_OLD_TAG,
                             m_Components.emplace_back(decayRate, bucketLength, traverser))
        } while (traverser.next());
        m_PredictionErrors.resize(m_Components.size());
    }
    return true;
}

void CTimeSeriesDecompositionDetail::CComponents::CCalendar::acceptPersistInserter(
    core::CStatePersistInserter& inserter) const {
    inserter.insertValue(VERSION_6_4_TAG, "");
    for (const auto& component : m_Components) {
        inserter.insertLevel(COMPONENT_6_4_TAG,
                             std::bind(&CCalendarComponent::acceptPersistInserter,
                                       &component, std::placeholders::_1));
    }
    core::CPersistUtils::persist(ERRORS_6_4_TAG, m_PredictionErrors, inserter);
}

void CTimeSeriesDecompositionDetail::CComponents::CCalendar::decayRate(double decayRate) {
    for (auto& component : m_Components) {
        component.decayRate(decayRate);
    }
}

void CTimeSeriesDecompositionDetail::CComponents::CCalendar::propagateForwards(core_t::TTime start,
                                                                               core_t::TTime end) {
    for (std::size_t i = 0u; i < m_Components.size(); ++i) {
        double time{stepwisePropagateForwards(&m_Components[i], start, end, MONTH, 1.0 / 3.0)};
        if (time > 0.0) {
            m_PredictionErrors[i].age(std::exp(-m_Components[i].decayRate() * time));
        }
    }
}

void CTimeSeriesDecompositionDetail::CComponents::CCalendar::clearPredictionErrors() {
    for (auto& errors : m_PredictionErrors) {
        errors.clear();
    }
}

std::size_t CTimeSeriesDecompositionDetail::CComponents::CCalendar::size() const {
    std::size_t result{0};
    for (const auto& component : m_Components) {
        result += component.size();
    }
    return result;
}

const maths_t::TCalendarComponentVec&
CTimeSeriesDecompositionDetail::CComponents::CCalendar::components() const {
    return m_Components;
}

bool CTimeSeriesDecompositionDetail::CComponents::CCalendar::haveComponent(CCalendarFeature feature) const {
    for (const auto& component : m_Components) {
        if (component.feature() == feature) {
            return true;
        }
    }
    return false;
}

void CTimeSeriesDecompositionDetail::CComponents::CCalendar::componentsAndErrors(
    core_t::TTime time,
    TCalendarComponentPtrVec& components,
    TComponentErrorsPtrVec& errors) {
    std::size_t n = m_Components.size();
    components.reserve(n);
    errors.reserve(n);
    for (std::size_t i = 0u; i < n; ++i) {
        if (m_Components[i].feature().inWindow(time)) {
            components.push_back(&m_Components[i]);
            errors.push_back(&m_PredictionErrors[i]);
        }
    }
}

void CTimeSeriesDecompositionDetail::CComponents::CCalendar::appendPredictions(
    core_t::TTime time,
    TDoubleVec& predictions) const {
    predictions.reserve(predictions.size() + m_Components.size());
    for (const auto& component : m_Components) {
        if (component.feature().inWindow(time)) {
            predictions.push_back(CBasicStatistics::mean(component.value(time, 0.0)) -
                                  component.meanValue());
        }
    }
}

bool CTimeSeriesDecompositionDetail::CComponents::CCalendar::shouldInterpolate(
    core_t::TTime time,
    core_t::TTime last) const {
    for (const auto& component : m_Components) {
        CCalendarFeature feature = component.feature();
        if (!feature.inWindow(time) && feature.inWindow(last)) {
            return true;
        }
    }
    return false;
}

void CTimeSeriesDecompositionDetail::CComponents::CCalendar::interpolate(core_t::TTime time,
                                                                         core_t::TTime lastTime,
                                                                         bool refine) {
    for (auto& component : m_Components) {
        CCalendarFeature feature = component.feature();
        if (!feature.inWindow(time) && feature.inWindow(lastTime)) {
            component.interpolate(time - feature.offset(time), refine);
        }
    }
}

bool CTimeSeriesDecompositionDetail::CComponents::CCalendar::initialized() const {
    for (const auto& component : m_Components) {
        if (component.initialized()) {
            return true;
        }
    }
    return false;
}

bool CTimeSeriesDecompositionDetail::CComponents::CCalendar::prune(core_t::TTime time,
                                                                   core_t::TTime bucketLength) {
    TBoolVec remove(m_Components.size(), false);
    for (std::size_t i = 0u; i < m_Components.size(); ++i) {
        if (m_PredictionErrors[i].remove(bucketLength, m_Components[i].feature().window())) {
            LOG_DEBUG(<< "Removing calendar component"
                      << " '" << m_Components[i].feature().print() << "' at " << time);
            remove[i] = true;
        }
    }

    CSetTools::simultaneousRemoveIf(remove, m_Components, m_PredictionErrors,
                                    [](bool remove_) { return remove_; });

    return m_Components.empty();
}

bool CTimeSeriesDecompositionDetail::CComponents::CCalendar::removeComponentsWithBadValues(core_t::TTime time) {

    TBoolVec remove(m_Components.size(), false);
    bool anyBadComponentsFound{false};
    for (std::size_t i = 0u; i < m_Components.size(); ++i) {
        if (m_Components[i].isBad()) {
            LOG_DEBUG(<< "Removing calendar component"
                      << " '" << m_Components[i].feature().print() << "' at "
                      << time << ". Invalid value detected.");
            remove[i] = true;
            anyBadComponentsFound |= true;
        }
    }

    if (anyBadComponentsFound) {
        CSetTools::simultaneousRemoveIf(remove, m_Components, m_PredictionErrors,
                                        [](bool remove_) { return remove_; });
        return true;
    }

    return false;
}

void CTimeSeriesDecompositionDetail::CComponents::CCalendar::add(const CCalendarFeature& feature,
                                                                 std::size_t size,
                                                                 double decayRate,
                                                                 double bucketLength) {
    m_Components.emplace_back(feature, size, decayRate, bucketLength, CSplineTypes::E_Natural);
    m_Components.back().initialize();
    m_PredictionErrors.resize(m_Components.size());
}

void CTimeSeriesDecompositionDetail::CComponents::CCalendar::linearScale(core_t::TTime time,
                                                                         double scale) {
    for (auto& component : m_Components) {
        component.linearScale(time, scale);
    }
}

uint64_t CTimeSeriesDecompositionDetail::CComponents::CCalendar::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_Components);
    return CChecksum::calculate(seed, m_PredictionErrors);
}

void CTimeSeriesDecompositionDetail::CComponents::CCalendar::debugMemoryUsage(
    core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CCalendar");
    core::CMemoryDebug::dynamicSize("m_Components", m_Components, mem);
    core::CMemoryDebug::dynamicSize("m_PredictionErrors", m_PredictionErrors, mem);
}

std::size_t CTimeSeriesDecompositionDetail::CComponents::CCalendar::memoryUsage() const {
    return core::CMemory::dynamicSize(m_Components) +
           core::CMemory::dynamicSize(m_PredictionErrors);
}
}
}
