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

#include <maths/time_series/CTimeSeriesDecomposition.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/Constants.h>
#include <core/RestoreMacros.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CBasicStatisticsPersist.h>
#include <maths/common/CChecksum.h>
#include <maths/common/CIntegerTools.h>
#include <maths/common/CMathsFuncs.h>
#include <maths/common/CMathsFuncsForMatrixAndVectorTypes.h>
#include <maths/common/CPrior.h>
#include <maths/common/CRestoreParams.h>

#include <maths/time_series/CSeasonalTime.h>

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
const core::TPersistenceTag LAST_VALUE_TIME_7_11_TAG{"a", "last_value_time"};
const core::TPersistenceTag LAST_PROPAGATION_TIME_7_11_TAG{"b", "last_propagation_time"};
const core::TPersistenceTag CHANGE_POINT_TEST_7_11_TAG{"c", "change_point_test"};
const core::TPersistenceTag SEASONALITY_TEST_7_11_TAG{"d", "seasonality_test"};
const core::TPersistenceTag CALENDAR_CYCLIC_TEST_7_11_TAG{"e", "calendar_cyclic_test"};
const core::TPersistenceTag COMPONENTS_7_11_TAG{"f", "components"};
const core::TPersistenceTag TIME_SHIFT_7_11_TAG{"g", "time_shift"};
// Version 6.3
const std::string VERSION_6_3_TAG("6.3");
const core::TPersistenceTag LAST_VALUE_TIME_6_3_TAG{"a", "last_value_time"};
const core::TPersistenceTag LAST_PROPAGATION_TIME_6_3_TAG{"b", "last_propagation_time"};
const core::TPersistenceTag SEASONALITY_TEST_6_3_TAG{"c", "periodicity_test"};
const core::TPersistenceTag CALENDAR_CYCLIC_TEST_6_3_TAG{"d", "calendar_cyclic_test"};
const core::TPersistenceTag COMPONENTS_6_3_TAG{"e", "components"};
const core::TPersistenceTag TIME_SHIFT_6_3_TAG{"f", "time_shift"};

const std::string EMPTY_STRING;
}

CTimeSeriesDecomposition::CTimeSeriesDecomposition(double decayRate,
                                                   core_t::TTime bucketLength,
                                                   std::size_t seasonalComponentSize)
    : m_TimeShift{0}, m_LastValueTime{0}, m_LastPropagationTime{0},
      m_ChangePointTest{decayRate, bucketLength}, m_SeasonalityTest{decayRate, bucketLength},
      m_CalendarCyclicTest{decayRate, bucketLength}, m_Components{decayRate, bucketLength,
                                                                  seasonalComponentSize} {
    this->initializeMediator();
}

CTimeSeriesDecomposition::CTimeSeriesDecomposition(const common::STimeSeriesDecompositionRestoreParams& params,
                                                   core::CStateRestoreTraverser& traverser)
    : m_TimeShift{0}, m_LastValueTime{0}, m_LastPropagationTime{0},
      m_ChangePointTest{params.s_DecayRate, params.s_MinimumBucketLength},
      m_SeasonalityTest{params.s_DecayRate, params.s_MinimumBucketLength},
      m_CalendarCyclicTest{params.s_DecayRate, params.s_MinimumBucketLength},
      m_Components{params.s_DecayRate, params.s_MinimumBucketLength, params.s_ComponentSize} {
    if (traverser.traverseSubLevel([&](auto& traverser_) {
            return this->acceptRestoreTraverser(params.s_ChangeModelParams, traverser_);
        }) == false) {
        traverser.setBadState();
        return;
    }
    this->initializeMediator();
}

CTimeSeriesDecomposition::CTimeSeriesDecomposition(const CTimeSeriesDecomposition& other,
                                                   bool isForForecast)
    : m_TimeShift{other.m_TimeShift}, m_LastValueTime{other.m_LastValueTime},
      m_LastPropagationTime{other.m_LastPropagationTime},
      m_ChangePointTest{other.m_ChangePointTest, isForForecast},
      m_SeasonalityTest{other.m_SeasonalityTest, isForForecast},
      m_CalendarCyclicTest{other.m_CalendarCyclicTest, isForForecast}, m_Components{
                                                                           other.m_Components} {
    this->initializeMediator();
}

bool CTimeSeriesDecomposition::acceptRestoreTraverser(const common::SDistributionRestoreParams& params,
                                                      core::CStateRestoreTraverser& traverser) {
    if (traverser.name() == VERSION_7_11_TAG) {
        while (traverser.next()) {
            const std::string& name{traverser.name()};
            RESTORE_BUILT_IN(TIME_SHIFT_7_11_TAG, m_TimeShift)
            RESTORE_BUILT_IN(LAST_VALUE_TIME_7_11_TAG, m_LastValueTime)
            RESTORE_BUILT_IN(LAST_PROPAGATION_TIME_7_11_TAG, m_LastPropagationTime)
            RESTORE(CHANGE_POINT_TEST_7_11_TAG,
                    traverser.traverseSubLevel([this](auto& traverser_) {
                        return m_ChangePointTest.acceptRestoreTraverser(traverser_);
                    }))
            RESTORE(SEASONALITY_TEST_7_11_TAG,
                    traverser.traverseSubLevel([this](auto& traverser_) {
                        return m_SeasonalityTest.acceptRestoreTraverser(traverser_);
                    }))
            RESTORE(CALENDAR_CYCLIC_TEST_7_11_TAG,
                    traverser.traverseSubLevel([this](auto& traverser_) {
                        return m_CalendarCyclicTest.acceptRestoreTraverser(traverser_);
                    }))
            RESTORE(COMPONENTS_7_11_TAG, traverser.traverseSubLevel([&](auto& traverser_) {
                return m_Components.acceptRestoreTraverser(params, traverser_);
            }))
        }
    } else if (traverser.name() == VERSION_6_3_TAG) {
        while (traverser.next()) {
            const std::string& name{traverser.name()};
            RESTORE_BUILT_IN(TIME_SHIFT_6_3_TAG, m_TimeShift)
            RESTORE_BUILT_IN(LAST_VALUE_TIME_6_3_TAG, m_LastValueTime)
            RESTORE_BUILT_IN(LAST_PROPAGATION_TIME_6_3_TAG, m_LastPropagationTime)
            RESTORE(SEASONALITY_TEST_6_3_TAG, traverser.traverseSubLevel([this](auto& traverser_) {
                return m_SeasonalityTest.acceptRestoreTraverser(traverser_);
            }))
            RESTORE(CALENDAR_CYCLIC_TEST_6_3_TAG,
                    traverser.traverseSubLevel([this](auto& traverser_) {
                        return m_CalendarCyclicTest.acceptRestoreTraverser(traverser_);
                    }))
            RESTORE(COMPONENTS_6_3_TAG, traverser.traverseSubLevel([&](auto& traverser_) {
                return m_Components.acceptRestoreTraverser(params, traverser_);
            }))
        }
    } else {
        LOG_ERROR(<< "Unsupported version '" << traverser.name() << "'");
        return false;
    }
    return true;
}

void CTimeSeriesDecomposition::swap(CTimeSeriesDecomposition& other) {
    std::swap(m_TimeShift, other.m_TimeShift);
    std::swap(m_LastValueTime, other.m_LastValueTime);
    std::swap(m_LastPropagationTime, other.m_LastPropagationTime);
    m_ChangePointTest.swap(other.m_ChangePointTest);
    m_SeasonalityTest.swap(other.m_SeasonalityTest);
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
    inserter.insertValue(VERSION_7_11_TAG, "");
    inserter.insertValue(TIME_SHIFT_7_11_TAG, m_TimeShift);
    inserter.insertValue(LAST_VALUE_TIME_7_11_TAG, m_LastValueTime);
    inserter.insertValue(LAST_PROPAGATION_TIME_7_11_TAG, m_LastPropagationTime);
    inserter.insertLevel(CHANGE_POINT_TEST_7_11_TAG, [this](auto& inserter_) {
        m_ChangePointTest.acceptPersistInserter(inserter_);
    });
    inserter.insertLevel(SEASONALITY_TEST_7_11_TAG, [this](auto& inserter_) {
        m_SeasonalityTest.acceptPersistInserter(inserter_);
    });
    inserter.insertLevel(CALENDAR_CYCLIC_TEST_7_11_TAG, [this](auto& inserter_) {
        m_CalendarCyclicTest.acceptPersistInserter(inserter_);
    });
    inserter.insertLevel(COMPONENTS_7_11_TAG, [this](auto& inserter_) {
        m_Components.acceptPersistInserter(inserter_);
    });
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

void CTimeSeriesDecomposition::addPoint(core_t::TTime time,
                                        double value,
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
    CComponents::CScopeAttachComponentChangeCallback attach{
        m_Components, componentChangeCallback, modelAnnotationCallback};

    time += m_TimeShift;

    core_t::TTime lastTime{std::max(m_LastValueTime, m_LastPropagationTime)};

    m_LastValueTime = std::max(m_LastValueTime, time);
    this->propagateForwardsTo(time);

    auto testForSeasonality = m_Components.makeTestForSeasonality([this] {
        auto predictor_ = this->predictor(E_Seasonal);
        return [ predictor = std::move(predictor_),
                 this ](core_t::TTime time_, const TBoolVec& removedSeasonalMask) {
            return predictor(time_, removedSeasonalMask) +
                   this->smooth(
                       [&](core_t::TTime shiftedTime) {
                           return predictor(shiftedTime - m_TimeShift, removedSeasonalMask);
                       },
                       time_ + m_TimeShift, E_Seasonal);
        };
    });

    SAddValue message{time,
                      lastTime,
                      m_TimeShift,
                      value,
                      weights,
                      occupancy,
                      firstValueTime,
                      this->value(time, 0.0, E_TrendForced, true).mean(),
                      this->value(time, 0.0, E_Seasonal, true).mean(),
                      this->value(time, 0.0, E_Calendar, true).mean(),
                      *this,
                      [this] {
                          auto predictor_ = this->predictor(E_All | E_TrendForced);
                          return [predictor = std::move(predictor_)](core_t::TTime time_) {
                              return predictor(time_, {});
                          };
                      },
                      [this] { return this->predictor(E_Seasonal | E_Calendar); },
                      testForSeasonality};

    m_ChangePointTest.handle(message);
    m_Components.handle(message);
    m_SeasonalityTest.handle(message);
    m_CalendarCyclicTest.handle(message);
}

void CTimeSeriesDecomposition::shiftTime(core_t::TTime time, core_t::TTime shift) {
    m_SeasonalityTest.shiftTime(time, shift);
    m_TimeShift += shift;
    m_LastValueTime += shift;
    m_LastPropagationTime += shift;
}

void CTimeSeriesDecomposition::propagateForwardsTo(core_t::TTime time) {
    if (time > m_LastPropagationTime) {
        m_ChangePointTest.propagateForwards(m_LastPropagationTime, time);
        m_SeasonalityTest.propagateForwards(m_LastPropagationTime, time);
        m_CalendarCyclicTest.propagateForwards(m_LastPropagationTime, time);
        m_Components.propagateForwards(m_LastPropagationTime, time);
    }
    m_LastPropagationTime = std::max(m_LastPropagationTime, time);
}

double CTimeSeriesDecomposition::meanValue(core_t::TTime time) const {
    return m_Components.meanValue(time);
}

CTimeSeriesDecomposition::TVector2x1
CTimeSeriesDecomposition::value(core_t::TTime time, double confidence, int components, bool smooth) const {

    TVector2x1 result{0.0};

    time += m_TimeShift;

    if ((components & E_TrendForced) != 0) {
        result += m_Components.trend().value(time, confidence);
    } else if ((components & E_Trend) != 0) {
        if (m_Components.usingTrendForPrediction()) {
            result += m_Components.trend().value(time, confidence);
        }
    }

    if ((components & E_Seasonal) != 0) {
        for (const auto& component : m_Components.seasonal()) {
            if (component.initialized() && component.time().inWindow(time)) {
                result += component.value(time, confidence);
            }
        }
    }

    if ((components & E_Calendar) != 0) {
        for (const auto& component : m_Components.calendar()) {
            if (component.initialized() && component.feature().inWindow(time)) {
                result += component.value(time, confidence);
            }
        }
    }

    if (smooth) {
        result += this->smooth(
            [&](core_t::TTime time_) {
                return this->value(time_ - m_TimeShift, confidence,
                                   components & E_Seasonal, false);
            },
            time, components);
    }

    return result;
}

CTimeSeriesDecomposition::TVector2x1
CTimeSeriesDecomposition::value(core_t::TTime time, double confidence, bool isNonNegative) const {
    auto result = this->value(time, confidence, E_All, true);
    return isNonNegative ? max(result, 0.0) : result;
}

CTimeSeriesDecomposition::TFilteredPredictor
CTimeSeriesDecomposition::predictor(int components) const {

    auto trend_ = (((components & E_TrendForced) != 0) || ((components & E_Trend) != 0))
                      ? m_Components.trend().predictor()
                      : [](core_t::TTime) { return 0.0; };

    return [ components, trend = std::move(trend_),
             this ](core_t::TTime time, const TBoolVec& removedSeasonalMask) {

        double result{0.0};

        time += m_TimeShift;

        if ((components & E_TrendForced) != 0) {
            result += trend(time);
        } else if ((components & E_Trend) != 0) {
            if (m_Components.usingTrendForPrediction()) {
                result += trend(time);
            }
        }

        if ((components & E_Seasonal) != 0) {
            const auto& seasonal = m_Components.seasonal();
            for (std::size_t i = 0; i < seasonal.size(); ++i) {
                if (seasonal[i].initialized() &&
                    (removedSeasonalMask.empty() || removedSeasonalMask[i] == false) &&
                    seasonal[i].time().inWindow(time)) {
                    result += seasonal[i].value(time, 0.0).mean();
                }
            }
        }

        if ((components & E_Calendar) != 0) {
            for (const auto& component : m_Components.calendar()) {
                if (component.initialized() && component.feature().inWindow(time)) {
                    result += component.value(time, 0.0).mean();
                }
            }
        }

        return result;
    };
}

core_t::TTime CTimeSeriesDecomposition::maximumForecastInterval() const {
    return m_Components.trend().maximumForecastInterval();
}

void CTimeSeriesDecomposition::forecast(core_t::TTime startTime,
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

    auto seasonal = [this, confidence](core_t::TTime time) -> TVector2x1 {
        TVector2x1 prediction{0.0};
        for (const auto& component : m_Components.seasonal()) {
            if (component.initialized() && component.time().inWindow(time)) {
                prediction += component.value(time, confidence);
            }
        }
        for (const auto& component : m_Components.calendar()) {
            if (component.initialized() && component.feature().inWindow(time)) {
                prediction += component.value(time, confidence);
            }
        }
        return prediction;
    };

    startTime += m_TimeShift;
    endTime += m_TimeShift;
    endTime = startTime + common::CIntegerTools::ceil(endTime - startTime, step);

    auto forecastSeasonal = [&](core_t::TTime time) -> TDouble3Vec {
        m_Components.interpolateForForecast(time);

        TVector2x1 bounds{seasonal(time)};

        // Decompose the smoothing into shift plus stretch and ensure that the
        // smoothed interval between the prediction bounds remains positive length.
        TVector2x1 smoothing{this->smooth(seasonal, time, E_Seasonal)};
        double shift{smoothing.mean()};
        double stretch{std::max(smoothing(1) - smoothing(0), bounds(0) - bounds(1))};
        bounds += TVector2x1{{shift - stretch / 2.0, shift + stretch / 2.0}};

        double variance{this->meanVariance()};
        double boundsScale{std::sqrt(std::max(
            minimumScale, this->varianceScaleWeight(time, variance, 0.0).mean()))};
        double prediction{(bounds(0) + bounds(1)) / 2.0};
        double interval{boundsScale * (bounds(1) - bounds(0))};

        return {prediction - interval / 2.0, prediction, prediction + interval / 2.0};
    };

    m_Components.trend().forecast(startTime, endTime, step, confidence,
                                  isNonNegative, forecastSeasonal, writer);
}

double CTimeSeriesDecomposition::detrend(core_t::TTime time,
                                         double value,
                                         double confidence,
                                         bool isNonNegative,
                                         core_t::TTime maximumTimeShift) const {
    if (this->initialized() == false) {
        return value;
    }

    TVector2x1 interval{this->value(time, confidence, E_All ^ E_Seasonal, true)};
    auto shiftedInterval = [&](core_t::TTime shift) -> TVector2x1 {
        TVector2x1 result{interval + this->value(time + shift, confidence, E_Seasonal, true)};
        return isNonNegative ? max(result, 0.0) : result;
    };

    core_t::TTime shift{0};
    if (maximumTimeShift > 0) {
        auto loss = [&](double shift_) -> double {
            TVector2x1 interval_{shiftedInterval(static_cast<core_t::TTime>(shift_ + 0.5))};
            return std::fabs(std::min(value - interval_(0), 0.0) +
                             std::max(value - interval_(1), 0.0));
        };
        std::tie(shift, std::ignore) =
            CSeasonalComponent::likelyShift(maximumTimeShift, 0, loss);
    }

    interval = shiftedInterval(shift);

    return std::min(value - interval(0), 0.0) + std::max(value - interval(1), 0.0);
}

double CTimeSeriesDecomposition::meanVariance() const {
    return m_Components.meanVarianceScale() * m_Components.meanVariance();
}

CTimeSeriesDecomposition::TVector2x1
CTimeSeriesDecomposition::varianceScaleWeight(core_t::TTime time,
                                              double variance,
                                              double confidence,
                                              bool smooth) const {
    if (this->initialized() == false) {
        return TVector2x1{1.0};
    }

    if (common::CMathsFuncs::isFinite(variance) == false) {
        LOG_ERROR(<< "Supplied variance is " << variance << ".");
        return TVector2x1{1.0};
    }
    double mean{this->meanVariance()};
    if (mean <= 0.0 || variance <= 0.0) {
        return TVector2x1{1.0};
    }

    time += m_TimeShift;

    double components{0.0};
    TVector2x1 scale(0.0);
    if (m_Components.usingTrendForPrediction()) {
        scale += m_Components.trend().variance(confidence);
    }
    for (const auto& component : m_Components.seasonal()) {
        if (component.initialized() && component.time().inWindow(time)) {
            scale += component.variance(time, confidence);
            components += 1.0;
        }
    }
    for (const auto& component : m_Components.calendar()) {
        if (component.initialized() && component.feature().inWindow(time)) {
            scale += component.variance(time, confidence);
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
    scale = max(TVector2x1{1.0} + bias * (scale - TVector2x1{1.0}), TVector2x1{0.0});

    if (smooth) {
        scale += this->smooth(
            [&](core_t::TTime time_) {
                return this->varianceScaleWeight(time_ - m_TimeShift, variance,
                                                 confidence, false);
            },
            time, E_All);
    }

    // If anything overflowed just bail and don't scale.
    return common::CMathsFuncs::isFinite(scale) ? scale : TVector2x1{1.0};
}

CTimeSeriesDecomposition::TVector2x1
CTimeSeriesDecomposition::varianceScaleWeight(core_t::TTime time,
                                              double variance,
                                              double confidence) const {
    return this->varianceScaleWeight(time, variance, confidence, true);
}

double CTimeSeriesDecomposition::countWeight(core_t::TTime time) const {
    return m_ChangePointTest.countWeight(time);
}

double CTimeSeriesDecomposition::winsorisationDerate(core_t::TTime time, double error) const {
    return m_ChangePointTest.winsorisationDerate(time, error);
}

CTimeSeriesDecomposition::TFloatMeanAccumulatorVec
CTimeSeriesDecomposition::residuals(bool isNonNegative) const {
    return m_SeasonalityTest.residuals([&](core_t::TTime time) {
        return this->value(time, 0.0, isNonNegative).mean();
    });
}

void CTimeSeriesDecomposition::skipTime(core_t::TTime skipInterval) {
    m_LastValueTime += skipInterval;
    m_LastPropagationTime += skipInterval;
}

std::uint64_t CTimeSeriesDecomposition::checksum(std::uint64_t seed) const {
    seed = common::CChecksum::calculate(seed, m_LastValueTime);
    seed = common::CChecksum::calculate(seed, m_LastPropagationTime);
    seed = common::CChecksum::calculate(seed, m_ChangePointTest);
    seed = common::CChecksum::calculate(seed, m_SeasonalityTest);
    seed = common::CChecksum::calculate(seed, m_CalendarCyclicTest);
    return common::CChecksum::calculate(seed, m_Components);
}

void CTimeSeriesDecomposition::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CTimeSeriesDecomposition");
    core::CMemoryDebug::dynamicSize("m_Mediator", m_Mediator, mem);
    core::CMemoryDebug::dynamicSize("m_ChangePointTest", m_ChangePointTest, mem);
    core::CMemoryDebug::dynamicSize("m_SeasonalityTest", m_SeasonalityTest, mem);
    core::CMemoryDebug::dynamicSize("m_CalendarCyclicTest", m_CalendarCyclicTest, mem);
    core::CMemoryDebug::dynamicSize("m_Components", m_Components, mem);
}

std::size_t CTimeSeriesDecomposition::memoryUsage() const {
    return core::CMemory::dynamicSize(m_Mediator) +
           core::CMemory::dynamicSize(m_ChangePointTest) +
           core::CMemory::dynamicSize(m_SeasonalityTest) +
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

core_t::TTime CTimeSeriesDecomposition::lastValueTime() const {
    return m_LastValueTime;
}

void CTimeSeriesDecomposition::initializeMediator() {
    m_Mediator = std::make_unique<CMediator>();
    m_Mediator->registerHandler(m_ChangePointTest);
    m_Mediator->registerHandler(m_SeasonalityTest);
    m_Mediator->registerHandler(m_CalendarCyclicTest);
    m_Mediator->registerHandler(m_Components);
}

template<typename F>
auto CTimeSeriesDecomposition::smooth(const F& f, core_t::TTime time, int components) const
    -> decltype(f(time)) {

    using TResultType = decltype(f(time));

    if ((components & E_Seasonal) != E_Seasonal) {
        return TResultType{0.0};
    }

    auto offset = [&f, time](core_t::TTime discontinuity) {
        auto baselineMinusEps = f(discontinuity - 1);
        auto baselinePlusEps = f(discontinuity + 1);
        return 0.5 *
               std::max((1.0 - static_cast<double>(std::abs(time - discontinuity)) /
                                   static_cast<double>(SMOOTHING_INTERVAL)),
                        0.0) *
               (baselinePlusEps - baselineMinusEps);
    };

    for (const auto& component : m_Components.seasonal()) {
        if (component.initialized() == false ||
            component.time().windowRepeat() <= SMOOTHING_INTERVAL) {
            continue;
        }

        const CSeasonalTime& times{component.time()};

        bool timeInWindow{times.inWindow(time)};
        bool inWindowBefore{times.inWindow(time - SMOOTHING_INTERVAL)};
        bool inWindowAfter{times.inWindow(time + SMOOTHING_INTERVAL)};
        if (timeInWindow == false && inWindowBefore) {
            core_t::TTime discontinuity{times.startOfWindow(time - SMOOTHING_INTERVAL) +
                                        times.windowLength()};
            return -offset(discontinuity);
        }
        if (timeInWindow == false && inWindowAfter) {
            core_t::TTime discontinuity{component.time().startOfWindow(time + SMOOTHING_INTERVAL)};
            return offset(discontinuity);
        }
    }

    return TResultType{0.0};
}

const core_t::TTime CTimeSeriesDecomposition::SMOOTHING_INTERVAL{14400};
}
}
}
