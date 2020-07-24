/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CTimeSeriesTestForSeasonality.h>

#include <core/CContainerPrinter.h>
#include <core/CTimeUtils.h>
#include <core/Constants.h>

#include <maths/CBasicStatistics.h>
#include <maths/CFuzzyLogic.h>
#include <maths/CIntegerTools.h>
#include <maths/COrderings.h>
#include <maths/CSeasonalTime.h>
#include <maths/CSetTools.h>
#include <maths/CSignal.h>
#include <maths/CStatisticalTests.h>
#include <maths/CTimeSeriesSegmentation.h>
#include <maths/CTools.h>
#include <maths/MathsTypes.h>

#include <boost/math/distributions/binomial.hpp>
#include <boost/math/distributions/normal.hpp>

#include <algorithm>
#include <limits>
#include <numeric>

namespace ml {
namespace maths {
namespace {
using TDoubleVec = std::vector<double>;
using TSegmentation = CTimeSeriesSegmentation;
using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

//! \brief Accumulates the minimum amplitude.
class CMinAmplitude {
public:
    CMinAmplitude(double meanRepeats, double level)
        : m_Level{level}, m_Min{this->targetCount(meanRepeats)}, m_Max{this->targetCount(meanRepeats)} {
    }

    void add(double x, double n) {
        if (n > 0.0) {
            ++m_Count;
            m_Min.add(x - m_Level);
            m_Max.add(x - m_Level);
        }
    }

    std::size_t count() const { return m_Min.count(); }
    std::size_t capacity() const { return m_Min.capacity(); }

    double amplitude() const {
        if (this->count() == this->capacity()) {
            return std::max(std::max(-m_Min.biggest(), 0.0),
                            std::max(m_Max.biggest(), 0.0));
        }
        return 0.0;
    }

    double significance(const boost::math::normal& normal) const {
        if (this->count() < this->capacity()) {
            return 1.0;
        }
        double f{2.0 * CTools::safeCdf(normal, -this->amplitude())};
        if (f == 0.0) {
            return 0.0;
        }
        double n{static_cast<double>(this->count())};
        boost::math::binomial binomial(static_cast<double>(m_Count), f);
        return CTools::safeCdfComplement(binomial, n - 1.0);
    }

private:
    using TMinAccumulator = CBasicStatistics::COrderStatisticsHeap<double>;
    using TMaxAccumulator =
        CBasicStatistics::COrderStatisticsHeap<double, std::greater<double>>;

private:
    std::size_t targetCount(double meanRepeats) const {
        return std::max(static_cast<std::size_t>(std::ceil(meanRepeats / 3.0)),
                        std::size_t{5});
    }

private:
    //! The mean of the trend.
    double m_Level = 0.0;
    //! The total count of values added.
    std::size_t m_Count = 0;
    //! The smallest values.
    TMinAccumulator m_Min;
    //! The largest values.
    TMaxAccumulator m_Max;
};

using TAmplitudeVec = std::vector<CMinAmplitude>;
}

CNewTrendSummary::CNewTrendSummary(core_t::TTime startTime,
                                   core_t::TTime bucketLength,
                                   TFloatMeanAccumulatorVec initialValues)
    : m_StartTime{startTime}, m_BucketLength{bucketLength}, m_InitialValues{std::move(initialValues)} {
}

CNewTrendSummary::CInitialValueConstIterator CNewTrendSummary::beginInitialValues() const {
    return {*this, 0};
}

CNewTrendSummary::CInitialValueConstIterator CNewTrendSummary::endInitialValues() const {
    return {*this, m_InitialValues.size()};
}

CNewSeasonalComponentSummary::CNewSeasonalComponentSummary(std::string annotationText,
                                                           const TSeasonalComponent& period,
                                                           std::size_t size,
                                                           bool diurnal,
                                                           core_t::TTime startTime,
                                                           core_t::TTime bucketLength,
                                                           TFloatMeanAccumulatorVec initialValues,
                                                           double precedence)
    : m_AnnotationText{std::move(annotationText)}, m_Period{period}, m_Size{size},
      m_Diurnal{diurnal}, m_StartTime{startTime}, m_BucketLength{bucketLength},
      m_InitialValues{std::move(initialValues)}, m_Precedence{precedence} {
}

const std::string& CNewSeasonalComponentSummary::annotationText() const {
    return m_AnnotationText;
}

std::size_t CNewSeasonalComponentSummary::size() const {
    return m_Size;
}

CNewSeasonalComponentSummary::TSeasonalTimeUPtr
CNewSeasonalComponentSummary::seasonalTime() const {
    if (m_Diurnal) {
        return std::make_unique<CDiurnalTime>(
            m_Period.windowed()
                ? (m_StartTime + static_cast<core_t::TTime>(m_Period.s_StartOfWeek) * m_BucketLength) %
                      m_Period.s_WindowRepeat
                : 0,
            static_cast<core_t::TTime>(m_Period.s_Window.first) * m_BucketLength,
            static_cast<core_t::TTime>(m_Period.s_Window.first) * m_BucketLength,
            static_cast<core_t::TTime>(m_Period.s_Period) * m_BucketLength, m_Precedence);
    }
    return std::make_unique<CGeneralPeriodTime>(
        static_cast<core_t::TTime>(m_Period.s_Period) * m_BucketLength, m_Precedence);
}

CNewSeasonalComponentSummary::CInitialValueConstIterator
CNewSeasonalComponentSummary::beginInitialValues() const {
    auto windows = m_Period.windows(m_InitialValues.size());
    return {*this, windows.front().first, m_InitialValues.begin(), std::move(windows)};
}

CNewSeasonalComponentSummary::CInitialValueConstIterator
CNewSeasonalComponentSummary::endInitialValues() const {
    return {*this, CInitialValueConstIterator::END, m_InitialValues.end(), {}};
}

std::string CNewSeasonalComponentSummary::print() const {
    std::ostringstream result;
    result << m_BucketLength * m_Period.s_Period;
    if (m_Period.windowed()) {
        result << "/(" << m_BucketLength * m_Period.s_Window.first << ","
               << m_BucketLength * m_Period.s_Window.second << ")";
    }
    return result.str();
}

void CSeasonalHypotheses::add(CNewTrendSummary trend) {
    m_Trend = std::move(trend);
}

void CSeasonalHypotheses::add(std::string annotationText,
                              const TSeasonalComponent& period,
                              std::size_t size,
                              bool diurnal,
                              core_t::TTime startTime,
                              core_t::TTime bucketLength,
                              TFloatMeanAccumulatorVec initialValues,
                              double precedence) {
    m_Components.emplace_back(std::move(annotationText), period, size, diurnal, startTime,
                              bucketLength, std::move(initialValues), precedence);
}

const CNewTrendSummary* CSeasonalHypotheses::trend() const {
    return m_Trend != boost::none ? &(*m_Trend) : nullptr;
}

const CSeasonalHypotheses::TNewSeasonalComponentVec& CSeasonalHypotheses::components() const {
    return m_Components;
}

std::string CSeasonalHypotheses::print() const {
    return core::CContainerPrinter::print(m_Components);
}

CTimeSeriesTestForSeasonality::CTimeSeriesTestForSeasonality(core_t::TTime startTime,
                                                             core_t::TTime bucketLength,
                                                             TFloatMeanAccumulatorVec values,
                                                             double outlierFraction)
    : m_StartTime{startTime}, m_BucketLength{bucketLength},
      m_OutlierFraction{outlierFraction}, m_Values{std::move(values)} {
}

void CTimeSeriesTestForSeasonality::addModelledSeasonality(const CSeasonalTime& period) {
    m_ModelledPeriods.push_back(static_cast<std::size_t>(
        (period.period() + m_BucketLength / 2) / m_BucketLength));
}

CSeasonalHypotheses CTimeSeriesTestForSeasonality::decompose() {

    using TRemoveTrend = std::function<bool(TFloatMeanAccumulatorVec&)>;

    LOG_TRACE(<< "Decompose into seasonal components");

    TSizeVec trendSegments{TSegmentation::piecewiseLinear(m_Values)};
    LOG_TRACE(<< "trend segments = " << core::CContainerPrinter::print(trendSegments));

    std::size_t numberTrendSegments{1};

    TRemoveTrend trendModels[]{
        [&](TFloatMeanAccumulatorVec&) {
            LOG_TRACE(<< "No trend");
            numberTrendSegments = 1;
            return true;
        },
        [&](TFloatMeanAccumulatorVec& values) {
            LOG_TRACE(<< "Linear trend");
            numberTrendSegments = 1;
            CSignal::removeLinearTrend(values);
            return true;
        },
        [&](TFloatMeanAccumulatorVec& values) {
            numberTrendSegments = trendSegments.size() - 1;
            if (trendSegments.size() > 2) {
                LOG_TRACE(<< trendSegments.size() - 1 << " linear trend segments");
                values = TSegmentation::removePiecewiseLinear(std::move(values), trendSegments);
                return true;
            }
            return false;
        }};

    // Increase the weight of diurnal and already modelled seasonal components.
    std::sort(m_ModelledPeriods.begin(), m_ModelledPeriods.end());
    CSignal::TWeightFunction weight = [this](std::size_t period) {
        if (this->isDiurnal(period) ||
            std::binary_search(m_ModelledPeriods.begin(), m_ModelledPeriods.end(), period)) {
            return 1.1;
        }
        return 1.0;
    };

    TFloatMeanAccumulatorVec valuesToTest;
    TSeasonalComponentVec periods;

    TFloatMeanAccumulatorVecHypothesisStatsVecPrVec hypotheses;
    hypotheses.reserve(3);

    for (const auto& removeTrend : trendModels) {

        valuesToTest.assign(m_Values.begin(), m_Values.end());

        if (removeTrend(valuesToTest)) {
            periods = CSignal::seasonalDecomposition(
                valuesToTest, m_OutlierFraction, this->week(), weight, m_StartOfWeek);
            this->appendDiurnalComponents(valuesToTest, periods);
            periods.erase(std::remove_if(periods.begin(), periods.end(),
                                         [this](const auto& period) {
                                             return this->seenSufficientData(period) == false;
                                         }),
                          periods.end());

            hypotheses.push_back(
                this->testDecomposition(valuesToTest, numberTrendSegments, periods));
        }
    }

    return this->select(hypotheses);
}

CSeasonalHypotheses CTimeSeriesTestForSeasonality::select(
    TFloatMeanAccumulatorVecHypothesisStatsVecPrVec& hypotheses) const {

    // Choose the trend hypothesis which is most true.

    std::size_t selected{hypotheses.size()};
    CFuzzyTruthValue mostTrue{CFuzzyTruthValue::FALSE};

    for (std::size_t i = 0; i < hypotheses.size(); ++i) {
        CFuzzyTruthValue truth{CFuzzyTruthValue::FALSE};
        if (hypotheses[i].second.size() > 0) {
            truth = std::accumulate(
                hypotheses[i].second.begin(), hypotheses[i].second.end(),
                CFuzzyTruthValue::TRUE,
                [](CFuzzyTruthValue partialTruth, const SHypothesisStats& hypothesis) {
                    return partialTruth && hypothesis.s_Truth;
                });
        }
        if (mostTrue < truth) {
            std::tie(selected, mostTrue) = std::make_pair(i, truth);
        }
    }

    CSeasonalHypotheses result;

    if (selected < hypotheses.size()) {

        TFloatMeanAccumulatorVec trendInitialValues;
        THypothesisStatsVec selectedHypotheses;
        std::tie(trendInitialValues, selectedHypotheses) = std::move(hypotheses[selected]);

        result.add(CNewTrendSummary{m_StartTime, m_BucketLength, std::move(trendInitialValues)});
        for (const auto& hypothesis : selectedHypotheses) {
            result.add(this->annotationText(hypothesis.s_Period),
                       hypothesis.s_Period, hypothesis.s_ComponentSize,
                       this->isDiurnal(hypothesis.s_Period.s_Period), m_StartTime,
                       m_BucketLength, std::move(hypothesis.s_InitialValues),
                       this->precedence(hypothesis.s_Period));
        }
    }

    return result;
}

void CTimeSeriesTestForSeasonality::truth(SHypothesisStats& hypothesis) const {
    double repeatsPerSegment{hypothesis.s_MeanNumberRepeats /
                             static_cast<double>(hypothesis.s_NumberTrendSegments +
                                                 hypothesis.s_NumberScaleSegments)};
    hypothesis.s_Truth =
        (fuzzyGreaterThan(repeatsPerSegment / m_MinimumRepeatsPerSegmentForVariance, 1.0, 0.2) &&
         fuzzyGreaterThan(hypothesis.s_Autocorrelation / m_MinimumAutocorrelation, 1.0, 0.1) &&
         fuzzyGreaterThan(hypothesis.s_ExplainedVariance / m_MinimumExplainedVariance, 1.0, 0.1) &&
         fuzzyLessThan(std::log(hypothesis.s_ExplainedVariancePValue / m_MaximumExplainedVariancePValue),
                       0.0, 0.1)) ||
        (fuzzyGreaterThan(repeatsPerSegment / m_MinimumRepeatsPerSegmentForAmplitude, 1.0, 0.2) &&
         fuzzyLessThan(std::log(hypothesis.s_AmplitudePValue / m_MaximumAmplitudePValue),
                       0.0, 0.1));
}

CTimeSeriesTestForSeasonality::TFloatMeanAccumulatorVecHypothesisStatsVecPr
CTimeSeriesTestForSeasonality::testDecomposition(TFloatMeanAccumulatorVec& valuesToTest,
                                                 std::size_t numberTrendSegments,
                                                 const TSeasonalComponentVec& periods) const {
    using TComputeScaling =
        std::function<bool(TFloatMeanAccumulatorVec&, const TSeasonalComponent&)>;

    LOG_TRACE(<< "testing " << core::CContainerPrinter::print(periods));

    CSignal::TMeanAccumulatorVec1Vec components;
    CSignal::fitSeasonalComponents(periods, valuesToTest, components);

    TFloatMeanAccumulatorVec valuesToTestComponent;

    TSizeVec scaleSegments;
    TDoubleVec scales;
    double componentInitialValuesScale{1.0};

    TComputeScaling scalings[]{
        [&](TFloatMeanAccumulatorVec& values, const TSeasonalComponent&) {
            scaleSegments.assign({0, values.size()});
            componentInitialValuesScale = 1.0;
            return true;
        },
        [&](TFloatMeanAccumulatorVec& values, const TSeasonalComponent& period) {
            scaleSegments = TSegmentation::piecewiseLinearScaledSeasonal(
                values, period.period());
            componentInitialValuesScale = 1.0;
            if (scaleSegments.size() > 2) {
                std::tie(values, scales) = TSegmentation::meanScalePiecewiseLinearScaledSeasonal(
                    std::move(values), period.period(), scaleSegments);
                auto decay = [&](std::size_t i) {
                    return std::pow(0.9, static_cast<double>(values.size() - i));
                };
                auto unit = [](std::size_t) { return 1.0; };
                componentInitialValuesScale =
                    TSegmentation::meanScale(scaleSegments, scales, decay) /
                    TSegmentation::meanScale(scaleSegments, scales, unit);
                return true;
            }
            return false;
        }};

    auto removeComponentPredictions = [&](std::size_t j, TFloatMeanAccumulatorVec& values) {
        for (std::size_t i = 0; i < values.size(); ++i) {
            CBasicStatistics::moment<0>(values[i]) -= periods[j].value(components[j], i);
        }
    };

    TFloatMeanAccumulatorVec trendInitialValues(valuesToTest.size());
    THypothesisStatsVec hypotheses;
    hypotheses.reserve(components.size());

    for (std::size_t i = 0; i < components.size(); ++i) {

        LOG_TRACE(<< "testing " << periods[i].print());

        // Precondition by removing all other components.
        valuesToTestComponent.assign(valuesToTest.begin(), valuesToTest.end());
        for (std::size_t j = 0; j < i; ++j) {
            removeComponentPredictions(j, valuesToTestComponent);
        }
        for (std::size_t j = i + 1; j < components.size(); ++j) {
            removeComponentPredictions(j, valuesToTestComponent);
        }

        // Restrict to component's windows.
        CSignal::restrictTo(periods[i], valuesToTestComponent);
        TSeasonalComponent period{CSignal::seasonalComponentSummary(periods[i].period())};

        SHypothesisStats bestHypothesis{periods[i]};

        for (const auto& scale : scalings) {
            if (scale(valuesToTestComponent, period)) {
                LOG_TRACE(<< "scale segments = " << core::CContainerPrinter::print(scaleSegments));

                SHypothesisStats hypothesis{periods[i]};
                hypothesis.s_ComponentInitialValuesScale = componentInitialValuesScale;
                hypothesis.s_NumberTrendSegments = numberTrendSegments;
                hypothesis.s_NumberScaleSegments = scaleSegments.size();
                hypothesis.s_MeanNumberRepeats =
                    CSignal::meanNumberRepeatedValues(valuesToTestComponent, period);

                this->testExplainedVariance(valuesToTestComponent, period, hypothesis);
                this->testAutocorrelation(valuesToTestComponent, period, hypothesis);
                this->testAmplitude(valuesToTestComponent, period, hypothesis);
                this->truth(hypothesis);
                LOG_TRACE(<< "truth = " << hypothesis.s_Truth.value());

                if (hypothesis.s_Truth.boolean() &&
                    hypothesis.s_Truth.value() > bestHypothesis.s_Truth.value()) {
                    hypothesis.s_InitialValues = valuesToTestComponent;
                    bestHypothesis = std::move(hypothesis);
                }
            }
        }

        if (bestHypothesis.s_Truth.boolean()) {
            bestHypothesis.s_ComponentSize = CSignal::selectComponentSize(
                bestHypothesis.s_InitialValues, period.period());
            this->updateInitialValues(period, bestHypothesis, trendInitialValues);
            hypotheses.push_back(std::move(bestHypothesis));
        }
    }

    return {std::move(trendInitialValues), std::move(hypotheses)};
}

void CTimeSeriesTestForSeasonality::updateInitialValues(const TSeasonalComponent& period,
                                                        SHypothesisStats& hypothesis,
                                                        TFloatMeanAccumulatorVec& trendInitialValues) const {
    auto& initialValues = hypothesis.s_InitialValues;

    CSignal::TMeanAccumulatorVec1Vec component;
    CSignal::fitSeasonalComponents({period}, initialValues, component);

    std::size_t i{0};
    std::size_t n{CIntegerTools::floor(trendInitialValues.size(), period.s_WindowRepeat)};
    for (const auto& window : hypothesis.s_Period.windows(initialValues.size())) {
        for (std::size_t j = window.first;
             i < initialValues.size() && j < window.second; ++i, ++j) {
            auto& residual = trendInitialValues[j % n];
            residual = initialValues[i];
            CBasicStatistics::moment<0>(residual) -= period.value(component[0], i);
        }
        CBasicStatistics::moment<0>(initialValues[i]) +=
            (hypothesis.s_ComponentInitialValuesScale - 1.0) *
            period.value(component[0], i);
    }
}

void CTimeSeriesTestForSeasonality::testExplainedVariance(const TFloatMeanAccumulatorVec& valuesToTest,
                                                          const TSeasonalComponent& period,
                                                          SHypothesisStats& hypothesis) const {
    CSignal::TMeanAccumulatorVec1Vec component;
    CSignal::fitSeasonalComponents({period}, valuesToTest, component);

    std::size_t numberValues{CSignal::countNotMissing(valuesToTest)};
    std::size_t parameters{CSignal::countNotMissing(component[0])};

    double degreesFreedom[]{static_cast<double>(numberValues - 1),
                            static_cast<double>(numberValues - parameters)};
    double variances[2];
    std::tie(variances[0], variances[1]) =
        CSignal::residualVariance(valuesToTest, {period}, component);

    hypothesis.s_ResidualVariance = variances[1];
    hypothesis.s_ExplainedVariance =
        CBasicStatistics::varianceAtPercentile(10.0, variances[0], degreesFreedom[0]) /
        CBasicStatistics::varianceAtPercentile(90.0, variances[1], degreesFreedom[1]);
    hypothesis.s_ExplainedVariancePValue = CStatisticalTests::rightTailFTest(
        variances[0] / variances[1], degreesFreedom[0], degreesFreedom[1]);
    LOG_TRACE(<< "variances = " << core::CContainerPrinter::print(variances)
              << ", p-value = " << hypothesis.s_ExplainedVariancePValue);
}

void CTimeSeriesTestForSeasonality::testAutocorrelation(const TFloatMeanAccumulatorVec& valuesToTest,
                                                        const TSeasonalComponent& period,
                                                        SHypothesisStats& hypothesis) const {
    std::size_t m{CIntegerTools::floor(valuesToTest.size(), period.period())};
    CSignal::TFloatMeanAccumulatorCRng valuesToTestAutocorrelation{valuesToTest, 0, m};
    std::size_t n{CSignal::countNotMissing(valuesToTestAutocorrelation)};
    hypothesis.s_Autocorrelation = CSignal::autocorrelationAtPercentile(
        10.0, CSignal::cyclicAutocorrelation(period.period(), valuesToTestAutocorrelation),
        static_cast<double>(n));
    LOG_TRACE(<< "autocorrelation = " << hypothesis.s_Autocorrelation);
}

void CTimeSeriesTestForSeasonality::testAmplitude(const TFloatMeanAccumulatorVec& valuesToTest,
                                                  const TSeasonalComponent& period,
                                                  SHypothesisStats& hypothesis) const {

    double level{CBasicStatistics::mean(std::accumulate(
        valuesToTest.begin(), valuesToTest.end(), TMeanAccumulator{},
        [](TMeanAccumulator partialLevel, const TFloatMeanAccumulator& value) {
            partialLevel.add(CBasicStatistics::mean(value), CBasicStatistics::count(value));
            return partialLevel;
        }))};

    TAmplitudeVec amplitudes(period.period(), {hypothesis.s_MeanNumberRepeats, level});
    for (std::size_t i = 0; i < valuesToTest.size(); ++i) {
        if (period.contains(i)) {
            amplitudes[period.offset(i)].add(CBasicStatistics::mean(valuesToTest[i]),
                                             CBasicStatistics::count(valuesToTest[i]));
        }
    }

    double pvalue{1.0};
    boost::math::normal normal(0.0, std::sqrt(hypothesis.s_ResidualVariance));
    for (const auto& amplitude : amplitudes) {
        if (amplitude.amplitude() >= 2.0 * boost::math::standard_deviation(normal)) {
            pvalue = std::min(pvalue, amplitude.significance(normal));
        }
    }

    hypothesis.s_AmplitudePValue = CTools::oneMinusPowOneMinusX(
        pvalue, static_cast<double>(std::count_if(
                    amplitudes.begin(), amplitudes.end(),
                    [](const auto& amplitude) { return amplitude.count() > 0; })));
    LOG_TRACE(<< "amplitude p-value = " << hypothesis.s_AmplitudePValue);
}

void CTimeSeriesTestForSeasonality::appendDiurnalComponents(TFloatMeanAccumulatorVec& valuesToTest,
                                                            TSeasonalComponentVec& periods) const {
    auto periodsInclude = [&](std::size_t period) {
        return std::find_if(periods.begin(), periods.end(), [&period](const auto& entry) {
                   return entry.period() == period;
               }) != periods.end();
    };
    auto periodsIncludeTradingDayDecomposition = [&] {
        return std::find_if(periods.begin(), periods.end(), [](const auto& entry) {
                   return entry.windowed();
               }) != periods.end();
    };

    if (periodsInclude(this->day()) == false) {
        CSignal::appendSeasonalComponentSummary(this->day(), periods);
    }

    if (periodsInclude(this->week()) == false &&
        periodsIncludeTradingDayDecomposition() == false) {
        auto decomposition = CSignal::tradingDayDecomposition(
            valuesToTest, 0.0, this->week(), m_StartOfWeek);
        if (decomposition.empty()) {
            CSignal::appendSeasonalComponentSummary(this->week(), periods);
        } else {
            periods.insert(periods.end(), decomposition.begin(), decomposition.end());
        }
    }

    if (periodsInclude(this->year()) == false) {
        CSignal::appendSeasonalComponentSummary(this->year(), periods);
    }
}

bool CTimeSeriesTestForSeasonality::isDiurnal(std::size_t period) const {
    return period == this->day() || period == this->week() || period == this->year();
}

bool CTimeSeriesTestForSeasonality::isDiurnal(const TSeasonalComponent& period) const {
    return this->isDiurnal(period.s_Period);
}

bool CTimeSeriesTestForSeasonality::isWeekend(const TSeasonalComponent& period) const {
    return period.s_Window == this->weekendWindow();
}

bool CTimeSeriesTestForSeasonality::isWeekday(const TSeasonalComponent& period) const {
    return period.s_Window == this->weekdayWindow();
}

bool CTimeSeriesTestForSeasonality::seenSufficientData(const TSeasonalComponent& period) const {
    return 2 * period.s_WindowRepeat <= this->observedRange();
}

bool CTimeSeriesTestForSeasonality::seenSufficientDataToTestForTradingDayDecomposition() const {
    return 2 * this->week() <= this->observedRange();
}

double CTimeSeriesTestForSeasonality::precedence(const TSeasonalComponent& period) const {
    return this->seenSufficientDataToTestForTradingDayDecomposition() &&
                   period.period() != this->day()
               ? 2.0
               : 1.0;
}

std::string CTimeSeriesTestForSeasonality::annotationText(const TSeasonalComponent& period) const {
    return "Detected periodicity with period " +
           core::CTimeUtils::durationToString(period.s_Period) +
           (this->isWeekend(period) ? " (weekend)"
                                    : (this->isWeekday(period) ? " (weekdays)" : ""));
}

std::size_t CTimeSeriesTestForSeasonality::day() const {
    return (core::constants::DAY + m_BucketLength / 2) / m_BucketLength;
}

std::size_t CTimeSeriesTestForSeasonality::week() const {
    return (core::constants::WEEK + m_BucketLength / 2) / m_BucketLength;
}

std::size_t CTimeSeriesTestForSeasonality::year() const {
    return (core::constants::YEAR + m_BucketLength / 2) / m_BucketLength;
}

CTimeSeriesTestForSeasonality::TSizeSizePr CTimeSeriesTestForSeasonality::weekdayWindow() const {
    return {0, 2 * this->day()};
}

CTimeSeriesTestForSeasonality::TSizeSizePr CTimeSeriesTestForSeasonality::weekendWindow() const {
    return {2 * this->day(), 7 * this->day()};
}

std::size_t CTimeSeriesTestForSeasonality::observedRange() const {
    int begin{0};
    int end{static_cast<int>(m_Values.size())};
    int size{static_cast<int>(m_Values.size())};
    for (/**/; begin < size && CBasicStatistics::count(m_Values[begin]) == 0.0; ++begin) {
    }
    for (/**/; end > begin && CBasicStatistics::count(m_Values[end - 1]) == 0.0; --end) {
    }
    return static_cast<std::size_t>(end - begin);
}
}
}
