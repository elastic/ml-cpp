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

#include <algorithm>
#include <limits>
#include <numeric>

namespace ml {
namespace maths {

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

    LOG_TRACE(<< "decompose into seasonal components");

    TSizeVec trendSegments{TSegmentation::piecewiseLinear(m_Values)};
    LOG_TRACE(<< "trend segments = " << core::CContainerPrinter::print(trendSegments));

    std::size_t numberTrendSegments{1};

    TRemoveTrend trendModels[]{
        [&](TFloatMeanAccumulatorVec&) {
            LOG_TRACE(<< "no trend");
            numberTrendSegments = 1;
            return true;
        },
        [&](TFloatMeanAccumulatorVec& values) {
            LOG_TRACE(<< "linear trend");
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
    CSignal::TWeightFunc weight = [this](std::size_t period) {
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

        valuesToTest = m_Values;

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
                                                 hypothesis.s_NumberScaleSegments - 1)};
    hypothesis.s_Truth =
        (fuzzyGreaterThan(repeatsPerSegment / m_MinimumRepeatsPerSegmentForVariance, 1.0, 0.2) &&
         fuzzyGreaterThan(hypothesis.s_Autocorrelation / m_MinimumAutocorrelation, 1.0, 0.1) &&
         fuzzyLessThan(hypothesis.s_ExplainedVariance / m_MaximumExplainedVariance, 1.0, 0.1) &&
         fuzzyLessThan(std::log(hypothesis.s_ExplainedVariancePValue / m_MaximumExplainedVariancePValue),
                       0.0, 0.1)) ||
        (fuzzyGreaterThan(repeatsPerSegment / m_MinimumRepeatsPerSegmentForAmplitude, 1.0, 0.2) &&
         fuzzyGreaterThan(std::max(hypothesis.s_Autocorrelation, hypothesis.s_AbsAutocorrelation) /
                              m_MinimumAutocorrelation,
                          1.0, 0.15) &&
         fuzzyLessThan(std::log(hypothesis.s_AmplitudePValue / m_MaximumAmplitudePValue),
                       0.0, 0.1));
}

CTimeSeriesTestForSeasonality::TFloatMeanAccumulatorVecHypothesisStatsVecPr
CTimeSeriesTestForSeasonality::testDecomposition(TFloatMeanAccumulatorVec& valuesToTest,
                                                 std::size_t numberTrendSegments,
                                                 const TSeasonalComponentVec& periods) const {
    using TComputeScaling =
        std::function<bool(TFloatMeanAccumulatorVec&, SHypothesisStats&)>;

    LOG_TRACE(<< "testing " << core::CContainerPrinter::print(periods));

    TComputeScaling scalings[]{
        [&](TFloatMeanAccumulatorVec& values, SHypothesisStats& stats) {
            stats.s_ScaleSegments.assign({0, values.size()});
            return true;
        },
        [&](TFloatMeanAccumulatorVec& values, SHypothesisStats& hypothesis) {
            std::size_t period{hypothesis.s_Period.period()};
            hypothesis.s_ScaleSegments =
                TSegmentation::piecewiseLinearScaledSeasonal(values, period);
            return this->meanScale(values, hypothesis,
                                   [](std::size_t) { return 1.0; });
        }};

    TFloatMeanAccumulatorVec residuals{valuesToTest};
    THypothesisStatsVec hypotheses;
    hypotheses.reserve(periods.size());

    for (std::size_t i = 0; i < periods.size(); ++i) {

        LOG_TRACE(<< "testing " << periods[i].print());

        // Precondition by removing all remaining components.
        m_ValuesToTestComponent = residuals;
        m_Periods.assign(periods.begin() + i + 1, periods.end());
        if (m_Periods.size() > 0) {
            CSignal::fitSeasonalComponents(m_Periods, m_ValuesToTestComponent, m_Components);
            this->removeComponentPredictions(m_Periods.size(), m_Periods,
                                             m_Components, m_ValuesToTestComponent);
        }

        // Restrict to the component to test time windows.
        m_WindowIndices.resize(m_ValuesToTestComponent.size());
        std::iota(m_WindowIndices.begin(), m_WindowIndices.end(), 0);
        CSignal::restrictTo(periods[i], m_ValuesToTestComponent);
        CSignal::restrictTo(periods[i], m_WindowIndices);
        TSeasonalComponent period{CSignal::seasonalComponentSummary(periods[i].period())};

        SHypothesisStats bestHypothesis{periods[i]};

        for (const auto& scale : scalings) {

            SHypothesisStats hypothesis{periods[i]};

            if (scale(m_ValuesToTestComponent, hypothesis) &&
                CSignal::countNotMissing(m_ValuesToTestComponent) > 0) {

                LOG_TRACE(<< "scale segments = "
                          << core::CContainerPrinter::print(hypothesis.s_ScaleSegments));

                hypothesis.s_NumberTrendSegments = numberTrendSegments;
                hypothesis.s_NumberScaleSegments = hypothesis.s_ScaleSegments.size() - 1;
                hypothesis.s_MeanNumberRepeats =
                    CSignal::meanNumberRepeatedValues(m_ValuesToTestComponent, period);

                this->testExplainedVariance(m_ValuesToTestComponent, period, hypothesis);
                this->testAutocorrelation(m_ValuesToTestComponent, period, hypothesis);
                this->testAmplitude(m_ValuesToTestComponent, period, hypothesis);
                this->truth(hypothesis);
                LOG_TRACE(<< "truth = " << hypothesis.s_Truth.value());

                if (hypothesis.s_Truth.boolean() &&
                    hypothesis.s_Truth.value() > bestHypothesis.s_Truth.value()) {
                    bestHypothesis = std::move(hypothesis);
                }
            }
        }

        if (bestHypothesis.s_Truth.boolean()) {
            LOG_DEBUG(<< periods[i].print() << " "
                      << "repeats = " << bestHypothesis.s_MeanNumberRepeats << " "
                      << "ns = " << bestHypothesis.s_NumberScaleSegments << " "
                      << "nt = " << bestHypothesis.s_NumberTrendSegments << " "
                      << "ve = " << bestHypothesis.s_ExplainedVariance << " "
                      << "R = " << bestHypothesis.s_Autocorrelation << " "
                      << "p(a) = " << bestHypothesis.s_AmplitudePValue << " "
                      << "p(v) = " << bestHypothesis.s_ExplainedVariancePValue
                      << " " << bestHypothesis.s_Truth.print());

            this->updateResiduals(periods[i], bestHypothesis, residuals);
            hypotheses.push_back(std::move(bestHypothesis));
        }
    }

    if (hypotheses.size() > 0) {
        this->finalizeHypotheses(hypotheses);
    }

    return {std::move(residuals), std::move(hypotheses)};
}

void CTimeSeriesTestForSeasonality::updateResiduals(const TSeasonalComponent& period,
                                                    const SHypothesisStats& hypothesis,
                                                    TFloatMeanAccumulatorVec& residuals) const {
    m_ValuesToTestComponent.assign(residuals.begin(), residuals.end());
    CSignal::restrictTo(period, m_ValuesToTestComponent);
    this->meanScale(m_ValuesToTestComponent, hypothesis,
                    [](std::size_t) { return 1.0; });
    m_Periods.assign(1, CSignal::seasonalComponentSummary(period.period()));
    CSignal::fitSeasonalComponents(m_Periods, m_ValuesToTestComponent, m_Components);

    for (std::size_t i = 0; i < m_ValuesToTestComponent.size(); ++i) {
        auto& moments = residuals[m_WindowIndices[i]];
        moments = m_ValuesToTestComponent[i];
        CBasicStatistics::moment<0>(moments) -= m_Periods[0].value(m_Components[0], i);
    }
}

void CTimeSeriesTestForSeasonality::finalizeHypotheses(THypothesisStatsVec& hypotheses) const {
    m_Periods.clear();
    for (auto& hypothesis : hypotheses) {
        m_Periods.push_back(hypothesis.s_Period);
    }
    m_ValuesToTestComponent = m_Values;
    CSignal::fitSeasonalComponentsRobust(m_Periods, m_OutlierFraction,
                                         m_ValuesToTestComponent, m_Components);

    for (std::size_t i = 0; i < hypotheses.size(); ++i) {

        hypotheses[i].s_InitialValues = m_ValuesToTestComponent;

        std::swap(m_Periods[i], m_Periods.back());
        std::swap(m_Components[i], m_Components.back());
        this->removeComponentPredictions(m_Periods.size() - 1, m_Periods, m_Components,
                                         hypotheses[i].s_InitialValues);
        std::swap(m_Periods[i], m_Periods.back());
        std::swap(m_Components[i], m_Components.back());
        CSignal::restrictTo(m_Periods[i], hypotheses[i].s_InitialValues);

        this->meanScale(hypotheses[i].s_InitialValues, hypotheses[i], [&](std::size_t j) {
            return std::pow(
                0.9, static_cast<double>(hypotheses[i].s_InitialValues.size() - j - 1));
        });
        hypotheses[i].s_ComponentSize = CSignal::selectComponentSize(
            hypotheses[i].s_InitialValues, hypotheses[i].s_Period.period());
    }
}

bool CTimeSeriesTestForSeasonality::meanScale(TFloatMeanAccumulatorVec& values,
                                              const SHypothesisStats& hypothesis,
                                              const TWeightFunc& weight) const {
    if (hypothesis.s_ScaleSegments.size() > 2) {
        values = TSegmentation::meanScalePiecewiseLinearScaledSeasonal(
            values, hypothesis.s_Period.period(), hypothesis.s_ScaleSegments, weight);
        return true;
    }
    return false;
}

void CTimeSeriesTestForSeasonality::removeComponentPredictions(
    std::size_t numberOfPeriodsToRemove,
    const TSeasonalComponentVec& periodsToRemove,
    const TMeanAccumulatorVec1Vec& componentsToRemove,
    TFloatMeanAccumulatorVec& values) const {
    for (std::size_t i = 0; i < values.size(); ++i) {
        for (std::size_t j = 0; j < numberOfPeriodsToRemove; ++j) {
            CBasicStatistics::moment<0>(values[i]) -=
                periodsToRemove[j].value(componentsToRemove[j], i);
        }
    }
}

void CTimeSeriesTestForSeasonality::testExplainedVariance(const TFloatMeanAccumulatorVec& valuesToTest,
                                                          const TSeasonalComponent& period,
                                                          SHypothesisStats& hypothesis) const {
    m_Periods.assign(1, period);
    CSignal::fitSeasonalComponents(m_Periods, valuesToTest, m_Components);

    std::size_t numberValues{CSignal::countNotMissing(valuesToTest)};
    std::size_t parameters{CSignal::countNotMissing(m_Components[0])};

    double variances[2];
    std::tie(variances[0], variances[1]) =
        CSignal::residualVariance(valuesToTest, m_Periods, m_Components);
    double degreesFreedom[]{static_cast<double>(numberValues - 1),
                            static_cast<double>(numberValues - parameters)};

    hypothesis.s_ResidualVariance = variances[1];
    if (numberValues <= parameters) {
        hypothesis.s_ExplainedVariance = 1.0;
        hypothesis.s_ExplainedVariancePValue = 1.0;
    } else {
        hypothesis.s_ExplainedVariance =
            CBasicStatistics::varianceAtPercentile(90.0, variances[1], degreesFreedom[1]) /
            CBasicStatistics::varianceAtPercentile(10.0, variances[0], degreesFreedom[0]);
        hypothesis.s_ExplainedVariancePValue = CStatisticalTests::rightTailFTest(
            variances[0] / variances[1], degreesFreedom[1], degreesFreedom[0]);
    }
    LOG_TRACE(<< "variances = " << core::CContainerPrinter::print(variances)
              << ", p-value = " << hypothesis.s_ExplainedVariancePValue);
}

void CTimeSeriesTestForSeasonality::testAutocorrelation(const TFloatMeanAccumulatorVec& valuesToTest,
                                                        const TSeasonalComponent& period,
                                                        SHypothesisStats& hypothesis) const {
    CSignal::TFloatMeanAccumulatorCRng valuesToTestAutocorrelation{
        valuesToTest, 0, CIntegerTools::floor(valuesToTest.size(), period.period())};
    hypothesis.s_Autocorrelation = CSignal::autocorrelationAtPercentile(
        10.0, CSignal::cyclicAutocorrelation(period.period(), valuesToTestAutocorrelation),
        static_cast<double>(CSignal::countNotMissing(valuesToTestAutocorrelation)));
    hypothesis.s_AbsAutocorrelation = CSignal::autocorrelationAtPercentile(
        10.0,
        CSignal::cyclicAutocorrelation(period.period(), valuesToTestAutocorrelation,
                                       [](const TFloatMeanAccumulator& value) {
                                           return std::fabs(CBasicStatistics::mean(value));
                                       }),
        static_cast<double>(CSignal::countNotMissing(valuesToTestAutocorrelation)));
    LOG_TRACE(<< "autocorrelation = " << hypothesis.s_Autocorrelation
              << ", abs autocorrelation = " << hypothesis.s_AbsAutocorrelation);
}

void CTimeSeriesTestForSeasonality::testAmplitude(const TFloatMeanAccumulatorVec& valuesToTest,
                                                  const TSeasonalComponent& period,
                                                  SHypothesisStats& hypothesis) const {

    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

    double level{CBasicStatistics::mean(std::accumulate(
        valuesToTest.begin(), valuesToTest.end(), TMeanAccumulator{},
        [](TMeanAccumulator partialLevel, const TFloatMeanAccumulator& value) {
            partialLevel.add(CBasicStatistics::mean(value), CBasicStatistics::count(value));
            return partialLevel;
        }))};

    m_Amplitudes.assign(period.period(),
                        {valuesToTest.size(), hypothesis.s_MeanNumberRepeats, level});
    for (std::size_t i = 0; i < valuesToTest.size(); ++i) {
        if (period.contains(i)) {
            m_Amplitudes[period.offset(i)].add(i, valuesToTest[i]);
        }
    }

    double pvalue{1.0};
    boost::math::normal normal(0.0, std::sqrt(hypothesis.s_ResidualVariance));
    for (const auto& amplitude : m_Amplitudes) {
        if (amplitude.amplitude() >= 2.0 * boost::math::standard_deviation(normal)) {
            pvalue = std::min(pvalue, amplitude.significance(normal));
        }
    }

    hypothesis.s_AmplitudePValue = CTools::oneMinusPowOneMinusX(
        pvalue, static_cast<double>(std::count_if(
                    m_Amplitudes.begin(), m_Amplitudes.end(), [](const auto& amplitude) {
                        return amplitude.amplitude() > 0.0;
                    })));
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
