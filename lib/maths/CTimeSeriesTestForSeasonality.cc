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
#include <sstream>

namespace ml {
namespace maths {
namespace {
using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

double rightTailFTest(double v0, double v1, double df0, double df1) {
    if (df1 <= 0.0) {
        return 1.0;
    }
    double F{v0 == v1 ? 1.0 : v0 / v1};
    return CStatisticalTests::rightTailFTest(F, df0, df1);
}
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
      m_OutlierFraction{outlierFraction}, m_Values{std::move(values)},
      m_Outliers{static_cast<std::size_t>(
          outlierFraction * static_cast<double>(CSignal::countNotMissing(m_Values)) + 0.5)} {
}

void CTimeSeriesTestForSeasonality::startOfWeek(core_t::TTime value) {
    m_StartOfWeek = static_cast<std::size_t>(
        ((core::constants::WEEK + value - (m_StartTime % core::constants::WEEK)) %
         core::constants::WEEK) /
        m_BucketLength);
}

void CTimeSeriesTestForSeasonality::addModelledSeasonality(const CSeasonalTime& period) {
    auto buckets = [this](core_t::TTime interval) {
        return static_cast<std::size_t>((interval + m_BucketLength / 2) / m_BucketLength);
    };
    std::size_t periodInBuckets{buckets(period.period())};
    std::size_t startOfWindowInBuckets{
        buckets(period.windowed() ? period.startOfWindow(m_StartTime) : 0)};
    std::size_t windowRepeatInBuckets{buckets(period.windowRepeat())};
    TSizeSizePr windowInBuckets{buckets(period.window().first),
                                buckets(period.window().second)};
    m_ModelledPeriods.emplace_back(periodInBuckets, startOfWindowInBuckets,
                                   windowRepeatInBuckets, windowInBuckets);
}

CSeasonalHypotheses CTimeSeriesTestForSeasonality::decompose() {

    using TRemoveTrend = std::function<bool(TFloatMeanAccumulatorVec&)>;

    LOG_TRACE(<< "decompose into seasonal components");

    TSizeVec trendSegments{TSegmentation::piecewiseLinear(m_Values)};
    std::size_t numberTrendSegments{0};
    LOG_TRACE(<< "trend segments = " << core::CContainerPrinter::print(trendSegments));

    TRemoveTrend trendModels[]{
        [&](TFloatMeanAccumulatorVec&) {
            LOG_TRACE(<< "no trend");
            numberTrendSegments = 0;
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

    TModelVec decompositions;
    decompositions.reserve(4 * std::size(trendModels));

    std::stable_sort(m_ModelledPeriods.begin(), m_ModelledPeriods.end(),
                     [](const auto& lhs, const auto& rhs) {
                         return lhs.s_Period < rhs.s_Period;
                     });
    TFloatMeanAccumulatorVec valuesMinusTrend;
    TFloatMeanAccumulatorVec valuesToDecompose;
    TSeasonalComponentVec periods;

    for (const auto& removeTrend : trendModels) {

        valuesMinusTrend = m_Values;

        if (removeTrend(valuesMinusTrend)) {
            this->addNotSeasonal(valuesMinusTrend, numberTrendSegments, decompositions);
            this->addModelled(valuesMinusTrend, numberTrendSegments, periods, decompositions);
            this->addDiurnal(valuesMinusTrend, numberTrendSegments,
                             valuesToDecompose, periods, decompositions);
            this->addDecomposition(valuesMinusTrend, numberTrendSegments,
                                   valuesToDecompose, periods, decompositions);
        }
    }

    return this->select(decompositions);
}

CSeasonalHypotheses CTimeSeriesTestForSeasonality::select(TModelVec& decompositions) const {

    // Choose the trend hypothesis which is the most parsimonious explanation of
    // the values variance.

    // Sort by increasing complexity.
    std::stable_sort(decompositions.begin(), decompositions.end(),
                     [](const auto& lhs, const auto& rhs) {
                         return lhs.numberParameters() < rhs.numberParameters();
                     });

    double eps{std::numeric_limits<double>::epsilon()};
    std::size_t numberValues{static_cast<std::size_t>(
        (1.0 - m_OutlierFraction) * static_cast<double>(CSignal::countNotMissing(m_Values)) + 0.5)};

    auto computePValue = [&](const SModel& H0, const SModel& H1) {
        double degreesFreedom[]{H0.degreesFreedom(numberValues),
                                H1.degreesFreedom(numberValues)};
        double variances[]{H0.s_TruncatedResidualVariance, H0.s_ResidualVariance,
                           H1.s_TruncatedResidualVariance, H1.s_ResidualVariance};
        return std::min(rightTailFTest(variances[0], variances[2],
                                       degreesFreedom[0], degreesFreedom[1]),
                        rightTailFTest(variances[1], variances[3],
                                       degreesFreedom[0], degreesFreedom[1]));
    };
    auto computePValueProxy = [&](const SModel& H0, const SModel& H1) {
        double degreesFreedom[]{H0.degreesFreedom(numberValues),
                                H1.degreesFreedom(numberValues)};
        double variances[]{
            H0.s_TruncatedResidualVariance, H0.s_ResidualVariance,
            std::max(H1.s_TruncatedResidualVariance, eps * H0.s_TruncatedResidualVariance),
            std::max(H1.s_ResidualVariance, eps * H0.s_ResidualVariance)};
        return std::min(
            (variances[0] * degreesFreedom[0]) / (variances[2] * degreesFreedom[1]),
            (variances[1] * degreesFreedom[0]) / (variances[3] * degreesFreedom[1]));
    };

    // Select the best null hypothesis. This is either no seasonality or modelling
    // the same seasonal components.
    std::size_t H0{decompositions.size()};
    double pValueH0{1.0};
    for (std::size_t H1 = 0; H1 < decompositions.size(); ++H1) {
        if (decompositions[H1].isNull(numberValues)) {
            double pValue{[&] {
                if (H0 == decompositions.size()) {
                    return 0.5;
                }
                return rightTailFTest(decompositions[H0].s_ResidualVariance,
                                      decompositions[H1].s_ResidualVariance,
                                      decompositions[H0].degreesFreedom(numberValues),
                                      decompositions[H1].degreesFreedom(numberValues));
            }()};
            LOG_TRACE(<< "hypothesis = " << H1
                      << ", variance = " << decompositions[H1].s_ResidualVariance
                      << ", # parameters = " << decompositions[H1].numberParameters()
                      << ", p-value = " << pValue);
            if (pValue < pValueH0) {
                std::tie(H0, pValueH0) = std::make_pair(H1, pValue);
            }
        }
    }

    // Select the best new decomposition if it is a statistically significant
    // improvement.
    std::size_t selected{decompositions.size()};
    double qualitySelected{-std::numeric_limits<double>::max()};
    for (std::size_t H1 = 0; H1 < decompositions.size(); ++H1) {
        if (decompositions[H1].isAlternative(numberValues)) {
            double pValue{computePValue(decompositions[H0], decompositions[H1])};
            LOG_TRACE(<< "hypothesis = "
                      << core::CContainerPrinter::print(decompositions[H1].s_Hypotheses));
            LOG_TRACE(<< "p-value = " << pValue);
            if (pValue >= m_AcceptedFalsePostiveRate) {
                continue;
            }

            // We know that the model is statistically significant however none
            // of the various alternatives have the status of a null hypothesis.
            // We therefore choose the best model based on the following criteria:
            //   1. The log p-value. This captures information about both the model
            //      size and the variance with and without the model, but has some
            //      blind spots for which the other criteria account.
            //   2. (var(H0) * degrees_freedom(H1)) / (var(H1) * degrees_freedom(H0)).
            //      It is possible for the p-value to underflow. In these cases this
            //      is a proxy which captures something about the relative significance
            //      of the hypotheses.
            //   3. The amount of variance explained per parameter. Models with a
            //      small period which explain most of the variance are preferred
            //      because they can be modelled more accurately.
            //   4. The total model size. The p-value is less sensitive to model
            //      size as the window length increases. However, for both accuracy
            //      and efficiency considerations we strongly prefer smaller models.
            double logPValue{pValue == 0.0 ? std::log(std::numeric_limits<double>::min())
                                           : std::log(pValue)};
            double pValueProxy{computePValueProxy(decompositions[H0], decompositions[H1])};
            double explainedVariance{decompositions[H0].s_ResidualVariance -
                                     decompositions[H1].s_ResidualVariance};
            double explainedTruncatedVariance{decompositions[H0].s_TruncatedResidualVariance -
                                              decompositions[H1].s_TruncatedResidualVariance};
            double explainedVariancePerParameter{
                decompositions[H1].explainedVariancePerParameter(explainedVariance)};
            double explainedTruncatedVariancePerParameter{
                decompositions[H1].explainedVariancePerParameter(explainedTruncatedVariance)};
            double quality{0.5 * std::log(-logPValue) + 0.2 * std::log(pValueProxy) +
                           1.2 * std::log(explainedVariancePerParameter) +
                           1.2 * std::log(explainedTruncatedVariancePerParameter) -
                           0.7 * std::log(decompositions[H1].numberParameters())};
            LOG_TRACE(<< "p-value proxy = " << pValueProxy);
            LOG_TRACE(<< "explained variance = " << explainedVariance
                      << ", per param = " << explainedVariancePerParameter);
            LOG_TRACE(<< "explained truncated variance = " << explainedTruncatedVariance
                      << ", per param = " << explainedTruncatedVariancePerParameter);
            LOG_TRACE(<< "number parameters = " << decompositions[H1].numberParameters());
            LOG_TRACE(<< "quality = " << quality);

            if (quality > qualitySelected) {
                std::tie(selected, qualitySelected) = std::make_pair(H1, quality);
            }
        }
    }

    CSeasonalHypotheses result;

    if (selected < decompositions.size()) {
        LOG_TRACE(<< "selected = "
                  << core::CContainerPrinter::print(decompositions[selected].s_Hypotheses));

        result.add(CNewTrendSummary{m_StartTime, m_BucketLength,
                                    std::move(decompositions[selected].s_TrendInitialValues)});
        for (const auto& hypothesis : decompositions[selected].s_Hypotheses) {
            LOG_TRACE(<< "Adding " << hypothesis.s_Period.print());
            result.add(this->annotationText(hypothesis.s_Period),
                       hypothesis.s_Period, hypothesis.s_ComponentSize,
                       this->isDiurnal(hypothesis.s_Period.s_Period), m_StartTime, m_BucketLength,
                       std::move(hypothesis.s_InitialValues), this->precedence());
        }
    }

    return result;
}

void CTimeSeriesTestForSeasonality::addNotSeasonal(const TFloatMeanAccumulatorVec& valuesMinusTrend,
                                                   std::size_t numberTrendSegments,
                                                   TModelVec& decompositions) const {
    decompositions.emplace_back(
        this->truncatedVariance(0.0, valuesMinusTrend),
        this->truncatedVariance(m_OutlierFraction, valuesMinusTrend),
        numberTrendSegments, TFloatMeanAccumulatorVec{}, THypothesisStatsVec{});
}

void CTimeSeriesTestForSeasonality::addModelled(const TFloatMeanAccumulatorVec& valuesMinusTrend,
                                                std::size_t numberTrendSegments,
                                                TSeasonalComponentVec& periods,
                                                TModelVec& decompositions) const {
    if (m_ModelledPeriods.empty() == false) {
        periods = m_ModelledPeriods;
        this->removeIfInsufficientData(periods);
        auto decomposition = this->testDecomposition(valuesMinusTrend,
                                                     numberTrendSegments, periods);
        if (decomposition.seasonal()) {
            decomposition.s_AlreadyModelled = true;
            decompositions.push_back(std::move(decomposition));
        }
    }
}

void CTimeSeriesTestForSeasonality::addDiurnal(const TFloatMeanAccumulatorVec& valuesMinusTrend,
                                               std::size_t numberTrendSegments,
                                               TFloatMeanAccumulatorVec& valuesToDecompose,
                                               TSeasonalComponentVec& periods,
                                               TModelVec& decompositions) const {
    valuesToDecompose = valuesMinusTrend;
    periods = CSignal::tradingDayDecomposition(valuesToDecompose, m_OutlierFraction,
                                               this->week(), m_StartOfWeek);
    periods.push_back(CSignal::seasonalComponentSummary(this->year()));
    this->removeIfInsufficientData(periods);

    if (periods.size() > 0 && this->alreadyModelled(periods) == false) {
        auto decomposition = this->testDecomposition(valuesMinusTrend,
                                                     numberTrendSegments, periods);
        if (decomposition.seasonal(this->day())) {
            decompositions.push_back(std::move(decomposition));
        }
    }

    periods.assign({CSignal::seasonalComponentSummary(this->day()),
                    CSignal::seasonalComponentSummary(this->week()),
                    CSignal::seasonalComponentSummary(this->year())});
    this->removeIfInsufficientData(periods);

    if (periods.size() > 0 && this->alreadyModelled(periods) == false) {
        auto decomposition = this->testDecomposition(valuesMinusTrend,
                                                     numberTrendSegments, periods);
        if (decomposition.seasonal()) {
            decompositions.push_back(std::move(decomposition));
        }
    }
}

void CTimeSeriesTestForSeasonality::addDecomposition(const TFloatMeanAccumulatorVec& valuesMinusTrend,
                                                     std::size_t numberTrendSegments,
                                                     TFloatMeanAccumulatorVec& valuesToDecompose,
                                                     TSeasonalComponentVec& periods,
                                                     TModelVec& decompositions) const {
    valuesToDecompose = valuesMinusTrend;
    periods = CSignal::seasonalDecomposition(
        valuesToDecompose, m_OutlierFraction, {this->day(), this->week(), this->year()},
        [](std::size_t) { return 1.0; }, m_StartOfWeek);
    this->removeIfInsufficientData(periods);

    if (periods.size() > 0 && this->alreadyModelled(periods) == false &&
        this->onlyDiurnal(periods) == false) {
        auto decomposition = this->testDecomposition(valuesMinusTrend,
                                                     numberTrendSegments, periods);
        if (decomposition.seasonal()) {
            decompositions.push_back(std::move(decomposition));
        }
    }
}

CTimeSeriesTestForSeasonality::SModel
CTimeSeriesTestForSeasonality::testDecomposition(const TFloatMeanAccumulatorVec& valuesToTest,
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
        m_ValuesWithoutComponent = residuals;
        m_Periods.clear();
        for (std::size_t j = i + 1; j < periods.size(); ++j) {
            if (periods[j].period() % periods[i].period() != 0 &&
                periods[i].period() % periods[j].period() != 0) {
                m_Periods.push_back(periods[j]);
            }
        }
        if (m_Periods.size() > 0) {
            CSignal::fitSeasonalComponents(m_Periods, m_ValuesWithoutComponent, m_Components);
            this->removeComponentPredictions(m_Periods.size(), m_Periods,
                                             m_Components, m_ValuesWithoutComponent);
        }

        // Restrict to the component to test time windows.
        m_WindowIndices.resize(m_ValuesWithoutComponent.size());
        std::iota(m_WindowIndices.begin(), m_WindowIndices.end(), 0);
        CSignal::restrictTo(periods[i], m_ValuesWithoutComponent);
        CSignal::restrictTo(periods[i], m_WindowIndices);
        m_ValuesWithComponent = m_ValuesWithoutComponent;
        TSeasonalComponent period{CSignal::seasonalComponentSummary(periods[i].period())};

        // Compute the null hypothesis residual variance statistics.
        m_Periods.assign(1, CSignal::seasonalComponentSummary(1));
        CSignal::fitSeasonalComponentsRobust(m_Periods, m_OutlierFraction,
                                             m_ValuesWithoutComponent, m_Components);
        auto H0 = CSignal::residualVarianceStats(m_ValuesWithoutComponent,
                                                 m_Periods, m_Components);

        SHypothesisStats bestHypothesis{periods[i]};

        for (const auto& scale : scalings) {

            SHypothesisStats hypothesis{periods[i]};

            if (scale(m_ValuesWithComponent, hypothesis) &&
                CSignal::countNotMissing(m_ValuesWithComponent) > 0) {

                LOG_TRACE(<< "scale segments = "
                          << core::CContainerPrinter::print(hypothesis.s_ScaleSegments));

                hypothesis.s_NumberTrendSegments = numberTrendSegments;
                hypothesis.s_NumberScaleSegments = hypothesis.s_ScaleSegments.size() - 1;
                hypothesis.s_MeanNumberRepeats =
                    CSignal::meanNumberRepeatedValues(m_ValuesWithComponent, period);

                m_Periods.assign(1, period);
                CSignal::fitSeasonalComponentsRobust(m_Periods, m_OutlierFraction,
                                                     m_ValuesWithComponent, m_Components);

                this->testExplainedVariance(H0, hypothesis);
                this->testAutocorrelation(hypothesis);
                this->testAmplitude(hypothesis);
                hypothesis.s_Truth = hypothesis.testVariance(*this) ||
                                     hypothesis.testAmplitude(*this);
                LOG_TRACE(<< "truth = " << hypothesis.s_Truth.print());

                if (bestHypothesis.s_Truth.value() <= hypothesis.s_Truth.value()) {
                    bestHypothesis = std::move(hypothesis);
                }
            }
        }

        if (bestHypothesis.s_Truth.boolean()) {
            LOG_TRACE(<< "selected " << periods[i].print());
            this->updateResiduals(bestHypothesis, residuals);
            hypotheses.push_back(std::move(bestHypothesis));
        } else if (bestHypothesis.s_Period.windowed()) {
            hypotheses.push_back(std::move(bestHypothesis));
        }
    }

    if (std::count_if(hypotheses.begin(), hypotheses.end(), [](const auto& hypothesis) {
            return hypothesis.s_Truth.boolean();
        }) == 0) {
        return {};
    }

    this->finalizeHypotheses(hypotheses);
    double variance{this->truncatedVariance(0.0, residuals)};
    double truncatedVariance{this->truncatedVariance(m_OutlierFraction, residuals)};
    for (std::size_t i = 0; i < m_Values.size(); ++i) {
        double offset{CBasicStatistics::mean(m_Values[i]) -
                      CBasicStatistics::mean(valuesToTest[i])};
        CBasicStatistics::moment<0>(residuals[i]) += offset;
    }

    return {variance, truncatedVariance, 2 * numberTrendSegments,
            std::move(residuals), std::move(hypotheses)};
}

void CTimeSeriesTestForSeasonality::updateResiduals(const SHypothesisStats& hypothesis,
                                                    TFloatMeanAccumulatorVec& residuals) const {
    m_ValuesWithComponent = residuals;
    CSignal::restrictTo(hypothesis.s_Period, m_ValuesWithComponent);
    this->meanScale(m_ValuesWithComponent, hypothesis,
                    [](std::size_t) { return 1.0; });
    m_Periods.assign(1, CSignal::seasonalComponentSummary(hypothesis.s_Period.period()));
    CSignal::fitSeasonalComponentsRobust(m_Periods, m_OutlierFraction,
                                         m_ValuesWithComponent, m_Components);

    for (std::size_t i = 0; i < m_ValuesWithComponent.size(); ++i) {
        auto& moments = residuals[m_WindowIndices[i]];
        CBasicStatistics::moment<0>(moments) =
            CBasicStatistics::mean(m_ValuesWithComponent[i]) -
            m_Periods[0].value(m_Components[0], i);
    }
}

void CTimeSeriesTestForSeasonality::finalizeHypotheses(THypothesisStatsVec& hypotheses) const {

    // Ensure that we only keep "false" hypotheses which are needed because they
    // are the best hypothesis for their time window.

    for (std::size_t i = 0, removedCount = 0; i < hypotheses.size();
         i += (removedCount > 0 ? 0 : 1)) {
        removedCount = 0;
        const auto& hypothesis = hypotheses[i];
        if (hypothesis.s_Period.windowed()) {
            const auto& cutoff = std::max_element(
                hypotheses.begin(), hypotheses.end(),
                [&](const SHypothesisStats& lhs, const SHypothesisStats& rhs) {
                    return COrderings::lexicographical_compare(
                        lhs.s_Period.s_Window == hypothesis.s_Period.s_Window, lhs.s_Truth,
                        rhs.s_Period.s_Window == hypothesis.s_Period.s_Window, rhs.s_Truth);
                });
            removedCount = hypotheses.size();
            hypotheses.erase(
                std::remove_if(hypotheses.begin(), hypotheses.end(),
                               [&](const SHypothesisStats& candidate) {
                                   return candidate.s_Period.s_Window ==
                                              hypothesis.s_Period.s_Window &&
                                          candidate.s_Truth.boolean() == false &&
                                          candidate.s_Truth < cutoff->s_Truth;
                               }),
                hypotheses.end());
            removedCount -= hypotheses.size();
        }
    }

    m_Periods.clear();
    for (auto& hypothesis : hypotheses) {
        m_Periods.push_back(hypothesis.s_Period);
    }
    m_ValuesWithComponent = m_Values;
    CSignal::fitSeasonalComponentsRobust(m_Periods, m_OutlierFraction,
                                         m_ValuesWithComponent, m_Components);

    for (std::size_t i = 0; i < hypotheses.size(); ++i) {

        hypotheses[i].s_InitialValues = m_ValuesWithComponent;

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

void CTimeSeriesTestForSeasonality::testExplainedVariance(const TVarianceStats& H0,
                                                          SHypothesisStats& hypothesis) const {

    auto H1 = CSignal::residualVarianceStats(m_ValuesWithComponent, m_Periods, m_Components);

    hypothesis.s_FractionNotMissing = static_cast<double>(H1.s_NumberParameters) /
                                      static_cast<double>(m_Components[0].size());
    hypothesis.s_ResidualVariance = H1.s_ResidualVariance;
    hypothesis.s_ExplainedVariance = CBasicStatistics::maximumLikelihoodVariance(
        std::accumulate(m_Components[0].begin(), m_Components[0].end(),
                        TMeanVarAccumulator{}, [](auto result, const auto& value) {
                            if (CBasicStatistics::count(value) > 0.0) {
                                result.add(CBasicStatistics::mean(value));
                            }
                            return result;
                        }));
    hypothesis.s_NumberParametersToExplainVariance = H1.s_NumberParameters;
    hypothesis.s_ExplainedVariancePValue = CSignal::rightTailFTest(H0, H1);
    LOG_TRACE(<< "fraction not missing = " << hypothesis.s_FractionNotMissing);
    LOG_TRACE(<< H1.print() << " vs " << H0.print());
    LOG_TRACE(<< "p-value = " << hypothesis.s_ExplainedVariancePValue);
}

void CTimeSeriesTestForSeasonality::testAutocorrelation(SHypothesisStats& hypothesis) const {
    CSignal::TFloatMeanAccumulatorCRng valuesToTestAutocorrelation{
        m_ValuesWithComponent, 0,
        CIntegerTools::floor(m_ValuesWithComponent.size(), m_Periods[0].period())};

    double autocorrelations[]{
        CSignal::cyclicAutocorrelation(m_Periods[0], valuesToTestAutocorrelation),
        CSignal::cyclicAutocorrelation( // Not reweighting outliers
            m_Periods[0], valuesToTestAutocorrelation,
            [](const TFloatMeanAccumulator& value) {
                return CBasicStatistics::mean(value);
            },
            [](const TFloatMeanAccumulator&) { return 1.0; }),
        CSignal::cyclicAutocorrelation( // Absolute values
            m_Periods[0], valuesToTestAutocorrelation,
            [](const TFloatMeanAccumulator& value) {
                return std::fabs(CBasicStatistics::mean(value));
            }),
        CSignal::cyclicAutocorrelation( // Not reweighting outliers and absolute values
            m_Periods[0], valuesToTestAutocorrelation,
            [](const TFloatMeanAccumulator& value) {
                return std::fabs(CBasicStatistics::mean(value));
            },
            [](const TFloatMeanAccumulator&) { return 1.0; })};
    LOG_TRACE(<< "autocorrelations = " << core::CContainerPrinter::print(autocorrelations));

    hypothesis.s_Autocorrelation = *std::max_element(
        std::begin(autocorrelations), std::begin(autocorrelations) + 2);
    hypothesis.s_AutocorrelationUpperBound =
        *std::max_element(std::begin(autocorrelations), std::end(autocorrelations));
    LOG_TRACE(<< "autocorrelation = " << hypothesis.s_Autocorrelation
              << ", autocorrelation upper bound = " << hypothesis.s_AutocorrelationUpperBound);
}

void CTimeSeriesTestForSeasonality::testAmplitude(SHypothesisStats& hypothesis) const {

    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

    hypothesis.s_SeenSufficientDataToTestAmplitude = CMinAmplitude::seenSufficientDataToTestAmplitude(
        this->observedRange(m_ValuesWithComponent), m_Periods[0].s_Period);
    if (hypothesis.s_SeenSufficientDataToTestAmplitude == false) {
        return;
    }

    double level{CBasicStatistics::mean(std::accumulate(
        m_ValuesWithComponent.begin(), m_ValuesWithComponent.end(), TMeanAccumulator{},
        [](TMeanAccumulator partialLevel, const TFloatMeanAccumulator& value) {
            partialLevel.add(CBasicStatistics::mean(value), CBasicStatistics::count(value));
            return partialLevel;
        }))};

    m_Amplitudes.assign(m_Periods[0].period(), {m_ValuesWithComponent.size(),
                                                hypothesis.s_MeanNumberRepeats, level});
    for (std::size_t i = 0; i < m_ValuesWithComponent.size(); ++i) {
        if (m_Periods[0].contains(i)) {
            m_Amplitudes[m_Periods[0].offset(i)].add(i, m_ValuesWithComponent[i]);
        }
    }

    double pValue{1.0};
    if (hypothesis.s_ResidualVariance <= 0.0) {
        pValue = std::find_if(m_Amplitudes.begin(), m_Amplitudes.end(),
                              [](const auto& amplitude) {
                                  return amplitude.amplitude() > 0.0;
                              }) != m_Amplitudes.end()
                     ? 0.0
                     : 1.0;
    } else {
        boost::math::normal normal(0.0, std::sqrt(hypothesis.s_ResidualVariance));
        for (const auto& amplitude : m_Amplitudes) {
            if (amplitude.amplitude() >= 2.0 * boost::math::standard_deviation(normal)) {
                pValue = std::min(pValue, amplitude.significance(normal));
            }
        }
    }

    hypothesis.s_AmplitudePValue = CTools::oneMinusPowOneMinusX(
        pValue, static_cast<double>(std::count_if(
                    m_Amplitudes.begin(), m_Amplitudes.end(), [](const auto& amplitude) {
                        return amplitude.amplitude() > 0.0;
                    })));
    LOG_TRACE(<< "amplitude p-value = " << hypothesis.s_AmplitudePValue);
}

double CTimeSeriesTestForSeasonality::truncatedVariance(double outlierFraction,
                                                        const TFloatMeanAccumulatorVec& residuals) const {
    double cutoff{std::numeric_limits<double>::max()};
    std::size_t count{CSignal::countNotMissing(residuals)};
    if (outlierFraction > 0.0) {
        m_Outliers.clear();
        m_Outliers.resize(static_cast<std::size_t>(
            outlierFraction * static_cast<double>(CSignal::countNotMissing(residuals)) + 0.5));
        for (const auto& value : residuals) {
            if (CBasicStatistics::count(value) > 0.0) {
                m_Outliers.add(std::fabs(CBasicStatistics::mean(value)));
            }
        }
        cutoff = m_Outliers.biggest();
        count -= m_Outliers.count();
    }
    LOG_TRACE(<< "cutoff = " << cutoff << ", count = " << count);

    TMeanVarAccumulator moments;
    for (const auto& value : residuals) {
        if (CBasicStatistics::count(value) > 0.0 &&
            std::fabs(CBasicStatistics::mean(value)) < cutoff) {
            moments.add(CBasicStatistics::mean(value));
        }
    }
    if (m_OutlierFraction > 0.0) {
        moments.add(cutoff, static_cast<double>(count) - CBasicStatistics::count(moments));
    }
    return CBasicStatistics::maximumLikelihoodVariance(moments);
};

bool CTimeSeriesTestForSeasonality::alreadyModelled(const TSeasonalComponentVec& periods) const {
    for (const auto& period : periods) {
        if (std::find(m_ModelledPeriods.begin(), m_ModelledPeriods.end(), period) ==
            m_ModelledPeriods.end()) {
            return false;
        }
    }
    return true;
}

bool CTimeSeriesTestForSeasonality::onlyDiurnal(const TSeasonalComponentVec& periods) const {
    return std::find_if(periods.begin(), periods.end(), [this](const auto& period) {
               return this->isDiurnal(period) == false;
           }) == periods.end();
}

void CTimeSeriesTestForSeasonality::removeIfInsufficientData(TSeasonalComponentVec& periods) const {
    periods.erase(std::remove_if(periods.begin(), periods.end(),
                                 [this](const auto& period) {
                                     return this->seenSufficientData(period) == false;
                                 }),
                  periods.end());
}

bool CTimeSeriesTestForSeasonality::isDiurnal(std::size_t period) const {
    core_t::TTime periodInSeconds{static_cast<core_t::TTime>(period) * m_BucketLength};
    return periodInSeconds == core::constants::DAY ||
           periodInSeconds == core::constants::WEEK ||
           periodInSeconds == core::constants::YEAR;
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
    return 2 * period.s_WindowRepeat <= this->observedRange(m_Values);
}

bool CTimeSeriesTestForSeasonality::seenSufficientDataToTestForTradingDayDecomposition() const {
    return 2 * this->week() <= this->observedRange(m_Values);
}

double CTimeSeriesTestForSeasonality::precedence() const {
    return static_cast<double>(this->observedRange(m_Values));
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

std::size_t CTimeSeriesTestForSeasonality::observedRange(const TFloatMeanAccumulatorVec& values) const {
    int begin{0};
    int end{static_cast<int>(values.size())};
    int size{static_cast<int>(values.size())};
    for (/**/; begin < size && CBasicStatistics::count(values[begin]) == 0.0; ++begin) {
    }
    for (/**/; end > begin && CBasicStatistics::count(values[end - 1]) == 0.0; --end) {
    }
    return static_cast<std::size_t>(end - begin);
}

std::size_t CTimeSeriesTestForSeasonality::CMinAmplitude::seenSufficientDataToTestAmplitude(
    std::size_t range,
    std::size_t period) {
    return range >= MINIMUM_REPEATS * period;
}

void CTimeSeriesTestForSeasonality::CMinAmplitude::add(std::size_t index,
                                                       const TFloatMeanAccumulator& value) {
    if (CBasicStatistics::count(value) > 0.0) {
        std::size_t bucket{index / m_BucketLength};
        if (bucket < m_BucketAmplitudes.size()) {
            ++m_Count;
            m_BucketAmplitudes[bucket].add(CBasicStatistics::mean(value) - m_Level);
        }
    }
}

double CTimeSeriesTestForSeasonality::CMinAmplitude::amplitude() const {
    double amplitudes[]{INF, INF};
    for (const auto& bucket : m_BucketAmplitudes) {
        if (bucket.initialized()) {
            amplitudes[0] = std::min(amplitudes[0], std::max(-bucket.min(), 0.0));
            amplitudes[1] = std::min(amplitudes[1], std::max(bucket.max(), 0.0));
        } else {
            amplitudes[0] = amplitudes[1] = 0.0;
            break;
        }
    }
    return std::max(amplitudes[0], amplitudes[1]);
}

double CTimeSeriesTestForSeasonality::CMinAmplitude::significance(const boost::math::normal& normal) const {
    double amplitude{this->amplitude()};
    if (amplitude == 0.0) {
        return 1.0;
    }
    double twoTailPValue{2.0 * CTools::safeCdf(normal, -amplitude)};
    if (twoTailPValue == 0.0) {
        return 0.0;
    }
    boost::math::binomial binomial(static_cast<double>(m_Count), twoTailPValue);
    return CTools::safeCdfComplement(
        binomial, static_cast<double>(m_BucketAmplitudes.size()) - 1.0);
}

std::string CTimeSeriesTestForSeasonality::CMinAmplitude::print() const {
    auto appendBucket = [](const TMinMaxAccumulator& bucket, std::ostringstream& result) {
        if (bucket.initialized()) {
            result << "(" << bucket.min() << "," << bucket.max() << ")";
        } else {
            result << "-";
        }
    };
    std::ostringstream result;
    result << "count = " << m_Count << " [";
    appendBucket(m_BucketAmplitudes[0], result);
    for (std::size_t i = 1; i < m_BucketAmplitudes.size(); ++i) {
        result << ", ";
        appendBucket(m_BucketAmplitudes[i], result);
    }
    result << "]";
    return result.str();
}

CFuzzyTruthValue CTimeSeriesTestForSeasonality::SHypothesisStats::testVariance(
    const CTimeSeriesTestForSeasonality& params) const {
    double repeatsPerSegment{
        s_MeanNumberRepeats /
        static_cast<double>(std::max(s_NumberTrendSegments, std::size_t{1}) +
                            s_NumberScaleSegments - 1)};
    double minimumRepeatsPerSegment{params.m_MinimumRepeatsPerSegmentToTestVariance};
    double mediumAutocorrelation{params.m_MediumAutocorrelation};
    double highAutocorrelation{params.m_HighAutocorrelation};
    double logPValue{std::log(s_ExplainedVariancePValue)};
    double logSignificantPValue{std::log(params.m_SignificantPValue)};
    double logVerySignificantPValue{std::log(params.m_VerySignificantPValue)};
    LOG_TRACE(<< "repeats per segment = " << repeatsPerSegment);
    return fuzzyGreaterThan(repeatsPerSegment / minimumRepeatsPerSegment, 1.0, 0.2) &&
           fuzzyGreaterThan(std::min(repeatsPerSegment / 2.0, 1.0), 1.0, 0.1) &&
           fuzzyGreaterThan(s_FractionNotMissing, 1.0, 0.5) &&
           fuzzyGreaterThan(logPValue / logSignificantPValue, 1.0, 0.1) &&
           fuzzyGreaterThan(std::max(logPValue / logVerySignificantPValue, 1.0), 1.0, 0.1) &&
           fuzzyGreaterThan(s_Autocorrelation / mediumAutocorrelation, 1.0, 0.2) &&
           fuzzyGreaterThan(std::max(s_Autocorrelation / highAutocorrelation, 1.0), 1.0, 0.1);
}

CFuzzyTruthValue CTimeSeriesTestForSeasonality::SHypothesisStats::testAmplitude(
    const CTimeSeriesTestForSeasonality& params) const {
    if (s_SeenSufficientDataToTestAmplitude == false) {
        return CFuzzyTruthValue::OR_UNDETERMINED;
    }
    double repeatsPerSegment{
        s_MeanNumberRepeats /
        static_cast<double>(std::max(s_NumberTrendSegments, std::size_t{1}) +
                            s_NumberScaleSegments - 1)};
    double minimumRepeatsPerSegment{params.m_MinimumRepeatsPerSegmentToTestAmplitude};
    double lowAutocorrelation{params.m_LowAutocorrelation};
    double autocorrelation{s_AutocorrelationUpperBound};
    double logPValue{std::log(s_AmplitudePValue)};
    double logSignificantPValue{std::log(params.m_SignificantPValue)};
    double logVerySignificantPValue{std::log(params.m_VerySignificantPValue)};
    LOG_TRACE(<< "repeats per segment = " << repeatsPerSegment);
    return fuzzyGreaterThan(repeatsPerSegment / minimumRepeatsPerSegment, 1.0, 0.2) &&
           fuzzyGreaterThan(autocorrelation / lowAutocorrelation, 1.0, 0.2) &&
           fuzzyGreaterThan(logPValue / logSignificantPValue, 1.0, 0.1) &&
           fuzzyGreaterThan(std::max(logPValue / logVerySignificantPValue, 1.0), 1.0, 0.1);
}

std::string CTimeSeriesTestForSeasonality::SHypothesisStats::print() const {
    return s_Period.print();
}

bool CTimeSeriesTestForSeasonality::SModel::seasonal(std::size_t period) const {
    return std::find_if(s_Hypotheses.begin(), s_Hypotheses.end(), [&](const auto& hypothesis) {
               return hypothesis.s_Period.s_Period == period;
           }) != s_Hypotheses.end();
}

bool CTimeSeriesTestForSeasonality::SModel::isNull(std::size_t numberValues) const {
    return (s_Hypotheses.empty() || s_AlreadyModelled) &&
           this->degreesFreedom(numberValues) > 0.0;
}

bool CTimeSeriesTestForSeasonality::SModel::isAlternative(std::size_t numberValues) const {
    return s_Hypotheses.size() > 0 && s_AlreadyModelled == false &&
           this->degreesFreedom(numberValues) > 0.0;
}

double CTimeSeriesTestForSeasonality::SModel::degreesFreedom(std::size_t numberValues) const {
    return static_cast<double>(numberValues) - this->numberParameters() - 1.0;
}

double CTimeSeriesTestForSeasonality::SModel::numberParameters() const {
    return static_cast<double>(std::accumulate(
        s_Hypotheses.begin(), s_Hypotheses.end(), s_NumberTrendParameters + 1,
        [](std::size_t result, const auto& component) {
            return result + component.s_NumberParametersToExplainVariance +
                   (component.s_NumberScaleSegments - 1);
        }));
}

double CTimeSeriesTestForSeasonality::SModel::meanRepeats() const {
    double result{std::numeric_limits<double>::max()};
    for (const auto& hypothesis : s_Hypotheses) {
        result = std::min(result, hypothesis.s_MeanNumberRepeats);
    }
    return result;
}

double CTimeSeriesTestForSeasonality::SModel::explainedVariancePerParameter(double explainedVariance) const {
    double result{0.0};
    double Z{0.0};
    for (const auto& hypothesis : s_Hypotheses) {
        result += hypothesis.s_ExplainedVariance * explainedVariance /
                  static_cast<double>(hypothesis.s_NumberParametersToExplainVariance);
        Z += hypothesis.s_ExplainedVariance;
    }
    return std::max(result / Z, std::numeric_limits<double>::min());
}
}
}
