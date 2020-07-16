/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CTimeSeriesTestForSeasonality.h>

#include <core/Constants.h>

#include <maths/CBasicStatistics.h>
#include <maths/CFuzzyLogic.h>
#include <maths/CIntegerTools.h>
#include <maths/COrderings.h>
#include <maths/CSetTools.h>
#include <maths/CSignal.h>
#include <maths/CStatisticalTests.h>
#include <maths/CTimeSeriesSegmentation.h>
#include <maths/CTools.h>
#include <maths/MathsTypes.h>

#include <boost/math/distributions/binomial.hpp>
#include <boost/math/distributions/normal.hpp>

#include <algorithm>

namespace ml {
namespace maths {
namespace {
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
        return std::max(static_cast<std::size_t>(std::ceil(0.5 * meanRepeats)),
                        std::size_t{4});
    }
    std::size_t capacity() const { return m_Min.capacity(); }
    std::size_t count() const { return m_Min.count(); }

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

SSeasonalComponentSummary::SSeasonalComponentSummary(const std::string& description,
                                                     bool diurnal,
                                                     bool piecewiseScaled,
                                                     std::size_t startOfWeek,
                                                     std::size_t period,
                                                     const TSizeSizePr& window)
    : s_Description{description}, s_Diurnal{diurnal}, s_PiecewiseScaled{piecewiseScaled},
      s_StartOfWeek{startOfWeek}, s_Period{period}, s_Window{window} {
}

bool SSeasonalComponentSummary::operator==(const SSeasonalComponentSummary& other) const {
    // TODO
    return true;
}

CSeasonalTime* SSeasonalComponentSummary::seasonalTime() const {
    // TODO
    return nullptr;
}

CTimeSeriesTestForSeasonality::CTimeSeriesTestForSeasonality(core_t::TTime startTime,
                                                             core_t::TTime bucketLength,
                                                             TFloatMeanAccumulatorVec values,
                                                             double outlierFraction)
    : m_StartTime{startTime}, m_BucketLength{bucketLength},
      m_OutlierFraction{outlierFraction}, m_Values{std::move(values)} {
}

CSeasonalHypotheses CTimeSeriesTestForSeasonality::test() {

    using TRemoveTrend = std::function<bool(TFloatMeanAccumulatorVec&)>;

    TSizeVec trendSegments{TSegmentation::piecewiseLinear(m_Values)};

    TSizeVec hypothesisTrendSegments;
    std::size_t numberTrendSegments{1};

    TRemoveTrend trendModels[]{[&](TFloatMeanAccumulatorVec&) {
                                   hypothesisTrendSegments.assign(0, m_Values.size());
                                   numberTrendSegments = 1;
                                   return true;
                               },
                               [&](TFloatMeanAccumulatorVec& values) {
                                   hypothesisTrendSegments.assign(0, m_Values.size());
                                   numberTrendSegments = 1;
                                   CSignal::removeLinearTrend(values);
                                   return true;
                               },
                               [&](TFloatMeanAccumulatorVec& values) {
                                   hypothesisTrendSegments = trendSegments;
                                   numberTrendSegments = trendSegments.size() - 1;
                                   if (trendSegments.size() > 2) {
                                       values = TSegmentation::removePiecewiseLinear(
                                           std::move(values), trendSegments);
                                       return true;
                                   }
                                   return false;
                               }};

    TFloatMeanAccumulatorVec valuesToTest;
    TSeasonalComponentVec periods;

    TSizeVecHypothesisStatsVecPrVec hypotheses;
    hypotheses.reserve(3);

    for (const auto& removeTrend : trendModels) {

        valuesToTest.assign(m_Values.begin(), m_Values.end());

        if (removeTrend(valuesToTest)) {
            periods = CSignal::seasonalDecomposition(
                valuesToTest, m_OutlierFraction, this->week(), [this](std::size_t period) {
                    return this->isDiurnal(period) ? 1.1 : 1.0;
                });
            this->appendDiurnalComponents(valuesToTest, periods);
            periods.erase(std::remove_if(periods.begin(), periods.end(),
                                         [this](const auto& period) {
                                             return this->seenSufficientData(
                                                        period.s_WindowRepeat) == false;
                                         }),
                          periods.end());

            hypotheses.emplace_back(hypothesisTrendSegments,
                                    this->testDecomposition(valuesToTest, numberTrendSegments, periods));
        }
    }

    return this->select(hypotheses);
}

CSeasonalHypotheses CTimeSeriesTestForSeasonality::select(const THypothesisStatsVecVec& hypotheses) const {
    // Choose the trend hypothesis with the highest mean truth value.

    for (const auto& hypothesis: hypotheses) {
        dou
        if (hypotheses.size() > 0) {

        }
    }
}

void CTimeSeriesTestForSeasonality::truth(const SHypothesisStats& hypothesis) const {
    hypothesis.s_TruthValue =
        fuzzyGreaterThan(hypothesis.s_MeanNumberRepeats /
                             static_cast<double>(hypothesis.s_NumberTrendSegments +
                                                 hypothesis.s_ScaleSegments.size()) /
                             m_MinimumNumberRepeatsPerSegment,
                         1.0, 0.2) && // 20% margin
        fuzzyGreaterThan(hypothesis.s_Autocorrelation / m_MinimumAutocorrelation,
                         1.0, 0.1) && // 10% margin
        (fuzzyGreaterThan(hypothesis.s_ExplainedVariance / m_MinimumExplainedVariance,
                          1.0, 0.1) && // 10% margin
             fuzzyLessThan(std::log(hypothesis.s_ExplainedVariancePValue / m_MaximumExplainedVariancePValue),
                           0.0, 0.1) || // 10% margin
         fuzzyLessThan(std::log(hypothesis.s_AmplitudePValue / m_MaximumAmplitudePValue),
                       0.0, 0.1)); // 10% margin
}

CTimeSeriesTestForSeasonality::THypothesisStatsVec
CTimeSeriesTestForSeasonality::testDecomposition(TFloatMeanAccumulatorVec& valuesToTest,
                                                 std::size_t numberTrendSegments,
                                                 const TSeasonalComponentVec& periods) const {

    using TComputeScaling =
        std::function<bool(const TFloatMeanAccumulatorVec&, const TSeasonalComponent&)>;

    CSignal::TMeanAccumulatorVec1Vec components;
    CSignal::fitSeasonalComponentsRobust(periods, m_OutlierFraction, valuesToTest, components);

    TFloatMeanAccumulatorVec valuesToTestComponent;

    auto removeComponentPredictions = [&](std::size_t i, TFloatMeanAccumulatorVec& values) {
        for (std::size_t j = 0; j < values.size(); ++j) {
            if (periods[j].contains(j)) {
                CBasicStatistics::moment<0>(values[j]) -=
                    CBasicStatistics::mean(components[i][periods[j].offset(j)]);
            }
        }
    };

    TSizeVec scaleSegments;

    TComputeScaling scalings[]{
        [&](const TFloatMeanAccumulatorVec&, const CSignal::SSeasonalComponentSummary&) {
            scaleSegments.assign({0, valuesToTest.size()});
            return true;
        },
        [&](const TFloatMeanAccumulatorVec& values,
            const CSignal::SSeasonalComponentSummary& period) {
            scaleSegments.assign({0, valuesToTest.size()});
            scaleSegments = TSegmentation::piecewiseLinearScaledSeasonal(
                values, period.period());
            if (scaleSegments.size() > 2) {
                TSegmentation::meanScalePiecewiseLinearScaledSeasonal(
                    values, period.period(), scaleSegments);
                return true;
            }
            return false;
        }};

    THypothesisStatsVec result;
    result.reserve(std::size(scalings) * components.size());

    for (std::size_t i = 0; i < components.size(); ++i) {

        // Precondition by removing all other components.
        valuesToTestComponent.assign(valuesToTest.begin(), valuesToTest.end());
        for (std::size_t j = 0; j < i; ++j) {
            removeComponentPredictions(j, valuesToTestComponent);
        }
        for (std::size_t j = i + 1; j < components.size(); ++j) {
            removeComponentPredictions(j, valuesToTestComponent);
        }
        CSignal::restrictTo(periods[i], valuesToTestComponent);
        TSeasonalComponent period{CSignal::seasonalComponentSummary(periods[i].period())};

        for (const auto& scale : scalings) {
            if (scale(valuesToTestComponent, period)) {
                SHypothesisStats hypothesis;
                hypothesis.s_NumberTrendSegments = numberTrendSegments;
                hypothesis.s_HasWeekend = periods[i].windowed();
                hypothesis.s_StartOfWeek = periods[i].s_StartOfWeek;
                hypothesis.s_ScaleSegments = scaleSegments;
                hypothesis.s_MeanNumberRepeats =
                    CSignal::meanNumberRepeatedValues(valuesToTestComponent, period);

                this->testExplainedVariance(valuesToTestComponent, period, hypothesis);
                this->testAutocorrelation(valuesToTestComponent, period, hypothesis);
                this->testAmplitude(valuesToTestComponent, period, hypothesis);
                this->truth(hypothesis);

                if (hypothesis.s_TruthValue.boolean()) {
                    result.push_back(hypothesis);
                }
            }
        }
    }

    return result;
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
        CSignal::varianceAtPercentile(10.0, variances[0], degreesFreedom[0]) /
        CSignal::varianceAtPercentile(90.0, variances[1], degreesFreedom[1]);
    hypothesis.s_ExplainedVariancePValue = CStatisticalTests::rightTailFTest(
        variances[0] / variances[1], degreesFreedom[0], degreesFreedom[1]);
}

void CTimeSeriesTestForSeasonality::testAutocorrelation(const TFloatMeanAccumulatorVec& valuesToTest,
                                                        const TSeasonalComponent& period,
                                                        SHypothesisStats& hypothesis) const {
    CSignal::TFloatMeanAccumulatorCRng valuesToTestAutocorrelation{
        valuesToTest, 0, CIntegerTools::floor(valuesToTest.size(), period.period())};
    std::size_t n{CSignal::countNotMissing(valuesToTestAutocorrelation)};
    hypothesis.s_Autocorrelation = CSignal::autocorrelationAtPercentile(
        10.0, CSignal::cyclicAutocorrelation(period.period(), valuesToTestAutocorrelation),
        static_cast<double>(n));
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
    hypothesis.s_AmplitudePValue = pvalue;
}

void CTimeSeriesTestForSeasonality::appendDiurnalComponents(const TFloatMeanAccumulatorVec& valuesToTest,
                                                            TSeasonalComponentVec& periods) const {
    auto has = [&](std::size_t period) {
        return std::find_if(periods.begin(), periods.end(), [&period](const auto& entry) {
                   return entry.period() == period;
               }) == periods.end();
    };
    auto hasTradingDayDecomposition = [&] {
        return std::find_if(periods.begin(), periods.end(), [](const auto& entry) {
                   return entry.windowed();
               }) == periods.end();
    };

    if (has(this->day()) == false) {
        CSignal::appendSeasonalComponentSummary(this->day(), periods);
    }

    if (has(this->week()) == false && hasTradingDayDecomposition() == false) {
        auto decomposition = CSignal::tradingDayDecomposition(
            valuesToTest, m_OutlierFraction, this->week());
        if (decomposition.empty()) {
            CSignal::appendSeasonalComponentSummary(this->week(), periods);
        } else {
            periods.insert(periods.end(), decomposition.begin(), decomposition.end());
        }
    }

    if (has(this->year()) == false) {
        CSignal::appendSeasonalComponentSummary(this->year(), periods);
    }
}

bool CTimeSeriesTestForSeasonality::isDiurnal(std::size_t period) const {
    return period == this->day() || period == this->week() || period == this->year();
}

bool CTimeSeriesTestForSeasonality::dividesDiurnal(std::size_t period) const {
    return (this->day() % period) == 0 ||  // divides daily
           (this->week() % period) == 0 || // divides weekly
           (this->year() % period) == 0;   // divides yearly
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

bool CTimeSeriesTestForSeasonality::seenSufficientData(std::size_t period) const {
    return 2 * period >= this->observedRange();
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

std::string CTimeSeriesTestForSeasonality::describe(std::size_t period) const {
    if (period == this->day()) {
        return "daily";
    }
    if (period == this->week()) {
        return "weekly";
    }
    if (period == this->year()) {
        return "annual";
    }
    return "period " + std::to_string(period);
}
}
}
