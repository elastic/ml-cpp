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
#include <maths/CLeastSquaresOnlineRegression.h>
#include <maths/CLeastSquaresOnlineRegressionDetail.h>
#include <maths/COrderings.h>
#include <maths/CSeasonalTime.h>
#include <maths/CSetTools.h>
#include <maths/CSignal.h>
#include <maths/CStatisticalTests.h>
#include <maths/CTimeSeriesSegmentation.h>
#include <maths/CTools.h>
#include <maths/MathsTypes.h>

#include <boost/iterator/counting_iterator.hpp>

#include <algorithm>
#include <cstddef>
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

bool almostDivisor(std::size_t i, std::size_t j, double eps) {
    if (i > j) {
        return false;
    }
    double diff{static_cast<double>(std::min(j % i, i - (j % i))) / static_cast<double>(j)};
    return diff < eps;
}

double calibrateTruncatedVariance(double meanRepeats) {
    double x{1.0 - 1.0 / (1.0 + meanRepeats)};
    return 1.0 / std::min(0.5457652 - 1.057418 * x + 1.536299 * x * x, 1.0);
}
}

CNewTrendSummary::CNewTrendSummary(core_t::TTime initialValuesStartTime,
                                   core_t::TTime bucketLength,
                                   TFloatMeanAccumulatorVec initialValues)
    : m_InitialValuesStartTime{initialValuesStartTime},
      m_BucketLength{bucketLength}, m_InitialValues{std::move(initialValues)} {
}

core_t::TTime CNewTrendSummary::initialValuesStartTime() const {
    return m_InitialValuesStartTime;
}

core_t::TTime CNewTrendSummary::initialValuesEndTime() const {
    return m_InitialValuesStartTime +
           static_cast<core_t::TTime>(m_InitialValues.size()) * m_BucketLength;
}

core_t::TTime CNewTrendSummary::bucketLength() const {
    return m_BucketLength;
}

const CNewTrendSummary::TFloatMeanAccumulatorVec& CNewTrendSummary::initialValues() const {
    return m_InitialValues;
}

CNewSeasonalComponentSummary::CNewSeasonalComponentSummary(
    std::string annotationText,
    const TSeasonalComponent& period,
    std::size_t size,
    EPeriodDescriptor periodDescriptor,
    core_t::TTime initialValuesStartTime,
    core_t::TTime bucketStartTime,
    core_t::TTime bucketLength,
    TFloatMeanAccumulatorVec initialValues)
    : m_AnnotationText{std::move(annotationText)}, m_Period{period}, m_Size{size},
      m_PeriodDescriptor{periodDescriptor},
      m_InitialValuesStartTime{initialValuesStartTime}, m_BucketStartTime{bucketStartTime},
      m_BucketLength{bucketLength}, m_InitialValues{std::move(initialValues)} {
}

const std::string& CNewSeasonalComponentSummary::annotationText() const {
    return m_AnnotationText;
}

std::size_t CNewSeasonalComponentSummary::size() const {
    return m_Size;
}

CNewSeasonalComponentSummary::TSeasonalTimeUPtr
CNewSeasonalComponentSummary::seasonalTime() const {
    auto interval = [this](std::size_t i) {
        return m_BucketLength * static_cast<core_t::TTime>(i);
    };
    if ((m_PeriodDescriptor & E_Diurnal) != 0) {
        core_t::TTime period{[this] {
            if (m_PeriodDescriptor == E_Day) {
                return core::constants::DAY;
            }
            if (m_PeriodDescriptor == E_Week) {
                return core::constants::WEEK;
            }
            LOG_ERROR(<< "Unexpected descriptor '" << m_PeriodDescriptor << "'");
            return core::constants::DAY;
        }()};
        core_t::TTime windowStart{[&] {
            core_t::TTime times[]{0, 2 * core::constants::DAY};
            core_t::TTime start{interval(m_Period.s_Window.first)};
            return *std::min_element(std::begin(times), std::end(times),
                                     [&](const auto& lhs, const auto& rhs) {
                                         return std::abs(lhs - start) <
                                                std::abs(rhs - start);
                                     });
        }()};
        core_t::TTime windowEnd{[&] {
            core_t::TTime times[]{2 * core::constants::DAY, 7 * core::constants::DAY};
            core_t::TTime start{interval(m_Period.s_Window.second)};
            return *std::min_element(std::begin(times), std::end(times),
                                     [&](const auto& lhs, const auto& rhs) {
                                         return std::abs(lhs - start) <
                                                std::abs(rhs - start);
                                     });
        }()};
        if (m_Period.windowed()) {
            core_t::TTime startOfWeek{(m_BucketStartTime + interval(m_Period.s_StartOfWeek)) %
                                      core::constants::WEEK};
            return std::make_unique<CDiurnalTime>(startOfWeek, windowStart, windowEnd, period);
        }
        return std::make_unique<CDiurnalTime>(0, 0, core::constants::WEEK, period);
    }
    if (m_PeriodDescriptor == E_Year) {
        return std::make_unique<CGeneralPeriodTime>(core::constants::YEAR);
    }
    return std::make_unique<CGeneralPeriodTime>(interval(m_Period.s_Period));
}

core_t::TTime CNewSeasonalComponentSummary::initialValuesStartTime() const {
    return m_InitialValuesStartTime;
}

core_t::TTime CNewSeasonalComponentSummary::initialValuesEndTime() const {
    return m_InitialValuesStartTime +
           static_cast<core_t::TTime>(m_InitialValues.size()) * m_BucketLength;
}

const CNewSeasonalComponentSummary::TFloatMeanAccumulatorVec&
CNewSeasonalComponentSummary::initialValues() const {
    return m_InitialValues;
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

void CSeasonalDecomposition::removeModelled() {
    m_RemoveModelled = true;
}

void CSeasonalDecomposition::add(CNewTrendSummary trend) {
    m_Trend = std::move(trend);
}

void CSeasonalDecomposition::add(std::string annotationText,
                                 const TSeasonalComponent& period,
                                 std::size_t size,
                                 TPeriodDescriptor periodDescriptor,
                                 core_t::TTime initialValuesStartTime,
                                 core_t::TTime bucketStartTime,
                                 core_t::TTime bucketLength,
                                 TFloatMeanAccumulatorVec initialValues) {
    m_Seasonal.emplace_back(std::move(annotationText), period, size,
                            periodDescriptor, initialValuesStartTime, bucketStartTime,
                            bucketLength, std::move(initialValues));
}

void CSeasonalDecomposition::add(TBoolVec seasonalToRemoveMask) {
    m_SeasonalToRemoveMask = std::move(seasonalToRemoveMask);
}

bool CSeasonalDecomposition::componentsChanged() const {
    return m_Seasonal.size() > 0 || m_RemoveModelled;
}

const CNewTrendSummary* CSeasonalDecomposition::trend() const {
    return m_Trend != boost::none ? &(*m_Trend) : nullptr;
}

const CSeasonalDecomposition::TNewSeasonalComponentVec& CSeasonalDecomposition::seasonal() const {
    return m_Seasonal;
}

const CSeasonalDecomposition::TBoolVec& CSeasonalDecomposition::seasonalToRemoveMask() const {
    return m_SeasonalToRemoveMask;
}

std::string CSeasonalDecomposition::print() const {
    return core::CContainerPrinter::print(m_Seasonal);
}

CTimeSeriesTestForSeasonality::CTimeSeriesTestForSeasonality(core_t::TTime valuesStartTime,
                                                             core_t::TTime bucketStartTime,
                                                             core_t::TTime bucketLength,
                                                             TFloatMeanAccumulatorVec values,
                                                             double outlierFraction)
    : m_ValuesStartTime{valuesStartTime}, m_BucketStartTime{bucketStartTime}, m_BucketLength{bucketLength},
      m_OutlierFraction{outlierFraction}, m_Values{std::move(values)},
      m_Outliers{static_cast<std::size_t>(
          outlierFraction * static_cast<double>(CSignal::countNotMissing(m_Values)) + 0.5)} {
    TMeanVarAccumulator moments;
    for (const auto& value : m_Values) {
        if (CBasicStatistics::count(value) > 0.0) {
            moments.add(CBasicStatistics::mean(value));
        }
    }
    m_EpsVariance = CTools::pow2(1000.0 * std::numeric_limits<double>::epsilon()) *
                    CBasicStatistics::maximumLikelihoodVariance(moments);
    LOG_TRACE(<< "eps variance = " << m_EpsVariance);
}

bool CTimeSeriesTestForSeasonality::canTestComponent(const TFloatMeanAccumulatorVec& values,
                                                     core_t::TTime bucketStartTime,
                                                     core_t::TTime bucketLength,
                                                     const CSeasonalTime& component) {
    return 10 * (component.period() % bucketLength) < component.period() &&
           canTestPeriod(values, toPeriod(bucketStartTime, bucketLength, component));
}

void CTimeSeriesTestForSeasonality::minimumPeriod(core_t::TTime minimumPeriod) {
    m_MinimumPeriod = minimumPeriod;
}

void CTimeSeriesTestForSeasonality::addModelledSeasonality(const CSeasonalTime& component,
                                                           std::size_t size) {
    auto period = toPeriod(m_BucketStartTime, m_BucketLength, component);
    m_ModelledPeriods.push_back(period);
    m_ModelledPeriodsSizes.push_back(size);
    m_ModelledPeriodsTestable.push_back(
        canTestComponent(m_Values, m_BucketStartTime, m_BucketLength, component));
    if (period.windowed()) {
        m_StartOfWeekOverride = period.s_StartOfWeek;
    }
}

void CTimeSeriesTestForSeasonality::modelledSeasonalityPredictor(const TPredictor& predictor) {
    m_ModelledPredictor = predictor;
}

CSeasonalDecomposition CTimeSeriesTestForSeasonality::decompose() const {

    LOG_TRACE(<< "decomposing " << m_Values.size()
              << " values, bucket length = " << m_BucketLength);

    // The quality of anomaly detection is sensitive to bias variance tradeoff.
    // If you care about point predictions you can get away with erring on the side
    // of overfitting slightly to avoid bias. We however care about the predicted
    // distribution and are very sensitive to prediction variance. We therefore want
    // to be careful to only model seasonality which adds significant predictive
    // value.
    //
    // This is the main entry point to decomposing a signal into its seasonal
    // components. This whole process is somewhat involved. Complications include
    // all the various data characteristics we would like to deal with automatically
    // these include: signals polluted with outliers, discontinuities in the trend,
    // discontinuities in the seasonality (i.e. scaling up or down) and missing
    // values. We'd also like very high power for detecting the most common seasonal
    // components, i.e. daily, weekly, weekday/weekend modulation and yearly
    // (predictive calendar features are handled separately). We also run this
    // continuously on various window lengths and it needs to produce a stable
    // result if the seasonality is not changing whilst still being able to detect
    // changes and to initialize new components with the right size (bias variance
    // tradeoff) and seed values.
    //
    // The high-level strategy is:
    //   1. For various trend assumptions, no trend, linear and piecewise linear,
    //   2. Test for the seasonalities we already model, common diurnal seasonality
    //      and the best decomposition based on serial autocorrelation and select
    //      those hypotheses for which there is strong evidence.
    //   3. Ensure there is good evidence that the signal is seasonal vs the best
    //      explanation for the values which only uses a trend.
    //   4. Given a set of statistically significant seasonal hypotheses choose
    //      the one which will lead to the best modelling and avoids churn.
    //
    // I found it was far more effective to consider each hypothesis separately.
    // The alternative pushes much more complexity into the step to actually fit
    // the model. For example, one might try and simultaneously fit piecewise linear
    // trend and scaled seasonality, but determining the right break points (if any)
    // in the trend together with the appropriate scaled seasonal components is a
    // non trivial estimation problem. We take an ordered approach, first fitting
    // the trend then seasonality and trying at each stage to fit significant
    // patterns. However, we also test simpler hypotheses such that there is no
    // trend at all explicitly. This is much more forgiving to the estimation process
    // since if the data doesn't have a trend not trying to fit one can easily be
    // identified as a better choice after the fact. The final selection is based
    // on a number of criterion which are geared towards our modelling techniques
    // and are described in select.

    TSizeVec trendSegments{TSegmentation::piecewiseLinear(
        m_Values, m_SignificantPValue, m_OutlierFraction)};

    TRemoveTrend removeTrendModels[]{
        [&](const TSeasonalComponentVec&, TFloatMeanAccumulatorVec& values,
            TSizeVec& modelTrendSegments) {
            LOG_TRACE(<< "no trend");
            values = m_Values;
            modelTrendSegments.clear();
            return true;
        },
        [this](const TSeasonalComponentVec& periods,
               TFloatMeanAccumulatorVec& values, TSizeVec& modelTrendSegments) {
            // We wish to solve argmin_{t,s}{sum_i w_i (y_i - t(i) - s(i))^2}.
            // Since t and s are linear functions of their parameters this is
            // convex. The following makes use of the fact that the minimizers
            // of sum_i w_i (y_i - t(i))^2 and sum_i w_i (y_i - s(i))^2 are
            // trivial to perform a sequence of line searches. In practice,
            // this converges quickly so we use a fixed number of iterations.
            LOG_TRACE(<< "quadratic trend");
            using TRegression = CLeastSquaresOnlineRegression<2, double>;

            modelTrendSegments.assign({0, values.size()});

            TRegression trend;
            auto predictor = [&](std::size_t i) {
                return trend.predict(static_cast<double>(i));
            };

            if (periods.empty()) {
                values = m_Values;
                for (std::size_t j = 0; j < values.size(); ++j) {
                    trend.add(static_cast<double>(j), CBasicStatistics::mean(values[j]),
                              CBasicStatistics::count(values[j]));
                }
                this->removePredictions(predictor, values);
                return true;
            }

            for (std::size_t i = 0; i < 2; ++i) {
                values = m_Values;
                this->removePredictions(predictor, values);
                CSignal::fitSeasonalComponents(periods, values, m_Components);
                values = m_Values;
                this->removePredictions({periods, 0, periods.size()},
                                        {m_Components, 0, m_Components.size()}, values);
                trend = TRegression{};
                for (std::size_t j = 0; j < values.size(); ++j) {
                    trend.add(static_cast<double>(j), CBasicStatistics::mean(values[j]),
                              CBasicStatistics::count(values[j]));
                }
            }
            values = m_Values;
            this->removePredictions(predictor, values);
            return true;
        },
        [&](const TSeasonalComponentVec& periods,
            TFloatMeanAccumulatorVec& values, TSizeVec& modelTrendSegments) {
            if (trendSegments.size() <= 2) {
                return false;
            }

            // We're only interested in applying segmentation when the number of
            // segments is small w.r.t. the number of repeats. In such cases we
            // can fit the trend first and suffer little loss in accuracy.
            LOG_TRACE(<< trendSegments.size() - 1 << " linear trend segments");
            modelTrendSegments = trendSegments;

            auto predictor = [&](std::size_t i) {
                double result{0.0};
                for (std::size_t j = 0; j < periods.size(); ++j) {
                    result += periods[j].value(m_Components[j], i);
                }
                return result;
            };

            values = m_Values;
            values = TSegmentation::removePiecewiseLinear(std::move(values), modelTrendSegments);

            if (periods.size() > 0) {
                CSignal::fitSeasonalComponents(periods, values, m_Components);
                values = m_Values;
                this->removePredictions(predictor, values);
                values = TSegmentation::removePiecewiseLinear(std::move(values),
                                                              modelTrendSegments);
                this->removePredictions(
                    [&](std::size_t j) { return -predictor(j); }, values);
            }

            return true;
        }};

    TModelVec decompositions;
    decompositions.reserve(8 * std::size(removeTrendModels));

    for (const auto& removeTrend : removeTrendModels) {
        this->addNotSeasonal(removeTrend, decompositions);
        this->addModelled(removeTrend, decompositions);
        this->addDiurnal(removeTrend, decompositions);
        this->addHighestAutocorrelation(removeTrend, decompositions);
    }

    return this->select(decompositions);
}

CSeasonalDecomposition CTimeSeriesTestForSeasonality::select(TModelVec& decompositions) const {

    // Choose the hypothesis which yields the best explanation of the values.

    // Sort by increasing complexity.
    std::stable_sort(decompositions.begin(), decompositions.end(),
                     [](const auto& lhs, const auto& rhs) {
                         return lhs.numberParameters() < rhs.numberParameters();
                     });

    double eps{std::numeric_limits<double>::epsilon()};
    std::size_t numberValues{CSignal::countNotMissing(m_Values)};
    std::size_t numberTruncatedValues{static_cast<std::size_t>(
        (1.0 - m_OutlierFraction) * static_cast<double>(numberValues) + 0.5)};

    auto computePValue = [&](const SModel& H0, const SModel& H1, double calibration = 1.0) {
        double degreesFreedom[]{H0.degreesFreedom(numberTruncatedValues),
                                H0.degreesFreedom(numberValues),
                                H1.degreesFreedom(numberTruncatedValues),
                                H1.degreesFreedom(numberValues)};
        double variances[]{H0.s_TruncatedResidualVariance, H0.s_ResidualVariance,
                           calibration * H1.s_TruncatedResidualVariance,
                           H1.s_ResidualVariance};
        return std::min(rightTailFTest(variances[0], variances[2],
                                       degreesFreedom[0], degreesFreedom[2]),
                        rightTailFTest(variances[1], variances[3],
                                       degreesFreedom[1], degreesFreedom[3]));
    };
    auto computePValueProxy = [&](const SModel& H0, const SModel& H1,
                                  double calibration = 1.0) {
        double degreesFreedom[]{H0.degreesFreedom(numberTruncatedValues),
                                H0.degreesFreedom(numberValues),
                                H1.degreesFreedom(numberTruncatedValues),
                                H1.degreesFreedom(numberValues)};
        double variances[]{H0.s_TruncatedResidualVariance, H0.s_ResidualVariance,
                           std::max(calibration * H1.s_TruncatedResidualVariance,
                                    eps * H0.s_TruncatedResidualVariance),
                           std::max(H1.s_ResidualVariance, eps * H0.s_ResidualVariance)};
        return std::min(
            (variances[0] * degreesFreedom[0]) / (variances[2] * degreesFreedom[2]),
            (variances[1] * degreesFreedom[1]) / (variances[3] * degreesFreedom[3]));
    };

    // Select the best null hypothesis.
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
                      << ", truncated variance = " << decompositions[H1].s_ResidualVariance
                      << ", # parameters = " << decompositions[H1].numberParameters()
                      << ", p-value = " << pValue);
            if (pValue < pValueH0) {
                std::tie(H0, pValueH0) = std::make_pair(H1, pValue);
            }
        }
    }

    // Select the best decomposition if it is a statistically significant improvement.
    std::size_t selected{decompositions.size()};
    double qualitySelected{-std::numeric_limits<double>::max()};
    double pValueAlreadyModelled{1.0};
    for (std::size_t H1 = 0; H1 < decompositions.size(); ++H1) {
        if (decompositions[H1].isAlternative(numberTruncatedValues)) {
            double calibration{calibrateTruncatedVariance(decompositions[H1].meanRepeats())};
            double pValue{computePValue(decompositions[H0], decompositions[H1], calibration)};
            double logPValue{pValue == 0.0
                                 ? std::log(std::numeric_limits<double>::min())
                                 : (pValue == 1.0 ? -std::numeric_limits<double>::min()
                                                  : std::log(pValue))};
            double logAcceptedFalsePostiveRate{std::log(m_AcceptedFalsePostiveRate)};
            double autocorrelation{decompositions[H1].autocorrelation()};
            pValueAlreadyModelled =
                std::min(pValueAlreadyModelled,
                         decompositions[H1].s_AlreadyModelled ? pValue : 1.0);
            LOG_TRACE(<< "hypothesis = "
                      << core::CContainerPrinter::print(decompositions[H1].s_Hypotheses));
            LOG_TRACE(<< "variance = " << decompositions[H1].s_ResidualVariance
                      << ", truncated variance = " << decompositions[H1].s_TruncatedResidualVariance
                      << ", # parameters = " << decompositions[H1].numberParameters()
                      << ", p-value = " << pValue);

            // It is possible that the null hypothesis uses a piecewise linear fit of
            // seasonal components in the data. In this case we accept the alternative
            // if it's autocorrelation is high and number of trend parameters is large
            // enough, i.e. O(0.05 * number values).
            if (pValue > m_AcceptedFalsePostiveRate &&
                (fuzzyGreaterThan(logPValue / logAcceptedFalsePostiveRate, 1.0, 1.0) &&
                 fuzzyGreaterThan(autocorrelation / m_HighAutocorrelation, 1.0, 0.1) &&
                 fuzzyGreaterThan(static_cast<double>(decompositions[H0].numberParameters()) /
                                      (0.05 * static_cast<double>(numberValues)),
                                  1.0, 0.2))
                        .boolean() == false) {
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
            //   4. The number of scalings and pieces in the trend model. These increase
            //      false positives so if we have an alternative similarly good hypothesis
            //      we use that one.
            //   5. The total model size. The p-value is less sensitive to model
            //      size as the window length increases. However, for both accuracy
            //      and efficiency considerations we strongly prefer smaller models.
            //   6. Some consideration of whether the components are already modelled
            //      to avoid churn on marginal decisions.
            double pValueProxy{computePValueProxy(decompositions[H0],
                                                  decompositions[H1], calibration)};
            double explainedVariance{decompositions[H0].s_ResidualVariance -
                                     decompositions[H1].s_ResidualVariance};
            double explainedTruncatedVariance{decompositions[H0].s_TruncatedResidualVariance -
                                              decompositions[H1].s_TruncatedResidualVariance};
            double explainedVariancePerParameter{
                decompositions[H1].explainedVariancePerParameter(explainedVariance)};
            double explainedTruncatedVariancePerParameter{
                decompositions[H1].explainedVariancePerParameter(explainedTruncatedVariance)};
            double numberTrendParameters{
                static_cast<double>(decompositions[H1].s_NumberTrendParameters)};
            double quality{0.7 * std::log(-logPValue) + 0.3 * std::log(pValueProxy) +
                           1.0 * std::log(explainedVariancePerParameter) +
                           1.0 * std::log(explainedTruncatedVariancePerParameter) -
                           0.5 * std::log(decompositions[H1].numberParameters()) -
                           0.3 * std::log(2.0 + decompositions[H1].numberScalings()) -
                           0.3 * std::log(2.0 + numberTrendParameters) +
                           0.2 * (decompositions[H1].s_AlreadyModelled ? 1.0 : 0.0)};
            LOG_TRACE(<< "p-value proxy = " << pValueProxy);
            LOG_TRACE(<< "explained variance = " << explainedVariance
                      << ", per param = " << explainedVariancePerParameter);
            LOG_TRACE(<< "explained truncated variance = " << explainedTruncatedVariance
                      << ", per param = " << explainedTruncatedVariancePerParameter);
            LOG_TRACE(<< "number parameters = " << decompositions[H1].numberParameters()
                      << ", modelled = " << decompositions[H1].s_AlreadyModelled);
            LOG_TRACE(<< "scalings = " << decompositions[H1].numberScalings()
                      << ", trend parameters = " << numberTrendParameters);
            LOG_TRACE(<< "quality = " << quality);

            if (quality > qualitySelected) {
                std::tie(selected, qualitySelected) = std::make_pair(H1, quality);
            }
        }
    }

    CSeasonalDecomposition result;

    LOG_TRACE(<< "p-value already modelled = " << pValueAlreadyModelled);
    std::ptrdiff_t numberModelled(m_ModelledPeriods.size());
    if (selected < decompositions.size()) {
        LOG_TRACE(<< "selected = "
                  << core::CContainerPrinter::print(decompositions[selected].s_Hypotheses));

        result.add(CNewTrendSummary{m_ValuesStartTime, m_BucketLength,
                                    std::move(decompositions[selected].s_TrendInitialValues)});
        for (auto& hypothesis : decompositions[selected].s_Hypotheses) {
            if (hypothesis.s_Model) {
                LOG_TRACE(<< "Adding " << hypothesis.s_Period.print());
                result.add(this->annotationText(hypothesis.s_Period),
                           hypothesis.s_Period, hypothesis.s_ComponentSize,
                           this->periodDescriptor(hypothesis.s_Period.s_Period),
                           m_ValuesStartTime, m_BucketStartTime, m_BucketLength,
                           std::move(hypothesis.s_InitialValues));
            }
        }
        result.add(std::move(decompositions[selected].s_RemoveComponentsMask));
    } else if (pValueAlreadyModelled > 0.3 && numberModelled > 0 &&
               std::count(m_ModelledPeriodsTestable.begin(),
                          m_ModelledPeriodsTestable.end(), true) == numberModelled) {
        // If the evidence for the current model is very weak remove it.
        result.removeModelled();
        result.add(CNewTrendSummary{m_ValuesStartTime, m_BucketLength,
                                    std::move(decompositions[H0].s_TrendInitialValues)});
        result.add(std::move(decompositions[H0].s_RemoveComponentsMask));
    }

    return result;
}

void CTimeSeriesTestForSeasonality::addNotSeasonal(const TRemoveTrend& removeTrend,
                                                   TModelVec& decompositions) const {
    if (removeTrend({}, m_ValuesMinusTrend, m_ModelTrendSegments)) {
        decompositions.emplace_back(
            this->truncatedVariance(0.0, m_ValuesMinusTrend),
            this->truncatedVariance(m_OutlierFraction, m_ValuesMinusTrend),
            m_ModelTrendSegments.empty() ? 0 : m_ModelTrendSegments.size() - 1,
            m_Values, THypothesisStatsVec{}, m_ModelledPeriodsTestable);
    }
}

void CTimeSeriesTestForSeasonality::addModelled(const TRemoveTrend& removeTrend,
                                                TModelVec& decompositions) const {
    m_CandidatePeriods = m_ModelledPeriods;
    this->removeIfNotTestable(m_CandidatePeriods);
    if (m_CandidatePeriods.size() > 0 &&
        removeTrend(m_CandidatePeriods, m_ValuesMinusTrend, m_ModelTrendSegments)) {

        // Already modelled seasonal components.
        std::stable_sort(m_CandidatePeriods.begin(), m_CandidatePeriods.end(),
                         [](const auto& lhs, const auto& rhs) {
                             return lhs.s_Period < rhs.s_Period;
                         });
        this->testAndAddDecomposition(m_CandidatePeriods, m_ModelTrendSegments,
                                      m_ValuesMinusTrend, true, decompositions);

        // Already modelled plus highest serial autocorrelation seasonal components.
        m_Periods = m_CandidatePeriods;
        CSignal::fitSeasonalComponents(m_Periods, m_ValuesMinusTrend, m_Components);
        m_TemporaryValues = m_ValuesMinusTrend;
        this->removePredictions({m_Periods, 0, m_Periods.size()},
                                {m_Components, 0, m_Components.size()}, m_TemporaryValues);
        auto diurnal = std::make_tuple(this->day(), this->week(), this->year());
        auto unit = [](std::size_t) { return 1.0; };
        for (const auto& period : CSignal::seasonalDecomposition(
                 m_TemporaryValues, m_OutlierFraction, diurnal, unit,
                 m_StartOfWeekOverride, 0.05, m_MaximumNumberComponents)) {
            if (std::find_if(m_Periods.begin(), m_Periods.end(), [&](const auto& modelledPeriod) {
                    return modelledPeriod.periodAlmostEqual(period, 0.05);
                }) == m_Periods.end()) {
                m_CandidatePeriods.push_back(period);
            }
        }
        this->removeIfNotTestable(m_CandidatePeriods);
        if (m_CandidatePeriods != m_Periods && // Have we already tested these periods?
            this->includesPermittedPeriod(m_CandidatePeriods) &&
            this->onlyDiurnal(m_CandidatePeriods) == false) {
            this->testAndAddDecomposition(m_CandidatePeriods, m_ModelTrendSegments,
                                          m_ValuesMinusTrend, false, decompositions);
        }
    }
}

void CTimeSeriesTestForSeasonality::addDiurnal(const TRemoveTrend& removeTrend,
                                               TModelVec& decompositions) const {

    // Weekday/weekend modulation removing trend after determining decomposition.
    m_TemporaryValues = m_Values;
    m_CandidatePeriods = CSignal::tradingDayDecomposition(
        m_TemporaryValues, m_OutlierFraction, this->week(), m_StartOfWeekOverride);
    m_CandidatePeriods.push_back(CSignal::seasonalComponentSummary(this->year()));
    this->removeIfNotTestable(m_CandidatePeriods);
    if (m_CandidatePeriods.size() > 0 && // Did we find candidate weekend/weekday split?
        this->includesPermittedPeriod(m_CandidatePeriods) &&
        this->alreadyModelled(m_CandidatePeriods) == false &&
        removeTrend(m_CandidatePeriods, m_ValuesMinusTrend, m_ModelTrendSegments)) {
        this->testAndAddDecomposition(m_CandidatePeriods, m_ModelTrendSegments,
                                      m_ValuesMinusTrend, false, decompositions);
    }

    // Weekday/weekend modulation removing trend before determining decomposition.
    if (removeTrend({}, m_ValuesMinusTrend, m_ModelTrendSegments)) {
        m_CandidatePeriods = CSignal::tradingDayDecomposition(
            m_TemporaryValues, m_OutlierFraction, this->week(), m_StartOfWeekOverride);
        m_CandidatePeriods.push_back(CSignal::seasonalComponentSummary(this->year()));
        this->removeIfNotTestable(m_CandidatePeriods);
        if (m_CandidatePeriods.size() > 0 && // Did we find candidate weekend/weekday split?
            m_CandidatePeriods != m_Periods && // Have we already tested these periods?
            this->includesPermittedPeriod(m_CandidatePeriods) &&
            this->alreadyModelled(m_CandidatePeriods) == false) {
            this->testAndAddDecomposition(m_CandidatePeriods, m_ModelTrendSegments,
                                          m_ValuesMinusTrend, false, decompositions);
        }
    }

    // Day, week, year.
    m_Periods = m_CandidatePeriods;
    m_CandidatePeriods.assign({CSignal::seasonalComponentSummary(this->day()),
                               CSignal::seasonalComponentSummary(this->week()),
                               CSignal::seasonalComponentSummary(this->year())});
    this->removeIfNotTestable(m_CandidatePeriods);
    if (m_CandidatePeriods.size() > 0 && // Is there sufficient data?
        m_CandidatePeriods != m_Periods && // Have we already tested these periods?
        this->includesPermittedPeriod(m_CandidatePeriods) &&
        this->alreadyModelled(m_CandidatePeriods) == false &&
        removeTrend(m_CandidatePeriods, m_ValuesMinusTrend, m_ModelTrendSegments)) {
        this->testAndAddDecomposition(m_CandidatePeriods, m_ModelTrendSegments,
                                      m_ValuesMinusTrend, false, decompositions);
    }

    // Week, year.
    m_Periods = m_CandidatePeriods;
    m_CandidatePeriods.assign({CSignal::seasonalComponentSummary(this->week()),
                               CSignal::seasonalComponentSummary(this->year())});
    this->removeIfNotTestable(m_CandidatePeriods);
    if (m_CandidatePeriods.size() > 0 && // Is there sufficient data?
        m_CandidatePeriods != m_Periods && // Have we already tested these periods?
        this->includesPermittedPeriod(m_CandidatePeriods) &&
        this->alreadyModelled(m_CandidatePeriods) == false &&
        removeTrend(m_CandidatePeriods, m_ValuesMinusTrend, m_ModelTrendSegments)) {
        this->testAndAddDecomposition(m_CandidatePeriods, m_ModelTrendSegments,
                                      m_ValuesMinusTrend, false, decompositions);
    }
}

void CTimeSeriesTestForSeasonality::addHighestAutocorrelation(const TRemoveTrend& removeTrend,
                                                              TModelVec& decompositions) const {
    // Highest serial autocorrelation components.
    if (removeTrend({}, m_ValuesMinusTrend, m_ModelTrendSegments)) {
        m_TemporaryValues = m_ValuesMinusTrend;
        auto diurnal = std::make_tuple(this->day(), this->week(), this->year());
        auto unit = [](std::size_t) { return 1.0; };
        m_CandidatePeriods = CSignal::seasonalDecomposition(
            m_TemporaryValues, m_OutlierFraction, diurnal, unit,
            m_StartOfWeekOverride, 0.05, m_MaximumNumberComponents);
        this->removeIfNotTestable(m_CandidatePeriods);
        if (m_CandidatePeriods.size() > 0 && // Did this identified any candidate components?
            this->includesPermittedPeriod(m_CandidatePeriods) && // Includes a sufficiently long period.
            this->alreadyModelled(m_CandidatePeriods) == false &&
            this->onlyDiurnal(m_CandidatePeriods) == false) {
            this->testAndAddDecomposition(m_CandidatePeriods, m_ModelTrendSegments,
                                          m_ValuesMinusTrend, false, decompositions);
        }
    }
}

void CTimeSeriesTestForSeasonality::testAndAddDecomposition(
    const TSeasonalComponentVec& periods,
    const TSizeVec& trendSegments,
    const TFloatMeanAccumulatorVec& valuesToTest,
    bool alreadyModelled,
    TModelVec& decompositions) const {
    std::size_t numberTrendSegments{trendSegments.empty() ? 0 : trendSegments.size() - 1};
    auto decomposition = this->testDecomposition(periods, numberTrendSegments,
                                                 valuesToTest, alreadyModelled);
    if (this->acceptDecomposition(decomposition)) {
        decomposition.s_AlreadyModelled = alreadyModelled;
        this->removeDiscontinuities(trendSegments, decomposition.s_TrendInitialValues);
        decompositions.push_back(std::move(decomposition));
    }
}

CTimeSeriesTestForSeasonality::SModel
CTimeSeriesTestForSeasonality::testDecomposition(const TSeasonalComponentVec& periods,
                                                 std::size_t numberTrendSegments,
                                                 const TFloatMeanAccumulatorVec& valuesToTest,
                                                 bool alreadyModelled) const {
    using TComputeScaling =
        std::function<bool(TFloatMeanAccumulatorVec&, SHypothesisStats&)>;

    LOG_TRACE(<< "testing " << core::CContainerPrinter::print(periods));

    TComputeScaling removeScalingModels[]{
        [&](TFloatMeanAccumulatorVec& values, SHypothesisStats& hypothesis) {
            hypothesis.s_ScaleSegments.assign({0, values.size()});
            return true;
        },
        [&](TFloatMeanAccumulatorVec& values, SHypothesisStats& hypothesis) {
            std::size_t period{hypothesis.s_Period.period()};
            hypothesis.s_ScaleSegments = TSegmentation::piecewiseLinearScaledSeasonal(
                values, period, m_SignificantPValue, m_OutlierFraction);
            return this->meanScale(hypothesis, [](std::size_t) { return 1.0; },
                                   values, m_Scales);
        }};

    // The following loop schematically does the following:
    //   1. Remove any out-of-phase seasonal components which haven't been tested from
    //      the values to test (these affect the tests we apply).
    //   2. Test the period with and without piecewise linear scaling and choose the
    //      best alternative.
    //   3. If the period was accepted update the values to remove the predictions of
    //      that component.
    //
    // Conceptually we are testing a sequence of nested hypotheses for modelling the
    // seasonality since any added component could be zeroed just by setting all its
    // predictions to zero. I chose not to express this as likelihood ratio test
    // because we can get away with a less powerful test and its errors are sensitive
    // to the distribution assumptions.
    //
    // For each component we test the significance of the variance it explains (F-test),
    // the cyclic autocorrelation on the window of values and the significance of an
    // amplitude test statistic. The amplitude test looks for frequently repeated spikes
    // or dips. Our model can represent such signals with seasonality in the variance
    // and modelling such approximate seasonality is very effective at reducing false
    // positives.
    //
    // The treatment of outliers is subtle. The variance and autocorrelation tests are
    // adversely affected by outliers. However, an unmodelled seasonality can easily
    // generate outliers. We compute test statistics with and without outliers and use
    // the most significant test. In this context, it is important to realize the outliers
    // need to be identified w.r.t. the model which is assumed for the data. We therefore
    // compute weights separately for the null hypothesis that the component is not present.
    //
    // The overall choice of whether to model a component depends on all these factors and
    // also the number of repeats we have observed, the number of values we observe per
    // period, etc (see testVariance and testAmplitude for details). Since the considerations
    // are heterogenous we combine them using fuzzy logic.

    TFloatMeanAccumulatorVec residuals{valuesToTest};
    THypothesisStatsVec hypotheses;
    hypotheses.reserve(periods.size());

    for (std::size_t i = 0; i < periods.size(); ++i) {

        LOG_TRACE(<< "testing " << periods[i].print());

        // Precondition by removing all remaining out-of-phase components.
        m_TemporaryValues = residuals;
        m_Periods.clear();
        for (std::size_t j = i + 1; j < periods.size(); ++j) {
            if (almostDivisor(periods[j].s_Period, periods[i].s_Period, 0.05) == false &&
                almostDivisor(periods[i].s_Period, periods[j].s_Period, 0.05) == false) {
                m_Periods.push_back(periods[j]);
            }
        }
        if (m_Periods.size() > 0) {
            LOG_TRACE(<< "removing " << core::CContainerPrinter::print(m_Periods));
            CSignal::fitSeasonalComponents(m_Periods, m_TemporaryValues, m_Components);
            this->removePredictions({m_Periods, 0, m_Periods.size()},
                                    {m_Components, 0, m_Components.size()},
                                    m_TemporaryValues);
        }

        // Restrict to the component to test time windows.
        m_WindowIndices.resize(m_TemporaryValues.size());
        std::iota(m_WindowIndices.begin(), m_WindowIndices.end(), 0);
        CSignal::restrictTo(periods[i], m_TemporaryValues);
        CSignal::restrictTo(periods[i], m_WindowIndices);
        m_ValuesToTest = m_TemporaryValues;
        auto period = CSignal::seasonalComponentSummary(periods[i].period());

        // Compute the null hypothesis residual variance statistics.
        m_Periods.assign(1, CSignal::seasonalComponentSummary(1));
        CSignal::fitSeasonalComponentsRobust(m_Periods, m_OutlierFraction,
                                             m_TemporaryValues, m_Components);
        auto H0 = this->residualVarianceStats(m_TemporaryValues);

        SHypothesisStats bestHypothesis{periods[i]};

        for (const auto& removeScaling : removeScalingModels) {

            SHypothesisStats hypothesis{periods[i]};

            if (removeScaling(m_ValuesToTest, hypothesis) &&
                CSignal::countNotMissing(m_ValuesToTest) > 0) {

                LOG_TRACE(<< "scale segments = "
                          << core::CContainerPrinter::print(hypothesis.s_ScaleSegments));

                hypothesis.s_NumberTrendSegments = numberTrendSegments;
                hypothesis.s_NumberScaleSegments = hypothesis.s_ScaleSegments.size() - 1;
                hypothesis.s_MeanNumberRepeats =
                    CSignal::meanNumberRepeatedValues(m_ValuesToTest, period);
                hypothesis.s_WindowRepeats =
                    static_cast<double>(this->observedRange(m_Values)) /
                    static_cast<double>(periods[i].s_WindowRepeat);

                m_Periods.assign(1, period);
                CSignal::fitSeasonalComponentsRobust(m_Periods, m_OutlierFraction,
                                                     m_ValuesToTest, m_Components);

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

        if (alreadyModelled || bestHypothesis.s_Truth.boolean()) {
            LOG_TRACE(<< "selected " << periods[i].print());
            this->updateResiduals(bestHypothesis, residuals);
            hypotheses.push_back(std::move(bestHypothesis));
        } else if (bestHypothesis.s_Period.windowed()) {
            hypotheses.push_back(std::move(bestHypothesis));
        }
    }

    if (alreadyModelled == false &&
        std::count_if(hypotheses.begin(), hypotheses.end(), [](const auto& hypothesis) {
            return hypothesis.s_Truth.boolean();
        }) == 0) {
        return {};
    }

    double variance{this->truncatedVariance(0.0, residuals) + m_EpsVariance};
    double truncatedVariance{this->truncatedVariance(m_OutlierFraction, residuals) + m_EpsVariance};
    LOG_TRACE(<< "variance = " << variance << " <variance> = " << truncatedVariance);

    TBoolVec componentsToRemoveMask{
        this->finalizeHypotheses(valuesToTest, hypotheses, residuals)};
    for (std::size_t i = 0; i < m_Values.size(); ++i) {
        double offset{CBasicStatistics::mean(m_Values[i]) -
                      CBasicStatistics::mean(valuesToTest[i])};
        CBasicStatistics::moment<0>(residuals[i]) += offset;
    }

    return {variance,
            truncatedVariance,
            2 * numberTrendSegments,
            std::move(residuals),
            std::move(hypotheses),
            std::move(componentsToRemoveMask)};
}

bool CTimeSeriesTestForSeasonality::acceptDecomposition(const SModel& decomposition) const {
    return decomposition.seasonal() &&
           std::count_if(
               decomposition.s_Hypotheses.begin(),
               decomposition.s_Hypotheses.end(), [this](const auto& hypothesis) {
                   return hypothesis.s_Period.windowed() &&
                          hypothesis.s_Period.s_Period == this->week();
               }) != static_cast<std::ptrdiff_t>(decomposition.s_Hypotheses.size());
}

void CTimeSeriesTestForSeasonality::updateResiduals(const SHypothesisStats& hypothesis,
                                                    TFloatMeanAccumulatorVec& residuals) const {
    m_TemporaryValues = residuals;
    CSignal::restrictTo(hypothesis.s_Period, m_TemporaryValues);
    this->meanScale(hypothesis, [](std::size_t) { return 1.0; }, m_TemporaryValues, m_Scales);
    m_Periods.assign(1, CSignal::seasonalComponentSummary(hypothesis.s_Period.period()));
    CSignal::fitSeasonalComponentsRobust(m_Periods, m_OutlierFraction,
                                         m_TemporaryValues, m_Components);

    for (std::size_t i = 0; i < m_TemporaryValues.size(); ++i) {
        auto& moments = residuals[m_WindowIndices[i]];
        CBasicStatistics::moment<0>(moments) =
            CBasicStatistics::mean(m_TemporaryValues[i]) -
            m_Periods[0].value(m_Components[0], i);
    }
}

CTimeSeriesTestForSeasonality::TBoolVec
CTimeSeriesTestForSeasonality::finalizeHypotheses(const TFloatMeanAccumulatorVec& values,
                                                  THypothesisStatsVec& hypotheses,
                                                  TFloatMeanAccumulatorVec& residuals) const {

    auto componentsToRemoveMask = this->selectModelledHypotheses(hypotheses);

    m_Periods.clear();
    for (std::size_t i = 0; i < hypotheses.size(); ++i) {
        if (hypotheses[i].s_Model) {
            m_Periods.push_back(hypotheses[i].s_Period);
        }
    }

    residuals = values;
    this->removeModelledPredictions(componentsToRemoveMask, m_ValuesStartTime, residuals);

    CSignal::fitSeasonalComponentsRobust(m_Periods, m_OutlierFraction, residuals, m_Components);

    auto nextModelled = [&](std::size_t i) {
        for (/**/; i < hypotheses.size() && hypotheses[i].s_Model == false; ++i) {
        }
        return i;
    };
    TSeasonalComponentVec period;
    TMeanAccumulatorVecVec component;

    for (std::size_t i = 0, j = nextModelled(0); i < m_Periods.size();
         ++i, j = nextModelled(j + 1)) {
        for (auto scale : {hypotheses[j].s_ScaleSegments.size() > 2, false}) {

            m_TemporaryValues = residuals;

            this->removePredictions({m_Periods, i + 1, m_Periods.size()},
                                    {m_Components, i + 1, m_Components.size()},
                                    m_TemporaryValues);

            m_WindowIndices.resize(m_TemporaryValues.size());
            std::iota(m_WindowIndices.begin(), m_WindowIndices.end(), 0);
            CSignal::restrictTo(m_Periods[i], m_TemporaryValues);
            CSignal::restrictTo(m_Periods[i], m_WindowIndices);

            auto weight = [&](std::size_t k) {
                return std::pow(0.9, static_cast<double>(m_TemporaryValues.size() - k - 1));
            };

            if (scale && this->meanScale(hypotheses[j], weight,
                                         m_TemporaryValues, m_Scales) == false) {
                continue;
            }

            period.assign(1, CSignal::seasonalComponentSummary(m_Periods[i].period()));
            CSignal::fitSeasonalComponents(period, m_TemporaryValues, component);

            hypotheses[j].s_ComponentSize =
                this->selectComponentSize(m_TemporaryValues, hypotheses[j].s_Period);
            hypotheses[j].s_InitialValues.resize(values.size());

            for (std::size_t k = 0; k < m_TemporaryValues.size(); ++k) {
                if (CBasicStatistics::count(m_TemporaryValues[k]) > 0.0) {
                    auto& value = CBasicStatistics::moment<0>(m_TemporaryValues[k]);
                    double prediction{period[0].value(component[0], k)};
                    value = prediction + (value - prediction) /
                                             static_cast<double>(m_Periods.size());
                }
                hypotheses[j].s_InitialValues[m_WindowIndices[k]] = m_TemporaryValues[k];
                if (CBasicStatistics::count(residuals[m_WindowIndices[k]]) > 0.0) {
                    CBasicStatistics::moment<0>(residuals[m_WindowIndices[k]]) -=
                        (scale ? TSegmentation::scaleAt(k, hypotheses[j].s_ScaleSegments, m_Scales)
                               : 1.0) *
                        period[0].value(component[0], k);
                }
            }
            break;
        }
    }

    return componentsToRemoveMask;
}

CTimeSeriesTestForSeasonality::TBoolVec
CTimeSeriesTestForSeasonality::selectModelledHypotheses(THypothesisStatsVec& hypotheses) const {

    // Ensure that we only keep "false" hypotheses which are needed because they
    // are the best hypothesis for their time window.
    //
    // For weekday/weekend modulation we use dedicated models for the weekend and
    // weekdays since it is more parsimonious than using one model for the whole
    // week. We do however need to ensure that we've selected a seasonal model for
    // say weekends if we have one for weekdays (even it is just a level prediction).
    // To achieve this we always add in all "windowed" seasonal components in
    // testDecomposition, whether they meet the conditions to be selected or not.
    // This prunes all those that aren't needed because we already have a model
    // for their window. They are kept in order of their actual truth value.
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

    std::ptrdiff_t excess{-m_MaximumNumberComponents};

    // Determine which periods from hypotheses will be modelled.
    std::size_t numberModelled{m_ModelledPeriods.size()};
    std::size_t numberSimilarToModelled{0};
    std::size_t numberEvictableModelled(std::count_if(
        boost::counting_iterator<std::size_t>(0),
        boost::counting_iterator<std::size_t>(numberModelled), [this](std::size_t i) {
            std::size_t period{m_ModelledPeriods[i].period()};
            std::size_t range{
                m_ModelledPeriods[i].fractionInWindow(this->observedRange(m_Values))};
            double repeats{static_cast<double>(range) / static_cast<double>(period)};
            double minimumRepeats{0.6 * std::max(m_MinimumRepeatsPerSegmentToTestVariance,
                                                 m_MinimumRepeatsPerSegmentToTestAmplitude)};
            return m_ModelledPeriodsTestable[i] && period >= 12 &&
                   CMinAmplitude::seenSufficientDataToTestAmplitude(range, period) &&
                   repeats >= minimumRepeats;
        }));
    for (std::size_t i = 0; i < hypotheses.size(); ++i) {
        const auto& period = hypotheses[i].s_Period;
        if (std::find_if(boost::counting_iterator<std::size_t>(0),
                         boost::counting_iterator<std::size_t>(numberModelled),
                         [&](const auto& j) {
                             return m_ModelledPeriods[j].almostEqual(period, 0.05) ||
                                    (m_ModelledPeriods[j].periodAlmostEqual(period, 0.05) &&
                                     m_ModelledPeriodsTestable[j] == false);
                         }) != boost::counting_iterator<std::size_t>(numberModelled)) {
            ++numberSimilarToModelled;
        }
    }
    for (std::size_t i = 0; i < hypotheses.size(); ++i) {
        // Have we found extra unmodelled components or should we be removing
        // currently modelled components?
        hypotheses[i].s_Model = (numberSimilarToModelled < hypotheses.size() ||
                                 numberSimilarToModelled < numberEvictableModelled);
        excess += hypotheses[i].s_Model ? 1 : 0;
    }

    // Mark which existing components we should remove if any.
    TBoolVec componentsToRemoveMask(m_ModelledPeriodsTestable);
    excess -= std::count(componentsToRemoveMask.begin(), componentsToRemoveMask.end(), true);

    LOG_TRACE(<< "to remove = " << core::CContainerPrinter::print(componentsToRemoveMask));
    LOG_TRACE(<< "excess = " << excess);

    // Don't exceed the maximum number of components discarding excess in order
    // of increasing explained variance.
    for (/**/; excess > 0; --excess) {
        auto hypothesis = std::min_element(
            hypotheses.begin(), hypotheses.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.s_ExplainedVariance < rhs.s_ExplainedVariance;
            });
        hypothesis->s_Model = false;
    }

    return componentsToRemoveMask;
}

std::size_t
CTimeSeriesTestForSeasonality::selectComponentSize(const TFloatMeanAccumulatorVec& valuesToTest,
                                                   const TSeasonalComponent& period) const {
    auto matchingModelled = std::find_if(
        m_ModelledPeriods.begin(), m_ModelledPeriods.end(), [&](const auto& modelledPeriod) {
            return modelledPeriod.s_Period == period.s_Period;
        });
    std::size_t modelledSize{
        matchingModelled == m_ModelledPeriods.end()
            ? 0
            : m_ModelledPeriodsSizes[matchingModelled - m_ModelledPeriods.begin()]};
    return std::max(modelledSize,
                    CSignal::selectComponentSize(valuesToTest, period.period()));
}

void CTimeSeriesTestForSeasonality::removeModelledPredictions(const TBoolVec& componentsToRemoveMask,
                                                              core_t::TTime startTime,
                                                              TFloatMeanAccumulatorVec& values) const {
    TBoolVec mask{m_ModelledPeriodsTestable};
    for (std::size_t i = 0; i < mask.size(); ++i) {
        mask[i] = mask[i] == false || componentsToRemoveMask[i];
    }
    core_t::TTime time{startTime};
    for (std::size_t i = 0; i < values.size(); ++i, time += m_BucketLength) {
        CBasicStatistics::moment<0>(values[i]) -= m_ModelledPredictor(time, mask);
    }
}

void CTimeSeriesTestForSeasonality::removeDiscontinuities(const TSizeVec& modelTrendSegments,
                                                          TFloatMeanAccumulatorVec& values) const {
    if (modelTrendSegments.size() > 2) {
        values = TSegmentation::removePiecewiseLinearDiscontinuities(
            std::move(values), modelTrendSegments, m_OutlierFraction);
    }
}

bool CTimeSeriesTestForSeasonality::meanScale(const SHypothesisStats& hypothesis,
                                              const TWeightFunc& weight,
                                              TFloatMeanAccumulatorVec& values,
                                              TDoubleVec& scales) const {
    if (hypothesis.s_ScaleSegments.size() > 2) {
        bool successful;
        std::tie(values, scales, successful) = TSegmentation::meanScalePiecewiseLinearScaledSeasonal(
            values, hypothesis.s_Period.period(), hypothesis.s_ScaleSegments, weight);
        return successful;
    }

    scales.assign(1, 1.0);
    return false;
}

void CTimeSeriesTestForSeasonality::testExplainedVariance(const TVarianceStats& H0,
                                                          SHypothesisStats& hypothesis) const {

    auto H1 = this->residualVarianceStats(m_ValuesToTest);

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
        m_ValuesToTest, 0,
        CIntegerTools::floor(m_ValuesToTest.size(), m_Periods[0].period())};

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
        this->observedRange(m_ValuesToTest), m_Periods[0].s_Period);
    if (hypothesis.s_SeenSufficientDataToTestAmplitude == false) {
        return;
    }

    double level{CBasicStatistics::mean(std::accumulate(
        m_ValuesToTest.begin(), m_ValuesToTest.end(), TMeanAccumulator{},
        [](TMeanAccumulator partialLevel, const TFloatMeanAccumulator& value) {
            partialLevel.add(CBasicStatistics::mean(value), CBasicStatistics::count(value));
            return partialLevel;
        }))};

    m_Amplitudes.assign(m_Periods[0].period(),
                        {m_ValuesToTest.size(), hypothesis.s_MeanNumberRepeats, level});
    for (std::size_t i = 0; i < m_ValuesToTest.size(); ++i) {
        if (m_Periods[0].contains(i)) {
            m_Amplitudes[m_Periods[0].offset(i)].add(i, m_ValuesToTest[i]);
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

CTimeSeriesTestForSeasonality::TVarianceStats
CTimeSeriesTestForSeasonality::residualVarianceStats(const TFloatMeanAccumulatorVec& values) const {
    auto result = CSignal::residualVarianceStats(values, m_Periods, m_Components);
    result.s_ResidualVariance += m_EpsVariance;
    result.s_TruncatedResidualVariance += m_EpsVariance;
    return result;
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
        if (this->alreadyModelled(period) == false) {
            return false;
        }
    }
    return true;
}

bool CTimeSeriesTestForSeasonality::alreadyModelled(const TSeasonalComponent& period) const {
    return std::find(m_ModelledPeriods.begin(), m_ModelledPeriods.end(), period) !=
           m_ModelledPeriods.end();
}

bool CTimeSeriesTestForSeasonality::onlyDiurnal(const TSeasonalComponentVec& periods) const {
    return std::find_if(periods.begin(), periods.end(), [this](const auto& period) {
               return this->isDiurnal(period) == false;
           }) == periods.end();
}

void CTimeSeriesTestForSeasonality::removeIfNotTestable(TSeasonalComponentVec& periods) const {
    periods.erase(std::remove_if(periods.begin(), periods.end(),
                                 [this](const auto& period) {
                                     return canTestPeriod(m_Values, period) == false;
                                 }),
                  periods.end());
}

CTimeSeriesTestForSeasonality::TPeriodDescriptor
CTimeSeriesTestForSeasonality::periodDescriptor(std::size_t period) const {
    if (period == this->day()) {
        return CNewSeasonalComponentSummary::E_Day;
    }
    if (period == this->week()) {
        return CNewSeasonalComponentSummary::E_Week;
    }
    if (period == this->year()) {
        return CNewSeasonalComponentSummary::E_Year;
    }
    return CNewSeasonalComponentSummary::E_General;
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

bool CTimeSeriesTestForSeasonality::includesPermittedPeriod(const TSeasonalComponentVec& periods) const {
    return m_MinimumPeriod == boost::none ||
           std::find_if(periods.begin(), periods.end(), [this](const auto& period) {
               return m_MinimumPeriod == boost::none ||
                      static_cast<core_t::TTime>(period.s_WindowRepeat) * m_BucketLength >=
                          *m_MinimumPeriod;
           }) != periods.end();
}

std::string CTimeSeriesTestForSeasonality::annotationText(const TSeasonalComponent& period) const {
    return "Detected seasonal component with period " +
           core::CTimeUtils::durationToString(
               m_BucketLength * static_cast<core_t::TTime>(period.s_Period)) +
           (this->isWeekend(period) ? " (weekend)"
                                    : (this->isWeekday(period) ? " (weekdays)" : ""));
}

std::size_t CTimeSeriesTestForSeasonality::day() const {
    return buckets(m_BucketLength, core::constants::DAY);
}

std::size_t CTimeSeriesTestForSeasonality::week() const {
    return buckets(m_BucketLength, core::constants::WEEK);
}

std::size_t CTimeSeriesTestForSeasonality::year() const {
    return buckets(m_BucketLength, core::constants::YEAR);
}

CTimeSeriesTestForSeasonality::TSizeSizePr CTimeSeriesTestForSeasonality::weekdayWindow() const {
    return {2 * this->day(), 7 * this->day()};
}

CTimeSeriesTestForSeasonality::TSizeSizePr CTimeSeriesTestForSeasonality::weekendWindow() const {
    return {0, 2 * this->day()};
}

CTimeSeriesTestForSeasonality::TSeasonalComponent
CTimeSeriesTestForSeasonality::toPeriod(core_t::TTime startTime,
                                        core_t::TTime bucketLength,
                                        const CSeasonalTime& component) {
    std::size_t periodInBuckets{buckets(bucketLength, component.period())};
    if (component.windowed()) {
        std::size_t startOfWindowInBuckets{
            buckets(bucketLength,
                    adjustForStartTime(startTime, component.windowRepeatStart())) %
            buckets(bucketLength, core::constants::WEEK)};
        std::size_t windowRepeatInBuckets{buckets(bucketLength, component.windowRepeat())};
        TSizeSizePr windowInBuckets{buckets(bucketLength, component.window().first),
                                    buckets(bucketLength, component.window().second)};
        return {periodInBuckets, startOfWindowInBuckets, windowRepeatInBuckets, windowInBuckets};
    }
    TSizeSizePr windowInBuckets{0, periodInBuckets};
    return {periodInBuckets, 0, periodInBuckets, windowInBuckets};
}

core_t::TTime CTimeSeriesTestForSeasonality::adjustForStartTime(core_t::TTime startTime,
                                                                core_t::TTime startOfWeek) {
    return (core::constants::WEEK + startOfWeek - (startTime % core::constants::WEEK)) %
           core::constants::WEEK;
}

std::size_t CTimeSeriesTestForSeasonality::buckets(core_t::TTime bucketLength,
                                                   core_t::TTime interval) {
    return static_cast<std::size_t>((interval + bucketLength / 2) / bucketLength);
}

bool CTimeSeriesTestForSeasonality::canTestPeriod(const TFloatMeanAccumulatorVec& values,
                                                  const TSeasonalComponent& period) {
    return 190 * period.s_WindowRepeat < 100 * observedRange(values) &&
           period.s_Period >= 2;
}

std::size_t CTimeSeriesTestForSeasonality::observedRange(const TFloatMeanAccumulatorVec& values) {
    int begin{0};
    int end{static_cast<int>(values.size())};
    int size{static_cast<int>(values.size())};
    for (/**/; begin < size && CBasicStatistics::count(values[begin]) == 0.0; ++begin) {
    }
    for (/**/; end > begin && CBasicStatistics::count(values[end - 1]) == 0.0; --end) {
    }
    return static_cast<std::size_t>(end - begin);
}

void CTimeSeriesTestForSeasonality::removePredictions(const TSeasonalComponentCRng& periodsToRemove,
                                                      const TMeanAccumulatorVecCRng& componentsToRemove,
                                                      TFloatMeanAccumulatorVec& values) {
    removePredictions(
        [&](std::size_t i) {
            double value{0.0};
            for (std::size_t j = 0; j < periodsToRemove.size(); ++j) {
                value += periodsToRemove[j].value(componentsToRemove[j], i);
            }
            return value;
        },
        values);
}

void CTimeSeriesTestForSeasonality::removePredictions(const TBucketPredictor& predictor,
                                                      TFloatMeanAccumulatorVec& values) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        CBasicStatistics::moment<0>(values[i]) -= predictor(i);
    }
}

bool CTimeSeriesTestForSeasonality::CMinAmplitude::seenSufficientDataToTestAmplitude(
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

    // We have the following hard constraints:
    //   1. We need to see at least m_MinimumRepeatsPerSegmentToTestVariance
    //      repeats of the seasonality.
    //   2. The test p-value needs to be less than m_SignificantPValue.
    //   3. The autocorrelation needs to be higher than m_MediumAutocorrelation.
    //
    // We also get more confident the more non-missing values we see per repeat,
    // if we have very high signficance or very high autocorrelation and less
    // confident if we've seen very few repeats or have low autocorrelation.
    //
    // In order to make final decision we soften the hard constraints using a fuzzy
    // logic approach. This uses a standard form with a logistic function to represent
    // the truth value of a proposition and multiplicative AND. This has the effect
    // that missing any constraint significantly means the test fails, but we can
    // still take advantage of a constraint which nearly meets if some other one
    // is comfortably satisfied. In this context, the width of the fuzzy region is
    // relatively small, typically 10% of the constraint value.
    //
    // The other considerations are one-sided, i.e. they either *only* increase
    // or decrease the truth value of the overall proposition. This is done by
    // setting them to the max or min of the constraint value and the decision
    // boundary.

    double minimumRepeatsPerSegment{params.m_MinimumRepeatsPerSegmentToTestVariance};
    double mediumAutocorrelation{params.m_MediumAutocorrelation};
    double lowAutocorrelation{params.m_LowAutocorrelation};
    double highAutocorrelation{params.m_HighAutocorrelation};
    double logSignificantPValue{std::log(params.m_SignificantPValue)};
    double logVerySignificantPValue{std::log(params.m_VerySignificantPValue)};
    std::size_t segments{std::max(s_NumberTrendSegments, std::size_t{1}) +
                         s_NumberScaleSegments - 1};
    double repeatsPerSegment{s_MeanNumberRepeats / static_cast<double>(segments)};
    double windowRepeatsPerSegment{segments > 1 ? s_WindowRepeats / static_cast<double>(segments)
                                                : minimumRepeatsPerSegment};
    double logPValue{std::log(s_ExplainedVariancePValue)};
    LOG_TRACE(<< "repeats per segment = " << repeatsPerSegment);
    return fuzzyGreaterThan(repeatsPerSegment / minimumRepeatsPerSegment, 1.0, 0.2) &&
           fuzzyGreaterThan(std::min(repeatsPerSegment / 2.0, 1.0), 1.0, 0.1) &&
           fuzzyGreaterThan(std::min(windowRepeatsPerSegment / minimumRepeatsPerSegment, 1.0),
                            1.0, 0.1) &&
           fuzzyGreaterThan(s_FractionNotMissing, 1.0, 0.5) &&
           fuzzyGreaterThan(logPValue / logSignificantPValue, 1.0, 0.1) &&
           fuzzyGreaterThan(std::max(logPValue / logVerySignificantPValue, 1.0), 1.0, 0.1) &&
           fuzzyGreaterThan(s_Autocorrelation / mediumAutocorrelation, 1.0, 0.2) &&
           fuzzyGreaterThan(std::min(s_Autocorrelation / lowAutocorrelation, 1.0), 1.0, 0.1) &&
           fuzzyGreaterThan(std::max(s_Autocorrelation / highAutocorrelation, 1.0), 1.0, 0.1);
}

CFuzzyTruthValue CTimeSeriesTestForSeasonality::SHypothesisStats::testAmplitude(
    const CTimeSeriesTestForSeasonality& params) const {
    if (s_SeenSufficientDataToTestAmplitude == false) {
        return CFuzzyTruthValue::OR_UNDETERMINED;
    }

    // Compare with the discussion in testVariance.

    double minimumRepeatsPerSegment{params.m_MinimumRepeatsPerSegmentToTestAmplitude};
    double lowAutocorrelation{params.m_LowAutocorrelation};
    double logSignificantPValue{std::log(params.m_SignificantPValue)};
    double logVerySignificantPValue{std::log(params.m_VerySignificantPValue)};
    std::size_t segments{std::max(s_NumberTrendSegments, std::size_t{1}) +
                         s_NumberScaleSegments - 1};
    double repeatsPerSegment{s_MeanNumberRepeats / static_cast<double>(segments)};
    double windowRepeatsPerSegment{segments > 1 ? s_WindowRepeats / static_cast<double>(segments)
                                                : minimumRepeatsPerSegment};
    double autocorrelation{s_AutocorrelationUpperBound};
    double logPValue{std::log(s_AmplitudePValue)};
    LOG_TRACE(<< "repeats per segment = " << repeatsPerSegment);
    return fuzzyGreaterThan(repeatsPerSegment / minimumRepeatsPerSegment, 1.0, 0.1) &&
           fuzzyGreaterThan(std::min(repeatsPerSegment / 2.0, 1.0), 1.0, 0.1) &&
           fuzzyGreaterThan(std::min(windowRepeatsPerSegment / minimumRepeatsPerSegment, 1.0),
                            1.0, 0.1) &&
           fuzzyGreaterThan(autocorrelation / lowAutocorrelation, 1.0, 0.2) &&
           fuzzyGreaterThan(logPValue / logSignificantPValue, 1.0, 0.1) &&
           fuzzyGreaterThan(std::max(logPValue / logVerySignificantPValue, 1.0), 1.0, 0.1);
}

std::string CTimeSeriesTestForSeasonality::SHypothesisStats::print() const {
    return s_Period.print();
}

bool CTimeSeriesTestForSeasonality::SModel::isNull(std::size_t numberValues) const {
    return s_Hypotheses.empty() && this->degreesFreedom(numberValues) > 0.0;
}

bool CTimeSeriesTestForSeasonality::SModel::isAlternative(std::size_t numberValues) const {
    return this->isNull(numberValues) == false && this->degreesFreedom(numberValues) > 0.0;
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

double CTimeSeriesTestForSeasonality::SModel::numberScalings() const {
    std::size_t segments{0};
    for (const auto& hypothesis : s_Hypotheses) {
        segments += hypothesis.s_NumberScaleSegments - 1;
    }
    return static_cast<double>(segments);
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

double CTimeSeriesTestForSeasonality::SModel::autocorrelation() const {
    double result{0.0};
    double Z{0.0};
    for (const auto& hypothesis : s_Hypotheses) {
        result += hypothesis.s_ExplainedVariance * hypothesis.s_Autocorrelation;
        Z += hypothesis.s_ExplainedVariance;
    }
    return result / Z;
}
}
}
