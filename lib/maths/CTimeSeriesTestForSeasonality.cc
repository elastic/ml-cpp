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
#include <maths/Constants.h>
#include <maths/MathsTypes.h>

#include <boost/iterator/counting_iterator.hpp>
#include <boost/math/distributions/fisher_f.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>

namespace ml {
namespace maths {
namespace {
double rightTailFTest(double v0, double v1, double df0, double df1) {
    // If there is insufficient data for either hypothesis treat we are conservative
    // and say the alternative hypothesis is not provable.
    if (df0 <= 0.0 || df1 <= 0.0) {
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
    core_t::TTime bucketsStartTime,
    core_t::TTime bucketLength,
    TOptionalTime startOfWeekTime,
    TFloatMeanAccumulatorVec initialValues)
    : m_AnnotationText{std::move(annotationText)}, m_Period{period}, m_Size{size},
      m_PeriodDescriptor{periodDescriptor}, m_InitialValuesStartTime{initialValuesStartTime},
      m_BucketsStartTime{bucketsStartTime}, m_BucketLength{bucketLength},
      m_StartOfWeekTime{startOfWeekTime}, m_InitialValues{std::move(initialValues)} {
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
            core_t::TTime startOfWeek{
                m_StartOfWeekTime ? *m_StartOfWeekTime
                                  : (m_BucketsStartTime + interval(m_Period.s_StartOfWeek)) %
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

void CSeasonalDecomposition::add(CNewTrendSummary trend) {
    m_Trend = std::move(trend);
}

void CSeasonalDecomposition::add(std::string annotationText,
                                 const TSeasonalComponent& period,
                                 std::size_t size,
                                 TPeriodDescriptor periodDescriptor,
                                 core_t::TTime initialValuesStartTime,
                                 core_t::TTime bucketsStartTime,
                                 core_t::TTime bucketLength,
                                 TOptionalTime startOfWeekTime,
                                 TFloatMeanAccumulatorVec initialValues) {
    m_Seasonal.emplace_back(std::move(annotationText), period, size, periodDescriptor,
                            initialValuesStartTime, bucketsStartTime, bucketLength,
                            startOfWeekTime, std::move(initialValues));
}

void CSeasonalDecomposition::add(TBoolVec seasonalToRemoveMask) {
    m_SeasonalToRemoveMask = std::move(seasonalToRemoveMask);
}

void CSeasonalDecomposition::withinBucketVariance(double variance) {
    m_WithinBucketVariance = variance;
}

bool CSeasonalDecomposition::componentsChanged() const {
    return m_Seasonal.size() > 0 || std::count(m_SeasonalToRemoveMask.begin(),
                                               m_SeasonalToRemoveMask.end(), true) > 0;
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

double CSeasonalDecomposition::withinBucketVariance() const {
    return m_WithinBucketVariance;
}

std::string CSeasonalDecomposition::print() const {
    return core::CContainerPrinter::print(m_Seasonal);
}

CTimeSeriesTestForSeasonality::CTimeSeriesTestForSeasonality(core_t::TTime valuesStartTime,
                                                             core_t::TTime bucketsStartTime,
                                                             core_t::TTime bucketLength,
                                                             core_t::TTime sampleInterval,
                                                             TFloatMeanAccumulatorVec values,
                                                             double sampleVariance,
                                                             double outlierFraction)
    : m_ValuesStartTime{valuesStartTime}, m_BucketsStartTime{bucketsStartTime},
      m_BucketLength{bucketLength}, m_SampleInterval{sampleInterval}, m_SampleVariance{sampleVariance},
      m_OutlierFraction{outlierFraction}, m_Values{std::move(values)},
      m_Outliers{static_cast<std::size_t>(std::max(
          outlierFraction * static_cast<double>(CSignal::countNotMissing(m_Values)) + 0.5,
          1.0))} {

    TMeanVarAccumulator moments{this->truncatedMoments(m_OutlierFraction, m_Values)};
    TMeanVarAccumulator meanAbs{this->truncatedMoments(
        m_OutlierFraction, m_Values, [](const TFloatMeanAccumulator& value) {
            return std::fabs(CBasicStatistics::mean(value));
        })};

    // Note we don't bother modelling seasonality whose amplitude is too small
    // compared to the absolute values. We won't raise anomalies for differences
    // from our predictions which are smaller than this anyway.
    m_EpsVariance = std::max(
        CTools::pow2(1000.0 * std::numeric_limits<double>::epsilon()) *
            CBasicStatistics::maximumLikelihoodVariance(moments),
        CTools::pow2(MINIMUM_COEFFICIENT_OF_VARIATION * CBasicStatistics::mean(meanAbs)));
    LOG_TRACE(<< "eps variance = " << m_EpsVariance);
}

bool CTimeSeriesTestForSeasonality::canTestComponent(const TFloatMeanAccumulatorVec& values,
                                                     core_t::TTime bucketsStartTime,
                                                     core_t::TTime bucketLength,
                                                     core_t::TTime minimumPeriod,
                                                     const CSeasonalTime& component) {
    return 10 * (component.period() % bucketLength) < component.period() &&
           canTestPeriod(values, buckets(bucketLength, minimumPeriod),
                         toPeriod(bucketsStartTime, bucketLength, component));
}

void CTimeSeriesTestForSeasonality::addModelledSeasonality(const CSeasonalTime& component,
                                                           std::size_t size) {
    auto period = toPeriod(m_BucketsStartTime, m_BucketLength, component);
    m_ModelledPeriods.push_back(period);
    m_ModelledPeriodsSizes.push_back(size);
    m_ModelledPeriodsTestable.push_back(canTestComponent(
        m_Values, m_BucketsStartTime, m_BucketLength, m_MinimumPeriod, component));
    if (period.windowed()) {
        m_StartOfWeekOverride = period.s_StartOfWeek;
        // We need the actual time in case it isn't a multiple of the bucket length
        // after the start of the window.
        m_StartOfWeekTimeOverride = component.windowRepeatStart();
    }
}

void CTimeSeriesTestForSeasonality::modelledSeasonalityPredictor(const TPredictor& predictor) {
    m_ModelledPredictor = predictor;
}

void CTimeSeriesTestForSeasonality::fitAndRemoveUntestableModelledComponents() {

    // Although we precondition by removing untestable modelled component predictions
    // there can be errors. This is problematic when the period is too short to test
    // and can induce spurious apparent seasonality at a multiple of its period. This
    // fits and removes any seasonality from the prediction error.

    TSeasonalComponentVec untestable;
    untestable.reserve(m_ModelledPeriods.size());
    for (const auto& period : m_ModelledPeriods) {
        if (period.period() > 1 && periodTooShortToTest(m_MinimumPeriod, period)) {
            untestable.push_back(period);
        }
    }
    if (untestable.size() > 0) {
        TMeanAccumulatorVecVec components;
        CSignal::fitSeasonalComponentsRobust(untestable, m_OutlierFraction,
                                             m_Values, components);
        this->removePredictions({untestable, 0, untestable.size()},
                                {components, 0, components.size()}, m_Values);
    }
}

CSeasonalDecomposition CTimeSeriesTestForSeasonality::decompose() const {

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
    //   1. For various trend assumptions, no trend, quadratic and piecewise linear,
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
    // trend and seasonality, but for example determining the right break points
    // (if any) in the trend together with the appropriate scaled seasonal components
    // is a non trivial estimation problem. We take an ordered approach, first fitting
    // the trend then seasonality and selecting at each stage only to fit significant
    // effects. However, we also test simpler hypotheses, such that there is no
    // trend at all, explicitly. This is much more forgiving to the estimation
    // process since if the data doesn't have a trend not trying to fit one can
    // easily be identified as a better choice after the fact. The final selection
    // is based on a number of criterion which are geared towards our modelling
    // techniques and are described in select.

    LOG_TRACE(<< "decomposing " << m_Values.size()
              << " values, bucket length = " << m_BucketLength);

    // Shortcircuit if we can't test any periods.
    if (canTestPeriod(m_Values, 0, CSignal::seasonalComponentSummary(2)) == false) {
        return {};
    }

    TSizeVec trendSegments{TSegmentation::piecewiseLinear(
        m_Values, m_SignificantPValue, m_OutlierFraction, MAXIMUM_NUMBER_SEGMENTS)};
    LOG_TRACE(<< "trend segments = " << core::CContainerPrinter::print(trendSegments));

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
            // trivial to perform a sequence of "line searches". In practice,
            // this converges quickly so we use a fixed number of iterations.
            LOG_TRACE(<< "quadratic trend");
            using TRegression = CLeastSquaresOnlineRegression<2, double>;

            modelTrendSegments.assign({0, values.size()});

            TRegression trend;
            TRegression::TArray parameters;
            parameters.fill(0.0);
            auto predictor = [&](std::size_t i) {
                return TRegression::predict(parameters, static_cast<double>(i));
            };

            if (periods.empty()) {
                values = m_Values;
                for (std::size_t j = 0; j < values.size(); ++j) {
                    trend.add(static_cast<double>(j), CBasicStatistics::mean(values[j]),
                              CBasicStatistics::count(values[j]));
                }
                // Note that parameters are referenced by predictor. Reading them
                // here refreshes the values used for prediction. Computing the
                // parameters is the bottleneck in this code and the same values
                // are used for each prediction. We optimise removePredictions by
                // reading them once upfront.
                trend.parameters(parameters);
                this->removePredictions(predictor, values);
                return true;
            }

            for (std::size_t i = 0; i < 3; ++i) {
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
                // See above.
                trend.parameters(parameters);
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
            values = TSegmentation::removePiecewiseLinear(
                std::move(values), modelTrendSegments, m_OutlierFraction);

            if (periods.size() > 0) {
                CSignal::fitSeasonalComponents(periods, values, m_Components);
                values = m_Values;
                this->removePredictions(predictor, values);
                values = TSegmentation::removePiecewiseLinear(
                    std::move(values), modelTrendSegments, m_OutlierFraction);
                this->removePredictions(
                    [&](std::size_t j) { return -predictor(j); }, values);
            }

            return true;
        }};

    try {
        TModelVec decompositions;
        decompositions.reserve(8 * std::size(removeTrendModels));

        for (const auto& removeTrend : removeTrendModels) {
            this->addNotSeasonal(removeTrend, decompositions);
            this->addModelled(removeTrend, decompositions);
            this->addDiurnal(removeTrend, decompositions);
            this->addHighestAutocorrelation(removeTrend, decompositions);
        }

        return this->select(decompositions);

    } catch (const std::exception& e) {
        LOG_ERROR(<< "Seasonal decomposition failed: " << e.what());
    }

    return {};
}

CSeasonalDecomposition CTimeSeriesTestForSeasonality::select(TModelVec& decompositions) const {

    // If the existing seasonality couldn't be tested short circuit: we'll keep it.

    if (std::find_if(decompositions.begin(), decompositions.end(), [](const auto& decomposition) {
            return decomposition.s_AlreadyModelled && decomposition.isTestable() == false;
        }) != decompositions.end()) {
        return {};
    }

    // Choose the hypothesis which yields the best explanation of the values.

    // Sort by increasing complexity.
    std::stable_sort(decompositions.begin(), decompositions.end(),
                     [](const auto& lhs, const auto& rhs) {
                         return lhs.numberParameters() < rhs.numberParameters();
                     });

    auto computePValue = [&](std::size_t H1) {
        double pValueMax{-1.0};
        double logPValueProxyMax{-std::numeric_limits<double>::max()};
        std::size_t pValueMaxHypothesis{decompositions.size()};
        for (std::size_t H0 = 0; H0 < decompositions.size(); ++H0) {
            if (decompositions[H0].isNull()) {
                double pValue{decompositions[H1].pValue(decompositions[H0])};
                double logPValueProxy{decompositions[H1].logPValueProxy(decompositions[H0])};
                if (pValue > pValueMax ||
                    (pValue == pValueMax && logPValueProxy > logPValueProxyMax)) {
                    std::tie(pValueMax, logPValueProxyMax, pValueMaxHypothesis) =
                        std::make_tuple(pValue, logPValueProxy, H0);
                }
            }
        }
        return std::make_tuple(pValueMax, logPValueProxyMax, pValueMaxHypothesis);
    };

    double variance{CBasicStatistics::maximumLikelihoodVariance(
        this->truncatedMoments(0.0, m_Values))};
    double truncatedVariance{CBasicStatistics::maximumLikelihoodVariance(
        this->truncatedMoments(m_OutlierFraction, m_Values))};
    LOG_TRACE(<< "variance = " << variance << " truncated variance = " << truncatedVariance);

    // Select the best decomposition if it is a statistically significant improvement.
    std::size_t selected{decompositions.size()};
    double qualitySelected{-std::numeric_limits<double>::max()};
    double minPValue{1.0};
    std::size_t h0ForMinPValue{0};
    for (std::size_t H1 = 0; H1 < decompositions.size(); ++H1) {
        if (decompositions[H1].isAlternative()) {
            double pValueVsH0;
            double logPValueProxy;
            std::size_t H0;
            std::tie(pValueVsH0, logPValueProxy, H0) = computePValue(H1);
            double logPValue{pValueVsH0 == 0.0
                                 ? std::log(std::numeric_limits<double>::min())
                                 : (pValueVsH0 == 1.0 ? -std::numeric_limits<double>::min()
                                                      : std::log(pValueVsH0))};
            logPValueProxy = std::min(logPValueProxy, -std::numeric_limits<double>::min());
            double pValueToAccept{decompositions[H1].s_AlreadyModelled
                                      ? m_PValueToEvict
                                      : m_AcceptedFalsePostiveRate};
            if (pValueVsH0 < minPValue) {
                std::tie(minPValue, h0ForMinPValue) = std::make_pair(pValueVsH0, H0);
            }
            LOG_TRACE(<< "hypothesis = "
                      << core::CContainerPrinter::print(decompositions[H1].s_Hypotheses));
            LOG_TRACE(<< "p-value vs not periodic = " << pValueVsH0
                      << ", log proxy = " << logPValueProxy);

            if (pValueVsH0 > pValueToAccept) {
                continue;
            }

            // We've rejected the hypothesis that the signal is not periodic.
            //
            // First check if this is significantly better than the currently selected
            // model from an explained variance standpoint. If the test is ambiguous
            // then fallback to selecting based on a number of criteria.
            //
            // We therefore choose the best model based on the following criteria:
            //   1. The amount of variance explained per parameter. Models with a small
            //      period which explain most of the variance are preferred because they
            //      are more accurate and robust.
            //   2. Whether the components are already modelled to avoid churn on marginal
            //      decisions.
            //   3. The log p-value vs non seasonal. This captures information about both
            //      the model size and the variance with and without the model.
            //   4. Standard deviations above the mean of the F-distribution. This captures
            //      similar information to the log p-value but won't underflow.
            //   5. The total target model size. The p-value is less sensitive to model
            //      size as the window length increases. However, for both stability
            //      and efficiency considerations we strongly prefer smaller models.
            //   6. The number of scalings and pieces in the trend model. These increase
            //      false positives so if we have an alternative similarly good hypothesis
            //      we use that one.
            //   7. The number of repeats of the superposition of components we've seen.
            //      We penalise seeing fewer than two repeats to avoid using interference
            //      to fit changes in the seasonal pattern.
            //
            // Why sum the logs you might ask. This makes the decision dimensionless.
            // Consider that sum_i{ log(f_i) } < sum_i{ log(f_i') } is equivalent to
            // sum_i{ log(f_i / f_i')} < 0 so if we scale each feature by a constant
            // they cancel and we still make the same decision.
            //
            // One can think of this as a smooth lexicographical order with the weights
            // playing the role of the order: smaller weight values "break ties".

            auto explainedVariancePerParameter =
                decompositions[H1].explainedVariancePerParameter(variance, truncatedVariance);
            double leastCommonRepeat{decompositions[H1].leastCommonRepeat()};
            double pValueVsSelected{
                selected < decompositions.size()
                    ? decompositions[H1].pValue(decompositions[selected], 1e-3, m_SampleVariance)
                    : 1.0};
            double scalings{decompositions[H1].numberScalings()};
            double segments{
                std::max(static_cast<double>(decompositions[H1].s_NumberTrendParameters / 2), 1.0) -
                1.0};
            LOG_TRACE(<< "explained variance per param = " << explainedVariancePerParameter
                      << ", scalings = " << scalings << ", segments = " << segments
                      << ", number parameters = " << decompositions[H1].numberParameters()
                      << ", p-value H1 vs selected = " << pValueVsSelected);
            LOG_TRACE(<< "residual moments = " << decompositions[H1].s_ResidualMoments
                      << ", truncated residual moments = " << decompositions[H1].s_TruncatedResidualMoments
                      << ", sample variance = " << m_SampleVariance);
            LOG_TRACE(<< "least common repeat = " << leastCommonRepeat);

            double quality{1.0 * std::log(explainedVariancePerParameter(0)) +
                           1.0 * std::log(explainedVariancePerParameter(1)) +
                           0.7 * decompositions[H1].componentsSimilarity() +
                           0.5 * std::log(-logPValue) + 0.2 * std::log(-logPValueProxy) -
                           0.5 * std::log(decompositions[H1].targetModelSize()) -
                           0.3 * std::log(0.2 + CTools::pow2(scalings)) -
                           0.3 * std::log(1.0 + CTools::pow2(segments)) -
                           0.3 * std::log(std::max(leastCommonRepeat, 0.5))};
            double qualityToAccept{
                1.0 * qualitySelected -
                1.0 * std::log(1.0 + std::max(std::log(0.01 / pValueVsSelected), 0.0))};
            LOG_TRACE(<< "target size = " << decompositions[H1].targetModelSize()
                      << ", modelled = " << decompositions[H1].s_AlreadyModelled);
            LOG_TRACE(<< "quality = " << quality << " to accept = " << qualityToAccept);

            if (quality > qualityToAccept) {
                std::tie(selected, qualitySelected) = std::make_pair(H1, quality);
                LOG_TRACE(<< "selected " << selected);
            }
        }
    }

    CSeasonalDecomposition result;

    if (selected < decompositions.size() &&
        std::count_if(decompositions[selected].s_Hypotheses.begin(), // Are there new components?
                      decompositions[selected].s_Hypotheses.end(), [this](const auto& hypothesis) {
                          return hypothesis.s_Model &&
                                 this->permittedPeriod(hypothesis.s_Period);
                      }) > 0) {
        LOG_TRACE(<< "selected = "
                  << core::CContainerPrinter::print(decompositions[selected].s_Hypotheses));

        result.add(CNewTrendSummary{m_ValuesStartTime, m_BucketLength,
                                    std::move(decompositions[selected].s_TrendInitialValues)});
        for (auto& hypothesis : decompositions[selected].s_Hypotheses) {
            if (hypothesis.s_Model && this->permittedPeriod(hypothesis.s_Period)) {
                LOG_TRACE(<< "Adding " << hypothesis.s_Period.print());
                result.add(this->annotationText(hypothesis.s_Period),
                           hypothesis.s_Period, hypothesis.s_ModelSize,
                           this->periodDescriptor(hypothesis.s_Period.s_Period),
                           m_ValuesStartTime, m_BucketsStartTime,
                           m_BucketLength, m_StartOfWeekTimeOverride,
                           std::move(hypothesis.s_InitialValues));
            }
        }
        result.add(std::move(decompositions[selected].s_RemoveComponentsMask));
        result.withinBucketVariance(m_SampleVariance);
        return result;
    }

    LOG_TRACE(<< "p-value min = " << minPValue);

    // If the evidence for seasonality is weak and we can test all components
    // in this window remove them.
    std::ptrdiff_t numberModelled(m_ModelledPeriods.size());
    if (minPValue > m_PValueToEvict && numberModelled > 0 &&
        std::count(m_ModelledPeriodsTestable.begin(),
                   m_ModelledPeriodsTestable.end(), true) == numberModelled) {
        result.add(CNewTrendSummary{
            m_ValuesStartTime, m_BucketLength,
            std::move(decompositions[h0ForMinPValue].s_TrendInitialValues)});
        result.add(std::move(decompositions[h0ForMinPValue].s_RemoveComponentsMask));
        result.withinBucketVariance(m_SampleVariance);
    }

    return result;
}

void CTimeSeriesTestForSeasonality::addNotSeasonal(const TRemoveTrend& removeTrend,
                                                   TModelVec& decompositions) const {
    if (removeTrend({}, m_ValuesMinusTrend, m_ModelTrendSegments)) {
        decompositions.emplace_back(
            *this, this->truncatedMoments(0.0, m_ValuesMinusTrend),
            this->truncatedMoments(m_OutlierFraction, m_ValuesMinusTrend),
            this->numberTrendParameters(
                m_ModelTrendSegments.empty() ? 0 : m_ModelTrendSegments.size() - 1),
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
        this->testAndAddDecomposition(m_CandidatePeriods, m_ModelTrendSegments, m_ValuesMinusTrend,
                                      true,  // Already modelled
                                      false, // Is diurnal
                                      decompositions);

        // Already modelled plus highest serial autocorrelation seasonal components.
        m_Periods = m_CandidatePeriods;
        CSignal::fitSeasonalComponents(m_Periods, m_ValuesMinusTrend, m_Components);
        m_TemporaryValues = m_ValuesMinusTrend;
        this->removePredictions({m_Periods, 0, m_Periods.size()},
                                {m_Components, 0, m_Components.size()}, m_TemporaryValues);
        auto diurnal = std::make_tuple(this->day(), this->week(), this->year());
        for (const auto& period : CSignal::seasonalDecomposition(
                 m_TemporaryValues, m_OutlierFraction, diurnal,
                 m_StartOfWeekOverride, 0.05, m_MaximumNumberComponents)) {
            if (std::find_if(m_Periods.begin(), m_Periods.end(), [&](const auto& modelledPeriod) {
                    return modelledPeriod.periodAlmostEqual(period, 0.05);
                }) == m_Periods.end()) {
                m_CandidatePeriods.push_back(period);
            }
        }
        this->removeIfNotTestable(m_CandidatePeriods);
        if (this->includesNewComponents(m_CandidatePeriods) &&
            this->onlyDiurnal(m_CandidatePeriods) == false) {
            this->testAndAddDecomposition(m_CandidatePeriods,
                                          m_ModelTrendSegments, m_ValuesMinusTrend,
                                          false, // Already modelled
                                          false, // Is diurnal
                                          decompositions);
        }
    }
}

void CTimeSeriesTestForSeasonality::addDiurnal(const TRemoveTrend& removeTrend,
                                               TModelVec& decompositions) const {
    // day + year.
    m_CandidatePeriods.assign({CSignal::seasonalComponentSummary(this->day()),
                               CSignal::seasonalComponentSummary(this->year())});
    this->removeIfNotTestable(m_CandidatePeriods);
    if (this->includesNewComponents(m_CandidatePeriods) &&
        removeTrend(m_CandidatePeriods, m_ValuesMinusTrend, m_ModelTrendSegments)) {
        this->testAndAddDecomposition(m_CandidatePeriods, m_ModelTrendSegments, m_ValuesMinusTrend,
                                      false, // Already modelled
                                      true,  // Is diurnal
                                      decompositions);
    }

    m_CandidatePeriods.assign({CSignal::seasonalComponentSummary(this->week()),
                               CSignal::seasonalComponentSummary(this->year())});
    this->removeIfNotTestable(m_CandidatePeriods);
    if (removeTrend(m_CandidatePeriods, m_ValuesMinusTrend, m_ModelTrendSegments)) {
        m_CandidatePeriods = CSignal::tradingDayDecomposition(
            m_ValuesMinusTrend, m_OutlierFraction, this->week(), m_StartOfWeekOverride);

        // weekday/weekend modulation + year.
        if (m_CandidatePeriods.size() > 0) {
            CSignal::appendSeasonalComponentSummary(this->year(), m_CandidatePeriods);
            this->removeIfNotTestable(m_CandidatePeriods);
            if (this->includesNewComponents(m_CandidatePeriods)) {
                this->testAndAddDecomposition(m_CandidatePeriods,
                                              m_ModelTrendSegments, m_ValuesMinusTrend,
                                              false, // Already modelled
                                              true,  // Is diurnal
                                              decompositions);
            }
        }

        // week + year.
        m_CandidatePeriods.assign({CSignal::seasonalComponentSummary(this->week()),
                                   CSignal::seasonalComponentSummary(this->year())});
        this->removeIfNotTestable(m_CandidatePeriods);
        if (this->includesNewComponents(m_CandidatePeriods)) {
            this->testAndAddDecomposition(m_CandidatePeriods,
                                          m_ModelTrendSegments, m_ValuesMinusTrend,
                                          false, // Already modelled
                                          true,  // Is diurnal
                                          decompositions);
        }
    }
}

void CTimeSeriesTestForSeasonality::addHighestAutocorrelation(const TRemoveTrend& removeTrend,
                                                              TModelVec& decompositions) const {
    // Highest serial autocorrelation components.
    if (removeTrend({}, m_ValuesMinusTrend, m_ModelTrendSegments)) {
        auto diurnal = std::make_tuple(this->day(), this->week(), this->year());
        m_CandidatePeriods = CSignal::seasonalDecomposition(
            m_ValuesMinusTrend, m_OutlierFraction, diurnal,
            m_StartOfWeekOverride, 0.05, m_MaximumNumberComponents);
        this->removeIfNotTestable(m_CandidatePeriods);
        if (removeTrend(m_CandidatePeriods, m_ValuesMinusTrend, m_ModelTrendSegments) &&
            this->includesNewComponents(m_CandidatePeriods) &&
            this->onlyDiurnal(m_CandidatePeriods) == false) {
            this->testAndAddDecomposition(m_CandidatePeriods,
                                          m_ModelTrendSegments, m_ValuesMinusTrend,
                                          false, // Already modelled
                                          false, // Is diurnal
                                          decompositions);
        }
    }
}

void CTimeSeriesTestForSeasonality::testAndAddDecomposition(
    const TSeasonalComponentVec& periods,
    const TSizeVec& trendSegments,
    const TFloatMeanAccumulatorVec& valuesToTest,
    bool alreadyModelled,
    bool isDiurnal,
    TModelVec& decompositions) const {
    std::size_t numberTrendSegments{trendSegments.empty() ? 0 : trendSegments.size() - 1};
    auto decomposition = this->testDecomposition(periods, numberTrendSegments,
                                                 valuesToTest, alreadyModelled);
    if (this->considerDecompositionForSelection(decomposition, alreadyModelled, isDiurnal)) {
        decomposition.s_AlreadyModelled = alreadyModelled;
        this->removeDiscontinuities(trendSegments, decomposition.s_TrendInitialValues);
        decompositions.push_back(std::move(decomposition));
    }
}

bool CTimeSeriesTestForSeasonality::considerDecompositionForSelection(const SModel& decomposition,
                                                                      bool alreadyModelled,
                                                                      bool isDiurnal) const {
    return decomposition.seasonal() &&
           std::count_if(decomposition.s_Hypotheses.begin(),
                         decomposition.s_Hypotheses.end(),
                         [this](const auto& hypothesis) {
                             return hypothesis.s_Period.windowed() &&
                                    hypothesis.s_Period.s_Period == this->week();
                         }) !=
               static_cast<std::ptrdiff_t>(decomposition.s_Hypotheses.size()) &&
           std::count_if(decomposition.s_Hypotheses.begin(),
                         decomposition.s_Hypotheses.end(), [&](const auto& hypothesis) {
                             return alreadyModelled ||
                                    this->isDiurnal(hypothesis.s_Period) == isDiurnal;
                         }) > 0;
}

CTimeSeriesTestForSeasonality::SModel
CTimeSeriesTestForSeasonality::testDecomposition(const TSeasonalComponentVec& periods,
                                                 std::size_t numberTrendSegments,
                                                 const TFloatMeanAccumulatorVec& valuesToTest,
                                                 bool alreadyModelled) const {

    // The main loop schematically does the following:
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

    using TMeanScaleHypothesis =
        std::function<bool(TFloatMeanAccumulatorVec&, const TSeasonalComponentVec&,
                           const TMeanAccumulatorVecVec&, SHypothesisStats&)>;

    LOG_TRACE(<< "testing " << core::CContainerPrinter::print(periods));

    auto meanScale = [](const TSizeVec& segmentation, const TDoubleVec& scales) {
        return TSegmentation::meanScale(segmentation, scales);
    };
    TMeanScaleHypothesis constantScales[]{
        [&](TFloatMeanAccumulatorVec& values, const TSeasonalComponentVec&,
            const TMeanAccumulatorVecVec&, SHypothesisStats& hypothesis) {
            hypothesis.s_ScaleSegments.assign({0, values.size()});
            return true;
        },
        [&](TFloatMeanAccumulatorVec& values, const TSeasonalComponentVec& period,
            const TMeanAccumulatorVecVec& component, SHypothesisStats& hypothesis) {
            hypothesis.s_ScaleSegments = TSegmentation::piecewiseLinearScaledSeasonal(
                values,
                [&](std::size_t i) { return period[0].value(component[0], i); },
                m_SignificantPValue, MAXIMUM_NUMBER_SEGMENTS);
            return this->constantScale(meanScale, m_Periods, hypothesis.s_ScaleSegments,
                                       values, m_ScaledComponent, m_ComponentScales);
        }};

    TFloatMeanAccumulatorVec residuals{valuesToTest};
    THypothesisStatsVec hypotheses;
    hypotheses.reserve(periods.size());

    // If the superposition of periodicities doesn't repeat in window we need to be careful
    // not to use it to model non-seasonal changes. We start to penalise the selection as
    // lcm(periods) exceeds the window length.
    std::size_t leastCommonRepeat{1};
    std::size_t observedRange{this->observedRange(m_Values)};

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

        // Compute the null hypothesis residual variance statistics.
        m_Periods.assign(1, CSignal::seasonalComponentSummary(1));
        CSignal::fitSeasonalComponentsRobust(m_Periods, m_OutlierFraction,
                                             m_TemporaryValues, m_Components);
        auto H0 = this->residualVarianceStats(m_TemporaryValues);

        SHypothesisStats bestHypothesis{periods[i]};

        bool componentAlreadyModelled{this->alreadyModelled(periods[i])};
        for (const auto& constantScale : constantScales) {

            SHypothesisStats hypothesis{periods[i]};

            auto period = CSignal::seasonalComponentSummary(periods[i].period());
            m_Periods.assign(1, period);
            CSignal::fitSeasonalComponentsRobust(m_Periods, m_OutlierFraction,
                                                 m_ValuesToTest, m_Components);

            if (constantScale(m_ValuesToTest, m_Periods, m_Components, hypothesis) &&
                CSignal::countNotMissing(m_ValuesToTest) > 0) {

                LOG_TRACE(<< "scale segments = "
                          << core::CContainerPrinter::print(hypothesis.s_ScaleSegments));

                hypothesis.s_IsTestable = true;
                hypothesis.s_NumberTrendSegments = numberTrendSegments;
                hypothesis.s_NumberScaleSegments = hypothesis.s_ScaleSegments.size() - 1;
                hypothesis.s_MeanNumberRepeats =
                    CSignal::meanNumberRepeatedValues(m_ValuesToTest, period);
                hypothesis.s_WindowRepeats = static_cast<double>(observedRange) /
                                             static_cast<double>(periods[i].s_WindowRepeat);
                hypothesis.s_LeastCommonRepeat =
                    static_cast<double>(componentAlreadyModelled
                                            ? leastCommonRepeat
                                            : CIntegerTools::lcm(leastCommonRepeat,
                                                                 periods[i].s_WindowRepeat)) /
                    static_cast<double>(observedRange);
                hypothesis.testExplainedVariance(*this, H0);
                hypothesis.testAutocorrelation(*this);
                hypothesis.testAmplitude(*this);
                hypothesis.s_Truth = hypothesis.varianceTestResult(*this) ||
                                     hypothesis.amplitudeTestResult(*this);
                LOG_TRACE(<< "truth = " << hypothesis.s_Truth.print());

                if (hypothesis.isBetter(bestHypothesis)) {
                    bestHypothesis = std::move(hypothesis);
                }
            }
        }

        if (alreadyModelled && bestHypothesis.evict(*this, i)) {
            LOG_TRACE(<< "discarding " << periods[i].print());
            bestHypothesis.s_DiscardingModel = true;
            hypotheses.push_back(std::move(bestHypothesis));
        } else if (alreadyModelled || bestHypothesis.s_Truth.boolean()) {
            leastCommonRepeat = componentAlreadyModelled
                                    ? leastCommonRepeat
                                    : CIntegerTools::lcm(leastCommonRepeat,
                                                         periods[i].s_WindowRepeat);
            LOG_TRACE(<< "selected " << periods[i].print());
            this->updateResiduals(bestHypothesis, residuals);
            hypotheses.push_back(std::move(bestHypothesis));
        } else if (bestHypothesis.s_Period.windowed()) {
            hypotheses.push_back(std::move(bestHypothesis));
        }
    }

    if (alreadyModelled == false &&
        std::count_if(hypotheses.begin(), hypotheses.end(), [this](const auto& hypothesis) {
            return hypothesis.s_Truth.boolean() &&
                   this->alreadyModelled(hypothesis.s_Period) == false;
        }) == 0) {
        return {};
    }

    auto residualMoments = this->truncatedMoments(0.0, residuals);
    auto truncatedResidualMoments = this->truncatedMoments(m_OutlierFraction, residuals);
    LOG_TRACE(<< "variance = " << residualMoments << " <variance> = " << truncatedResidualMoments);

    TBoolVec componentsToRemoveMask{this->finalizeHypotheses(
        valuesToTest, alreadyModelled, hypotheses, residuals)};
    for (std::size_t i = 0; i < m_Values.size(); ++i) {
        double offset{CBasicStatistics::mean(m_Values[i]) -
                      CBasicStatistics::mean(valuesToTest[i])};
        CBasicStatistics::moment<0>(residuals[i]) += offset;
    }

    return {*this,
            residualMoments,
            truncatedResidualMoments,
            this->numberTrendParameters(numberTrendSegments),
            std::move(residuals),
            std::move(hypotheses),
            std::move(componentsToRemoveMask)};
}

void CTimeSeriesTestForSeasonality::updateResiduals(const SHypothesisStats& hypothesis,
                                                    TFloatMeanAccumulatorVec& residuals) const {
    m_TemporaryValues = residuals;
    m_WindowIndices.resize(residuals.size());
    std::iota(m_WindowIndices.begin(), m_WindowIndices.end(), 0);
    const TSizeVec& scaleSegments{hypothesis.s_ScaleSegments};
    CSignal::restrictTo(hypothesis.s_Period, m_TemporaryValues);
    CSignal::restrictTo(hypothesis.s_Period, m_WindowIndices);
    m_Periods.assign(1, CSignal::seasonalComponentSummary(hypothesis.s_Period.period()));

    auto meanScale = [](const TSizeVec& segmentation, const TDoubleVec& scales) {
        return TSegmentation::meanScale(segmentation, scales);
    };
    bool scale{this->constantScale(meanScale, m_Periods, scaleSegments, m_TemporaryValues,
                                   m_ScaledComponent, m_ComponentScales)};

    if (scale == false) {
        CSignal::fitSeasonalComponentsRobust(m_Periods, m_OutlierFraction,
                                             m_TemporaryValues, m_Components);
    }

    for (std::size_t i = 0; i < m_TemporaryValues.size(); ++i) {
        CBasicStatistics::moment<0>(residuals[m_WindowIndices[i]]) -=
            (scale ? TSegmentation::scaleAt(i, scaleSegments, m_ComponentScales) *
                         m_ScaledComponent[0][m_Periods[0].offset(i)]
                   : m_Periods[0].value(m_Components[0], i));
    }
}

CTimeSeriesTestForSeasonality::TBoolVec
CTimeSeriesTestForSeasonality::finalizeHypotheses(const TFloatMeanAccumulatorVec& values,
                                                  bool alreadyModelled,
                                                  THypothesisStatsVec& hypotheses,
                                                  TFloatMeanAccumulatorVec& residuals) const {

    auto componentsToRemoveMask = this->selectModelledHypotheses(alreadyModelled, hypotheses);
    auto componentsExcludedMask = componentsToRemoveMask;

    m_Periods.clear();
    TSizeVec periodsHypotheses;
    periodsHypotheses.reserve(hypotheses.size());

    for (std::size_t i = 0; i < hypotheses.size(); ++i) {
        // We always fit here even components we are already modelling because it's
        // a fairer comparison with new components.
        if (hypotheses[i].s_Model) {
            m_Periods.push_back(hypotheses[i].s_Period);
            periodsHypotheses.push_back(i);
        } else if (hypotheses[i].s_DiscardingModel == false &&
                   m_ModelledPeriodsTestable[hypotheses[i].s_SimilarModelled]) {
            componentsExcludedMask[hypotheses[i].s_SimilarModelled] = true;
            m_Periods.push_back(hypotheses[i].s_Period);
            periodsHypotheses.push_back(i);
        }
    }
    LOG_TRACE(<< "periods to model = " << core::CContainerPrinter::print(m_Periods)
              << ", period hypotheses = " << core::CContainerPrinter::print(periodsHypotheses)
              << ", not preconditioned = "
              << core::CContainerPrinter::print(componentsExcludedMask));

    residuals = values;
    this->removeModelledPredictions(componentsExcludedMask, residuals);

    CSignal::fitSeasonalComponentsRobust(m_Periods, m_OutlierFraction, residuals, m_Components);

    TSeasonalComponentVec period;
    TMeanAccumulatorVecVec component;

    for (std::size_t i = 0; i < m_Periods.size(); ++i) {

        std::size_t j{periodsHypotheses[i]};
        const TSizeVec& scaleSegments{hypotheses[j].s_ScaleSegments};

        for (auto scale : {scaleSegments.size() > 2, false}) {

            m_TemporaryValues = residuals;
            m_WindowIndices.resize(m_TemporaryValues.size());
            std::iota(m_WindowIndices.begin(), m_WindowIndices.end(), 0);
            this->removePredictions({m_Periods, i + 1, m_Periods.size()},
                                    {m_Components, i + 1, m_Components.size()},
                                    m_TemporaryValues);
            CSignal::restrictTo(m_Periods[i], m_TemporaryValues);
            CSignal::restrictTo(m_Periods[i], m_WindowIndices);
            period.assign(1, CSignal::seasonalComponentSummary(m_Periods[i].period()));

            if (scale) {
                auto weightedMeanScale = [this](const TSizeVec& segmentation,
                                                const TDoubleVec& scales) {
                    return TSegmentation::meanScale(segmentation, scales, [&](std::size_t k) {
                        return std::pow(
                            0.9, static_cast<double>(m_TemporaryValues.size() - k - 1));
                    });
                };
                if (this->constantScale(weightedMeanScale, period, scaleSegments, m_TemporaryValues,
                                        m_ScaledComponent, m_ComponentScales) == false) {
                    continue;
                }
            } else {
                CSignal::fitSeasonalComponents(period, m_TemporaryValues, component);
            }

            hypotheses[j].s_ModelSize =
                this->selectComponentSize(m_TemporaryValues, period[0]);
            hypotheses[j].s_InitialValues.resize(values.size());

            double overlapping{static_cast<double>(std::count_if(
                m_Periods.begin(), m_Periods.end(), [&](const auto& period_) {
                    return period_.windowed() == false ||
                           period_.s_Window == m_Periods[i].s_Window;
                }))};

            for (std::size_t k = 0; k < m_TemporaryValues.size(); ++k) {
                if (CBasicStatistics::count(m_TemporaryValues[k]) > 0.0) {
                    auto& value = CBasicStatistics::moment<0>(m_TemporaryValues[k]);
                    double prediction{
                        scale ? TSegmentation::scaleAt(k, scaleSegments, m_ComponentScales) *
                                    m_ScaledComponent[0][period[0].offset(k)]
                              : period[0].value(component[0], k)};
                    value = prediction + (value - prediction) / overlapping;
                    CBasicStatistics::moment<0>(residuals[m_WindowIndices[k]]) -= prediction;
                }
                hypotheses[j].s_InitialValues[m_WindowIndices[k]] = m_TemporaryValues[k];
            }
            break;
        }
    }

    return componentsToRemoveMask;
}

CTimeSeriesTestForSeasonality::TBoolVec
CTimeSeriesTestForSeasonality::selectModelledHypotheses(bool alreadyModelled,
                                                        THypothesisStatsVec& hypotheses) const {

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
    for (std::size_t i = 0, removedCount = 0;
         alreadyModelled == false && i < hypotheses.size();
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

    LOG_TRACE(<< "selecting from " << core::CContainerPrinter::print(hypotheses));

    // Check if there are any extra unmodelled components.
    bool change{false};
    std::size_t numberModelled{m_ModelledPeriods.size()};
    for (std::size_t i = 0; i < hypotheses.size(); ++i) {
        const auto& period = hypotheses[i].s_Period;
        std::size_t similar{this->similarModelled(period)};
        hypotheses[i].s_SimilarModelled = similar;
        change |= (similar == numberModelled) ||
                  (m_ModelledPeriods[similar].almostEqual(period, 0.05) == false);
    }
    LOG_TRACE(<< "change = " << change);

    std::ptrdiff_t excess{-m_MaximumNumberComponents};
    for (auto& hypothesis : hypotheses) {
        hypothesis.s_Model = change;
        excess += hypothesis.s_Model ? 1 : 0;
    }

    // Check if there are any existing components to evict.
    TBoolVec componentsToRemoveMask(numberModelled, false);
    for (std::size_t i = 0; i < numberModelled; ++i) {
        auto hypothesis = std::find_if(hypotheses.begin(), hypotheses.end(),
                                       [&](const auto& candidate) {
                                           return candidate.s_SimilarModelled == i;
                                       });
        componentsToRemoveMask[i] = m_ModelledPeriodsTestable[i] &&
                                    (change || hypothesis->s_DiscardingModel);
        if (componentsToRemoveMask[i]) {
            for (std::size_t j = 0; j < hypotheses.size(); ++j) {
                hypotheses[j].s_DiscardingModel |= (hypotheses[j].s_SimilarModelled == i);
            }
        }
        excess -= componentsToRemoveMask[i] ? 1 : 0;
    }
    if (alreadyModelled && std::count(componentsToRemoveMask.begin(),
                                      componentsToRemoveMask.end(), true) > 0) {

        // Ensure we still have one component per window.
        if (*std::find_if(
                boost::counting_iterator<std::size_t>(0),
                boost::counting_iterator<std::size_t>(m_ModelledPeriods.size()), [&](auto i) {
                    return m_ModelledPeriods[i].windowed() &&
                           componentsToRemoveMask[i] == false;
                }) != m_ModelledPeriods.size()) {
            for (std::size_t i = 0; i < numberModelled; ++i) {
                if (m_ModelledPeriods[i].windowed() && componentsToRemoveMask[i] &&
                    // There is no other component for this window.
                    *std::find_if(boost::counting_iterator<std::size_t>(0),
                                  boost::counting_iterator<std::size_t>(
                                      m_ModelledPeriods.size()),
                                  [&](auto j) {
                                      return m_ModelledPeriods[j].s_Window ==
                                                 m_ModelledPeriods[i].s_Window &&
                                             componentsToRemoveMask[j] == false;
                                  }) == m_ModelledPeriods.size()) {
                    componentsToRemoveMask[i] = false;
                    hypotheses[i].s_DiscardingModel = false;
                    ++excess;
                }
            }
        }

        // If we're discarding some components we reinitalise the ones we keep.
        if (std::count(componentsToRemoveMask.begin(),
                       componentsToRemoveMask.end(), true) > 0) {
            for (auto& hypothesis : hypotheses) {
                std::size_t similar{hypothesis.s_SimilarModelled};
                bool remove{componentsToRemoveMask[similar]};
                hypothesis.s_Model = m_ModelledPeriodsTestable[similar] && remove == false;
                componentsToRemoveMask[similar] = remove || hypothesis.s_Model;
            }
        }
        LOG_TRACE(<< "components to remove = "
                  << core::CContainerPrinter::print(componentsToRemoveMask));
    }
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

std::size_t CTimeSeriesTestForSeasonality::similarModelled(const TSeasonalComponent& period) const {
    std::size_t exact{*std::find_if(
        boost::counting_iterator<std::size_t>(0),
        boost::counting_iterator<std::size_t>(m_ModelledPeriods.size()), [&](const auto& j) {
            return m_ModelledPeriods[j].almostEqual(period, 0.05);
        })};
    return exact < m_ModelledPeriods.size()
               ? exact
               : *std::find_if(
                     boost::counting_iterator<std::size_t>(0),
                     boost::counting_iterator<std::size_t>(m_ModelledPeriods.size()),
                     [&](const auto& j) {
                         return m_ModelledPeriods[j].periodAlmostEqual(period, 0.05);
                     });
}

void CTimeSeriesTestForSeasonality::removeModelledPredictions(const TBoolVec& componentsExcludedMask,
                                                              TFloatMeanAccumulatorVec& values) const {
    // We've already conditioned values on any non-testable components so we
    // *always* exclude them from the prediction.
    TBoolVec mask{m_ModelledPeriodsTestable};
    for (std::size_t i = 0; i < mask.size(); ++i) {
        mask[i] = mask[i] == false || componentsExcludedMask[i];
    }

    auto bucketPredictor = CSignal::bucketPredictor(
        [&](core_t::TTime time) { return m_ModelledPredictor(time, mask); }, m_BucketsStartTime,
        m_BucketLength, m_ValuesStartTime - m_BucketsStartTime, m_SampleInterval);

    for (std::size_t i = 0; i < values.size(); ++i) {
        CBasicStatistics::moment<0>(values[i]) -=
            bucketPredictor(m_BucketLength * static_cast<core_t::TTime>(i));
    }
}

void CTimeSeriesTestForSeasonality::removeDiscontinuities(const TSizeVec& modelTrendSegments,
                                                          TFloatMeanAccumulatorVec& values) const {
    if (modelTrendSegments.size() > 2) {
        values = TSegmentation::removePiecewiseLinearDiscontinuities(
            std::move(values), modelTrendSegments, m_OutlierFraction);
    }
}

bool CTimeSeriesTestForSeasonality::constantScale(const TConstantScale& scale,
                                                  const TSeasonalComponentVec& periods,
                                                  const TSizeVec& scaleSegments,
                                                  TFloatMeanAccumulatorVec& values,
                                                  TDoubleVecVec& components,
                                                  TDoubleVec& scales) const {
    if (scaleSegments.size() > 2) {
        values = TSegmentation::constantScalePiecewiseLinearScaledSeasonal(
            values, periods, scaleSegments, scale, m_OutlierFraction, components, scales);
        return true;
    }
    return false;
}

CTimeSeriesTestForSeasonality::TVarianceStats
CTimeSeriesTestForSeasonality::residualVarianceStats(const TFloatMeanAccumulatorVec& values) const {
    auto result = CSignal::residualVarianceStats(values, m_Periods, m_Components);
    result.s_ResidualVariance += m_EpsVariance;
    result.s_TruncatedResidualVariance += m_EpsVariance;
    return result;
}

CTimeSeriesTestForSeasonality::TMeanVarAccumulator
CTimeSeriesTestForSeasonality::truncatedMoments(double outlierFraction,
                                                const TFloatMeanAccumulatorVec& residuals,
                                                const TTransform& transform) const {
    double cutoff{std::numeric_limits<double>::max()};
    std::size_t count{CSignal::countNotMissing(residuals)};
    std::size_t numberOutliers{
        static_cast<std::size_t>(outlierFraction * static_cast<double>(count) + 0.5)};
    if (numberOutliers > 0) {
        m_Outliers.clear();
        m_Outliers.resize(numberOutliers);
        for (const auto& value : residuals) {
            if (CBasicStatistics::count(value) > 0.0) {
                m_Outliers.add(std::fabs(transform(value)));
            }
        }
        cutoff = m_Outliers.biggest();
        count -= m_Outliers.count();
    }
    LOG_TRACE(<< "cutoff = " << cutoff << ", count = " << count);

    TMeanVarAccumulator moments;
    for (const auto& value : residuals) {
        if (CBasicStatistics::count(value) > 0.0 && std::fabs(transform(value)) < cutoff) {
            moments.add(transform(value));
        }
    }
    if (numberOutliers > 0) {
        moments.add(cutoff, static_cast<double>(count) - CBasicStatistics::count(moments));
    }
    CBasicStatistics::moment<1>(moments) += m_EpsVariance;

    return moments;
}

std::size_t CTimeSeriesTestForSeasonality::numberTrendParameters(std::size_t numberTrendSegments) const {
    return numberTrendSegments == 1 ? 3 : 2 * numberTrendSegments;
}

bool CTimeSeriesTestForSeasonality::includesNewComponents(const TSeasonalComponentVec& periods) const {
    return periods.size() > 0 && this->includesPermittedPeriod(periods) &&
           this->alreadyModelled(periods) == false;
}

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
    // We don't try to test components which aren't testable or nearly match
    // modelled components which aren't testable.
    periods.erase(
        std::remove_if(periods.begin(), periods.end(),
                       [this](const auto& period) {
                           std::size_t similar{this->similarModelled(period)};
                           return canTestPeriod(m_Values, buckets(m_BucketLength, m_MinimumPeriod),
                                                period) == false ||
                                  (similar < m_ModelledPeriodsTestable.size() &&
                                   m_ModelledPeriodsTestable[similar] == false);
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
    return period.windowed() && period.s_Window == this->weekendWindow();
}

bool CTimeSeriesTestForSeasonality::isWeekday(const TSeasonalComponent& period) const {
    return period.windowed() && period.s_Window == this->weekdayWindow();
}

bool CTimeSeriesTestForSeasonality::permittedPeriod(const TSeasonalComponent& period) const {
    return m_MinimumPeriod == 0 ||
           static_cast<core_t::TTime>(period.s_WindowRepeat) * m_BucketLength >= m_MinimumPeriod;
}

bool CTimeSeriesTestForSeasonality::includesPermittedPeriod(const TSeasonalComponentVec& periods) const {
    return m_MinimumPeriod == 0 ||
           std::find_if(periods.begin(), periods.end(), [this](const auto& period) {
               return this->permittedPeriod(period);
           }) != periods.end();
}

std::string CTimeSeriesTestForSeasonality::annotationText(const TSeasonalComponent& period) const {
    std::ostringstream result;
    result << "Detected seasonal component with period ";
    if (period.s_Period == this->day()) {
        result << core::CTimeUtils::durationToString(core::constants::DAY);
    } else if (period.s_Period == this->week()) {
        result << core::CTimeUtils::durationToString(core::constants::WEEK);
    } else if (period.s_Period == this->year()) {
        result << core::CTimeUtils::durationToString(core::constants::YEAR);
    } else {
        result << core::CTimeUtils::durationToString(
            m_BucketLength * static_cast<core_t::TTime>(period.s_Period));
    }
    result << (this->isWeekend(period) ? " (weekend)"
                                       : (this->isWeekday(period) ? " (weekdays)" : ""));
    return result.str();
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
                                                  std::size_t minimumPeriod,
                                                  const TSeasonalComponent& period) {
    return periodTooShortToTest(minimumPeriod, period) == false &&
           periodTooLongToTest(values, period) == false;
}

bool CTimeSeriesTestForSeasonality::periodTooLongToTest(const TFloatMeanAccumulatorVec& values,
                                                        const TSeasonalComponent& period) {
    std::size_t range{observedRange(values)};
    std::size_t gap{longestGap(values)};
    return 190 * period.s_WindowRepeat + 100 * gap > 100 * range;
}

bool CTimeSeriesTestForSeasonality::periodTooShortToTest(std::size_t minimumPeriod,
                                                         const TSeasonalComponent& period) {
    return period.s_Period < std::max(minimumPeriod, std::size_t{2});
}

std::size_t CTimeSeriesTestForSeasonality::observedRange(const TFloatMeanAccumulatorVec& values) {
    std::size_t begin;
    std::size_t end;
    std::tie(begin, end) = observedInterval(values);
    return end - begin;
}

std::size_t CTimeSeriesTestForSeasonality::longestGap(const TFloatMeanAccumulatorVec& values) {
    std::size_t result{0};
    std::size_t begin;
    std::size_t end;
    std::tie(begin, end) = observedInterval(values);
    for (std::size_t i = begin, j = end; j < end; i = j) {
        for (++j; j < end; ++j) {
            if (CBasicStatistics::count(values[j]) > 0.0) {
                break;
            }
        }
        result = std::max(result, j - i - 1);
    }
    return result;
}

CTimeSeriesTestForSeasonality::TSizeSizePr
CTimeSeriesTestForSeasonality::observedInterval(const TFloatMeanAccumulatorVec& values) {
    std::size_t begin{0};
    std::size_t end{values.size()};
    std::size_t size{values.size()};
    for (/**/; begin < size && CBasicStatistics::count(values[begin]) == 0.0; ++begin) {
    }
    for (/**/; end > begin && CBasicStatistics::count(values[end - 1]) == 0.0; --end) {
    }
    return {begin, end};
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
        if (CBasicStatistics::count(values[i]) > 0.0) {
            CBasicStatistics::moment<0>(values[i]) -= predictor(i);
        }
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
    if (amplitude == 0.0 || m_Count == 0) {
        return 1.0;
    }
    double twoTailPValue{2.0 * CTools::safeCdf(normal, -amplitude)};
    if (twoTailPValue <= 0.0) {
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

void CTimeSeriesTestForSeasonality::SHypothesisStats::testExplainedVariance(
    const CTimeSeriesTestForSeasonality& params,
    const TVarianceStats& H0) {

    auto H1 = params.residualVarianceStats(params.m_ValuesToTest);

    s_FractionNotMissing = static_cast<double>(H1.s_NumberParameters) /
                           static_cast<double>(params.m_Components[0].size());
    s_ResidualVariance = H1.s_ResidualVariance;
    s_ExplainedVariance = CBasicStatistics::maximumLikelihoodVariance(
        std::accumulate(params.m_Components[0].begin(), params.m_Components[0].end(),
                        TMeanVarAccumulator{}, [](auto result, const auto& value) {
                            if (CBasicStatistics::count(value) > 0.0) {
                                result.add(CBasicStatistics::mean(value));
                            }
                            return result;
                        }));
    s_NumberParametersToExplainVariance = H1.s_NumberParameters + s_NumberScaleSegments - 1;
    s_ExplainedVariancePValue = CSignal::nestedDecompositionPValue(H0, H1);
    LOG_TRACE(<< "fraction not missing = " << s_FractionNotMissing);
    LOG_TRACE(<< H1.print() << " vs " << H0.print());
    LOG_TRACE(<< "p-value = " << s_ExplainedVariancePValue);
}

void CTimeSeriesTestForSeasonality::SHypothesisStats::testAutocorrelation(
    const CTimeSeriesTestForSeasonality& params) {

    CSignal::TFloatMeanAccumulatorCRng valuesToTestAutocorrelation{
        params.m_ValuesToTest, 0,
        CIntegerTools::floor(params.m_ValuesToTest.size(), params.m_Periods[0].period())};

    double autocorrelations[]{
        CSignal::cyclicAutocorrelation( // Normal
            params.m_Periods[0], valuesToTestAutocorrelation,
            [](const TFloatMeanAccumulator& value) {
                return CBasicStatistics::mean(value);
            },
            [](const TFloatMeanAccumulator& value) {
                return CBasicStatistics::count(value);
            },
            params.m_EpsVariance),
        CSignal::cyclicAutocorrelation( // Not reweighting outliers
            params.m_Periods[0], valuesToTestAutocorrelation,
            [](const TFloatMeanAccumulator& value) {
                return CBasicStatistics::mean(value);
            },
            [](const TFloatMeanAccumulator&) { return 1.0; }, params.m_EpsVariance),
        CSignal::cyclicAutocorrelation( // Absolute values
            params.m_Periods[0], valuesToTestAutocorrelation,
            [](const TFloatMeanAccumulator& value) {
                return std::fabs(CBasicStatistics::mean(value));
            },
            [](const TFloatMeanAccumulator& value) {
                return CBasicStatistics::count(value);
            },
            params.m_EpsVariance),
        CSignal::cyclicAutocorrelation( // Not reweighting outliers and absolute values
            params.m_Periods[0], valuesToTestAutocorrelation,
            [](const TFloatMeanAccumulator& value) {
                return std::fabs(CBasicStatistics::mean(value));
            },
            [](const TFloatMeanAccumulator&) { return 1.0; }, params.m_EpsVariance)};
    LOG_TRACE(<< "autocorrelations = " << core::CContainerPrinter::print(autocorrelations));

    s_Autocorrelation = *std::max_element(std::begin(autocorrelations),
                                          std::begin(autocorrelations) + 2);
    s_AutocorrelationUpperBound = *std::max_element(std::begin(autocorrelations),
                                                    std::end(autocorrelations));
    LOG_TRACE(<< "autocorrelation = " << s_Autocorrelation
              << ", autocorrelation upper bound = " << s_AutocorrelationUpperBound);
}

void CTimeSeriesTestForSeasonality::SHypothesisStats::testAmplitude(const CTimeSeriesTestForSeasonality& params) {

    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

    s_SeenSufficientDataToTestAmplitude = CMinAmplitude::seenSufficientDataToTestAmplitude(
        observedRange(params.m_ValuesToTest), params.m_Periods[0].s_Period);
    if (s_SeenSufficientDataToTestAmplitude == false) {
        return;
    }

    double level{CBasicStatistics::mean(std::accumulate(
        params.m_ValuesToTest.begin(), params.m_ValuesToTest.end(), TMeanAccumulator{},
        [](TMeanAccumulator partialLevel, const TFloatMeanAccumulator& value) {
            partialLevel.add(CBasicStatistics::mean(value), CBasicStatistics::count(value));
            return partialLevel;
        }))};

    params.m_Amplitudes.assign(params.m_Periods[0].period(),
                               {params.m_ValuesToTest.size(), s_MeanNumberRepeats, level});
    for (std::size_t i = 0; i < params.m_ValuesToTest.size(); ++i) {
        if (params.m_Periods[0].contains(i)) {
            params.m_Amplitudes[params.m_Periods[0].offset(i)].add(
                i, params.m_ValuesToTest[i]);
        }
    }

    double pValue{1.0};
    if (s_ResidualVariance <= 0.0) {
        pValue = std::find_if(params.m_Amplitudes.begin(), params.m_Amplitudes.end(),
                              [](const auto& amplitude) {
                                  return amplitude.amplitude() > 0.0;
                              }) != params.m_Amplitudes.end()
                     ? 0.0
                     : 1.0;
    } else {
        boost::math::normal normal(0.0, std::sqrt(s_ResidualVariance));
        for (const auto& amplitude : params.m_Amplitudes) {
            if (amplitude.amplitude() >= 2.0 * boost::math::standard_deviation(normal)) {
                pValue = std::min(pValue, amplitude.significance(normal));
            }
        }
    }

    s_AmplitudePValue = CTools::oneMinusPowOneMinusX(
        pValue,
        static_cast<double>(std::count_if(
            params.m_Amplitudes.begin(), params.m_Amplitudes.end(),
            [](const auto& amplitude) { return amplitude.amplitude() > 0.0; })));
    LOG_TRACE(<< "amplitude p-value = " << s_AmplitudePValue);
}

CFuzzyTruthValue CTimeSeriesTestForSeasonality::SHypothesisStats::varianceTestResult(
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
    double windowRepeatsPerSegment{
        segments > 1 ? s_WindowRepeats / static_cast<double>(segments) : 2.0};
    double logPValue{std::log(std::max(s_ExplainedVariancePValue,
                                       std::numeric_limits<double>::min()))};
    return fuzzyGreaterThan(repeatsPerSegment / minimumRepeatsPerSegment, 1.0, 0.3) &&
           fuzzyGreaterThan(std::min(repeatsPerSegment / 2.0, 1.0), 1.0, 0.1) &&
           fuzzyGreaterThan(std::min(windowRepeatsPerSegment / 2.0, 1.0), 1.0, 0.1) &&
           fuzzyLessThan(std::max(s_LeastCommonRepeat / 0.5, 1.0), 1.0, 0.5) &&
           fuzzyGreaterThan(s_FractionNotMissing, 1.0, 0.5) &&
           fuzzyGreaterThan(logPValue / logSignificantPValue, 1.0, 0.1) &&
           fuzzyGreaterThan(std::max(logPValue / logVerySignificantPValue, 1.0), 1.0, 0.1) &&
           fuzzyGreaterThan(s_Autocorrelation / mediumAutocorrelation, 1.0, 0.2) &&
           fuzzyGreaterThan(std::min(s_Autocorrelation / lowAutocorrelation, 1.0), 1.0, 0.1) &&
           fuzzyGreaterThan(std::max(s_Autocorrelation / highAutocorrelation, 1.0), 1.0, 0.1);
}

CFuzzyTruthValue CTimeSeriesTestForSeasonality::SHypothesisStats::amplitudeTestResult(
    const CTimeSeriesTestForSeasonality& params) const {
    if (s_SeenSufficientDataToTestAmplitude == false) {
        return CFuzzyTruthValue::OR_UNDETERMINED_VALUE;
    }

    // Compare with the discussion in testVariance.

    double minimumRepeatsPerSegment{params.m_MinimumRepeatsPerSegmentToTestAmplitude};
    double lowAutocorrelation{params.m_LowAutocorrelation};
    double logSignificantPValue{std::log(params.m_SignificantPValue)};
    double logVerySignificantPValue{std::log(params.m_VerySignificantPValue)};
    std::size_t segments{std::max(s_NumberTrendSegments, std::size_t{1}) +
                         s_NumberScaleSegments - 1};
    double repeatsPerSegment{s_MeanNumberRepeats / static_cast<double>(segments)};
    double windowRepeatsPerSegment{
        segments > 1 ? s_WindowRepeats / static_cast<double>(segments) : 2.0};
    double autocorrelation{s_AutocorrelationUpperBound};
    double logPValue{
        std::log(std::max(s_AmplitudePValue, std::numeric_limits<double>::min()))};
    return fuzzyGreaterThan(repeatsPerSegment / minimumRepeatsPerSegment, 1.0, 0.1) &&
           fuzzyGreaterThan(std::min(repeatsPerSegment / 2.0, 1.0), 1.0, 0.1) &&
           fuzzyGreaterThan(std::min(windowRepeatsPerSegment / 2.0, 1.0), 1.0, 0.1) &&
           fuzzyLessThan(std::max(s_LeastCommonRepeat / 0.5, 1.0), 1.0, 0.5) &&
           fuzzyGreaterThan(s_FractionNotMissing, 1.0, 0.5) &&
           fuzzyGreaterThan(autocorrelation / lowAutocorrelation, 1.0, 0.2) &&
           fuzzyGreaterThan(logPValue / logSignificantPValue, 1.0, 0.1) &&
           fuzzyGreaterThan(std::max(logPValue / logVerySignificantPValue, 1.0), 1.0, 0.1);
}

bool CTimeSeriesTestForSeasonality::SHypothesisStats::isBetter(const SHypothesisStats& other) const {
    // The truth value alone is not discriminative when all conditions are met.
    double min{std::numeric_limits<double>::min()};
    return COrderings::lexicographical_compare(
        other.s_Truth.boolean(),
        1.0 * std::log(std::max(other.s_Truth.value(), min)) +
            0.5 * std::log(-std::log(std::max(other.s_ExplainedVariancePValue, min))),
        s_Truth.boolean(),
        1.0 * std::log(std::max(s_Truth.value(), min)) +
            0.5 * std::log(-std::log(std::max(s_ExplainedVariancePValue, min))));
}

bool CTimeSeriesTestForSeasonality::SHypothesisStats::evict(const CTimeSeriesTestForSeasonality& params,
                                                            std::size_t modelledIndex) const {
    std::size_t range{params.m_ModelledPeriods[modelledIndex].fractionInWindow(
        observedRange(params.m_Values))};
    std::size_t period{params.m_ModelledPeriods[modelledIndex].period()};
    return params.m_ModelledPeriodsTestable[modelledIndex] &&
           3 * period >= params.m_ModelledPeriodsSizes[modelledIndex] &&
           CMinAmplitude::seenSufficientDataToTestAmplitude(range, period) &&
           s_ExplainedVariancePValue > params.m_PValueToEvict &&
           s_AmplitudePValue > params.m_PValueToEvict;
}

double CTimeSeriesTestForSeasonality::SHypothesisStats::weight() const {
    return s_ExplainedVariance *
           static_cast<double>(s_Period.s_Window.second - s_Period.s_Window.first) /
           static_cast<double>(s_Period.s_WindowRepeat);
}

std::string CTimeSeriesTestForSeasonality::SHypothesisStats::print() const {
    return s_Period.print();
}

bool CTimeSeriesTestForSeasonality::SModel::isTestable() const {
    for (const auto& hypothesis : s_Hypotheses) {
        if (hypothesis.s_IsTestable == false) {
            return false;
        }
    }
    return true;
}

bool CTimeSeriesTestForSeasonality::SModel::isNull() const {
    return s_Hypotheses.empty() &&
           CBasicStatistics::count(s_ResidualMoments) > this->numberParameters();
}

bool CTimeSeriesTestForSeasonality::SModel::isAlternative() const {
    return this->isNull() == false &&
           CBasicStatistics::count(s_ResidualMoments) > this->numberParameters();
}

double CTimeSeriesTestForSeasonality::SModel::componentsSimilarity() const {
    if (s_AlreadyModelled == false) {
        return 0.0;
    }
    for (const auto& hypothesis : s_Hypotheses) {
        if (hypothesis.s_DiscardingModel) {
            return 0.5;
        }
    }
    return 1.0;
}

double CTimeSeriesTestForSeasonality::SModel::pValue(const SModel& H0,
                                                     double minimumRelativeTruncatedVariance,
                                                     double unexplainedVariance) const {

    double n[]{CBasicStatistics::count(H0.s_ResidualMoments),
               CBasicStatistics::count(H0.s_TruncatedResidualMoments)};
    double v0[]{CBasicStatistics::maximumLikelihoodVariance(H0.s_ResidualMoments) + unexplainedVariance,
                CBasicStatistics::maximumLikelihoodVariance(H0.s_TruncatedResidualMoments) +
                    unexplainedVariance};
    double v1[]{std::max(CBasicStatistics::maximumLikelihoodVariance(s_ResidualMoments),
                         std::numeric_limits<double>::epsilon() * v0[0]) +
                    unexplainedVariance,
                std::max(CBasicStatistics::maximumLikelihoodVariance(s_TruncatedResidualMoments),
                         std::numeric_limits<double>::epsilon() * v0[1]) +
                    unexplainedVariance};
    double df0[]{n[0] - H0.numberParameters(), n[1] - H0.numberParameters()};
    double df1[]{CBasicStatistics::count(s_ResidualMoments) - this->numberParameters(),
                 CBasicStatistics::count(s_TruncatedResidualMoments) -
                     this->numberParameters()};
    v0[1] += minimumRelativeTruncatedVariance * v0[0];
    v1[1] += minimumRelativeTruncatedVariance * v1[0];

    return std::min(rightTailFTest(v0[0] / df0[0], v1[0] / df1[0], df0[0], df1[0]),
                    rightTailFTest(v0[1] / df0[1], v1[1] / df1[1], df0[1], df1[1]));
}

double CTimeSeriesTestForSeasonality::SModel::logPValueProxy(const SModel& H0) const {
    // We use minus the number of standard deviations above the mean of the F-distribution.
    double v0[]{CBasicStatistics::maximumLikelihoodVariance(H0.s_ResidualMoments),
                CBasicStatistics::maximumLikelihoodVariance(H0.s_TruncatedResidualMoments)};
    double v1[]{std::max(CBasicStatistics::maximumLikelihoodVariance(s_ResidualMoments),
                         1e-3 * v0[0]),
                std::max(CBasicStatistics::maximumLikelihoodVariance(s_TruncatedResidualMoments),
                         1e-3 * v0[1])};
    double df0[]{CBasicStatistics::count(H0.s_ResidualMoments) - H0.numberParameters(),
                 CBasicStatistics::count(H0.s_TruncatedResidualMoments) -
                     H0.numberParameters()};
    double df1[]{CBasicStatistics::count(s_ResidualMoments) - this->numberParameters(),
                 CBasicStatistics::count(s_TruncatedResidualMoments) -
                     this->numberParameters()};

    double result{0.0};
    for (auto i : {0, 1}) {
        // d2 needs to be > 4 for finite variance. We can happily use 0.0 for the log(p-value)
        // proxy if this condition is not satisfied.
        if (df1[i] > 4.0 && df0[i] > 0.0) {
            boost::math::fisher_f f{df0[i], df1[i]};
            double mean{boost::math::mean(f)};
            double sd{boost::math::standard_deviation(f)};
            result = std::max(result, ((v0[i] * df1[i]) / (v1[i] * df0[i]) - mean) / sd);
        }
    }
    return -result;
};

CTimeSeriesTestForSeasonality::TVector2x1
CTimeSeriesTestForSeasonality::SModel::explainedVariancePerParameter(double variance,
                                                                     double truncatedVariance) const {
    TVector2x1 explainedVariance;
    explainedVariance(0) =
        variance - CBasicStatistics::maximumLikelihoodVariance(s_ResidualMoments);
    explainedVariance(1) = truncatedVariance - CBasicStatistics::maximumLikelihoodVariance(
                                                   s_TruncatedResidualMoments);
    TVector2x1 result{0.0};
    double Z{0.0};
    for (const auto& hypothesis : s_Hypotheses) {
        if (hypothesis.s_Model || hypothesis.s_DiscardingModel == false) {
            double weight{hypothesis.weight()};
            result += weight * explainedVariance /
                      static_cast<double>(hypothesis.s_NumberParametersToExplainVariance);
            Z += weight;
        }
    }
    return max(result == TVector2x1{0.0} ? result : result / Z,
               TVector2x1{std::numeric_limits<double>::min()});
}

double CTimeSeriesTestForSeasonality::SModel::numberParameters() const {
    return static_cast<double>(std::accumulate(
        s_Hypotheses.begin(), s_Hypotheses.end(), s_NumberTrendParameters + 1,
        [this](std::size_t partialNumber, const auto& hypothesis) {
            auto i = std::find_if(
                s_Hypotheses.begin(), s_Hypotheses.end(), [&](const auto& hypothesis_) {
                    return hypothesis.s_Period.nested(hypothesis_.s_Period);
                });
            return partialNumber +
                   ((hypothesis.s_Model || hypothesis.s_DiscardingModel == false) &&
                            i == s_Hypotheses.end()
                        ? hypothesis.s_NumberParametersToExplainVariance
                        : 0);
        }));
}

double CTimeSeriesTestForSeasonality::SModel::targetModelSize() const {
    return static_cast<double>(std::accumulate(
        s_Hypotheses.begin(), s_Hypotheses.end(), 0, [&](auto partialSize, const auto& hypothesis) {
            if (hypothesis.s_Model) {
                partialSize += std::max(hypothesis.s_ModelSize, s_Params->m_MinimumModelSize);
            } else if (hypothesis.s_DiscardingModel == false) {
                partialSize += s_Params->m_ModelledPeriodsSizes[hypothesis.s_SimilarModelled];
            }
            return partialSize;
        }));
}

double CTimeSeriesTestForSeasonality::SModel::numberScalings() const {
    std::size_t segments{0};
    for (const auto& hypothesis : s_Hypotheses) {
        if (hypothesis.s_Model || hypothesis.s_DiscardingModel == false) {
            segments += hypothesis.s_NumberScaleSegments - 1;
        }
    }
    return static_cast<double>(segments);
}

double CTimeSeriesTestForSeasonality::SModel::autocorrelation() const {
    double result{0.0};
    double Z{0.0};
    for (const auto& hypothesis : s_Hypotheses) {
        if (hypothesis.s_Model || hypothesis.s_DiscardingModel == false) {
            double weight{hypothesis.weight()};
            result += weight * hypothesis.s_Autocorrelation;
            Z += weight;
        }
    }
    return result == 0.0 ? 0.0 : result / Z;
}

double CTimeSeriesTestForSeasonality::SModel::leastCommonRepeat() const {
    std::size_t result{1};
    for (const auto& hypothesis : s_Hypotheses) {
        if (hypothesis.s_Model || hypothesis.s_DiscardingModel == false) {
            result = CIntegerTools::lcm(result, hypothesis.s_Period.s_WindowRepeat);
        }
    }
    return static_cast<double>(result) /
           static_cast<double>(observedRange(s_Params->m_Values));
}
}
}
