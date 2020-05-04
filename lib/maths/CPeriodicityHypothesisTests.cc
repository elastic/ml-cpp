/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CPeriodicityHypothesisTests.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CPersistUtils.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/Constants.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CLeastSquaresOnlineRegression.h>
#include <maths/CLeastSquaresOnlineRegressionDetail.h>
#include <maths/CSeasonalTime.h>
#include <maths/CSignal.h>
#include <maths/CStatisticalTests.h>
#include <maths/CTimeSeriesSegmentation.h>
#include <maths/CTools.h>
#include <maths/Constants.h>

#include <boost/circular_buffer.hpp>
#include <boost/math/distributions/binomial.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/fisher_f.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/range.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>
#include <vector>

namespace ml {
namespace maths {
namespace {

using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;
using TTimeVec = std::vector<core_t::TTime>;
using TSizeSizePr = std::pair<std::size_t, std::size_t>;
using TSizeSizePr2Vec = core::CSmallVector<TSizeSizePr, 2>;
using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;
using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
using TMeanVarAccumulatorVec = std::vector<TMeanVarAccumulator>;
using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
using TTimeTimePr = std::pair<core_t::TTime, core_t::TTime>;
using TTimeTimePr2Vec = core::CSmallVector<TTimeTimePr, 2>;
using TTimeTimePrMeanVarAccumulatorPr = std::pair<TTimeTimePr, TMeanVarAccumulator>;

//! The confidence interval used for test statistic values.
const double CONFIDENCE_INTERVAL{80.0};
//! The soft minimum number of repeats which we'll use to test
//! for periodicity using the variance test.
const double MINIMUM_REPEATS_TO_TEST_VARIANCE{3.0};
//! The minimum number of repeats which we'll use to test for
//! periodicity using the amplitude test.
const std::size_t MINIMUM_REPEATS_TO_TEST_AMPLITUDE{4};
//! A high priority for components we want to take precendence.
double HIGH_PRIORITY{2.0};

//! \brief Fuzzy logical expression with multiplicative AND.
//!
//! DESCRIPTION:
//! This isn't strictly a fuzzy logical expression since we don't ensure
//! that the range of truth values is [0,1]. In fact, we arrange for TRUE
//! to correspond to value > 1. We roll in an implicit threshold such that
//! if individual conditions have values > 0.5 then the expression (just)
//! maps to true.
class CFuzzyExpression {
public:
    explicit CFuzzyExpression(double value = 0.0) : m_Value{value} {}

    operator bool() const { return m_Value > 1.0; }
    bool operator<(const CFuzzyExpression& rhs) const {
        return m_Value < rhs.m_Value;
    }

    double truthValue() const { return m_Value; }

    friend CFuzzyExpression operator&&(const CFuzzyExpression& lhs,
                                       const CFuzzyExpression& rhs) {
        return CFuzzyExpression{lhs.m_Value * rhs.m_Value};
    }

private:
    double m_Value;
};

//! Fuzzy check if \p value is greater than \p threshold.
CFuzzyExpression softGreaterThan(double value, double threshold, double margin) {
    return CFuzzyExpression{2.0 * CTools::logisticFunction(value, margin, threshold, +1.0)};
}

//! Fuzzy check if \p value is less than \p threshold.
CFuzzyExpression softLessThan(double value, double threshold, double margin) {
    return CFuzzyExpression{2.0 * CTools::logisticFunction(value, margin, threshold, -1.0)};
}

//! \brief Accumulates the minimum amplitude.
class CMinAmplitude {
public:
    CMinAmplitude(std::size_t n, double level)
        : m_Level(level), m_Count(0),
          m_Min(std::max(n, MINIMUM_REPEATS_TO_TEST_AMPLITUDE)),
          m_Max(std::max(n, MINIMUM_REPEATS_TO_TEST_AMPLITUDE)) {}

    void add(double x, double n) {
        if (n > 0.0) {
            ++m_Count;
            m_Min.add(x - m_Level);
            m_Max.add(x - m_Level);
        }
    }

    double amplitude() const {
        if (this->count() >= MINIMUM_REPEATS_TO_TEST_AMPLITUDE) {
            return std::max(std::max(-m_Min.biggest(), 0.0),
                            std::max(m_Max.biggest(), 0.0));
        }
        return 0.0;
    }

    double significance(const boost::math::normal& normal) const {
        if (this->count() < MINIMUM_REPEATS_TO_TEST_AMPLITUDE) {
            return 1.0;
        }
        double F{2.0 * CTools::safeCdf(normal, -this->amplitude())};
        if (F == 0.0) {
            return 0.0;
        }
        double n{static_cast<double>(this->count())};
        boost::math::binomial binomial(static_cast<double>(m_Count), F);
        return CTools::safeCdfComplement(binomial, n - 1.0);
    }

private:
    using TMinAccumulator = CBasicStatistics::COrderStatisticsHeap<double>;
    using TMaxAccumulator =
        CBasicStatistics::COrderStatisticsHeap<double, std::greater<double>>;

private:
    std::size_t count() const { return m_Min.count(); }

private:
    //! The mean of the trend.
    double m_Level;
    //! The total count of values added.
    std::size_t m_Count;
    //! The smallest values.
    TMinAccumulator m_Min;
    //! The largest values.
    TMaxAccumulator m_Max;
};

using TMinAmplitudeVec = std::vector<CMinAmplitude>;

//! \brief Holds the relevant summary for choosing between alternative
//! (non-nested) hypotheses.
struct SHypothesisSummary {
    double s_V;
    double s_R;
    double s_DF;
    double s_VarianceThreshold;
    double s_AutocorrelationThreshold;
    double s_TrendSegments;
    double s_ScaleSegments;
    CPeriodicityHypothesisTestsResult s_H;
};

using THypothesisSummaryVec = std::vector<SHypothesisSummary>;

enum EDiurnalComponents {
    E_WeekendDay,
    E_WeekendWeek,
    E_WeekdayDay,
    E_WeekdayWeek,
    E_Day,
    E_Week,
};

using TComponent4Vec = core::CSmallVector<EDiurnalComponents, 4>;

enum EThreshold { E_LowThreshold, E_HighThreshold };

// Copy constants into scope.
const core_t::TTime DAY{core::constants::DAY};
const core_t::TTime WEEKEND{core::constants::WEEKEND};
const core_t::TTime WEEK{core::constants::WEEK};
//! The periods of the diurnal components.
const core_t::TTime DIURNAL_PERIODS[]{DAY, WEEK};
//! The weekend/day windows.
const TTimeTimePr DIURNAL_WINDOWS[]{{0, WEEKEND}, {WEEKEND, WEEK}, {0, WEEK}};
//! The names of the the diurnal periodic components.
const std::string DIURNAL_COMPONENT_NAMES[] = {
    "weekend daily",  "weekend weekly", "weekday daily",
    "weekday weekly", "daily",          "weekly"};

//! Fit and remove a linear trend from \p values.
void removeLinearTrend(TFloatMeanAccumulatorVec& values) {
    using TRegression = CLeastSquaresOnlineRegression<1, double>;
    TRegression trend;
    double dt{10.0 / static_cast<double>(values.size())};
    double time{dt / 2.0};
    for (const auto& value : values) {
        trend.add(time, CBasicStatistics::mean(value), CBasicStatistics::count(value));
        time += dt;
    }
    time = dt / 2.0;
    for (auto& value : values) {
        if (CBasicStatistics::count(value) > 0.0) {
            CBasicStatistics::moment<0>(value) -= trend.predict(time);
        }
        time += dt;
    }
}

//! Get the correction to apply to the partition variance test
//! statistic if there are \p bucketsPerRepeat buckets in the
//! in one repeat of the partitioning pattern.
double weekendPartitionVarianceCorrection(std::size_t bucketsPerWeek) {
    static const std::size_t BUCKETS_PER_WEEK[]{7, 14, 21, 28, 42, 56, 84, 168};
    static const double CORRECTIONS[]{1.0,  1.0,  1.0,  1.12,
                                      1.31, 1.31, 1.31, 1.31};
    std::ptrdiff_t index{std::min(
        std::lower_bound(std::begin(BUCKETS_PER_WEEK), std::end(BUCKETS_PER_WEEK), bucketsPerWeek) -
            std::begin(BUCKETS_PER_WEEK),
        std::ptrdiff_t(boost::size(BUCKETS_PER_WEEK) - 1))};
    return CORRECTIONS[index];
}

//! Compute the \p percentage % variance for a chi-squared random
//! variance with \p df degrees of freedom.
double varianceAtPercentile(double variance, double df, double percentage) {
    try {
        boost::math::chi_squared chi(df);
        return boost::math::quantile(chi, percentage / 100.0) / df * variance;
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Bad input: " << e.what() << ", df = " << df
                  << ", percentage = " << percentage);
    }
    return variance;
}

//! Compute the \p percentage % autocorrelation for a F distributed
//! random autocorrelation with parameters \p n - 1 and \p n - 1.
double autocorrelationAtPercentile(double autocorrelation, double n, double percentage) {
    try {
        boost::math::fisher_f f(n - 1.0, n - 1.0);
        return boost::math::quantile(f, percentage / 100.0) * autocorrelation;
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Bad input: " << e.what() << ", n = " << n
                  << ", percentage = " << percentage);
    }
    return autocorrelation;
}

//! Get the length of the \p window.
template<typename T>
T length(const std::pair<T, T>& window) {
    return window.second - window.first;
}

//! Get the total length of the \p windows.
template<typename T>
T length(const core::CSmallVector<std::pair<T, T>, 2>& windows) {
    return std::accumulate(windows.begin(), windows.end(), 0,
                           [](core_t::TTime length_, const TTimeTimePr& window) {
                               return length_ + length(window);
                           });
}

//! Get the length of \p buckets.
template<typename T>
core_t::TTime length(const T& buckets, core_t::TTime bucketLength) {
    return static_cast<core_t::TTime>(buckets.size()) * bucketLength;
}

//! Compute the windows at repeat \p repeat with length \p length.
TTimeTimePr2Vec calculateWindows(core_t::TTime startOfWeek,
                                 core_t::TTime window,
                                 core_t::TTime repeat,
                                 const TTimeTimePr& interval) {
    core_t::TTime a{startOfWeek + interval.first};
    core_t::TTime b{startOfWeek + window};
    core_t::TTime l{length(interval)};
    TTimeTimePr2Vec result;
    result.reserve((b - a) / repeat);
    for (core_t::TTime time = a; time < b; time += repeat) {
        result.emplace_back(time, time + l);
    }
    return result;
}

//! Get the index ranges corresponding to \p windows.
std::size_t calculateIndexWindows(const TTimeTimePr2Vec& windows,
                                  core_t::TTime bucketLength,
                                  TSizeSizePr2Vec& result) {
    std::size_t l(0);
    result.reserve(windows.size());
    for (const auto& window : windows) {
        core_t::TTime a{window.first / bucketLength};
        core_t::TTime b{window.second / bucketLength};
        result.emplace_back(a, b);
        l += b - a;
    }
    return l;
}

//! Compute the projection of \p values to \p windows.
void project(const TFloatMeanAccumulatorVec& values,
             const TTimeTimePr2Vec& windows_,
             core_t::TTime bucketLength,
             TFloatMeanAccumulatorVec& result) {
    result.clear();
    if (values.size() > 0) {
        TSizeSizePr2Vec windows;
        calculateIndexWindows(windows_, bucketLength, windows);
        result.reserve(length(windows));
        std::size_t n{values.size()};
        for (const auto& window : windows) {
            std::size_t a{window.first};
            std::size_t b{window.second};
            for (std::size_t j = a; j < b; ++j) {
                const TFloatMeanAccumulator& value{values[j % n]};
                result.push_back(value);
            }
        }
    }
}

//! Calculate the number of non-empty buckets at each bucket offset in
//! the period for the \p values in \p windows.
TDoubleVec calculateRepeats(const TSizeSizePr2Vec& windows,
                            std::size_t period,
                            const TFloatMeanAccumulatorVec& values) {
    TDoubleVec result(std::min(period, length(windows[0])), 0);
    std::size_t n{values.size()};
    for (const auto& window : windows) {
        std::size_t a{window.first};
        std::size_t b{window.second};
        for (std::size_t i = a; i < b; ++i) {
            double count{CBasicStatistics::count(values[i % n])};
            result[(i - a) % period] += std::min(count, 1.0);
        }
    }
    return result;
}

//! Calculate the number of non-empty buckets at each bucket offset in
//! the period for the \p values in \p windows.
TDoubleVec calculateRepeats(const TTimeTimePr2Vec& windows_,
                            core_t::TTime period,
                            core_t::TTime bucketLength,
                            const TFloatMeanAccumulatorVec& values) {
    TSizeSizePr2Vec windows;
    calculateIndexWindows(windows_, bucketLength, windows);
    return calculateRepeats(windows, period / bucketLength, values);
}

//! Reweight outliers from \p values.
//!
//! These are defined as some fraction of the values which are most
//! different from the periodic trend on the time windows \p windows_.
template<typename T>
void reweightOutliers(const std::vector<T>& trend,
                      const TTimeTimePr2Vec& windows_,
                      core_t::TTime bucketLength,
                      TFloatMeanAccumulatorVec& values) {
    using TDoubleSizePr = std::pair<double, std::size_t>;
    using TMaxAccumulator =
        CBasicStatistics::COrderStatisticsHeap<TDoubleSizePr, std::greater<TDoubleSizePr>>;

    std::size_t period{trend.size()};
    std::size_t numberOutliers{static_cast<std::size_t>([&period, &values] {
        std::size_t count(std::count_if(
            values.begin(), values.end(), [](const TFloatMeanAccumulator& value) {
                return CBasicStatistics::count(value) > 0.0;
            }));
        return SEASONAL_OUTLIER_FRACTION *
               static_cast<double>(count - std::min(count, period));
    }())};
    LOG_TRACE(<< "Number outliers = " << numberOutliers);

    if (numberOutliers > 0) {
        TSizeSizePr2Vec windows;
        calculateIndexWindows(windows_, bucketLength, windows);
        std::size_t n{values.size()};

        TMaxAccumulator outliers{numberOutliers};
        TMeanAccumulator meanDifference;
        for (const auto& window : windows) {
            std::size_t a{window.first};
            std::size_t b{window.second};
            for (std::size_t j = a; j < b; ++j) {
                const TFloatMeanAccumulator& value{values[j % n]};
                if (CBasicStatistics::count(value) > 0.0) {
                    std::size_t offset{(j - a) % period};
                    double difference{std::fabs(CBasicStatistics::mean(value) -
                                                CBasicStatistics::mean(trend[offset]))};
                    outliers.add({difference, j});
                    meanDifference.add(difference);
                }
            }
        }
        TMeanAccumulator meanDifferenceOfOutliers;
        for (const auto& outlier : outliers) {
            meanDifferenceOfOutliers.add(outlier.first);
        }
        meanDifference -= meanDifferenceOfOutliers;
        LOG_TRACE(<< "mean difference = " << CBasicStatistics::mean(meanDifference));
        LOG_TRACE(<< "outliers = " << core::CContainerPrinter::print(outliers));

        for (const auto& outlier : outliers) {
            if (outlier.first > SEASONAL_OUTLIER_DIFFERENCE_THRESHOLD *
                                    CBasicStatistics::mean(meanDifference)) {
                CBasicStatistics::count(values[outlier.second % n]) *= SEASONAL_OUTLIER_WEIGHT;
            }
        }
        LOG_TRACE(<< "Values - outliers = " << core::CContainerPrinter::print(values));
    }
}

//! Compute the periodic trend from \p values falling in \p windows.
template<typename T, typename CONTAINER>
void periodicTrend(const std::vector<T>& values,
                   const TSizeSizePr2Vec& windows_,
                   core_t::TTime bucketLength,
                   CONTAINER& trend) {
    if (trend.size() > 0) {
        TSizeSizePr2Vec windows;
        calculateIndexWindows(windows_, bucketLength, windows);
        std::size_t period{trend.size()};
        std::size_t n{values.size()};
        for (const auto& window : windows) {
            std::size_t a{window.first};
            std::size_t b{window.second};
            for (std::size_t j = a; j < b; ++j) {
                const TFloatMeanAccumulator& value{values[j % n]};
                trend[(j - a) % period].add(CBasicStatistics::mean(value),
                                            CBasicStatistics::count(value));
            }
        }
    }
}

//! Compute the periodic trend on \p values minus outliers.
template<typename T, typename CONTAINER>
void periodicTrendMinusOutliers(std::vector<T>& values,
                                const TSizeSizePr2Vec& windows,
                                core_t::TTime bucketLength,
                                CONTAINER& trend) {
    periodicTrend(values, windows, bucketLength, trend);
    reweightOutliers(trend, windows, bucketLength, values);
    trend.assign(trend.size(), TMeanVarAccumulator{});
    periodicTrend(values, windows, bucketLength, trend);
}

//! Compute the average of the values at \p times.
void averageValue(const TFloatMeanAccumulatorVec& values,
                  const TTimeVec& times,
                  core_t::TTime bucketLength,
                  TMeanVarAccumulator& value) {
    for (const auto time : times) {
        std::size_t index(time / bucketLength);
        value.add(CBasicStatistics::mean(values[index]),
                  CBasicStatistics::count(values[index]));
    }
}

//! Extract the residual variance from the mean of a collection
//! of residual variances.
double residualVariance(const TMeanAccumulator& mean) {
    double n{CBasicStatistics::count(mean)};
    return n <= 1.0 ? 0.0 : n / (n - 1.0) * std::max(CBasicStatistics::mean(mean), 0.0);
}

//! Extract the residual variance of \p bucket of a trend.
TMeanAccumulator residualVariance(const TMeanVarAccumulator& bucket, double scale) {
    return CBasicStatistics::momentsAccumulator(
        scale * CBasicStatistics::count(bucket),
        CBasicStatistics::maximumLikelihoodVariance(bucket));
}

//! \brief Partially specialized helper class to get the trend
//! residual variance as a specified type.
template<typename R>
struct SResidualVarianceImpl {};

//! \brief Get the residual variance as a double.
template<>
struct SResidualVarianceImpl<double> {
    static double get(const TMeanAccumulator& mean) {
        return residualVariance(mean);
    }
};

//! \brief Get the residual variance as a mean accumulator.
template<>
struct SResidualVarianceImpl<TMeanAccumulator> {
    static TMeanAccumulator get(const TMeanAccumulator& mean) { return mean; }
};

//! Compute the residual variance of the trend \p trend.
template<typename R, typename CONTAINER>
R residualVariance(const CONTAINER& trend, double scale) {
    TMeanAccumulator result;
    for (const auto& bucket : trend) {
        result.add(CBasicStatistics::maximumLikelihoodVariance(bucket),
                   CBasicStatistics::count(bucket));
    }
    result.s_Count *= scale;
    return SResidualVarianceImpl<R>::get(result);
}
}

CTrendHypothesis::CTrendHypothesis(std::size_t segments)
    : m_Segments{segments} {
}

CTrendHypothesis::EType CTrendHypothesis::type() const {
    return m_Segments == 0 ? E_None : (m_Segments == 1 ? E_Linear : E_PiecewiseLinear);
}

std::size_t CTrendHypothesis::segments() const {
    return m_Segments;
}

bool CPeriodicityHypothesisTestsResult::
operator==(const CPeriodicityHypothesisTestsResult& other) const {
    return m_Components == other.m_Components;
}

void CPeriodicityHypothesisTestsResult::add(const std::string& description,
                                            bool diurnal,
                                            bool piecewiseScaled,
                                            core_t::TTime startOfPartition,
                                            core_t::TTime period,
                                            const TTimeTimePr& window,
                                            double precedence) {
    m_Components.emplace_back(description, diurnal, piecewiseScaled,
                              startOfPartition, period, window, precedence);
}

void CPeriodicityHypothesisTestsResult::remove(const TRemoveCondition& condition) {
    m_Components.erase(std::remove_if(m_Components.begin(), m_Components.end(), condition),
                       m_Components.end());
}

void CPeriodicityHypothesisTestsResult::trend(CTrendHypothesis value) {
    m_Trend = value;
}

void CPeriodicityHypothesisTestsResult::removeTrend(TFloatMeanAccumulatorVec& values) const {
    if (m_Trend.type() == CTrendHypothesis::E_Linear) {
        removeLinearTrend(values);
    } else if (m_Trend.type() == CTrendHypothesis::E_PiecewiseLinear) {
        TSizeVec segmentation(CTimeSeriesSegmentation::piecewiseLinear(values));
        values = CTimeSeriesSegmentation::removePiecewiseLinear(std::move(values), segmentation);
    }
}

void CPeriodicityHypothesisTestsResult::removeDiscontinuities(TFloatMeanAccumulatorVec& values) const {
    TSizeVec segmentation(CTimeSeriesSegmentation::piecewiseLinear(values));
    values = CTimeSeriesSegmentation::removePiecewiseLinearDiscontinuities(
        std::move(values), segmentation);
}

bool CPeriodicityHypothesisTestsResult::periodic() const {
    return m_Components.size() > 0;
}

const CPeriodicityHypothesisTestsResult::TComponent5Vec&
CPeriodicityHypothesisTestsResult::components() const {
    return m_Components;
}

std::string CPeriodicityHypothesisTestsResult::print() const {
    std::string result("{");
    for (const auto& component : m_Components) {
        result += " '" + component.s_Description + "'";
    }
    result += " }";
    return result;
}

CPeriodicityHypothesisTestsResult::SComponent::SComponent(const std::string& description,
                                                          bool diurnal,
                                                          bool piecewiseScaled,
                                                          core_t::TTime startOfPartition,
                                                          core_t::TTime period,
                                                          const TTimeTimePr& window,
                                                          double precedence)
    : s_Description(description), s_Diurnal(diurnal),
      s_PiecewiseScaled(piecewiseScaled), s_StartOfPartition(startOfPartition),
      s_Period(period), s_Window(window), s_Precedence(precedence) {
}

bool CPeriodicityHypothesisTestsResult::SComponent::operator==(const SComponent& other) const {
    return s_Description == other.s_Description && s_StartOfPartition == other.s_StartOfPartition;
}

CSeasonalTime* CPeriodicityHypothesisTestsResult::SComponent::seasonalTime() const {
    if (s_Diurnal) {
        return new CDiurnalTime(s_StartOfPartition, s_Window.first,
                                s_Window.second, s_Period, s_Precedence);
    }
    return new CGeneralPeriodTime(s_Period, s_Precedence);
}

void CPeriodicityHypothesisTestsConfig::disableDiurnal() {
    m_TestForDiurnal = false;
}

void CPeriodicityHypothesisTestsConfig::hasDaily(bool value) {
    m_HasDaily = value;
}

void CPeriodicityHypothesisTestsConfig::hasWeekend(bool value) {
    m_HasWeekend = value;
}

void CPeriodicityHypothesisTestsConfig::hasWeekly(bool value) {
    m_HasWeekly = value;
}

void CPeriodicityHypothesisTestsConfig::startOfWeek(core_t::TTime value) {
    m_StartOfWeek = value;
}

bool CPeriodicityHypothesisTestsConfig::testForDiurnal() const {
    return m_TestForDiurnal;
}

bool CPeriodicityHypothesisTestsConfig::hasDaily() const {
    return m_HasDaily;
}

bool CPeriodicityHypothesisTestsConfig::hasWeekend() const {
    return m_HasWeekend;
}

bool CPeriodicityHypothesisTestsConfig::hasWeekly() const {
    return m_HasWeekly;
}

core_t::TTime CPeriodicityHypothesisTestsConfig::startOfWeek() const {
    return m_StartOfWeek;
}

CPeriodicityHypothesisTests::CPeriodicityHypothesisTests(const CPeriodicityHypothesisTestsConfig& config)
    : m_Config(config) {
}

bool CPeriodicityHypothesisTests::initialized() const {
    return m_BucketValues.size() > 0;
}

void CPeriodicityHypothesisTests::initialize(core_t::TTime startTime,
                                             core_t::TTime bucketLength,
                                             core_t::TTime windowLength,
                                             core_t::TTime period) {
    m_StartTime = startTime;
    m_BucketLength = bucketLength;
    m_WindowLength = windowLength;
    m_BucketValues.resize(static_cast<std::size_t>(windowLength / m_BucketLength));
    m_Period = period;
}

void CPeriodicityHypothesisTests::add(core_t::TTime time, double value, double weight) {
    if (m_BucketValues.size() > 0) {
        std::size_t i((time % m_WindowLength) / m_BucketLength);
        m_BucketValues[i].add(value, weight);
        if (weight > 0.0) {
            m_TimeRange.add(time);
        }
    }
}

CPeriodicityHypothesisTestsResult CPeriodicityHypothesisTests::test() const {
    // We perform a series of tests of nested hypotheses about
    // the periodic components and weekday/end patterns. To test
    // for periodic components we compare the residual variance
    // with and without trend. This must be reduced in significant
    // absolute sense to make it worthwhile modelling and in a
    // statistical sense. We use an F-test for this purpose. Note
    // that since the buckets contain the mean of multiple samples
    // we expect them to tend to Gaussian. We also test the amplitude.
    // Again this must be significant in both an absolute and
    // statistical sense. We assume the bucket values are Gaussian,
    // with the same rationale, for the purpose of the statistical
    // test. Each time we accept a simpler hypothesis about the
    // data we test the nested hypothesis w.r.t. this. This entails
    // removing any periodic component we've already found from the
    // data.

    if (this->initialized() == false) {
        return CPeriodicityHypothesisTestsResult();
    }

    auto window = [this](core_t::TTime period) {
        std::size_t bucketsPerPeriod(period / m_BucketLength);
        std::size_t repeats{bucketsPerPeriod == 0 ? 0 : m_BucketValues.size() / bucketsPerPeriod};
        core_t::TTime windowLength{static_cast<core_t::TTime>(repeats) * period};
        return TTimeTimePr2Vec{{0, windowLength}};
    };
    auto buckets = [this](core_t::TTime period) {
        std::size_t bucketsPerPeriod(period / m_BucketLength);
        std::size_t repeats{bucketsPerPeriod == 0 ? 0 : m_BucketValues.size() / bucketsPerPeriod};
        return bucketsPerPeriod * repeats;
    };

    TFloatMeanAccumulatorVec bucketValuesMinusLinearTrend(m_BucketValues);
    removeLinearTrend(bucketValuesMinusLinearTrend);

    TFloatMeanAccumulatorVec bucketValuesMinusPiecewiseLinearTrend;
    TSizeVec segmentation(CTimeSeriesSegmentation::piecewiseLinear(m_BucketValues));
    if (segmentation.size() > 2) {
        bucketValuesMinusPiecewiseLinearTrend =
            CTimeSeriesSegmentation::removePiecewiseLinear(m_BucketValues, segmentation);
    }
    LOG_TRACE(<< "trend segmentation = " << core::CContainerPrinter::print(segmentation));

    std::size_t numberHypotheses(segmentation.size() > 2 ? 3 : 2);

    CTrendHypothesis trendHypotheses[]{CTrendHypothesis{0}, CTrendHypothesis{1},
                                       CTrendHypothesis{segmentation.size()}};

    TTimeTimePr2Vec windowForTestingDaily(window(DAY));
    TTimeTimePr2Vec windowForTestingWeekly(window(WEEK));
    TTimeTimePr2Vec windowForTestingPeriod(window(m_Period));
    TFloatMeanAccumulatorCRng bucketsForTestingDaily[]{
        {m_BucketValues, 0, buckets(DAY)},
        {bucketValuesMinusLinearTrend, 0, buckets(DAY)},
        {bucketValuesMinusPiecewiseLinearTrend, 0, buckets(DAY)}};
    TFloatMeanAccumulatorCRng bucketsForTestingWeekly[]{
        {m_BucketValues, 0, buckets(WEEK)},
        {bucketValuesMinusLinearTrend, 0, buckets(WEEK)},
        {bucketValuesMinusPiecewiseLinearTrend, 0, buckets(WEEK)}};
    TFloatMeanAccumulatorCRng bucketsForTestingPeriod[]{
        {m_BucketValues, 0, buckets(m_Period)},
        {bucketValuesMinusLinearTrend, 0, buckets(m_Period)},
        {bucketValuesMinusPiecewiseLinearTrend, 0, buckets(m_Period)}};

    LOG_TRACE(<< "Testing periodicity hypotheses");
    LOG_TRACE(<< "window for daily = "
              << core::CContainerPrinter::print(windowForTestingDaily));
    LOG_TRACE(<< "window for weekly = "
              << core::CContainerPrinter::print(windowForTestingWeekly));
    LOG_TRACE(<< "window for period = "
              << core::CContainerPrinter::print(windowForTestingPeriod));

    TNestedHypothesesVec hypotheses;

    for (std::size_t i = 0; i < numberHypotheses; ++i) {
        TNestedHypothesesVec hypotheses_;

        if (this->seenSufficientDataToTest(WEEK, bucketsForTestingWeekly[i])) {
            this->hypothesesForWeekly(windowForTestingWeekly,
                                      bucketsForTestingWeekly[i], windowForTestingPeriod,
                                      bucketsForTestingPeriod[i], hypotheses_);
        } else if (this->seenSufficientDataToTest(DAY, bucketsForTestingDaily[i])) {
            this->hypothesesForDaily(windowForTestingDaily,
                                     bucketsForTestingDaily[i], windowForTestingPeriod,
                                     bucketsForTestingPeriod[i], hypotheses_);
        } else if (this->seenSufficientDataToTest(m_Period, bucketsForTestingPeriod[i])) {
            this->hypothesesForPeriod(windowForTestingPeriod,
                                      bucketsForTestingPeriod[i], hypotheses_);
        }

        CTrendHypothesis trendHypothesis{trendHypotheses[i]};
        std::for_each(hypotheses_.begin(), hypotheses_.end(),
                      [&trendHypothesis](CNestedHypotheses& hypothesis) {
                          hypothesis.trend(trendHypothesis);
                      });

        LOG_TRACE(<< "# hypotheses = " << hypotheses_.size());

        if (hypotheses_.size() > 0) {
            hypotheses.insert(hypotheses.end(), hypotheses_.begin(), hypotheses_.end());
        }
    }

    return this->best(hypotheses);
}

void CPeriodicityHypothesisTests::hypothesesForWeekly(
    const TTimeTimePr2Vec& windowForTestingWeekly,
    const TFloatMeanAccumulatorCRng& bucketsForTestingWeekly,
    const TTimeTimePr2Vec& windowForTestingPeriod,
    const TFloatMeanAccumulatorCRng& bucketsForTestingPeriod,
    TNestedHypothesesVec& hypotheses) const {

    if (WEEK % m_Period == 0) {
        auto testForNull = std::bind(&CPeriodicityHypothesisTests::testForNull,
                                     this, std::cref(windowForTestingWeekly),
                                     std::cref(bucketsForTestingWeekly),
                                     std::placeholders::_1);
        auto testForPeriod = std::bind(&CPeriodicityHypothesisTests::testForPeriod,
                                       this, std::cref(windowForTestingWeekly),
                                       std::cref(bucketsForTestingWeekly),
                                       false, std::placeholders::_1);
        auto testForDaily = std::bind(&CPeriodicityHypothesisTests::testForDaily,
                                      this, std::cref(windowForTestingWeekly),
                                      std::cref(bucketsForTestingWeekly), false,
                                      std::placeholders::_1);
        auto testForWeekly = std::bind(&CPeriodicityHypothesisTests::testForWeekly,
                                       this, std::cref(windowForTestingWeekly),
                                       std::cref(bucketsForTestingWeekly),
                                       false, std::placeholders::_1);
        auto testForPeriodWithScaling = std::bind(
            &CPeriodicityHypothesisTests::testForPeriod, this,
            std::cref(windowForTestingWeekly),
            std::cref(bucketsForTestingWeekly), true, std::placeholders::_1);
        auto testForDailyWithScaling = std::bind(
            &CPeriodicityHypothesisTests::testForDaily, this,
            std::cref(windowForTestingWeekly),
            std::cref(bucketsForTestingWeekly), true, std::placeholders::_1);
        auto testForWeeklyWithScaling = std::bind(
            &CPeriodicityHypothesisTests::testForWeekly, this,
            std::cref(windowForTestingWeekly),
            std::cref(bucketsForTestingWeekly), true, std::placeholders::_1);
        auto testForDailyWithWeekend =
            std::bind(&CPeriodicityHypothesisTests::testForDailyWithWeekend, this,
                      std::cref(bucketsForTestingWeekly), std::placeholders::_1);
        auto testForWeeklyGivenWeekend =
            std::bind(&CPeriodicityHypothesisTests::testForWeeklyGivenDailyWithWeekend,
                      this, std::cref(windowForTestingWeekly),
                      std::cref(bucketsForTestingWeekly), std::placeholders::_1);

        hypotheses.resize(2);
        if (DAY % m_Period == 0) {
            // clang-format off
            hypotheses[0].null(testForNull)
                             .addNested(testForPeriod)
                                 .addNested(testForDaily)
                                     .addNested(testForDailyWithWeekend)
                                         .addNested(testForWeeklyGivenWeekend)
                                         .finishedNested()
                                     .addAlternative(testForWeekly)
                                     .finishedNested()
                                 .finishedNested()
                             .addAlternative(testForDaily)
                                 .addNested(testForDailyWithWeekend)
                                     .addNested(testForWeeklyGivenWeekend)
                                     .finishedNested()
                                 .addAlternative(testForWeekly)
                                 .finishedNested()
                             .addAlternative(testForDailyWithWeekend)
                                 .addNested(testForWeeklyGivenWeekend)
                                 .finishedNested()
                             .addAlternative(testForWeekly);
            hypotheses[1].null(testForNull)
                             .addNested(testForPeriodWithScaling)
                                 .addNested(testForDailyWithScaling)
                                     .addNested(testForWeeklyWithScaling)
                                     .finishedNested()
                                 .finishedNested()
                             .addAlternative(testForDailyWithScaling)
                                 .addNested(testForWeeklyWithScaling)
                                 .finishedNested()
                             .addAlternative(testForWeeklyWithScaling);
            // clang-format on
        } else {
            // clang-format off
            hypotheses[0].null(testForNull)
                             .addNested(testForDaily)
                                 .addNested(testForDailyWithWeekend)
                                     .addNested(testForWeeklyGivenWeekend)
                                     .finishedNested()
                                 .addAlternative(testForWeekly)
                                 .finishedNested()
                             .addAlternative(testForDailyWithWeekend)
                                 .addNested(testForWeeklyGivenWeekend)
                                 .finishedNested()
                             .addAlternative(testForPeriod)
                                 .addNested(testForWeekly)
                                 .finishedNested()
                             .addAlternative(testForWeekly);
            hypotheses[1].null(testForNull)
                             .addNested(testForDailyWithScaling)
                                 .addNested(testForWeeklyWithScaling)
                                 .finishedNested()
                             .addAlternative(testForWeeklyWithScaling)
                             .addAlternative(testForPeriodWithScaling);
            // clang-format on
        }
    } else if (m_Period % WEEK == 0) {
        auto testForNull = std::bind(&CPeriodicityHypothesisTests::testForNull,
                                     this, std::cref(windowForTestingPeriod),
                                     std::cref(bucketsForTestingPeriod),
                                     std::placeholders::_1);
        auto testForPeriod = std::bind(&CPeriodicityHypothesisTests::testForPeriod,
                                       this, std::cref(windowForTestingPeriod),
                                       std::cref(bucketsForTestingPeriod),
                                       false, std::placeholders::_1);
        auto testForDaily = std::bind(&CPeriodicityHypothesisTests::testForDaily,
                                      this, std::cref(windowForTestingPeriod),
                                      std::cref(bucketsForTestingPeriod), false,
                                      std::placeholders::_1);
        auto testForWeekly = std::bind(&CPeriodicityHypothesisTests::testForWeekly,
                                       this, std::cref(windowForTestingPeriod),
                                       std::cref(bucketsForTestingPeriod),
                                       false, std::placeholders::_1);
        auto testForPeriodWithScaling = std::bind(
            &CPeriodicityHypothesisTests::testForPeriod, this,
            std::cref(windowForTestingPeriod),
            std::cref(bucketsForTestingPeriod), true, std::placeholders::_1);
        auto testForDailyWithScaling = std::bind(
            &CPeriodicityHypothesisTests::testForDaily, this,
            std::cref(windowForTestingPeriod),
            std::cref(bucketsForTestingPeriod), true, std::placeholders::_1);
        auto testForWeeklyWithScaling = std::bind(
            &CPeriodicityHypothesisTests::testForWeekly, this,
            std::cref(windowForTestingPeriod),
            std::cref(bucketsForTestingPeriod), true, std::placeholders::_1);
        auto testForDailyWithWeekend =
            std::bind(&CPeriodicityHypothesisTests::testForDailyWithWeekend, this,
                      std::cref(bucketsForTestingPeriod), std::placeholders::_1);
        auto testForWeeklyGivenWeekend =
            std::bind(&CPeriodicityHypothesisTests::testForWeeklyGivenDailyWithWeekend,
                      this, std::cref(windowForTestingPeriod),
                      std::cref(bucketsForTestingPeriod), std::placeholders::_1);

        // clang-format off
        hypotheses.resize(2);
        hypotheses[0].null(testForNull)
                         .addNested(testForDaily)
                             .addNested(testForDailyWithWeekend)
                                 .addNested(testForWeeklyGivenWeekend)
                                     .addNested(testForPeriod)
                                     .finishedNested()
                                 .finishedNested()
                             .addAlternative(testForWeekly)
                                 .addNested(testForPeriod)
                                 .finishedNested()
                             .finishedNested()
                         .addAlternative(testForDailyWithWeekend)
                             .addNested(testForWeeklyGivenWeekend)
                                 .addNested(testForPeriod)
                                 .finishedNested()
                             .finishedNested()
                         .addAlternative(testForWeekly)
                             .addNested(testForPeriod)
                             .finishedNested()
                         .addAlternative(testForPeriod);
        hypotheses[1].null(testForNull)
                         .addNested(testForDailyWithScaling)
                             .addNested(testForWeeklyWithScaling)
                                 .addNested(testForPeriodWithScaling)
                                 .finishedNested()
                             .finishedNested()
                         .addAlternative(testForWeeklyWithScaling)
                             .addNested(testForPeriodWithScaling)
                             .finishedNested()
                         .addAlternative(testForPeriodWithScaling);
        // clang-format on
    } else {
        hypotheses.resize(4);
        {
            auto testForNull = std::bind(&CPeriodicityHypothesisTests::testForNull,
                                         this, std::cref(windowForTestingWeekly),
                                         std::cref(bucketsForTestingWeekly),
                                         std::placeholders::_1);
            auto testForDaily = std::bind(&CPeriodicityHypothesisTests::testForDaily,
                                          this, std::cref(windowForTestingWeekly),
                                          std::cref(bucketsForTestingWeekly),
                                          false, std::placeholders::_1);
            auto testForWeekly = std::bind(&CPeriodicityHypothesisTests::testForWeekly,
                                           this, std::cref(windowForTestingWeekly),
                                           std::cref(bucketsForTestingWeekly),
                                           false, std::placeholders::_1);
            auto testForDailyWithScaling = std::bind(
                &CPeriodicityHypothesisTests::testForDaily, this,
                std::cref(windowForTestingWeekly),
                std::cref(bucketsForTestingWeekly), true, std::placeholders::_1);
            auto testForWeeklyWithScaling = std::bind(
                &CPeriodicityHypothesisTests::testForWeekly, this,
                std::cref(windowForTestingWeekly),
                std::cref(bucketsForTestingWeekly), true, std::placeholders::_1);
            auto testForDailyWithWeekend = std::bind(
                &CPeriodicityHypothesisTests::testForDailyWithWeekend, this,
                std::cref(bucketsForTestingWeekly), std::placeholders::_1);
            auto testForWeeklyGivenWeekend = std::bind(
                &CPeriodicityHypothesisTests::testForWeeklyGivenDailyWithWeekend,
                this, std::cref(windowForTestingWeekly),
                std::cref(bucketsForTestingWeekly), std::placeholders::_1);

            // clang-format off
            hypotheses[0].null(testForNull)
                             .addNested(testForDaily)
                                 .addNested(testForDailyWithWeekend)
                                     .addNested(testForWeeklyGivenWeekend)
                                     .finishedNested()
                                 .addAlternative(testForWeekly)
                                 .finishedNested()
                             .addAlternative(testForDailyWithWeekend)
                                 .addNested(testForWeeklyGivenWeekend)
                                 .finishedNested()
                             .addAlternative(testForWeekly);
            hypotheses[1].null(testForNull)
                             .addNested(testForDailyWithScaling)
                                 .addNested(testForWeeklyWithScaling)
                                 .finishedNested()
                             .addAlternative(testForWeeklyWithScaling);
            // clang-format on
        }
        if (m_Period % DAY == 0) {
            auto testForNull = std::bind(&CPeriodicityHypothesisTests::testForNull,
                                         this, std::cref(windowForTestingPeriod),
                                         std::cref(bucketsForTestingPeriod),
                                         std::placeholders::_1);
            auto testForDaily = std::bind(&CPeriodicityHypothesisTests::testForDaily,
                                          this, std::cref(windowForTestingPeriod),
                                          std::cref(bucketsForTestingPeriod),
                                          false, std::placeholders::_1);
            auto testForPeriod = std::bind(&CPeriodicityHypothesisTests::testForPeriod,
                                           this, std::cref(windowForTestingPeriod),
                                           std::cref(bucketsForTestingPeriod),
                                           false, std::placeholders::_1);
            auto testForDailyWithScaling = std::bind(
                &CPeriodicityHypothesisTests::testForDaily, this,
                std::cref(windowForTestingPeriod),
                std::cref(bucketsForTestingPeriod), true, std::placeholders::_1);
            auto testForPeriodWithScaling = std::bind(
                &CPeriodicityHypothesisTests::testForPeriod, this,
                std::cref(windowForTestingPeriod),
                std::cref(bucketsForTestingPeriod), true, std::placeholders::_1);

            // clang-format off
            hypotheses[2].null(testForNull)
                             .addNested(testForDaily)
                                 .addNested(testForPeriod)
                                 .finishedNested()
                             .addAlternative(testForPeriod);
            hypotheses[3].null(testForNull)
                             .addNested(testForDailyWithScaling)
                                 .addNested(testForPeriod)
                                 .finishedNested()
                             .addAlternative(testForPeriodWithScaling);
            // clang-format on
        } else {
            auto testForNull = std::bind(&CPeriodicityHypothesisTests::testForNull,
                                         this, std::cref(windowForTestingPeriod),
                                         std::cref(bucketsForTestingPeriod),
                                         std::placeholders::_1);
            auto testForPeriod = std::bind(&CPeriodicityHypothesisTests::testForPeriod,
                                           this, std::cref(windowForTestingPeriod),
                                           std::cref(bucketsForTestingPeriod),
                                           false, std::placeholders::_1);
            auto testForPeriodWithScaling = std::bind(
                &CPeriodicityHypothesisTests::testForPeriod, this,
                std::cref(windowForTestingPeriod),
                std::cref(bucketsForTestingPeriod), true, std::placeholders::_1);

            // clang-format off
            hypotheses[2].null(testForNull)
                             .addNested(testForPeriod);
            hypotheses[3].null(testForNull)
                             .addNested(testForPeriodWithScaling);
            // clang-format on
        }
    }
}

void CPeriodicityHypothesisTests::hypothesesForDaily(
    const TTimeTimePr2Vec& windowForTestingDaily,
    const TFloatMeanAccumulatorCRng& bucketsForTestingDaily,
    const TTimeTimePr2Vec& windowForTestingPeriod,
    const TFloatMeanAccumulatorCRng& bucketsForTestingPeriod,
    TNestedHypothesesVec& hypotheses) const {
    if (DAY % m_Period == 0) {
        auto testForNull = std::bind(&CPeriodicityHypothesisTests::testForNull,
                                     this, std::cref(windowForTestingDaily),
                                     std::cref(bucketsForTestingDaily),
                                     std::placeholders::_1);
        auto testForPeriod = std::bind(&CPeriodicityHypothesisTests::testForPeriod,
                                       this, std::cref(windowForTestingDaily),
                                       std::cref(bucketsForTestingDaily), false,
                                       std::placeholders::_1);
        auto testForDaily = std::bind(&CPeriodicityHypothesisTests::testForDaily,
                                      this, std::cref(windowForTestingDaily),
                                      std::cref(bucketsForTestingDaily), false,
                                      std::placeholders::_1);
        auto testForPeriodWithScaling = std::bind(
            &CPeriodicityHypothesisTests::testForPeriod, this,
            std::cref(windowForTestingDaily), std::cref(bucketsForTestingDaily),
            true, std::placeholders::_1);
        auto testForDailyWithScaling = std::bind(
            &CPeriodicityHypothesisTests::testForDaily, this,
            std::cref(windowForTestingDaily), std::cref(bucketsForTestingDaily),
            true, std::placeholders::_1);

        hypotheses.resize(2);
        // clang-format off
        hypotheses[0].null(testForNull)
                         .addNested(testForPeriod)
                             .addNested(testForDaily)
                             .finishedNested()
                         .addAlternative(testForDaily);
        hypotheses[1].null(testForNull)
                         .addNested(testForPeriodWithScaling)
                             .addNested(testForDailyWithScaling)
                             .finishedNested()
                         .addAlternative(testForDailyWithScaling);
        // clang-format on
    } else if (m_Period % DAY == 0) {
        auto testForNull = std::bind(&CPeriodicityHypothesisTests::testForNull,
                                     this, std::cref(windowForTestingPeriod),
                                     std::cref(bucketsForTestingPeriod),
                                     std::placeholders::_1);
        auto testForPeriod = std::bind(&CPeriodicityHypothesisTests::testForPeriod,
                                       this, std::cref(windowForTestingPeriod),
                                       std::cref(bucketsForTestingPeriod),
                                       false, std::placeholders::_1);
        auto testForDaily = std::bind(&CPeriodicityHypothesisTests::testForDaily,
                                      this, std::cref(windowForTestingPeriod),
                                      std::cref(bucketsForTestingPeriod), false,
                                      std::placeholders::_1);
        auto testForPeriodWithScaling = std::bind(
            &CPeriodicityHypothesisTests::testForPeriod, this,
            std::cref(windowForTestingDaily), std::cref(bucketsForTestingDaily),
            true, std::placeholders::_1);
        auto testForDailyWithScaling = std::bind(
            &CPeriodicityHypothesisTests::testForDaily, this,
            std::cref(windowForTestingDaily), std::cref(bucketsForTestingDaily),
            true, std::placeholders::_1);

        hypotheses.resize(2);
        // clang-format off
        hypotheses[0].null(testForNull)
                         .addNested(testForDaily)
                             .addNested(testForPeriod);
        hypotheses[1].null(testForNull)
                         .addNested(testForDailyWithScaling)
                             .addNested(testForPeriodWithScaling);
        // clang-format on
    } else {
        hypotheses.resize(4);
        {
            auto testForNull = std::bind(&CPeriodicityHypothesisTests::testForNull,
                                         this, std::cref(windowForTestingDaily),
                                         std::cref(bucketsForTestingDaily),
                                         std::placeholders::_1);
            auto testForDaily = std::bind(&CPeriodicityHypothesisTests::testForDaily,
                                          this, std::cref(windowForTestingDaily),
                                          std::cref(bucketsForTestingDaily),
                                          false, std::placeholders::_1);
            auto testForDailyWithScaling = std::bind(
                &CPeriodicityHypothesisTests::testForDaily, this,
                std::cref(windowForTestingDaily),
                std::cref(bucketsForTestingDaily), true, std::placeholders::_1);

            // clang-format off
            hypotheses[0].null(testForNull)
                             .addNested(testForDaily);
            hypotheses[1].null(testForNull)
                             .addNested(testForDailyWithScaling);
            // clang-format on
        }
        {
            auto testForNull = std::bind(&CPeriodicityHypothesisTests::testForNull,
                                         this, std::cref(windowForTestingPeriod),
                                         std::cref(bucketsForTestingPeriod),
                                         std::placeholders::_1);
            auto testForPeriod = std::bind(&CPeriodicityHypothesisTests::testForPeriod,
                                           this, std::cref(windowForTestingPeriod),
                                           std::cref(bucketsForTestingPeriod),
                                           false, std::placeholders::_1);
            auto testForPeriodWithScaling = std::bind(
                &CPeriodicityHypothesisTests::testForPeriod, this,
                std::cref(windowForTestingPeriod),
                std::cref(bucketsForTestingPeriod), true, std::placeholders::_1);
            // clang-format off
            hypotheses[2].null(testForNull)
                             .addNested(testForPeriod);
            hypotheses[3].null(testForNull)
                             .addNested(testForPeriodWithScaling);
            // clang-format on
        }
    }
}

void CPeriodicityHypothesisTests::hypothesesForPeriod(const TTimeTimePr2Vec& windows,
                                                      const TFloatMeanAccumulatorCRng& buckets,
                                                      TNestedHypothesesVec& hypotheses) const {
    auto testForNull = std::bind(&CPeriodicityHypothesisTests::testForNull,
                                 this, std::cref(windows), std::cref(buckets),
                                 std::placeholders::_1);
    auto testForPeriod = std::bind(&CPeriodicityHypothesisTests::testForPeriod,
                                   this, std::cref(windows), std::cref(buckets),
                                   false, std::placeholders::_1);
    auto testForPeriodWithScaling = std::bind(
        &CPeriodicityHypothesisTests::testForPeriod, this, std::cref(windows),
        std::cref(buckets), true, std::placeholders::_1);

    hypotheses.resize(2);
    // clang-format off
    hypotheses[0].null(testForNull)
                     .addNested(testForPeriod);
    hypotheses[1].null(testForNull)
                     .addNested(testForPeriodWithScaling);
    // clang-format on
}

CPeriodicityHypothesisTestsResult
CPeriodicityHypothesisTests::best(const TNestedHypothesesVec& hypotheses) const {
    // We are comparing different accepted hypotheses here. In particular,
    // diurnal and the best non-diurnal components with and without fitting
    // a linear ramp to the values. We use a smooth decision function to
    // select between them preferring:
    //   1) The hypothesis which explains the most variance,
    //   2) The simplest hypothesis (fewest parameters),
    //   3) The hypothesis which most exceeds the minimum autocorrelation
    //      to accept.
    //   4) Hypotheses with fewer segments.

    using TMinAccumulator = CBasicStatistics::SMin<double>::TAccumulator;

    LOG_TRACE(<< "# hypotheses = " << hypotheses.size());

    CPeriodicityHypothesisTestsResult result;

    THypothesisSummaryVec summaries;
    summaries.reserve(hypotheses.size());

    double meanMagnitude{CBasicStatistics::mean(std::accumulate(
        m_BucketValues.begin(), m_BucketValues.end(), TMeanAccumulator{},
        [](TMeanAccumulator partial, const TFloatMeanAccumulator& value) {
            partial.add(std::fabs(CBasicStatistics::mean(value)),
                        CBasicStatistics::count(value));
            return partial;
        }))};

    for (const auto& hypothesis : hypotheses) {
        STestStats stats{meanMagnitude};
        stats.s_TrendSegments = static_cast<double>(hypothesis.trendSegments());
        CPeriodicityHypothesisTestsResult resultForHypothesis{hypothesis.test(stats)};
        if (stats.s_NonEmptyBuckets > stats.s_DF0) {
            if (resultForHypothesis.periodic() == false) {
                stats.setThresholds(
                    COMPONENT_SIGNIFICANT_VARIANCE_REDUCTION[E_HighThreshold],
                    SEASONAL_SIGNIFICANT_AMPLITUDE[E_HighThreshold],
                    SEASONAL_SIGNIFICANT_AUTOCORRELATION[E_HighThreshold]);
                stats.s_R0 = stats.s_AutocorrelationThreshold;
            }
            LOG_TRACE(<< resultForHypothesis.print()
                      << (hypothesis.trendSegments() > 1 ? " piecewise linear trend" : ""));
            summaries.push_back(SHypothesisSummary{
                stats.s_V0, stats.s_R0, stats.s_NonEmptyBuckets - stats.s_DF0,
                stats.s_VarianceThreshold, stats.s_AutocorrelationThreshold,
                stats.s_TrendSegments - 1.0,
                std::max(static_cast<double>(stats.s_Segmentation.size()), 1.0) - 1.0,
                std::move(resultForHypothesis)});
        }
    }

    if (summaries.size() > 0) {
        TMinAccumulator vmin;
        TMinAccumulator DFmin;
        for (const auto& summary : summaries) {
            double v{varianceAtPercentile(summary.s_V, summary.s_DF,
                                          50.0 + CONFIDENCE_INTERVAL / 2.0)};
            vmin.add(v == summary.s_VarianceThreshold ? 1.0 : v / summary.s_VarianceThreshold);
            DFmin.add(summary.s_DF);
        }

        TMinAccumulator minMinusTruth;
        for (const auto& summary : summaries) {
            double v{varianceAtPercentile(summary.s_V, summary.s_DF,
                                          50.0 - CONFIDENCE_INTERVAL / 2.0)};
            v = v == summary.s_VarianceThreshold * vmin[0]
                    ? 1.0
                    : v / summary.s_VarianceThreshold / vmin[0];
            double truth{(softLessThan(v, 1.0, 0.2) &&
                          softGreaterThan(summary.s_R, summary.s_AutocorrelationThreshold, 0.1) &&
                          softGreaterThan(summary.s_DF / DFmin[0], 1.0, 0.2) &&
                          softLessThan(summary.s_TrendSegments, 0.0, 0.3) &&
                          softLessThan(summary.s_ScaleSegments, 0.0, 0.3))
                             .truthValue()};
            LOG_TRACE(<< "truth(hypothesis) = " << truth);
            if (minMinusTruth.add(-truth)) {
                result = summary.s_H;
            }
        }
    }
    return result;
}

CPeriodicityHypothesisTestsResult
CPeriodicityHypothesisTests::testForNull(const TTimeTimePr2Vec& window,
                                         const TFloatMeanAccumulatorCRng& buckets,
                                         STestStats& stats) const {
    LOG_TRACE(<< "Testing null on " << core::CContainerPrinter::print(window));
    this->nullHypothesis(window, buckets, stats);
    return CPeriodicityHypothesisTestsResult();
}

CPeriodicityHypothesisTestsResult
CPeriodicityHypothesisTests::testForDaily(const TTimeTimePr2Vec& windows,
                                          const TFloatMeanAccumulatorCRng& buckets,
                                          bool scaling,
                                          STestStats& stats) const {
    LOG_TRACE(<< "Testing daily on " << core::CContainerPrinter::print(windows));

    CPeriodicityHypothesisTestsResult result{stats.s_H0};

    stats.s_HasPeriod = m_Config.hasDaily();
    stats.setThresholds(COMPONENT_SIGNIFICANT_VARIANCE_REDUCTION[E_LowThreshold],
                        SEASONAL_SIGNIFICANT_AMPLITUDE[E_LowThreshold],
                        SEASONAL_SIGNIFICANT_AUTOCORRELATION[E_LowThreshold]);

    if (m_Config.testForDiurnal() && m_BucketLength <= DAY / 4 &&
        this->seenSufficientDataToTest(DAY, buckets) &&
        (scaling ? this->testPeriodWithScaling(windows, buckets, DAY, stats)
                 : this->testPeriod(windows, buckets, DAY, stats))) {
        stats.s_StartOfPartition = 0;
        stats.s_Partition.assign(1, {0, length(buckets, m_BucketLength)});
        this->hypothesis({DAY}, buckets, stats);
        result.add(DIURNAL_COMPONENT_NAMES[E_Day], true /*diurnal*/,
                   stats.s_Segmentation.size() > 0, 0 /*startOfWeek*/,
                   DIURNAL_PERIODS[static_cast<int>(E_Day) % 2],
                   DIURNAL_WINDOWS[static_cast<int>(E_Day) / 2]);
    }

    return result;
}

CPeriodicityHypothesisTestsResult
CPeriodicityHypothesisTests::testForWeekly(const TTimeTimePr2Vec& windows,
                                           const TFloatMeanAccumulatorCRng& buckets,
                                           bool scaling,
                                           STestStats& stats) const {
    LOG_TRACE(<< "Testing weekly on " << core::CContainerPrinter::print(windows));

    CPeriodicityHypothesisTestsResult result{stats.s_H0};

    stats.s_HasPeriod = m_Config.hasWeekly();
    stats.setThresholds(COMPONENT_SIGNIFICANT_VARIANCE_REDUCTION[E_LowThreshold],
                        SEASONAL_SIGNIFICANT_AMPLITUDE[E_LowThreshold],
                        SEASONAL_SIGNIFICANT_AUTOCORRELATION[E_LowThreshold]);

    if (m_Config.testForDiurnal() && m_BucketLength <= WEEK / 4 &&
        this->seenSufficientDataToTest(WEEK, buckets) &&
        (scaling ? this->testPeriodWithScaling(windows, buckets, WEEK, stats)
                 : this->testPeriod(windows, buckets, WEEK, stats))) {
        stats.s_StartOfPartition = 0;
        stats.s_Partition.assign(1, {0, length(buckets, m_BucketLength)});
        this->hypothesis({WEEK}, buckets, stats);
        result.add(DIURNAL_COMPONENT_NAMES[E_Week], true /*diurnal*/,
                   stats.s_Segmentation.size() > 0, 0 /*startOfWeek*/,
                   DIURNAL_PERIODS[static_cast<int>(E_Week) % 2],
                   DIURNAL_WINDOWS[static_cast<int>(E_Week) / 2]);
    }

    return result;
}

CPeriodicityHypothesisTestsResult
CPeriodicityHypothesisTests::testForDailyWithWeekend(const TFloatMeanAccumulatorCRng& buckets,
                                                     STestStats& stats) const {
    LOG_TRACE(<< "Testing for weekend");

    CPeriodicityHypothesisTestsResult result{stats.s_H0};

    stats.s_HasPartition = m_Config.hasWeekend();
    stats.s_StartOfPartition = m_Config.hasWeekend() ? m_Config.startOfWeek() : 0;
    stats.setThresholds(COMPONENT_SIGNIFICANT_VARIANCE_REDUCTION[E_HighThreshold],
                        SEASONAL_SIGNIFICANT_AMPLITUDE[E_HighThreshold],
                        SEASONAL_SIGNIFICANT_AUTOCORRELATION[E_HighThreshold]);

    TTimeTimePr2Vec partition{{0, WEEKEND}, {WEEKEND, WEEK}};
    std::size_t bucketsPerWeek(WEEK / m_BucketLength);

    if (m_Config.testForDiurnal() && m_BucketLength <= DAY / 4 &&
        this->seenSufficientDataToTest(WEEK, buckets) &&
        this->testPartition(partition, buckets, DAY,
                            weekendPartitionVarianceCorrection(bucketsPerWeek), stats)) {
        stats.s_Partition = partition;
        this->hypothesis({DAY, DAY}, buckets, stats);
        core_t::TTime startOfWeek{stats.s_StartOfPartition};
        result.remove([](const CPeriodicityHypothesisTestsResult::SComponent& component) {
            return component.s_Period == DAY;
        });
        result.add(DIURNAL_COMPONENT_NAMES[E_WeekendDay], true /*diurnal*/,
                   false /*piecewiseConstant*/, startOfWeek,
                   DIURNAL_PERIODS[static_cast<int>(E_WeekendDay) % 2],
                   DIURNAL_WINDOWS[static_cast<int>(E_WeekendDay) / 2], HIGH_PRIORITY);
        result.add(DIURNAL_COMPONENT_NAMES[E_WeekdayDay], true /*diurnal*/,
                   false /*piecewiseConstant*/, startOfWeek,
                   DIURNAL_PERIODS[static_cast<int>(E_WeekdayDay) % 2],
                   DIURNAL_WINDOWS[static_cast<int>(E_WeekdayDay) / 2], HIGH_PRIORITY);
    }

    return result;
}

CPeriodicityHypothesisTestsResult CPeriodicityHypothesisTests::testForWeeklyGivenDailyWithWeekend(
    const TTimeTimePr2Vec& windows,
    const TFloatMeanAccumulatorCRng& buckets,
    STestStats& stats) const {
    LOG_TRACE(<< "Testing for weekly given weekend on "
              << core::CContainerPrinter::print(windows));

    CPeriodicityHypothesisTestsResult result(stats.s_H0);

    if (m_Config.testForDiurnal() == false) {
        return result;
    }

    core_t::TTime startOfWeek{stats.s_StartOfPartition};

    CPeriodicityHypothesisTestsResult resultForWeekly{
        this->testForWeekly(windows, buckets, false, stats)};
    if (resultForWeekly != result) {
        // Note that testForWeekly sets up the hypothesis for us.
        result.add(DIURNAL_COMPONENT_NAMES[E_WeekendWeek], true /*diurnal*/,
                   false /*piecewiseConstant*/, startOfWeek,
                   DIURNAL_PERIODS[static_cast<int>(E_WeekendWeek) % 2],
                   DIURNAL_WINDOWS[static_cast<int>(E_WeekendWeek) / 2], HIGH_PRIORITY);
        result.add(DIURNAL_COMPONENT_NAMES[E_WeekdayWeek], true /*diurnal*/,
                   false /*piecewiseConstant*/, startOfWeek,
                   DIURNAL_PERIODS[static_cast<int>(E_WeekdayWeek) % 2],
                   DIURNAL_WINDOWS[static_cast<int>(E_WeekdayWeek) / 2], HIGH_PRIORITY);
        return result;
    }

    core_t::TTime windowLength{length(windows)};
    TTimeTimePr2Vec partition{{0, WEEKEND}, {WEEKEND, WEEK}};

    TTimeTimePr2Vec weekday(
        calculateWindows(startOfWeek, windowLength, WEEK, {WEEKEND, WEEK}));
    CPeriodicityHypothesisTestsResult resultForWeekday{
        this->testForWeekly(weekday, buckets, false, stats)};
    if (resultForWeekday != result) {
        stats.s_StartOfPartition = startOfWeek;
        stats.s_Partition = partition;
        this->hypothesis({DAY, WEEK}, buckets, stats);
        result.add(DIURNAL_COMPONENT_NAMES[E_WeekdayWeek], true /*diurnal*/,
                   false /*piecewiseConstant*/, startOfWeek,
                   DIURNAL_PERIODS[static_cast<int>(E_WeekdayWeek) % 2],
                   DIURNAL_WINDOWS[static_cast<int>(E_WeekdayWeek) / 2], HIGH_PRIORITY);
        return result;
    }

    TTimeTimePr2Vec weekend(calculateWindows(startOfWeek, windowLength, WEEK, {0, WEEKEND}));
    CPeriodicityHypothesisTestsResult resultForWeekend{
        this->testForWeekly(weekend, buckets, false, stats)};
    if (resultForWeekend != result) {
        stats.s_StartOfPartition = startOfWeek;
        stats.s_Partition = partition;
        this->hypothesis({WEEK, DAY}, buckets, stats);
        result.add(DIURNAL_COMPONENT_NAMES[E_WeekendWeek], true /*diurnal*/,
                   false /*piecewiseConstant*/, startOfWeek,
                   DIURNAL_PERIODS[static_cast<int>(E_WeekendWeek) % 2],
                   DIURNAL_WINDOWS[static_cast<int>(E_WeekendWeek) / 2], HIGH_PRIORITY);
    }

    return result;
}

CPeriodicityHypothesisTestsResult
CPeriodicityHypothesisTests::testForPeriod(const TTimeTimePr2Vec& windows,
                                           const TFloatMeanAccumulatorCRng& buckets,
                                           bool scaling,
                                           STestStats& stats) const {
    LOG_TRACE(<< "Testing for " << m_Period << " on "
              << core::CContainerPrinter::print(windows));

    CPeriodicityHypothesisTestsResult result{stats.s_H0};

    if (m_Period != DAY && m_Period != WEEK && m_BucketLength <= m_Period / 4 &&
        this->seenSufficientDataToTest(m_Period, buckets)) {
        stats.s_HasPeriod = false;
        EThreshold index{m_Period % DAY == 0 ? E_LowThreshold : E_HighThreshold};
        stats.setThresholds(COMPONENT_SIGNIFICANT_VARIANCE_REDUCTION[index],
                            SEASONAL_SIGNIFICANT_AMPLITUDE[index],
                            SEASONAL_SIGNIFICANT_AUTOCORRELATION[index]);
        if (scaling ? this->testPeriodWithScaling(windows, buckets, m_Period, stats)
                    : this->testPeriod(windows, buckets, m_Period, stats)) {
            stats.s_StartOfPartition = 0;
            stats.s_Partition.assign(1, {0, length(buckets, m_BucketLength)});
            this->hypothesis({m_Period}, buckets, stats);
            result.add(core::CStringUtils::typeToString(m_Period),
                       false /*diurnal*/, stats.s_Segmentation.size() > 0,
                       0 /*startOfWeek*/, m_Period, {0, m_Period});
        }
    }

    return result;
}

bool CPeriodicityHypothesisTests::seenSufficientDataToTest(core_t::TTime period,
                                                           const TFloatMeanAccumulatorCRng& buckets) const {
    return (buckets.size() * m_BucketLength) / period >= 2 &&
           m_TimeRange.initialized() &&
           static_cast<double>(m_TimeRange.range()) >=
               2.0 * ACCURATE_TEST_POPULATED_FRACTION * static_cast<double>(period);
}

template<typename CONTAINER>
bool CPeriodicityHypothesisTests::seenSufficientPeriodicallyPopulatedBucketsToTest(
    const CONTAINER& buckets,
    std::size_t period) const {
    double repeats{0.0};
    for (std::size_t i = 0u; i < period; ++i) {
        for (std::size_t j = i + period; j < buckets.size(); j += period) {
            if (CBasicStatistics::count(buckets[j]) *
                    CBasicStatistics::count(buckets[j - period]) >
                0.0) {
                repeats += 1.0;
                break;
            }
        }
    }
    LOG_TRACE(<< "repeated values = " << repeats);
    return repeats >= static_cast<double>(period) * ACCURATE_TEST_POPULATED_FRACTION / 3.0;
}

bool CPeriodicityHypothesisTests::testStatisticsFor(const TFloatMeanAccumulatorCRng& buckets,
                                                    STestStats& stats) const {
    CBasicStatistics::CMinMax<double> range;
    double populated{0.0};
    double count{0.0};
    for (std::size_t i = 0u; i < buckets.size(); ++i) {
        double ni{CBasicStatistics::count(buckets[i])};
        count += ni;
        if (ni > 0.0) {
            populated += 1.0;
            range.add(static_cast<double>(i));
        }
    }

    if (populated == 0.0) {
        return false;
    }

    LOG_TRACE(<< "populated = "
              << 100.0 * populated / static_cast<double>(buckets.size()) << "%");

    stats.s_Range = range.max() - range.min();
    stats.s_NonEmptyBuckets = populated;
    stats.s_MeasurementsPerBucket = count / stats.s_NonEmptyBuckets;
    LOG_TRACE(<< "range = " << stats.s_Range << ", populatedBuckets = " << stats.s_NonEmptyBuckets
              << ", valuesPerBucket = " << stats.s_MeasurementsPerBucket);

    return true;
}

void CPeriodicityHypothesisTests::nullHypothesis(const TTimeTimePr2Vec& window,
                                                 const TFloatMeanAccumulatorCRng& buckets,
                                                 STestStats& stats) const {
    if (this->testStatisticsFor(buckets, stats)) {
        TMeanVarAccumulatorVec trend(1);
        TFloatMeanAccumulatorVec values(buckets.begin(), buckets.end());
        periodicTrendMinusOutliers(values, window, m_BucketLength, trend);
        double v0{CBasicStatistics::variance(trend[0])};
        LOG_TRACE(<< "variance = " << v0);
        stats.s_DF0 = 1.0;
        stats.s_V0 = v0;
        stats.s_T0.assign(1, {0.0});
        stats.s_Partition = window;
    }
}

void CPeriodicityHypothesisTests::hypothesis(const TTime2Vec& periods,
                                             const TFloatMeanAccumulatorCRng& buckets,
                                             STestStats& stats) const {
    if (this->testStatisticsFor(buckets, stats)) {
        stats.s_V0 = 0.0;
        stats.s_DF0 = 0.0;
        stats.s_T0 = TDoubleVec2Vec(stats.s_Partition.size());

        if (stats.s_Segmentation.size() > 0) {
            // We've found a piecewise linear scaling.

            std::size_t period{static_cast<std::size_t>(periods[0] / m_BucketLength)};
            TFloatMeanAccumulatorVec values(buckets.begin(), buckets.end());
            std::tie(stats.s_T0[0], stats.s_Scales) =
                CTimeSeriesSegmentation::piecewiseLinearScaledPeriodic(
                    values, period, stats.s_Segmentation);
            values = CTimeSeriesSegmentation::removePiecewiseLinearScaledPeriodic(
                values, stats.s_Segmentation, stats.s_T0[0], stats.s_Scales);

            TMeanVarAccumulator moments;
            double b{0.0};
            for (std::size_t i = 0; i < period; ++i) {
                TMeanVarAccumulator momentsAtOffset;
                for (std::size_t j = i; j < values.size(); j += period) {
                    momentsAtOffset.add(CBasicStatistics::mean(values[j]),
                                        CBasicStatistics::count(values[j]));
                }
                moments += momentsAtOffset;
                b += (CBasicStatistics::count(momentsAtOffset) > 0.0 ? 0.0 : 1.0);
            }

            stats.s_V0 = CBasicStatistics::variance(moments);
            stats.s_DF0 = b + static_cast<double>(stats.s_Segmentation.size() - 2);
        } else {
            // We've got a standard periodic component which may also have
            // a periodically repeating partition.

            for (std::size_t i = 0; i < stats.s_Partition.size(); ++i) {
                std::size_t period{static_cast<std::size_t>(
                    std::min(periods[i], length(stats.s_Partition[i])) / m_BucketLength)};
                TTimeTimePr2Vec windows(calculateWindows(
                    stats.s_StartOfPartition, length(buckets, m_BucketLength),
                    length(stats.s_Partition), stats.s_Partition[i]));

                TMeanVarAccumulatorVec trend(periods[i] / m_BucketLength);
                TFloatMeanAccumulatorVec values(buckets.begin(), buckets.end());
                periodicTrendMinusOutliers(values, windows, m_BucketLength, trend);

                stats.s_V0 += residualVariance<double>(trend, 1.0 / stats.s_MeasurementsPerBucket);
                stats.s_T0[i].reserve(period);
                std::for_each(trend.begin(), trend.end(), [&stats, i](const TMeanVarAccumulator& value) {
                    stats.s_T0[i].push_back(CBasicStatistics::mean(value));
                    stats.s_DF0 += (CBasicStatistics::count(value) > 0.0 ? 1.0 : 0.0);
                });
            }
            stats.s_V0 /= static_cast<double>(periods.size());
        }
    }
}

void CPeriodicityHypothesisTests::conditionOnHypothesis(const STestStats& stats,
                                                        TFloatMeanAccumulatorVec& buckets) const {
    std::size_t n{buckets.size()};
    core_t::TTime windowLength{static_cast<core_t::TTime>(n) * m_BucketLength};

    if (stats.s_Segmentation.size() > 0) {
        buckets = CTimeSeriesSegmentation::removePiecewiseLinearScaledPeriodic(
            buckets, stats.s_Segmentation, stats.s_T0[0], stats.s_Scales);
    } else {
        for (std::size_t i = 0; i < stats.s_Partition.size(); ++i) {
            TTimeTimePr2Vec windows_(
                calculateWindows(stats.s_StartOfPartition, windowLength,
                                 length(stats.s_Partition), stats.s_Partition[i]));
            TSizeSizePr2Vec indexWindows;
            calculateIndexWindows(windows_, m_BucketLength, indexWindows);

            std::size_t period{stats.s_T0[i].size()};
            LOG_TRACE(<< "Conditioning on period = " << period
                      << " in windows = " << core::CContainerPrinter::print(windows_));
            for (const auto& window : indexWindows) {
                std::size_t a{window.first};
                std::size_t b{window.second};
                for (std::size_t j = a; j < b; ++j) {
                    CBasicStatistics::moment<0>(buckets[j % n]) -=
                        stats.s_T0[i][(j - a) % period];
                }
            }
        }
    }
}

bool CPeriodicityHypothesisTests::testPeriod(const TTimeTimePr2Vec& windows,
                                             const TFloatMeanAccumulatorCRng& buckets,
                                             core_t::TTime period_,
                                             STestStats& stats) const {
    // We use two tests to check for the period:
    //   1) That it explains both a non-negligible absolute and statistically
    //      significant amount of variance and the cyclic autocorrelation at
    //      that repeat is high enough OR
    //   2) There is a large absolute and statistically significant periodic
    //      spike or trough.

    LOG_TRACE(<< "Testing period " << period_);

    if (this->testStatisticsFor(buckets, stats) == false ||
        stats.nullHypothesisGoodEnough()) {
        return false;
    }
    if (stats.s_HasPeriod) {
        stats.s_R0 = stats.s_AutocorrelationThreshold;
        return true;
    }

    // Compute the effective period in buckets.
    period_ = std::min(period_, length(windows[0]));
    std::size_t period{static_cast<std::size_t>(period_ / m_BucketLength)};

    // We need to observe a minimum number of repeated values to test with
    // an acceptable false positive rate.
    if (this->seenSufficientPeriodicallyPopulatedBucketsToTest(buckets, period) == false) {
        return false;
    }

    core_t::TTime windowLength{length(windows)};
    TTimeTimePr2Vec window{{0, windowLength}};

    // Get the values minus the trend for the null hypothesis.
    TFloatMeanAccumulatorVec values(buckets.begin(), buckets.end());
    this->conditionOnHypothesis(stats, values);

    // If necessary project the set of all buckets onto the set which
    // intersect the windows.
    if (windowLength < length(buckets, m_BucketLength)) {
        LOG_TRACE(<< "Projecting onto " << core::CContainerPrinter::print(windows));
        TFloatMeanAccumulatorVec projection;
        project(values, windows, m_BucketLength, projection);
        values = std::move(projection);
    }

    // Compute the number of non-empty buckets. This needs to account for
    // projecting onto windows.
    stats.s_NonEmptyBuckets = static_cast<double>(std::count_if(
        values.begin(), values.end(), [](const TFloatMeanAccumulator& value) {
            return CBasicStatistics::count(value) > 0.0;
        }));

    // We need fewer degrees of freedom in the null hypothesis trend model
    // we're fitting than non-empty buckets.
    if (stats.s_NonEmptyBuckets <= stats.s_DF0) {
        return false;
    }

    // Fit the periodic trend and compute the degrees of freedom given the
    // alternative hypothesis.
    TMeanVarAccumulatorVec trend(period);
    periodicTrend(values, window, m_BucketLength, trend);
    double b{static_cast<double>(std::count_if(
        trend.begin(), trend.end(), [](const TMeanVarAccumulator& value) {
            return CBasicStatistics::count(value) > 0.0;
        }))};
    double df1{stats.s_NonEmptyBuckets - b};
    LOG_TRACE(<< "  populated = " << b);

    // We need fewer points in the trend model we're fitting than non-empty
    // buckets.
    if (df1 <= 0.0) {
        return false;
    }

    // Compute the residual variance in the alternative hypothesis *without*
    // removing outliers.
    double scale{1.0 / stats.s_MeasurementsPerBucket};
    LOG_TRACE(<< "  scale = " << scale);
    double v{residualVariance<double>(trend, scale)};
    v = varianceAtPercentile(v, df1, 50.0 + CONFIDENCE_INTERVAL / 2.0);

    // Fit the periodic trend re-weighting outliers.
    reweightOutliers(trend, window, m_BucketLength, values);
    trend.assign(period, TMeanVarAccumulator{});
    periodicTrend(values, window, m_BucketLength, trend);
    double v1{residualVariance<double>(trend, scale)};

    double R;
    double meanRepeats;
    double truthVariance;
    return this->testVariance(window, values, period_, df1, v1, stats, R,
                              meanRepeats, truthVariance) ||
           this->testAmplitude(window, values, period_, b, v, R, meanRepeats,
                               truthVariance, stats);
}

bool CPeriodicityHypothesisTests::testPeriodWithScaling(const TTimeTimePr2Vec& windows,
                                                        const TFloatMeanAccumulatorCRng& buckets,
                                                        core_t::TTime period_,
                                                        STestStats& stats) const {
    LOG_TRACE(<< "Testing period " << period_ << " with linear scales");

    // We check to see if the bucket values are better explained by a
    // piecewise constant linear scaling of an underlying component with
    // specified period. If they are we check that the component explains
    // both a non-negligible absolute and statistically significant amount
    // of variance and the cyclic autocorrelation at that repeat is high
    // enough.

    auto scaledPeriodic = [&stats](const TFloatMeanAccumulatorVec& values, std::size_t period) {
        // If we've already chosen a segmentation then use that otherwise
        // fit a piecewise linear scaled periodic trend.
        return stats.s_Segmentation.empty()
                   ? CTimeSeriesSegmentation::piecewiseLinearScaledPeriodic(values, period)
                   : stats.s_Segmentation;
    };
    auto variance = [](const TFloatMeanAccumulatorVec& values) {
        TMeanVarAccumulator moments;
        for (const auto& value : values) {
            moments.add(CBasicStatistics::mean(value), CBasicStatistics::count(value));
        }
        return CBasicStatistics::variance(moments);
    };
    auto amplitude = [](const TDoubleVec& trend) {
        CBasicStatistics::CMinMax<double> minmax;
        minmax.add(trend);
        return minmax.range() / 2.0;
    };

    if (this->testStatisticsFor(buckets, stats) == false ||
        stats.nullHypothesisGoodEnough()) {
        return false;
    }

    // Compute the period in buckets.
    std::size_t period{static_cast<std::size_t>(period_ / m_BucketLength)};

    if (stats.s_HasPeriod) {
        TFloatMeanAccumulatorVec values(buckets.begin(), buckets.end());
        stats.s_R0 = stats.s_AutocorrelationThreshold;
        stats.s_Segmentation = scaledPeriodic(values, period);
        if (stats.s_Segmentation.size() == 2) {
            stats.s_Segmentation.clear();
        }
        return true;
    }

    // We need fewer degrees of freedom in the null hypothesis trend model
    // we're fitting than non-empty buckets.
    if (stats.s_NonEmptyBuckets <= stats.s_DF0) {
        return false;
    }

    // We need to observe a minimum number of repeated values to test with
    // an acceptable false positive rate.
    if (this->seenSufficientPeriodicallyPopulatedBucketsToTest(buckets, period) == false) {
        return false;
    }

    // Get the values minus the trend for the null hypothesis.
    TFloatMeanAccumulatorVec values(buckets.begin(), buckets.end());
    this->conditionOnHypothesis(stats, values);

    TSizeVec segmentation(scaledPeriodic(values, period));
    LOG_TRACE(<< "  segmentation = " << core::CContainerPrinter::print(segmentation));

    // Check if we have a (suitable) segmentation.
    if (segmentation.size() == 2) {
        return false;
    }

    // Compute the residual variance and remove the "scaling" from the
    // periodic component.
    TDoubleVec trend;
    TDoubleVec scales;
    std::tie(trend, scales) = CTimeSeriesSegmentation::piecewiseLinearScaledPeriodic(
        values, period, segmentation);
    LOG_TRACE(<< "  trend = " << core::CContainerPrinter::print(trend));
    LOG_TRACE(<< "  scales = " << core::CContainerPrinter::print(scales));
    TMeanAccumulator scale_;
    for (std::size_t i = 1; i < segmentation.size(); ++i) {
        scale_.add(scales[i - 1],
                   static_cast<double>(segmentation[i] - segmentation[i - 1]));
    }
    double scale{CBasicStatistics::mean(scale_)};
    LOG_TRACE(<< "  scale = " << scale);
    values = CTimeSeriesSegmentation::removePiecewiseLinearScaledPeriodic(
        values, segmentation, trend, scales);
    double v1{variance(values)};
    double noise{std::sqrt(v1) / amplitude(trend)};
    LOG_TRACE(<< "  noise = " << noise);
    for (std::size_t i = 0; i < values.size(); ++i) {
        // If the component is "scaled away" in a segment we treat that
        // segment as missing so we don't erroneously detect a periodic
        // signal when we've seen too few real repeats.
        auto index = std::upper_bound(segmentation.begin(), segmentation.end(), i);
        if (scales[(index - segmentation.begin()) - 1] <= noise) {
            values[i] = TFloatMeanAccumulator{};
        } else {
            CBasicStatistics::moment<0>(values[i]) += scale * trend[i % period];
        }
    }

    // Re-check after (potentially) removing buckets.
    if (this->seenSufficientPeriodicallyPopulatedBucketsToTest(values, period) == false) {
        return false;
    }

    // Compute the degrees of freedom given the alternative hypothesis.
    double b{[&windows, &period_, &values, this] {
        TDoubleVec repeats(calculateRepeats(windows, period_, m_BucketLength, values));
        return static_cast<double>(
            std::count_if(repeats.begin(), repeats.end(),
                          [](double repeat) { return repeat > 0.0; }));
    }()};
    double df1{stats.s_NonEmptyBuckets - b - static_cast<double>(segmentation.size() - 2)};
    LOG_TRACE(<< "  populated = " << b);

    // We need fewer degrees of freedom in the trend model we're fitting
    // than non-empty buckets.
    if (df1 <= 0.0) {
        return false;
    }

    double R;
    double meanRepeats;
    double truthVariance;
    return this->testVariance({{0, length(windows)}}, values, period_, df1, v1,
                              stats, R, meanRepeats, truthVariance, segmentation);
}

bool CPeriodicityHypothesisTests::testPartition(const TTimeTimePr2Vec& partition,
                                                const TFloatMeanAccumulatorCRng& buckets,
                                                core_t::TTime period_,
                                                double correction,
                                                STestStats& stats) const {
    // We check to see if partitioning the data as well as fitting a periodic
    // component with period "period_" explains a non-negligible absolute and
    // statistically significant amount of variance and the cyclic correlation
    // at the repeat of at least one of the partition's periodic components
    // is high enough. In order to do this we search over all permitted offsets
    // for the start of the split.

    using TDoubleTimePr = std::pair<double, core_t::TTime>;
    using TDoubleTimePrVec = std::vector<TDoubleTimePr>;
    using TMinAccumulator = CBasicStatistics::SMin<TDoubleTimePr>::TAccumulator;
    using TMeanVarAccumulatorBuffer = boost::circular_buffer<TMeanVarAccumulator>;

    LOG_TRACE(<< "Testing partition " << core::CContainerPrinter::print(partition)
              << " with period " << period_);

    if (this->testStatisticsFor(buckets, stats) == false ||
        stats.nullHypothesisGoodEnough()) {
        return false;
    }
    if (stats.s_HasPartition) {
        stats.s_R0 = stats.s_AutocorrelationThreshold;
        return true;
    }

    std::size_t period{static_cast<std::size_t>(period_ / m_BucketLength)};
    core_t::TTime windowLength{length(buckets, m_BucketLength)};
    core_t::TTime repeat{length(partition)};
    double scale{1.0 / stats.s_MeasurementsPerBucket};
    LOG_TRACE(<< "scale = " << scale);

    // We need to observe a minimum number of repeated values to test with
    // an acceptable false positive rate.
    if (this->seenSufficientPeriodicallyPopulatedBucketsToTest(buckets, period) == false) {
        return false;
    }

    // Find the partition of the data such that the residual variance
    // w.r.t. the period is minimized and check if there is significant
    // evidence that it reduces the residual variance and repeats.

    double B{stats.s_NonEmptyBuckets};
    double df0{B - stats.s_DF0};

    // We need fewer degrees of freedom in the null hypothesis trend model
    // we're fitting than non-empty buckets.
    if (df0 <= 0.0) {
        return false;
    }

    TFloatMeanAccumulatorVec values(buckets.begin(), buckets.end());
    this->conditionOnHypothesis(stats, values);
    {
        TTimeTimePr2Vec window{{0, windowLength}};
        TMeanAccumulatorVec trend(period);
        periodicTrend(values, window, m_BucketLength, trend);
        reweightOutliers(trend, window, m_BucketLength, values);
    }

    double v0{varianceAtPercentile(stats.s_V0, df0, 50.0 + CONFIDENCE_INTERVAL / 2.0)};
    double vt{stats.s_VarianceThreshold * v0};
    LOG_TRACE(<< "period = " << period);

    core_t::TTime startOfPartition{stats.s_StartOfPartition};
    TTimeTimePr2Vec windows[]{
        calculateWindows(startOfPartition, windowLength, repeat, partition[0]),
        calculateWindows(startOfPartition, windowLength, repeat, partition[1])};
    LOG_TRACE(<< "windows = " << core::CContainerPrinter::print(windows));

    TTimeVec deltas[2];
    deltas[0].reserve((length(partition[0]) * windowLength) / (period_ * repeat));
    deltas[1].reserve((length(partition[1]) * windowLength) / (period_ * repeat));
    for (std::size_t j = 0u; j < 2; ++j) {
        for (const auto& window : windows[j]) {
            core_t::TTime a_{window.first};
            core_t::TTime b_{window.second};
            for (core_t::TTime t = a_ + period_; t <= b_; t += period_) {
                deltas[j].push_back(t - m_BucketLength);
            }
        }
    }
    LOG_TRACE(<< "deltas = " << core::CContainerPrinter::print(deltas));

    TMeanVarAccumulatorBuffer trends[]{
        TMeanVarAccumulatorBuffer(period, TMeanVarAccumulator()),
        TMeanVarAccumulatorBuffer(period, TMeanVarAccumulator())};
    periodicTrend(values, windows[0], m_BucketLength, trends[0]);
    periodicTrend(values, windows[1], m_BucketLength, trends[1]);

    TMeanAccumulator variances[]{residualVariance<TMeanAccumulator>(trends[0], scale),
                                 residualVariance<TMeanAccumulator>(trends[1], scale)};
    LOG_TRACE(<< "variances = " << core::CContainerPrinter::print(variances));

    TMinAccumulator minimum;
    TDoubleTimePrVec candidates;
    candidates.reserve(period);

    double variance{(residualVariance(variances[0]) + residualVariance(variances[1])) / 2.0};
    minimum.add({variance, 0});
    candidates.emplace_back(variance, 0);
    for (core_t::TTime time = m_BucketLength; time < repeat; time += m_BucketLength) {
        for (std::size_t j = 0u; j < 2; ++j) {
            for (auto& delta : deltas[j]) {
                delta = (delta + m_BucketLength) % windowLength;
            }
            TMeanVarAccumulator oldBucket{trends[j].front()};
            TMeanVarAccumulator newBucket;
            averageValue(values, deltas[j], m_BucketLength, newBucket);

            trends[j].pop_front();
            trends[j].push_back(newBucket);
            variances[j] -= residualVariance(oldBucket, scale);
            variances[j] += residualVariance(newBucket, scale);
        }
        variance = (residualVariance(variances[0]) + residualVariance(variances[1])) / 2.0;
        minimum.add({variance, time});
        if (variance <= 1.05 * minimum[0].first) {
            candidates.emplace_back(variance, time);
        }
    }

    double b{0.0};
    TMinAccumulator best;

    for (const auto& candidate : candidates) {
        if (candidate.first <= 1.05 * minimum[0].first) {
            startOfPartition = candidate.second;
            TMeanAccumulator cost;
            for (const auto& window : calculateWindows(startOfPartition, windowLength,
                                                       repeat, partition[0])) {
                core_t::TTime a_{window.first / m_BucketLength};
                core_t::TTime b_{window.second / m_BucketLength - 1};
                double va{CBasicStatistics::mean(values[a_ % values.size()])};
                double vb{CBasicStatistics::mean(values[b_ % values.size()])};
                cost.add(std::fabs(va) + std::fabs(vb) + std::fabs(vb - va));
            }
            if (best.add({CBasicStatistics::mean(cost), startOfPartition})) {
                b = 0.0;
                for (std::size_t i = 0u; i < partition.size(); ++i) {
                    windows[i] = calculateWindows(startOfPartition, windowLength,
                                                  repeat, partition[i]);
                    TMeanVarAccumulatorVec trend(period);
                    periodicTrend(values, windows[i], m_BucketLength, trend);
                    b += static_cast<double>(std::count_if(
                        trend.begin(), trend.end(), [](const TMeanVarAccumulator& value) {
                            return CBasicStatistics::count(value) > 0.0;
                        }));
                }
            }
        }
    }

    double df1{B - b};

    // We need fewer degrees of freedom in the trend model we're fitting
    // than non-empty buckets.
    if (df1 <= 0.0) {
        return false;
    }

    // It's possible that none of the candidates are <= 1.05 times the minimum,
    // this would be the case if a NaN were present in the values say.
    // NaNs are detected and purged elsewhere so we simply return false here.
    if (best.count() == 0) {
        return false;
    }

    startOfPartition = (m_StartTime + best[0].second) % repeat;
    double v1{varianceAtPercentile(correction * minimum[0].first, df1,
                                   50.0 + CONFIDENCE_INTERVAL / 2.0)};
    LOG_TRACE(<< "  start of partition = " << startOfPartition);
    LOG_TRACE(<< "  variance          = " << v1);
    LOG_TRACE(<< "  varianceThreshold = " << vt);
    LOG_TRACE(<< "  significance      = "
              << CStatisticalTests::leftTailFTest(v1 / v0, df1, df0));

    values.assign(buckets.begin(), buckets.end());
    TMeanAccumulatorVec trend;
    for (const auto& window : windows) {
        trend.assign(period, TMeanAccumulator{});
        periodicTrend(values, window, m_BucketLength, trend);
        reweightOutliers(trend, window, m_BucketLength, values);
    }

    // In the following we're trading off:
    //   1) The cyclic autocorrelation of each periodic component in the
    //      partition,
    //   2) The number of repeats we've observed of each periodic component
    //      in the partition,
    //   3) The significance of the variance reduction, and
    //   4) The amount of variance reduction.

    auto calculateMeanRepeats = [&values, this](const TTimeTimePr2Vec& w, core_t::TTime p) {
        TMeanAccumulator result;
        result.add(calculateRepeats(w, p, m_BucketLength, values));
        return CBasicStatistics::mean(result);
    };

    CFuzzyExpression correlationCondition;
    double R{-1.0};

    TFloatMeanAccumulatorVec partitionValues;
    for (const auto& window : windows) {
        project(values, window, m_BucketLength, partitionValues);

        double RW{-1.0};
        double BW{std::accumulate(
            partitionValues.begin(), partitionValues.end(), 0.0,
            [](double n, const TFloatMeanAccumulator& value) {
                return n + (CBasicStatistics::count(value) > 0.0 ? 1.0 : 0.0);
            })};
        if (BW > 1.0) {
            RW = CSignal::autocorrelation(length(window[0]) / m_BucketLength + period,
                                          partitionValues);
            RW = autocorrelationAtPercentile(RW, BW, 50.0 - CONFIDENCE_INTERVAL / 2.0);
            LOG_TRACE(<< "  autocorrelation          = " << RW);
            LOG_TRACE(<< "  autocorrelationThreshold = " << stats.s_AutocorrelationThreshold);
        }

        double meanRepeats{calculateMeanRepeats(window, period_)};
        double meanRepeatsPerSegment{meanRepeats / std::max(stats.s_TrendSegments, 1.0) /
                                     MINIMUM_REPEATS_TO_TEST_VARIANCE};
        LOG_TRACE(<< "  mean repeats per segment = " << meanRepeatsPerSegment);

        correlationCondition =
            std::max(correlationCondition,
                     softGreaterThan(R, stats.s_AutocorrelationThreshold, 0.1) &&
                         softGreaterThan(meanRepeatsPerSegment, 1.0, 0.2));
        R = std::max(R, RW);
    }

    double logSignificance{
        CTools::fastLog(CStatisticalTests::leftTailFTest(v1 / v0, df1, df0)) /
        LOG_COMPONENT_STATISTICALLY_SIGNIFICANCE};

    if (correlationCondition && softGreaterThan(logSignificance, 1.0, 0.1) &&
        (vt > v1 ? softGreaterThan(vt / v1, 1.0, 1.0) : softLessThan(v1 / vt, 1.0, 0.1))) {
        stats.s_StartOfPartition = startOfPartition;
        stats.s_R0 = R;
        return true;
    }

    return false;
}

bool CPeriodicityHypothesisTests::testVariance(const TTimeTimePr2Vec& window,
                                               const TFloatMeanAccumulatorVec& buckets,
                                               core_t::TTime period_,
                                               double df1,
                                               double v1,
                                               STestStats& stats,
                                               double& R,
                                               double& meanRepeats,
                                               double& truthVariance,
                                               const TSizeVec& segmentation) const {
    std::size_t period{static_cast<std::size_t>(period_ / m_BucketLength)};

    double df0{stats.s_NonEmptyBuckets - stats.s_DF0};
    double v0{varianceAtPercentile(stats.s_V0, df0, 50.0 + CONFIDENCE_INTERVAL / 2.0)};
    double vt{stats.s_VarianceThreshold * v0};
    v1 = varianceAtPercentile(v1, df1, 50.0 + CONFIDENCE_INTERVAL / 2.0);
    LOG_TRACE(<< "  variance          = " << v1);
    LOG_TRACE(<< "  varianceThreshold = " << vt);
    LOG_TRACE(<< "  significance      = "
              << CStatisticalTests::leftTailFTest(v1 / v0, df1, df0));

    R = CSignal::autocorrelation(period, buckets);
    R = autocorrelationAtPercentile(R, stats.s_NonEmptyBuckets,
                                    50.0 - CONFIDENCE_INTERVAL / 2.0);
    LOG_TRACE(<< "  autocorrelation          = " << R);
    LOG_TRACE(<< "  autocorrelationThreshold = " << stats.s_AutocorrelationThreshold);

    meanRepeats = [&window, &period_, &buckets, this] {
        TMeanAccumulator result;
        result.add(calculateRepeats(window, period_, m_BucketLength, buckets));
        return CBasicStatistics::mean(result);
    }();

    // We're trading off:
    //   1) The significance of the variance reduction,
    //   2) The cyclic autocorrelation of the periodic component,
    //   3) The amount of variance reduction
    //   4) The number of repeats we've observed and
    //   5) The number of segments we've used.
    //
    // Specifically, the period will just be accepted if the p-value is equal
    // to the threshold to be statistically significant, the autocorrelation
    // is equal to the threshold, the variance reduction is equal to the
    // threshold and we've observed three periods on average.

    double logSignificance{
        CTools::fastLog(CStatisticalTests::leftTailFTest(v1 / v0, df1, df0)) /
        LOG_COMPONENT_STATISTICALLY_SIGNIFICANCE};
    double meanRepeatsPerSegment{
        meanRepeats /
        std::max(stats.s_TrendSegments + static_cast<double>(segmentation.size()), 1.0) /
        MINIMUM_REPEATS_TO_TEST_VARIANCE};
    LOG_TRACE(<< "  mean repeats per segment = " << meanRepeatsPerSegment);

    auto condition = softGreaterThan(logSignificance, 1.0, 0.1) &&
                     softGreaterThan(R, stats.s_AutocorrelationThreshold, 0.1) &&
                     (vt > v1 ? softGreaterThan(vt / v1, 1.0, 1.0)
                              : softLessThan(v1 / vt, 0.1, 1.0)) &&
                     softGreaterThan(meanRepeatsPerSegment, 1.0, 0.2);
    truthVariance = condition.truthValue();
    LOG_TRACE(<< "  truth(variance) = " << truthVariance);

    if (condition) {
        stats.s_R0 = R;
        stats.s_Segmentation = segmentation;
        return true;
    }
    return false;
}

bool CPeriodicityHypothesisTests::testAmplitude(const TTimeTimePr2Vec& window,
                                                const TFloatMeanAccumulatorVec& buckets,
                                                core_t::TTime period_,
                                                double b,
                                                double v,
                                                double R,
                                                double meanRepeats,
                                                double truthVariance,
                                                STestStats& stats) const {
    core_t::TTime windowLength{length(window)};
    std::size_t period{static_cast<std::size_t>(period_ / m_BucketLength)};

    double F1{1.0};
    if (v > 0.0) {
        try {
            std::size_t n{static_cast<std::size_t>(
                std::ceil(stats.s_AutocorrelationThreshold *
                          static_cast<double>(windowLength / period_)))};
            double scale{1.0 / stats.s_MeasurementsPerBucket};
            double df0{stats.s_NonEmptyBuckets - stats.s_DF0};
            double v0{varianceAtPercentile(stats.s_V0, df0, 50.0 + CONFIDENCE_INTERVAL / 2.0)};
            double at{stats.s_AmplitudeThreshold * std::sqrt(v0 / scale)};
            LOG_TRACE(<< "  n = " << n << ", at = " << at << ", v = " << v);
            TMeanAccumulator level;
            for (const auto& value : buckets) {
                level.add(CBasicStatistics::mean(value), CBasicStatistics::count(value));
            }
            TMinAmplitudeVec amplitudes(period, {n, CBasicStatistics::mean(level)});
            periodicTrend(buckets, window, m_BucketLength, amplitudes);
            boost::math::normal normal(0.0, std::sqrt(v));
            std::for_each(amplitudes.begin(), amplitudes.end(),
                          [&F1, &normal, at](CMinAmplitude& x) {
                              if (x.amplitude() >= at) {
                                  F1 = std::min(F1, x.significance(normal));
                              }
                          });
        } catch (const std::exception& e) {
            LOG_ERROR(<< "Unable to compute significance of amplitude: " << e.what());
        }
    }
    LOG_TRACE(<< "  F(amplitude) = " << F1);

    // Trade off the test significance and the mean number of repeats
    // we've observed.
    double logSignificance{CTools::fastLog(CTools::oneMinusPowOneMinusX(F1, b)) /
                           LOG_COMPONENT_STATISTICALLY_SIGNIFICANCE};
    double meanRepeatsPerSegment{meanRepeats / std::max(stats.s_TrendSegments, 1.0) /
                                 static_cast<double>(MINIMUM_REPEATS_TO_TEST_AMPLITUDE)};
    double minusLogTruthVariance{-CTools::fastLog(truthVariance)};
    LOG_TRACE(<< "  mean repeats per segment = " << meanRepeatsPerSegment);

    if (softLessThan(minusLogTruthVariance, 0.0, 2.0) &&
        softGreaterThan(logSignificance, 1.0, 0.2) &&
        softGreaterThan(meanRepeatsPerSegment, 1.0, 0.2)) {
        stats.s_R0 = R;
        return true;
    }
    return false;
}

const double CPeriodicityHypothesisTests::ACCURATE_TEST_POPULATED_FRACTION{0.9};
const double CPeriodicityHypothesisTests::MINIMUM_COEFFICIENT_OF_VARIATION{1e-4};

CPeriodicityHypothesisTests::STestStats::STestStats(double meanMagnitude)
    : s_TrendSegments(1.0), s_HasPeriod(false), s_HasPartition(false),
      s_VarianceThreshold(COMPONENT_SIGNIFICANT_VARIANCE_REDUCTION[E_HighThreshold]),
      s_AmplitudeThreshold(SEASONAL_SIGNIFICANT_AMPLITUDE[E_HighThreshold]),
      s_AutocorrelationThreshold(SEASONAL_SIGNIFICANT_AUTOCORRELATION[E_HighThreshold]),
      s_Range(0.0), s_NonEmptyBuckets(0.0), s_MeasurementsPerBucket(0.0),
      s_MeanMagnitude(meanMagnitude), s_V0(0.0), s_R0(0.0), s_DF0(0.0),
      s_StartOfPartition(0) {
}

void CPeriodicityHypothesisTests::STestStats::setThresholds(double vt, double at, double Rt) {
    s_VarianceThreshold = vt;
    s_AmplitudeThreshold = at;
    s_AutocorrelationThreshold = Rt;
}

bool CPeriodicityHypothesisTests::STestStats::nullHypothesisGoodEnough() const {
    return std::sqrt(s_V0) <= MINIMUM_COEFFICIENT_OF_VARIATION * s_MeanMagnitude;
}

CPeriodicityHypothesisTests::CNestedHypotheses::CNestedHypotheses(TTestFunc test)
    : m_Test(test), m_AlwaysTestNested(false) {
}

CPeriodicityHypothesisTests::CNestedHypotheses::CBuilder
CPeriodicityHypothesisTests::CNestedHypotheses::null(TTestFunc test) {
    m_Test = test;
    m_AlwaysTestNested = true;
    return CBuilder(*this);
}

CPeriodicityHypothesisTests::CNestedHypotheses&
CPeriodicityHypothesisTests::CNestedHypotheses::addNested(TTestFunc test) {
    m_Nested.emplace_back(test);
    return m_Nested.back();
}

CPeriodicityHypothesisTestsResult
CPeriodicityHypothesisTests::CNestedHypotheses::test(STestStats& stats) const {
    CPeriodicityHypothesisTestsResult result{m_Test(stats)};
    if (m_AlwaysTestNested || result != stats.s_H0) {
        stats.s_H0 = result;
        for (const auto& child : m_Nested) {
            CPeriodicityHypothesisTestsResult childResult{child.test(stats)};
            if (result != childResult) {
                childResult.trend(m_Trend);
                return childResult;
            }
        }
    }
    result.trend(m_Trend);
    return result;
}

void CPeriodicityHypothesisTests::CNestedHypotheses::trend(CTrendHypothesis value) {
    m_Trend = value;
}

std::size_t CPeriodicityHypothesisTests::CNestedHypotheses::trendSegments() const {
    return m_Trend.segments();
}

CPeriodicityHypothesisTests::CNestedHypotheses::CBuilder::CBuilder(CNestedHypotheses& hypothesis) {
    m_Levels.push_back(&hypothesis);
}

CPeriodicityHypothesisTests::CNestedHypotheses::CBuilder&
CPeriodicityHypothesisTests::CNestedHypotheses::CBuilder::addNested(TTestFunc test) {
    m_Levels.push_back(&m_Levels.back()->addNested(test));
    return *this;
}

CPeriodicityHypothesisTests::CNestedHypotheses::CBuilder&
CPeriodicityHypothesisTests::CNestedHypotheses::CBuilder::addAlternative(TTestFunc test) {
    m_Levels.pop_back();
    return this->addNested(test);
}

CPeriodicityHypothesisTests::CNestedHypotheses::CBuilder&
CPeriodicityHypothesisTests::CNestedHypotheses::CBuilder::finishedNested() {
    m_Levels.pop_back();
    return *this;
}

namespace {

//! Compute the mean of the autocorrelation for \f${P, 2P, ...}\f$
//! where \f$P\f$ is \p period.
double meanAutocorrelationForPeriodicOffsets(const TDoubleVec& correlations,
                                             std::size_t window,
                                             std::size_t period) {
    auto correctForPad = [window](double correlation, std::size_t offset) {
        return correlation * static_cast<double>(window) /
               static_cast<double>(window - offset);
    };
    TMeanAccumulator result;
    for (std::size_t offset = period; offset < correlations.size(); offset += period) {
        result.add(correctForPad(correlations[offset - 1], offset));
    }
    return CBasicStatistics::mean(result);
}

//! Find the single periodic component which explains the most
//! cyclic autocorrelation.
std::size_t mostSignificantPeriodicComponent(TFloatMeanAccumulatorVec values) {
    using TDoubleSizePr = std::pair<double, std::size_t>;
    using TMaxAccumulator =
        CBasicStatistics::COrderStatisticsHeap<TDoubleSizePr, std::greater<TDoubleSizePr>>;
    using TFloatMeanAccumulatorCRng = core::CVectorRange<const TFloatMeanAccumulatorVec>;

    std::size_t n{values.size()};
    std::size_t pad{n / 3};

    // Compute the serial autocorrelations padding to the maximum offset
    // to avoid windowing effects.
    TDoubleVec correlations;
    values.resize(n + pad);
    CSignal::autocorrelations(values, correlations);
    values.resize(n);

    // We retain the top 15 serial autocorrelations so we have a high
    // chance of finding the highest cyclic autocorrelation. Note, we
    // average over offsets which are integer multiples of the period
    // since these should have high autocorrelation if the signal is
    // periodic.
    TMaxAccumulator candidates(15);
    correlations.resize(pad);
    for (std::size_t p = 4u; p < correlations.size(); ++p) {
        double correlation{meanAutocorrelationForPeriodicOffsets(correlations, n, p)};
        LOG_TRACE(<< "correlation(" << p << ") = " << correlation);
        candidates.add({correlation, p});
    }

    // Sort by decreasing cyclic autocorrelation.
    TSizeVec candidatePeriods(candidates.count());
    std::transform(
        candidates.begin(), candidates.end(), candidatePeriods.begin(),
        [](const TDoubleSizePr& candidate_) { return candidate_.second; });
    candidates.clear();
    for (const auto period : candidatePeriods) {
        TFloatMeanAccumulatorCRng window(values, 0, period * (n / period));
        candidates.add({CSignal::autocorrelation(period, window), period});
    }
    candidates.sort();
    LOG_TRACE(<< "candidate periods = " << candidates.print());

    // We prefer shorter periods if the decision is close because
    // if there is some periodic component with period p in the
    // signal it is possible by chance that n * p + eps for n > 1
    // ends up with higher autocorrelation due to additive noise.
    std::size_t result{candidates[0].second};
    double cutoff{0.9 * candidates[0].first};
    for (auto i = candidates.begin() + 1; i != candidates.end() && i->first > cutoff; ++i) {
        if (i->second < result && candidates[0].second % i->second == 0) {
            result = i->second;
        }
    }

    return result;
}
}

CPeriodicityHypothesisTestsResult
testForPeriods(const CPeriodicityHypothesisTestsConfig& config,
               core_t::TTime startTime,
               core_t::TTime bucketLength,
               const TFloatMeanAccumulatorVec& values) {

    // Find the single periodic component which explains the
    // most cyclic autocorrelation.
    std::size_t period_{mostSignificantPeriodicComponent(values)};
    core_t::TTime windowLength{static_cast<core_t::TTime>(values.size()) * bucketLength};
    core_t::TTime period{static_cast<core_t::TTime>(period_) * bucketLength};
    LOG_TRACE(<< "bucket length = " << bucketLength << ", windowLength = " << windowLength
              << ", periods to test = " << period << ", # values = " << values.size());

    // Set up the hypothesis tests.
    CPeriodicityHypothesisTests test{config};
    test.initialize(startTime, bucketLength, windowLength, period);
    core_t::TTime time{bucketLength / 2};
    for (const auto& value : values) {
        test.add(time, CBasicStatistics::mean(value), CBasicStatistics::count(value));
        time += bucketLength;
    }

    return test.test();
}
}
}
