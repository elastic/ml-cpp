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
#include <maths/CRegression.h>
#include <maths/CRegressionDetail.h>
#include <maths/CSeasonalTime.h>
#include <maths/CSignal.h>
#include <maths/CStatisticalTests.h>
#include <maths/CTools.h>
#include <maths/Constants.h>

#include <boost/circular_buffer.hpp>
#include <boost/math/distributions/binomial.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/fisher_f.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/range.hpp>
#include <boost/ref.hpp>

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
    double s_Vt;
    double s_Rt;
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
    using TRegression = CRegression::CLeastSquaresOnline<1, double>;
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
    std::ptrdiff_t index{std::min(std::lower_bound(boost::begin(BUCKETS_PER_WEEK),
                                                   boost::end(BUCKETS_PER_WEEK), bucketsPerWeek) -
                                      boost::begin(BUCKETS_PER_WEEK),
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
    if (!values.empty()) {
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
TSizeVec calculateRepeats(const TSizeSizePr2Vec& windows,
                          std::size_t period,
                          TFloatMeanAccumulatorVec& values) {
    TSizeVec result(std::min(period, length(windows[0])), 0);
    std::size_t n{values.size()};
    for (const auto& window : windows) {
        std::size_t a{window.first};
        std::size_t b{window.second};
        for (std::size_t i = a; i < b; ++i) {
            if (CBasicStatistics::count(values[i % n]) > 0.0) {
                ++result[(i - a) % period];
            }
        }
    }
    return result;
}

//! Calculate the number of non-empty buckets at each bucket offset in
//! the period for the \p values in \p windows.
TSizeVec calculateRepeats(const TTimeTimePr2Vec& windows_,
                          core_t::TTime period,
                          core_t::TTime bucketLength,
                          TFloatMeanAccumulatorVec& values) {
    TSizeSizePr2Vec windows;
    calculateIndexWindows(windows_, bucketLength, windows);
    return calculateRepeats(windows, period / bucketLength, values);
}

//! Reweight outliers from \p values.
//!
//! These are defined as some fraction of the values which are most
//! different from the periodic trend on the time windows \p windows_.
void reweightOutliers(const TMeanVarAccumulatorVec& trend,
                      const TTimeTimePr2Vec& windows_,
                      core_t::TTime bucketLength,
                      TFloatMeanAccumulatorVec& values) {
    using TDoubleSizePr = std::pair<double, std::size_t>;
    using TMaxAccumulator =
        CBasicStatistics::COrderStatisticsHeap<TDoubleSizePr, std::greater<TDoubleSizePr>>;

    if (!values.empty()) {
        TSizeSizePr2Vec windows;
        calculateIndexWindows(windows_, bucketLength, windows);
        std::size_t period{trend.size()};
        std::size_t n{values.size()};

        TSizeVec repeats{calculateRepeats(windows, period, values)};
        double excess{std::accumulate(
            repeats.begin(), repeats.end(), 0.0, [](double excess_, std::size_t repeat) {
                return excess_ + static_cast<double>(repeat > 1 ? repeat - 1 : 0);
            })};
        std::size_t numberOutliers{static_cast<std::size_t>(SEASONAL_OUTLIER_FRACTION * excess)};
        LOG_TRACE(<< "Number outliers = " << numberOutliers);

        if (numberOutliers > 0) {
            TMaxAccumulator outliers{numberOutliers};
            TMeanAccumulator meanDifference;
            for (const auto& window : windows) {
                std::size_t a{window.first};
                std::size_t b{window.second};
                for (std::size_t j = a; j < b; ++j) {
                    const TFloatMeanAccumulator& value{values[j % n]};
                    std::size_t offset{(j - a) % period};
                    double difference{std::fabs(CBasicStatistics::mean(value) -
                                                CBasicStatistics::mean(trend[offset]))};
                    if (CBasicStatistics::count(value) > 0.0) {
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
}

//! Compute the periodic trend from \p values falling in \p windows.
template<typename U, typename V>
void periodicTrend(const U& values,
                   const TSizeSizePr2Vec& windows_,
                   core_t::TTime bucketLength,
                   V& trend) {
    if (!trend.empty()) {
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
template<typename U, typename V>
void periodicTrendMinusOutliers(U& values,
                                const TSizeSizePr2Vec& windows,
                                core_t::TTime bucketLength,
                                V& trend) {
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
    return CBasicStatistics::accumulator(scale * CBasicStatistics::count(bucket),
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
template<typename R, typename T>
R residualVariance(const T& trend, double scale) {
    TMeanAccumulator result;
    for (const auto& bucket : trend) {
        result.add(CBasicStatistics::maximumLikelihoodVariance(bucket),
                   CBasicStatistics::count(bucket));
    }
    result.s_Count *= scale;
    return SResidualVarianceImpl<R>::get(result);
}
}

bool CPeriodicityHypothesisTestsResult::
operator==(const CPeriodicityHypothesisTestsResult& other) const {
    return m_Components == other.m_Components;
}

const CPeriodicityHypothesisTestsResult& CPeriodicityHypothesisTestsResult::
operator+=(const CPeriodicityHypothesisTestsResult& other) {
    m_Components.insert(m_Components.end(), other.m_Components.begin(),
                        other.m_Components.end());
    return *this;
}

void CPeriodicityHypothesisTestsResult::add(const std::string& description,
                                            bool diurnal,
                                            core_t::TTime startOfPartition,
                                            core_t::TTime period,
                                            const TTimeTimePr& window,
                                            double precedence) {
    m_Components.emplace_back(description, diurnal, startOfPartition, period,
                              window, precedence);
}

void CPeriodicityHypothesisTestsResult::remove(const std::string& description) {
    auto i = std::find_if(m_Components.begin(), m_Components.end(),
                          [&description](const SComponent& component) {
                              return component.s_Description == description;
                          });
    if (i != m_Components.end()) {
        m_Components.erase(i);
    }
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

CPeriodicityHypothesisTestsResult::SComponent::SComponent()
    : s_Description(""), s_Diurnal(false), s_StartOfPartition(0), s_Period(0),
      s_Precedence(0.0) {
}

CPeriodicityHypothesisTestsResult::SComponent::SComponent(const std::string& description,
                                                          bool diurnal,
                                                          core_t::TTime startOfPartition,
                                                          core_t::TTime period,
                                                          const TTimeTimePr& window,
                                                          double precedence)
    : s_Description(description), s_Diurnal(diurnal),
      s_StartOfPartition(startOfPartition), s_Period(period), s_Window(window),
      s_Precedence(precedence) {
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

CPeriodicityHypothesisTestsConfig::CPeriodicityHypothesisTestsConfig()
    : m_TestForDiurnal(true), m_HasDaily(false), m_HasWeekend(false),
      m_HasWeekly(false), m_StartOfWeek(0) {
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

CPeriodicityHypothesisTests::CPeriodicityHypothesisTests()
    : m_BucketLength(0), m_WindowLength(0), m_Period(0) {
}
CPeriodicityHypothesisTests::CPeriodicityHypothesisTests(const CPeriodicityHypothesisTestsConfig& config)
    : m_Config(config), m_BucketLength(0), m_WindowLength(0), m_Period(0) {
}

bool CPeriodicityHypothesisTests::initialized() const {
    return m_BucketValues.size() > 0;
}

void CPeriodicityHypothesisTests::initialize(core_t::TTime bucketLength,
                                             core_t::TTime windowLength,
                                             core_t::TTime period) {
    m_BucketLength = bucketLength;
    m_WindowLength = windowLength;
    m_BucketValues.resize(static_cast<std::size_t>(windowLength / m_BucketLength));
    m_Period = period;
}

void CPeriodicityHypothesisTests::add(core_t::TTime time, double value, double weight) {
    if (!m_BucketValues.empty()) {
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

    if (!this->initialized()) {
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

    TFloatMeanAccumulatorVec detrendedBucketValues(m_BucketValues);
    removeLinearTrend(detrendedBucketValues);

    TTimeTimePr2Vec windowForTestingDaily(window(DAY));
    TTimeTimePr2Vec windowForTestingWeekly(window(WEEK));
    TTimeTimePr2Vec windowForTestingPeriod(window(m_Period));
    TFloatMeanAccumulatorCRng bucketsForTestingDaily[]{
        {m_BucketValues, 0, buckets(DAY)}, {detrendedBucketValues, 0, buckets(DAY)}};
    TFloatMeanAccumulatorCRng bucketsForTestingWeekly[]{
        {m_BucketValues, 0, buckets(WEEK)}, {detrendedBucketValues, 0, buckets(WEEK)}};
    TFloatMeanAccumulatorCRng bucketsForTestingPeriod[]{
        {m_BucketValues, 0, buckets(m_Period)},
        {detrendedBucketValues, 0, buckets(m_Period)}};

    LOG_TRACE(<< "Testing periodicity hypotheses");
    LOG_TRACE(<< "window for daily = "
              << core::CContainerPrinter::print(windowForTestingDaily));
    LOG_TRACE(<< "window for weekly = "
              << core::CContainerPrinter::print(windowForTestingWeekly));
    LOG_TRACE(<< "window for period = "
              << core::CContainerPrinter::print(windowForTestingPeriod));

    TNestedHypothesesVec hypotheses;

    for (std::size_t i : {0, 1}) {
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

        hypotheses.insert(hypotheses.end(), hypotheses_.begin(), hypotheses_.end());
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
        auto testForNull = boost::bind(&CPeriodicityHypothesisTests::testForNull,
                                       this, boost::cref(windowForTestingWeekly),
                                       boost::cref(bucketsForTestingWeekly), _1);
        auto testForPeriod = boost::bind(&CPeriodicityHypothesisTests::testForPeriod,
                                         this, boost::cref(windowForTestingWeekly),
                                         boost::cref(bucketsForTestingWeekly), _1);
        auto testForDaily = boost::bind(&CPeriodicityHypothesisTests::testForDaily,
                                        this, boost::cref(windowForTestingWeekly),
                                        boost::cref(bucketsForTestingWeekly), _1);
        auto testForWeekly = boost::bind(&CPeriodicityHypothesisTests::testForWeekly,
                                         this, boost::cref(windowForTestingWeekly),
                                         boost::cref(bucketsForTestingWeekly), _1);
        auto testForDailyWithWeekend =
            boost::bind(&CPeriodicityHypothesisTests::testForDailyWithWeekend,
                        this, boost::cref(bucketsForTestingWeekly), _1);
        auto testForWeeklyGivenWeekend = boost::bind(
            &CPeriodicityHypothesisTests::testForWeeklyGivenDailyWithWeekend,
            this, boost::cref(windowForTestingWeekly),
            boost::cref(bucketsForTestingWeekly), _1);

        hypotheses.resize(1);
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
            // clang-format on
        }
    } else if (m_Period % WEEK == 0) {
        auto testForNull = boost::bind(&CPeriodicityHypothesisTests::testForNull,
                                       this, boost::cref(windowForTestingPeriod),
                                       boost::cref(bucketsForTestingPeriod), _1);
        auto testForPeriod = boost::bind(&CPeriodicityHypothesisTests::testForPeriod,
                                         this, boost::cref(windowForTestingPeriod),
                                         boost::cref(bucketsForTestingPeriod), _1);
        auto testForDaily = boost::bind(&CPeriodicityHypothesisTests::testForDaily,
                                        this, boost::cref(windowForTestingPeriod),
                                        boost::cref(bucketsForTestingPeriod), _1);
        auto testForWeekly = boost::bind(&CPeriodicityHypothesisTests::testForWeekly,
                                         this, boost::cref(windowForTestingPeriod),
                                         boost::cref(bucketsForTestingPeriod), _1);
        auto testForDailyWithWeekend =
            boost::bind(&CPeriodicityHypothesisTests::testForDailyWithWeekend,
                        this, boost::cref(bucketsForTestingPeriod), _1);
        auto testForWeeklyGivenWeekend = boost::bind(
            &CPeriodicityHypothesisTests::testForWeeklyGivenDailyWithWeekend,
            this, boost::cref(windowForTestingPeriod),
            boost::cref(bucketsForTestingPeriod), _1);

        hypotheses.resize(1);
        // clang-format off
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
        // clang-format on
    } else {
        {
            auto testForNull = boost::bind(&CPeriodicityHypothesisTests::testForNull,
                                           this, boost::cref(windowForTestingWeekly),
                                           boost::cref(bucketsForTestingWeekly), _1);
            auto testForDaily = boost::bind(&CPeriodicityHypothesisTests::testForDaily,
                                            this, boost::cref(windowForTestingWeekly),
                                            boost::cref(bucketsForTestingWeekly), _1);
            auto testForWeekly =
                boost::bind(&CPeriodicityHypothesisTests::testForWeekly, this,
                            boost::cref(windowForTestingWeekly),
                            boost::cref(bucketsForTestingWeekly), _1);
            auto testForDailyWithWeekend =
                boost::bind(&CPeriodicityHypothesisTests::testForDailyWithWeekend,
                            this, boost::cref(bucketsForTestingWeekly), _1);
            auto testForWeeklyGivenWeekend = boost::bind(
                &CPeriodicityHypothesisTests::testForWeeklyGivenDailyWithWeekend,
                this, boost::cref(windowForTestingWeekly),
                boost::cref(bucketsForTestingWeekly), _1);

            hypotheses.resize(2);
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
            // clang-format on
        }
        if (m_Period % DAY == 0) {
            auto testForNull = boost::bind(&CPeriodicityHypothesisTests::testForNull,
                                           this, boost::cref(windowForTestingPeriod),
                                           boost::cref(bucketsForTestingPeriod), _1);
            auto testForDaily = boost::bind(&CPeriodicityHypothesisTests::testForDaily,
                                            this, boost::cref(windowForTestingPeriod),
                                            boost::cref(bucketsForTestingPeriod), _1);
            auto testForPeriod =
                boost::bind(&CPeriodicityHypothesisTests::testForPeriod, this,
                            boost::cref(windowForTestingPeriod),
                            boost::cref(bucketsForTestingPeriod), _1);

            // clang-format off
            hypotheses[1].null(testForNull)
                             .addNested(testForDaily)
                                 .addNested(testForPeriod)
                                 .finishedNested()
                             .addAlternative(testForPeriod);
            // clang-format on
        } else {
            auto testForNull = boost::bind(&CPeriodicityHypothesisTests::testForNull,
                                           this, boost::cref(windowForTestingPeriod),
                                           boost::cref(bucketsForTestingPeriod), _1);
            auto testForPeriod =
                boost::bind(&CPeriodicityHypothesisTests::testForPeriod, this,
                            boost::cref(windowForTestingPeriod),
                            boost::cref(bucketsForTestingPeriod), _1);

            // clang-format off
            hypotheses[1].null(testForNull)
                             .addNested(testForPeriod);
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
        auto testForNull = boost::bind(&CPeriodicityHypothesisTests::testForNull,
                                       this, boost::cref(windowForTestingDaily),
                                       boost::cref(bucketsForTestingDaily), _1);
        auto testForPeriod = boost::bind(&CPeriodicityHypothesisTests::testForPeriod,
                                         this, boost::cref(windowForTestingDaily),
                                         boost::cref(bucketsForTestingDaily), _1);
        auto testForDaily = boost::bind(&CPeriodicityHypothesisTests::testForDaily,
                                        this, boost::cref(windowForTestingDaily),
                                        boost::cref(bucketsForTestingDaily), _1);

        hypotheses.resize(1);
        // clang-format off
        hypotheses[0].null(testForNull)
                         .addNested(testForPeriod)
                             .addNested(testForDaily)
                             .finishedNested()
                         .addAlternative(testForDaily);
        // clang-format on
    } else if (m_Period % DAY == 0) {
        auto testForNull = boost::bind(&CPeriodicityHypothesisTests::testForNull,
                                       this, boost::cref(windowForTestingPeriod),
                                       boost::cref(bucketsForTestingPeriod), _1);
        auto testForPeriod = boost::bind(&CPeriodicityHypothesisTests::testForPeriod,
                                         this, boost::cref(windowForTestingPeriod),
                                         boost::cref(bucketsForTestingPeriod), _1);
        auto testForDaily = boost::bind(&CPeriodicityHypothesisTests::testForDaily,
                                        this, boost::cref(windowForTestingPeriod),
                                        boost::cref(bucketsForTestingPeriod), _1);

        hypotheses.resize(1);
        // clang-format off
        hypotheses[0].null(testForNull)
                         .addNested(testForDaily)
                             .addNested(testForPeriod);
        // clang-format on
    } else {
        {
            auto testForNull = boost::bind(&CPeriodicityHypothesisTests::testForNull,
                                           this, boost::cref(windowForTestingDaily),
                                           boost::cref(bucketsForTestingDaily), _1);
            auto testForDaily = boost::bind(&CPeriodicityHypothesisTests::testForDaily,
                                            this, boost::cref(windowForTestingDaily),
                                            boost::cref(bucketsForTestingDaily), _1);

            hypotheses.resize(2);
            // clang-format off
            hypotheses[0].null(testForNull)
                             .addNested(testForDaily);
            // clang-format on
        }
        {
            auto testForNull = boost::bind(&CPeriodicityHypothesisTests::testForNull,
                                           this, boost::cref(windowForTestingPeriod),
                                           boost::cref(bucketsForTestingPeriod), _1);
            auto testForPeriod =
                boost::bind(&CPeriodicityHypothesisTests::testForPeriod, this,
                            boost::cref(windowForTestingPeriod),
                            boost::cref(bucketsForTestingPeriod), _1);
            // clang-format off
            hypotheses[1].null(testForNull)
                             .addNested(testForPeriod);
            // clang-format on
        }
    }
}

void CPeriodicityHypothesisTests::hypothesesForPeriod(const TTimeTimePr2Vec& windows,
                                                      const TFloatMeanAccumulatorCRng& buckets,
                                                      TNestedHypothesesVec& hypotheses) const {
    auto testForNull = boost::bind(&CPeriodicityHypothesisTests::testForNull, this,
                                   boost::cref(windows), boost::cref(buckets), _1);
    auto testForPeriod = boost::bind(&CPeriodicityHypothesisTests::testForPeriod, this,
                                     boost::cref(windows), boost::cref(buckets), _1);

    hypotheses.resize(1);
    // clang-format off
    hypotheses[0].null(testForNull)
                     .addNested(testForPeriod);
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

    using TMinAccumulator = CBasicStatistics::SMin<double>::TAccumulator;

    LOG_TRACE(<< "# hypotheses = " << hypotheses.size());

    CPeriodicityHypothesisTestsResult result;

    THypothesisSummaryVec summaries;
    summaries.reserve(hypotheses.size());

    for (const auto& hypothesis : hypotheses) {
        STestStats stats;
        CPeriodicityHypothesisTestsResult resultForHypothesis{hypothesis.test(stats)};
        if (stats.s_B > stats.s_DF0) {
            if (!resultForHypothesis.periodic()) {
                stats.setThresholds(
                    COMPONENT_SIGNIFICANT_VARIANCE_REDUCTION[E_HighThreshold],
                    SEASONAL_SIGNIFICANT_AMPLITUDE[E_HighThreshold],
                    SEASONAL_SIGNIFICANT_AUTOCORRELATION[E_HighThreshold]);
                stats.s_R0 = stats.s_Rt;
            }
            LOG_TRACE(<< resultForHypothesis.print());
            summaries.push_back(SHypothesisSummary{
                stats.s_V0, stats.s_R0, stats.s_B - stats.s_DF0, stats.s_Vt,
                stats.s_Rt, std::move(resultForHypothesis)});
        }
    }

    if (!summaries.empty()) {
        TMinAccumulator vmin;
        TMinAccumulator DFmin;
        for (const auto& summary : summaries) {
            vmin.add(varianceAtPercentile(summary.s_V, summary.s_DF,
                                          50.0 + CONFIDENCE_INTERVAL / 2.0) /
                     summary.s_Vt);
            DFmin.add(summary.s_DF);
        }

        TMinAccumulator pmin;
        for (const auto& summary : summaries) {
            double v{varianceAtPercentile(summary.s_V, summary.s_DF,
                                          50.0 - CONFIDENCE_INTERVAL / 2.0) /
                     summary.s_Vt / vmin[0]};
            double R{summary.s_R / summary.s_Rt};
            double DF{summary.s_DF / DFmin[0]};
            double p{CTools::logisticFunction(v, 0.2, 1.0, -1.0) *
                     CTools::logisticFunction(R, 0.2, 1.0, +1.0) *
                     CTools::logisticFunction(DF, 0.2, 1.0, +1.0)};
            LOG_TRACE(<< "p = " << p);
            if (pmin.add(-p)) {
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
                                          STestStats& stats) const {
    LOG_TRACE(<< "Testing daily on " << core::CContainerPrinter::print(windows));

    CPeriodicityHypothesisTestsResult result{stats.s_H0};

    stats.s_HasPeriod = m_Config.hasDaily();
    stats.setThresholds(COMPONENT_SIGNIFICANT_VARIANCE_REDUCTION[E_LowThreshold],
                        SEASONAL_SIGNIFICANT_AMPLITUDE[E_LowThreshold],
                        SEASONAL_SIGNIFICANT_AUTOCORRELATION[E_LowThreshold]);

    if (m_Config.testForDiurnal() && m_BucketLength <= DAY / 4 &&
        this->seenSufficientDataToTest(DAY, buckets) &&
        this->testPeriod(windows, buckets, DAY, stats)) {
        this->hypothesis({DAY}, buckets, stats);
        result.add(DIURNAL_COMPONENT_NAMES[E_Day], true, 0,
                   DIURNAL_PERIODS[static_cast<int>(E_Day) % 2],
                   DIURNAL_WINDOWS[static_cast<int>(E_Day) / 2]);
    }

    return result;
}

CPeriodicityHypothesisTestsResult
CPeriodicityHypothesisTests::testForWeekly(const TTimeTimePr2Vec& windows,
                                           const TFloatMeanAccumulatorCRng& buckets,
                                           STestStats& stats) const {
    LOG_TRACE(<< "Testing weekly on " << core::CContainerPrinter::print(windows));

    CPeriodicityHypothesisTestsResult result{stats.s_H0};

    stats.s_HasPeriod = m_Config.hasWeekly();
    stats.setThresholds(COMPONENT_SIGNIFICANT_VARIANCE_REDUCTION[E_LowThreshold],
                        SEASONAL_SIGNIFICANT_AMPLITUDE[E_LowThreshold],
                        SEASONAL_SIGNIFICANT_AUTOCORRELATION[E_LowThreshold]);

    if (m_Config.testForDiurnal() && m_BucketLength <= WEEK / 4 &&
        this->seenSufficientDataToTest(WEEK, buckets) &&
        this->testPeriod(windows, buckets, WEEK, stats)) {
        stats.s_StartOfPartition = 0;
        stats.s_Partition.assign(1, {0, length(buckets, m_BucketLength)});
        this->hypothesis({WEEK}, buckets, stats);
        result.add(DIURNAL_COMPONENT_NAMES[E_Week], true, 0,
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
        result.remove(DIURNAL_COMPONENT_NAMES[E_Day]);
        result.add(DIURNAL_COMPONENT_NAMES[E_WeekendDay], true, startOfWeek,
                   DIURNAL_PERIODS[static_cast<int>(E_WeekendDay) % 2],
                   DIURNAL_WINDOWS[static_cast<int>(E_WeekendDay) / 2], HIGH_PRIORITY);
        result.add(DIURNAL_COMPONENT_NAMES[E_WeekdayDay], true, startOfWeek,
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

    if (!m_Config.testForDiurnal()) {
        return result;
    }

    core_t::TTime startOfWeek{stats.s_StartOfPartition};

    CPeriodicityHypothesisTestsResult resultForWeekly{
        this->testForWeekly(windows, buckets, stats)};
    if (resultForWeekly != result) {
        // Note that testForWeekly sets up the hypothesis for us.
        result.add(DIURNAL_COMPONENT_NAMES[E_WeekendWeek], true, startOfWeek,
                   DIURNAL_PERIODS[static_cast<int>(E_WeekendWeek) % 2],
                   DIURNAL_WINDOWS[static_cast<int>(E_WeekendWeek) / 2], HIGH_PRIORITY);
        result.add(DIURNAL_COMPONENT_NAMES[E_WeekdayWeek], true, startOfWeek,
                   DIURNAL_PERIODS[static_cast<int>(E_WeekdayWeek) % 2],
                   DIURNAL_WINDOWS[static_cast<int>(E_WeekdayWeek) / 2], HIGH_PRIORITY);
        return result;
    }

    core_t::TTime windowLength{length(windows)};
    TTimeTimePr2Vec partition{{0, WEEKEND}, {WEEKEND, WEEK}};

    TTimeTimePr2Vec weekday(
        calculateWindows(startOfWeek, windowLength, WEEK, {WEEKEND, WEEK}));
    CPeriodicityHypothesisTestsResult resultForWeekday{
        this->testForWeekly(weekday, buckets, stats)};
    if (resultForWeekday != result) {
        stats.s_StartOfPartition = startOfWeek;
        stats.s_Partition = partition;
        this->hypothesis({DAY, WEEK}, buckets, stats);
        result.add(DIURNAL_COMPONENT_NAMES[E_WeekdayWeek], true, startOfWeek,
                   DIURNAL_PERIODS[static_cast<int>(E_WeekdayWeek) % 2],
                   DIURNAL_WINDOWS[static_cast<int>(E_WeekdayWeek) / 2], HIGH_PRIORITY);
        return result;
    }

    TTimeTimePr2Vec weekend(calculateWindows(startOfWeek, windowLength, WEEK, {0, WEEKEND}));
    CPeriodicityHypothesisTestsResult resultForWeekend{
        this->testForWeekly(weekend, buckets, stats)};
    if (resultForWeekend != result) {
        stats.s_StartOfPartition = startOfWeek;
        stats.s_Partition = partition;
        this->hypothesis({WEEK, DAY}, buckets, stats);
        result.add(DIURNAL_COMPONENT_NAMES[E_WeekendWeek], true, startOfWeek,
                   DIURNAL_PERIODS[static_cast<int>(E_WeekendWeek) % 2],
                   DIURNAL_WINDOWS[static_cast<int>(E_WeekendWeek) / 2], HIGH_PRIORITY);
    }

    return result;
}

CPeriodicityHypothesisTestsResult
CPeriodicityHypothesisTests::testForPeriod(const TTimeTimePr2Vec& windows,
                                           const TFloatMeanAccumulatorCRng& buckets,
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
        if (this->testPeriod(windows, buckets, m_Period, stats)) {
            stats.s_StartOfPartition = 0;
            stats.s_Partition.assign(1, {0, length(buckets, m_BucketLength)});
            this->hypothesis({m_Period}, buckets, stats);
            result.add(core::CStringUtils::typeToString(m_Period), false, 0,
                       m_Period, {0, m_Period});
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

bool CPeriodicityHypothesisTests::seenSufficientPeriodicallyPopulatedBucketsToTest(
    const TFloatMeanAccumulatorCRng& buckets,
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
    stats.s_B = populated;
    stats.s_M = count / stats.s_B;
    LOG_TRACE(<< "range = " << stats.s_Range << ", populatedBuckets = " << stats.s_B
              << ", valuesPerBucket = " << stats.s_M);

    return true;
}

void CPeriodicityHypothesisTests::nullHypothesis(const TTimeTimePr2Vec& window,
                                                 const TFloatMeanAccumulatorCRng& buckets,
                                                 STestStats& stats) const {
    if (this->testStatisticsFor(buckets, stats)) {
        TMeanVarAccumulatorVec trend(1);
        TFloatMeanAccumulatorVec values(buckets.begin(), buckets.end());
        periodicTrendMinusOutliers(values, window, m_BucketLength, trend);
        double mean{CBasicStatistics::mean(trend[0])};
        double v0{CBasicStatistics::variance(trend[0])};
        LOG_TRACE(<< "mean = " << mean);
        LOG_TRACE(<< "variance = " << v0);
        stats.s_DF0 = 1.0;
        stats.s_V0 = v0;
        stats.s_T0.assign(1, {mean});
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
        for (std::size_t i = 0u; i < stats.s_Partition.size(); ++i) {
            core_t::TTime period{std::min(periods[i], length(stats.s_Partition[i])) / m_BucketLength};
            TTimeTimePr2Vec windows(calculateWindows(
                stats.s_StartOfPartition, length(buckets, m_BucketLength),
                length(stats.s_Partition), stats.s_Partition[i]));

            TMeanVarAccumulatorVec trend(periods[i] / m_BucketLength);
            TFloatMeanAccumulatorVec values(buckets.begin(), buckets.end());
            periodicTrendMinusOutliers(values, windows, m_BucketLength, trend);

            stats.s_V0 += residualVariance<double>(trend, 1.0 / stats.s_M);
            stats.s_T0[i].reserve(period);
            std::for_each(trend.begin(), trend.end(), [&stats, i](const TMeanVarAccumulator& value) {
                stats.s_T0[i].push_back(CBasicStatistics::mean(value));
                stats.s_DF0 += (CBasicStatistics::count(value) > 0.0 ? 1.0 : 0.0);
            });
        }
        stats.s_V0 /= static_cast<double>(periods.size());
    }
}

void CPeriodicityHypothesisTests::conditionOnHypothesis(const TTimeTimePr2Vec& windows,
                                                        const STestStats& stats,
                                                        TFloatMeanAccumulatorVec& buckets) const {
    std::size_t n{buckets.size()};
    core_t::TTime windowLength{static_cast<core_t::TTime>(n) * m_BucketLength};
    for (std::size_t i = 0u; i < stats.s_Partition.size(); ++i) {
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

    if (length(windows) < windowLength) {
        LOG_TRACE(<< "Projecting onto " << core::CContainerPrinter::print(windows));
        TFloatMeanAccumulatorVec projection;
        project(buckets, windows, m_BucketLength, projection);
        buckets = std::move(projection);
        LOG_TRACE(<< "# values = " << buckets.size());
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

    if (!this->testStatisticsFor(buckets, stats) || stats.nullHypothesisGoodEnough()) {
        return false;
    }
    if (stats.s_HasPeriod) {
        stats.s_R0 = stats.s_Rt;
        return true;
    }

    period_ = std::min(period_, length(windows[0]));
    std::size_t period{static_cast<std::size_t>(period_ / m_BucketLength)};

    // We need to observe a minimum number of repeated values to test with
    // an acceptable false positive rate.
    if (!this->seenSufficientPeriodicallyPopulatedBucketsToTest(buckets, period)) {
        return false;
    }

    double B{static_cast<double>(
        std::count_if(buckets.begin(), buckets.end(),
                      [](const TFloatMeanAccumulator& value) {
                          return CBasicStatistics::count(value) > 0.0;
                      }))};
    double df0{B - stats.s_DF0};

    // We need fewer degrees of freedom in the null hypothesis trend model
    // we're fitting than non-empty buckets.
    if (df0 <= 0.0) {
        return false;
    }

    TTimeTimePr2Vec window{{0, length(windows)}};
    TFloatMeanAccumulatorVec values(buckets.begin(), buckets.end());
    this->conditionOnHypothesis(window, stats, values);
    TMeanVarAccumulatorVec trend(period);
    periodicTrend(values, window, m_BucketLength, trend);

    double b{static_cast<double>(std::count_if(
        trend.begin(), trend.end(), [](const TMeanVarAccumulator& value) {
            return CBasicStatistics::count(value) > 0.0;
        }))};
    double df1{B - b};
    LOG_TRACE(<< "  populated = " << b);

    // We need fewer degrees of freedom in the trend model we're fitting
    // than non-empty buckets.
    if (df1 <= 0.0) {
        return false;
    }

    double scale{1.0 / stats.s_M};
    LOG_TRACE(<< "scale = " << scale);

    double v{residualVariance<double>(trend, scale)};
    v = varianceAtPercentile(v, df1, 50.0 + CONFIDENCE_INTERVAL / 2.0);
    reweightOutliers(trend, windows, m_BucketLength, values);
    trend.assign(period, TMeanVarAccumulator{});
    periodicTrend(values, window, m_BucketLength, trend);

    // The Variance Test

    double v0{varianceAtPercentile(stats.s_V0, df0, 50.0 + CONFIDENCE_INTERVAL / 2.0)};
    double vt{stats.s_Vt * v0};
    double v1{varianceAtPercentile(residualVariance<double>(trend, scale), df1,
                                   50.0 + CONFIDENCE_INTERVAL / 2.0)};
    LOG_TRACE(<< "  variance          = " << v1);
    LOG_TRACE(<< "  varianceThreshold = " << vt);
    LOG_TRACE(<< "  significance      = "
              << CStatisticalTests::leftTailFTest(v1 / v0, df1, df0));

    double R{CSignal::autocorrelation(period, values)};
    R = autocorrelationAtPercentile(R, B, 50.0 - CONFIDENCE_INTERVAL / 2.0);
    double Rt{stats.s_Rt};
    LOG_TRACE(<< "  autocorrelation          = " << R);
    LOG_TRACE(<< "  autocorrelationThreshold = " << Rt);

    TSizeVec repeats{calculateRepeats(windows, period_, m_BucketLength, values)};
    double meanRepeats{CBasicStatistics::mean(
        std::accumulate(repeats.begin(), repeats.end(), TMeanAccumulator{},
                        [](TMeanAccumulator mean, std::size_t repeat) {
                            mean.add(static_cast<double>(repeat));
                            return mean;
                        }))};
    LOG_TRACE(<< "  relative mean repeats = " << meanRepeats);

    // We're trading off:
    //   1) The significance of the variance reduction,
    //   2) The cyclic autocorrelation of the periodic component,
    //   3) The amount of variance reduction and
    //   4) The number of repeats we've observed.
    //
    // Specifically, the period will just be accepted if the p-value is equal
    // to the threshold to be statistically significant, the autocorrelation
    // is equal to the threshold, the variance reduction is equal to the
    // threshold and we've observed three periods on average.

    double relativeLogSignificance{
        CTools::fastLog(CStatisticalTests::leftTailFTest(v1 / v0, df1, df0)) /
        LOG_COMPONENT_STATISTICALLY_SIGNIFICANCE};
    double relativeMeanRepeats{meanRepeats / MINIMUM_REPEATS_TO_TEST_VARIANCE};
    double pVariance{CTools::logisticFunction(relativeLogSignificance, 0.1, 1.0) *
                     CTools::logisticFunction(R / Rt, 0.15, 1.0) *
                     (vt > v1 ? CTools::logisticFunction(vt / v1, 1.0, 1.0, +1.0)
                              : CTools::logisticFunction(v1 / vt, 0.1, 1.0, -1.0)) *
                     CTools::logisticFunction(relativeMeanRepeats, 0.25, 1.0)};
    LOG_TRACE(<< "  p(variance) = " << pVariance);

    if (pVariance >= 0.0625) {
        stats.s_R0 = R;
        return true;
    }

    // The Amplitude Test

    double F1{1.0};
    if (v > 0.0) {
        try {
            std::size_t n{static_cast<std::size_t>(
                std::ceil(Rt * static_cast<double>(length(window) / period_)))};
            double at{stats.s_At * std::sqrt(v0 / scale)};
            LOG_TRACE(<< " n = " << n << ", at = " << at << ", v = " << v);
            TMeanAccumulator level;
            for (const auto& value : values) {
                if (CBasicStatistics::count(value) > 0.0) {
                    level.add(CBasicStatistics::mean(value));
                }
            }
            TMinAmplitudeVec amplitudes(period, {n, CBasicStatistics::mean(level)});
            periodicTrend(values, window, m_BucketLength, amplitudes);
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
    relativeLogSignificance = CTools::fastLog(1.0 - std::pow(1.0 - F1, b)) /
                              LOG_COMPONENT_STATISTICALLY_SIGNIFICANCE;
    relativeMeanRepeats = meanRepeats / static_cast<double>(2 * MINIMUM_REPEATS_TO_TEST_AMPLITUDE);
    double pAmplitude{CTools::logisticFunction(relativeLogSignificance, 0.2, 1.0) *
                      CTools::logisticFunction(relativeMeanRepeats, 0.5, 1.0)};
    LOG_TRACE(<< "  p(amplitude) = " << pAmplitude);

    if (pAmplitude >= 0.25) {
        stats.s_R0 = R;
        return true;
    }
    return false;
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
    using TMinAccumulator = CBasicStatistics::COrderStatisticsStack<TDoubleTimePr, 1>;
    using TMeanVarAccumulatorBuffer = boost::circular_buffer<TMeanVarAccumulator>;

    LOG_TRACE(<< "Testing partition " << core::CContainerPrinter::print(partition)
              << " with period " << period_);

    if (!this->testStatisticsFor(buckets, stats) || stats.nullHypothesisGoodEnough()) {
        return false;
    }
    if (stats.s_HasPartition) {
        stats.s_R0 = stats.s_Rt;
        return true;
    }

    std::size_t period{static_cast<std::size_t>(period_ / m_BucketLength)};
    core_t::TTime windowLength{length(buckets, m_BucketLength)};
    core_t::TTime repeat{length(partition)};
    double scale{1.0 / stats.s_M};
    LOG_TRACE(<< "scale = " << scale);

    // We need to observe a minimum number of repeated values to test with
    // an acceptable false positive rate.
    if (!this->seenSufficientPeriodicallyPopulatedBucketsToTest(buckets, period)) {
        return false;
    }

    // Find the partition of the data such that the residual variance
    // w.r.t. the period is minimized and check if there is significant
    // evidence that it reduces the residual variance and repeats.

    double B{static_cast<double>(std::count_if(
        buckets.begin(), buckets.end(), [](const TFloatMeanAccumulator& value) {
            return CBasicStatistics::count(value) > 0.0;
        }))};
    double df0{B - stats.s_DF0};

    // We need fewer degrees of freedom in the null hypothesis trend model
    // we're fitting than non-empty buckets.
    if (df0 <= 0.0) {
        return false;
    }

    TFloatMeanAccumulatorVec values(buckets.begin(), buckets.end());
    this->conditionOnHypothesis({{0, windowLength}}, stats, values);
    {
        TTimeTimePr2Vec window{{0, windowLength}};
        TMeanVarAccumulatorVec trend(period);
        periodicTrend(values, window, m_BucketLength, trend);
        reweightOutliers(trend, window, m_BucketLength, values);
    }

    double v0{varianceAtPercentile(stats.s_V0, df0, 50.0 + CONFIDENCE_INTERVAL / 2.0)};
    double vt{stats.s_Vt * v0};
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
    minimum.add({(residualVariance(variances[0]) + residualVariance(variances[1])) / 2.0, 0});

    TDoubleTimePrVec candidates;
    candidates.reserve(period);
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
        double variance{
            (residualVariance(variances[0]) + residualVariance(variances[1])) / 2.0};
        minimum.add({variance, time});
        if (variance <= 1.05 * minimum[0].first) {
            candidates.emplace_back(variance, time);
        }
    }

    double b{0.0};
    TMinAccumulator best;

    TTimeTimePr2Vec candidateWindows;
    for (const auto& candidate : candidates) {
        if (candidate.first <= 1.05 * minimum[0].first) {
            core_t::TTime candidateStartOfPartition{candidate.second};
            candidateWindows = calculateWindows(candidateStartOfPartition,
                                                windowLength, repeat, partition[0]);
            TMeanAccumulator cost;
            for (const auto& window : candidateWindows) {
                core_t::TTime a_{window.first / m_BucketLength};
                core_t::TTime b_{window.second / m_BucketLength - 1};
                double va{CBasicStatistics::mean(values[a_ % values.size()])};
                double vb{CBasicStatistics::mean(values[b_ % values.size()])};
                cost.add(std::fabs(va) + std::fabs(vb) + std::fabs(vb - va));
            }
            if (best.add({CBasicStatistics::mean(cost), candidateStartOfPartition})) {
                b = 0.0;
                for (const auto& subset : partition) {
                    candidateWindows = calculateWindows(
                        candidateStartOfPartition, windowLength, repeat, subset);

                    TMeanVarAccumulatorVec trend(period);
                    periodicTrend(values, candidateWindows, m_BucketLength, trend);

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

    double variance{correction * minimum[0].first};
    double v1{varianceAtPercentile(variance, df1, 50.0 + CONFIDENCE_INTERVAL / 2.0)};
    LOG_TRACE(<< "  variance          = " << v1);
    LOG_TRACE(<< "  varianceThreshold = " << vt);
    LOG_TRACE(<< "  significance      = "
              << CStatisticalTests::leftTailFTest(v1 / v0, df1, df0));

    startOfPartition = best[0].second;
    windows[0] = calculateWindows(startOfPartition, windowLength, repeat, partition[0]);
    windows[1] = calculateWindows(startOfPartition, windowLength, repeat, partition[1]);
    LOG_TRACE(<< "  start of partition = " << startOfPartition);

    // In the following we're trading off:
    //   1) The cyclic autocorrelation of each periodic component in the
    //      partition,
    //   2) The number of repeats we've observed of each periodic component
    //      in the partition,
    //   3) The significance of the variance reduction, and
    //   4) The amount of variance reduction.

    double p{0.0};
    double R{-1.0};
    double Rt{stats.s_Rt};

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
            LOG_TRACE(<< "  autocorrelationThreshold = " << Rt);
        }

        TSizeVec repeats{calculateRepeats(window, period_, m_BucketLength, values)};
        double relativeMeanRepeats{CBasicStatistics::mean(std::accumulate(
                                       repeats.begin(), repeats.end(), TMeanAccumulator{},
                                       [](TMeanAccumulator mean, std::size_t repeat_) {
                                           mean.add(static_cast<double>(repeat_));
                                           return mean;
                                       })) /
                                   MINIMUM_REPEATS_TO_TEST_VARIANCE};
        LOG_TRACE(<< "  relative mean repeats = " << relativeMeanRepeats);

        p = std::max(p, CTools::logisticFunction(RW / Rt, 0.15, 1.0) *
                            CTools::logisticFunction(relativeMeanRepeats, 0.25, 1.0));
        R = std::max(R, RW);
    }

    double relativeLogSignificance{
        CTools::fastLog(CStatisticalTests::leftTailFTest(v1 / v0, df1, df0)) /
        LOG_COMPONENT_STATISTICALLY_SIGNIFICANCE};
    p *= CTools::logisticFunction(relativeLogSignificance, 0.1, 1.0) *
         (vt > v1 ? CTools::logisticFunction(vt / v1, 1.0, 1.0, +1.0)
                  : CTools::logisticFunction(v1 / vt, 0.1, 1.0, -1.0));
    LOG_TRACE(<< "  p(partition) = " << p);

    if (p >= 0.0625) {
        stats.s_StartOfPartition = startOfPartition;
        stats.s_R0 = R;
        return true;
    }
    return false;
}

const double CPeriodicityHypothesisTests::ACCURATE_TEST_POPULATED_FRACTION{0.9};
const double CPeriodicityHypothesisTests::MINIMUM_COEFFICIENT_OF_VARIATION{1e-4};

CPeriodicityHypothesisTests::STestStats::STestStats()
    : s_HasPeriod(false), s_HasPartition(false),
      s_Vt(COMPONENT_SIGNIFICANT_VARIANCE_REDUCTION[E_HighThreshold]),
      s_At(SEASONAL_SIGNIFICANT_AMPLITUDE[E_HighThreshold]),
      s_Rt(SEASONAL_SIGNIFICANT_AUTOCORRELATION[E_HighThreshold]), s_Range(0.0),
      s_B(0.0), s_M(0.0), s_V0(0.0), s_R0(0.0), s_DF0(0.0), s_StartOfPartition(0) {
}

void CPeriodicityHypothesisTests::STestStats::setThresholds(double vt, double at, double Rt) {
    s_Vt = vt;
    s_At = at;
    s_Rt = Rt;
}

bool CPeriodicityHypothesisTests::STestStats::nullHypothesisGoodEnough() const {
    TMeanAccumulator mean;
    for (const auto& t : s_T0) {
        mean += std::accumulate(t.begin(), t.end(), TMeanAccumulator(),
                                [](TMeanAccumulator m, double x) {
                                    m.add(std::fabs(x));
                                    return m;
                                });
    }
    return std::sqrt(s_V0) <=
           MINIMUM_COEFFICIENT_OF_VARIATION * CBasicStatistics::mean(mean);
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
                return childResult;
            }
        }
    }

    return result;
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
    using TSizeVec = std::vector<std::size_t>;
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
    TSizeVec candidatePeriods(15);
    std::transform(
        candidates.begin(), candidates.end(), candidatePeriods.begin(),
        [](const TDoubleSizePr& candidate_) { return candidate_.second; });
    candidates.clear();
    for (const auto period : candidatePeriods) {
        TFloatMeanAccumulatorCRng window(values, 0, period * (values.size() / period));
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
    core_t::TTime window{static_cast<core_t::TTime>(values.size()) * bucketLength};
    core_t::TTime period{static_cast<core_t::TTime>(period_) * bucketLength};
    LOG_TRACE(<< "bucket length = " << bucketLength << ", window = " << window
              << ", periods to test = " << period << ", # values = " << values.size());

    // Set up the hypothesis tests.
    CPeriodicityHypothesisTests test{config};
    test.initialize(bucketLength, window, period);
    core_t::TTime time{startTime + bucketLength / 2};
    for (const auto& value : values) {
        test.add(time, CBasicStatistics::mean(value), CBasicStatistics::count(value));
        time += bucketLength;
    }

    return test.test();
}
}
}
