/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CTrendTests.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/Constants.h>
#include <core/CPersistUtils.h>
#include <core/CScopedLock.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CTimezone.h>
#include <core/CTriple.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>
#include <maths/CIntegerTools.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraPersist.h>
#include <maths/CLinearAlgebraTools.h>
#include <maths/CMathsFuncs.h>
#include <maths/CRegressionDetail.h>
#include <maths/CSampling.h>
#include <maths/CSeasonalTime.h>
#include <maths/CSignal.h>
#include <maths/CStatisticalTests.h>
#include <maths/CTools.h>

#include <boost/bind.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/math/distributions/binomial.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/fisher_f.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/range.hpp>
#include <boost/ref.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace ml
{
namespace maths
{
namespace
{

using TDoubleVec = std::vector<double>;
using TTimeVec = std::vector<core_t::TTime>;
using TSizeSizePr = std::pair<std::size_t, std::size_t>;
using TSizeSizePr2Vec = core::CSmallVector<TSizeSizePr, 2>;
using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;
using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
using TMeanVarAccumulatorVec = std::vector<TMeanVarAccumulator>;
using TTimeTimePr = std::pair<core_t::TTime, core_t::TTime>;
using TTimeTimePr2Vec = core::CSmallVector<TTimeTimePr, 2>;
using TTimeTimePrMeanVarAccumulatorPr = std::pair<TTimeTimePr, TMeanVarAccumulator>;

const core_t::TTime HOUR    = core::constants::HOUR;
const core_t::TTime DAY     = core::constants::DAY;
const core_t::TTime WEEKEND = core::constants::WEEKEND;
const core_t::TTime WEEK    = core::constants::WEEK;

//! The diurnal components.
enum EDiurnalComponents
{
    E_WeekendDay,
    E_WeekendWeek,
    E_WeekdayDay,
    E_WeekdayWeek,
    E_Day,
    E_Week,
};
//! The periods of the diurnal components.
const core_t::TTime DIURNAL_PERIODS[]{DAY, WEEK};
//! The weekend/day windows.
const TTimeTimePr   DIURNAL_WINDOWS[]{{0, WEEKEND}, {WEEKEND, WEEK}, {0, WEEK}};
//! The names of the the diurnal periodic components.
const std::string DIURNAL_PERIOD_NAMES[]{"daily", "weekly", ""};
//! The names of the weekday/end partitions for diurnal components.
const std::string DIURNAL_WINDOW_NAMES[]{"weekend", "weekday", ""};

//! \brief Sets the timezone to a specified value in a constructor
//! call so it can be called once by static initialisation.
struct SSetTimeZone
{
    SSetTimeZone(const std::string &zone)
    {
        core::CTimezone::instance().timezoneName(zone);
    }
};

//! \brief Accumulates the minimum amplitude.
class CMinAmplitude
{
    public:
        CMinAmplitude(std::size_t n, double level) :
                m_Level(level),
                m_Count(0),
                m_Min(std::max(n, MINIMUM_COUNT_TO_TEST)),
                m_Max(std::max(n, MINIMUM_COUNT_TO_TEST))
        {}

        void add(double x, double n)
        {
            if (n > 0.0)
            {
                ++m_Count;
                m_Min.add(x - m_Level);
                m_Max.add(x - m_Level);
            }
        }

        double amplitude(void) const
        {
            if (this->count() >= MINIMUM_COUNT_TO_TEST)
            {
                return std::max(std::max(-m_Min.biggest(), 0.0),
                                std::max( m_Max.biggest(), 0.0));
            }
            return 0.0;
        }

        double significance(const boost::math::normal &normal) const
        {
            if (this->count() < MINIMUM_COUNT_TO_TEST)
            {
                return 1.0;
            }

            double F{2.0 * CTools::safeCdf(normal, -this->amplitude())};
            if (F == 0.0)
            {
                return 0.0;
            }

            double n{static_cast<double>(this->count())};
            boost::math::binomial binomial(static_cast<double>(m_Count), F);
            return CTools::safeCdfComplement(binomial, n - 1.0);
        }

    private:
        using TMinAccumulator = CBasicStatistics::COrderStatisticsHeap<double>;
        using TMaxAccumulator = CBasicStatistics::COrderStatisticsHeap<double, std::greater<double>>;

    private:
        std::size_t count(void) const { return m_Min.count(); }

    private:
        //! The minimum number of repeats for which we'll test.
        static const std::size_t MINIMUM_COUNT_TO_TEST;

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

const std::size_t CMinAmplitude::MINIMUM_COUNT_TO_TEST{3};

using TMinAmplitudeVec = std::vector<CMinAmplitude>;

//! Generate \p n samples uniformly in the interval [\p a, \p b].
template<typename ITR>
void generateUniformSamples(boost::random::mt19937_64 &rng,
                            double a,
                            double b,
                            std::size_t n,
                            ITR samples)
{
    boost::random::uniform_real_distribution<> uniform(a, b);
    std::generate_n(samples, n, boost::bind(uniform, boost::ref(rng)));
}

//! Force the sample mean to zero.
void zeroMean(TDoubleVec &samples)
{
    TMeanAccumulator mean;
    for (auto sample : samples)
    {
        mean.add(sample);
    }
    for (auto &&sample : samples)
    {
        sample -= CBasicStatistics::mean(mean);
    }
}

//! Compute the \p percentage % variance for a chi-squared random
//! variance with \p df degrees of freedom.
double varianceAtPercentile(double variance, double df, double percentage)
{
    try
    {
        boost::math::chi_squared chi(df);
        return boost::math::quantile(chi, percentage / 100.0) / df * variance;
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Bad input: " << e.what()
                  << ", df = " << df
                  << ", percentage = " << percentage);
    }
    return variance;
}

//! Compute the \p percentage % autocorrelation for a F distributed
//! random autocorrelation with parameters \p n - 1 and \p n - 1.
double autocorrelationAtPercentile(double autocorrelation, double n, double percentage)
{
    try
    {
        boost::math::fisher_f f(n - 1.0, n - 1.0);
        return boost::math::quantile(f, percentage / 100.0) * autocorrelation;
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Bad input: " << e.what()
                  << ", n = " << n
                  << ", percentage = " << percentage);
    }
    return autocorrelation;
}

//! Get the length of the \p window.
template<typename T>
T length(const std::pair<T, T> &window)
{
    return window.second - window.first;
}

//! Get the total length of the \p windows.
template<typename T>
T length(const core::CSmallVector<std::pair<T, T>, 2> &windows)
{
    return std::accumulate(windows.begin(), windows.end(), 0,
                           [](core_t::TTime length_, const TTimeTimePr &window)
                           { return length_ + length(window); });
}

//! Compute the windows at repeat \p repeat with length \p length.
TTimeTimePr2Vec calculateWindows(core_t::TTime startOfWeek,
                                 core_t::TTime window,
                                 core_t::TTime repeat,
                                 const TTimeTimePr &interval)
{
    core_t::TTime a{startOfWeek + interval.first};
    core_t::TTime b{startOfWeek + window};
    core_t::TTime l{length(interval)};
    TTimeTimePr2Vec result;
    result.reserve((b - a) / repeat);
    for (core_t::TTime time = a; time < b; time += repeat)
    {
        result.emplace_back(time, time + l);
    }
    return result;
}

//! Get the index ranges corresponding to \p windows.
std::size_t calculateIndexWindows(const TTimeTimePr2Vec &windows,
                                  core_t::TTime bucketLength,
                                  TSizeSizePr2Vec &result)
{
    std::size_t l(0);
    result.reserve(windows.size());
    for (const auto &window : windows)
    {
        core_t::TTime a{window.first  / bucketLength};
        core_t::TTime b{window.second / bucketLength};
        result.emplace_back(a, b);
        l += b - a;
    }
    return l;
}

//! Compute the projection of \p values to \p windows.
void project(const TFloatMeanAccumulatorVec &values,
             const TTimeTimePr2Vec &windows_,
             core_t::TTime bucketLength,
             TFloatMeanAccumulatorVec &result)
{
    TSizeSizePr2Vec windows;
    calculateIndexWindows(windows_, bucketLength, windows);
    result.clear();
    result.reserve(length(windows));
    std::size_t n{values.size()};
    for (std::size_t i = 0u; i < windows.size(); ++i)
    {
        std::size_t a{windows[i].first};
        std::size_t b{windows[i].second};
        for (std::size_t j = a; j < b; ++j)
        {
            const TFloatMeanAccumulator &value{values[j % n]};
            result.push_back(value);
        }
    }
}

//! Compute the periodic trend from \p values falling in \p windows.
template<typename T>
void periodicTrend(const TFloatMeanAccumulatorVec &values,
                   const TSizeSizePr2Vec &windows_,
                   core_t::TTime bucketLength, T &trend)
{
    if (!trend.empty())
    {
        TSizeSizePr2Vec windows;
        calculateIndexWindows(windows_, bucketLength, windows);
        std::size_t period{trend.size()};
        std::size_t n{values.size()};
        for (std::size_t i = 0u; i < windows.size(); ++i)
        {
            std::size_t a{windows[i].first};
            std::size_t b{windows[i].second};
            for (std::size_t j = a; j < b; ++j)
            {
                const TFloatMeanAccumulator &value{values[j % n]};
                trend[(j - a) % period].add(CBasicStatistics::mean(value),
                                            CBasicStatistics::count(value));
            }
        }
    }
}

//! Compute the average of the values at \p times.
void averageValue(const TFloatMeanAccumulatorVec &values,
                  const TTimeVec &times,
                  core_t::TTime bucketLength,
                  TMeanVarAccumulator &value)
{
    for (auto time : times)
    {
        std::size_t index(time / bucketLength);
        value.add(CBasicStatistics::mean(values[index]),
                  CBasicStatistics::count(values[index]));
    }
}

//! Compute the variance of the \p trend values.
template<typename T>
double trendVariance(const T &trend)
{
    TMeanVarAccumulator result;
    for (const auto &value : trend)
    {
        result.add(CBasicStatistics::mean(value),
                   CBasicStatistics::count(value));
    }
    return CBasicStatistics::variance(result);
}

//! Get the maximum residual of \p trend.
template<typename T>
double trendAmplitude(const T &trend)
{
    using TMaxAccumulator = CBasicStatistics::SMax<double>::TAccumulator;

    TMeanAccumulator level;
    for (const auto &bucket : trend)
    {
        level.add(mean(bucket), count(bucket));
    }

    TMaxAccumulator result;
    result.add(0.0);
    for (const auto &bucket : trend)
    {
        if (count(bucket) > 0.0)
        {
            result.add(std::fabs(mean(bucket) - CBasicStatistics::mean(level)));
        }
    }

    return result[0];
}

//! Extract the residual variance from the mean of a collection
//! of residual variances.
double residualVariance(const TMeanAccumulator &mean)
{
    double n{CBasicStatistics::count(mean)};
    return std::max(n / (n - 1.0) * CBasicStatistics::mean(mean), 0.0);
}

//! Extract the residual variance of \p bucket of a trend.
TMeanAccumulator residualVariance(const TMeanVarAccumulator &bucket,
                                  double scale)
{
    return CBasicStatistics::accumulator(scale * CBasicStatistics::count(bucket),
                                         CBasicStatistics::maximumLikelihoodVariance(bucket));
}

//! \brief Partially specialized helper class to get the trend
//! residual variance as a specified type.
template<typename R> struct SResidualVarianceImpl {};

//! \brief Get the residual variance as a double.
template<>
struct SResidualVarianceImpl<double>
{
    static double get(const TMeanAccumulator &mean)
    {
        return residualVariance(mean);
    }
};

//! \brief Get the residual variance as a mean accumulator.
template<>
struct SResidualVarianceImpl<TMeanAccumulator>
{
    static TMeanAccumulator get(const TMeanAccumulator &mean)
    {
        return mean;
    }
};

//! Compute the residual variance of the trend \p trend.
template<typename R, typename T>
R residualVariance(const T &trend, double scale)
{
    TMeanAccumulator result;
    for (const auto &bucket : trend)
    {
        result.add(CBasicStatistics::maximumLikelihoodVariance(bucket),
                   CBasicStatistics::count(bucket));
    }
    result.s_Count *= scale;
    return SResidualVarianceImpl<R>::get(result);
}

//! Get a diurnal result.
CPeriodicityTestResult diurnalResult(core::CSmallVector<EDiurnalComponents, 4> components,
                                     core_t::TTime startOfWeek = 0)
{
    CPeriodicityTestResult result;
    for (auto component : components)
    {
        result.add(component, startOfWeek,
                   DIURNAL_PERIODS[static_cast<int>(component) % 2],
                   DIURNAL_WINDOWS[static_cast<int>(component) / 2]);

    }
    return result;
}

//! Cyclic permutation of \p values with shift \p shift.
void cyclicShift(std::size_t shift, TFloatMeanAccumulatorVec &values)
{
    std::size_t n = values.size();
    TFloatMeanAccumulatorVec result(n);
    for (std::size_t i = 0u; i < n; ++i)
    {
        result[(i + shift) % n] = values[i];
    }
    values.swap(result);
}

// CTrendTest
const std::string DECAY_RATE_TAG("a");
const std::string TIME_ORIGIN_TAG("b");
const std::string TREND_TAG("c");
const std::string VARIANCES_TAG("d");

// CRandomizedPeriodicityTest
// statics
const std::string RNG_TAG("a");
const std::string DAY_RANDOM_PROJECTIONS_TAG("b");
const std::string DAY_PERIODIC_PROJECTIONS_TAG("c");
const std::string DAY_RESAMPLED_TAG("d");
const std::string WEEK_RANDOM_PROJECTIONS_TAG("e");
const std::string WEEK_PERIODIC_PROJECTIONS_TAG("f");
const std::string WEEK_RESAMPLED_TAG("g");
const std::string ARRAY_INDEX_TAG("h");
// non-statics
const std::string DAY_PROJECTIONS_TAG("a");
const std::string DAY_STATISTICS_TAG("b");
const std::string DAY_REFRESHED_PROJECTIONS_TAG("c");
const std::string WEEK_PROJECTIONS_TAG("d");
const std::string WEEK_STATISTICS_TAG("e");
const std::string WEEK_REFRESHED_PROJECTIONS_TAG("f");

// CPeriodicityTestResult
const std::string COMPONENTS_TAG("a");

// CPeriodicityTest
const std::string WINDOW_LENGTH_TAG("a");
const std::string BUCKET_LENGTH_TAG("b");
const std::string BUCKET_VALUE_TAG("c");

// CScanningPeriodicityTest
const std::string BUCKET_LENGTH_INDEX_TAG("a");
//const std::string BUCKET_VALUE_TAG("c");
const std::string START_TIME_TAG("d");

// CCalendarCyclicTest
const std::string ERROR_QUANTILES_TAG("a");
const std::string BUCKET_TAG("c");
const std::string ERROR_COUNTS_TAG("d");
const std::string ERROR_SUMS_TAG("e");

//! The maximum significance of a test statistic.
const double MAXIMUM_SIGNIFICANCE = 0.001;
//! The confidence interval used for test statistic values.
const double CONFIDENCE_INTERVAL = 80.0;
//! The test bucket lengths for the diurnal periodicity test.
//! The data bucketing interval is snapped to the least longer
//! bucket length if there is one.
const core_t::TTime DIURNAL_PERMITTED_BUCKET_LENGTHS[]
    {
        HOUR, 2 * HOUR, 3 * HOUR, 4 * HOUR, 6 * HOUR, 8 * HOUR, 12 * HOUR, DAY
    };
//! Scales to apply to the minimum partitioned variance when
//! testing the significance.
const double DIURNAL_VARIANCE_CORRECTIONS[][8]
    {
        { 1.08, 1.08, 1.08, 1.08, 1.09, 1.1, 1.11, 1.15 },
        { 1.31, 1.31, 1.31, 1.31, 1.12, 1.0, 1.0,  1.0  }
    };

}

//////// CTrendTest ////////

CTrendTest::CTrendTest(double decayRate) :
        m_DecayRate(decayRate), m_TimeOrigin(0)
{}

bool CTrendTest::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser)
{
    do
    {
        const std::string &name = traverser.name();
        RESTORE_BUILT_IN(DECAY_RATE_TAG, m_DecayRate)
        RESTORE_BUILT_IN(TIME_ORIGIN_TAG, m_TimeOrigin)
        RESTORE(TREND_TAG, traverser.traverseSubLevel(
                               boost::bind(&TRegression::acceptRestoreTraverser, &m_Trend, _1)))
        RESTORE(VARIANCES_TAG, m_Variances.fromDelimited(traverser.value()))
    }
    while (traverser.next());
    return true;
}

void CTrendTest::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertValue(DECAY_RATE_TAG, m_DecayRate);
    inserter.insertValue(TIME_ORIGIN_TAG, m_TimeOrigin);
    inserter.insertLevel(TREND_TAG, boost::bind(&TRegression::acceptPersistInserter, &m_Trend, _1));
    inserter.insertValue(VARIANCES_TAG, m_Variances.toDelimited());
}

void CTrendTest::decayRate(double decayRate)
{
    m_DecayRate = decayRate;
}

void CTrendTest::propagateForwardsByTime(double time)
{
    if (!CMathsFuncs::isFinite(time) || time < 0.0)
    {
        LOG_ERROR("Bad propagation time " << time);
        return;
    }
    double factor = std::exp(-m_DecayRate * time);
    m_Trend.age(factor);
    m_Variances.age(factor);
}

void CTrendTest::add(core_t::TTime time, double value, double weight)
{
    core_t::TTime origin = CIntegerTools::floor(time, WEEK);
    double shift = this->time(origin);
    if (shift > 0.0)
    {
        m_Trend.shiftAbscissa(-shift);
        m_TimeOrigin = origin;
    }
    m_Trend.add(this->time(time), value, weight);
}

void CTrendTest::captureVariance(core_t::TTime time, double value, double weight)
{
    double prediction = CRegression::predict(m_Trend, this->time(time));
    TVector values;
    values(0) = value;
    values(1) = value - prediction;
    m_Variances.add(values, weight);
}

void CTrendTest::shift(double shift)
{
    m_Trend.shiftOrdinate(shift);
}

bool CTrendTest::test(void) const
{
    double n   = CBasicStatistics::count(m_Variances);
    double df0 = n - 1.0;
    double df1 = n - ORDER - 1.0;
    double v0  = CBasicStatistics::maximumLikelihoodVariance(m_Variances)(0);
    double v1  = CBasicStatistics::maximumLikelihoodVariance(m_Variances)(1);
    return   n > 3 * ORDER
          && varianceAtPercentile(v1, df1, 80.0) < MAXIMUM_TREND_VARIANCE_RATIO * v0
          && CStatisticalTests::leftTailFTest(v1 / v0, df1, df0) <= MAXIMUM_SIGNIFICANCE;
}

const CTrendTest::TRegression &CTrendTest::trend(void) const
{
    return m_Trend;
}

core_t::TTime CTrendTest::origin(void) const
{
    return m_TimeOrigin;
}

double CTrendTest::variance(void) const
{
    return CBasicStatistics::maximumLikelihoodVariance(m_Variances)(1);
}

uint64_t CTrendTest::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, m_DecayRate);
    seed = CChecksum::calculate(seed, m_TimeOrigin);
    seed = CChecksum::calculate(seed, m_Trend);
    return CChecksum::calculate(seed, m_Variances);
}

double CTrendTest::time(core_t::TTime time) const
{
    return static_cast<double>(time - m_TimeOrigin) / static_cast<double>(WEEK);
}

const double CTrendTest::MAXIMUM_TREND_VARIANCE_RATIO{0.5};

//////// CRandomizedPeriodicitytest ////////

CRandomizedPeriodicityTest::CRandomizedPeriodicityTest(void) :
        m_DayRefreshedProjections(-DAY_RESAMPLE_INTERVAL),
        m_WeekRefreshedProjections(-DAY_RESAMPLE_INTERVAL)
{
    resample(0);
}

bool CRandomizedPeriodicityTest::staticsAcceptRestoreTraverser(core::CStateRestoreTraverser &traverser)
{
    // Note we require that we only ever do one persistence per process.

    std::size_t index = 0;
    reset();

    core::CScopedLock lock(ms_Lock);

    do
    {
        const std::string &name = traverser.name();

        if (name == RNG_TAG)
        {
            // Replace '_' with space
            std::string value(traverser.value());
            std::replace(value.begin(), value.end(), '_', ' ');
            std::stringstream ss;
            ss << value;
            ss >> ms_Rng;
            continue;
        }
        RESTORE_SETUP_TEARDOWN(DAY_RESAMPLED_TAG,
                               core_t::TTime resampled,
                               core::CStringUtils::stringToType(traverser.value(), resampled),
                               ms_DayResampled.store(resampled))
        RESTORE_SETUP_TEARDOWN(WEEK_RESAMPLED_TAG,
                               core_t::TTime resampled,
                               core::CStringUtils::stringToType(traverser.value(), resampled),
                               ms_WeekResampled.store(resampled))
        RESTORE_BUILT_IN(ARRAY_INDEX_TAG, index)
        RESTORE_SETUP_TEARDOWN(DAY_RANDOM_PROJECTIONS_TAG,
                               double d,
                               core::CStringUtils::stringToType(traverser.value(), d),
                               ms_DayRandomProjections[index].push_back(d))
        RESTORE_SETUP_TEARDOWN(DAY_PERIODIC_PROJECTIONS_TAG,
                               double d,
                               core::CStringUtils::stringToType(traverser.value(), d),
                               ms_DayPeriodicProjections[index].push_back(d))
        RESTORE_SETUP_TEARDOWN(WEEK_RANDOM_PROJECTIONS_TAG,
                               double d,
                               core::CStringUtils::stringToType(traverser.value(), d),
                               ms_WeekRandomProjections[index].push_back(d))
        RESTORE_SETUP_TEARDOWN(WEEK_PERIODIC_PROJECTIONS_TAG,
                               double d,
                               core::CStringUtils::stringToType(traverser.value(), d),
                               ms_WeekPeriodicProjections[index].push_back(d))
    }
    while (traverser.next());

    return true;
}

void CRandomizedPeriodicityTest::staticsAcceptPersistInserter(core::CStatePersistInserter &inserter)
{
    // Note we require that we only ever do one persistence per process.

    core::CScopedLock lock(ms_Lock);

    std::ostringstream ss;
    ss << ms_Rng;
    std::string rng(ss.str());
    // Replace spaces else JSON parsers get confused
    std::replace(rng.begin(), rng.end(), ' ', '_');
    inserter.insertValue(RNG_TAG, rng);
    inserter.insertValue(DAY_RESAMPLED_TAG, ms_DayResampled.load());
    inserter.insertValue(WEEK_RESAMPLED_TAG, ms_WeekResampled.load());
    for (std::size_t i = 0; i < N; ++i)
    {
        inserter.insertValue(ARRAY_INDEX_TAG, i);
        for (auto rand : ms_DayRandomProjections[i])
        {
            inserter.insertValue(DAY_RANDOM_PROJECTIONS_TAG, rand);
        }
        for (auto rand : ms_DayPeriodicProjections[i])
        {
            inserter.insertValue(DAY_PERIODIC_PROJECTIONS_TAG, rand);
        }
        for (auto rand : ms_WeekRandomProjections[i])
        {
            inserter.insertValue(WEEK_RANDOM_PROJECTIONS_TAG, rand);
        }
        for (auto rand : ms_WeekPeriodicProjections[i])
        {
            inserter.insertValue(WEEK_PERIODIC_PROJECTIONS_TAG, rand);
        }
    }
}

bool CRandomizedPeriodicityTest::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser)
{
    do
    {
        const std::string &name = traverser.name();

        RESTORE(DAY_PROJECTIONS_TAG, m_DayProjections.fromDelimited(traverser.value()))
        RESTORE(DAY_STATISTICS_TAG, m_DayStatistics.fromDelimited(traverser.value()))
        RESTORE(DAY_REFRESHED_PROJECTIONS_TAG,
                core::CStringUtils::stringToType(traverser.value(),
                                                 m_DayRefreshedProjections))
        RESTORE(WEEK_PROJECTIONS_TAG, m_WeekProjections.fromDelimited(traverser.value()))
        RESTORE(WEEK_STATISTICS_TAG, m_WeekStatistics.fromDelimited(traverser.value()))
        RESTORE(DAY_STATISTICS_TAG, m_DayStatistics.fromDelimited(traverser.value()))
        RESTORE(WEEK_REFRESHED_PROJECTIONS_TAG,
                core::CStringUtils::stringToType(traverser.value(),
                                                 m_WeekRefreshedProjections))
    }
    while (traverser.next());

    return true;
}

void CRandomizedPeriodicityTest::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertValue(DAY_PROJECTIONS_TAG, m_DayProjections.toDelimited());
    inserter.insertValue(DAY_STATISTICS_TAG, m_DayStatistics.toDelimited());
    inserter.insertValue(DAY_REFRESHED_PROJECTIONS_TAG, m_DayRefreshedProjections);
    inserter.insertValue(WEEK_PROJECTIONS_TAG, m_WeekProjections.toDelimited());
    inserter.insertValue(WEEK_STATISTICS_TAG, m_WeekStatistics.toDelimited());
    inserter.insertValue(WEEK_REFRESHED_PROJECTIONS_TAG, m_WeekRefreshedProjections);
}

void CRandomizedPeriodicityTest::add(core_t::TTime time, double value)
{
    resample(time);

    if (time >= m_DayRefreshedProjections + DAY_RESAMPLE_INTERVAL)
    {
        LOG_TRACE("Updating day statistics");
        updateStatistics(m_DayProjections, m_DayStatistics);
        m_DayRefreshedProjections = CIntegerTools::floor(time, DAY_RESAMPLE_INTERVAL);
    }
    if (time >= m_WeekRefreshedProjections + WEEK_RESAMPLE_INTERVAL)
    {
        LOG_TRACE("Updating week statistics");
        updateStatistics(m_WeekProjections, m_WeekStatistics);
        m_WeekRefreshedProjections = CIntegerTools::floor(time, WEEK_RESAMPLE_INTERVAL);
    }

    TVector2N daySample;
    TVector2N weekSample;
    std::size_t td = static_cast<std::size_t>( (time % DAY_RESAMPLE_INTERVAL)
                                              / SAMPLE_INTERVAL);
    std::size_t d  = static_cast<std::size_t>( (time % DAY)
                                              / SAMPLE_INTERVAL);
    std::size_t tw = static_cast<std::size_t>( (time % WEEK_RESAMPLE_INTERVAL)
                                              / SAMPLE_INTERVAL);
    std::size_t w  = static_cast<std::size_t>( (time % WEEK)
                                              / SAMPLE_INTERVAL);

    for (std::size_t i = 0u; i < N; ++i)
    {
        daySample(2*i+0)  = ms_DayRandomProjections[i][td] * value;
        daySample(2*i+1)  = ms_DayPeriodicProjections[i][d] * value;
        weekSample(2*i+0) = ms_WeekRandomProjections[i][tw] * value;
        weekSample(2*i+1) = ms_WeekPeriodicProjections[i][w] * value;
    }

    m_DayProjections.add(daySample);
    m_WeekProjections.add(weekSample);
}

bool CRandomizedPeriodicityTest::test(void) const
{
    static const double SIGNIFICANCE = 1e-3;

    try
    {
        double nd = CBasicStatistics::count(m_DayStatistics);
        if (nd >= 1.0)
        {
            TVector2 S = CBasicStatistics::mean(m_DayStatistics);
            LOG_TRACE("Day test statistic, S = " << S << ", n = " << nd);
            double ratio = S(0) == S(1) ?
                           1.0 : (S(0) == 0.0 ? boost::numeric::bounds<double>::highest() :
                                                static_cast<double>(S(1) / S(0)));
            double significance = CStatisticalTests::rightTailFTest(ratio, nd, nd);
            LOG_TRACE("Daily significance = " << significance);
            if (significance < SIGNIFICANCE)
            {
                return true;
            }
        }

        double nw = CBasicStatistics::count(m_WeekStatistics);
        if (nw >= 1.0)
        {
            TVector2 S = CBasicStatistics::mean(m_WeekStatistics);
            LOG_TRACE("Week test statistic, S = " << S);
            double ratio = S(0) == S(1) ?
                           1.0 : (S(0) == 0.0 ? boost::numeric::bounds<double>::highest() :
                                                static_cast<double>(S(1) / S(0)));
            double significance = CStatisticalTests::rightTailFTest(ratio, nw, nw);
            LOG_TRACE("Weekly significance = " << significance);
            if (significance < SIGNIFICANCE)
            {
                return true;
            }
        }
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Failed to test for periodicity: " << e.what());
    }

    return false;
}

void CRandomizedPeriodicityTest::reset(void)
{
    core::CScopedLock lock(ms_Lock);

    ms_Rng = boost::random::mt19937_64();
    for (std::size_t i = 0u; i < N; ++i)
    {
        ms_DayRandomProjections[i].clear();
        ms_DayPeriodicProjections[i].clear();
        ms_WeekRandomProjections[i].clear();
        ms_WeekPeriodicProjections[i].clear();
    }
    ms_DayResampled = -DAY_RESAMPLE_INTERVAL;
    ms_WeekResampled = -WEEK_RESAMPLE_INTERVAL;
}

uint64_t CRandomizedPeriodicityTest::checksum(uint64_t seed) const
{
    // This checksum is problematic until we switch to using our
    // own rng for each test.
    //seed = CChecksum::calculate(seed, m_DayProjections);
    //seed = CChecksum::calculate(seed, m_DayStatistics);
    //seed = CChecksum::calculate(seed, m_DayRefreshedProjections);
    //seed = CChecksum::calculate(seed, m_WeekProjections);
    //seed = CChecksum::calculate(seed, m_WeekStatistics);
    //return CChecksum::calculate(seed, m_WeekRefreshedProjections);
    return seed;
}

void CRandomizedPeriodicityTest::updateStatistics(TVector2NMeanAccumulator &projections,
                                                  TVector2MeanAccumulator &statistics)
{
    static const double ALPHA = 0.1;

    if (CBasicStatistics::count(projections) > 0.0)
    {
        const TVector2N &mean = CBasicStatistics::mean(projections);
        LOG_TRACE("mean = " << mean);

        TVector2MeanAccumulator statistic;
        for (std::size_t i = 0u; i < N; ++i)
        {
            TVector2 s;
            s(0) = mean(2*i+0) * mean(2*i+0);
            s(1) = mean(2*i+1) * mean(2*i+1);
            statistic.add(s);
        }
        statistics += statistic;
        statistics.age(1.0 - ALPHA);
        LOG_TRACE("statistics = " << statistics);
    }

    projections = TVector2NMeanAccumulator();
}

void CRandomizedPeriodicityTest::resample(core_t::TTime time)
{
    if (time >= ms_DayResampled.load(std::memory_order_acquire) + DAY_RESAMPLE_INTERVAL)
    {
        core::CScopedLock lock(ms_Lock);

        LOG_TRACE("Updating daily random projections at " << time);
        if (time >= ms_DayResampled.load(std::memory_order_relaxed) + DAY_RESAMPLE_INTERVAL)
        {
            resample(DAY,
                     DAY_RESAMPLE_INTERVAL,
                     ms_DayPeriodicProjections,
                     ms_DayRandomProjections);
            ms_DayResampled.store(CIntegerTools::floor(time, DAY_RESAMPLE_INTERVAL),
                                  std::memory_order_release);
        }
    }

    if (time >= ms_WeekResampled.load(std::memory_order_acquire) + WEEK_RESAMPLE_INTERVAL)
    {
        core::CScopedLock lock(ms_Lock);

        LOG_TRACE("Updating weekly random projections at " << time);
        if (time >= ms_WeekResampled.load(std::memory_order_relaxed) + WEEK_RESAMPLE_INTERVAL)
        {
            resample(WEEK,
                     WEEK_RESAMPLE_INTERVAL,
                     ms_WeekPeriodicProjections,
                     ms_WeekRandomProjections);
            ms_WeekResampled.store(CIntegerTools::floor(time, WEEK_RESAMPLE_INTERVAL),
                                   std::memory_order_release);
        }
    }
}

void CRandomizedPeriodicityTest::resample(core_t::TTime period,
                                          core_t::TTime resampleInterval,
                                          TDoubleVec (&periodicProjections)[N],
                                          TDoubleVec (&randomProjections)[N])
{
    std::size_t n = static_cast<std::size_t>(period / SAMPLE_INTERVAL);
    std::size_t t = static_cast<std::size_t>(resampleInterval / SAMPLE_INTERVAL);
    std::size_t p = static_cast<std::size_t>(resampleInterval / period);
    for (std::size_t i = 0u; i < N; ++i)
    {
        periodicProjections[i].resize(n);
        generateUniformSamples(ms_Rng, -1.0, 1.0, n, periodicProjections[i].begin());
        zeroMean(periodicProjections[i]);
        randomProjections[i].resize(t);
        for (std::size_t j = 0u; j < p; ++j)
        {
            std::copy(periodicProjections[i].begin(),
                      periodicProjections[i].end(),
                      randomProjections[i].begin() + j * n);
            CSampling::random_shuffle(ms_Rng,
                                      randomProjections[i].begin() + j * n,
                                      randomProjections[i].begin() + (j+1) * n);
        }
    }
}

const core_t::TTime CRandomizedPeriodicityTest::SAMPLE_INTERVAL(3600);
const core_t::TTime CRandomizedPeriodicityTest::DAY_RESAMPLE_INTERVAL(1209600);
const core_t::TTime CRandomizedPeriodicityTest::WEEK_RESAMPLE_INTERVAL(2419200);
boost::random::mt19937_64 CRandomizedPeriodicityTest::ms_Rng = boost::random::mt19937_64();
TDoubleVec CRandomizedPeriodicityTest::ms_DayRandomProjections[N] = {};
TDoubleVec CRandomizedPeriodicityTest::ms_DayPeriodicProjections[N] = {};
std::atomic<core_t::TTime> CRandomizedPeriodicityTest::ms_DayResampled(-DAY_RESAMPLE_INTERVAL);
TDoubleVec CRandomizedPeriodicityTest::ms_WeekRandomProjections[N] = {};
TDoubleVec CRandomizedPeriodicityTest::ms_WeekPeriodicProjections[N] = {};
std::atomic<core_t::TTime> CRandomizedPeriodicityTest::ms_WeekResampled(-WEEK_RESAMPLE_INTERVAL);
core::CMutex CRandomizedPeriodicityTest::ms_Lock;

//////// CPeriodicityResultTest ////////

bool CPeriodicityTestResult::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser)
{
    do
    {
        const std::string name = traverser.name();
        RESTORE(COMPONENTS_TAG, core::CPersistUtils::fromString(traverser.value(),
                                                                SComponent::fromString,
                                                                m_Components))
    }
    while (traverser.next());
    return true;
}

void CPeriodicityTestResult::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertValue(COMPONENTS_TAG, core::CPersistUtils::toString(m_Components,
                                                                       SComponent::toString));
}

bool CPeriodicityTestResult::operator==(const CPeriodicityTestResult &other) const
{
    return m_Components == other.m_Components;
}

const CPeriodicityTestResult &CPeriodicityTestResult::operator+=(const CPeriodicityTestResult &other)
{
    m_Components.insert(m_Components.end(), other.m_Components.begin(), other.m_Components.end());
    return *this;
}

void CPeriodicityTestResult::add(unsigned int id,
                                 core_t::TTime startOfPartition,
                                 core_t::TTime period,
                                 const TTimeTimePr &window)
{
    m_Components.emplace_back(id, startOfPartition, period, window);
}

bool CPeriodicityTestResult::periodic(void) const
{
    return m_Components.size() > 0;
}

const CPeriodicityTestResult::TComponent4Vec &CPeriodicityTestResult::components(void) const
{
    return m_Components;
}

uint64_t CPeriodicityTestResult::checksum(void) const
{
    return CChecksum::calculate(0, m_Components);
}

CPeriodicityTestResult::SComponent::SComponent(void) :
        s_Id(0), s_StartOfPartition(0), s_Period(0)
{}

CPeriodicityTestResult::SComponent::SComponent(unsigned int id,
                                               core_t::TTime startOfPartition,
                                               core_t::TTime period,
                                               const TTimeTimePr &window) :
        s_Id(id),
        s_StartOfPartition(startOfPartition),
        s_Period(period),
        s_Window(window)
{}

bool CPeriodicityTestResult::SComponent::fromString(const std::string &value,
                                                    SComponent &result)
{
    boost::array<long long, 5> state;
    if (core::CPersistUtils::fromString(value, state, core::CPersistUtils::PAIR_DELIMITER))
    {
        result.s_Id = static_cast<unsigned int>(state[0]);
        result.s_StartOfPartition = static_cast<core_t::TTime>(state[1]);
        result.s_Period = static_cast<core_t::TTime>(state[2]);
        result.s_Window.first = static_cast<core_t::TTime>(state[3]);
        result.s_Window.second = static_cast<core_t::TTime>(state[4]);
        return true;
    }
    return false;
}

std::string CPeriodicityTestResult::SComponent::toString(const SComponent &component)
{
    boost::array<long long, 5> state;
    state[0] = static_cast<long long>(component.s_Id);
    state[1] = component.s_StartOfPartition;
    state[2] = component.s_Period;
    state[3] = component.s_Window.first;
    state[4] = component.s_Window.second;
    return core::CPersistUtils::toString(state, core::CPersistUtils::PAIR_DELIMITER);
}

bool CPeriodicityTestResult::SComponent::operator==(const SComponent &other) const
{
    return s_Id == other.s_Id && s_StartOfPartition == other.s_StartOfPartition;
}

uint64_t CPeriodicityTestResult::SComponent::checksum(void) const
{
    uint64_t seed = CChecksum::calculate(0, s_Id);
    seed = CChecksum::calculate(seed, s_StartOfPartition);
    seed = CChecksum::calculate(seed, s_Period);
    return CChecksum::calculate(seed, s_Window);
}

//////// CPeriodicityTest ////////

CPeriodicityTest::CPeriodicityTest(void) :
        m_DecayRate(0.0),
        m_BucketLength(0),
        m_WindowLength(0)
{}

CPeriodicityTest::CPeriodicityTest(double decayRate) :
        m_DecayRate(decayRate),
        m_BucketLength(0),
        m_WindowLength(0)
{}

bool CPeriodicityTest::initialized(void) const
{
    return m_BucketValues.size() > 0;
}

void CPeriodicityTest::propagateForwardsByTime(double time)
{
    if (!CMathsFuncs::isFinite(time) || time < 0.0)
    {
        LOG_ERROR("Bad propagation time " << time);
        return;
    }
    if (this->initialized())
    {
        double factor = std::exp(-m_DecayRate * time);
        std::for_each(m_BucketValues.begin(), m_BucketValues.end(),
                      [factor](TFloatMeanAccumulator &value) { value.age(factor); });
    }
}

void CPeriodicityTest::add(core_t::TTime time, double value, double weight)
{
    if (!m_BucketValues.empty())
    {
        std::size_t i((time % m_WindowLength) / m_BucketLength);
        m_BucketValues[i].add(value, weight);
    }
}

double CPeriodicityTest::populatedRatio(void) const
{
    if (!m_BucketValues.empty())
    {
        return static_cast<double>(
                std::count_if(m_BucketValues.begin(), m_BucketValues.end(),
                              [](const TFloatMeanAccumulator &value)
                              { return CBasicStatistics::count(value) > 0.0; })
              / static_cast<double>(m_BucketValues.size()));
    }
    return 0.0;
}

bool CPeriodicityTest::seenSufficientData(void) const
{
    double populated{0.0};
    CBasicStatistics::CMinMax<double> range;
    for (std::size_t i = 0u; i < m_BucketValues.size(); ++i)
    {
        if (CBasicStatistics::count(m_BucketValues[i]) > 0.0)
        {
            populated += 1.0;
            range.add(static_cast<double>(i));
        }
    }
    return   range.max() - range.min() >=  ACCURATE_TEST_POPULATED_FRACTION
                                         * static_cast<double>(m_WindowLength / m_BucketLength)
          && populated >  ACCURATE_TEST_POPULATED_FRACTION
                        * static_cast<double>(m_BucketValues.size());
}

void CPeriodicityTest::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    mem->setName("CPeriodicityTest");
    core::CMemoryDebug::dynamicSize("m_BucketValues", m_BucketValues, mem);
}

std::size_t CPeriodicityTest::memoryUsage(void) const
{
    return core::CMemory::dynamicSize(m_BucketValues);
}

std::size_t CPeriodicityTest::staticSize(void) const
{
    return sizeof(*this);
}

core_t::TTime CPeriodicityTest::bucketLength(void) const
{
    return m_BucketLength;
}

core_t::TTime CPeriodicityTest::windowLength(void) const
{
    return m_WindowLength;
}

const CPeriodicityTest::TFloatMeanAccumulatorVec &CPeriodicityTest::bucketValues(void) const
{
    return m_BucketValues;
}

bool CPeriodicityTest::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser)
{
    m_BucketValues.clear();
    do
    {
        const std::string &name = traverser.name();
        RESTORE_BUILT_IN(BUCKET_LENGTH_TAG, m_BucketLength)
        RESTORE_BUILT_IN(WINDOW_LENGTH_TAG, m_WindowLength)
        RESTORE(BUCKET_VALUE_TAG, core::CPersistUtils::restore(BUCKET_VALUE_TAG, m_BucketValues, traverser))
    }
    while (traverser.next());
    return true;
}

void CPeriodicityTest::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertValue(BUCKET_LENGTH_TAG, m_BucketLength);
    inserter.insertValue(WINDOW_LENGTH_TAG, m_WindowLength);
    core::CPersistUtils::persist(BUCKET_VALUE_TAG, m_BucketValues, inserter);
}

uint64_t CPeriodicityTest::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, m_DecayRate);
    seed = CChecksum::calculate(seed, m_BucketLength);
    seed = CChecksum::calculate(seed, m_WindowLength);
    return CChecksum::calculate(seed, m_BucketValues);
}

void CPeriodicityTest::initialize(core_t::TTime bucketLength,
                                  core_t::TTime windowLength,
                                  const TFloatMeanAccumulatorVec &initial)
{
    m_BucketLength = bucketLength;
    m_WindowLength = windowLength;
    m_BucketValues.resize(static_cast<std::size_t>(windowLength / m_BucketLength));
    std::copy(initial.begin(),
              initial.begin() + std::min(initial.size(), m_BucketValues.size()),
              m_BucketValues.begin());
}

bool CPeriodicityTest::initializeTestStatistics(STestStats &stats) const
{
    CBasicStatistics::CMinMax<double> range;
    double populated{0.0};
    double count{0.0};
    for (std::size_t i = 0u; i < m_BucketValues.size(); ++i)
    {
        double ni{CBasicStatistics::count(m_BucketValues[i])};
        count += ni;
        if (ni > 0.0)
        {
            populated += 1.0;
            range.add(static_cast<double>(i));
        }
    }

    if (populated == 0.0)
    {
        return false;
    }

    LOG_TRACE("populated = " << 100.0 * populated << "%");

    stats.s_Range = range.max() - range.min();
    stats.s_B = populated;
    stats.s_M = count / stats.s_B;
    LOG_TRACE("range = " << stats.s_Range
              << ", populatedBuckets = " << stats.s_B
              << ", valuesPerBucket = " << stats.s_M);

    return true;
}

bool CPeriodicityTest::nullHypothesis(STestStats &stats) const
{
    TMeanVarAccumulatorVec trend(1);
    TTimeTimePr2Vec window{{0, this->windowLength()}};
    periodicTrend(m_BucketValues, window, m_BucketLength, trend);
    double mean{CBasicStatistics::mean(trend[0])};
    double v0{CBasicStatistics::variance(trend[0])};
    LOG_TRACE("mean = " << mean);
    LOG_TRACE("variance = " << v0);
    if (std::sqrt(v0) <= MINIMUM_COEFFICIENT_OF_VARIATION * std::fabs(mean))
    {
        return false;
    }
    stats.s_DF0 = stats.s_B - 1.0;
    stats.s_V0  = varianceAtPercentile(v0, stats.s_DF0, 50.0 + CONFIDENCE_INTERVAL / 2.0);
    stats.s_T0.assign(1, TDoubleVec{mean});
    stats.s_Partition = window;
    return true;
}

bool CPeriodicityTest::testPeriod(const TTimeTimePr2Vec &windows,
                                  core_t::TTime period_,
                                  STestStats &stats) const
{
    // We use two tests to check for the period:
    //   1) That it explains both a non-negligible absolute and statistically
    //      significant amount of variance and the cyclic autocorrelation at
    //      that repeat is high enough OR
    //   2) There is a large absolute and statistically significant periodic
    //      spike or trough.

    LOG_TRACE("Testing period " << period_);

    period_ = std::min(period_, length(windows[0]));
    std::size_t period{static_cast<std::size_t>(period_ / m_BucketLength)};

    // We need to observe a minimum number of repeated values to test with
    // an acceptable false positive rate.
    double repeats{0.0};
    for (std::size_t i = 0u; i < period; ++i)
    {
        for (std::size_t j = i + period; j < m_BucketValues.size(); j += period)
        {
            if (  CBasicStatistics::count(m_BucketValues[j])
                * CBasicStatistics::count(m_BucketValues[j - period]) > 0.0)
            {
                repeats += 1.0;
                break;
            }
        }
    }
    LOG_TRACE("repeats = " << repeats);
    if (repeats < static_cast<double>(period) * ACCURATE_TEST_POPULATED_FRACTION / 3.0)
    {
        return false;
    }

    TTimeTimePr2Vec window{{0, length(windows)}};
    double M{stats.s_M};
    double scale{1.0 / M};
    double v0{stats.s_V0}, df0{stats.s_DF0};
    double vt{stats.s_Vt * v0};
    double at{stats.s_At * std::sqrt(v0 / scale)};
    LOG_TRACE("M = " << M);

    TFloatMeanAccumulatorVec values(m_BucketValues);
    this->conditionOnNullHypothesis(windows, stats, values);
    double B{static_cast<double>(
                 std::count_if(values.begin(), values.end(),
                               [](const TFloatMeanAccumulator &value)
                               { return CBasicStatistics::count(value) > 0.0; }))};

    // The variance test.

    TMeanVarAccumulatorVec trend(period);
    periodicTrend(values, window, m_BucketLength, trend);
    double b{static_cast<double>(
                 std::count_if(trend.begin(), trend.end(),
                               [](const TMeanVarAccumulator &value)
                               { return CBasicStatistics::count(value) > 0.0; }))};
    LOG_TRACE("populated = " << b);

    double df1{B - b};
    double v1{varianceAtPercentile(residualVariance<double>(trend, scale),
                                   df1, 50.0 + CONFIDENCE_INTERVAL / 2.0)};
    LOG_TRACE("  variance          = " << v1);
    LOG_TRACE("  varianceThreshold = " << vt);
    LOG_TRACE("  significance      = " << CStatisticalTests::leftTailFTest(v1 / v0, df1, df0));

    double Rt{stats.s_Rt * CTools::truncate(1.0 - 0.5 * (vt - v1) / vt, 0.9, 1.0)};
    if (v1 < vt && CStatisticalTests::leftTailFTest(v1 / v0, df1, df0) <= MAXIMUM_SIGNIFICANCE)
    {
        double R{CSignal::autocorrelation(period, values)};
        R = autocorrelationAtPercentile(R, B, 50.0 - CONFIDENCE_INTERVAL / 2.0);
        LOG_TRACE("  autocorrelation          = " << R);
        LOG_TRACE("  autocorrelationThreshold = " << Rt);
        if (R > Rt)
        {
            stats.s_V0  = v1;
            stats.s_DF0 = df1;
            return true;
        }
    }

    // The amplitude test.

    double F1{1.0};
    if (v1 > 0.0)
    {
        try
        {
            std::size_t n{static_cast<std::size_t>(
                    std::ceil(Rt * static_cast<double>(length(windows) / period_)))};
            TMeanAccumulator level;
            for (const auto &value : values)
            {
                if (CBasicStatistics::count(value) > 0.0)
                {
                    level.add(CBasicStatistics::mean(value));
                }
            }
            TMinAmplitudeVec amplitudes(period, {n, CBasicStatistics::mean(level)});
            periodicTrend(values, window, m_BucketLength, amplitudes);
            boost::math::normal normal(0.0, std::sqrt(v1));
            std::for_each(amplitudes.begin(), amplitudes.end(),
                          [&F1, &normal, at](CMinAmplitude &x)
                          {
                              if (x.amplitude() >= at)
                              {
                                  F1 = std::min(F1, x.significance(normal));
                              }
                          });
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("Unable to compute significance of amplitude: " << e.what());
        }
    }
    LOG_TRACE("  F(amplitude)       = " << F1);

    if (1.0 - std::pow(1.0 - F1, b) <= MAXIMUM_SIGNIFICANCE)
    {
        stats.s_V0  = v1;
        stats.s_DF0 = df1;
        return true;
    }
    return false;
}

bool CPeriodicityTest::testPartition(const TTimeTimePr2Vec &partition,
                                     core_t::TTime period_,
                                     double correction,
                                     STestStats &stats) const
{
    using TDoubleTimePr = std::pair<double, core_t::TTime>;
    using TDoubleTimePrVec = std::vector<TDoubleTimePr>;
    using TMinAccumulator = CBasicStatistics::COrderStatisticsStack<TDoubleTimePr, 1>;
    using TMeanVarAccumulatorBuffer = boost::circular_buffer<TMeanVarAccumulator>;

    LOG_TRACE("Testing partition " << core::CContainerPrinter::print(partition)
              << " with period " << period_);

    // Find the partition of the data such that the residual variance
    // w.r.t. the period is minimized and check if there is significant
    // evidence that it reduces the residual variance and repeats.

    std::size_t period{static_cast<std::size_t>(period_ / m_BucketLength)};
    core_t::TTime repeat{length(partition)};
    double b{static_cast<double>(period)};
    double B{stats.s_B};
    double scale{1.0 / stats.s_M};
    double v0{stats.s_V0}, df0{stats.s_DF0};
    double vt{stats.s_Vt * v0};
    LOG_TRACE("period = " << period);
    LOG_TRACE("scale = " << scale);

    TFloatMeanAccumulatorVec values(m_BucketValues);
    this->conditionOnNullHypothesis(TTimeTimePr2Vec{{0, m_WindowLength}}, stats, values);

    double df1{B - 2.0 * b};

    TTimeTimePr2Vec windows[]{calculateWindows(0, m_WindowLength, repeat, partition[0]),
                              calculateWindows(0, m_WindowLength, repeat, partition[1])};
    LOG_TRACE("windows = " << core::CContainerPrinter::print(windows));

    TTimeVec deltas[2];
    deltas[0].reserve((length(partition[0]) * m_WindowLength) / (period_ * repeat));
    deltas[1].reserve((length(partition[1]) * m_WindowLength) / (period_ * repeat));
    for (std::size_t i = 0u; i < 2; ++i)
    {
        for (const auto &window : windows[i])
        {
            core_t::TTime a_{window.first};
            core_t::TTime b_{window.second};
            for (core_t::TTime t = a_ + period_; t <= b_; t += period_)
            {
                deltas[i].push_back(t - m_BucketLength);
            }
        }
    }
    LOG_TRACE("deltas = " << core::CContainerPrinter::print(deltas));

    TMeanVarAccumulatorBuffer trends[]
        {
            TMeanVarAccumulatorBuffer(period, TMeanVarAccumulator()),
            TMeanVarAccumulatorBuffer(period, TMeanVarAccumulator())
        };
    periodicTrend(values, windows[0], m_BucketLength, trends[0]);
    periodicTrend(values, windows[1], m_BucketLength, trends[1]);

    TMeanAccumulator variances[]
        {
            residualVariance<TMeanAccumulator>(trends[0], scale),
            residualVariance<TMeanAccumulator>(trends[1], scale)
        };
    LOG_TRACE("variances = " << core::CContainerPrinter::print(variances));

    TMinAccumulator minimum;
    minimum.add({(  residualVariance(variances[0])
                  + residualVariance(variances[1])) / 2.0, 0});

    TDoubleTimePrVec candidates;
    candidates.reserve(period);
    for (core_t::TTime time = m_BucketLength;
         time < repeat;
         time += m_BucketLength)
    {
        for (std::size_t i = 0u; i < 2; ++i)
        {
            for (auto &&delta : deltas[i])
            {
                delta = (delta + m_BucketLength) % m_WindowLength;
            }
            TMeanVarAccumulator oldBucket{trends[i].front()};
            TMeanVarAccumulator newBucket;
            averageValue(values, deltas[i], m_BucketLength, newBucket);

            trends[i].pop_front();
            trends[i].push_back(newBucket);
            variances[i] -= residualVariance(oldBucket, scale);
            variances[i] += residualVariance(newBucket, scale);
        }
        double variance{(  residualVariance(variances[0])
                         + residualVariance(variances[1])) / 2.0};
        minimum.add({variance, time});
        if (variance <= 1.05 * minimum[0].first)
        {
            candidates.emplace_back(variance, time);
        }
    }

    TMinAccumulator lowest;
    for (const auto &candidate : candidates)
    {
        if (candidate.first <= 1.05 * minimum[0].first)
        {
            core_t::TTime time{candidate.second};
            std::size_t j{static_cast<std::size_t>(time / m_BucketLength)};
            lowest.add({std::fabs(CBasicStatistics::mean(values[j])), time});
        }
    }

    double v1{correction * minimum[0].first};
    LOG_TRACE("  variance          = " << v1);
    LOG_TRACE("  varianceThreshold = " << vt);
    LOG_TRACE("  significance      = " << CStatisticalTests::leftTailFTest(v1 / v0, df1, df0));

    if (v1 <= vt && CStatisticalTests::leftTailFTest(v1 / v0, df1, df0) <= MAXIMUM_SIGNIFICANCE)
    {
        double R{-1.0};
        double Rt{stats.s_Rt * CTools::truncate(1.0 - 0.5 * (vt - v1) / vt, 0.9, 1.0)};

        core_t::TTime startOfPartition{lowest[0].second};
        windows[0] = calculateWindows(startOfPartition, m_WindowLength, repeat, partition[0]);
        windows[1] = calculateWindows(startOfPartition, m_WindowLength, repeat, partition[1]);
        for (const auto &windows_ : windows)
        {
            TFloatMeanAccumulatorVec partitionValues;
            project(values, windows_, m_BucketLength, partitionValues);
            std::size_t windowLength(length(windows_[0]) / m_BucketLength);
            double BW{std::accumulate(partitionValues.begin(), partitionValues.end(), 0.0,
                                      [](double n, const TFloatMeanAccumulator &value)
                                      { return n + (CBasicStatistics::count(value) > 0.0 ? 1.0 : 0.0); })};
            R = std::max(R, autocorrelationAtPercentile(CSignal::autocorrelation(
                                windowLength + period, partitionValues),
                                BW, 50.0 - CONFIDENCE_INTERVAL / 2.0));
            LOG_TRACE("  autocorrelation          = " << R);
            LOG_TRACE("  autocorrelationThreshold = " << Rt);
        }

        if (R > Rt)
        {
            stats.s_V0  = v1;
            stats.s_DF0 = df1;
            stats.s_StartOfPartition = startOfPartition;
            return true;
        }
    }
    return false;
}

void CPeriodicityTest::conditionOnNullHypothesis(const TTimeTimePr2Vec &windows,
                                                 const STestStats &stats,
                                                 TFloatMeanAccumulatorVec &values) const
{
    std::size_t n{values.size()};
    for (std::size_t i = 0u; i < stats.s_Partition.size(); ++i)
    {
        TTimeTimePr2Vec windows_(calculateWindows(stats.s_StartOfPartition,
                                                  m_WindowLength,
                                                  length(stats.s_Partition),
                                                  stats.s_Partition[i]));
        TSizeSizePr2Vec indexWindows;
        calculateIndexWindows(windows_, m_BucketLength, indexWindows);

        std::size_t period{stats.s_T0[i].size()};
        for (const auto &window : indexWindows)
        {
            std::size_t a{window.first};
            std::size_t b{window.second};
            for (std::size_t j = a; j < b; ++j)
            {
                CBasicStatistics::moment<0>(values[j % n]) -= stats.s_T0[i][(j - a) % period];
            }
        }
    }
    if (length(windows) < m_WindowLength)
    {
        LOG_TRACE("Projecting onto " << core::CContainerPrinter::print(windows));
        TFloatMeanAccumulatorVec projection;
        project(values, windows, m_BucketLength, projection);
        values = std::move(projection);
        LOG_TRACE("# values = " << values.size());
    }
}

void CPeriodicityTest::periodicBucketing(core_t::TTime period_,
                                         const TTimeTimePr2Vec &windows,
                                         TTimeTimePrMeanVarAccumulatorPrVec &trend) const
{
    trend.clear();

    if (windows.empty())
    {
        return;
    }

    period_ = std::min(period_, length(windows[0]));
    std::size_t period(period_ / m_BucketLength);

    initializeBuckets(period, windows, trend);
    TMeanAccumulatorVec varianceScales(trend.size());
    std::size_t n{m_BucketValues.size()};
    for (std::size_t i = 0u, j = 0u; i < windows.size(); ++i)
    {
        std::size_t a{static_cast<std::size_t>(windows[i].first  / m_BucketLength)};
        std::size_t b{static_cast<std::size_t>(windows[i].second / m_BucketLength)};
        for (std::size_t k = a; k < b; ++j, ++k)
        {
            const TFloatMeanAccumulator &bucket{m_BucketValues[k % n]};
            double count{CBasicStatistics::count(bucket)};
            double mean{CBasicStatistics::mean(bucket)};
            if (count > 0.0)
            {
                std::size_t pj{j % period};
                trend[pj].second.add(mean, count);
                varianceScales[pj].add(1.0 / count);
            }
        }
    }
    for (std::size_t i = 0u; i < trend.size(); ++i)
    {
        CBasicStatistics::moment<1>(trend[i].second) /= CBasicStatistics::mean(varianceScales[i]);
    }
}

void CPeriodicityTest::periodicBucketing(TTime2Vec periods_,
                                         const TTimeTimePr2Vec &windows,
                                         TTimeTimePrMeanVarAccumulatorPrVec &shortTrend,
                                         TTimeTimePrMeanVarAccumulatorPrVec &longTrend) const
{
    // For periods P1 and P2 then, given the window is a whole number
    // of P2, we have that
    //   x(i) = d(i mod P2, j) * m2'(j) + d(i mod P1, j) * m1'(j)
    //
    // where d(.) denotes the Kronecker delta, and m1' and m2' are the
    // adjusted periodic baselines, respectively. This gives an over
    // determined matrix equation which can be solved using the standard
    // least squares approach, i.e. using the Moore-Penrose pseudo-inverse.
    // There is one complication which is that the matrix is singular.
    // This is because there is a degeneracy among possible solutions.
    // Specifically, if we add c(j) to m1'(j) and subtract c(j) from
    // m(j + k * D) we end up with the same total baseline. One strategy
    // is to add eps * I and let eps -> 0 which gives a well defined linear
    // combination of {x(i)} to compute the mean, i.e. for the j'th bucket
    // in period P2
    //   m1'(j) = (N/N1) / (N/N1 + N/N2) * m1(j)
    //   m2'(j) = m2(j) - (N/N1) / (N/N1 + N/N2) * m1(j mod n1)
    //
    // Since we have lower resolution to model m2' we prefer to subsequently
    // adjust m1' to make m2' as smooth as possible.

    shortTrend.clear();
    longTrend.clear();

    if (windows.empty())
    {
        return;
    }

    core_t::TTime window{length(windows)};

    core_t::TTime w0{length(windows[0])};
    periods_[0] = std::min(periods_[0], w0);
    periods_[1] = std::min(periods_[1], w0);

    std::size_t periods[2];
    periods[0] = static_cast<std::size_t>(periods_[0] / m_BucketLength);
    periods[1] = static_cast<std::size_t>(periods_[1] / m_BucketLength);

    std::size_t length = m_BucketValues.size();
    double S{static_cast<double>(window / periods_[0])};
    double L{static_cast<double>(window / periods_[1])};
    double scale{S / (S + L)};

    TMeanAccumulatorVec trends[]{TMeanAccumulatorVec(periods[0]),
                                 TMeanAccumulatorVec(periods[1])};
    periodicTrend(m_BucketValues, windows, m_BucketLength, trends[0]);
    periodicTrend(m_BucketValues, windows, m_BucketLength, trends[1]);

    for (auto &&bucket : trends[0])
    {
        CBasicStatistics::moment<0>(bucket) *= scale;
    }
    for (std::size_t i = 0u; i < trends[1].size(); /**/)
    {
        for (std::size_t j = 0u; i < trends[1].size() && j < trends[0].size(); ++i, ++j)
        {
            TMeanAccumulator &bucket{trends[1][i]};
            if (CBasicStatistics::count(bucket) > 0.0)
            {
                CBasicStatistics::moment<0>(bucket) -= CBasicStatistics::mean(trends[0][j]);
            }
        }
    }

    TMeanAccumulatorVec shifts(periods[0]);
    for (std::size_t i = 0u; i < trends[1].size(); /**/)
    {
        for (std::size_t j = 0u; i < trends[1].size() && j < shifts.size(); ++i, ++j)
        {
            const TMeanAccumulator &bucket{trends[1][i]};
            if (CBasicStatistics::count(bucket) > 0.0)
            {
                shifts[j].add(CBasicStatistics::mean(bucket));
            }
        }
    }
    for (std::size_t i = 0u; i < trends[0].size(); ++i)
    {
        double shift{CBasicStatistics::mean(shifts[i])};
        if (shift != 0.0)
        {
            CBasicStatistics::moment<0>(trends[0][i]) += shift;
            for (std::size_t j = 0u; j < trends[1].size(); j += trends[0].size())
            {
                TMeanAccumulator &bucket{trends[1][i + j]};
                if (CBasicStatistics::count(bucket) > 0.0)
                {
                    CBasicStatistics::moment<0>(bucket) -= shift;
                }
            }
        }
    }

    this->initializeBuckets(periods[0], windows, shortTrend);
    this->initializeBuckets(periods[1], windows, longTrend);
    TMeanAccumulatorVec varianceScales(shortTrend.size());
    for (std::size_t i = 0u, j = 0u, k = 0u; i < windows.size(); ++i)
    {
        std::size_t a(windows[i].first  / m_BucketLength);
        std::size_t b(windows[i].second / m_BucketLength);
        for (std::size_t l = a; l < b; ++j, ++k, ++l)
        {
            const TFloatMeanAccumulator &bucket{m_BucketValues[l % length]};
            double count{CBasicStatistics::count(bucket)};
            double mean{CBasicStatistics::mean(bucket)};
            if (count > 0.0)
            {
                std::size_t pj{j % periods[0]};
                std::size_t pk{k % periods[1]};
                shortTrend[pj].second.add(mean - CBasicStatistics::mean(trends[1][pk]), count);
                varianceScales[pj].add(1.0 / count);
            }
        }
    }
    for (std::size_t i = 0u; i < shortTrend.size(); ++i)
    {
        CBasicStatistics::moment<1>(shortTrend[i].second) /= CBasicStatistics::mean(varianceScales[i]);
    }
    for (std::size_t i = 0u; i < trends[1].size(); ++i)
    {
        longTrend[i].second.add(CBasicStatistics::mean(trends[1][i]));
    }
}

void CPeriodicityTest::initializeBuckets(std::size_t period,
                                         const TTimeTimePr2Vec &windows,
                                         TTimeTimePrMeanVarAccumulatorPrVec &trend) const
{
    trend.resize(period);
    core_t::TTime bucket = windows[0].first;
    for (auto &&value : trend)
    {
        value.first = {bucket, bucket + m_BucketLength};
        bucket += m_BucketLength;
    }
}

const double CPeriodicityTest::ACCURATE_TEST_POPULATED_FRACTION{0.9};
const double CPeriodicityTest::MINIMUM_COEFFICIENT_OF_VARIATION{1e-4};

CPeriodicityTest::STestStats::STestStats(double vt, double at, double Rt) :
        s_Vt(vt), s_At(at), s_Rt(Rt),
        s_Range(0.0), s_B(0.0), s_M(0.0), s_V0(0.0), s_DF0(0.0),
        s_StartOfPartition(0)
{}

//////// CDiurnalPeriodicityTest ////////

CDiurnalPeriodicityTest::CDiurnalPeriodicityTest(double decayRate) :
        CPeriodicityTest(decayRate)
{
    std::fill_n(m_VarianceCorrections, 2, 1.0);
}

bool CDiurnalPeriodicityTest::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser)
{
    bool result{this->CPeriodicityTest::acceptRestoreTraverser(traverser)};
    if (result)
    {
        std::ptrdiff_t i{  std::find(boost::begin(DIURNAL_PERMITTED_BUCKET_LENGTHS),
                                     boost::end(DIURNAL_PERMITTED_BUCKET_LENGTHS),
                                     this->bucketLength())
                         - boost::begin(DIURNAL_PERMITTED_BUCKET_LENGTHS)};
        m_VarianceCorrections[0] = DIURNAL_VARIANCE_CORRECTIONS[0][i];
        m_VarianceCorrections[1] = DIURNAL_VARIANCE_CORRECTIONS[1][i];
    }
    return result;
}

void CDiurnalPeriodicityTest::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    this->CPeriodicityTest::acceptPersistInserter(inserter);
}

CDiurnalPeriodicityTest *CDiurnalPeriodicityTest::create(core_t::TTime bucketLength, double decayRate)
{
    std::size_t n{boost::size(DIURNAL_PERMITTED_BUCKET_LENGTHS)};
    if (bucketLength > DIURNAL_PERMITTED_BUCKET_LENGTHS[n - 1])
    {
        return 0;
    }

    std::ptrdiff_t index{  std::lower_bound(DIURNAL_PERMITTED_BUCKET_LENGTHS,
                                            DIURNAL_PERMITTED_BUCKET_LENGTHS + n,
                                            bucketLength)
                         - DIURNAL_PERMITTED_BUCKET_LENGTHS};
    bucketLength = DIURNAL_PERMITTED_BUCKET_LENGTHS[index];
    double corrections[]{DIURNAL_VARIANCE_CORRECTIONS[0][index],
                         DIURNAL_VARIANCE_CORRECTIONS[1][index]};

    CDiurnalPeriodicityTest *result{new CDiurnalPeriodicityTest(decayRate)};
    core_t::TTime window{std::min(2 * WEEK * (bucketLength / HOUR), 8 * WEEK)};
    if (!result->initialize(bucketLength, window, corrections))
    {
        delete result;
        result = 0;
    }
    return result;
}

CPeriodicityTestResult CDiurnalPeriodicityTest::test(void) const
{
    // We perform a series of tests of nested hypotheses about
    // the periodic components and weekday/end patterns. To test
    // for periodic components we compare the residual variance
    // with and without trend. This must be reduced in significant
    // absolute sense to make it worthwhile modelling and in a
    // statistical sense. We use an F-test for this purpose. Note
    // that since the buckets contain the mean of multiple samples
    // we expect them to tend to Gaussian over time. We also test
    // the amplitude. Again this must be significant in both an
    // absolute and statistical sense. We again assume the bucket
    // values are Gaussian for the purpose of the statistical test.
    // Each time we accept a simpler hypothesis about the data we
    // test the nested hypothesis w.r.t. this. This entails removing
    // any periodic component we've already found from the data.

    if (this->bucketLength() > DAY)
    {
        return CPeriodicityTestResult();
    }

    STestStats stats(MAXIMUM_PERIOD_VARIANCE,
                     MINIMUM_PERIOD_AMPLITUDE,
                     MINIMUM_AUTOCORRELATION);
    if (!this->initializeTestStatistics(stats)
        || stats.s_Range < 2.9 * static_cast<double>(DAY / this->bucketLength())
        || !this->nullHypothesis(stats))
    {
        return CPeriodicityTestResult();
    }

    TTimeTimePr2Vec window{{0, this->windowLength()}};
    if (this->testDaily(stats))
    {
        if (this->testWeekend(true, stats))
        {
            return this->testWeeklyGivenDailyAndWeekend(stats);
        }
        if (this->testWeekly(window, stats))
        {
            return diurnalResult({E_Day, E_Week});
        }
        return diurnalResult({E_Day});
    }

    if (this->testWeekend(false, stats))
    {
        if (this->testWeekend(true, stats))
        {
            return this->testWeeklyGivenDailyAndWeekend(stats);
        }
        if (this->testWeekly(window, stats))
        {
            return diurnalResult({E_Week});
        }
        return diurnalResult({E_WeekdayWeek, E_WeekendWeek}, stats.s_StartOfPartition);
    }

    if (this->testWeekend(true, stats))
    {
        return this->testWeeklyGivenDailyAndWeekend(stats);
    }

    if (this->testWeekly(window, stats))
    {
        return diurnalResult({E_Week});
    }

    return CPeriodicityTestResult();
}

CDiurnalPeriodicityTest::TTime2Vec CDiurnalPeriodicityTest::periods(void) const
{
    return TTime2Vec{DAY, WEEK};
}

CSeasonalTime *CDiurnalPeriodicityTest::seasonalTime(const TComponent &component) const
{
    return new CDiurnalTime(component.s_StartOfPartition,
                            component.s_Window.first,
                            component.s_Window.second,
                            component.s_Period);
}

void CDiurnalPeriodicityTest::trends(const CPeriodicityTestResult &required,
                                     TTimeTimePrMeanVarAccumulatorPrVecVec &result) const
{
    auto &components = required.components();

    result.resize(components.size());

    std::size_t table[6];
    std::fill_n(boost::begin(table), 6, result.size());
    for (std::size_t i = 0u; i < components.size(); ++i)
    {
        table[components[i].s_Id] = i;
    }

    for (std::size_t i = 0u; i < 6; i += 2)
    {
        if (table[i] < result.size() && table[i+1] < result.size())
        {
            std::size_t day{table[i]};
            std::size_t week{table[i+1]};
            core_t::TTime startOfWeek{components[day].s_StartOfPartition};
            const TTimeTimePr &window{components[day].s_Window};
            this->CPeriodicityTest::periodicBucketing(this->periods(),
                                                      calculateWindows(startOfWeek,
                                                                       this->windowLength(),
                                                                       WEEK, window),
                                                      result[day], result[week]);

        }
        else if (table[i] < result.size() || table[i+1] < result.size())
        {
            std::size_t period{std::min(table[i], table[i+1])};
            core_t::TTime startOfWeek{components[period].s_StartOfPartition};
            const TTimeTimePr &window{components[period].s_Window};
            this->CPeriodicityTest::periodicBucketing(components[period].s_Period,
                                                      calculateWindows(startOfWeek,
                                                                       this->windowLength(),
                                                                       WEEK, window),
                                                      result[period]);
        }
    }
}

std::size_t CDiurnalPeriodicityTest::staticSize(void) const
{
    return sizeof(*this);
}

uint64_t CDiurnalPeriodicityTest::checksum(uint64_t seed) const
{
    return this->CPeriodicityTest::checksum(seed);
}

std::string CDiurnalPeriodicityTest::print(const CPeriodicityTestResult &result) const
{
    std::string desc = "{";
    for (const auto &component : result.components())
    {
        std::string partition{DIURNAL_WINDOW_NAMES[component.s_Id / 2]};
        std::string period{DIURNAL_PERIOD_NAMES[component.s_Id % 2]};
        desc +=  " '" + partition + (partition.empty() ? "" : " ") + period + "'";
    }
    desc += " }";
    return desc;
}

bool CDiurnalPeriodicityTest::initialize(core_t::TTime bucketLength,
                                         core_t::TTime window,
                                         const double (&corrections)[2])
{
    // The following conditions need to hold:
    //   - The window needs to be at least two weeks,
    //   - The window needs to be a whole number of weeks,
    //   - The periods needs to be multiples of the bucket length.
    if (   window < 2 * WEEK
        || window % WEEK != 0
        || DAY % bucketLength != 0
        || WEEK % bucketLength != 0)
    {
        return false;
    }

    TFloatMeanAccumulatorVec initial;
    this->CPeriodicityTest::initialize(bucketLength, window, initial);
    std::copy(corrections, corrections + 2, m_VarianceCorrections);

    return true;
}

CPeriodicityTestResult
CDiurnalPeriodicityTest::testWeeklyGivenDailyAndWeekend(STestStats &stats) const
{
    core_t::TTime startOfWeek{stats.s_StartOfPartition};
    TTimeTimePr2Vec window{{0, this->windowLength()}};
    if (this->testWeekly(window, stats))
    {
        return diurnalResult({E_WeekendDay, E_WeekendWeek, E_WeekdayDay, E_WeekdayWeek}, startOfWeek);
    }
    TTimeTimePr2Vec weekday(
            calculateWindows(startOfWeek, this->windowLength(), WEEK, {WEEKEND, WEEK}));
    if (this->testWeekly(weekday, stats))
    {
        return diurnalResult({E_WeekendDay, E_WeekdayDay, E_WeekdayWeek}, startOfWeek);
    }
    TTimeTimePr2Vec weekend(
            calculateWindows(startOfWeek, this->windowLength(), WEEK, {0, WEEKEND}));
    if (this->testWeekly(weekend, stats))
    {
        return diurnalResult({E_WeekendDay, E_WeekendWeek, E_WeekdayDay}, startOfWeek);

    }
    return diurnalResult({E_WeekendDay, E_WeekdayDay}, startOfWeek);
}

bool CDiurnalPeriodicityTest::testDaily(STestStats &stats) const
{
    std::size_t period(DAY / this->bucketLength());
    TTimeTimePr2Vec window{{0, this->windowLength()}};
    if (   this->bucketLength() <= DAY / 4
        && stats.s_Range >= 2.9 * static_cast<double>(period)
        && this->testPeriod(window, DAY, stats))
    {
        TMeanAccumulatorVec trend(period);
        periodicTrend(this->bucketValues(), window, this->bucketLength(), trend);
        stats.s_T0 = TDoubleVec2Vec(1);
        stats.s_T0[0].reserve(period);
        std::for_each(trend.begin(), trend.end(),
                      [&stats](const TMeanAccumulator &value)
                      { stats.s_T0[0].push_back(CBasicStatistics::mean(value)); });
        return true;
    }
    return false;
}

bool CDiurnalPeriodicityTest::testWeekend(bool daily, STestStats &stats) const
{
    core_t::TTime period_{daily ? DAY : this->bucketLength()};
    std::size_t period(period_ / this->bucketLength());
    TTimeTimePr2Vec partition{{0, WEEKEND}, {WEEKEND, WEEK}};
    if (  (!daily || this->bucketLength() <= DAY / 4)
        && this->seenSufficientData()
        && this->testPartition(
                partition, period_, m_VarianceCorrections[daily ? 0 : 1], stats))
    {

        stats.s_Partition = partition;
        stats.s_T0 = TDoubleVec2Vec(2);
        for (std::size_t i = 0u; i < partition.size(); ++i)
        {
            TMeanAccumulatorVec trend(period);
            TTimeTimePr2Vec windows(calculateWindows(stats.s_StartOfPartition,
                                                     this->windowLength(),
                                                     WEEK, partition[i]));
            periodicTrend(this->bucketValues(), windows, this->bucketLength(), trend);
            stats.s_T0[i].reserve(period);
            std::for_each(trend.begin(), trend.end(),
                          [&stats, i](const TMeanAccumulator &value)
                          { stats.s_T0[i].push_back(CBasicStatistics::mean(value)); });
        }
        return true;
    }
    return false;
}

bool CDiurnalPeriodicityTest::testWeekly(const TTimeTimePr2Vec &window,
                                         STestStats &stats) const
{
    return   stats.s_Range >=  ACCURATE_TEST_POPULATED_FRACTION
                             * static_cast<double>(2 * WEEK / this->bucketLength())
          && this->testPeriod(window, WEEK, stats);
}

const double CDiurnalPeriodicityTest::MAXIMUM_PARTITION_VARIANCE{0.5};
const double CDiurnalPeriodicityTest::MAXIMUM_PERIOD_VARIANCE{0.7};
const double CDiurnalPeriodicityTest::MINIMUM_PERIOD_AMPLITUDE{1.0};
const double CDiurnalPeriodicityTest::MINIMUM_AUTOCORRELATION{0.5};

//////// CGeneralPeriodicityTest ////////

bool CGeneralPeriodicityTest::initialize(core_t::TTime bucketLength,
                                         core_t::TTime window,
                                         core_t::TTime period,
                                         const TFloatMeanAccumulatorVec &initial)
{
    // The following conditions need to hold:
    //   - The window needs to be at least two periods,
    //   - The window needs to be a whole number of periods,
    //   - The period needs to be multiples of the bucket length.
    if (   window < 2 * period
        || window % period != 0
        || period % bucketLength != 0)
    {
        return false;
    }

    this->CPeriodicityTest::initialize(bucketLength, window, initial);
    m_Period = period;
    return true;
}

CPeriodicityTestResult CGeneralPeriodicityTest::test(void) const
{
    std::size_t period = static_cast<std::size_t>(m_Period / this->bucketLength());

    STestStats stats(MAXIMUM_PERIOD_VARIANCE,
                     MINIMUM_PERIOD_AMPLITUDE,
                     MINIMUM_AUTOCORRELATION);
    if (!this->initializeTestStatistics(stats)
        || stats.s_B < 2.9 * static_cast<double>(period)
        || !this->nullHypothesis(stats))
    {
        return CPeriodicityTestResult();
    }

    LOG_TRACE("Testing " << m_Period);

    CPeriodicityTestResult result;
    TTimeTimePr2Vec window{{0, this->windowLength()}};
    if (this->testPeriod(window, m_Period, stats))
    {
        result.add(0, 0, m_Period, {0, m_Period});
    }
    return result;
}

CGeneralPeriodicityTest::TTime2Vec CGeneralPeriodicityTest::periods(void) const
{
    return TTime2Vec{m_Period};
}

CSeasonalTime *CGeneralPeriodicityTest::seasonalTime(const TComponent &/*component*/) const
{
    return new CGeneralPeriodTime(m_Period);
}

void CGeneralPeriodicityTest::trends(const CPeriodicityTestResult &required,
                                     TTimeTimePrMeanVarAccumulatorPrVecVec &result) const
{
    if (required.periodic())
    {
        result.resize(1);
        TTimeTimePr2Vec window{{0, this->windowLength()}};
        this->CPeriodicityTest::periodicBucketing(m_Period, window, result[0]);
    }
}

std::size_t CGeneralPeriodicityTest::staticSize(void) const
{
    return sizeof(*this);
}

uint64_t CGeneralPeriodicityTest::checksum(uint64_t seed) const
{
    seed = this->CPeriodicityTest::checksum(seed);
    return CChecksum::calculate(seed, m_Period);
}

std::string CGeneralPeriodicityTest::print(const CPeriodicityTestResult &result) const
{
    return result.periodic() ?
           "{ " + core::CStringUtils::typeToString(result.components()[0].s_Period) + " }" : "{ }";
}

const TFloatMeanAccumulatorVec CGeneralPeriodicityTest::NO_BUCKET_VALUES{};
const double CGeneralPeriodicityTest::MAXIMUM_PERIOD_VARIANCE{0.5};
const double CGeneralPeriodicityTest::MINIMUM_PERIOD_AMPLITUDE{2.0};
const double CGeneralPeriodicityTest::MINIMUM_AUTOCORRELATION{0.7};

//////// CScanningPeriodicityTest ////////

CScanningPeriodicityTest::CScanningPeriodicityTest(TTimeCRng bucketLengths,
                                                   std::size_t size,
                                                   double decayRate) :
        m_DecayRate(decayRate),
        m_BucketLengths(bucketLengths),
        m_BucketLengthIndex(0),
        m_StartTime(boost::numeric::bounds<core_t::TTime>::lowest()),
        m_BucketValues(size % 2 == 0 ? size : size + 1)
{}

bool CScanningPeriodicityTest::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser)
{
    m_BucketValues.clear();
    do
    {
        const std::string &name = traverser.name();
        RESTORE_BUILT_IN(BUCKET_LENGTH_INDEX_TAG, m_BucketLengthIndex)
        RESTORE_BUILT_IN(START_TIME_TAG, m_StartTime)
        RESTORE(BUCKET_VALUE_TAG, core::CPersistUtils::restore(BUCKET_VALUE_TAG, m_BucketValues, traverser));
    }
    while (traverser.next());
    return true;
}

void CScanningPeriodicityTest::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertValue(BUCKET_LENGTH_INDEX_TAG, m_BucketLengthIndex);
    inserter.insertValue(START_TIME_TAG, m_StartTime);
    core::CPersistUtils::persist(BUCKET_VALUE_TAG, m_BucketValues, inserter);
}

void CScanningPeriodicityTest::initialize(core_t::TTime time)
{
    m_StartTime = time;
}

void CScanningPeriodicityTest::propagateForwardsByTime(double time)
{
    if (!CMathsFuncs::isFinite(time) || time < 0.0)
    {
        LOG_ERROR("Bad propagation time " << time);
    }
    double factor = std::exp(-m_DecayRate * time);
    for (auto &&value : m_BucketValues)
    {
        value.age(factor);
    }
}

void CScanningPeriodicityTest::add(core_t::TTime time, double value, double weight)
{
    if (time >= m_StartTime)
    {
        while (this->needToCompress(time))
        {
            m_BucketLengthIndex = (m_BucketLengthIndex + 1) % m_BucketLengths.size();
            auto end = m_BucketValues.begin();

            if (m_BucketLengthIndex == 0)
            {
                m_StartTime = CIntegerTools::floor(time, m_BucketLengths[0]);
            }
            else
            {
                std::size_t compression =  m_BucketLengths[m_BucketLengthIndex]
                                         / m_BucketLengths[m_BucketLengthIndex - 1];
                for (std::size_t i = 0u; i < m_BucketValues.size(); i += compression, ++end)
                {
                    std::swap(*end, m_BucketValues[i]);
                    for (std::size_t j = 1u; j < compression && i + j < m_BucketValues.size(); ++j)
                    {
                        *end += m_BucketValues[i + j];
                    }
                }
            }
            std::fill(end, m_BucketValues.end(), TFloatMeanAccumulator());
        }

        m_BucketValues[(time - m_StartTime) / m_BucketLengths[m_BucketLengthIndex]].add(value, weight);
    }
}

bool CScanningPeriodicityTest::needToCompress(core_t::TTime time) const
{
    return time >= m_StartTime +   static_cast<core_t::TTime>(m_BucketValues.size())
                                 * m_BucketLengths[m_BucketLengthIndex];
}

CScanningPeriodicityTest::TPeriodicityResultPr CScanningPeriodicityTest::test(void) const
{
    using TSizeVec = std::vector<std::size_t>;
    using TDoubleSizePr = std::pair<double, std::size_t>;
    using TMaxAccumulator = CBasicStatistics::COrderStatisticsHeap<TDoubleSizePr, std::greater<TDoubleSizePr>>;

    // Compute the serial autocorrelations padding to the maximum offset
    // to avoid windowing effects.

    core_t::TTime bucketLength = m_BucketLengths[m_BucketLengthIndex];

    LOG_TRACE("Testing with bucket length " << bucketLength);

    std::size_t n = m_BucketValues.size();
    std::size_t pad = n / 3;

    TFloatMeanAccumulatorVec values(m_BucketValues);

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
    for (std::size_t p = 4u; p < correlations.size(); ++p)
    {
        double correlation = this->meanForPeriodicOffsets(correlations, p);
        LOG_TRACE("correlation(" << p << ") = " << correlation);
        candidates.add({correlation, p});
    }

    TSizeVec candidatePeriods(15);
    std::transform(candidates.begin(), candidates.end(),
                   candidatePeriods.begin(),
                   [](const TDoubleSizePr &candidate_) { return candidate_.second; });
    candidates.clear();
    for (auto period : candidatePeriods)
    {
        this->resize(n - n % period, values);
        candidates.add({CSignal::autocorrelation(period, values), period});
    }
    candidates.sort();
    LOG_TRACE("candidate periods = " << candidates.print());

    std::size_t period_ = candidates[0].second;
    double cutoff = 0.9 * candidates[0].first;
    for (auto i = candidates.begin() + 1; i != candidates.end() && i->first > cutoff; ++i)
    {
        if (i->second < period_ && candidates[0].second % i->second == 0)
        {
            period_ = i->second;
        }
    }

    // Configure the full periodicity test.

    std::size_t window_ = n - n % period_;
    core_t::TTime window = static_cast<core_t::TTime>(window_) * bucketLength;
    core_t::TTime period = static_cast<core_t::TTime>(period_) * bucketLength;
    this->resize(window_, values);

    // We define times relative to an integer multiple of the period so need
    // to shift values to account for the offset of m_StartTime relative to
    // this pattern.
    std::size_t offset = static_cast<std::size_t>(
            (m_StartTime - CIntegerTools::floor(m_StartTime, period)) / bucketLength);
    cyclicShift(offset, values);

    LOG_TRACE("bucket length = " << bucketLength
              << ", window = " << window
              << ", periods to test = " << period
              << ", # values = " << values.size());

    CGeneralPeriodicityTest test;
    test.initialize(bucketLength, window, period, values);

    return {test, test.test()};
}

void CScanningPeriodicityTest::skipTime(core_t::TTime skipInterval)
{
    m_StartTime += skipInterval;
}

uint64_t CScanningPeriodicityTest::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, m_BucketLengthIndex);
    seed = CChecksum::calculate(seed, m_StartTime);
    return CChecksum::calculate(seed, m_BucketValues);
}

void CScanningPeriodicityTest::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    mem->setName("CScanningPeriodicityTest");
    core::CMemoryDebug::dynamicSize("m_BucketValues", m_BucketValues, mem);
}

std::size_t CScanningPeriodicityTest::memoryUsage(void) const
{
    return core::CMemory::dynamicSize(m_BucketValues);
}

void CScanningPeriodicityTest::resize(std::size_t size, TFloatMeanAccumulatorVec &values) const
{
    std::size_t n = values.size();
    values.resize(size);
    for (std::size_t i = n; i < size; ++i)
    {
        values[i] = m_BucketValues[i];
    }
}

double CScanningPeriodicityTest::meanForPeriodicOffsets(const TDoubleVec &correlations,
                                                                 std::size_t period) const
{
    TMeanAccumulator result;
    for (std::size_t offset = period; offset < correlations.size(); offset += period)
    {
        result.add(this->correctForPad(correlations[offset - 1], offset));
    }
    return CBasicStatistics::mean(result);
}

double CScanningPeriodicityTest::correctForPad(double correlation, std::size_t offset) const
{
    return correlation * static_cast<double>(m_BucketValues.size())
                       / static_cast<double>(m_BucketValues.size() - offset);
}

//////// CCalendarCyclicTest ////////

CCalendarCyclicTest::CCalendarCyclicTest(double decayRate) :
        m_DecayRate(decayRate),
        m_Bucket(0),
        m_ErrorQuantiles(CQuantileSketch::E_Linear, 20),
        m_ErrorCounts(WINDOW / BUCKET)
{
    static const SSetTimeZone timezone("GMT");
    m_ErrorSums.reserve(WINDOW / BUCKET / 10);
}

bool CCalendarCyclicTest::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser)
{
    do
    {
        const std::string &name = traverser.name();
        RESTORE_BUILT_IN(BUCKET_TAG, m_Bucket)
        RESTORE(ERROR_QUANTILES_TAG, traverser.traverseSubLevel(
                                         boost::bind(&CQuantileSketch::acceptRestoreTraverser, &m_ErrorQuantiles, _1)))
        RESTORE(ERROR_COUNTS_TAG, core::CPersistUtils::restore(ERROR_COUNTS_TAG, m_ErrorCounts, traverser))
        RESTORE(ERROR_SUMS_TAG, core::CPersistUtils::fromString(traverser.value(), m_ErrorSums))
    }
    while (traverser.next());
    return true;
}

void CCalendarCyclicTest::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertValue(BUCKET_TAG, m_Bucket);
    inserter.insertLevel(ERROR_QUANTILES_TAG,
                         boost::bind(&CQuantileSketch::acceptPersistInserter, &m_ErrorQuantiles, _1));
    core::CPersistUtils::persist(ERROR_COUNTS_TAG, m_ErrorCounts, inserter);
    inserter.insertValue(ERROR_SUMS_TAG, core::CPersistUtils::toString(m_ErrorSums));
}

void CCalendarCyclicTest::propagateForwardsByTime(double time)
{
    if (!CMathsFuncs::isFinite(time) || time < 0.0)
    {
        LOG_ERROR("Bad propagation time " << time);
        return;
    }
    m_ErrorQuantiles.age(std::exp(-m_DecayRate * time));
}

void CCalendarCyclicTest::add(core_t::TTime time, double error, double weight)
{
    error = std::fabs(error);

    m_ErrorQuantiles.add(error, weight);

    if (m_ErrorQuantiles.count() > 100.0)
    {
        core_t::TTime bucket = CIntegerTools::floor(time, BUCKET);
        if (m_ErrorCounts.empty())
        {
            m_ErrorCounts.push_back(0);
        }
        else
        {
            for (core_t::TTime i = m_Bucket; i < bucket; i += BUCKET)
            {
                m_ErrorCounts.push_back(0);
            }
        }

        uint32_t &count = m_ErrorCounts.back();
        count += (count % COUNT_BITS < COUNT_BITS - 1) ? 1 : 0;

        double high;
        m_ErrorQuantiles.quantile(LARGE_ERROR_PERCENTILE, high);

        m_ErrorSums.erase(m_ErrorSums.begin(),
                          std::find_if(m_ErrorSums.begin(), m_ErrorSums.end(),
                                       [bucket](const TTimeFloatPr &error_)
                                       { return error_.first + WINDOW > bucket; }));
        if (error >= high)
        {
            count += (count < 0x100000000 - COUNT_BITS) ? COUNT_BITS : 0;
            m_ErrorSums[bucket] += this->winsorise(error);
        }

        m_Bucket = bucket;
    }
}

CCalendarCyclicTest::TOptionalFeature CCalendarCyclicTest::test(void) const
{
    // The statistics we need in order to be able to test for calendar
    // features.
    struct SStats
    {
        SStats(void) :
            s_Offset(0), s_Repeats(0), s_Sum(0.0), s_Count(0.0), s_Significance(0.0)
        {}
        core_t::TTime s_Offset;
        unsigned int s_Repeats;
        double s_Sum;
        double s_Count;
        double s_Significance;
    };
    using TFeatureStatsFMap = boost::container::flat_map<CCalendarFeature, SStats>;
    using TDoubleTimeCalendarFeatureTr = core::CTriple<double, core_t::TTime, CCalendarFeature>;
    using TMaxAccumulator = CBasicStatistics::SMax<TDoubleTimeCalendarFeatureTr>::TAccumulator;

    TMaxAccumulator result;

    // Most features get the same count. The odd ones out are features
    // which happen sporadically because of the variation of the days
    // in a month and the day of week on which the first of the month
    // falls. The test therefore isn't that sensitive to the exact value
    // of this threshold.

    TFeatureStatsFMap stats;
    stats.reserve(m_ErrorSums.size());

    for (auto offset : TIMEZONE_OFFSETS)
    {
        for (const auto &error : m_ErrorSums)
        {
            std::size_t i =  m_ErrorCounts.size() - 1
                           - static_cast<std::size_t>((m_Bucket - error.first) / BUCKET);
            double n = static_cast<double>(m_ErrorCounts[i] % COUNT_BITS);
            double x = static_cast<double>(m_ErrorCounts[i] / COUNT_BITS);
            double s = this->significance(n, x);
            for (auto feature : CCalendarFeature::features(error.first + BUCKET / 2 + offset))
            {
                SStats &stat = stats[feature];
                ++stat.s_Repeats;
                stat.s_Offset = offset;
                stat.s_Sum += error.second;
                stat.s_Count += x;
                stat.s_Significance = std::max(stat.s_Significance, s);
            }
        }
    }

    double errorThreshold;
    m_ErrorQuantiles.quantile(50.0, errorThreshold);
    errorThreshold *= 2.0;

    for (const auto &stat : stats)
    {
        CCalendarFeature feature = stat.first;
        double r = static_cast<double>(stat.second.s_Repeats);
        double x = stat.second.s_Count;
        double e = stat.second.s_Sum;
        double s = stat.second.s_Significance;
        if (   stat.second.s_Repeats >= MINIMUM_REPEATS
            && e > errorThreshold * x
            && ::pow(s, r) < MAXIMUM_SIGNIFICANCE)
        {
            result.add({e, stat.second.s_Offset, feature});
        }
    }

    return result.count() > 0 ? result[0].third : TOptionalFeature();
}

uint64_t CCalendarCyclicTest::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, m_ErrorQuantiles);
    seed = CChecksum::calculate(seed, m_ErrorCounts);
    return CChecksum::calculate(seed, m_ErrorSums);
}

void CCalendarCyclicTest::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    mem->setName("CCalendarCyclicTest");
    core::CMemoryDebug::dynamicSize("m_ErrorQuantiles", m_ErrorQuantiles, mem);
    core::CMemoryDebug::dynamicSize("m_ErrorCounts", m_ErrorCounts, mem);
    core::CMemoryDebug::dynamicSize("m_ErrorSums", m_ErrorSums, mem);
}

std::size_t CCalendarCyclicTest::memoryUsage(void) const
{
    return  core::CMemory::dynamicSize(m_ErrorQuantiles)
          + core::CMemory::dynamicSize(m_ErrorCounts)
          + core::CMemory::dynamicSize(m_ErrorSums);
}

double CCalendarCyclicTest::winsorise(double error) const
{
    double high;
    m_ErrorQuantiles.quantile(99.5, high);
    return std::min(error, high);
}

double CCalendarCyclicTest::significance(double n, double x) const
{
    try
    {
        boost::math::binomial binom(n, 1.0 - LARGE_ERROR_PERCENTILE / 100.0);
        return std::min(2.0 * CTools::safeCdfComplement(binom, x - 1.0), 1.0);
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Failed to calculate significance: " << e.what()
                  << " n = " << n << " x = " << x);
    }
    return 1.0;
}

const core_t::TTime CCalendarCyclicTest::BUCKET{core::constants::DAY};
const core_t::TTime CCalendarCyclicTest::WINDOW{124 * BUCKET};
const double CCalendarCyclicTest::LARGE_ERROR_PERCENTILE(99.0);
const unsigned int CCalendarCyclicTest::MINIMUM_REPEATS{4};
const uint32_t CCalendarCyclicTest::COUNT_BITS{0x100000};
// TODO support offsets are +/- 12hrs for time zones.
const TTimeVec CCalendarCyclicTest::TIMEZONE_OFFSETS{0};

}
}
