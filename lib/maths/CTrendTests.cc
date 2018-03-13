/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */

#include <maths/CTrendTests.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CPersistUtils.h>
#include <core/CScopedLock.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CTimezone.h>
#include <core/CTriple.h>
#include <core/Constants.h>
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
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/range.hpp>
#include <boost/ref.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace ml {
namespace maths {
namespace {

using TDoubleVec = std::vector<double>;
using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
using TTimeVec = std::vector<core_t::TTime>;

//! \brief Sets the timezone to a specified value in a constructor
//! call so it can be called once by static initialisation.
struct SSetTimeZone {
    SSetTimeZone(const std::string &zone) { core::CTimezone::instance().timezoneName(zone); }
};

//! Generate \p n samples uniformly in the interval [\p a, \p b].
template <typename ITR>
void generateUniformSamples(boost::random::mt19937_64 &rng,
                            double a,
                            double b,
                            std::size_t n,
                            ITR samples) {
    boost::random::uniform_real_distribution<> uniform(a, b);
    std::generate_n(samples, n, boost::bind(uniform, boost::ref(rng)));
}

//! Force the sample mean to zero.
void zeroMean(TDoubleVec &samples) {
    TMeanAccumulator mean;
    for (auto sample : samples) {
        mean.add(sample);
    }
    for (auto &&sample : samples) {
        sample -= CBasicStatistics::mean(mean);
    }
}

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

// CCalendarCyclicTest
const std::string ERROR_QUANTILES_TAG("a");
const std::string BUCKET_TAG("c");
const std::string ERROR_COUNTS_TAG("d");
const std::string ERROR_SUMS_TAG("e");

//! The maximum significance of a test statistic.
const double MAXIMUM_SIGNIFICANCE = 0.001;
//! Forward day in seconds into scope.
const core_t::TTime DAY = core::constants::DAY;
//! Forward day in seconds into scope.
const core_t::TTime WEEK = core::constants::WEEK;
}

//////// CRandomizedPeriodicitytest ////////

CRandomizedPeriodicityTest::CRandomizedPeriodicityTest(void)
    : m_DayRefreshedProjections(-DAY_RESAMPLE_INTERVAL),
      m_WeekRefreshedProjections(-DAY_RESAMPLE_INTERVAL) {
    resample(0);
}

bool CRandomizedPeriodicityTest::staticsAcceptRestoreTraverser(
    core::CStateRestoreTraverser &traverser) {
    // Note we require that we only ever do one persistence per process.

    std::size_t index = 0;
    reset();

    core::CScopedLock lock(ms_Lock);

    do {
        const std::string &name = traverser.name();

        if (name == RNG_TAG) {
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
    } while (traverser.next());

    return true;
}

void CRandomizedPeriodicityTest::staticsAcceptPersistInserter(
    core::CStatePersistInserter &inserter) {
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
    for (std::size_t i = 0; i < N; ++i) {
        inserter.insertValue(ARRAY_INDEX_TAG, i);
        for (auto rand : ms_DayRandomProjections[i]) {
            inserter.insertValue(DAY_RANDOM_PROJECTIONS_TAG, rand);
        }
        for (auto rand : ms_DayPeriodicProjections[i]) {
            inserter.insertValue(DAY_PERIODIC_PROJECTIONS_TAG, rand);
        }
        for (auto rand : ms_WeekRandomProjections[i]) {
            inserter.insertValue(WEEK_RANDOM_PROJECTIONS_TAG, rand);
        }
        for (auto rand : ms_WeekPeriodicProjections[i]) {
            inserter.insertValue(WEEK_PERIODIC_PROJECTIONS_TAG, rand);
        }
    }
}

bool CRandomizedPeriodicityTest::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser) {
    do {
        const std::string &name = traverser.name();

        RESTORE(DAY_PROJECTIONS_TAG, m_DayProjections.fromDelimited(traverser.value()))
        RESTORE(DAY_STATISTICS_TAG, m_DayStatistics.fromDelimited(traverser.value()))
        RESTORE(DAY_REFRESHED_PROJECTIONS_TAG,
                core::CStringUtils::stringToType(traverser.value(), m_DayRefreshedProjections))
        RESTORE(WEEK_PROJECTIONS_TAG, m_WeekProjections.fromDelimited(traverser.value()))
        RESTORE(WEEK_STATISTICS_TAG, m_WeekStatistics.fromDelimited(traverser.value()))
        RESTORE(DAY_STATISTICS_TAG, m_DayStatistics.fromDelimited(traverser.value()))
        RESTORE(WEEK_REFRESHED_PROJECTIONS_TAG,
                core::CStringUtils::stringToType(traverser.value(), m_WeekRefreshedProjections))
    } while (traverser.next());

    return true;
}

void CRandomizedPeriodicityTest::acceptPersistInserter(
    core::CStatePersistInserter &inserter) const {
    inserter.insertValue(DAY_PROJECTIONS_TAG, m_DayProjections.toDelimited());
    inserter.insertValue(DAY_STATISTICS_TAG, m_DayStatistics.toDelimited());
    inserter.insertValue(DAY_REFRESHED_PROJECTIONS_TAG, m_DayRefreshedProjections);
    inserter.insertValue(WEEK_PROJECTIONS_TAG, m_WeekProjections.toDelimited());
    inserter.insertValue(WEEK_STATISTICS_TAG, m_WeekStatistics.toDelimited());
    inserter.insertValue(WEEK_REFRESHED_PROJECTIONS_TAG, m_WeekRefreshedProjections);
}

void CRandomizedPeriodicityTest::add(core_t::TTime time, double value) {
    resample(time);

    if (time >= m_DayRefreshedProjections + DAY_RESAMPLE_INTERVAL) {
        LOG_TRACE("Updating day statistics");
        updateStatistics(m_DayProjections, m_DayStatistics);
        m_DayRefreshedProjections = CIntegerTools::floor(time, DAY_RESAMPLE_INTERVAL);
    }
    if (time >= m_WeekRefreshedProjections + WEEK_RESAMPLE_INTERVAL) {
        LOG_TRACE("Updating week statistics");
        updateStatistics(m_WeekProjections, m_WeekStatistics);
        m_WeekRefreshedProjections = CIntegerTools::floor(time, WEEK_RESAMPLE_INTERVAL);
    }

    TVector2N daySample;
    TVector2N weekSample;
    std::size_t td = static_cast<std::size_t>((time % DAY_RESAMPLE_INTERVAL) / SAMPLE_INTERVAL);
    std::size_t d = static_cast<std::size_t>((time % DAY) / SAMPLE_INTERVAL);
    std::size_t tw = static_cast<std::size_t>((time % WEEK_RESAMPLE_INTERVAL) / SAMPLE_INTERVAL);
    std::size_t w = static_cast<std::size_t>((time % WEEK) / SAMPLE_INTERVAL);

    for (std::size_t i = 0u; i < N; ++i) {
        daySample(2 * i + 0) = ms_DayRandomProjections[i][td] * value;
        daySample(2 * i + 1) = ms_DayPeriodicProjections[i][d] * value;
        weekSample(2 * i + 0) = ms_WeekRandomProjections[i][tw] * value;
        weekSample(2 * i + 1) = ms_WeekPeriodicProjections[i][w] * value;
    }

    m_DayProjections.add(daySample);
    m_WeekProjections.add(weekSample);
}

bool CRandomizedPeriodicityTest::test(void) const {
    static const double SIGNIFICANCE = 1e-3;

    try {
        double nd = CBasicStatistics::count(m_DayStatistics);
        if (nd >= 1.0) {
            TVector2 S = CBasicStatistics::mean(m_DayStatistics);
            LOG_TRACE("Day test statistic, S = " << S << ", n = " << nd);
            double ratio = S(0) == S(1) ? 1.0
                                        : (S(0) == 0.0 ? boost::numeric::bounds<double>::highest()
                                                       : static_cast<double>(S(1) / S(0)));
            double significance = CStatisticalTests::rightTailFTest(ratio, nd, nd);
            LOG_TRACE("Daily significance = " << significance);
            if (significance < SIGNIFICANCE) {
                return true;
            }
        }

        double nw = CBasicStatistics::count(m_WeekStatistics);
        if (nw >= 1.0) {
            TVector2 S = CBasicStatistics::mean(m_WeekStatistics);
            LOG_TRACE("Week test statistic, S = " << S);
            double ratio = S(0) == S(1) ? 1.0
                                        : (S(0) == 0.0 ? boost::numeric::bounds<double>::highest()
                                                       : static_cast<double>(S(1) / S(0)));
            double significance = CStatisticalTests::rightTailFTest(ratio, nw, nw);
            LOG_TRACE("Weekly significance = " << significance);
            if (significance < SIGNIFICANCE) {
                return true;
            }
        }
    } catch (const std::exception &e) { LOG_ERROR("Failed to test for periodicity: " << e.what()); }

    return false;
}

void CRandomizedPeriodicityTest::reset(void) {
    core::CScopedLock lock(ms_Lock);

    ms_Rng = boost::random::mt19937_64();
    for (std::size_t i = 0u; i < N; ++i) {
        ms_DayRandomProjections[i].clear();
        ms_DayPeriodicProjections[i].clear();
        ms_WeekRandomProjections[i].clear();
        ms_WeekPeriodicProjections[i].clear();
    }
    ms_DayResampled = -DAY_RESAMPLE_INTERVAL;
    ms_WeekResampled = -WEEK_RESAMPLE_INTERVAL;
}

uint64_t CRandomizedPeriodicityTest::checksum(uint64_t seed) const {
    // This checksum is problematic until we switch to using our
    // own rng for each test.
    // seed = CChecksum::calculate(seed, m_DayProjections);
    // seed = CChecksum::calculate(seed, m_DayStatistics);
    // seed = CChecksum::calculate(seed, m_DayRefreshedProjections);
    // seed = CChecksum::calculate(seed, m_WeekProjections);
    // seed = CChecksum::calculate(seed, m_WeekStatistics);
    // return CChecksum::calculate(seed, m_WeekRefreshedProjections);
    return seed;
}

void CRandomizedPeriodicityTest::updateStatistics(TVector2NMeanAccumulator &projections,
                                                  TVector2MeanAccumulator &statistics) {
    static const double ALPHA = 0.1;

    if (CBasicStatistics::count(projections) > 0.0) {
        const TVector2N &mean = CBasicStatistics::mean(projections);
        LOG_TRACE("mean = " << mean);

        TVector2MeanAccumulator statistic;
        for (std::size_t i = 0u; i < N; ++i) {
            TVector2 s;
            s(0) = mean(2 * i + 0) * mean(2 * i + 0);
            s(1) = mean(2 * i + 1) * mean(2 * i + 1);
            statistic.add(s);
        }
        statistics += statistic;
        statistics.age(1.0 - ALPHA);
        LOG_TRACE("statistics = " << statistics);
    }

    projections = TVector2NMeanAccumulator();
}

void CRandomizedPeriodicityTest::resample(core_t::TTime time) {
    if (time >= ms_DayResampled.load(std::memory_order_acquire) + DAY_RESAMPLE_INTERVAL) {
        core::CScopedLock lock(ms_Lock);

        LOG_TRACE("Updating daily random projections at " << time);
        if (time >= ms_DayResampled.load(std::memory_order_relaxed) + DAY_RESAMPLE_INTERVAL) {
            resample(
                DAY, DAY_RESAMPLE_INTERVAL, ms_DayPeriodicProjections, ms_DayRandomProjections);
            ms_DayResampled.store(CIntegerTools::floor(time, DAY_RESAMPLE_INTERVAL),
                                  std::memory_order_release);
        }
    }

    if (time >= ms_WeekResampled.load(std::memory_order_acquire) + WEEK_RESAMPLE_INTERVAL) {
        core::CScopedLock lock(ms_Lock);

        LOG_TRACE("Updating weekly random projections at " << time);
        if (time >= ms_WeekResampled.load(std::memory_order_relaxed) + WEEK_RESAMPLE_INTERVAL) {
            resample(
                WEEK, WEEK_RESAMPLE_INTERVAL, ms_WeekPeriodicProjections, ms_WeekRandomProjections);
            ms_WeekResampled.store(CIntegerTools::floor(time, WEEK_RESAMPLE_INTERVAL),
                                   std::memory_order_release);
        }
    }
}

void CRandomizedPeriodicityTest::resample(core_t::TTime period,
                                          core_t::TTime resampleInterval,
                                          TDoubleVec (&periodicProjections)[N],
                                          TDoubleVec (&randomProjections)[N]) {
    std::size_t n = static_cast<std::size_t>(period / SAMPLE_INTERVAL);
    std::size_t t = static_cast<std::size_t>(resampleInterval / SAMPLE_INTERVAL);
    std::size_t p = static_cast<std::size_t>(resampleInterval / period);
    for (std::size_t i = 0u; i < N; ++i) {
        periodicProjections[i].resize(n);
        generateUniformSamples(ms_Rng, -1.0, 1.0, n, periodicProjections[i].begin());
        zeroMean(periodicProjections[i]);
        randomProjections[i].resize(t);
        for (std::size_t j = 0u; j < p; ++j) {
            std::copy(periodicProjections[i].begin(),
                      periodicProjections[i].end(),
                      randomProjections[i].begin() + j * n);
            CSampling::random_shuffle(ms_Rng,
                                      randomProjections[i].begin() + j * n,
                                      randomProjections[i].begin() + (j + 1) * n);
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

//////// CCalendarCyclicTest ////////

CCalendarCyclicTest::CCalendarCyclicTest(double decayRate)
    : m_DecayRate(decayRate),
      m_Bucket(0),
      m_ErrorQuantiles(CQuantileSketch::E_Linear, 20),
      m_ErrorCounts(WINDOW / BUCKET) {
    static const SSetTimeZone timezone("GMT");
    m_ErrorSums.reserve(WINDOW / BUCKET / 10);
}

bool CCalendarCyclicTest::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser) {
    do {
        const std::string &name = traverser.name();
        RESTORE_BUILT_IN(BUCKET_TAG, m_Bucket)
        RESTORE(ERROR_QUANTILES_TAG,
                traverser.traverseSubLevel(
                    boost::bind(&CQuantileSketch::acceptRestoreTraverser, &m_ErrorQuantiles, _1)))
        RESTORE(ERROR_COUNTS_TAG,
                core::CPersistUtils::restore(ERROR_COUNTS_TAG, m_ErrorCounts, traverser))
        RESTORE(ERROR_SUMS_TAG, core::CPersistUtils::fromString(traverser.value(), m_ErrorSums))
    } while (traverser.next());
    return true;
}

void CCalendarCyclicTest::acceptPersistInserter(core::CStatePersistInserter &inserter) const {
    inserter.insertValue(BUCKET_TAG, m_Bucket);
    inserter.insertLevel(
        ERROR_QUANTILES_TAG,
        boost::bind(&CQuantileSketch::acceptPersistInserter, &m_ErrorQuantiles, _1));
    core::CPersistUtils::persist(ERROR_COUNTS_TAG, m_ErrorCounts, inserter);
    inserter.insertValue(ERROR_SUMS_TAG, core::CPersistUtils::toString(m_ErrorSums));
}

void CCalendarCyclicTest::propagateForwardsByTime(double time) {
    if (!CMathsFuncs::isFinite(time) || time < 0.0) {
        LOG_ERROR("Bad propagation time " << time);
        return;
    }
    m_ErrorQuantiles.age(std::exp(-m_DecayRate * time));
}

void CCalendarCyclicTest::add(core_t::TTime time, double error, double weight) {
    error = std::fabs(error);

    m_ErrorQuantiles.add(error, weight);

    if (m_ErrorQuantiles.count() > 100.0) {
        core_t::TTime bucket = CIntegerTools::floor(time, BUCKET);
        if (m_ErrorCounts.empty()) {
            m_ErrorCounts.push_back(0);
        } else {
            for (core_t::TTime i = m_Bucket; i < bucket; i += BUCKET) {
                m_ErrorCounts.push_back(0);
            }
        }

        uint32_t &count = m_ErrorCounts.back();
        count += (count % COUNT_BITS < COUNT_BITS - 1) ? 1 : 0;

        double high;
        m_ErrorQuantiles.quantile(LARGE_ERROR_PERCENTILE, high);

        m_ErrorSums.erase(m_ErrorSums.begin(),
                          std::find_if(m_ErrorSums.begin(),
                                       m_ErrorSums.end(),
                                       [bucket](const TTimeFloatPr &error_) {
                                           return error_.first + WINDOW > bucket;
                                       }));
        if (error >= high) {
            count += (count < 0x100000000 - COUNT_BITS) ? COUNT_BITS : 0;
            m_ErrorSums[bucket] += this->winsorise(error);
        }

        m_Bucket = bucket;
    }
}

CCalendarCyclicTest::TOptionalFeature CCalendarCyclicTest::test(void) const {
    // The statistics we need in order to be able to test for calendar
    // features.
    struct SStats {
        SStats(void) : s_Offset(0), s_Repeats(0), s_Sum(0.0), s_Count(0.0), s_Significance(0.0) {}
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

    for (auto offset : TIMEZONE_OFFSETS) {
        for (const auto &error : m_ErrorSums) {
            std::size_t i = m_ErrorCounts.size() - 1 -
                            static_cast<std::size_t>((m_Bucket - error.first) / BUCKET);
            double n = static_cast<double>(m_ErrorCounts[i] % COUNT_BITS);
            double x = static_cast<double>(m_ErrorCounts[i] / COUNT_BITS);
            double s = this->significance(n, x);
            for (auto feature : CCalendarFeature::features(error.first + BUCKET / 2 + offset)) {
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

    for (const auto &stat : stats) {
        CCalendarFeature feature = stat.first;
        double r = static_cast<double>(stat.second.s_Repeats);
        double x = stat.second.s_Count;
        double e = stat.second.s_Sum;
        double s = stat.second.s_Significance;
        if (stat.second.s_Repeats >= MINIMUM_REPEATS && e > errorThreshold * x &&
            ::pow(s, r) < MAXIMUM_SIGNIFICANCE) {
            result.add({e, stat.second.s_Offset, feature});
        }
    }

    return result.count() > 0 ? result[0].third : TOptionalFeature();
}

uint64_t CCalendarCyclicTest::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_ErrorQuantiles);
    seed = CChecksum::calculate(seed, m_ErrorCounts);
    return CChecksum::calculate(seed, m_ErrorSums);
}

void CCalendarCyclicTest::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CCalendarCyclicTest");
    core::CMemoryDebug::dynamicSize("m_ErrorQuantiles", m_ErrorQuantiles, mem);
    core::CMemoryDebug::dynamicSize("m_ErrorCounts", m_ErrorCounts, mem);
    core::CMemoryDebug::dynamicSize("m_ErrorSums", m_ErrorSums, mem);
}

std::size_t CCalendarCyclicTest::memoryUsage(void) const {
    return core::CMemory::dynamicSize(m_ErrorQuantiles) +
           core::CMemory::dynamicSize(m_ErrorCounts) + core::CMemory::dynamicSize(m_ErrorSums);
}

double CCalendarCyclicTest::winsorise(double error) const {
    double high;
    m_ErrorQuantiles.quantile(99.5, high);
    return std::min(error, high);
}

double CCalendarCyclicTest::significance(double n, double x) const {
    try {
        boost::math::binomial binom(n, 1.0 - LARGE_ERROR_PERCENTILE / 100.0);
        return std::min(2.0 * CTools::safeCdfComplement(binom, x - 1.0), 1.0);
    } catch (const std::exception &e) {
        LOG_ERROR("Failed to calculate significance: " << e.what() << " n = " << n << " x = " << x);
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
