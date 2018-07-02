/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CRandomizedPeriodicityTest.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CScopedLock.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/Constants.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>
#include <maths/CIntegerTools.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraPersist.h>
#include <maths/CLinearAlgebraTools.h>
#include <maths/CSampling.h>
#include <maths/CStatisticalTests.h>

#include <boost/bind.hpp>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/range.hpp>
#include <boost/ref.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace ml {
namespace maths {
namespace {

using TDoubleVec = std::vector<double>;
using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
using TTimeVec = std::vector<core_t::TTime>;

//! Generate \p n samples uniformly in the interval [\p a, \p b].
template<typename ITR>
void generateUniformSamples(boost::random::mt19937_64& rng, double a, double b, std::size_t n, ITR samples) {
    boost::random::uniform_real_distribution<> uniform(a, b);
    std::generate_n(samples, n, boost::bind(uniform, boost::ref(rng)));
}

//! Force the sample mean to zero.
void zeroMean(TDoubleVec& samples) {
    TMeanAccumulator mean;
    for (auto sample : samples) {
        mean.add(sample);
    }
    for (auto& sample : samples) {
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
}

//////// CRandomizedPeriodicitytest ////////

CRandomizedPeriodicityTest::CRandomizedPeriodicityTest()
    : m_DayRefreshedProjections(-DAY_RESAMPLE_INTERVAL),
      m_WeekRefreshedProjections(-DAY_RESAMPLE_INTERVAL) {
    resample(0);
}

bool CRandomizedPeriodicityTest::staticsAcceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    // Note we require that we only ever do one persistence per process.

    std::size_t index = 0;
    reset();

    core::CScopedLock lock(ms_Lock);

    do {
        const std::string& name = traverser.name();

        if (name == RNG_TAG) {
            // Replace '_' with space
            std::string value(traverser.value());
            std::replace(value.begin(), value.end(), '_', ' ');
            std::stringstream ss;
            ss << value;
            ss >> ms_Rng;
            continue;
        }
        RESTORE_SETUP_TEARDOWN(DAY_RESAMPLED_TAG, core_t::TTime resampled,
                               core::CStringUtils::stringToType(traverser.value(), resampled),
                               ms_DayResampled.store(resampled))
        RESTORE_SETUP_TEARDOWN(WEEK_RESAMPLED_TAG, core_t::TTime resampled,
                               core::CStringUtils::stringToType(traverser.value(), resampled),
                               ms_WeekResampled.store(resampled))
        RESTORE_BUILT_IN(ARRAY_INDEX_TAG, index)
        RESTORE_SETUP_TEARDOWN(DAY_RANDOM_PROJECTIONS_TAG, double d,
                               core::CStringUtils::stringToType(traverser.value(), d),
                               ms_DayRandomProjections[index].push_back(d))
        RESTORE_SETUP_TEARDOWN(DAY_PERIODIC_PROJECTIONS_TAG, double d,
                               core::CStringUtils::stringToType(traverser.value(), d),
                               ms_DayPeriodicProjections[index].push_back(d))
        RESTORE_SETUP_TEARDOWN(WEEK_RANDOM_PROJECTIONS_TAG, double d,
                               core::CStringUtils::stringToType(traverser.value(), d),
                               ms_WeekRandomProjections[index].push_back(d))
        RESTORE_SETUP_TEARDOWN(WEEK_PERIODIC_PROJECTIONS_TAG, double d,
                               core::CStringUtils::stringToType(traverser.value(), d),
                               ms_WeekPeriodicProjections[index].push_back(d))
    } while (traverser.next());

    return true;
}

void CRandomizedPeriodicityTest::staticsAcceptPersistInserter(core::CStatePersistInserter& inserter) {
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

bool CRandomizedPeriodicityTest::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();

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

void CRandomizedPeriodicityTest::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
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
        LOG_TRACE(<< "Updating day statistics");
        updateStatistics(m_DayProjections, m_DayStatistics);
        m_DayRefreshedProjections = CIntegerTools::floor(time, DAY_RESAMPLE_INTERVAL);
    }
    if (time >= m_WeekRefreshedProjections + WEEK_RESAMPLE_INTERVAL) {
        LOG_TRACE(<< "Updating week statistics");
        updateStatistics(m_WeekProjections, m_WeekStatistics);
        m_WeekRefreshedProjections = CIntegerTools::floor(time, WEEK_RESAMPLE_INTERVAL);
    }

    TVector2N daySample;
    TVector2N weekSample;
    std::size_t td = static_cast<std::size_t>((time % DAY_RESAMPLE_INTERVAL) / SAMPLE_INTERVAL);
    std::size_t d = static_cast<std::size_t>((time % core::constants::DAY) / SAMPLE_INTERVAL);
    std::size_t tw = static_cast<std::size_t>((time % WEEK_RESAMPLE_INTERVAL) / SAMPLE_INTERVAL);
    std::size_t w = static_cast<std::size_t>((time % core::constants::WEEK) / SAMPLE_INTERVAL);

    for (std::size_t i = 0u; i < N; ++i) {
        daySample(2 * i + 0) = ms_DayRandomProjections[i][td] * value;
        daySample(2 * i + 1) = ms_DayPeriodicProjections[i][d] * value;
        weekSample(2 * i + 0) = ms_WeekRandomProjections[i][tw] * value;
        weekSample(2 * i + 1) = ms_WeekPeriodicProjections[i][w] * value;
    }

    m_DayProjections.add(daySample);
    m_WeekProjections.add(weekSample);
}

bool CRandomizedPeriodicityTest::test() const {
    static const double SIGNIFICANCE = 1e-3;

    try {
        double nd = CBasicStatistics::count(m_DayStatistics);
        if (nd >= 1.0) {
            TVector2 S = CBasicStatistics::mean(m_DayStatistics);
            LOG_TRACE(<< "Day test statistic, S = " << S << ", n = " << nd);
            double ratio = S(0) == S(1)
                               ? 1.0
                               : (S(0) == 0.0 ? boost::numeric::bounds<double>::highest()
                                              : static_cast<double>(S(1) / S(0)));
            double significance = CStatisticalTests::rightTailFTest(ratio, nd, nd);
            LOG_TRACE(<< "Daily significance = " << significance);
            if (significance < SIGNIFICANCE) {
                return true;
            }
        }

        double nw = CBasicStatistics::count(m_WeekStatistics);
        if (nw >= 1.0) {
            TVector2 S = CBasicStatistics::mean(m_WeekStatistics);
            LOG_TRACE(<< "Week test statistic, S = " << S);
            double ratio = S(0) == S(1)
                               ? 1.0
                               : (S(0) == 0.0 ? boost::numeric::bounds<double>::highest()
                                              : static_cast<double>(S(1) / S(0)));
            double significance = CStatisticalTests::rightTailFTest(ratio, nw, nw);
            LOG_TRACE(<< "Weekly significance = " << significance);
            if (significance < SIGNIFICANCE) {
                return true;
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to test for periodicity: " << e.what());
    }

    return false;
}

void CRandomizedPeriodicityTest::reset() {
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

std::uint64_t CRandomizedPeriodicityTest::checksum(std::uint64_t seed) const {
    // This checksum is problematic until we switch to using our
    // own rng for each test.
    seed = CChecksum::calculate(seed, m_DayProjections);
    seed = CChecksum::calculate(seed, m_DayStatistics);
    seed = CChecksum::calculate(seed, m_DayRefreshedProjections);
    seed = CChecksum::calculate(seed, m_WeekProjections);
    seed = CChecksum::calculate(seed, m_WeekStatistics);
    return CChecksum::calculate(seed, m_WeekRefreshedProjections);
}

void CRandomizedPeriodicityTest::updateStatistics(TVector2NMeanAccumulator& projections,
                                                  TVector2MeanAccumulator& statistics) {
    static const double ALPHA = 0.1;

    if (CBasicStatistics::count(projections) > 0.0) {
        const TVector2N& mean = CBasicStatistics::mean(projections);
        LOG_TRACE(<< "mean = " << mean);

        TVector2MeanAccumulator statistic;
        for (std::size_t i = 0u; i < N; ++i) {
            TVector2 s;
            s(0) = mean(2 * i + 0) * mean(2 * i + 0);
            s(1) = mean(2 * i + 1) * mean(2 * i + 1);
            statistic.add(s);
        }
        statistics += statistic;
        statistics.age(1.0 - ALPHA);
        LOG_TRACE(<< "statistics = " << statistics);
    }

    projections = TVector2NMeanAccumulator();
}

void CRandomizedPeriodicityTest::resample(core_t::TTime time) {
    if (time >= ms_DayResampled.load(atomic_t::memory_order_acquire) + DAY_RESAMPLE_INTERVAL) {
        core::CScopedLock lock(ms_Lock);

        LOG_TRACE(<< "Updating daily random projections at " << time);
        if (time >= ms_DayResampled.load(atomic_t::memory_order_relaxed) + DAY_RESAMPLE_INTERVAL) {
            resample(core::constants::DAY, DAY_RESAMPLE_INTERVAL,
                     ms_DayPeriodicProjections, ms_DayRandomProjections);
            ms_DayResampled.store(CIntegerTools::floor(time, DAY_RESAMPLE_INTERVAL),
                                  atomic_t::memory_order_release);
        }
    }

    if (time >= ms_WeekResampled.load(atomic_t::memory_order_acquire) + WEEK_RESAMPLE_INTERVAL) {
        core::CScopedLock lock(ms_Lock);

        LOG_TRACE(<< "Updating weekly random projections at " << time);
        if (time >= ms_WeekResampled.load(atomic_t::memory_order_relaxed) + WEEK_RESAMPLE_INTERVAL) {
            resample(core::constants::WEEK, WEEK_RESAMPLE_INTERVAL,
                     ms_WeekPeriodicProjections, ms_WeekRandomProjections);
            ms_WeekResampled.store(CIntegerTools::floor(time, WEEK_RESAMPLE_INTERVAL),
                                   atomic_t::memory_order_release);
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
            std::copy(periodicProjections[i].begin(), periodicProjections[i].end(),
                      randomProjections[i].begin() + j * n);
            CSampling::random_shuffle(ms_Rng, randomProjections[i].begin() + j * n,
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
atomic_t::atomic<core_t::TTime> CRandomizedPeriodicityTest::ms_DayResampled(-DAY_RESAMPLE_INTERVAL);
TDoubleVec CRandomizedPeriodicityTest::ms_WeekRandomProjections[N] = {};
TDoubleVec CRandomizedPeriodicityTest::ms_WeekPeriodicProjections[N] = {};
atomic_t::atomic<core_t::TTime> CRandomizedPeriodicityTest::ms_WeekResampled(-WEEK_RESAMPLE_INTERVAL);
core::CMutex CRandomizedPeriodicityTest::ms_Lock;
}
}
