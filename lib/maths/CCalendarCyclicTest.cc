/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CCalendarCyclicTest.h>

#include <core/CLogger.h>
#include <core/CPersistUtils.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CTimezone.h>
#include <core/CTriple.h>
#include <core/Constants.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CChecksum.h>
#include <maths/CIntegerTools.h>
#include <maths/CMathsFuncs.h>
#include <maths/CTools.h>
#include <maths/Constants.h>

#include <boost/bind.hpp>
#include <boost/math/distributions/binomial.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

namespace ml {
namespace maths {
namespace {
//! \brief Sets the timezone to a specified value in a constructor
//! call so it can be called once by static initialisation.
struct SSetTimeZone {
    SSetTimeZone(const std::string& zone) {
        core::CTimezone::instance().timezoneName(zone);
    }
};

const std::string ERROR_QUANTILES_TAG("a");
const std::string BUCKET_TAG("c");
const std::string ERROR_COUNTS_TAG("d");
const std::string ERROR_SUMS_TAG("e");
}

CCalendarCyclicTest::CCalendarCyclicTest(double decayRate)
    : m_DecayRate(decayRate), m_Bucket(0),
      m_ErrorQuantiles(CQuantileSketch::E_Linear, 20), m_ErrorCounts(WINDOW / BUCKET) {
    static const SSetTimeZone timezone("GMT");
    m_ErrorSums.reserve(WINDOW / BUCKET / 10);
}

bool CCalendarCyclicTest::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE_BUILT_IN(BUCKET_TAG, m_Bucket)
        RESTORE(ERROR_QUANTILES_TAG,
                traverser.traverseSubLevel(boost::bind(&CQuantileSketch::acceptRestoreTraverser,
                                                       &m_ErrorQuantiles, _1)))
        RESTORE(ERROR_COUNTS_TAG,
                core::CPersistUtils::restore(ERROR_COUNTS_TAG, m_ErrorCounts, traverser))
        RESTORE(ERROR_SUMS_TAG, core::CPersistUtils::fromString(traverser.value(), m_ErrorSums))
    } while (traverser.next());
    return true;
}

void CCalendarCyclicTest::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(BUCKET_TAG, m_Bucket);
    inserter.insertLevel(ERROR_QUANTILES_TAG, boost::bind(&CQuantileSketch::acceptPersistInserter,
                                                          &m_ErrorQuantiles, _1));
    core::CPersistUtils::persist(ERROR_COUNTS_TAG, m_ErrorCounts, inserter);
    inserter.insertValue(ERROR_SUMS_TAG, core::CPersistUtils::toString(m_ErrorSums));
}

void CCalendarCyclicTest::propagateForwardsByTime(double time) {
    if (!CMathsFuncs::isFinite(time) || time < 0.0) {
        LOG_ERROR(<< "Bad propagation time " << time);
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

        std::uint32_t& count = m_ErrorCounts.back();
        count += (count % COUNT_BITS < COUNT_BITS - 1) ? 1 : 0;

        double high;
        m_ErrorQuantiles.quantile(LARGE_ERROR_PERCENTILE, high);

        m_ErrorSums.erase(m_ErrorSums.begin(),
                          std::find_if(m_ErrorSums.begin(), m_ErrorSums.end(),
                                       [bucket](const TTimeFloatPr& error_) {
                                           return error_.first + WINDOW > bucket;
                                       }));
        if (error >= high) {
            count += (count < 0x100000000 - COUNT_BITS) ? COUNT_BITS : 0;
            m_ErrorSums[bucket] += this->winsorise(error);
        }

        m_Bucket = bucket;
    }
}

CCalendarCyclicTest::TOptionalFeature CCalendarCyclicTest::test() const {
    // The statistics we need in order to be able to test for calendar
    // features.
    struct SStats {
        SStats()
            : s_Offset(0), s_Repeats(0), s_Sum(0.0), s_Count(0.0),
              s_Significance(0.0) {}
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
        for (const auto& error : m_ErrorSums) {
            std::size_t i = m_ErrorCounts.size() - 1 -
                            static_cast<std::size_t>((m_Bucket - error.first) / BUCKET);
            double n = static_cast<double>(m_ErrorCounts[i] % COUNT_BITS);
            double x = static_cast<double>(m_ErrorCounts[i] / COUNT_BITS);
            double s = this->significance(n, x);
            for (auto feature :
                 CCalendarFeature::features(error.first + BUCKET / 2 + offset)) {
                SStats& stat = stats[feature];
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

    for (const auto& stat : stats) {
        CCalendarFeature feature = stat.first;
        double r = static_cast<double>(stat.second.s_Repeats);
        double x = stat.second.s_Count;
        double e = stat.second.s_Sum;
        double s = stat.second.s_Significance;
        if (stat.second.s_Repeats >= MINIMUM_REPEATS && e > errorThreshold * x &&
            std::pow(s, r) < COMPONENT_STATISTICALLY_SIGNIFICANT) {
            result.add({e, stat.second.s_Offset, feature});
        }
    }

    return result.count() > 0 ? result[0].third : TOptionalFeature();
}

std::uint64_t CCalendarCyclicTest::checksum(std::uint64_t seed) const {
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

std::size_t CCalendarCyclicTest::memoryUsage() const {
    return core::CMemory::dynamicSize(m_ErrorQuantiles) +
           core::CMemory::dynamicSize(m_ErrorCounts) +
           core::CMemory::dynamicSize(m_ErrorSums);
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
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to calculate significance: " << e.what()
                  << " n = " << n << " x = " << x);
    }
    return 1.0;
}

const core_t::TTime CCalendarCyclicTest::BUCKET{core::constants::DAY};
const core_t::TTime CCalendarCyclicTest::WINDOW{124 * BUCKET};
const double CCalendarCyclicTest::LARGE_ERROR_PERCENTILE(99.0);
const unsigned int CCalendarCyclicTest::MINIMUM_REPEATS{4};
const std::uint32_t CCalendarCyclicTest::COUNT_BITS{0x100000};
// TODO support offsets are +/- 12hrs for time zones.
const CCalendarCyclicTest::TTimeVec CCalendarCyclicTest::TIMEZONE_OFFSETS{0};
}
}
