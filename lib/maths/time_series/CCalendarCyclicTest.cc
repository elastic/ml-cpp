/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <maths/time_series/CCalendarCyclicTest.h>

#include <core/CLogger.h>
#include <core/CPersistUtils.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CTimezone.h>
#include <core/CTriple.h>
#include <core/CompressUtils.h>
#include <core/Constants.h>
#include <core/RestoreMacros.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CChecksum.h>
#include <maths/common/CIntegerTools.h>
#include <maths/common/CMathsFuncs.h>
#include <maths/common/CTools.h>
#include <maths/common/Constants.h>

#include <boost/math/distributions/binomial.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/unordered_map.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace ml {
namespace maths {
namespace time_series {
namespace {
using TTimeVec = std::vector<core_t::TTime>;

//! \brief Sets the time zone to a specified value in a constructor
//! call so it can be called once by static initialisation.
struct SSetTimeZone {
    explicit SSetTimeZone(const std::string& zone) {
        core::CTimezone::instance().timezoneName(zone);
    }
};

//! \brief Hashes a calendar feature.
struct SHashFeature {
    std::size_t operator()(const CCalendarFeature& feature) const {
        return feature.checksum(0);
    }
};

const std::string VERSION_6_4_TAG("6.4");
// Version 6.4
const core::TPersistenceTag ERROR_QUANTILES_6_4_TAG("a", "error_quantiles");
const core::TPersistenceTag CURRENT_BUCKET_TIME_6_4_TAG("b", "current_bucket_time");
const core::TPersistenceTag CURRENT_BUCKET_INDEX_6_4_TAG("c", "current_bucket_index");
const core::TPersistenceTag CURRENT_BUCKET_ERROR_STATS_6_4_TAG("d", "current_bucket_error_stats");
const core::TPersistenceTag ERRORS_6_4_TAG("e", "errors");
// Version < 6.4
const std::string ERROR_QUANTILES_OLD_TAG("a");
// Everything else gets default initialised.

const std::string DELIMITER{","};
const core_t::TTime SIZE{155};
const core_t::TTime BUCKET{core::constants::DAY};
const core_t::TTime WINDOW{SIZE * BUCKET};
// We can't determine the exact time zone given the state we maintain but we can
// determine if the whole pattern is shifted 1 day late or early.
const TTimeVec TIME_ZONE_OFFSETS{0, -12 * core::constants::HOUR - 1,
                                 12 * core::constants::HOUR + 1};

//! The percentile of a large error.
const double LARGE_ERROR_PERCENTILE{98.5};
//! The percentile of a very large error.
const double VERY_LARGE_ERROR_PERCENTILE{99.99};
//! The minimum number of repeats to test a feature.
const unsigned int MINIMUM_REPEATS{4};
//! The maximum significance to accept a feature.
const double MAXIMUM_SIGNIFICANCE{0.01};
}

CCalendarCyclicTest::CCalendarCyclicTest(double decayRate)
    : m_DecayRate{decayRate}, m_ErrorQuantiles{common::CQuantileSketch::E_Linear, 20},
      m_CurrentBucketTime{0}, m_CurrentBucketIndex{0} {
    static const SSetTimeZone timezone("GMT");
    TErrorStatsVec stats(SIZE);
    this->deflate(stats);
}

bool CCalendarCyclicTest::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    TErrorStatsVec errors;
    if (traverser.name() == VERSION_6_4_TAG) {
        while (traverser.next()) {
            const std::string& name = traverser.name();
            RESTORE(ERROR_QUANTILES_6_4_TAG, traverser.traverseSubLevel([this](auto& traverser_) {
                return m_ErrorQuantiles.acceptRestoreTraverser(traverser_);
            }))
            RESTORE_BUILT_IN(CURRENT_BUCKET_TIME_6_4_TAG, m_CurrentBucketTime)
            RESTORE_BUILT_IN(CURRENT_BUCKET_INDEX_6_4_TAG, m_CurrentBucketIndex)
            RESTORE(CURRENT_BUCKET_ERROR_STATS_6_4_TAG,
                    m_CurrentBucketErrorStats.fromDelimited(traverser.value()))
            RESTORE(ERRORS_6_4_TAG,
                    core::CPersistUtils::restore(ERRORS_6_4_TAG, errors, traverser))
        }
    } else {
        do {
            const std::string& name = traverser.name();
            RESTORE(ERROR_QUANTILES_OLD_TAG, traverser.traverseSubLevel([this](auto& traverser_) {
                return m_ErrorQuantiles.acceptRestoreTraverser(traverser_);
            }))
        } while (traverser.next());
        errors.resize(SIZE);
    }
    this->checkRestoredInvariants(errors);
    this->deflate(errors);
    return true;
}

void CCalendarCyclicTest::checkRestoredInvariants(const TErrorStatsVec& errors) const {
    VIOLATES_INVARIANT(m_CurrentBucketIndex, >=,
                       static_cast<core_t::TTime>(errors.size()));
    VIOLATES_INVARIANT(errors.size(), !=, SIZE);
}

void CCalendarCyclicTest::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(VERSION_6_4_TAG, "");
    inserter.insertLevel(ERROR_QUANTILES_6_4_TAG, [this](auto& inserter_) {
        m_ErrorQuantiles.acceptPersistInserter(inserter_);
    });
    inserter.insertValue(CURRENT_BUCKET_TIME_6_4_TAG, m_CurrentBucketTime);
    inserter.insertValue(CURRENT_BUCKET_INDEX_6_4_TAG, m_CurrentBucketIndex);
    inserter.insertValue(CURRENT_BUCKET_ERROR_STATS_6_4_TAG,
                         m_CurrentBucketErrorStats.toDelimited());
    TErrorStatsVec errors{this->inflate()};
    core::CPersistUtils::persist(ERRORS_6_4_TAG, errors, inserter);
}

void CCalendarCyclicTest::propagateForwardsByTime(double time) {
    if (common::CMathsFuncs::isFinite(time) == false || time < 0.0) {
        LOG_ERROR(<< "Bad propagation time " << time);
        return;
    }
    m_ErrorQuantiles.age(std::exp(-m_DecayRate * time));
}

void CCalendarCyclicTest::add(core_t::TTime time, double error, double weight) {
    error = std::fabs(error);

    m_ErrorQuantiles.add(error, weight);

    if (m_ErrorQuantiles.count() > 100.0) {
        time = common::CIntegerTools::floor(time, BUCKET);
        if (time > m_CurrentBucketTime) {
            TErrorStatsVec errors{this->inflate()};
            do {
                errors[m_CurrentBucketIndex] = m_CurrentBucketErrorStats;
                m_CurrentBucketErrorStats = SErrorStats{};
                m_CurrentBucketTime += BUCKET;
                m_CurrentBucketIndex = (m_CurrentBucketIndex + 1) % SIZE;
            } while (m_CurrentBucketTime < time);
            this->deflate(errors);
        }

        ++m_CurrentBucketErrorStats.s_Count;

        double largeError;
        m_ErrorQuantiles.quantile(LARGE_ERROR_PERCENTILE, largeError);
        if (error >= largeError) {
            bool isVeryLarge{100.0 * (1.0 - this->survivalFunction(error)) >=
                             VERY_LARGE_ERROR_PERCENTILE};
            m_CurrentBucketErrorStats.s_LargeErrorCount +=
                std::min(std::numeric_limits<std::size_t>::max() -
                             m_CurrentBucketErrorStats.s_LargeErrorCount,
                         static_cast<std::size_t>(isVeryLarge ? (1 << 17) + 1 : 1));
            m_CurrentBucketErrorStats.s_LargeErrorSum += this->winsorise(error);
        }
    }
}

CCalendarCyclicTest::TOptionalFeatureTimePr CCalendarCyclicTest::test() const {

    // The statistics we need in order to be able to test for calendar
    // features.
    struct SStats {
        core_t::TTime s_Offset{0};
        unsigned int s_Repeats{0};
        double s_Sum{0.0};
        double s_Count{0.0};
        double s_PValue{0.0};
    };
    using TFeatureStatsUMap = boost::unordered_map<CCalendarFeature, SStats, SHashFeature>;

    TErrorStatsVec errors{this->inflate()};

    double mostSignificantError{0.0};
    CCalendarFeature mostSignificantFeature;
    core_t::TTime mostSignificantOffset{0};

    double errorThreshold;
    m_ErrorQuantiles.quantile(50.0, errorThreshold);
    errorThreshold *= 2.0;

    for (auto offset : TIME_ZONE_OFFSETS) {
        // Note that the current index points to the next bucket to overwrite,
        // i.e. the earliest bucket error statistics we have. The start of
        // this bucket is WINDOW before the start time of the current partial
        // bucket.
        TFeatureStatsUMap stats{errors.size()};
        for (core_t::TTime i = m_CurrentBucketIndex, time = m_CurrentBucketTime - WINDOW;
             time < m_CurrentBucketTime; i = (i + 1) % SIZE, time += BUCKET) {
            if (errors[i].s_Count > 0) {
                double n{static_cast<double>(errors[i].s_Count)};
                double nl{static_cast<double>(errors[i].s_LargeErrorCount % (1 << 17))};
                double nv{static_cast<double>(errors[i].s_LargeErrorCount / (1 << 17))};
                double pValue{this->significance(n, nl, nv)};
                core_t::TTime midpoint{time + BUCKET / 2 + offset};
                for (auto feature : CCalendarFeature::features(midpoint)) {
                    if (feature.testForTimeZoneOffset(offset)) {
                        SStats& stat = stats[feature];
                        ++stat.s_Repeats;
                        stat.s_Sum += errors[i].s_LargeErrorSum;
                        stat.s_Count += nl;
                        stat.s_PValue = std::max(stat.s_PValue, pValue);
                    }
                }
            }
        }

        double mostSignificantErrorForOffset{0.0};
        CCalendarFeature mostSignificantFeatureForOffset;
        for (const auto& stat : stats) {
            CCalendarFeature feature = stat.first;
            double r{static_cast<double>(stat.second.s_Repeats)};
            double nl{stat.second.s_Count};
            double sl{stat.second.s_Sum};
            double pValue{stat.second.s_PValue};
            if (stat.second.s_Repeats >= MINIMUM_REPEATS &&
                sl > errorThreshold * nl && sl > mostSignificantErrorForOffset &&
                std::pow(pValue, r) < MAXIMUM_SIGNIFICANCE) {
                mostSignificantErrorForOffset = sl;
                mostSignificantFeatureForOffset = feature;
            }
        }
        if (mostSignificantErrorForOffset > mostSignificantError) {
            mostSignificantError = mostSignificantErrorForOffset;
            mostSignificantFeature = mostSignificantFeatureForOffset;
            mostSignificantOffset = offset;
        }
    }

    return mostSignificantError > 0
               ? std::make_pair(mostSignificantFeature, mostSignificantOffset)
               : TOptionalFeatureTimePr();
}

std::uint64_t CCalendarCyclicTest::checksum(std::uint64_t seed) const {
    seed = common::CChecksum::calculate(seed, m_DecayRate);
    seed = common::CChecksum::calculate(seed, m_ErrorQuantiles);
    seed = common::CChecksum::calculate(seed, m_CurrentBucketTime);
    seed = common::CChecksum::calculate(seed, m_CurrentBucketIndex);
    seed = common::CChecksum::calculate(seed, m_CurrentBucketErrorStats);
    TErrorStatsVec errors{this->inflate()};
    return common::CChecksum::calculate(seed, errors);
}

void CCalendarCyclicTest::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CCalendarCyclicTest");
    core::CMemoryDebug::dynamicSize("m_ErrorQuantiles", m_ErrorQuantiles, mem);
    core::CMemoryDebug::dynamicSize("m_CompressedBucketErrorStats",
                                    m_CompressedBucketErrorStats, mem);
}

std::size_t CCalendarCyclicTest::memoryUsage() const {
    std::size_t mem{core::CMemory::dynamicSize(m_ErrorQuantiles)};
    mem += core::CMemory::dynamicSize(m_CompressedBucketErrorStats);
    return mem;
}

double CCalendarCyclicTest::winsorise(double error) const {
    double high;
    m_ErrorQuantiles.quantile(99.5, high);
    return std::min(error, high);
}

double CCalendarCyclicTest::survivalFunction(double error) const {
    using TMomentsAccumulator = common::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

    // We use an approximation for the right tail of a KDE from the error
    // percentiles to estimate the survival function.
    TMomentsAccumulator tailMoments;
    for (double i = 0.0; i < 5.0; i += 1.0) {
        double eq;
        m_ErrorQuantiles.quantile(
            LARGE_ERROR_PERCENTILE + i * (100.0 - LARGE_ERROR_PERCENTILE) / 5.0, eq);
        tailMoments.add(eq);
    }
    try {
        boost::math::normal normal{common::CBasicStatistics::mean(tailMoments),
                                   std::sqrt(common::CBasicStatistics::variance(tailMoments))};
        return (100.0 - LARGE_ERROR_PERCENTILE) / 100.0 *
               common::CTools::safeCdfComplement(normal, error);

    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to compute tail distribution '" << e.what() << "'");
    }
    return 1.0;
}

double CCalendarCyclicTest::significance(double n, double nl, double nv) const {
    if (n > 0.0) {
        try {
            // We have roughly 31 independent error samples, one for each
            // day of the month, so the chance of seeing as extreme an event
            // among all of them is:
            //   1 - P("don't see as extreme event") = 1 - (1 - P("event"))^31
            boost::math::binomial bl{n, 1.0 - LARGE_ERROR_PERCENTILE / 100.0};
            boost::math::binomial bv{n, 1.0 - VERY_LARGE_ERROR_PERCENTILE / 100.0};
            double p{std::min({2.0 * common::CTools::safeCdfComplement(bl, nl - 1.0),
                               2.0 * common::CTools::safeCdfComplement(bv, nv - 1.0), 1.0})};
            return common::CTools::oneMinusPowOneMinusX(p, 31.0);
        } catch (const std::exception& e) {
            LOG_ERROR(<< "Failed to calculate significance: " << e.what()
                      << " n = " << n << " nl = " << nl << " nv = " << nv);
        }
    }
    return 1.0;
}

void CCalendarCyclicTest::deflate(const TErrorStatsVec& stats) {
    bool lengthOnly{false};
    core::CDeflator deflator{lengthOnly};
    deflator.addVector(stats);
    deflator.finishAndTakeData(m_CompressedBucketErrorStats);
    m_CompressedBucketErrorStats.shrink_to_fit();
}

CCalendarCyclicTest::TErrorStatsVec CCalendarCyclicTest::inflate() const {
    bool lengthOnly{false};
    core::CInflator inflator{lengthOnly};
    inflator.addVector(m_CompressedBucketErrorStats);
    TByteVec decompressed;
    inflator.finishAndTakeData(decompressed);
    TErrorStatsVec result(decompressed.size() / sizeof(SErrorStats));
    std::copy(decompressed.begin(), decompressed.end(),
              reinterpret_cast<TByte*>(result.data()));
    return result;
}

std::uint64_t CCalendarCyclicTest::SErrorStats::checksum() const {
    std::uint64_t seed{static_cast<std::uint64_t>(s_Count)};
    seed = common::CChecksum::calculate(seed, s_LargeErrorCount);
    return common::CChecksum::calculate(seed, s_LargeErrorSum);
}

std::string CCalendarCyclicTest::SErrorStats::toDelimited() const {
    return core::CStringUtils::typeToString(s_Count) + DELIMITER +
           core::CStringUtils::typeToString(s_LargeErrorCount) + DELIMITER +
           s_LargeErrorSum.toString();
}

bool CCalendarCyclicTest::SErrorStats::fromDelimited(const std::string& str_) {
    std::string str{str_};
    std::size_t delimiter{str.find(DELIMITER)};
    if (core::CStringUtils::stringToType(str.substr(0, delimiter), s_Count) == false) {
        LOG_ERROR(<< "Failed to parse '" << str_ << "'");
        return false;
    }
    str = str.substr(delimiter + 1);
    delimiter = str.find(DELIMITER);
    if (core::CStringUtils::stringToType(str.substr(0, delimiter), s_LargeErrorCount) == false) {
        LOG_ERROR(<< "Failed to parse '" << str_ << "'");
        return false;
    }
    str = str.substr(delimiter + 1);
    if (s_LargeErrorSum.fromString(str) == false) {
        LOG_ERROR(<< "Failed to parse '" << str_ << "'");
        return false;
    }
    return true;
}
}
}
}
