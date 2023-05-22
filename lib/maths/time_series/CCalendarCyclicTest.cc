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

#include "maths/common/COrderings.h"
#include <maths/time_series/CCalendarCyclicTest.h>

#include <core/CHashing.h>
#include <core/CLogger.h>
#include <core/CMemoryDef.h>
#include <core/CPersistUtils.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CTimezone.h>
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
#include <iterator>
#include <limits>
#include <mutex>
#include <string>

namespace ml {
namespace maths {
namespace time_series {
namespace {
using TTimeVec = std::vector<core_t::TTime>;

//! \brief Hashes a calendar feature.
struct SHashAndOffsetFeature {
    std::size_t operator()(const std::pair<CCalendarFeature, core_t::TTime>& featureAndOffset) const {
        return core::CHashing::hashCombine(
            featureAndOffset.first.checksum(0),
            static_cast<std::uint64_t>(featureAndOffset.second));
    }
};

double binomialPValueAdj(double n, double p, double np) {
    try {
        boost::math::binomial binom{n, p};
        p = std::min(2.0 * common::CTools::safeCdfComplement(binom, np - 1.0), 1.0);
        // We have roughly 31 independent error samples, one for each
        // day of the month, so the chance of seeing as extreme an event
        // among all of them is:
        //   1 - P("don't see as extreme event") = 1 - (1 - P("event"))^31
        return common::CTools::oneMinusPowOneMinusX(p, 31.0);
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to calculate significance: " << e.what()
                  << " n = " << n << " p = " << p << " np = " << np);
    }
    return 1.0;
}

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
const core_t::TTime LONG_BUCKET{4 * core::constants::HOUR};
// We can't determine the exact time zone given the state we maintain but we can
// determine if the whole pattern is shifted 1 day late or early.
const TTimeVec TIME_ZONE_OFFSETS{0, -12 * core::constants::HOUR - 1,
                                 12 * core::constants::HOUR + 1};

//! The percentile of a large error.
constexpr double LARGE_ERROR_PERCENTILE{98.5};
//! The percentile of a very large error.
constexpr double VERY_LARGE_ERROR_PERCENTILE{99.99};
//! The minimum number of repeats to test a feature.
constexpr unsigned int MINIMUM_REPEATS{4};
//! The maximum significance to accept a feature.
constexpr double MAXIMUM_P_VALUE{0.01};
//! Used to guard setting the timezone.
std::once_flag setTimeZone;
}

CCalendarCyclicTest::CCalendarCyclicTest(core_t::TTime bucketLength, double decayRate)
    : m_DecayRate{decayRate}, m_BucketLength{bucketLength}, m_ErrorQuantiles{20} {
    std::call_once(setTimeZone,
                   [] { core::CTimezone::instance().timezoneName("GMT"); });
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

void CCalendarCyclicTest::forgetErrorDistribution() {
    m_ErrorQuantiles = common::CQuantileSketch{20};
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

    if (m_ErrorQuantiles.count() > this->sufficientCountToMeasureLargeErrors()) {
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
        m_ErrorQuantiles.quantile(this->largeErrorPercentile(), largeError);
        if (error >= largeError) {
            bool isVeryLarge{100.0 * (1.0 - this->survivalFunction(error)) >=
                             this->veryLargeErrorPercentile()};
            m_CurrentBucketErrorStats.s_LargeErrorCount += std::min(
                std::numeric_limits<std::uint32_t>::max() -
                    m_CurrentBucketErrorStats.s_LargeErrorCount,
                static_cast<std::uint32_t>(isVeryLarge ? (1 << 17) + 1 : 1));
            m_CurrentBucketErrorStats.s_LargeErrorSum += this->winsorise(error);
        }
    }
}

CCalendarCyclicTest::TFeatureTimePrVec CCalendarCyclicTest::test() const {

    if (m_ErrorQuantiles.count() == 0) {
        return {};
    }

    // The statistics we need in order to be able to test for calendar features.
    using TMaxDoubleAccumulator =
        common::CBasicStatistics::SMax<double, MINIMUM_REPEATS>::TAccumulator;
    struct SStats {
        unsigned int s_Repeats{0};
        unsigned int s_RepeatsWithLargeErrors{0};
        double s_ErrorSum{0.0};
        double s_ErrorCount{0.0};
        double s_LargeErrorCount{0.0};
        double s_VeryLargeErrorCount{0.0};
        double s_PValue{0.0};
        TMaxDoubleAccumulator s_Cdf;
    };
    using TFeatureStatsUMap =
        boost::unordered_map<std::pair<CCalendarFeature, core_t::TTime>, SStats, SHashAndOffsetFeature>;

    TErrorStatsVec errors{this->inflate()};

    struct SFeatureAndError {
        bool operator<(const SFeatureAndError& rhs) const {
            return common::COrderings::lexicographicalCompare(
                -s_ErrorSum, std::fabs(s_Offset), -rhs.s_ErrorSum, std::fabs(rhs.s_Offset));
        }
        CCalendarFeature s_Feature;
        core_t::TTime s_Offset;
        double s_ErrorSum;
    };
    using TMaxErrorFeatureAccumulator =
        common::CBasicStatistics::COrderStatisticsStack<SFeatureAndError, 3>;

    TMaxErrorFeatureAccumulator largestErrorFeatures;

    double errorThreshold;
    m_ErrorQuantiles.quantile(50.0, errorThreshold);
    errorThreshold *= 2.0;

    TFeatureStatsUMap stats{TIME_ZONE_OFFSETS.size() * errors.size()};

    // Note that the current index points to the next bucket to overwrite,
    // i.e. the earliest bucket error statistics we have. The start of
    // this bucket is WINDOW before the start time of the current partial
    // bucket.
    for (core_t::TTime i = m_CurrentBucketIndex, time = m_CurrentBucketTime - WINDOW;
         time < m_CurrentBucketTime; i = (i + 1) % SIZE, time += BUCKET) {
        if (errors[i].s_Count > 0) {
            double n{static_cast<double>(errors[i].s_Count)};
            double nl{static_cast<double>(errors[i].s_LargeErrorCount % (1 << 17))};
            double nv{static_cast<double>(errors[i].s_LargeErrorCount / (1 << 17))};
            double pValue{this->errorsPValue(n, nl, nv)};
            // It is that the maximum value is at least as large as the mean
            // We use this to compute a lower bound for the chance of seeing
            // a larger error on the interval.
            double cdf;
            m_ErrorQuantiles.cdf(errors[i].s_LargeErrorSum / n, cdf);

            for (auto offset : TIME_ZONE_OFFSETS) {
                core_t::TTime midpoint{time + BUCKET / 2 + offset};
                for (auto feature : CCalendarFeature::features(midpoint)) {
                    if (feature.testForTimeZoneOffset(offset)) {
                        SStats& stat = stats[std::make_pair(feature, offset)];
                        stat.s_Repeats += 1;
                        stat.s_RepeatsWithLargeErrors += nl > 0 ? 1 : 0;
                        stat.s_ErrorSum += errors[i].s_LargeErrorSum;
                        stat.s_ErrorCount += n;
                        stat.s_LargeErrorCount += nl;
                        stat.s_VeryLargeErrorCount += nv;
                        stat.s_PValue = std::max(stat.s_PValue, pValue);
                        stat.s_Cdf.add(cdf);
                    }
                }
            }
        }
    }

    for (const auto& stat : stats) {
        auto[feature, offset] = stat.first;
        unsigned int repeats{stat.second.s_Repeats};
        unsigned int repeatsWithLargeErrors{stat.second.s_RepeatsWithLargeErrors};
        double n{stat.second.s_ErrorCount};
        double nl{stat.second.s_LargeErrorCount};
        double nv{stat.second.s_VeryLargeErrorCount};
        double sl{stat.second.s_ErrorSum};
        double maxBucketPValue{stat.second.s_PValue};
        double cdf{stat.second.s_Cdf.biggest()};

        if (repeats < MINIMUM_REPEATS || // Insufficient repeats
            sl <= errorThreshold * nl) { // Error too small to bother modelling
            continue;
        }
        if (std::pow(maxBucketPValue, repeats) < MAXIMUM_P_VALUE) { // High significance for each repeat
            largestErrorFeatures.add(SFeatureAndError{feature, offset, sl});
            continue;
        }
        if (m_BucketLength < LONG_BUCKET || // Short raw data buckets
            repeatsWithLargeErrors < MINIMUM_REPEATS) { // Too few repeats with large errors
            continue;
        }
        double windowPValue{std::min(this->errorsPValue(n, nl, nv),
                                     binomialPValueAdj(n, 1.0 - cdf, repeats))};
        if (windowPValue < MAXIMUM_P_VALUE) { // High joint significance for all repeats
            largestErrorFeatures.add(SFeatureAndError{feature, offset, sl});
        }
    }

    TFeatureTimePrVec result;
    largestErrorFeatures.sort();
    std::transform(largestErrorFeatures.begin(), largestErrorFeatures.end(),
                   std::back_inserter(result), [](const auto& featureAndError) {
                       return std::make_pair(featureAndError.s_Feature,
                                             featureAndError.s_Offset);
                   });
    if (result.size() > 1) {
        auto offset = result[0].second;
        result.erase(std::remove_if(result.begin(), result.end(),
                                    [&](const auto& featureAndOffset) {
                                        return featureAndOffset.second != offset;
                                    }),
                     result.end());
    }
    return result;
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
    core::memory_debug::dynamicSize("m_ErrorQuantiles", m_ErrorQuantiles, mem);
    core::memory_debug::dynamicSize("m_CompressedBucketErrorStats",
                                    m_CompressedBucketErrorStats, mem);
}

std::size_t CCalendarCyclicTest::memoryUsage() const {
    std::size_t mem{core::memory::dynamicSize(m_ErrorQuantiles)};
    mem += core::memory::dynamicSize(m_CompressedBucketErrorStats);
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
        m_ErrorQuantiles.quantile(this->largeErrorPercentile() +
                                      i * (100.0 - this->largeErrorPercentile()) / 5.0,
                                  eq);
        tailMoments.add(eq);
    }

    if (common::CBasicStatistics::variance(tailMoments) == 0.0) {
        return error > common::CBasicStatistics::mean(tailMoments) ? 0.0 : 1.0;
    }
    try {
        boost::math::normal normal{common::CBasicStatistics::mean(tailMoments),
                                   std::sqrt(common::CBasicStatistics::variance(tailMoments))};
        return (100.0 - this->largeErrorPercentile()) / 100.0 *
               common::CTools::safeCdfComplement(normal, error);

    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to compute tail distribution '" << e.what() << "'");
    }
    return 1.0;
}

double CCalendarCyclicTest::errorsPValue(double n, double nl, double nv) const {
    return n > 0.0
               ? std::min(
                     binomialPValueAdj(n, 1.0 - this->largeErrorPercentile() / 100.0, nl),
                     binomialPValueAdj(n, 1.0 - this->veryLargeErrorPercentile() / 100.0, nv))
               : 1.0;
}

double CCalendarCyclicTest::sufficientCountToMeasureLargeErrors() const {
    // Cap the how long we'll wait identify large errors.
    return std::min(static_cast<double>(10 * core::constants::DAY) / m_BucketLength, 100.0);
}

double CCalendarCyclicTest::largeErrorPercentile() const {
    // For long bucket lengths we see very few high percentile errors
    // in the full window. In this case the power of the tests we use
    // drop greatly. So we adjust the threshold to hold the expected
    // count of large errors fixed.
    return 100.0 - (100.0 - LARGE_ERROR_PERCENTILE) *
                       static_cast<double>(std::max(LONG_BUCKET, m_BucketLength)) /
                       static_cast<double>(LONG_BUCKET);
}

double CCalendarCyclicTest::veryLargeErrorPercentile() const {
    return 100.0 - (100.0 - VERY_LARGE_ERROR_PERCENTILE) *
                       static_cast<double>(std::max(LONG_BUCKET, m_BucketLength)) /
                       static_cast<double>(LONG_BUCKET);
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
