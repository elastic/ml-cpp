/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CCalendarCyclicTest_h
#define INCLUDED_ml_maths_CCalendarCyclicTest_h

#include <core/CoreTypes.h>

#include <maths/CCalendarFeature.h>
#include <maths/CQuantileSketch.h>
#include <maths/ImportExport.h>

#include <boost/optional.hpp>

#include <cstdint>
#include <vector>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {

//! \brief The basic idea of this test is to see if there is stronger
//! than expected temporal correlation between large prediction errors
//! and calendar features.
//!
//! DESCRIPTION:\n
//! This maintains prediction error statistics for a collection of
//! calendar features. These are things like "day of month",
//! ("day of week", "week month") pairs and so on. The test checks to
//! see if the number of large prediction errors is statistically high,
//! i.e. are there many more errors exceeding a specified percentile
//! than one would expect given that this is expected to be binomial.
//! Amongst features with statistically significant frequencies of large
//! errors it returns the feature with the highest mean prediction error.
class MATHS_EXPORT CCalendarCyclicTest {
public:
    using TOptionalFeature = boost::optional<CCalendarFeature>;

public:
    explicit CCalendarCyclicTest(double decayRate = 0.0);

    //! Initialize by reading state from \p traverser.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Persist state by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Age the bucket values to account for \p time elapsed time.
    void propagateForwardsByTime(double time);

    //! Add \p error at \p time.
    void add(core_t::TTime time, double error, double weight = 1.0);

    //! Check if there are calendar components.
    TOptionalFeature test() const;

    //! Get a checksum for this object.
    std::uint64_t checksum(std::uint64_t seed = 0) const;

    //! Debug the memory used by this object.
    void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

    //! Get the memory used by this object.
    std::size_t memoryUsage() const;

private:
    using TTimeVec = std::vector<core_t::TTime>;
    using TByte = unsigned char;
    using TByteVec = std::vector<TByte>;

    //! \brief Records the daily error statistics.
    struct MATHS_EXPORT SErrorStats {
        //! Get a checksum for this object.
        std::uint64_t checksum() const;
        //! Convert to a delimited string.
        std::string toDelimited() const;
        //! Initialize from a delimited string.
        bool fromDelimited(const std::string& str);

        std::uint32_t s_Count = 0;
        std::uint32_t s_LargeErrorCount = 0;
        CFloatStorage s_LargeErrorSum = 0.0;
    };
    using TErrorStatsVec = std::vector<SErrorStats>;

private:
    //! Winsorise \p error.
    double winsorise(double error) const;

    //! Get the significance of \p x large errors given \p n samples.
    double significance(double n, double x) const;

    //! Convert to a compressed representation.
    void deflate(const TErrorStatsVec& stats);

    //! Extract from the compressed representation.
    TErrorStatsVec inflate() const;

private:
    //! The rate at which the error counts are aged.
    double m_DecayRate;

    //! Used to estimate large error thresholds.
    CQuantileSketch m_ErrorQuantiles;

    //! The start time of the bucket to which the last error
    //! was added.
    core_t::TTime m_CurrentBucketTime;

    //! The start time of the earliest bucket for which we have
    //! error statistics.
    core_t::TTime m_CurrentBucketIndex;

    //! The bucket statistics currently being updated.
    SErrorStats m_CurrentBucketErrorStats;

    //! The compressed error statistics.
    TByteVec m_CompressedBucketErrorStats;
};
}
}

#endif // INCLUDED_ml_maths_CCalendarCyclicTest_h
