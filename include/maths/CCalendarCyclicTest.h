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

#include <boost/circular_buffer.hpp>
#include <boost/container/flat_map.hpp>
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
    using TUInt32CBuf = boost::circular_buffer<std::uint32_t>;
    using TTimeFloatPr = std::pair<core_t::TTime, CFloatStorage>;
    using TTimeFloatFMap = boost::container::flat_map<core_t::TTime, CFloatStorage>;

private:
    //! Winsorise \p error.
    double winsorise(double error) const;

    //! Get the significance of \p x large errors given \p n samples.
    double significance(double n, double x) const;

private:
    //! The error bucketing interval.
    static const core_t::TTime BUCKET;
    //! The window length in buckets.
    static const core_t::TTime WINDOW;
    //! The percentile of a large error.
    static const double LARGE_ERROR_PERCENTILE;
    //! The minimum number of repeats for a testable feature.
    static const unsigned int MINIMUM_REPEATS;
    //! The bits used to count added values.
    static const std::uint32_t COUNT_BITS;
    //! The offsets that are used for different timezones.
    static const TTimeVec TIMEZONE_OFFSETS;

private:
    //! The rate at which the error counts are aged.
    double m_DecayRate;

    //! The time of the last error added.
    core_t::TTime m_Bucket;

    //! Used to estimate large error thresholds.
    CQuantileSketch m_ErrorQuantiles;

    //! The counts of errors and large errors in a sliding window.
    TUInt32CBuf m_ErrorCounts;

    //! The bucket large error sums.
    TTimeFloatFMap m_ErrorSums;
};
}
}

#endif // INCLUDED_ml_maths_CCalendarCyclicTest_h
