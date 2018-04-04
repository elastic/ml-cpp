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

#ifndef INCLUDED_ml_maths_CTrendTests_h
#define INCLUDED_ml_maths_CTrendTests_h

#include <core/AtomicTypes.h>
#include <core/CMutex.h>
#include <core/CVectorRange.h>
#include <core/CoreTypes.h>

#include <maths/CBasicStatistics.h>
#include <maths/CCalendarFeature.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CPRNG.h>
#include <maths/CQuantileSketch.h>
#include <maths/CRegression.h>
#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <cstddef>
#include <vector>

#include <boost/circular_buffer.hpp>
#include <boost/container/flat_map.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/shared_ptr.hpp>

#include <stdint.h>

class CTrendTestsTest;

namespace ml {
namespace maths {
class CSeasonalTime;

//! \brief A low memory footprint randomized test for probability.
//!
//! This is based on the idea of random projection in a Hilbert
//! space.
//!
//! If we choose a direction uniformly at random, \f$r\f$,
//! verses in the subspace of periodic functions, \f$p\$, we
//! expect, with probability 1, that the inner product with
//! a periodic function with matched period, \$f\f$, will be
//! larger. In particular, if we imagine taking the expectation
//! over a each family of functions then with probability 1:
//! <pre class="fragment">
//!     \f$E\left[\frac{\|r^t f\|}{\|r\|}\right] < E\left[\frac{\|p^t f\|}{\|p\|}\right]\f$
//! </pre>
//!
//! Therefore, if we sample independently many such random
//! projections of the function and the function is periodic
//! then we can test the for periodicity by comparing means.
//! The variance of the means will tend to zero as the number
//! of samples grows so the significance for rejecting the
//! null hypothesis (that the function is a-periodic) will
//! shrink to zero.
class MATHS_EXPORT CRandomizedPeriodicityTest {
public:
    //! The size of the projection sample coefficients
    static const std::size_t N = 5;

public:
    CRandomizedPeriodicityTest();

    //! \name Persistence
    //@{
    //! Restore the static members by reading state from \p traverser.
    static bool staticsAcceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Persist the static members by passing information to \p inserter.
    static void staticsAcceptPersistInserter(core::CStatePersistInserter& inserter);

    //! Initialize by reading state from \p traverser.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Persist state by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;
    //@}

    //! Add a new value \p value at \p time.
    void add(core_t::TTime time, double value);

    //! Test whether there is a periodic trend.
    bool test() const;

    //! Reset the test static random vectors.
    //!
    //! \note For unit testing only.
    static void reset();

    //! Get a checksum for this object.
    uint64_t checksum(uint64_t seed = 0) const;

private:
    using TDoubleVec = std::vector<double>;
    using TVector2 = CVectorNx1<CFloatStorage, 2>;
    using TVector2MeanAccumulator = CBasicStatistics::SSampleMean<TVector2>::TAccumulator;
    using TVector2N = CVectorNx1<CFloatStorage, 2 * N>;
    using TVector2NMeanAccumulator = CBasicStatistics::SSampleMean<TVector2N>::TAccumulator;
    using TAtomicTime = atomic_t::atomic<core_t::TTime>;

private:
    //! The length over which the periodic random projection decoheres.
    static const core_t::TTime SAMPLE_INTERVAL;
    //! The time between day resample events.
    static const core_t::TTime DAY_RESAMPLE_INTERVAL;
    //! The time between week resample events.
    static const core_t::TTime WEEK_RESAMPLE_INTERVAL;
    //! The random number generator.
    static boost::random::mt19937_64 ms_Rng;
    //! The permutations daily projections.
    static TDoubleVec ms_DayRandomProjections[N];
    //! The daily periodic projections.
    static TDoubleVec ms_DayPeriodicProjections[N];
    //! The time at which we re-sampled day projections.
    static TAtomicTime ms_DayResampled;
    //! The permutations weekly projections.
    static TDoubleVec ms_WeekRandomProjections[N];
    //! The weekly periodic projections.
    static TDoubleVec ms_WeekPeriodicProjections[N];
    //! The time at which we re-sampled week projections.
    static TAtomicTime ms_WeekResampled;
    //! The mutex for protecting state update.
    static core::CMutex ms_Lock;

private:
    //! Refresh \p projections and update \p statistics.
    static void updateStatistics(TVector2NMeanAccumulator& projections, TVector2MeanAccumulator& statistics);

    //! Re-sample the projections.
    static void resample(core_t::TTime time);

    //! Re-sample the specified projections.
    static void resample(core_t::TTime period,
                         core_t::TTime resampleInterval,
                         TDoubleVec (&periodicProjections)[N],
                         TDoubleVec (&randomProjections)[N]);

private:
    //! The day projections.
    TVector2NMeanAccumulator m_DayProjections;
    //! The sample mean of the square day projections.
    TVector2MeanAccumulator m_DayStatistics;
    //! The last time the day projections were updated.
    core_t::TTime m_DayRefreshedProjections;
    //! The week projections.
    TVector2NMeanAccumulator m_WeekProjections;
    //! The sample mean of the square week projections.
    TVector2MeanAccumulator m_WeekStatistics;
    //! The last time the day projections were updated.
    core_t::TTime m_WeekRefreshedProjections;

    friend class ::CTrendTestsTest;
};

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
    uint64_t checksum(uint64_t seed = 0) const;

    //! Debug the memory used by this object.
    void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

    //! Get the memory used by this object.
    std::size_t memoryUsage() const;

private:
    using TTimeVec = std::vector<core_t::TTime>;
    using TUInt32CBuf = boost::circular_buffer<uint32_t>;
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
    static const uint32_t COUNT_BITS;
    //! The offsets that are used for different timezone offsets.
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

#endif // INCLUDED_ml_maths_CTrendTests_h
