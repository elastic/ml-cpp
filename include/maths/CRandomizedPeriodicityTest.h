/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CRandomizedPeriodicityTest_h
#define INCLUDED_ml_maths_CRandomizedPeriodicityTest_h

#include <core/CMutex.h>
#include <core/CoreTypes.h>

#include <maths/CBasicStatistics.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CPRNG.h>
#include <maths/ImportExport.h>

#include <boost/random/mersenne_twister.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace CRandomizedPeriodicityTestTest {
struct testPersist;
}

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
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
    std::uint64_t checksum(std::uint64_t seed = 0) const;

private:
    using TDoubleVec = std::vector<double>;
    using TVector2 = CVectorNx1<CFloatStorage, 2>;
    using TVector2MeanAccumulator = CBasicStatistics::SSampleMean<TVector2>::TAccumulator;
    using TVector2N = CVectorNx1<CFloatStorage, 2 * N>;
    using TVector2NMeanAccumulator = CBasicStatistics::SSampleMean<TVector2N>::TAccumulator;
    using TAtomicTime = std::atomic<core_t::TTime>;

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
    static void updateStatistics(TVector2NMeanAccumulator& projections,
                                 TVector2MeanAccumulator& statistics);

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

    friend struct CRandomizedPeriodicityTestTest::testPersist;
};
}
}

#endif // INCLUDED_ml_maths_CRandomizedPeriodicityTest_h
