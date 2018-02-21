/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CTrendTests_h
#define INCLUDED_ml_maths_CTrendTests_h

#include <core/CoreTypes.h>
#include <core/CMutex.h>
#include <core/CVectorRange.h>

#include <maths/CBasicStatistics.h>
#include <maths/CCalendarFeature.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CPRNG.h>
#include <maths/CQuantileSketch.h>
#include <maths/CRegression.h>
#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <boost/circular_buffer.hpp>
#include <boost/container/flat_map.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/shared_ptr.hpp>

#include <atomic>
#include <cstddef>
#include <vector>

#include <stdint.h>

class CTrendTestsTest;

namespace ml
{
namespace maths
{
class CSeasonalTime;

//! \brief Implements a simple test for whether a random
//! process has a trend.
//!
//! DESCRIPTION:\n
//! A process with a trend is defined as one which can be
//! modeled as follows:
//! <pre class="fragment">
//!   \f$Y_i = f(t_i) + X_i\f$
//! </pre>
//! Here, \f$f(.)\f$ is some smoothly varying function and
//! the \f$X_i = X\f$ are IID. The approach we take is to
//! perform a variance ratio test on \f${ Y_i - f(t_i) }\f$
//! verses \f${ Y_i }\f$. We are interested in the case that
//! modeling f(.), using an exponentially decaying cubic
//! regression with the current decay rate, will materially
//! affect our results. We therefore test to see if the
//! reduction in variance, as a proxy for the full model
//! confidence bounds, is both large enough and statistically
//! significant.
class MATHS_EXPORT CTrendTest
{
    public:
        //! The order of the trend regression.
        static const std::size_t ORDER = 3u;

    public:
        using TRegression = CRegression::CLeastSquaresOnline<ORDER>;

    public:
        explicit CTrendTest(double decayRate = 0.0);

        //! Initialize by reading state from \p traverser.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

        //! Persist state by passing information to \p inserter.
        void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

        //! Set the decay rate.
        void decayRate(double decayRate);

        //! Age the state to account for \p time elapsed time.
        void propagateForwardsByTime(double time);

        //! Add a new value \p value at \p time.
        void add(core_t::TTime time, double value, double weight = 1.0);

        //! Capture the variance in the prediction error.
        void captureVariance(core_t::TTime time, double value, double weight = 1.0);

        //! Translate the trend by \p shift.
        void shift(double shift);

        //! Test whether there is a trend.
        bool test(void) const;

        //! Get the regression model of the trend.
        const TRegression &trend(void) const;

        //! Get the origin of the time coordinate system.
        core_t::TTime origin(void) const;

        //! Get the variance after removing the trend.
        double variance(void) const;

        //! Get a checksum for this object.
        uint64_t checksum(uint64_t seed = 0) const;

    private:
        //! The smallest decrease in variance after removing the trend
        //! which is consider significant.
        static const double MAXIMUM_TREND_VARIANCE_RATIO;

    private:
        using TVector = CVectorNx1<double, 2>;
        using TVectorMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<TVector>::TAccumulator;

    private:
        //! Get the time at which to evaluate the regression model
        //! of the trend.
        double time(core_t::TTime time) const;

    private:
        //! The rate at which the regression model is aged.
        double m_DecayRate;

        //! The origin of the time coordinate system.
        core_t::TTime m_TimeOrigin;

        //! The current regression model.
        TRegression m_Trend;

        //! The values' mean and variance.
        TVectorMeanVarAccumulator m_Variances;
};

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
class MATHS_EXPORT CRandomizedPeriodicityTest
{
    public:
        //! The size of the projection sample coefficients
        static const std::size_t N = 5;

    public:
        CRandomizedPeriodicityTest(void);

        //! \name Persistence
        //@{
        //! Restore the static members by reading state from \p traverser.
        static bool staticsAcceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

        //! Persist the static members by passing information to \p inserter.
        static void staticsAcceptPersistInserter(core::CStatePersistInserter &inserter);

        //! Initialize by reading state from \p traverser.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

        //! Persist state by passing information to \p inserter.
        void acceptPersistInserter(core::CStatePersistInserter &inserter) const;
        //@}

        //! Add a new value \p value at \p time.
        void add(core_t::TTime time, double value);

        //! Test whether there is a periodic trend.
        bool test(void) const;

        //! Reset the test static random vectors.
        //!
        //! \note For unit testing only.
        static void reset(void);

        //! Get a checksum for this object.
        uint64_t checksum(uint64_t seed = 0) const;

    private:
        using TDoubleVec = std::vector<double>;
        using TVector2 = CVectorNx1<CFloatStorage, 2>;
        using TVector2MeanAccumulator = CBasicStatistics::SSampleMean<TVector2>::TAccumulator;
        using TVector2N = CVectorNx1<CFloatStorage, 2*N>;
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
        static void updateStatistics(TVector2NMeanAccumulator &projections,
                                     TVector2MeanAccumulator &statistics);

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

//! \brief Represents the result of running a periodicity test.
class MATHS_EXPORT CPeriodicityTestResult : boost::equality_comparable<CPeriodicityTestResult,
                                            boost::addable<CPeriodicityTestResult> >
{
    public:
        using TTimeTimePr = std::pair<core_t::TTime, core_t::TTime>;

    public:
        //! \brief Component data.
        struct MATHS_EXPORT SComponent
        {
            SComponent(void);
            SComponent(unsigned int id,
                       core_t::TTime startOfPartition,
                       core_t::TTime period,
                       const TTimeTimePr &window);

            //! Initialize from a string created by persist.
            static bool fromString(const std::string &value, SComponent &result);
            //! Convert to a string.
            static std::string toString(const SComponent &component);

            //! Check if this is equal to \p other.
            bool operator==(const SComponent &other) const;

            //! Get a checksum of this object.
            uint64_t checksum(void) const;

            //! An identifier for the component used by the test.
            unsigned int  s_Id;
            //! The start of the partition.
            core_t::TTime s_StartOfPartition;
            //! The period of the component.
            core_t::TTime s_Period;
            //! The component window.
            TTimeTimePr   s_Window;
        };

        using TComponent4Vec = core::CSmallVector<SComponent, 4>;

    public:
        //! Initialize by reading state from \p traverser.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

        //! Persist state by passing information to \p inserter.
        void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

        //! Check if this is equal to \p other.
        bool operator==(const CPeriodicityTestResult &other) const;

        //! Sets to the union of the periodic components present.
        //!
        //! \warning This only makes sense if the this and the other result
        //! share the start of the partition time.
        const CPeriodicityTestResult &operator+=(const CPeriodicityTestResult &other);

        //! Add a component if and only if \p hasPeriod is true.
        void add(unsigned int id,
                 core_t::TTime startOfWeek,
                 core_t::TTime period,
                 const TTimeTimePr &window);

        //! Check if there are any periodic components.
        bool periodic(void) const;

        //! Get the binary representation of the periodic components.
        const TComponent4Vec &components(void) const;

        //! Get a checksum for this object.
        uint64_t checksum(void) const;

    private:
        //! The periodic components.
        TComponent4Vec m_Components;
};

//! \brief Implements shared functionality for our periodicity tests.
class MATHS_EXPORT CPeriodicityTest
{
    public:
        using TDouble2Vec = core::CSmallVector<double, 2>;
        using TTimeTimePr = std::pair<core_t::TTime, core_t::TTime>;
        using TTimeTimePr2Vec = core::CSmallVector<TTimeTimePr, 2>;
        using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
        using TMeanVarAccumulatorVec = std::vector<TMeanVarAccumulator>;
        using TTimeTimePrMeanVarAccumulatorPr = std::pair<TTimeTimePr, TMeanVarAccumulator>;
        using TTimeTimePrMeanVarAccumulatorPrVec = std::vector<TTimeTimePrMeanVarAccumulatorPr>;
        using TTimeTimePrMeanVarAccumulatorPrVecVec = std::vector<TTimeTimePrMeanVarAccumulatorPrVec>;
        using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
        using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
        using TTime2Vec = core::CSmallVector<core_t::TTime, 2>;
        using TComponent = CPeriodicityTestResult::SComponent;

    public:
        CPeriodicityTest(void);
        explicit CPeriodicityTest(double decayRate);
        virtual ~CPeriodicityTest(void) = default;

        //! Check if the test is initialized.
        bool initialized(void) const;

        //! Age the bucket values to account for \p time elapsed time.
        void propagateForwardsByTime(double time);

        //! Add \p value at \p time.
        void add(core_t::TTime time, double value, double weight = 1.0);

        //! Get the periods being tested.
        virtual TTime2Vec periods(void) const = 0;

        //! Get a seasonal time for the specified results.
        //!
        //! \warning The caller owns the returned object.
        virtual CSeasonalTime *seasonalTime(const TComponent &component) const = 0;

        //! Get the periodic trends corresponding to \p periods.
        virtual void trends(const CPeriodicityTestResult &periods,
                            TTimeTimePrMeanVarAccumulatorPrVecVec &result) const = 0;

        //! Get the fraction of populated test slots
        double populatedRatio(void) const;

        //! Check we've seen sufficient data to test accurately.
        bool seenSufficientData(void) const;

        //! Debug the memory used by this object.
        virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the memory used by this object.
        virtual std::size_t memoryUsage(void) const;

        //! Get the static size of this object.
        virtual std::size_t staticSize(void) const;

        //! Print a description of \p result.
        virtual std::string print(const CPeriodicityTestResult &result) const = 0;

    protected:
        using TDoubleVec = std::vector<double>;
        using TDoubleVec2Vec = core::CSmallVector<TDoubleVec, 2>;

        //! \brief A collection of statistics used during testing.
        struct MATHS_EXPORT STestStats
        {
            STestStats(double vt, double at, double Rt);
            //! The maximum variance to accept the alternative hypothesis.
            double s_Vt;
            //! The minimum amplitude to accept the alternative hypothesis.
            double s_At;
            //! The minimum autocorrelation to accept the alternative
            //! hypothesis.
            double s_Rt;
            //! The data range.
            double s_Range;
            //! The number of buckets with at least one measurement.
            double s_B;
            //! The average number of measurements per bucket value.
            double s_M;
            //! The variance estimate of H0.
            double s_V0;
            //! The degrees of freedom in the variance estimate of H0.
            double s_DF0;
            //! The trend for the null hypothesis.
            TDoubleVec2Vec s_T0;
            //! The partition for the null hypothesis.
            TTimeTimePr2Vec s_Partition;
            //! The start of the repeating partition.
            core_t::TTime s_StartOfPartition;
        };

    protected:
        //! The minimum proportion of populated buckets for which
        //! the test is accurate.
        static const double ACCURATE_TEST_POPULATED_FRACTION;

    protected:
        //! Get the bucket length.
        core_t::TTime bucketLength(void) const;

        //! Get the window length.
        core_t::TTime windowLength(void) const;

        //! Get the bucket values.
        const TFloatMeanAccumulatorVec &bucketValues(void) const;

        //! Initialize by reading state from \p traverser.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

        //! Persist state by passing information to \p inserter.
        void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

        //! Get a checksum for this object.
        uint64_t checksum(uint64_t seed) const;

        //! Initialize the bucket values.
        void initialize(core_t::TTime bucketLength,
                        core_t::TTime window,
                        const TFloatMeanAccumulatorVec &initial);

        //! Compute various ancillary statistics for testing.
        bool initializeTestStatistics(STestStats &stats) const;

        //! Get the variance and degrees freedom for the null hypothesis
        //! that there is no trend or repeating partition of any kind.
        bool nullHypothesis(STestStats &stats) const;

        //! Test to see if there is significant evidence for a component
        //! with period \p period.
        bool testPeriod(const TTimeTimePr2Vec &window,
                        core_t::TTime period, STestStats &stats) const;

        //! Test to see if there is significant evidence for a repeating
        //! partition of the data into windows defined by \p partition.
        bool testPartition(const TTimeTimePr2Vec &partition,
                           core_t::TTime period,
                           double correction, STestStats &stats) const;

        //! Condition \p values assuming the null hypothesis is true.
        //!
        //! This removes any trend associated with the null hypothesis.
        void conditionOnNullHypothesis(const TTimeTimePr2Vec &window,
                                       const STestStats &stats,
                                       TFloatMeanAccumulatorVec &values) const;

        //! Get the trend for period \p period.
        void periodicBucketing(core_t::TTime period,
                               const TTimeTimePr2Vec &windows,
                               TTimeTimePrMeanVarAccumulatorPrVec &trend) const;

        //! Get the decomposition into a short and long period trend.
        void periodicBucketing(TTime2Vec periods,
                               const TTimeTimePr2Vec &windows,
                               TTimeTimePrMeanVarAccumulatorPrVec &shortTrend,
                               TTimeTimePrMeanVarAccumulatorPrVec &longTrend) const;

        //! Initialize the buckets in \p trend.
        void initializeBuckets(std::size_t period,
                               const TTimeTimePr2Vec &windows,
                               TTimeTimePrMeanVarAccumulatorPrVec &trend) const;

    private:
        //! The minimum coefficient of variation to bother to test.
        static const double MINIMUM_COEFFICIENT_OF_VARIATION;

    private:
        //! The rate at which the bucket values are aged.
        double m_DecayRate;

        //! The bucketing interval.
        core_t::TTime m_BucketLength;

        //! The window length for which to maintain bucket values.
        core_t::TTime m_WindowLength;

        //! The mean bucket values.
        TFloatMeanAccumulatorVec m_BucketValues;
};

//! \brief Implements test for diurnal periodic components.
//!
//! DESCRIPTION:\n
//! This tests whether there are daily and/or weekly components
//! in a time series. It also checks to see if there is a partition
//! of the time series into disjoint weekend and weekday intervals.
//! Tests include various forms of analysis of variance and a test
//! of the amplitude.
class MATHS_EXPORT CDiurnalPeriodicityTest : public CPeriodicityTest
{
    public:
        explicit CDiurnalPeriodicityTest(double decayRate = 0.0);

        //! \name Persistence
        //@{
        //! Initialize by reading state from \p traverser.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

        //! Persist state by passing information to \p inserter.
        void acceptPersistInserter(core::CStatePersistInserter &inserter) const;
        //@}

        //! Get a test for daily and weekly periodic components.
        //!
        //! \warning The caller owns the returned object.
        static CDiurnalPeriodicityTest *create(core_t::TTime bucketLength,
                                               double decayRate = 0.0);

        //! Check if there periodic components.
        CPeriodicityTestResult test(void) const;

        //! Get the periods being tested.
        virtual TTime2Vec periods(void) const;

        //! Get a seasonal time for the specified results.
        //!
        //! \warning The caller owns the returned object.
        virtual CSeasonalTime *seasonalTime(const TComponent &component) const;

        //! Get the periodic trends corresponding to \p required.
        virtual void trends(const CPeriodicityTestResult &required,
                            TTimeTimePrMeanVarAccumulatorPrVecVec &result) const;

        //! Get the static size of this object.
        virtual std::size_t staticSize(void) const;

        //! Get a checksum for this object.
        uint64_t checksum(uint64_t seed = 0) const;

        //! Print a description of \p result.
        virtual std::string print(const CPeriodicityTestResult &result) const;

    private:
        //! Initialize the bucket values.
        bool initialize(core_t::TTime bucketLength,
                        core_t::TTime window,
                        const double (&corrections)[2]);

        //! Test for weekly components given a daily periodic component
        //! and weekday/end partition.
        CPeriodicityTestResult testWeeklyGivenDailyAndWeekend(STestStats &stats) const;

        //! Test for a daily periodic component.
        bool testDaily(STestStats &stats) const;
        //! Test for a weekday/end partition.
        bool testWeekend(bool daily, STestStats &stats) const;
        //! Test for a weekly periodic component.
        bool testWeekly(const TTimeTimePr2Vec &window, STestStats &stats) const;

    private:
        //! The minimum proportion of explained variance including the
        //! partition into weekdays and weekends for the test to pass.
        static const double MAXIMUM_PARTITION_VARIANCE;
        //! The minimum proportion of explained variance including the
        //! periodic component for the test to pass.
        static const double MAXIMUM_PERIOD_VARIANCE;
        //! The threshold for the amplitude of the periodic component,
        //! as a multiple of the residual standard deviation, used to
        //! test for the presence of periodic spikes.
        static const double MINIMUM_PERIOD_AMPLITUDE;
        //! The minimum permitted autocorrelation for the test to pass.
        static const double MINIMUM_AUTOCORRELATION;

    private:
        //! The scales to apply when to the weekend variance when using
        //! {simple partition, partition and daily periodicity}.
        double m_VarianceCorrections[2];
};

//! \brief Tests to see whether there is a specified periodic component.
class MATHS_EXPORT CGeneralPeriodicityTest : public CPeriodicityTest
{
    public:
        //! An empty collection of bucket values.
        static const TFloatMeanAccumulatorVec NO_BUCKET_VALUES;

    public:
        //! Initialize the bucket values.
        bool initialize(core_t::TTime bucketLength,
                        core_t::TTime window,
                        core_t::TTime period,
                        const TFloatMeanAccumulatorVec &initial = NO_BUCKET_VALUES);

        //! Check if there periodic components.
        CPeriodicityTestResult test(void) const;

        //! Get the periods being tested.
        virtual TTime2Vec periods(void) const;

        //! Get a seasonal time for the specified results.
        //!
        //! \warning The caller owns the returned object.
        CSeasonalTime *seasonalTime(const TComponent &component) const;

        //! Get the periodic trend corresponding to \p required.
        virtual void trends(const CPeriodicityTestResult &required,
                            TTimeTimePrMeanVarAccumulatorPrVecVec &result) const;

        //! Get the static size of this object.
        virtual std::size_t staticSize(void) const;

        //! Get a checksum for this object.
        uint64_t checksum(uint64_t seed = 0) const;

        //! Print a description of \p result.
        virtual std::string print(const CPeriodicityTestResult &result) const;

    private:
        //! The minimum proportion of explained variance including the
        //! periodic component for the test to pass.
        static const double MAXIMUM_PERIOD_VARIANCE;
        //! The threshold for the amplitude of the periodic component,
        //! as a multiple of the residual standard deviation, used to
        //! test for the presence of periodic spikes.
        static const double MINIMUM_PERIOD_AMPLITUDE;
        //! The minimum permitted autocorrelation for the test to pass.
        static const double MINIMUM_AUTOCORRELATION;

    private:
        //! The candidate period.
        core_t::TTime m_Period;
};

//! \brief Implements a test that scans through a range of frequencies
//! looking for different periodic components in the data.
//!
//! DESCRIPTION:\n
//! This performs a scan for increasingly low frequency periodic
//! components maintaining a fixed size buffer. We find the most
//! promising candidate periods using linear autocorrelation and
//! then test them using our standard periodicity test.
//!
//! In order to maintain a fixed space the bucket length is increased
//! as soon as the observed data span exceeds the test size multiplied
//! by the current bucket span.
class MATHS_EXPORT CScanningPeriodicityTest
{
    public:
        using TDoubleVec = std::vector<double>;
        using TTimeVec = std::vector<core_t::TTime>;
        using TTimeCRng = core::CVectorRange<const TTimeVec>;
        using TPeriodicityResultPr = std::pair<CGeneralPeriodicityTest, CPeriodicityTestResult>;

    public:
        CScanningPeriodicityTest(TTimeCRng bucketLengths,
                                 std::size_t size,
                                 double decayRate = 0.0);

        //! Initialize by reading state from \p traverser.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

        //! Persist state by passing information to \p inserter.
        void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

        //! Set the start time to \p time.
        void initialize(core_t::TTime time);

        //! Age the bucket values to account for \p time elapsed time.
        void propagateForwardsByTime(double time);

        //! Add \p value at \p time.
        void add(core_t::TTime time, double value, double weight = 1.0);

        //! Check if we need to compress by increasing the bucket span.
        bool needToCompress(core_t::TTime) const;

        //! Check if there periodic components.
        TPeriodicityResultPr test(void) const;

        //! Roll time forwards by \p skipInterval.
        void skipTime(core_t::TTime skipInterval);

        //! Get a checksum for this object.
        uint64_t checksum(uint64_t seed = 0) const;

        //! Debug the memory used by this object.
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the memory used by this object.
        std::size_t memoryUsage(void) const;

    private:
        using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
        using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;

    private:
        //! Resize the bucket values to \p size.
        void resize(std::size_t size, TFloatMeanAccumulatorVec &values) const;

        //! Compute the mean of the autocorrelation at integer multiples
        //! of \p period.
        double meanForPeriodicOffsets(const TDoubleVec &correlations, std::size_t period) const;

        //! Correct the autocorrelation calculated on padded data.
        double correctForPad(double correlation, std::size_t offset) const;

    private:
        //! The rate at which the bucket values are aged.
        double m_DecayRate;

        //! The bucket lengths to test.
        TTimeCRng m_BucketLengths;

        //! The index in m_BucketLengths of the current bucketing interval.
        std::size_t m_BucketLengthIndex;

        //! The time of the first data point.
        core_t::TTime m_StartTime;

        //! The bucket values.
        TFloatMeanAccumulatorVec m_BucketValues;
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
class MATHS_EXPORT CCalendarCyclicTest
{
    public:
        using TOptionalFeature = boost::optional<CCalendarFeature>;

    public:
        explicit CCalendarCyclicTest(double decayRate = 0.0);

        //! Initialize by reading state from \p traverser.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

        //! Persist state by passing information to \p inserter.
        void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

        //! Age the bucket values to account for \p time elapsed time.
        void propagateForwardsByTime(double time);

        //! Add \p error at \p time.
        void add(core_t::TTime time, double error, double weight = 1.0);

        //! Check if there are calendar components.
        TOptionalFeature test(void) const;

        //! Get a checksum for this object.
        uint64_t checksum(uint64_t seed = 0) const;

        //! Debug the memory used by this object.
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the memory used by this object.
        std::size_t memoryUsage(void) const;

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
