/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CPeriodicityHypothesisTests_h
#define INCLUDED_ml_maths_CPeriodicityHypothesisTests_h

#include <core/CSmallVector.h>
#include <core/CVectorRange.h>
#include <core/CoreTypes.h>

#include <maths/CBasicStatistics.h>
#include <maths/ImportExport.h>

#include <boost/function.hpp>
#include <boost/operators.hpp>

#include <string>
#include <utility>
#include <vector>

namespace ml {
namespace maths {
class CSeasonalTime;

//! \brief Represents the result of running the periodicity
//! hypothesis tests.
class MATHS_EXPORT CPeriodicityHypothesisTestsResult
    : boost::equality_comparable<CPeriodicityHypothesisTestsResult> {
public:
    using TTimeTimePr = std::pair<core_t::TTime, core_t::TTime>;
    using TSizeVec = std::vector<std::size_t>;

    //! \brief Component data.
    struct MATHS_EXPORT SComponent {
        SComponent() = default;
        SComponent(const std::string& description,
                   bool diurnal,
                   bool piecewiseScaled,
                   core_t::TTime startOfPartition,
                   core_t::TTime period,
                   const TTimeTimePr& window,
                   double precedence = 1.0);

        //! Check if this is equal to \p other.
        bool operator==(const SComponent& other) const;

        //! Get a seasonal time for the specified results.
        //!
        //! \warning The caller owns the returned object.
        CSeasonalTime* seasonalTime() const;

        //! An identifier for the component used by the test.
        std::string s_Description;
        //! True if this is a diurnal component false otherwise.
        bool s_Diurnal = false;
        //! The segmentation of the window into intervals of constant
        //! scaling.
        bool s_PiecewiseScaled = false;
        //! The start of the partition.
        core_t::TTime s_StartOfPartition = 0;
        //! The period of the component.
        core_t::TTime s_Period = 0;
        //! The component window.
        TTimeTimePr s_Window;
        //! The precedence to apply to this component when deciding
        //! which to keep.
        double s_Precedence = 0.0;
    };

    using TComponent5Vec = core::CSmallVector<SComponent, 5>;
    using TRemoveCondition = std::function<bool(const SComponent&)>;

public:
    //! Check if this is equal to \p other.
    bool operator==(const CPeriodicityHypothesisTestsResult& other) const;

    //! Add a component.
    void add(const std::string& description,
             bool diurnal,
             bool piecewiseScaled,
             core_t::TTime startOfWeek,
             core_t::TTime period,
             const TTimeTimePr& window,
             double precedence = 1.0);

    //! Remove the component with \p description.
    void remove(const TRemoveCondition& condition);

    //! Set if this is a piecewise linear trend.
    void piecewiseLinearTrend(bool value);

    //! Check if this is a piecewise linear trend.
    bool piecewiseLinearTrend() const;

    //! Check if there are any periodic components.
    bool periodic() const;

    //! Get the binary representation of the periodic components.
    const TComponent5Vec& components() const;

    //! Get a human readable description of the result.
    std::string print() const;

private:
    //! If true then the hypothesis used a piecewise linear trend.
    bool m_PiecewiseLinearTrend = false;

    //! The periodic components.
    TComponent5Vec m_Components;
};

//! \brief Configures the periodicity testing.
class MATHS_EXPORT CPeriodicityHypothesisTestsConfig {
public:
    CPeriodicityHypothesisTestsConfig();

    //! Disable diurnal periodicity tests.
    void disableDiurnal();
    //! Test given we know there is daily periodic component.
    void hasDaily(bool value);
    //! Test given we know there is a weekend.
    void hasWeekend(bool value);
    //! Test given we know there is a weekly periodic component.
    void hasWeekly(bool value);
    //! Set the start of the week.
    void startOfWeek(core_t::TTime value);

    //! Check if we should test for diurnal periodic components.
    bool testForDiurnal() const;
    //! Check if we know there is a daily component.
    bool hasDaily() const;
    //! Check if we know there is a weekend.
    bool hasWeekend() const;
    //! Check if we know there is a weekly component.
    bool hasWeekly() const;
    //! Get the start of the week.
    core_t::TTime startOfWeek() const;

private:
    //! True if we should test for diurnal periodicity.
    bool m_TestForDiurnal;
    //! True if we know there is a daily component.
    bool m_HasDaily;
    //! True if we know there is a weekend.
    bool m_HasWeekend;
    //! True if we know there is a weekly component.
    bool m_HasWeekly;
    //! The start of the week.
    core_t::TTime m_StartOfWeek;
};

//! \brief Implements a set of hypothesis tests to discover the
//! most plausible explanation of the periodic patterns in a
//! time window of a time series.
//!
//! DESCRIPTION:\n
//! This tests whether there are daily and/or weekly components
//! in a time series. It also checks to see if there is a partition
//! of the time series into disjoint weekend and weekday intervals.
//! Tests include various forms of analysis of variance and a test
//! of the amplitude. It also compares these possibilities with a
//! specified period (typically found by examining the cyclic
//! autocorrelation).
class MATHS_EXPORT CPeriodicityHypothesisTests {
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
    using TComponent = CPeriodicityHypothesisTestsResult::SComponent;

public:
    CPeriodicityHypothesisTests() = default;
    explicit CPeriodicityHypothesisTests(const CPeriodicityHypothesisTestsConfig& config);

    //! Check if the test is initialized.
    bool initialized() const;

    //! Initialize the bucket values.
    void initialize(core_t::TTime startTime,
                    core_t::TTime bucketLength,
                    core_t::TTime window,
                    core_t::TTime period);

    //! Add \p value at \p time.
    void add(core_t::TTime time, double value, double weight = 1.0);

    //! Check if there periodic components and, if there are,
    //! which best describe the periodic patterns in the data.
    CPeriodicityHypothesisTestsResult test() const;

private:
    using TDoubleVec = std::vector<double>;
    using TDoubleVec2Vec = core::CSmallVector<TDoubleVec, 2>;
    using TSizeVec = std::vector<std::size_t>;
    using TFloatMeanAccumulatorCRng = core::CVectorRange<const TFloatMeanAccumulatorVec>;
    using TMinMaxAccumulator = maths::CBasicStatistics::CMinMax<core_t::TTime>;

    //! \brief A collection of statistics used during testing.
    struct STestStats {
        explicit STestStats(double meanMagnitude);
        //! Set the various test thresholds.
        void setThresholds(double vt, double at, double Rt);
        //! Check if the null hypothesis is good enough to not need an
        //! alternative.
        bool nullHypothesisGoodEnough() const;
        //! The number of segments in the trend.
        double s_TrendSegments;
        //! True if a known periodic component is tested.
        bool s_HasPeriod;
        //! True if a known repeating partition is tested.
        bool s_HasPartition;
        //! The maximum variance to accept the alternative hypothesis.
        double s_VarianceThreshold;
        //! The minimum amplitude to accept the alternative hypothesis.
        double s_AmplitudeThreshold;
        //! The minimum autocorrelation to accept the alternative
        //! hypothesis.
        double s_AutocorrelationThreshold;
        //! The data range.
        double s_Range;
        //! The number of buckets with at least one measurement.
        double s_NonEmptyBuckets;
        //! The average number of measurements per bucket value.
        double s_MeasurementsPerBucket;
        //! The mean magnitude of the bucket values.
        double s_MeanMagnitude;
        //! The null hypothesis periodic components.
        CPeriodicityHypothesisTestsResult s_H0;
        //! The variance estimate of H0.
        double s_V0;
        //! The autocorrelation estimate of H0.
        double s_R0;
        //! The degrees of freedom in the variance estimate of H0.
        double s_DF0;
        //! The trend for the null hypothesis.
        TDoubleVec2Vec s_T0;
        //! The linear scales if any.
        TDoubleVec s_Scales;
        //! The partition for the null hypothesis.
        TTimeTimePr2Vec s_Partition;
        //! The start of the repeating partition.
        core_t::TTime s_StartOfPartition;
        //! The segmentation of the interval if any.
        TSizeVec s_Segmentation;
    };

    //! \brief Manages the testing of a set of nested hypotheses.
    class CNestedHypotheses {
    public:
        // Using boost::function rather than std::function here due to a problem with VS2013
        using TTestFunc = boost::function<CPeriodicityHypothesisTestsResult(STestStats&)>;

        //! \brief Manages the building of a collection of nested
        //! hypotheses.
        class CBuilder {
        public:
            explicit CBuilder(CNestedHypotheses& hypothesis);
            CBuilder& addNested(TTestFunc test);
            CBuilder& addAlternative(TTestFunc test);
            CBuilder& finishedNested();

        private:
            using TNestedHypothesesPtrVec = std::vector<CNestedHypotheses*>;

        private:
            TNestedHypothesesPtrVec m_Levels;
        };

    public:
        explicit CNestedHypotheses(TTestFunc test = TTestFunc());

        //! Set the null hypothesis.
        CBuilder null(TTestFunc test);
        //! Add a nested hypothesis for \p test.
        CNestedHypotheses& addNested(TTestFunc test);
        //! Test the hypotheses.
        CPeriodicityHypothesisTestsResult test(STestStats& stats) const;
        //! Set if the hypothesis uses a piecewise linear trend.
        void trendSegments(std::size_t segments);
        //! Check if the hypothesis uses a piecewise linear trend.
        std::size_t trendSegments() const;

    private:
        using THypothesisVec = std::vector<CNestedHypotheses>;

    private:
        //! The test.
        TTestFunc m_Test;
        //! The number of segments in the trend.
        std::size_t m_TrendSegments;
        //! If true always test the nested hypotheses.
        bool m_AlwaysTestNested;
        //! The nested hypotheses to test.
        THypothesisVec m_Nested;
    };

    using TNestedHypothesesVec = std::vector<CNestedHypotheses>;

private:
    //! Get the hypotheses to test for period/daily/weekly components.
    void hypothesesForWeekly(const TTimeTimePr2Vec& windowForTestingWeekly,
                             const TFloatMeanAccumulatorCRng& bucketsForTestingWeekly,
                             const TTimeTimePr2Vec& windowForTestingPeriod,
                             const TFloatMeanAccumulatorCRng& bucketsForTestingPeriod,
                             TNestedHypothesesVec& hypotheses) const;

    //! Get the hypotheses to test for period/daily components.
    void hypothesesForDaily(const TTimeTimePr2Vec& windowForTestingDaily,
                            const TFloatMeanAccumulatorCRng& bucketsForTestingDaily,
                            const TTimeTimePr2Vec& windowForTestingPeriod,
                            const TFloatMeanAccumulatorCRng& bucketsForTestingPeriod,
                            TNestedHypothesesVec& hypotheses) const;

    //! Get the hypotheses to test for period components.
    void hypothesesForPeriod(const TTimeTimePr2Vec& windows,
                             const TFloatMeanAccumulatorCRng& buckets,
                             TNestedHypothesesVec& hypotheses) const;

    //! Extract the best hypothesis.
    CPeriodicityHypothesisTestsResult best(const TNestedHypothesesVec& hypotheses) const;

    //! The null hypothesis of the various tests.
    CPeriodicityHypothesisTestsResult testForNull(const TTimeTimePr2Vec& window,
                                                  const TFloatMeanAccumulatorCRng& buckets,
                                                  STestStats& stats) const;

    //! Test for a daily periodic component.
    CPeriodicityHypothesisTestsResult testForDaily(const TTimeTimePr2Vec& window,
                                                   const TFloatMeanAccumulatorCRng& buckets,
                                                   bool scaling,
                                                   STestStats& stats) const;

    //! Test for a weekly periodic component.
    CPeriodicityHypothesisTestsResult testForWeekly(const TTimeTimePr2Vec& window,
                                                    const TFloatMeanAccumulatorCRng& buckets,
                                                    bool scaling,
                                                    STestStats& stats) const;

    //! Test for a weekday/end partition.
    CPeriodicityHypothesisTestsResult
    testForDailyWithWeekend(const TFloatMeanAccumulatorCRng& buckets, STestStats& stats) const;

    //! Test for a weekly period given we think there is a
    //! weekday/end partition.
    CPeriodicityHypothesisTestsResult
    testForWeeklyGivenDailyWithWeekend(const TTimeTimePr2Vec& window,
                                       const TFloatMeanAccumulatorCRng& buckets,
                                       STestStats& stats) const;

    //! Test for the specified period given we think there is diurnal
    //! periodicity.
    CPeriodicityHypothesisTestsResult testForPeriod(const TTimeTimePr2Vec& window,
                                                    const TFloatMeanAccumulatorCRng& buckets,
                                                    bool scaling,
                                                    STestStats& stats) const;

    //! Check we've seen sufficient data to test accurately.
    bool seenSufficientDataToTest(core_t::TTime period,
                                  const TFloatMeanAccumulatorCRng& buckets) const;

    //! Check if there are enough non-empty buckets which are repeated
    //! at at least one \p period in \p buckets.
    template<typename CONTAINER>
    bool seenSufficientPeriodicallyPopulatedBucketsToTest(const CONTAINER& buckets,
                                                          std::size_t period) const;

    //! Compute various ancillary statistics for testing.
    bool testStatisticsFor(const TFloatMeanAccumulatorCRng& buckets, STestStats& stats) const;

    //! Get the variance and degrees freedom for the null hypothesis
    //! that there is no trend or repeating partition of any kind.
    void nullHypothesis(const TTimeTimePr2Vec& window,
                        const TFloatMeanAccumulatorCRng& buckets,
                        STestStats& stats) const;

    //! Compute the variance and degrees freedom for the hypothesis.
    void hypothesis(const TTime2Vec& periods,
                    const TFloatMeanAccumulatorCRng& buckets,
                    STestStats& stats) const;

    //! Condition \p buckets assuming the null hypothesis is true.
    //!
    //! This removes any trend associated with the null hypothesis.
    void conditionOnHypothesis(const STestStats& stats, TFloatMeanAccumulatorVec& buckets) const;

    //! Test to see if there is significant evidence for a component
    //! with period \p period.
    bool testPeriod(const TTimeTimePr2Vec& window,
                    const TFloatMeanAccumulatorCRng& buckets,
                    core_t::TTime period,
                    STestStats& stats) const;

    //! Test to see if there is significant evidence for a component
    //! with period \p period which is piecewise linearly scaled.
    bool testPeriodWithScaling(const TTimeTimePr2Vec& windows,
                               const TFloatMeanAccumulatorCRng& buckets,
                               core_t::TTime period,
                               STestStats& stats) const;

    //! Test to see if there is significant evidence for a repeating
    //! partition of the data into windows defined by \p partition.
    bool testPartition(const TTimeTimePr2Vec& partition,
                       const TFloatMeanAccumulatorCRng& buckets,
                       core_t::TTime period,
                       double correction,
                       STestStats& stats) const;

    //! Run the explained variance test on an alternative hypothesis.
    bool testVariance(const TTimeTimePr2Vec& window,
                      const TFloatMeanAccumulatorVec& buckets,
                      core_t::TTime period,
                      double df1,
                      double v1,
                      STestStats& stats,
                      double& R,
                      double& meanRepeats,
                      double& pVariance,
                      const TSizeVec& segmentation = TSizeVec{}) const;

    //! Run the component amplitude test on the alternative hypothesis.
    bool testAmplitude(const TTimeTimePr2Vec& window,
                       const TFloatMeanAccumulatorVec& buckets,
                       core_t::TTime period,
                       double b,
                       double v,
                       double R,
                       double meanRepeats,
                       double pVariance,
                       STestStats& stats) const;

private:
    //! The minimum proportion of populated buckets for which
    //! the test is accurate.
    static const double ACCURATE_TEST_POPULATED_FRACTION;

    //! The minimum coefficient of variation to bother to test.
    static const double MINIMUM_COEFFICIENT_OF_VARIATION;

private:
    //! Configures the tests to run.
    CPeriodicityHypothesisTestsConfig m_Config;

    //! The start time of the window.
    core_t::TTime m_StartTime = 0;

    //! The bucketing interval.
    core_t::TTime m_BucketLength = 0;

    //! The window length for which to maintain bucket values.
    core_t::TTime m_WindowLength = 0;

    //! The specified period to test.
    core_t::TTime m_Period = 0;

    //! The time range of values added to the test.
    TMinMaxAccumulator m_TimeRange;

    //! The mean bucket values.
    TFloatMeanAccumulatorVec m_BucketValues;
};

//! Test for periodic components in \p values.
MATHS_EXPORT
CPeriodicityHypothesisTestsResult
testForPeriods(const CPeriodicityHypothesisTestsConfig& config,
               core_t::TTime startTime,
               core_t::TTime bucketLength,
               const std::vector<CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator>& values);
}
}

#endif // INCLUDED_ml_maths_CPeriodicityHypothesisTests_h
