/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CTimeSeriesTestForSeasonality_h
#define INCLUDED_ml_maths_CTimeSeriesTestForSeasonality_h

#include <core/CSmallVector.h>
#include <core/CVectorRange.h>
#include <core/Constants.h>
#include <core/CoreTypes.h>

#include <maths/CBasicStatistics.h>
#include <maths/CFuzzyLogic.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CSignal.h>
#include <maths/CTimeSeriesSegmentation.h>
#include <maths/ImportExport.h>

#include <boost/math/distributions/binomial.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/optional.hpp>

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace ml {
namespace maths {

class CSeasonalTime;

//! \brief A summary of the trend on the test window.
class MATHS_EXPORT CNewTrendSummary final {
public:
    using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;

public:
    CNewTrendSummary(core_t::TTime startTime,
                     core_t::TTime bucketLength,
                     TFloatMeanAccumulatorVec initialValues);

    //! Get the start time of the initial values.
    core_t::TTime initialValuesStartTime() const;

    //! Get the end time of the initial values.
    core_t::TTime initialValuesEndTime() const;

    //! Get the initial values bucket length.
    core_t::TTime bucketLength() const;

    //! Get the values to use to initialize the component.
    const TFloatMeanAccumulatorVec& initialValues() const;

private:
    core_t::TTime m_InitialValuesStartTime = 0;
    core_t::TTime m_BucketLength = 0;
    TFloatMeanAccumulatorVec m_InitialValues;
};

//! \brief A summary of a new seasonal component.
class MATHS_EXPORT CNewSeasonalComponentSummary {
public:
    using TOptionalTime = boost::optional<core_t::TTime>;
    using TSeasonalComponent = CSignal::SSeasonalComponentSummary;
    using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
    using TSeasonalTimeUPtr = std::unique_ptr<CSeasonalTime>;

    enum EPeriodDescriptor {
        E_Day = 0x1,
        E_Week = 0x2,
        E_Year = 0x4,
        E_General = 0x8,
        E_Diurnal = E_Day | E_Week
    };

public:
    CNewSeasonalComponentSummary(std::string annotationText,
                                 const TSeasonalComponent& period,
                                 std::size_t size,
                                 EPeriodDescriptor periodDescriptor,
                                 core_t::TTime initialValuesStartTime,
                                 core_t::TTime bucketsStartTime,
                                 core_t::TTime bucketLength,
                                 TOptionalTime startOfWeekTime,
                                 TFloatMeanAccumulatorVec initialValues);

    //! Get the annotation for this component.
    const std::string& annotationText() const;

    //! Get the desired component size.
    std::size_t size() const;

    //! Get a seasonal time for the specified results.
    //!
    //! \warning The caller owns the returned object.
    TSeasonalTimeUPtr seasonalTime() const;

    //! Get the start time of the initial values.
    core_t::TTime initialValuesStartTime() const;

    //! Get the end time of the initial values.
    core_t::TTime initialValuesEndTime() const;

    //! Get the values to use to initialize the component.
    const TFloatMeanAccumulatorVec& initialValues() const;

    //! Get a description of the component.
    std::string print() const;

private:
    std::string m_AnnotationText;
    TSeasonalComponent m_Period;
    std::size_t m_Size = 0;
    EPeriodDescriptor m_PeriodDescriptor = E_General;
    core_t::TTime m_InitialValuesStartTime = 0;
    core_t::TTime m_BucketsStartTime = 0;
    core_t::TTime m_BucketLength = 0;
    TOptionalTime m_StartOfWeekTime;
    TFloatMeanAccumulatorVec m_InitialValues;
};

//! \brief Represents the decomposition of a collection of values into zero or more
//! seasonal components.
class MATHS_EXPORT CSeasonalDecomposition {
public:
    using TBoolVec = std::vector<bool>;
    using TOptionalTime = boost::optional<core_t::TTime>;
    using TSeasonalComponent = CSignal::SSeasonalComponentSummary;
    using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
    using TNewSeasonalComponentVec = std::vector<CNewSeasonalComponentSummary>;
    using TPeriodDescriptor = CNewSeasonalComponentSummary::EPeriodDescriptor;

public:
    //! Add the new trend summary.
    void add(CNewTrendSummary value);

    //! Add a new seasonal component.
    void add(std::string annotationText,
             const TSeasonalComponent& period,
             std::size_t size,
             TPeriodDescriptor periodDescriptor,
             core_t::TTime initialValuesStartTime,
             core_t::TTime bucketsStartTime,
             core_t::TTime bucketLength,
             TOptionalTime startOfWeekTime,
             TFloatMeanAccumulatorVec initialValues);

    //! Add a mask of any seasonal components which should be removed.
    void add(TBoolVec seasonalToRemoveMask);

    //! Set the within bucket value variance.
    void withinBucketVariance(double variance);

    //! Return true if the test thinks the components have changed.
    bool componentsChanged() const;

    //! Get the summary of any trend component.
    const CNewTrendSummary* trend() const;

    //! Get the summaries of any periodic components.
    const TNewSeasonalComponentVec& seasonal() const;

    //! A mask of any currently modelled components to remove.
    const TBoolVec& seasonalToRemoveMask() const;

    //! Get the within bucket value variance.
    double withinBucketVariance() const;

    //! Get a description of the seasonal components.
    std::string print() const;

private:
    using TOptionalNewTrendSummary = boost::optional<CNewTrendSummary>;

private:
    TOptionalNewTrendSummary m_Trend;
    TNewSeasonalComponentVec m_Seasonal;
    TBoolVec m_SeasonalToRemoveMask;
    double m_WithinBucketVariance = 0.0;
};

//! \brief Discovers the seasonal components present in the values in a time window
//! of a discrete time series.
class MATHS_EXPORT CTimeSeriesTestForSeasonality {
public:
    using TBoolVec = std::vector<bool>;
    using TPredictor = std::function<double(core_t::TTime, const TBoolVec&)>;
    using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;

public:
    static constexpr double OUTLIER_FRACTION = 0.1;
    static constexpr std::size_t MAXIMUM_NUMBER_SEGMENTS = 4;

public:
    CTimeSeriesTestForSeasonality(core_t::TTime valuesStartTime,
                                  core_t::TTime bucketsStartTime,
                                  core_t::TTime bucketLength,
                                  core_t::TTime sampleInterval,
                                  TFloatMeanAccumulatorVec values,
                                  double sampleVariance = 0.0,
                                  double outlierFraction = OUTLIER_FRACTION);

    //! Check if it is possible to test for \p component given the window \p values.
    static bool canTestComponent(const TFloatMeanAccumulatorVec& values,
                                 core_t::TTime bucketsStartTime,
                                 core_t::TTime bucketLength,
                                 core_t::TTime minimumPeriod,
                                 const CSeasonalTime& component);

    //! Register a seasonal component which is already being modelled.
    void addModelledSeasonality(const CSeasonalTime& period, std::size_t size);

    //! Add a predictor for the currently modelled seasonal conponents.
    void modelledSeasonalityPredictor(const TPredictor& predictor);

    //! Fit and remove any seasonality we're modelling and can't test.
    void fitAndRemoveUntestableModelledComponents();

    //! Run the test and return the new components found if any.
    CSeasonalDecomposition decompose() const;

    //! \name Parameters
    //@{
    CTimeSeriesTestForSeasonality& lowAutocorrelation(double value) {
        m_LowAutocorrelation = value;
        return *this;
    }
    CTimeSeriesTestForSeasonality& mediumAutocorrelation(double value) {
        m_MediumAutocorrelation = value;
        return *this;
    }
    CTimeSeriesTestForSeasonality& highAutocorrelation(double value) {
        m_HighAutocorrelation = value;
        return *this;
    }
    CTimeSeriesTestForSeasonality& significantPValue(double value) {
        m_SignificantPValue = value;
        return *this;
    }
    CTimeSeriesTestForSeasonality& verySignificantPValue(double value) {
        m_VerySignificantPValue = value;
        return *this;
    }
    CTimeSeriesTestForSeasonality& acceptedFalsePostiveRate(double value) {
        m_AcceptedFalsePostiveRate = value;
        return *this;
    }
    CTimeSeriesTestForSeasonality& minimumPeriod(core_t::TTime value) {
        m_MinimumPeriod = value;
        return *this;
    }
    CTimeSeriesTestForSeasonality& minimumModelSize(std::size_t value) {
        m_MinimumModelSize = value;
        return *this;
    }
    CTimeSeriesTestForSeasonality& maximumNumberOfComponents(std::ptrdiff_t value) {
        m_MaximumNumberComponents = value;
        return *this;
    }
    //@}

private:
    using TDoubleVec = std::vector<double>;
    using TDoubleVecVec = std::vector<TDoubleVec>;
    using TSizeSizePr = std::pair<std::size_t, std::size_t>;
    using TSizeVec = std::vector<std::size_t>;
    using TOptionalSize = boost::optional<std::size_t>;
    using TOptionalTime = boost::optional<core_t::TTime>;
    using TMaxAccumulator =
        CBasicStatistics::COrderStatisticsHeap<double, std::greater<double>>;
    using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using TVector2x1 = CVectorNx1<double, 2>;
    using TVarianceStats = CSignal::SVarianceStats;
    using TSeasonalComponent = CSignal::SSeasonalComponentSummary;
    using TSeasonalComponentVec = CSignal::TSeasonalComponentVec;
    using TSeasonalComponentCRng = core::CVectorRange<const TSeasonalComponentVec>;
    using TMeanAccumulatorVecVec = CSignal::TMeanAccumulatorVecVec;
    using TMeanAccumulatorVecCRng = core::CVectorRange<const TMeanAccumulatorVecVec>;
    using TPeriodDescriptor = CNewSeasonalComponentSummary::EPeriodDescriptor;
    using TSegmentation = CTimeSeriesSegmentation;
    using TConstantScale = TSegmentation::TConstantScale;
    using TBucketPredictor = std::function<double(std::size_t)>;
    using TTransform = std::function<double(const TFloatMeanAccumulator&)>;
    using TRemoveTrend =
        std::function<bool(const TSeasonalComponentVec&, TFloatMeanAccumulatorVec&, TSizeVec&)>;
    using TMeanScale =
        std::function<void(const TSeasonalComponentVec&, TFloatMeanAccumulatorVec&)>;

    //! \brief Accumulates the minimum amplitude.
    //!
    //! This is used for finding and testing the significance of semi-periodic spikes
    //! (or dips), i.e. frequently repeated values that are much further from the mean
    //! than the datas' typical variation.
    class CMinAmplitude {
    public:
        static constexpr double INF = std::numeric_limits<double>::max();
        static constexpr std::size_t MINIMUM_REPEATS{4};

    public:
        CMinAmplitude(std::size_t numberValues, double meanRepeats, double level)
            : m_Level{level}, m_BucketLength{numberValues / this->targetCount(meanRepeats)},
              m_BucketAmplitudes(numberValues / m_BucketLength) {}

        //! Have we seen enough data to test for semi-periodic spikes?
        static bool seenSufficientDataToTestAmplitude(std::size_t range, std::size_t period);
        //! Update with \p value.
        void add(std::size_t index, const TFloatMeanAccumulator& value);
        //! Compute the amplitude.
        double amplitude() const;
        //! Compute the significance of the amplitude given residual distribution \p normal.
        double significance(const boost::math::normal& normal) const;
        //! Get a readable description.
        std::string print() const;

    private:
        using TMinMaxAccumulator = CBasicStatistics::CMinMax<double>;
        using TMinMaxAccumulatorVec = std::vector<TMinMaxAccumulator>;

    private:
        std::size_t targetCount(double meanRepeats) const {
            return std::max(static_cast<std::size_t>(std::ceil(meanRepeats / 3.0)),
                            MINIMUM_REPEATS);
        }

    private:
        //! The mean of the trend.
        double m_Level = 0.0;
        //! The total count of values added.
        std::size_t m_Count = 0;
        //! The size of the index buckets we divide the values into.
        std::size_t m_BucketLength;
        //! The smallest values.
        TMinMaxAccumulatorVec m_BucketAmplitudes;
    };

    using TAmplitudeVec = std::vector<CMinAmplitude>;

    //! \brief A summary of test statistics related to a new seasonal component.
    struct SHypothesisStats {
        explicit SHypothesisStats(const TSeasonalComponent& period)
            : s_Period{period} {}

        //! Test the variance explained by this component.
        void testExplainedVariance(const CTimeSeriesTestForSeasonality& params,
                                   const TVarianceStats& H0);
        //! Test the cyclic autocorrelation of this component.
        void testAutocorrelation(const CTimeSeriesTestForSeasonality& params);
        //! Test for semi-periodic components.
        void testAmplitude(const CTimeSeriesTestForSeasonality& params);

        //! Test the variance reduction of this hypothesis.
        CFuzzyTruthValue varianceTestResult(const CTimeSeriesTestForSeasonality& params) const;
        //! Test the amplitude of this hypothesis.
        CFuzzyTruthValue amplitudeTestResult(const CTimeSeriesTestForSeasonality& params) const;

        //! Check if this is better than \p other.
        bool isBetter(const SHypothesisStats& other) const;

        //! Check if we should evict an existing component from the model.
        bool evict(const CTimeSeriesTestForSeasonality& params, std::size_t modelledIndex) const;

        //! The weight of this hypothesis for compouting decomposition properties.
        double weight() const;

        //! Get a readable description.
        std::string print() const;

        //! Set to true if the hypothesis can tested.
        bool s_IsTestable = false;
        //! If true then this should be modelled.
        bool s_Model = false;
        //! True if we're going to discard this model.
        bool s_DiscardingModel = false;
        //! The index of a similar modelled component if there is one.
        std::size_t s_SimilarModelled = 0;
        //! A summary of the seasonal component period.
        TSeasonalComponent s_Period;
        //! The desired component model size.
        std::size_t s_ModelSize = 0;
        //! The number of segments in the trend.
        std::size_t s_NumberTrendSegments = 0;
        //! The number of scale segments in the component.
        std::size_t s_NumberScaleSegments = 0;
        //! The number of buckets we'd have to wait for a repeat of all the seasonal
        //! components we've detected up to and including this one normalized by the
        //! observed window.
        double s_LeastCommonRepeat = 0;
        //! The mean number of repeats of buckets with at least one value.
        double s_MeanNumberRepeats = 0.0;
        //! The number of repeats of the period window with at least one value.
        double s_WindowRepeats = 0.0;
        //! The autocorrelation estimate of the hypothesis.
        double s_Autocorrelation = 0.0;
        //! The autocorrelation estimate of absolute values for the hypothesis.
        double s_AutocorrelationUpperBound = 0.0;
        //! The proportion of values in the seasonal component which are not missing.
        double s_FractionNotMissing = 1.0;
        //! The residual variance after removing this component.
        double s_ResidualVariance = 0.0;
        //! The amount of variance this component explains.
        double s_ExplainedVariance = 0.0;
        //! The number of values used to example the variance.
        std::size_t s_NumberParametersToExplainVariance = 0;
        //! The explained variance significance.
        double s_ExplainedVariancePValue = 1.0;
        //! The amplitude significance.
        double s_AmplitudePValue = 1.0;
        //! Have we seen enough values to test this hypothesis's amplitude.
        bool s_SeenSufficientDataToTestAmplitude = false;
        //! The truth value for the hypothesis.
        CFuzzyTruthValue s_Truth = CFuzzyTruthValue::FALSE_VALUE;
        //! The segmentation into constant linear scales.
        TSizeVec s_ScaleSegments;
        //! The values to use to initialize the component.
        TFloatMeanAccumulatorVec s_InitialValues;
    };

    using THypothesisStatsVec = std::vector<SHypothesisStats>;

    //! \brief A candidate model for the values being decomposed.
    struct SModel {
        SModel() = default;
        SModel(const CTimeSeriesTestForSeasonality& params,
               TMeanVarAccumulator residualMoments,
               TMeanVarAccumulator truncatedResidualMoments,
               std::size_t numberTrendParameters,
               TFloatMeanAccumulatorVec trendInitialValues,
               THypothesisStatsVec hypotheses,
               TBoolVec removeComponentsMask)
            : s_Params{&params}, s_ResidualMoments{residualMoments},
              s_TruncatedResidualMoments{truncatedResidualMoments}, s_NumberTrendParameters{numberTrendParameters},
              s_TrendInitialValues{std::move(trendInitialValues)},
              s_Hypotheses{std::move(hypotheses)}, s_RemoveComponentsMask{std::move(
                                                       removeComponentsMask)} {}

        //! Does this include seasonality?
        bool seasonal() const { return s_Hypotheses.size() > 0; }
        //! True if every seasonal component could be tested.
        bool isTestable() const;
        //! Should this behave as a null hypothesis?
        bool isNull() const;
        //! Should this behave as an alternative hypothesis?
        bool isAlternative() const;
        //! The similarity of the components after applying this hypothesis.
        double componentsSimilarity() const;
        //! The p-value of this model vs H0.
        double pValue(const SModel& H0,
                      double minimumRelativeTruncatedVariance = 0.0,
                      double unexplainedVariance = 0.0) const;
        //! A proxy for p-value of this model vs H0 which doesn't underflow.
        double logPValueProxy(const SModel& H0) const;
        //! Get the variance explained per parameter weighted by the variance explained
        //! by each component of the model.
        TVector2x1 explainedVariancePerParameter(double variance, double truncatedVariance) const;
        //! The number of parameters in model.
        double numberParameters() const;
        //! The target model size.
        double targetModelSize() const;
        //! Get the total number of linear scalings of the periods which were used.
        double numberScalings() const;
        //! Get the average autocorrelation weighted by the variance explained by each
        //! component of the model.
        double autocorrelation() const;
        //! The least common repeat of the seasonal components in this hypothesis.
        double leastCommonRepeat() const;

        //! The test to which this decomposition applies.
        const CTimeSeriesTestForSeasonality* s_Params = nullptr;
        //! Are the seasonal components already modelled?
        bool s_AlreadyModelled = false;
        //! The residual variance after removing the seasonal components.
        TMeanVarAccumulator s_ResidualMoments;
        //! The residual variance after removing the seasonal components and outliers.
        TMeanVarAccumulator s_TruncatedResidualMoments;
        //! The number of parameters in the trend model.
        std::size_t s_NumberTrendParameters = 1;
        //! The values with which to initialize the trend.
        TFloatMeanAccumulatorVec s_TrendInitialValues;
        //! The seasonal components which have been found.
        THypothesisStatsVec s_Hypotheses;
        //! A mask of the currently modelled components to remove.
        TBoolVec s_RemoveComponentsMask;
    };

    using TModelVec = std::vector<SModel>;

private:
    CSeasonalDecomposition select(TModelVec& hypotheses) const;
    void addNotSeasonal(const TRemoveTrend& removeTrend, TModelVec& decompositions) const;
    void addModelled(const TRemoveTrend& removeTrend, TModelVec& decompositions) const;
    void addDiurnal(const TRemoveTrend& removeTrend, TModelVec& decompositions) const;
    void addHighestAutocorrelation(const TRemoveTrend& removeTrend,
                                   TModelVec& decompositions) const;
    void testAndAddDecomposition(const TSeasonalComponentVec& periods,
                                 const TSizeVec& modelTrendSegments,
                                 const TFloatMeanAccumulatorVec& valuesToTest,
                                 bool alreadyModelled,
                                 bool isDiurnal,
                                 TModelVec& decompositions) const;
    bool considerDecompositionForSelection(const SModel& decomposition,
                                           bool alreadyModelled,
                                           bool isDiurnal) const;
    SModel testDecomposition(const TSeasonalComponentVec& periods,
                             std::size_t numberTrendSegments,
                             const TFloatMeanAccumulatorVec& valueToTest,
                             bool alreadyModelled) const;
    void updateResiduals(const SHypothesisStats& hypothesis,
                         TFloatMeanAccumulatorVec& residuals) const;
    TBoolVec finalizeHypotheses(const TFloatMeanAccumulatorVec& valuesToTest,
                                bool alreadyModelled,
                                THypothesisStatsVec& hypotheses,
                                TFloatMeanAccumulatorVec& residuals) const;
    TBoolVec selectModelledHypotheses(bool alreadyModelled,
                                      THypothesisStatsVec& hypotheses) const;
    std::size_t selectComponentSize(const TFloatMeanAccumulatorVec& valuesToTest,
                                    const TSeasonalComponent& period) const;
    std::size_t similarModelled(const TSeasonalComponent& period) const;
    void removeModelledPredictions(const TBoolVec& componentsToRemoveMask,
                                   TFloatMeanAccumulatorVec& values) const;
    void removeDiscontinuities(const TSizeVec& modelTrendSegments,
                               TFloatMeanAccumulatorVec& values) const;
    bool constantScale(const TConstantScale& scale,
                       const TSeasonalComponentVec& periods,
                       const TSizeVec& scaleSegments,
                       TFloatMeanAccumulatorVec& values,
                       TDoubleVecVec& components,
                       TDoubleVec& scales) const;
    TVarianceStats residualVarianceStats(const TFloatMeanAccumulatorVec& values) const;
    TMeanVarAccumulator
    truncatedMoments(double outlierFraction,
                     const TFloatMeanAccumulatorVec& residuals,
                     const TTransform& transform = [](const TFloatMeanAccumulator& value) {
                         return CBasicStatistics::mean(value);
                     }) const;
    std::size_t numberTrendParameters(std::size_t numberTrendSegments) const;
    bool includesNewComponents(const TSeasonalComponentVec& periods) const;
    bool alreadyModelled(const TSeasonalComponentVec& periods) const;
    bool alreadyModelled(const TSeasonalComponent& period) const;
    bool onlyDiurnal(const TSeasonalComponentVec& periods) const;
    void removeIfNotTestable(TSeasonalComponentVec& periods) const;
    TPeriodDescriptor periodDescriptor(std::size_t period) const;
    bool isDiurnal(std::size_t period) const;
    bool isDiurnal(const TSeasonalComponent& period) const;
    bool isWeekend(const TSeasonalComponent& period) const;
    bool isWeekday(const TSeasonalComponent& period) const;
    bool permittedPeriod(const TSeasonalComponent& period) const;
    bool includesPermittedPeriod(const TSeasonalComponentVec& period) const;
    std::string annotationText(const TSeasonalComponent& period) const;
    std::size_t day() const;
    std::size_t week() const;
    std::size_t year() const;
    TSizeSizePr weekdayWindow() const;
    TSizeSizePr weekendWindow() const;
    static TSeasonalComponent toPeriod(core_t::TTime startTime,
                                       core_t::TTime bucketLength,
                                       const CSeasonalTime& component);
    static core_t::TTime adjustForStartTime(core_t::TTime startTime, core_t::TTime startOfWeek);
    static std::size_t buckets(core_t::TTime bucketLength, core_t::TTime interval);
    static bool canTestPeriod(const TFloatMeanAccumulatorVec& values,
                              std::size_t minimumPeriod,
                              const TSeasonalComponent& period);
    static bool periodTooLongToTest(const TFloatMeanAccumulatorVec& values,
                                    const TSeasonalComponent& period);
    static bool periodTooShortToTest(std::size_t minimumPeriod,
                                     const TSeasonalComponent& period);
    static std::size_t observedRange(const TFloatMeanAccumulatorVec& values);
    static std::size_t longestGap(const TFloatMeanAccumulatorVec& values);
    static TSizeSizePr observedInterval(const TFloatMeanAccumulatorVec& values);
    static void removePredictions(const TSeasonalComponentCRng& periodsToRemove,
                                  const TMeanAccumulatorVecCRng& componentsToRemove,
                                  TFloatMeanAccumulatorVec& values);
    static void removePredictions(const TBucketPredictor& predictor,
                                  TFloatMeanAccumulatorVec& values);

private:
    double m_MinimumRepeatsPerSegmentToTestVariance = 3.0;
    double m_MinimumRepeatsPerSegmentToTestAmplitude = 5.0;
    double m_LowAutocorrelation = 0.3;
    double m_MediumAutocorrelation = 0.5;
    double m_HighAutocorrelation = 0.7;
    double m_PValueToEvict = 0.5;
    double m_SignificantPValue = 5e-3;
    double m_VerySignificantPValue = 1e-6;
    double m_AcceptedFalsePostiveRate = 1e-4;
    std::ptrdiff_t m_MaximumNumberComponents = 10;
    std::size_t m_MinimumModelSize = 24;
    TOptionalSize m_StartOfWeekOverride;
    TOptionalTime m_StartOfWeekTimeOverride;
    core_t::TTime m_MinimumPeriod = 0;
    core_t::TTime m_ValuesStartTime = 0;
    core_t::TTime m_BucketsStartTime = 0;
    core_t::TTime m_BucketLength = 0;
    core_t::TTime m_SampleInterval = 0;
    double m_SampleVariance = 0.0;
    double m_OutlierFraction = OUTLIER_FRACTION;
    double m_EpsVariance = 0.0;
    TPredictor m_ModelledPredictor = [](core_t::TTime, const TBoolVec&) {
        return 0.0;
    };
    TSeasonalComponentVec m_ModelledPeriods;
    TSizeVec m_ModelledPeriodsSizes;
    TBoolVec m_ModelledPeriodsTestable;
    TFloatMeanAccumulatorVec m_Values;
    // The follow are member data to avoid repeatedly reinitialising.
    mutable TAmplitudeVec m_Amplitudes;
    mutable TSeasonalComponentVec m_Periods;
    mutable TSeasonalComponentVec m_CandidatePeriods;
    mutable TMeanAccumulatorVecVec m_Components;
    mutable TFloatMeanAccumulatorVec m_ValuesToTest;
    mutable TFloatMeanAccumulatorVec m_TemporaryValues;
    mutable TFloatMeanAccumulatorVec m_ValuesMinusTrend;
    mutable TSizeVec m_ModelTrendSegments;
    mutable TMaxAccumulator m_Outliers;
    mutable TSizeVec m_WindowIndices;
    mutable TDoubleVecVec m_ScaledComponent;
    mutable TDoubleVec m_ComponentScales;
};
}
}

#endif //INCLUDED_ml_maths_CTimeSeriesTestForSeasonality_h
