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

namespace CTimeSeriesTestForSeasonalityTest {
struct calibrateTruncatedVariance;
}

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
    core_t::TTime m_StartTime = 0;
    core_t::TTime m_BucketLength = 0;
    TFloatMeanAccumulatorVec m_InitialValues;
};

//! \brief A summary of a new seasonal component.
class MATHS_EXPORT CNewSeasonalComponentSummary {
public:
    using TSeasonalComponent = CSignal::SSeasonalComponentSummary;
    using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
    using TSeasonalTimeUPtr = std::unique_ptr<CSeasonalTime>;

public:
    CNewSeasonalComponentSummary(std::string annotationText,
                                 const TSeasonalComponent& period,
                                 std::size_t size,
                                 bool diurnal,
                                 core_t::TTime startTime,
                                 core_t::TTime bucketLength,
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
    bool m_Diurnal = false;
    core_t::TTime m_StartTime = 0;
    core_t::TTime m_BucketLength = 0;
    TFloatMeanAccumulatorVec m_InitialValues;
};

//! \brief Represents the decomposition of a collection of values into zero or more
//! seasonal components.
class MATHS_EXPORT CSeasonalDecomposition {
public:
    using TBoolVec = std::vector<bool>;
    using TSeasonalComponent = CSignal::SSeasonalComponentSummary;
    using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
    using TNewSeasonalComponentVec = std::vector<CNewSeasonalComponentSummary>;

public:
    //! Add the new trend summary.
    void add(CNewTrendSummary value);

    //! Add a new seasonal component.
    void add(std::string annotationText,
             const TSeasonalComponent& period,
             std::size_t size,
             bool diurnal,
             core_t::TTime startTime,
             core_t::TTime bucketLength,
             TFloatMeanAccumulatorVec initialValues);

    //! Add a mask of any seasonal components which should be removed.
    void add(TBoolVec seasonalToRemoveMask);

    //! Get the summary of any trend component.
    const CNewTrendSummary* trend() const;

    //! Get the summaries of any periodic components.
    const TNewSeasonalComponentVec& seasonal() const;

    //! A mask of any currently modelled components to remove.
    const TBoolVec& seasonalToRemoveMask() const;

    //! Get a description of the seasonal components.
    std::string print() const;

private:
    using TOptionalNewTrendSummary = boost::optional<CNewTrendSummary>;

private:
    TOptionalNewTrendSummary m_Trend;
    TNewSeasonalComponentVec m_Seasonal;
    TBoolVec m_SeasonalToRemoveMask;
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

public:
    CTimeSeriesTestForSeasonality(core_t::TTime startTime,
                                  core_t::TTime bucketLength,
                                  TFloatMeanAccumulatorVec values,
                                  double outlierFraction = OUTLIER_FRACTION);

    //! Check if it is possible to test for \p component given the window \p values.
    static bool canTestComponent(const TFloatMeanAccumulatorVec& values,
                                 core_t::TTime startTime,
                                 core_t::TTime bucketLength,
                                 const CSeasonalTime& component);

    //! The minimum seasonal component period to consider.
    void minimumPeriod(core_t::TTime minimumPeriod);

    //! Register a seasonal component which is already being modelled.
    void addModelledSeasonality(const CSeasonalTime& period);

    //! Add a predictor for the currently modelled seasonal conponents.
    void modelledSeasonalityPredictor(const TPredictor& predictor);

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
    CTimeSeriesTestForSeasonality& maximumNumberOfComponents(std::ptrdiff_t value) {
        m_MaximumNumberComponents = value;
        return *this;
    }
    //@}

private:
    using TDoubleVec = std::vector<double>;
    using TSizeSizePr = std::pair<std::size_t, std::size_t>;
    using TSizeVec = std::vector<std::size_t>;
    using TOptionalSize = boost::optional<std::size_t>;
    using TOptionalTime = boost::optional<core_t::TTime>;
    using TMaxAccumulator =
        CBasicStatistics::COrderStatisticsHeap<double, std::greater<double>>;
    using TVarianceStats = CSignal::SVarianceStats;
    using TSeasonalComponent = CSignal::SSeasonalComponentSummary;
    using TSeasonalComponentVec = CSignal::TSeasonalComponentVec;
    using TSeasonalComponentCRng = core::CVectorRange<const TSeasonalComponentVec>;
    using TMeanAccumulatorVecVec = CSignal::TMeanAccumulatorVecVec;
    using TMeanAccumulatorVecCRng = core::CVectorRange<const TMeanAccumulatorVecVec>;
    using TSegmentation = CTimeSeriesSegmentation;
    using TWeightFunc = TSegmentation::TWeightFunc;

    //! \brief Accumulates the minimum amplitude.
    //!
    //! This is used for finding and testing the significance of semi-periodic spikes
    //! (or dips), i.e. frequently repeated values that are much further from the mean
    //! than the datas' typical variation.
    class CMinAmplitude {
    public:
        static constexpr double INF = std::numeric_limits<double>::max();
        static constexpr std::size_t MINIMUM_REPEATS{5};

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

        //! Test the variance reduction of this hypothesis.
        CFuzzyTruthValue testVariance(const CTimeSeriesTestForSeasonality& params) const;
        //! Test the amplitude of this hypothesis.
        CFuzzyTruthValue testAmplitude(const CTimeSeriesTestForSeasonality& params) const;
        //! Get a readable description.
        std::string print() const;

        //! If true then this should be modelled.
        bool s_Model = false;
        //! A summary of the seasonal component period.
        TSeasonalComponent s_Period;
        //! The desired component size.
        std::size_t s_ComponentSize = 0;
        //! The number of segments in the trend.
        std::size_t s_NumberTrendSegments = 0;
        //! The number of scale segments in the component.
        std::size_t s_NumberScaleSegments = 0;
        //! The mean number of repeats of buckets with at least one measurement.
        double s_MeanNumberRepeats = 0.0;
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
        CFuzzyTruthValue s_Truth = CFuzzyTruthValue::FALSE;
        //! The segmentation into constant linear scales.
        TSizeVec s_ScaleSegments;
        //! The values to use to initialize the component.
        TFloatMeanAccumulatorVec s_InitialValues;
    };

    using THypothesisStatsVec = std::vector<SHypothesisStats>;

    //! \brief A candidate model for the values being decomposed.
    struct SModel {
        SModel() = default;
        SModel(double residualVariance,
               double truncatedResidualVariance,
               std::size_t numberTrendParameters,
               TFloatMeanAccumulatorVec trendInitialValues,
               THypothesisStatsVec hypotheses,
               TBoolVec removeComponentsMask)
            : s_ResidualVariance{residualVariance}, s_TruncatedResidualVariance{truncatedResidualVariance},
              s_NumberTrendParameters{numberTrendParameters},
              s_TrendInitialValues{std::move(trendInitialValues)},
              s_Hypotheses{std::move(hypotheses)}, s_RemoveComponentsMask{std::move(
                                                       removeComponentsMask)} {}

        //! Does this include seasonality?
        bool seasonal() const { return s_Hypotheses.size() > 0; }
        //! Should this behave as a null hypothesis?
        bool isNull(std::size_t numberValues) const;
        //! Should this behave as an alternative hypothesis?
        bool isAlternative(std::size_t numberValues) const;
        //! The number of degrees of freedom in the variance estimate.
        double degreesFreedom(std::size_t numberValues) const;
        //! The number of parameters in model.
        double numberParameters() const;
        //! The minimum mean number of repeats of any seasonal component.
        double meanRepeats() const;
        //! Get the variance explained per parameter weighted by the variance explained
        //! by each component of the model.
        double explainedVariancePerParameter(double explainedVariance) const;
        //! Get the average autocorrelation weighted by the variance explained by each
        //! component of the model.
        double autocorrelation() const;

        //! Are the seasonal components already modelled?
        bool s_AlreadyModelled = false;
        //! The residual variance after removing the seasonal components.
        double s_ResidualVariance = std::numeric_limits<double>::max();
        //! The residual variance after removing the seasonal components and outliers.
        double s_TruncatedResidualVariance = std::numeric_limits<double>::max();
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
    void addNotSeasonal(const TFloatMeanAccumulatorVec& valuesMinusTrend,
                        const TSizeVec& modelTrendSegments,
                        TModelVec& decompositions) const;
    void addModelled(const TFloatMeanAccumulatorVec& valuesMinusTrend,
                     const TSizeVec& modelTrendSegments,
                     TSeasonalComponentVec& periods,
                     TModelVec& decompositions) const;
    void addDiurnal(const TFloatMeanAccumulatorVec& valuesMinusTrend,
                    const TSizeVec& modelTrendSegments,
                    TSeasonalComponentVec& periods,
                    TModelVec& decompositions) const;
    void addDecomposition(const TFloatMeanAccumulatorVec& valuesMinusTrend,
                          const TSizeVec& modelTrendSegments,
                          TSeasonalComponentVec& periods,
                          TModelVec& decompositions) const;
    void testAndAddDecomposition(const TSeasonalComponentVec& periods,
                                 const TSizeVec& modelTrendSegments,
                                 const TFloatMeanAccumulatorVec& valuesToTest,
                                 TModelVec& decompositions,
                                 bool alreadyModelled) const;
    SModel testDecomposition(const TSeasonalComponentVec& periods,
                             std::size_t numberTrendSegments,
                             const TFloatMeanAccumulatorVec& valueToTest) const;
    bool acceptDecomposition(const SModel& decomposition) const;
    void updateResiduals(const SHypothesisStats& hypothesis,
                         TFloatMeanAccumulatorVec& residuals) const;
    TBoolVec finalizeHypotheses(const TFloatMeanAccumulatorVec& valuesToTest,
                                THypothesisStatsVec& hypotheses,
                                TFloatMeanAccumulatorVec& residuals) const;
    TBoolVec selectModelledHypotheses(THypothesisStatsVec& hypotheses) const;
    void removeModelledPredictions(const TBoolVec& componentsToRemoveMask,
                                   core_t::TTime startTime,
                                   TFloatMeanAccumulatorVec& values) const;
    void removeDiscontinuities(const TSizeVec& modelTrendSegments,
                               TFloatMeanAccumulatorVec& values) const;
    bool meanScale(const SHypothesisStats& hypothesis,
                   const TWeightFunc& weight,
                   TFloatMeanAccumulatorVec& values,
                   TDoubleVec& scales) const;
    void removePredictions(const TSeasonalComponentCRng& periodsToRemove,
                           const TMeanAccumulatorVecCRng& componentsToRemove,
                           TFloatMeanAccumulatorVec& values) const;
    void testExplainedVariance(const TVarianceStats& H0, SHypothesisStats& hypothesis) const;
    void testAutocorrelation(SHypothesisStats& hypothesis) const;
    void testAmplitude(SHypothesisStats& hypothesis) const;
    TVarianceStats residualVarianceStats(const TFloatMeanAccumulatorVec& values) const;
    double truncatedVariance(double outlierFraction,
                             const TFloatMeanAccumulatorVec& residuals) const;
    bool alreadyModelled(const TSeasonalComponentVec& periods) const;
    bool alreadyModelled(const TSeasonalComponent& period) const;
    bool onlyDiurnal(const TSeasonalComponentVec& periods) const;
    void removeIfNotTestable(TSeasonalComponentVec& periods) const;
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
                              const TSeasonalComponent& period);
    static std::size_t observedRange(const TFloatMeanAccumulatorVec& values);

private:
    double m_MinimumRepeatsPerSegmentToTestVariance = 3.0;
    double m_MinimumRepeatsPerSegmentToTestAmplitude = 5.0;
    double m_LowAutocorrelation = 0.3;
    double m_MediumAutocorrelation = 0.5;
    double m_HighAutocorrelation = 0.7;
    double m_SignificantPValue = 1e-3;
    double m_VerySignificantPValue = 1e-8;
    double m_AcceptedFalsePostiveRate = 1e-4;
    std::ptrdiff_t m_MaximumNumberComponents = 10;
    TOptionalSize m_StartOfWeekOverride;
    TOptionalTime m_MinimumPeriod;
    core_t::TTime m_StartTime = 0;
    core_t::TTime m_BucketLength = 0;
    double m_OutlierFraction = OUTLIER_FRACTION;
    double m_EpsVariance = 0.0;
    TPredictor m_ModelledPredictor = [](core_t::TTime, const TBoolVec&) {
        return 0.0;
    };
    TSeasonalComponentVec m_ModelledPeriods;
    TBoolVec m_ModelledPeriodsTestable;
    TFloatMeanAccumulatorVec m_Values;
    // The follow are member data to avoid repeatedly reinitialising.
    mutable TSizeVec m_WindowIndices;
    mutable TSeasonalComponentVec m_Periods;
    mutable TAmplitudeVec m_Amplitudes;
    mutable TMeanAccumulatorVecVec m_Components;
    mutable TFloatMeanAccumulatorVec m_ValuesToTest;
    mutable TFloatMeanAccumulatorVec m_TemporaryValues;
    mutable TMaxAccumulator m_Outliers;
    mutable TDoubleVec m_Scales;

    friend struct CTimeSeriesTestForSeasonalityTest::calibrateTruncatedVariance;
};
}
}

#endif //INCLUDED_ml_maths_CTimeSeriesTestForSeasonality_h
