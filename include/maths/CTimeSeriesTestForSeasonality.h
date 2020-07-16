/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CTimeSeriesTestForSeasonality_h
#define INCLUDED_ml_maths_CTimeSeriesTestForSeasonality_h

#include <core/CSmallVector.h>
#include <core/CoreTypes.h>

#include <maths/CBasicStatistics.h>
#include <maths/CFuzzyLogic.h>
#include <maths/CSignal.h>
#include <maths/ImportExport.h>

#include <boost/operators.hpp>
#include <boost/optional.hpp>

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace ml {
namespace maths {

class CSeasonalTime;

//! \brief Represents an accepted trend hypothesis.
class MATHS_EXPORT CTrendHypothesis final {
public:
    enum EType { E_None, E_Linear, E_PiecewiseLinear };

public:
    explicit CTrendHypothesis(std::size_t segments = 0);
    EType type() const;
    std::size_t segments() const;

private:
    std::size_t m_Segments;
};

//! \brief Component data.
struct MATHS_EXPORT SSeasonalComponentSummary {
    using TSizeSizePr = std::pair<std::size_t, std::size_t>;

    SSeasonalComponentSummary() = default;
    SSeasonalComponentSummary(const std::string& description,
                              bool diurnal,
                              bool piecewiseScaled,
                              std::size_t startOfWeek,
                              std::size_t period,
                              const TSizeSizePr& window);

    //! Check if this is equal to \p other.
    bool operator==(const SSeasonalComponentSummary& other) const;

    //! Get a seasonal time for the specified results.
    //!
    //! \warning The caller owns the returned object.
    CSeasonalTime* seasonalTime() const;

    //! An identifier for the component used by the test.
    std::string s_Description;
    //! True if this is a diurnal component false otherwise.
    bool s_Diurnal = false;
    //! The segmentation of the window into intervals of constant scaling.
    bool s_PiecewiseScaled = false;
    //! The start of the week if decomposing into trading days and weekend.
    std::size_t s_StartOfWeek = 0;
    //! The period of the component.
    std::size_t s_Period = 0;
    //! The window offset relative to the start of the repeat.
    TSizeSizePr s_Window;
};

//! \brief Represents a collection of accepted seasonal hypotheses.
class MATHS_EXPORT CSeasonalHypotheses final
    : boost::equality_comparable<CSeasonalHypotheses> {
public:
    using TTimeTimePr = std::pair<core_t::TTime, core_t::TTime>;
    using TSizeVec = std::vector<std::size_t>;
    using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
    using TSeasonalComponentVec = std::vector<SSeasonalComponentSummary>;

public:
    //! Check if this is equal to \p other.
    bool operator==(const CSeasonalHypotheses& other) const;

    //! Add a component.
    void add(const std::string& description,
             bool diurnal,
             bool piecewiseScaled,
             core_t::TTime startOfWeek,
             core_t::TTime period,
             const TTimeTimePr& window);

    //! Set the type of trend the hypothesis assumes.
    void trend(CTrendHypothesis value);

    //! Fit and remove the appropriate trend from \p values.
    void removeTrendFrom(TFloatMeanAccumulatorVec& values) const;

    //! Remove any discontinuities in \p values.
    void removeDiscontinuitiesFrom(TFloatMeanAccumulatorVec& values) const;

    //! Check if there are any periodic components.
    bool isSeasonal() const;

    //! Get the binary representation of the periodic components.
    const TSeasonalComponentVec& components() const;

    //! Get a human readable description of the result.
    std::string print() const;

private:
    //! The selected trend hypothesis.
    CTrendHypothesis m_Trend;

    //! The selected seasonal components if any.
    TSeasonalComponentVec m_Components;
};

class MATHS_EXPORT CTimeSeriesTestForSeasonality {
public:
    using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;

    static constexpr double MINIMUM_NUMBER_REPEATS_PER_SEGMENT{3.0};
    static constexpr double MINIMUM_AUTOCORRELATION{0.1};
    static constexpr double MINIMUM_EXPLAINED_VARIANCE{0.1};
    static constexpr double MAXIMUM_EXPLAINED_VARIANCE_PVALUE{1e-3};
    static constexpr double MAXIMUM_AMPLITUDE_PVALUE{1e-4};
    static constexpr double OUTLIER_FRACTION{0.1};

public:
    CTimeSeriesTestForSeasonality(core_t::TTime startTime,
                                  core_t::TTime bucketLength,
                                  TFloatMeanAccumulatorVec values,
                                  double outlierFraction = OUTLIER_FRACTION);

    //! Run the test and return the 
    CSeasonalHypotheses test();

    CTimeSeriesTestForSeasonality& startOfWeek(core_t::TTime value) {
        m_StartOfWeek = value;
        return *this;
    }
    CTimeSeriesTestForSeasonality& minimumNumberRepeatsPerSegment(double value) {
        m_MinimumNumberRepeatsPerSegment = value;
        return *this;
    }
    CTimeSeriesTestForSeasonality& minimumAutocorrelation(double value) {
        m_MinimumAutocorrelation = value;
        return *this;
    }
    CTimeSeriesTestForSeasonality& minimumExplainedVariance(double value) {
        m_MinimumExplainedVariance = value;
        return *this;
    }
    CTimeSeriesTestForSeasonality& maximumExplainedVariancePValue(double value) {
        m_MaximumExplainedVariancePValue = value;
        return *this;
    }
    CTimeSeriesTestForSeasonality& maximumAmplitudePValue(double value) {
        m_MaximumAmplitudePValue = value;
        return *this;
    }

private:
    using TSizeSizePr = std::pair<std::size_t, std::size_t>;
    using TSizeVec = std::vector<std::size_t>;
    using TOptionalTime = boost::optional<core_t::TTime>;
    using TSeasonalComponent = CSignal::SSeasonalComponentSummary;
    using TSeasonalComponentVec = CSignal::TSeasonalComponentVec;

    struct SHypothesisStats {
        //! The number of segments in the trend.
        std::size_t s_NumberTrendSegments = 0;
        //! True if a known repeating partition is tested.
        bool s_HasWeekend = false;
        //! The start of the repeating week if partitioned into trading days
        //! and weekeds.
        std::size_t s_StartOfWeek = 0;
        //! The segmentation of the interval if any.
        TSizeVec s_ScaleSegments;
        //! The mean number of repeats of buckets with at least one measurement.
        double s_MeanNumberRepeats = 0.0;
        //! The autocorrelation estimate of the hypothesis.
        double s_Autocorrelation = 0.0;
        //! The residual variance after removing this component.
        double s_ResidualVariance = 0.0;
        //! The variance estimate of hypothesis.
        double s_ExplainedVariance = 0.0;
        //! The explained variance significance.
        double s_ExplainedVariancePValue = 0.0;
        //! The amplitude significance.
        double s_AmplitudePValue = 0.0;
        //! The truth value for hypothesis.
        CFuzzyTruthValue s_TruthValue = CFuzzyTruthValue::FALSE;
    };

    using THypothesisStatsVec = std::vector<SHypothesisStats>;
    using TSizeVecHypothesisStatsVecPrVec = std::vector<std::pair<TSizeVec, THypothesisStatsVec>>;

private:
    CSeasonalHypotheses select(const TSizeVecHypothesisStatsVecPrVec& hypotheses) const;
    void truth(const SHypothesisStats& hypothesis) const;
    THypothesisStatsVec testDecomposition(TFloatMeanAccumulatorVec& valueToTest,
                                          std::size_t numberTrendSegments,
                                          const TSeasonalComponentVec& periods) const;
    void testExplainedVariance(const TFloatMeanAccumulatorVec& valueToTest,
                               const TSeasonalComponent& period,
                               SHypothesisStats& hypothesis) const;
    void testAutocorrelation(const TFloatMeanAccumulatorVec& valuesToTest,
                             const TSeasonalComponent& period,
                             SHypothesisStats& hypothesis) const;
    void testAmplitude(const TFloatMeanAccumulatorVec& valueToTest,
                       const TSeasonalComponent& period,
                       SHypothesisStats& hypothesis) const;
    void appendDiurnalComponents(const TFloatMeanAccumulatorVec& valuesToTest,
                                 TSeasonalComponentVec& periods) const;
    bool isDiurnal(std::size_t period) const;
    bool dividesDiurnal(std::size_t period) const;
    std::size_t day() const;
    std::size_t week() const;
    std::size_t year() const;
    TSizeSizePr weekdayWindow() const;
    TSizeSizePr weekendWindow() const;
    bool seenSufficientData(std::size_t period) const;
    std::size_t observedRange() const;
    std::string describe(std::size_t period) const;

private:
    double m_MinimumNumberRepeatsPerSegment = MINIMUM_NUMBER_REPEATS_PER_SEGMENT;
    double m_MinimumAutocorrelation = MINIMUM_AUTOCORRELATION;
    double m_MinimumExplainedVariance = MINIMUM_EXPLAINED_VARIANCE;
    double m_MaximumExplainedVariancePValue = MAXIMUM_EXPLAINED_VARIANCE_PVALUE;
    double m_MaximumAmplitudePValue = MAXIMUM_AMPLITUDE_PVALUE;
    TOptionalTime m_StartOfWeek;
    core_t::TTime m_StartTime = 0;
    core_t::TTime m_BucketLength = 0;
    double m_OutlierFraction = OUTLIER_FRACTION;
    TFloatMeanAccumulatorVec m_Values;
};
}
}

#endif //INCLUDED_ml_maths_CTimeSeriesTestForSeasonality_h
