/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CTimeSeriesTestForSeasonality_h
#define INCLUDED_ml_maths_CTimeSeriesTestForSeasonality_h

#include <core/CSmallVector.h>
#include <core/Constants.h>
#include <core/CoreTypes.h>

#include <maths/CBasicStatistics.h>
#include <maths/CFuzzyLogic.h>
#include <maths/CSignal.h>
#include <maths/ImportExport.h>

#include <boost/operators.hpp>
#include <boost/optional.hpp>

#include <algorithm>
#include <functional>
#include <limits>
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
    using TTimeFloatMeanAccumulatorPr = std::pair<core_t::TTime, TFloatMeanAccumulator>;

    //! \brief Iterate over the (time, value) to use to initialize the component.
    class CInitialValueConstIterator
        : std::iterator<std::forward_iterator_tag, const TTimeFloatMeanAccumulatorPr, std::ptrdiff_t> {
    public:
        CInitialValueConstIterator() = default;
        CInitialValueConstIterator(const CNewTrendSummary& summary, std::size_t index)
            : m_Summary{&summary}, m_Index{index} {
            m_Value.first = summary.m_StartTime + static_cast<core_t::TTime>(index) *
                                                      summary.m_BucketLength;
            m_Value.second = summary.m_InitialValues[index];
        }

        //! \name Forward Iterator Contract
        //@{
        bool operator==(const CInitialValueConstIterator& rhs) const {
            return m_Index == rhs.m_Index && m_Summary == rhs.m_Summary;
        }
        bool operator!=(const CInitialValueConstIterator& rhs) const {
            return m_Index != rhs.m_Index || m_Summary != rhs.m_Summary;
        }
        const TTimeFloatMeanAccumulatorPr& operator*() const { return m_Value; }
        const TTimeFloatMeanAccumulatorPr* operator->() const {
            return &m_Value;
        }
        CInitialValueConstIterator& operator++() {
            m_Value.first += m_Summary->m_BucketLength;
            m_Value.second = m_Summary->m_InitialValues[++m_Index];
            return *this;
        }
        CInitialValueConstIterator operator++(int) {
            CInitialValueConstIterator result{*this};
            this->operator++();
            return result;
        }
        //@}

    private:
        const CNewTrendSummary* m_Summary = nullptr;
        std::size_t m_Index = 0;
        TTimeFloatMeanAccumulatorPr m_Value;
    };

public:
    CNewTrendSummary(core_t::TTime startTime,
                     core_t::TTime bucketLength,
                     TFloatMeanAccumulatorVec initialValues);

    //! Get an iterator over the values with which to initialize the new trend.
    CInitialValueConstIterator beginInitialValues() const;

    //! Get an iterator to the end of the values with which to initialize the new trend.
    CInitialValueConstIterator endInitialValues() const;

private:
    core_t::TTime m_StartTime = 0;
    core_t::TTime m_BucketLength = 0;
    TFloatMeanAccumulatorVec m_InitialValues;
};

//! \brief A summary of a new seasonal component.
class MATHS_EXPORT CNewSeasonalComponentSummary {
public:
    using TSizeSizePr = CSignal::TSizeSizePr;
    using TSizeSizePr2Vec = CSignal::TSizeSizePr2Vec;
    using TSizeSizePr2VecCItr = TSizeSizePr2Vec::const_iterator;
    using TSeasonalComponent = CSignal::SSeasonalComponentSummary;
    using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
    using TFloatMeanAccumulatorVecCItr = TFloatMeanAccumulatorVec::const_iterator;
    using TTimeFloatMeanAccumulatorPr = std::pair<core_t::TTime, TFloatMeanAccumulator>;
    using TSeasonalTimeUPtr = std::unique_ptr<CSeasonalTime>;

    //! \brief Iterate over the (time, value) to use to initialize the component.
    class CInitialValueConstIterator
        : std::iterator<std::forward_iterator_tag, const TTimeFloatMeanAccumulatorPr, std::ptrdiff_t> {
    public:
        class CTimeFloatMeanAccumulatorPrCPtr {
        public:
            CTimeFloatMeanAccumulatorPrCPtr(const TTimeFloatMeanAccumulatorPr& value)
                : m_Value{value} {}
            const TTimeFloatMeanAccumulatorPr* operator->() const {
                return &m_Value;
            }

        private:
            TTimeFloatMeanAccumulatorPr m_Value;
        };

    public:
        static constexpr std::size_t END{std::numeric_limits<std::size_t>::max()};

    public:
        CInitialValueConstIterator() = default;
        CInitialValueConstIterator(const CNewSeasonalComponentSummary& summary,
                                   std::size_t index,
                                   TFloatMeanAccumulatorVecCItr value,
                                   TSizeSizePr2Vec windows)
            : m_Summary{&summary}, m_Index{index}, m_Value{value},
              m_CurrentWindow{this->window()}, m_Windows{std::move(windows)} {}

        //! \name Forward Iterator Contract
        //@{
        bool operator==(const CInitialValueConstIterator& rhs) const {
            return m_Value == rhs.m_Value;
        }
        bool operator!=(const CInitialValueConstIterator& rhs) const {
            return m_Value != rhs.m_Value;
        }
        TTimeFloatMeanAccumulatorPr operator*() const {
            std::size_t index{m_Index % m_Summary->m_InitialValues.size()};
            core_t::TTime offset{m_Summary->m_BucketLength *
                                 static_cast<core_t::TTime>(index)};
            return {m_Summary->m_StartTime + offset, m_Summary->m_InitialValues[index]};
        }
        CTimeFloatMeanAccumulatorPrCPtr operator->() const {
            return {this->operator*()};
        }
        CInitialValueConstIterator& operator++() {
            ++m_Index;
            ++m_Value;
            if (m_Index >= m_CurrentWindow->second) {
                m_CurrentWindow = this->window();
                m_Index = m_CurrentWindow != m_Windows.end()
                              ? CTools::truncate(m_Index, m_CurrentWindow->first,
                                                 m_CurrentWindow->second)
                              : END;
            }
            return *this;
        }
        CInitialValueConstIterator operator++(int) {
            CInitialValueConstIterator result{*this};
            this->operator++();
            return result;
        }
        //@}

    private:
        TSizeSizePr2VecCItr window() const {
            return std::upper_bound(m_Windows.begin(), m_Windows.end(), m_Index,
                                    [](std::size_t i, const TSizeSizePr& window) {
                                        return i < window.second;
                                    });
        }

    private:
        const CNewSeasonalComponentSummary* m_Summary = nullptr;
        std::size_t m_Index = 0;
        TFloatMeanAccumulatorVecCItr m_Value;
        TSizeSizePr2VecCItr m_CurrentWindow;
        TSizeSizePr2Vec m_Windows;
    };

public:
    CNewSeasonalComponentSummary(std::string annotationText,
                                 const TSeasonalComponent& period,
                                 std::size_t size,
                                 bool diurnal,
                                 core_t::TTime startTime,
                                 core_t::TTime bucketLength,
                                 TFloatMeanAccumulatorVec initialValues,
                                 double precedence);

    //! Get the annotation for this component.
    const std::string& annotationText() const;

    //! Get the desired component size.
    std::size_t size() const;

    //! Get a seasonal time for the specified results.
    //!
    //! \warning The caller owns the returned object.
    TSeasonalTimeUPtr seasonalTime() const;

    //! Get an iterator over the values with which to initialize the new component.
    CInitialValueConstIterator beginInitialValues() const;

    //! Get an iterator to the end of the values with which to initialize the new component.
    CInitialValueConstIterator endInitialValues() const;

    //! Get a description of the seasonal.
    std::string print() const;

private:
    std::string m_AnnotationText;
    TSeasonalComponent m_Period;
    std::size_t m_Size = 0;
    bool m_Diurnal = false;
    core_t::TTime m_StartTime = 0;
    core_t::TTime m_BucketLength = 0;
    TFloatMeanAccumulatorVec m_InitialValues;
    double m_Precedence = 0.0;
};

//! \brief Represents a collection of accepted seasonal hypotheses.
class MATHS_EXPORT CSeasonalHypotheses {
public:
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
             TFloatMeanAccumulatorVec initialValues,
             double precedence);

    //! Get the summary of any trend component.
    const CNewTrendSummary* trend() const;

    //! Get the summaries of any periodic components.
    const TNewSeasonalComponentVec& components() const;

    //! Get a description of the seasonal components.
    std::string print() const;

private:
    using TOptionalNewTrendSummary = boost::optional<CNewTrendSummary>;

private:
    //! The selected trend summary.
    TOptionalNewTrendSummary m_Trend;

    //! The selected seasonal components if any.
    TNewSeasonalComponentVec m_Components;
};

class MATHS_EXPORT CTimeSeriesTestForSeasonality {
public:
    using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;

public:
    static constexpr double MINIMUM_REPEATS_PER_SEGMENT_FOR_VARIANCE{3.0};
    static constexpr double MINIMUM_REPEATS_PER_SEGMENT_FOR_AMPLITUDE{5.0};
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

    CTimeSeriesTestForSeasonality& startOfWeek(core_t::TTime value) {
        m_StartOfWeek = static_cast<std::size_t>(
            ((core::constants::WEEK + value - (m_StartTime % core::constants::WEEK)) %
             core::constants::WEEK) /
            m_BucketLength);
        return *this;
    }
    CTimeSeriesTestForSeasonality& minimumRepeatsPerSegmentForVariance(double value) {
        m_MinimumRepeatsPerSegmentForVariance = value;
        return *this;
    }
    CTimeSeriesTestForSeasonality& minimumRepeatsPerSegmentForAmplitude(double value) {
        m_MinimumRepeatsPerSegmentForAmplitude = value;
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

    //! Register a seasonal component which is already being modelled.
    void addModelledSeasonality(const CSeasonalTime& period);

    //! Run the test and return the new components found if any.
    CSeasonalHypotheses decompose();

private:
    using TSizeSizePr = std::pair<std::size_t, std::size_t>;
    using TSizeVec = std::vector<std::size_t>;
    using TOptionalSize = boost::optional<std::size_t>;
    using TSeasonalComponent = CSignal::SSeasonalComponentSummary;
    using TSeasonalComponentVec = CSignal::TSeasonalComponentVec;

    //! \brief A summary of a test statistics related to a component.
    struct SHypothesisStats {
        explicit SHypothesisStats(const TSeasonalComponent& period)
            : s_Period{period} {}
        //! A summary of the seasonal component period.
        TSeasonalComponent s_Period;
        //! The desired component size.
        std::size_t s_ComponentSize = 0;
        //! The scale to apply to the component's initial values.
        double s_ComponentInitialValuesScale;
        //! The number of segments in the trend.
        std::size_t s_NumberTrendSegments = 0;
        //! The number of scale segments in the component.
        std::size_t s_NumberScaleSegments = 0;
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
        //! The truth value for the hypothesis.
        CFuzzyTruthValue s_Truth = CFuzzyTruthValue::FALSE;
        //! The values to use to initialize the component.
        TFloatMeanAccumulatorVec s_InitialValues;
    };

    using THypothesisStatsVec = std::vector<SHypothesisStats>;
    using TFloatMeanAccumulatorVecHypothesisStatsVecPr =
        std::pair<TFloatMeanAccumulatorVec, THypothesisStatsVec>;
    using TFloatMeanAccumulatorVecHypothesisStatsVecPrVec =
        std::vector<TFloatMeanAccumulatorVecHypothesisStatsVecPr>;

private:
    CSeasonalHypotheses select(TFloatMeanAccumulatorVecHypothesisStatsVecPrVec& hypotheses) const;
    void truth(SHypothesisStats& hypothesis) const;
    TFloatMeanAccumulatorVecHypothesisStatsVecPr
    testDecomposition(TFloatMeanAccumulatorVec& valueToTest,
                      std::size_t numberTrendSegments,
                      const TSeasonalComponentVec& periods) const;
    void updateInitialValues(const TSeasonalComponent& period,
                             SHypothesisStats& hypothesis,
                             TFloatMeanAccumulatorVec& trendInitialValues) const;
    void testExplainedVariance(const TFloatMeanAccumulatorVec& valueToTest,
                               const TSeasonalComponent& period,
                               SHypothesisStats& hypothesis) const;
    void testAutocorrelation(const TFloatMeanAccumulatorVec& valuesToTest,
                             const TSeasonalComponent& period,
                             SHypothesisStats& hypothesis) const;
    void testAmplitude(const TFloatMeanAccumulatorVec& valueToTest,
                       const TSeasonalComponent& period,
                       SHypothesisStats& hypothesis) const;
    void appendDiurnalComponents(TFloatMeanAccumulatorVec& valuesToTest,
                                 TSeasonalComponentVec& periods) const;
    bool isDiurnal(std::size_t period) const;
    bool isDiurnal(const TSeasonalComponent& period) const;
    bool isWeekend(const TSeasonalComponent& period) const;
    bool isWeekday(const TSeasonalComponent& period) const;
    bool seenSufficientData(const TSeasonalComponent& period) const;
    bool seenSufficientDataToTestForTradingDayDecomposition() const;
    double precedence(const TSeasonalComponent& period) const;
    std::string annotationText(const TSeasonalComponent& period) const;
    std::size_t day() const;
    std::size_t week() const;
    std::size_t year() const;
    TSizeSizePr weekdayWindow() const;
    TSizeSizePr weekendWindow() const;
    std::size_t observedRange() const;

private:
    double m_MinimumRepeatsPerSegmentForVariance = MINIMUM_REPEATS_PER_SEGMENT_FOR_VARIANCE;
    double m_MinimumRepeatsPerSegmentForAmplitude = MINIMUM_REPEATS_PER_SEGMENT_FOR_AMPLITUDE;
    double m_MinimumAutocorrelation = MINIMUM_AUTOCORRELATION;
    double m_MinimumExplainedVariance = MINIMUM_EXPLAINED_VARIANCE;
    double m_MaximumExplainedVariancePValue = MAXIMUM_EXPLAINED_VARIANCE_PVALUE;
    double m_MaximumAmplitudePValue = MAXIMUM_AMPLITUDE_PVALUE;
    TOptionalSize m_StartOfWeek;
    core_t::TTime m_StartTime = 0;
    core_t::TTime m_BucketLength = 0;
    double m_OutlierFraction = OUTLIER_FRACTION;
    TFloatMeanAccumulatorVec m_Values;
    TSizeVec m_ModelledPeriods;
};
}
}

#endif //INCLUDED_ml_maths_CTimeSeriesTestForSeasonality_h
