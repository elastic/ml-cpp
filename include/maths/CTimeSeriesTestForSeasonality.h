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

//! \brief A summary of the trend on the test window.
class MATHS_EXPORT CNewTrendSummary final {
public:
    using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
    using TTimeFloatMeanAccumulatorPr = std::pair<core_t::TTime, TFloatMeanAccumulator>;

    //! \brief Iterate over the (time, value) to use to initialize the component.
    class MATHS_EXPORT CInitialValueConstIterator
        : std::iterator<std::forward_iterator_tag, const TTimeFloatMeanAccumulatorPr, std::ptrdiff_t> {
    public:
        CInitialValueConstIterator() = default;
        CInitialValueConstIterator(std::size_t index, const CNewTrendSummary& summary)
            : m_Index{index}, m_Summary{&summary} {
            m_Value.first = summary.m_StartOfInitialValues +
                            static_cast<core_t::TTime>(index) * summary.m_InitialValuesInterval;
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
            m_Value.first += m_Summary->m_InitialValuesInterval;
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
        std::size_t m_Index = 0;
        const CNewTrendSummary* m_Summary = nullptr;
        TTimeFloatMeanAccumulatorPr m_Value;
    };

public:
    CNewTrendSummary(core_t::TTime startOfInitialValues,
                     core_t::TTime initialValuesInterval,
                     TFloatMeanAccumulatorVec initialValues);

    //! Get an iterator over the values with which to initialize the new trend.
    CInitialValueConstIterator beginInitialValues() const;

    //! Get an iterator to the end of the values with which to initialize the new trend.
    CInitialValueConstIterator endInitialValues() const;

private:
    core_t::TTime m_StartOfInitialValues = 0;
    core_t::TTime m_InitialValuesInterval = 0;
    TFloatMeanAccumulatorVec m_InitialValues;
};

//! \brief A summary of a new seasonal component.
class MATHS_EXPORT CNewSeasonalComponentSummary {
public:
    using TTimeTimePr = std::pair<core_t::TTime, core_t::TTime>;
    using TTimeTimePrVec = std::vector<TTimeTimePr>;
    using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
    using TTimeFloatMeanAccumulatorPr = std::pair<core_t::TTime, TFloatMeanAccumulator>;
    using TSeasonalTimeUPtr = std::unique_ptr<CSeasonalTime>;

    //! \brief Iterate over the (time, value) to use to initialize the component.
    class MATHS_EXPORT CInitialValueConstIterator
        : std::iterator<std::forward_iterator_tag, const TTimeFloatMeanAccumulatorPr, std::ptrdiff_t> {
    public:
        CInitialValueConstIterator() = default;
        CInitialValueConstIterator(std::size_t index, const CNewSeasonalComponentSummary& summary)
            : m_Index{index}, m_Summary{&summary} {
            // TODO
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
            do {
                m_Value.first += m_Summary->m_InitialValuesInterval;
            } while (this->inWindow(m_Value.first) == false);
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
        core_t::TTime inWindow(core_t::TTime time) const {
            time = (time - m_Summary->m_InitialValuesInterval) % m_Summary->m_WindowRepeat;
            return time >= m_Summary->m_Window.first &&
                   time < m_Summary->m_Window.second;
        }

    private:
        std::size_t m_Index = 0;
        const CNewSeasonalComponentSummary* m_Summary = nullptr;
        TTimeFloatMeanAccumulatorPr m_Value;
    };

public:
    CNewSeasonalComponentSummary(const std::string& description,
                                 std::size_t size,
                                 bool diurnal,
                                 const TTimeTimePr& window,
                                 core_t::TTime windowRepeat,
                                 core_t::TTime period,
                                 core_t::TTime startOfWeek,
                                 core_t::TTime startOfInitialValues,
                                 core_t::TTime initialValuesInterval,
                                 TFloatMeanAccumulatorVec initialValues);

    //! Get a description of this component.
    const std::string& description() const;

    //! Get a seasonal time for the specified results.
    //!
    //! \warning The caller owns the returned object.
    TSeasonalTimeUPtr seasonalTime() const;

    //! Get an iterator over the values with which to initialize the new component.
    CInitialValueConstIterator beginInitialValues() const;

    //! Get an iterator to the end of the values with which to initialize the new component.
    CInitialValueConstIterator endInitialValues() const;

private:
    std::string m_Description;
    std::size_t m_Size = 0;
    bool m_Diurnal = false;
    TTimeTimePr m_Window;
    core_t::TTime m_WindowRepeat = 0;
    core_t::TTime m_Period = 0;
    core_t::TTime m_StartOfWeek = 0;
    core_t::TTime m_StartOfInitialValues = 0;
    core_t::TTime m_InitialValuesInterval = 0;
    TFloatMeanAccumulatorVec m_InitialValues;
};

//! \brief Represents a collection of accepted seasonal hypotheses.
class MATHS_EXPORT CSeasonalHypotheses {
public:
    using TSizeVec = std::vector<std::size_t>;
    using TTimeTimePr = std::pair<core_t::TTime, core_t::TTime>;
    using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
    using TNewSeasonalComponentVec = std::vector<CNewSeasonalComponentSummary>;

public:
    //! Set the new trend.
    void trend(CNewTrendSummary value);

    //! Add a new seasonal component.
    void add(const std::string& description,
             std::size_t size,
             bool diurnal,
             const TTimeTimePr& window,
             core_t::TTime windowRepeat,
             core_t::TTime period,
             core_t::TTime startOfWeek,
             core_t::TTime startOfInitialValues,
             core_t::TTime initialValuesInterval,
             TFloatMeanAccumulatorVec initialValues);

    //! Get the binary representation of the periodic components.
    const TNewSeasonalComponentVec& components() const;

private:
    //! The selected trend summary.
    CNewTrendSummary m_Trend;

    //! The selected seasonal components if any.
    TNewSeasonalComponentVec m_Components;
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

    //! Run the test and return the new components found if any.
    CSeasonalHypotheses test();

private:
    using TSizeSizePr = std::pair<std::size_t, std::size_t>;
    using TSizeVec = std::vector<std::size_t>;
    using TOptionalTime = boost::optional<core_t::TTime>;
    using TSeasonalComponent = CSignal::SSeasonalComponentSummary;
    using TSeasonalComponentVec = CSignal::TSeasonalComponentVec;

    //! \brief A summary of a test statistics related to a component.
    struct SHypothesisStats {
        explicit SHypothesisStats(const TSeasonalComponent& component)
            : s_Component{component} {}
        //! A summary of the seasonal component.
        TSeasonalComponent s_Component;
        //! The number of segments in the trend.
        std::size_t s_NumberTrendSegments = 0;
        //! The number of scale segments in the component.
        std::size_t s_NumberScaleSegments;
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
    using TSizeVecHypothesisStatsVecPrVec =
        std::vector<std::pair<TSizeVec, THypothesisStatsVec>>;

private:
    CSeasonalHypotheses select(TSizeVecHypothesisStatsVecPrVec& hypotheses) const;
    void truth(SHypothesisStats& hypothesis) const;
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
