/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CTimeSeriesTestForChange_h
#define INCLUDED_ml_maths_CTimeSeriesTestForChange_h

#include <core/Constants.h>
#include <core/CoreTypes.h>

#include <maths/CBasicStatistics.h>
#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <functional>
#include <memory>
#include <tuple>
#include <vector>

namespace ml {
namespace maths {
class CCalendarComponent;
class CSeasonalComponent;
class CTimeSeriesDecomposition;
class CTrendComponent;

//! \brief Represents a sudden change to a time series model.
class MATHS_EXPORT CChangePoint {
public:
    using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;

public:
    CChangePoint(std::size_t index, core_t::TTime time, TFloatMeanAccumulatorVec initialValues)
        : m_Index{index}, m_Time{time}, m_InitialValues{std::move(initialValues)} {}
    virtual ~CChangePoint() = default;

    virtual bool apply(CTimeSeriesDecomposition&) const { return false; }
    virtual bool apply(CTrendComponent&) const { return false; }
    virtual bool apply(CSeasonalComponent&) const { return false; }
    virtual bool apply(CCalendarComponent&) const { return false; }
    virtual bool largeEnough(double threshold) const = 0;
    bool longEnough(core_t::TTime time, core_t::TTime minimumDuration) const {
        return time > m_Time + minimumDuration;
    }
    virtual const std::string& type() const = 0;
    virtual double value() const = 0;
    virtual std::string print() const = 0;
    std::size_t index() const { return m_Index; }
    core_t::TTime time() const { return m_Time; }
    const TFloatMeanAccumulatorVec& initialValues() const {
        return m_InitialValues;
    }

private:
    std::size_t m_Index = 0;
    core_t::TTime m_Time = 0;
    TFloatMeanAccumulatorVec m_InitialValues;
};

//! \brief Represents a level shift of a time series.
class MATHS_EXPORT CLevelShift : public CChangePoint {
public:
    static const std::string TYPE;

public:
    CLevelShift(std::size_t index, core_t::TTime time, double valueAtShift, double shift, TFloatMeanAccumulatorVec initialValues)
        : CChangePoint{index, time, std::move(initialValues)}, m_Shift{shift}, m_ValueAtShift{valueAtShift} {
    }

    bool apply(CTrendComponent& component) const override;
    bool largeEnough(double threshold) const override {
        return m_Shift > threshold;
    }
    const std::string& type() const override;
    std::string print() const override;
    double value() const override { return m_Shift; }

private:
    double m_Shift = 0.0;
    double m_ValueAtShift = 0.0;
};

//! \brief Represents a linear scale of a time series.
class MATHS_EXPORT CScale : public CChangePoint {
public:
    static const std::string TYPE;

public:
    CScale(std::size_t index, core_t::TTime time, double scale, double magnitude, TFloatMeanAccumulatorVec initialValues)
        : CChangePoint{index, time, std::move(initialValues)}, m_Scale{scale}, m_Magnitude{magnitude} {
    }

    bool apply(CTrendComponent& component) const override;
    bool apply(CSeasonalComponent& component) const override;
    bool apply(CCalendarComponent& component) const override;
    bool largeEnough(double threshold) const override {
        return m_Magnitude > threshold;
    }
    const std::string& type() const override;
    std::string print() const override;
    double value() const override { return m_Scale; }

private:
    double m_Scale = 1.0;
    double m_Magnitude = 0.0;
};

//! \brief Represents a time shift of a time series.
class MATHS_EXPORT CTimeShift : public CChangePoint {
public:
    static const std::string TYPE;

public:
    CTimeShift(std::size_t index, core_t::TTime time, core_t::TTime shift, TFloatMeanAccumulatorVec initialValues)
        : CChangePoint{index, time, std::move(initialValues)}, m_Shift{shift} {}

    bool apply(CTimeSeriesDecomposition& decomposition) const override;
    bool largeEnough(double) const override { return m_Shift > 0; }
    const std::string& type() const override;
    std::string print() const override;
    double value() const override { return static_cast<double>(m_Shift); }

private:
    core_t::TTime m_Shift = 0;
};

//! \brief Test for sudden changes or shocks to a time series.
//!
//! DESCRIPTION\n
//! This checks a window of bucketed average samples values looking for sudden changes.
//! It tests for level shift, scaling and time shift events. If the variance explained
//! by any hypothesis is statistically significant vs a null hypothesis that there
//! is a smooth trend then it is considered for selection. We test candidate hypotheses
//! in complexity order. A hypothesis will be selected if it explains a statistically
//! significant additional proportion of the variance to an already selected hypothesis
//! otherwise we fallback to minimizing the Akaike Information Criterion assuming residual
//! errors are normally distributed. In practice this is a reasonable assumption since
//! they are typically the average of multiple individual samples and so the CLT comes
//! into play.
//!
//! Some noteworthy features:
//!   -# If the outlier fraction is greater than zero then change models are fitted
//!      reweighting this proportion of the data (assuming there is strong evidence
//!      they come from a different distribution). P-values are computed with and
//!      without removing outliers and the minimum is selected. The reason for using
//!      both is that spikes in a seasonal pattern will often be treated as outliers
//!      if the model is a poor fit. This significantly affects the test power if the
//!      null is unable to fit them.
//!   -# If sample variance is supplied any explained variance has to be significant
//!      on the order of the sample variance.
class MATHS_EXPORT CTimeSeriesTestForChange {
public:
    using TBoolVec = std::vector<bool>;
    using TPredictor = std::function<double(core_t::TTime)>;
    using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
    using TChangePointUPtr = std::unique_ptr<CChangePoint>;

public:
    static constexpr double OUTLIER_FRACTION = 0.1;

public:
    //! \param[in] valuesStartTime The average offset of samples in each time bucket.
    //! \param[in] bucketStartTime The start first time bucket.
    //! \param[in] bucketLength The length of the time buckets.
    //! \param[in] sampleInterval The interval between samples of the time series.
    //! \param[in] predictor The current model of the time series.
    //! \param[in] values The average of values falling in each time bucket.
    //! \param[in] sampleVariance The residual variance of the samples after removing
    //! predictions.
    //! \param[in] outlierFraction The proportion of values to treat as outliers.
    //! This must be in the range (0.0, 1.0).
    CTimeSeriesTestForChange(core_t::TTime valuesStartTime,
                             core_t::TTime bucketsStartTime,
                             core_t::TTime bucketLength,
                             core_t::TTime sampleInterval,
                             TPredictor predictor,
                             TFloatMeanAccumulatorVec values,
                             double sampleVariance = 0.0,
                             double outlierFraction = OUTLIER_FRACTION);

    //! Test the values supplied to the constructor for a change.
    TChangePointUPtr test() const;

    //! \name Parameters
    //@{
    CTimeSeriesTestForChange& significantPValue(double value) {
        m_SignificantPValue = value;
        return *this;
    }
    CTimeSeriesTestForChange& acceptedFalsePostiveRate(double value) {
        m_AcceptedFalsePostiveRate = value;
        return *this;
    }
    //@}

private:
    using TDoubleDoublePr = std::pair<double, double>;
    using TDoubleDoubleDoubleTr = std::tuple<double, double, double>;
    using TSizeVec = std::vector<std::size_t>;
    using TMaxAccumulator =
        CBasicStatistics::COrderStatisticsHeap<double, std::greater<double>>;
    using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using TBucketPredictor = std::function<double(std::size_t)>;
    using TTransform = std::function<double(const TFloatMeanAccumulator&)>;

    enum EType { E_NoChangePoint, E_LevelShift, E_Scale, E_TimeShift };

    struct SChangePoint {
        SChangePoint() = default;
        SChangePoint(EType type,
                     std::size_t index,
                     double residualVariance,
                     double truncatedResidualVariance,
                     double numberParameters,
                     TFloatMeanAccumulatorVec initialValues)
            : s_Type{type}, s_Index{index}, s_ResidualVariance{residualVariance},
              s_TruncatedResidualVariance{truncatedResidualVariance},
              s_NumberParameters{numberParameters}, s_InitialValues{initialValues} {}

        EType s_Type = E_NoChangePoint;
        std::size_t s_Index = 0;
        double s_ResidualVariance = 0.0;
        double s_TruncatedResidualVariance = 0.0;
        double s_NumberParameters = 0.0;
        double s_LevelShift = 0.0;
        double s_Scale = 0.0;
        double s_ScaleMagnitude = 0.0;
        core_t::TTime s_TimeShift = 0;
        TFloatMeanAccumulatorVec s_InitialValues;
    };

private:
    TDoubleDoubleDoubleTr quadraticTrend() const;
    SChangePoint levelShift(double varianceH0, double truncatedVarianceH0, double parametersH0) const;
    SChangePoint scale(double varianceH0, double truncatedVarianceH0, double parametersH0) const;
    SChangePoint timeShift(double varianceH0, double truncatedVarianceH0, double parametersH0) const;
    TBucketPredictor bucketPredictor() const;
    TMeanVarAccumulator truncatedMoments(double outlierFraction,
                                         const TFloatMeanAccumulatorVec& residuals,
                                         const TTransform& transform = mean) const;
    core_t::TTime changeTime(std::size_t changeIndex) const;
    double valueAtChange(std::size_t changeIndex) const;
    TDoubleDoublePr variances(const TFloatMeanAccumulatorVec& residuals) const;
    double pValue(double varianceH0,
                  double truncatedVarianceH0,
                  double parametersH0,
                  double varianceH1,
                  double truncatedVarianceH1,
                  double parametersH1,
                  double n) const;
    static TFloatMeanAccumulatorVec removePredictions(const TBucketPredictor& predictor,
                                                      TFloatMeanAccumulatorVec values);
    static std::size_t buckets(core_t::TTime bucketLength, core_t::TTime interval);
    static double aic(const SChangePoint& change);
    static double mean(const TFloatMeanAccumulator& value) {
        return CBasicStatistics::mean(value);
    }
    static const std::string& print(EType type);

private:
    double m_SignificantPValue = 1e-3;
    double m_AcceptedFalsePostiveRate = 1e-4;
    core_t::TTime m_ValuesStartTime = 0;
    core_t::TTime m_BucketsStartTime = 0;
    core_t::TTime m_BucketLength = 0;
    core_t::TTime m_SampleInterval = 0;
    double m_SampleVariance = 0.0;
    double m_OutlierFraction = OUTLIER_FRACTION;
    double m_EpsVariance = 0.0;
    TPredictor m_Predictor = [](core_t::TTime) { return 0.0; };
    TFloatMeanAccumulatorVec m_Values;
    // The follow are member data to avoid repeatedly reinitialising.
    mutable TFloatMeanAccumulatorVec m_ValuesMinusPredictions;
    mutable TMaxAccumulator m_Outliers;
};
}
}

#endif // INCLUDED_ml_maths_CTimeSeriesTestForChange_h
