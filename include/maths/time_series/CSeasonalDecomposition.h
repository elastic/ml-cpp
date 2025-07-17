/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#ifndef INCLUDED_ml_maths_time_series_CSeasonalDecomposition_h
#define INCLUDED_ml_maths_time_series_CSeasonalDecomposition_h

#include <maths/time_series/CTimeSeriesDecompositionBase.h>
#include <maths/time_series/CTimeSeriesDecompositionDetail.h>
#include <maths/time_series/ImportExport.h>
#include <maths/common/Constants.h>

namespace ml {
namespace maths {
namespace time_series {

//! \brief Implements time series decomposition focused on seasonal components
//!
//! DESCRIPTION:\n
//! This class specializes in detecting and modeling seasonal components in a time series.
//! It detects patterns on daily, weekly, weekend/weekday, and yearly time scales,
//! providing methods to predict values based on these seasonal patterns.
class MATHS_TIME_SERIES_EXPORT EMPTY_BASE_OPT CSeasonalDecomposition
    : public CTimeSeriesDecompositionBase {
public:
    //! \param[in] decayRate The rate at which information is lost.
    //! \param[in] bucketLength The data bucketing length.
    //! \param[in] seasonalComponentSize The number of buckets to use to estimate a
    //! seasonal component.
    explicit CSeasonalDecomposition(double decayRate = 0.0, 
                                    core_t::TTime bucketLength = 0,
                                    std::size_t seasonalComponentSize = common::COMPONENT_SIZE);

    //! Construct from part of a state document.
    CSeasonalDecomposition(const common::STimeSeriesDecompositionRestoreParams& params,
                           core::CStateRestoreTraverser& traverser);

    //! Deep copy constructor.
    CSeasonalDecomposition(const CSeasonalDecomposition& other, 
                          bool isForForecast = false);

    //! Efficient swap the state of this and \p other.
    void swap(CSeasonalDecomposition& other);

    //! Assignment operator.
    CSeasonalDecomposition& operator=(const CSeasonalDecomposition& other);

    //! Persist state by passing information to the supplied inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const override;

    //! Clone this decomposition.
    CSeasonalDecomposition* clone(bool isForForecast = false) const override;

    //! Set the data type.
    void dataType(maths_t::EDataType dataType) override;

    //! Set the decay rate.
    void decayRate(double decayRate) override;

    //! Get the decay rate.
    double decayRate() const override;

    //! Check if the decomposition has any initialized components.
    bool initialized() const override;

    //! Adds a time series point \f$(t, f(t))\f$.
    void addPoint(core_t::TTime time,
                  double value,
                  const core::CMemoryCircuitBreaker& allocator = core::CMemoryCircuitBreakerStub::instance(),
                  const maths_t::TDoubleWeightsAry& weights = TWeights::UNIT,
                  const TComponentChangeCallback& componentChangeCallback = noopComponentChange,
                  const maths_t::TModelAnnotationCallback& modelAnnotationCallback = noopModelAnnotation,
                  double occupancy = 1.0,
                  core_t::TTime firstValueTime = MIN_TIME) override;

    //! Shift seasonality by \p shift at \p time.
    void shiftTime(core_t::TTime time, core_t::TTime shift) override;

    //! Propagate the seasonal components forwards to \p time.
    void propagateForwardsTo(core_t::TTime time) override;

    //! Get the mean value of the time series in the vicinity of \p time.
    double meanValue(core_t::TTime time) const override;

    //! Get the predicted value of the time series at \p time.
    TVector2x1 value(core_t::TTime time, double confidence, bool isNonNegative) const override;

    //! Get the maximum interval for which the time series can be forecast.
    core_t::TTime maximumForecastInterval() const override;

    //! Forecast from \p start to \p end at \p dt intervals.
    void forecast(core_t::TTime startTime,
                  core_t::TTime endTime,
                  core_t::TTime step,
                  double confidence,
                  double minimumScale,
                  bool isNonNegative,
                  const TWriteForecastResult& writer) override;

    //! Remove the seasonal prediction at \p time from \p value.
    double detrend(core_t::TTime time,
                   double value,
                   double confidence,
                   bool isNonNegative,
                   core_t::TTime maximumTimeShift = 0) const override;

    //! Get the mean variance of the baseline.
    double meanVariance() const override;

    //! Compute the variance scale weight to apply at \p time.
    TVector2x1 varianceScaleWeight(core_t::TTime time, double variance, double confidence) const override;

    //! Get the count weight to apply at \p time.
    double countWeight(core_t::TTime time) const override;

    //! Get the derate to apply to the outlier weight at \p time.
    double outlierWeightDerate(core_t::TTime time, double error) const override;

    //! Get the prediction residuals in a recent time window.
    TFloatMeanAccumulatorVec residuals(bool isNonNegative) const override;

    //! Roll time forwards by \p skipInterval.
    void skipTime(core_t::TTime skipInterval) override;

    //! Debug the memory used by this object.
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const override;

    //! Get the memory used by this object.
    std::size_t memoryUsage() const override;

    //! Get the static size of this object.
    std::size_t staticSize() const override;

    //! Get the time shift which is being applied.
    core_t::TTime timeShift() const override;

    //! Get the seasonal components.
    const maths_t::TSeasonalComponentVec& seasonalComponents() const override;

    //! Get the calendar components.
    const maths_t::TCalendarComponentVec& calendarComponents() const override;

    //! Get a filtered predictor function for the seasonal components
    TFilteredPredictor predictor() const;

    //! Interpolate components for forecast
    void interpolateForForecast(core_t::TTime time);

private:
    //! Calculate the seasonal prediction at a given time
    TVector2x1 calculateSeasonalPrediction(core_t::TTime time, double confidence) const;

    //! Calculate seasonal forecast with confidence interval
    TDouble3Vec calculateSeasonalForecastWithConfidenceInterval(core_t::TTime time, 
                                                               double confidence,
                                                               double minimumScale) const;

    //! Smooth the seasonal prediction to ensure continuity
    template<typename F>
    auto smooth(const F& f, core_t::TTime time) const -> decltype(f(time));

private:
    //! The time over which discontinuities between weekdays
    //! and weekends are smoothed out.
    static const core_t::TTime SMOOTHING_INTERVAL;

private:
    //! Any time shift to supplied times.
    core_t::TTime m_TimeShift;

    //! The decay rate for the seasonal components.
    double m_DecayRate;

    //! The time of the latest value added.
    core_t::TTime m_LastValueTime;

    //! The time to which the components have been propagated.
    core_t::TTime m_LastPropagationTime;

    //! The test for seasonal components.
    CTimeSeriesDecompositionDetail::CSeasonalityTest m_SeasonalityTest;

    //! The seasonal component collection
    CTimeSeriesDecompositionDetail::CSeasonalComponents m_SeasonalComponents;
};

}
}
}

#endif // INCLUDED_ml_maths_time_series_CSeasonalDecomposition_h
