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

#ifndef INCLUDED_ml_maths_CTimeSeriesDecomposition_h
#define INCLUDED_ml_maths_CTimeSeriesDecomposition_h

#include <maths/CTimeSeriesDecompositionDetail.h>
#include <maths/CTimeSeriesDecompositionInterface.h>
#include <maths/Constants.h>
#include <maths/ImportExport.h>

#include <memory>

namespace CTimeSeriesDecompositionTest {
class CNanInjector;
}

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {
class CPrior;
struct STimeSeriesDecompositionRestoreParams;

//! \brief Decomposes a time series into a linear combination of trend and cyclical
//! functions and a stationary random process.
//!
//! DESCRIPTION:\n
//! This looks for a trend and daily, weekly, weekend/weekday, and yearly seasonality.
//! It also checks for offsets with high autocorrelation and predictive calendar
//! components, such as day of month, last Friday of month, etc. It maintains the
//! state to do this in a streaming fashion and will both add and remove predictive
//! features as they come and go from the time series.
//!
//! It uses an additive combination of the predictive features it has selected to
//! predict both the values and residual variance w.r.t. these predictions. It uses
//! exponential decay to remove old data from the component models. The schedule is
//! different for each component and the overall rate is controlled by the decay
//! rate.
//!
//! The main functionality it provides are to predict the value and variance at
//! a specified time, value and scale, to remove its prediction from a sample of
//! the time series, see detrend, and forecasting, see forecast.
class MATHS_EXPORT EMPTY_BASE_OPT CTimeSeriesDecomposition
    : public CTimeSeriesDecompositionInterface,
      private CTimeSeriesDecompositionDetail {
public:
    //! \param[in] decayRate The rate at which information is lost.
    //! \param[in] bucketLength The data bucketing length.
    //! \param[in] seasonalComponentSize The number of buckets to use to estimate a
    //! seasonal component.
    explicit CTimeSeriesDecomposition(double decayRate = 0.0,
                                      core_t::TTime bucketLength = 0,
                                      std::size_t seasonalComponentSize = COMPONENT_SIZE);

    //! Construct from part of a state document.
    CTimeSeriesDecomposition(const STimeSeriesDecompositionRestoreParams& params,
                             core::CStateRestoreTraverser& traverser);

    //! Deep copy.
    CTimeSeriesDecomposition(const CTimeSeriesDecomposition& other,
                             bool isForForecast = false);

    //! An efficient swap of the state of this and \p other.
    void swap(CTimeSeriesDecomposition& other);

    //! Assign this object (using deep copy).
    CTimeSeriesDecomposition& operator=(const CTimeSeriesDecomposition& other);

    //! Persist state by passing information to the supplied inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Clone this decomposition.
    CTimeSeriesDecomposition* clone(bool isForForecast = false) const override;

    //! Set the data type.
    void dataType(maths_t::EDataType dataType) override;

    //! Set the decay rate.
    void decayRate(double decayRate) override;

    //! Get the decay rate.
    double decayRate() const override;

    //! Check if the decomposition has any initialized components.
    bool initialized() const override;

    //! Adds a time series point \f$(t, f(t))\f$.
    //!
    //! \param[in] time The time of the data point.
    //! \param[in] value The value of the data point.
    //! \param[in] weights The weights of \p value. The smaller the count weight the
    //! less influence \p value has on the decomposition.
    //! \param[in] componentChangeCallback Supplied with samples of the prediction
    //! residuals if a new component is added as a result of adding the data point.
    //! \param[in] modelAnnotationCallback Supplied with an annotation if a new
    //! component is added as a result of adding the data point.
    void addPoint(core_t::TTime time,
                  double value,
                  const maths_t::TDoubleWeightsAry& weights = TWeights::UNIT,
                  const TComponentChangeCallback& componentChangeCallback = noopComponentChange,
                  const maths_t::TModelAnnotationCallback& modelAnnotationCallback = noopModelAnnotation) override;

    //! Shift seasonality by \p shift at \p time.
    void shiftTime(core_t::TTime time, core_t::TTime shift) override;

    //! Propagate the decomposition forwards to \p time.
    void propagateForwardsTo(core_t::TTime time) override;

    //! Get the mean value of the time series in the vicinity of \p time.
    double meanValue(core_t::TTime time) const override;

    //! Get the predicted value of the time series at \p time.
    //!
    //! \param[in] time The time of interest.
    //! \param[in] confidence The symmetric confidence interval for the prediction
    //! the baseline as a percentage.
    //! \param[in] components The components to include in the baseline.
    //! \param[in] removedSeasonalMask A bit mask of specific seasonal components
    //! to remove. This is only intended for use by CTimeSeriesTestForSeasonlity.
    //! \param[in] smooth Detail do not supply.
    maths_t::TDoubleDoublePr value(core_t::TTime time,
                                   double confidence = 0.0,
                                   int components = E_All,
                                   const TBoolVec& removedSeasonalMask = {},
                                   bool smooth = true) const override;

    //! Get a function which returns the decomposition value as a function of time.
    //!
    //! This caches the expensive part of the calculation and so is much faster
    //! than repeatedly calling value.
    //!
    //! \warning This can only be used as long as the trend component isn't updated.
    TFilteredPredictor predictor(int components) const;

    //! Get the maximum interval for which the time series can be forecast.
    core_t::TTime maximumForecastInterval() const override;

    //! Forecast from \p start to \p end at \p dt intervals.
    //!
    //! \param[in] startTime The start of the forecast.
    //! \param[in] endTime The end of the forecast.
    //! \param[in] step The time increment.
    //! \param[in] confidence The forecast confidence interval.
    //! \param[in] minimumScale The minimum permitted seasonal scale.
    //! \param[in] writer Forecast results are passed to this callback.
    void forecast(core_t::TTime startTime,
                  core_t::TTime endTime,
                  core_t::TTime step,
                  double confidence,
                  double minimumScale,
                  const TWriteForecastResult& writer) override;

    //! Detrend \p value by the prediction of the modelled features at \p time.
    //!
    //! \note That detrending preserves the time series mean.
    double detrend(core_t::TTime time, double value, double confidence, int components = E_All) const override;

    //! Get the mean variance of the baseline.
    double meanVariance() const override;

    //! Compute the variance scale weight to apply at \p time.
    //!
    //! \param[in] time The time of interest.
    //! \param[in] variance The variance of the distribution to scale.
    //! \param[in] confidence The symmetric confidence interval for the variance
    //! scale as a percentage.
    maths_t::TDoubleDoublePr varianceScaleWeight(core_t::TTime time,
                                                 double variance,
                                                 double confidence,
                                                 bool smooth = true) const override;

    //! Get the count weight to apply at \p time.
    double countWeight(core_t::TTime time) const override;

    //! Get the derate to apply to the Winsorisation weight at \p time.
    double winsorisationDerate(core_t::TTime time) const override;

    //! Get the prediction residuals in a recent time window.
    TFloatMeanAccumulatorVec residuals() const override;

    //! Roll time forwards by \p skipInterval.
    void skipTime(core_t::TTime skipInterval) override;

    //! Get a checksum for this object.
    std::uint64_t checksum(std::uint64_t seed = 0) const override;

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

    //! Get the time of the last value.
    core_t::TTime lastValueTime() const;

private:
    using TMediatorPtr = std::unique_ptr<CMediator>;

private:
    //! Set up the communication mediator.
    void initializeMediator();

    //! Create from part of a state document.
    bool acceptRestoreTraverser(const SDistributionRestoreParams& params,
                                core::CStateRestoreTraverser& traverser);

    //! The correction to produce a smooth join between periodic
    //! repeats and partitions.
    template<typename F>
    maths_t::TDoubleDoublePr smooth(const F& f, core_t::TTime time, int components) const;

private:
    //! The time over which discontinuities between weekdays
    //! and weekends are smoothed out.
    static const core_t::TTime SMOOTHING_INTERVAL;

private:
    //! Any time shift to supplied times.
    core_t::TTime m_TimeShift;

    //! The time of the latest value added.
    core_t::TTime m_LastValueTime;

    //! The time to which the trend has been propagated.
    core_t::TTime m_LastPropagationTime;

    //! Handles the communication between the various tests and components.
    TMediatorPtr m_Mediator;

    //! The test for sudden change events.
    CChangePointTest m_ChangePointTest;

    //! The test for seasonal components.
    CSeasonalityTest m_SeasonalityTest;

    //! The test for calendar cyclic components.
    CCalendarTest m_CalendarCyclicTest;

    //! The state for modeling the components of the decomposition.
    CComponents m_Components;

    //! Befriend a helper class used by the unit tests
    friend class CTimeSeriesDecompositionTest::CNanInjector;
};
}
}

#endif // INCLUDED_ml_maths_CTimeSeriesDecomposition_h
