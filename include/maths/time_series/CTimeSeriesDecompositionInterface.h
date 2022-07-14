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

#ifndef INCLUDED_ml_maths_time_series_CTimeSeriesDecompositionInterface_h
#define INCLUDED_ml_maths_time_series_CTimeSeriesDecompositionInterface_h

#include <core/CMemoryFwd.h>
#include <core/CSmallVector.h>
#include <core/CoreTypes.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/MathsTypes.h>

#include <maths/time_series/ImportExport.h>

#include <boost/array.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace ml {
namespace maths {
namespace common {
class CMultivariatePrior;
class CPrior;
}
namespace time_series {
class CSeasonalComponent;
struct SChangeDescription;

//! \brief Type definitions shared by the CTimeSeriesDecompositionInterface
//! hierarchy and CTimeSeriesDecompositionDetails.
class MATHS_TIME_SERIES_EXPORT CTimeSeriesDecompositionTypes {
public:
    using TBoolVec = std::vector<bool>;
    using TPredictor = std::function<double(core_t::TTime)>;
    using TFloatMeanAccumulator =
        common::CBasicStatistics::SSampleMean<common::CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
    using TComponentChangeCallback = std::function<void(TFloatMeanAccumulatorVec)>;
};

//! \brief The interface for decomposing times series into seasonal, calendar
//! cyclical and trend components.
class MATHS_TIME_SERIES_EXPORT CTimeSeriesDecompositionInterface
    : public CTimeSeriesDecompositionTypes {
public:
    using TVector2x1 = common::CVectorNx1<double, 2>;
    using TDouble3Vec = core::CSmallVector<double, 3>;
    using TDouble3VecVec = std::vector<TDouble3Vec>;
    using TWeights = maths_t::CUnitWeights;
    using TWriteForecastResult = std::function<void(core_t::TTime, const TDouble3Vec&)>;

    //! The components of the decomposition.
    enum EComponents {
        E_Seasonal = 0x1,
        E_Trend = 0x2,
        E_Calendar = 0x4,
        E_All = 0x7,
        E_TrendForced = 0x8 //!< Force get the trend component (if
                            //!< it's not being used for prediction).
                            //!< This needs to be bigger than E_All.
    };

public:
    static constexpr core_t::TTime MIN_TIME{std::numeric_limits<core_t::TTime>::min()};

public:
    virtual ~CTimeSeriesDecompositionInterface() = default;

    //! Clone this decomposition.
    virtual CTimeSeriesDecompositionInterface* clone(bool isForForecast = false) const = 0;

    //! Set the data type.
    virtual void dataType(maths_t::EDataType dataType) = 0;

    //! Set the decay rate.
    virtual void decayRate(double decayRate) = 0;

    //! Get the decay rate.
    virtual double decayRate() const = 0;

    //! Check if this is initialized.
    virtual bool initialized() const = 0;

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
    //! \param[in] occupancy The proportion of non-empty buckets.
    //! \param[in] firstValueTime The time of the first value added to the decomposition.
    virtual void
    addPoint(core_t::TTime time,
             double value,
             const maths_t::TDoubleWeightsAry& weights = TWeights::UNIT,
             const TComponentChangeCallback& componentChangeCallback = noopComponentChange,
             const maths_t::TModelAnnotationCallback& modelAnnotationCallback = noopModelAnnotation,
             double occupancy = 1.0,
             core_t::TTime firstValueTime = MIN_TIME) = 0;

    //! Shift seasonality by \p shift at \p time.
    virtual void shiftTime(core_t::TTime time, core_t::TTime shift) = 0;

    //! Propagate the decomposition forwards to \p time.
    virtual void propagateForwardsTo(core_t::TTime time) = 0;

    //! Get the mean value of the time series in the vicinity of \p time.
    virtual double meanValue(core_t::TTime time) const = 0;

    //! Get the predicted value of the time series at \p time.
    //!
    //! \param[in] time The time of interest.
    //! \param[in] confidence The symmetric confidence interval for the prediction
    //! the baseline as a percentage.
    //! \param[in] isNonNegative True if the data being modelled are known to be
    //! non-negative.
    virtual TVector2x1
    value(core_t::TTime time, double confidence, bool isNonNegative) const = 0;

    //! Get the maximum interval for which the time series can be forecast.
    virtual core_t::TTime maximumForecastInterval() const = 0;

    //! Forecast from \p start to \p end at \p dt intervals.
    //!
    //! \param[in] startTime The start of the forecast.
    //! \param[in] endTime The end of the forecast.
    //! \param[in] step The time increment.
    //! \param[in] confidence The forecast confidence interval.
    //! \param[in] minimumScale The minimum permitted seasonal scale.
    //! \param[in] isNonNegative True if the data being modelled are known to be
    //! non-negative.
    //! \param[in] writer Forecast results are passed to this callback.
    virtual void forecast(core_t::TTime startTime,
                          core_t::TTime endTime,
                          core_t::TTime step,
                          double confidence,
                          double minimumScale,
                          bool isNonNegative,
                          const TWriteForecastResult& writer) = 0;

    //! Remove the prediction of the component models at \p time from \p value.
    //!
    //! \param[in] time The time of \p value.
    //! \param[in] value The value from which to remove the prediction.
    //! \param[in] confidence The prediction confidence interval as a percentage.
    //! The closest point to \p value in the confidence interval is removed.
    //! \param[in] isNonNegative True if the data being modelled are known to be
    //! non-negative.
    //! \param[in] maximumTimeShift The maximum amount by which we will shift
    //! \p time in order to minimize the difference between the prediction and
    //! \p value.
    virtual double detrend(core_t::TTime time,
                           double value,
                           double confidence,
                           bool isNonNegative,
                           core_t::TTime maximumTimeShift = 0) const = 0;

    //! Get the mean variance of the baseline.
    virtual double meanVariance() const = 0;

    //! Compute the variance scale to apply at \p time.
    //!
    //! \param[in] time The time of interest.
    //! \param[in] variance The variance of the distribution to scale.
    //! \param[in] confidence The symmetric confidence interval for the variance
    //! scale as a percentage.
    virtual TVector2x1
    varianceScaleWeight(core_t::TTime time, double variance, double confidence) const = 0;

    //! Get the count weight to apply at \p time.
    virtual double countWeight(core_t::TTime time) const = 0;

    //! Get the derate to apply to the outlier weight at \p time.
    virtual double outlierWeightDerate(core_t::TTime time, double derate) const = 0;

    //! Get the prediction residuals in a recent time window.
    //!
    //! \param[in] isNonNegative True if the data being modelled are known to be
    //! non-negative.
    virtual TFloatMeanAccumulatorVec residuals(bool isNonNegative) const = 0;

    //! Roll time forwards by \p skipInterval.
    virtual void skipTime(core_t::TTime skipInterval) = 0;

    //! Get a checksum for this object.
    virtual std::uint64_t checksum(std::uint64_t seed = 0) const = 0;

    //! Get the memory used by this instance
    virtual void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const = 0;

    //! Get the memory used by this instance
    virtual std::size_t memoryUsage() const = 0;

    //! Get the static size of this object.
    virtual std::size_t staticSize() const = 0;

    //! Get the time shift which is being applied.
    virtual core_t::TTime timeShift() const = 0;

    //! Get the seasonal components.
    virtual const maths_t::TSeasonalComponentVec& seasonalComponents() const = 0;

protected:
    static void noopComponentChange(TFloatMeanAccumulatorVec) {}
    static void noopModelAnnotation(const std::string&) {}
};
}
}
}

#endif // INCLUDED_ml_maths_time_series_CTimeSeriesDecompositionInterface_h
