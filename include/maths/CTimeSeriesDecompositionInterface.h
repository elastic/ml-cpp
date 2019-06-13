/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CTimeSeriesDecompositionInterface_h
#define INCLUDED_ml_maths_CTimeSeriesDecompositionInterface_h

#include <core/CMemory.h>
#include <core/CSmallVector.h>
#include <core/CoreTypes.h>

#include <maths/CBasicStatistics.h>
#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <boost/array.hpp>

#include <cstddef>
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

namespace ml {
namespace maths {
class CMultivariatePrior;
class CPrior;
class CSeasonalComponent;
struct SChangeDescription;

//! \brief Type definitions shared by the CTimeSeriesDecompositionInterface
//! hierarchy and CTimeSeriesDecompositionDetails.
class MATHS_EXPORT CTimeSeriesDecompositionTypes {
public:
    using TPredictor = std::function<double(core_t::TTime)>;
    using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
    using TComponentChangeCallback = std::function<void(TFloatMeanAccumulatorVec)>;
};

//! \brief The interface for decomposing times series into periodic,
//! calendar periodic and trend components.
class MATHS_EXPORT CTimeSeriesDecompositionInterface : public CTimeSeriesDecompositionTypes {
public:
    using TDouble3Vec = core::CSmallVector<double, 3>;
    using TDouble3VecVec = std::vector<TDouble3Vec>;
    using TWeights = maths_t::CUnitWeights;
    using TWriteForecastResult = std::function<void(core_t::TTime, const TDouble3Vec&)>;

    //! The components of the decomposition.
    enum EComponents {
        E_Diurnal = 0x1,
        E_NonDiurnal = 0x2,
        E_Seasonal = 0x3,
        E_Trend = 0x4,
        E_Calendar = 0x8,
        E_All = 0xf,
        E_TrendForced = 0x10 //!< Force get the trend component (if
                             //!< it's not being used for prediction).
                             //!< This needs to be bigger than E_All.
    };

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

    //! Set whether or not we're testing for a change.
    virtual void testingForChange(bool value) = 0;

    //! Adds a time series point \f$(t, f(t))\f$.
    //!
    //! \param[in] time The time of the data point.
    //! \param[in] value The value of the data point.
    //! \param[in] weights The weights of \p value. The smaller
    //! the product count weight the less influence \p value has
    //! on the trend and it's local variance.
    //! \param[in] componentChangeCallback Called if the components
    //! change as a result of adding the data point.
    virtual void
    addPoint(core_t::TTime time,
             double value,
             const maths_t::TDoubleWeightsAry& weights = TWeights::UNIT,
             const TComponentChangeCallback& componentChangeCallback = noop) = 0;

    //! Apply \p change at \p time.
    //!
    //! \param[in] time The time of the change point.
    //! \param[in] value The value immediately before the change
    //! point.
    //! \param[in] change A description of the change to apply.
    //! \return True if a new component was detected.
    virtual bool
    applyChange(core_t::TTime time, double value, const SChangeDescription& change) = 0;

    //! Propagate the decomposition forwards to \p time.
    virtual void propagateForwardsTo(core_t::TTime time) = 0;

    //! Get the mean value of the time series in the vicinity of \p time.
    virtual double meanValue(core_t::TTime time) const = 0;

    //! Get the predicted value of the time series at \p time.
    //!
    //! \param[in] time The time of interest.
    //! \param[in] confidence The symmetric confidence interval for the
    //! prediction the baseline as a percentage.
    //! \param[in] components The components to include in the baseline.
    virtual maths_t::TDoubleDoublePr value(core_t::TTime time,
                                           double confidence = 0.0,
                                           int components = E_All,
                                           bool smooth = true) const = 0;

    //! Get the maximum interval for which the time series can be forecast.
    virtual core_t::TTime maximumForecastInterval() const = 0;

    //! Forecast from \p start to \p end at \p dt intervals.
    //!
    //! \param[in] startTime The start of the forecast.
    //! \param[in] endTime The end of the forecast.
    //! \param[in] step The time increment.
    //! \param[in] confidence The forecast confidence interval.
    //! \param[in] minimumScale The minimum permitted seasonal scale.
    //! \param[in] writer Forecast results are passed to this callback.
    virtual void forecast(core_t::TTime startTime,
                          core_t::TTime endTime,
                          core_t::TTime step,
                          double confidence,
                          double minimumScale,
                          const TWriteForecastResult& writer) = 0;

    //! Detrend \p value from the time series being modeled by removing
    //! any periodic component at \p time.
    //!
    //! \note That detrending preserves the time series mean.
    virtual double detrend(core_t::TTime time,
                           double value,
                           double confidence,
                           int components = E_All) const = 0;

    //! Get the mean variance of the baseline.
    virtual double meanVariance() const = 0;

    //! Compute the variance scale at \p time.
    //!
    //! \param[in] time The time of interest.
    //! \param[in] variance The variance of the distribution to scale.
    //! \param[in] confidence The symmetric confidence interval for the
    //! variance scale as a percentage.
    virtual maths_t::TDoubleDoublePr
    scale(core_t::TTime time, double variance, double confidence, bool smooth = true) const = 0;

    //! Get the values in a recent time window.
    virtual TFloatMeanAccumulatorVec windowValues(const TPredictor& predictor) const = 0;

    //! Roll time forwards by \p skipInterval.
    virtual void skipTime(core_t::TTime skipInterval) = 0;

    //! Get a checksum for this object.
    virtual uint64_t checksum(uint64_t seed = 0) const = 0;

    //! Get the memory used by this instance
    virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const = 0;

    //! Get the memory used by this instance
    virtual std::size_t memoryUsage() const = 0;

    //! Get the static size of this object.
    virtual std::size_t staticSize() const = 0;

    //! Get the time shift which is being applied.
    virtual core_t::TTime timeShift() const = 0;

    //! Get the seasonal components.
    virtual const maths_t::TSeasonalComponentVec& seasonalComponents() const = 0;

    //! This is the latest time of any point added to this object or
    //! the time skipped to.
    virtual core_t::TTime lastValueTime() const = 0;

protected:
    static void noop(TFloatMeanAccumulatorVec) {}
};
}
}

#endif // INCLUDED_ml_maths_CTimeSeriesDecompositionInterface_h
