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

#ifndef INCLUDED_ml_maths_time_series_CTimeSeriesDecompositionStub_h
#define INCLUDED_ml_maths_time_series_CTimeSeriesDecompositionStub_h

#include <maths/time_series/ImportExport.h>

#include <maths/time_series/CTimeSeriesDecompositionInterface.h>

namespace ml {
namespace maths {
namespace time_series {

//! \brief Stub out the interface if it is known that the time series
//! being modeled can't have seasonality.
//!
//! DESCRIPTION:\n
//! This is a lightweight (empty) class which implements the interface
//! for the case that the time series being modeled is known a-priori
//! not to have seasonality.
class MATHS_TIME_SERIES_EXPORT CTimeSeriesDecompositionStub
    : public CTimeSeriesDecompositionInterface {
public:
    //! Clone this decomposition.
    CTimeSeriesDecompositionStub* clone(bool isForForecast = false) const override;

    //! No-op.
    void dataType(maths_t::EDataType dataType) override;

    //! No-op.
    void decayRate(double decayRate) override;

    //! Get the decay rate.
    double decayRate() const override;

    //! Returns false.
    bool initialized() const override;

    //! No-op returning false.
    void addPoint(core_t::TTime time,
                  double value,
                  const maths_t::TDoubleWeightsAry& weights = TWeights::UNIT,
                  const TComponentChangeCallback& componentChangeCallback = noopComponentChange,
                  const maths_t::TModelAnnotationCallback& modelAnnotationCallback = noopModelAnnotation,
                  double occupancy = 1.0,
                  core_t::TTime firstValueTime = MIN_TIME) override;

    //! No-op.
    void shiftTime(core_t::TTime time, core_t::TTime shift) override;

    //! No-op.
    void propagateForwardsTo(core_t::TTime time) override;

    //! Returns 0.
    double meanValue(core_t::TTime time) const override;

    //! Returns zero vector.
    TVector2x1 value(core_t::TTime time, double confidence, bool isNonNegative) const override;

    //! Returns 0.
    core_t::TTime maximumForecastInterval() const override;

    //! No-op.
    void forecast(core_t::TTime startTime,
                  core_t::TTime endTime,
                  core_t::TTime step,
                  double confidence,
                  double minimumScale,
                  bool isNonNegative,
                  const TWriteForecastResult& writer) override;

    //! Returns \p value.
    double detrend(core_t::TTime time,
                   double value,
                   double confidence,
                   bool isNonNegative,
                   core_t::TTime maximumTimeShift = 0) const override;

    //! Returns 0.0.
    double meanVariance() const override;

    //! Returns ones vector.
    TVector2x1 varianceScaleWeight(core_t::TTime time, double variance, double confidence) const override;

    //! Returns 1.0.
    double countWeight(core_t::TTime time) const override;

    //! Returns 0.0.
    double outlierWeightDerate(core_t::TTime time, double error) const override;

    //! Returns an empty vector.
    TFloatMeanAccumulatorVec residuals(bool isNonNegative) const override;

    //! No-op.
    void skipTime(core_t::TTime skipInterval) override;

    //! Get a checksum for this object.
    std::uint64_t checksum(std::uint64_t seed = 0) const override;

    //! Debug the memory used by this object.
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const override;

    //! Get the memory used by this object.
    std::size_t memoryUsage() const override;

    //! Get the static size of this object.
    std::size_t staticSize() const override;

    //! Returns zero.
    core_t::TTime timeShift() const override;

    //! Returns an empty vector.
    const maths_t::TSeasonalComponentVec& seasonalComponents() const override;
};
}
}
}

#endif // INCLUDED_ml_maths_time_series_CTimeSeriesDecompositionStub_h
