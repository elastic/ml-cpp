/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CTimeSeriesDecompositionStub_h
#define INCLUDED_ml_maths_CTimeSeriesDecompositionStub_h

#include <maths/CTimeSeriesDecompositionInterface.h>
#include <maths/ImportExport.h>

namespace ml {
namespace maths {

//! \brief Stub out the interface if it is known that the time series
//! being modeled can't have seasonality.
//!
//! DESCRIPTION:\n
//! This is a lightweight (empty) class which implements the interface
//! for the case that the time series being modeled is known a-priori
//! not to have seasonality.
class MATHS_EXPORT CTimeSeriesDecompositionStub : public CTimeSeriesDecompositionInterface {
public:
    //! Clone this decomposition.
    virtual CTimeSeriesDecompositionStub* clone() const;

    //! No-op.
    virtual void decayRate(double decayRate);

    //! Get the decay rate.
    virtual double decayRate() const;

    //! Returns false.
    virtual bool initialized() const;

    //! No-op returning false.
    virtual bool addPoint(core_t::TTime time,
                          double value,
                          const maths_t::TWeightStyleVec& weightStyles = TWeights::COUNT,
                          const maths_t::TDouble4Vec& weights = TWeights::UNIT);

    //! No-op.
    virtual void propagateForwardsTo(core_t::TTime time);

    //! Returns 0.
    virtual double mean(core_t::TTime time) const;

    //! Returns (0.0, 0.0).
    virtual maths_t::TDoubleDoublePr
    baseline(core_t::TTime time, double confidence = 0.0, int components = E_All, bool smooth = true) const;

    //! Clears \p result.
    virtual void forecast(core_t::TTime startTime,
                          core_t::TTime endTime,
                          core_t::TTime step,
                          double confidence,
                          double minimumScale,
                          TDouble3VecVec& result);

    //! Returns \p value.
    virtual double detrend(core_t::TTime time, double value, double confidence) const;

    //! Returns 0.0.
    virtual double meanVariance() const;

    //! Returns (1.0, 1.0).
    virtual maths_t::TDoubleDoublePr scale(core_t::TTime time, double variance, double confidence, bool smooth = true) const;

    //! No-op.
    virtual void skipTime(core_t::TTime skipInterval);

    //! Get a checksum for this object.
    virtual uint64_t checksum(uint64_t seed = 0) const;

    //! Debug the memory used by this object.
    virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

    //! Get the memory used by this object.
    virtual std::size_t memoryUsage() const;

    //! Get the static size of this object.
    virtual std::size_t staticSize() const;

    //! Get the seasonal components.
    virtual const maths_t::TSeasonalComponentVec& seasonalComponents() const;

    //! Returns 0.
    virtual core_t::TTime lastValueTime() const;
};
}
}

#endif // INCLUDED_ml_maths_CTimeSeriesDecompositionStub_h
