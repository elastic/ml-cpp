/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */

#ifndef INCLUDED_ml_maths_CTimeSeriesDecompositionStub_h
#define INCLUDED_ml_maths_CTimeSeriesDecompositionStub_h

#include <maths/CTimeSeriesDecompositionInterface.h>
#include <maths/ImportExport.h>

namespace ml
{
namespace maths
{

//! \brief Stub out the interface if it is known that the time series
//! being modeled can't have seasonality.
//!
//! DESCRIPTION:\n
//! This is a lightweight (empty) class which implements the interface
//! for the case that the time series being modeled is known a-priori
//! not to have seasonality.
class MATHS_EXPORT CTimeSeriesDecompositionStub : public CTimeSeriesDecompositionInterface
{
    public:
        //! Clone this decomposition.
        virtual CTimeSeriesDecompositionStub *clone(bool isForForecast = false) const;

        //! No-op.
        virtual void dataType(maths_t::EDataType dataType);

        //! No-op.
        virtual void decayRate(double decayRate);

        //! Get the decay rate.
        virtual double decayRate(void) const;

        //! Returns false.
        virtual bool initialized(void) const;

        //! No-op returning false.
        virtual bool addPoint(core_t::TTime time,
                              double value,
                              const maths_t::TWeightStyleVec &weightStyles = TWeights::COUNT,
                              const maths_t::TDouble4Vec &weights = TWeights::UNIT);

        //! No-op returning false.
        virtual bool applyChange(core_t::TTime time, double value,
                                 const SChangeDescription &change);

        //! No-op.
        virtual void propagateForwardsTo(core_t::TTime time);

        //! Returns 0.
        virtual double meanValue(core_t::TTime time) const;

        //! Returns (0.0, 0.0).
        virtual maths_t::TDoubleDoublePr value(core_t::TTime time,
                                               double confidence = 0.0,
                                               int components = E_All,
                                               bool smooth = true) const;

        //! No-op.
        virtual void forecast(core_t::TTime startTime,
                              core_t::TTime endTime,
                              core_t::TTime step,
                              double confidence,
                              double minimumScale,
                              const TWriteForecastResult &writer);

        //! Returns \p value.
        virtual double detrend(core_t::TTime time,
                               double value,
                               double confidence,
                               int components = E_All) const;

        //! Returns 0.0.
        virtual double meanVariance(void) const;

        //! Returns (1.0, 1.0).
        virtual maths_t::TDoubleDoublePr scale(core_t::TTime time,
                                               double variance,
                                               double confidence,
                                               bool smooth = true) const;

        //! No-op.
        virtual void skipTime(core_t::TTime skipInterval);

        //! Get a checksum for this object.
        virtual uint64_t checksum(uint64_t seed = 0) const;

        //! Debug the memory used by this object.
        virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the memory used by this object.
        virtual std::size_t memoryUsage(void) const;

        //! Get the static size of this object.
        virtual std::size_t staticSize(void) const;

        //! Returns zero.
        virtual core_t::TTime timeShift(void) const;

        //! Returns an empty vector.
        virtual const maths_t::TSeasonalComponentVec &seasonalComponents(void) const;

        //! Returns 0.
        virtual core_t::TTime lastValueTime(void) const;
};

}
}

#endif // INCLUDED_ml_maths_CTimeSeriesDecompositionStub_h
