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

#ifndef INCLUDED_ml_maths_CTimeSeriesDecompositionInterface_h
#define INCLUDED_ml_maths_CTimeSeriesDecompositionInterface_h

#include <core/CMemory.h>
#include <core/CoreTypes.h>
#include <core/CSmallVector.h>

#include <maths/Constants.h>
#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <boost/array.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>
#include <stdint.h>

namespace ml
{
namespace maths
{
class CMultivariatePrior;
class CPrior;
class CSeasonalComponent;

//! \brief The interface for decomposing times series into periodic,
//! calendar periodic and trend components.
class MATHS_EXPORT CTimeSeriesDecompositionInterface
{
    public:
        using TDoubleAry = boost::array<double, 2>;
        using TWeights = CConstantWeights;

        //! The components of the decomposition.
        enum EComponents
        {
            E_Diurnal    = 0x1,
            E_NonDiurnal = 0x2,
            E_Seasonal   = 0x3,
            E_Trend      = 0x4,
            E_Calendar   = 0x8,
            E_All        = 0xf
        };

    public:
        virtual ~CTimeSeriesDecompositionInterface(void) = default;

        //! Clone this decomposition.
        virtual CTimeSeriesDecompositionInterface *clone(void) const = 0;

        //! Set the decay rate.
        virtual void decayRate(double decayRate) = 0;

        //! Get the decay rate.
        virtual double decayRate(void) const = 0;

        //! Switch to using this trend to forecast.
        //!
        //! \warning This is an irreversible action so if the trend
        //! is still need it should be copied first.
        virtual void forForecasting(void) = 0;

        //! Check if this is initialized.
        virtual bool initialized(void) const = 0;

        //! Adds a time series point \f$(t, f(t))\f$.
        //!
        //! \param[in] time The time of the function point.
        //! \param[in] value The function value at \p time.
        //! \param[in] weightStyles The styles of \p weights. Both the
        //! count and the Winsorisation weight styles have an effect.
        //! See maths_t::ESampleWeightStyle for more details.
        //! \param[in] weights The weights of \p value. The smaller
        //! the product count weight the less influence \p value has
        //! on the trend and it's local variance.
        //! \return True if number of estimated components changed
        //! and false otherwise.
        virtual bool addPoint(core_t::TTime time,
                              double value,
                              const maths_t::TWeightStyleVec &weightStyles = TWeights::COUNT,
                              const maths_t::TDouble4Vec &weights = TWeights::UNIT) = 0;

        //! Propagate the decomposition forwards to \p time.
        virtual void propagateForwardsTo(core_t::TTime time) = 0;

        //! May be test to see if there are any new seasonal components
        //! and interpolate.
        //!
        //! \param[in] time The current time.
        //! \return True if the number of seasonal components changed
        //! and false otherwise.
        virtual bool testAndInterpolate(core_t::TTime time) = 0;

        //! Get the mean value of the baseline in the vicinity of \p time.
        virtual double mean(core_t::TTime time) const = 0;

        //! Get the value of the time series baseline at \p time.
        //!
        //! \param[in] time The time of interest.
        //! \param[in] predictionConfidence The symmetric confidence interval
        //! for the prediction the baseline as a percentage.
        //! \param[in] forecastConfidence The symmetric confidence interval
        //! for long range forecasts as a percentage.
        //! \param[in] components The components to include in the baseline.
        virtual maths_t::TDoubleDoublePr baseline(core_t::TTime time,
                                                  double predictionConfidence = 0.0,
                                                  double forecastConfidence = 0.0,
                                                  EComponents components = E_All,
                                                  bool smooth = true) const = 0;

        //! Detrend \p value from the time series being modeled by removing
        //! any periodic component at \p time.
        //!
        //! \note That detrending preserves the time series mean.
        virtual double detrend(core_t::TTime time, double value, double confidence) const = 0;

        //! Get the mean variance of the baseline.
        virtual double meanVariance(void) const = 0;

        //! Compute the variance scale at \p time.
        //!
        //! \param[in] time The time of interest.
        //! \param[in] variance The variance of the distribution to scale.
        //! \param[in] confidence The symmetric confidence interval for the
        //! variance scale as a percentage.
        virtual maths_t::TDoubleDoublePr scale(core_t::TTime time,
                                               double variance,
                                               double confidence,
                                               bool smooth = true) const = 0;

        //! Roll time forwards by \p skipInterval.
        virtual void skipTime(core_t::TTime skipInterval) = 0;

        //! Get a checksum for this object.
        virtual uint64_t checksum(uint64_t seed = 0) const = 0;

        //! Get the memory used by this instance
        virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const = 0;

        //! Get the memory used by this instance
        virtual std::size_t memoryUsage(void) const = 0;

        //! Get the static size of this object.
        virtual std::size_t staticSize(void) const = 0;

        //! Get the seasonal components.
        virtual const maths_t::TSeasonalComponentVec &seasonalComponents(void) const = 0;

        //! This is the latest time of any point added to this object or the time skipped to.
        virtual core_t::TTime lastValueTime(void) const = 0;
};

using TDecompositionPtr = boost::shared_ptr<maths::CTimeSeriesDecompositionInterface>;
using TDecompositionPtr10Vec = core::CSmallVector<TDecompositionPtr, 10>;

//! Initialize a univariate prior to match the moments of \p decomposition.
MATHS_EXPORT
bool initializePrior(core_t::TTime bucketLength,
                     double learnRate,
                     const CTimeSeriesDecompositionInterface &decomposition,
                     CPrior &prior);

//! Initialize a multivariate prior to match the moments of \p decomposition.
MATHS_EXPORT
bool initializePrior(core_t::TTime bucketLength,
                     double learnRate,
                     const TDecompositionPtr10Vec &decomposition,
                     CMultivariatePrior &prior);

}
}

#endif // INCLUDED_ml_maths_CTimeSeriesDecompositionInterface_h
