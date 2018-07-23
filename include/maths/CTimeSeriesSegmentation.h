/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDE_ml_maths_CTimeSeriesSegmentation_h
#define INCLUDE_ml_maths_CTimeSeriesSegmentation_h

#include <maths/CBasicStatistics.h>
#include <maths/MathsTypes.h>

#include <vector>

namespace ml {
namespace maths {

//! \brief Utility functionality to perform segmentation of a time series.
class CTimeSeriesSegmentation {
public:
    using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;

    //! Perform top-down recursive segmentation.
    //!
    //! The time series is segmented using piecewise linear models in a top down
    //! fashion with each break point being chosen to maximize r-squared and model
    //! selection happening for each candidate segmentation. Model selection is
    //! achieved by examining the delta BIC for the segmentation versus non-segmented
    //! model.
    //!
    //! \param[in] values The time series values to segment. These are assumed to
    //! be equally spaced in time order.
    //! \param[in] bicGainToSegment The minimum delta BIC to accept a partition, i.e.
    //! increase in BIC of non-segmented model.
    //! \param[in] outliersFraction The proportion of values to treat as outliers.
    //! Outliers are re-weighted when doing model fit and computing unexplained
    //! variance for model selection. Must be in the range [0.0, 1.0).
    //! \param[in] outlierWeight The weight to apply to outliers when doing model
    //! fit and computing unexplained variance. Must be in the range [0.0, 1.0].
    //! \return The values minus the model predictions.
    static TFloatMeanAccumulatorVec
    topDownPiecewiseLinear(const TFloatMeanAccumulatorVec& values,
                           double bicGainToSegment = 6.0,
                           double outliersFraction = 0.0,
                           double outlierWeight = 1.0);

    static TFloatMeanAccumulatorVec
    topDownPeriodicPiecewiseLinearScaling(const TFloatMeanAccumulatorVec& values,
                                          std::size_t period,
                                          double bicGainToSegment = 6.0,
                                          double outliersFraction = 0.0,
                                          double outlierWeight = 1.0);

private:
    using TDoubleVec = std::vector<double>;
    using TSizeVec = std::vector<std::size_t>;

private:
    //! Implements top-down recursive segmentation with linear models.
    template<typename ITR>
    static void topDownPiecewiseLinear(ITR begin,
                                       ITR end,
                                       std::size_t offset,
                                       double startTime,
                                       double dt,
                                       double bicGainToSegment,
                                       double outliersFraction,
                                       double outlierWeight,
                                       TSizeVec& segmentation);

    //! Implements top-down recursive segmentation with linear models.
    template<typename ITR>
    static void topDownPeriodicPiecewiseLinearScaling(ITR begin,
                                                      ITR end,
                                                      std::size_t offset,
                                                      const TDoubleVec& model,
                                                      double bicGainToSegment,
                                                      TSizeVec& segmentation);
};
}
}

#endif // INCLUDE_ml_maths_CTimeSeriesSegmentation_h
