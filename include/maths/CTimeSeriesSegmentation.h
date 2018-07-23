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
    using TSizeVec = std::vector<std::size_t>;
    using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;

    //! Perform top-down recursive segmentation into linear models.
    //!
    //! The time series is segmented using piecewise linear models in a top down
    //! fashion with each break point being chosen to maximize r-squared and model
    //! selection happening for each candidate segmentation. Model selection
    //! is achieved by thresholding the significance of the residual variance
    //! ratio for segmented versus non-segmented models.
    //!
    //! \param[in] values The time series values to segment. These are assumed to
    //! be equally spaced in time order.
    //! \param[in] significanceToSegment The maximum significance of the decrease
    //! in the residual variance to accept a partition.
    //! \param[in] outlierFraction The proportion of values to treat as outliers.
    //! Outliers are re-weighted when doing model fit and computing unexplained
    //! variance for model selection. Must be in the range [0.0, 1.0).
    //! \param[in] outlierWeight The weight to apply to outliers when doing model
    //! fit and computing unexplained variance. Must be in the range [0.0, 1.0].
    //! \return The segmentation of values otherwise the start and end indices.
    static TSizeVec topDownPiecewiseLinear(const TFloatMeanAccumulatorVec& values,
                                           double significanceToSegment = 0.01,
                                           double outlierFraction = 0.0,
                                           double outlierWeight = 0.1);

    //! Remove the predictions of a piecewise linear model.
    //!
    //! \param[in] values The time series values to segment. These are assumed to
    //! be equally spaced in time order.
    //! \param[in] segmentation The segmentation of \p values to use.
    //! \param[in] outlierFraction The proportion of values to treat as outliers.
    //! Outliers are re-weighted when doing model fit and computing unexplained
    //! variance for model selection. Must be in the range [0.0, 1.0).
    //! \param[in] outlierWeight The weight to apply to outliers when doing model
    //! fit and computing unexplained variance. Must be in the range [0.0, 1.0].
    //! \return The values minus the model predictions.
    static TFloatMeanAccumulatorVec
    removePredictionsOfPiecewiseLinear(TFloatMeanAccumulatorVec values,
                                       const TSizeVec& segmentation,
                                       double outlierFraction = 0.0,
                                       double outlierWeight = 0.1);

    //! Perform top-down recursive segmentation into a scaled periodic model.
    //!
    //! The time series is segmented using piecewise linear scaled periodic model
    //! in a top down fashion with break points being chosen to maximize r-squared
    //! and model selection happening for each candidate segmentation. Model selection
    //! is achieved by thresholding the significance of the residual variance ratio
    //! for the segmented versus non-segmented models. A scaled periodic model is
    //! defined here as follows:
    //! <pre class="fragment">
    //!   \f$ \sum_i 1\{t_i \leq t < t_{i+1}\} s_i f_p(t) \f$
    //! </pre>
    //! where \f$\{s_i\} are a collection of scales and \f$f_p(\cdot)\f$ denotes a
    //! function with period \p period.
    //!
    //! \param[in] values The time series values to segment. These are assumed to
    //! be equally spaced in time order.
    //! \param[in] period The period of the underlying periodic component.
    //! \param[in] significanceToSegment The maximum significance of the decrease
    //! in the residual variance to accept a partition.
    //! \param[in] outlierFraction The proportion of values to treat as outliers.
    //! Outliers are re-weighted when doing model fit and computing unexplained
    //! variance for model selection. Must be in the range [0.0, 1.0).
    //! \param[in] outlierWeight The weight to apply to outliers when doing model
    //! fit and computing unexplained variance. Must be in the range [0.0, 1.0].
    //! \return The segmentation of values otherwise the start and end indices.
    static TSizeVec
    topDownPeriodicPiecewiseLinearScaled(const TFloatMeanAccumulatorVec& values,
                                         std::size_t period,
                                         double significanceToSegment = 0.01,
                                         double outlierFraction = 0.0,
                                         double outlierWeight = 0.1);

    //! Remove the predictions of a periodic model with piecewise linear scaling.
    //!
    //! \param[in] values The time series values to segment. These are assumed to
    //! be equally spaced in time order.
    //! \param[in] period The period of the underlying periodic component.
    //! \param[in] segmentation The segmentation of \p values to use.
    //! \param[in] outlierFraction The proportion of values to treat as outliers.
    //! Outliers are re-weighted when doing model fit and computing unexplained
    //! variance for model selection. Must be in the range [0.0, 1.0).
    //! \param[in] outlierWeight The weight to apply to outliers when doing model
    //! fit and computing unexplained variance. Must be in the range [0.0, 1.0].
    //! \return The values minus the model predictions.
    static TFloatMeanAccumulatorVec
    removePredictionsOfPiecewiseLinearScaled(const TFloatMeanAccumulatorVec& values,
                                             std::size_t period,
                                             const TSizeVec& segmentation,
                                             double outlierFraction = 0.0,
                                             double outlierWeight = 0.1);

private:
    using TDoubleVec = std::vector<double>;

private:
    //! Implements top-down recursive segmentation with linear models.
    template<typename ITR>
    static void topDownPiecewiseLinear(ITR begin,
                                       ITR end,
                                       std::size_t offset,
                                       double startTime,
                                       double dt,
                                       double significanceToSegment,
                                       double outliersFraction,
                                       double outlierWeight,
                                       TSizeVec& segmentation);

    //! Implements top-down recursive segmentation with linear models.
    template<typename ITR>
    static void topDownPeriodicPiecewiseLinearScaled(ITR begin,
                                                     ITR end,
                                                     std::size_t offset,
                                                     const TDoubleVec& model,
                                                     double significanceToSegment,
                                                     TSizeVec& segmentation);
};
}
}

#endif // INCLUDE_ml_maths_CTimeSeriesSegmentation_h
