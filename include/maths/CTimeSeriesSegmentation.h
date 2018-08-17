/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDE_ml_maths_CTimeSeriesSegmentation_h
#define INCLUDE_ml_maths_CTimeSeriesSegmentation_h

#include <maths/CBasicStatistics.h>
#include <maths/Constants.h>
#include <maths/MathsTypes.h>

#include <vector>

namespace ml {
namespace maths {

//! \brief Utility functionality to perform segmentation of a time series.
class MATHS_EXPORT CTimeSeriesSegmentation {
public:
    using TDoubleVec = std::vector<double>;
    using TDoubleVecDoubleVecPr = std::pair<TDoubleVec, TDoubleVec>;
    using TSizeVec = std::vector<std::size_t>;
    using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;

    //! Perform top-down recursive segmentation with linear models.
    //!
    //! The time series is segmented using piecewise linear models in a top down
    //! fashion with each break point being chosen to maximize r-squared and model
    //! selection happening for each candidate segmentation. Model selection
    //! is achieved by thresholding the significance of the unexplained variance
    //! ratio for segmented versus non-segmented models.
    //!
    //! \param[in] values The time series values to segment. These are assumed to
    //! be equally spaced in time order.
    //! \param[in] significanceToSegment The maximum significance of the decrease
    //! in the unexplained variance to accept a partition.
    //! \param[in] outlierFraction The proportion of values to treat as outliers.
    //! Outliers are re-weighted when doing model fit and computing unexplained
    //! variance for model selection. This must be in the range [0.0, 1.0).
    //! \param[in] outlierWeight The weight to apply to outliers when doing model
    //! fit and computing unexplained variance. This must be in the range [0.0, 1.0].
    //! \return The sorted segmentation indices. This includes the start and end
    //! indices of \p values, i.e. 0 and values.size().
    static TSizeVec piecewiseLinear(const TFloatMeanAccumulatorVec& values,
                                    double significanceToSegment = COMPONENT_STATISTICALLY_SIGNIFICANT,
                                    double outlierFraction = SEASONAL_OUTLIER_FRACTION,
                                    double outlierWeight = SEASONAL_OUTLIER_WEIGHT);

    //! Remove the predictions of a piecewise linear model.
    //!
    //! \param[in] segmentation The segmentation of \p values to use.
    //! \return The values minus the model predictions.
    static TFloatMeanAccumulatorVec
    removePiecewiseLinear(TFloatMeanAccumulatorVec values,
                          const TSizeVec& segmentation,
                          double outlierFraction = SEASONAL_OUTLIER_FRACTION,
                          double outlierWeight = SEASONAL_OUTLIER_WEIGHT);

    //! Remove only the jump discontinuities in the segmented model.
    //!
    //! This removes discontinuities corresponding to the models in adjacent segments
    //! in backwards pass such that values in each preceding segment are adjusted
    //! relative to each succeeding segment.
    //!
    //! \param[in] segmentation The segmentation of \p values to use.
    //! \return The values minus discontinuities.
    static TFloatMeanAccumulatorVec
    removePiecewiseLinearDiscontinuities(TFloatMeanAccumulatorVec values,
                                         const TSizeVec& segmentation,
                                         double outlierFraction = SEASONAL_OUTLIER_FRACTION,
                                         double outlierWeight = SEASONAL_OUTLIER_WEIGHT);

    //! Perform top-down recursive segmentation with a scaled periodic model.
    //!
    //! The time series is segmented using piecewise constant linear scaled periodic
    //! model in a top down fashion with break points being chosen to maximize r-squared
    //! and model selection happening for each candidate segmentation. Model selection
    //! is achieved by thresholding the significance of the unexplained variance ratio
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
    //! \param[in] period The period of the underlying model.
    //! \param[in] significanceToSegment The maximum significance of the decrease
    //! in the unexplained variance to accept a partition.
    //! \param[in] outlierFraction The proportion of values to treat as outliers.
    //! Outliers are re-weighted when doing model fit and computing unexplained
    //! variance for model selection. Must be in the range [0.0, 1.0).
    //! \param[in] outlierWeight The weight to apply to outliers when doing model
    //! fit and computing unexplained variance. Must be in the range [0.0, 1.0].
    //! \return The sorted segmentation indices. This includes the start and end
    //! indices of \p values, i.e. 0 and values.size().
    static TSizeVec
    piecewiseLinearScaledPeriodic(const TFloatMeanAccumulatorVec& values,
                                  std::size_t period,
                                  double significanceToSegment = COMPONENT_STATISTICALLY_SIGNIFICANT,
                                  double outlierFraction = SEASONAL_OUTLIER_FRACTION,
                                  double outlierWeight = SEASONAL_OUTLIER_WEIGHT);

    //! Compute the underlying periodic model with period \p period and piecewise
    //! constant linear scales on the segments defined by \p segmentation.
    //!
    //! \param[in] segmentation The segmentation of \p values into intervals
    //! with constant scale.
    //! \return A pair comprising (periodic model, scales).
    static TDoubleVecDoubleVecPr
    piecewiseLinearScaledPeriodic(const TFloatMeanAccumulatorVec& values,
                                  std::size_t period,
                                  const TSizeVec& segmentation,
                                  double outlierFraction = SEASONAL_OUTLIER_FRACTION,
                                  double outlierWeight = SEASONAL_OUTLIER_WEIGHT);

    //! Remove the predictions of a periodic model with period \p period and
    //! piecewise constant linear scales on the segments defined by \p segmentation.
    //!
    //! \param[in] segmentation The segmentation of \p values into intervals with
    //! constant scale.
    //! \return The values minus the model predictions.
    static TFloatMeanAccumulatorVec
    removePiecewiseLinearScaledPeriodic(const TFloatMeanAccumulatorVec& values,
                                        std::size_t period,
                                        const TSizeVec& segmentation,
                                        double outlierFraction = 0.0,
                                        double outlierWeight = 0.1);

    //! Remove the predictions of a periodic \p model with piecewise constant
    //! linear scales \p scales.
    //!
    //! \param[in] segmentation The segmentation of \p values into intervals with
    //! constant scale.
    //! \param[in] model The underlying periodic model.
    //! \param[in] scales The piecewise constant linear scales to apply to \p model.
    //! \return The values minus the scaled model predictions.
    static TFloatMeanAccumulatorVec
    removePiecewiseLinearScaledPeriodic(TFloatMeanAccumulatorVec values,
                                        const TSizeVec& segmentation,
                                        const TDoubleVec& model,
                                        const TDoubleVec& scales);

private:
    using TPredictor = std::function<double(double)>;
    using TDoubleVecFloatMeanAccumulatorVecPr = std::pair<TDoubleVec, TFloatMeanAccumulatorVec>;

private:
    //! Implements top-down recursive segmentation with linear models.
    template<typename ITR>
    static void fitTopDownPiecewiseLinear(ITR begin,
                                          ITR end,
                                          std::size_t offset,
                                          double startTime,
                                          double dt,
                                          double significanceToSegment,
                                          double outliersFraction,
                                          double outlierWeight,
                                          TSizeVec& segmentation);

    //! Fit a piecewise linear model to \p values for the segmentation
    //! \p segmentation.
    static TPredictor fitPiecewiseLinear(TFloatMeanAccumulatorVec& values,
                                         const TSizeVec& segmentation,
                                         double outlierFraction,
                                         double outlierWeight);

    //! Implements top-down recursive segmentation with a periodic model
    //! with piecewise constant linear scaling.
    template<typename ITR>
    static void fitTopDownPiecewiseLinearScaledPeriodic(ITR begin,
                                                        ITR end,
                                                        std::size_t offset,
                                                        const TDoubleVec& model,
                                                        double significanceToSegment,
                                                        TSizeVec& segmentation);

    //! Fit a periodic model with piecewise constant linear scaling to
    //! \p values for the segmentation \p segmentation.
    static void fitPiecewiseLinearScaledPeriodic(const TFloatMeanAccumulatorVec& values,
                                                 std::size_t period,
                                                 const TSizeVec& segmentation,
                                                 double outlierFraction,
                                                 double outlierWeight,
                                                 TFloatMeanAccumulatorVec& reweighted,
                                                 TDoubleVec& model,
                                                 TDoubleVec& scales);
};
}
}

#endif // INCLUDE_ml_maths_CTimeSeriesSegmentation_h
