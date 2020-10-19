/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDE_ml_maths_CTimeSeriesSegmentation_h
#define INCLUDE_ml_maths_CTimeSeriesSegmentation_h

#include <maths/CBasicStatistics.h>
#include <maths/CLeastSquaresOnlineRegression.h>
#include <maths/CSignal.h>
#include <maths/Constants.h>
#include <maths/MathsTypes.h>

#include <utility>
#include <vector>

namespace ml {
namespace maths {

//! \brief Utility functionality to perform segmentation of a time series.
class MATHS_EXPORT CTimeSeriesSegmentation {
public:
    using TDoubleVec = std::vector<double>;
    using TDoubleVecVec = std::vector<TDoubleVec>;
    using TDoubleVecDoubleVecPr = std::pair<TDoubleVec, TDoubleVec>;
    using TSizeVec = std::vector<std::size_t>;
    using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
    using TFloatMeanAccumulatorVecDoubleVecPr = std::pair<TFloatMeanAccumulatorVec, TDoubleVec>;
    using TSeasonalComponent = CSignal::SSeasonalComponentSummary;
    using TSeasonalComponentVec = std::vector<TSeasonalComponent>;
    using TSeasonality = std::function<double(std::size_t)>;
    using TIndexWeight = std::function<double(std::size_t)>;

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
    //! \param[in] pValueToSegment The maximum p-value of the explained variance
    //! to accept a trend segment.
    //! \param[in] outlierFraction The proportion of values to treat as outliers.
    //! Outliers are re-weighted when doing model fit and computing unexplained
    //! variance for model selection. This must be in the range [0.0, 1.0).
    //! \param[in] maxSegments The maximum number of segments to divide \p values
    //! into.
    //! \return The sorted segmentation indices. This includes the start and end
    //! indices of \p values, i.e. 0 and values.size().
    static TSizeVec
    piecewiseLinear(const TFloatMeanAccumulatorVec& values,
                    double pValueToSegment,
                    double outlierFraction,
                    std::size_t maxSegments = std::numeric_limits<std::size_t>::max());

    //! Remove the predictions of a piecewise linear model.
    //!
    //! \param[in] segmentation The segmentation of \p values to use.
    //! \return The values minus the model predictions.
    static TFloatMeanAccumulatorVec removePiecewiseLinear(TFloatMeanAccumulatorVec values,
                                                          const TSizeVec& segmentation,
                                                          double outlierFraction);

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
                                         double outlierFraction);

    //! Perform top-down recursive segmentation of seasonal model into segments with
    //! constant linear scale.
    //!
    //! The time series is segmented using piecewise constant linear scaled seasonal
    //! model in a top down fashion with break points being chosen to maximize r-squared
    //! and model selection happening for each candidate segmentation. Since each split
    //! generates a nested model we check the significance using the explained variance
    //! divided by segmented model's residual variance. A scaled seasonal model is
    //! defined here as:
    //! <pre class="fragment">
    //!   \f$ \sum_i 1\{t_i \leq t < t_{i+1}\} s_i f_p(t) \f$
    //! </pre>
    //! where \f$\{s_i\}\f$ are a collection of constant scales and \f$f_p(\cdot)\f$
    //! denotes a function with period \p period.
    //!
    //! \param[in] values The time series values to segment.
    //! \param[in] seasonality A model of the seasonality to segment which returns
    //! its value for the i'th bucket of \p values.
    //! \param[in] pValueToSegment The maximum p-value of the F-test to accept a
    //! scaling segment.
    //! \param[in] maxSegments The maximum number of segments to divide \p values
    //! into.
    //! \return The sorted segmentation indices. This includes the start and end
    //! indices of \p values, i.e. 0 and values.size().
    static TSizeVec piecewiseLinearScaledSeasonal(
        const TFloatMeanAccumulatorVec& values,
        const TSeasonality& seasonality,
        double pValueToSegment,
        std::size_t maxSegments = std::numeric_limits<std::size_t>::max());

    //! Rescale the piecewise linear scaled seasonal component of \p values with
    //! period \p period to its mean scale.
    //!
    //! \param[in] values The time series values to segment.
    //! \param[in] periods The seasonal components to model.
    //! \param[in] segmentation The segmentation of \p values into intervals with
    //! constant scale.
    //! \param[in] indexWeight A function used to weight indices of \p segmentation
    //! when computing the mean scale over \p values.
    //! \param[in] outlierFraction The proportion of values to treat as outliers.
    //! Outliers are re-weighted when doing model fit. This must be in the range
    //! [0.0, 1.0).
    //! \param[out] models Set to the fitted models for each period in \p periods.
    //! \param[out] scales Set to the scales for each segment of \p segmentation.
    //! \return The values with the mean scaled seasonal component.
    static TFloatMeanAccumulatorVec
    meanScalePiecewiseLinearScaledSeasonal(const TFloatMeanAccumulatorVec& values,
                                           const TSeasonalComponentVec& periods,
                                           const TSizeVec& segmentation,
                                           const TIndexWeight& indexWeight,
                                           double outlierFraction,
                                           TDoubleVecVec& models,
                                           TDoubleVec& scales);

    //! Compute the weighted mean scale for the piecewise linear \p scales on
    //! \p segmentation.
    //!
    //! \param[in] segmentation The segmentation into intervals with constant scale.
    //! \param[in] scales The piecewise constant linear scales.
    //! \param[in] indexWeight A function used to weight indices of \p segmentation.
    static double meanScale(const TSizeVec& segmentation,
                            const TDoubleVec& scales,
                            const TIndexWeight& indexWeight);

    //! Compute the scale at \p index for the piecewise linear \p scales on
    //! \p segmentation.
    //!
    //! \param[in] index The index at which to compute the scale.
    //! \param[in] segmentation The segmentation into intervals with constant scale.
    //! \param[in] scales The piecewise constant linear scales to apply.
    //! \return The scale at \p index.
    static double scaleAt(std::size_t index, const TSizeVec& segmentation, const TDoubleVec& scales);

private:
    using TDoubleDoublePr = std::pair<double, double>;
    using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using TRegression = CLeastSquaresOnlineRegression<1, double>;
    using TPredictor = std::function<double(double)>;
    using TScale = std::function<double(std::size_t)>;

private:
    //! Implements top-down recursive segmentation with linear models.
    template<typename ITR>
    static void fitTopDownPiecewiseLinear(ITR begin,
                                          ITR end,
                                          std::size_t depth,
                                          std::size_t offset,
                                          std::size_t maxDepth,
                                          double pValueToSegment,
                                          double outliersFraction,
                                          TSizeVec& segmentation,
                                          TDoubleDoublePrVec& depthAndPValue,
                                          TFloatMeanAccumulatorVec& values);

    //! Fit a piecewise linear model to \p values for the segmentation \p segmentation.
    static TPredictor fitPiecewiseLinear(const TSizeVec& segmentation,
                                         double outlierFraction,
                                         TFloatMeanAccumulatorVec& values);

    //! Implements top-down recursive segmentation of a seasonal model with
    //! piecewise constant linear scales.
    template<typename ITR>
    static void fitTopDownPiecewiseLinearScaledSeasonal(ITR begin,
                                                        ITR end,
                                                        std::size_t depth,
                                                        std::size_t offset,
                                                        const TSeasonality& model,
                                                        std::size_t maxDepth,
                                                        double pValueToSegment,
                                                        TSizeVec& segmentation,
                                                        TDoubleDoublePrVec& depthAndPValue);

    //! Fit a seasonal model with piecewise constant linear scaling to \p values
    //! for the segmentation \p segmentation.
    static void fitPiecewiseLinearScaledSeasonal(const TFloatMeanAccumulatorVec& values,
                                                 const TSeasonalComponentVec& periods,
                                                 const TSizeVec& segmentation,
                                                 double outlierFraction,
                                                 TFloatMeanAccumulatorVec& reweighted,
                                                 TDoubleVecVec& model,
                                                 TDoubleVec& scales);

    //! Compute the residual moments of a least squares linear model fit to
    //! [\p begin, \p end).
    template<typename ITR>
    static TMeanVarAccumulator centredResidualMoments(ITR begin, ITR end, double startTime);

    //! Compute the residual moments of a least squares scaled seasonal model fit to
    //! [\p begin, \p end).
    template<typename ITR>
    static TMeanVarAccumulator
    centredResidualMoments(ITR begin, ITR end, std::size_t offset, const TSeasonality& model);

    //! Compute the moments of the values in [\p begin, \p end) after subtracting
    //! the predictions of \p model.
    template<typename ITR>
    static TMeanVarAccumulator
    residualMoments(ITR begin, ITR end, double startTime, const TRegression& model);

    //! Fit a linear model to the values in [\p begin, \p end).
    template<typename ITR>
    static TRegression fitLinearModel(ITR begin, ITR end, double startTime);

    //! Fit a seasonal model of period \p period to the values [\p begin, \p end).
    template<typename ITR>
    static void fitSeasonalModel(ITR begin,
                                 ITR end,
                                 const TSeasonalComponent& period,
                                 const TPredictor& predictor,
                                 const TScale& scale,
                                 TDoubleVec& result);
};
}
}

#endif // INCLUDE_ml_maths_CTimeSeriesSegmentation_h
