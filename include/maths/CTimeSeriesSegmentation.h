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

#include <limits>
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
    using TTimeVec = std::vector<core_t::TTime>;
    using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
    using TFloatMeanAccumulatorVecDoubleVecPr = std::pair<TFloatMeanAccumulatorVec, TDoubleVec>;
    using TSeasonalComponent = CSignal::SSeasonalComponentSummary;
    using TSeasonalComponentVec = std::vector<TSeasonalComponent>;
    using TConstantScale = std::function<double(const TSizeVec&, const TDoubleVec&)>;
    using TIndexWeight = std::function<double(std::size_t)>;
    using TSeasonality = std::function<double(std::size_t)>;
    using TModel = std::function<double(core_t::TTime)>;

    //! Perform top-down recursive segmentation with linear models.
    //!
    //! The time series is segmented using piecewise linear models in a top down
    //! fashion with each break point being chosen to minimise the residual variance.
    //! Model selection happens for each candidate segment. This is uses the p-value
    //! of the variance the segment explains.
    //!
    //! \param[in] values The time series values to segment. These are assumed to
    //! be equally spaced in time order.
    //! \param[in] pValueToSegment The maximum p-value of the explained variance
    //! to accept a trend segment.
    //! \param[in] outlierFraction The proportion of values to treat as outliers.
    //! This must be in the range (0.0, 1.0).
    //! \param[in] maxSegments The maximum number of segments to divide \p values into.
    //! \return The sorted segmentation indices. This includes the start and end
    //! indices of \p values, i.e. 0 and values.size().
    static TSizeVec
    piecewiseLinear(const TFloatMeanAccumulatorVec& values,
                    double pValueToSegment,
                    double outlierFraction,
                    std::size_t maxSegments = std::numeric_limits<std::size_t>::max());

    //! Remove the predictions of a piecewise linear model.
    //!
    //! \param[in] values The time series values.
    //! \param[in] segmentation The segmentation of \p values to use.
    //! \param[in] outlierFraction The proportion of values to treat as outliers.
    //! This must be in the range (0.0, 1.0).
    //! \param[out] shifts Filled in with the level shifts between each segment.
    //! \return \p values minus the model predictions.
    static TFloatMeanAccumulatorVec removePiecewiseLinear(TFloatMeanAccumulatorVec values,
                                                          const TSizeVec& segmentation,
                                                          double outlierFraction,
                                                          TDoubleVec& shifts);

    //! Overload if shifts aren't wanted.
    static TFloatMeanAccumulatorVec removePiecewiseLinear(TFloatMeanAccumulatorVec values,
                                                          const TSizeVec& segmentation,
                                                          double outlierFraction) {
        TDoubleVec shifts;
        return removePiecewiseLinear(values, segmentation, outlierFraction, shifts);
    }

    //! Remove only the jump discontinuities in the segmented model.
    //!
    //! This removes discontinuities corresponding to the models in adjacent segments
    //! in backwards pass such that values in each preceding segment are adjusted
    //! relative to each succeeding segment.
    //!
    //! \param[in] values The time series values.
    //! \param[in] segmentation The segmentation of \p values to use.
    //! \param[in] outlierFraction The proportion of values to treat as outliers.
    //! This must be in the range (0.0, 1.0).
    //! \return \p values minus discontinuities at the trend knot points.
    static TFloatMeanAccumulatorVec
    removePiecewiseLinearDiscontinuities(TFloatMeanAccumulatorVec values,
                                         const TSizeVec& segmentation,
                                         double outlierFraction);

    //! Perform top-down recursive segmentation of a seasonal model into segments with
    //! constant linear scale.
    //!
    //! The time series is segmented using piecewise constant linear scaled seasonal
    //! model in a top down fashion with break points being chosen to minimise residual
    //! Model selection happens for each candidate segment. This is via the p-value
    //! of the variance the segment explains. Formally, a scaled seasonal model is
    //! defined here as:
    //! <pre class="fragment">
    //!   \f$ \sum_i 1\{t_i \leq t < t_{i+1}\} s_i f_p(t) \f$
    //! </pre>
    //! where \f$\{s_i\}\f$ are a collection of constant scales and \f$f_p(\cdot)\f$
    //! denotes a function with period \p period.
    //!
    //! \param[in] values The time series values to segment. These are assumed to be
    //! equally spaced in time order.
    //! \param[in] model A model of the seasonality to segment which returns its value
    //! for the i'th bucket of \p values.
    //! \param[in] pValueToSegment The maximum p-value of the explained variance
    //! to accept a scaling segment.
    //! \param[in] maxSegments The maximum number of segments to divide \p values into.
    //! \return The sorted segmentation indices. This includes the start and end
    //! indices of \p values, i.e. 0 and values.size().
    static TSizeVec piecewiseLinearScaledSeasonal(
        const TFloatMeanAccumulatorVec& values,
        const TSeasonality& model,
        double pValueToSegment,
        std::size_t maxSegments = std::numeric_limits<std::size_t>::max());

    //! Remove the scaled predictions of \p model from \p values.
    //!
    //! This fits a piecewise linear scaled \p model on \p segmentation to minimise the
    //! residuals from \p values and returns \p values minus the scaled predictions.
    //!
    //! \param[in] model A model of the seasonality to remove which returns its value
    //! for the i'th bucket of \p values.
    //! \param[in] segmentation The segmentation of \p values into intervals with
    //! constant scale.
    //! \param[in] outlierFraction The proportion of values to treat as outliers.
    //! This must be in the range (0.0, 1.0).
    //! \param[out] scales Filled in with the scales for each segment.
    //! \return The values minus the scaled model predictions.
    static TFloatMeanAccumulatorVec
    removePiecewiseLinearScaledSeasonal(TFloatMeanAccumulatorVec values,
                                        const TSeasonality& model,
                                        const TSizeVec& segmentation,
                                        double outlierFraction,
                                        TDoubleVec& scales);

    //! Overload if scales aren't wanted.
    static TFloatMeanAccumulatorVec
    removePiecewiseLinearScaledSeasonal(TFloatMeanAccumulatorVec values,
                                        const TSeasonality& model,
                                        const TSizeVec& segmentation,
                                        double outlierFraction) {
        TDoubleVec scales;
        return removePiecewiseLinearScaledSeasonal(values, model, segmentation,
                                                   outlierFraction, scales);
    }

    //! Rescale \p values on \p segmentation to minimise residual variance for seasonal
    //! components with periods \p periods.
    //!
    //! \param[in] values The values to scale.
    //! \param[in] periods The seasonal components present in \p values.
    //! \param[in] segmentation The segmentation of \p values into intervals with
    //! constant scale.
    //! \param[in] computeConstantScale Computes the constant scale to apply to the
    //! seasonal components of \p values.
    //! \param[in] outlierFraction The proportion of values to treat as outliers.
    //! This must be in the range (0.0, 1.0).
    //! \param[out] models The component models.
    //! \param[out] scales The scales to apply to \p models in each segment.
    //! \return The values with the mean scaled seasonal component.
    static TFloatMeanAccumulatorVec
    constantScalePiecewiseLinearScaledSeasonal(const TFloatMeanAccumulatorVec& values,
                                               const TSeasonalComponentVec& periods,
                                               const TSizeVec& segmentation,
                                               const TConstantScale& computeConstantScale,
                                               double outlierFraction,
                                               TDoubleVecVec& models,
                                               TDoubleVec& scales);

    //! Compute the weighted mean scale for the piecewise linear \p scales on
    //! \p segmentation.
    //!
    //! \param[in] segmentation The segmentation into intervals with constant scale.
    //! \param[in] scales The piecewise constant linear scales.
    //! \param[in] indexWeight A function used to weight indices of \p segmentation.
    static double
    meanScale(const TSizeVec& segmentation,
              const TDoubleVec& scales,
              const TIndexWeight& indexWeight = [](std::size_t) { return 1.0; });

    //! Compute the scale to use at \p index for the piecewise linear \p scales on
    //! \p segmentation.
    //!
    //! \param[in] index The index at which to compute the scale.
    //! \param[in] segmentation The segmentation into intervals with constant scale.
    //! \param[in] scales The piecewise constant linear scales to apply.
    //! \return The scale at \p index.
    static double scaleAt(std::size_t index, const TSizeVec& segmentation, const TDoubleVec& scales);

    //! Perform top-down recursive segmentation of a seasonal model into segments with
    //! constant time shift.
    //!
    //! \param[in] bucketLength The time bucket length of \p values.
    //! \param[in] candidateShifts The time shifts we'll consider applying.
    //! \param[in] model A model of the seasonality to segment which returns its value
    //! for the i'th bucket of \p values.
    //! \param[in] pValueToSegment The maximum p-value of the explained variance
    //! to accept a scaling segment.
    //! \param[in] maxSegments The maximum number of segments to divide \p values into.
    //! \param[out] shifts If not null filled in with the shift for each segment.
    //! \return The sorted segmentation indices. This includes the start and end
    //! indices of \p values, i.e. 0 and values.size().
    static TSizeVec
    piecewiseTimeShifted(const TFloatMeanAccumulatorVec& values,
                         core_t::TTime bucketLength,
                         const TTimeVec& candidateShifts,
                         const TModel& model,
                         double pValueToSegment,
                         std::size_t maxSegments = std::numeric_limits<std::size_t>::max(),
                         TTimeVec* shifts = nullptr);

    //! Compute the time shift to use at \p index for the piecewise constant \p shifts
    //! on \p segmentation.
    //!
    //! \param[in] index The index at which to compute the shift.
    //! \param[in] segmentation The segmentation into intervals with constant time shifts.
    //! \param[in] shifts The piecewise constant time shift to apply.
    //! \return The shift at \p index.
    static core_t::TTime
    shiftAt(std::size_t index, const TSizeVec& segmentation, const TTimeVec& shifts);

private:
    using TDoubleDoublePr = std::pair<double, double>;
    using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using TMeanVarAccumulatorSizePr = std::pair<TMeanVarAccumulator, std::size_t>;
    using TRegression = CLeastSquaresOnlineRegression<1, double>;
    using TRegressionArray = TRegression::TArray;
    using TPredictor = std::function<double(double)>;
    using TScale = std::function<double(std::size_t)>;

private:
    //! Choose the final segmentation we'll use.
    static void selectSegmentation(std::size_t maxSegments,
                                   TSizeVec& segmentation,
                                   TDoubleDoublePrVec& depthAndPValue);

    //! Implements top-down recursive segmentation of [\p begin, \p end) to minimise
    //! square residuals for a linear model.
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

    //! Implements top-down recursive segmentation of [\p begin, \p end) to minimise
    //! square residuals for linear scales of \p model.
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

    //! Fit \p model with piecewise constant linear scaling to \p values for the
    //! segmentation \p segmentation.
    static void fitPiecewiseLinearScaledSeasonal(const TFloatMeanAccumulatorVec& values,
                                                 const TSeasonality& model,
                                                 const TSizeVec& segmentation,
                                                 double outlierFraction,
                                                 TFloatMeanAccumulatorVec& reweighted,
                                                 TDoubleVec& scales);

    //! Fit a seasonal model with piecewise constant linear scaling to \p values
    //! for the segmentation \p segmentation.
    static void fitPiecewiseLinearScaledSeasonal(const TFloatMeanAccumulatorVec& values,
                                                 const TSeasonalComponentVec& periods,
                                                 const TSizeVec& segmentation,
                                                 double outlierFraction,
                                                 TFloatMeanAccumulatorVec& reweighted,
                                                 TDoubleVecVec& model,
                                                 TDoubleVec& scales);

    //! Implements top-down recursive segmentation of [\p begin, \p end) to minimise
    //! square residuals for time shifts of a base model \p predictions.
    template<typename ITR>
    static void fitTopDownPiecewiseTimeShifted(ITR begin,
                                               ITR end,
                                               std::size_t depth,
                                               std::size_t offset,
                                               const TDoubleVecVec& predictions,
                                               std::size_t maxDepth,
                                               double pValueToSegment,
                                               TSizeVec& segmentation,
                                               TDoubleDoublePrVec& depthAndPValue);

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

    //! Compute the moments of the values in [\p begin, \p end) after subtracting
    //! \p predictions minimising variance over the outer index.
    template<typename ITR>
    static TMeanVarAccumulatorSizePr
    residualMoments(ITR begin, ITR end, std::size_t offset, const TDoubleVecVec& predictions);

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
