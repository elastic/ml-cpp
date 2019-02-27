/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CTrendComponent_h
#define INCLUDED_ml_maths_CTrendComponent_h

#include <core/CoreTypes.h>

#include <maths/CBasicStatistics.h>
#include <maths/CLeastSquaresOnlineRegression.h>
#include <maths/CLinearAlgebraFwd.h>
#include <maths/CNaiveBayes.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/CPRNG.h>
#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <vector>

namespace ml {
namespace maths {
struct SDistributionRestoreParams;

//! \brief Models the trend component of a time series.
//!
//! DESCRIPTION:\n
//! This is an ensemble of trend models fitted over different time scales.
//! In particular, we age data at different rates from each of the models.
//! A prediction is then a weighted average of the different models. We
//! adjust the weighting of components based on the difference in their
//! decay rate and the target decay rate.
//!
//! The key advantage of this approach is that we can also adjust the
//! weighting over a forecast based on how far ahead we are predicting.
//! This means at each time scale we can revert to the trend for that time
//! scale. It also allows us to accurately estimate confidence intervals
//! (since these can be estimated from the variation of observed values
//! we see w.r.t. the predictions from the next longer time scale component).
//! This produces plausible looking forecasts and this sort of mean reversion
//! is common in many real world time series.
class MATHS_EXPORT CTrendComponent {
public:
    using TDoubleDoublePr = maths_t::TDoubleDoublePr;
    using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
    using TDoubleVec = std::vector<double>;
    using TDoubleVecVec = std::vector<TDoubleVec>;
    using TDouble3Vec = core::CSmallVector<double, 3>;
    using TDouble3VecVec = std::vector<TDouble3Vec>;
    using TVector = CVectorNx1<double, 3>;
    using TVectorVec = std::vector<TVector>;
    using TVectorVecVec = std::vector<TVectorVec>;
    using TMatrix = CSymmetricMatrixNxN<double, 3>;
    using TMatrixVec = std::vector<TMatrix>;
    using TSeasonalForecast = std::function<TDouble3Vec(core_t::TTime)>;
    using TWriteForecastResult = std::function<void(core_t::TTime, const TDouble3Vec&)>;

public:
    CTrendComponent(double decayRate);

    //! Efficiently swap the state of this and \p other.
    void swap(CTrendComponent& other);

    //! Persist state by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Initialize by reading state from \p traverser.
    bool acceptRestoreTraverser(const SDistributionRestoreParams& params,
                                core::CStateRestoreTraverser& traverser);

    //!  Check if the trend has been estimated.
    bool initialized() const;

    //! Clear all data.
    void clear();

    //! Shift the regression models' time origins to \p time.
    void shiftOrigin(core_t::TTime time);

    //! Shift the slope of the regression models' keeping the prediction
    //! at \p time fixed.
    void shiftSlope(core_t::TTime time, double shift);

    //! Apply a level shift of \p value at \p time and \p value.
    void shiftLevel(core_t::TTime time, double value, double shift);

    //! Apply no level shift at \p time and \p value.
    //!
    //! This updates the model for the probability of a level shift.
    void dontShiftLevel(core_t::TTime time, double value);

    //! Apply a linear scale by \p scale.
    void linearScale(double scale);

    //! Adds a value \f$(t, f(t))\f$ to this component.
    //!
    //! \param[in] time The time of the point.
    //! \param[in] value The value at \p time.
    //! \param[in] weight The weight of \p value. The smaller this is the
    //! less influence it has on the component.
    void add(core_t::TTime time, double value, double weight = 1.0);

    //! Set the data type.
    void dataType(maths_t::EDataType dataType);

    //! Get the base rate at which models lose information.
    double defaultDecayRate() const;

    //! Set the rate base rate at which models lose information.
    void decayRate(double decayRate);

    //! Age the trend to account for \p interval elapsed time.
    void propagateForwardsByTime(core_t::TTime interval);

    //! Get the predicted value at \p time.
    //!
    //! \param[in] time The time of interest.
    //! \param[in] confidence The symmetric confidence interval for the variance
    //! as a percentage.
    TDoubleDoublePr value(core_t::TTime time, double confidence) const;

    //! Get the variance of the residual about the predicted value at \p time.
    //!
    //! \param[in] confidence The symmetric confidence interval for the
    //! variance as a percentage.
    TDoubleDoublePr variance(double confidence) const;

    //! Forecast the trend model from \p startTime to \p endTime.
    //!
    //! \param[in] startTime The start time of the forecast interval.
    //! \param[in] endTime The end time of the forecast interval.
    //! \param[in] step The time step.
    //! \param[in] confidence The confidence interval to calculate.
    //! \param[in] seasonal Forecasts seasonal components.
    //! \param[in] writer Writes out forecast results.
    void forecast(core_t::TTime startTime,
                  core_t::TTime endTime,
                  core_t::TTime step,
                  double confidence,
                  const TSeasonalForecast& seasonal,
                  const TWriteForecastResult& writer) const;

    //! Get the interval which has been observed so far.
    core_t::TTime observedInterval() const;

    //! Get the number of parameters used to describe the trend.
    double parameters() const;

    //! Get a checksum for this object.
    uint64_t checksum(uint64_t seed = 0) const;

    //! Get a debug description of this object.
    std::string print() const;

private:
    using TRegression = CLeastSquaresOnlineRegression<2, double>;
    using TRegressionArray = TRegression::TArray;
    using TRegressionArrayVec = std::vector<TRegressionArray>;
    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;
    using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

    //! \brief A model of the trend at a specific time scale.
    struct SModel {
        explicit SModel(double weight);
        void acceptPersistInserter(core::CStatePersistInserter& inserter) const;
        bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);
        uint64_t checksum(uint64_t seed) const;
        TMeanAccumulator s_Weight;
        TRegression s_Regression;
        TMeanVarAccumulator s_ResidualMoments;
    };
    using TModelVec = std::vector<SModel>;

    //! \brief Forecasts the level model by path roll out.
    class CForecastLevel : private core::CNonCopyable {
    public:
        //! The default number of roll out paths to use.
        static const std::size_t DEFAULT_NUMBER_PATHS{100u};

    public:
        CForecastLevel(const CNaiveBayes& probability,
                       const CNormalMeanPrecConjugate& magnitude,
                       core_t::TTime timeOfLastChange,
                       std::size_t numberPaths = DEFAULT_NUMBER_PATHS);

        //! Forecast the time series level at \p time.
        TDouble3Vec forecast(core_t::TTime time, double prediction, double confidence);

    private:
        using TTimeVec = std::vector<core_t::TTime>;

    private:
        //! The model of the change probability.
        const CNaiveBayes& m_Probability;
        //! The model of the change magnitude.
        const CNormalMeanPrecConjugate& m_Magnitude;
        //! A random number generator for generating roll outs.
        CPRNG::CXorOShiro128Plus m_Rng;
        //! The current roll outs forecasted levels.
        TDoubleVec m_Levels;
        //! The current roll outs times of last change.
        TTimeVec m_TimesOfLastChange;
        //! Maintains the current bucket probability of change.
        TDoubleVec m_ProbabilitiesOfChange;
        //! Place holder for sampling.
        TDoubleVec m_Uniform01;
    };

private:
    //! Get the factors by which to age the different regression models.
    TDoubleVec factors(core_t::TTime interval) const;

    //! Get the initial weights to use for forecast predictions.
    TDoubleVec initialForecastModelWeights() const;

    //! Get the initial weights to use for forecast prediction errors.
    TDoubleVec initialForecastErrorWeights() const;

    //! Get the mean count of samples in the prediction.
    double count() const;

    //! Get the predicted value at \p time.
    double value(const TDoubleVec& weights, const TRegressionArrayVec& models, double time) const;

    //! Get the weight to assign to the prediction verses the long term mean.
    double weightOfPrediction(core_t::TTime time) const;

private:
    //! The default rate at which information is aged out of the trend models.
    double m_DefaultDecayRate;

    //! The target rate at which information is aged out of the ensemble.
    double m_TargetDecayRate;

    //! The time the model was first updated.
    core_t::TTime m_FirstUpdate;
    //! The time the model was last updated.
    core_t::TTime m_LastUpdate;

    //! The start time of the regression models.
    core_t::TTime m_RegressionOrigin;
    //! The regression models (we have them for multiple time scales).
    TModelVec m_TrendModels;
    //! The variance of the prediction errors.
    double m_PredictionErrorVariance;
    //! The mean and variance of the values added to the trend component.
    TMeanVarAccumulator m_ValueMoments;

    //! The time of the last level change.
    core_t::TTime m_TimeOfLastLevelChange;
    //! A model of probability of level changes for the trend.
    CNaiveBayes m_ProbabilityOfLevelChangeModel;
    //! A model of magnitude of level changes for the trend.
    CNormalMeanPrecConjugate m_MagnitudeOfLevelChangeModel;
};
}
}

#endif // INCLUDED_ml_maths_CTrendComponent_h
