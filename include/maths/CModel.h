/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CModel_h
#define INCLUDED_ml_maths_CModel_h

#include <core/CSmallVector.h>
#include <core/CTriple.h>
#include <core/CoreTypes.h>

#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <boost/optional.hpp>

#include <cstddef>
#include <cstdint>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {
class CMultivariatePrior;
class CPrior;
class CTimeSeriesCorrelations;

//! \brief Data describing a prediction error bar.
struct MATHS_EXPORT SErrorBar {
    core_t::TTime s_Time;
    core_t::TTime s_BucketLength;
    double s_LowerBound;
    double s_Predicted;
    double s_UpperBound;
};

using TForecastPushDatapointFunc = std::function<void(SErrorBar)>;

//! \brief Model parameters.
class MATHS_EXPORT CModelParams {
public:
    CModelParams(core_t::TTime bucketLength,
                 double learnRate,
                 double decayRate,
                 double minimumSeasonalVarianceScale,
                 core_t::TTime minimumTimeToDetectChange,
                 core_t::TTime maximumTimeToTestForChange);

    //! Get the bucket length.
    core_t::TTime bucketLength() const;

    //! Get the model learn rate.
    double learnRate() const;

    //! Get the model decay rate.
    double decayRate() const;

    //! Get the decay rate to use for time averaging the model decay.
    double averagingDecayRate() const;

    //! Get the minimum seasonal variance scale.
    double minimumSeasonalVarianceScale() const;

    //! Check if we should start testing for a change point in the model.
    bool testForChange(core_t::TTime changeInterval) const;

    //! Get the minimum time to detect a change point in the model.
    core_t::TTime minimumTimeToDetectChange() const;

    //! Get the maximum time to test for a change point in the model.
    core_t::TTime maximumTimeToTestForChange() const;

private:
    //! The data bucketing length.
    core_t::TTime m_BucketLength;
    //! The model learn rate.
    double m_LearnRate;
    //! The model decay rate.
    double m_DecayRate;
    //! The minimum seasonal variance scale.
    double m_MinimumSeasonalVarianceScale;
    //! The minimum time permitted to detect a change in the model.
    core_t::TTime m_MinimumTimeToDetectChange;
    //! The maximum time permitted to test for a change in the model.
    core_t::TTime m_MaximumTimeToTestForChange;
};

//! \brief The extra parameters needed by CModel::addSamples.
class MATHS_EXPORT CModelAddSamplesParams {
public:
    using TDouble2Vec = core::CSmallVector<double, 2>;
    using TDouble2VecWeightsAryVec = std::vector<maths_t::TDouble2VecWeightsAry>;

public:
    //! Set whether or not the data are integer valued.
    CModelAddSamplesParams& integer(bool integer);
    //! Get the data type.
    maths_t::EDataType type() const;

    //! Set whether or not the data are non-negative.
    CModelAddSamplesParams& nonNegative(bool nonNegative);
    //! Get the whether the data are non-negative.
    bool isNonNegative() const;

    //! Set the model propagation interval.
    CModelAddSamplesParams& propagationInterval(double interval);
    //! Get the model propagation interval.
    double propagationInterval() const;

    //! Set the trend samples weights.
    CModelAddSamplesParams& trendWeights(const TDouble2VecWeightsAryVec& weights);
    //! Get the trend sample weights.
    const TDouble2VecWeightsAryVec& trendWeights() const;

    //! Set the prior samples weights.
    CModelAddSamplesParams& priorWeights(const TDouble2VecWeightsAryVec& weights);
    //! Get the prior sample weights.
    const TDouble2VecWeightsAryVec& priorWeights() const;

    //! Set the model annotation callback.
    CModelAddSamplesParams&
    annotationCallback(const maths_t::TModelAnnotationCallback& modelAnnotationCallback);
    //! Get the model annotation callback.
    const maths_t::TModelAnnotationCallback& annotationCallback() const;

private:
    //! The data type.
    maths_t::EDataType m_Type = maths_t::E_MixedData;
    //! True if the data are non-negative false otherwise.
    bool m_IsNonNegative = false;
    //! The propagation interval.
    double m_PropagationInterval = 1.0;
    //! The trend sample weights.
    const TDouble2VecWeightsAryVec* m_TrendWeights = nullptr;
    //! The prior sample weights.
    const TDouble2VecWeightsAryVec* m_PriorWeights = nullptr;
    //! The add annotation callback.
    maths_t::TModelAnnotationCallback m_ModelAnnotationCallback = [](const std::string&) {};
};

//! \brief The extra parameters needed by CModel::probability.
class MATHS_EXPORT CModelProbabilityParams {
public:
    using TOptionalSize = boost::optional<std::size_t>;
    using TBool2Vec = core::CSmallVector<bool, 2>;
    using TBool2Vec1Vec = core::CSmallVector<TBool2Vec, 2>;
    using TSize2Vec = core::CSmallVector<std::size_t, 2>;
    using TDouble2VecWeightsAry = maths_t::TDouble2VecWeightsAry;
    using TDouble2VecWeightsAry1Vec = maths_t::TDouble2VecWeightsAry1Vec;
    using TProbabilityCalculation2Vec =
        core::CSmallVector<maths_t::EProbabilityCalculation, 2>;

public:
    CModelProbabilityParams();

    //! Add a coordinate's calculation style.
    CModelProbabilityParams& addCalculation(maths_t::EProbabilityCalculation calculation);
    //! Get the number of calculations.
    std::size_t calculations() const;
    //! Get the \p i'th coordinate's calculation style.
    maths_t::EProbabilityCalculation calculation(std::size_t i) const;

    //! Set the confidence interval to use when detrending.
    CModelProbabilityParams& seasonalConfidenceInterval(double confidence);
    //! Get the confidence interval to use when detrending.
    double seasonalConfidenceInterval() const;

    //! Add a value's weights.
    CModelProbabilityParams& addWeights(const TDouble2VecWeightsAry& weights);
    //! Set the values' weights.
    CModelProbabilityParams& weights(const TDouble2VecWeightsAry1Vec& weights);
    //! Get the values' weights.
    const TDouble2VecWeightsAry1Vec& weights() const;
    //! Get writable values' weights.
    TDouble2VecWeightsAry1Vec& weights();

    //! Add a coordinate for which to compute probability.
    CModelProbabilityParams& addCoordinate(std::size_t coordinate);
    //! Get the coordinates for which to compute probability.
    const TSize2Vec& coordinates() const;

    //! Set the most anomalous correlate.
    CModelProbabilityParams& mostAnomalousCorrelate(std::size_t correlate);
    //! Get the most anomalous correlate if there is one.
    TOptionalSize mostAnomalousCorrelate() const;

    //! Set whether or not to use multibucket features.
    CModelProbabilityParams& useMultibucketFeatures(bool use);
    //! Get whether or not to use multibucket features.
    bool useMultibucketFeatures() const;

    //! Set whether or not to use the anomaly model.
    CModelProbabilityParams& useAnomalyModel(bool use);
    //! Get whether or not to use the anomaly model.
    bool useAnomalyModel() const;

    //! Set whether or not to skip updating the anomaly model.
    CModelProbabilityParams& skipAnomalyModelUpdate(bool skipAnomalyModelUpdate);
    //! Get whether or not to skip updating the anomaly model.
    bool skipAnomalyModelUpdate() const;

private:
    //! The coordinates' probability calculations.
    TProbabilityCalculation2Vec m_Calculations;
    //! The confidence interval to use when detrending.
    double m_SeasonalConfidenceInterval;
    //! The sample weights.
    TDouble2VecWeightsAry1Vec m_Weights;
    //! The coordinates for which to compute the probability.
    TSize2Vec m_Coordinates;
    //! The most anomalous coordinate (if there is one).
    TOptionalSize m_MostAnomalousCorrelate;
    //! Whether or not to use multibucket features.
    bool m_UseMultibucketFeatures = true;
    //! Whether or not to use the anomaly model.
    bool m_UseAnomalyModel = true;
    //! Whether or not to skip updating the anomaly model
    //! because a rule triggered.
    bool m_SkipAnomalyModelUpdate = false;
};

//! \brief Describes the result of the model probability calculation.
struct MATHS_EXPORT SModelProbabilityResult {
    using TDouble4Vec = core::CSmallVector<double, 4>;
    using TSize1Vec = core::CSmallVector<std::size_t, 1>;
    using TTail2Vec = core::CSmallVector<maths_t::ETail, 2>;

    //! Labels for different contributions to the overall probability.
    enum EFeatureProbabilityLabel {
        E_SingleBucketProbability,
        E_MultiBucketProbability,
        E_AnomalyModelProbability,
        E_UndefinedProbability
    };

    //! \brief Wraps up a feature label and probability.
    struct MATHS_EXPORT SFeatureProbability {
        SFeatureProbability();
        SFeatureProbability(EFeatureProbabilityLabel label, double probability);
        EFeatureProbabilityLabel s_Label;
        double s_Probability = 1.0;
    };
    using TFeatureProbability4Vec = core::CSmallVector<SFeatureProbability, 4>;

    //! The overall result probability.
    double s_Probability = 1.0;
    //! True if the probability depends on the correlation between two
    //! time series and false otherwise.
    bool s_Conditional = false;
    //! The probabilities for each individual feature.
    TFeatureProbability4Vec s_FeatureProbabilities;
    //! The tail of the current bucket probability.
    TTail2Vec s_Tail;
    //! The identifier of the time series correlated with this one which
    //! has the smallest probability in the current bucket (if and only
    //! if the result depends on the correlation structure).
    TSize1Vec s_MostAnomalousCorrelate;
};

//! \brief The model interface.
//!
//! DESCRIPTION:\n
//! The aim of the model hierarchy is to wrap up functionality to model
//! various types of object behind a simple interface which is used to
//! implement anomaly detection. In particular,
//! models can be:
//!   -# Updated with new values
//!   -# Predicted
//!   -# Forecast over an extended range
//!   -# Used to compute the probability of seeing a less likely event.
//!
//! Specific implementations exist for different types of object. For example,
//! for univariate and multivariate time series.
class MATHS_EXPORT CModel {
public:
    using TBool2Vec = core::CSmallVector<bool, 2>;
    using TDouble2Vec = core::CSmallVector<double, 2>;
    using TDouble10Vec = core::CSmallVector<double, 10>;
    using TDouble2Vec1Vec = core::CSmallVector<TDouble2Vec, 1>;
    using TDouble2Vec3Vec = core::CSmallVector<TDouble2Vec, 3>;
    using TSize1Vec = core::CSmallVector<std::size_t, 1>;
    using TSize2Vec = core::CSmallVector<std::size_t, 2>;
    using TSize2Vec1Vec = core::CSmallVector<TSize2Vec, 1>;
    using TTime2Vec = core::CSmallVector<core_t::TTime, 2>;
    using TTime2Vec1Vec = core::CSmallVector<TTime2Vec, 1>;
    using TSizeDoublePr = std::pair<std::size_t, double>;
    using TSizeDoublePr1Vec = core::CSmallVector<TSizeDoublePr, 1>;
    using TTimeDouble2VecSizeTr = core::CTriple<core_t::TTime, TDouble2Vec, std::size_t>;
    using TTimeDouble2VecSizeTrVec = std::vector<TTimeDouble2VecSizeTr>;
    using TDouble2VecWeightsAry = maths_t::TDouble2VecWeightsAry;
    using TDouble2VecWeightsAry1Vec = maths_t::TDouble2VecWeightsAry1Vec;
    using TTail2Vec = core::CSmallVector<maths_t::ETail, 2>;

    //! Possible statuses for updating a model.
    enum EUpdateResult {
        E_Failure, //!< Update failed.
        E_Success, //!< Update succeeded.
        E_Reset    //!< Model reset.
    };

public:
    explicit CModel(const CModelParams& params);
    virtual ~CModel() = default;

    //! These don't need to be and shouldn't be copied.
    const CModel& operator=(const CModel&) = delete;

    //! Get the effective count per correlate model for calibrating aggregation.
    static double effectiveCount(std::size_t n);

    //! Get the model identifier.
    virtual std::size_t identifier() const = 0;

    //! Create a copy of this model passing ownership to the caller.
    virtual CModel* clone(std::size_t id) const = 0;

    //! Create a copy of the state we need to persist passing ownership to the caller.
    virtual CModel* cloneForPersistence() const = 0;

    //! Create a copy of the state we need to run forecasting.
    virtual CModel* cloneForForecast() const = 0;

    //! Return true if forecast is currently possible for this model.
    virtual bool isForecastPossible() const = 0;

    //! Tell this to model correlations.
    virtual void modelCorrelations(CTimeSeriesCorrelations& model) = 0;

    //! Get the correlated time series identifier pairs if any.
    virtual TSize2Vec1Vec correlates() const = 0;

    //! Update the model with the bucket \p value.
    virtual void addBucketValue(const TTimeDouble2VecSizeTrVec& value) = 0;

    //! Update the model with new samples.
    virtual EUpdateResult addSamples(const CModelAddSamplesParams& params,
                                     TTimeDouble2VecSizeTrVec samples) = 0;

    //! Advance time by \p gap.
    virtual void skipTime(core_t::TTime gap) = 0;

    //! Get the most likely value for the time series at \p time.
    virtual TDouble2Vec mode(core_t::TTime time,
                             const TDouble2VecWeightsAry& weights) const = 0;

    //! Get the most likely value for each correlate time series at
    //! \p time, if there are any.
    virtual TDouble2Vec1Vec
    correlateModes(core_t::TTime time, const TDouble2VecWeightsAry1Vec& weights) const = 0;

    //! Get the local maxima of the residual distribution.
    virtual TDouble2Vec1Vec residualModes(const TDouble2VecWeightsAry& weights) const = 0;

    //! Remove any trend components from \p value.
    virtual void detrend(const TTime2Vec1Vec& time,
                         double confidenceInterval,
                         TDouble2Vec1Vec& value) const = 0;

    //! Get the best (least MSE) predicted value at \p time.
    virtual TDouble2Vec predict(core_t::TTime time,
                                const TSizeDoublePr1Vec& correlated = TSizeDoublePr1Vec(),
                                TDouble2Vec hint = TDouble2Vec()) const = 0;

    //! Get the prediction and \p confidenceInterval percentage
    //! confidence interval for the time series at \p time.
    virtual TDouble2Vec3Vec confidenceInterval(core_t::TTime time,
                                               double confidenceInterval,
                                               const TDouble2VecWeightsAry& weights) const = 0;

    //! Forecast the time series and get its \p confidenceInterval
    //! percentage confidence interval between \p startTime and
    //! \p endTime.
    //!
    //! Data is pushed to the given \p forecastPushDataPointFunc
    //!
    //! \param[in] firstDataTime The first time data was added to the model.
    //! \param[in] lastDataTime The last time data was added to the model.
    //! \param[in] startTime The start time of the forecast.
    //! \param[in] endTime The start time of the forecast.
    //! \param[in] confidenceInterval The forecast confidence interval.
    //! \param[in] minimum The minimum permitted forecast value.
    //! \param[in] maximum The minimum permitted forecast value.
    //! \param[out] messageOut Filled in with any message generated
    //! generated whilst forecasting.
    //! \return true if forecast completed, false otherwise, in
    //! which case \p[out] messageOut is set.
    virtual bool forecast(core_t::TTime firstDataTime,
                          core_t::TTime lastDataTime,
                          core_t::TTime startTime,
                          core_t::TTime endTime,
                          double confidenceInterval,
                          const TDouble2Vec& minimum,
                          const TDouble2Vec& maximum,
                          const TForecastPushDatapointFunc& forecastPushDataPointFunc,
                          std::string& messageOut) = 0;

    //! Compute the probability of drawing \p value at \p time.
    virtual bool probability(const CModelProbabilityParams& params,
                             const TTime2Vec1Vec& time,
                             const TDouble2Vec1Vec& value,
                             SModelProbabilityResult& result) const = 0;

    //! Get the Winsorisation weight to apply to \p value,
    //! if appropriate.
    virtual TDouble2Vec winsorisationWeight(double derate,
                                            core_t::TTime time,
                                            const TDouble2Vec& value) const = 0;

    //! Get the seasonal variance scale at \p time.
    virtual TDouble2Vec seasonalWeight(double confidence, core_t::TTime time) const = 0;

    //! Compute a checksum for this object.
    virtual std::uint64_t checksum(std::uint64_t seed = 0) const = 0;

    //! Debug the memory used by this object.
    virtual void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const = 0;

    //! Get the memory used by this object.
    virtual std::size_t memoryUsage() const = 0;

    //! Persist by passing information to \p inserter.
    virtual void acceptPersistInserter(core::CStatePersistInserter& inserter) const = 0;

    //! Persist the state of the residual models only.
    virtual void persistModelsState(core::CStatePersistInserter& inserter) const = 0;

    //! Get the type of data being modeled.
    virtual maths_t::EDataType dataType() const = 0;

    //! Get read only model parameters.
    const CModelParams& params() const;

    //! Get writable model parameters.
    CModelParams& params();

protected:
    CModel(const CModel&) = default;

    //! Get the nearest mean of \p prior to \p detrended.
    template<typename VECTOR>
    static VECTOR marginalLikelihoodMean(const CPrior& prior);

    //! Get the nearest mean of \p prior to \p detrended.
    template<typename VECTOR>
    static VECTOR marginalLikelihoodMean(const CMultivariatePrior& prior);

    //! Get the error in the trend prediction for \p sample.
    template<typename TREND, typename VECTOR>
    static boost::optional<VECTOR> predictionError(const TREND& trend, const VECTOR& sample);

    //! Get the error in the prior prediction for \p sample.
    template<typename PRIOR, typename VECTOR>
    static boost::optional<VECTOR>
    predictionError(double propagationInterval, const PRIOR& prior, const VECTOR& sample);

private:
    //! The model parameters.
    CModelParams m_Params;
};

//! A stateless lightweight model which stubs the interface.
class MATHS_EXPORT CModelStub : public CModel {
public:
    CModelStub();

    //! Returns 0.
    std::size_t identifier() const override;

    //! Create a copy of this model passing ownership to the caller.
    CModelStub* clone(std::size_t id) const override;

    //! Create a copy of the state we need to persist passing ownership to the caller.
    CModelStub* cloneForPersistence() const override;

    //! Create a copy of the state we need to run forecasting.
    CModelStub* cloneForForecast() const override;

    //! Return false;
    bool isForecastPossible() const override;

    //! No-op.
    void modelCorrelations(CTimeSeriesCorrelations& model) override;

    //! Returns empty.
    TSize2Vec1Vec correlates() const override;

    //! No-op.
    void addBucketValue(const TTimeDouble2VecSizeTrVec& value) override;

    //! No-op.
    EUpdateResult addSamples(const CModelAddSamplesParams& params,
                             TTimeDouble2VecSizeTrVec samples) override;

    //! No-op.
    void skipTime(core_t::TTime gap) override;

    //! Returns empty.
    TDouble2Vec mode(core_t::TTime time, const TDouble2VecWeightsAry& weights) const override;

    //! Returns empty.
    TDouble2Vec1Vec correlateModes(core_t::TTime time,
                                   const TDouble2VecWeightsAry1Vec& weights) const override;

    //! Returns empty.
    TDouble2Vec1Vec residualModes(const TDouble2VecWeightsAry& weights) const override;

    //! No-op.
    void detrend(const TTime2Vec1Vec& time,
                 double confidenceInterval,
                 TDouble2Vec1Vec& value) const override;

    //! Returns empty.
    TDouble2Vec predict(core_t::TTime time,
                        const TSizeDoublePr1Vec& correlated,
                        TDouble2Vec hint = TDouble2Vec()) const override;

    //! Returns empty.
    TDouble2Vec3Vec confidenceInterval(core_t::TTime time,
                                       double confidenceInterval,
                                       const TDouble2VecWeightsAry& weights) const override;

    //! Returns empty.
    bool forecast(core_t::TTime firstDataTime,
                  core_t::TTime lastDataTime,
                  core_t::TTime startTime,
                  core_t::TTime endTime,
                  double confidenceInterval,
                  const TDouble2Vec& minimum,
                  const TDouble2Vec& maximum,
                  const TForecastPushDatapointFunc& forecastPushDataPointFunc,
                  std::string& messageOut) override;

    //! Returns true.
    bool probability(const CModelProbabilityParams& params,
                     const TTime2Vec1Vec& time,
                     const TDouble2Vec1Vec& value,
                     SModelProbabilityResult& result) const override;

    //! Returns empty.
    TDouble2Vec winsorisationWeight(double derate,
                                    core_t::TTime time,
                                    const TDouble2Vec& value) const override;

    //! Returns empty.
    TDouble2Vec seasonalWeight(double confidence, core_t::TTime time) const override;

    //! Returns the seed.
    std::uint64_t checksum(std::uint64_t seed = 0) const override;

    //! Debug the memory used by this object.
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const override;

    //! Get the memory used by this object.
    std::size_t memoryUsage() const override;

    //! No-op.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const override;

    //! Persist the state of the residual models only.
    void persistModelsState(core::CStatePersistInserter& inserter) const override;

    //! Returns mixed data type since we don't know.
    maths_t::EDataType dataType() const override;
};
}
}

#endif // INCLUDED_ml_maths_CModel_h
