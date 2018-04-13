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
    core_t::TTime minimumTimeToDetectChange(void) const;

    //! Get the maximum time to test for a change point in the model.
    core_t::TTime maximumTimeToTestForChange(void) const;

    //! Set the probability that the bucket will be empty for the model.
    void probabilityBucketEmpty(double probability);

    //! Get the probability that the bucket will be empty for the model.
    double probabilityBucketEmpty() const;

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
    //! The probability that a bucket will be empty for the model.
    double m_ProbabilityBucketEmpty;
};

//! \brief The extra parameters needed by CModel::addSamples.
class MATHS_EXPORT CModelAddSamplesParams {
public:
    using TDouble2Vec = core::CSmallVector<double, 2>;
    using TDouble2Vec4Vec = core::CSmallVector<TDouble2Vec, 4>;
    using TDouble2Vec4VecVec = std::vector<TDouble2Vec4Vec>;

public:
    CModelAddSamplesParams();

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

    //! Set the weight styles.
    CModelAddSamplesParams& weightStyles(const maths_t::TWeightStyleVec& styles);
    //! Get the weight styles.
    const maths_t::TWeightStyleVec& weightStyles() const;

    //! Set the trend samples weights.
    CModelAddSamplesParams& trendWeights(const TDouble2Vec4VecVec& weights);
    //! Get the trend sample weights.
    const TDouble2Vec4VecVec& trendWeights() const;

    //! Set the prior samples weights.
    CModelAddSamplesParams& priorWeights(const TDouble2Vec4VecVec& weights);
    //! Get the prior sample weights.
    const TDouble2Vec4VecVec& priorWeights() const;

private:
    //! The data type.
    maths_t::EDataType m_Type;
    //! True if the data are non-negative false otherwise.
    bool m_IsNonNegative;
    //! The propagation interval.
    double m_PropagationInterval;
    //! Controls the interpretation of the weights.
    const maths_t::TWeightStyleVec* m_WeightStyles;
    //! The trend sample weights.
    const TDouble2Vec4VecVec* m_TrendWeights;
    //! The prior sample weights.
    const TDouble2Vec4VecVec* m_PriorWeights;
};

//! \brief The extra parameters needed by CModel::probability.
class MATHS_EXPORT CModelProbabilityParams {
public:
    using TOptionalSize = boost::optional<std::size_t>;
    using TBool2Vec = core::CSmallVector<bool, 2>;
    using TBool2Vec1Vec = core::CSmallVector<TBool2Vec, 2>;
    using TDouble2Vec = core::CSmallVector<double, 2>;
    using TDouble2Vec4Vec = core::CSmallVector<TDouble2Vec, 4>;
    using TDouble2Vec4Vec1Vec = core::CSmallVector<TDouble2Vec4Vec, 2>;
    using TSize2Vec = core::CSmallVector<std::size_t, 2>;
    using TProbabilityCalculation2Vec = core::CSmallVector<maths_t::EProbabilityCalculation, 2>;

public:
    CModelProbabilityParams();

    //! Set the tag for the entity for which to compute the probability.
    CModelProbabilityParams& tag(std::size_t tag);
    //! Get the tag for the entity for which to compute the probability.
    std::size_t tag() const;

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

    //! Add whether a value's bucket is empty.
    CModelProbabilityParams& addBucketEmpty(const TBool2Vec& empty);
    //! Get whether the values' bucket is empty.
    const TBool2Vec1Vec& bucketEmpty() const;

    //! Set the weight styles.
    CModelProbabilityParams& weightStyles(const maths_t::TWeightStyleVec& styles);
    //! Get the weight styles.
    const maths_t::TWeightStyleVec& weightStyles() const;

    //! Add a value's weights.
    CModelProbabilityParams& addWeights(const TDouble2Vec4Vec& weights);
    //! Set the values' weights.
    CModelProbabilityParams& weights(const TDouble2Vec4Vec1Vec& weights);
    //! Get the values' weights.
    const TDouble2Vec4Vec1Vec& weights() const;
    //! Get writable values' weights.
    TDouble2Vec4Vec1Vec& weights();

    //! Add a coordinate for which to compute probability.
    CModelProbabilityParams& addCoordinate(std::size_t coordinate);
    //! Get the coordinates for which to compute probability.
    const TSize2Vec& coordinates() const;

    //! Set the most anomalous correlate.
    CModelProbabilityParams& mostAnomalousCorrelate(std::size_t correlate);
    //! Get the most anomalous correlate if there is one.
    TOptionalSize mostAnomalousCorrelate() const;

    //! Set whether or not to update the anomaly model.
    CModelProbabilityParams& updateAnomalyModel(bool update);
    //! Get whether or not to update the anomaly model.
    bool updateAnomalyModel() const;

private:
    //! The entity tag (if relevant otherwise 0).
    std::size_t m_Tag;
    //! The coordinates' probability calculations.
    TProbabilityCalculation2Vec m_Calculations;
    //! The confidence interval to use when detrending.
    double m_SeasonalConfidenceInterval;
    //! True if the bucket is empty and false otherwise.
    TBool2Vec1Vec m_BucketEmpty;
    //! Controls the interpretation of the weights.
    const maths_t::TWeightStyleVec* m_WeightStyles;
    //! The sample weights.
    TDouble2Vec4Vec1Vec m_Weights;
    //! The coordinates for which to compute the probability.
    TSize2Vec m_Coordinates;
    //! The most anomalous coordinate (if there is one).
    TOptionalSize m_MostAnomalousCorrelate;
    //! Whether or not to update the anomaly model.
    bool m_UpdateAnomalyModel;
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
    using TDouble2Vec4Vec = core::CSmallVector<TDouble2Vec, 4>;
    using TDouble2Vec4Vec1Vec = core::CSmallVector<TDouble2Vec4Vec, 1>;
    using TSize1Vec = core::CSmallVector<std::size_t, 1>;
    using TSize2Vec = core::CSmallVector<std::size_t, 2>;
    using TSize2Vec1Vec = core::CSmallVector<TSize2Vec, 1>;
    using TTime2Vec = core::CSmallVector<core_t::TTime, 2>;
    using TTime2Vec1Vec = core::CSmallVector<TTime2Vec, 1>;
    using TSizeDoublePr = std::pair<std::size_t, double>;
    using TSizeDoublePr1Vec = core::CSmallVector<TSizeDoublePr, 1>;
    using TTimeDouble2VecSizeTr = core::CTriple<core_t::TTime, TDouble2Vec, std::size_t>;
    using TTimeDouble2VecSizeTrVec = std::vector<TTimeDouble2VecSizeTr>;
    using TTail2Vec = core::CSmallVector<maths_t::ETail, 2>;

    //! Possible statuses for updating a model.
    enum EUpdateResult {
        E_Failure, //!< Update failed.
        E_Success, //!< Update succeeded.
        E_Reset    //!< Model reset.
    };

    //! Combine the results \p lhs and \p rhs.
    static EUpdateResult combine(EUpdateResult lhs, EUpdateResult rhs);

public:
    CModel(const CModelParams& params);
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
    virtual EUpdateResult addSamples(const CModelAddSamplesParams& params, TTimeDouble2VecSizeTrVec samples) = 0;

    //! Advance time by \p gap.
    virtual void skipTime(core_t::TTime gap) = 0;

    //! Get the most likely value for the time series at \p time.
    virtual TDouble2Vec mode(core_t::TTime time, const maths_t::TWeightStyleVec& weightStyles, const TDouble2Vec4Vec& weights) const = 0;

    //! Get the most likely value for each correlate time series at
    //! \p time, if there are any.
    virtual TDouble2Vec1Vec
    correlateModes(core_t::TTime time, const maths_t::TWeightStyleVec& weightStyles, const TDouble2Vec4Vec1Vec& weights) const = 0;

    //! Get the local maxima of the residual distribution.
    virtual TDouble2Vec1Vec residualModes(const maths_t::TWeightStyleVec& weightStyles, const TDouble2Vec4Vec& weights) const = 0;

    //! Remove any trend components from \p value.
    virtual void detrend(const TTime2Vec1Vec& time, double confidenceInterval, TDouble2Vec1Vec& value) const = 0;

    //! Get the best (least MSE) predicted value at \p time.
    virtual TDouble2Vec
    predict(core_t::TTime time, const TSizeDoublePr1Vec& correlated = TSizeDoublePr1Vec(), TDouble2Vec hint = TDouble2Vec()) const = 0;

    //! Get the prediction and \p confidenceInterval percentage
    //! confidence interval for the time series at \p time.
    virtual TDouble2Vec3Vec confidenceInterval(core_t::TTime time,
                                               double confidenceInterval,
                                               const maths_t::TWeightStyleVec& weightStyles,
                                               const TDouble2Vec4Vec& weights) const = 0;

    //! Forecast the time series and get its \p confidenceInterval
    //! percentage confidence interval between \p startTime and
    //! \p endTime.
    //! Data is pushed to the given \p forecastPushDataPointFunc
    //! \return true if forecast completed, false otherwise, in
    //! which case \p[out] messageOut is set.
    virtual bool forecast(core_t::TTime startTime,
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
                             double& probability,
                             TTail2Vec& tail,
                             bool& conditional,
                             TSize1Vec& mostAnomalousCorrelate) const = 0;

    //! Get the Winsorisation weight to apply to \p value,
    //! if appropriate.
    virtual TDouble2Vec winsorisationWeight(double derate, core_t::TTime time, const TDouble2Vec& value) const = 0;

    //! Get the seasonal variance scale at \p time.
    virtual TDouble2Vec seasonalWeight(double confidence, core_t::TTime time) const = 0;

    //! Compute a checksum for this object.
    virtual std::uint64_t checksum(std::uint64_t seed = 0) const = 0;

    //! Debug the memory used by this object.
    virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const = 0;

    //! Get the memory used by this object.
    virtual std::size_t memoryUsage() const = 0;

    //! Persist by passing information to \p inserter.
    virtual void acceptPersistInserter(core::CStatePersistInserter& inserter) const = 0;

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
    static boost::optional<VECTOR> predictionError(double propagationInterval, const PRIOR& prior, const VECTOR& sample);

    //! Correct \p probability with \p probabilityEmptyBucket.
    static double correctForEmptyBucket(maths_t::EProbabilityCalculation calculation,
                                        const TDouble2Vec& value,
                                        bool bucketEmpty,
                                        double probabilityBucketEmpty,
                                        double probability);

    //! Correct \p probability with \p probabilityEmptyBucket.
    static double correctForEmptyBucket(maths_t::EProbabilityCalculation calculation,
                                        double value,
                                        const TBool2Vec& bucketEmpty,
                                        const TDouble2Vec& probabilityEmptyBucket,
                                        double probability);

private:
    //! The model parameters.
    CModelParams m_Params;
};

//! A stateless lightweight model which stubs the interface.
class MATHS_EXPORT CModelStub : public CModel {
public:
    CModelStub();

    //! Returns 0.
    virtual std::size_t identifier() const;

    //! Create a copy of this model passing ownership to the caller.
    virtual CModelStub* clone(std::size_t id) const;

    //! Create a copy of the state we need to persist passing ownership to the caller.
    virtual CModelStub* cloneForPersistence() const;

    //! Create a copy of the state we need to run forecasting.
    virtual CModelStub* cloneForForecast() const;

    //! Return false;
    virtual bool isForecastPossible() const;

    //! No-op.
    virtual void modelCorrelations(CTimeSeriesCorrelations& model);

    //! Returns empty.
    virtual TSize2Vec1Vec correlates() const;

    //! No-op.
    virtual void addBucketValue(const TTimeDouble2VecSizeTrVec& value);

    //! No-op.
    virtual EUpdateResult addSamples(const CModelAddSamplesParams& params, TTimeDouble2VecSizeTrVec samples);

    //! No-op.
    virtual void skipTime(core_t::TTime gap);

    //! Returns empty.
    virtual TDouble2Vec mode(core_t::TTime time, const maths_t::TWeightStyleVec& weightStyles, const TDouble2Vec4Vec& weights) const;

    //! Returns empty.
    virtual TDouble2Vec1Vec
    correlateModes(core_t::TTime time, const maths_t::TWeightStyleVec& weightStyles, const TDouble2Vec4Vec1Vec& weights) const;

    //! Returns empty.
    virtual TDouble2Vec1Vec residualModes(const maths_t::TWeightStyleVec& weightStyles, const TDouble2Vec4Vec& weights) const;

    //! No-op.
    virtual void detrend(const TTime2Vec1Vec& time, double confidenceInterval, TDouble2Vec1Vec& value) const;

    //! Returns empty.
    virtual TDouble2Vec predict(core_t::TTime time, const TSizeDoublePr1Vec& correlated, TDouble2Vec hint = TDouble2Vec()) const;

    //! Returns empty.
    virtual TDouble2Vec3Vec confidenceInterval(core_t::TTime time,
                                               double confidenceInterval,
                                               const maths_t::TWeightStyleVec& weightStyles,
                                               const TDouble2Vec4Vec& weights) const;
    //! Returns empty.
    virtual bool forecast(core_t::TTime startTime,
                          core_t::TTime endTime,
                          double confidenceInterval,
                          const TDouble2Vec& minimum,
                          const TDouble2Vec& maximum,
                          const TForecastPushDatapointFunc& forecastPushDataPointFunc,
                          std::string& messageOut);

    //! Returns 1.0.
    virtual bool probability(const CModelProbabilityParams& params,
                             const TTime2Vec1Vec& time,
                             const TDouble2Vec1Vec& value,
                             double& probability,
                             TTail2Vec& tail,
                             bool& conditional,
                             TSize1Vec& mostAnomalousCorrelate) const;

    //! Returns empty.
    virtual TDouble2Vec winsorisationWeight(double derate, core_t::TTime time, const TDouble2Vec& value) const;

    //! Returns empty.
    virtual TDouble2Vec seasonalWeight(double confidence, core_t::TTime time) const;

    //! Returns the seed.
    virtual std::uint64_t checksum(std::uint64_t seed = 0) const;

    //! Debug the memory used by this object.
    virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

    //! Get the memory used by this object.
    virtual std::size_t memoryUsage() const;

    //! No-op.
    virtual void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Returns mixed data type since we don't know.
    virtual maths_t::EDataType dataType() const;
};
}
}

#endif // INCLUDED_ml_maths_CModel_h
