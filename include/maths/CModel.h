/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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
                 const double& learnRate,
                 const double& decayRate,
                 double minimumSeasonalVarianceScale);

    //! Get the bucket length.
    core_t::TTime bucketLength(void) const;

    //! Get the model learn rate.
    double learnRate(void) const;

    //! Get the model decay rate.
    double decayRate(void) const;

    //! Get the decay rate to use for time averaging the model decay.
    double averagingDecayRate(void) const;

    //! Get the minimum seasonal variance scale.
    double minimumSeasonalVarianceScale(void) const;

    //! Set the probability that the bucket will be empty for the model.
    void probabilityBucketEmpty(double probability);

    //! Get the probability that the bucket will be empty for the model.
    double probabilityBucketEmpty(void) const;

private:
    //! The data bucketing length.
    core_t::TTime m_BucketLength;
    //! The model learn rate.
    double m_LearnRate;
    //! The model decay rate.
    double m_DecayRate;
    //! The minimum seasonal variance scale.
    double m_MinimumSeasonalVarianceScale;
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
    CModelAddSamplesParams(void);
    CModelAddSamplesParams(const CModelAddSamplesParams&) = delete;
    const CModelAddSamplesParams& operator=(const CModelAddSamplesParams&) = delete;

    //! Set whether or not the data are integer valued.
    CModelAddSamplesParams& integer(bool integer);
    //! Get the data type.
    maths_t::EDataType type(void) const;

    //! Set whether or not the data are non-negative.
    CModelAddSamplesParams& nonNegative(bool nonNegative);
    //! Get the whether the data are non-negative.
    bool isNonNegative(void) const;

    //! Set the model propagation interval.
    CModelAddSamplesParams& propagationInterval(double interval);
    //! Get the model propagation interval.
    double propagationInterval(void) const;

    //! Set the weight styles.
    CModelAddSamplesParams& weightStyles(const maths_t::TWeightStyleVec& styles);
    //! Get the weight styles.
    const maths_t::TWeightStyleVec& weightStyles(void) const;

    //! Set the trend samples weights.
    CModelAddSamplesParams& trendWeights(const TDouble2Vec4VecVec& weights);
    //! Get the trend sample weights.
    const TDouble2Vec4VecVec& trendWeights(void) const;

    //! Set the prior samples weights.
    CModelAddSamplesParams& priorWeights(const TDouble2Vec4VecVec& weights);
    //! Get the prior sample weights.
    const TDouble2Vec4VecVec& priorWeights(void) const;

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
    CModelProbabilityParams(void);
    CModelProbabilityParams(const CModelAddSamplesParams&) = delete;
    const CModelProbabilityParams& operator=(const CModelAddSamplesParams&) = delete;

    //! Set the tag for the entity for which to compute the probability.
    CModelProbabilityParams& tag(std::size_t tag);
    //! Get the tag for the entity for which to compute the probability.
    std::size_t tag(void) const;

    //! Add a coordinate's calculation style.
    CModelProbabilityParams& addCalculation(maths_t::EProbabilityCalculation calculation);
    //! Get the number of calculations.
    std::size_t calculations(void) const;
    //! Get the \p i'th coordinate's calculation style.
    maths_t::EProbabilityCalculation calculation(std::size_t i) const;

    //! Set the confidence interval to use when detrending.
    CModelProbabilityParams& seasonalConfidenceInterval(double confidence);
    //! Get the confidence interval to use when detrending.
    double seasonalConfidenceInterval(void) const;

    //! Add whether a value's bucket is empty.
    CModelProbabilityParams& addBucketEmpty(const TBool2Vec& empty);
    //! Get whether the values' bucket is empty.
    const TBool2Vec1Vec& bucketEmpty(void) const;

    //! Set the weight styles.
    CModelProbabilityParams& weightStyles(const maths_t::TWeightStyleVec& styles);
    //! Get the weight styles.
    const maths_t::TWeightStyleVec& weightStyles(void) const;

    //! Add a value's weights.
    CModelProbabilityParams& addWeights(const TDouble2Vec4Vec& weights);
    //! Set the values' weights.
    CModelProbabilityParams& weights(const TDouble2Vec4Vec1Vec& weights);
    //! Get the values' weights.
    const TDouble2Vec4Vec1Vec& weights(void) const;
    //! Get writable values' weights.
    TDouble2Vec4Vec1Vec& weights(void);

    //! Add a coordinate for which to compute probability.
    CModelProbabilityParams& addCoordinate(std::size_t coordinate);
    //! Get the coordinates for which to compute probability.
    const TSize2Vec& coordinates(void) const;

    //! Set the most anomalous correlate.
    CModelProbabilityParams& mostAnomalousCorrelate(std::size_t correlate);
    //! Get the most anomalous correlate if there is one.
    TOptionalSize mostAnomalousCorrelate(void) const;

    //! Set whether or not to update the anomaly model.
    CModelProbabilityParams& updateAnomalyModel(bool update);
    //! Get whether or not to update the anomaly model.
    bool updateAnomalyModel(void) const;

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

public:
    CModel(const CModelParams& params);
    virtual ~CModel(void) = default;

    //! These don't need to be and shouldn't be copied.
    const CModel& operator=(const CModel&) = delete;

    //! Get the effective count per correlate model for calibrating aggregation.
    static double effectiveCount(std::size_t n);

    //! Get the model identifier.
    virtual std::size_t identifier(void) const = 0;

    //! Create a copy of this model passing ownership to the caller.
    virtual CModel* clone(std::size_t id) const = 0;

    //! Create a copy of the state we need to persist passing ownership to the caller.
    virtual CModel* cloneForPersistence(void) const = 0;

    //! Create a copy of the state we need to run forecasting.
    virtual CModel* cloneForForecast(void) const = 0;

    //! Return true if forecast is currently possible for this model.
    virtual bool isForecastPossible(void) const = 0;

    //! Tell this to model correlations.
    virtual void modelCorrelations(CTimeSeriesCorrelations& model) = 0;

    //! Get the correlated time series identifier pairs if any.
    virtual TSize2Vec1Vec correlates(void) const = 0;

    //! Update the model with the bucket \p value.
    virtual void addBucketValue(const TTimeDouble2VecSizeTrVec& value) = 0;

    //! Update the model with new samples.
    virtual EUpdateResult addSamples(const CModelAddSamplesParams& params,
                                     TTimeDouble2VecSizeTrVec samples) = 0;

    //! Advance time by \p gap.
    virtual void skipTime(core_t::TTime gap) = 0;

    //! Get the most likely value for the time series at \p time.
    virtual TDouble2Vec mode(core_t::TTime time,
                             const maths_t::TWeightStyleVec& weightStyles,
                             const TDouble2Vec4Vec& weights) const = 0;

    //! Get the most likely value for each correlate time series at
    //! \p time, if there are any.
    virtual TDouble2Vec1Vec correlateModes(core_t::TTime time,
                                           const maths_t::TWeightStyleVec& weightStyles,
                                           const TDouble2Vec4Vec1Vec& weights) const = 0;

    //! Get the local maxima of the residual distribution.
    virtual TDouble2Vec1Vec residualModes(const maths_t::TWeightStyleVec& weightStyles,
                                          const TDouble2Vec4Vec& weights) const = 0;

    //! Remove any trend components from \p value.
    virtual void
    detrend(const TTime2Vec1Vec& time, double confidenceInterval, TDouble2Vec1Vec& value) const = 0;

    //! Get the best (least MSE) predicted value at \p time.
    virtual TDouble2Vec predict(core_t::TTime time,
                                const TSizeDoublePr1Vec& correlated = TSizeDoublePr1Vec(),
                                TDouble2Vec hint = TDouble2Vec()) const = 0;

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
    virtual TDouble2Vec
    winsorisationWeight(double derate, core_t::TTime time, const TDouble2Vec& value) const = 0;

    //! Get the seasonal variance scale at \p time.
    virtual TDouble2Vec seasonalWeight(double confidence, core_t::TTime time) const = 0;

    //! Compute a checksum for this object.
    virtual std::uint64_t checksum(std::uint64_t seed = 0) const = 0;

    //! Debug the memory used by this object.
    virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const = 0;

    //! Get the memory used by this object.
    virtual std::size_t memoryUsage(void) const = 0;

    //! Persist by passing information to \p inserter.
    virtual void acceptPersistInserter(core::CStatePersistInserter& inserter) const = 0;

    //! Get the type of data being modeled.
    virtual maths_t::EDataType dataType(void) const = 0;

    //! Get read only model parameters.
    const CModelParams& params(void) const;

    //! Get writable model parameters.
    CModelParams& params(void);

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
    CModelStub(void);

    //! Returns 0.
    virtual std::size_t identifier(void) const;

    //! Create a copy of this model passing ownership to the caller.
    virtual CModelStub* clone(std::size_t id) const;

    //! Create a copy of the state we need to persist passing ownership to the caller.
    virtual CModelStub* cloneForPersistence(void) const;

    //! Create a copy of the state we need to run forecasting.
    virtual CModelStub* cloneForForecast(void) const;

    //! Return false;
    virtual bool isForecastPossible(void) const;

    //! No-op.
    virtual void modelCorrelations(CTimeSeriesCorrelations& model);

    //! Returns empty.
    virtual TSize2Vec1Vec correlates(void) const;

    //! No-op.
    virtual void addBucketValue(const TTimeDouble2VecSizeTrVec& value);

    //! No-op.
    virtual EUpdateResult addSamples(const CModelAddSamplesParams& params,
                                     TTimeDouble2VecSizeTrVec samples);

    //! No-op.
    virtual void skipTime(core_t::TTime gap);

    //! Returns empty.
    virtual TDouble2Vec mode(core_t::TTime time,
                             const maths_t::TWeightStyleVec& weightStyles,
                             const TDouble2Vec4Vec& weights) const;

    //! Returns empty.
    virtual TDouble2Vec1Vec correlateModes(core_t::TTime time,
                                           const maths_t::TWeightStyleVec& weightStyles,
                                           const TDouble2Vec4Vec1Vec& weights) const;

    //! Returns empty.
    virtual TDouble2Vec1Vec residualModes(const maths_t::TWeightStyleVec& weightStyles,
                                          const TDouble2Vec4Vec& weights) const;

    //! No-op.
    virtual void
    detrend(const TTime2Vec1Vec& time, double confidenceInterval, TDouble2Vec1Vec& value) const;

    //! Returns empty.
    virtual TDouble2Vec predict(core_t::TTime time,
                                const TSizeDoublePr1Vec& correlated,
                                TDouble2Vec hint = TDouble2Vec()) const;

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
    virtual TDouble2Vec
    winsorisationWeight(double derate, core_t::TTime time, const TDouble2Vec& value) const;

    //! Returns empty.
    virtual TDouble2Vec seasonalWeight(double confidence, core_t::TTime time) const;

    //! Returns the seed.
    virtual std::uint64_t checksum(std::uint64_t seed = 0) const;

    //! Debug the memory used by this object.
    virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

    //! Get the memory used by this object.
    virtual std::size_t memoryUsage(void) const;

    //! No-op.
    virtual void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Returns mixed data type since we don't know.
    virtual maths_t::EDataType dataType(void) const;
};
}
}

#endif // INCLUDED_ml_maths_CModel_h
