/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDE_ml_maths_CTimeSeriesModel_h
#define INCLUDE_ml_maths_CTimeSeriesModel_h

#include <maths/CKMostCorrelated.h>
#include <maths/CModel.h>
#include <maths/CMultivariatePrior.h>
#include <maths/ImportExport.h>

#include <boost/array.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/unordered_map.hpp>

#include <cstddef>
#include <memory>

namespace ml {
namespace maths {
class CDecayRateController;
class CPrior;
class CTimeSeriesDecompositionInterface;
class CTimeSeriesAnomalyModel;
class CUnivariateTimeSeriesChangeDetector;
struct SChangeDescription;
struct SDistributionRestoreParams;
struct SModelRestoreParams;

namespace winsorisation {
//! Computes a Winsorisation weight for \p value based on its
//! one tail p-value.
MATHS_EXPORT
double weight(const CPrior& prior, double derate, double scale, double value);

//! Computes a Winsorisation weight for \p value based on its
//! marginal for \p dimension one tail p-value.
MATHS_EXPORT
double weight(const CMultivariatePrior& prior,
              std::size_t dimension,
              double derate,
              double scale,
              const core::CSmallVector<double, 10>& value);
}

//! \brief A CModel implementation for modeling a univariate time series.
class MATHS_EXPORT CUnivariateTimeSeriesModel : public CModel {
public:
    using TTimeDoublePr = std::pair<core_t::TTime, double>;
    using TTimeDoublePrCBuf = boost::circular_buffer<TTimeDoublePr>;
    using TDoubleWeightsAry = maths_t::TDoubleWeightsAry;
    using TDecompositionPtr = std::shared_ptr<CTimeSeriesDecompositionInterface>;
    using TDecayRateController2Ary = boost::array<CDecayRateController, 2>;

public:
    //! \param[in] params The model parameters.
    //! \param[in] id The *unique* identifier for this time series.
    //! \param[in] trendModel The time series trend decomposition.
    //! \param[in] residualModel The prior for the time series residual model.
    //! \param[in] controllers Optional decay rate controllers for the trend
    //! and residual model.
    //! \param[in] modelAnomalies If true we use a separate model to capture
    //! the characteristics of anomalous time periods.
    CUnivariateTimeSeriesModel(const CModelParams& params,
                               std::size_t id,
                               const CTimeSeriesDecompositionInterface& trendModel,
                               const CPrior& residualModel,
                               const TDecayRateController2Ary* controllers = nullptr,
                               bool modelAnomalies = true);
    CUnivariateTimeSeriesModel(const SModelRestoreParams& params,
                               core::CStateRestoreTraverser& traverser);
    ~CUnivariateTimeSeriesModel();

    const CUnivariateTimeSeriesModel& operator=(const CUnivariateTimeSeriesModel&) = delete;

    //! Get the model identifier.
    virtual std::size_t identifier() const;

    //! Create a copy of this model passing ownership to the caller.
    virtual CUnivariateTimeSeriesModel* clone(std::size_t id) const;

    //! Create a copy of the state we need to persist passing ownership
    //! to the caller.
    virtual CUnivariateTimeSeriesModel* cloneForPersistence() const;

    //! Create a copy of the state we need to run forecasting.
    virtual CUnivariateTimeSeriesModel* cloneForForecast() const;

    //! Return true if forecast is currently possible for this model.
    virtual bool isForecastPossible() const;

    //! Tell this to model correlations.
    virtual void modelCorrelations(CTimeSeriesCorrelations& model);

    //! Get the correlated time series identifier pairs if any.
    virtual TSize2Vec1Vec correlates() const;

    //! Update the model with the bucket \p value.
    virtual void addBucketValue(const TTimeDouble2VecSizeTrVec& value);

    //! Update the model with new samples.
    virtual EUpdateResult addSamples(const CModelAddSamplesParams& params,
                                     TTimeDouble2VecSizeTrVec samples);

    //! Advance time by \p gap.
    virtual void skipTime(core_t::TTime gap);

    //! Get the most likely value for the time series at \p time.
    virtual TDouble2Vec mode(core_t::TTime time, const TDouble2VecWeightsAry& weights) const;

    //! Get the most likely value for each correlate time series
    //! at \p time, if there are any.
    virtual TDouble2Vec1Vec
    correlateModes(core_t::TTime time, const TDouble2VecWeightsAry1Vec& weights) const;

    //! Get the local maxima of the residual distribution.
    virtual TDouble2Vec1Vec residualModes(const TDouble2VecWeightsAry& weights) const;

    //! Remove any trend components from \p value.
    virtual void detrend(const TTime2Vec1Vec& time,
                         double confidenceInterval,
                         TDouble2Vec1Vec& value) const;

    //! Get the best (least MSE) predicted value at \p time.
    virtual TDouble2Vec predict(core_t::TTime time,
                                const TSizeDoublePr1Vec& correlated = TSizeDoublePr1Vec(),
                                TDouble2Vec hint = TDouble2Vec()) const;

    //! Get the prediction and \p confidenceInterval percentage
    //! confidence interval for the time series at \p time.
    virtual TDouble2Vec3Vec confidenceInterval(core_t::TTime time,
                                               double confidenceInterval,
                                               const TDouble2VecWeightsAry& weights) const;

    //! Forecast the time series and get its \p confidenceInterval
    //! percentage confidence interval between \p startTime and
    //! \p endTime.
    virtual bool forecast(core_t::TTime startTime,
                          core_t::TTime endTime,
                          double confidenceInterval,
                          const TDouble2Vec& minimum,
                          const TDouble2Vec& maximum,
                          const TForecastPushDatapointFunc& forecastPushDataPointFunc,
                          std::string& messageOut);

    //! Compute the probability of drawing \p value at \p time.
    virtual bool probability(const CModelProbabilityParams& params,
                             const TTime2Vec1Vec& time,
                             const TDouble2Vec1Vec& value,
                             double& probability,
                             TTail2Vec& tail,
                             bool& conditional,
                             TSize1Vec& mostAnomalousCorrelate) const;

    //! Get the Winsorisation weight to apply to \p value.
    virtual TDouble2Vec
    winsorisationWeight(double derate, core_t::TTime time, const TDouble2Vec& value) const;

    //! Get the seasonal variance scale at \p time.
    virtual TDouble2Vec seasonalWeight(double confidence, core_t::TTime time) const;

    //! Compute a checksum for this object.
    virtual uint64_t checksum(uint64_t seed = 0) const;

    //! Debug the memory used by this object.
    virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

    //! Get the memory used by this object.
    virtual std::size_t memoryUsage() const;

    //! Initialize reading state from \p traverser.
    bool acceptRestoreTraverser(const SModelRestoreParams& params,
                                core::CStateRestoreTraverser& traverser);

    //! Persist by passing information to \p inserter.
    virtual void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Get the type of data being modeled.
    virtual maths_t::EDataType dataType() const;

    //! \name Helpers
    //@{
    //! Unpack the weights in \p weights.
    static TDoubleWeightsAry unpack(const TDouble2VecWeightsAry& weights);

    //! Reinitialize \p residualModel using the detrended values
    //! from \p slidingWindow.
    static void reinitializeResidualModel(double learnRate,
                                          const TDecompositionPtr& trend,
                                          const TTimeDoublePrCBuf& slidingWindow,
                                          CPrior& residualModel);
    //@}

    //! \name Test Functions
    //@{
    //! Get the sliding window of recent values.
    const TTimeDoublePrCBuf& slidingWindow() const;

    //! Get the trend.
    const CTimeSeriesDecompositionInterface& trendModel() const;

    //! Get the residual model.
    const CPrior& residualModel(void) const;
    //@}

private:
    using TSizeVec = std::vector<std::size_t>;
    using TDouble1Vec = core::CSmallVector<double, 1>;
    using TDouble1VecVec = std::vector<TDouble1Vec>;
    using TDouble2VecWeightsAryVec = std::vector<TDouble2VecWeightsAry>;
    using TVector = CVectorNx1<double, 2>;
    using TVectorMeanAccumulator = CBasicStatistics::SSampleMean<TVector>::TAccumulator;
    using TDecayRateController2AryPtr = std::unique_ptr<TDecayRateController2Ary>;
    using TPriorPtr = std::shared_ptr<CPrior>;
    using TAnomalyModelPtr = std::unique_ptr<CTimeSeriesAnomalyModel>;
    using TMultivariatePriorCPtrSizePr = std::pair<const CMultivariatePrior*, std::size_t>;
    using TMultivariatePriorCPtrSizePr1Vec =
        core::CSmallVector<TMultivariatePriorCPtrSizePr, 1>;
    using TModelCPtr1Vec = core::CSmallVector<const CUnivariateTimeSeriesModel*, 1>;
    using TChangeDetectorPtr = std::unique_ptr<CUnivariateTimeSeriesChangeDetector>;

private:
    CUnivariateTimeSeriesModel(const CUnivariateTimeSeriesModel& other,
                               std::size_t id,
                               bool isForForecast = false);

    //! Test for and apply any change we find.
    EUpdateResult testAndApplyChange(const CModelAddSamplesParams& params,
                                     const TSizeVec& order,
                                     const TTimeDouble2VecSizeTrVec& samples);

    //! Apply \p change to this model.
    EUpdateResult applyChange(const SChangeDescription& change);

    //! Update the trend with \p samples.
    EUpdateResult updateTrend(const TTimeDouble2VecSizeTrVec& samples,
                              const TDouble2VecWeightsAryVec& trendWeights);

    //! Compute the prediction errors for \p sample.
    void appendPredictionErrors(double interval, double sample, TDouble1VecVec (&result)[2]);

    //! Reinitialize state after detecting a new component of the trend
    //! decomposition.
    void reinitializeStateGivenNewComponent(void);

    //! Get the models for the correlations and the models of the correlated
    //! time series.
    bool correlationModels(TSize1Vec& correlated,
                           TSize2Vec1Vec& variables,
                           TMultivariatePriorCPtrSizePr1Vec& correlationDistributionModels,
                           TModelCPtr1Vec& correlatedTimeSeriesModels) const;

private:
    //! A unique identifier for this model.
    std::size_t m_Id;

    //! True if the data are non-negative.
    bool m_IsNonNegative;

    //! True if the model can be forecast.
    bool m_IsForecastable;

    //! A random number generator for sampling the sliding window.
    CPRNG::CXorOShiro128Plus m_Rng;

    //! These control the trend and residual model decay rates (see
    //! CDecayRateController for more details).
    TDecayRateController2AryPtr m_Controllers;

    //! The time series trend decomposition.
    //!
    //! \note This can be temporarily be shared with the change detector.
    TDecompositionPtr m_TrendModel;

    //! The time series' residual model.
    //!
    //! \note This can be temporarily be shared with the change detector.
    TPriorPtr m_ResidualModel;

    //! A model for time periods when the basic model can't predict the
    //! value of the time series.
    TAnomalyModelPtr m_AnomalyModel;

    //! The last "normal" time and median value.
    TTimeDoublePr m_CandidateChangePoint;

    //! If the time series appears to be undergoing change, the contiguous
    //! interval of unpredictable values.
    core_t::TTime m_CurrentChangeInterval;

    //! Used to test for changes in the time series.
    TChangeDetectorPtr m_ChangeDetector;

    //! A sliding window of the recent samples (used to reinitialize the
    //! residual model when a new trend component is detected).
    TTimeDoublePrCBuf m_SlidingWindow;

    //! Models the correlations between time series.
    CTimeSeriesCorrelations* m_Correlations;
};

//! \brief Manages the creation correlate models.
class MATHS_EXPORT CTimeSeriesCorrelateModelAllocator {
public:
    using TMultivariatePriorPtr = std::unique_ptr<CMultivariatePrior>;

public:
    virtual ~CTimeSeriesCorrelateModelAllocator() = default;

    //! Check if we can still allocate any correlations.
    virtual bool areAllocationsAllowed() const = 0;

    //! Check if \p correlations exceeds the memory limit.
    virtual bool exceedsLimit(std::size_t correlations) const = 0;

    //! Get the maximum number of correlations we should model.
    virtual std::size_t maxNumberCorrelations() const = 0;

    //! Get the chunk size in which to allocate correlations.
    virtual std::size_t chunkSize() const = 0;

    //! Create a new prior for a correlation model.
    virtual TMultivariatePriorPtr newPrior() const = 0;
};

//! \brief A model of the top k correlates.
//!
//! DESCRIPTION:\n
//! This estimates the (Pearson) correlations between a collection of univariate
//! time series and manages life-cycle of the models for the k most correlated
//! pairs. Note that the allocator (supplied to refresh) defines how many correlates
//! can be modeled.
//!
//! IMPLEMENTATION:\n
//! The individual time series models hold a reference to this and update it with
//! their samples, add and remove themselves as part of their life-cycle management
//! and use it to correct their predictions and probability calculation as appropriate.
//! The user of this class simply needs to pass it to CUnivariateTimeSeriesModel on
//! construction and manage the calls to update it after a batch of samples has been
//! added and to refresh it before a batch of samples is added to the individual models.
class MATHS_EXPORT CTimeSeriesCorrelations {
public:
    using TTime1Vec = core::CSmallVector<core_t::TTime, 1>;
    using TDouble1Vec = core::CSmallVector<double, 1>;
    using TDouble2Vec = core::CSmallVector<double, 2>;
    using TDoubleWeightsAry1Vec = maths_t::TDoubleWeightsAry1Vec;
    using TSize1Vec = core::CSmallVector<std::size_t, 1>;
    using TSizeSize1VecUMap = boost::unordered_map<std::size_t, TSize1Vec>;
    using TSize2Vec = core::CSmallVector<std::size_t, 2>;
    using TSize2Vec1Vec = core::CSmallVector<TSize2Vec, 1>;
    using TSizeSizePr = std::pair<std::size_t, std::size_t>;
    using TMultivariatePriorPtr = std::unique_ptr<CMultivariatePrior>;
    using TMultivariatePriorPtrDoublePr = std::pair<TMultivariatePriorPtr, double>;
    using TSizeSizePrMultivariatePriorPtrDoublePrUMap =
        boost::unordered_map<TSizeSizePr, TMultivariatePriorPtrDoublePr>;
    using TMultivariatePriorCPtrSizePr = std::pair<const CMultivariatePrior*, std::size_t>;
    using TMultivariatePriorCPtrSizePr1Vec =
        core::CSmallVector<TMultivariatePriorCPtrSizePr, 1>;

    //! \brief Wraps up the sampled data for a feature.
    struct MATHS_EXPORT SSampleData {
        //! The data type.
        maths_t::EDataType s_Type;
        //! The times of the samples.
        TTime1Vec s_Times;
        //! The detrended samples.
        TDouble1Vec s_Samples;
        //! The tags for each sample.
        TSize1Vec s_Tags;
        //! The sample weights.
        TDoubleWeightsAry1Vec s_Weights;
        //! The interval by which to age the correlation model.
        double s_Interval;
        //! The decay rate multiplier.
        double s_Multiplier;
    };

    using TSizeSampleDataUMap = boost::unordered_map<std::size_t, SSampleData>;

public:
    CTimeSeriesCorrelations(double minimumSignificantCorrelation, double decayRate);
    const CTimeSeriesCorrelations& operator=(const CTimeSeriesCorrelations&) = delete;

    //! Create a copy of this model passing ownership to the caller.
    CTimeSeriesCorrelations* clone() const;

    //! Create a copy of the state we need to persist passing ownership
    //! to the caller.
    CTimeSeriesCorrelations* cloneForPersistence() const;

    //! Process all samples added from individual time series models.
    //!
    //! \note This should be called exactly once after every univariate
    //! time series model has added its samples.
    void processSamples();

    //! Refresh the models to account for any changes to the correlation
    //! estimates.
    //!
    //! \note This should be called exactly once before every univariate
    //! time series model adds its samples.
    void refresh(const CTimeSeriesCorrelateModelAllocator& allocator);

    //! Get the correlation joint distribution models.
    const TSizeSizePrMultivariatePriorPtrDoublePrUMap& correlationModels() const;

    //! Debug the memory used by this object.
    void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

    //! Get the memory used by this object.
    std::size_t memoryUsage() const;

    //! Initialize reading state from \p traverser.
    bool acceptRestoreTraverser(const SDistributionRestoreParams& params,
                                core::CStateRestoreTraverser& traverser);

    //! Persist by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

private:
    using TTimeDouble2VecSizeTr = core::CTriple<core_t::TTime, TDouble2Vec, std::size_t>;
    using TTimeDouble2VecSizeTrVec = std::vector<TTimeDouble2VecSizeTr>;
    using TModelCPtrVec = std::vector<const CUnivariateTimeSeriesModel*>;
    using TModelCPtr1Vec = core::CSmallVector<const CUnivariateTimeSeriesModel*, 1>;
    using TSizeSizePrMultivariatePriorPtrDoublePrPr =
        std::pair<TSizeSizePr, TMultivariatePriorPtrDoublePr>;
    using TConstSizeSizePrMultivariatePriorPtrDoublePrPr =
        std::pair<const TSizeSizePr, TMultivariatePriorPtrDoublePr>;

private:
    CTimeSeriesCorrelations(const CTimeSeriesCorrelations& other,
                            bool isForPersistence = false);

    //! Restore the correlation distribution models reading state from
    //! \p traverser.
    bool restoreCorrelationModels(const SDistributionRestoreParams& params,
                                  core::CStateRestoreTraverser& traverser);

    //! Persist the correlation distribution models passing information
    //! to \p inserter.
    void persistCorrelationModels(core::CStatePersistInserter& inserter) const;

    //! Restore the \p model reading state from \p traverser.
    static bool restore(const SDistributionRestoreParams& params,
                        TSizeSizePrMultivariatePriorPtrDoublePrPr& model,
                        core::CStateRestoreTraverser& traverser);

    //! Persist the \p model passing information to \p inserter.
    static void persist(const TConstSizeSizePrMultivariatePriorPtrDoublePrPr& model,
                        core::CStatePersistInserter& inserter);

    //! Add the time series identified by \p id.
    void addTimeSeries(std::size_t id, const CUnivariateTimeSeriesModel& model);

    //! Remove the correlates of \p id.
    void removeTimeSeries(std::size_t id);

    //! Add a sample for the time series identified by \p id.
    void addSamples(std::size_t id,
                    const CModelAddSamplesParams& params,
                    const TTimeDouble2VecSizeTrVec& samples,
                    double multiplier);

    //! Get the ids of the time series correlated with \p id.
    TSize1Vec correlated(std::size_t id) const;

    //! Get the correlation models and the correlated time series models
    //! for for \p id.
    bool correlationModels(std::size_t id,
                           TSize1Vec& correlated,
                           TSize2Vec1Vec& variables,
                           TMultivariatePriorCPtrSizePr1Vec& correlationDistributionModels,
                           TModelCPtr1Vec& correlatedTimeSeriesModels) const;

    //! Refresh the mapping from time series identifier to correlate
    //! identifiers.
    void refreshLookup();

private:
    //! The minimum significant Pearson correlation.
    double m_MinimumSignificantCorrelation;

    //! Filled in with the sample data if we are modeling correlates.
    TSizeSampleDataUMap m_SampleData;

    //! Estimates the Pearson correlations of the k-most correlated
    //! time series.
    CKMostCorrelated m_Correlations;

    //! A lookup by time series identifier for correlated time series.
    TSizeSize1VecUMap m_CorrelatedLookup;

    //! Models of the joint distribution (of the residuals) of the pairs
    //! of time series which have significant correlation.
    TSizeSizePrMultivariatePriorPtrDoublePrUMap m_CorrelationDistributionModels;

    //! A collection of univariate time series models for which this is
    //! modeling correlations (indexed by their identifier).
    TModelCPtrVec m_TimeSeriesModels;

    friend class CUnivariateTimeSeriesModel;
};

//! \brief A CModel implementation for modeling a multivariate time series.
class MATHS_EXPORT CMultivariateTimeSeriesModel : public CModel {
public:
    using TDouble10Vec = core::CSmallVector<double, 10>;
    using TTimeDouble2VecPr = std::pair<core_t::TTime, TDouble2Vec>;
    using TTimeDouble2VecPrCBuf = boost::circular_buffer<TTimeDouble2VecPr>;
    using TDouble10VecWeightsAry = maths_t::TDouble10VecWeightsAry;
    using TDecompositionPtr = std::shared_ptr<CTimeSeriesDecompositionInterface>;
    using TDecompositionPtr10Vec = core::CSmallVector<TDecompositionPtr, 10>;
    using TDecayRateController2Ary = boost::array<CDecayRateController, 2>;

public:
    //! \param[in] params The model parameters.
    //! \param[in] trendModel The time series trend decomposition.
    //! \param[in] residualModel The prior for the time series residual model.
    //! \param[in] controllers Optional decay rate controllers for the trend
    //! and residual model.
    //! \param[in] modelAnomalies If true we use a separate model to capture
    //! the characteristics of anomalous time periods.
    CMultivariateTimeSeriesModel(const CModelParams& params,
                                 const CTimeSeriesDecompositionInterface& trendModel,
                                 const CMultivariatePrior& residualModel,
                                 const TDecayRateController2Ary* controllers = nullptr,
                                 bool modelAnomalies = true);
    CMultivariateTimeSeriesModel(const CMultivariateTimeSeriesModel& other);
    CMultivariateTimeSeriesModel(const SModelRestoreParams& params,
                                 core::CStateRestoreTraverser& traverser);
    ~CMultivariateTimeSeriesModel();

    const CMultivariateTimeSeriesModel& operator=(const CMultivariateTimeSeriesModel&) = delete;

    //! Returns 0 since these models don't need a unique identifier.
    virtual std::size_t identifier() const;

    //! Create a copy of this model passing ownership to the caller.
    virtual CMultivariateTimeSeriesModel* clone(std::size_t id) const;

    //! Create a copy of the state we need to persist passing ownership
    //! to the caller.
    virtual CMultivariateTimeSeriesModel* cloneForPersistence() const;

    //! Create a copy of the state we need to run forecasting.
    virtual CMultivariateTimeSeriesModel* cloneForForecast() const;

    //! Returns false (not currently supported for multivariate features).
    virtual bool isForecastPossible() const;

    //! No-op.
    virtual void modelCorrelations(CTimeSeriesCorrelations& model);

    //! Returns empty.
    virtual TSize2Vec1Vec correlates() const;

    //! Update the model with the bucket \p value.
    virtual void addBucketValue(const TTimeDouble2VecSizeTrVec& value);

    //! Update the model with new samples.
    virtual EUpdateResult addSamples(const CModelAddSamplesParams& params,
                                     TTimeDouble2VecSizeTrVec samples);

    //! Advance time by \p gap.
    virtual void skipTime(core_t::TTime gap);

    //! Get the most likely value for the time series at \p time.
    virtual TDouble2Vec mode(core_t::TTime time, const TDouble2VecWeightsAry& weights) const;

    //! Returns empty.
    virtual TDouble2Vec1Vec
    correlateModes(core_t::TTime time, const TDouble2VecWeightsAry1Vec& weights) const;

    //! Get the local maxima of the residual distribution.
    virtual TDouble2Vec1Vec residualModes(const TDouble2VecWeightsAry& weights) const;

    //! Remove any trend components from \p value.
    virtual void detrend(const TTime2Vec1Vec& time,
                         double confidenceInterval,
                         TDouble2Vec1Vec& value) const;

    //! Get the best (least MSE) predicted value at \p time.
    virtual TDouble2Vec predict(core_t::TTime time,
                                const TSizeDoublePr1Vec& correlated = TSizeDoublePr1Vec(),
                                TDouble2Vec hint = TDouble2Vec()) const;

    //! Get the prediction and \p confidenceInterval percentage
    //! confidence interval for the time series at \p time.
    virtual TDouble2Vec3Vec confidenceInterval(core_t::TTime time,
                                               double confidenceInterval,
                                               const TDouble2VecWeightsAry& weights) const;

    //! Not currently supported.
    virtual bool forecast(core_t::TTime startTime,
                          core_t::TTime endTime,
                          double confidenceInterval,
                          const TDouble2Vec& minimum,
                          const TDouble2Vec& maximum,
                          const TForecastPushDatapointFunc& forecastPushDataPointFunc,
                          std::string& messageOut);

    //! Compute the probability of drawing \p value at \p time.
    virtual bool probability(const CModelProbabilityParams& params,
                             const TTime2Vec1Vec& time,
                             const TDouble2Vec1Vec& value,
                             double& probability,
                             TTail2Vec& tail,
                             bool& conditional,
                             TSize1Vec& mostAnomalousCorrelate) const;

    //! Get the Winsorisation weight to apply to \p value.
    virtual TDouble2Vec
    winsorisationWeight(double derate, core_t::TTime time, const TDouble2Vec& value) const;

    //! Get the seasonal variance scale at \p time.
    virtual TDouble2Vec seasonalWeight(double confidence, core_t::TTime time) const;

    //! Compute a checksum for this object.
    virtual uint64_t checksum(uint64_t seed = 0) const;

    //! Debug the memory used by this object.
    virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

    //! Get the memory used by this object.
    virtual std::size_t memoryUsage() const;

    //! Initialize reading state from \p traverser.
    bool acceptRestoreTraverser(const SModelRestoreParams& params,
                                core::CStateRestoreTraverser& traverser);

    //! Persist by passing information to \p inserter.
    virtual void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Get the type of data being modeled.
    virtual maths_t::EDataType dataType() const;

    //! \name Helpers
    //@{
    //! Unpack the weights in \p weights.
    static TDouble10VecWeightsAry unpack(const TDouble2VecWeightsAry& weights);

    //! Reinitialize \p residualModel using the detrended values
    //! from \p slidingWindow.
    static void reinitializeResidualModel(double learnRate,
                                          const TDecompositionPtr10Vec& trend,
                                          const TTimeDouble2VecPrCBuf& slidingWindow,
                                          CMultivariatePrior& residualModel);
    //@}

    //! \name Test Functions
    //@{
    //! Get the sliding window of recent values.
    const TTimeDouble2VecPrCBuf& slidingWindow() const;

    //! Get the trend.
    const TDecompositionPtr10Vec& trendModel() const;

    //! Get the residual model.
    const CMultivariatePrior& residualModel() const;
    //@}

private:
    using TDouble1Vec = core::CSmallVector<double, 1>;
    using TDouble1VecVec = std::vector<TDouble1Vec>;
    using TDouble2VecWeightsAryVec = std::vector<TDouble2VecWeightsAry>;
    using TVector = CVectorNx1<double, 2>;
    using TVectorMeanAccumulator = CBasicStatistics::SSampleMean<TVector>::TAccumulator;
    using TDecayRateController2AryPtr = std::unique_ptr<TDecayRateController2Ary>;
    using TMultivariatePriorPtr = std::unique_ptr<CMultivariatePrior>;
    using TAnomalyModelPtr = std::unique_ptr<CTimeSeriesAnomalyModel>;

private:
    //! Update the trend with \p samples.
    EUpdateResult updateTrend(const TTimeDouble2VecSizeTrVec& samples,
                              const TDouble2VecWeightsAryVec& trendWeights);

    //! Compute the prediction errors for \p sample.
    void appendPredictionErrors(double interval,
                                const TDouble2Vec& sample,
                                TDouble1VecVec (&result)[2]);

    //! Reinitialize state after detecting a new component of the trend
    //! decomposition.
    void reinitializeStateGivenNewComponent(void);

    //! Get the model dimension.
    std::size_t dimension() const;

private:
    //! True if the data are non-negative.
    bool m_IsNonNegative;

    //! A random number generator for sampling the sliding window.
    CPRNG::CXorOShiro128Plus m_Rng;

    //! These control the trend and residual model decay rates (see
    //! CDecayRateController for more details).
    TDecayRateController2AryPtr m_Controllers;

    //! The time series trend decomposition.
    TDecompositionPtr10Vec m_TrendModel;

    //! The time series residual model.
    TMultivariatePriorPtr m_ResidualModel;

    //! A model for time periods when the basic model can't predict the
    //! value of the time series.
    TAnomalyModelPtr m_AnomalyModel;

    //! A sliding window of the recent samples (used to reinitialize the
    //! residual model when a new trend component is detected).
    TTimeDouble2VecPrCBuf m_SlidingWindow;
};
}
}

#endif // INCLUDE_ml_maths_CTimeSeriesModel_h
