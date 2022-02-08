/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#ifndef INCLUDE_ml_maths_time_series_CTimeSeriesModel_h
#define INCLUDE_ml_maths_time_series_CTimeSeriesModel_h

#include <maths/common/CKMostCorrelated.h>
#include <maths/common/CModel.h>
#include <maths/common/CMultivariatePrior.h>

#include <maths/time_series/CTimeSeriesMultibucketFeaturesFwd.h>
#include <maths/time_series/ImportExport.h>

#include <boost/array.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/unordered_map.hpp>

#include <cstddef>
#include <memory>

namespace ml {
namespace maths {
namespace common {
class CPrior;
struct SModelRestoreParams;
struct SDistributionRestoreParams;
}
namespace time_series {
class CDecayRateController;
class CTimeSeriesDecompositionInterface;
class CTimeSeriesAnomalyModel;

//! \brief A CModel implementation for modeling a univariate time series.
class MATHS_TIME_SERIES_EXPORT CUnivariateTimeSeriesModel : public common::CModel {
public:
    using TFloatMeanAccumulator =
        common::CBasicStatistics::SSampleMean<common::CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
    using TDoubleWeightsAry = maths_t::TDoubleWeightsAry;
    using TDecompositionPtr = std::shared_ptr<CTimeSeriesDecompositionInterface>;
    using TDecayRateController2Ary = std::array<CDecayRateController, 2>;
    using TMultibucketFeature = CTimeSeriesMultibucketScalarFeature;

public:
    //! \param[in] params The model parameters.
    //! \param[in] id The *unique* identifier for this time series.
    //! \param[in] trendModel The time series trend decomposition.
    //! \param[in] residualModel The prior for the time series residual model.
    //! \param[in] controllers Optional decay rate controllers for the trend
    //! and residual model.
    //! \param[in] multibucketFeature The multi-bucket feature to analyse if any.
    //! \param[in] modelAnomalies If true we use a separate model to capture
    //! the characteristics of anomalous time periods.
    CUnivariateTimeSeriesModel(const common::CModelParams& params,
                               std::size_t id,
                               const CTimeSeriesDecompositionInterface& trendModel,
                               const common::CPrior& residualModel,
                               const TDecayRateController2Ary* controllers = nullptr,
                               const TMultibucketFeature* multibucketFeature = nullptr,
                               bool modelAnomalies = true);
    CUnivariateTimeSeriesModel(const common::SModelRestoreParams& params,
                               core::CStateRestoreTraverser& traverser);
    ~CUnivariateTimeSeriesModel() override;

    const CUnivariateTimeSeriesModel& operator=(const CUnivariateTimeSeriesModel&) = delete;

    //! Get the model identifier.
    std::size_t identifier() const override;

    //! Create a copy of this model passing ownership to the caller.
    CUnivariateTimeSeriesModel* clone(std::size_t id) const override;

    //! Create a copy of the state we need to persist passing ownership
    //! to the caller.
    CUnivariateTimeSeriesModel* cloneForPersistence() const override;

    //! Create a copy of the state we need to run forecasting.
    CUnivariateTimeSeriesModel* cloneForForecast() const override;

    //! Return true if forecast is currently possible for this model.
    bool isForecastPossible() const override;

    //! Tell this to model correlations.
    void modelCorrelations(CTimeSeriesCorrelations& model) override;

    //! Get the correlated time series identifier pairs if any.
    TSize2Vec1Vec correlates() const override;

    //! Update the model with the bucket \p value.
    void addBucketValue(const TTimeDouble2VecSizeTrVec& value) override;

    //! Update the model with new samples.
    EUpdateResult addSamples(const common::CModelAddSamplesParams& params,
                             TTimeDouble2VecSizeTrVec samples) override;

    //! Advance time by \p gap.
    void skipTime(core_t::TTime gap) override;

    //! Get the most likely value for the time series at \p time.
    TDouble2Vec mode(core_t::TTime time, const TDouble2VecWeightsAry& weights) const override;

    //! Get the most likely value for each correlate time series
    //! at \p time, if there are any.
    TDouble2Vec1Vec correlateModes(core_t::TTime time,
                                   const TDouble2VecWeightsAry1Vec& weights) const override;

    //! Get the local maxima of the residual distribution.
    TDouble2Vec1Vec residualModes(const TDouble2VecWeightsAry& weights) const override;

    //! Remove any trend components from \p value.
    void detrend(const TTime2Vec1Vec& time,
                 double confidenceInterval,
                 TDouble2Vec1Vec& value) const override;

    //! Get the best (least MSE) predicted value at \p time.
    TDouble2Vec predict(core_t::TTime time,
                        const TSizeDoublePr1Vec& correlated = TSizeDoublePr1Vec(),
                        TDouble2Vec hint = TDouble2Vec()) const override;

    //! Get the prediction and \p confidenceInterval percentage
    //! confidence interval for the time series at \p time.
    TDouble2Vec3Vec confidenceInterval(core_t::TTime time,
                                       double confidenceInterval,
                                       const TDouble2VecWeightsAry& weights) const override;

    //! Forecast the time series and get its \p confidenceInterval
    //! percentage confidence interval between \p startTime and
    //! \p endTime.
    bool forecast(core_t::TTime firstDataTime,
                  core_t::TTime lastDataTime,
                  core_t::TTime startTime,
                  core_t::TTime endTime,
                  double confidenceInterval,
                  const TDouble2Vec& minimum,
                  const TDouble2Vec& maximum,
                  const common::TForecastPushDatapointFunc& forecastPushDataPointFunc,
                  std::string& messageOut) override;

    //! Compute the probability of drawing \p value at \p time.
    bool probability(const common::CModelProbabilityParams& params,
                     const TTime2Vec1Vec& time,
                     const TDouble2Vec1Vec& value,
                     common::SModelProbabilityResult& result) const override;

    //! Fill in \p trendWeights and \p residualWeights with the count related
    //! weights for \p value.
    void countWeights(core_t::TTime time,
                      const TDouble2Vec& value,
                      double trendCountWeight,
                      double residualCountWeight,
                      double winsorisationDerate,
                      double countVarianceScale,
                      TDouble2VecWeightsAry& trendWeights,
                      TDouble2VecWeightsAry& residualWeights) const override;

    //! Add to \p trendWeights and \p residualWeights.
    void addCountWeights(core_t::TTime time,
                         double trendCountWeight,
                         double residualCountWeight,
                         double countVarianceScale,
                         TDouble2VecWeightsAry& trendWeights,
                         TDouble2VecWeightsAry& residualWeights) const override;

    //! Fill in the seasonal variance scale at \p time.
    void seasonalWeight(double confidence, core_t::TTime time, TDouble2Vec& weight) const override;

    //! Compute a checksum for this object.
    std::uint64_t checksum(std::uint64_t seed = 0) const override;

    //! Debug the memory used by this object.
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const override;

    //! Get the memory used by this object.
    std::size_t memoryUsage() const override;

    //! Initialize reading state from \p traverser.
    bool acceptRestoreTraverser(const common::SModelRestoreParams& params,
                                core::CStateRestoreTraverser& traverser);

    //! Persist by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const override;

    //! Persist the state of the residual models only.
    void persistModelsState(core::CStatePersistInserter& inserter) const override;

    //! Get the type of data being modeled.
    maths_t::EDataType dataType() const override;

    //! Unpack the weights in \p weights.
    static TDoubleWeightsAry unpack(const TDouble2VecWeightsAry& weights);

    //! \name Test Functions
    //@{
    //! Get the trend.
    const CTimeSeriesDecompositionInterface& trendModel() const;

    //! Get the residual model.
    const common::CPrior& residualModel() const;

    //! Get the decay rate controllers.
    const TDecayRateController2Ary* decayRateControllers() const;
    //@}

private:
    using TDouble1Vec = core::CSmallVector<double, 1>;
    using TDouble1VecVec = std::vector<TDouble1Vec>;
    using TTimeDouble2VecSizeTrVecDoublePr = std::pair<TTimeDouble2VecSizeTrVec, double>;
    using TMultibucketFeaturePtr = std::unique_ptr<TMultibucketFeature>;
    using TDecayRateController2AryPtr = std::unique_ptr<TDecayRateController2Ary>;
    using TPriorPtr = std::shared_ptr<common::CPrior>;
    using TAnomalyModelPtr = std::unique_ptr<CTimeSeriesAnomalyModel>;
    using TMultivariatePriorCPtrSizePr =
        std::pair<const common::CMultivariatePrior*, std::size_t>;
    using TMultivariatePriorCPtrSizePr1Vec =
        core::CSmallVector<TMultivariatePriorCPtrSizePr, 1>;
    using TModelCPtr1Vec = core::CSmallVector<const CUnivariateTimeSeriesModel*, 1>;

private:
    CUnivariateTimeSeriesModel(const CUnivariateTimeSeriesModel& other,
                               std::size_t id,
                               bool isForForecast = false);

    //! Update the trend with \p samples.
    EUpdateResult updateTrend(const common::CModelAddSamplesParams& params,
                              const TTimeDouble2VecSizeTrVec& samples);

    //! Update the residual models.
    TTimeDouble2VecSizeTrVecDoublePr
    updateResidualModels(const common::CModelAddSamplesParams& params,
                         TTimeDouble2VecSizeTrVec samples);

    //! Update the various model decay rates based on the prediction errors
    //! for \p samples.
    double updateDecayRates(const common::CModelAddSamplesParams& params,
                            core_t::TTime time,
                            const TDouble1Vec& samples);

    //! Compute the prediction errors for \p sample.
    void appendPredictionErrors(double interval, double sample, TDouble1VecVec (&result)[2]);

    //! Reinitialize state after detecting a new component of the trend
    //! decomposition.
    void reinitializeStateGivenNewComponent(TFloatMeanAccumulatorVec residuals);

    //! Compute the probability for uncorrelated series.
    bool uncorrelatedProbability(const common::CModelProbabilityParams& params,
                                 const TTime2Vec1Vec& time,
                                 const TDouble2Vec1Vec& value,
                                 common::SModelProbabilityResult& result) const;

    //! Compute the probability for correlated series.
    bool correlatedProbability(const common::CModelProbabilityParams& params,
                               const TTime2Vec1Vec& time,
                               const TDouble2Vec1Vec& value,
                               common::SModelProbabilityResult& result) const;

    //! Get the models for the correlations and the models of the correlated
    //! time series.
    bool correlationModels(TSize1Vec& correlated,
                           TSize2Vec1Vec& variables,
                           TMultivariatePriorCPtrSizePr1Vec& correlationDistributionModels,
                           TModelCPtr1Vec& correlatedTimeSeriesModels) const;

    //! Check the state invariants after restoration
    //! Abort on failure.
    void checkRestoredInvariants() const;

private:
    //! A unique identifier for this model.
    std::size_t m_Id;

    //! True if the data are non-negative.
    bool m_IsNonNegative;

    //! True if the model can be forecast.
    bool m_IsForecastable;

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

    //! The multi-bucket feature to use.
    TMultibucketFeaturePtr m_MultibucketFeature;

    //! A model of the multi-bucket feature.
    TPriorPtr m_MultibucketFeatureModel;

    //! A model for time periods when the basic model can't predict the
    //! value of the time series.
    TAnomalyModelPtr m_AnomalyModel;

    //! Models the correlations between time series.
    CTimeSeriesCorrelations* m_Correlations;
};

//! \brief Manages the creation correlate models.
class MATHS_TIME_SERIES_EXPORT CTimeSeriesCorrelateModelAllocator {
public:
    using TMultivariatePriorPtr = std::unique_ptr<common::CMultivariatePrior>;

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
class MATHS_TIME_SERIES_EXPORT CTimeSeriesCorrelations {
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
    using TMultivariatePriorPtr = std::unique_ptr<common::CMultivariatePrior>;
    using TMultivariatePriorPtrDoublePr = std::pair<TMultivariatePriorPtr, double>;
    using TSizeSizePrMultivariatePriorPtrDoublePrUMap =
        boost::unordered_map<TSizeSizePr, TMultivariatePriorPtrDoublePr>;
    using TMultivariatePriorCPtrSizePr =
        std::pair<const common::CMultivariatePrior*, std::size_t>;
    using TMultivariatePriorCPtrSizePr1Vec =
        core::CSmallVector<TMultivariatePriorCPtrSizePr, 1>;

    //! \brief Wraps up the sampled data for a feature.
    struct MATHS_TIME_SERIES_EXPORT SSampleData {
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
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const;

    //! Get the memory used by this object.
    std::size_t memoryUsage() const;

    //! Initialize reading state from \p traverser.
    bool acceptRestoreTraverser(const common::SDistributionRestoreParams& params,
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
    bool restoreCorrelationModels(const common::SDistributionRestoreParams& params,
                                  core::CStateRestoreTraverser& traverser);

    //! Persist the correlation distribution models passing information
    //! to \p inserter.
    void persistCorrelationModels(core::CStatePersistInserter& inserter) const;

    //! Restore the \p model reading state from \p traverser.
    static bool restore(const common::SDistributionRestoreParams& params,
                        TSizeSizePrMultivariatePriorPtrDoublePrPr& model,
                        core::CStateRestoreTraverser& traverser);

    //! Persist the \p model passing information to \p inserter.
    static void persist(const TConstSizeSizePrMultivariatePriorPtrDoublePrPr& model,
                        core::CStatePersistInserter& inserter);

    //! Add the time series identified by \p id.
    void addTimeSeries(std::size_t id, const CUnivariateTimeSeriesModel& model);

    //! Remove the time series identified by \p id.
    void removeTimeSeries(std::size_t id);

    //! Clear all correlation information for time series identified \p id.
    void clearCorrelationModels(std::size_t id);

    //! Add a sample for the time series identified by \p id.
    void addSamples(std::size_t id,
                    const common::CModelAddSamplesParams& params,
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
    common::CKMostCorrelated m_Correlations;

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
class MATHS_TIME_SERIES_EXPORT CMultivariateTimeSeriesModel : public common::CModel {
public:
    using TDouble10Vec = core::CSmallVector<double, 10>;
    using TDouble10Vec1Vec = core::CSmallVector<TDouble10Vec, 1>;
    using TFloatMeanAccumulator =
        common::CBasicStatistics::SSampleMean<common::CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
    using TFloatMeanAccumulatorVec10Vec = core::CSmallVector<TFloatMeanAccumulatorVec, 10>;
    using TDouble10VecWeightsAry = maths_t::TDouble10VecWeightsAry;
    using TDouble10VecWeightsAry1Vec = core::CSmallVector<TDouble10VecWeightsAry, 1>;
    using TDecompositionPtr = std::shared_ptr<CTimeSeriesDecompositionInterface>;
    using TDecompositionPtr10Vec = core::CSmallVector<TDecompositionPtr, 10>;
    using TDecayRateController2Ary = std::array<CDecayRateController, 2>;
    using TMultibucketFeature = CTimeSeriesMultibucketVectorFeature;

public:
    //! \param[in] params The model parameters.
    //! \param[in] trendModel The time series trend decomposition.
    //! \param[in] residualModel The prior for the time series residual model.
    //! \param[in] controllers Optional decay rate controllers for the trend
    //! and residual model.
    //! \param[in] multibucketFeature The multi-bucket feature to analyse if any.
    //! \param[in] modelAnomalies If true we use a separate model to capture
    //! the characteristics of anomalous time periods.
    CMultivariateTimeSeriesModel(const common::CModelParams& params,
                                 const CTimeSeriesDecompositionInterface& trendModel,
                                 const common::CMultivariatePrior& residualModel,
                                 const TDecayRateController2Ary* controllers = nullptr,
                                 const TMultibucketFeature* multibucketFeature = nullptr,
                                 bool modelAnomalies = true);
    CMultivariateTimeSeriesModel(const CMultivariateTimeSeriesModel& other);
    CMultivariateTimeSeriesModel(const common::SModelRestoreParams& params,
                                 core::CStateRestoreTraverser& traverser);
    ~CMultivariateTimeSeriesModel() override;

    const CMultivariateTimeSeriesModel& operator=(const CMultivariateTimeSeriesModel&) = delete;

    //! Returns 0 since these models don't need a unique identifier.
    std::size_t identifier() const override;

    //! Create a copy of this model passing ownership to the caller.
    CMultivariateTimeSeriesModel* clone(std::size_t id) const override;

    //! Create a copy of the state we need to persist passing ownership
    //! to the caller.
    CMultivariateTimeSeriesModel* cloneForPersistence() const override;

    //! Create a copy of the state we need to run forecasting.
    CMultivariateTimeSeriesModel* cloneForForecast() const override;

    //! Returns false (not currently supported for multivariate features).
    bool isForecastPossible() const override;

    //! No-op.
    void modelCorrelations(CTimeSeriesCorrelations& model) override;

    //! Returns empty.
    TSize2Vec1Vec correlates() const override;

    //! Update the model with the bucket \p value.
    void addBucketValue(const TTimeDouble2VecSizeTrVec& value) override;

    //! Update the model with new samples.
    EUpdateResult addSamples(const common::CModelAddSamplesParams& params,
                             TTimeDouble2VecSizeTrVec samples) override;

    //! Advance time by \p gap.
    void skipTime(core_t::TTime gap) override;

    //! Get the most likely value for the time series at \p time.
    TDouble2Vec mode(core_t::TTime time, const TDouble2VecWeightsAry& weights) const override;

    //! Returns empty.
    TDouble2Vec1Vec correlateModes(core_t::TTime time,
                                   const TDouble2VecWeightsAry1Vec& weights) const override;

    //! Get the local maxima of the residual distribution.
    TDouble2Vec1Vec residualModes(const TDouble2VecWeightsAry& weights) const override;

    //! Remove any trend components from \p value.
    void detrend(const TTime2Vec1Vec& time,
                 double confidenceInterval,
                 TDouble2Vec1Vec& value) const override;

    //! Get the best (least MSE) predicted value at \p time.
    TDouble2Vec predict(core_t::TTime time,
                        const TSizeDoublePr1Vec& correlated = TSizeDoublePr1Vec(),
                        TDouble2Vec hint = TDouble2Vec()) const override;

    //! Get the prediction and \p confidenceInterval percentage
    //! confidence interval for the time series at \p time.
    TDouble2Vec3Vec confidenceInterval(core_t::TTime time,
                                       double confidenceInterval,
                                       const TDouble2VecWeightsAry& weights) const override;

    //! Not currently supported.
    bool forecast(core_t::TTime firstDataTime,
                  core_t::TTime lastDataTime,
                  core_t::TTime startTime,
                  core_t::TTime endTime,
                  double confidenceInterval,
                  const TDouble2Vec& minimum,
                  const TDouble2Vec& maximum,
                  const common::TForecastPushDatapointFunc& forecastPushDataPointFunc,
                  std::string& messageOut) override;

    //! Compute the probability of drawing \p value at \p time.
    bool probability(const common::CModelProbabilityParams& params,
                     const TTime2Vec1Vec& time,
                     const TDouble2Vec1Vec& value,
                     common::SModelProbabilityResult& result) const override;

    //! Fill in \p trendWeights and \p residualWeights with the count related
    //! weights for \p value.
    void countWeights(core_t::TTime time,
                      const TDouble2Vec& value,
                      double trendCountWeight,
                      double residualCountWeight,
                      double winsorisationDerate,
                      double countVarianceScale,
                      TDouble2VecWeightsAry& trendWeights,
                      TDouble2VecWeightsAry& residualWeights) const override;

    //! Add to \p trendWeights and \p residualWeights.
    void addCountWeights(core_t::TTime time,
                         double trendCountWeight,
                         double residualCountWeight,
                         double countVarianceScale,
                         TDouble2VecWeightsAry& trendWeights,
                         TDouble2VecWeightsAry& residualWeights) const override;

    //! Fill in the seasonal variance scale at \p time.
    void seasonalWeight(double confidence, core_t::TTime time, TDouble2Vec& weight) const override;

    //! Compute a checksum for this object.
    std::uint64_t checksum(std::uint64_t seed = 0) const override;

    //! Debug the memory used by this object.
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const override;

    //! Get the memory used by this object.
    std::size_t memoryUsage() const override;

    //! Initialize reading state from \p traverser.
    bool acceptRestoreTraverser(const common::SModelRestoreParams& params,
                                core::CStateRestoreTraverser& traverser);

    //! Persist by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const override;

    //! Persist the state of the residual models only.
    void persistModelsState(core::CStatePersistInserter& inserter) const override;

    //! Get the type of data being modeled.
    maths_t::EDataType dataType() const override;

    //! Unpack the weights in \p weights.
    static TDouble10VecWeightsAry unpack(const TDouble2VecWeightsAry& weights);

    //! \name Test Functions
    //@{
    //! Get the trend.
    const TDecompositionPtr10Vec& trendModel() const;

    //! Get the residual model.
    const common::CMultivariatePrior& residualModel() const;

    //! Get the decay rate controllers.
    const TDecayRateController2Ary* decayRateControllers() const;
    //@}

private:
    using TDouble1Vec = core::CSmallVector<double, 1>;
    using TDouble1VecVec = std::vector<TDouble1Vec>;
    using TMultibucketFeaturePtr = std::unique_ptr<TMultibucketFeature>;
    using TDecayRateController2AryPtr = std::unique_ptr<TDecayRateController2Ary>;
    using TMultivariatePriorPtr = std::unique_ptr<common::CMultivariatePrior>;
    using TAnomalyModelPtr = std::unique_ptr<CTimeSeriesAnomalyModel>;

private:
    //! Update the trend with \p samples.
    EUpdateResult updateTrend(const common::CModelAddSamplesParams& params,
                              const TTimeDouble2VecSizeTrVec& samples);

    //! Update the residual models.
    void updateResidualModels(const common::CModelAddSamplesParams& params,
                              TTimeDouble2VecSizeTrVec samples);

    //! Update the various model decay rates based on the prediction errors
    //! for \p samples.
    void updateDecayRates(const common::CModelAddSamplesParams& params,
                          core_t::TTime time,
                          const TDouble10Vec1Vec& samples);

    //! Compute the prediction errors for \p sample.
    void appendPredictionErrors(double interval,
                                const TDouble10Vec& sample,
                                TDouble1VecVec (&result)[2]);

    //! Reinitialize state after detecting a new component of the trend
    //! decomposition.
    void reinitializeStateGivenNewComponent(TFloatMeanAccumulatorVec10Vec residuals);

    //! Get the model dimension.
    std::size_t dimension() const;

    //! Check the state invariants after restoration
    //! Abort on failure.
    void checkRestoredInvariants() const;

private:
    //! True if the data are non-negative.
    bool m_IsNonNegative;

    //! These control the trend and residual model decay rates (see
    //! CDecayRateController for more details).
    TDecayRateController2AryPtr m_Controllers;

    //! The time series trend decomposition.
    TDecompositionPtr10Vec m_TrendModel;

    //! The time series' residual model.
    TMultivariatePriorPtr m_ResidualModel;

    //! The multi-bucket feature to use.
    TMultibucketFeaturePtr m_MultibucketFeature;

    //! A model of the multi-bucket feature.
    TMultivariatePriorPtr m_MultibucketFeatureModel;

    //! A model for time periods when the basic model can't predict the
    //! value of the time series.
    TAnomalyModelPtr m_AnomalyModel;
};
}
}
}

#endif // INCLUDE_ml_maths_time_series_CTimeSeriesModel_h
