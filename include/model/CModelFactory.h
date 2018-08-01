/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CModelFactory_h
#define INCLUDED_ml_model_CModelFactory_h

#include <core/CNonCopyable.h>
#include <core/CoreTypes.h>

#include <maths/COrderings.h>
#include <maths/MathsTypes.h>

#include <model/CModelParams.h>
#include <model/CSearchKey.h>
#include <model/ImportExport.h>
#include <model/ModelTypes.h>

#include <boost/optional.hpp>
#include <boost/ref.hpp>
#include <boost/unordered_map.hpp>

#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace ml {
namespace core {
class CStateRestoreTraverser;
}

namespace maths {
class CModel;
class CMultinomialConjugate;
class CMultivariatePrior;
class CPrior;
class CTimeSeriesCorrelations;
class CTimeSeriesDecompositionInterface;
}

namespace model {
class CAnomalyDetectorModel;
class CDataGatherer;
class CDetectionRule;
class CInfluenceCalculator;
class CInterimBucketCorrector;
class CSearchKey;

//! \brief A factory class interface for the CAnomalyDetectorModel hierarchy.
//!
//! DESCRIPTION:\n
//! The interface for the factory classes for making concrete objects
//! in the CAnomalyDetectorModel hierarchy.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The CModelConfig object is able to dynamically configure the
//! anomaly detection classes:
//!   -# CAnomalyDetector,
//!   -# CAnomalyDetectorModel,
//!      ...
//!
//! to either compute online or delta probabilities for log messages,
//! metric values, etc. This hierarchy implements the factory pattern
//! for the CAnomalyDetectorModel hierarchy for this purpose.
class MODEL_EXPORT CModelFactory {
public:
    using TFeatureVec = std::vector<model_t::EFeature>;
    using TStrVec = std::vector<std::string>;
    using TOptionalUInt = boost::optional<unsigned int>;
    using TStrCRef = boost::reference_wrapper<const std::string>;
    using TStrCRefVec = std::vector<TStrCRef>;
    using TDataGathererPtr = std::shared_ptr<CDataGatherer>;
    using TPriorPtr = std::unique_ptr<maths::CPrior>;
    using TMultivariatePriorSPtr = std::shared_ptr<maths::CMultivariatePrior>;
    using TMultivariatePriorUPtr = std::unique_ptr<maths::CMultivariatePrior>;
    using TFeatureMultivariatePriorSPtrPr = std::pair<model_t::EFeature, TMultivariatePriorSPtr>;
    using TFeatureMultivariatePriorSPtrPrVec = std::vector<TFeatureMultivariatePriorSPtrPr>;
    using TDecompositionCPtr = std::shared_ptr<const maths::CTimeSeriesDecompositionInterface>;
    using TMathsModelPtr = std::shared_ptr<maths::CModel>;
    using TFeatureMathsModelPtrPr = std::pair<model_t::EFeature, TMathsModelPtr>;
    using TFeatureMathsModelPtrPrVec = std::vector<TFeatureMathsModelPtrPr>;
    using TCorrelationsPtr = std::unique_ptr<maths::CTimeSeriesCorrelations>;
    using TFeatureCorrelationsPtrPr = std::pair<model_t::EFeature, TCorrelationsPtr>;
    using TFeatureCorrelationsPtrPrVec = std::vector<TFeatureCorrelationsPtrPr>;
    using TModelPtr = std::shared_ptr<CAnomalyDetectorModel>;
    using TModelCPtr = std::shared_ptr<const CAnomalyDetectorModel>;
    using TInfluenceCalculatorCPtr = std::shared_ptr<const CInfluenceCalculator>;
    using TFeatureInfluenceCalculatorCPtrPr =
        std::pair<model_t::EFeature, TInfluenceCalculatorCPtr>;
    using TFeatureInfluenceCalculatorCPtrPrVec = std::vector<TFeatureInfluenceCalculatorCPtrPr>;
    using TFeatureInfluenceCalculatorCPtrPrVecVec =
        std::vector<TFeatureInfluenceCalculatorCPtrPrVec>;
    using TInterimBucketCorrectorWPtr = std::weak_ptr<CInterimBucketCorrector>;
    using TInterimBucketCorrectorPtr = std::shared_ptr<CInterimBucketCorrector>;
    using TDetectionRuleVec = std::vector<CDetectionRule>;
    using TDetectionRuleVecCRef = boost::reference_wrapper<const TDetectionRuleVec>;
    using TStrDetectionRulePr = std::pair<std::string, model::CDetectionRule>;
    using TStrDetectionRulePrVec = std::vector<TStrDetectionRulePr>;
    using TStrDetectionRulePrVecCRef = boost::reference_wrapper<const TStrDetectionRulePrVec>;

public:
    //! Wrapper around the model initialization data.
    //!
    //! IMPLEMENTATION DECISIONS:\n
    //! We wrap up the initialization data in an object so we don't
    //! need to change the signature of every factory function each
    //! time we need extra data to initialize a model.
    struct MODEL_EXPORT SModelInitializationData {
        SModelInitializationData(const TDataGathererPtr& dataGatherer);

        TDataGathererPtr s_DataGatherer;
    };

    //! Wrapper around the data gatherer initialization data.
    //!
    //! IMPLEMENTATION DECISIONS:\n
    //! We wrap up the initialization data in an object so we don't
    //! need to change the signature of every factory function each
    //! time we need extra data to initialize a data gatherer.
    struct MODEL_EXPORT SGathererInitializationData {
        SGathererInitializationData(core_t::TTime startTime,
                                    const std::string& partitionFieldValue,
                                    unsigned int sampleOverrideCount = 0u);

        //! This constructor is to simplify unit testing.
        SGathererInitializationData(const core_t::TTime startTime);

        core_t::TTime s_StartTime;
        const std::string& s_PartitionFieldValue;
        unsigned int s_SampleOverrideCount;
    };

public:
    static const std::string EMPTY_STRING;

public:
    //! \warning The user must ensure that \p interimBucketCorrector
    //! outlives this object. If model factories are obtained from
    //! CModelConfig this is ensured for you.
    CModelFactory(const SModelParams& params,
                  const TInterimBucketCorrectorWPtr& interimBucketCorrector);
    virtual ~CModelFactory() = default;

    //! Create a copy of the factory owned by the calling code.
    virtual CModelFactory* clone() const = 0;

    //! \name Factory Methods
    //@{
    //! Make a new model.
    //!
    //! \param[in] initData The parameters needed to initialize the model.
    //! \warning It is owned by the calling code.
    virtual CAnomalyDetectorModel*
    makeModel(const SModelInitializationData& initData) const = 0;

    //! Make a new model from part of a state document.
    //!
    //! \param[in] initData Additional parameters needed to initialize
    //! the model.
    //! \param[in,out] traverser A state document traverser.
    //! \warning It is owned by the calling code.
    virtual CAnomalyDetectorModel*
    makeModel(const SModelInitializationData& initData,
              core::CStateRestoreTraverser& traverser) const = 0;

    //! Make a new data gatherer.
    //!
    //! \param[in] initData The parameters needed to initialize the
    //! data gatherer.
    //! \warning It is owned by the calling code.
    virtual CDataGatherer*
    makeDataGatherer(const SGathererInitializationData& initData) const = 0;

    //! Make a new data gatherer from part of a state document.
    //!
    //! \param[in,out] traverser A state document traverser.
    //! \param[in] partitionFieldValue The partition field value.
    //! \warning It is owned by the calling code.
    virtual CDataGatherer* makeDataGatherer(const std::string& partitionFieldValue,
                                            core::CStateRestoreTraverser& traverser) const = 0;
    //@}

    //! \name Defaults
    //@{
    //! Get the default models to use for \p features and \p bucketLength.
    const TFeatureMathsModelPtrPrVec&
    defaultFeatureModels(const TFeatureVec& features,
                         core_t::TTime bucketLength,
                         double minimumSeasonalVarianceScale,
                         bool modelAnomalies,
                         std::size_t multibucketFeaturesWindowLength) const;

    //! Get the default model to use for \p features and \p bucketLength.
    TMathsModelPtr defaultFeatureModel(model_t::EFeature feature,
                                       core_t::TTime bucketLength,
                                       double minimumSeasonalVarianceScale,
                                       bool modelAnomalies,
                                       std::size_t multibucketFeaturesWindowLength) const;

    //! Get the default correlate priors to use for correlated pairs of time
    //! series of \p features.
    const TFeatureMultivariatePriorSPtrPrVec&
    defaultCorrelatePriors(const TFeatureVec& features) const;

    //! Get the default models for correlations of \p features.
    TFeatureCorrelationsPtrPrVec defaultCorrelates(const TFeatureVec& features) const;

    //! Get the default prior to use for \p feature.
    TPriorPtr defaultPrior(model_t::EFeature feature) const;

    //! Get the default prior to use for multivariate \p feature.
    TMultivariatePriorUPtr defaultMultivariatePrior(model_t::EFeature feature) const;

    //! Get the default prior to use for correlared pairs of time
    //! series for univariate \p feature.
    TMultivariatePriorUPtr defaultCorrelatePrior(model_t::EFeature feature) const;

    //! Get the default prior for \p feature.
    //!
    //! \param[in] feature The feature for which to get the prior.
    //! \param[in] params The model parameters.
    virtual TPriorPtr defaultPrior(model_t::EFeature feature,
                                   const SModelParams& params) const = 0;

    //! Get the default prior for multivariate \p feature.
    //!
    //! \param[in] feature The feature for which to get the prior.
    //! \param[in] params The model parameters.
    virtual TMultivariatePriorUPtr
    defaultMultivariatePrior(model_t::EFeature feature, const SModelParams& params) const = 0;

    //! Get the default prior for pairs of correlated time series
    //! of \p feature.
    //!
    //! \param[in] feature The feature for which to get the prior.
    //! \param[in] params The model parameters.
    virtual TMultivariatePriorUPtr
    defaultCorrelatePrior(model_t::EFeature feature, const SModelParams& params) const = 0;

    //! Get the default prior to use for categorical data.
    maths::CMultinomialConjugate defaultCategoricalPrior() const;

    //! Get the default time series decomposition.
    //!
    //! \param[in] feature The feature for which to get the decomposition.
    //! \param[in] bucketLength The data bucketing length.
    TDecompositionCPtr defaultDecomposition(model_t::EFeature feature,
                                            core_t::TTime bucketLength) const;

    //! Get the influence calculators to use for each feature in \p features.
    const TFeatureInfluenceCalculatorCPtrPrVec&
    defaultInfluenceCalculators(const std::string& influencerName,
                                const TFeatureVec& features) const;
    //@}

    //! Get the search key corresponding to this factory.
    virtual const CSearchKey& searchKey() const = 0;

    //! Check if this makes the model used for a simple counting search.
    virtual bool isSimpleCount() const = 0;

    //! Check the pre-summarisation mode for this factory.
    virtual model_t::ESummaryMode summaryMode() const = 0;

    //! Get the default data type for models from this factory.
    virtual maths_t::EDataType dataType() const = 0;

    //! \name Customization by a specific search
    //@{
    //! Set the identifier of the search for which this generates models.
    virtual void identifier(int identifier) = 0;

    //! Set the record field names which will be modeled.
    virtual void fieldNames(const std::string& partitionFieldName,
                            const std::string& overFieldName,
                            const std::string& byFieldName,
                            const std::string& valueFieldName,
                            const TStrVec& influenceFieldNames) = 0;

    //! Set whether the model should process missing field values.
    virtual void useNull(bool useNull) = 0;

    //! Set the features which will be modeled.
    virtual void features(const TFeatureVec& features) = 0;

    //! Set the amount by which metric sample count is reduced for
    //! fine-grained sampling when there is latency.
    void sampleCountFactor(std::size_t sampleCountFactor);

    //! Set the bucket results delay
    virtual void bucketResultsDelay(std::size_t bucketResultsDelay) = 0;

    //! Set whether the model should exclude frequent hitters from the
    //! calculations.
    void excludeFrequent(model_t::EExcludeFrequent excludeFrequent);

    //! Set the detection rules for a detector.
    void detectionRules(TDetectionRuleVecCRef detectionRules);
    //@}

    //! Set the scheduled events.
    void scheduledEvents(TStrDetectionRulePrVecCRef scheduledEvents);

    //! Set the interim bucket corrector.
    //!
    //! \warning The caller must ensure that \p interimBucketCorrector
    //! outlives this object. If model factories are obtained from
    //! CModelConfig this is ensured for you.
    void interimBucketCorrector(const TInterimBucketCorrectorWPtr& interimBucketCorrector);

    //! \name Customization
    //@{
    //! Set the learn rate used for initializing models.
    void learnRate(double learnRate);

    //! Set the decay rate used for initializing the models.
    void decayRate(double decayRate);

    //! Set the initial decay rate multiplier used for initializing
    //! models.
    void initialDecayRateMultiplier(double multiplier);

    //! Set the maximum number of times we'll update a person's model
    //! in a bucketing interval.
    void maximumUpdatesPerBucket(double maximumUpdatesPerBucket);

    //! Set the prune window scale factor minimum
    void pruneWindowScaleMinimum(double factor);

    //! Set the prune window scale factor maximum
    void pruneWindowScaleMaximum(double factor);

    //! Set the window length to use for multibucket features.
    //!
    //! \note A length of zero disables modeling of multibucket features altogether.
    void multibucketFeaturesWindowLength(std::size_t length);

    //! Set whether multivariate analysis of correlated 'by' fields should
    //! be performed.
    void multivariateByFields(bool enabled);

    //! Set the minimum mode fraction used for initializing the models.
    void minimumModeFraction(double minimumModeFraction);

    //! Set the minimum mode count used for initializing the models.
    void minimumModeCount(double minimumModeCount);

    //! Set the periods and the number of points we'll use to model
    //! of the seasonal components in the data.
    void componentSize(std::size_t componentSize);
    //@}

    //! Update the bucket length, for ModelAutoConfig's benefit
    void updateBucketLength(core_t::TTime length);

    //! Get global model configuration parameters.
    const SModelParams& modelParams() const;

    //! Get the minimum mode fraction used for initializing the models.
    double minimumModeFraction() const;

    //! Set the minimum mode count used for initializing the models.
    double minimumModeCount() const;

    //! Get the number of points to use for approximating each seasonal
    //! component.
    std::size_t componentSize() const;

    //! Get the minimum seasonal variance scale, specific to the model
    virtual double minimumSeasonalVarianceScale() const = 0;

protected:
    using TMultivariatePriorUPtrVec = std::vector<TMultivariatePriorUPtr>;
    using TOptionalSearchKey = boost::optional<CSearchKey>;

protected:
    //! Get the singleton interim bucket correction calculator.
    TInterimBucketCorrectorPtr interimBucketCorrector() const;

    //! Get a multivariate normal prior with dimension \p dimension.
    //!
    //! \param[in] dimension The dimension.
    //! \param[in] params The model parameters.
    //! \warning Up to ten dimensions are supported.
    TMultivariatePriorUPtr multivariateNormalPrior(std::size_t dimension,
                                                   const SModelParams& params) const;

    //! Get a multivariate multimodal prior with dimension \p dimension.
    //!
    //! \param[in] dimension The dimension.
    //! \param[in] params The model parameters.
    //! \warning Up to ten dimensions are supported.
    TMultivariatePriorUPtr
    multivariateMultimodalPrior(std::size_t dimension,
                                const SModelParams& params,
                                const maths::CMultivariatePrior& modePrior) const;

    //! Get a multivariate 1-of-n prior with dimension \p dimension.
    //!
    //! \param[in] dimension The dimension.
    //! \param[in] params The model parameters.
    //! \param[in] models The component models to select between.
    TMultivariatePriorUPtr multivariateOneOfNPrior(std::size_t dimension,
                                                   const SModelParams& params,
                                                   const TMultivariatePriorUPtrVec& models) const;

    //! Get the default prior for time-of-day and time-of-week modeling.
    //! This is just a mixture of normals which allows more modes than
    //! we typically do.
    //!
    //! \param[in] params The model parameters.
    TPriorPtr timeOfDayPrior(const SModelParams& params) const;

    //! Get the default prior for latitude and longitude modeling.
    //! This is just a mixture of correlate normals which allows more
    //! modes than we typically do.
    //!
    //! \param[in] params The model parameters.
    TMultivariatePriorUPtr latLongPrior(const SModelParams& params) const;

private:
    using TFeatureVecMathsModelMap = std::map<TFeatureVec, TFeatureMathsModelPtrPrVec>;
    using TFeatureVecMultivariatePriorMap =
        std::map<TFeatureVec, TFeatureMultivariatePriorSPtrPrVec>;
    using TStrFeatureVecPr = std::pair<std::string, TFeatureVec>;
    using TStrFeatureVecPrInfluenceCalculatorCPtrMap =
        std::map<TStrFeatureVecPr, TFeatureInfluenceCalculatorCPtrPrVec, maths::COrderings::SLess>;

private:
    //! Get the field values which partition the data for modeling.
    virtual TStrCRefVec partitioningFields() const = 0;

private:
    //! The global model configuration parameters.
    SModelParams m_ModelParams;

    //! A reference to the singleton interim bucket correction calculator.
    //!
    //! \note It is the responsibility of the user of the factory class
    //! to ensure that the interim bucket corrector is not deleted whilst
    //! still in use. We store it here by weak pointer since we don't want
    //! this to update the reference count so we properly account for its
    //! memory usage in the objects this creates.
    TInterimBucketCorrectorWPtr m_InterimBucketCorrector;

    //! A cache of models for collections of features.
    mutable TFeatureVecMathsModelMap m_MathsModelCache;

    //! A cache of priors for correlate pairs of collections of features.
    mutable TFeatureVecMultivariatePriorMap m_CorrelatePriorCache;

    //! A cache of influence calculators for collections of features.
    mutable TStrFeatureVecPrInfluenceCalculatorCPtrMap m_InfluenceCalculatorCache;
};
}
}

#endif // INCLUDED_ml_model_CModelFactory_h
