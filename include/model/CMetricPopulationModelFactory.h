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

#ifndef INCLUDED_ml_model_CMetricPopulationModelFactory_h
#define INCLUDED_ml_model_CMetricPopulationModelFactory_h

#include <model/CModelFactory.h>
#include <model/ImportExport.h>

namespace ml {
namespace core {
class CStateRestoreTraverser;
}
namespace model {

//! \brief A factory class implementation for CMetricPopulationModel.
//!
//! DESCRIPTION:\n
//! This concrete factory implements the methods to make new models
//! and data gatherers, and create default priors suitable for the
//! CMetricPopulationModel class.
class MODEL_EXPORT CMetricPopulationModelFactory final : public CModelFactory {
public:
    //! Lift all overloads into scope.
    using CModelFactory::defaultMultivariatePrior;
    using CModelFactory::defaultPrior;

public:
    //! \note The default arguments supplied to the constructor are
    //! intended for unit testing and are not necessarily good defaults.
    //! The CModelConfig class is responsible for providing sensible
    //! default values for the factory for use within our products.
    CMetricPopulationModelFactory(const SModelParams& params,
                                  const TInterimBucketCorrectorWPtr& interimBucketCorrector,
                                  model_t::ESummaryMode summaryMode = model_t::E_None,
                                  const std::string& summaryCountFieldName = "");

    //! Create a copy of the factory owned by the calling code.
    CMetricPopulationModelFactory* clone() const override;

    //! \name Factory Methods
    //@{
    //! Make a new population model for metric values.
    //!
    //! \param[in] initData The parameters needed to initialize the model.
    //! \warning It is owned by the calling code.
    CAnomalyDetectorModel* makeModel(const SModelInitializationData& initData) const override;

    //! Make a new metric population model from part of a state document.
    //!
    //! \param[in] initData Additional parameters needed to initialize
    //! the model.
    //! \param[in,out] traverser A state document traverser.
    //! \warning It is owned by the calling code.
    CAnomalyDetectorModel* makeModel(const SModelInitializationData& initData,
                                     core::CStateRestoreTraverser& traverser) const override;

    //! Make a new metric population data gatherer.
    //!
    //! \param[in] initData The parameters needed to initialize the data
    //! gatherer.
    //! \warning It is owned by the calling code.
    CDataGatherer* makeDataGatherer(const SGathererInitializationData& initData) const override;

    //! Make a new metric population data gatherer from part of a state
    //! document.
    //!
    //! \param[in] partitionFieldValue The partition field value.
    //! \param[in,out] traverser A state document traverser.
    //! \warning It is owned by the calling code.
    CDataGatherer* makeDataGatherer(const std::string& partitionFieldValue,
                                    core::CStateRestoreTraverser& traverser) const override;
    //@}

    //! \name Defaults
    //@{
    //! Get the default prior for \p feature.
    //!
    //! \param[in] feature The feature for which to get the prior.
    //! \param[in] params The model parameters.
    TPriorPtr defaultPrior(model_t::EFeature feature, const SModelParams& params) const override;

    //! Get the default multivariate prior for \p feature.
    //!
    //! \param[in] feature The feature for which to get the prior.
    //! \param[in] params The model parameters.
    TMultivariatePriorUPtr defaultMultivariatePrior(model_t::EFeature feature,
                                                    const SModelParams& params) const override;

    //! Get the default prior for pairs of correlated time series
    //! of \p feature.
    //!
    //! \param[in] feature The feature for which to get the prior.
    //! \param[in] params The model parameters.
    TMultivariatePriorUPtr defaultCorrelatePrior(model_t::EFeature feature,
                                                 const SModelParams& params) const override;
    //@}

    //! Get the search key corresponding to this factory.
    const CSearchKey& searchKey() const override;

    //! Returns false.
    bool isSimpleCount() const override;

    //! Check the pre-summarisation mode for this factory.
    model_t::ESummaryMode summaryMode() const override;

    //! Get the default data type for models from this factory.
    maths_t::EDataType dataType() const override;

    //! \name Customization
    //@{
    //! Set the identifier of the search for which this generates models.
    void detectorIndex(int detectorIndex) override;

    //! Set the name of the field whose values will be counted.
    void fieldNames(const std::string& partitionFieldName,
                    const std::string& overFieldName,
                    const std::string& byFieldName,
                    const std::string& valueFieldName,
                    const TStrVec& influenceFieldNames) override;

    //! Set whether the models should process missing person and
    //! attribute fields.
    void useNull(bool useNull) override;

    //! Set the features which will be modeled.
    void features(const TFeatureVec& features) override;
    //@}

    //! Get the minimum seasonal variance scale
    double minimumSeasonalVarianceScale() const override;

private:
    //! Get the field values which partition the data for modeling.
    TStrCRefVec partitioningFields() const override;

    //! Restores a single population metric model.
    bool modelAcceptRestoreTraverser(const SModelInitializationData& initData,
                                     core::CStateRestoreTraverser& traverser,
                                     CAnomalyDetectorModel*& model) const;

private:
    //! The identifier of the search for which this generates models.
    int m_DetectorIndex = 0;

    //! Indicates whether the data being gathered are already summarized
    //! by an external aggregation process
    model_t::ESummaryMode m_SummaryMode;

    //! If m_SummaryMode is E_Manual then this is the name of the field
    //! holding the summary count.
    std::string m_SummaryCountFieldName;

    //! The name of the field which splits the data.
    std::string m_PartitionFieldName;

    //! The name of the field which defines the population which
    //! will be analyzed.
    std::string m_PersonFieldName;

    //! The name of the field which defines the person attributes
    //! which will be analyzed.
    std::string m_AttributeFieldName;

    //! The name of field whose values define the metric series'
    //! values which will be analyzed.
    std::string m_ValueFieldName;

    //! The field names for which we are computing influence. These are
    //! the fields which can be used to join results across different
    //! searches.
    TStrVec m_InfluenceFieldNames;

    //! If true the models will process missing person and attribute
    //! fields.
    bool m_UseNull = false;

    //! The count features which will be modeled.
    TFeatureVec m_Features;

    //! A cached search key.
    mutable TOptionalSearchKey m_SearchKeyCache;
};
}
}

#endif // INCLUDED_ml_model_CMetricPopulationModelFactory_h
