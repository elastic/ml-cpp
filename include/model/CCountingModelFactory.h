/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CCountingModelFactory_h
#define INCLUDED_ml_model_CCountingModelFactory_h

#include <model/CModelFactory.h>
#include <model/ImportExport.h>

namespace ml {
namespace core {
class CStateRestoreTraverser;
}
namespace model {

//! \brief A factory class implementation for CCountingModel.
//!
//! DESCRIPTION:\n
//! This concrete factory implements the methods to make new models
//! and data gatherers, and create default priors suitable for the
//! CCountingModel class.
class MODEL_EXPORT CCountingModelFactory : public CModelFactory {
public:
    //! Lift all overloads of the defaultPrior function into scope.
    using CModelFactory::defaultPrior;

public:
    //! \note The default arguments supplied to the constructor are
    //! intended for unit testing and are not necessarily good defaults.
    //! The CModelConfig class is responsible for providing sensible
    //! default values for the factory for use within our products.
    explicit CCountingModelFactory(const SModelParams& params,
                                   model_t::ESummaryMode summaryMode = model_t::E_None,
                                   const std::string& summaryCountFieldName = "");

    //! Create a copy of the factory owned by the calling code.
    virtual CCountingModelFactory* clone() const;

    //! \name Factory Methods
    //@{
    //! Make a new counting model.
    //!
    //! \param[in] initData The parameters needed to initialize the model.
    //! \warning It is owned by the calling code.
    virtual CAnomalyDetectorModel* makeModel(const SModelInitializationData& initData) const;

    //! Make a new counting model from part of a state document.
    //!
    //! \param[in] initData Additional parameters needed to initialize
    //! the model.
    //! \param[in,out] traverser A state document traverser.
    //! \warning It is owned by the calling code.
    virtual CAnomalyDetectorModel* makeModel(const SModelInitializationData& initData, core::CStateRestoreTraverser& traverser) const;

    //! Make a new event rate data gatherer.
    //!
    //! \param[in] initData The parameters needed to initialize the data
    //! gatherer.
    //! \warning It is owned by the calling code.
    virtual CDataGatherer* makeDataGatherer(const SGathererInitializationData& initData) const;

    //! Make a new event rate data gatherer from part of a state document.
    //!
    //! \param[in] partitionFieldValue The partition field value.
    //! \param[in,out] traverser A state document traverser.
    //! \warning It is owned by the calling code.
    virtual CDataGatherer* makeDataGatherer(const std::string& partitionFieldValue, core::CStateRestoreTraverser& traverser) const;
    //@}

    //! \name Defaults
    //@{
    //! Get the default prior for \p feature which is a stub.
    virtual TPriorPtr defaultPrior(model_t::EFeature feature, const SModelParams& params) const;

    //! Get the default prior for \p feature which is a stub.
    virtual TMultivariatePriorPtr defaultMultivariatePrior(model_t::EFeature feature, const SModelParams& params) const;

    //! Get the default prior for pairs of correlated time series
    //! of \p feature which is a stub.
    virtual TMultivariatePriorPtr defaultCorrelatePrior(model_t::EFeature feature, const SModelParams& params) const;
    //@}

    //! Get the search key corresponding to this factory.
    virtual const CSearchKey& searchKey() const;

    //! Check if this makes the model used for a simple counting search.
    virtual bool isSimpleCount() const;

    //! Check the pre-summarisation mode for this factory.
    virtual model_t::ESummaryMode summaryMode() const;

    //! Get the default data type for models from this factory.
    virtual maths_t::EDataType dataType() const;

    //! \name Customization by a specific search
    //@{
    //! Set the identifier of the search for which this generates models.
    virtual void identifier(int identifier);

    //! Set the name of the field whose values will be counted.
    virtual void fieldNames(const std::string& partitionFieldName,
                            const std::string& overFieldName,
                            const std::string& byFieldName,
                            const std::string& valueFieldName,
                            const TStrVec& influenceFieldNames);

    //! Set whether the models should process missing person fields.
    virtual void useNull(bool useNull);

    //! Set the features which will be modeled.
    virtual void features(const TFeatureVec& features);

    //! Set the bucket results delay
    virtual void bucketResultsDelay(std::size_t bucketResultsDelay);
    //@}

private:
    //! Get the field values which partition the data for modeling.
    virtual TStrCRefVec partitioningFields() const;

private:
    //! The identifier of the search for which this generates models.
    int m_Identifier;

    //! Indicates whether the data being gathered are already summarized
    //! by an external aggregation process.
    model_t::ESummaryMode m_SummaryMode;

    //! If m_SummaryMode is E_Manual then this is the name of the field
    //! holding the summary count.
    std::string m_SummaryCountFieldName;

    //! The name of the field which splits the data.
    std::string m_PartitionFieldName;

    //! The name of the field whose values will be counted.
    std::string m_PersonFieldName;

    //! If true the models will process missing person fields.
    bool m_UseNull;

    //! The count features which will be modeled.
    TFeatureVec m_Features;

    //! The bucket results delay.
    std::size_t m_BucketResultsDelay;

    //! A cached search key.
    mutable TOptionalSearchKey m_SearchKeyCache;
};
}
}

#endif // INCLUDED_ml_model_CCountingModelFactory_h
