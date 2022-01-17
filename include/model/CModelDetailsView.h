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

#ifndef INCLUDED_ml_model_CModelDetailsView_h
#define INCLUDED_ml_model_CModelDetailsView_h

#include <model/CAnomalyDetectorModel.h>
#include <model/CModelPlotData.h>
#include <model/ImportExport.h>

#include <cstddef>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace ml {
namespace model {
class CEventRateModel;
class CEventRateOnlineModel;
class CEventRatePopulationModel;
class CMetricModel;
class CMetricOnlineModel;
class CMetricPopulationModel;

//! \brief A view into the model details.
//!
//! DESCRIPTION:\n
//! This defines the interface to extract various details of the CAnomalyDetectorModel to
//! avoid cluttering the CAnomalyDetectorModel interface. The intention is to expose all
//! aspects of the mathematical models of both individual and population
//! models for visualization purposes.
class MODEL_EXPORT CModelDetailsView {
public:
    using TTimeTimePr = std::pair<core_t::TTime, core_t::TTime>;
    using TFeatureVec = std::vector<model_t::EFeature>;
    using TStrSet = std::set<std::string>;

public:
    virtual ~CModelDetailsView() = default;

    //! Get the identifier of the person called \p name if they exist.
    bool personId(const std::string& person, std::size_t& result) const;

    //! Get the identifier of the person called \p name if they exist.
    bool categoryId(const std::string& attribute, std::size_t& result) const;

    //! Get the collection of features for which data is being gathered.
    const TFeatureVec& features() const;

    //! Get data for creating a model plot error bar at \p time for the
    //! confidence interval \p boundsPercentile and the by fields identified
    //! by \p terms.
    //!
    //! \note If \p terms is empty all by field error bars are returned.
    void modelPlot(core_t::TTime time,
                   double boundsPercentile,
                   const TStrSet& terms,
                   CModelPlotData& modelPlotData) const;

    //! Get the time interval from the first to last data point of \p byFieldId.
    virtual TTimeTimePr dataTimeInterval(std::size_t byFieldId) const = 0;

    //! Get the time series model for \p feature and \p byFieldId.
    virtual const maths::common::CModel* model(model_t::EFeature feature,
                                               std::size_t byFieldId) const = 0;

private:
    //! Add the model plot data for all by field values which match \p terms.
    void addCurrentBucketValues(core_t::TTime time,
                                model_t::EFeature feature,
                                const TStrSet& terms,
                                CModelPlotData& modelPlotData) const;

    //! Get the model plot data for the specified by field value.
    void modelPlotForByFieldId(core_t::TTime,
                               double boundsPercentile,
                               model_t::EFeature feature,
                               std::size_t byFieldId,
                               CModelPlotData& modelPlotData) const;

    //! Get the underlying model.
    virtual const CAnomalyDetectorModel& base() const = 0;

    //! Get the count variance scale.
    virtual double countVarianceScale(model_t::EFeature feature,
                                      std::size_t byFieldId,
                                      core_t::TTime time) const = 0;

    //! Returns true if the terms are empty or they contain the key.
    static bool contains(const TStrSet& terms, const std::string& key);

    //! Check if the model has a by field.
    bool hasByField() const;
    //! Get the maximum by field identifier.
    std::size_t maxByFieldId() const;
    //! Try to get the by field identifier corresponding to \p byFieldValue.
    bool byFieldId(const std::string& byFieldValue, std::size_t& result) const;
    //! Get the by field value corresponding to \p byFieldId.
    const std::string& byFieldValue(std::size_t byFieldId) const;
    //! Get the by field corresponding to (\p pid, \p cid).
    const std::string& byFieldValue(std::size_t pid, std::size_t cid) const;
    //! Check if the by field identified by \p byFieldId is currently in use.
    bool isByFieldIdActive(std::size_t byFieldId) const;
};

//! \brief A view into the details of a CEventRateModel object.
//!
//! \sa CModelDetailsView.
class MODEL_EXPORT CEventRateModelDetailsView : public CModelDetailsView {
public:
    CEventRateModelDetailsView(const CEventRateModel& model);

    //! Get the time interval from the first to last data point of \p byFieldId.
    TTimeTimePr dataTimeInterval(std::size_t byFieldId) const override;

    //! Get the time series model for \p feature and \p byFieldId.
    const maths::common::CModel* model(model_t::EFeature feature,
                                       std::size_t byFieldId) const override;

private:
    const CAnomalyDetectorModel& base() const override;
    double countVarianceScale(model_t::EFeature feature,
                              std::size_t byFieldId,
                              core_t::TTime time) const override;

private:
    //! The model.
    const CEventRateModel* m_Model;
};

//! \brief A view into the details of a CEventRatePopulationModel object.
//!
//! \sa CModelDetailsView.
class MODEL_EXPORT CEventRatePopulationModelDetailsView : public CModelDetailsView {
public:
    CEventRatePopulationModelDetailsView(const CEventRatePopulationModel& model);

    //! Get the time interval from the first to last data point of \p byFieldId.
    TTimeTimePr dataTimeInterval(std::size_t byFieldId) const override;

    //! Get the time series model for \p feature and \p byFieldId.
    const maths::common::CModel* model(model_t::EFeature feature,
                                       std::size_t byFieldId) const override;

private:
    const CAnomalyDetectorModel& base() const override;
    double countVarianceScale(model_t::EFeature feature,
                              std::size_t byFieldId,
                              core_t::TTime time) const override;

private:
    //! The model.
    const CEventRatePopulationModel* m_Model;
};

//! \brief A view into the details of a CMetricModel object.
//!
//! \sa CModelDetailsView.
class MODEL_EXPORT CMetricModelDetailsView : public CModelDetailsView {
public:
    CMetricModelDetailsView(const CMetricModel& model);

    //! Get the time interval from the first to last data point of \p byFieldId.
    TTimeTimePr dataTimeInterval(std::size_t byFieldId) const override;

    //! Get the time series model for \p feature and \p byFieldId.
    const maths::common::CModel* model(model_t::EFeature feature,
                                       std::size_t byFieldId) const override;

private:
    const CAnomalyDetectorModel& base() const override;
    double countVarianceScale(model_t::EFeature feature,
                              std::size_t byFieldId,
                              core_t::TTime time) const override;

private:
    //! The model.
    const CMetricModel* m_Model;
};

//! \brief A view into the details of a CMetricPopulationModel object.
//!
//! \sa CModelDetailsView.
class MODEL_EXPORT CMetricPopulationModelDetailsView : public CModelDetailsView {
public:
    CMetricPopulationModelDetailsView(const CMetricPopulationModel& model);

    //! Get the time interval from the first to last data point of \p byFieldId.
    TTimeTimePr dataTimeInterval(std::size_t byFieldId) const override;

    //! Get the time series model for \p feature and \p byFieldId.
    const maths::common::CModel* model(model_t::EFeature feature,
                                       std::size_t byFieldId) const override;

private:
    const CAnomalyDetectorModel& base() const override;
    double countVarianceScale(model_t::EFeature feature,
                              std::size_t byFieldId,
                              core_t::TTime time) const override;

private:
    //! The model.
    const CMetricPopulationModel* m_Model;
};
}
}

#endif // INCLUDED_ml_model_CModelDetailsView_h
