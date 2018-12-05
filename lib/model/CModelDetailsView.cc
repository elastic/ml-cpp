/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CModelDetailsView.h>

#include <core/CSmallVector.h>

#include <maths/CBasicStatistics.h>
#include <maths/CTimeSeriesDecomposition.h>

#include <model/CDataGatherer.h>
#include <model/CEventRateModel.h>
#include <model/CEventRatePopulationModel.h>
#include <model/CMetricModel.h>
#include <model/CMetricPopulationModel.h>

namespace ml {
namespace model {
namespace {
const std::string EMPTY_STRING("");
}

using TDouble1Vec = core::CSmallVector<double, 1>;
using TOptionalUInt64 = CAnomalyDetectorModel::TOptionalUInt64;
using TDoubleDoublePr = std::pair<double, double>;

////////// CModelDetailsView Implementation //////////

bool CModelDetailsView::personId(const std::string& name, std::size_t& result) const {
    return this->base().dataGatherer().personId(name, result);
}

bool CModelDetailsView::categoryId(const std::string& attribute, std::size_t& result) const {
    return this->base().dataGatherer().attributeId(attribute, result);
}

const CModelDetailsView::TFeatureVec& CModelDetailsView::features() const {
    return this->base().dataGatherer().features();
}

void CModelDetailsView::modelPlot(core_t::TTime time,
                                  double boundsPercentile,
                                  const TStrSet& terms,
                                  CModelPlotData& modelPlotData) const {
    for (auto feature : this->features()) {
        if (!model_t::isConstant(feature) && !model_t::isCategorical(feature)) {
            if (terms.empty() || !this->hasByField()) {
                for (std::size_t byFieldId = 0; byFieldId < this->maxByFieldId(); ++byFieldId) {
                    this->modelPlotForByFieldId(time, boundsPercentile, feature,
                                                byFieldId, modelPlotData);
                }
            } else {
                for (const auto& term : terms) {
                    std::size_t byFieldId(0);
                    if (this->byFieldId(term, byFieldId)) {
                        this->modelPlotForByFieldId(time, boundsPercentile, feature,
                                                    byFieldId, modelPlotData);
                    }
                }
            }
            this->addCurrentBucketValues(time, feature, terms, modelPlotData);
        }
    }
}

void CModelDetailsView::modelPlotForByFieldId(core_t::TTime time,
                                              double boundsPercentile,
                                              model_t::EFeature feature,
                                              std::size_t byFieldId,
                                              CModelPlotData& modelPlotData) const {
    using TDouble1VecDouble1VecPr = std::pair<TDouble1Vec, TDouble1Vec>;
    using TDouble2Vec = core::CSmallVector<double, 2>;
    using TDouble2Vec3Vec = core::CSmallVector<TDouble2Vec, 3>;

    if (this->isByFieldIdActive(byFieldId)) {
        const maths::CModel* model = this->model(feature, byFieldId);
        if (model == nullptr) {
            return;
        }

        std::size_t dimension = model_t::dimension(feature);
        time = model_t::sampleTime(feature, time, model->params().bucketLength());

        maths_t::TDouble2VecWeightsAry weights(
            maths_t::CUnitWeights::unit<TDouble2Vec>(dimension));
        maths_t::setSeasonalVarianceScale(
            model->seasonalWeight(maths::DEFAULT_SEASONAL_CONFIDENCE_INTERVAL, time), weights);
        maths_t::setCountVarianceScale(
            TDouble2Vec(dimension, this->countVarianceScale(feature, byFieldId, time)), weights);

        TDouble1VecDouble1VecPr support(model_t::support(feature));
        TDouble2Vec supportLower(support.first);
        TDouble2Vec supportUpper(support.second);

        TDouble2Vec3Vec interval(model->confidenceInterval(time, boundsPercentile, weights));

        if (interval.size() == 3) {
            TDouble2Vec lower = maths::CTools::truncate(interval[0], supportLower, supportUpper);
            TDouble2Vec upper = maths::CTools::truncate(interval[2], lower, supportUpper);
            TDouble2Vec median = maths::CTools::truncate(interval[1], lower, upper);

            // TODO This data structure should support multivariate features.
            modelPlotData.get(feature, this->byFieldValue(byFieldId)) =
                CModelPlotData::SByFieldData(lower[0], upper[0], median[0]);
        }
    }
}

void CModelDetailsView::addCurrentBucketValues(core_t::TTime time,
                                               model_t::EFeature feature,
                                               const TStrSet& terms,
                                               CModelPlotData& modelPlotData) const {
    const CDataGatherer& gatherer = this->base().dataGatherer();
    if (!gatherer.dataAvailable(time)) {
        return;
    }

    bool isPopulation{gatherer.isPopulation()};

    auto addCurrentBucketValue = [&](std::size_t pid, std::size_t cid) {
        const std::string& byFieldValue{this->byFieldValue(pid, cid)};
        if (this->contains(terms, byFieldValue)) {
            TDouble1Vec value(this->base().currentBucketValue(feature, pid, cid, time));
            if (!value.empty()) {
                const std::string& overFieldValue{
                    isPopulation ? this->base().personName(pid) : EMPTY_STRING};
                modelPlotData.get(feature, byFieldValue).addValue(overFieldValue, value[0]);
            }
        }
    };

    if (model_t::countsEmptyBuckets(feature)) {
        for (std::size_t pid = 0u; pid < gatherer.numberPeople(); ++pid) {
            if (gatherer.isPersonActive(pid)) {
                if (isPopulation) {
                    for (std::size_t cid = 0u; cid < gatherer.numberAttributes(); ++cid) {
                        if (gatherer.isAttributeActive(cid)) {
                            addCurrentBucketValue(pid, cid);
                        }
                    }
                } else {
                    addCurrentBucketValue(pid, model_t::INDIVIDUAL_ANALYSIS_ATTRIBUTE_ID);
                }
            }
        }
    } else {
        for (const auto& count : gatherer.bucketCounts(time)) {
            std::size_t pid{gatherer.extractPersonId(count)};
            std::size_t cid{gatherer.extractAttributeId(count)};
            addCurrentBucketValue(pid, cid);
        }
    }
}

bool CModelDetailsView::contains(const TStrSet& terms, const std::string& key) {
    return terms.empty() || key.empty() || terms.find(key) != terms.end();
}

bool CModelDetailsView::hasByField() const {
    const std::string& byField = this->base().isPopulation()
                                     ? this->base().dataGatherer().attributeFieldName()
                                     : this->base().dataGatherer().personFieldName();
    return byField.empty() == false;
}

std::size_t CModelDetailsView::maxByFieldId() const {
    return this->base().isPopulation()
               ? this->base().dataGatherer().numberAttributes()
               : this->base().dataGatherer().numberPeople();
}

bool CModelDetailsView::byFieldId(const std::string& byFieldValue, std::size_t& result) const {
    return this->base().isPopulation()
               ? this->base().dataGatherer().attributeId(byFieldValue, result)
               : this->base().dataGatherer().personId(byFieldValue, result);
}

const std::string& CModelDetailsView::byFieldValue(std::size_t byFieldId) const {
    return this->base().isPopulation() ? this->base().attributeName(byFieldId)
                                       : this->base().personName(byFieldId);
}

const std::string& CModelDetailsView::byFieldValue(std::size_t pid, std::size_t cid) const {
    return this->base().isPopulation() ? this->base().attributeName(cid)
                                       : this->base().personName(pid);
}

bool CModelDetailsView::isByFieldIdActive(std::size_t byFieldId) const {
    return this->base().isPopulation()
               ? this->base().dataGatherer().isAttributeActive(byFieldId)
               : this->base().dataGatherer().isPersonActive(byFieldId);
}

////////// CEventRateModelDetailsView Implementation //////////

CEventRateModelDetailsView::CEventRateModelDetailsView(const CEventRateModel& model)
    : m_Model(&model) {
}

const maths::CModel* CEventRateModelDetailsView::model(model_t::EFeature feature,
                                                       std::size_t byFieldId) const {
    return m_Model->model(feature, byFieldId);
}

const CAnomalyDetectorModel& CEventRateModelDetailsView::base() const {
    return *m_Model;
}

double CEventRateModelDetailsView::countVarianceScale(model_t::EFeature /*feature*/,
                                                      std::size_t /*byFieldId*/,
                                                      core_t::TTime /*time*/) const {
    return 1.0;
}

////////// CEventRatePopulationModelDetailsView Implementation //////////

CEventRatePopulationModelDetailsView::CEventRatePopulationModelDetailsView(const CEventRatePopulationModel& model)
    : m_Model(&model) {
}

const maths::CModel* CEventRatePopulationModelDetailsView::model(model_t::EFeature feature,
                                                                 std::size_t byFieldId) const {
    return m_Model->model(feature, byFieldId);
}

const CAnomalyDetectorModel& CEventRatePopulationModelDetailsView::base() const {
    return *m_Model;
}

double CEventRatePopulationModelDetailsView::countVarianceScale(model_t::EFeature /*feature*/,
                                                                std::size_t /*byFieldId*/,
                                                                core_t::TTime /*time*/) const {
    return 1.0;
}

////////// CMetricModelDetailsView Implementation //////////

CMetricModelDetailsView::CMetricModelDetailsView(const CMetricModel& model)
    : m_Model(&model) {
}

const maths::CModel* CMetricModelDetailsView::model(model_t::EFeature feature,
                                                    std::size_t byFieldId) const {
    return m_Model->model(feature, byFieldId);
}

const CAnomalyDetectorModel& CMetricModelDetailsView::base() const {
    return *m_Model;
}

double CMetricModelDetailsView::countVarianceScale(model_t::EFeature feature,
                                                   std::size_t byFieldId,
                                                   core_t::TTime time) const {
    TOptionalUInt64 count = m_Model->currentBucketCount(byFieldId, time);
    if (!count) {
        return 1.0;
    }
    return model_t::varianceScale(feature,
                                  m_Model->dataGatherer().effectiveSampleCount(byFieldId),
                                  static_cast<double>(*count));
}

////////// CMetricPopulationModelDetailsView Implementation //////////

CMetricPopulationModelDetailsView::CMetricPopulationModelDetailsView(const CMetricPopulationModel& model)
    : m_Model(&model) {
}

const maths::CModel* CMetricPopulationModelDetailsView::model(model_t::EFeature feature,
                                                              std::size_t byFieldId) const {
    return m_Model->model(feature, byFieldId);
}

const CAnomalyDetectorModel& CMetricPopulationModelDetailsView::base() const {
    return *m_Model;
}

double CMetricPopulationModelDetailsView::countVarianceScale(model_t::EFeature feature,
                                                             std::size_t byFieldId,
                                                             core_t::TTime time) const {
    TOptionalUInt64 count = m_Model->currentBucketCount(byFieldId, time);
    if (!count) {
        return 1.0;
    }
    return model_t::varianceScale(feature,
                                  m_Model->dataGatherer().effectiveSampleCount(byFieldId),
                                  static_cast<double>(*count));
}
}
}
