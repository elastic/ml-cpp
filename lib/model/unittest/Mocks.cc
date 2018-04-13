/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "Mocks.h"

#include <model/CModelDetailsView.h>

namespace ml {
namespace model {

CMockModel::CMockModel(const SModelParams& params,
                       const TDataGathererPtr& dataGatherer,
                       const TFeatureInfluenceCalculatorCPtrPrVecVec& influenceCalculators)
    : CAnomalyDetectorModel(params, dataGatherer, influenceCalculators),
      m_IsPopulation(false) {
}

void CMockModel::acceptPersistInserter(core::CStatePersistInserter& /*inserter*/) const {
}

bool CMockModel::acceptRestoreTraverser(core::CStateRestoreTraverser& /*traverser*/) {
    return false;
}

CAnomalyDetectorModel* CMockModel::cloneForPersistence() const {
    return nullptr;
}

model_t::EModelType CMockModel::category() const {
    return model_t::E_MetricOnline;
}

bool CMockModel::isPopulation() const {
    return m_IsPopulation;
}

bool CMockModel::isEventRate() const {
    return false;
}

bool CMockModel::isMetric() const {
    return false;
}

CMockModel::TOptionalUInt64
CMockModel::currentBucketCount(std::size_t /*pid*/, core_t::TTime /*time*/) const {
    CAnomalyDetectorModel::TOptionalUInt64 count;
    return count;
}

CMockModel::TOptionalDouble CMockModel::baselineBucketCount(std::size_t /*pid*/) const {
    CAnomalyDetectorModel::TOptionalDouble count;
    return count;
}

CMockModel::TDouble1Vec CMockModel::currentBucketValue(model_t::EFeature feature,
                                                       std::size_t pid,
                                                       std::size_t cid,
                                                       core_t::TTime time) const {
    auto i = m_BucketValues.find({feature, core::make_triple(pid, cid, time)});
    return i != m_BucketValues.end() ? i->second : TDouble1Vec();
}

CMockModel::TDouble1Vec CMockModel::baselineBucketMean(model_t::EFeature feature,
                                                       std::size_t pid,
                                                       std::size_t cid,
                                                       model_t::CResultType /*type*/,
                                                       const TSizeDoublePr1Vec& /*correlated*/,
                                                       core_t::TTime time) const {
    auto i = m_BucketBaselineMeans.find({feature, core::make_triple(pid, cid, time)});
    return i != m_BucketBaselineMeans.end() ? i->second : TDouble1Vec();
}

bool CMockModel::bucketStatsAvailable(core_t::TTime /*time*/) const {
    return false;
}

void CMockModel::currentBucketPersonIds(core_t::TTime /*time*/, TSizeVec& /*result*/) const {
}

void CMockModel::sampleBucketStatistics(core_t::TTime /*startTime*/,
                                        core_t::TTime /*endTime*/,
                                        CResourceMonitor& /*resourceMonitor*/) {
}

void CMockModel::sample(core_t::TTime /*startTime*/,
                        core_t::TTime /*endTime*/,
                        CResourceMonitor& /*resourceMonitor*/) {
}

void CMockModel::sampleOutOfPhase(core_t::TTime /*startTime*/,
                                  core_t::TTime /*endTime*/,
                                  CResourceMonitor& /*resourceMonitor*/) {
}

void CMockModel::prune(std::size_t /*maximumAge*/) {
}

bool CMockModel::computeProbability(std::size_t /*pid*/,
                                    core_t::TTime /*startTime*/,
                                    core_t::TTime /*endTime*/,
                                    CPartitioningFields& /*partitioningFields*/,
                                    std::size_t /*numberAttributeProbabilities*/,
                                    SAnnotatedProbability& /*result*/) const {
    return false;
}

bool CMockModel::computeTotalProbability(const std::string& /*person*/,
                                         std::size_t /*numberAttributeProbabilities*/,
                                         TOptionalDouble& /*probability*/,
                                         TAttributeProbability1Vec& /*attributeProbabilities*/) const {
    return false;
}

uint64_t CMockModel::checksum(bool /*includeCurrentBucketStats*/) const {
    return 0;
}

void CMockModel::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr /*mem*/) const {
}

std::size_t CMockModel::memoryUsage() const {
    return 0;
}

std::size_t CMockModel::computeMemoryUsage() const {
    return 0;
}

std::size_t CMockModel::staticSize() const {
    return 0;
}

CMockModel::CModelDetailsViewPtr CMockModel::details() const {
    CModelDetailsViewPtr result{new CMockModelDetailsView(*this)};
    return result;
}

double CMockModel::attributeFrequency(std::size_t /*cid*/) const {
    return 0.0;
}

core_t::TTime CMockModel::currentBucketStartTime() const {
    return 0;
}

void CMockModel::currentBucketStartTime(core_t::TTime /*time*/) {
}

void CMockModel::createNewModels(std::size_t /*n*/, std::size_t /*m*/) {
}

void CMockModel::updateRecycledModels() {
}

void CMockModel::clearPrunedResources(const TSizeVec& /*people*/, const TSizeVec& /*attributes*/) {
}

void CMockModel::currentBucketTotalCount(uint64_t /*totalCount*/) {
}

void CMockModel::doSkipSampling(core_t::TTime /*startTime*/, core_t::TTime /*endTime*/) {
}

const maths::CModel* CMockModel::model(std::size_t id) const {
    return m_Models[id].get();
}

void CMockModel::mockPopulation(bool isPopulation) {
    m_IsPopulation = isPopulation;
}

void CMockModel::mockAddBucketValue(model_t::EFeature feature,
                                    std::size_t pid,
                                    std::size_t cid,
                                    core_t::TTime time,
                                    const TDouble1Vec& value) {
    m_BucketValues[{feature, core::make_triple(pid, cid, time)}] = value;
}

void CMockModel::mockAddBucketBaselineMean(model_t::EFeature feature,
                                           std::size_t pid,
                                           std::size_t cid,
                                           core_t::TTime time,
                                           const TDouble1Vec& value) {
    m_BucketBaselineMeans[{feature, core::make_triple(pid, cid, time)}] = value;
}

void CMockModel::mockTimeSeriesModels(const TMathsModelPtrVec& models) {
    m_Models = models;
}

CMemoryUsageEstimator* CMockModel::memoryUsageEstimator() const {
    return nullptr;
}

CMockModelDetailsView::CMockModelDetailsView(const CMockModel& model)
    : m_Model{&model} {
}

const maths::CModel* CMockModelDetailsView::model(model_t::EFeature /*feature*/,
                                                  std::size_t byFieldId) const {
    return m_Model->model(byFieldId);
}

const CAnomalyDetectorModel& CMockModelDetailsView::base() const {
    return *m_Model;
}

double CMockModelDetailsView::countVarianceScale(model_t::EFeature /*feature*/,
                                                 std::size_t /*byFieldId*/,
                                                 core_t::TTime /*time*/) const {
    return 1.0;
}
}
}
