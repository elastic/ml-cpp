/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_Mocks_h
#define INCLUDED_ml_model_Mocks_h

#include <core/CTriple.h>

#include <model/CAnomalyDetectorModel.h>
#include <model/CInterimBucketCorrector.h>
#include <model/CModelDetailsView.h>

#include <boost/unordered_map.hpp>

#include <utility>

namespace ml {
namespace model {

//! \brief Mock a model and allow setting of bucket values
//! and baselines.
class CMockModel : public CAnomalyDetectorModel {
public:
    CMockModel(const SModelParams& params,
               const TDataGathererPtr& dataGatherer,
               const TFeatureInfluenceCalculatorCPtrPrVecVec& influenceCalculators);

    void persistModelsState(core::CStatePersistInserter& inserter) const override;

    void acceptPersistInserter(core::CStatePersistInserter& inserter) const override;

    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) override;

    CAnomalyDetectorModel* cloneForPersistence() const override;

    model_t::EModelType category() const override;

    bool isPopulation() const override;

    bool isEventRate() const override;

    bool isMetric() const override;

    TOptionalUInt64 currentBucketCount(std::size_t pid, core_t::TTime time) const override;

    TOptionalDouble baselineBucketCount(std::size_t pid) const override;

    TDouble1Vec currentBucketValue(model_t::EFeature feature,
                                   std::size_t pid,
                                   std::size_t cid,
                                   core_t::TTime time) const override;

    TDouble1Vec baselineBucketMean(model_t::EFeature feature,
                                   std::size_t pid,
                                   std::size_t cid,
                                   model_t::CResultType type,
                                   const TSizeDoublePr1Vec& correlated,
                                   core_t::TTime time) const override;

    bool bucketStatsAvailable(core_t::TTime time) const override;

    void currentBucketPersonIds(core_t::TTime time, TSizeVec& result) const override;

    void sampleBucketStatistics(core_t::TTime startTime,
                                core_t::TTime endTime,
                                CResourceMonitor& resourceMonitor) override;

    void sample(core_t::TTime startTime, core_t::TTime endTime, CResourceMonitor& resourceMonitor) override;

    void prune(std::size_t maximumAge) override;

    bool computeProbability(std::size_t pid,
                            core_t::TTime startTime,
                            core_t::TTime endTime,
                            CPartitioningFields& partitioningFields,
                            std::size_t numberAttributeProbabilities,
                            SAnnotatedProbability& result) const override;

    bool computeTotalProbability(const std::string& person,
                                 std::size_t numberAttributeProbabilities,
                                 TOptionalDouble& probability,
                                 TAttributeProbability1Vec& attributeProbabilities) const override;

    uint64_t checksum(bool includeCurrentBucketStats = true) const override;

    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const override;

    std::size_t memoryUsage() const override;

    std::size_t computeMemoryUsage() const override;

    std::size_t staticSize() const override;

    TModelDetailsViewUPtr details() const override;

    double attributeFrequency(std::size_t cid) const override;

    const maths::CModel* model(std::size_t id) const;

    // Setter methods to allow mocking

    void mockPopulation(bool isPopulation);

    void mockAddBucketValue(model_t::EFeature feature,
                            std::size_t pid,
                            std::size_t cid,
                            core_t::TTime time,
                            const TDouble1Vec& value);

    void mockAddBucketBaselineMean(model_t::EFeature feature,
                                   std::size_t pid,
                                   std::size_t cid,
                                   core_t::TTime time,
                                   const TDouble1Vec& value);

    void mockTimeSeriesModels(TMathsModelUPtrVec&& model);

private:
    using TDouble1Vec = CAnomalyDetectorModel::TDouble1Vec;
    using TSizeSizeTimeTriple = core::CTriple<std::size_t, std::size_t, core_t::TTime>;
    using TFeatureSizeSizeTimeTriplePr = std::pair<model_t::EFeature, TSizeSizeTimeTriple>;
    using TFeatureSizeSizeTimeTriplePrDouble1VecUMap =
        boost::unordered_map<TFeatureSizeSizeTimeTriplePr, TDouble1Vec>;

private:
    core_t::TTime currentBucketStartTime() const override;
    void currentBucketStartTime(core_t::TTime time) override;
    void createNewModels(std::size_t n, std::size_t m) override;
    void updateRecycledModels() override;
    void clearPrunedResources(const TSizeVec& people, const TSizeVec& attributes) override;
    const model::CInterimBucketCorrector& interimValueCorrector() const override;
    void doSkipSampling(core_t::TTime startTime, core_t::TTime endTime) override;
    CMemoryUsageEstimator* memoryUsageEstimator() const override;

private:
    bool m_IsPopulation;
    TFeatureSizeSizeTimeTriplePrDouble1VecUMap m_BucketValues;
    TFeatureSizeSizeTimeTriplePrDouble1VecUMap m_BucketBaselineMeans;
    TMathsModelUPtrVec m_Models;
    model::CInterimBucketCorrector m_InterimBucketCorrector;
};

//! \brief A details view for a mock model.
class CMockModelDetailsView : public CModelDetailsView {
public:
    explicit CMockModelDetailsView(const CMockModel& model);

private:
    const maths::CModel* model(model_t::EFeature feature, std::size_t byFieldId) const override;
    TTimeTimePr dataTimeInterval(std::size_t byFieldId) const override;
    const CAnomalyDetectorModel& base() const override;
    double countVarianceScale(model_t::EFeature feature,
                              std::size_t byFieldId,
                              core_t::TTime time) const override;

private:
    //! The model.
    const CMockModel* m_Model;
};
}
}

#endif // INCLUDED_ml_model_Mocks_h
