/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_Mocks_h
#define INCLUDED_ml_model_Mocks_h

#include <core/CTriple.h>

#include <model/CAnomalyDetectorModel.h>
#include <model/CModelDetailsView.h>

#include <boost/unordered_map.hpp>

#include <utility>

namespace ml
{
namespace model
{

//! \brief Mock a model and allow setting of bucket values
//! and baselines.
class CMockModel : public CAnomalyDetectorModel
{
    public:
        CMockModel(const SModelParams &params,
                   const TDataGathererPtr &dataGatherer,
                   const TFeatureInfluenceCalculatorCPtrPrVecVec &influenceCalculators);

        virtual void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

        virtual bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

        virtual CAnomalyDetectorModel *cloneForPersistence(void) const;

        virtual model_t::EModelType category(void) const;

        virtual bool isPopulation(void) const;

        virtual bool isEventRate(void) const;

        virtual bool isMetric(void) const;

        virtual TOptionalUInt64 currentBucketCount(std::size_t pid,
                                                   core_t::TTime time) const;

        virtual TOptionalDouble baselineBucketCount(std::size_t pid) const;

        virtual TDouble1Vec currentBucketValue(model_t::EFeature feature,
                                               std::size_t pid,
                                               std::size_t cid,
                                               core_t::TTime time) const;

        virtual TDouble1Vec baselineBucketMean(model_t::EFeature feature,
                                               std::size_t pid,
                                               std::size_t cid,
                                               model_t::CResultType type,
                                               const TSizeDoublePr1Vec &correlated,
                                               core_t::TTime time) const;

        virtual bool bucketStatsAvailable(core_t::TTime time) const;

        virtual void currentBucketPersonIds(core_t::TTime time, TSizeVec &result) const;

        virtual void sampleBucketStatistics(core_t::TTime startTime,
                                            core_t::TTime endTime,
                                            CResourceMonitor &resourceMonitor);

        virtual void sample(core_t::TTime startTime,
                            core_t::TTime endTime,
                            CResourceMonitor &resourceMonitor);

        virtual void sampleOutOfPhase(core_t::TTime startTime,
                                      core_t::TTime endTime,
                                      CResourceMonitor &resourceMonitor);

        virtual void prune(std::size_t maximumAge);

        virtual bool computeProbability(std::size_t pid,
                                        core_t::TTime startTime,
                                        core_t::TTime endTime,
                                        CPartitioningFields &partitioningFields,
                                        std::size_t numberAttributeProbabilities,
                                        SAnnotatedProbability &result) const;

        virtual bool computeTotalProbability(const std::string &person,
                                             std::size_t numberAttributeProbabilities,
                                             TOptionalDouble &probability,
                                             TAttributeProbability1Vec &attributeProbabilities) const;

        virtual uint64_t checksum(bool includeCurrentBucketStats = true) const;

        virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        virtual std::size_t memoryUsage(void) const;

        virtual std::size_t computeMemoryUsage(void) const;

        virtual std::size_t staticSize(void) const;

        virtual CModelDetailsViewPtr details(void) const;

        virtual double attributeFrequency(std::size_t cid) const;

        const maths::CModel *model(std::size_t id) const;

        // Setter methods to allow mocking

        void mockPopulation(bool isPopulation);

        void mockAddBucketValue(model_t::EFeature feature,
                                std::size_t pid,
                                std::size_t cid,
                                core_t::TTime time,
                                const TDouble1Vec &value);

        void mockAddBucketBaselineMean(model_t::EFeature feature,
                                       std::size_t pid,
                                       std::size_t cid,
                                       core_t::TTime time,
                                       const TDouble1Vec &value);

        void mockTimeSeriesModels(const TMathsModelPtrVec &model);

    protected:
        virtual core_t::TTime currentBucketStartTime(void) const;
        virtual void currentBucketStartTime(core_t::TTime time);
        virtual void createNewModels(std::size_t n, std::size_t m);
        virtual void updateRecycledModels(void);
        virtual void clearPrunedResources(const TSizeVec &people,
                                          const TSizeVec &attributes);

    private:
        using TDouble1Vec = CAnomalyDetectorModel::TDouble1Vec;
        using TSizeSizeTimeTriple = core::CTriple<std::size_t, std::size_t, core_t::TTime>;
        using TFeatureSizeSizeTimeTriplePr = std::pair<model_t::EFeature, TSizeSizeTimeTriple>;
        using TFeatureSizeSizeTimeTriplePrDouble1VecUMap = boost::unordered_map<TFeatureSizeSizeTimeTriplePr, TDouble1Vec>;

    private:
        virtual void currentBucketTotalCount(uint64_t totalCount);
        virtual void doSkipSampling(core_t::TTime startTime, core_t::TTime endTime);
        virtual CMemoryUsageEstimator *memoryUsageEstimator(void) const;

    private:
        bool m_IsPopulation;
        TFeatureSizeSizeTimeTriplePrDouble1VecUMap m_BucketValues;
        TFeatureSizeSizeTimeTriplePrDouble1VecUMap m_BucketBaselineMeans;
        TMathsModelPtrVec m_Models;
};

//! \brief A details view for a mock model.
class CMockModelDetailsView : public CModelDetailsView
{
    public:
        CMockModelDetailsView(const CMockModel &model);

    private:
        virtual const maths::CModel *model(model_t::EFeature feature,
                                           std::size_t byFieldId) const;
        virtual const CAnomalyDetectorModel &base() const;
        virtual double countVarianceScale(model_t::EFeature feature,
                                          std::size_t byFieldId,
                                          core_t::TTime time) const;

    private:
        //! The model.
        const CMockModel *m_Model;
};

}
}

#endif // INCLUDED_ml_model_Mocks_h
