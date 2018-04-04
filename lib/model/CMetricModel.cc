/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CMetricModel.h>

#include <core/CContainerPrinter.h>
#include <core/CFunctional.h>
#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CStatistics.h>
#include <core/CoreTypes.h>

#include <maths/CChecksum.h>
#include <maths/CMultivariatePrior.h>
#include <maths/COrderings.h>
#include <maths/CPrior.h>
#include <maths/CTools.h>
#include <maths/ProbabilityAggregators.h>

#include <model/CAnnotatedProbabilityBuilder.h>
#include <model/CDataGatherer.h>
#include <model/CGathererTools.h>
#include <model/CIndividualModelDetail.h>
#include <model/CInterimBucketCorrector.h>
#include <model/CModelDetailsView.h>
#include <model/CModelTools.h>
#include <model/CProbabilityAndInfluenceCalculator.h>
#include <model/CSampleGatherer.h>
#include <model/CResourceMonitor.h>
#include <model/FrequencyPredicates.h>

#include <boost/bind.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/ref.hpp>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace ml
{
namespace model
{

namespace
{

using TTime1Vec = core::CSmallVector<core_t::TTime, 1>;
using TDouble1Vec = core::CSmallVector<double, 1>;
using TDouble2Vec = core::CSmallVector<double, 2>;
using TDouble4Vec = core::CSmallVector<double, 4>;
using TDouble2Vec1Vec = core::CSmallVector<TDouble2Vec, 1>;
using TDouble2Vec4Vec = core::CSmallVector<TDouble2Vec, 4>;
using TDouble4Vec1Vec = core::CSmallVector<TDouble4Vec, 1>;
using TBool2Vec = core::CSmallVector<bool, 2>;

// We use short field names to reduce the state size
const std::string INDIVIDUAL_STATE_TAG("a");

const maths_t::TWeightStyleVec SAMPLE_WEIGHT_STYLES{maths_t::E_SampleCountWeight,
                                                    maths_t::E_SampleWinsorisationWeight,
                                                    maths_t::E_SampleCountVarianceScaleWeight};
const maths_t::TWeightStyleVec PROBABILITY_WEIGHT_STYLES{maths_t::E_SampleSeasonalVarianceScaleWeight,
                                                         maths_t::E_SampleCountVarianceScaleWeight};

}

CMetricModel::CMetricModel(const SModelParams &params,
                           const TDataGathererPtr &dataGatherer,
                           const TFeatureMathsModelPtrPrVec &newFeatureModels,
                           const TFeatureMultivariatePriorPtrPrVec &newFeatureCorrelateModelPriors,
                           const TFeatureCorrelationsPtrPrVec &featureCorrelatesModels,
                           const TFeatureInfluenceCalculatorCPtrPrVecVec &influenceCalculators) :
        CIndividualModel(params, dataGatherer,
                         newFeatureModels,
                         newFeatureCorrelateModelPriors,
                         featureCorrelatesModels,
                         influenceCalculators),
        m_CurrentBucketStats(CAnomalyDetectorModel::TIME_UNSET)
{}

CMetricModel::CMetricModel(const SModelParams &params,
                           const TDataGathererPtr &dataGatherer,
                           const TFeatureMathsModelPtrPrVec &newFeatureModels,
                           const TFeatureMultivariatePriorPtrPrVec &newFeatureCorrelateModelPriors,
                           const TFeatureCorrelationsPtrPrVec &featureCorrelatesModels,
                           const TFeatureInfluenceCalculatorCPtrPrVecVec &influenceCalculators,
                           core::CStateRestoreTraverser &traverser) :
        CIndividualModel(params, dataGatherer,
                         newFeatureModels,
                         newFeatureCorrelateModelPriors,
                         featureCorrelatesModels,
                         influenceCalculators),
        m_CurrentBucketStats(CAnomalyDetectorModel::TIME_UNSET)
{
    traverser.traverseSubLevel(boost::bind(&CMetricModel::acceptRestoreTraverser,
                                           this, _1));
}

CMetricModel::CMetricModel(bool isForPersistence, const CMetricModel &other) :
        CIndividualModel(isForPersistence, other),
        m_CurrentBucketStats(0) // Not needed for persistence so minimally constructed
{
    if (!isForPersistence)
    {
        LOG_ABORT("This constructor only creates clones for persistence");
    }
}

void CMetricModel::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertLevel(INDIVIDUAL_STATE_TAG, boost::bind(&CMetricModel::doAcceptPersistInserter, this, _1));
}

bool CMetricModel::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser)
{
    do
    {
        const std::string &name = traverser.name();
        if (name == INDIVIDUAL_STATE_TAG)
        {
            if (traverser.traverseSubLevel(boost::bind(&CMetricModel::doAcceptRestoreTraverser,
                                                       this,
                                                       _1)) == false)
            {
                // Logging handled already.
                return false;
            }
        }
    }
    while (traverser.next());

    return true;
}

CAnomalyDetectorModel *CMetricModel::cloneForPersistence() const
{
    return new CMetricModel(true, *this);
}

model_t::EModelType CMetricModel::category() const
{
    return model_t::E_MetricOnline;
}

bool CMetricModel::isEventRate() const
{
    return false;
}

bool CMetricModel::isMetric() const
{
    return true;
}

void CMetricModel::currentBucketPersonIds(core_t::TTime time, TSizeVec &result) const
{
    this->CIndividualModel::currentBucketPersonIds(time, m_CurrentBucketStats.s_FeatureData, result);
}

CMetricModel::TOptionalDouble
    CMetricModel::baselineBucketCount(const std::size_t /*pid*/) const
{
    return TOptionalDouble();
}

CMetricModel::TDouble1Vec CMetricModel::currentBucketValue(model_t::EFeature feature,
                                                           std::size_t pid,
                                                           std::size_t /*cid*/,
                                                           core_t::TTime time) const
{
    const TFeatureData *data = this->featureData(feature, pid, time);
    if (data)
    {
        const TOptionalSample &value = data->s_BucketValue;
        return value ? value->value(model_t::dimension(feature)) : TDouble1Vec();
    }
    return TDouble1Vec();
}

CMetricModel::TDouble1Vec CMetricModel::baselineBucketMean(model_t::EFeature feature,
                                                           std::size_t pid,
                                                           std::size_t /*cid*/,
                                                           model_t::CResultType type,
                                                           const TSizeDoublePr1Vec &correlated,
                                                           core_t::TTime time) const
{
    const maths::CModel *model{this->model(feature, pid)};
    if (!model)
    {
        return TDouble1Vec();
    }
    static const TSizeDoublePr1Vec NO_CORRELATED;
    TDouble1Vec result(model->predict(time, type.isUnconditional() ? NO_CORRELATED : correlated));
    this->correctBaselineForInterim(feature, pid, type, correlated,
                                    this->currentBucketInterimCorrections(), result);
    TDouble1VecDouble1VecPr support = model_t::support(feature);
    return maths::CTools::truncate(result, support.first, support.second);
}

void CMetricModel::sampleBucketStatistics(core_t::TTime startTime,
                                          core_t::TTime endTime,
                                          CResourceMonitor &resourceMonitor)
{
    this->createUpdateNewModels(startTime, resourceMonitor);
    m_CurrentBucketStats.s_InterimCorrections.clear();
    this->CIndividualModel::sampleBucketStatistics(startTime, endTime,
                                                   this->personFilter(),
                                                   m_CurrentBucketStats.s_FeatureData,
                                                   resourceMonitor);
}

void CMetricModel::sample(core_t::TTime startTime,
                          core_t::TTime endTime,
                          CResourceMonitor &resourceMonitor)
{
    CDataGatherer &gatherer = this->dataGatherer();
    core_t::TTime bucketLength = gatherer.bucketLength();

    if (!gatherer.validateSampleTimes(startTime, endTime))
    {
        return;
    }

    this->createUpdateNewModels(startTime, resourceMonitor);
    m_CurrentBucketStats.s_InterimCorrections.clear();

    for (core_t::TTime time = startTime; time < endTime; time += bucketLength)
    {
        LOG_TRACE("Sampling [" << time << "," << time + bucketLength << ")");

        gatherer.sampleNow(time);
        gatherer.featureData(time, bucketLength, m_CurrentBucketStats.s_FeatureData);

        const CIndividualModel::TTimeVec &preSampleLastBucketTimes = this->lastBucketTimes();
        CIndividualModel::TSizeTimeUMap lastBucketTimesMap;
        for (const auto &featureData : m_CurrentBucketStats.s_FeatureData)
        {
            for (const auto &data : featureData.second)
            {
                std::size_t pid = data.first;
                lastBucketTimesMap[pid] = preSampleLastBucketTimes[pid];
            }
        }

        this->CIndividualModel::sample(time, time + bucketLength, resourceMonitor);

        // Declared outside the loop to minimize the number of times they are created.
        maths::CModel::TTimeDouble2VecSizeTrVec values;
        maths::CModelAddSamplesParams::TDouble2Vec4VecVec trendWeights;
        maths::CModelAddSamplesParams::TDouble2Vec4VecVec priorWeights;

        for (auto &featureData : m_CurrentBucketStats.s_FeatureData)
        {
            model_t::EFeature feature = featureData.first;
            TSizeFeatureDataPrVec &data = featureData.second;
            std::size_t dimension = model_t::dimension(feature);
            LOG_TRACE(model_t::print(feature) << " data = " << core::CContainerPrinter::print(data));
            this->applyFilter(model_t::E_XF_By, true, this->personFilter(), data);

            for (const auto &data_ : data)
            {
                std::size_t pid = data_.first;
                const CGathererTools::TSampleVec &samples = data_.second.s_Samples;

                maths::CModel *model = this->model(feature, pid);
                if (!model)
                {
                    LOG_ERROR("Missing model for " << this->personName(pid));
                    continue;
                }

                core_t::TTime sampleTime = model_t::sampleTime(feature, time, bucketLength);
                if (this->shouldIgnoreSample(feature, pid, model_t::INDIVIDUAL_ANALYSIS_ATTRIBUTE_ID, sampleTime))
                {
                    model->skipTime(time - lastBucketTimesMap[pid]);
                    continue;
                }

                const TOptionalSample &bucket = data_.second.s_BucketValue;
                if (model_t::isSampled(feature) && bucket)
                {
                    values.assign(1, core::make_triple(bucket->time(),
                                                       TDouble2Vec(bucket->value(dimension)),
                                                       model_t::INDIVIDUAL_ANALYSIS_ATTRIBUTE_ID));
                    model->addBucketValue(values);
                }

                double emptyBucketWeight = this->emptyBucketWeight(feature, pid, time);
                if (emptyBucketWeight == 0.0)
                {
                    continue;
                }

                double derate = this->derate(pid, sampleTime);
                double interval = (1.0 + (this->params().s_InitialDecayRateMultiplier - 1.0) * derate) * emptyBucketWeight;
                double count = this->params().s_MaximumUpdatesPerBucket > 0.0 && samples.size() > 0 ?
                               this->params().s_MaximumUpdatesPerBucket / static_cast<double>(samples.size()) : 1.0;

                LOG_TRACE("Bucket = " << gatherer.printCurrentBucket(time)
                          << ", feature = " << model_t::print(feature)
                          << ", samples = " << core::CContainerPrinter::print(samples)
                          << ", isInteger = " << data_.second.s_IsInteger
                          << ", person = " << this->personName(pid)
                          << ", count weight = " << count
                          << ", dimension = " << dimension
                          << ", empty bucket weight = " << emptyBucketWeight);

                model->params().probabilityBucketEmpty(this->probabilityBucketEmpty(feature, pid));

                values.resize(samples.size());
                trendWeights.resize(samples.size(), TDouble2Vec4Vec(3));
                priorWeights.resize(samples.size(), TDouble2Vec4Vec(3));
                for (std::size_t i = 0u; i < samples.size(); ++i)
                {
                    core_t::TTime ti = samples[i].time();
                    TDouble2Vec vi(samples[i].value(dimension));
                    double vs = samples[i].varianceScale();
                    values[i] = core::make_triple(model_t::sampleTime(feature, time, bucketLength, ti),
                                                  vi, model_t::INDIVIDUAL_ANALYSIS_ATTRIBUTE_ID);
                    trendWeights[i][0].assign(dimension, emptyBucketWeight * count * this->learnRate(feature) / vs);
                    trendWeights[i][1] = model->winsorisationWeight(derate, ti, vi);
                    trendWeights[i][2].assign(dimension, vs);
                    priorWeights[i][0].assign(dimension, emptyBucketWeight * count * this->learnRate(feature));
                    priorWeights[i][1] = trendWeights[i][1];
                    priorWeights[i][2].assign(dimension, vs);
                }

                maths::CModelAddSamplesParams params;
                params.integer(data_.second.s_IsInteger)
                      .nonNegative(data_.second.s_IsNonNegative)
                      .propagationInterval(interval)
                      .weightStyles(SAMPLE_WEIGHT_STYLES)
                      .trendWeights(trendWeights)
                      .priorWeights(priorWeights);

                if (model->addSamples(params, values) == maths::CModel::E_Reset)
                {
                    gatherer.resetSampleCount(pid);
                }
            }
        }

        this->sampleCorrelateModels(SAMPLE_WEIGHT_STYLES);
    }
}

bool CMetricModel::computeProbability(const std::size_t pid,
                                      core_t::TTime startTime,
                                      core_t::TTime endTime,
                                      CPartitioningFields &partitioningFields,
                                      const std::size_t /*numberAttributeProbabilities*/,
                                      SAnnotatedProbability &result) const
{
    CAnnotatedProbabilityBuilder resultBuilder(result);

    const CDataGatherer &gatherer = this->dataGatherer();
    core_t::TTime bucketLength = gatherer.bucketLength();

    if (endTime != startTime + bucketLength)
    {
        LOG_ERROR("Can only compute probability for single bucket");
        return false;
    }

    if (pid >= this->firstBucketTimes().size())
    {
        // This is not necessarily an error: the person might have been added
        // only in an out of phase bucket so far
        LOG_TRACE("No first time for person = " << gatherer.personName(pid));
        return false;
    }

    CProbabilityAndInfluenceCalculator pJoint(this->params().s_InfluenceCutoff);
    pJoint.addAggregator(maths::CJointProbabilityOfLessLikelySamples());
    pJoint.addAggregator(maths::CProbabilityOfExtremeSample());

    for (std::size_t i = 0u, n = gatherer.numberFeatures(); i < n; ++i)
    {
        model_t::EFeature feature = gatherer.feature(i);
        if (model_t::isCategorical(feature))
        {
            continue;
        }
        const TFeatureData *data = this->featureData(feature, pid, startTime);
        if (!data || !data->s_BucketValue)
        {
            continue;
        }
        const TOptionalSample &bucket = data->s_BucketValue;
        if (this->shouldIgnoreResult(feature,
                                     result.s_ResultType,
                                     pid,
                                     model_t::INDIVIDUAL_ANALYSIS_ATTRIBUTE_ID,
                                     model_t::sampleTime(feature, startTime, bucketLength, bucket->time())))
        {
            continue;
        }

        LOG_TRACE("Compute probability for " << data->print());

        if (this->correlates(feature, pid, startTime))
        {
            CProbabilityAndInfluenceCalculator::SCorrelateParams params(partitioningFields);
            TStrCRefDouble1VecDouble1VecPrPrVecVecVec influenceValues;
            this->fill(feature, pid, startTime, result.isInterim(), params, influenceValues);
            this->addProbabilityAndInfluences(pid, params, influenceValues, pJoint, resultBuilder);
        }
        else
        {
            CProbabilityAndInfluenceCalculator::SParams params(partitioningFields);
            this->fill(feature, pid, startTime, result.isInterim(), params);
            this->addProbabilityAndInfluences(pid, params, data->s_InfluenceValues, pJoint, resultBuilder);
        }
    }

    if (pJoint.empty())
    {
        LOG_TRACE("No samples in [" << startTime << "," << endTime << ")");
        return false;
    }

    double p;
    if (!pJoint.calculate(p, result.s_Influences))
    {
        LOG_ERROR("Failed to compute probability");
        return false;
    }
    LOG_TRACE("probability(" << this->personName(pid) << ") = " << p);

    resultBuilder.probability(p);
    resultBuilder.build();

    return true;
}

uint64_t CMetricModel::checksum(bool includeCurrentBucketStats) const
{
    using TStrCRefUInt64Map = std::map<TStrCRef, uint64_t, maths::COrderings::SLess>;

    uint64_t seed = this->CIndividualModel::checksum(includeCurrentBucketStats);

#define KEY(pid) boost::cref(this->personName(pid))

    TStrCRefUInt64Map hashes;
    if (includeCurrentBucketStats)
    {
        const TFeatureSizeFeatureDataPrVecPrVec &featureData = m_CurrentBucketStats.s_FeatureData;
        for (std::size_t i = 0u; i < featureData.size(); ++i)
        {
            for (std::size_t j = 0u; j < featureData[i].second.size(); ++j)
            {
                uint64_t &hash = hashes[KEY(featureData[i].second[j].first)];
                const TFeatureData &data = featureData[i].second[j].second;
                hash = maths::CChecksum::calculate(hash, data.s_BucketValue);
                hash = core::CHashing::hashCombine(hash, static_cast<uint64_t>(data.s_IsInteger));
                hash = maths::CChecksum::calculate(hash, data.s_Samples);
            }
        }
    }

#undef KEY

    LOG_TRACE("seed = " << seed);
    LOG_TRACE("hashes = " << core::CContainerPrinter::print(hashes));

    return maths::CChecksum::calculate(seed, hashes);
}

void CMetricModel::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    mem->setName("CMetricModel");
    this->CIndividualModel::debugMemoryUsage(mem->addChild());
    core::CMemoryDebug::dynamicSize("m_CurrentBucketStats.s_PersonCounts",
                                    m_CurrentBucketStats.s_PersonCounts, mem);
    core::CMemoryDebug::dynamicSize("m_CurrentBucketStats.s_FeatureData",
                                    m_CurrentBucketStats.s_FeatureData, mem);
    core::CMemoryDebug::dynamicSize("m_CurrentBucketStats.s_InterimCorrections",
                                        m_CurrentBucketStats.s_InterimCorrections, mem);
}

std::size_t CMetricModel::memoryUsage() const
{
    return this->CIndividualModel::memoryUsage();
}

std::size_t CMetricModel::computeMemoryUsage() const
{
    std::size_t mem = this->CIndividualModel::computeMemoryUsage();
    mem += core::CMemory::dynamicSize(m_CurrentBucketStats.s_PersonCounts);
    mem += core::CMemory::dynamicSize(m_CurrentBucketStats.s_FeatureData);
    mem += core::CMemory::dynamicSize(m_CurrentBucketStats.s_InterimCorrections);
    return mem;
}

std::size_t CMetricModel::staticSize() const
{
    return sizeof(*this);
}

CMetricModel::CModelDetailsViewPtr CMetricModel::details() const
{
    return CModelDetailsViewPtr(new CMetricModelDetailsView(*this));
}

const CMetricModel::TFeatureData *CMetricModel::featureData(model_t::EFeature feature,
                                                            std::size_t pid,
                                                            core_t::TTime time) const
{
    return this->CIndividualModel::featureData(feature, pid, time, m_CurrentBucketStats.s_FeatureData);
}

void CMetricModel::createNewModels(std::size_t n, std::size_t m)
{
    this->CIndividualModel::createNewModels(n, m);
}

void CMetricModel::updateRecycledModels()
{
    this->CIndividualModel::updateRecycledModels();
}

void CMetricModel::clearPrunedResources(const TSizeVec &people,
                                        const TSizeVec &attributes)
{
    CDataGatherer &gatherer = this->dataGatherer();

    // Stop collecting for these people and add them to the free list.
    gatherer.recyclePeople(people);
    if (gatherer.dataAvailable(m_CurrentBucketStats.s_StartTime))
    {
        gatherer.featureData(m_CurrentBucketStats.s_StartTime,
                             gatherer.bucketLength(),
                             m_CurrentBucketStats.s_FeatureData);
    }

    this->CIndividualModel::clearPrunedResources(people, attributes);
}

core_t::TTime CMetricModel::currentBucketStartTime() const
{
    return m_CurrentBucketStats.s_StartTime;
}

void CMetricModel::currentBucketStartTime(core_t::TTime time)
{
    m_CurrentBucketStats.s_StartTime = time;
}

uint64_t CMetricModel::currentBucketTotalCount() const
{
    return m_CurrentBucketStats.s_TotalCount;
}

CIndividualModel::TFeatureSizeSizeTripleDouble1VecUMap &CMetricModel::currentBucketInterimCorrections() const
{
    return m_CurrentBucketStats.s_InterimCorrections;
}

const CMetricModel::TSizeUInt64PrVec &CMetricModel::currentBucketPersonCounts() const
{
    return m_CurrentBucketStats.s_PersonCounts;
}

CMetricModel::TSizeUInt64PrVec &CMetricModel::currentBucketPersonCounts()
{
    return m_CurrentBucketStats.s_PersonCounts;
}

void CMetricModel::currentBucketTotalCount(uint64_t totalCount)
{
    m_CurrentBucketStats.s_TotalCount = totalCount;
}

bool CMetricModel::correlates(model_t::EFeature feature, std::size_t pid, core_t::TTime time) const
{
    if (model_t::dimension(feature) > 1 || !this->params().s_MultivariateByFields)
    {
        return false;
    }

    const maths::CModel *model{this->model(feature, pid)};
    for (const auto &correlate : model->correlates())
    {
        if (this->featureData(feature, pid == correlate[0] ? correlate[1] : correlate[0], time))
        {
            return true;
        }
    }
    return false;
}

void CMetricModel::fill(model_t::EFeature feature,
                        std::size_t pid,
                        core_t::TTime bucketTime,
                        bool interim,
                        CProbabilityAndInfluenceCalculator::SParams &params) const
{
    std::size_t dimension{model_t::dimension(feature)};
    const TFeatureData *data{this->featureData(feature, pid, bucketTime)};
    const TOptionalSample &bucket{data->s_BucketValue};
    const maths::CModel *model{this->model(feature, pid)};
    core_t::TTime time{model_t::sampleTime(feature, bucketTime, this->bucketLength(), bucket->time())};
    TDouble2Vec4Vec weights(2);
    weights[0] = model->seasonalWeight(maths::DEFAULT_SEASONAL_CONFIDENCE_INTERVAL, time);
    weights[1].assign(dimension, bucket->varianceScale());
    TOptionalUInt64 count{this->currentBucketCount(pid, bucketTime)};

    params.s_Feature = feature;
    params.s_Model = model;
    params.s_ElapsedTime = bucketTime - this->firstBucketTimes()[pid];
    params.s_Time.assign(1, TTime2Vec{time});
    params.s_Value.assign(1, bucket->value());
    if (interim && model_t::requiresInterimResultAdjustment(feature))
    {
        TDouble2Vec mode(params.s_Model->mode(time, PROBABILITY_WEIGHT_STYLES, weights));
        TDouble2Vec correction(this->interimValueCorrector().corrections(
                                         time, this->currentBucketTotalCount(),
                                         mode, bucket->value(dimension)));
        params.s_Value[0] += correction;
        this->currentBucketInterimCorrections().emplace(core::make_triple(feature, pid, pid), correction);
    }
    params.s_Count = bucket->count();
    params.s_ComputeProbabilityParams.addCalculation(model_t::probabilityCalculation(feature))
                                     .weightStyles(PROBABILITY_WEIGHT_STYLES)
                                     .addBucketEmpty(TBool2Vec(1, !count || *count == 0))
                                     .addWeights(weights);
}

void CMetricModel::fill(model_t::EFeature feature,
                        std::size_t pid,
                        core_t::TTime bucketTime,
                        bool interim,
                        CProbabilityAndInfluenceCalculator::SCorrelateParams &params,
                        TStrCRefDouble1VecDouble1VecPrPrVecVecVec &influenceValues) const
{
    using TStrCRefDouble1VecDoublePrPr = std::pair<TStrCRef, TDouble1VecDoublePr>;

    const CDataGatherer &gatherer{this->dataGatherer()};
    const maths::CModel *model{this->model(feature, pid)};
    const TSize2Vec1Vec &correlates{model->correlates()};
    const TTimeVec &firstBucketTimes{this->firstBucketTimes()};
    core_t::TTime bucketLength{gatherer.bucketLength()};

    params.s_Feature = feature;
    params.s_Model = model;
    params.s_ElapsedTime = boost::numeric::bounds<core_t::TTime>::highest();
    params.s_Times.resize(correlates.size());
    params.s_Values.resize(correlates.size());
    params.s_Counts.resize(correlates.size());
    params.s_Variables.resize(correlates.size());
    params.s_CorrelatedLabels.resize(correlates.size());
    params.s_Correlated.resize(correlates.size());
    params.s_ComputeProbabilityParams.addCalculation(model_t::probabilityCalculation(feature))
                                     .weightStyles(PROBABILITY_WEIGHT_STYLES);

    // These are indexed as follows:
    //   influenceValues["influencer name"]["correlate"]["influence value"]
    // This is because we aren't guaranteed that each influence is present for
    // each feature.
    influenceValues.resize(this->featureData(feature, pid, bucketTime)->s_InfluenceValues.size(),
                           TStrCRefDouble1VecDouble1VecPrPrVecVec(correlates.size()));

    // Declared outside the loop to minimize the number of times it is created.
    TDouble1VecDouble1VecPr value;

    for (std::size_t i = 0u; i < correlates.size(); ++i)
    {
        TSize2Vec variables(pid == correlates[i][0] ? TSize2Vec{0, 1} : TSize2Vec{1, 0});
        params.s_CorrelatedLabels[i] = gatherer.personNamePtr(correlates[i][variables[1]]);
        params.s_Correlated[i] = correlates[i][variables[1]];
        params.s_Variables[i] = variables;
        const maths::CModel *models[]{model, this->model(feature, correlates[i][variables[1]])};
        TDouble2Vec4Vec weight(2, TDouble2Vec(2, 1.0));
        weight[0][variables[0]] = models[0]->seasonalWeight(maths::DEFAULT_SEASONAL_CONFIDENCE_INTERVAL, bucketTime)[0];
        weight[0][variables[1]] = models[1]->seasonalWeight(maths::DEFAULT_SEASONAL_CONFIDENCE_INTERVAL, bucketTime)[0];

        const TFeatureData *data[2];
        data[0] = this->featureData(feature, correlates[i][0], bucketTime);
        data[1] = this->featureData(feature, correlates[i][1], bucketTime);
        if (data[0] && data[1] && data[0]->s_BucketValue && data[1]->s_BucketValue)
        {
            const TOptionalSample &bucket0{data[0]->s_BucketValue};
            const TOptionalSample &bucket1{data[1]->s_BucketValue};
            core_t::TTime times[] =
                {
                    model_t::sampleTime(feature, bucketTime, bucketLength, bucket0->time()),
                    model_t::sampleTime(feature, bucketTime, bucketLength, bucket1->time())
                };
            params.s_ElapsedTime = std::min(params.s_ElapsedTime, times[0] - firstBucketTimes[correlates[i][0]]);
            params.s_ElapsedTime = std::min(params.s_ElapsedTime, times[1] - firstBucketTimes[correlates[i][1]]);
            params.s_Times[i] = TTime2Vec{times[0], times[1]};
            params.s_Values[i].resize(2 * bucket0->value().size());
            for (std::size_t j = 0u; j < bucket0->value().size(); ++j)
            {
                params.s_Values[i][2*j+0] = bucket0->value()[j];
                params.s_Values[i][2*j+1] = bucket1->value()[j];
            }
            weight[1][variables[0]] = bucket0->varianceScale();
            weight[1][variables[1]] = bucket1->varianceScale();
            for (std::size_t j = 0u; j < data[0]->s_InfluenceValues.size(); ++j)
            {
                for (const auto &influenceValue : data[0]->s_InfluenceValues[j])
                {
                    TStrCRef influence = influenceValue.first;
                    std::size_t match = static_cast<std::size_t>(
                                            std::find_if(data[1]->s_InfluenceValues[j].begin(),
                                                         data[1]->s_InfluenceValues[j].end(),
                                                         [influence](const TStrCRefDouble1VecDoublePrPr &value_)
                                                         {
                                                             return value_.first.get() == influence.get();
                                                         }) - data[1]->s_InfluenceValues[j].begin());
                    if (match < data[1]->s_InfluenceValues[j].size())
                    {
                        const TDouble1VecDoublePr &value0 = influenceValue.second;
                        const TDouble1VecDoublePr &value1 = data[1]->s_InfluenceValues[j][match].second;
                        value.first.resize(2 * value0.first.size());
                        for (std::size_t k = 0u; k < value0.first.size(); ++k)
                        {
                            value.first[2*k+0] = value0.first[k];
                            value.first[2*k+1] = value1.first[k];
                        }
                        value.second = TDouble1Vec{value0.second, value1.second};
                        influenceValues[j][i].emplace_back(influence, value);
                    }
                }
            }
        }
        TOptionalUInt64 count[2];
        count[0] = this->currentBucketCount(correlates[i][0], bucketTime);
        count[1] = this->currentBucketCount(correlates[i][1], bucketTime);
        params.s_ComputeProbabilityParams.addBucketEmpty(TBool2Vec{!count[0] || *count[0] == 0,
                                                                   !count[1] || *count[1] == 0})
                                         .addWeights(weight);
    }
    if (interim && model_t::requiresInterimResultAdjustment(feature))
    {
        core_t::TTime time{bucketTime + bucketLength / 2};
        TDouble2Vec1Vec modes(params.s_Model->correlateModes(time, PROBABILITY_WEIGHT_STYLES,
                                                             params.s_ComputeProbabilityParams.weights()));
        for (std::size_t i = 0u; i < modes.size(); ++i)
        {
            if (!params.s_Values.empty())
            {
                TDouble2Vec value_{params.s_Values[i][0], params.s_Values[i][1]};
                TDouble2Vec correction(this->interimValueCorrector().corrections(
                                                 time, this->currentBucketTotalCount(), modes[i], value_));
                for (std::size_t j = 0u; j < 2; ++j)
                {
                    params.s_Values[i][j] += correction[j];
                }
                this->currentBucketInterimCorrections().emplace(core::make_triple(feature, pid, params.s_Correlated[i]),
                                                                TDouble1Vec(1, correction[params.s_Variables[i][0]]));
            }
        }
    }
}

////////// CMetricModel::SBucketStats Implementation //////////

CMetricModel::SBucketStats::SBucketStats(core_t::TTime startTime) :
        s_StartTime(startTime),
        s_PersonCounts(),
        s_TotalCount(0),
        s_FeatureData(),
        s_InterimCorrections(1)
{}

}
}
