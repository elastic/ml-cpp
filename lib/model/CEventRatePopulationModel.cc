/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CEventRatePopulationModel.h>

#include <core/CAllocationStrategy.h>
#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CCategoricalTools.h>
#include <maths/CChecksum.h>
#include <maths/COrderings.h>
#include <maths/CRestoreParams.h>
#include <maths/CTimeSeriesDecomposition.h>
#include <maths/CTimeSeriesDecompositionStateSerialiser.h>
#include <maths/CTools.h>
#include <maths/ProbabilityAggregators.h>

#include <model/CAnnotatedProbabilityBuilder.h>
#include <model/CAnnotation.h>
#include <model/CEventRateBucketGatherer.h>
#include <model/CInterimBucketCorrector.h>
#include <model/CModelDetailsView.h>
#include <model/CPopulationModelDetail.h>
#include <model/CProbabilityAndInfluenceCalculator.h>
#include <model/CSearchKey.h>
#include <model/FrequencyPredicates.h>

#include <algorithm>

namespace ml {
namespace model {

namespace {

using TDouble2Vec = core::CSmallVector<double, 2>;
using TDouble2Vec1Vec = core::CSmallVector<TDouble2Vec, 1>;
using TTime2Vec = core::CSmallVector<core_t::TTime, 2>;
using TSizeSizePrFeatureDataPrVec = CEventRatePopulationModel::TSizeSizePrFeatureDataPrVec;
using TFeatureSizeSizePrFeatureDataPrVecPr =
    std::pair<model_t::EFeature, TSizeSizePrFeatureDataPrVec>;
using TFeatureSizeSizePrFeatureDataPrVecPrVec = std::vector<TFeatureSizeSizePrFeatureDataPrVecPr>;
using TSizeFuzzyDeduplicateUMap =
    boost::unordered_map<std::size_t, CModelTools::CFuzzyDeduplicate>;

//! \brief The values and weights for an attribute.
struct SValuesAndWeights {
    maths::CModel::TTimeDouble2VecSizeTrVec s_Values;
    maths::CModelAddSamplesParams::TDouble2VecWeightsAryVec s_TrendWeights;
    maths::CModelAddSamplesParams::TDouble2VecWeightsAryVec s_ResidualWeights;
};
using TSizeValuesAndWeightsUMap = boost::unordered_map<std::size_t, SValuesAndWeights>;

// We use short field names to reduce the state size
const std::string POPULATION_STATE_TAG("a");
const std::string NEW_ATTRIBUTE_PROBABILITY_PRIOR_TAG("b");
const std::string ATTRIBUTE_PROBABILITY_PRIOR_TAG("c");
const std::string FEATURE_MODELS_TAG("d");
const std::string FEATURE_CORRELATE_MODELS_TAG("e");
const std::string MEMORY_ESTIMATOR_TAG("f");
}

CEventRatePopulationModel::CEventRatePopulationModel(
    const SModelParams& params,
    const TDataGathererPtr& dataGatherer,
    const TFeatureMathsModelSPtrPrVec& newFeatureModels,
    const TFeatureMultivariatePriorSPtrPrVec& newFeatureCorrelateModelPriors,
    TFeatureCorrelationsPtrPrVec&& featureCorrelatesModels,
    const TFeatureInfluenceCalculatorCPtrPrVecVec& influenceCalculators,
    const TInterimBucketCorrectorCPtr& interimBucketCorrector)
    : CPopulationModel(params, dataGatherer, influenceCalculators),
      m_CurrentBucketStats(dataGatherer->currentBucketStartTime() -
                           dataGatherer->bucketLength()),
      m_NewAttributeProbabilityPrior(maths::CMultinomialConjugate::nonInformativePrior(
          boost::numeric::bounds<int>::highest(),
          params.s_DecayRate)),
      m_AttributeProbabilityPrior(maths::CMultinomialConjugate::nonInformativePrior(
          boost::numeric::bounds<int>::highest(),
          params.s_DecayRate)),
      m_InterimBucketCorrector(interimBucketCorrector), m_Probabilities(0.05) {
    this->initialize(newFeatureModels, newFeatureCorrelateModelPriors,
                     std::move(featureCorrelatesModels));
}

CEventRatePopulationModel::CEventRatePopulationModel(
    const SModelParams& params,
    const TDataGathererPtr& dataGatherer,
    const TFeatureMathsModelSPtrPrVec& newFeatureModels,
    const TFeatureMultivariatePriorSPtrPrVec& newFeatureCorrelateModelPriors,
    TFeatureCorrelationsPtrPrVec&& featureCorrelatesModels,
    const TFeatureInfluenceCalculatorCPtrPrVecVec& influenceCalculators,
    const TInterimBucketCorrectorCPtr& interimBucketCorrector,
    core::CStateRestoreTraverser& traverser)
    : CPopulationModel(params, dataGatherer, influenceCalculators),
      m_CurrentBucketStats(dataGatherer->currentBucketStartTime() -
                           dataGatherer->bucketLength()),
      m_InterimBucketCorrector(interimBucketCorrector), m_Probabilities(0.05) {
    this->initialize(newFeatureModels, newFeatureCorrelateModelPriors,
                     std::move(featureCorrelatesModels));
    traverser.traverseSubLevel(std::bind(&CEventRatePopulationModel::acceptRestoreTraverser,
                                         this, std::placeholders::_1));
}

void CEventRatePopulationModel::initialize(
    const TFeatureMathsModelSPtrPrVec& newFeatureModels,
    const TFeatureMultivariatePriorSPtrPrVec& newFeatureCorrelateModelPriors,
    TFeatureCorrelationsPtrPrVec&& featureCorrelatesModels) {
    m_FeatureModels.reserve(newFeatureModels.size());
    for (const auto& model : newFeatureModels) {
        m_FeatureModels.emplace_back(model.first, model.second);
    }
    std::sort(m_FeatureModels.begin(), m_FeatureModels.end(),
              [](const SFeatureModels& lhs, const SFeatureModels& rhs) {
                  return lhs.s_Feature < rhs.s_Feature;
              });

    if (this->params().s_MultivariateByFields) {
        m_FeatureCorrelatesModels.reserve(featureCorrelatesModels.size());
        for (std::size_t i = 0; i < featureCorrelatesModels.size(); ++i) {
            m_FeatureCorrelatesModels.emplace_back(
                featureCorrelatesModels[i].first,
                newFeatureCorrelateModelPriors[i].second,
                std::move(featureCorrelatesModels[i].second));
        }
        std::sort(m_FeatureCorrelatesModels.begin(), m_FeatureCorrelatesModels.end(),
                  [](const SFeatureCorrelateModels& lhs, const SFeatureCorrelateModels& rhs) {
                      return lhs.s_Feature < rhs.s_Feature;
                  });
    }
}

CEventRatePopulationModel::CEventRatePopulationModel(bool isForPersistence,
                                                     const CEventRatePopulationModel& other)
    : CPopulationModel(isForPersistence, other),
      m_CurrentBucketStats(0), // Not needed for persistence so minimally constructed
      m_NewAttributeProbabilityPrior(other.m_NewAttributeProbabilityPrior),
      m_AttributeProbabilityPrior(other.m_AttributeProbabilityPrior),
      m_Probabilities(0.05), // Not needed for persistence so minimally construct
      m_MemoryEstimator(other.m_MemoryEstimator) {
    if (!isForPersistence) {
        LOG_ABORT(<< "This constructor only creates clones for persistence");
    }

    m_FeatureModels.reserve(m_FeatureModels.size());
    for (const auto& feature : other.m_FeatureModels) {
        m_FeatureModels.emplace_back(feature.s_Feature, feature.s_NewModel);
        m_FeatureModels.back().s_Models.reserve(feature.s_Models.size());
        for (const auto& model : feature.s_Models) {
            m_FeatureModels.back().s_Models.emplace_back(model->cloneForPersistence());
        }
    }

    m_FeatureCorrelatesModels.reserve(other.m_FeatureCorrelatesModels.size());
    for (const auto& feature : other.m_FeatureCorrelatesModels) {
        m_FeatureCorrelatesModels.emplace_back(
            feature.s_Feature, feature.s_ModelPrior,
            TCorrelationsPtr(feature.s_Models->cloneForPersistence()));
    }
}

void CEventRatePopulationModel::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertLevel(POPULATION_STATE_TAG,
                         std::bind(&CEventRatePopulationModel::doAcceptPersistInserter,
                                   this, std::placeholders::_1));
    inserter.insertLevel(NEW_ATTRIBUTE_PROBABILITY_PRIOR_TAG,
                         std::bind(&maths::CMultinomialConjugate::acceptPersistInserter,
                                   &m_NewAttributeProbabilityPrior, std::placeholders::_1));
    inserter.insertLevel(ATTRIBUTE_PROBABILITY_PRIOR_TAG,
                         std::bind(&maths::CMultinomialConjugate::acceptPersistInserter,
                                   &m_AttributeProbabilityPrior, std::placeholders::_1));
    for (const auto& feature : m_FeatureModels) {
        inserter.insertLevel(FEATURE_MODELS_TAG,
                             std::bind(&SFeatureModels::acceptPersistInserter,
                                       &feature, std::placeholders::_1));
    }
    for (const auto& feature : m_FeatureCorrelatesModels) {
        inserter.insertLevel(FEATURE_CORRELATE_MODELS_TAG,
                             std::bind(&SFeatureCorrelateModels::acceptPersistInserter,
                                       &feature, std::placeholders::_1));
    }
    core::CPersistUtils::persist(MEMORY_ESTIMATOR_TAG, m_MemoryEstimator, inserter);
}

bool CEventRatePopulationModel::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    std::size_t i = 0u, j = 0;
    do {
        const std::string& name = traverser.name();
        RESTORE(POPULATION_STATE_TAG,
                traverser.traverseSubLevel(std::bind(&CEventRatePopulationModel::doAcceptRestoreTraverser,
                                                     this, std::placeholders::_1)))
        RESTORE_NO_ERROR(
            NEW_ATTRIBUTE_PROBABILITY_PRIOR_TAG,
            maths::CMultinomialConjugate restored(
                this->params().distributionRestoreParams(maths_t::E_DiscreteData), traverser);
            m_NewAttributeProbabilityPrior.swap(restored))
        RESTORE_NO_ERROR(
            ATTRIBUTE_PROBABILITY_PRIOR_TAG,
            maths::CMultinomialConjugate restored(
                this->params().distributionRestoreParams(maths_t::E_DiscreteData), traverser);
            m_AttributeProbabilityPrior.swap(restored))
        RESTORE(FEATURE_MODELS_TAG,
                i == m_FeatureModels.size() ||
                    traverser.traverseSubLevel(std::bind(
                        &SFeatureModels::acceptRestoreTraverser, &m_FeatureModels[i++],
                        std::cref(this->params()), std::placeholders::_1)))
        RESTORE(FEATURE_CORRELATE_MODELS_TAG,
                j == m_FeatureCorrelatesModels.size() ||
                    traverser.traverseSubLevel(std::bind(
                        &SFeatureCorrelateModels::acceptRestoreTraverser,
                        &m_FeatureCorrelatesModels[j++],
                        std::cref(this->params()), std::placeholders::_1)))
        RESTORE(MEMORY_ESTIMATOR_TAG,
                core::CPersistUtils::restore(MEMORY_ESTIMATOR_TAG, m_MemoryEstimator, traverser))
    } while (traverser.next());

    for (auto& feature : m_FeatureModels) {
        for (auto& model : feature.s_Models) {
            for (const auto& correlates : m_FeatureCorrelatesModels) {
                if (feature.s_Feature == correlates.s_Feature) {
                    model->modelCorrelations(*correlates.s_Models);
                }
            }
        }
    }

    return true;
}

CAnomalyDetectorModel* CEventRatePopulationModel::cloneForPersistence() const {
    return new CEventRatePopulationModel(true, *this);
}

model_t::EModelType CEventRatePopulationModel::category() const {
    return model_t::E_EventRateOnline;
}

bool CEventRatePopulationModel::isEventRate() const {
    return true;
}

bool CEventRatePopulationModel::isMetric() const {
    return false;
}

CEventRatePopulationModel::TDouble1Vec
CEventRatePopulationModel::currentBucketValue(model_t::EFeature feature,
                                              std::size_t pid,
                                              std::size_t cid,
                                              core_t::TTime time) const {
    const TSizeSizePrFeatureDataPrVec& featureData = this->featureData(feature, time);
    auto i = find(featureData, pid, cid);
    return i != featureData.end() ? extractValue(feature, *i) : TDouble1Vec(1, 0.0);
}

CEventRatePopulationModel::TDouble1Vec
CEventRatePopulationModel::baselineBucketMean(model_t::EFeature feature,
                                              std::size_t pid,
                                              std::size_t cid,
                                              model_t::CResultType type,
                                              const TSizeDoublePr1Vec& correlated,
                                              core_t::TTime time) const {
    const maths::CModel* model{this->model(feature, cid)};
    if (!model) {
        return TDouble1Vec();
    }

    static const TSizeDoublePr1Vec NO_CORRELATED;
    TDouble2Vec hint;
    if (model_t::isDiurnal(feature)) {
        hint = this->currentBucketValue(feature, pid, cid, time);
    }
    TDouble1Vec result(model->predict(
        time, type.isUnconditional() ? NO_CORRELATED : correlated, hint));

    double probability = 1.0;
    if (model_t::isConstant(feature) && !m_AttributeProbabilities.lookup(pid, probability)) {
        probability = 1.0;
    }
    for (auto& coord : result) {
        coord = probability * model_t::inverseOffsetCountToZero(feature, coord);
    }
    this->correctBaselineForInterim(feature, pid, cid, type, correlated,
                                    this->currentBucketInterimCorrections(), result);

    TDouble1VecDouble1VecPr support{model_t::support(feature)};
    return maths::CTools::truncate(result, support.first, support.second);
}

bool CEventRatePopulationModel::bucketStatsAvailable(core_t::TTime time) const {
    return time >= m_CurrentBucketStats.s_StartTime &&
           time < m_CurrentBucketStats.s_StartTime + this->bucketLength();
}

void CEventRatePopulationModel::sampleBucketStatistics(core_t::TTime startTime,
                                                       core_t::TTime endTime,
                                                       CResourceMonitor& resourceMonitor) {
    CDataGatherer& gatherer = this->dataGatherer();
    core_t::TTime bucketLength = gatherer.bucketLength();
    if (!gatherer.dataAvailable(startTime)) {
        return;
    }

    this->createUpdateNewModels(startTime, resourceMonitor);
    this->currentBucketInterimCorrections().clear();

    for (core_t::TTime time = startTime; time < endTime; time += bucketLength) {
        // Currently, we only remember one bucket.
        m_CurrentBucketStats.s_StartTime = time;
        TSizeUInt64PrVec& personCounts = m_CurrentBucketStats.s_PersonCounts;
        gatherer.personNonZeroCounts(time, personCounts);
        this->applyFilter(model_t::E_XF_Over, false, this->personFilter(), personCounts);

        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        gatherer.featureData(time, bucketLength, featureData);
        for (auto& featureData_ : featureData) {
            model_t::EFeature feature = featureData_.first;
            TSizeSizePrFeatureDataPrVec& data = m_CurrentBucketStats.s_FeatureData[feature];
            data.swap(featureData_.second);
            LOG_TRACE(<< model_t::print(feature) << ": "
                      << core::CContainerPrinter::print(data));
            this->applyFilters(false, this->personFilter(), this->attributeFilter(), data);
        }
    }
}

void CEventRatePopulationModel::sample(core_t::TTime startTime,
                                       core_t::TTime endTime,
                                       CResourceMonitor& resourceMonitor) {
    CDataGatherer& gatherer = this->dataGatherer();
    core_t::TTime bucketLength = gatherer.bucketLength();
    if (!gatherer.validateSampleTimes(startTime, endTime)) {
        return;
    }

    this->createUpdateNewModels(startTime, resourceMonitor);
    this->currentBucketInterimCorrections().clear();
    m_CurrentBucketStats.s_Annotations.clear();

    for (core_t::TTime time = startTime; time < endTime; time += bucketLength) {
        LOG_TRACE(<< "Sampling [" << time << "," << time + bucketLength << ")");

        gatherer.sampleNow(time);
        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        gatherer.featureData(time, bucketLength, featureData);

        this->CPopulationModel::sample(time, time + bucketLength, resourceMonitor);
        const TTimeVec& preSampleAttributeLastBucketTimes = this->attributeLastBucketTimes();
        TSizeTimeUMap attributeLastBucketTimesMap;
        for (const auto& featureData_ : featureData) {
            TSizeSizePrFeatureDataPrVec& data =
                m_CurrentBucketStats.s_FeatureData[featureData_.first];
            for (const auto& data_ : data) {
                std::size_t cid = CDataGatherer::extractAttributeId(data_);
                attributeLastBucketTimesMap[cid] = preSampleAttributeLastBucketTimes[cid];
            }
        }

        // Currently, we only remember one bucket.
        m_CurrentBucketStats.s_StartTime = time;
        TSizeUInt64PrVec& personCounts = m_CurrentBucketStats.s_PersonCounts;
        gatherer.personNonZeroCounts(time, personCounts);
        this->applyFilter(model_t::E_XF_Over, true, this->personFilter(), personCounts);

        for (auto& featureData_ : featureData) {
            model_t::EFeature feature = featureData_.first;
            TSizeSizePrFeatureDataPrVec& data = m_CurrentBucketStats.s_FeatureData[feature];
            data.swap(featureData_.second);
            LOG_TRACE(<< model_t::print(feature) << ": "
                      << core::CContainerPrinter::print(data));

            if (feature == model_t::E_PopulationUniquePersonCountByAttribute) {
                TDoubleVec categories;
                TDoubleVec concentrations;
                categories.reserve(data.size());
                concentrations.reserve(data.size());
                for (const auto& tuple : data) {
                    categories.push_back(static_cast<double>(
                        CDataGatherer::extractAttributeId(tuple)));
                    concentrations.push_back(static_cast<double>(
                        CDataGatherer::extractData(tuple).s_Count));
                }
                maths::CMultinomialConjugate prior(boost::numeric::bounds<int>::highest(),
                                                   categories, concentrations);
                m_AttributeProbabilityPrior.swap(prior);
                continue;
            }
            if (model_t::isCategorical(feature)) {
                continue;
            }

            this->applyFilters(true, this->personFilter(), this->attributeFilter(), data);

            core_t::TTime sampleTime = model_t::sampleTime(feature, time, bucketLength);

            TSizeValuesAndWeightsUMap attributeValuesAndWeights;
            TSizeFuzzyDeduplicateUMap duplicates;

            if (data.size() >= this->params().s_MinimumToFuzzyDeduplicate) {
                // Set up fuzzy de-duplication.
                for (const auto& data_ : data) {
                    std::size_t cid = CDataGatherer::extractAttributeId(data_);
                    uint64_t count = CDataGatherer::extractData(data_).s_Count;
                    duplicates[cid].add({static_cast<double>(count)});
                }
                for (auto& attribute : duplicates) {
                    attribute.second.computeEpsilons(
                        bucketLength, this->params().s_MinimumToFuzzyDeduplicate);
                }
            }

            for (const auto& data_ : data) {
                std::size_t pid = CDataGatherer::extractPersonId(data_);
                std::size_t cid = CDataGatherer::extractAttributeId(data_);

                maths::CModel* model{this->model(feature, cid)};
                if (!model) {
                    LOG_ERROR(<< "Missing model for " << this->attributeName(cid));
                    continue;
                }
                if (this->shouldIgnoreSample(feature, pid, cid, sampleTime)) {
                    core_t::TTime skipTime = sampleTime - attributeLastBucketTimesMap[cid];
                    if (skipTime > 0) {
                        model->skipTime(skipTime);
                        // Update the last time so we don't advance the same model
                        // multiple times (once per person)
                        attributeLastBucketTimesMap[cid] = sampleTime;
                    }
                    continue;
                }

                double count =
                    static_cast<double>(CDataGatherer::extractData(data_).s_Count);
                double value = model_t::offsetCountToZero(feature, count);
                double countWeight = this->sampleRateWeight(pid, cid) *
                                     this->learnRate(feature);
                LOG_TRACE(<< "Adding " << value
                          << " for person = " << gatherer.personName(pid)
                          << " and attribute = " << gatherer.attributeName(cid));

                SValuesAndWeights& attribute = attributeValuesAndWeights[cid];
                std::size_t duplicate = duplicates[cid].duplicate(sampleTime, {value});

                if (duplicate < attribute.s_Values.size()) {
                    model->addCountWeights(sampleTime, countWeight, countWeight,
                                           1.0, attribute.s_TrendWeights[duplicate],
                                           attribute.s_ResidualWeights[duplicate]);
                } else {
                    attribute.s_Values.emplace_back(sampleTime, TDouble2Vec{value}, pid);
                    attribute.s_TrendWeights.push_back(
                        maths_t::CUnitWeights::unit<TDouble2Vec>(1));
                    attribute.s_ResidualWeights.push_back(
                        maths_t::CUnitWeights::unit<TDouble2Vec>(1));
                    model->countWeights(sampleTime, {value}, countWeight,
                                        countWeight, 1.0, // winsorisation derate
                                        1.0, // count variance scale
                                        attribute.s_TrendWeights.back(),
                                        attribute.s_ResidualWeights.back());
                }
            }

            for (auto& attribute : attributeValuesAndWeights) {
                std::size_t cid = attribute.first;
                auto annotationCallback = [&](const std::string& annotation) {
                    if (this->params().s_AnnotationsEnabled) {
                        m_CurrentBucketStats.s_Annotations.emplace_back(
                            time, CAnnotation::E_ModelChange, annotation,
                            gatherer.searchKey().detectorIndex(),
                            gatherer.searchKey().partitionFieldName(),
                            gatherer.partitionFieldValue(),
                            gatherer.searchKey().overFieldName(),
                            gatherer.attributeName(cid),
                            gatherer.searchKey().byFieldName(), EMPTY_STRING);
                    }
                };

                maths::CModelAddSamplesParams params;
                params.integer(true)
                    .nonNegative(true)
                    .propagationInterval(this->propagationTime(cid, sampleTime))
                    .trendWeights(attribute.second.s_TrendWeights)
                    .priorWeights(attribute.second.s_ResidualWeights)
                    .annotationCallback([&](const std::string& annotation) {
                        annotationCallback(annotation);
                    });

                maths::CModel* model{this->model(feature, cid)};
                if (model == nullptr) {
                    LOG_TRACE(<< "Model unexpectedly null");
                    continue;
                }
                if (model->addSamples(params, attribute.second.s_Values) ==
                    maths::CModel::E_Reset) {
                    gatherer.resetSampleCount(cid);
                }
            }
        }

        for (const auto& feature : m_FeatureCorrelatesModels) {
            feature.s_Models->processSamples();
        }

        m_AttributeProbabilities = TCategoryProbabilityCache(m_AttributeProbabilityPrior);
        m_Probabilities.clear();
    }
}

void CEventRatePopulationModel::prune(std::size_t maximumAge) {
    CDataGatherer& gatherer = this->dataGatherer();

    TSizeVec peopleToRemove;
    TSizeVec attributesToRemove;
    this->peopleAndAttributesToRemove(m_CurrentBucketStats.s_StartTime, maximumAge,
                                      peopleToRemove, attributesToRemove);

    if (peopleToRemove.empty() && attributesToRemove.empty()) {
        return;
    }

    std::sort(peopleToRemove.begin(), peopleToRemove.end());
    std::sort(attributesToRemove.begin(), attributesToRemove.end());
    LOG_DEBUG(<< "Removing people {" << this->printPeople(peopleToRemove, 20) << '}');
    LOG_DEBUG(<< "Removing attributes {"
              << this->printAttributes(attributesToRemove, 20) << '}');

    // Stop collecting for these people/attributes and add them
    // to the free list.
    gatherer.recyclePeople(peopleToRemove);
    gatherer.recycleAttributes(attributesToRemove);

    if (gatherer.dataAvailable(m_CurrentBucketStats.s_StartTime)) {
        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        gatherer.featureData(m_CurrentBucketStats.s_StartTime,
                             gatherer.bucketLength(), featureData);
        for (auto& feature : featureData) {
            m_CurrentBucketStats.s_FeatureData[feature.first].swap(feature.second);
        }
    }

    TDoubleVec categoriesToRemove;
    categoriesToRemove.reserve(attributesToRemove.size());
    for (auto attribute : attributesToRemove) {
        categoriesToRemove.push_back(static_cast<double>(attribute));
    }
    std::sort(categoriesToRemove.begin(), categoriesToRemove.end());
    m_AttributeProbabilityPrior.removeCategories(categoriesToRemove);
    m_AttributeProbabilities = TCategoryProbabilityCache(m_AttributeProbabilityPrior);

    this->clearPrunedResources(peopleToRemove, attributesToRemove);
    this->removePeople(peopleToRemove);
}

bool CEventRatePopulationModel::computeProbability(std::size_t pid,
                                                   core_t::TTime startTime,
                                                   core_t::TTime endTime,
                                                   CPartitioningFields& partitioningFields,
                                                   std::size_t numberAttributeProbabilities,
                                                   SAnnotatedProbability& result) const {
    const CDataGatherer& gatherer = this->dataGatherer();
    core_t::TTime bucketLength = gatherer.bucketLength();

    if (endTime != startTime + bucketLength) {
        LOG_ERROR(<< "Can only compute probability for single bucket");
        return false;
    }
    if (pid > gatherer.numberPeople()) {
        LOG_TRACE(<< "No person for pid = " << pid);
        return false;
    }

    LOG_TRACE(<< "computeProbability(" << gatherer.personName(pid) << ")");

    using TStoredStringPtr1Vec = core::CSmallVector<core::CStoredStringPtr, 1>;
    using TSizeProbabilityAndInfluenceUMap =
        boost::unordered_map<std::size_t, CProbabilityAndInfluenceCalculator>;
    using TDoubleFeaturePr = std::pair<double, model_t::EFeature>;
    using TDoubleFeaturePrMinAccumulator =
        maths::CBasicStatistics::SMin<TDoubleFeaturePr>::TAccumulator;
    using TSizeDoubleFeaturePrMinAccumulatorUMap =
        boost::unordered_map<std::size_t, TDoubleFeaturePrMinAccumulator>;

    static const TStoredStringPtr1Vec NO_CORRELATED_ATTRIBUTES;
    static const TSizeDoublePr1Vec NO_CORRELATES;

    partitioningFields.add(gatherer.attributeFieldName(), EMPTY_STRING);

    CProbabilityAndInfluenceCalculator pConditionalTemplate(this->params().s_InfluenceCutoff);
    pConditionalTemplate.addAggregator(maths::CJointProbabilityOfLessLikelySamples());
    pConditionalTemplate.addAggregator(maths::CProbabilityOfExtremeSample());
    if (this->params().s_CacheProbabilities) {
        pConditionalTemplate.addCache(m_Probabilities);
    }
    TSizeProbabilityAndInfluenceUMap pConditional;

    TSizeDoubleFeaturePrMinAccumulatorUMap minimumProbabilityFeatures;

    maths::CMultinomialConjugate personAttributeProbabilityPrior(m_NewAttributeProbabilityPrior);

    CAnnotatedProbabilityBuilder resultBuilder(
        result, std::max(numberAttributeProbabilities, std::size_t(1)),
        function_t::function(gatherer.features()), gatherer.numberActivePeople());
    resultBuilder.attributeProbabilityPrior(&m_AttributeProbabilityPrior);
    resultBuilder.personAttributeProbabilityPrior(&personAttributeProbabilityPrior);

    for (std::size_t i = 0; i < gatherer.numberFeatures(); ++i) {
        model_t::EFeature feature = gatherer.feature(i);
        LOG_TRACE(<< "feature = " << model_t::print(feature));

        if (feature == model_t::E_PopulationAttributeTotalCountByPerson) {
            const TSizeSizePrFeatureDataPrVec& data = this->featureData(feature, startTime);
            TSizeSizePr range = personRange(data, pid);
            for (std::size_t j = range.first; j < range.second; ++j) {
                TDouble1Vec category{
                    static_cast<double>(CDataGatherer::extractAttributeId(data[j]))};
                maths_t::TDoubleWeightsAry1Vec weights{maths_t::countWeight(
                    static_cast<double>(CDataGatherer::extractData(data[j]).s_Count))};
                personAttributeProbabilityPrior.addSamples(category, weights);
            }
            continue;
        }
        if (model_t::isCategorical(feature)) {
            continue;
        }

        const TSizeSizePrFeatureDataPrVec& featureData = this->featureData(feature, startTime);
        TSizeSizePr range = personRange(featureData, pid);

        for (std::size_t j = range.first; j < range.second; ++j) {
            std::size_t cid = CDataGatherer::extractAttributeId(featureData[j]);

            if (this->shouldIgnoreResult(feature, result.s_ResultType, pid, cid,
                                         model_t::sampleTime(feature, startTime, bucketLength))) {
                continue;
            }

            partitioningFields.back().second = TStrCRef(gatherer.attributeName(cid));

            if (this->correlates(feature, pid, cid, startTime)) {
                // TODO
            } else {
                CProbabilityAndInfluenceCalculator::SParams params(partitioningFields);
                if (this->fill(feature, pid, cid, startTime, result.isInterim(),
                               params) == false) {
                    continue;
                }
                model_t::CResultType type;
                TSize1Vec mostAnomalousCorrelate;
                if (pConditional.emplace(cid, pConditionalTemplate)
                        .first->second.addProbability(
                            feature, cid, *params.s_Model, params.s_ElapsedTime,
                            params.s_ComputeProbabilityParams, params.s_Time,
                            params.s_Value, params.s_Probability, params.s_Tail,
                            type, mostAnomalousCorrelate)) {
                    LOG_TRACE(<< "P(" << params.describe()
                              << ", attribute = " << gatherer.attributeName(cid)
                              << ", person = " << gatherer.personName(pid)
                              << ") = " << params.s_Probability);
                    CProbabilityAndInfluenceCalculator& calculator =
                        pConditional.emplace(cid, pConditionalTemplate).first->second;
                    const auto& influenceValues =
                        CDataGatherer::extractData(featureData[j]).s_InfluenceValues;
                    for (std::size_t k = 0; k < influenceValues.size(); ++k) {
                        if (const CInfluenceCalculator* influenceCalculator =
                                this->influenceCalculator(feature, k)) {
                            calculator.plugin(*influenceCalculator);
                            calculator.addInfluences(*(gatherer.beginInfluencers() + k),
                                                     influenceValues[k], params);
                        }
                    }
                    minimumProbabilityFeatures[cid].add({params.s_Probability, feature});
                } else {
                    LOG_ERROR(<< "Unable to compute P(" << params.describe()
                              << ", attribute = " << gatherer.attributeName(cid)
                              << ", person = " << gatherer.personName(pid) << ")");
                }
            }
        }
    }

    CProbabilityAndInfluenceCalculator pJoint(this->params().s_InfluenceCutoff);
    pJoint.addAggregator(maths::CJointProbabilityOfLessLikelySamples());

    for (const auto& pConditional_ : pConditional) {
        std::size_t cid = pConditional_.first;
        CProbabilityAndInfluenceCalculator pPersonAndAttribute(this->params().s_InfluenceCutoff);
        pPersonAndAttribute.addAggregator(maths::CJointProbabilityOfLessLikelySamples());
        pPersonAndAttribute.add(pConditional_.second);
        double pAttribute;
        if (m_AttributeProbabilities.lookup(cid, pAttribute)) {
            pPersonAndAttribute.addProbability(pAttribute);
        }
        LOG_TRACE(<< "P(" << gatherer.attributeName(cid) << ") = " << pAttribute);

        // The idea is we imagine drawing n samples from the person's total
        // attribute set, where n is the size of the person's attribute set,
        // and we weight each sample according to the probability it occurs
        // assuming the attributes are distributed according to the supplied
        // multinomial distribution.
        double w = 1.0;
        double pAttributeGivenPerson;
        if (personAttributeProbabilityPrior.probability(static_cast<double>(cid),
                                                        pAttributeGivenPerson)) {
            w = maths::CCategoricalTools::probabilityOfCategory(
                pConditional.size(), pAttributeGivenPerson);
        }
        LOG_TRACE(<< "w = " << w);

        pJoint.add(pPersonAndAttribute, w);

        auto feature = minimumProbabilityFeatures.find(cid);
        if (feature == minimumProbabilityFeatures.end()) {
            LOG_ERROR(<< "No feature for " << gatherer.attributeName(cid));
        } else {
            double p;
            pPersonAndAttribute.calculate(p);
            resultBuilder.addAttributeProbability(
                cid, gatherer.attributeNamePtr(cid), pAttribute, p,
                model_t::CResultType::E_Unconditional, (feature->second)[0].second,
                NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
        }
    }

    if (pJoint.empty()) {
        LOG_TRACE(<< "No samples in [" << startTime << "," << endTime << ")");
        return false;
    }

    double p;
    if (!pJoint.calculate(p, result.s_Influences)) {
        LOG_ERROR(<< "Failed to compute probability of " << this->personName(pid));
        return false;
    }
    LOG_TRACE(<< "probability(" << this->personName(pid) << ") = " << p);
    resultBuilder.probability(p);
    resultBuilder.build();

    return true;
}

bool CEventRatePopulationModel::computeTotalProbability(
    const std::string& /*person*/,
    std::size_t /*numberAttributeProbabilities*/,
    TOptionalDouble& probability,
    TAttributeProbability1Vec& attributeProbabilities) const {
    probability = TOptionalDouble();
    attributeProbabilities.clear();
    return true;
}

uint64_t CEventRatePopulationModel::checksum(bool includeCurrentBucketStats) const {
    uint64_t seed = this->CPopulationModel::checksum(includeCurrentBucketStats);
    seed = maths::CChecksum::calculate(seed, m_NewAttributeProbabilityPrior);
    if (includeCurrentBucketStats) {
        seed = maths::CChecksum::calculate(seed, m_CurrentBucketStats.s_StartTime);
    }

    using TStrCRefStrCRefPr = std::pair<TStrCRef, TStrCRef>;
    using TStrCRefStrCRefPrUInt64Map =
        std::map<TStrCRefStrCRefPr, uint64_t, maths::COrderings::SLess>;

    const CDataGatherer& gatherer = this->dataGatherer();

    TStrCRefStrCRefPrUInt64Map hashes;

    const TDoubleVec& categories = m_AttributeProbabilityPrior.categories();
    const TDoubleVec& concentrations = m_AttributeProbabilityPrior.concentrations();
    for (std::size_t i = 0; i < categories.size(); ++i) {
        std::size_t cid = static_cast<std::size_t>(categories[i]);
        uint64_t& hash =
            hashes[{std::cref(EMPTY_STRING), std::cref(this->attributeName(cid))}];
        hash = maths::CChecksum::calculate(hash, concentrations[i]);
    }

    for (const auto& feature : m_FeatureModels) {
        for (std::size_t cid = 0; cid < feature.s_Models.size(); ++cid) {
            if (gatherer.isAttributeActive(cid)) {
                uint64_t& hash =
                    hashes[{std::cref(EMPTY_STRING), std::cref(gatherer.attributeName(cid))}];
                hash = maths::CChecksum::calculate(hash, feature.s_Models[cid]);
            }
        }
    }

    for (const auto& feature : m_FeatureCorrelatesModels) {
        for (const auto& model : feature.s_Models->correlationModels()) {
            std::size_t cids[]{model.first.first, model.first.second};
            if (gatherer.isAttributeActive(cids[0]) &&
                gatherer.isAttributeActive(cids[1])) {
                uint64_t& hash = hashes[{std::cref(gatherer.attributeName(cids[0])),
                                         std::cref(gatherer.attributeName(cids[1]))}];
                hash = maths::CChecksum::calculate(hash, model.second);
            }
        }
    }

    if (includeCurrentBucketStats) {
        for (const auto& personCount : this->personCounts()) {
            uint64_t& hash =
                hashes[{std::cref(gatherer.personName(personCount.first)), std::cref(EMPTY_STRING)}];
            hash = maths::CChecksum::calculate(hash, personCount.second);
        }
        for (const auto& feature : m_CurrentBucketStats.s_FeatureData) {
            for (const auto& data : feature.second) {
                std::size_t pid = CDataGatherer::extractPersonId(data);
                std::size_t cid = CDataGatherer::extractAttributeId(data);
                uint64_t& hash =
                    hashes[{std::cref(this->personName(pid)), std::cref(this->attributeName(cid))}];
                hash = maths::CChecksum::calculate(
                    hash, CDataGatherer::extractData(data).s_Count);
            }
        }
    }

    LOG_TRACE(<< "seed = " << seed);
    LOG_TRACE(<< "hashes = " << core::CContainerPrinter::print(hashes));

    return maths::CChecksum::calculate(seed, hashes);
}

void CEventRatePopulationModel::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CEventRatePopulationModel");
    this->CPopulationModel::debugMemoryUsage(mem->addChild());
    core::CMemoryDebug::dynamicSize("m_CurrentBucketStats.s_PersonCounts",
                                    m_CurrentBucketStats.s_PersonCounts, mem);
    core::CMemoryDebug::dynamicSize("m_CurrentBucketStats.s_FeatureData",
                                    m_CurrentBucketStats.s_FeatureData, mem);
    core::CMemoryDebug::dynamicSize("m_CurrentBucketStats.s_InterimCorrections",
                                    m_CurrentBucketStats.s_InterimCorrections, mem);
    core::CMemoryDebug::dynamicSize("m_CurrentBucketStats.s_Annotations",
                                    m_CurrentBucketStats.s_Annotations, mem);
    core::CMemoryDebug::dynamicSize("m_AttributeProbabilities",
                                    m_AttributeProbabilities, mem);
    core::CMemoryDebug::dynamicSize("m_NewPersonAttributePrior",
                                    m_NewAttributeProbabilityPrior, mem);
    core::CMemoryDebug::dynamicSize("m_AttributeProbabilityPrior",
                                    m_AttributeProbabilityPrior, mem);
    core::CMemoryDebug::dynamicSize("m_FeatureModels", m_FeatureModels, mem);
    core::CMemoryDebug::dynamicSize("m_FeatureCorrelatesModels",
                                    m_FeatureCorrelatesModels, mem);
    core::CMemoryDebug::dynamicSize("m_InterimBucketCorrector",
                                    m_InterimBucketCorrector, mem);
    core::CMemoryDebug::dynamicSize("m_MemoryEstimator", m_MemoryEstimator, mem);
}

std::size_t CEventRatePopulationModel::memoryUsage() const {
    const CDataGatherer& gatherer = this->dataGatherer();
    TOptionalSize estimate = this->estimateMemoryUsage(
        gatherer.numberActivePeople(), gatherer.numberActiveAttributes(),
        0); // # correlations
    return estimate ? estimate.get() : this->computeMemoryUsage();
}

std::size_t CEventRatePopulationModel::computeMemoryUsage() const {
    std::size_t mem = this->CPopulationModel::memoryUsage();
    mem += core::CMemory::dynamicSize(m_CurrentBucketStats.s_PersonCounts);
    mem += core::CMemory::dynamicSize(m_CurrentBucketStats.s_FeatureData);
    mem += core::CMemory::dynamicSize(m_CurrentBucketStats.s_InterimCorrections);
    mem += core::CMemory::dynamicSize(m_CurrentBucketStats.s_Annotations);
    mem += core::CMemory::dynamicSize(m_AttributeProbabilities);
    mem += core::CMemory::dynamicSize(m_NewAttributeProbabilityPrior);
    mem += core::CMemory::dynamicSize(m_AttributeProbabilityPrior);
    mem += core::CMemory::dynamicSize(m_FeatureModels);
    mem += core::CMemory::dynamicSize(m_FeatureCorrelatesModels);
    mem += core::CMemory::dynamicSize(m_InterimBucketCorrector);
    mem += core::CMemory::dynamicSize(m_MemoryEstimator);
    return mem;
}

const CEventRatePopulationModel::TAnnotationVec& CEventRatePopulationModel::annotations() const {
    return m_CurrentBucketStats.s_Annotations;
}

CMemoryUsageEstimator* CEventRatePopulationModel::memoryUsageEstimator() const {
    return &m_MemoryEstimator;
}

std::size_t CEventRatePopulationModel::staticSize() const {
    return sizeof(*this);
}

CEventRatePopulationModel::TModelDetailsViewUPtr CEventRatePopulationModel::details() const {
    return TModelDetailsViewUPtr(new CEventRatePopulationModelDetailsView(*this));
}

const CEventRatePopulationModel::TSizeSizePrFeatureDataPrVec&
CEventRatePopulationModel::featureData(model_t::EFeature feature, core_t::TTime time) const {
    static const TSizeSizePrFeatureDataPrVec EMPTY;
    if (!this->bucketStatsAvailable(time)) {
        LOG_ERROR(<< "No statistics at " << time << ", current bucket = ["
                  << m_CurrentBucketStats.s_StartTime << ","
                  << m_CurrentBucketStats.s_StartTime + this->bucketLength() << ")");
        return EMPTY;
    }
    auto result = m_CurrentBucketStats.s_FeatureData.find(feature);
    return result == m_CurrentBucketStats.s_FeatureData.end() ? EMPTY : result->second;
}

core_t::TTime CEventRatePopulationModel::currentBucketStartTime() const {
    return m_CurrentBucketStats.s_StartTime;
}

void CEventRatePopulationModel::currentBucketStartTime(core_t::TTime startTime) {
    m_CurrentBucketStats.s_StartTime = startTime;
}

const CEventRatePopulationModel::TSizeUInt64PrVec&
CEventRatePopulationModel::personCounts() const {
    return m_CurrentBucketStats.s_PersonCounts;
}

CEventRatePopulationModel::TCorrectionKeyDouble1VecUMap&
CEventRatePopulationModel::currentBucketInterimCorrections() const {
    return m_CurrentBucketStats.s_InterimCorrections;
}

void CEventRatePopulationModel::createNewModels(std::size_t n, std::size_t m) {
    if (m > 0) {
        for (auto& feature : m_FeatureModels) {
            std::size_t newM = feature.s_Models.size() + m;
            core::CAllocationStrategy::reserve(feature.s_Models, newM);
            for (std::size_t cid = feature.s_Models.size(); cid < newM; ++cid) {
                feature.s_Models.emplace_back(feature.s_NewModel->clone(cid));
                for (const auto& correlates : m_FeatureCorrelatesModels) {
                    if (feature.s_Feature == correlates.s_Feature) {
                        feature.s_Models.back()->modelCorrelations(*correlates.s_Models);
                    }
                }
            }
        }
    }
    this->CPopulationModel::createNewModels(n, m);
}

void CEventRatePopulationModel::updateRecycledModels() {
    CDataGatherer& gatherer = this->dataGatherer();
    for (auto cid : gatherer.recycledAttributeIds()) {
        for (auto& feature : m_FeatureModels) {
            if (cid < feature.s_Models.size()) {
                feature.s_Models[cid].reset(feature.s_NewModel->clone(cid));
                for (const auto& correlates : m_FeatureCorrelatesModels) {
                    if (feature.s_Feature == correlates.s_Feature) {
                        feature.s_Models.back()->modelCorrelations(*correlates.s_Models);
                    }
                }
            }
        }
    }
    this->CPopulationModel::updateRecycledModels();
}

void CEventRatePopulationModel::refreshCorrelationModels(std::size_t resourceLimit,
                                                         CResourceMonitor& resourceMonitor) {
    std::size_t n = this->numberOfPeople();
    double maxNumberCorrelations = this->params().s_CorrelationModelsOverhead *
                                   static_cast<double>(n);
    auto memoryUsage = std::bind(&CAnomalyDetectorModel::estimateMemoryUsageOrComputeAndUpdate,
                                 this, n, 0, std::placeholders::_1);
    CTimeSeriesCorrelateModelAllocator allocator(
        resourceMonitor, memoryUsage, resourceLimit,
        static_cast<std::size_t>(maxNumberCorrelations + 0.5));
    for (auto& feature : m_FeatureCorrelatesModels) {
        allocator.prototypePrior(feature.s_ModelPrior);
        feature.s_Models->refresh(allocator);
    }
}

void CEventRatePopulationModel::clearPrunedResources(const TSizeVec& /*people*/,
                                                     const TSizeVec& attributes) {
    for (auto cid : attributes) {
        for (auto& feature : m_FeatureModels) {
            if (cid < feature.s_Models.size()) {
                feature.s_Models[cid].reset(this->tinyModel());
            }
        }
    }
}

const CInterimBucketCorrector& CEventRatePopulationModel::interimValueCorrector() const {
    return *m_InterimBucketCorrector;
}

void CEventRatePopulationModel::doSkipSampling(core_t::TTime startTime, core_t::TTime endTime) {
    core_t::TTime gap = endTime - startTime;
    for (auto& feature : m_FeatureModels) {
        for (auto& model : feature.s_Models) {
            model->skipTime(gap);
        }
    }
    this->CPopulationModel::doSkipSampling(startTime, endTime);
}

const maths::CModel* CEventRatePopulationModel::model(model_t::EFeature feature,
                                                      std::size_t cid) const {
    return const_cast<CEventRatePopulationModel*>(this)->model(feature, cid);
}

maths::CModel* CEventRatePopulationModel::model(model_t::EFeature feature, std::size_t cid) {
    auto i = std::find_if(m_FeatureModels.begin(), m_FeatureModels.end(),
                          [feature](const SFeatureModels& model) {
                              return model.s_Feature == feature;
                          });
    return i != m_FeatureModels.end() && cid < i->s_Models.size()
               ? i->s_Models[cid].get()
               : nullptr;
}

bool CEventRatePopulationModel::correlates(model_t::EFeature feature,
                                           std::size_t pid,
                                           std::size_t cid,
                                           core_t::TTime time) const {
    if (model_t::dimension(feature) > 1 || !this->params().s_MultivariateByFields) {
        return false;
    }

    const maths::CModel* model{this->model(feature, cid)};
    if (model == nullptr) {
        LOG_TRACE(<< "Model unexpectedly null");
        return false;
    }
    const TSizeSizePrFeatureDataPrVec& data = this->featureData(feature, time);
    TSizeSizePr range = personRange(data, pid);

    for (std::size_t j = range.first; j < range.second; ++j) {
        std::size_t cids[]{cid, CDataGatherer::extractAttributeId(data[j])};
        for (const auto& correlate : model->correlates()) {
            if ((cids[0] == correlate[0] && cids[1] == correlate[1]) ||
                (cids[1] == correlate[0] && cids[0] == correlate[1])) {
                return true;
            }
        }
    }
    return false;
}

bool CEventRatePopulationModel::fill(model_t::EFeature feature,
                                     std::size_t pid,
                                     std::size_t cid,
                                     core_t::TTime bucketTime,
                                     bool interim,
                                     CProbabilityAndInfluenceCalculator::SParams& params) const {
    auto data = find(this->featureData(feature, bucketTime), pid, cid);
    const maths::CModel* model{this->model(feature, cid)};
    if (model == nullptr) {
        LOG_TRACE(<< "Model unexpectedly null");
        return false;
    }
    core_t::TTime time{model_t::sampleTime(feature, bucketTime, this->bucketLength())};
    maths_t::TDouble2VecWeightsAry weight{[&] {
        TDouble2Vec result;
        model->seasonalWeight(maths::DEFAULT_SEASONAL_CONFIDENCE_INTERVAL, time, result);
        return maths_t::seasonalVarianceScaleWeight(result);
    }()};
    double value{model_t::offsetCountToZero(
        feature, static_cast<double>(CDataGatherer::extractData(*data).s_Count))};
    bool skipAnomalyModelUpdate = this->shouldIgnoreSample(feature, pid, cid, time);

    params.s_Feature = feature;
    params.s_Model = model;
    params.s_ElapsedTime = bucketTime - this->attributeFirstBucketTimes()[cid];
    params.s_Time.assign(1, {time});
    params.s_Value.assign(1, {value});
    if (interim && model_t::requiresInterimResultAdjustment(feature)) {
        double mode{params.s_Model->mode(time, weight)[0]};
        TDouble2Vec correction{this->interimValueCorrector().corrections(mode, value)};
        params.s_Value[0] += correction;
        this->currentBucketInterimCorrections().emplace(
            CCorrectionKey(feature, pid, cid), correction);
    }
    params.s_Count = 1.0;
    params.s_ComputeProbabilityParams
        .addCalculation(model_t::probabilityCalculation(feature))
        .addWeights(weight)
        .skipAnomalyModelUpdate(skipAnomalyModelUpdate);

    return true;
}

////////// CEventRatePopulationModel::SBucketStats Implementation //////////

CEventRatePopulationModel::SBucketStats::SBucketStats(core_t::TTime startTime)
    : s_StartTime(startTime), s_InterimCorrections(1) {
}
}
}
