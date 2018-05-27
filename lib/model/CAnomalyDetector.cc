/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CAnomalyDetector.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CStatistics.h>

#include <maths/CIntegerTools.h>
#include <maths/COrderings.h>
#include <maths/CSampling.h>

#include <model/CAnomalyDetectorModel.h>
#include <model/CAnomalyScore.h>
#include <model/CDataGatherer.h>
#include <model/CForecastModelPersist.h>
#include <model/CModelDetailsView.h>
#include <model/CModelPlotData.h>
#include <model/CSearchKey.h>

#include <boost/bind.hpp>

#include <limits>
#include <sstream>
#include <vector>

namespace ml {
namespace model {

// We use short field names to reduce the state size
namespace {

using TModelDetailsViewPtr = CAnomalyDetectorModel::CModelDetailsViewPtr;

// tag 'a' was previously used for persisting first time;
// DO NOT USE; unless it is decided to break model state BWC
const std::string MODEL_AND_GATHERER_TAG("b");
const std::string PARTITION_FIELD_VALUE_TAG("c");
const std::string KEY_TAG("d");
const std::string SIMPLE_COUNT_STATICS("f");

// classes containing static members needing persistence
//const std::string RANDOMIZED_PERIODIC_TAG("a"); // No longer used
const std::string STATISTICS_TAG("b");
const std::string SAMPLING_TAG("c");

// tags for the parts that used to be in model ensemble.
// !!! NOTE: Tags 'c' & 'e' were previously used for removed
// state. If new state is added here, tags from `f` onwards
// should be used in order not to break model state BWC.
const std::string DATA_GATHERER_TAG("a");
const std::string MODELS_TAG("b");
const std::string MODEL_TAG("d");

CAnomalyDetector::TDataGathererPtr
makeDataGatherer(const CAnomalyDetector::TModelFactoryCPtr& factory,
                 core_t::TTime startTime,
                 const std::string& partitionFieldValue) {
    CModelFactory::SGathererInitializationData initData(startTime, partitionFieldValue);
    return CAnomalyDetector::TDataGathererPtr(factory->makeDataGatherer(initData));
}

CAnomalyDetector::TModelPtr makeModel(const CAnomalyDetector::TModelFactoryCPtr& factory,
                                      const CAnomalyDetector::TDataGathererPtr& dataGatherer) {
    CModelFactory::SModelInitializationData initData(dataGatherer);
    return CAnomalyDetector::TModelPtr(factory->makeModel(initData));
}
}

// Increment this every time a change to the state is made that requires
// existing state to be discarded
const std::string CAnomalyDetector::STATE_VERSION("34");

const std::string CAnomalyDetector::COUNT_NAME("count");
const std::string CAnomalyDetector::TIME_NAME("time");
const std::string CAnomalyDetector::DISTINCT_COUNT_NAME("distinct_count");
const std::string CAnomalyDetector::RARE_NAME("rare");
const std::string CAnomalyDetector::INFO_CONTENT_NAME("info_content");
const std::string CAnomalyDetector::MEAN_NAME("mean");
const std::string CAnomalyDetector::MEDIAN_NAME("median");
const std::string CAnomalyDetector::MIN_NAME("min");
const std::string CAnomalyDetector::MAX_NAME("max");
const std::string CAnomalyDetector::VARIANCE_NAME("varp");
const std::string CAnomalyDetector::SUM_NAME("sum");
const std::string CAnomalyDetector::LAT_LONG_NAME("lat_long");
const std::string CAnomalyDetector::EMPTY_STRING;

CAnomalyDetector::CAnomalyDetector(int detectorIndex,
                                   CLimits& limits,
                                   const CAnomalyDetectorModelConfig& modelConfig,
                                   const std::string& partitionFieldValue,
                                   core_t::TTime firstTime,
                                   const TModelFactoryCPtr& modelFactory)
    : m_Limits(limits), m_DetectorIndex(detectorIndex), m_ModelConfig(modelConfig),
      m_LastBucketEndTime(maths::CIntegerTools::ceil(firstTime, modelConfig.bucketLength())),
      m_DataGatherer(makeDataGatherer(modelFactory, m_LastBucketEndTime, partitionFieldValue)),
      m_ModelFactory(modelFactory),
      m_Model(makeModel(modelFactory, m_DataGatherer)), m_IsForPersistence(false) {
    if (m_DataGatherer == nullptr) {
        LOG_ABORT(<< "Failed to construct data gatherer for detector: "
                  << this->description());
    }
    if (m_Model == nullptr) {
        LOG_ABORT(<< "Failed to construct model for detector: " << this->description());
    }
    limits.resourceMonitor().registerComponent(*this);
    LOG_DEBUG(<< "CAnomalyDetector(): " << this->description() << " for '"
              << m_DataGatherer->partitionFieldValue() << "'"
              << ", first time = " << firstTime
              << ", bucketLength = " << modelConfig.bucketLength()
              << ", m_LastBucketEndTime = " << m_LastBucketEndTime);
}

CAnomalyDetector::CAnomalyDetector(bool isForPersistence, const CAnomalyDetector& other)
    : m_Limits(other.m_Limits), m_DetectorIndex(other.m_DetectorIndex),
      m_ModelConfig(other.m_ModelConfig),
      // Empty result function is fine in this case
      // Empty result count function is fine in this case
      m_LastBucketEndTime(other.m_LastBucketEndTime),
      m_DataGatherer(other.m_DataGatherer->cloneForPersistence()),
      m_ModelFactory(other.m_ModelFactory), // Shallow copy of model factory is OK
      m_Model(other.m_Model->cloneForPersistence()),
      // Empty message propagation function is fine in this case
      m_IsForPersistence(isForPersistence) {
    if (!isForPersistence) {
        LOG_ABORT(<< "This constructor only creates clones for persistence");
    }
}

CAnomalyDetector::~CAnomalyDetector() {
    if (!m_IsForPersistence) {
        m_Limits.resourceMonitor().unRegisterComponent(*this);
    }
}

size_t CAnomalyDetector::numberActivePeople() const {
    return m_DataGatherer->numberActivePeople();
}

size_t CAnomalyDetector::numberActiveAttributes() const {
    return m_DataGatherer->numberActiveAttributes();
}

size_t CAnomalyDetector::maxDimension() const {
    return m_DataGatherer->maxDimension();
}

void CAnomalyDetector::zeroModelsToTime(core_t::TTime time) {
    // If there has been a big gap in the times, we might need to sample
    // many buckets; if there has been no gap, the loop may legitimately
    // have no iterations.

    core_t::TTime bucketLength = m_ModelConfig.bucketLength();

    while (time >= (m_LastBucketEndTime + bucketLength)) {
        core_t::TTime bucketStartTime = m_LastBucketEndTime;
        m_LastBucketEndTime += bucketLength;

        LOG_TRACE(<< "sample: m_DetectorKey = '" << this->description()
                  << "', bucketStartTime = " << bucketStartTime
                  << ", m_LastBucketEndTime = " << m_LastBucketEndTime);

        // Update the statistical models.
        m_Model->sample(bucketStartTime, m_LastBucketEndTime, m_Limits.resourceMonitor());
    }
}

bool CAnomalyDetector::acceptRestoreTraverser(const std::string& partitionFieldValue,
                                              core::CStateRestoreTraverser& traverser) {
    // As the model pointer will change during restore, we unregister
    // the detector from the resource monitor. We can register it
    // again at the end of restore.
    m_Limits.resourceMonitor().unRegisterComponent(*this);

    m_DataGatherer->clear();
    m_Model.reset();

    // We expect tags immediately below the root storing the first time the
    // models were created and the models IN THAT ORDER.

    do {
        const std::string& name = traverser.name();
        if (name == MODEL_AND_GATHERER_TAG) {
            if (traverser.traverseSubLevel(boost::bind(
                    &CAnomalyDetector::legacyModelEnsembleAcceptRestoreTraverser,
                    this, boost::cref(partitionFieldValue), _1)) == false) {
                LOG_ERROR(<< "Invalid model ensemble section in " << traverser.value());
                return false;
            }
        } else if (name == SIMPLE_COUNT_STATICS) {
            if (traverser.traverseSubLevel(boost::bind(&CAnomalyDetector::staticsAcceptRestoreTraverser,
                                                       this, _1)) == false) {
                LOG_ERROR(<< "Invalid simple count statics in " << traverser.value());
                return false;
            }
        }
    } while (traverser.next());

    m_Limits.resourceMonitor().registerComponent(*this);

    return true;
}

bool CAnomalyDetector::legacyModelEnsembleAcceptRestoreTraverser(const std::string& partitionFieldValue,
                                                                 core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        if (name == DATA_GATHERER_TAG) {
            m_DataGatherer.reset(
                m_ModelFactory->makeDataGatherer(partitionFieldValue, traverser));
            if (!m_DataGatherer) {
                LOG_ERROR(<< "Failed to restore the data gatherer from "
                          << traverser.value());
                return false;
            }
        } else if (name == MODELS_TAG) {
            if (traverser.traverseSubLevel(boost::bind(&CAnomalyDetector::legacyModelsAcceptRestoreTraverser,
                                                       this, _1)) == false) {
                LOG_ERROR(<< "Failed to restore live models from " << traverser.value());
                return false;
            }
        }
    } while (traverser.next());

    return true;
}

bool CAnomalyDetector::legacyModelsAcceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        if (name == MODEL_TAG) {
            CModelFactory::SModelInitializationData initData(m_DataGatherer);
            m_Model.reset(m_ModelFactory->makeModel(initData, traverser));
            if (!m_Model) {
                LOG_ERROR(<< "Failed to extract model from " << traverser.value());
                return false;
            }
        }
    } while (traverser.next());

    return true;
}

bool CAnomalyDetector::staticsAcceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        if (name == STATISTICS_TAG) {
            if (traverser.traverseSubLevel(
                    &core::CStatistics::staticsAcceptRestoreTraverser) == false) {
                LOG_ERROR(<< "Failed to restore statistics");
                return false;
            }
        } else if (name == SAMPLING_TAG) {
            if (traverser.traverseSubLevel(
                    &maths::CSampling::staticsAcceptRestoreTraverser) == false) {
                LOG_ERROR(<< "Failed to restore sampling state");
                return false;
            }
        }
    } while (traverser.next());

    return true;
}

bool CAnomalyDetector::partitionFieldAcceptRestoreTraverser(core::CStateRestoreTraverser& traverser,
                                                            std::string& partitionFieldValue) {
    do {
        const std::string& name = traverser.name();
        if (name == PARTITION_FIELD_VALUE_TAG) {
            partitionFieldValue = traverser.value();
            return true;
        }
    } while (traverser.next());

    return false;
}

bool CAnomalyDetector::keyAcceptRestoreTraverser(core::CStateRestoreTraverser& traverser,
                                                 CSearchKey& key) {
    do {
        const std::string& name = traverser.name();
        if (name == KEY_TAG) {
            bool successful(true);
            key = CSearchKey(traverser, successful);
            if (successful == false) {
                LOG_ERROR(<< "Invalid key in " << traverser.value());
                return false;
            }
            return true;
        }
    } while (traverser.next());

    return false;
}

void CAnomalyDetector::keyAcceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertLevel(KEY_TAG, boost::bind(&CSearchKey::acceptPersistInserter,
                                              &m_DataGatherer->searchKey(), _1));
}

void CAnomalyDetector::partitionFieldAcceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(PARTITION_FIELD_VALUE_TAG, m_DataGatherer->partitionFieldValue());
}

void CAnomalyDetector::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    // Persist static members only once within the simple count detector
    // and do this first so that other model components can use
    // static strings
    if (this->isSimpleCount()) {
        inserter.insertLevel(
            SIMPLE_COUNT_STATICS,
            boost::bind(&CAnomalyDetector::staticsAcceptPersistInserter, this, _1));
    }

    // Persist what used to belong in model ensemble at a separate level to ensure BWC
    inserter.insertLevel(MODEL_AND_GATHERER_TAG, boost::bind(&CAnomalyDetector::legacyModelEnsembleAcceptPersistInserter,
                                                             this, _1));
}

void CAnomalyDetector::staticsAcceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertLevel(STATISTICS_TAG, &core::CStatistics::staticsAcceptPersistInserter);
    inserter.insertLevel(SAMPLING_TAG, &maths::CSampling::staticsAcceptPersistInserter);
}

void CAnomalyDetector::legacyModelEnsembleAcceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertLevel(DATA_GATHERER_TAG,
                         boost::bind(&CDataGatherer::acceptPersistInserter,
                                     boost::cref(*m_DataGatherer), _1));
    // This level seems redundant but it is simulating state as it was when CModelEnsemble
    // was around.
    inserter.insertLevel(MODELS_TAG, boost::bind(&CAnomalyDetector::legacyModelsAcceptPersistInserter,
                                                 this, _1));
}

void CAnomalyDetector::legacyModelsAcceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertLevel(MODEL_TAG, boost::bind(&CAnomalyDetectorModel::acceptPersistInserter,
                                                m_Model.get(), _1));
}

const CAnomalyDetector::TStrVec& CAnomalyDetector::fieldsOfInterest() const {
    return m_DataGatherer->fieldsOfInterest();
}

void CAnomalyDetector::addRecord(core_t::TTime time, const TStrCPtrVec& fieldValues) {
    const TStrCPtrVec& processedFieldValues = this->preprocessFieldValues(fieldValues);

    CEventData eventData;
    eventData.time(time);

    m_DataGatherer->addArrival(processedFieldValues, eventData, m_Limits.resourceMonitor());
}

const CAnomalyDetector::TStrCPtrVec&
CAnomalyDetector::preprocessFieldValues(const TStrCPtrVec& fieldValues) {
    return fieldValues;
}

void CAnomalyDetector::buildResults(core_t::TTime bucketStartTime,
                                    core_t::TTime bucketEndTime,
                                    CHierarchicalResults& results) {
    core_t::TTime bucketLength = m_ModelConfig.bucketLength();
    if (m_ModelConfig.bucketResultsDelay()) {
        bucketLength /= 2;
    }
    bucketStartTime = maths::CIntegerTools::floor(bucketStartTime, bucketLength);
    bucketEndTime = maths::CIntegerTools::floor(bucketEndTime, bucketLength);
    if (bucketEndTime <= m_LastBucketEndTime) {
        return;
    }

    m_Limits.resourceMonitor().clearExtraMemory();

    this->buildResultsHelper(
        bucketStartTime, bucketEndTime,
        boost::bind(&CAnomalyDetector::sample, this, _1, _2,
                    boost::ref(m_Limits.resourceMonitor())),
        boost::bind(&CAnomalyDetector::updateLastSampledBucket, this, _1), results);
}

void CAnomalyDetector::sample(core_t::TTime startTime,
                              core_t::TTime endTime,
                              CResourceMonitor& resourceMonitor) {
    if (endTime <= startTime) {
        // Nothing to sample.
        return;
    }

    core_t::TTime bucketLength = m_ModelConfig.bucketLength();

    bool isEndOfBucketSample = endTime % bucketLength == 0;
    if (isEndOfBucketSample) {
        LOG_TRACE(<< "Going to do end-of-bucket sample");
    } else {
        LOG_TRACE(<< "Going to do out-of-phase sampleBucketStatistics");
    }

    for (core_t::TTime time = startTime; time < endTime; time += bucketLength) {
        if (isEndOfBucketSample) {
            m_Model->sample(time, time + bucketLength, resourceMonitor);
        } else {
            m_Model->sampleBucketStatistics(time, time + bucketLength, resourceMonitor);
        }
    }

    if ((endTime / bucketLength) % 10 == 0) {
        // Even if memory limiting is disabled, force a refresh every 10 buckets
        // so the user has some idea what's going on with memory.  (Note: the
        // 10 bucket interval is inexact as sampling may not take place for
        // every bucket.  However, it's probably good enough.)
        resourceMonitor.forceRefresh(*this);
    } else {
        resourceMonitor.refresh(*this);
    }
}

void CAnomalyDetector::sampleBucketStatistics(core_t::TTime startTime,
                                              core_t::TTime endTime,
                                              CResourceMonitor& resourceMonitor) {
    if (endTime <= startTime) {
        // Nothing to sample.
        return;
    }

    core_t::TTime bucketLength = m_ModelConfig.bucketLength();
    for (core_t::TTime time = startTime; time < endTime; time += bucketLength) {
        m_Model->sampleBucketStatistics(time, time + bucketLength, resourceMonitor);
    }
    resourceMonitor.refresh(*this);
}

void CAnomalyDetector::generateModelPlot(core_t::TTime bucketStartTime,
                                         core_t::TTime bucketEndTime,
                                         double boundsPercentile,
                                         const TStrSet& terms,
                                         TModelPlotDataVec& modelPlots) const {
    if (bucketEndTime <= bucketStartTime) {
        return;
    }
    if (terms.empty() || m_DataGatherer->partitionFieldValue().empty() ||
        terms.find(m_DataGatherer->partitionFieldValue()) != terms.end()) {
        const CSearchKey& key = m_DataGatherer->searchKey();
        TModelDetailsViewPtr view = m_Model.get()->details();
        if (view.get()) {
            core_t::TTime bucketLength = m_ModelConfig.bucketLength();
            for (core_t::TTime time = bucketStartTime; time < bucketEndTime;
                 time += bucketLength) {
                modelPlots.emplace_back(time, key.partitionFieldName(),
                                        m_DataGatherer->partitionFieldValue(),
                                        key.overFieldName(), key.byFieldName(),
                                        bucketLength, m_DetectorIndex);
                view->modelPlot(time, boundsPercentile, terms, modelPlots.back());
            }
        }
    }
}

CForecastDataSink::SForecastModelPrerequisites
CAnomalyDetector::getForecastPrerequisites() const {
    CForecastDataSink::SForecastModelPrerequisites prerequisites{0, 0, 0, true, false};

    TModelDetailsViewPtr view = m_Model->details();

    // The view can be empty, e.g. for the counting model.
    if (view.get() == nullptr) {
        return prerequisites;
    }

    prerequisites.s_IsPopulation = m_DataGatherer->isPopulation();

    if (prerequisites.s_IsPopulation) {
        return prerequisites;
    }

    const CSearchKey& key = m_DataGatherer->searchKey();

    prerequisites.s_IsSupportedFunction = function_t::isForecastSupported(key.function());
    if (prerequisites.s_IsSupportedFunction == false) {
        return prerequisites;
    }

    for (std::size_t pid = 0u, maxPid = m_DataGatherer->numberPeople(); pid < maxPid; ++pid) {
        // todo: Add terms filtering here
        if (m_DataGatherer->isPersonActive(pid)) {
            for (auto feature : view->features()) {
                const maths::CModel* model = view->model(feature, pid);

                // The model might not exist, e.g. for categorical features.
                if (model != nullptr) {
                    ++prerequisites.s_NumberOfModels;
                    if (model->isForecastPossible()) {
                        ++prerequisites.s_NumberOfForecastableModels;
                    }
                    prerequisites.s_MemoryUsageForDetector += model->memoryUsage();
                }
            }
        }
    }

    return prerequisites;
}

CForecastDataSink::SForecastResultSeries
CAnomalyDetector::getForecastModels(bool persistOnDisk,
                                    const std::string& persistenceFolder) const {
    CForecastDataSink::SForecastResultSeries series(m_ModelFactory->modelParams());

    if (m_DataGatherer->isPopulation()) {
        return series;
    }

    TModelDetailsViewPtr view = m_Model.get()->details();

    // The view can be empty, e.g. for the counting model.
    if (view.get() == nullptr) {
        return series;
    }

    const CSearchKey& key = m_DataGatherer->searchKey();
    series.s_ByFieldName = key.byFieldName();
    series.s_DetectorIndex = m_DetectorIndex;
    series.s_PartitionFieldName = key.partitionFieldName();
    series.s_PartitionFieldValue = m_DataGatherer->partitionFieldValue();
    series.s_MinimumSeasonalVarianceScale = m_ModelFactory->minimumSeasonalVarianceScale();

    if (persistOnDisk) {
        CForecastModelPersist::CPersist persister(persistenceFolder);

        for (std::size_t pid = 0u, maxPid = m_DataGatherer->numberPeople();
             pid < maxPid; ++pid) {
            // todo: Add terms filtering here
            if (m_DataGatherer->isPersonActive(pid)) {
                for (auto feature : view->features()) {
                    const maths::CModel* model = view->model(feature, pid);
                    if (model != nullptr && model->isForecastPossible()) {
                        persister.addModel(model, feature, m_DataGatherer->personName(pid));
                    }
                }
            }
        }

        series.s_ToForecastPersisted = persister.finalizePersistAndGetFile();
    } else {
        for (std::size_t pid = 0u, maxPid = m_DataGatherer->numberPeople();
             pid < maxPid; ++pid) {
            // todo: Add terms filtering here
            if (m_DataGatherer->isPersonActive(pid)) {
                for (auto feature : view->features()) {
                    const maths::CModel* model = view->model(feature, pid);
                    if (model != nullptr && model->isForecastPossible()) {
                        series.s_ToForecast.emplace_back(
                            feature,
                            CForecastDataSink::TMathsModelPtr(model->cloneForForecast()),
                            m_DataGatherer->personName(pid));
                    }
                }
            }
        }
    }

    return series;
}

void CAnomalyDetector::buildInterimResults(core_t::TTime bucketStartTime,
                                           core_t::TTime bucketEndTime,
                                           CHierarchicalResults& results) {
    this->buildResultsHelper(
        bucketStartTime, bucketEndTime,
        boost::bind(&CAnomalyDetector::sampleBucketStatistics, this, _1, _2,
                    boost::ref(m_Limits.resourceMonitor())),
        boost::bind(&CAnomalyDetector::noUpdateLastSampledBucket, this, _1), results);
}

void CAnomalyDetector::pruneModels() {
    // Purge out any ancient models which are effectively dead.
    m_Model->prune(m_Model->defaultPruneWindow());
}

void CAnomalyDetector::resetBucket(core_t::TTime bucketStart) {
    m_DataGatherer->resetBucket(bucketStart);
}

void CAnomalyDetector::releaseMemory(core_t::TTime samplingCutoffTime) {
    m_DataGatherer->releaseMemory(samplingCutoffTime);
}

void CAnomalyDetector::showMemoryUsage(std::ostream& stream) const {
    core::CMemoryUsage mem;
    this->debugMemoryUsage(mem.addChild());
    mem.compress();
    mem.print(stream);
    if (mem.usage() != this->memoryUsage()) {
        LOG_ERROR(<< "Discrepancy in memory report: " << mem.usage()
                  << " from debug, but " << this->memoryUsage() << " from normal");
    }
}

void CAnomalyDetector::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("Anomaly Detector Memory Usage");
    core::CMemoryDebug::dynamicSize("m_Model", m_Model, mem);
}

std::size_t CAnomalyDetector::memoryUsage() const {
    // We only account for the model in CResourceMonitor,
    // so we just include that here.
    std::size_t mem = core::CMemory::dynamicSize(m_Model);
    return mem;
}

const core_t::TTime& CAnomalyDetector::lastBucketEndTime() const {
    return m_LastBucketEndTime;
}

core_t::TTime& CAnomalyDetector::lastBucketEndTime() {
    return m_LastBucketEndTime;
}

core_t::TTime CAnomalyDetector::modelBucketLength() const {
    return m_ModelConfig.bucketLength();
}

std::string CAnomalyDetector::description() const {
    auto beginInfluencers = m_DataGatherer->beginInfluencers();
    auto endInfluencers = m_DataGatherer->endInfluencers();
    return m_DataGatherer->description() +
           (m_DataGatherer->partitionFieldValue().empty() ? "" : "/") +
           m_DataGatherer->partitionFieldValue() +
           (beginInfluencers != endInfluencers
                ? (" " + core::CContainerPrinter::print(beginInfluencers, endInfluencers))
                : "");
}

void CAnomalyDetector::timeNow(core_t::TTime time) {
    m_DataGatherer->timeNow(time);
}

void CAnomalyDetector::skipSampling(core_t::TTime endTime) {
    m_Model->skipSampling(endTime);
    m_LastBucketEndTime = endTime;
}

template<typename SAMPLE_FUNC, typename LAST_SAMPLED_BUCKET_UPDATE_FUNC>
void CAnomalyDetector::buildResultsHelper(core_t::TTime bucketStartTime,
                                          core_t::TTime bucketEndTime,
                                          SAMPLE_FUNC sampleFunc,
                                          LAST_SAMPLED_BUCKET_UPDATE_FUNC lastSampledBucketUpdateFunc,
                                          CHierarchicalResults& results) {
    core_t::TTime bucketLength = m_ModelConfig.bucketLength();

    LOG_TRACE(<< "sample: m_DetectorKey = '" << this->description() << "', bucketStartTime = "
              << bucketStartTime << ", bucketEndTime = " << bucketEndTime);

    // Update the statistical models.
    sampleFunc(bucketStartTime, bucketEndTime);

    LOG_TRACE(<< "detect: m_DetectorKey = '" << this->description() << "'");

    CSearchKey key = m_DataGatherer->searchKey();
    LOG_TRACE(<< "OutputResults, for " << key.toCue());

    if (m_Model->addResults(m_DetectorIndex, bucketStartTime, bucketEndTime,
                            10, // TODO max number of attributes
                            results)) {
        if (bucketEndTime % bucketLength == 0) {
            lastSampledBucketUpdateFunc(bucketEndTime);
        }
    }
}

void CAnomalyDetector::updateLastSampledBucket(core_t::TTime bucketEndTime) {
    m_LastBucketEndTime = std::max(m_LastBucketEndTime, bucketEndTime);
}

void CAnomalyDetector::noUpdateLastSampledBucket(core_t::TTime /*bucketEndTime*/) const {
    // Do nothing
}

std::string CAnomalyDetector::toCue() const {
    return m_DataGatherer->searchKey().toCue() + m_DataGatherer->searchKey().CUE_DELIMITER +
           m_DataGatherer->partitionFieldValue();
}

std::string CAnomalyDetector::debug() const {
    return m_DataGatherer->searchKey().debug() + '/' + m_DataGatherer->partitionFieldValue();
}

bool CAnomalyDetector::isSimpleCount() const {
    return false;
}

void CAnomalyDetector::initSimpleCounting() {
    bool addedPerson = false;
    m_DataGatherer->addPerson(COUNT_NAME, m_Limits.resourceMonitor(), addedPerson);
}

const CAnomalyDetector::TModelPtr& CAnomalyDetector::model() const {
    return m_Model;
}

CAnomalyDetector::TModelPtr& CAnomalyDetector::model() {
    return m_Model;
}

std::ostream& operator<<(std::ostream& strm, const CAnomalyDetector& detector) {
    strm << detector.m_DataGatherer->searchKey() << '/'
         << detector.m_DataGatherer->partitionFieldValue();
    return strm;
}
}
}
