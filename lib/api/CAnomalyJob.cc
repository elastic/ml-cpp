/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CAnomalyJob.h>

#include <core/CDataAdder.h>
#include <core/CDataSearcher.h>
#include <core/CFunctional.h>
#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CLogger.h>
#include <core/CPersistUtils.h>
#include <core/CProgramCounters.h>
#include <core/CScopedRapidJsonPoolAllocator.h>
#include <core/CStateCompressor.h>
#include <core/CStateDecompressor.h>
#include <core/CStringUtils.h>
#include <core/CTimeUtils.h>
#include <core/Constants.h>
#include <core/UnwrapRef.h>

#include <maths/CIntegerTools.h>
#include <maths/COrderings.h>
#include <maths/CTools.h>

#include <model/CAnomalyScore.h>
#include <model/CForecastDataSink.h>
#include <model/CHierarchicalResultsAggregator.h>
#include <model/CHierarchicalResultsPopulator.h>
#include <model/CHierarchicalResultsProbabilityFinalizer.h>
#include <model/CLimits.h>
#include <model/CSearchKey.h>
#include <model/CSimpleCountDetector.h>
#include <model/CStringStore.h>

#include <api/CBackgroundPersister.h>
#include <api/CConfigUpdater.h>
#include <api/CFieldConfig.h>
#include <api/CHierarchicalResultsWriter.h>
#include <api/CJsonOutputWriter.h>
#include <api/CModelPlotDataJsonWriter.h>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <fstream>
#include <sstream>
#include <string>

namespace ml {
namespace api {

// We use short field names to reduce the state size
namespace {
using TStrCRef = std::reference_wrapper<const std::string>;

//! Convert a (string, key) pair to something readable.
template<typename T>
inline std::string pairDebug(const T& t) {
    return ml::core::unwrap_ref(t.second).debug() + '/' + ml::core::unwrap_ref(t.first);
}

const std::string TOP_LEVEL_DETECTOR_TAG("detector"); // do not shorten this
const std::string RESULTS_AGGREGATOR_TAG("aggregator");
const std::string TIME_TAG("a");
const std::string VERSION_TAG("b");
const std::string KEY_TAG("c");
const std::string PARTITION_FIELD_TAG("d");
const std::string DETECTOR_TAG("e");

// This is no longer used - removed in 6.6
// const std::string HIERARCHICAL_RESULTS_TAG("f");
const std::string LATEST_RECORD_TIME_TAG("h");

// This is no longer used - removed in 6.6
// const std::string MODEL_PLOT_TAG("i");

const std::string LAST_RESULTS_TIME_TAG("j");
const std::string INTERIM_BUCKET_CORRECTOR_TAG("k");

//! The minimum version required to read the state corresponding to a model snapshot.
//! This should be updated every time there is a breaking change to the model state.
const std::string MODEL_SNAPSHOT_MIN_VERSION("6.4.0");

// Persist state as JSON with meaningful tag names.
class CReadableJsonStatePersistInserter : public core::CJsonStatePersistInserter {
public:
    explicit CReadableJsonStatePersistInserter(std::ostream& outputStream)
        : core::CJsonStatePersistInserter(outputStream) {}
    virtual bool readableTags() const { return true; }
};
}

// Statics
const std::string CAnomalyJob::ML_STATE_INDEX(".ml-state");
const std::string CAnomalyJob::STATE_TYPE("model_state");
const std::string CAnomalyJob::DEFAULT_TIME_FIELD_NAME("time");
const std::string CAnomalyJob::EMPTY_STRING;

const CAnomalyJob::TAnomalyDetectorPtr CAnomalyJob::NULL_DETECTOR;

CAnomalyJob::CAnomalyJob(const std::string& jobId,
                         model::CLimits& limits,
                         CFieldConfig& fieldConfig,
                         model::CAnomalyDetectorModelConfig& modelConfig,
                         core::CJsonOutputStreamWrapper& outputStream,
                         const TPersistCompleteFunc& persistCompleteFunc,
                         CBackgroundPersister* periodicPersister,
                         core_t::TTime maxQuantileInterval,
                         const std::string& timeFieldName,
                         const std::string& timeFieldFormat,
                         size_t maxAnomalyRecords)
    : m_JobId(jobId), m_Limits(limits), m_OutputStream(outputStream),
      m_ForecastRunner(m_JobId, m_OutputStream, limits.resourceMonitor()),
      m_JsonOutputWriter(m_JobId, m_OutputStream), m_FieldConfig(fieldConfig),
      m_ModelConfig(modelConfig), m_NumRecordsHandled(0),
      m_LastFinalisedBucketEndTime(0), m_PersistCompleteFunc(persistCompleteFunc),
      m_TimeFieldName(timeFieldName), m_TimeFieldFormat(timeFieldFormat),
      m_MaxDetectors(std::numeric_limits<size_t>::max()),
      m_PeriodicPersister(periodicPersister),
      m_MaxQuantileInterval(maxQuantileInterval),
      m_LastNormalizerPersistTime(core::CTimeUtils::now()), m_LatestRecordTime(0),
      m_LastResultsTime(0), m_Aggregator(modelConfig), m_Normalizer(modelConfig) {
    m_JsonOutputWriter.limitNumberRecords(maxAnomalyRecords);

    m_Limits.resourceMonitor().memoryUsageReporter(std::bind(
        &CJsonOutputWriter::reportMemoryUsage, &m_JsonOutputWriter, std::placeholders::_1));
}

CAnomalyJob::~CAnomalyJob() {
    m_ForecastRunner.finishForecasts();
}

void CAnomalyJob::newOutputStream() {
    m_JsonOutputWriter.newOutputStream();
}

COutputHandler& CAnomalyJob::outputHandler() {
    return m_JsonOutputWriter;
}

bool CAnomalyJob::handleRecord(const TStrStrUMap& dataRowFields) {
    // Non-empty control fields take precedence over everything else
    TStrStrUMapCItr iter = dataRowFields.find(CONTROL_FIELD_NAME);
    if (iter != dataRowFields.end() && !iter->second.empty()) {
        return this->handleControlMessage(iter->second);
    }

    core_t::TTime time(0);
    iter = dataRowFields.find(m_TimeFieldName);
    if (iter == dataRowFields.end()) {
        ++core::CProgramCounters::counter(counter_t::E_TSADNumberRecordsNoTimeField);
        LOG_ERROR(<< "Found record with no " << m_TimeFieldName << " field:"
                  << core_t::LINE_ENDING << this->debugPrintRecord(dataRowFields));
        return true;
    }
    if (m_TimeFieldFormat.empty()) {
        if (core::CStringUtils::stringToType(iter->second, time) == false) {
            ++core::CProgramCounters::counter(counter_t::E_TSADNumberTimeFieldConversionErrors);
            LOG_ERROR(<< "Cannot interpret " << m_TimeFieldName
                      << " field in record:" << core_t::LINE_ENDING
                      << this->debugPrintRecord(dataRowFields));
            return true;
        }
    } else {
        // Use this library function instead of raw strptime() as it works
        // around many operating system specific issues.
        if (core::CTimeUtils::strptime(m_TimeFieldFormat, iter->second, time) == false) {
            ++core::CProgramCounters::counter(counter_t::E_TSADNumberTimeFieldConversionErrors);
            LOG_ERROR(<< "Cannot interpret " << m_TimeFieldName << " field using format "
                      << m_TimeFieldFormat << " in record:" << core_t::LINE_ENDING
                      << this->debugPrintRecord(dataRowFields));
            return true;
        }
    }

    // This record must be within the specified latency. If latency
    // is zero, then it should be after the current bucket end. If
    // latency is non-zero, then it should be after the current bucket
    // end minus the latency.
    if (time < m_LastFinalisedBucketEndTime) {
        ++core::CProgramCounters::counter(counter_t::E_TSADNumberTimeOrderErrors);
        std::ostringstream ss;
        ss << "Records must be in ascending time order. "
           << "Record '" << this->debugPrintRecord(dataRowFields) << "' time "
           << time << " is before bucket time " << m_LastFinalisedBucketEndTime;
        LOG_ERROR(<< ss.str());
        return true;
    }

    this->outputBucketResultsUntil(time);

    if (m_DetectorKeys.empty()) {
        this->populateDetectorKeys(m_FieldConfig, m_DetectorKeys);
    }

    for (std::size_t i = 0u; i < m_DetectorKeys.size(); ++i) {
        const std::string& partitionFieldName(m_DetectorKeys[i].partitionFieldName());

        // An empty partitionFieldName means no partitioning
        TStrStrUMapCItr itr = partitionFieldName.empty()
                                  ? dataRowFields.end()
                                  : dataRowFields.find(partitionFieldName);
        const std::string& partitionFieldValue(
            itr == dataRowFields.end() ? EMPTY_STRING : itr->second);

        // TODO - should usenull apply to the partition field too?

        const TAnomalyDetectorPtr& detector = this->detectorForKey(
            false, // not restoring
            time, m_DetectorKeys[i], partitionFieldValue, m_Limits.resourceMonitor());
        if (detector == nullptr) {
            // There wasn't enough memory to create the detector
            continue;
        }

        this->addRecord(detector, time, dataRowFields);
    }

    ++core::CProgramCounters::counter(counter_t::E_TSADNumberApiRecordsHandled);

    ++m_NumRecordsHandled;
    m_LatestRecordTime = std::max(m_LatestRecordTime, time);

    return true;
}

void CAnomalyJob::finalise() {
    // Persist final state of normalizer iff an input record has been handled or time has been advanced.
    if (this->isPersistenceNeeded("quantiles state")) {
        m_JsonOutputWriter.persistNormalizer(m_Normalizer, m_LastNormalizerPersistTime);
    }

    // Prune the models so that the final persisted state is as neat as possible
    this->pruneAllModels();

    this->refreshMemoryAndReport();

    // Wait for any ongoing periodic persist to complete, so that the data adder
    // is not used by both a periodic periodic persist and final persist at the
    // same time
    if (m_PeriodicPersister != nullptr) {
        m_PeriodicPersister->waitForIdle();
    }
}

bool CAnomalyJob::initNormalizer(const std::string& quantilesStateFile) {
    std::ifstream inputStream(quantilesStateFile.c_str());
    return m_Normalizer.fromJsonStream(inputStream) ==
           model::CHierarchicalResultsNormalizer::E_Ok;
}

uint64_t CAnomalyJob::numRecordsHandled() const {
    return m_NumRecordsHandled;
}

void CAnomalyJob::description() const {
    if (m_Detectors.empty()) {
        return;
    }

    TKeyCRefAnomalyDetectorPtrPrVec detectors;
    this->sortedDetectors(detectors);

    LOG_INFO(<< "Anomaly detectors:");
    TStrCRef partition = detectors[0].first.first;
    LOG_INFO(<< "\tpartition " << partition.get());
    LOG_INFO(<< "\t\tkey " << detectors[0].first.second.get());
    LOG_INFO(<< "\t\t\t" << detectors[0].second->description());
    for (std::size_t i = 1u; i < detectors.size(); ++i) {
        if (detectors[i].first.first.get() != partition.get()) {
            partition = detectors[i].first.first;
            LOG_INFO(<< "\tpartition " << partition.get());
        }
        LOG_INFO(<< "\t\tkey " << detectors[i].first.second.get());
        LOG_INFO(<< "\t\t\t" << detectors[i].second->description());
    }
}

void CAnomalyJob::descriptionAndDebugMemoryUsage() const {
    if (m_Detectors.empty()) {
        LOG_INFO(<< "No detectors");
        return;
    }

    TKeyCRefAnomalyDetectorPtrPrVec detectors;
    this->sortedDetectors(detectors);

    std::ostringstream ss;
    ss << "Anomaly detectors:" << std::endl;
    TStrCRef partition = detectors[0].first.first;
    ss << "\tpartition " << partition.get() << std::endl;
    ss << "\t\tkey " << detectors[0].first.second.get() << std::endl;
    ss << "\t\t\t" << detectors[0].second->description() << std::endl;
    detectors[0].second->showMemoryUsage(ss);

    for (std::size_t i = 1u; i < detectors.size(); ++i) {
        ss << std::endl;
        if (detectors[i].first.first.get() != partition.get()) {
            partition = detectors[i].first.first;
            ss << "\tpartition " << partition.get() << std::endl;
        }
        ss << "\t\tkey " << detectors[i].first.second.get() << std::endl;
        ss << "\t\t\t" << detectors[i].second->description() << std::endl;
        detectors[i].second->showMemoryUsage(ss);
    }
    LOG_INFO(<< ss.str());
}

const CAnomalyJob::SRestoredStateDetail& CAnomalyJob::restoreStateStatus() const {
    return m_RestoredStateDetail;
}

bool CAnomalyJob::handleControlMessage(const std::string& controlMessage) {
    if (controlMessage.empty()) {
        LOG_ERROR(<< "Programmatic error - handleControlMessage should only be "
                     "called with non-empty control messages");
        return false;
    }

    switch (controlMessage[0]) {
    case ' ':
        // Spaces are just used to fill the buffers and force prior messages
        // through the system - we don't need to do anything else
        LOG_TRACE(<< "Received space control message of length "
                  << controlMessage.length());
        break;
    case CONTROL_FIELD_NAME_CHAR:
        // Silent no-op.  This is a simple way to ignore repeated header
        // rows in input.
        break;
    case 'f':
        // Flush ID comes after the initial f
        this->acknowledgeFlush(controlMessage.substr(1));
        break;
    case 'i':
        this->generateInterimResults(controlMessage);
        break;
    case 'r':
        this->resetBuckets(controlMessage);
        break;
    case 's':
        this->skipTime(controlMessage.substr(1));
        break;
    case 't':
        this->advanceTime(controlMessage.substr(1));
        break;
    case 'u':
        this->updateConfig(controlMessage.substr(1));
        break;
    case 'p':
        this->doForecast(controlMessage);
        break;
    case 'w': {
        if (m_PeriodicPersister != nullptr) {
            m_PeriodicPersister->startBackgroundPersist();
        }
    } break;
    default:
        LOG_WARN(<< "Ignoring unknown control message of length "
                 << controlMessage.length() << " beginning with '"
                 << controlMessage[0] << '\'');
        // Don't return false here (for the time being at least), as it
        // seems excessive to cause the entire job to fail
        break;
    }

    return true;
}

void CAnomalyJob::acknowledgeFlush(const std::string& flushId) {
    if (flushId.empty()) {
        LOG_ERROR(<< "Received flush control message with no ID");
    } else {
        LOG_TRACE(<< "Received flush control message with ID " << flushId);
    }
    m_JsonOutputWriter.acknowledgeFlush(flushId, m_LastFinalisedBucketEndTime);
}

void CAnomalyJob::updateConfig(const std::string& config) {
    LOG_DEBUG(<< "Received update config request: " << config);
    CConfigUpdater configUpdater(m_FieldConfig, m_ModelConfig);
    if (configUpdater.update(config) == false) {
        LOG_ERROR(<< "Failed to update configuration");
    }
}

void CAnomalyJob::advanceTime(const std::string& time_) {
    if (time_.empty()) {
        LOG_ERROR(<< "Received request to advance time with no time");
        return;
    }

    core_t::TTime time(0);
    if (core::CStringUtils::stringToType(time_, time) == false) {
        LOG_ERROR(<< "Received request to advance time to invalid time " << time_);
        return;
    }

    if (m_LastFinalisedBucketEndTime == 0) {
        LOG_DEBUG(<< "Manually advancing time to " << time
                  << " before any valid data has been seen");
    } else {
        LOG_TRACE(<< "Received request to advance time to " << time);
    }

    m_TimeAdvanced = true;

    this->outputBucketResultsUntil(time);

    this->timeNow(time);
}

bool CAnomalyJob::isPersistenceNeeded(const std::string& description) const {
    if ((m_NumRecordsHandled == 0) && (m_TimeAdvanced == false)) {
        LOG_DEBUG(<< "Will not attempt to persist " << description
                  << ". Zero records were handled and time has not been advanced.");
        return false;
    }

    return true;
}

void CAnomalyJob::outputBucketResultsUntil(core_t::TTime time) {
    // If the bucket time has increased, output results for all field names
    core_t::TTime bucketLength = m_ModelConfig.bucketLength();
    core_t::TTime latency = m_ModelConfig.latency();

    if (m_LastFinalisedBucketEndTime == 0) {
        m_LastFinalisedBucketEndTime =
            std::max(m_LastFinalisedBucketEndTime,
                     maths::CIntegerTools::floor(time, bucketLength) - latency);
    }

    m_Normalizer.resetBigChange();

    for (core_t::TTime lastBucketEndTime = m_LastFinalisedBucketEndTime;
         lastBucketEndTime + bucketLength + latency <= time;
         lastBucketEndTime += bucketLength) {
        this->outputResults(lastBucketEndTime);
        m_Limits.resourceMonitor().decreaseMargin(bucketLength);
        m_Limits.resourceMonitor().sendMemoryUsageReportIfSignificantlyChanged(lastBucketEndTime);
        m_LastFinalisedBucketEndTime = lastBucketEndTime + bucketLength;

        // Check for periodic persistence immediately after calculating results
        // for the last bucket but before adding the first piece of data for the
        // next bucket
        if (m_PeriodicPersister != nullptr) {
            m_PeriodicPersister->startBackgroundPersistIfAppropriate();
        }
    }

    if (m_Normalizer.hasLastUpdateCausedBigChange() ||
        (m_MaxQuantileInterval > 0 &&
         core::CTimeUtils::now() > m_LastNormalizerPersistTime + m_MaxQuantileInterval)) {
        m_JsonOutputWriter.persistNormalizer(m_Normalizer, m_LastNormalizerPersistTime);
    }
}

void CAnomalyJob::skipTime(const std::string& time_) {
    if (time_.empty()) {
        LOG_ERROR(<< "Received request to skip time with no time");
        return;
    }

    core_t::TTime time(0);
    if (core::CStringUtils::stringToType(time_, time) == false) {
        LOG_ERROR(<< "Received request to skip time to invalid time " << time_);
        return;
    }

    this->skipSampling(maths::CIntegerTools::ceil(time, m_ModelConfig.bucketLength()));
}

void CAnomalyJob::skipSampling(core_t::TTime endTime) {
    LOG_INFO(<< "Skipping time to: " << endTime);

    for (const auto& detector_ : m_Detectors) {
        model::CAnomalyDetector* detector(detector_.second.get());
        if (detector == nullptr) {
            LOG_ERROR(<< "Unexpected NULL pointer for key '"
                      << pairDebug(detector_.first) << '\'');
            continue;
        }
        detector->skipSampling(endTime);
    }

    m_LastFinalisedBucketEndTime = endTime;
}

void CAnomalyJob::timeNow(core_t::TTime time) {
    for (const auto& detector_ : m_Detectors) {
        model::CAnomalyDetector* detector(detector_.second.get());
        if (detector == nullptr) {
            LOG_ERROR(<< "Unexpected NULL pointer for key '"
                      << pairDebug(detector_.first) << '\'');
            continue;
        }
        detector->timeNow(time);
    }
}

void CAnomalyJob::generateInterimResults(const std::string& controlMessage) {
    LOG_TRACE(<< "Generating interim results");

    if (m_LastFinalisedBucketEndTime == 0) {
        LOG_TRACE(<< "Cannot create interim results having seen data for less than one bucket ever");
        return;
    }

    core_t::TTime start = m_LastFinalisedBucketEndTime;
    core_t::TTime end = m_LastFinalisedBucketEndTime +
                        (m_ModelConfig.latencyBuckets() + 1) * m_ModelConfig.bucketLength();

    if (this->parseTimeRangeInControlMessage(controlMessage, start, end)) {
        LOG_TRACE(<< "Time range for results: " << start << " : " << end);
        this->outputResultsWithinRange(true, start, end);
    }
}

bool CAnomalyJob::parseTimeRangeInControlMessage(const std::string& controlMessage,
                                                 core_t::TTime& start,
                                                 core_t::TTime& end) {
    using TStrVec = core::CStringUtils::TStrVec;
    TStrVec tokens;
    std::string remainder;
    core::CStringUtils::tokenise(" ", controlMessage.substr(1, std::string::npos),
                                 tokens, remainder);
    if (!remainder.empty()) {
        tokens.push_back(remainder);
    }
    std::size_t tokensSize = tokens.size();
    if (tokensSize == 0) {
        // Default range
        return true;
    }
    if (tokensSize != 2) {
        LOG_ERROR(<< "Control message " << controlMessage << " has " << tokensSize
                  << " parameters when only zero or two are allowed.");
        return false;
    }
    if (core::CStringUtils::stringToType(tokens[0], start) &&
        core::CStringUtils::stringToType(tokens[1], end)) {
        return true;
    }
    LOG_ERROR(<< "Cannot parse control message: " << controlMessage);
    return false;
}

void CAnomalyJob::doForecast(const std::string& controlMessage) {
    // make a copy of the detectors vector, note: this is a shallow, not a deep copy
    TAnomalyDetectorPtrVec detectorVector;
    this->detectors(detectorVector);

    // push request into forecast queue, validates
    if (!m_ForecastRunner.pushForecastJob(controlMessage, detectorVector, m_LastResultsTime)) {
        // ForecastRunner already logged about it and send a status, so no need to log at info here
        LOG_DEBUG(<< "Forecast request failed");
    }
}

void CAnomalyJob::outputResults(core_t::TTime bucketStartTime) {
    core::CStopWatch timer(true);

    core_t::TTime bucketLength = m_ModelConfig.bucketLength();

    model::CHierarchicalResults results;
    TModelPlotDataVec modelPlotData;

    TKeyCRefAnomalyDetectorPtrPrVec detectors;
    this->sortedDetectors(detectors);

    for (const auto& detector_ : detectors) {
        model::CAnomalyDetector* detector(detector_.second.get());
        if (detector == nullptr) {
            LOG_ERROR(<< "Unexpected NULL pointer for key '"
                      << pairDebug(detector_.first) << '\'');
            continue;
        }
        detector->buildResults(bucketStartTime, bucketStartTime + bucketLength, results);
        detector->releaseMemory(bucketStartTime - m_ModelConfig.samplingAgeCutoff());

        this->generateModelPlot(bucketStartTime, bucketStartTime + bucketLength,
                                *detector, modelPlotData);
    }

    if (!results.empty()) {
        results.buildHierarchy();

        this->updateAggregatorAndAggregate(false, results);

        model::CHierarchicalResultsProbabilityFinalizer finalizer;
        results.bottomUpBreadthFirst(finalizer);
        results.pivotsBottomUpBreadthFirst(finalizer);

        model::CHierarchicalResultsPopulator populator(m_Limits);
        results.bottomUpBreadthFirst(populator);
        results.pivotsBottomUpBreadthFirst(populator);

        this->updateNormalizerAndNormalizeResults(false, results);
    }

    uint64_t processingTime = timer.stop();

    // Model plots must be written first so the Java persists them
    // once the bucket result is processed
    this->writeOutModelPlot(modelPlotData);
    this->writeOutResults(false, results, bucketStartTime, processingTime);

    m_Limits.resourceMonitor().pruneIfRequired(bucketStartTime);
    model::CStringStore::tidyUpNotThreadSafe();
}

void CAnomalyJob::outputInterimResults(core_t::TTime bucketStartTime) {
    core::CStopWatch timer(true);

    core_t::TTime bucketLength = m_ModelConfig.bucketLength();

    model::CHierarchicalResults results;
    results.setInterim();

    TKeyCRefAnomalyDetectorPtrPrVec detectors;
    this->sortedDetectors(detectors);

    for (const auto& detector_ : detectors) {
        model::CAnomalyDetector* detector(detector_.second.get());
        if (detector == nullptr) {
            LOG_ERROR(<< "Unexpected NULL pointer for key '"
                      << pairDebug(detector_.first) << '\'');
            continue;
        }
        detector->buildInterimResults(bucketStartTime, bucketStartTime + bucketLength, results);
    }

    if (!results.empty()) {
        results.buildHierarchy();

        this->updateAggregatorAndAggregate(true, results);

        model::CHierarchicalResultsProbabilityFinalizer finalizer;
        results.bottomUpBreadthFirst(finalizer);
        results.pivotsBottomUpBreadthFirst(finalizer);

        model::CHierarchicalResultsPopulator populator(m_Limits);
        results.bottomUpBreadthFirst(populator);
        results.pivotsBottomUpBreadthFirst(populator);

        this->updateNormalizerAndNormalizeResults(true, results);
    }

    uint64_t processingTime = timer.stop();
    this->writeOutResults(true, results, bucketStartTime, processingTime);
}

void CAnomalyJob::writeOutResults(bool interim,
                                  model::CHierarchicalResults& results,
                                  core_t::TTime bucketTime,
                                  uint64_t processingTime) {
    if (!results.empty()) {
        LOG_TRACE(<< "Got results object here: " << results.root()->s_RawAnomalyScore
                  << " / " << results.root()->s_NormalizedAnomalyScore
                  << ", count " << results.resultCount() << " at " << bucketTime);

        using TScopedAllocator = ml::core::CScopedRapidJsonPoolAllocator<CJsonOutputWriter>;
        static const std::string ALLOCATOR_ID("CAnomalyJob::writeOutResults");
        TScopedAllocator scopedAllocator(ALLOCATOR_ID, m_JsonOutputWriter);

        api::CHierarchicalResultsWriter writer(
            m_Limits,
            std::bind(&CJsonOutputWriter::acceptResult, &m_JsonOutputWriter,
                      std::placeholders::_1),
            std::bind(&CJsonOutputWriter::acceptInfluencer, &m_JsonOutputWriter,
                      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
        results.bottomUpBreadthFirst(writer);
        results.pivotsBottomUpBreadthFirst(writer);

        // Add the bucketTime bucket influencer.
        // Note that the influencer will only be accepted if there are records.
        m_JsonOutputWriter.acceptBucketTimeInfluencer(
            bucketTime, results.root()->s_AnnotatedProbability.s_Probability,
            results.root()->s_RawAnomalyScore, results.root()->s_NormalizedAnomalyScore);

        if (m_JsonOutputWriter.endOutputBatch(interim, processingTime) == false) {
            LOG_ERROR(<< "Problem writing anomaly output");
        }
        m_LastResultsTime = bucketTime;
    }
}

void CAnomalyJob::resetBuckets(const std::string& controlMessage) {
    if (controlMessage.length() == 1) {
        LOG_ERROR(<< "Received reset buckets control message without time range");
        return;
    }
    core_t::TTime start = 0;
    core_t::TTime end = 0;
    if (this->parseTimeRangeInControlMessage(controlMessage, start, end)) {
        core_t::TTime bucketLength = m_ModelConfig.bucketLength();
        core_t::TTime time = maths::CIntegerTools::floor(start, bucketLength);
        core_t::TTime bucketEnd = maths::CIntegerTools::ceil(end, bucketLength);
        while (time < bucketEnd) {
            for (const auto& detector_ : m_Detectors) {
                model::CAnomalyDetector* detector = detector_.second.get();
                if (detector == nullptr) {
                    LOG_ERROR(<< "Unexpected NULL pointer for key '"
                              << pairDebug(detector_.first) << '\'');
                    continue;
                }
                LOG_TRACE(<< "Resetting bucket = " << time);
                detector->resetBucket(time);
            }
            time += bucketLength;
        }
    }
}

bool CAnomalyJob::restoreState(core::CDataSearcher& restoreSearcher,
                               core_t::TTime& completeToTime) {
    // Pass on the request in case we're chained
    if (this->outputHandler().restoreState(restoreSearcher, completeToTime) == false) {
        return false;
    }

    size_t numDetectors(0);
    try {
        // Restore from Elasticsearch compressed data
        core::CStateDecompressor decompressor(restoreSearcher);
        decompressor.setStateRestoreSearch(ML_STATE_INDEX);

        core::CDataSearcher::TIStreamP strm(decompressor.search(1, 1));
        if (strm == nullptr) {
            LOG_ERROR(<< "Unable to connect to data store");
            return false;
        }

        if (strm->bad()) {
            LOG_ERROR(<< "State restoration search returned bad stream");
            return false;
        }

        if (strm->fail()) {
            // This is fatal. If the stream exists and has failed then state is missing
            LOG_ERROR(<< "State restoration search returned failed stream");
            return false;
        }

        // We're dealing with streaming JSON state
        core::CJsonStateRestoreTraverser traverser(*strm);

        if (this->restoreState(traverser, completeToTime, numDetectors) == false) {
            LOG_ERROR(<< "Failed to restore detectors");
            return false;
        }
        LOG_DEBUG(<< "Finished restoration, with " << numDetectors << " detectors");

        if (numDetectors == 1 && m_Detectors.empty()) {
            // non fatal error
            m_RestoredStateDetail.s_RestoredStateStatus = E_NoDetectorsRecovered;
            return true;
        }

        if (completeToTime > 0) {
            core_t::TTime lastBucketEndTime(maths::CIntegerTools::ceil(
                completeToTime, m_ModelConfig.bucketLength()));

            for (const auto& detector_ : m_Detectors) {
                model::CAnomalyDetector* detector(detector_.second.get());
                if (detector == nullptr) {
                    LOG_ERROR(<< "Unexpected NULL pointer for key '"
                              << pairDebug(detector_.first) << '\'');
                    continue;
                }

                LOG_DEBUG(<< "Setting lastBucketEndTime to " << lastBucketEndTime
                          << " in detector for '" << detector->description() << '\'');
                detector->lastBucketEndTime() = lastBucketEndTime;
            }
        } else {
            if (!m_Detectors.empty()) {
                LOG_ERROR(<< "Inconsistency - " << m_Detectors.size()
                          << " detectors have been restored but completeToTime is "
                          << completeToTime);
            }
        }
    } catch (std::exception& e) {
        LOG_ERROR(<< "Failed to restore state! " << e.what());
        return false;
    }

    return true;
}

bool CAnomalyJob::restoreState(core::CStateRestoreTraverser& traverser,
                               core_t::TTime& completeToTime,
                               std::size_t& numDetectors) {
    m_RestoredStateDetail.s_RestoredStateStatus = E_Failure;
    m_RestoredStateDetail.s_Extra = boost::none;

    // Call name() to prime the traverser if it hasn't started
    traverser.name();
    if (traverser.isEof()) {
        m_RestoredStateDetail.s_RestoredStateStatus = E_NoDetectorsRecovered;
        LOG_ERROR(<< "Expected persisted state but no state exists");
        return false;
    }

    core_t::TTime lastBucketEndTime(0);
    if (traverser.name() != TIME_TAG ||
        core::CStringUtils::stringToType(traverser.value(), lastBucketEndTime) == false) {
        m_RestoredStateDetail.s_RestoredStateStatus = E_UnexpectedTag;
        LOG_ERROR(<< "Cannot restore anomaly detector - '" << TIME_TAG << "' element expected but found "
                  << traverser.name() << '=' << traverser.value());
        return false;
    }
    m_LastFinalisedBucketEndTime = lastBucketEndTime;

    if (lastBucketEndTime > completeToTime) {
        LOG_INFO(<< "Processing is already complete to time " << lastBucketEndTime);
        completeToTime = lastBucketEndTime;
    }

    if ((traverser.next() == false) || (traverser.name() != VERSION_TAG)) {
        m_RestoredStateDetail.s_RestoredStateStatus = E_UnexpectedTag;
        LOG_ERROR(<< "Cannot restore anomaly detector " << VERSION_TAG << " was expected");
        return false;
    }

    const std::string& stateVersion = traverser.value();
    if (stateVersion != model::CAnomalyDetector::STATE_VERSION) {
        m_RestoredStateDetail.s_RestoredStateStatus = E_IncorrectVersion;
        LOG_ERROR(<< "Restored anomaly detector state version is "
                  << stateVersion << " - ignoring it as current state version is "
                  << model::CAnomalyDetector::STATE_VERSION);

        // This counts as successful restoration
        return true;
    }

    while (traverser.next()) {
        const std::string& name = traverser.name();
        if (name == INTERIM_BUCKET_CORRECTOR_TAG) {
            // Note that this has to be persisted and restored before any detectors.
            auto interimBucketCorrector = std::make_shared<model::CInterimBucketCorrector>(
                m_ModelConfig.bucketLength());
            if (traverser.traverseSubLevel(std::bind(
                    &model::CInterimBucketCorrector::acceptRestoreTraverser,
                    interimBucketCorrector.get(), std::placeholders::_1)) == false) {
                LOG_ERROR(<< "Cannot restore interim bucket corrector");
                return false;
            }
            m_ModelConfig.interimBucketCorrector(interimBucketCorrector);
        } else if (name == TOP_LEVEL_DETECTOR_TAG) {
            if (traverser.traverseSubLevel(std::bind(&CAnomalyJob::restoreSingleDetector,
                                                     this, std::placeholders::_1)) == false) {
                LOG_ERROR(<< "Cannot restore anomaly detector");
                return false;
            }
            ++numDetectors;
        } else if (name == RESULTS_AGGREGATOR_TAG) {
            if (traverser.traverseSubLevel(std::bind(
                    &model::CHierarchicalResultsAggregator::acceptRestoreTraverser,
                    &m_Aggregator, std::placeholders::_1)) == false) {
                LOG_ERROR(<< "Cannot restore results aggregator");
                return false;
            }
        } else if (name == LATEST_RECORD_TIME_TAG) {
            core::CPersistUtils::restore(LATEST_RECORD_TIME_TAG, m_LatestRecordTime, traverser);
        } else if (name == LAST_RESULTS_TIME_TAG) {
            core::CPersistUtils::restore(LAST_RESULTS_TIME_TAG, m_LastResultsTime, traverser);
        }
    }

    m_RestoredStateDetail.s_RestoredStateStatus = E_Success;

    return true;
}

bool CAnomalyJob::restoreSingleDetector(core::CStateRestoreTraverser& traverser) {
    if (traverser.name() != KEY_TAG) {
        LOG_ERROR(<< "Cannot restore anomaly detector - " << KEY_TAG << " element expected but found "
                  << traverser.name() << '=' << traverser.value());

        m_RestoredStateDetail.s_RestoredStateStatus = E_UnexpectedTag;
        return false;
    }

    model::CSearchKey key;
    if (traverser.traverseSubLevel(std::bind(&model::CAnomalyDetector::keyAcceptRestoreTraverser,
                                             std::placeholders::_1, std::ref(key))) == false) {
        LOG_ERROR(<< "Cannot restore anomaly detector - no key found in " << KEY_TAG);

        m_RestoredStateDetail.s_RestoredStateStatus = E_UnexpectedTag;
        return false;
    }

    if (traverser.next() == false) {
        LOG_ERROR(<< "Cannot restore anomaly detector - end of object reached when "
                  << PARTITION_FIELD_TAG << " was expected");

        m_RestoredStateDetail.s_RestoredStateStatus = E_UnexpectedTag;
        return false;
    }

    if (traverser.name() != PARTITION_FIELD_TAG) {
        LOG_ERROR(<< "Cannot restore anomaly detector - " << PARTITION_FIELD_TAG << " element expected but found "
                  << traverser.name() << '=' << traverser.value());

        m_RestoredStateDetail.s_RestoredStateStatus = E_UnexpectedTag;
        return false;
    }

    std::string partitionFieldValue;
    if (traverser.traverseSubLevel(std::bind(
            &model::CAnomalyDetector::partitionFieldAcceptRestoreTraverser,
            std::placeholders::_1, std::ref(partitionFieldValue))) == false) {
        LOG_ERROR(<< "Cannot restore anomaly detector - "
                     "no partition field value found in "
                  << PARTITION_FIELD_TAG);

        m_RestoredStateDetail.s_RestoredStateStatus = E_UnexpectedTag;
        return false;
    }

    if (traverser.next() == false) {
        LOG_ERROR(<< "Cannot restore anomaly detector - end of object reached when "
                  << DETECTOR_TAG << " was expected");

        m_RestoredStateDetail.s_RestoredStateStatus = E_UnexpectedTag;
        return false;
    }

    if (traverser.name() != DETECTOR_TAG) {
        LOG_ERROR(<< "Cannot restore anomaly detector - " << DETECTOR_TAG << " element expected but found "
                  << traverser.name() << '=' << traverser.value());

        m_RestoredStateDetail.s_RestoredStateStatus = E_UnexpectedTag;
        return false;
    }

    if (this->restoreDetectorState(key, partitionFieldValue, traverser) == false ||
        traverser.haveBadState()) {
        LOG_ERROR(<< "Delegated portion of anomaly detector restore failed");
        m_RestoredStateDetail.s_RestoredStateStatus = E_Failure;
        return false;
    }

    LOG_TRACE(<< "Restored state for " << key.toCue() << "/" << partitionFieldValue);
    return true;
}

bool CAnomalyJob::restoreDetectorState(const model::CSearchKey& key,
                                       const std::string& partitionFieldValue,
                                       core::CStateRestoreTraverser& traverser) {
    const TAnomalyDetectorPtr& detector =
        this->detectorForKey(true, // for restoring
                             0,    // time reset later
                             key, partitionFieldValue, m_Limits.resourceMonitor());
    if (!detector) {
        LOG_ERROR(<< "Detector with key '" << key.debug() << '/' << partitionFieldValue
                  << "' was not recreated on restore - "
                     "memory limit is too low to continue this job");

        m_RestoredStateDetail.s_RestoredStateStatus = E_MemoryLimitReached;
        return false;
    }

    LOG_DEBUG(<< "Restoring state for detector with key '" << key.debug() << '/'
              << partitionFieldValue << '\'');

    if (traverser.traverseSubLevel(std::bind(
            &model::CAnomalyDetector::acceptRestoreTraverser, detector.get(),
            std::cref(partitionFieldValue), std::placeholders::_1)) == false) {
        LOG_ERROR(<< "Error restoring anomaly detector for key '" << key.debug()
                  << '/' << partitionFieldValue << '\'');
        return false;
    }

    return true;
}

bool CAnomalyJob::persistResidualModelsState(core::CDataAdder& persister) {
    TKeyCRefAnomalyDetectorPtrPrVec detectors;
    this->sortedDetectors(detectors);

    // Persistence operates on a cached collection of counters rather than on the live counters directly.
    // This is in order that background persistence operates on a consistent set of counters however we
    // also must ensure that foreground persistence has access to an up-to-date cache of counters as well.
    core::CProgramCounters::cacheCounters();

    return this->persistResidualModelsState(detectors, persister);
}

bool CAnomalyJob::persistState(core::CDataAdder& persister) {
    if (m_PeriodicPersister != nullptr) {
        // This will not happen if finalise() was called before persisting state
        if (m_PeriodicPersister->isBusy()) {
            LOG_ERROR(<< "Cannot do final persistence of state - periodic "
                         "persister still busy");
            return false;
        }
    }

    // Pass on the request in case we're chained
    if (this->outputHandler().persistState(persister) == false) {
        return false;
    }

    if (m_LastFinalisedBucketEndTime == 0) {
        LOG_INFO(<< "Will not persist detectors as no results have been output");
        return true;
    }

    TKeyCRefAnomalyDetectorPtrPrVec detectors;
    this->sortedDetectors(detectors);
    std::string normaliserState;
    m_Normalizer.toJson(m_LastResultsTime, "api", normaliserState, true);

    // Persistence operates on a cached collection of counters rather than on the live counters directly.
    // This is in order that background persistence operates on a consistent set of counters however we
    // also must ensure that foreground persistence has access to an up-to-date cache of counters as well.
    core::CProgramCounters::cacheCounters();

    return this->persistState(
        "State persisted due to job close at ", m_LastFinalisedBucketEndTime, detectors,
        m_Limits.resourceMonitor().createMemoryUsageReport(
            m_LastFinalisedBucketEndTime - m_ModelConfig.bucketLength()),
        m_ModelConfig.interimBucketCorrector(), m_Aggregator, normaliserState,
        m_LatestRecordTime, m_LastResultsTime, persister);
}

bool CAnomalyJob::backgroundPersistState(CBackgroundPersister& backgroundPersister) {
    LOG_INFO(<< "Background persist starting data copy");

    // Pass arguments by value: this is what we want for
    // passing to a new thread.
    // Do NOT add std::ref wrappers around these arguments - they
    // MUST be copied for thread safety
    TBackgroundPersistArgsPtr args = std::make_shared<SBackgroundPersistArgs>(
        m_LastFinalisedBucketEndTime,
        m_Limits.resourceMonitor().createMemoryUsageReport(
            m_LastFinalisedBucketEndTime - m_ModelConfig.bucketLength()),
        m_ModelConfig.interimBucketCorrector(), m_Aggregator,
        m_LatestRecordTime, m_LastResultsTime);

    // The normaliser is non-copyable, so we have to make do with JSONifying it now;
    // it should be relatively fast though
    m_Normalizer.toJson(m_LastResultsTime, "api", args->s_NormalizerState, true);

    TKeyCRefAnomalyDetectorPtrPrVec& copiedDetectors = args->s_Detectors;
    copiedDetectors.reserve(m_Detectors.size());

    for (const auto& detector_ : m_Detectors) {
        model::CAnomalyDetector* detector(detector_.second.get());
        if (detector == nullptr) {
            LOG_ERROR(<< "Unexpected NULL pointer for key '"
                      << pairDebug(detector_.first) << '\'');
            continue;
        }
        model::CSearchKey::TStrCRefKeyCRefPr key(std::cref(detector_.first.first),
                                                 std::cref(detector_.first.second));
        if (detector->isSimpleCount()) {
            copiedDetectors.push_back(TKeyCRefAnomalyDetectorPtrPr(
                key, TAnomalyDetectorPtr(new model::CSimpleCountDetector(true, *detector))));
        } else {
            copiedDetectors.push_back(TKeyCRefAnomalyDetectorPtrPr(
                key, TAnomalyDetectorPtr(new model::CAnomalyDetector(true, *detector))));
        }
    }
    std::sort(copiedDetectors.begin(), copiedDetectors.end(),
              maths::COrderings::SFirstLess());

    if (backgroundPersister.addPersistFunc(std::bind(
            &CAnomalyJob::runBackgroundPersist, this, args, std::placeholders::_1)) == false) {
        LOG_ERROR(<< "Failed to add anomaly detector background persistence function");
        return false;
    }

    return true;
}

bool CAnomalyJob::runBackgroundPersist(TBackgroundPersistArgsPtr args,
                                       core::CDataAdder& persister) {
    if (!args) {
        LOG_ERROR(<< "Unexpected NULL pointer passed to background persist");
        return false;
    }

    return this->persistState("Periodic background persist at ", args->s_Time,
                              args->s_Detectors, args->s_ModelSizeStats,
                              args->s_InterimBucketCorrector, args->s_Aggregator,
                              args->s_NormalizerState, args->s_LatestRecordTime,
                              args->s_LastResultsTime, persister);
}

bool CAnomalyJob::persistResidualModelsState(const TKeyCRefAnomalyDetectorPtrPrVec& detectors,
                                             core::CDataAdder& persister) {
    try {
        core_t::TTime snapshotTimestamp(core::CTimeUtils::now());
        const std::string snapShotId(core::CStringUtils::typeToString(snapshotTimestamp));
        core::CDataAdder::TOStreamP strm = persister.addStreamed(
            ML_STATE_INDEX, m_JobId + '_' + STATE_TYPE + '_' + snapShotId);
        if (strm != nullptr) {
            {
                // The JSON inserter must be destroyed before the stream is complete
                CReadableJsonStatePersistInserter inserter(*strm);
                for (const auto& detector_ : detectors) {
                    const model::CAnomalyDetector* detector(detector_.second.get());
                    if (detector == nullptr) {
                        LOG_ERROR(<< "Unexpected NULL pointer for key '"
                                  << pairDebug(detector_.first) << '\'');
                        continue;
                    }

                    detector->persistResidualModelsState(inserter);

                    LOG_DEBUG(<< "Persisted state for '" << detector->description() << "'");
                }
            }

            if (persister.streamComplete(strm, true) == false || strm->bad()) {
                LOG_ERROR(<< "Failed to complete last persistence stream");
                return false;
            }
        }
    } catch (std::exception& e) {
        LOG_ERROR(<< "Failed to persist state! " << e.what());
        return false;
    }

    return true;
}

bool CAnomalyJob::persistState(const std::string& descriptionPrefix,
                               core_t::TTime lastFinalisedBucketEnd,
                               const TKeyCRefAnomalyDetectorPtrPrVec& detectors,
                               const model::CResourceMonitor::SResults& modelSizeStats,
                               const model::CInterimBucketCorrector& interimBucketCorrector,
                               const model::CHierarchicalResultsAggregator& aggregator,
                               const std::string& normalizerState,
                               core_t::TTime latestRecordTime,
                               core_t::TTime lastResultsTime,
                               core::CDataAdder& persister) {
    // Persist state for each detector separately by streaming
    try {
        core::CStateCompressor compressor(persister);

        core_t::TTime snapshotTimestamp(core::CTimeUtils::now());
        const std::string snapShotId(core::CStringUtils::typeToString(snapshotTimestamp));
        core::CDataAdder::TOStreamP strm = compressor.addStreamed(
            ML_STATE_INDEX, m_JobId + '_' + STATE_TYPE + '_' + snapShotId);
        if (strm != nullptr) {
            // IMPORTANT - this method can run in a background thread while the
            // analytics carries on processing new buckets in the main thread.
            // Therefore, this method must NOT access any member variables whose
            // values can change.  There should be no use of m_ variables in the
            // following code block.
            {
                // The JSON inserter must be destructed before the stream is complete
                core::CJsonStatePersistInserter inserter(*strm);
                inserter.insertValue(TIME_TAG, lastFinalisedBucketEnd);
                inserter.insertValue(VERSION_TAG, model::CAnomalyDetector::STATE_VERSION);
                inserter.insertLevel(
                    INTERIM_BUCKET_CORRECTOR_TAG,
                    std::bind(&model::CInterimBucketCorrector::acceptPersistInserter,
                              &interimBucketCorrector, std::placeholders::_1));

                for (const auto& detector_ : detectors) {
                    const model::CAnomalyDetector* detector(detector_.second.get());
                    if (detector == nullptr) {
                        LOG_ERROR(<< "Unexpected NULL pointer for key '"
                                  << pairDebug(detector_.first) << '\'');
                        continue;
                    }
                    inserter.insertLevel(
                        TOP_LEVEL_DETECTOR_TAG,
                        std::bind(&CAnomalyJob::persistIndividualDetector,
                                  std::cref(*detector), std::placeholders::_1));

                    LOG_DEBUG(<< "Persisted state for '" << detector->description() << "'");
                }

                inserter.insertLevel(RESULTS_AGGREGATOR_TAG,
                                     std::bind(&model::CHierarchicalResultsAggregator::acceptPersistInserter,
                                               &aggregator, std::placeholders::_1));

                core::CPersistUtils::persist(LATEST_RECORD_TIME_TAG,
                                             latestRecordTime, inserter);
                core::CPersistUtils::persist(LAST_RESULTS_TIME_TAG, lastResultsTime, inserter);
            }

            if (compressor.streamComplete(strm, true) == false || strm->bad()) {
                LOG_ERROR(<< "Failed to complete last persistence stream");
                return false;
            }

            if (m_PersistCompleteFunc) {
                CModelSnapshotJsonWriter::SModelSnapshotReport modelSnapshotReport{
                    MODEL_SNAPSHOT_MIN_VERSION, snapshotTimestamp,
                    descriptionPrefix + core::CTimeUtils::toIso8601(snapshotTimestamp),
                    snapShotId, compressor.numCompressedDocs(), modelSizeStats,
                    normalizerState, latestRecordTime,
                    // This needs to be the last final result time as it serves
                    // as the time after which all results are deleted when a
                    // model snapshot is reverted
                    lastFinalisedBucketEnd - m_ModelConfig.bucketLength()};

                m_PersistCompleteFunc(modelSnapshotReport);
            }
        }
    } catch (std::exception& e) {
        LOG_ERROR(<< "Failed to persist state! " << e.what());
        return false;
    }

    return true;
}

bool CAnomalyJob::periodicPersistState(CBackgroundPersister& persister) {
    // Pass on the request in case we're chained
    if (this->outputHandler().periodicPersistState(persister) == false) {
        return false;
    }

    // Prune the models so that the persisted state is as neat as possible
    this->pruneAllModels();

    // Make sure model size stats are up to date
    for (const auto& detector_ : m_Detectors) {
        model::CAnomalyDetector* detector = detector_.second.get();
        if (detector == nullptr) {
            LOG_ERROR(<< "Unexpected NULL pointer for key '"
                      << pairDebug(detector_.first) << '\'');
            continue;
        }
        m_Limits.resourceMonitor().forceRefresh(*detector);
    }

    return this->backgroundPersistState(persister);
}

void CAnomalyJob::updateAggregatorAndAggregate(bool isInterim,
                                               model::CHierarchicalResults& results) {
    m_Aggregator.refresh(m_ModelConfig);

    m_Aggregator.setJob(model::CHierarchicalResultsAggregator::E_Correct);

    // The equalizers are NOT updated with interim results.
    if (isInterim == false) {
        m_Aggregator.setJob(model::CHierarchicalResultsAggregator::E_UpdateAndCorrect);
        m_Aggregator.propagateForwardByTime(1.0);
    }

    results.bottomUpBreadthFirst(m_Aggregator);
    results.createPivots();
    results.pivotsBottomUpBreadthFirst(m_Aggregator);
}

void CAnomalyJob::updateNormalizerAndNormalizeResults(bool isInterim,
                                                      model::CHierarchicalResults& results) {
    m_Normalizer.setJob(model::CHierarchicalResultsNormalizer::E_RefreshSettings);
    results.bottomUpBreadthFirst(m_Normalizer);
    results.pivotsBottomUpBreadthFirst(m_Normalizer);

    // The normalizers are NOT updated with interim results, in other
    // words interim results are normalized with respect to previous
    // final results.
    if (isInterim == false) {
        m_Normalizer.propagateForwardByTime(1.0);
        m_Normalizer.setJob(model::CHierarchicalResultsNormalizer::E_UpdateQuantiles);
        results.bottomUpBreadthFirst(m_Normalizer);
        results.pivotsBottomUpBreadthFirst(m_Normalizer);
    }

    m_Normalizer.setJob(model::CHierarchicalResultsNormalizer::E_NormalizeScores);
    results.bottomUpBreadthFirst(m_Normalizer);
    results.pivotsBottomUpBreadthFirst(m_Normalizer);
}

void CAnomalyJob::outputResultsWithinRange(bool isInterim, core_t::TTime start, core_t::TTime end) {
    if (m_LastFinalisedBucketEndTime <= 0) {
        return;
    }
    if (start < m_LastFinalisedBucketEndTime) {
        LOG_WARN(<< "Cannot output results for range (" << start << ", " << m_LastFinalisedBucketEndTime
                 << "): Start time is before last finalized bucket end time "
                 << m_LastFinalisedBucketEndTime << '.');
        start = m_LastFinalisedBucketEndTime;
    }
    if (start > end) {
        LOG_ERROR(<< "Cannot output results for range (" << start << ", " << end
                  << "): Start time is later than end time.");
        return;
    }
    core_t::TTime bucketLength = m_ModelConfig.bucketLength();
    core_t::TTime time = maths::CIntegerTools::floor(start, bucketLength);
    core_t::TTime bucketEnd = maths::CIntegerTools::ceil(end, bucketLength);
    while (time < bucketEnd) {
        if (isInterim) {
            this->outputInterimResults(time);
        } else {
            this->outputResults(time);
        }
        m_Limits.resourceMonitor().sendMemoryUsageReportIfSignificantlyChanged(time);
        time += bucketLength;
    }
}

void CAnomalyJob::generateModelPlot(core_t::TTime startTime,
                                    core_t::TTime endTime,
                                    const model::CAnomalyDetector& detector,
                                    TModelPlotDataVec& modelPlotData) {
    double modelPlotBoundsPercentile(m_ModelConfig.modelPlotBoundsPercentile());
    if (modelPlotBoundsPercentile > 0.0) {
        LOG_TRACE(<< "Generating model debug data at " << startTime);
        detector.generateModelPlot(startTime, endTime,
                                   m_ModelConfig.modelPlotBoundsPercentile(),
                                   m_ModelConfig.modelPlotTerms(), modelPlotData);
    }
}

void CAnomalyJob::writeOutModelPlot(const TModelPlotDataVec& modelPlotData) {
    CModelPlotDataJsonWriter modelPlotWriter(m_OutputStream);
    for (const auto& plot : modelPlotData) {
        modelPlotWriter.writeFlat(m_JobId, plot);
    }
}

void CAnomalyJob::refreshMemoryAndReport() {
    // Make sure model size stats are up to date and then send a final memory
    // usage report
    for (const auto& detector_ : m_Detectors) {
        model::CAnomalyDetector* detector = detector_.second.get();
        if (detector == nullptr) {
            LOG_ERROR(<< "Unexpected NULL pointer for key '"
                      << pairDebug(detector_.first) << '\'');
            continue;
        }
        m_Limits.resourceMonitor().forceRefresh(*detector);
    }
    m_Limits.resourceMonitor().sendMemoryUsageReport(
        m_LastFinalisedBucketEndTime - m_ModelConfig.bucketLength());
}

void CAnomalyJob::persistIndividualDetector(const model::CAnomalyDetector& detector,
                                            core::CStatePersistInserter& inserter) {
    inserter.insertLevel(KEY_TAG, std::bind(&model::CAnomalyDetector::keyAcceptPersistInserter,
                                            &detector, std::placeholders::_1));
    inserter.insertLevel(PARTITION_FIELD_TAG,
                         std::bind(&model::CAnomalyDetector::partitionFieldAcceptPersistInserter,
                                   &detector, std::placeholders::_1));
    inserter.insertLevel(DETECTOR_TAG, std::bind(&model::CAnomalyDetector::acceptPersistInserter,
                                                 &detector, std::placeholders::_1));
}

void CAnomalyJob::detectors(TAnomalyDetectorPtrVec& detectors) const {
    detectors.clear();
    detectors.reserve(m_Detectors.size());
    for (const auto& detector : m_Detectors) {
        detectors.push_back(detector.second);
    }
}

void CAnomalyJob::sortedDetectors(TKeyCRefAnomalyDetectorPtrPrVec& detectors) const {
    detectors.reserve(m_Detectors.size());
    for (const auto& detector : m_Detectors) {
        detectors.push_back(TKeyCRefAnomalyDetectorPtrPr(
            model::CSearchKey::TStrCRefKeyCRefPr(std::cref(detector.first.first),
                                                 std::cref(detector.first.second)),
            detector.second));
    }
    std::sort(detectors.begin(), detectors.end(), maths::COrderings::SFirstLess());
}

const CAnomalyJob::TKeyAnomalyDetectorPtrUMap& CAnomalyJob::detectorPartitionMap() const {
    return m_Detectors;
}

const CAnomalyJob::TAnomalyDetectorPtr&
CAnomalyJob::detectorForKey(bool isRestoring,
                            core_t::TTime time,
                            const model::CSearchKey& key,
                            const std::string& partitionFieldValue,
                            model::CResourceMonitor& resourceMonitor) {
    // The simple count detector always lives in a special null partition.
    const std::string& partition = key.isSimpleCount() ? EMPTY_STRING : partitionFieldValue;

    // Try and get the detector.
    auto itr = m_Detectors.find(
        model::CSearchKey::TStrCRefKeyCRefPr(std::cref(partition), std::cref(key)),
        model::CStrKeyPrHash(), model::CStrKeyPrEqual());

    // Check if we need to and are allowed to create a new detector.
    if (itr == m_Detectors.end() && resourceMonitor.areAllocationsAllowed()) {
        // Create an placeholder for the anomaly detector.
        TAnomalyDetectorPtr& detector =
            m_Detectors
                .emplace(model::CSearchKey::TStrKeyPr(partition, key), TAnomalyDetectorPtr())
                .first->second;

        LOG_TRACE(<< "Creating new detector for key '" << key.debug() << '/'
                  << partition << '\'' << ", time " << time);
        LOG_TRACE(<< "Detector count " << m_Detectors.size());

        detector = this->makeDetector(key.identifier(), m_ModelConfig, m_Limits,
                                      partition, time, m_ModelConfig.factory(key));
        if (detector == nullptr) {
            // This should never happen as CAnomalyDetectorUtils::makeDetector()
            // contracts to never return NULL
            LOG_ABORT(<< "Failed to create anomaly detector for key '"
                      << key.debug() << '\'');
        }

        detector->zeroModelsToTime(time - m_ModelConfig.latency());

        if (isRestoring == false) {
            m_Limits.resourceMonitor().forceRefresh(*detector);
        }
        return detector;
    } else if (itr == m_Detectors.end()) {
        LOG_TRACE(<< "No memory to create new detector for key '" << key.debug()
                  << '/' << partition << '\'');
        return NULL_DETECTOR;
    }

    return itr->second;
}

void CAnomalyJob::pruneAllModels() {
    LOG_INFO(<< "Pruning all models");

    for (const auto& detector_ : m_Detectors) {
        model::CAnomalyDetector* detector = detector_.second.get();
        if (detector == nullptr) {
            LOG_ERROR(<< "Unexpected NULL pointer for key '"
                      << pairDebug(detector_.first) << '\'');
            continue;
        }
        detector->pruneModels();
    }
}

CAnomalyJob::TAnomalyDetectorPtr
CAnomalyJob::makeDetector(int identifier,
                          const model::CAnomalyDetectorModelConfig& modelConfig,
                          model::CLimits& limits,
                          const std::string& partitionFieldValue,
                          core_t::TTime firstTime,
                          const model::CAnomalyDetector::TModelFactoryCPtr& modelFactory) {
    return modelFactory->isSimpleCount()
               ? std::make_shared<model::CSimpleCountDetector>(
                     identifier, modelFactory->summaryMode(), modelConfig,
                     std::ref(limits), partitionFieldValue, firstTime, modelFactory)
               : std::make_shared<model::CAnomalyDetector>(
                     identifier, std::ref(limits), modelConfig,
                     partitionFieldValue, firstTime, modelFactory);
}

void CAnomalyJob::populateDetectorKeys(const CFieldConfig& fieldConfig, TKeyVec& keys) {
    keys.clear();

    // Add a key for the simple count detector.
    keys.push_back(model::CSearchKey::simpleCountKey());

    for (const auto& fieldOptions : fieldConfig.fieldOptions()) {
        keys.emplace_back(fieldOptions.configKey(), fieldOptions.function(),
                          fieldOptions.useNull(), fieldOptions.excludeFrequent(),
                          fieldOptions.fieldName(), fieldOptions.byFieldName(),
                          fieldOptions.overFieldName(), fieldOptions.partitionFieldName(),
                          fieldConfig.influencerFieldNames());
    }
}

const std::string* CAnomalyJob::fieldValue(const std::string& fieldName,
                                           const TStrStrUMap& dataRowFields) {
    TStrStrUMapCItr itr = fieldName.empty() ? dataRowFields.end()
                                            : dataRowFields.find(fieldName);
    const std::string& fieldValue(itr == dataRowFields.end() ? EMPTY_STRING : itr->second);
    return !fieldName.empty() && fieldValue.empty() ? nullptr : &fieldValue;
}

void CAnomalyJob::addRecord(const TAnomalyDetectorPtr detector,
                            core_t::TTime time,
                            const TStrStrUMap& dataRowFields) {
    model::CAnomalyDetector::TStrCPtrVec fieldValues;
    const TStrVec& fieldNames = detector->fieldsOfInterest();
    fieldValues.reserve(fieldNames.size());
    for (std::size_t i = 0u; i < fieldNames.size(); ++i) {
        fieldValues.push_back(fieldValue(fieldNames[i], dataRowFields));
    }

    detector->addRecord(time, fieldValues);
}

CAnomalyJob::SBackgroundPersistArgs::SBackgroundPersistArgs(
    core_t::TTime time,
    const model::CResourceMonitor::SResults& modelSizeStats,
    const model::CInterimBucketCorrector& interimBucketCorrector,
    const model::CHierarchicalResultsAggregator& aggregator,
    core_t::TTime latestRecordTime,
    core_t::TTime lastResultsTime)
    : s_Time(time), s_ModelSizeStats(modelSizeStats),
      s_InterimBucketCorrector(interimBucketCorrector), s_Aggregator(aggregator),
      s_LatestRecordTime(latestRecordTime), s_LastResultsTime(lastResultsTime) {
}
}
}
