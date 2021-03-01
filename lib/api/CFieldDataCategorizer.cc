/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CFieldDataCategorizer.h>

#include <core/CDataAdder.h>
#include <core/CDataSearcher.h>
#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CLogger.h>
#include <core/CPersistUtils.h>
#include <core/CStateCompressor.h>
#include <core/CStateDecompressor.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CStringUtils.h>

#include <model/CTokenListReverseSearchCreator.h>

#include <api/CAnomalyJobConfig.h>
#include <api/CNoopCategoryIdMapper.h>
#include <api/CPerPartitionCategoryIdMapper.h>
#include <api/CPersistenceManager.h>

#include <memory>
#include <sstream>

namespace ml {
namespace api {

namespace {
// For historical reasons, these tags must not duplicate those used in
// CSingleFieldDataCategorizer.cc
const std::string VERSION_TAG{"a"};
// b, c, d used in CSingleFieldDataCategorizer.cc
const std::string PARTITION_FIELD_VALUE_TAG{"e"};
const std::string HIGHEST_GLOBAL_ID_TAG{"f"};
const std::string CATEGORIZER_ALLOCATION_FAILURES{"g"};
const std::string EMPTY_STRING;
} // unnamed

// Initialise statics
const std::string CFieldDataCategorizer::MLCATEGORY_NAME{"mlcategory"};
const double CFieldDataCategorizer::SIMILARITY_THRESHOLD{0.7};
const std::string CFieldDataCategorizer::STATE_TYPE{"categorizer_state"};
const std::string CFieldDataCategorizer::STATE_VERSION{"1"};

CFieldDataCategorizer::CFieldDataCategorizer(std::string jobId,
                                             const CAnomalyJobConfig::CAnalysisConfig& analysisConfig,
                                             model::CLimits& limits,
                                             const std::string& timeFieldName,
                                             const std::string& timeFieldFormat,
                                             CDataProcessor* chainedProcessor,
                                             core::CJsonOutputStreamWrapper& outputStream,
                                             CPersistenceManager* persistenceManager,
                                             bool stopCategorizationOnWarnStatus)
    : CDataProcessor{timeFieldName, timeFieldFormat}, m_JobId{std::move(jobId)}, m_Limits{limits},
      m_ChainedProcessor{chainedProcessor}, m_OutputStream{outputStream},
      m_StopCategorizationOnWarnStatus{stopCategorizationOnWarnStatus},
      m_JsonOutputWriter{m_JobId, m_OutputStream}, m_AnnotationJsonWriter{m_OutputStream},
      m_PartitionFieldName{analysisConfig.categorizationPartitionFieldName()},
      m_CategorizationFieldName{analysisConfig.categorizationFieldName()}, m_PersistenceManager{persistenceManager} {

    if (analysisConfig.categorizationFilters().empty() == false) {
        LOG_DEBUG(<< "Configuring categorization filtering");
        m_CategorizationFilter.configure(analysisConfig.categorizationFilters());
    }
}

CFieldDataCategorizer::~CFieldDataCategorizer() {
    for (const auto& dataCategorizerEntry : m_DataCategorizers) {
        dataCategorizerEntry.second->dumpStats();
    }
}

void CFieldDataCategorizer::registerMutableField(const std::string& fieldName,
                                                 std::string& fieldValue) {
    if (fieldName == MLCATEGORY_NAME) {
        m_OutputFieldCategory = &fieldValue;
    }
    if (m_ChainedProcessor != nullptr) {
        m_ChainedProcessor->registerMutableField(fieldName, fieldValue);
    }
}

bool CFieldDataCategorizer::handleRecord(const TStrStrUMap& dataRowFields, TOptionalTime time) {

    // Non-empty control fields take precedence over everything else
    auto iter = dataRowFields.find(CONTROL_FIELD_NAME);
    if (iter != dataRowFields.end() && !iter->second.empty()) {
        // Always handle control messages, but signal completion of handling ONLY if we are the last handler
        // e.g. flush requests are acknowledged here if this is the last handler
        bool msgHandled{this->handleControlMessage(iter->second, m_ChainedProcessor == nullptr)};
        if (m_ChainedProcessor != nullptr) {
            return m_ChainedProcessor->handleRecord(dataRowFields, time);
        }
        return msgHandled;
    }

    if (time == boost::none) {
        time = this->parseTime(dataRowFields);
    }

    CGlobalCategoryId globalCategoryId{this->computeAndUpdateCategory(dataRowFields, time)};
    if (globalCategoryId.isHardFailure() == false) {
        if (m_OutputFieldCategory != nullptr) {
            *m_OutputFieldCategory =
                core::CStringUtils::typeToString(globalCategoryId.globalId());
        }
        if (m_ChainedProcessor != nullptr &&
            m_ChainedProcessor->handleRecord(dataRowFields, time) == false) {
            return false;
        }
        ++m_NumRecordsHandled;
    }

    if (m_PersistenceManager != nullptr) {
        m_PersistenceManager->startPersistIfAppropriate();
    }

    // We return true even if we had a hard failure for the current input,
    // because to return false would fail the whole job
    return true;
}

void CFieldDataCategorizer::finalise() {

    // Make sure model size stats are up to date
    for (const auto& dataCategorizerEntry : m_DataCategorizers) {
        dataCategorizerEntry.second->forceResourceRefresh(m_Limits.resourceMonitor());
    }
    writeChanges();

    // Pass on the request in case we're chained
    if (m_ChainedProcessor != nullptr) {
        m_ChainedProcessor->finalise();
    }

    // Wait for any ongoing periodic persist to complete, so that the data adder
    // is not used by both a periodic periodic persist and final persist at the
    // same time
    if (m_PersistenceManager != nullptr) {
        m_PersistenceManager->waitForIdle();
    }

    m_JsonOutputWriter.finalise();
}

std::uint64_t CFieldDataCategorizer::numRecordsHandled() const {
    return m_NumRecordsHandled;
}

CGlobalCategoryId
CFieldDataCategorizer::computeAndUpdateCategory(const TStrStrUMap& dataRowFields,
                                                const TOptionalTime& time) {
    CGlobalCategoryId globalCategoryId;

    auto fieldIter = dataRowFields.find(m_CategorizationFieldName);
    if (fieldIter == dataRowFields.end()) {
        LOG_WARN(<< "Assigning ML category " << globalCategoryId << " to record with no "
                 << m_CategorizationFieldName << " field:" << core_t::LINE_ENDING
                 << this->debugPrintRecord(dataRowFields));
        return globalCategoryId;
    }

    const std::string& fieldValue{fieldIter->second};
    if (fieldValue.empty()) {
        LOG_WARN(<< "Assigning ML category " << globalCategoryId << " to record with blank "
                 << m_CategorizationFieldName << " field:" << core_t::LINE_ENDING
                 << this->debugPrintRecord(dataRowFields));
        return globalCategoryId;
    }

    const std::string& partitionFieldValue{this->categorizerKeyForRecord(dataRowFields)};
    CSingleFieldDataCategorizer* dataCategorizer{this->categorizerPtrForKey(partitionFieldValue)};
    if (dataCategorizer == nullptr) {
        ++m_CategorizerAllocationFailures;
        if (m_PartitionFieldName.empty()) {
            if (m_Limits.resourceMonitor().categorizerAllocationFailures() == 0) {
                LOG_WARN(<< "Categorization not possible due to lack of memory");
            }
        } else {
            bool partitionNotFailedBefore{m_CategorizerAllocationFailedPartitions
                                              .insert(partitionFieldValue)
                                              .second};
            if (partitionNotFailedBefore) {
                LOG_WARN(<< "Categorization not possible for partition '"
                         << partitionFieldValue << "' due to lack of memory");
            }
        }
        m_Limits.resourceMonitor().categorizerAllocationFailures(m_CategorizerAllocationFailures);
        return CGlobalCategoryId::hardFailure();
    }
    if (m_StopCategorizationOnWarnStatus &&
        dataCategorizer->categorizationStatus() == model_t::E_CategorizationStatusWarn) {
        LOG_TRACE(<< "Ignoring input record as its categorizer has a 'warn' status:"
                  << core_t::LINE_ENDING << this->debugPrintRecord(dataRowFields));
        return CGlobalCategoryId::hardFailure();
    }
    if (m_CategorizationFilter.empty()) {
        globalCategoryId = dataCategorizer->computeAndUpdateCategory(
            false, dataRowFields, time, fieldValue, fieldValue,
            m_Limits.resourceMonitor(), m_JsonOutputWriter);
    } else {
        std::string filtered{m_CategorizationFilter.apply(fieldValue)};
        globalCategoryId = dataCategorizer->computeAndUpdateCategory(
            false, dataRowFields, time, filtered, fieldValue,
            m_Limits.resourceMonitor(), m_JsonOutputWriter);
    }
    if (globalCategoryId.isValid()) {
        dataCategorizer->writeStatsIfUrgent(m_JsonOutputWriter, m_AnnotationJsonWriter);
    }
    return globalCategoryId;
}

const std::string& CFieldDataCategorizer::categorizerKeyForRecord(const TStrStrUMap& dataRowFields) {
    if (m_PartitionFieldName.empty()) {
        return EMPTY_STRING;
    }
    auto partitionFieldIter = dataRowFields.find(m_PartitionFieldName);
    if (partitionFieldIter == dataRowFields.end()) {
        return EMPTY_STRING;
    }
    return partitionFieldIter->second;
}

CSingleFieldDataCategorizer*
CFieldDataCategorizer::categorizerPtrForKey(const std::string& partitionFieldValue) {
    auto iter = m_DataCategorizers.find(partitionFieldValue);
    if (iter != m_DataCategorizers.end()) {
        return iter->second.get();
    }

    if (m_Limits.resourceMonitor().areAllocationsAllowed() == false) {
        // The categorizer doesn't exist, but we are not allowed to create it
        return nullptr;
    }

    CCategoryIdMapper::TCategoryIdMapperPtr idMapper;
    if (m_PartitionFieldName.empty()) {
        idMapper = std::make_shared<CNoopCategoryIdMapper>();
    } else {
        idMapper = std::make_shared<CPerPartitionCategoryIdMapper>(
            partitionFieldValue, [this]() { return this->nextGlobalId(); });
    }

    // TODO - if we ever have more than one data categorizer class, this
    // should be replaced with a factory
    auto localCategorizer = std::make_unique<TTokenListDataCategorizerKeepsFields>(
        m_Limits, std::make_shared<model::CTokenListReverseSearchCreator>(m_CategorizationFieldName),
        SIMILARITY_THRESHOLD, m_CategorizationFieldName);

    auto globalCategorizer = std::make_unique<CSingleFieldDataCategorizer>(
        m_PartitionFieldName, std::move(localCategorizer), std::move(idMapper));
    iter = m_DataCategorizers
               .emplace(partitionFieldValue, std::move(globalCategorizer))
               .first;
    LOG_TRACE(<< "Created new categorizer for '" << partitionFieldValue << '/'
              << m_CategorizationFieldName << "'");
    return iter->second.get();
}

CSingleFieldDataCategorizer&
CFieldDataCategorizer::categorizerForKey(const std::string& partitionFieldValue) {
    return *this->categorizerPtrForKey(partitionFieldValue);
}

bool CFieldDataCategorizer::restoreState(core::CDataSearcher& restoreSearcher,
                                         core_t::TTime& completeToTime) {
    // Pass on the request in case we're chained
    if (m_ChainedProcessor != nullptr &&
        m_ChainedProcessor->restoreState(restoreSearcher, completeToTime) == false) {
        return false;
    }

    LOG_DEBUG(<< "Restore categorizer state");

    try {
        // Restore from Elasticsearch compressed data
        core::CStateDecompressor decompressor(restoreSearcher);

        core::CDataSearcher::TIStreamP strm(decompressor.search(1, 1));
        if (strm == nullptr) {
            LOG_ERROR(<< "Unable to connect to data store");
            return false;
        }

        if (strm->bad()) {
            LOG_ERROR(<< "Categorizer state restoration returned a bad stream");
            return false;
        }

        if (strm->fail()) {
            // This is fatal. If the stream exists and has failed then state is missing
            LOG_ERROR(<< "Categorizer state restoration returned a failed stream");
            return false;
        }

        // We're dealing with streaming JSON state
        core::CJsonStateRestoreTraverser traverser(*strm);

        if (this->acceptRestoreTraverser(traverser) == false) {
            LOG_ERROR(<< "JSON restore failed");
            return false;
        }
    } catch (std::exception& e) {
        LOG_ERROR(<< "Failed to restore state! " << e.what());
        // This is fatal in terms of the categorizer we attempted to restore,
        // but returning false here can throw the system into a repeated cycle
        // of failure.  It's better to reset the categorizer and re-categorize from
        // scratch.
        this->resetAfterCorruptRestore();
        return true;
    }

    return true;
}

bool CFieldDataCategorizer::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    const std::string& firstFieldName = traverser.name();
    if (traverser.isEof()) {
        LOG_ERROR(<< "Expected categorizer persisted state but no state exists");
        return false;
    }

    if (firstFieldName == VERSION_TAG) {
        std::string version;
        if (core::CStringUtils::stringToType(traverser.value(), version) == false) {
            LOG_ERROR(<< "Cannot restore categorizer, invalid version: "
                      << traverser.value());
            return false;
        }
        if (version != STATE_VERSION) {
            LOG_DEBUG(<< "Categorizer has not been restored as the version has changed");
            return true;
        }
    } else {
        LOG_ERROR(<< "Cannot restore categorizer - " << VERSION_TAG << " element expected but found "
                  << traverser.name() << '=' << traverser.value());
        return false;
    }

    TStrVec categorizerKeys;
    if (m_PartitionFieldName.empty()) {
        // The sole categorizer key when per-partition categorization is
        // disabled is the empty string
        categorizerKeys.resize(1);
    } else {
        if (traverser.next() == false) {
            LOG_ERROR(<< "Cannot restore categorizer - end of object reached when "
                      << PARTITION_FIELD_VALUE_TAG << " was expected");
            return false;
        }

        if (core::CPersistUtils::restore(PARTITION_FIELD_VALUE_TAG,
                                         categorizerKeys, traverser) == false) {
            LOG_ERROR(<< "Invalid partition values in " << traverser.value());
            return false;
        }

        if (traverser.next() == false) {
            LOG_ERROR(<< "Cannot restore categorizer - end of object reached when "
                      << HIGHEST_GLOBAL_ID_TAG << " was expected");
            return false;
        }

        if (traverser.name() == HIGHEST_GLOBAL_ID_TAG) {
            if (core::CStringUtils::stringToType(traverser.value(), m_HighestGlobalId) == false) {
                LOG_ERROR(<< "Invalid highest global ID in " << traverser.value());
                return false;
            }
        }
    }

    for (const auto& categorizerKey : categorizerKeys) {
        CSingleFieldDataCategorizer* dataCategorizer{this->categorizerPtrForKey(categorizerKey)};
        if (dataCategorizer == nullptr) {
            // This is extremely unlikely to happen.  It implies the model
            // memory limit was reduced since the job was last run, or else the
            // anomaly detector state was reverted to a model snapshot that
            // requires more memory than the one in use at the time the
            // categorizer state was generated.
            LOG_ERROR(<< "Memory limit hit while restoring categorizer state");
            return false;
        }
        // Unlike most nested objects this doesn't traverse a sub-level.
        // This dates back to the time when there was only one categorizer.
        // To do otherwise now would break state compatibility.
        if (dataCategorizer->acceptRestoreTraverser(traverser) == false) {
            LOG_ERROR(<< "Cannot restore categorizer from " << traverser.value());
            return false;
        }
    }

    // This won't be present in pre-7.9 state, but that's OK because the
    // default value of 0 is correct for this case (there was no per-partition
    // categorization and the single categorizer was allocated unconditionally)
    if (traverser.next() && traverser.name() == CATEGORIZER_ALLOCATION_FAILURES) {
        if (core::CStringUtils::stringToType(
                traverser.value(), m_CategorizerAllocationFailures) == false) {
            LOG_ERROR(<< "Invalid categorizer allocation failure count in "
                      << traverser.value());
            return false;
        }
    }

    return true;
}

bool CFieldDataCategorizer::persistStateInForeground(core::CDataAdder& persister,
                                                     const std::string& descriptionPrefix) {
    if (m_PersistenceManager != nullptr) {
        // This will not happen if finalise() was called before persisting state
        if (m_PersistenceManager->isBusy()) {
            LOG_ERROR(<< "Cannot do final persistence of state - periodic "
                         "persister still busy");
            return false;
        }
    }

    // Pass on the request in case we're chained
    if (m_ChainedProcessor != nullptr &&
        m_ChainedProcessor->persistStateInForeground(persister, descriptionPrefix) == false) {
        return false;
    }

    LOG_DEBUG(<< "Persist categorizer state in foreground");

    TStrVec partitionFieldValues;
    TPersistFuncVec dataCategorizerPersistFuncs;

    if (m_PartitionFieldName.empty()) {
        CSingleFieldDataCategorizer& dataCategorizer{categorizerForKey(EMPTY_STRING)};
        dataCategorizerPersistFuncs.emplace_back(dataCategorizer.makeForegroundPersistFunc());
    } else {
        if (m_DataCategorizers.empty()) {
            LOG_WARN(<< "No partition-specific categorizers found");
            return true;
        }
        partitionFieldValues.reserve(m_DataCategorizers.size());
        dataCategorizerPersistFuncs.reserve(m_DataCategorizers.size());
        for (auto& dataCategorizerEntry : m_DataCategorizers) {
            partitionFieldValues.push_back(dataCategorizerEntry.first);
            dataCategorizerPersistFuncs.emplace_back(
                dataCategorizerEntry.second->makeForegroundPersistFunc());
        }
    }

    return this->doPersistState(partitionFieldValues, dataCategorizerPersistFuncs,
                                m_CategorizerAllocationFailures, persister);
}

bool CFieldDataCategorizer::isPersistenceNeeded(const std::string& description) const {
    // Pass on the request in case we're chained
    if (m_ChainedProcessor != nullptr && m_ChainedProcessor->isPersistenceNeeded(description)) {
        return true;
    }

    if (m_NumRecordsHandled == 0) {
        LOG_DEBUG(<< "Zero records were handled - will not attempt to persist "
                  << description << ".");
        return false;
    }
    return true;
}

bool CFieldDataCategorizer::doPersistState(const TStrVec& partitionFieldValues,
                                           const TPersistFuncVec& dataCategorizerPersistFuncs,
                                           std::size_t categorizerAllocationFailures,
                                           core::CDataAdder& persister) {

    // TODO: if the standalone categorize program is ever progressed, a mechanism needs
    // to be added that does the following:
    // 1. Caches program counters in the foreground before starting background persistence
    // 2. Calls core::CProgramCounters::staticsAcceptPersistInserter once and only once per persist
    // 3. Clears the program counter cache after persistence is complete

    // The two input vectors should have the same size _unless_ we are not
    // doing per-partition categorization, in which case partition field values
    // should be empty and there should be exactly one categorizer
    if (partitionFieldValues.size() != dataCategorizerPersistFuncs.size() &&
        (dataCategorizerPersistFuncs.size() != 1 || partitionFieldValues.empty() == false)) {
        LOG_ERROR(<< "Programmatic error - doPersistState called with "
                  << dataCategorizerPersistFuncs.size() << " categorizer persistence functions and "
                  << partitionFieldValues.size() << " partition field values");
        return false;
    }
    try {
        core::CStateCompressor compressor{persister};

        core::CDataAdder::TOStreamP strm{compressor.addStreamed(m_JobId + '_' + STATE_TYPE)};

        if (strm == nullptr) {
            LOG_ERROR(<< "Failed to create persistence stream");
            return false;
        }

        if (!strm->good()) {
            LOG_ERROR(<< "Persistence stream is bad before stream of "
                         "state for the categorizer");
            return false;
        }

        {
            // Keep the JSON inserter scoped as it only finishes the stream
            // when it is destructed
            core::CJsonStatePersistInserter inserter{*strm};
            inserter.insertValue(VERSION_TAG, STATE_VERSION);
            if (partitionFieldValues.empty() == false) {
                core::CPersistUtils::persist(PARTITION_FIELD_VALUE_TAG,
                                             partitionFieldValues, inserter);
                inserter.insertValue(HIGHEST_GLOBAL_ID_TAG, m_HighestGlobalId);
            }

            for (const auto& dataCategorizerPersistFunc : dataCategorizerPersistFuncs) {
                dataCategorizerPersistFunc(inserter);
            }

            inserter.insertValue(CATEGORIZER_ALLOCATION_FAILURES, categorizerAllocationFailures);
            // Note that m_CategorizerAllocationFailedPartitions is deliberately
            // not persisted here.  This means that failures to categorize at
            // all for a given partition will be logged once per invocation of
            // the program.  Given that people may close and reopen the job
            // in an attempt to fix the problem of categorization not being done
            // at all for a partition, this is a good thing.
        }

        if (strm->bad()) {
            LOG_ERROR(<< "Persistence stream went bad during stream of "
                         "state for the categorizer");
            return false;
        }

        if (compressor.streamComplete(strm, true) == false || strm->bad()) {
            LOG_ERROR(<< "Failed to complete last persistence stream");
            return false;
        }
    } catch (std::exception& e) {
        LOG_ERROR(<< "Failed to persist state! " << e.what());
        return false;
    }
    return true;
}

bool CFieldDataCategorizer::periodicPersistStateInBackground() {
    LOG_DEBUG(<< "Periodic persist categorizer state in background");

    // Make sure that the model size stats are up to date
    for (auto& dataCategorizerEntry : m_DataCategorizers) {
        dataCategorizerEntry.second->forceResourceRefresh(m_Limits.resourceMonitor());
    }

    // Pass on the request in case we're chained
    if (m_ChainedProcessor != nullptr &&
        m_ChainedProcessor->periodicPersistStateInBackground() == false) {
        return false;
    }

    if (m_PersistenceManager == nullptr) {
        LOG_ERROR(<< "NULL persistence manager");
        return false;
    }

    TStrVec partitionFieldValues;
    TPersistFuncVec dataCategorizerPersistFuncs;

    if (m_PartitionFieldName.empty()) {
        CSingleFieldDataCategorizer& dataCategorizer{categorizerForKey(EMPTY_STRING)};
        dataCategorizerPersistFuncs.emplace_back(dataCategorizer.makeBackgroundPersistFunc());
    } else {
        if (m_DataCategorizers.empty()) {
            LOG_WARN(<< "No partition-specific categorizers found");
            return true;
        }

        partitionFieldValues.reserve(m_DataCategorizers.size());
        dataCategorizerPersistFuncs.reserve(m_DataCategorizers.size());
        for (auto& dataCategorizerEntry : m_DataCategorizers) {
            partitionFieldValues.push_back(dataCategorizerEntry.first);
            CSingleFieldDataCategorizer& dataCategorizer{*dataCategorizerEntry.second};
            dataCategorizer.forceResourceRefresh(m_Limits.resourceMonitor());
            dataCategorizerPersistFuncs.emplace_back(
                dataCategorizer.makeBackgroundPersistFunc());
        }
    }

    // Do NOT pass the captures by reference - they
    // MUST be copied for thread safety
    if (m_PersistenceManager->addPersistFunc([
            this, partitionFieldValues = std::move(partitionFieldValues),
            dataCategorizerPersistFuncs = std::move(dataCategorizerPersistFuncs),
            categorizerAllocationFailures = m_CategorizerAllocationFailures
        ](core::CDataAdder & persister) {
            return this->doPersistState(partitionFieldValues, dataCategorizerPersistFuncs,
                                        categorizerAllocationFailures, persister);
        }) == false) {
        LOG_ERROR(<< "Failed to add categorizer background persistence function");
        return false;
    }

    m_PersistenceManager->useBackgroundPersistence();

    return true;
}

bool CFieldDataCategorizer::periodicPersistStateInForeground() {
    LOG_DEBUG(<< "Periodic persist categorizer state in foreground");

    if (m_PersistenceManager == nullptr) {
        return false;
    }

    // Do NOT pass this request on to the output chainer.
    // That logic is already present in persistStateInForeground.
    if (m_PersistenceManager->addPersistFunc([&](core::CDataAdder& persister) {
            return this->persistStateInForeground(persister, "Periodic foreground persist at ");
        }) == false) {
        LOG_ERROR(<< "Failed to add categorizer foreground persistence function");
        return false;
    }

    m_PersistenceManager->useForegroundPersistence();

    return true;
}

void CFieldDataCategorizer::resetAfterCorruptRestore() {
    LOG_WARN(<< "Discarding corrupt categorizer state - will re-categorize from scratch");

    m_HighestGlobalId = 0;
    m_DataCategorizers.clear();
    m_CategorizerAllocationFailures = 0;
    m_CategorizerAllocationFailedPartitions.clear();
}

bool CFieldDataCategorizer::handleControlMessage(const std::string& controlMessage,
                                                 bool lastHandler) {
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
    case 'c':
        this->parseStopOnWarnControlMessage(controlMessage.substr(1));
        break;
    case 'f':
        // Flush ID comes after the initial f
        this->acknowledgeFlush(controlMessage.substr(1), lastHandler);
        break;
    default:
        if (lastHandler) {
            LOG_WARN(<< "Ignoring unknown control message of length "
                     << controlMessage.length() << " beginning with '"
                     << controlMessage[0] << '\'');
        }
        // Don't return false here (for the time being at least), as it
        // seems excessive to cause the entire job to fail
        break;
    }

    return true;
}

void CFieldDataCategorizer::acknowledgeFlush(const std::string& flushId, bool lastHandler) {
    if (flushId.empty()) {
        LOG_ERROR(<< "Received flush control message with no ID");
    } else {
        LOG_TRACE(<< "Received flush control message with ID " << flushId);
    }
    writeChanges();
    if (lastHandler) {
        m_JsonOutputWriter.acknowledgeFlush(flushId, 0);
    }
}

void CFieldDataCategorizer::parseStopOnWarnControlMessage(const std::string& enabledStr) {
    bool enabled{false};
    if (core::CStringUtils::stringToType(enabledStr, enabled) == false) {
        LOG_ERROR(<< "Failed to parse stop-on-warn control message: " << enabledStr);
        return;
    }
    if (m_StopCategorizationOnWarnStatus != enabled) {
        LOG_INFO(<< "Categorization stop-on-warn now: " << std::boolalpha << enabled);
        m_StopCategorizationOnWarnStatus = enabled;
    }
}

void CFieldDataCategorizer::writeChanges() {
    for (auto& dataCategorizerEntry : m_DataCategorizers) {
        dataCategorizerEntry.second->writeChanges(m_JsonOutputWriter, m_AnnotationJsonWriter);
    }
}

int CFieldDataCategorizer::nextGlobalId() {
    return ++m_HighestGlobalId;
}
}
}
