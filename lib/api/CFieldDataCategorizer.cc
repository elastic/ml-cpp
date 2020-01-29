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
#include <core/CStateCompressor.h>
#include <core/CStateDecompressor.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CStringUtils.h>

#include <model/CTokenListReverseSearchCreator.h>

#include <api/CFieldConfig.h>
#include <api/CJsonOutputWriter.h>
#include <api/COutputHandler.h>
#include <api/CPersistenceManager.h>

#include <memory>
#include <sstream>

namespace ml {
namespace api {

namespace {

const std::string VERSION_TAG("a");
const std::string CATEGORIZER_TAG("b");
const std::string EXAMPLES_COLLECTOR_TAG("c");
} // unnamed

// Initialise statics
const std::string CFieldDataCategorizer::ML_STATE_INDEX(".ml-state");
const std::string CFieldDataCategorizer::MLCATEGORY_NAME("mlcategory");
const double CFieldDataCategorizer::SIMILARITY_THRESHOLD(0.7);
const std::string CFieldDataCategorizer::STATE_TYPE("categorizer_state");
const std::string CFieldDataCategorizer::STATE_VERSION("1");

CFieldDataCategorizer::CFieldDataCategorizer(const std::string& jobId,
                                             const CFieldConfig& config,
                                             model::CLimits& limits,
                                             COutputHandler& outputHandler,
                                             CJsonOutputWriter& jsonOutputWriter,
                                             CPersistenceManager* periodicPersister)
    : m_JobId(jobId), m_Limits(limits), m_OutputHandler(outputHandler),
      m_ExtraFieldNames(1, MLCATEGORY_NAME), m_WriteFieldNames(true),
      m_NumRecordsHandled(0), m_OutputFieldCategory(m_Overrides[MLCATEGORY_NAME]),
      m_MaxMatchingLength(0), m_JsonOutputWriter(jsonOutputWriter),
      m_CategorizationFieldName(config.categorizationFieldName()),
      m_CategorizationFilter(), m_PeriodicPersister(periodicPersister) {
    this->createCategorizer(m_CategorizationFieldName);

    LOG_DEBUG(<< "Configuring categorization filtering");
    m_CategorizationFilter.configure(config.categorizationFilters());
}

CFieldDataCategorizer::~CFieldDataCategorizer() {
    m_DataCategorizer->dumpStats();
}

void CFieldDataCategorizer::newOutputStream() {
    m_WriteFieldNames = true;
    m_OutputHandler.newOutputStream();
}

bool CFieldDataCategorizer::handleRecord(const TStrStrUMap& dataRowFields) {
    // First time through we output the field names
    if (m_WriteFieldNames) {
        TStrVec fieldNames;
        fieldNames.reserve(dataRowFields.size() + 1);
        for (const auto& entry : dataRowFields) {
            fieldNames.push_back(entry.first);
        }

        if (m_OutputHandler.fieldNames(fieldNames, m_ExtraFieldNames) == false) {
            LOG_ERROR(<< "Unable to set field names for output:" << core_t::LINE_ENDING
                      << this->debugPrintRecord(dataRowFields));
            return false;
        }
        m_WriteFieldNames = false;
    }

    // Non-empty control fields take precedence over everything else
    TStrStrUMapCItr iter = dataRowFields.find(CONTROL_FIELD_NAME);
    if (iter != dataRowFields.end() && !iter->second.empty()) {
        if (m_OutputHandler.consumesControlMessages()) {
            return m_OutputHandler.writeRow(dataRowFields, m_Overrides);
        }
        return this->handleControlMessage(iter->second);
    }

    m_OutputFieldCategory =
        core::CStringUtils::typeToString(this->computeCategory(dataRowFields));

    if (m_OutputHandler.writeRow(dataRowFields, m_Overrides) == false) {
        LOG_ERROR(<< "Unable to write output with type " << m_OutputFieldCategory
                  << " for input:" << core_t::LINE_ENDING
                  << this->debugPrintRecord(dataRowFields));
        return false;
    }
    ++m_NumRecordsHandled;
    return true;
}

void CFieldDataCategorizer::finalise() {

    // Make sure model size stats are up to date
    m_Limits.resourceMonitor().forceRefresh(*m_DataCategorizer);

    // Pass on the request in case we're chained
    m_OutputHandler.finalise();

    // Wait for any ongoing periodic persist to complete, so that the data adder
    // is not used by both a periodic periodic persist and final persist at the
    // same time
    if (m_PeriodicPersister != nullptr) {
        m_PeriodicPersister->waitForIdle();
    }
}

std::uint64_t CFieldDataCategorizer::numRecordsHandled() const {
    return m_NumRecordsHandled;
}

COutputHandler& CFieldDataCategorizer::outputHandler() {
    return m_OutputHandler;
}

int CFieldDataCategorizer::computeCategory(const TStrStrUMap& dataRowFields) {
    const std::string& categorizationFieldName{m_DataCategorizer->fieldName()};
    auto fieldIter = dataRowFields.find(categorizationFieldName);
    if (fieldIter == dataRowFields.end()) {
        LOG_WARN(<< "Assigning ML category -1 to record with no "
                 << categorizationFieldName << " field:" << core_t::LINE_ENDING
                 << this->debugPrintRecord(dataRowFields));
        return -1;
    }

    const std::string& fieldValue{fieldIter->second};
    if (fieldValue.empty()) {
        LOG_WARN(<< "Assigning ML category -1 to record with blank "
                 << categorizationFieldName << " field:" << core_t::LINE_ENDING
                 << this->debugPrintRecord(dataRowFields));
        return -1;
    }

    int categoryId{-1};
    if (m_CategorizationFilter.empty()) {
        categoryId = m_DataCategorizer->computeCategory(
            false, dataRowFields, fieldValue, fieldValue.length());
    } else {
        std::string filtered = m_CategorizationFilter.apply(fieldValue);
        categoryId = m_DataCategorizer->computeCategory(false, dataRowFields, filtered,
                                                        fieldValue.length());
    }
    if (categoryId < 1) {
        return -1;
    }

    bool exampleAdded{m_DataCategorizer->addExample(categoryId, fieldValue)};
    bool searchTermsChanged{this->createReverseSearch(categoryId)};
    if (exampleAdded || searchTermsChanged) {
        m_JsonOutputWriter.writeCategoryDefinition(
            categoryId, m_SearchTerms, m_SearchTermsRegex, m_MaxMatchingLength,
            m_DataCategorizer->examplesCollector().examples(categoryId));
        if (categoryId % 10 == 0) {
            // Even if memory limiting is disabled, force a refresh occasionally
            // so the user has some idea what's going on with memory.
            m_Limits.resourceMonitor().forceRefresh(*m_DataCategorizer);
        } else {
            m_Limits.resourceMonitor().refresh(*m_DataCategorizer);
        }
    }

    // Check if a periodic persist is due.
    if (m_PeriodicPersister != nullptr) {
        m_PeriodicPersister->startPersistIfAppropriate();
    }

    return categoryId;
}

void CFieldDataCategorizer::createCategorizer(const std::string& fieldName) {
    // TODO - if we ever have more than one data categorizer class, this should
    // be replaced with a factory
    m_DataCategorizer = std::make_shared<TTokenListDataCategorizerKeepsFields>(
        m_Limits, std::make_shared<model::CTokenListReverseSearchCreator>(fieldName),
        SIMILARITY_THRESHOLD, fieldName);

    LOG_TRACE(<< "Created new categorizer for field '" << fieldName << "'");
}

bool CFieldDataCategorizer::createReverseSearch(int categoryId) {
    bool wasCached(false);
    if (m_DataCategorizer->createReverseSearch(categoryId, m_SearchTerms, m_SearchTermsRegex,
                                               m_MaxMatchingLength, wasCached) == false) {
        m_SearchTerms.clear();
        m_SearchTermsRegex.clear();
    }
    return !wasCached;
}

bool CFieldDataCategorizer::restoreState(core::CDataSearcher& restoreSearcher,
                                         core_t::TTime& completeToTime) {
    // Pass on the request in case we're chained
    if (m_OutputHandler.restoreState(restoreSearcher, completeToTime) == false) {
        return false;
    }

    LOG_DEBUG(<< "Restore categorizer state");

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

    if (traverser.next() == false) {
        LOG_ERROR(<< "Cannot restore categorizer - end of object reached when "
                  << CATEGORIZER_TAG << " was expected");
        return false;
    }

    if (traverser.name() == CATEGORIZER_TAG) {
        if (traverser.traverseSubLevel(
                std::bind(&model::CDataCategorizer::acceptRestoreTraverser,
                          m_DataCategorizer, std::placeholders::_1)) == false) {
            LOG_ERROR(<< "Cannot restore categorizer, unexpected element: "
                      << traverser.value());
            return false;
        }
    } else {
        LOG_ERROR(<< "Cannot restore categorizer - " << CATEGORIZER_TAG << " element expected but found "
                  << traverser.name() << '=' << traverser.value());
        return false;
    }

    if (traverser.next() == false) {
        LOG_ERROR(<< "Cannot restore categorizer - end of object reached when "
                  << EXAMPLES_COLLECTOR_TAG << " was expected");
        return false;
    }

    if (traverser.name() == EXAMPLES_COLLECTOR_TAG) {
        if (m_DataCategorizer->restoreExamplesCollector(traverser) == false) {
            return false;
        }
    } else {
        LOG_ERROR(<< "Cannot restore categorizer - " << EXAMPLES_COLLECTOR_TAG << " element expected but found "
                  << traverser.name() << '=' << traverser.value());
        return false;
    }

    return true;
}

bool CFieldDataCategorizer::persistState(core::CDataAdder& persister,
                                         const std::string& descriptionPrefix) {
    if (m_PeriodicPersister != nullptr) {
        // This will not happen if finalise() was called before persisting state
        if (m_PeriodicPersister->isBusy()) {
            LOG_ERROR(<< "Cannot do final persistence of state - periodic "
                         "persister still busy");
            return false;
        }
    }

    // Pass on the request in case we're chained
    if (m_OutputHandler.persistState(persister, descriptionPrefix) == false) {
        return false;
    }

    LOG_DEBUG(<< "Persist categorizer state");

    return this->doPersistState(m_DataCategorizer->makeForegroundPersistFunc(),
                                m_DataCategorizer->examplesCollector(), persister);
}

bool CFieldDataCategorizer::isPersistenceNeeded(const std::string& description) const {
    // Pass on the request in case we're chained
    if (m_OutputHandler.isPersistenceNeeded(description)) {
        return true;
    }

    if (m_NumRecordsHandled == 0) {
        LOG_DEBUG(<< "Zero records were handled - will not attempt to persist "
                  << description << ".");
        return false;
    }
    return true;
}

bool CFieldDataCategorizer::doPersistState(const model::CDataCategorizer::TPersistFunc& dataCategorizerPersistFunc,
                                           const model::CCategoryExamplesCollector& examplesCollector,
                                           core::CDataAdder& persister) {
    try {
        core::CStateCompressor compressor{persister};

        core::CDataAdder::TOStreamP strm{
            compressor.addStreamed(ML_STATE_INDEX, m_JobId + '_' + STATE_TYPE)};

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
            this->acceptPersistInserter(dataCategorizerPersistFunc,
                                        examplesCollector, inserter);
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

void CFieldDataCategorizer::acceptPersistInserter(
    const model::CDataCategorizer::TPersistFunc& dataCategorizerPersistFunc,
    const model::CCategoryExamplesCollector& examplesCollector,
    core::CStatePersistInserter& inserter) const {
    inserter.insertValue(VERSION_TAG, STATE_VERSION);
    inserter.insertLevel(CATEGORIZER_TAG, dataCategorizerPersistFunc);
    inserter.insertLevel(EXAMPLES_COLLECTOR_TAG,
                         std::bind(&model::CCategoryExamplesCollector::acceptPersistInserter,
                                   &examplesCollector, std::placeholders::_1));
}

bool CFieldDataCategorizer::periodicPersistStateInBackground() {
    LOG_DEBUG(<< "Periodic persist categorizer state");

    // Make sure that the model size stats are up to date
    m_Limits.resourceMonitor().forceRefresh(*m_DataCategorizer);

    // Pass on the request in case we're chained
    if (m_OutputHandler.periodicPersistStateInBackground() == false) {
        return false;
    }

    if (m_PeriodicPersister == nullptr) {
        LOG_ERROR(<< "NULL persistence manager");
        return false;
    }

    if (m_PeriodicPersister->addPersistFunc(std::bind(
            &CFieldDataCategorizer::doPersistState, this,
            // Do NOT add std::ref wrappers
            // around these arguments - they
            // MUST be copied for thread safety
            m_DataCategorizer->makeBackgroundPersistFunc(),
            m_DataCategorizer->examplesCollector(), std::placeholders::_1)) == false) {
        LOG_ERROR(<< "Failed to add categorizer background persistence function");
        return false;
    }

    m_PeriodicPersister->useBackgroundPersistence();

    return true;
}

bool CFieldDataCategorizer::periodicPersistStateInForeground() {
    LOG_DEBUG(<< "Periodic persist categorizer state");

    if (m_PeriodicPersister == nullptr) {
        return false;
    }

    // Do NOT pass this request on to the output chainer. That logic is already present in persistState.
    if (m_PeriodicPersister->addPersistFunc([&](core::CDataAdder& persister) {
            return this->persistState(persister, "Periodic foreground persist at ");
        }) == false) {
        LOG_ERROR(<< "Failed to add categorizer foreground persistence function");
        return false;
    }

    m_PeriodicPersister->useForegroundPersistence();

    return true;
}

void CFieldDataCategorizer::resetAfterCorruptRestore() {
    LOG_WARN(<< "Discarding corrupt categorizer state - will re-categorize from scratch");

    m_SearchTerms.clear();
    m_SearchTermsRegex.clear();
    this->createCategorizer(m_CategorizationFieldName);
}

bool CFieldDataCategorizer::handleControlMessage(const std::string& controlMessage) {
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

void CFieldDataCategorizer::acknowledgeFlush(const std::string& flushId) {
    if (flushId.empty()) {
        LOG_ERROR(<< "Received flush control message with no ID");
    } else {
        LOG_TRACE(<< "Received flush control message with ID " << flushId);
    }
    m_JsonOutputWriter.acknowledgeFlush(flushId, 0);
}
}
}
