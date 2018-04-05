/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */
#include <api/CFieldDataTyper.h>

#include <core/CDataAdder.h>
#include <core/CDataSearcher.h>
#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CLogger.h>
#include <core/CStateCompressor.h>
#include <core/CStateDecompressor.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CStringUtils.h>

#include <api/CBackgroundPersister.h>
#include <api/CFieldConfig.h>
#include <api/CJsonOutputWriter.h>
#include <api/COutputHandler.h>
#include <api/CTokenListReverseSearchCreator.h>

#include <boost/bind.hpp>

#include <sstream>

namespace ml {
namespace api {

namespace {

const std::string VERSION_TAG("a");
const std::string TYPER_TAG("b");
const std::string EXAMPLES_COLLECTOR_TAG("c");
} // unnamed

// Initialise statics
const std::string CFieldDataTyper::ML_STATE_INDEX(".ml-state");
const std::string CFieldDataTyper::MLCATEGORY_NAME("mlcategory");
const double CFieldDataTyper::SIMILARITY_THRESHOLD(0.7);
const std::string CFieldDataTyper::STATE_TYPE("categorizer_state");
const std::string CFieldDataTyper::STATE_VERSION("1");

CFieldDataTyper::CFieldDataTyper(const std::string& jobId,
                                 const CFieldConfig& config,
                                 const model::CLimits& limits,
                                 COutputHandler& outputHandler,
                                 CJsonOutputWriter& jsonOutputWriter,
                                 CBackgroundPersister* periodicPersister)
    : m_JobId(jobId),
      m_OutputHandler(outputHandler),
      m_ExtraFieldNames(1, MLCATEGORY_NAME),
      m_WriteFieldNames(true),
      m_NumRecordsHandled(0),
      m_OutputFieldCategory(m_Overrides[MLCATEGORY_NAME]),
      m_MaxMatchingLength(0),
      m_JsonOutputWriter(jsonOutputWriter),
      m_ExamplesCollector(limits.maxExamples()),
      m_CategorizationFieldName(config.categorizationFieldName()),
      m_CategorizationFilter(),
      m_PeriodicPersister(periodicPersister) {
    this->createTyper(m_CategorizationFieldName);

    LOG_DEBUG("Configuring categorization filtering");
    m_CategorizationFilter.configure(config.categorizationFilters());
}

CFieldDataTyper::~CFieldDataTyper() {
    m_DataTyper->dumpStats();
}

void CFieldDataTyper::newOutputStream() {
    m_WriteFieldNames = true;
    m_OutputHandler.newOutputStream();
}

bool CFieldDataTyper::handleRecord(const TStrStrUMap& dataRowFields) {
    // First time through we output the field names
    if (m_WriteFieldNames) {
        TStrVec fieldNames;
        fieldNames.reserve(dataRowFields.size() + 1);
        for (const auto& entry : dataRowFields) {
            fieldNames.push_back(entry.first);
        }

        if (m_OutputHandler.fieldNames(fieldNames, m_ExtraFieldNames) == false) {
            LOG_ERROR("Unable to set field names for output:" << core_t::LINE_ENDING << this->debugPrintRecord(dataRowFields));
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

    m_OutputFieldCategory = core::CStringUtils::typeToString(this->computeType(dataRowFields));

    if (m_OutputHandler.writeRow(dataRowFields, m_Overrides) == false) {
        LOG_ERROR("Unable to write output with type " << m_OutputFieldCategory << " for input:" << core_t::LINE_ENDING
                                                      << this->debugPrintRecord(dataRowFields));
        return false;
    }
    ++m_NumRecordsHandled;
    return true;
}

void CFieldDataTyper::finalise() {
    // Pass on the request in case we're chained
    m_OutputHandler.finalise();

    // Wait for any ongoing periodic persist to complete, so that the data adder
    // is not used by both a periodic periodic persist and final persist at the
    // same time
    if (m_PeriodicPersister != nullptr) {
        m_PeriodicPersister->waitForIdle();
    }
}

uint64_t CFieldDataTyper::numRecordsHandled() const {
    return m_NumRecordsHandled;
}

COutputHandler& CFieldDataTyper::outputHandler() {
    return m_OutputHandler;
}

int CFieldDataTyper::computeType(const TStrStrUMap& dataRowFields) {
    const std::string& categorizationFieldName = m_DataTyper->fieldName();
    TStrStrUMapCItr fieldIter = dataRowFields.find(categorizationFieldName);
    if (fieldIter == dataRowFields.end()) {
        LOG_WARN("Assigning type -1 to record with no " << categorizationFieldName << " field:" << core_t::LINE_ENDING
                                                        << this->debugPrintRecord(dataRowFields));
        return -1;
    }

    const std::string& fieldValue = fieldIter->second;
    if (fieldValue.empty()) {
        LOG_WARN("Assigning type -1 to record with blank " << categorizationFieldName << " field:" << core_t::LINE_ENDING
                                                           << this->debugPrintRecord(dataRowFields));
        return -1;
    }

    int type = -1;
    if (m_CategorizationFilter.empty()) {
        type = m_DataTyper->computeType(false, dataRowFields, fieldValue, fieldValue.length());
    } else {
        std::string filtered = m_CategorizationFilter.apply(fieldValue);
        type = m_DataTyper->computeType(false, dataRowFields, filtered, fieldValue.length());
    }
    if (type < 1) {
        return -1;
    }

    bool exampleAdded = m_ExamplesCollector.add(static_cast<std::size_t>(type), fieldValue);
    bool searchTermsChanged = this->createReverseSearch(type);
    if (exampleAdded || searchTermsChanged) {
        const TStrSet& examples = m_ExamplesCollector.examples(static_cast<std::size_t>(type));
        m_JsonOutputWriter.writeCategoryDefinition(type, m_SearchTerms, m_SearchTermsRegex, m_MaxMatchingLength, examples);
    }

    // Check if a periodic persist is due.
    if (m_PeriodicPersister != nullptr) {
        m_PeriodicPersister->startBackgroundPersistIfAppropriate();
    }

    return type;
}

void CFieldDataTyper::createTyper(const std::string& fieldName) {
    // TODO - if we ever have more than one data typer class, this should be
    // replaced with a factory
    TTokenListDataTyperKeepsFields::TTokenListReverseSearchCreatorIntfCPtr reverseSearchCreator(
        new CTokenListReverseSearchCreator(fieldName));
    m_DataTyper.reset(new TTokenListDataTyperKeepsFields(reverseSearchCreator, SIMILARITY_THRESHOLD, fieldName));

    LOG_TRACE("Created new categorizer for field '" << fieldName << "'");
}

bool CFieldDataTyper::createReverseSearch(int type) {
    bool wasCached(false);
    if (m_DataTyper->createReverseSearch(type, m_SearchTerms, m_SearchTermsRegex, m_MaxMatchingLength, wasCached) == false) {
        m_SearchTerms.clear();
        m_SearchTermsRegex.clear();
    }
    return !wasCached;
}

bool CFieldDataTyper::restoreState(core::CDataSearcher& restoreSearcher, core_t::TTime& completeToTime) {
    // Pass on the request in case we're chained
    if (m_OutputHandler.restoreState(restoreSearcher, completeToTime) == false) {
        return false;
    }

    LOG_DEBUG("Restore typer state");

    try {
        // Restore from Elasticsearch compressed data
        core::CStateDecompressor decompressor(restoreSearcher);
        decompressor.setStateRestoreSearch(ML_STATE_INDEX);

        core::CDataSearcher::TIStreamP strm(decompressor.search(1, 1));
        if (strm == 0) {
            LOG_ERROR("Unable to connect to data store");
            return false;
        }

        if (strm->bad()) {
            LOG_ERROR("Categorizer state restoration returned a bad stream");
            return false;
        }

        if (strm->fail()) {
            // This is fatal. If the stream exists and has failed then state is missing
            LOG_ERROR("Categorizer state restoration returned a failed stream");
            return false;
        }

        // We're dealing with streaming JSON state
        core::CJsonStateRestoreTraverser traverser(*strm);

        if (this->acceptRestoreTraverser(traverser) == false) {
            LOG_ERROR("JSON restore failed");
            return false;
        }
    } catch (std::exception& e) {
        LOG_ERROR("Failed to restore state! " << e.what());
        // This is fatal in terms of the categorizer we attempted to restore,
        // but returning false here can throw the system into a repeated cycle
        // of failure.  It's better to reset the categorizer and re-categorize from
        // scratch.
        this->resetAfterCorruptRestore();
        return true;
    }

    return true;
}

bool CFieldDataTyper::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    const std::string& firstFieldName = traverser.name();
    if (traverser.isEof()) {
        LOG_ERROR("Expected categorizer persisted state but no state exists");
        return false;
    }

    if (firstFieldName == VERSION_TAG) {
        std::string version;
        if (core::CStringUtils::stringToType(traverser.value(), version) == false) {
            LOG_ERROR("Cannot restore categorizer, invalid version: " << traverser.value());
            return false;
        }
        if (version != STATE_VERSION) {
            LOG_DEBUG("Categorizer has not been restored as the version has changed");
            return true;
        }
    } else {
        LOG_ERROR("Cannot restore categorizer - " << VERSION_TAG << " element expected but found " << traverser.name() << '='
                                                  << traverser.value());
        return false;
    }

    if (traverser.next() == false) {
        LOG_ERROR("Cannot restore categorizer - end of object reached when " << TYPER_TAG << " was expected");
        return false;
    }

    if (traverser.name() == TYPER_TAG) {
        if (traverser.traverseSubLevel(boost::bind(&CDataTyper::acceptRestoreTraverser, m_DataTyper, _1)) == false) {
            LOG_ERROR("Cannot restore categorizer, unexpected element: " << traverser.value());
            return false;
        }
    } else {
        LOG_ERROR("Cannot restore categorizer - " << TYPER_TAG << " element expected but found " << traverser.name() << '='
                                                  << traverser.value());
        return false;
    }

    if (traverser.next() == false) {
        LOG_ERROR("Cannot restore categorizer - end of object reached when " << EXAMPLES_COLLECTOR_TAG << " was expected");
        return false;
    }

    if (traverser.name() == EXAMPLES_COLLECTOR_TAG) {
        if (traverser.traverseSubLevel(
                boost::bind(&CCategoryExamplesCollector::acceptRestoreTraverser, boost::ref(m_ExamplesCollector), _1)) == false ||
            traverser.haveBadState()) {
            LOG_ERROR("Cannot restore categorizer, unexpected element: " << traverser.value());
            return false;
        }
    } else {
        LOG_ERROR("Cannot restore categorizer - " << EXAMPLES_COLLECTOR_TAG << " element expected but found " << traverser.name() << '='
                                                  << traverser.value());
        return false;
    }

    return true;
}

bool CFieldDataTyper::persistState(core::CDataAdder& persister) {
    if (m_PeriodicPersister != nullptr) {
        // This will not happen if finalise() was called before persisting state
        if (m_PeriodicPersister->isBusy()) {
            LOG_ERROR("Cannot do final persistence of state - periodic "
                      "persister still busy");
            return false;
        }
    }

    // Pass on the request in case we're chained
    if (m_OutputHandler.persistState(persister) == false) {
        return false;
    }

    LOG_DEBUG("Persist typer state");

    return this->doPersistState(m_DataTyper->makePersistFunc(), m_ExamplesCollector, persister);
}

bool CFieldDataTyper::doPersistState(const CDataTyper::TPersistFunc& dataTyperPersistFunc,
                                     const CCategoryExamplesCollector& examplesCollector,
                                     core::CDataAdder& persister) {
    try {
        core::CStateCompressor compressor(persister);

        core::CDataAdder::TOStreamP strm = compressor.addStreamed(ML_STATE_INDEX, m_JobId + '_' + STATE_TYPE);

        if (strm == 0) {
            LOG_ERROR("Failed to create persistence stream");
            return false;
        }

        if (!strm->good()) {
            LOG_ERROR("Persistence stream is bad before stream of "
                      "state for the categorizer");
            return false;
        }

        {
            // Keep the JSON inserter scoped as it only finishes the stream
            // when it is desctructed
            core::CJsonStatePersistInserter inserter(*strm);
            this->acceptPersistInserter(dataTyperPersistFunc, examplesCollector, inserter);
        }

        if (strm->bad()) {
            LOG_ERROR("Persistence stream went bad during stream of "
                      "state for the categorizer");
            return false;
        }

        if (compressor.streamComplete(strm, true) == false || strm->bad()) {
            LOG_ERROR("Failed to complete last persistence stream");
            return false;
        }
    } catch (std::exception& e) {
        LOG_ERROR("Failed to persist state! " << e.what());
        return false;
    }
    return true;
}

void CFieldDataTyper::acceptPersistInserter(const CDataTyper::TPersistFunc& dataTyperPersistFunc,
                                            const CCategoryExamplesCollector& examplesCollector,
                                            core::CStatePersistInserter& inserter) const {
    inserter.insertValue(VERSION_TAG, STATE_VERSION);
    inserter.insertLevel(TYPER_TAG, dataTyperPersistFunc);
    inserter.insertLevel(EXAMPLES_COLLECTOR_TAG, boost::bind(&CCategoryExamplesCollector::acceptPersistInserter, &examplesCollector, _1));
}

bool CFieldDataTyper::periodicPersistState(CBackgroundPersister& persister) {
    LOG_DEBUG("Periodic persist typer state");

    // Pass on the request in case we're chained
    if (m_OutputHandler.periodicPersistState(persister) == false) {
        return false;
    }

    if (persister.addPersistFunc(boost::bind(&CFieldDataTyper::doPersistState,
                                             this,
                                             // Do NOT add boost::ref wrappers
                                             // around these arguments - they
                                             // MUST be copied for thread safety
                                             m_DataTyper->makePersistFunc(),
                                             m_ExamplesCollector,
                                             _1)) == false) {
        LOG_ERROR("Failed to add categorizer background persistence function");
        return false;
    }

    return true;
}

void CFieldDataTyper::resetAfterCorruptRestore() {
    LOG_WARN("Discarding corrupt categorizer state - will re-categorize from scratch");

    m_SearchTerms.clear();
    m_SearchTermsRegex.clear();
    this->createTyper(m_CategorizationFieldName);
    m_ExamplesCollector.clear();
}

bool CFieldDataTyper::handleControlMessage(const std::string& controlMessage) {
    if (controlMessage.empty()) {
        LOG_ERROR("Programmatic error - handleControlMessage should only be "
                  "called with non-empty control messages");
        return false;
    }

    switch (controlMessage[0]) {
    case ' ':
        // Spaces are just used to fill the buffers and force prior messages
        // through the system - we don't need to do anything else
        LOG_TRACE("Received space control message of length " << controlMessage.length());
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
        LOG_WARN("Ignoring unknown control message of length " << controlMessage.length() << " beginning with '" << controlMessage[0]
                                                               << '\'');
        // Don't return false here (for the time being at least), as it
        // seems excessive to cause the entire job to fail
        break;
    }

    return true;
}

void CFieldDataTyper::acknowledgeFlush(const std::string& flushId) {
    if (flushId.empty()) {
        LOG_ERROR("Received flush control message with no ID");
    } else {
        LOG_TRACE("Received flush control message with ID " << flushId);
    }
    m_JsonOutputWriter.acknowledgeFlush(flushId, 0);
}
}
}
