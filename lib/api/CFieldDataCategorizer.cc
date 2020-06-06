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

#include <api/CFieldConfig.h>
#include <api/CJsonOutputWriter.h>
#include <api/CNoopCategoryIdMapper.h>
#include <api/COutputHandler.h>
#include <api/CPerPartitionCategoryIdMapper.h>
#include <api/CPersistenceManager.h>

#include <memory>
#include <sstream>

namespace ml {
namespace api {

namespace {

const std::string VERSION_TAG{"a"};
const std::string CATEGORIZER_TAG{"b"};
const std::string EXAMPLES_COLLECTOR_TAG{"c"};
const std::string PARTITION_FIELD_VALUE_TAG{"d"};
const std::string CATEGORY_ID_MAPPER_TAG{"e"};
const std::string CATEGORIZER_ALLOCATION_FAILURES{"f"};
const std::string EMPTY_STRING;
} // unnamed

// Initialise statics
const std::string CFieldDataCategorizer::ML_STATE_INDEX{".ml-state"};
const std::string CFieldDataCategorizer::MLCATEGORY_NAME{"mlcategory"};
const double CFieldDataCategorizer::SIMILARITY_THRESHOLD{0.7};
const std::string CFieldDataCategorizer::STATE_TYPE{"categorizer_state"};
const std::string CFieldDataCategorizer::STATE_VERSION{"1"};

CFieldDataCategorizer::CFieldDataCategorizer(const std::string& jobId,
                                             const CFieldConfig& config,
                                             model::CLimits& limits,
                                             COutputHandler& outputHandler,
                                             CJsonOutputWriter& jsonOutputWriter,
                                             CPersistenceManager* persistenceManager)
    : m_JobId{jobId}, m_Limits{limits}, m_OutputHandler{outputHandler}, m_ExtraFieldNames{1, MLCATEGORY_NAME},
      m_OutputFieldCategory{m_Overrides[MLCATEGORY_NAME]}, m_JsonOutputWriter{jsonOutputWriter},
      m_PartitionFieldName{config.categorizationPartitionFieldName()},
      m_CategorizationFieldName{config.categorizationFieldName()}, m_PersistenceManager{persistenceManager} {

    if (config.categorizationFilters().empty() == false) {
        LOG_DEBUG(<< "Configuring categorization filtering");
        m_CategorizationFilter.configure(config.categorizationFilters());
    }

    if (m_PartitionFieldName.empty()) {
        m_CategoryIdMapper = std::make_unique<CNoopCategoryIdMapper>();
    } else {
        m_CategoryIdMapper = std::make_unique<CPerPartitionCategoryIdMapper>();
    }
}

CFieldDataCategorizer::~CFieldDataCategorizer() {
    for (const auto& dataCategorizerEntry : m_DataCategorizers) {
        dataCategorizerEntry.second->dumpStats(
            [this, &dataCategorizerEntry](model::CLocalCategoryId localCategoryId) {
                return m_CategoryIdMapper
                    ->map(dataCategorizerEntry.first, localCategoryId)
                    .print();
            });
    }
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
    auto iter = dataRowFields.find(CONTROL_FIELD_NAME);
    if (iter != dataRowFields.end() && !iter->second.empty()) {
        // Always handle control messages, but signal completion of handling ONLY if we are the last handler
        // e.g. flush requests are acknowledged here if this is the last handler
        bool msgHandled = this->handleControlMessage(
            iter->second, !m_OutputHandler.consumesControlMessages());
        if (m_OutputHandler.consumesControlMessages()) {
            return m_OutputHandler.writeRow(dataRowFields, m_Overrides);
        }
        return msgHandled;
    }

    CGlobalCategoryId globalCategoryId{this->computeCategory(dataRowFields)};
    if (globalCategoryId.isHardFailure()) {
        // Still return true here, because false would fail the entire job
        return true;
    }

    m_OutputFieldCategory = core::CStringUtils::typeToString(globalCategoryId.globalId());
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
    for (const auto& dataCategorizerEntry : m_DataCategorizers) {
        m_Limits.resourceMonitor().forceRefresh(*dataCategorizerEntry.second);
    }
    writeOutChangedCategories();
    // Pass on the request in case we're chained
    m_OutputHandler.finalise();

    // Wait for any ongoing periodic persist to complete, so that the data adder
    // is not used by both a periodic periodic persist and final persist at the
    // same time
    if (m_PersistenceManager != nullptr) {
        m_PersistenceManager->waitForIdle();
    }
}

std::uint64_t CFieldDataCategorizer::numRecordsHandled() const {
    return m_NumRecordsHandled;
}

COutputHandler& CFieldDataCategorizer::outputHandler() {
    return m_OutputHandler;
}

CGlobalCategoryId CFieldDataCategorizer::computeCategory(const TStrStrUMap& dataRowFields) {
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
    model::CDataCategorizer* dataCategorizer{this->categorizerPtrForKey(partitionFieldValue)};
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
    model::CLocalCategoryId localCategoryId;
    if (m_CategorizationFilter.empty()) {
        localCategoryId = dataCategorizer->computeCategory(
            false, dataRowFields, fieldValue, fieldValue.length());
    } else {
        std::string filtered = m_CategorizationFilter.apply(fieldValue);
        localCategoryId = dataCategorizer->computeCategory(
            false, dataRowFields, filtered, fieldValue.length());
    }
    globalCategoryId = m_CategoryIdMapper->map(partitionFieldValue, localCategoryId);
    if (globalCategoryId.isValid() == false) {
        return globalCategoryId;
    }

    bool exampleAdded{dataCategorizer->addExample(localCategoryId, fieldValue)};
    bool searchTermsChanged{this->createReverseSearch(*dataCategorizer, localCategoryId)};
    if (exampleAdded || searchTermsChanged) {
        //! signal that we noticed the change and are persisting here
        dataCategorizer->categoryChangedAndReset(localCategoryId);
        m_JsonOutputWriter.writeCategoryDefinition(
            m_PartitionFieldName, partitionFieldValue, globalCategoryId,
            m_SearchTerms, m_SearchTermsRegex, m_MaxMatchingLength,
            dataCategorizer->examplesCollector().examples(localCategoryId),
            dataCategorizer->numMatches(localCategoryId),
            m_CategoryIdMapper->mapVec(partitionFieldValue,
                                       dataCategorizer->usurpedCategories(localCategoryId)));
        if (localCategoryId.id() % 10 == 0) {
            // Even if memory limiting is disabled, force a refresh occasionally
            // so the user has some idea what's going on with memory.
            m_Limits.resourceMonitor().forceRefresh(*dataCategorizer);
        } else {
            m_Limits.resourceMonitor().refresh(*dataCategorizer);
        }
    }

    // Check if a periodic persist is due.
    if (m_PersistenceManager != nullptr) {
        m_PersistenceManager->startPersistIfAppropriate();
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

model::CDataCategorizer*
CFieldDataCategorizer::categorizerPtrForKey(const std::string& partitionFieldValue) {
    auto iter = m_DataCategorizers.find(partitionFieldValue);
    if (iter != m_DataCategorizers.end()) {
        return iter->second.get();
    }

    if (m_Limits.resourceMonitor().areAllocationsAllowed() == false) {
        // The categorizer doesn't exist, but we are not allowed to create it
        return nullptr;
    }

    // TODO - if we ever have more than one data categorizer class, this
    // should be replaced with a factory
    auto ptr = std::make_shared<TTokenListDataCategorizerKeepsFields>(
        m_Limits, std::make_shared<model::CTokenListReverseSearchCreator>(m_CategorizationFieldName),
        SIMILARITY_THRESHOLD, m_CategorizationFieldName);
    m_DataCategorizers.emplace(partitionFieldValue, ptr);
    LOG_TRACE(<< "Created new categorizer for '" << partitionFieldValue << '/'
              << m_CategorizationFieldName << "'");
    return ptr.get();
}

model::CDataCategorizer&
CFieldDataCategorizer::categorizerForKey(const std::string& partitionFieldValue) {
    return *this->categorizerPtrForKey(partitionFieldValue);
}

bool CFieldDataCategorizer::createReverseSearch(model::CDataCategorizer& dataCategorizer,
                                                model::CLocalCategoryId localCategoryId) {
    bool wasCached(false);
    if (dataCategorizer.createReverseSearch(localCategoryId, m_SearchTerms, m_SearchTermsRegex,
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
    }

    for (const auto& categorizerKey : categorizerKeys) {
        model::CDataCategorizer& dataCategorizer{this->categorizerForKey(categorizerKey)};
        if (traverser.next() == false) {
            LOG_ERROR(<< "Cannot restore categorizer - end of object reached when "
                      << CATEGORIZER_TAG << " was expected");
            return false;
        }

        if (traverser.name() == CATEGORIZER_TAG) {
            if (traverser.traverseSubLevel([&dataCategorizer](core::CStateRestoreTraverser& traverser_) {
                    return dataCategorizer.acceptRestoreTraverser(traverser_);
                }) == false) {
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
            if (dataCategorizer.restoreExamplesCollector(traverser) == false) {
                return false;
            }
        } else {
            LOG_ERROR(<< "Cannot restore categorizer - " << EXAMPLES_COLLECTOR_TAG << " element expected but found "
                      << traverser.name() << '=' << traverser.value());
            return false;
        }
    }

    // This won't be present in pre-7.9 state, but that's OK because pre-7.9
    // jobs must be using the no-op category ID mapper, and that is stateless
    // (restore is a no-op for that mapper)
    if (traverser.next() && traverser.name() == CATEGORY_ID_MAPPER_TAG) {
        if (traverser.traverseSubLevel([this](core::CStateRestoreTraverser& traverser_) {
                return m_CategoryIdMapper->acceptRestoreTraverser(traverser_);
            }) == false) {
            LOG_ERROR(<< "Cannot restore category ID mapper, unexpected element: "
                      << traverser.value());
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
    if (m_OutputHandler.persistStateInForeground(persister, descriptionPrefix) == false) {
        return false;
    }

    LOG_DEBUG(<< "Persist categorizer state");

    TStrVec partitionFieldValues;
    TPersistFuncVec dataCategorizerPersistFuncs;
    TCategoryExamplesCollectorsCRefVec examplesCollectors;

    if (m_PartitionFieldName.empty()) {
        model::CDataCategorizer& dataCategorizer{categorizerForKey(EMPTY_STRING)};
        dataCategorizerPersistFuncs.emplace_back(dataCategorizer.makeForegroundPersistFunc());
        examplesCollectors.push_back(dataCategorizer.examplesCollector());
    } else {
        if (m_DataCategorizers.empty()) {
            LOG_WARN(<< "No partition-specific categorizers found");
            return true;
        }
        partitionFieldValues.reserve(m_DataCategorizers.size());
        for (auto& dataCategorizerEntry : m_DataCategorizers) {
            partitionFieldValues.push_back(dataCategorizerEntry.first);
        }

        // Persist in sorted order to ensure consistency
        std::sort(partitionFieldValues.begin(), partitionFieldValues.end());
        dataCategorizerPersistFuncs.reserve(partitionFieldValues.size());
        examplesCollectors.reserve(partitionFieldValues.size());
        for (const auto& partitionFieldValue : partitionFieldValues) {
            model::CDataCategorizer& dataCategorizer{categorizerForKey(partitionFieldValue)};
            dataCategorizerPersistFuncs.emplace_back(
                dataCategorizer.makeForegroundPersistFunc());
            examplesCollectors.push_back(std::cref(dataCategorizer.examplesCollector()));
        }
    }

    return this->doPersistState(partitionFieldValues, dataCategorizerPersistFuncs,
                                examplesCollectors, *m_CategoryIdMapper,
                                m_CategorizerAllocationFailures, persister);
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

template<typename EXAMPLES_COLLECTOR_VEC>
bool CFieldDataCategorizer::doPersistState(const TStrVec& partitionFieldValues,
                                           const TPersistFuncVec& dataCategorizerPersistFuncs,
                                           const EXAMPLES_COLLECTOR_VEC& examplesCollectors,
                                           const CCategoryIdMapper& categoryIdMapper,
                                           std::size_t categorizerAllocationFailures,
                                           core::CDataAdder& persister) {
    // All three input vectors should have the same size _unless_ we are not
    // doing per-partition categorization, in which case partition field values
    // should be empty and there should be exactly one categorizer
    if (partitionFieldValues.size() != dataCategorizerPersistFuncs.size() &&
        (dataCategorizerPersistFuncs.size() != 1 || partitionFieldValues.empty() == false)) {
        LOG_ERROR(<< "Programmatic error - doPersistState called with "
                  << dataCategorizerPersistFuncs.size() << " categorizer persistence functions and "
                  << partitionFieldValues.size() << " partition field values");
        return false;
    }
    if (examplesCollectors.size() != dataCategorizerPersistFuncs.size()) {
        LOG_ERROR(<< "Programmatic error - doPersistState called with "
                  << dataCategorizerPersistFuncs.size() << " categorizer persistence functions and "
                  << examplesCollectors.size() << " examples collectors");
        return false;
    }
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
            inserter.insertValue(VERSION_TAG, STATE_VERSION);
            if (partitionFieldValues.empty() == false) {
                core::CPersistUtils::persist(PARTITION_FIELD_VALUE_TAG,
                                             partitionFieldValues, inserter);
            }
            for (std::size_t i = 0; i < dataCategorizerPersistFuncs.size(); ++i) {
                CFieldDataCategorizer::persistSingleCategorizer(
                    dataCategorizerPersistFuncs[i], examplesCollectors[i], inserter);
            }
            inserter.insertLevel(CATEGORY_ID_MAPPER_TAG,
                                 [&categoryIdMapper](core::CStatePersistInserter& inserter_) {
                                     categoryIdMapper.acceptPersistInserter(inserter_);
                                 });
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

void CFieldDataCategorizer::persistSingleCategorizer(
    const model::CDataCategorizer::TPersistFunc& dataCategorizerPersistFunc,
    const model::CCategoryExamplesCollector& examplesCollector,
    core::CStatePersistInserter& inserter) {
    inserter.insertLevel(CATEGORIZER_TAG, dataCategorizerPersistFunc);
    inserter.insertLevel(EXAMPLES_COLLECTOR_TAG,
                         [&examplesCollector](core::CStatePersistInserter& inserter_) {
                             examplesCollector.acceptPersistInserter(inserter_);
                         });
}

bool CFieldDataCategorizer::periodicPersistStateInBackground() {
    LOG_DEBUG(<< "Periodic persist categorizer state");

    // Make sure that the model size stats are up to date
    TStrVec partitionFieldValues;
    if (m_PartitionFieldName.empty() == false) {
        partitionFieldValues.reserve(m_DataCategorizers.size());
    }
    for (auto& dataCategorizerEntry : m_DataCategorizers) {
        m_Limits.resourceMonitor().forceRefresh(*dataCategorizerEntry.second);
        if (m_PartitionFieldName.empty() == false) {
            partitionFieldValues.push_back(dataCategorizerEntry.first);
        }
    }

    // Pass on the request in case we're chained
    if (m_OutputHandler.periodicPersistStateInBackground() == false) {
        return false;
    }

    if (m_PersistenceManager == nullptr) {
        LOG_ERROR(<< "NULL persistence manager");
        return false;
    }

    TPersistFuncVec dataCategorizerPersistFuncs;
    TCategoryExamplesCollectorsVec examplesCollectors;

    if (m_PartitionFieldName.empty()) {
        model::CDataCategorizer& dataCategorizer{categorizerForKey(EMPTY_STRING)};
        dataCategorizerPersistFuncs.emplace_back(dataCategorizer.makeBackgroundPersistFunc());
        examplesCollectors.push_back(dataCategorizer.examplesCollector());
    } else {
        if (partitionFieldValues.empty()) {
            LOG_WARN(<< "No partition-specific categorizers found");
            return true;
        }
        // Persist in sorted order to ensure consistency
        std::sort(partitionFieldValues.begin(), partitionFieldValues.end());
        dataCategorizerPersistFuncs.reserve(partitionFieldValues.size());
        examplesCollectors.reserve(partitionFieldValues.size());
        for (const auto& partitionFieldValue : partitionFieldValues) {
            model::CDataCategorizer& dataCategorizer{categorizerForKey(partitionFieldValue)};
            dataCategorizerPersistFuncs.emplace_back(
                dataCategorizer.makeBackgroundPersistFunc());
            examplesCollectors.push_back(dataCategorizer.examplesCollector());
        }
    }

    // std::function is required to be copyable, so we need a shared pointer
    // to the cloned category ID mapper rather than a unique pointer (which
    // is only movable)
    using TCategoryIdMapperPtr = std::shared_ptr<CCategoryIdMapper>;
    TCategoryIdMapperPtr clonedCategoryIdMapper{m_CategoryIdMapper->clone()};
    // Do NOT pass the captures by reference - they
    // MUST be copied for thread safety
    if (m_PersistenceManager->addPersistFunc([
            this, partitionFieldValues = std::move(partitionFieldValues),
            dataCategorizerPersistFuncs = std::move(dataCategorizerPersistFuncs),
            examplesCollectors = std::move(examplesCollectors), clonedCategoryIdMapper,
            categorizerAllocationFailures = m_CategorizerAllocationFailures
        ](core::CDataAdder & persister) {
            return this->doPersistState(partitionFieldValues, dataCategorizerPersistFuncs,
                                        examplesCollectors, *clonedCategoryIdMapper,
                                        categorizerAllocationFailures, persister);
        }) == false) {
        LOG_ERROR(<< "Failed to add categorizer background persistence function");
        return false;
    }

    m_PersistenceManager->useBackgroundPersistence();

    return true;
}

bool CFieldDataCategorizer::periodicPersistStateInForeground() {
    LOG_DEBUG(<< "Periodic persist categorizer state");

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

    m_SearchTerms.clear();
    m_SearchTermsRegex.clear();
    m_DataCategorizers.clear();
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
    writeOutChangedCategories();
    if (lastHandler) {
        m_JsonOutputWriter.acknowledgeFlush(flushId, 0);
    }
}

void CFieldDataCategorizer::writeOutChangedCategories() {
    for (auto& dataCategorizerEntry : m_DataCategorizers) {
        model::CDataCategorizer& dataCategorizer = *dataCategorizerEntry.second;
        std::size_t numCategories{dataCategorizer.numCategories()};
        if (numCategories == 0) {
            continue;
        }
        std::string searchTerms;
        std::string searchTermsRegex;
        std::size_t maxLength;
        bool wasCached{false};
        for (std::size_t index = 0; index < numCategories; ++index) {
            model::CLocalCategoryId localCategoryId{index};
            if (dataCategorizer.categoryChangedAndReset(localCategoryId)) {
                if (dataCategorizer.createReverseSearch(localCategoryId, searchTerms, searchTermsRegex,
                                                        maxLength, wasCached) == false) {
                    LOG_WARN(<< "Unable to create or retrieve reverse search for storing for category: "
                             << m_CategoryIdMapper
                                    ->map(dataCategorizerEntry.first, localCategoryId)
                                    .print());
                    continue;
                }
                LOG_TRACE(<< "Writing out changed category: "
                          << m_CategoryIdMapper
                                 ->map(dataCategorizerEntry.first, localCategoryId)
                                 .print());
                CGlobalCategoryId globalCategoryId{m_CategoryIdMapper->map(
                    dataCategorizerEntry.first, localCategoryId)};
                m_JsonOutputWriter.writeCategoryDefinition(
                    m_PartitionFieldName, dataCategorizerEntry.first,
                    globalCategoryId, searchTerms, searchTermsRegex, maxLength,
                    dataCategorizer.examplesCollector().examples(localCategoryId),
                    dataCategorizer.numMatches(localCategoryId),
                    m_CategoryIdMapper->mapVec(
                        dataCategorizerEntry.first,
                        dataCategorizer.usurpedCategories(localCategoryId)));
            }
        }
    }
}
}
}
