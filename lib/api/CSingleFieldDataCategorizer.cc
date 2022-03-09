/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <api/CSingleFieldDataCategorizer.h>

#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CTimeUtils.h>

#include <model/CCategoryExamplesCollector.h>
#include <model/ModelTypes.h>

#include <api/CAnnotationJsonWriter.h>
#include <api/CJsonOutputWriter.h>

#include <sstream>

namespace {
// For historical reasons, these tags must not duplicate those used in
// CFieldDataCategorizer.cc
// a used in CFieldDataCategorizer.cc
const std::string CATEGORIZER_TAG{"b"};
const std::string EXAMPLES_COLLECTOR_TAG{"c"};
const std::string CATEGORY_ID_MAPPER_TAG{"d"};
// e, f, g used in CFieldDataCategorizer.cc

const std::string EMPTY_STRING;
}

namespace ml {
namespace api {

CSingleFieldDataCategorizer::CSingleFieldDataCategorizer(
    std::string partitionFieldName,
    model::CDataCategorizer::TDataCategorizerUPtr dataCategorizer,
    CCategoryIdMapper::TCategoryIdMapperPtr categoryIdMapper)
    : m_PartitionFieldName{std::move(partitionFieldName)},
      m_DataCategorizer{std::move(dataCategorizer)}, m_CategoryIdMapper{std::move(categoryIdMapper)} {
}

void CSingleFieldDataCategorizer::dumpStats() const {
    m_DataCategorizer->dumpStats([this](model::CLocalCategoryId localCategoryId) {
        return m_CategoryIdMapper->map(localCategoryId).print();
    });
}

CGlobalCategoryId CSingleFieldDataCategorizer::computeAndUpdateCategory(
    bool isDryRun,
    const model::CDataCategorizer::TStrStrUMap& fields,
    const TOptionalTime& messageTime,
    const std::string& messageToCategorize,
    const std::string& rawMessage,
    model::CResourceMonitor& resourceMonitor,
    CJsonOutputWriter& jsonOutputWriter) {
    model::CLocalCategoryId localCategoryId{m_DataCategorizer->computeCategory(
        isDryRun, fields, messageToCategorize, rawMessage.length())};
    CGlobalCategoryId globalCategoryId{m_CategoryIdMapper->map(localCategoryId)};
    if (globalCategoryId.isValid() == false) {
        return globalCategoryId;
    }

    if (messageTime.has_value()) {
        m_LastMessageTime = messageTime;
    }
    bool exampleAdded{m_DataCategorizer->addExample(localCategoryId, rawMessage)};
    bool searchTermsChanged{m_DataCategorizer->cacheReverseSearch(localCategoryId)};
    if (exampleAdded || searchTermsChanged) {
        // In this case we are certain that there will have been a change, as
        // the count of the chosen category will have been incremented
        m_DataCategorizer->writeCategoryIfChanged(
            localCategoryId,
            [this, &jsonOutputWriter](
                model::CLocalCategoryId localCategoryId_, const std::string& terms,
                const std::string& regex, std::size_t maxMatchingFieldLength,
                const model::CCategoryExamplesCollector::TStrFSet& examples, std::size_t numMatches,
                const model::CDataCategorizer::TLocalCategoryIdVec& usurpedCategories) {
                jsonOutputWriter.writeCategoryDefinition(
                    m_PartitionFieldName, m_CategoryIdMapper->categorizerKey(),
                    m_CategoryIdMapper->map(localCategoryId_), terms, regex,
                    maxMatchingFieldLength, examples, numMatches,
                    m_CategoryIdMapper->mapVec(usurpedCategories));
            });
        if (localCategoryId.id() % 10 == 0) {
            // Even if memory limiting is disabled, force a refresh occasionally
            // so the user has some idea what's going on with memory.
            resourceMonitor.forceRefresh(*m_DataCategorizer);
        } else {
            resourceMonitor.refresh(*m_DataCategorizer);
        }
    }
    return globalCategoryId;
}

CSingleFieldDataCategorizer::TPersistFunc
CSingleFieldDataCategorizer::makeForegroundPersistFunc() const {
    model::CDataCategorizer::TPersistFunc categorizerPersistFunc{
        m_DataCategorizer->makeForegroundPersistFunc()};

    return [ categorizerPersistFunc = std::move(categorizerPersistFunc),
             this ](core::CStatePersistInserter & inserter) {
        CSingleFieldDataCategorizer::acceptPersistInserter(
            categorizerPersistFunc, m_DataCategorizer->examplesCollector(),
            *m_CategoryIdMapper, inserter);
    };
}

CSingleFieldDataCategorizer::TPersistFunc
CSingleFieldDataCategorizer::makeBackgroundPersistFunc() const {
    model::CDataCategorizer::TPersistFunc categorizerPersistFunc{
        m_DataCategorizer->makeBackgroundPersistFunc()};
    model::CCategoryExamplesCollector examplesCollector{m_DataCategorizer->examplesCollector()};
    CCategoryIdMapper::TCategoryIdMapperPtr categoryIdMapperClone{
        m_CategoryIdMapper->clone()};

    // IMPORTANT: here we are moving the local variables into the lambda, but
    // the local variables must be copies of the underlying data structures.
    // Do NOT change this to avoid the copying.  The background persist
    // function must be able to operate in a different thread on a snapshot of
    // the data at the time it was created.
    return [
        categorizerPersistFunc = std::move(categorizerPersistFunc),
        examplesCollector = std::move(examplesCollector),
        categoryIdMapperClone = std::move(categoryIdMapperClone)
    ](core::CStatePersistInserter & inserter) {
        CSingleFieldDataCategorizer::acceptPersistInserter(
            categorizerPersistFunc, examplesCollector, *categoryIdMapperClone, inserter);
    };
}

bool CSingleFieldDataCategorizer::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    if (traverser.next() == false) {
        LOG_ERROR(<< "Cannot restore categorizer - end of object reached when "
                  << CATEGORIZER_TAG << " was expected");
        return false;
    }

    if (traverser.name() == CATEGORIZER_TAG) {
        if (traverser.traverseSubLevel([this](core::CStateRestoreTraverser& traverser_) {
                return m_DataCategorizer->acceptRestoreTraverser(traverser_);
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
        if (m_DataCategorizer->restoreExamplesCollector(traverser) == false) {
            return false;
        }
    } else {
        LOG_ERROR(<< "Cannot restore categorizer - " << EXAMPLES_COLLECTOR_TAG << " element expected but found "
                  << traverser.name() << '=' << traverser.value());
        return false;
    }

    // Only expect a category ID mapper when per-partition categorization
    // is being used
    if (m_PartitionFieldName.empty() == false) {
        if (traverser.next() == false) {
            LOG_ERROR(<< "Cannot restore categorizer - end of object reached when "
                      << CATEGORY_ID_MAPPER_TAG << " was expected");
            return false;
        }

        if (traverser.name() == CATEGORY_ID_MAPPER_TAG) {
            if (traverser.traverseSubLevel([this](core::CStateRestoreTraverser& traverser_) {
                    return m_CategoryIdMapper->acceptRestoreTraverser(traverser_);
                }) == false) {
                LOG_ERROR(<< "Cannot restore category ID mapper, unexpected element: "
                          << traverser.value());
                return false;
            }
        } else {
            LOG_ERROR(<< "Cannot restore categorizer - " << CATEGORY_ID_MAPPER_TAG << " element expected but found "
                      << traverser.name() << '=' << traverser.value());
            return false;
        }
    }

    return true;
}

void CSingleFieldDataCategorizer::writeChanges(CJsonOutputWriter& jsonOutputWriter,
                                               CAnnotationJsonWriter& annotationJsonWriter) {
    std::size_t numWritten{m_DataCategorizer->writeChangedCategories(
        [this, &jsonOutputWriter](
            model::CLocalCategoryId localCategoryId, const std::string& terms,
            const std::string& regex, std::size_t maxMatchingFieldLength,
            const model::CCategoryExamplesCollector::TStrFSet& examples, std::size_t numMatches,
            const model::CDataCategorizer::TLocalCategoryIdVec& usurpedCategories) {
            jsonOutputWriter.writeCategoryDefinition(
                m_PartitionFieldName, m_CategoryIdMapper->categorizerKey(),
                m_CategoryIdMapper->map(localCategoryId), terms, regex,
                maxMatchingFieldLength, examples, numMatches,
                m_CategoryIdMapper->mapVec(usurpedCategories));
        })};
    LOG_TRACE(<< numWritten << " changed categories written for categorizer "
              << m_CategoryIdMapper->categorizerKey());
    this->writeStatsIfChanged(jsonOutputWriter, annotationJsonWriter);
}

void CSingleFieldDataCategorizer::writeStatsIfUrgent(CJsonOutputWriter& jsonOutputWriter,
                                                     CAnnotationJsonWriter& annotationJsonWriter) {
    if (m_DataCategorizer->isStatsWriteUrgent()) {
        this->writeStatsIfChanged(jsonOutputWriter, annotationJsonWriter);
    }
}

void CSingleFieldDataCategorizer::writeStatsIfChanged(CJsonOutputWriter& jsonOutputWriter,
                                                      CAnnotationJsonWriter& annotationJsonWriter) {
    if (m_DataCategorizer->writeCategorizerStatsIfChanged(
            [this, &jsonOutputWriter, &annotationJsonWriter](
                const model::SCategorizerStats& categorizerStats, bool statusChanged) {
                jsonOutputWriter.writeCategorizerStats(
                    m_PartitionFieldName, m_CategoryIdMapper->categorizerKey(),
                    categorizerStats, m_LastMessageTime);
                if (statusChanged) {
                    std::ostringstream text;
                    text << "Categorization status changed to '"
                         << model_t::print(categorizerStats.s_CategorizationStatus)
                         << '\'';
                    if (m_PartitionFieldName.empty() == false) {
                        text << " for '" << m_PartitionFieldName << "' '"
                             << m_CategoryIdMapper->categorizerKey() << '\'';
                    }
                    model::CAnnotation annotation{
                        m_LastMessageTime.has_value() ? *m_LastMessageTime
                                                      : core::CTimeUtils::now(),
                        model::CAnnotation::E_CategorizationStatusChange,
                        text.str(),
                        model::CAnnotation::DETECTOR_INDEX_NOT_APPLICABLE,
                        m_PartitionFieldName,
                        m_CategoryIdMapper->categorizerKey(),
                        EMPTY_STRING,
                        EMPTY_STRING,
                        EMPTY_STRING,
                        EMPTY_STRING};
                    annotationJsonWriter.writeResult(jsonOutputWriter.jobId(), annotation);
                }
            })) {
        LOG_TRACE(<< "Wrote categorizer stats for categorizer "
                  << m_CategoryIdMapper->categorizerKey());
    }
}

void CSingleFieldDataCategorizer::forceResourceRefresh(model::CResourceMonitor& resourceMonitor) {
    resourceMonitor.forceRefresh(*m_DataCategorizer);
}

void CSingleFieldDataCategorizer::acceptPersistInserter(
    const model::CDataCategorizer::TPersistFunc& dataCategorizerPersistFunc,
    const model::CCategoryExamplesCollector& examplesCollector,
    const CCategoryIdMapper& categoryIdMapper,
    core::CStatePersistInserter& inserter) {
    inserter.insertLevel(CATEGORIZER_TAG, dataCategorizerPersistFunc);
    inserter.insertLevel(EXAMPLES_COLLECTOR_TAG,
                         [&examplesCollector](core::CStatePersistInserter& inserter_) {
                             examplesCollector.acceptPersistInserter(inserter_);
                         });
    inserter.insertLevel(CATEGORY_ID_MAPPER_TAG,
                         [&categoryIdMapper](core::CStatePersistInserter& inserter_) {
                             categoryIdMapper.acceptPersistInserter(inserter_);
                         });
}
}
}
