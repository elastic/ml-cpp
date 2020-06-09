/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CGlobalIdDataCategorizer.h>

#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>

#include <model/CCategoryExamplesCollector.h>

#include <api/CJsonOutputWriter.h>

namespace {
// For historical reasons, these tags must not duplicate those used in
// CFieldDataCategorizer.cc
// a used in CFieldDataCategorizer.cc
const std::string CATEGORIZER_TAG{"b"};
const std::string EXAMPLES_COLLECTOR_TAG{"c"};
const std::string CATEGORY_ID_MAPPER_TAG{"d"};
// e, f, g used in CFieldDataCategorizer.cc
}

namespace ml {
namespace api {

CGlobalIdDataCategorizer::CGlobalIdDataCategorizer(std::string partitionFieldName,
                                                   model::CDataCategorizer::TDataCategorizerPtr dataCategorizer,
                                                   CCategoryIdMapper::TCategoryIdMapperPtr categoryIdMapper)
    : m_PartitionFieldName{std::move(partitionFieldName)},
      m_DataCategorizer{std::move(dataCategorizer)}, m_CategoryIdMapper{std::move(categoryIdMapper)} {
}

void CGlobalIdDataCategorizer::dumpStats() const {
    m_DataCategorizer->dumpStats([this](model::CLocalCategoryId localCategoryId) {
        return m_CategoryIdMapper->map(localCategoryId).print();
    });
}

CGlobalCategoryId CGlobalIdDataCategorizer::computeAndUpdateCategory(
    bool isDryRun,
    const model::CDataCategorizer::TStrStrUMap& fields,
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

    bool exampleAdded{m_DataCategorizer->addExample(globalCategoryId.localId(), rawMessage)};
    std::size_t maxMatchingLength{0};
    bool searchTermsChanged{false};
    if (m_DataCategorizer->createReverseSearch(
            localCategoryId, m_SearchTermsScratchSpace, m_SearchTermsRegexScratchSpace,
            maxMatchingLength, searchTermsChanged) == false) {
        m_SearchTermsScratchSpace.clear();
        m_SearchTermsRegexScratchSpace.clear();
    }
    if (exampleAdded || searchTermsChanged) {
        //! signal that we noticed the change and are persisting here
        m_DataCategorizer->categoryChangedAndReset(localCategoryId);
        jsonOutputWriter.writeCategoryDefinition(
            m_PartitionFieldName, m_CategoryIdMapper->categorizerKey(), globalCategoryId,
            m_SearchTermsScratchSpace, m_SearchTermsRegexScratchSpace, maxMatchingLength,
            m_DataCategorizer->examplesCollector().examples(localCategoryId),
            m_DataCategorizer->numMatches(localCategoryId),
            m_CategoryIdMapper->mapVec(m_DataCategorizer->usurpedCategories(localCategoryId)));
        if (globalCategoryId.globalId() % 10 == 0) {
            // Even if memory limiting is disabled, force a refresh occasionally
            // so the user has some idea what's going on with memory.
            resourceMonitor.forceRefresh(*m_DataCategorizer);
        } else {
            resourceMonitor.refresh(*m_DataCategorizer);
        }
    }
    return globalCategoryId;
}

CGlobalIdDataCategorizer::TPersistFunc
CGlobalIdDataCategorizer::makeForegroundPersistFunc() const {
    model::CDataCategorizer::TPersistFunc categorizerPersistFunc{
        m_DataCategorizer->makeForegroundPersistFunc()};

    return [ categorizerPersistFunc = std::move(categorizerPersistFunc),
             this ](core::CStatePersistInserter & inserter) {
        CGlobalIdDataCategorizer::acceptPersistInserter(
            categorizerPersistFunc, m_DataCategorizer->examplesCollector(),
            *m_CategoryIdMapper, inserter);
    };
}

CGlobalIdDataCategorizer::TPersistFunc
CGlobalIdDataCategorizer::makeBackgroundPersistFunc() const {
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
        CGlobalIdDataCategorizer::acceptPersistInserter(
            categorizerPersistFunc, examplesCollector, *categoryIdMapperClone, inserter);
    };
}

bool CGlobalIdDataCategorizer::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
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
                }) == false ||
                traverser.haveBadState()) {
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

void CGlobalIdDataCategorizer::writeOutChangedCategories(CJsonOutputWriter& jsonOutputWriter) {
    std::size_t numCategories{m_DataCategorizer->numCategories()};
    if (numCategories == 0) {
        return;
    }

    std::size_t maxMatchingLength{0};
    bool wasCached{false};
    for (std::size_t index = 0; index < numCategories; ++index) {
        model::CLocalCategoryId localCategoryId{index};
        if (m_DataCategorizer->categoryChangedAndReset(localCategoryId)) {
            CGlobalCategoryId globalCategoryId{m_CategoryIdMapper->map(localCategoryId)};
            if (m_DataCategorizer->createReverseSearch(
                    localCategoryId, m_SearchTermsScratchSpace, m_SearchTermsRegexScratchSpace,
                    maxMatchingLength, wasCached) == false) {
                LOG_WARN(<< "Unable to create or retrieve reverse search to store for category: "
                         << globalCategoryId);
                continue;
            }
            LOG_TRACE(<< "Writing out changed category: " << globalCategoryId);
            jsonOutputWriter.writeCategoryDefinition(
                m_PartitionFieldName, m_CategoryIdMapper->categorizerKey(),
                globalCategoryId, m_SearchTermsScratchSpace,
                m_SearchTermsRegexScratchSpace, maxMatchingLength,
                m_DataCategorizer->examplesCollector().examples(localCategoryId),
                m_DataCategorizer->numMatches(localCategoryId),
                m_CategoryIdMapper->mapVec(m_DataCategorizer->usurpedCategories(localCategoryId)));
        }
    }
}

void CGlobalIdDataCategorizer::forceResourceRefresh(model::CResourceMonitor& resourceMonitor) {
    resourceMonitor.forceRefresh(*m_DataCategorizer);
}

void CGlobalIdDataCategorizer::acceptPersistInserter(
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
