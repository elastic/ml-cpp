/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CGlobalIdDataCategorizer_h
#define INCLUDED_ml_api_CGlobalIdDataCategorizer_h

#include <model/CDataCategorizer.h>

#include <api/CCategoryIdMapper.h>
#include <api/CGlobalCategoryId.h>
#include <api/ImportExport.h>

#include <functional>
#include <string>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace model {
class CCategoryExamplesCollector;
}
namespace api {
class CJsonOutputWriter;

//! \brief
//! Wraps a categorizer, mapping local IDs to global IDs.
//!
//! DESCRIPTION:\n
//! Exposes the functionality of the model library data categorizer
//! classes, but with local category IDs mapped to the corresponding
//! global category IDs.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The persistence for this class is unusual in that it is not
//! persisted within a new sub-level in the state.  The reason for
//! this is that the objects it contains used to be persisted by
//! the CFieldDataCategorizer when there was only one low level
//! categorizer.  An extra level cannot be introduced now, as that
//! would break backwards compatibility of the state.
//!
class API_EXPORT CGlobalIdDataCategorizer {
public:
    //! Function used for persisting objects of this class
    using TPersistFunc = std::function<void(core::CStatePersistInserter&)>;

public:
    CGlobalIdDataCategorizer(std::string partitionFieldName,
                             model::CDataCategorizer::TDataCategorizerPtr dataCategorizer,
                             CCategoryIdMapper::TCategoryIdMapperPtr categoryIdMapper);

    //! Dump stats
    void dumpStats() const;

    //! Compute a category from a string.  The raw string length may be longer
    //! than the length of the passed string, because the passed string may
    //! have the date stripped out of it.  If the category changes as a result
    //! write it to the output.
    CGlobalCategoryId
    computeAndUpdateCategory(bool isDryRun,
                             const model::CDataCategorizer::TStrStrUMap& fields,
                             const std::string& messageToCategorize,
                             const std::string& rawMessage,
                             model::CResourceMonitor& resourceMonitor,
                             CJsonOutputWriter& jsonOutputWriter);

    //! Make a function that can be called later to persist state in the
    //! foreground, i.e. in the knowledge that no other thread will be
    //! accessing the data structures this method accesses.
    TPersistFunc makeForegroundPersistFunc() const;

    //! Make a function that can be called later to persist state in the
    //! background, i.e. copying any required data such that other threads
    //! may modify the original data structures while persistence is taking
    //! place.
    TPersistFunc makeBackgroundPersistFunc() const;

    //! Populate the object from part of a state document
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Access to the categorizer key.
    const std::string& categorizerKey() const {
        return m_CategoryIdMapper->categorizerKey();
    }

    //! Writes out to the JSON output writer any category that has changed
    //! since the last time this method was called.
    void writeOutChangedCategories(CJsonOutputWriter& jsonOutputWriter);

    //! Force an update of the resource monitor.
    void forceResourceRefresh(model::CResourceMonitor& resourceMonitor);

private:
    //! Used by deferred persistence functions
    static void acceptPersistInserter(const model::CDataCategorizer::TPersistFunc& dataCategorizerPersistFunc,
                                      const model::CCategoryExamplesCollector& examplesCollector,
                                      const CCategoryIdMapper& categoryIdMapper,
                                      core::CStatePersistInserter& inserter);

private:
    //! Which field name are we partitioning on?  If empty, this means
    //! per-partition categorization is disabled and categories are
    //! determined across the entire data set.
    std::string m_PartitionFieldName;

    //! Pointer to the wrapped data categorizer.
    model::CDataCategorizer::TDataCategorizerPtr m_DataCategorizer;

    //! Pointer to the category ID mapper.
    CCategoryIdMapper::TCategoryIdMapperPtr m_CategoryIdMapper;

    //! String to store search terms.  By keeping this as a member variable
    //! instead of repeatedly creating local strings the buffer can learn the
    //! appropriate size and won't need to be reallocated repeatedly, this
    //! saving memory allocations.
    std::string m_SearchTermsScratchSpace;

    //! Regex to match values of the current category.  As with
    //! m_SearchTermsScratchSpace, this is a member to avoid repeated memory
    //! allocations.
    std::string m_SearchTermsRegexScratchSpace;
};
}
}

#endif // INCLUDED_ml_api_CGlobalIdDataCategorizer_h
