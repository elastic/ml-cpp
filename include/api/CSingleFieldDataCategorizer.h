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
#ifndef INCLUDED_ml_api_CSingleFieldDataCategorizer_h
#define INCLUDED_ml_api_CSingleFieldDataCategorizer_h

#include <model/CDataCategorizer.h>

#include <api/CCategoryIdMapper.h>
#include <api/CGlobalCategoryId.h>
#include <api/ImportExport.h>

#include <functional>
#include <optional>
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
class CAnnotationJsonWriter;
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
class API_EXPORT CSingleFieldDataCategorizer {
public:
    //! Function used for persisting objects of this class
    using TPersistFunc = std::function<void(core::CStatePersistInserter&)>;

    using TOptionalTime = std::optional<core_t::TTime>;

public:
    CSingleFieldDataCategorizer(std::string partitionFieldName,
                                model::CDataCategorizer::TDataCategorizerUPtr dataCategorizer,
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
                             const TOptionalTime& messageTime,
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

    //! Get the most recent categorization status.
    model_t::ECategorizationStatus categorizationStatus() const {
        return m_DataCategorizer->categorizationStatus();
    }

    //! Writes out to the JSON output writer any category definitions and stats
    //! that have changed since they were last written.
    void writeChanges(CJsonOutputWriter& jsonOutputWriter,
                      CAnnotationJsonWriter& annotationJsonWriter);

    //! If the lower level categorizer thinks it urgent, write the latest
    //! categorizer stats, plus an annotation if the categorization status has
    //! changed.
    void writeStatsIfUrgent(CJsonOutputWriter& jsonOutputWriter,
                            CAnnotationJsonWriter& annotationJsonWriter);

    //! Force an update of the resource monitor.
    void forceResourceRefresh(model::CResourceMonitor& resourceMonitor);

private:
    //! Used by deferred persistence functions
    static void acceptPersistInserter(const model::CDataCategorizer::TPersistFunc& dataCategorizerPersistFunc,
                                      const model::CCategoryExamplesCollector& examplesCollector,
                                      const CCategoryIdMapper& categoryIdMapper,
                                      core::CStatePersistInserter& inserter);

    //! Write the latest categorizer stats, plus an annotation if the
    //! categorization status has changed.
    void writeStatsIfChanged(CJsonOutputWriter& jsonOutputWriter,
                             CAnnotationJsonWriter& annotationJsonWriter);

private:
    //! Which field name are we partitioning on?  If empty, this means
    //! per-partition categorization is disabled and categories are
    //! determined across the entire data set.
    std::string m_PartitionFieldName;

    //! Pointer to the wrapped data categorizer.
    model::CDataCategorizer::TDataCategorizerUPtr m_DataCategorizer;

    //! Pointer to the category ID mapper.
    CCategoryIdMapper::TCategoryIdMapperPtr m_CategoryIdMapper;

    //! Last timestamp observed in input.
    TOptionalTime m_LastMessageTime;
};
}
}

#endif // INCLUDED_ml_api_CSingleFieldDataCategorizer_h
