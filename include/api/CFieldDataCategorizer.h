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
#ifndef INCLUDED_ml_api_CFieldDataCategorizer_h
#define INCLUDED_ml_api_CFieldDataCategorizer_h

#include <core/CRegexFilter.h>
#include <core/CWordDictionary.h>
#include <core/CoreTypes.h>

#include <model/CDataCategorizer.h>
#include <model/CTokenListDataCategorizer.h>

#include <api/CAnnotationJsonWriter.h>
#include <api/CAnomalyJobConfig.h>
#include <api/CCategoryIdMapper.h>
#include <api/CDataProcessor.h>
#include <api/CGlobalCategoryId.h>
#include <api/CJsonOutputWriter.h>
#include <api/CSingleFieldDataCategorizer.h>
#include <api/ImportExport.h>

#include <boost/unordered_set.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ml {
namespace core {
class CDataAdder;
class CDataSearcher;
class CJsonOutputStreamWrapper;
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace model {
class CLimits;
}
namespace api {
class CPersistenceManager;

//! \brief
//! Categorize input records and add categorization fields
//! before passing down the chain.
//!
//! DESCRIPTION:\n
//! Uses the lower level categorizer in the model library to
//! categorize input records and writes any new or changed
//! categories to the process output.
//!
//! Also adds a new field called mlcategory and assigns to it
//! an integer that corresponds to the chosen global category
//! ID.
//!
//! IMPLEMENTATION DECISIONS:\n
//! When per-partition categorization is used, each lower
//! level model library categorizer will produce category IDs
//! starting from 1, known as local category IDs.  This class
//! maps these to category IDs thatare globally unique within
//! the job: global category IDs.
//!
//! When per-partition categorization is not used, local and
//! global category IDs are identical, as there is only one
//! lower level model library categorizer.  This is keyed on
//! the empty string in the map of lower level categorizers.
//!
class API_EXPORT CFieldDataCategorizer : public CDataProcessor {
public:
    //! The name of the field where the category is going to be written
    static const std::string MLCATEGORY_NAME;

    static const double SIMILARITY_THRESHOLD;

    //! Discriminant for Elasticsearch IDs
    static const std::string STATE_TYPE;

    //! The current state version
    static const std::string STATE_VERSION;

public:
    // A type of token list data categorizer that DOESN'T exclude fields from
    // its analysis
    using TTokenListDataCategorizerKeepsFields =
        model::CTokenListDataCategorizer<true,  // Warping
                                         true,  // Underscores
                                         true,  // Dots
                                         true,  // Dashes
                                         true,  // Forward slashes
                                         true,  // Ignore leading digit
                                         true,  // Ignore hex
                                         true,  // Ignore date words
                                         false, // Ignore field names
                                         2,     // Min dictionary word length
                                         true,  // Truncate at first newline
                                         core::CWordDictionary::TWeightVerbs5Other2AdjacentBoost6>;

public:
    //! Construct without persistence capability
    CFieldDataCategorizer(std::string jobId,
                          const CAnomalyJobConfig::CAnalysisConfig& analysisConfig,
                          model::CLimits& limits,
                          const std::string& timeFieldName,
                          const std::string& timeFieldFormat,
                          CDataProcessor* chainedProcessor,
                          core::CJsonOutputStreamWrapper& outputStream,
                          CPersistenceManager* persistenceManager,
                          bool stopCategorizationOnWarnStatus);

    ~CFieldDataCategorizer() override;

    CGlobalCategoryId computeAndUpdateCategory(const TStrStrUMap& dataRowFields,
                                               const TOptionalTime& time);

    //! If any chained processor requires the computed "mlcategory" field be
    //! added to each record then this method must be called before
    //! handleRecord() to register "mlcategory" as a mutable field.  If there
    //! are no chained processors, or they are not interested in "mlcategory"
    //! then calling this method is not optional.  It is not harmful to register
    //! mutable fields other than "mlcategory" although this class will not do
    //! anything differently as a result.
    void registerMutableField(const std::string& fieldName, std::string& fieldValue) override;

    //! Receive a single record to be categorized, and output that record
    //! with its ML category field added
    bool handleRecord(const TStrStrUMap& dataRowFields, TOptionalTime time) override;

    //! Perform any final processing once all input data has been seen.
    void finalise() override;

    //! Restore previously saved state
    bool restoreState(core::CDataSearcher& restoreSearcher,
                      core_t::TTime& completeToTime) override;

    //! Is persistence needed?
    bool isPersistenceNeeded(const std::string& description) const override;

    //! Persist current state
    bool persistStateInForeground(core::CDataAdder& persister,
                                  const std::string& descriptionPrefix) override;

    //! Persist current state due to the periodic persistence being triggered.
    bool periodicPersistStateInBackground() override;
    bool periodicPersistStateInForeground() override;

    //! How many records did we handle?
    std::uint64_t numRecordsHandled() const override;

private:
    using TPersistFuncVec = std::vector<CSingleFieldDataCategorizer::TPersistFunc>;

    using TStrUSet = boost::unordered_set<std::string>;

    using TSingleFieldDataCategorizerUPtr = std::unique_ptr<CSingleFieldDataCategorizer>;
    using TStrSingleFieldDataCategorizerUPtrMap =
        std::map<std::string, TSingleFieldDataCategorizerUPtr>;

private:
    //! Get the appropriate categorizer key from the given input record
    const std::string& categorizerKeyForRecord(const TStrStrUMap& dataRowFields);

    //! Get a pointer to the categorizer to operate on the given partition
    //! field value.  If the categorizer does not already exist then this method
    //! will create it if the memory status is not hard_limit, and will return
    //! nullptr if hard_limit has been hit.
    CSingleFieldDataCategorizer* categorizerPtrForKey(const std::string& partitionFieldValue);

    //! Get (creating if necessary) the categorizer to operate on the given
    //! partition field value
    CSingleFieldDataCategorizer& categorizerForKey(const std::string& partitionFieldValue);

    bool doPersistState(const TStrVec& partitionFieldValues,
                        const TPersistFuncVec& dataCategorizerPersistFuncs,
                        std::size_t categorizerAllocationFailures,
                        core::CDataAdder& persister);
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Respond to an attempt to restore corrupt categorizer state by
    //! resetting the categorizer and re-categorizing from scratch.
    void resetAfterCorruptRestore();

    //! Handle a control message.  The first character of the control
    //! message indicates its type.  Currently defined types are:
    //! ' ' => Dummy message to force all previously uploaded data through
    //!        buffers
    //! 'f' => Echo a flush ID so that the attached process knows that data
    //!        sent previously has all been processed
    bool handleControlMessage(const std::string& controlMessage, bool lastHandler);

    //! Acknowledge a flush request
    void acknowledgeFlush(const std::string& flushId, bool lastHandler);

    //! Parse a stop-on-warn control message
    void parseStopOnWarnControlMessage(const std::string& enabledStr);

    //! Writes out to the JSON output writer any category definitions and stats
    //! that have changed since they were last written.
    void writeChanges();

    //! Get the next global ID to use.  This method only returns a useful
    //! result when per-partition categorization is being used.  When
    //! per-partition categorization is not used, there is no need for this
    //! value to be tracked.
    int nextGlobalId();

private:
    //! The job ID
    const std::string m_JobId;

    //! Configurable limits
    model::CLimits& m_Limits;

    //! Another processor to which the input is passed on.
    //! May be NULL if there is no further processing required.
    CDataProcessor* m_ChainedProcessor = nullptr;

    //! Stream used by the output writer
    core::CJsonOutputStreamWrapper& m_OutputStream;

    //! Should we stop categorizing when a model library categorizer's
    //! categorization status is "warn"?
    bool m_StopCategorizationOnWarnStatus = false;

    //! Highest previously used global ID.
    int m_HighestGlobalId = 0;

    //! Keep count of how many records we've handled
    std::uint64_t m_NumRecordsHandled = 0;

    //! Pointer to the mutable entry in the data row fields map that
    //! needs to be updated with the computed global category ID before
    //! chaining to the next processor.
    std::string* m_OutputFieldCategory = nullptr;

    //! Map of categorizer by partition field value.  If per-partition
    //! categorization is disabled this map will have one entry, keyed on
    //! the empty string.
    TStrSingleFieldDataCategorizerUPtrMap m_DataCategorizers;

    //! Object to which the overall process output is passed
    CJsonOutputWriter m_JsonOutputWriter;

    //! Writer for annotations
    CAnnotationJsonWriter m_AnnotationJsonWriter;

    //! Which field name are we partitioning on?  If empty, this means
    //! per-partition categorization is disabled and categories are
    //! determined across the entire data set.
    std::string m_PartitionFieldName;

    //! Which field name are we categorizing?
    std::string m_CategorizationFieldName;

    //! The categorization filter
    core::CRegexFilter m_CategorizationFilter;

    //! Pointer to the persistence manager. May be nullptr if state persistence
    //! is not required, for example in unit tests.
    CPersistenceManager* m_PersistenceManager;

    //! Number of times we have failed to allocate a lower level categorizer
    //! due to lack of memory.
    std::size_t m_CategorizerAllocationFailures = 0;

    //! Partition field values for which categorization is completely impossible
    //! due to lack of memory.  This is used to avoid excessive logging of
    //! warnings.  The set is not persisted, so warnings will be logged again
    //! on each invocation of the program if the same partition values are seen
    //! again.
    TStrUSet m_CategorizerAllocationFailedPartitions;
};
}
}

#endif // INCLUDED_ml_api_CFieldDataCategorizer_h
