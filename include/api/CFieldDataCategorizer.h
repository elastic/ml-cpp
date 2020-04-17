/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CFieldDataCategorizer_h
#define INCLUDED_ml_api_CFieldDataCategorizer_h

#include <core/CRegexFilter.h>
#include <core/CWordDictionary.h>
#include <core/CoreTypes.h>

#include <model/CDataCategorizer.h>
#include <model/CTokenListDataCategorizer.h>

#include <api/CDataProcessor.h>
#include <api/ImportExport.h>

#include <cstdint>
#include <string>

namespace ml {
namespace core {
class CDataAdder;
class CDataSearcher;
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace model {
class CLimits;
}
namespace api {
class CFieldConfig;
class CJsonOutputWriter;
class COutputHandler;
class CPersistenceManager;

//! \brief
//! Assign categorisation fields to input records
//!
//! DESCRIPTION:\n
//! Adds a new field called mlcategory and assigns to it
//! integers that correspond to the various cateogories
//!
class API_EXPORT CFieldDataCategorizer : public CDataProcessor {
public:
    //! The index where state is stored
    static const std::string ML_STATE_INDEX;
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
                                         true,  // Ignore leading digit
                                         true,  // Ignore hex
                                         true,  // Ignore date words
                                         false, // Ignore field names
                                         2,     // Min dictionary word length
                                         core::CWordDictionary::TWeightVerbs5Other2>;

public:
    //! Construct without persistence capability
    CFieldDataCategorizer(const std::string& jobId,
                          const CFieldConfig& config,
                          model::CLimits& limits,
                          COutputHandler& outputHandler,
                          CJsonOutputWriter& jsonOutputWriter,
                          CPersistenceManager* persistenceManager);

    ~CFieldDataCategorizer() override;

    //! We're going to be writing to a new output stream
    void newOutputStream() override;

    //! Receive a single record to be categorized, and output that record
    //! with its ML category field added
    bool handleRecord(const TStrStrUMap& dataRowFields) override;

    //! Perform any final processing once all input data has been seen.
    void finalise() override;

    //! Restore previously saved state
    bool restoreState(core::CDataSearcher& restoreSearcher,
                      core_t::TTime& completeToTime) override;

    //! Is persistence needed?
    bool isPersistenceNeeded(const std::string& description) const override;

    //! Persist current state
    bool persistState(core::CDataAdder& persister, const std::string& descriptionPrefix) override;

    //! Persist current state due to the periodic persistence being triggered.
    bool periodicPersistStateInBackground() override;
    bool periodicPersistStateInForeground() override;

    //! How many records did we handle?
    std::uint64_t numRecordsHandled() const override;

    //! Access the output handler
    COutputHandler& outputHandler() override;

private:
    //! Create the categorizer to operate on the categorization field
    void createCategorizer(const std::string& fieldName);

    //! Compute the category for a given record.
    int computeCategory(const TStrStrUMap& dataRowFields);

    //! Create the reverse search and return true if it has changed or false otherwise
    bool createReverseSearch(int categoryId);

    bool doPersistState(const model::CDataCategorizer::TPersistFunc& dataCategorizerPersistFunc,
                        const model::CCategoryExamplesCollector& examplesCollector,
                        core::CDataAdder& persister);
    void acceptPersistInserter(const model::CDataCategorizer::TPersistFunc& dataCategorizerPersistFunc,
                               const model::CCategoryExamplesCollector& examplesCollector,
                               core::CStatePersistInserter& inserter) const;
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

    //! Writes out to the JSON output writer any category that has changed
    //! since the last time this method was called.
    void writeOutChangedCategories();

private:
    //! The job ID
    std::string m_JobId;

    //! Configurable limits
    model::CLimits& m_Limits;

    //! Object to which the output is passed
    COutputHandler& m_OutputHandler;

    //! Cache extra field names to be added
    TStrVec m_ExtraFieldNames;

    //! Should we write the field names before the next output?
    bool m_WriteFieldNames;

    //! Keep count of how many records we've handled
    std::uint64_t m_NumRecordsHandled;

    //! Map holding fields to add/change in the output compared to the input
    TStrStrUMap m_Overrides;

    //! References to specific entries in the overrides map to save
    //! repeatedly searching for them
    std::string& m_OutputFieldCategory;

    //! Space separated list of search terms for the current category
    std::string m_SearchTerms;

    //! Regex to match values of the current category
    std::string m_SearchTermsRegex;

    //! The max matching length of the current category
    std::size_t m_MaxMatchingLength;

    //! Pointer to the actual categorizer
    model::CDataCategorizer::TDataCategorizerP m_DataCategorizer;

    //! Reference to the json output writer so that examples can be written
    CJsonOutputWriter& m_JsonOutputWriter;

    //! Which field name are we categorizing?
    std::string m_CategorizationFieldName;

    //! The categorization filter
    core::CRegexFilter m_CategorizationFilter;

    //! Pointer to the persistence manager. May be nullptr if state persistence
    //! is not required, for example in unit tests.
    CPersistenceManager* m_PersistenceManager;
};
}
}

#endif // INCLUDED_ml_api_CFieldDataCategorizer_h
