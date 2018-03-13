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
#ifndef INCLUDED_ml_api_CFieldDataTyper_h
#define INCLUDED_ml_api_CFieldDataTyper_h

#include <core/CRegexFilter.h>
#include <core/CWordDictionary.h>
#include <core/CoreTypes.h>

#include <api/CCategoryExamplesCollector.h>
#include <api/CDataProcessor.h>
#include <api/CDataTyper.h>
#include <api/CTokenListDataTyper.h>
#include <api/ImportExport.h>

#include <string>

#include <stdint.h>

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
class CBackgroundPersister;
class CFieldConfig;
class CJsonOutputWriter;
class COutputHandler;

//! \brief
//! Assign categorisation fields to input records
//!
//! DESCRIPTION:\n
//! Adds a new field called mlcategory and assigns to it
//! integers that correspond to the various cateogories
//!
class API_EXPORT CFieldDataTyper : public CDataProcessor {
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
    // A type of token list data typer that DOESN'T exclude fields from its
    // analysis
    typedef CTokenListDataTyper<true, // Warping
                                true, // Underscores
                                true, // Dots
                                true, // Dashes
                                true, // Ignore leading digit
                                true, // Ignore hex
                                true, // Ignore date words
                                false,// Ignore field names
                                2,    // Min dictionary word length
                                core::CWordDictionary::TWeightVerbs5Other2>
        TTokenListDataTyperKeepsFields;

public:
    //! Construct without persistence capability
    CFieldDataTyper(const std::string &jobId,
                    const CFieldConfig &config,
                    const model::CLimits &limits,
                    COutputHandler &outputHandler,
                    CJsonOutputWriter &jsonOutputWriter,
                    CBackgroundPersister *periodicPersister = nullptr);

    virtual ~CFieldDataTyper(void);

    //! We're going to be writing to a new output stream
    virtual void newOutputStream(void);

    //! Receive a single record to be typed, and output that record to
    //! STDOUT with its type field added
    virtual bool handleRecord(const TStrStrUMap &dataRowFields);

    //! Perform any final processing once all input data has been seen.
    virtual void finalise(void);

    //! Restore previously saved state
    virtual bool restoreState(core::CDataSearcher &restoreSearcher, core_t::TTime &completeToTime);

    //! Persist current state
    virtual bool persistState(core::CDataAdder &persister);

    //! Persist current state due to the periodic persistence being triggered.
    virtual bool periodicPersistState(CBackgroundPersister &persister);

    //! How many records did we handle?
    virtual uint64_t numRecordsHandled(void) const;

    //! Access the output handler
    virtual COutputHandler &outputHandler(void);

private:
    //! Create the typer to operate on the categorization field
    void createTyper(const std::string &fieldName);

    //! Compute the type for a given record.
    int computeType(const TStrStrUMap &dataRowFields);

    //! Create the reverse search and return true if it has changed or false otherwise
    bool createReverseSearch(int type);

    bool doPersistState(const CDataTyper::TPersistFunc &dataTyperPersistFunc,
                        const CCategoryExamplesCollector &examplesCollector,
                        core::CDataAdder &persister);
    void acceptPersistInserter(const CDataTyper::TPersistFunc &dataTyperPersistFunc,
                               const CCategoryExamplesCollector &examplesCollector,
                               core::CStatePersistInserter &inserter) const;
    bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

    //! Respond to an attempt to restore corrupt categorizer state by
    //! resetting the categorizer and re-categorizing from scratch.
    void resetAfterCorruptRestore(void);

    //! Handle a control message.  The first character of the control
    //! message indicates its type.  Currently defined types are:
    //! ' ' => Dummy message to force all previously uploaded data through
    //!        buffers
    //! 'f' => Echo a flush ID so that the attached process knows that data
    //!        sent previously has all been processed
    bool handleControlMessage(const std::string &controlMessage);

    //! Acknowledge a flush request
    void acknowledgeFlush(const std::string &flushId);

private:
    typedef CCategoryExamplesCollector::TStrSet TStrSet;

private:
    //! The job ID
    std::string m_JobId;

    //! Object to which the output is passed
    COutputHandler &m_OutputHandler;

    //! Cache extra field names to be added
    TStrVec m_ExtraFieldNames;

    //! Should we write the field names before the next output?
    bool m_WriteFieldNames;

    //! Keep count of how many records we've handled
    uint64_t m_NumRecordsHandled;

    //! Map holding fields to add/change in the output compared to the input
    TStrStrUMap m_Overrides;

    //! References to specific entries in the overrides map to save
    //! repeatedly searching for them
    std::string &m_OutputFieldCategory;

    //! Space separated list of search terms for the current category
    std::string m_SearchTerms;

    //! Regex to match values of the current category
    std::string m_SearchTermsRegex;

    //! The max matching length of the current category
    std::size_t m_MaxMatchingLength;

    //! Pointer to the actual typer
    CDataTyper::TDataTyperP m_DataTyper;

    //! Reference to the json output writer so that examples can be written
    CJsonOutputWriter &m_JsonOutputWriter;

    //! Collects up to a configurable number of examples per category
    CCategoryExamplesCollector m_ExamplesCollector;

    //! Which field name are we categorizing?
    std::string m_CategorizationFieldName;

    //! The categorization filter
    core::CRegexFilter m_CategorizationFilter;

    //! Pointer to periodic persister that works in the background.  May be
    //! nullptr if this object is not responsible for starting periodic
    //! persistence.
    CBackgroundPersister *m_PeriodicPersister;
};
}
}

#endif// INCLUDED_ml_api_CFieldDataTyper_h
