/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_COutputChainer_h
#define INCLUDED_ml_api_COutputChainer_h

#include <api/COutputHandler.h>
#include <api/ImportExport.h>

#include <functional>
#include <string>

namespace ml {
namespace core {
class CDataAdder;
class CDataSearcher;
}
namespace api {
class CBackgroundPersister;
class CDataProcessor;

//! \brief
//! Pass output data to another CDataProcessor object
//!
//! DESCRIPTION:\n
//! Pass the output of one CDataProcessor object to another one.
//! This enables multiple processing steps to be performed within a
//! single Ml process, avoiding the overhead of converting to CSV
//! and back where Ml custom search commands would be adjacent in
//! a search pipe.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The function to be called for each output record is encapsulated
//! in a std::function to reduce coupling.
//!
class API_EXPORT COutputChainer : public COutputHandler {
public:
    //! Construct with a reference to the next data processor in the chain
    COutputChainer(CDataProcessor& dataProcessor);

    //! We're going to be writing to a new output stream
    virtual void newOutputStream();

    // Bring the other overload of fieldNames() into scope
    using COutputHandler::fieldNames;

    //! Set field names, adding extra field names if they're not already
    //! present - this is only allowed once
    virtual bool fieldNames(const TStrVec& fieldNames, const TStrVec& extraFieldNames);

    // Bring the other overload of writeRow() into scope
    using COutputHandler::writeRow;

    //! Call the next data processor's input function with some output
    //! values, optionally overriding some of the original field values.
    //! Where the same field is present in both overrideDataRowFields and
    //! dataRowFields, the value in overrideDataRowFields will be written.
    virtual bool writeRow(const TStrStrUMap& dataRowFields,
                          const TStrStrUMap& overrideDataRowFields);

    //! Perform any final processing once all data for the current search
    //! has been seen.  Chained classes should NOT rely on this method being
    //! called - they should do the best they can on the assumption that
    //! this method will not be called, but may be able to improve their
    //! output if this method is called.
    virtual void finalise();

    //! Restore previously saved state
    virtual bool restoreState(core::CDataSearcher& restoreSearcher,
                              core_t::TTime& completeToTime);

    //! Persist current state
    virtual bool persistState(core::CDataAdder& persister);

    //! Persist current state due to the periodic persistence being triggered.
    virtual bool periodicPersistState(CBackgroundPersister& persister);

    //! Is persistence needed?
    virtual bool isPersistenceNeeded(const std::string& description) const;

    //! The chainer does consume control messages, because it passes them on
    //! to whatever processor it's chained to.
    virtual bool consumesControlMessages();

private:
    //! The function that will be called for every record output via this
    //! object
    CDataProcessor& m_DataProcessor;

    //! Field names in the order they are to be written to the output
    TStrVec m_FieldNames;

    //! Pre-computed hashes for each field name.  The pre-computed hashes
    //! are at the same index in this vector as the corresponding field name
    //! in the m_FieldNames vector.
    TPreComputedHashVec m_Hashes;

    //! Used to build up the full set of fields to pass on to the next data
    //! processor
    TStrStrUMap m_WorkRecordFields;

    using TStrRef = std::reference_wrapper<std::string>;
    using TStrRefVec = std::vector<TStrRef>;
    using TStrRefVecCItr = TStrRefVec::const_iterator;

    //! References to the strings within m_WorkRecordFields in the same
    //! order as the field names in m_FieldNames.  This avoids the need to
    //! do hash lookups when populating m_WorkRecordFields.
    TStrRefVec m_WorkRecordFieldRefs;
};
}
}

#endif // INCLUDED_ml_api_COutputChainer_h
