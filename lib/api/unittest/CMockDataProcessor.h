/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CMockDataProcessor_h
#define INCLUDED_ml_api_CMockDataProcessor_h

#include <core/CoreTypes.h>

#include <api/CDataProcessor.h>

#include <string>

#include <stdint.h>


namespace ml
{
namespace api
{
class COutputHandler;
}
}

//! \brief
//! Mock object for unit tests
//!
//! DESCRIPTION:\n
//! Mock object for testing the OutputChainer class.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Only the minimal set of required functions are implemented.
//!
class CMockDataProcessor : public ml::api::CDataProcessor
{
    public:
        CMockDataProcessor(ml::api::COutputHandler &outputHandler);

        //! We're going to be writing to a new output stream
        virtual void newOutputStream();

        virtual bool handleRecord(const TStrStrUMap &dataRowFields);

        virtual void finalise();

        //! Restore previously saved state
        virtual bool restoreState(ml::core::CDataSearcher &restoreSearcher,
                                  ml::core_t::TTime &completeToTime);

        //! Persist current state
        virtual bool persistState(ml::core::CDataAdder &persister);

        //! How many records did we handle?
        virtual uint64_t numRecordsHandled() const;

        //! Access the output handler
        virtual ml::api::COutputHandler &outputHandler();

    private:
        ml::api::COutputHandler &m_OutputHandler;

        //! Empty field overrides
        TStrStrUMap                  m_FieldOverrides;

        uint64_t                     m_NumRecordsHandled;

        bool                         m_WriteFieldNames;
};


#endif // INCLUDED_ml_api_CMockDataProcessor_h

