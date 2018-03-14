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
#ifndef INCLUDED_ml_api_CMockDataProcessor_h
#define INCLUDED_ml_api_CMockDataProcessor_h

#include <core/CoreTypes.h>

#include <api/CDataProcessor.h>

#include <string>

#include <stdint.h>

namespace ml {
namespace api {
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
class CMockDataProcessor : public ml::api::CDataProcessor {
public:
    CMockDataProcessor(ml::api::COutputHandler& outputHandler);

    //! We're going to be writing to a new output stream
    virtual void newOutputStream(void);

    virtual bool handleRecord(const TStrStrUMap& dataRowFields);

    virtual void finalise(void);

    //! Restore previously saved state
    virtual bool restoreState(ml::core::CDataSearcher& restoreSearcher,
                              ml::core_t::TTime& completeToTime);

    //! Persist current state
    virtual bool persistState(ml::core::CDataAdder& persister);

    //! How many records did we handle?
    virtual uint64_t numRecordsHandled(void) const;

    //! Access the output handler
    virtual ml::api::COutputHandler& outputHandler(void);

private:
    ml::api::COutputHandler& m_OutputHandler;

    //! Empty field overrides
    TStrStrUMap m_FieldOverrides;

    uint64_t m_NumRecordsHandled;

    bool m_WriteFieldNames;
};

#endif // INCLUDED_ml_api_CMockDataProcessor_h
