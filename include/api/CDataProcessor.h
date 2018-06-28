/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CDataProcessor_h
#define INCLUDED_ml_api_CDataProcessor_h

#include <core/CNonCopyable.h>
#include <core/CoreTypes.h>

#include <api/ImportExport.h>

#include <boost/unordered_map.hpp>

#include <string>
#include <vector>

#include <stdint.h>

namespace ml {
namespace core {
class CDataAdder;
class CDataSearcher;
}

namespace api {
class CBackgroundPersister;
class COutputHandler;

//! \brief
//! Abstract interface for classes that process data records
//!
//! DESCRIPTION:\n
//! Classes that process data records must implement this
//! interface so that they can fit into the CCmdSkeleton
//! framework.
//!
//! IMPLEMENTATION DECISIONS:\n
//!
class API_EXPORT CDataProcessor : private core::CNonCopyable {
public:
    static const char CONTROL_FIELD_NAME_CHAR = '.';
    static const std::string CONTROL_FIELD_NAME;

public:
    using TStrVec = std::vector<std::string>;
    using TStrVecItr = TStrVec::iterator;
    using TStrVecCItr = TStrVec::const_iterator;

    using TStrStrUMap = boost::unordered_map<std::string, std::string>;
    using TStrStrUMapItr = TStrStrUMap::iterator;
    using TStrStrUMapCItr = TStrStrUMap::const_iterator;

public:
    CDataProcessor();
    virtual ~CDataProcessor();

    //! We're going to be writing to a new output stream
    virtual void newOutputStream() = 0;

    //! Receive a single record to be processed, and produce output
    //! with any required modifications
    virtual bool handleRecord(const TStrStrUMap& dataRowFields) = 0;

    //! Perform any final processing once all input data has been seen.
    virtual void finalise() = 0;

    //! Restore previously saved state
    virtual bool restoreState(core::CDataSearcher& restoreSearcher,
                              core_t::TTime& completeToTime) = 0;

    //! Persist current state
    virtual bool persistState(core::CDataAdder& persister) = 0;

    //! Persist current state due to the periodic persistence being triggered.
    virtual bool periodicPersistState(CBackgroundPersister& persister);

    //! How many records did we handle?
    virtual uint64_t numRecordsHandled() const = 0;

    //! Access the output handler
    virtual COutputHandler& outputHandler() = 0;

    //! Create debug for a record.  This is expensive so should NOT be
    //! called for every record as a matter of course.
    static std::string debugPrintRecord(const TStrStrUMap& dataRowFields);
};
}
}

#endif // INCLUDED_ml_api_CDataProcessor_h
