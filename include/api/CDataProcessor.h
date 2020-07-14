/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CDataProcessor_h
#define INCLUDED_ml_api_CDataProcessor_h

#include <core/CoreTypes.h>

#include <api/ImportExport.h>

#include <boost/optional.hpp>
#include <boost/unordered_map.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace ml {
namespace core {
class CDataAdder;
class CDataSearcher;
}
namespace api {
class CPersistenceManager;
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
class API_EXPORT CDataProcessor {
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

    using TOptionalTime = boost::optional<core_t::TTime>;

public:
    CDataProcessor() = default;
    CDataProcessor(const std::string& timeFieldName, const std::string& timeFieldFormat);
    virtual ~CDataProcessor() = default;

    //! No copying
    CDataProcessor(const CDataProcessor&) = delete;
    CDataProcessor& operator=(const CDataProcessor&) = delete;

    //! Receive a single record to be processed, and produce output
    //! with any required modifications
    virtual bool handleRecord(const TStrStrUMap& dataRowFields, TOptionalTime time) = 0;

    //! Perform any final processing once all input data has been seen.
    virtual void finalise() = 0;

    //! Restore previously saved state
    virtual bool restoreState(core::CDataSearcher& restoreSearcher,
                              core_t::TTime& completeToTime) = 0;

    //! Persist current state
    virtual bool persistStateInForeground(core::CDataAdder& persister,
                                          const std::string& descriptionPrefix) = 0;

    //! Persist current state in the background due to the periodic persistence being triggered.
    virtual bool periodicPersistStateInBackground();

    //! Persist current state in the foreground due to the periodic persistence being triggered.
    virtual bool periodicPersistStateInForeground();

    //! How many records did we handle?
    virtual std::uint64_t numRecordsHandled() const = 0;

    //! Is persistence needed?
    virtual bool isPersistenceNeeded(const std::string& description) const = 0;

    //! Create debug for a record.  This is expensive so should NOT be
    //! called for every record as a matter of course.
    static std::string debugPrintRecord(const TStrStrUMap& dataRowFields);

    //! Parse the time from an input record.
    //! \return An empty optional on failure.
    TOptionalTime parseTime(const TStrStrUMap& dataRowFields) const;

private:
    //! Name of field holding the time.  An empty string, indicates the input
    //! contains no timestamp.  This may not be valid for some data processors,
    //! in which case the derived class must validate this field.
    std::string m_TimeFieldName;

    //! Time field format.  Blank means seconds since the epoch, i.e. the
    //! time field can be converted to a time_t by simply converting the
    //! string to a number.
    std::string m_TimeFieldFormat;
};
}
}

#endif // INCLUDED_ml_api_CDataProcessor_h
