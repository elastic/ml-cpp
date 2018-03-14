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
        typedef std::vector<std::string>                       TStrVec;
        typedef TStrVec::iterator TStrVecItr;
        typedef TStrVec::const_iterator TStrVecCItr;

        typedef boost::unordered_map<std::string, std::string> TStrStrUMap;
        typedef TStrStrUMap::iterator TStrStrUMapItr;
        typedef TStrStrUMap::const_iterator TStrStrUMapCItr;

    public:
        CDataProcessor(void);
        virtual ~CDataProcessor(void);

        //! We're going to be writing to a new output stream
        virtual void newOutputStream(void) = 0;

        //! Receive a single record to be processed, and produce output
        //! with any required modifications
        virtual bool handleRecord(const TStrStrUMap &dataRowFields) = 0;

        //! Perform any final processing once all input data has been seen.
        virtual void finalise(void) = 0;

        //! Restore previously saved state
        virtual bool restoreState(core::CDataSearcher &restoreSearcher,
                                  core_t::TTime &completeToTime) = 0;

        //! Persist current state
        virtual bool persistState(core::CDataAdder &persister) = 0;

        //! Persist current state due to the periodic persistence being triggered.
        virtual bool periodicPersistState(CBackgroundPersister &persister);

        //! How many records did we handle?
        virtual uint64_t numRecordsHandled(void) const = 0;

        //! Access the output handler
        virtual COutputHandler &outputHandler(void) = 0;

        //! Create debug for a record.  This is expensive so should NOT be
        //! called for every record as a matter of course.
        static std::string debugPrintRecord(const TStrStrUMap &dataRowFields);
};


}
}

#endif // INCLUDED_ml_api_CDataProcessor_h

