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
#ifndef INCLUDED_ml_api_COutputHandler_h
#define INCLUDED_ml_api_COutputHandler_h

#include <core/CNonCopyable.h>
#include <core/CoreTypes.h>
#include <core/CDataAdder.h>
#include <core/CDataSearcher.h>

#include <api/ImportExport.h>

#include <boost/unordered_map.hpp>

#include <functional>
#include <string>
#include <vector>


namespace ml {
namespace core {
class CDataAdder;
class CDataSearcher;
}
namespace api {
class CBackgroundPersister;

//! \brief
//! Interface for CDataProcessor output
//!
//! DESCRIPTION:\n
//! Interface to the various ways of outputting data from a CDataProcessor
//! object.  Usually the output will be to a C++ stream (which in turn will be
//! connected to either a named pipe or STDOUT).  However, there is also the
//! option for output to be sent to the input of another CDataProcessor
//! object, which allows for chaining multiple processing steps within a single
//! program.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Since the field names should never change for a given job,
//! it is possible to massively improve efficiency by pre-computing the hashes
//! for the strings that hold the field names.  The nested CPreComputedHash
//! class and TPreComputedHashVec typedef can be used to implement this.
//!
class API_EXPORT COutputHandler : private core::CNonCopyable {
    public:
        typedef std::vector<std::string>                       TStrVec;
        typedef TStrVec::iterator TStrVecItr;
        typedef TStrVec::const_iterator TStrVecCItr;

        typedef boost::unordered_map<std::string, std::string> TStrStrUMap;
        typedef TStrStrUMap::iterator TStrStrUMapItr;
        typedef TStrStrUMap::const_iterator TStrStrUMapCItr;

    public:
        COutputHandler(void);

        //! Virtual destructor for abstract base class
        virtual ~COutputHandler(void);

        //! We're going to be writing to a new output stream
        virtual void newOutputStream(void);

        //! Set field names - this must only be called once per output file
        bool fieldNames(const TStrVec &fieldNames);

        //! Set field names, adding extra field names if they're not already
        //! present - this is only allowed once
        virtual bool fieldNames(const TStrVec &fieldNames,
                                const TStrVec &extraFieldNames) = 0;

        //! Get field names
        virtual const TStrVec &fieldNames(void) const = 0;

        //! Write a row to the stream.  The supplied map must contain every
        //! field value.
        bool writeRow(const TStrStrUMap &dataRowFields);

        //! Write a row to the stream, optionally overriding some of the
        //! original field values.  Where the same field is present in both
        //! overrideDataRowFields and dataRowFields, the value in
        //! overrideDataRowFields will be written.
        virtual bool writeRow(const TStrStrUMap &dataRowFields,
                              const TStrStrUMap &overrideDataRowFields) = 0;

        //! Perform any final processing once all input data has been seen.
        virtual void finalise(void);

        //! Restore previously saved state
        virtual bool restoreState(core::CDataSearcher &restoreSearcher,
                                  core_t::TTime &completeToTime);

        //! Persist current state
        virtual bool persistState(core::CDataAdder &persister);

        //! Persist current state due to the periodic persistence being triggered.
        virtual bool periodicPersistState(CBackgroundPersister &persister);

        //! Does this handler deal with control messages?
        virtual bool consumesControlMessages();

    protected:
        //! Class to cache a hash value so that it doesn't have to be repeatedly
        //! recomputed
        class API_EXPORT CPreComputedHash : public std::unary_function<std::string, size_t> {
            public:
                //! Store the given hash
                CPreComputedHash(size_t hash);

                //! Return the hash regardless of what string is passed.  Use
                //! with care!
                size_t operator()(const std::string &) const;

            private:
                size_t m_Hash;
        };

    protected:
        //! Used when there are no extra fields
        static const TStrVec     EMPTY_FIELD_NAMES;

        //! Used when there are no field overrides
        static const TStrStrUMap EMPTY_FIELD_OVERRIDES;

        typedef std::vector<CPreComputedHash>                  TPreComputedHashVec;
        typedef TPreComputedHashVec::iterator TPreComputedHashVecItr;
        typedef TPreComputedHashVec::const_iterator TPreComputedHashVecCItr;
};


}
}

#endif // INCLUDED_ml_api_COutputHandler_h

