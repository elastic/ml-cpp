/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CSimpleOutputWriter_h
#define INCLUDED_ml_api_CSimpleOutputWriter_h

#include <core/CoreTypes.h>

#include <api/ImportExport.h>

#include <boost/optional.hpp>
#include <boost/unordered_map.hpp>

#include <string>
#include <vector>

namespace ml {
namespace api {
class CPersistenceManager;

//! \brief
//! Interface for simple output of homogenous key/value pairs
//!
//! DESCRIPTION:\n
//! Interface to the various ways of outputting key/value data.  Usually
//! the output will be to a C++ stream after formatting into NDJSON or CSV.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Since the field names should never change for a given job, it is
//! possible to massively improve efficiency by pre-computing the hashes
//! for the strings that hold the field names.  The nested CPreComputedHash
//! class and TPreComputedHashVec typedef can be used to implement this.
//!
class API_EXPORT CSimpleOutputWriter {
public:
    using TStrVec = std::vector<std::string>;
    using TStrStrUMap = boost::unordered_map<std::string, std::string>;

    using TOptionalTime = boost::optional<core_t::TTime>;

public:
    CSimpleOutputWriter() = default;

    //! Virtual destructor for abstract base class
    virtual ~CSimpleOutputWriter() = default;

    //! No copying
    CSimpleOutputWriter(const CSimpleOutputWriter&) = delete;
    CSimpleOutputWriter& operator=(const CSimpleOutputWriter&) = delete;

    //! Set field names - this must only be called once per output file
    bool fieldNames(const TStrVec& fieldNames);

    //! Set field names, adding extra field names if they're not already
    //! present - this is only allowed once
    virtual bool fieldNames(const TStrVec& fieldNames, const TStrVec& extraFieldNames) = 0;

    //! Write a row to the stream.  The supplied map must contain every field
    //! value.  The time will be passed as an empty optional, i.e. unknown.
    bool writeRow(const TStrStrUMap& dataRowFields);

    //! Write a row to the stream, optionally overriding some of the
    //! original field values.  Where the same field is present in both
    //! overrideDataRowFields and dataRowFields, the value in
    //! overrideDataRowFields will be written.  The time will be passed
    //! as an empty optional, i.e. unknown.
    bool writeRow(const TStrStrUMap& dataRowFields, const TStrStrUMap& overrideDataRowFields);

    //! As above, but with a pre-parsed time.
    virtual bool writeRow(const TStrStrUMap& dataRowFields,
                          const TStrStrUMap& overrideDataRowFields,
                          TOptionalTime time) = 0;

protected:
    //! Class to cache a hash value so that it doesn't have to be repeatedly
    //! recomputed
    class API_EXPORT CPreComputedHash {
    public:
        //! Store the given hash
        CPreComputedHash(std::size_t hash) : m_Hash{hash} {}

        //! Return the hash regardless of what string is passed.  Use
        //! with care!
        std::size_t operator()(const std::string&) const { return m_Hash; }

    private:
        std::size_t m_Hash;
    };

    using TPreComputedHashVec = std::vector<CPreComputedHash>;

protected:
    //! Used when there are no extra fields
    static const TStrVec EMPTY_FIELD_NAMES;

    //! Used when there are no field overrides
    static const TStrStrUMap EMPTY_FIELD_OVERRIDES;
};
}
}

#endif // INCLUDED_ml_api_CSimpleOutputWriter_h
