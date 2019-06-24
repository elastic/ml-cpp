/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_config_CDetectorRecord_h
#define INCLUDED_ml_config_CDetectorRecord_h

#include <core/CoreTypes.h>

#include <config/ConfigTypes.h>
#include <config/Constants.h>
#include <config/ImportExport.h>

#include <boost/unordered_map.hpp>

#include <array>
#include <string>
#include <utility>
#include <vector>

namespace ml {
namespace config {
class CDetectorSpecification;

//! \brief Holds the field values needed to update a detector.
//!
//! DESCRIPTION:\n
//! The state from a record need to update a detector's penalty functions.
//! This is its time and its field values corresponding to the argument
//! and partitioning field(s) used by the detector.
class CONFIG_EXPORT CDetectorRecord {
public:
    using TSizeAry = std::array<std::size_t, constants::NUMBER_FIELD_INDICES>;
    using TStrCPtrAry = std::array<const std::string*, constants::NUMBER_FIELD_INDICES>;
    using TStrStrUMap = boost::unordered_map<std::string, std::string>;

public:
    CDetectorRecord(core_t::TTime time,
                    config_t::EFunctionCategory function,
                    const TStrCPtrAry& fieldNames,
                    const TStrCPtrAry& fieldValues,
                    const TSizeAry& hashedFieldValues);

    //! Get the record time.
    core_t::TTime time() const;

    //! Get the function of the record detector.
    config_t::EFunctionCategory function() const;

    //! Get the name of the argument field.
    const std::string* argumentFieldName() const;

    //! Get the name of the by field.
    const std::string* byFieldName() const;

    //! Get the name of the over field.
    const std::string* overFieldName() const;

    //! Get the name of the partition field.
    const std::string* partitionFieldName() const;

    //! Get the argument field value if there is one or null.
    const std::string* argumentFieldValue() const;

    //! Get the by field value if there is one or null.
    const std::string* byFieldValue() const;

    //! Get the over field value if there is one or null.
    const std::string* overFieldValue() const;

    //! Get the partition field value if there is one or null.
    const std::string* partitionFieldValue() const;

    //! Get the argument field value hash.
    std::size_t argumentFieldValueHash() const;

    //! Get the by field value hash.
    std::size_t byFieldValueHash() const;

    //! Get the over field value hash.
    std::size_t overFieldValueHash() const;

    //! Get the partition field value hash.
    std::size_t partitionFieldValueHash() const;

    //! Print a description of this record for debug.
    std::string print() const;

private:
    //! The record time.
    core_t::TTime m_Time;

    //! The function of the record's detector.
    config_t::EFunctionCategory m_Function;

    //! The relevant field names.
    TStrCPtrAry m_FieldNames;

    //! The relevant field values.
    TStrCPtrAry m_FieldValues;

    //! Hashes of the field values.
    TSizeAry m_HashedFieldValues;
};

//! \brief Defines a fast scheme, which minimizes lookups in the field values
//! map, when building detector keys.
//!
//! DESCRIPTION:\n
//! There are many more distinct detectors than field names and we only want
//! to lookup each field value once per record. To do this we maintain a direct
//! address table from every detector, built once up front on the set of initial
//! candidate detectors, to a corresponding collection of entries in a field
//! value vector which we populate once per record.
class CONFIG_EXPORT CDetectorRecordDirectAddressTable {
public:
    using TStrStrUMap = boost::unordered_map<std::string, std::string>;
    using TDetectorSpecificationVec = std::vector<CDetectorSpecification>;
    using TDetectorRecordVec = std::vector<CDetectorRecord>;

public:
    //! Build the table from \p specs.
    void build(const TDetectorSpecificationVec& specs);

    //! Get the unique records from \p time and \p fieldValues for \p specs.
    void detectorRecords(core_t::TTime time,
                         const TStrStrUMap& fieldValues,
                         const TDetectorSpecificationVec& specs,
                         TDetectorRecordVec& result);

private:
    //! Clear the state (as a precursor to build).
    void clear();

private:
    using TSizeVec = std::vector<std::size_t>;
    using TStrSizePr = std::pair<std::string, std::size_t>;
    using TStrSizePrVec = std::vector<TStrSizePr>;
    using TSizeAry = std::array<std::size_t, constants::NUMBER_FIELD_INDICES>;
    using TSizeAryVec = std::vector<TSizeAry>;
    using TStrCPtrVec = std::vector<const std::string*>;

private:
    //! A map from field to its value entry in the field value table.
    TStrSizePrVec m_FieldSchema;

    //! A map from detectors to their field value entries in the field
    //! value table.
    TSizeAryVec m_DetectorFieldSchema;

    //! The table of field values populated once per record.
    TStrCPtrVec m_FieldValueTable;

    //! The table of field value hashes populated once per record.
    TSizeVec m_HashedFieldValueTable;
};
}
}

#endif // INCLUDED_ml_config_CDetectorRecord_h
