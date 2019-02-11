/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CInputParser_h
#define INCLUDED_ml_api_CInputParser_h

#include <core/CNonCopyable.h>

#include <api/ImportExport.h>

#include <boost/ref.hpp>
#include <boost/unordered_map.hpp>

#include <functional>
#include <list>
#include <string>
#include <vector>

namespace ml {
namespace api {

//! \brief
//! Input parser interface
//!
//! DESCRIPTION:\n
//! Abstract base class for input parser classes.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Abstract interface declares the readStreamIntoMaps and readStreamIntoVecs
//! methods that must be implemented in sub-classes.
//!
class API_EXPORT CInputParser : private core::CNonCopyable {
public:
    using TStrVec = std::vector<std::string>;
    using TStrVecItr = TStrVec::iterator;
    using TStrVecCItr = TStrVec::const_iterator;

    using TStrStrUMap = boost::unordered_map<std::string, std::string>;
    using TStrStrUMapItr = TStrStrUMap::iterator;
    using TStrStrUMapCItr = TStrStrUMap::const_iterator;

    //! For fast access to the field values without repeatedly computing the
    //! hash, we maintain references to the values in the hash map
    using TStrRef = boost::reference_wrapper<std::string>;
    using TStrRefVec = std::vector<TStrRef>;
    using TStrRefVecItr = TStrRefVec::iterator;
    using TStrRefVecCItr = TStrRefVec::const_iterator;

    //! Callback function prototype that gets called for each record read
    //! from the input stream when reading into a map.  Return false to exit
    //! reader loop.  The argument is a map of field name to field value.
    using TMapReaderFunc = std::function<bool(const TStrStrUMap&)>;

    //! Callback function prototype that gets called for each record read
    //! from the input stream when reading into vectors.  Return false to exit
    //! reader loop.  The arguments are vectors of field names and field values.
    using TVecReaderFunc = std::function<bool(const TStrVec&, const TStrVec&)>;

public:
    CInputParser();
    virtual ~CInputParser();

    //! Did we find the input field names?
    bool gotFieldNames() const;

    //! Did we find any data in the input?
    bool gotData() const;

    //! Get field names
    const TStrVec& fieldNames() const;

    //! Read records from the stream.  The supplied reader function is called
    //! once per record.  If the supplied reader function returns false, reading
    //! will stop.  This method keeps reading until it reaches the end of the
    //! stream or an error occurs.  If it successfully reaches the end of
    //! the stream it returns true, otherwise it returns false.
    virtual bool readStreamIntoMaps(const TMapReaderFunc& readerFunc) = 0;

    //! Read records from the stream.  The supplied reader function is called
    //! once per record.  If the supplied reader function returns false, reading
    //! will stop.  This method keeps reading until it reaches the end of the
    //! stream or an error occurs.  If it successfully reaches the end of
    //! the stream it returns true, otherwise it returns false.
    virtual bool readStreamIntoVecs(const TVecReaderFunc& readerFunc) = 0;

protected:
    //! Set the "got field names" flag
    void gotFieldNames(bool gotFieldNames);

    //! Set the "got data" flag
    void gotData(bool gotData);

    //! Writable access to the field names for derived classes only
    TStrVec& fieldNames();

private:
    //! Have we got the field names?
    bool m_GotFieldNames;

    //! Have we found any data?
    bool m_GotData;

    //! Field names parsed from the input
    TStrVec m_FieldNames;
};
}
}

#endif // INCLUDED_ml_api_CInputParser_h
