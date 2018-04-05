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
//! Abstract interface declares the readStream method that must be
//! implemented in sub-classes.
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

    //! Callback function prototype that gets called for each record
    //! read from the input stream.  Return false to exit reader loop.
    //! Arguments are:
    //! 1) Header row fields
    //! 2) Data row fields
    using TReaderFunc = std::function<bool(const TStrStrUMap&)>;

public:
    CInputParser();
    virtual ~CInputParser();

    //! Did we find the input field names?
    bool gotFieldNames() const;

    //! Did we find any data in the input?
    bool gotData() const;

    //! Get field names
    const TStrVec& fieldNames() const;

    //! Read records from the stream.  The supplied settings function is
    //! called only once.  The supplied reader function is called once per
    //! record.  If the supplied reader function returns false, reading will
    //! stop.  This method keeps reading until it reaches the end of the
    //! stream or an error occurs.  If it successfully reaches the end of
    //! the stream it returns true, otherwise it returns false.  If
    virtual bool readStream(const TReaderFunc& readerFunc) = 0;

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
