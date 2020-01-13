/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CDataSearcher_h
#define INCLUDED_ml_core_CDataSearcher_h

#include <core/CNonCopyable.h>
#include <core/ImportExport.h>

#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace ml {
namespace core {

//! \brief
//! Interface class for retrieving data by running a search.
//!
//! DESCRIPTION:\n
//! Interface for searching for data in some sort of data store.
//! Derived classes will supply the details of how to search a
//! specific type of data store.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Abstract class.
//!
//! The result of a successful search is a std::istream.
//!
class CORE_EXPORT CDataSearcher : private CNonCopyable {
public:
    using TStrVec = std::vector<std::string>;
    using TStrVecCItr = TStrVec::const_iterator;

    using TIStreamP = std::shared_ptr<std::istream>;

public:
    //! Empty string
    static const std::string EMPTY_STRING;

public:
    CDataSearcher();
    virtual ~CDataSearcher();

    //! Do a search that results in an input stream.
    //! A return value of NULL indicates a technical problem with the
    //! creation of the stream.  Other errors may be indicated by the
    //! returned stream going into the "bad" state.
    virtual TIStreamP search(size_t currentDocNum, size_t limit) = 0;

    //! Set the search for all documents in the index
    virtual void setStateRestoreSearch(const std::string& index);

    //! Set the search for all documents in the index with the ID
    virtual void setStateRestoreSearch(const std::string& index, const std::string& id);

protected:
    TStrVec m_SearchTerms;
};
}
}

#endif // INCLUDED_ml_core_CDataSearcher_h
