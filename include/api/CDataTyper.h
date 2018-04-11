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
#ifndef INCLUDED_ml_api_CDataTyper_h
#define INCLUDED_ml_api_CDataTyper_h

#include <core/CoreTypes.h>

#include <api/ImportExport.h>

#include <boost/shared_ptr.hpp>
#include <boost/unordered_map.hpp>

#include <functional>
#include <string>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace api {

//! \brief
//! Interface for classes that convert a raw event string to a type.
//!
//! DESCRIPTION:\n
//! Abstract interface for classes that convert a raw event string
//! to a type.
//!
//! IMPLEMENTATION DECISIONS:\n
//! At the time of writing, only the token list data typer implements
//! this interface.  However, it is not hard to imagine a time when
//! there are specialist data typers for XML, JSON or delimited files,
//! so it is good to have an abstract interface that they can all use.
//!
class API_EXPORT CDataTyper {
public:
    //! Used for storing distinct token IDs
    using TStrStrUMap = boost::unordered_map<std::string, std::string>;
    using TStrStrUMapCItr = TStrStrUMap::const_iterator;

    //! Shared pointer to an instance of this class
    using TDataTyperP = boost::shared_ptr<CDataTyper>;

    //! Shared pointer to an instance of this class
    using TPersistFunc = std::function<void(core::CStatePersistInserter&)>;

public:
    CDataTyper(const std::string& fieldName);

    //! Virtual destructor for an abstract base class
    virtual ~CDataTyper();

    //! Dump stats
    virtual void dumpStats() const = 0;

    //! Compute a type from a string.  The raw string length may be longer
    //! than the length of the passed string, because the passed string may
    //! have the date stripped out of it.
    int computeType(bool isDryRun, const std::string& str, size_t rawStringLen);

    //! As above, but also take into account field names/values.
    virtual int computeType(bool isDryRun,
                            const TStrStrUMap& fields,
                            const std::string& str,
                            size_t rawStringLen) = 0;

    //! Create reverse search commands that will (more or less) just
    //! select the records that are classified as the given type when
    //! combined with the original search.  Note that the reverse search is
    //! only approximate - it may select more records than have actually
    //! been classified as the returned type.
    virtual bool createReverseSearch(int type,
                                     std::string& part1,
                                     std::string& part2,
                                     size_t& maxMatchingLength,
                                     bool& wasCached) = 0;

    //! Has the data typer's state changed?
    virtual bool hasChanged() const = 0;

    //! Populate the object from part of a state document
    virtual bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) = 0;

    //! Persist state by passing information to the supplied inserter
    virtual void acceptPersistInserter(core::CStatePersistInserter& inserter) const = 0;

    //! Make a function that can be called later to persist state
    virtual TPersistFunc makePersistFunc() const = 0;

    //! Access to the field name
    const std::string& fieldName() const;

    //! Access to last persistence time
    core_t::TTime lastPersistTime() const;

    //! Set last persistence time
    void lastPersistTime(core_t::TTime lastPersistTime);

protected:
    //! Used if no fields are supplied to the computeType() method.
    static const TStrStrUMap EMPTY_FIELDS;

private:
    //! Which field name are we working on?
    std::string m_FieldName;

    //! When was data last persisted for this typer?  (0 means never.)
    core_t::TTime m_LastPersistTime;
};
}
}

#endif // INCLUDED_ml_api_CDataTyper_h
