/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */
#ifndef INCLUDED_ml_core_CRapidXmlStatePersistInserter_h
#define INCLUDED_ml_core_CRapidXmlStatePersistInserter_h

#include <core/CStatePersistInserter.h>
#include <core/CStringCache.h>
#include <core/ImportExport.h>

#include <rapidxml/rapidxml.hpp>

#include <map>

namespace ml {
namespace core {

//! \brief
//! For persisting state in XML format.
//!
//! DESCRIPTION:\n
//! Concrete implementation of the CStatePersistInserter interface
//! that persists state in XML format.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Directly uses RapidXml to avoid the inefficiency of an
//! intermediate node hierarchy.
//!
class CORE_EXPORT CRapidXmlStatePersistInserter : public CStatePersistInserter {
public:
    using TStrStrMap = std::map<std::string, std::string>;
    using TStrStrMapCItr = TStrStrMap::const_iterator;

public:
    //! Root node has no attributes
    CRapidXmlStatePersistInserter(const std::string& rootName);

    //! Root node has attributes
    CRapidXmlStatePersistInserter(const std::string& rootName, const TStrStrMap& rootAttributes);

    //! Store a name/value
    virtual void insertValue(const std::string& name, const std::string& value);

    // Bring extra base class overloads into scope
    using CStatePersistInserter::insertValue;

    //! Convert to UTF-8 XML representation
    void toXml(std::string& xml) const;

    //! Convert to UTF-8 XML representation, optionally without indentation
    void toXml(bool indent, std::string& xml) const;

protected:
    //! Start a new level with the given name
    virtual void newLevel(const std::string& name);

    //! End the current level
    virtual void endLevel();

private:
    //! Get a const char * version of a string that will last at least as
    //! long as the RapidXml document
    const char* nameFromCache(const std::string& name);

private:
    //! XML documents are likely to contain the same node names many times,
    //! so just store each unique name once for efficiency
    CStringCache m_NameCache;

    using TCharRapidXmlDocument = rapidxml::xml_document<char>;
    using TCharRapidXmlNode = rapidxml::xml_node<char>;

    //! The RapidXml data structure
    TCharRapidXmlDocument m_Doc;

    //! Parent of the level we're currently inserting to
    TCharRapidXmlNode* m_LevelParent;

    //! Approximate size of final string - used to reserve memory to
    //! minimise reallocations during conversion to string representation
    size_t m_ApproxLen;
};
}
}

#endif // INCLUDED_ml_core_CRapidXmlStatePersistInserter_h
