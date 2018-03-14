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
    typedef std::map<std::string, std::string> TStrStrMap;
    typedef TStrStrMap::const_iterator TStrStrMapCItr;

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
    virtual void endLevel(void);

private:
    //! Get a const char * version of a string that will last at least as
    //! long as the RapidXml document
    const char* nameFromCache(const std::string& name);

private:
    //! XML documents are likely to contain the same node names many times,
    //! so just store each unique name once for efficiency
    CStringCache m_NameCache;

    typedef rapidxml::xml_document<char> TCharRapidXmlDocument;
    typedef rapidxml::xml_node<char> TCharRapidXmlNode;

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
