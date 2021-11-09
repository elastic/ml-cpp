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
#ifndef INCLUDED_ml_core_CRapidXmlParser_h
#define INCLUDED_ml_core_CRapidXmlParser_h

#include <core/CXmlParserIntf.h>
#include <core/ImportExport.h>

#include <rapidxml/rapidxml.hpp>

#include <boost/scoped_array.hpp>

#include <map>
#include <string>

namespace ml {
namespace core {
class CRapidXmlStateRestoreTraverser;

//! \brief
//! Simple C++ wrapper around the RapidXml library.
//!
//! DESCRIPTION:\n
//! Simple C++ wrapper around the RapidXml library.
//!
//! http://rapidxml.sourceforge.net/
//!
//! The RapidXml library is a much faster parser than
//! libxml2, but only implements a small subset of the
//! XML standard.  In particular it has NO SUPPORT for:
//! 1) namespaces
//! 2) xi:include
//! 3) xPath
//!
//! This class may be used ONLY when:
//! 1) There is never any need to manipulate the XML
//!    using one of the technologies listed above
//! 2) We control the source of the XML, and hence know
//!    it won't use a feature that RapidXml cannot handle
//! 3) There is no requirement to change the contents of
//!    the XML document after creating it
//!
//! IMPLEMENTATION DECISIONS:\n
//! It is clear that there are situations where we need to be
//! able to parse XML incredibly quickly in order to
//! maintain overall system throughput.  Unfortunately,
//! libxml2 just doesn't cut the mustard in these
//! situations.
//!
//! The alternative to using a fast non-standards-compliant
//! XML parser like RapidXml would be to exchange data in
//! binary format, but this has negative implications for
//! integration with 3rd party systems, debugging, unit
//! testing, and general flexibility.  So, for the time being
//! at least, we'll try to get acceptable performance from
//! XML using RapidXml.
//!
class CORE_EXPORT CRapidXmlParser : public CXmlParserIntf {
public:
    using TStrStrMap = std::map<std::string, std::string>;
    using TStrStrMapCItr = TStrStrMap::const_iterator;

public:
    CRapidXmlParser();
    ~CRapidXmlParser() override;

    //! Parse XML stored in a string
    bool parseString(const std::string& xml) override;

    //! Parse XML stored in a char buffer
    bool parseBuffer(const char* begin, size_t length) override;

    //! Parse XML stored in a char buffer that may be modified by the
    //! parsing and will outlive this object
    bool parseBufferInSitu(char* begin, size_t length) override;

    //! Parse a string ignoring CDATA elements
    bool parseStringIgnoreCdata(const std::string& xml);

    //! Return the root element name (empty string if not parsed yet)
    std::string rootElementName() const override;

    //! Get the root element attributes (returns false if not parsed yet)
    bool rootElementAttributes(TStrStrMap& rootAttributes) const;

    //! Dump the document to string
    std::string dumpToString() const override;

    //! Convert the entire XML document into a hierarchy of node objects.
    //! This is much more efficient than making repeated calls to
    //! evalXPathExpression() to retrieve the entire contents of a parsed
    //! document.
    bool toNodeHierarchy(CXmlNodeWithChildren::TXmlNodeWithChildrenP& rootNodePtr) const override;

    //! As above, but use a pool to avoid XML node memory allocations where possible
    bool toNodeHierarchy(CXmlNodeWithChildrenPool& pool,
                         CXmlNodeWithChildren::TXmlNodeWithChildrenP& rootNodePtr) const override;

    //! As above, but use a string cache to avoid string representation memory
    //! allocations where possible
    bool toNodeHierarchy(CStringCache& cache,
                         CXmlNodeWithChildren::TXmlNodeWithChildrenP& rootNodePtr) const override;

    //! As above, but use both a node pool and a string cache
    bool toNodeHierarchy(CXmlNodeWithChildrenPool& pool,
                         CStringCache& cache,
                         CXmlNodeWithChildren::TXmlNodeWithChildrenP& rootNodePtr) const override;

    //! Functions for navigating an XML document without converting it to a
    //! node hierarchy
    bool navigateRoot() override;
    bool navigateFirstChild() override;
    bool navigateNext() override;
    bool navigateParent() override;
    bool currentNodeName(std::string& name) override;
    bool currentNodeValue(std::string& value) override;

    //! Convert a node hierarchy to XML.
    //! (This will escape the text correctly.)
    static void convert(const CXmlNodeWithChildren& root, std::string& result);

    //! Convert a node hierarchy to XML, optionally without indenting.
    //! (This will escape the text correctly.)
    static void convert(bool indent, const CXmlNodeWithChildren& root, std::string& result);

private:
    using TCharRapidXmlDocument = rapidxml::xml_document<char>;
    using TCharRapidXmlNode = rapidxml::xml_node<char>;
    using TCharRapidXmlAttribute = rapidxml::xml_attribute<char>;

    //! Called recursively by the public toNodeHierarchy() method
    bool toNodeHierarchy(const TCharRapidXmlNode& parentNode,
                         CXmlNodeWithChildrenPool& pool,
                         CStringCache* cache,
                         CXmlNodeWithChildren::TXmlNodeWithChildrenP& nodePtr) const;

    //! Called recursively by the convert() method
    static void convertChildren(const CXmlNodeWithChildren& current,
                                TCharRapidXmlDocument& doc,
                                TCharRapidXmlNode& xmlNode,
                                size_t& approxLen);

    //! Parse a buffer with some specified RapidXml flags set
    //! without modifying the contents of the buffer
    template<int FLAGS>
    bool parseBufferNonDestructive(const char* begin, size_t length);

    //! Parse a string with some specified RapidXml flags set
    //! and modifying the contents of the buffer
    template<int FLAGS>
    bool parseBufferDestructive(char* begin, size_t length);

private:
    //! RapidXml modifies the input data, so store it in an array rather
    //! than in a string to avoid any problems with reference counting in
    //! STL strings.  (Obviously the template parameter here needs to match
    //! the rapidxml typedef template arguments in the typedefs above.)
    using TScopedCharArray = boost::scoped_array<char>;

    //! RapidXml parses the XML in-situ, so keep a copy of the input
    TScopedCharArray m_XmlBuf;

    //! Size of array allocated
    size_t m_XmlBufSize;

    //! The RapidXml data structure
    TCharRapidXmlDocument m_Doc;

    //! Pointer to the current node accessed via the navigation API
    TCharRapidXmlNode* m_NavigatedNode;

    friend class CRapidXmlStateRestoreTraverser;
};
}
}

#endif // INCLUDED_ml_core_CRapidXmlParser_h
