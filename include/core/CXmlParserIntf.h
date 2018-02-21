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
#ifndef INCLUDED_ml_core_CXmlParserIntf_h
#define INCLUDED_ml_core_CXmlParserIntf_h

#include <core/CNonCopyable.h>
#include <core/CXmlNodeWithChildren.h>
#include <core/ImportExport.h>

#include <string>


namespace ml
{
namespace core
{
class CStringCache;
class CXmlNodeWithChildrenPool;


//! \brief
//! Polymorphic interface to XML parser classes.
//!
//! DESCRIPTION:\n
//! Interface class to allow interchangability for our
//! different XML parser classes.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The interface contains the subset that the RapidXml parser
//! encapsulation can currently handle.  It can be expanded if
//! more functionality is added to the RapidXml parser
//! encapsulation.
//!
class CORE_EXPORT CXmlParserIntf : private CNonCopyable
{
    public:
        //! The <?xml version="1.0"?> that goes at the top of XML files
        static const std::string XML_HEADER;

    public:
        CXmlParserIntf(void);
        virtual ~CXmlParserIntf(void);

        //! Parse XML stored in a string
        virtual bool parseString(const std::string &xml) = 0;

        //! Parse XML stored in a char buffer
        virtual bool parseBuffer(const char *begin, size_t length) = 0;

        //! Parse XML stored in a char buffer that may be modified by the
        //! parsing and will outlive this object
        virtual bool parseBufferInSitu(char *begin, size_t length) = 0;

        //! Return the root element name (empty string if not parsed yet)
        virtual std::string rootElementName(void) const = 0;

        //! Dump the document to string
        virtual std::string dumpToString(void) const = 0;

        //! Convert the entire XML document into a hierarchy of node objects.
        //! This is much more efficient than making repeated calls to
        //! evalXPathExpression() to retrieve the entire contents of a parsed
        //! document.
        virtual bool toNodeHierarchy(CXmlNodeWithChildren::TXmlNodeWithChildrenP &rootNodePtr) const = 0;

        //! As above, but use a pool to avoid XML node memory allocations where possible
        virtual bool toNodeHierarchy(CXmlNodeWithChildrenPool &pool,
                                     CXmlNodeWithChildren::TXmlNodeWithChildrenP &rootNodePtr) const = 0;

        //! As above, but use a string cache to avoid string representation memory
        //! allocations where possible
        virtual bool toNodeHierarchy(CStringCache &cache,
                                     CXmlNodeWithChildren::TXmlNodeWithChildrenP &rootNodePtr) const = 0;

        //! As above, but use both a node pool and a string cache
        virtual bool toNodeHierarchy(CXmlNodeWithChildrenPool &pool,
                                     CStringCache &cache,
                                     CXmlNodeWithChildren::TXmlNodeWithChildrenP &rootNodePtr) const = 0;

        //! Functions for navigating an XML document without converting it to a
        //! node hierarchy
        virtual bool navigateRoot(void) = 0;
        virtual bool navigateFirstChild(void) = 0;
        virtual bool navigateNext(void) = 0;
        virtual bool navigateParent(void) = 0;
        virtual bool currentNodeName(std::string &name) = 0;
        virtual bool currentNodeValue(std::string &value) = 0;

        //! Replace characters that are not valid in an XML element name
        //! with underscores
        static std::string makeValidName(const std::string &str);

        //! Reformat a piece of XML to a single line.  Useful for writing files
        //! where each line contains a complete XML document.
        static std::string toOneLine(const std::string &xml);
};


}
}

#endif // INCLUDED_ml_core_CXmlParserIntf_h

