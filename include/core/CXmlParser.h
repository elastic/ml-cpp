/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CXmlParser_h
#define INCLUDED_ml_core_CXmlParser_h

#include <core/CLogger.h>
#include <core/CStringUtils.h>
#include <core/CXmlNode.h>
#include <core/CXmlParserIntf.h>
#include <core/ImportExport.h>

#include <libxml/parser.h>
#include <libxml/xpath.h>

#include <map>
#include <set>
#include <string>
#include <vector>


namespace ml
{
namespace core
{

//! \brief
//! Simple C++ wrapper around the libxml2 library.
//!
//! DESCRIPTION:\n
//! Simple C++ wrapper around the libxml2 library.
//!
//! http://www.xmlsoft.org/
//!
//! IMPLEMENTATION DECISIONS:\n
//! Similar to example xpath1.c
//! Fully encapsulates the libxml2 library.
//! XPath used to simplify access methods.
//!
//! Uses std::string not Unicode.  This may have to change.
//! Within the implementation, there are lots of casts
//! between char* and xmlChar*.  These are technically
//! wrong, because char* will be in the character set
//! implied by $LANG, whereas xmlChar* is always UTF-8.
//! Everything is fine for the ASCII character set, i.e.
//! character codes up to 127, but for character codes
//! higher than this, the XML parser will either change
//! the characters to something different or throw an
//! error if the byte sequence is not valid UTF-8.
//! Really, every cast from char* to xmlChar* or back
//! should be replaced with a call to a character mapping
//! library, e.g. ICU4C, that can convert between UTF-8
//! and other character sets.
//!
//! Does not call xmlCleanupParser on application exit.
//!
class CORE_EXPORT CXmlParser : public CXmlParserIntf
{
    public:
        static const std::string ATTRIBUTE_SEPARATOR;
        static const std::string ATTRIBUTE_EQUALS;
        static const size_t      DEFAULT_INDENT_SPACES;
        static const size_t      MAX_INDENT_SPACES;
        static const char        *INDENT_SPACE_STR;

    public:
        using TStrVec = std::vector<std::string>;
        using TStrVecItr = TStrVec::iterator;
        using TStrVecCItr = TStrVec::const_iterator;

        using TStrSet = std::set<std::string>;
        using TStrSetItr = TStrSet::iterator;
        using TStrSetCItr = TStrSet::const_iterator;

        using TXmlNodeVec = std::vector<CXmlNode>;
        using TXmlNodeVecItr = TXmlNodeVec::iterator;
        using TXmlNodeVecCItr = TXmlNodeVec::const_iterator;

        using TStrStrMap = std::map<std::string, std::string>;
        using TStrStrMapCItr = TStrStrMap::const_iterator;

    public:
        CXmlParser(void);
        virtual ~CXmlParser(void);

        bool    parseFile(const std::string &fileName);

        //! Parse XML stored in a string
        virtual bool parseString(const std::string &xml);

        //! Parse XML stored in a char buffer
        virtual bool parseBuffer(const char *begin, size_t length);

        //! Parse XML stored in a char buffer that may be modified by the
        //! parsing and will outlive this object
        virtual bool parseBufferInSitu(char *begin, size_t length);

        //! Return the root element name (empty string if not parsed yet)
        virtual std::string rootElementName(void) const;

        //! Return result from an XPath expression, if the number of matches != 1
        //! return false.
        bool    evalXPathExpression(const std::string &xpath,
                                    CXmlNode &value) const;

        //! Return value result from an XPath expression, if the number of matches != 1
        //! return false.
        bool    evalXPathExpression(const std::string &xpath,
                                    std::string &value) const;

        //! Return value result from an XPath expression
        bool    evalXPathExpression(const std::string &xpath,
                                    TStrVec &value) const;

        //! Return value result from an XPath expression, if there are
        //! duplicates return false.
        bool    evalXPathExpression(const std::string &xpath,
                                    TStrSet &value) const;

        //! Return a value result from an XPath expression,
        //! if the number of matches != 1 or value is not of type
        //! return false.
        template<typename TYPE>
        bool    evalXPathExpression(const std::string &xpath,
                                    TYPE &ret) const
        {
            CXmlNode value;
            if (this->evalXPathExpression(xpath, value) == false)
            {
                return false;
            }

            if (CStringUtils::stringToType(value.value(), ret) == false)
            {
                LOG_ERROR("Conversion error for " << xpath);
                return false;
            }

            return true;
        }

        //! Return result from an XPath expression
        bool    evalXPathExpression(const std::string &,
                                    TXmlNodeVec &values) const;

        //! Return result from an XPath expression
        bool    evalXPathExpression(const std::string &,
                                    TStrStrMap &values) const;

        //! Dump the document to stdout
        void    dumpToStdout(void) const;

        //! Dump the document to string
        virtual std::string dumpToString(void) const;

        //! Convert a node hierarchy to XML.
        //! (This will escape the text correctly.)
        static void convert(const CXmlNodeWithChildren &root,
                            std::string &result);

        //! Convert a node hierarchy to XML.
        //! (This will escape the text correctly.)
        //! The maximum number of spaces per indent is 10.
        static void convert(size_t indentSpaces,
                            const CXmlNodeWithChildren &root,
                            std::string &result);

        //! Convert a map of name/value pairs to XML.
        //! (This will escape the text correctly.)
        //! Note root is the name of the enclosing value.
        //! Where a token name contains an @ symbol , the text prior to the @ is
        //! taken as the tag name, and the text following the @ is treated
        //! as an attribute of the tag.  For example, if the map key is
        //! field@name=idle cpu % and the map value is 99 then this will
        //! be converted to <field name="idle cpu %">99</field>
        static void convert(const std::string &root,
                            const TStrStrMap &values,
                            std::string &result);

        //! As above, but with the ability to customise the number of spaces
        //! per indent (up to a maximum of 10).
        static void convert(size_t indentSpaces,
                            const std::string &root,
                            const TStrStrMap &values,
                            std::string &result);

        //! Convert a map of name/value pairs to an XML
        //! parser.
        //! Note root is the name of the enclosing value
        bool    convert(const std::string &root,
                        const TStrStrMap &values);

        //! Convert the entire XML document into a hierarchy of node objects.
        //! This is much more efficient than making repeated calls to
        //! evalXPathExpression() to retrieve the entire contents of a parsed
        //! document.
        virtual bool toNodeHierarchy(CXmlNodeWithChildren::TXmlNodeWithChildrenP &rootNodePtr) const;

        //! As above, but use a pool to avoid XML node memory allocations where possible
        virtual bool toNodeHierarchy(CXmlNodeWithChildrenPool &pool,
                                     CXmlNodeWithChildren::TXmlNodeWithChildrenP &rootNodePtr) const;

        //! As above, but use a string cache to avoid string representation memory
        //! allocations where possible
        virtual bool toNodeHierarchy(CStringCache &cache,
                                     CXmlNodeWithChildren::TXmlNodeWithChildrenP &rootNodePtr) const;

        //! As above, but use both a node pool and a string cache
        virtual bool toNodeHierarchy(CXmlNodeWithChildrenPool &pool,
                                     CStringCache &cache,
                                     CXmlNodeWithChildren::TXmlNodeWithChildrenP &rootNodePtr) const;

        //! Functions for navigating an XML document without converting it to a
        //! node hierarchy
        virtual bool navigateRoot(void);
        virtual bool navigateFirstChild(void);
        virtual bool navigateNext(void);
        virtual bool navigateParent(void);
        virtual bool currentNodeName(std::string &name);
        virtual bool currentNodeValue(std::string &value);

        //! Set root name
        bool    setRootNode(const std::string &);

        //! Add new child element (to root).
        //! Restrict to simple <name>value</name>.
        bool    addNewChildNode(const std::string &name,
                                const std::string &value);

        //! Add new child element (to root).
        //! Allows attributes.
        bool    addNewChildNode(const std::string &name,
                                const std::string &value,
                                const TStrStrMap &attrs);

        //! Change the content of a child element (to root)
        //! Restrict to simple <name>value</name>
        bool    changeChildNodeValue(const std::string &name,
                                     const std::string &newValue);

        //! Make sure a string is in the UTF-8 character set.
        //! The XML parser is implemented internally using UTF-8, and will
        //! throw a fatal error if faced with a byte sequence that's not
        //! a valid UTF-8 character.  Hence the need to do _something_,
        //! even if it's not brilliant.
        //! TODO - in the long term the whole application needs to have
        //! proper support for different character sets.  Once that work
        //! is done, this function can be replaced with whatever we use
        //! for the overall solution.
        static bool stringLatin1ToUtf8(std::string &str);

    private:
        void    destroy(void);

        //! Called recursively by the convert() method
        static void convertChildren(const CXmlNodeWithChildren &current,
                                    xmlNode &xmlRep);

        //! Called recursively by the public toNodeHierarchy() method
        bool toNodeHierarchy(const xmlNode &parentNode,
                             CXmlNodeWithChildrenPool &pool,
                             CStringCache *cache,
                             CXmlNodeWithChildren::TXmlNodeWithChildrenP &nodePtr) const;

        //! Called on every error
        static void errorHandler(void *ctxt, const char *msg, ...);

    private:
        xmlDocPtr           m_Doc;
        xmlXPathContextPtr  m_XPathContext;

        //! Pointer to the current node accessed via the navigation API
        xmlNode             *m_NavigatedNode;
};


}
}

#endif // INCLUDED_ml_core_CXmlParser_h

