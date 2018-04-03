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
#include <core/CXmlParser.h>

#include <core/CLogger.h>
#include <core/CoreTypes.h>
#include <core/CStringCache.h>
#include <core/CXmlNodeWithChildrenPool.h>

#include <libxml/tree.h>
#include <libxml/xinclude.h>
#include <libxml/xmlIO.h>
#include <libxml/xpathInternals.h>

#include <boost/scoped_array.hpp>

#include <algorithm>

#include <string.h>


namespace ml
{
namespace core
{


const std::string CXmlParser::ATTRIBUTE_SEPARATOR("@");
const std::string CXmlParser::ATTRIBUTE_EQUALS("=");
// 4 spaces to match the Ml coding standards
const size_t      CXmlParser::DEFAULT_INDENT_SPACES(4);
const size_t      CXmlParser::MAX_INDENT_SPACES(10);
// The number of spaces in this constant MUST match the maximum above
const char        *CXmlParser::INDENT_SPACE_STR("          ");


CXmlParser::CXmlParser(void)
    : m_Doc(0),
      m_XPathContext(0),
      m_NavigatedNode(0)
{
    // Note that xmlLoadExtDtdDefaultValue needs to be set before parsing,
    // but is a per-thread setting
    // xmlLoadExtDtdDefaultValue = 1;
}

CXmlParser::~CXmlParser(void)
{
    this->destroy();
}

void CXmlParser::destroy(void)
{
    if (m_XPathContext != 0)
    {
        xmlXPathFreeContext(m_XPathContext);
        m_XPathContext = 0;
    }
    if (m_Doc != 0)
    {
        xmlFreeDoc(m_Doc);
        m_Doc = 0;
    }
    m_NavigatedNode = 0;
}

bool CXmlParser::parseFile(const std::string &fileName)
{
    this->destroy();

    // Initialise globals - NOTE this current prints a line for EVERY call to
    // the error handler.  This is done here rather than in the constructor,
    // because it needs to be called for every thread in a multi-threaded
    // program.
    xmlSetGenericErrorFunc(0, &CXmlParser::errorHandler);
    xmlLoadExtDtdDefaultValue = 1;

    m_Doc = xmlParseFile(fileName.c_str());
    if (m_Doc == 0)
    {
        LOG_ERROR("Unable to parse XML file " << fileName);
        return false;
    }

    // Resolve xincludes
    if (xmlXIncludeProcess(m_Doc) < 0)
    {
        LOG_ERROR("Unable to parse XML file " << fileName);
        return false;
    }

    m_XPathContext = xmlXPathNewContext(m_Doc);
    if (m_XPathContext == 0)
    {
        this->destroy();
        LOG_ERROR("Unable to parse XML file " << fileName);
        return false;
    }

    // This makes XPath operations on large documents much faster
    xmlXPathOrderDocElems(m_Doc);

    return true;
}

bool CXmlParser::parseString(const std::string &xml)
{
    return this->parseBuffer(xml.c_str(), xml.length());
}

bool CXmlParser::parseBuffer(const char *begin, size_t length)
{
    this->destroy();

    // Initialise globals - NOTE this current prints a line for EVERY call to
    // the error handler.  This is done here rather than in the constructor,
    // because it needs to be called for every thread in a multi-threaded
    // program.
    xmlSetGenericErrorFunc(0, &CXmlParser::errorHandler);
    xmlLoadExtDtdDefaultValue = 1;

    m_Doc = xmlParseMemory(begin, static_cast<int>(length));
    if (m_Doc == 0)
    {
        LOG_ERROR("Unable to parse XML of length " << length);
        // Only log the full XML string at the debug level, so that it doesn't
        // get sent to the socket logger
        LOG_DEBUG("XML that cannot be parsed is " <<
                  std::string(begin, length));
        return false;
    }

    // Don't resolve xincludes for string parsing

    m_XPathContext = xmlXPathNewContext(m_Doc);
    if (m_XPathContext == 0)
    {
        this->destroy();
        LOG_ERROR("Unable to parse XML of length " << length);
        // Only log the full XML string at the debug level, so that it doesn't
        // get sent to the socket logger
        LOG_DEBUG("XML that cannot be parsed is " <<
                  std::string(begin, length));
        return false;
    }

    // This makes XPath operations on large documents much faster
    xmlXPathOrderDocElems(m_Doc);

    return true;
}

bool CXmlParser::parseBufferInSitu(char *begin, size_t length)
{
    // With libxml2 there's no benefit to parsing in-situ
    return this->parseBuffer(begin, length);
}

std::string CXmlParser::rootElementName(void) const
{
    if (m_Doc == 0)
    {
        LOG_ERROR("Cannot get root element for unparsed document");
        return std::string();
    }

    xmlNode *root(xmlDocGetRootElement(m_Doc));
    if (root == 0)
    {
        LOG_ERROR("Error getting root element");
        return std::string();
    }

    const char *name(reinterpret_cast<const char *>(root->name));
    if (name == 0)
    {
        LOG_ERROR("Error getting root element name");
        return std::string();
    }

    return name;
}

bool CXmlParser::evalXPathExpression(const std::string &xpathExpr,
                                     std::string &ret) const
{
    CXmlNode value;
    if (this->evalXPathExpression(xpathExpr, value) == false)
    {
        LOG_ERROR("Unable to eval " << xpathExpr);
        return false;
    }

    ret = value.value();

    return true;
}

bool CXmlParser::evalXPathExpression(const std::string &xpathExpr,
                                     CXmlNode &ret) const
{
    TXmlNodeVec vec;

    if (this->evalXPathExpression(xpathExpr, vec) == false)
    {
        return false;
    }

    if (vec.size() != 1)
    {
        LOG_ERROR("Return for " << xpathExpr << " must be single value, not " << vec.size());
        return false;
    }

    ret = vec[0];

    return true;
}

bool CXmlParser::evalXPathExpression(const std::string &xpathExpr,
                                     TStrVec &ret) const
{
    ret.clear();

    TXmlNodeVec vec;

    if (this->evalXPathExpression(xpathExpr, vec) == false)
    {
        return false;
    }

    if (vec.empty())
    {
        // This is ok
        return true;
    }

    ret.reserve(vec.size());
    for (TXmlNodeVecItr itr = vec.begin(); itr != vec.end(); ++itr)
    {
        ret.push_back(itr->value());
    }

    return true;
}

bool CXmlParser::evalXPathExpression(const std::string &xpathExpr,
                                     TStrSet &ret) const
{
    ret.clear();

    TXmlNodeVec vec;

    if (this->evalXPathExpression(xpathExpr, vec) == false)
    {
        return false;
    }

    if (vec.empty())
    {
        // This is ok
        return true;
    }

    for (TXmlNodeVecItr itr = vec.begin(); itr != vec.end(); ++itr)
    {
        if (ret.insert(itr->value()).second == false)
        {
            LOG_ERROR("Duplicate value " << itr->value());
            return false;
        }
    }

    return true;
}

bool CXmlParser::evalXPathExpression(const std::string &xpathExpr,
                                     TStrStrMap &ret) const
{
    ret.clear();

    TXmlNodeVec values;

    if (this->evalXPathExpression(xpathExpr, values) == false)
    {
        LOG_ERROR("Unable to evaluate xpath expression " << xpathExpr);
        return false;
    }

    for (TXmlNodeVecCItr itr = values.begin(); itr != values.end(); ++itr)
    {
        if (ret.insert(TStrStrMap::value_type(itr->name(), itr->value())).second == false)
        {
            LOG_ERROR("Inappropriate method call.  Tags for " << xpathExpr <<
                      " must be unique");
            return false;
        }
    }

    return true;
}

bool CXmlParser::evalXPathExpression(const std::string &xpathExpr,
                                     CXmlParser::TXmlNodeVec &ret) const
{
    ret.clear();

    if (m_Doc == 0 || m_XPathContext == 0)
    {
        LOG_ERROR("Attempt to evaluate Xpath expression before ::parseFile is called");
        return false;
    }

    xmlXPathObject *xpathObj(xmlXPathEvalExpression(reinterpret_cast<const xmlChar *>(xpathExpr.c_str()),
                                                    m_XPathContext));
    if (xpathObj == 0)
    {
        LOG_ERROR("Unable to evaluate xpath expression " << xpathExpr);
        return false;
    }

    if (xpathObj->type != XPATH_NODESET)
    {
        xmlXPathFreeObject(xpathObj);
        LOG_ERROR("Unable to evaluate xpath expression " << xpathExpr << " " << xpathObj->type);
        return false;
    }

    xmlNodeSet *nodes = xpathObj->nodesetval;
    if (nodes == 0)
    {
        xmlXPathFreeObject(xpathObj);
        // Returning 0 results is not an error at this stage
        return true;
    }

    // Sort the node set into document order
    xmlXPathNodeSetSort(nodes);

    int numEntries(nodes->nodeNr);
    for (int i = 0; i < numEntries; ++i)
    {
        xmlElementType type(nodes->nodeTab[i]->type);
        if (type == XML_ELEMENT_NODE || type == XML_ATTRIBUTE_NODE)
        {
            const xmlChar *name(nodes->nodeTab[i]->name);
            xmlChar *value(xmlNodeGetContent(nodes->nodeTab[i]));

            CXmlNode node(reinterpret_cast<const char *>(name),
                          reinterpret_cast<char *>(value));

            ret.push_back(node);

            xmlFree(value);

            CXmlNode::TStrStrPrVec &attrs = ret.back().m_Attributes;

            xmlAttr *prop(nodes->nodeTab[i]->properties);
            while (prop != 0)
            {
                const xmlChar *propName(prop->name);
                xmlChar *propValue(xmlGetProp(nodes->nodeTab[i], propName));

                attrs.push_back(CXmlNode::TStrStrPr(reinterpret_cast<const char *>(propName),
                                                    reinterpret_cast<char *>(propValue)));

                xmlFree(propValue);

                prop = prop->next;
            }
        }
        else
        {
            LOG_ERROR("Node type " << type << " not supported");
        }
    }

    xmlXPathFreeObject(xpathObj);

    return true;
}

std::string CXmlParser::dumpToString(void) const
{
    // The xmlTreeIndentString "global" is really a per-thread variable.
    // 4 spaces per indent to match Ml standard.
    xmlTreeIndentString = "    ";

    std::string result;

    if (m_Doc != 0)
    {
        // Dump the root node to a buffer and print it
        xmlBuffer *buf(xmlBufferCreate());
        xmlNode   *rootNode(xmlDocGetRootElement(m_Doc));

        xmlNodeDump(buf, m_Doc, rootNode, 0, 0);

        // Set return
        result = reinterpret_cast<const char *>(buf->content);

        // Free buffer
        xmlBufferFree(buf);
    }

    return result;
}

void CXmlParser::dumpToStdout(void) const
{
    // The xmlTreeIndentString "global" is really a per-thread variable.
    // 4 spaces per indent to match Ml standard.
    xmlTreeIndentString = "    ";

    if (m_Doc != 0)
    {
        //! NB: This won't go to the standard log file, and will be completely
        //! discarded if the program is running as a Windows service!
        xmlDocDump(stdout, m_Doc);
    }
}

void CXmlParser::convert(const CXmlNodeWithChildren &root,
                         std::string &result)
{
    CXmlParser::convert(DEFAULT_INDENT_SPACES, root, result);
}

void CXmlParser::convert(size_t indentSpaces,
                         const CXmlNodeWithChildren &root,
                         std::string &result)
{
    // The xmlTreeIndentString "global" is really a per-thread variable.
    xmlTreeIndentString = INDENT_SPACE_STR +
                          MAX_INDENT_SPACES -
                          std::min(indentSpaces,
                                   MAX_INDENT_SPACES);

    // Create a temporary document
    xmlDoc *doc(xmlNewDoc(reinterpret_cast<const xmlChar *>("1.0")));

    // Root node
    xmlNode *rootNode(xmlNewNode(0,
                                 reinterpret_cast<const xmlChar *>(root.name().c_str())));

    const CXmlNode::TStrStrPrVec &attrs = root.attributes();

    for (CXmlNode::TStrStrPrVecCItr attrIter = attrs.begin();
         attrIter != attrs.end();
         ++attrIter)
    {
        xmlSetProp(rootNode,
                   reinterpret_cast<const xmlChar *>(attrIter->first.c_str()),
                   reinterpret_cast<const xmlChar *>(attrIter->second.c_str()));
    }

    // Create child nodes
    CXmlParser::convertChildren(root, *rootNode);

    xmlDocSetRootElement(doc, rootNode);

    // Dump the root node to a buffer
    xmlBuffer *buf(xmlBufferCreate());
    xmlNodeDump(buf, doc, rootNode, 0, 1);

    // Free associated memory.
    xmlFreeDoc(doc);
    doc = 0;

    // Set return
    result = reinterpret_cast<const char *>(buf->content);

    // Free buffer
    xmlBufferFree(buf);
    buf = 0;
}

void CXmlParser::convertChildren(const CXmlNodeWithChildren &current,
                                 xmlNode &xmlRep)
{
    const CXmlNodeWithChildren::TChildNodePVec &childVec = current.children();

    for (CXmlNodeWithChildren::TChildNodePVecCItr childIter = childVec.begin();
         childIter != childVec.end();
         ++childIter)
    {
        const CXmlNodeWithChildren *child = childIter->get();
        if (child != 0)
        {
            xmlNode *childRep(0);

            if (child->value().empty() &&
                !child->children().empty())
            {
                // It's crucial to specify the value as NULL rather than
                // an empty string, otherwise the formatting will be messed
                // up
                childRep = xmlNewChild(&xmlRep,
                                       0,
                                       reinterpret_cast<const xmlChar *>(child->name().c_str()),
                                       0);
            }
            else
            {
                childRep = xmlNewTextChild(&xmlRep,
                                           0,
                                           reinterpret_cast<const xmlChar *>(child->name().c_str()),
                                           reinterpret_cast<const xmlChar *>(child->value().c_str()));
            }

            const CXmlNode::TStrStrPrVec &attrs = child->attributes();

            for (CXmlNode::TStrStrPrVecCItr attrIter = attrs.begin();
                 attrIter != attrs.end();
                 ++attrIter)
            {
                xmlSetProp(childRep,
                           reinterpret_cast<const xmlChar *>(attrIter->first.c_str()),
                           reinterpret_cast<const xmlChar *>(attrIter->second.c_str()));
            }

            CXmlParser::convertChildren(*child, *childRep);
        }
    }
}

void CXmlParser::convert(const std::string &root,
                         const TStrStrMap &values,
                         std::string &result)
{
    CXmlParser::convert(DEFAULT_INDENT_SPACES,
                        root,
                        values,
                        result);
}

void CXmlParser::convert(size_t indentSpaces,
                         const std::string &root,
                         const TStrStrMap &values,
                         std::string &result)
{
    // The xmlTreeIndentString "global" is really a per-thread variable.
    xmlTreeIndentString = INDENT_SPACE_STR +
                          MAX_INDENT_SPACES -
                          std::min(indentSpaces,
                                   MAX_INDENT_SPACES);

    // Create a temporary document
    xmlDoc *doc(xmlNewDoc(reinterpret_cast<const xmlChar *>("1.0")));

    // Root node
    xmlNode *rootNode(xmlNewNode(0, reinterpret_cast<const xmlChar *>(root.c_str())));

    // Create child nodes
    for (TStrStrMapCItr itr = values.begin(); itr != values.end(); ++itr)
    {
        // Handle an optional attribute in the form tag@name=value
        std::string tag(itr->first);
        std::string attribute;
        size_t attrPos(tag.find(ATTRIBUTE_SEPARATOR));
        if (attrPos == 0)
        {
            LOG_ERROR("Attribute separator found at position zero in tag " <<
                      tag);
            continue;
        }

        if (attrPos != std::string::npos)
        {
            attribute.assign(tag, attrPos + 1, tag.length() - attrPos - 1);
            tag.erase(attrPos);
        }

        xmlNode *childRep(xmlNewTextChild(rootNode,
                                          0,
                                          reinterpret_cast<const xmlChar *>(tag.c_str()),
                                          reinterpret_cast<const xmlChar *>(itr->second.c_str())));

        if (!attribute.empty())
        {
            size_t eqPos(attribute.find(ATTRIBUTE_EQUALS));
            if (eqPos == std::string::npos || eqPos == 0)
            {
                LOG_ERROR("Attribute format does not contain '" << ATTRIBUTE_EQUALS <<
                          "' surrounded by name and value : " << attribute <<
                          core_t::LINE_ENDING << "Map key : " << itr->first <<
                          core_t::LINE_ENDING << "Map value : " << itr->second);
            }
            else
            {
                xmlSetProp(childRep,
                           reinterpret_cast<const xmlChar *>(attribute.substr(0, eqPos).c_str()),
                           reinterpret_cast<const xmlChar *>(attribute.substr(eqPos + 1).c_str()));
            }
        }
    }

    xmlDocSetRootElement(doc, rootNode);

    // Dump the root node to a buffer and print it
    xmlBuffer *buf(xmlBufferCreate());
    xmlNodeDump(buf, doc, rootNode, 0, 0);

    // Free associated memory.
    xmlFreeDoc(doc);
    doc = 0;

    // Set return
    result = reinterpret_cast<const char *>(buf->content);

    // Free buffer
    xmlBufferFree(buf);
}

bool CXmlParser::convert(const std::string &root,
                         const TStrStrMap &values)
{
    if (m_Doc != 0)
    {
        LOG_ERROR("convert requires an empty document");
        return false;
    }

    // Create a temporary document
    m_Doc = xmlNewDoc(reinterpret_cast<const xmlChar *>("1.0"));

    // Root node
    xmlNode *rootNode(xmlNewNode(0,
                                 reinterpret_cast<const xmlChar *>(root.c_str())));

    // Create child nodes
    for (TStrStrMapCItr itr = values.begin(); itr != values.end(); ++itr)
    {
        xmlNewTextChild(rootNode,
                        0,
                        reinterpret_cast<const xmlChar *>(itr->first.c_str()),
                        reinterpret_cast<const xmlChar *>(itr->second.c_str()));
    }

    xmlDocSetRootElement(m_Doc, rootNode);

    m_XPathContext = xmlXPathNewContext(m_Doc);
    if (m_XPathContext == 0)
    {
        this->destroy();
        LOG_ERROR("Unable to convert to XML");
        return false;
    }

    // This makes XPath operations on large documents much faster
    xmlXPathOrderDocElems(m_Doc);

    return true;
}

bool CXmlParser::toNodeHierarchy(CXmlNodeWithChildren::TXmlNodeWithChildrenP &rootNodePtr) const
{
    // Because both the pool and the nodes use shared pointers, it doesn't
    // matter if the pool that originally allocates the nodes is destroyed
    // before the nodes themselves.  Hence we can get away with implementing
    // this version of the method in terms of the one that takes a pool.
    CXmlNodeWithChildrenPool pool;

    return this->toNodeHierarchy(pool, rootNodePtr);
}

bool CXmlParser::toNodeHierarchy(CXmlNodeWithChildrenPool &pool,
                                 CXmlNodeWithChildren::TXmlNodeWithChildrenP &rootNodePtr) const
{
    rootNodePtr.reset();

    if (m_Doc == 0)
    {
        LOG_ERROR("Attempt to convert to node hierarchy before ::parseFile is called");
        return false;
    }

    const xmlNode *root(xmlDocGetRootElement(const_cast<xmlDoc *>(m_Doc)));
    if (root == 0)
    {
        LOG_ERROR("Error getting root element");
        return false;
    }

    if (root->type != XML_ELEMENT_NODE)
    {
        LOG_ERROR("Node type " << root->type << " not supported");
        return false;
    }

    return this->toNodeHierarchy(*root, pool, 0, rootNodePtr);
}

bool CXmlParser::toNodeHierarchy(CStringCache &cache,
                                 CXmlNodeWithChildren::TXmlNodeWithChildrenP &rootNodePtr) const
{
    // Because both the pool and the nodes use shared pointers, it doesn't
    // matter if the pool that originally allocates the nodes is destroyed
    // before the nodes themselves.  Hence we can get away with implementing
    // this version of the method in terms of the one that takes a pool.
    CXmlNodeWithChildrenPool pool;

    return this->toNodeHierarchy(pool, cache, rootNodePtr);
}

bool CXmlParser::toNodeHierarchy(CXmlNodeWithChildrenPool &pool,
                                 CStringCache &cache,
                                 CXmlNodeWithChildren::TXmlNodeWithChildrenP &rootNodePtr) const
{
    rootNodePtr.reset();

    if (m_Doc == 0)
    {
        LOG_ERROR("Attempt to convert to node hierarchy before ::parseFile is called");
        return false;
    }

    const xmlNode *root(xmlDocGetRootElement(const_cast<xmlDoc *>(m_Doc)));
    if (root == 0)
    {
        LOG_ERROR("Error getting root element");
        return false;
    }

    if (root->type != XML_ELEMENT_NODE)
    {
        LOG_ERROR("Node type " << root->type << " not supported");
        return false;
    }

    // Only use the cache if the current platform employs copy-on-write strings.
    // If all strings are distinct then the cache is pointless.
    CStringCache *cachePtr(cache.haveCopyOnWriteStrings() ? &cache : 0);

    return this->toNodeHierarchy(*root, pool, cachePtr, rootNodePtr);
}

bool CXmlParser::navigateRoot(void)
{
    if (m_Doc != 0)
    {
        m_NavigatedNode = xmlDocGetRootElement(m_Doc);
    }
    return m_NavigatedNode != 0;
}

bool CXmlParser::navigateFirstChild(void)
{
    if (m_NavigatedNode == 0)
    {
        return false;
    }

    xmlNode *childNode(m_NavigatedNode->children);
    while (childNode != 0)
    {
        if (childNode->type == XML_ELEMENT_NODE)
        {
            m_NavigatedNode = childNode;
            return true;
        }

        childNode = childNode->next;
    }

    return false;
}

bool CXmlParser::navigateNext(void)
{
    if (m_NavigatedNode == 0)
    {
        return false;
    }

    xmlNode *nextNode(m_NavigatedNode->next);
    while (nextNode != 0)
    {
        if (nextNode->type == XML_ELEMENT_NODE)
        {
            m_NavigatedNode = nextNode;
            return true;
        }

        nextNode = nextNode->next;
    }

    return false;
}

bool CXmlParser::navigateParent(void)
{
    if (m_NavigatedNode == 0)
    {
        return false;
    }

    xmlNode *parentNode(m_NavigatedNode->parent);
    while (parentNode != 0)
    {
        if (parentNode->type == XML_ELEMENT_NODE)
        {
            m_NavigatedNode = parentNode;
            return true;
        }

        parentNode = parentNode->parent;
    }

    return false;
}

bool CXmlParser::currentNodeName(std::string &name)
{
    if (m_NavigatedNode == 0)
    {
        return false;
    }

    name = reinterpret_cast<const char *>(m_NavigatedNode->name);

    return true;
}

bool CXmlParser::currentNodeValue(std::string &value)
{
    if (m_NavigatedNode == 0)
    {
        return false;
    }

    bool isValueSet(false);

    // For the value, just concatenate direct children that are text elements.
    // (If we used xmlNodeGetContent() we'd get the text of child nodes too,
    // which we don't want, as we'll be dealing with the text in the child
    // nodes recursively.)
    const xmlNode *child(m_NavigatedNode->children);
    while (child != 0)
    {
        if (child->type == XML_TEXT_NODE ||
            child->type == XML_CDATA_SECTION_NODE)
        {
            const xmlChar *textVal(child->content);
            if (textVal != 0)
            {
                if (isValueSet)
                {
                    value += reinterpret_cast<const char *>(textVal);
                }
                else
                {
                    value = reinterpret_cast<const char *>(textVal);
                    isValueSet = true;
                }
            }
        }

        child = child->next;
    }

    if (!isValueSet)
    {
        value.clear();
    }

    return true;
}

bool CXmlParser::setRootNode(const std::string &root)
{
    if (m_Doc != 0)
    {
        LOG_ERROR("setRootNode requires an empty document");
        return false;
    }

    // Create a temporary document
    m_Doc = xmlNewDoc(reinterpret_cast<const xmlChar *>("1.0"));

    // Root node
    xmlNode *rootNode(xmlNewNode(0,
                                 reinterpret_cast<const xmlChar *>(root.c_str())));

    xmlDocSetRootElement(m_Doc, rootNode);

    m_XPathContext = xmlXPathNewContext(m_Doc);
    if (m_XPathContext == 0)
    {
        this->destroy();
        LOG_ERROR("Unable to set root node");
        return false;
    }

    // This makes XPath operations on large documents much faster
    xmlXPathOrderDocElems(m_Doc);

    return true;
}

bool CXmlParser::addNewChildNode(const std::string &name,
                                 const std::string &value)
{
    if (m_Doc == 0)
    {
        LOG_ERROR("Cannot add to uninitialised document");
        return false;
    }

    xmlNode *root(xmlDocGetRootElement(m_Doc));
    if (root == 0)
    {
        LOG_ERROR("Error getting root element");
        return false;
    }

    // Note the namespace is NULL here
    if (xmlNewTextChild(root,
                        0,
                        reinterpret_cast<const xmlChar *>(name.c_str()),
                        reinterpret_cast<const xmlChar *>(value.c_str())) == 0)
    {
        LOG_ERROR("Unable to add new child to " << root);
        return false;
    }

    // This makes XPath operations on large documents much faster
    xmlXPathOrderDocElems(m_Doc);

    return true;
}

bool CXmlParser::addNewChildNode(const std::string &name,
                                 const std::string &value,
                                 const TStrStrMap &attrs)
{
    if (m_Doc == 0)
    {
        LOG_ERROR("Cannot add to uninitialised document");
        return false;
    }

    xmlNode *root(xmlDocGetRootElement(m_Doc));
    if (root == 0)
    {
        LOG_ERROR("Error getting root element");
        return false;
    }

    // Note the namespace is NULL here
    xmlNode *child(xmlNewTextChild(root,
                                   0,
                                   reinterpret_cast<const xmlChar *>(name.c_str()),
                                   reinterpret_cast<const xmlChar *>(value.c_str())));
    if (child == 0)
    {
        LOG_ERROR("Unable to add new child to " << root);
        return false;
    }

    for (TStrStrMapCItr attrIter = attrs.begin();
         attrIter != attrs.end();
         ++attrIter)
    {
        xmlSetProp(child,
                   reinterpret_cast<const xmlChar *>(attrIter->first.c_str()),
                   reinterpret_cast<const xmlChar *>(attrIter->second.c_str()));
    }

    // This makes XPath operations on large documents much faster
    xmlXPathOrderDocElems(m_Doc);

    return true;
}

bool CXmlParser::changeChildNodeValue(const std::string &name,
                                      const std::string &newValue)
{
    if (m_Doc == 0)
    {
        LOG_ERROR("Cannot add to uninitialised document");
        return false;
    }

    xmlNode *root(xmlDocGetRootElement(m_Doc));
    if (root == 0)
    {
        LOG_ERROR("Error getting root element");
        return false;
    }

    xmlNode *child(root->children);
    while (child != 0)
    {
        if (child->type == XML_ELEMENT_NODE &&
            name == reinterpret_cast<const char *>(child->name))
        {
            // Unlike xmlNewTextChild, xmlNodeSetContent doesn't escape special
            // characters, so we have to call xmlEncodeSpecialChars ourselves to
            // do this
            xmlChar *encoded(xmlEncodeSpecialChars(m_Doc,
                                                   reinterpret_cast<const xmlChar *>(newValue.c_str())));
            xmlNodeSetContent(child, encoded);

            xmlFree(encoded);
            encoded = 0;

            return true;
        }

        child = child->next;
    }

    return false;
}

// TODO this whole function should really be replaced with a proper character
// set conversion library
bool CXmlParser::stringLatin1ToUtf8(std::string &str)
{
    // The UTF-8 character corresponding to each Latin1 character will require
    // either 1 or 2 bytes of storage (but note that some UTF-8 characters can
    // require 3 bytes)
    using TCharArray = boost::scoped_array<char>;
    size_t bufferSize(1 + 2 * str.length());
    TCharArray buffer(new char[bufferSize]);
    ::memset(&buffer[0], 0, bufferSize);

    int inLen(static_cast<int>(str.length()));
    int outLen(static_cast<int>(bufferSize));

    // This function is provided by libxml2
    int ret = ::isolat1ToUTF8(reinterpret_cast<unsigned char *>(&buffer[0]),
                              &outLen,
                              reinterpret_cast<const unsigned char *>(str.c_str()),
                              &inLen);
    if (ret == -1 || inLen < static_cast<int>(str.length()))
    {
        LOG_ERROR("Failure converting Latin1 string to UTF-8" <<
                  core_t::LINE_ENDING << "Return code: " << ret <<
                  core_t::LINE_ENDING << "Remaining length: " << inLen <<
                  core_t::LINE_ENDING << "Original string: " << str <<
                  core_t::LINE_ENDING << "Result so far: " << &buffer[0]);

        return false;
    }

    str.assign(&buffer[0], outLen);

    return true;
}

bool CXmlParser::toNodeHierarchy(const xmlNode &parentNode,
                                 CXmlNodeWithChildrenPool &pool,
                                 CStringCache *cache,
                                 CXmlNodeWithChildren::TXmlNodeWithChildrenP &nodePtr) const
{
    // Create the parent node
    nodePtr = pool.newNode();

    // Here we take advantage of friendship to directly modify the CXmlNode's
    // name and value.
    if (cache != 0)
    {
        // Get the name from the cache if there is one, as we expect relatively
        // few distinct names repeated many times
        nodePtr->m_Name = cache->stringFor(reinterpret_cast<const char *>(parentNode.name));
    }
    else
    {
        nodePtr->m_Name = reinterpret_cast<const char *>(parentNode.name);
    }

    // Nodes from the pool may contain old values
    bool isValueSet(false);

    // For the value, just concatenate direct children that are text elements.
    // (If we used xmlNodeGetContent() we'd get the text of child nodes too,
    // which we don't want, as we'll be dealing with the text in the child
    // nodes recursively.)
    const xmlNode *child(parentNode.children);
    while (child != 0)
    {
        if (child->type == XML_TEXT_NODE ||
            child->type == XML_CDATA_SECTION_NODE)
        {
            const xmlChar *textVal(child->content);
            if (textVal != 0)
            {
                if (isValueSet)
                {
                    nodePtr->m_Value += reinterpret_cast<const char *>(textVal);
                }
                else
                {
                    nodePtr->m_Value = reinterpret_cast<const char *>(textVal);
                    isValueSet = true;
                }
            }
        }

        child = child->next;
    }

    if (!isValueSet)
    {
        nodePtr->m_Value.clear();
    }

    // Take advantage of friendship to add attributes directly to the parent
    // node
    const xmlAttr *prop(parentNode.properties);
    while (prop != 0)
    {
        // Only cover the likely case.
        // (If we ever need to cover unlikely cases then use:
        // xmlChar *propValue(xmlGetProp(const_cast<xmlNode *>(&parentNode), propName));
        // followed by:
        // xmlFree(propValue);
        // but obviously this involves a temporary memory allocation.)
        const xmlNode *propChildren(prop->children);
        if (propChildren != 0 &&
            propChildren->next == 0 &&
            propChildren->type == XML_TEXT_NODE)
        {
            const char *propName(reinterpret_cast<const char *>(prop->name));
            const char *propValue(reinterpret_cast<const char *>(propChildren->content));

            // Here we take advantage of friendship to directly modify the
            // CXmlNode's attributes map, thus avoiding the need to build a
            // separate map and then copy it
            if (cache != 0)
            {
                // Get attribute names and values from the cache if there is
                // one, as we expect relatively few distinct attributes repeated
                // many times
                nodePtr->m_Attributes.push_back(CXmlNode::TStrStrPr(cache->stringFor(propName),
                                                                    cache->stringFor(propValue)));
            }
            else
            {
                nodePtr->m_Attributes.push_back(CXmlNode::TStrStrPr(propName, propValue));
            }
        }

        prop = prop->next;
    }

    // Recursively add the children to the parent
    const xmlNode *childNode(parentNode.children);
    while (childNode != 0)
    {
        if (childNode->type == XML_ELEMENT_NODE)
        {
            CXmlNodeWithChildren::TXmlNodeWithChildrenP childPtr;

            if (this->toNodeHierarchy(*childNode, pool, cache, childPtr) == false)
            {
                return false;
            }

            nodePtr->addChildP(childPtr);
        }

        childNode = childNode->next;
    }

    return true;
}

// 'Ml' error handler
// Note, this is called on every error
// TODO print a consolidated error message
void CXmlParser::errorHandler(void * /* ctxt */, const char *msg, ...)
{
    static const size_t ERRBUF_SIZE(1024);
    char                errbuf[ERRBUF_SIZE] = { '\0' };

    va_list args;
    va_start(args, msg);
    ::vsnprintf(errbuf, ERRBUF_SIZE, msg, args);
    va_end(args);

    LOG_ERROR("XML error: " << errbuf);
}


}
}

