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
#include <core/CRapidXmlParser.h>

#include <core/CLogger.h>
#include <core/CStringCache.h>
#include <core/CXmlNode.h>
#include <core/CXmlNodeWithChildrenPool.h>

#include <rapidxml/rapidxml_print.hpp>

#include <iterator>

#include <string.h>


namespace ml
{
namespace core
{


CRapidXmlParser::CRapidXmlParser()
    : m_XmlBufSize(0),
      m_NavigatedNode(0)
{
}

CRapidXmlParser::~CRapidXmlParser()
{
}

bool CRapidXmlParser::parseString(const std::string &xml)
{
    return this->parseBufferNonDestructive<rapidxml::parse_no_string_terminators>(xml.c_str(),
                                                                                  xml.length());
}

bool CRapidXmlParser::parseBuffer(const char *begin, size_t length)
{
    return this->parseBufferNonDestructive<rapidxml::parse_no_string_terminators>(begin,
                                                                                  length);
}

bool CRapidXmlParser::parseBufferInSitu(char *begin, size_t length)
{
    return this->parseBufferDestructive<rapidxml::parse_no_string_terminators>(begin,
                                                                               length);
}

bool CRapidXmlParser::parseStringIgnoreCdata(const std::string &xml)
{
    return this->parseBufferNonDestructive<rapidxml::parse_no_string_terminators |
                                           rapidxml::parse_no_data_nodes>(xml.c_str(),
                                                                          xml.length());
}

std::string CRapidXmlParser::rootElementName() const
{
    const TCharRapidXmlNode *root(m_Doc.first_node());
    if (root == 0)
    {
        LOG_ERROR("Error getting root element");
        return std::string();
    }

    return std::string(root->name(), root->name_size());
}

bool CRapidXmlParser::rootElementAttributes(TStrStrMap &rootAttributes) const
{
    rootAttributes.clear();

    const TCharRapidXmlNode *root(m_Doc.first_node());
    if (root == 0)
    {
        LOG_ERROR("Error getting root element");
        return false;
    }

    for (const TCharRapidXmlAttribute *attr = root->first_attribute();
         attr != 0;
         attr = attr->next_attribute())
    {
        // NB: where there are multiple attributes with the same name this keeps
        //     the last one (only keeping one attribute with a given name is a
        //     limitation throughout our XML encapsulation classes, but it
        //     hasn't been a problem in practice to date)
        rootAttributes[std::string(attr->name(),
                                   attr->name_size())].assign(attr->value(),
                                                              attr->value_size());
    }

    return true;
}

std::string CRapidXmlParser::dumpToString() const
{
    std::string result;
    rapidxml::print(std::back_inserter(result),
                    m_Doc);
    return result;
}

bool CRapidXmlParser::toNodeHierarchy(CXmlNodeWithChildren::TXmlNodeWithChildrenP &rootNodePtr) const
{
    // Because both the pool and the nodes use shared pointers, it doesn't
    // matter if the pool that originally allocates the nodes is destroyed
    // before the nodes themselves.  Hence we can get away with implementing
    // this version of the method in terms of the one that takes a pool.
    CXmlNodeWithChildrenPool pool;

    return this->toNodeHierarchy(pool, rootNodePtr);
}

bool CRapidXmlParser::toNodeHierarchy(CStringCache &cache,
                                      CXmlNodeWithChildren::TXmlNodeWithChildrenP &rootNodePtr) const
{
    // Because both the pool and the nodes use shared pointers, it doesn't
    // matter if the pool that originally allocates the nodes is destroyed
    // before the nodes themselves.  Hence we can get away with implementing
    // this version of the method in terms of the one that takes a pool.
    CXmlNodeWithChildrenPool pool;

    return this->toNodeHierarchy(pool, cache, rootNodePtr);
}

bool CRapidXmlParser::toNodeHierarchy(CXmlNodeWithChildrenPool &pool,
                                      CXmlNodeWithChildren::TXmlNodeWithChildrenP &rootNodePtr) const
{
    rootNodePtr.reset();

    const TCharRapidXmlNode *root(m_Doc.first_node());
    if (root == 0)
    {
        LOG_ERROR("Error getting root element");
        return false;
    }

    if (root->type() != rapidxml::node_element)
    {
        LOG_ERROR("Node type " << root->type() << " not supported");
        return false;
    }

    return this->toNodeHierarchy(*root, pool, 0, rootNodePtr);
}

bool CRapidXmlParser::toNodeHierarchy(CXmlNodeWithChildrenPool &pool,
                                      CStringCache &cache,
                                      CXmlNodeWithChildren::TXmlNodeWithChildrenP &rootNodePtr) const
{
    rootNodePtr.reset();

    const TCharRapidXmlNode *root(m_Doc.first_node());
    if (root == 0)
    {
        LOG_ERROR("Error getting root element");
        return false;
    }

    if (root->type() != rapidxml::node_element)
    {
        LOG_ERROR("Node type " << root->type() << " not supported");
        return false;
    }

    // Only use the cache if the current platform employs copy-on-write strings.
    // If all strings are distinct then the cache is pointless.
    CStringCache *cachePtr(cache.haveCopyOnWriteStrings() ? &cache : 0);

    return this->toNodeHierarchy(*root, pool, cachePtr, rootNodePtr);
}

bool CRapidXmlParser::toNodeHierarchy(const TCharRapidXmlNode &parentNode,
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
        nodePtr->m_Name = cache->stringFor(parentNode.name(),
                                           parentNode.name_size());
    }
    else
    {
        nodePtr->m_Name.assign(parentNode.name(), parentNode.name_size());
    }

    // For the value, we're just taking the first text element, because we
    // parsed with the parse_no_data_nodes flag.  Note that CDATA nodes get
    // appended to this in the loop at the bottom of this method.  In the event
    // of the text of a node being split between CDATA and non-CDATA, this will
    // garble it.  If this is a problem, you'll have to use the proper XML
    // parser class instead.
    nodePtr->m_Value.assign(parentNode.value(), parentNode.value_size());

    size_t numAttributes(0);
    const TCharRapidXmlAttribute *attr(parentNode.first_attribute());
    while (attr != 0)
    {
        ++numAttributes;
        attr = attr->next_attribute();
    }

    nodePtr->m_Attributes.resize(numAttributes);

    // Take advantage of friendship to add attributes directly to the parent
    // node
    attr = parentNode.first_attribute();
    for (CXmlNode::TStrStrPrVecItr iter = nodePtr->m_Attributes.begin();
         iter != nodePtr->m_Attributes.end();
         ++iter)
    {
        // Here we take advantage of friendship to directly modify the
        // CXmlNode's attributes map, thus avoiding the need to build a
        // separate map and then copy it
        if (cache != 0)
        {
            // Get attribute names and values from the cache if there is one, as
            // we expect relatively few distinct attributes repeated many times
            iter->first = cache->stringFor(attr->name(),
                                           attr->name_size());
            iter->second = cache->stringFor(attr->value(),
                                            attr->value_size());
        }
        else
        {
            iter->first.assign(attr->name(), attr->name_size());
            iter->second.assign(attr->value(), attr->value_size());
        }

        attr = attr->next_attribute();
    }

    // Recursively add the children to the parent
    const TCharRapidXmlNode *childNode(parentNode.first_node());
    while (childNode != 0)
    {
        if (childNode->type() == rapidxml::node_element)
        {
            CXmlNodeWithChildren::TXmlNodeWithChildrenP childPtr;

            if (this->toNodeHierarchy(*childNode, pool, cache, childPtr) == false)
            {
                return false;
            }

            nodePtr->addChildP(childPtr);
        }
        else if (childNode->type() == rapidxml::node_cdata)
        {
            // Append CDATA text to the value - see comment above regarding
            // garbling in complex documents
            nodePtr->m_Value.append(childNode->value(),
                                    childNode->value_size());
        }

        childNode = childNode->next_sibling();
    }

    return true;
}

bool CRapidXmlParser::navigateRoot()
{
    m_NavigatedNode = m_Doc.first_node();
    return m_NavigatedNode != 0;
}

bool CRapidXmlParser::navigateFirstChild()
{
    if (m_NavigatedNode == 0)
    {
        return false;
    }

    TCharRapidXmlNode *childNode(m_NavigatedNode->first_node());
    while (childNode != 0)
    {
        if (childNode->type() == rapidxml::node_element)
        {
            m_NavigatedNode = childNode;
            return true;
        }

        childNode = childNode->next_sibling();
    }

    return false;
}

bool CRapidXmlParser::navigateNext()
{
    if (m_NavigatedNode == 0)
    {
        return false;
    }

    TCharRapidXmlNode *nextNode(m_NavigatedNode->next_sibling());
    while (nextNode != 0)
    {
        if (nextNode->type() == rapidxml::node_element)
        {
            m_NavigatedNode = nextNode;
            return true;
        }

        nextNode = nextNode->next_sibling();
    }

    return false;
}

bool CRapidXmlParser::navigateParent()
{
    if (m_NavigatedNode == 0)
    {
        return false;
    }

    TCharRapidXmlNode *parentNode(m_NavigatedNode->parent());
    while (parentNode != 0)
    {
        if (parentNode->type() == rapidxml::node_element)
        {
            m_NavigatedNode = parentNode;
            return true;
        }

        parentNode = parentNode->parent();
    }

    return false;
}

bool CRapidXmlParser::currentNodeName(std::string &name)
{
    if (m_NavigatedNode == 0)
    {
        return false;
    }

    name.assign(m_NavigatedNode->name(), m_NavigatedNode->name_size());

    return true;
}

bool CRapidXmlParser::currentNodeValue(std::string &value)
{
    if (m_NavigatedNode == 0)
    {
        return false;
    }

    // For the value, we're just taking the first text element, because we
    // parsed with the parse_no_data_nodes flag.  Note that CDATA nodes get
    // appended to this in the loop at the bottom of this method.  In the event
    // of the text of a node being split between CDATA and non-CDATA, this will
    // garble it.  If this is a problem, you'll have to use the proper XML
    // parser class instead.
    value.assign(m_NavigatedNode->value(), m_NavigatedNode->value_size());

    // Add any CDATA children to the value
    const TCharRapidXmlNode *childNode(m_NavigatedNode->first_node());
    while (childNode != 0)
    {
        if (childNode->type() == rapidxml::node_cdata)
        {
            // Append CDATA text to the value - see comment above regarding
            // garbling in complex documents
            value.append(childNode->value(), childNode->value_size());
        }

        childNode = childNode->next_sibling();
    }

    return true;
}

void CRapidXmlParser::convert(const CXmlNodeWithChildren &root,
                              std::string &result)
{
    CRapidXmlParser::convert(true, root, result);
}

void CRapidXmlParser::convert(bool indent,
                              const CXmlNodeWithChildren &root,
                              std::string &result)
{
    // Create a temporary document
    TCharRapidXmlDocument doc;

    size_t nameLen(root.name().length());
    size_t valueLen(root.value().length());
    size_t approxLen(12 + nameLen * 2 + valueLen);

    // Root node
    TCharRapidXmlNode *rootNode(doc.allocate_node(rapidxml::node_element,
                                                  root.name().c_str(),
                                                  root.value().empty() ? 0 : root.value().c_str(),
                                                  nameLen,
                                                  valueLen));
    doc.append_node(rootNode);

    const CXmlNode::TStrStrPrVec &attrs = root.attributes();

    for (CXmlNode::TStrStrPrVecCItr attrIter = attrs.begin();
         attrIter != attrs.end();
         ++attrIter)
    {
        nameLen = attrIter->first.length();
        valueLen = attrIter->second.length();
        approxLen += 5 + nameLen + valueLen;

        TCharRapidXmlAttribute *attr(doc.allocate_attribute(attrIter->first.c_str(),
                                                            attrIter->second.empty() ? 0 : attrIter->second.c_str(),
                                                            nameLen,
                                                            valueLen));
        rootNode->append_attribute(attr);
    }

    // Create child nodes
    CRapidXmlParser::convertChildren(root, doc, *rootNode, approxLen);

    // Print to the string
    result.clear();
    result.reserve(approxLen);
    if (indent)
    {
        rapidxml::print(std::back_inserter(result),
                        doc);
    }
    else
    {
        rapidxml::print(std::back_inserter(result),
                        doc,
                        rapidxml::print_no_indenting);
    }
}

void CRapidXmlParser::convertChildren(const CXmlNodeWithChildren &current,
                                      TCharRapidXmlDocument &doc,
                                      TCharRapidXmlNode &xmlNode,
                                      size_t &approxLen)
{
    const CXmlNodeWithChildren::TChildNodePVec &childVec = current.children();

    // If a node has both children and a value, RapidXML requires that we add a
    // data node containing the value
    if (xmlNode.value_size() > 0 && !childVec.empty())
    {
        TCharRapidXmlNode *dataNode(doc.allocate_node(rapidxml::node_data,
                                                      0,
                                                      xmlNode.value(),
                                                      0,
                                                      xmlNode.value_size()));
        xmlNode.append_node(dataNode);
    }

    for (CXmlNodeWithChildren::TChildNodePVecCItr childIter = childVec.begin();
         childIter != childVec.end();
         ++childIter)
    {
        const CXmlNodeWithChildren *child = childIter->get();
        if (child != 0)
        {
            size_t nameLen(child->name().length());
            size_t valueLen(child->value().length());
            approxLen += 10 + nameLen * 2 + valueLen;

            TCharRapidXmlNode *childNode(doc.allocate_node(rapidxml::node_element,
                                                           child->name().c_str(),
                                                           child->value().empty() ? 0 : child->value().c_str(),
                                                           nameLen,
                                                           valueLen));
            xmlNode.append_node(childNode);

            const CXmlNode::TStrStrPrVec &attrs = child->attributes();

            for (CXmlNode::TStrStrPrVecCItr attrIter = attrs.begin();
                 attrIter != attrs.end();
                 ++attrIter)
            {
                nameLen = attrIter->first.length();
                valueLen = attrIter->second.length();
                approxLen += 5 + nameLen + valueLen;

                TCharRapidXmlAttribute *attr(doc.allocate_attribute(attrIter->first.c_str(),
                                                                    attrIter->second.empty() ? 0 : attrIter->second.c_str(),
                                                                    nameLen,
                                                                    valueLen));
                childNode->append_attribute(attr);
            }

            CRapidXmlParser::convertChildren(*child, doc, *childNode, approxLen);
        }
    }
}

template<int FLAGS>
bool CRapidXmlParser::parseBufferNonDestructive(const char *begin, size_t length)
{
    if (m_XmlBufSize <= length)
    {
        m_XmlBufSize = length + 1;
        m_XmlBuf.reset(new char[m_XmlBufSize]);
    }
    ::memcpy(m_XmlBuf.get(), begin, length);
    m_XmlBuf[length] = '\0';

    if (this->parseBufferDestructive<FLAGS>(m_XmlBuf.get(), length) == false)
    {
        // Only log the full XML string at the debug level, so that it doesn't
        // get sent to the socket logger
        LOG_DEBUG("XML that cannot be parsed is " <<
                  std::string(begin, length));
        return false;
    }
    return true;
}

template<int FLAGS>
bool CRapidXmlParser::parseBufferDestructive(char *begin, size_t length)
{
    m_Doc.clear();
    m_NavigatedNode = 0;
    try
    {
        m_Doc.parse<FLAGS>(begin);
    }
    catch (rapidxml::parse_error &e)
    {
        LOG_ERROR("Unable to parse XML of length " << length << ": " <<
                  e.what());
        return false;
    }

    return true;
}


}
}

