/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <core/CLogger.h>

namespace ml {
namespace core {

CRapidXmlStateRestoreTraverser::CRapidXmlStateRestoreTraverser(const CRapidXmlParser& parser)
    : m_Parser(parser), m_CurrentNode(m_Parser.m_Doc.first_node()),
      m_IsNameCacheValid(false), m_IsValueCacheValid(false) {
    if (m_CurrentNode != nullptr && m_CurrentNode->type() != rapidxml::node_element) {
        LOG_ERROR(<< "Node type " << m_CurrentNode->type() << " not supported");
        m_CurrentNode = nullptr;
        this->setBadState();
    }
}

bool CRapidXmlStateRestoreTraverser::next() {
    CRapidXmlParser::TCharRapidXmlNode* next(this->nextNodeElement());
    if (next == nullptr) {
        return false;
    }

    m_CurrentNode = next;

    m_IsNameCacheValid = false;
    m_IsValueCacheValid = false;

    return true;
}

bool CRapidXmlStateRestoreTraverser::hasSubLevel() const {
    return this->firstChildNodeElement() != nullptr;
}

const std::string& CRapidXmlStateRestoreTraverser::name() const {
    if (!m_IsNameCacheValid) {
        if (m_CurrentNode != nullptr) {
            m_CachedName.assign(m_CurrentNode->name(), m_CurrentNode->name_size());
        } else {
            m_CachedName.clear();
        }
        m_IsNameCacheValid = true;
    }

    return m_CachedName;
}

const std::string& CRapidXmlStateRestoreTraverser::value() const {
    if (!m_IsValueCacheValid) {
        if (m_CurrentNode != nullptr) {
            // NB: this doesn't work for CDATA - see implementation decisions in
            //     the header
            m_CachedValue.assign(m_CurrentNode->value(), m_CurrentNode->value_size());
        } else {
            m_CachedValue.clear();
        }
        m_IsValueCacheValid = true;
    }
    return m_CachedValue;
}

bool CRapidXmlStateRestoreTraverser::descend() {
    CRapidXmlParser::TCharRapidXmlNode* child(this->firstChildNodeElement());
    if (child == nullptr) {
        return false;
    }

    m_CurrentNode = child;

    m_IsNameCacheValid = false;
    m_IsValueCacheValid = false;

    return true;
}

bool CRapidXmlStateRestoreTraverser::ascend() {
    if (m_CurrentNode == nullptr) {
        return false;
    }

    CRapidXmlParser::TCharRapidXmlNode* parent(m_CurrentNode->parent());
    if (parent == nullptr) {
        return false;
    }

    m_CurrentNode = parent;

    m_IsNameCacheValid = false;
    m_IsValueCacheValid = false;

    return true;
}

CRapidXmlParser::TCharRapidXmlNode* CRapidXmlStateRestoreTraverser::nextNodeElement() const {
    if (m_CurrentNode == nullptr) {
        return nullptr;
    }

    for (CRapidXmlParser::TCharRapidXmlNode* nextNode = m_CurrentNode->next_sibling();
         nextNode != nullptr; nextNode = nextNode->next_sibling()) {
        // We ignore comments, CDATA and any other type of node that's not an
        // element
        if (nextNode->type() == rapidxml::node_element) {
            return nextNode;
        }
    }

    return nullptr;
}

CRapidXmlParser::TCharRapidXmlNode*
CRapidXmlStateRestoreTraverser::firstChildNodeElement() const {
    if (m_CurrentNode == nullptr) {
        return nullptr;
    }

    for (CRapidXmlParser::TCharRapidXmlNode* child = m_CurrentNode->first_node();
         child != nullptr; child = child->next_sibling()) {
        // We ignore comments, CDATA and any other type of node that's not an
        // element
        if (child->type() == rapidxml::node_element) {
            return child;
        }
    }

    return nullptr;
}

bool CRapidXmlStateRestoreTraverser::isEof() const {
    return false;
}
}
}
