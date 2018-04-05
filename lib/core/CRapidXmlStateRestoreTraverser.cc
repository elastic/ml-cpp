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
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <core/CLogger.h>

namespace ml {
namespace core {

CRapidXmlStateRestoreTraverser::CRapidXmlStateRestoreTraverser(const CRapidXmlParser& parser)
    : m_Parser(parser), m_CurrentNode(m_Parser.m_Doc.first_node()), m_IsNameCacheValid(false), m_IsValueCacheValid(false) {
    if (m_CurrentNode != 0 && m_CurrentNode->type() != rapidxml::node_element) {
        LOG_ERROR("Node type " << m_CurrentNode->type() << " not supported");
        m_CurrentNode = 0;
        this->setBadState();
    }
}

bool CRapidXmlStateRestoreTraverser::next() {
    CRapidXmlParser::TCharRapidXmlNode* next(this->nextNodeElement());
    if (next == 0) {
        return false;
    }

    m_CurrentNode = next;

    m_IsNameCacheValid = false;
    m_IsValueCacheValid = false;

    return true;
}

bool CRapidXmlStateRestoreTraverser::hasSubLevel() const {
    return this->firstChildNodeElement() != 0;
}

const std::string& CRapidXmlStateRestoreTraverser::name() const {
    if (!m_IsNameCacheValid) {
        if (m_CurrentNode != 0) {
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
        if (m_CurrentNode != 0) {
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
    if (child == 0) {
        return false;
    }

    m_CurrentNode = child;

    m_IsNameCacheValid = false;
    m_IsValueCacheValid = false;

    return true;
}

bool CRapidXmlStateRestoreTraverser::ascend() {
    if (m_CurrentNode == 0) {
        return false;
    }

    CRapidXmlParser::TCharRapidXmlNode* parent(m_CurrentNode->parent());
    if (parent == 0) {
        return false;
    }

    m_CurrentNode = parent;

    m_IsNameCacheValid = false;
    m_IsValueCacheValid = false;

    return true;
}

CRapidXmlParser::TCharRapidXmlNode* CRapidXmlStateRestoreTraverser::nextNodeElement() const {
    if (m_CurrentNode == 0) {
        return 0;
    }

    for (CRapidXmlParser::TCharRapidXmlNode* nextNode = m_CurrentNode->next_sibling(); nextNode != 0; nextNode = nextNode->next_sibling()) {
        // We ignore comments, CDATA and any other type of node that's not an
        // element
        if (nextNode->type() == rapidxml::node_element) {
            return nextNode;
        }
    }

    return 0;
}

CRapidXmlParser::TCharRapidXmlNode* CRapidXmlStateRestoreTraverser::firstChildNodeElement() const {
    if (m_CurrentNode == 0) {
        return 0;
    }

    for (CRapidXmlParser::TCharRapidXmlNode* child = m_CurrentNode->first_node(); child != 0; child = child->next_sibling()) {
        // We ignore comments, CDATA and any other type of node that's not an
        // element
        if (child->type() == rapidxml::node_element) {
            return child;
        }
    }

    return 0;
}

bool CRapidXmlStateRestoreTraverser::isEof() const {
    return false;
}
}
}
