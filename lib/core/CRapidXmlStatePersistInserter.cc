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
#include <core/CRapidXmlStatePersistInserter.h>

#include <core/CLogger.h>

#include <rapidxml/rapidxml_print.hpp>

#include <iterator>

namespace ml {
namespace core {

CRapidXmlStatePersistInserter::CRapidXmlStatePersistInserter(const std::string& rootName)
    : m_LevelParent(m_Doc.allocate_node(rapidxml::node_element, this->nameFromCache(rootName), 0, rootName.length())),
      m_ApproxLen(12 + rootName.length() * 2) {
    m_Doc.append_node(m_LevelParent);
}

CRapidXmlStatePersistInserter::CRapidXmlStatePersistInserter(const std::string& rootName, const TStrStrMap& rootAttributes)
    : m_LevelParent(m_Doc.allocate_node(rapidxml::node_element, this->nameFromCache(rootName), 0, rootName.length())),
      m_ApproxLen(12 + rootName.length() * 2) {
    m_Doc.append_node(m_LevelParent);

    for (TStrStrMapCItr iter = rootAttributes.begin(); iter != rootAttributes.end(); ++iter) {
        const std::string& name = iter->first;
        const std::string& value = iter->second;
        m_LevelParent->append_attribute(m_Doc.allocate_attribute(m_Doc.allocate_string(name.c_str(), name.length()),
                                                                 value.empty() ? 0 : m_Doc.allocate_string(value.c_str(), value.length()),
                                                                 name.length(),
                                                                 value.length()));

        m_ApproxLen += 5 + name.length() + value.length();
    }
}

void CRapidXmlStatePersistInserter::insertValue(const std::string& name, const std::string& value) {
    m_LevelParent->append_node(m_Doc.allocate_node(rapidxml::node_element,
                                                   this->nameFromCache(name),
                                                   value.empty() ? 0 : m_Doc.allocate_string(value.c_str(), value.length()),

                                                   name.length(),
                                                   value.length()));

    m_ApproxLen += 5 + name.length() * 2 + value.length();
}

void CRapidXmlStatePersistInserter::toXml(std::string& xml) const {
    this->toXml(true, xml);
}

void CRapidXmlStatePersistInserter::toXml(bool indent, std::string& xml) const {
    xml.clear();
    // Hopefully the 4096 will be enough to cover any escaping required
    xml.reserve(m_ApproxLen + 4096);

    if (indent) {
        rapidxml::print(std::back_inserter(xml), m_Doc);
    } else {
        rapidxml::print(std::back_inserter(xml), m_Doc, rapidxml::print_no_indenting);
    }
}

void CRapidXmlStatePersistInserter::newLevel(const std::string& name) {
    TCharRapidXmlNode* child(m_Doc.allocate_node(rapidxml::node_element, this->nameFromCache(name), 0, name.length()));
    m_LevelParent->append_node(child);

    m_ApproxLen += 5 + name.length() * 2;

    // The child will now be the parent of everything at the new level
    m_LevelParent = child;
}

void CRapidXmlStatePersistInserter::endLevel(void) {
    TCharRapidXmlNode* levelGrandParent(m_LevelParent->parent());
    if (levelGrandParent == 0) {
        LOG_ERROR("Logic error - ending more levels than have been started");
        return;
    }

    // Step back to the level above
    m_LevelParent = levelGrandParent;
}

const char* CRapidXmlStatePersistInserter::nameFromCache(const std::string& name) {
    return m_NameCache.stringFor(name.c_str(), name.length()).c_str();
}
}
}
