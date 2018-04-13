/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CXmlNode.h>

namespace ml {
namespace core {

CXmlNode::CXmlNode() {
}

CXmlNode::CXmlNode(const std::string& name) : m_Name(name) {
}

CXmlNode::CXmlNode(const std::string& name, const std::string& value)
    : m_Name(name), m_Value(value) {
}

CXmlNode::CXmlNode(const std::string& name, const std::string& value, const TStrStrMap& attributes)
    : m_Name(name), m_Value(value),
      m_Attributes(attributes.begin(), attributes.end()) {
}

CXmlNode::~CXmlNode() {
}

const std::string& CXmlNode::name() const {
    return m_Name;
}

const std::string& CXmlNode::value() const {
    return m_Value;
}

const CXmlNode::TStrStrPrVec& CXmlNode::attributes() const {
    return m_Attributes;
}

void CXmlNode::name(const std::string& name) {
    m_Name = name;
}

void CXmlNode::value(const std::string& value) {
    m_Value = value;
}

std::string CXmlNode::dump() const {
    std::string strRep("name=");
    strRep += m_Name;
    strRep += ";value=";
    strRep += m_Value;
    strRep += ';';

    for (TStrStrPrVecCItr itr = m_Attributes.begin(); itr != m_Attributes.end(); ++itr) {
        strRep += itr->first;
        strRep += '=';
        strRep += itr->second;
        strRep += ';';
    }

    return strRep;
}
}
}
