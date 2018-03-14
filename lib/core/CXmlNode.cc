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
#include <core/CXmlNode.h>

namespace ml {
namespace core {

CXmlNode::CXmlNode(void) {}

CXmlNode::CXmlNode(const std::string& name) : m_Name(name) {}

CXmlNode::CXmlNode(const std::string& name, const std::string& value)
    : m_Name(name), m_Value(value) {}

CXmlNode::CXmlNode(const std::string& name, const std::string& value, const TStrStrMap& attributes)
    : m_Name(name), m_Value(value), m_Attributes(attributes.begin(), attributes.end()) {}

CXmlNode::~CXmlNode(void) {}

const std::string& CXmlNode::name(void) const {
    return m_Name;
}

const std::string& CXmlNode::value(void) const {
    return m_Value;
}

const CXmlNode::TStrStrPrVec& CXmlNode::attributes(void) const {
    return m_Attributes;
}

void CXmlNode::name(const std::string& name) {
    m_Name = name;
}

void CXmlNode::value(const std::string& value) {
    m_Value = value;
}

std::string CXmlNode::dump(void) const {
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
