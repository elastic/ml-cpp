/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CXmlNodeWithChildren.h>

#include <core/CoreTypes.h>

namespace ml {
namespace core {

CXmlNodeWithChildren::CXmlNodeWithChildren() : CXmlNode() {
}

CXmlNodeWithChildren::CXmlNodeWithChildren(const std::string& name)
    : CXmlNode(name) {
}

CXmlNodeWithChildren::CXmlNodeWithChildren(const std::string& name, const std::string& value)
    : CXmlNode(name, value) {
}

CXmlNodeWithChildren::CXmlNodeWithChildren(const std::string& name,
                                           const std::string& value,
                                           const CXmlNode::TStrStrMap& attributes)
    : CXmlNode(name, value, attributes) {
}

CXmlNodeWithChildren::CXmlNodeWithChildren(const CXmlNodeWithChildren& arg)
    : CXmlNode(arg), m_Children(arg.m_Children) {
}

CXmlNodeWithChildren::~CXmlNodeWithChildren() {
}

CXmlNodeWithChildren& CXmlNodeWithChildren::operator=(const CXmlNodeWithChildren& rhs) {
    if (this != &rhs) {
        this->CXmlNode::operator=(rhs);
        m_Children = rhs.m_Children;
    }

    return *this;
}

void CXmlNodeWithChildren::addChild(const CXmlNode& child) {
    m_Children.push_back(std::make_shared<CXmlNodeWithChildren>());
    m_Children.back()->CXmlNode::operator=(child);
}

void CXmlNodeWithChildren::addChild(const CXmlNodeWithChildren& child) {
    m_Children.push_back(std::make_shared<CXmlNodeWithChildren>(child));
}

void CXmlNodeWithChildren::addChildP(const TXmlNodeWithChildrenP& childP) {
    m_Children.push_back(childP);
}

const CXmlNodeWithChildren::TChildNodePVec& CXmlNodeWithChildren::children() const {
    return m_Children;
}

std::string CXmlNodeWithChildren::dump() const {
    return this->dump(0);
}

std::string CXmlNodeWithChildren::dump(size_t indent) const {
    std::string strRep(indent, '\t');

    // Call base class dump for name/value/attributes
    strRep += this->CXmlNode::dump();

    strRep += core_t::LINE_ENDING;

    // Now add children at next level of indenting
    for (TChildNodePVecCItr childIter = m_Children.begin();
         childIter != m_Children.end(); ++childIter) {
        const CXmlNodeWithChildren* child = childIter->get();
        if (child != nullptr) {
            strRep += child->dump(indent + 1);
        }
    }

    return strRep;
}
}
}
