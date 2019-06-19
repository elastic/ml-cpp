/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CXmlNodeWithChildrenPool.h>

#include <core/CLogger.h>


#include <algorithm>
#include <memory>

namespace ml {
namespace core {

CXmlNodeWithChildrenPool::CXmlNodeWithChildrenPool()
    : m_MaxRecycled(m_Recycled.max_size()) {
}

CXmlNodeWithChildrenPool::CXmlNodeWithChildrenPool(size_t maxRecycled)
    : m_MaxRecycled(std::min(maxRecycled, m_Recycled.max_size())) {
}

CXmlNodeWithChildren::TXmlNodeWithChildrenP CXmlNodeWithChildrenPool::newNode() {
    if (m_Recycled.empty()) {
        return std::make_shared<CXmlNodeWithChildren>();
    }

    CXmlNodeWithChildren::TXmlNodeWithChildrenP nodePtr(m_Recycled.back());
    m_Recycled.pop_back();
    return nodePtr;
}

CXmlNodeWithChildren::TXmlNodeWithChildrenP
CXmlNodeWithChildrenPool::newNode(std::string name, std::string value) {
    CXmlNodeWithChildren::TXmlNodeWithChildrenP nodePtr(this->newNode());

    // We take advantage of friendship here to set the node's name and value
    nodePtr->m_Name.swap(name);
    nodePtr->m_Value.swap(value);

    return nodePtr;
}

CXmlNodeWithChildren::TXmlNodeWithChildrenP
CXmlNodeWithChildrenPool::newNode(const std::string& name, double value, CIEEE754::EPrecision precision) {
    return this->newNode(name, CStringUtils::typeToStringPrecise(value, precision));
}

void CXmlNodeWithChildrenPool::recycle(CXmlNodeWithChildren::TXmlNodeWithChildrenP& nodePtr) {
    if (nodePtr == nullptr) {
        LOG_ERROR(<< "Unexpected NULL pointer");
        return;
    }

    if (m_Recycled.size() < m_MaxRecycled) {
        // We take advantage of friendship here to clear the node's attribute vector
        nodePtr->m_Attributes.clear();
        std::for_each(nodePtr->m_Children.rbegin(), nodePtr->m_Children.rend(),
                      std::bind(&CXmlNodeWithChildrenPool::recycle, this,
                                std::placeholders::_1));
        nodePtr->m_Children.clear();
        m_Recycled.push_back(nodePtr);

        // Note that the name and value are NOT cleared here - any class that
        // gets nodes from the pool must explicitly set them
    }

    // Don't allow the recycled nodes to be accessed again through the pointer
    // passed to this function.  This also has a benefit if further processing
    // is done in the function that called recycle, because the passed pointer
    // will not be preventing lots of memory being freed.
    nodePtr.reset();
}
}
}
