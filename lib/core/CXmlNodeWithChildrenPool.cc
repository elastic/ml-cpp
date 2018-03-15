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
#include <core/CXmlNodeWithChildrenPool.h>

#include <core/CLogger.h>

#include <boost/bind.hpp>
#include <boost/make_shared.hpp>

#include <algorithm>


namespace ml {
namespace core {


CXmlNodeWithChildrenPool::CXmlNodeWithChildrenPool(void)
    : m_MaxRecycled(m_Recycled.max_size()) {}

CXmlNodeWithChildrenPool::CXmlNodeWithChildrenPool(size_t maxRecycled)
    : m_MaxRecycled(std::min(maxRecycled, m_Recycled.max_size())) {}

CXmlNodeWithChildren::TXmlNodeWithChildrenP CXmlNodeWithChildrenPool::newNode(void) {
    if (m_Recycled.empty()) {
        return boost::make_shared<CXmlNodeWithChildren>();
    }

    CXmlNodeWithChildren::TXmlNodeWithChildrenP nodePtr(m_Recycled.back());
    m_Recycled.pop_back();
    return nodePtr;
}

CXmlNodeWithChildren::TXmlNodeWithChildrenP CXmlNodeWithChildrenPool::newNode(std::string name,
                                                                              std::string value) {
    CXmlNodeWithChildren::TXmlNodeWithChildrenP nodePtr(this->newNode());

    // We take advantage of friendship here to set the node's name and value
    nodePtr->m_Name.swap(name);
    nodePtr->m_Value.swap(value);

    return nodePtr;
}

CXmlNodeWithChildren::TXmlNodeWithChildrenP CXmlNodeWithChildrenPool::newNode(const std::string &name,
                                                                              double value,
                                                                              CIEEE754::EPrecision precision) {
    return this->newNode(name, CStringUtils::typeToStringPrecise(value, precision));
}

void CXmlNodeWithChildrenPool::recycle(CXmlNodeWithChildren::TXmlNodeWithChildrenP &nodePtr) {
    if (nodePtr == 0) {
        LOG_ERROR("Unexpected NULL pointer");
        return;
    }

    if (m_Recycled.size() < m_MaxRecycled) {
        // We take advantage of friendship here to clear the node's attribute vector
        nodePtr->m_Attributes.clear();
        std::for_each(nodePtr->m_Children.rbegin(),
                      nodePtr->m_Children.rend(),
                      boost::bind(&CXmlNodeWithChildrenPool::recycle,
                                  this,
                                  _1));
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

